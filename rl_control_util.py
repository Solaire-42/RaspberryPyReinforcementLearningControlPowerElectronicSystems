#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import time
import logging
import hardware
import numpy as np
import tensorflow as tf
from math import floor
from typing import Callable
from dataclasses import dataclass
from tf_agents.drivers import dynamic_step_driver
from tf_agents.utils import common
from tf_agents.agents import ddpg
from tf_agents.environments import py_environment, tf_py_environment
from tf_agents.trajectories import time_step as ts
from tf_agents.specs import array_spec
from tf_agents.replay_buffers import tf_uniform_replay_buffer
from tf_agents.trajectories import trajectory

tf.get_logger().setLevel(logging.ERROR)  # Suppress TF logger messages


@dataclass
class HardwareSettings:
	"""
	Attribute storage for the PWM and ADC hardware.
	
	Args:
		pwm_sw_freqency (int): PWM switching frequency in Hz.
		pwm_out_channel (int): PWM output channel, channel 0 = GPIO18, channel 1 = GPIO19.
		adc_ref (float): Modify according to actual voltage external AVDD and AVSS (Default), or internal 2.5V.
		adc_samples (int): ADC samples. Possible values: 5, 10, 20, 50, 60, 100, 400, 1200, 2400, 4800, 7200, 14400, 19200, 38400.
		adc_mode (int): ADC mode; 0 = singleChannel, 1 = diffChannel.
		adc_channel (int): ADC channel.
		adc_moving_average_window (int): Number of samples for moving average filter.
	"""
	pwm_sw_freqency: int
	pwm_out_channel: int
	adc_ref: float
	adc_samples: int
	adc_mode: int
	adc_channel: int
	adc_moving_average_window: int
	voltage_sensing_function: Callable[[float], float]


@dataclass
class Hyperparameter:
	"""
	Attribute storage for the hyperparameters.
	
	Args:
		num_iterations (int): Number of iterations.
		train_steps_per_iteration (int): Train steps per iteration.
		collect_steps_per_iteration (int): Collect steps per iteration.
		max_steps_per_episode (int): Maximal steps per episode.
		replay_buffer_max_length (int): Maximal replay buffer length.
		batch_size (int): Batch size.
		actor_learning_rate (float): Actor learning rate, needs careful, slower updates because if actor changes too quickly, it can destabilize training.
		critic_learning_rate (float): Critic learning rate, can learn faster since it is directly supervised with TD-errors from rewards and bootstrapping.
		target_update_tau (float): Factor for soft update of the target networks; small means smoother, more stable learning, but slower adaption.
		gamma (float): Discount factor for future rewards; close to 1 means future rewards are important.
		critic_nodes_per_layer (int): Nodes per layer in the critic network.
		actor_nodes_per_layer (int): Nodes per layer in the actor network.
	"""
	num_iterations: int
	train_steps_per_iteration: int
	collect_steps_per_iteration: int
	max_steps_per_episode: int
	replay_buffer_max_length: int
	batch_size: int
	replay_buffer_length_for_control: int
	actor_learning_rate: float
	critic_learning_rate: float
	target_update_tau: float
	gamma: float
	critic_nodes_per_layer: int
	actor_nodes_per_layer: int

def wrap_environment(raw_env):
	"""
	Wraps the given environment in a TFPyEnvironment

	Args:
		raw_env (HardwareEnvironment(py_environment.PyEnvironment)): Environment to wrap
	
	Returns:
		tf_py_environment.TFPyEnvironment: Wrapped environment
	"""	
	return tf_py_environment.TFPyEnvironment(raw_env)


class HardwareEnvironment(py_environment.PyEnvironment):
	"""
	Custom environment for the power converter hardware, implementing the PyEnvironment interface
	
	This class represents the hardware environment for the power converter
	It initializes the ADC and PWM components, defines the action and observation specifications, and implements the reset and step methods for interaction
	"""
	def __init__(self, logger, hardware_settings, hyperparameter, setpoint, find_system_limits_savemode):
		"""
		Initializes the hardware environment with ADC and PWM components

		Args:
			logger (logging.Logger): Logger instance for logging messages
			hardware_settings (HardwareSettings): Instance where pwm and adc settings are stored
			hyperparameter (Hyperparameters): Instance where hyperparameters are stored
			min_voltage_observed (float): Minimum voltage of the system
			max_voltage_observed (float): Maximum voltage of the system
		"""
		super().__init__()
		self.logger = logger
		self.logger.info(f'{__class__.__name__} - Initialization start')
		self.calculate_converter_output = hardware_settings.voltage_sensing_function
		self.max_steps_per_episode = hyperparameter.max_steps_per_episode
		self.adc = hardware.Adc(ref=hardware_settings.adc_ref, samples=hardware_settings.adc_samples, mode=hardware_settings.adc_mode, channel=hardware_settings.adc_channel, moving_average_window=hardware_settings.adc_moving_average_window)
		self.pwm = hardware.Pwm(switching_frequency=hardware_settings.pwm_sw_freqency, output_channel=hardware_settings.pwm_out_channel)
		self._reference = setpoint  # [V]
		self._save_mode_training_limit = setpoint  # [V]
		self.run_in_savemode = find_system_limits_savemode
		self._controller_state_run = False
		self._action_spec = array_spec.BoundedArraySpec(shape=(1,), dtype=np.float32, minimum=0.0, maximum=1.0, name='action')  # duty cycle in percent
		self._steps = 0
		self._accepted_control_error = 1  # [V]
		self._current_voltage = 0
		self._previous_duty_cycle = 0
		self._counter = 0
		self.episode_number = 0
		self.reward_per_episode = 0
		self._episode_ended = False
		self._counter_exploration_strategy_probabilistic_noise = 0
		self._find_system_limits()  # Find system limits and define self._min_voltage_observed, self._max_voltage_observed, self._observation_spec, self._state
		self.logger.info(f'{__class__.__name__} - Initialization done')
	
	def _find_system_limits(self):
		"""
		Finds system limits and sets instance variables self._min_voltage_observed, self._max_voltage_observed, self._observation_spec, self._state
		
		To find the system limits, an input ramp (duty cycle ramp) is executed and min. and max. converter output voltage is used as limits
		If self.run_in_savemode is true, the ramp stops at reference value, where input/output data are streched linearly to estimate full system limit range
		"""
		self.logger.info(f'{__class__.__name__} - Starting to determine system limits with duty cycle ramp (savemode = {self.run_in_savemode})')
		voltage_values = []
		duty_cycle_values = []
		for duty_cycle in range(0, 100):
			self.pwm.set_duty_cycle(duty_cycle/100)
			time.sleep(0.2)
			self.voltage()  # Update self._current_voltage
			duty_cycle_values.append(duty_cycle)
			voltage_values.append(self._current_voltage)
			if self.run_in_savemode and self._current_voltage >= self._reference:
				break
					
		if self.run_in_savemode:
			# Manipulate input ouput data to determine system limits
			# In save mode we didn't determine proper system limits because the limit was the reference value
			# This is sometimes the case because otherwise the connected hardware will be destroyed
			# To estimate the real system limits, input (duty cycle) and output (voltage) values are used to estimate real limits
			factor = 100/max(duty_cycle_values)  # Normalize our max. input value to 100% duty cycle
			duty_cycle_values = [dc * factor for dc in duty_cycle_values]  # Apply the factor and strech input to 0-100%
			# If the output already has an offset (e.g. boost converter) we have to remove the offset first before linearization
			if (offset := round(min(voltage_values))) > 0:
				voltage_values = [(u - offset) * factor + offset for u in voltage_values]  # Remove the offset, apply the factor and add back offset
			else:
				voltage_values = [u * factor for u in voltage_values]  # With no offset simply apply factor
			self._max_voltage_observed = float(floor(max(voltage_values)))
		else:
			self._max_voltage_observed = float(round(max(voltage_values)))
		
		self.duty_cycle_values_against_voltage_values = [duty_cycle_values, voltage_values]  # Store for later use during agents initial training
		self._min_voltage_observed = float(floor(min(voltage_values)))
		self._observation_spec = array_spec.BoundedArraySpec(shape=(4,), dtype=np.float32, minimum=self._min_voltage_observed, maximum=self._max_voltage_observed, name='observation')  # output, reference, error, action
		self.logger.info(f'{__class__.__name__} - System limits set from {self._min_voltage_observed}V to {self._max_voltage_observed}V')
		self._state = np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32)
	
	def _reset(self):
		"""
		Resets the environment to its initial state

		Returns:
			ts.restart: a TimeStep object representing the initial state of the environment
		"""
		self.episode_number += 1  # Increment episode number
		self.reward_per_episode = 0
		self._state = np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32)
		self._steps = 0
		self.pwm.off()
		self._previous_duty_cycle = 0
		# Wait until converters output voltage is < 1%
		while self.voltage() > (self._max_voltage_observed/100):
			time.sleep(0.1)
		self._episode_ended = False
		return ts.restart(self._state)
	
	def _reward_function(self):
		"""
		Reward function to reward or punish the agent for each step
		
		Returns:
			error (float): Control error (reference value - current value)
			reward (float): Specific if-statements to calculate reward out of error
		"""
		error = self._reference - self._current_voltage
		abs_error = abs(error)
		
		# --- 1. Smooth base reward (exponential decay) ---
		# Gives high reward near target, small but nonzero reward far away
		if self._reference > 0:
			reward = 1.5 * np.exp(- (abs_error / self._reference) * 3.0) - 1.0
		else:
			reward = 1.5 * np.exp(- (abs_error / 0.001) * 3.0) - 1.0
		
		# --- 2. Bonus when inside accepted control error band ---
		if abs_error < self._accepted_control_error:
			reward += 0.5
		
		# --- 3. Penalize overshoot stronger ---
		if self._current_voltage > (self._reference + self._accepted_control_error):
			reward -= 0.3 * (abs_error / self._reference)
		
		return error, reward
		
	def _step(self, action):
		"""
		Takes a step in the environment with the passed action

		Args:
			action (tf.Tensor): action tensor containing the duty cycle value in percent
		
		Returns:
			TimeStep object representing the next state and reward: 
			- ts.transition if the episode continues
			- ts.termination if the episode ends (max steps reached or control error within bounds)
			- ts.restart if _step() is called after the episode has ended and the controller is not in RUN state
		"""
		if self._episode_ended and not self._controller_state_run:
			return self.reset()  # Only reset if the controller is not in RUN state
		
		if not self._controller_state_run:
			self._steps += 1  # Increment if we are not in (infinite) RUN state
		
		if self._steps == 1:
			if (self._counter_exploration_strategy_probabilistic_noise == 0 or \
				self._counter_exploration_strategy_probabilistic_noise % 5 == 0) and \
				self._counter_exploration_strategy_probabilistic_noise <= 100:
				# Exploration strategy we call ”probabilistic noise" every 5th episode
				# Overwrite agents first action with a duty cycle to get close to the reference voltage
				duty_cycle_to_get_close_to_reference = self.map_voltage_to_duty_cycle(self._reference)
				duty_cycle_to_get_close_to_reference += np.random.uniform(-3, 0)  # Add some noise to the duty cycle
				self.pwm.set_duty_cycle(max(0, duty_cycle_to_get_close_to_reference/100))  # Overwrite agents action for first step
				self.logger.info(f'{__class__.__name__} - Overwrite agents first action in step {self._steps:03} for ref. {self._reference}V with DC of {self.pwm.duty_cycle()*100:.1f}%')
			else:
				self.pwm.set_duty_cycle(float(np.squeeze(action)))  # Set action from agent
			self._counter_exploration_strategy_probabilistic_noise += 1
		else:
			self.pwm.set_duty_cycle(float(np.squeeze(action)))  # Set action from agent			
		
		current_voltage = self.voltage()  # Read output (voltage is then also updated in instance variable)
		error, reward = self._reward_function()  # Calculate error and reward	
		self._state = np.array([current_voltage, self._reference, error, float(np.squeeze(action))], dtype=np.float32)
		self.reward_per_episode += reward
		
		if self._controller_state_run:
			return ts.transition(self._state, reward=reward, discount=1.0)
		else:
			# Inside the accepted control error range
			if (current_voltage > self._reference - self._accepted_control_error) and (current_voltage < self._reference + self._accepted_control_error):
				self._counter += 1
				
				# If we are n-times inside the accepted control error range, the episode is done
				if self._counter == 5:
					self._counter = 0
					self._episode_ended = True
					reward += 2 + (self.max_steps_per_episode-self._steps)/self.max_steps_per_episode  # Also give reward for fast settling with less steps needed
					self.logger.info(f'{__class__.__name__} - Episode {self.episode_number} done step {self._steps:03}, ref. {self._reference}V, u {current_voltage:.2f}V, DC of {self.pwm.duty_cycle()*100:.1f}%, err {error:.2f}, cumulative R {self.reward_per_episode:.1f}')
					self._steps = 0
					return ts.termination(self._state, reward=reward)
				else:
					return ts.transition(self._state, reward=reward+self._counter/10, discount=1.0)  # Better reward for inside accepted control error range
			
			# Outside the accepted control error range
			elif self._steps >= self.max_steps_per_episode:
				self._counter = 0
				self._episode_ended = True
				self.logger.info(f'{__class__.__name__} - Episode {self.episode_number} ended step {self._steps:03}, ref. {self._reference}V, u {current_voltage:.2f}V, DC of {self.pwm.duty_cycle()*100:.1f}%, err {error:.2f}, cumulative R {self.reward_per_episode:.1f}')
				self._steps = 0
				return ts.termination(self._state, reward=reward)
			else:
				return ts.transition(self._state, reward=reward, discount=1.0)
	
	def action_spec(self):
		"""
		Returns:
			array_spec.BoundedArraySpec: action specification for the environment
		"""
		return self._action_spec
	
	def observation_spec(self):
		"""
		Returns:
			array_spec.BoundedArraySpec: observation specification for the environment
		"""
		return self._observation_spec
	
	def voltage(self):
		"""
		Reads the current voltage from the ADC, stores it as instance variable and returns it

		Returns:
			float: current voltage value
		"""
		self._current_voltage = self.calculate_converter_output(self.adc.get())  # Get ADC voltage and calculate conveters output
		return self._current_voltage
	
	def set_reference_and_accepted_control_error(self, ref_voltage=None, acc_error=None):
		"""
		Sets the reference voltage and accepted control error for the environment if it differs from the current values

		Args:
			ref_voltage (float): reference voltage value to be set, if None random will be set, defaut=None
			acc_error (float): accepted control error to be set, if None nothing happens, default=None
		"""
		if ref_voltage == None:
			if self.run_in_savemode:
				# Calculate random number between system limits
				self._reference = round(np.random.uniform(self._min_voltage_observed, self._save_mode_training_limit), 2)
				# But we do not want too small numbers <= (min + 0.5% of max)
				if self._reference <= (self._min_voltage_observed + self._save_mode_training_limit/200):
					self._reference = round(np.random.uniform(self._save_mode_training_limit/2, self._save_mode_training_limit), 2)
			else:
				# Calculate random number between system limits
				self._reference = round(np.random.uniform(self._min_voltage_observed, self._max_voltage_observed), 2)
				# But we do not want too small numbers <= (min + 0.5% of max)
				if self._reference <= (self._min_voltage_observed + self._max_voltage_observed/200):
					self._reference = round(np.random.uniform(self._max_voltage_observed/2, self._max_voltage_observed), 2)
			
		elif ref_voltage != self._reference:
			self._reference = ref_voltage
			
		if acc_error != self._accepted_control_error and acc_error != None:
			self._accepted_control_error = acc_error
	
	def set_controller_state_to_run(self, controller_state_run: bool):
		"""
		Sets the current controller state in the environment
		Import for logging to console and to not reset the environment when the controller is in state 'RUN'

		Args:
			controller_state_run (bool): Controller state run
		"""
		self._controller_state_run = controller_state_run
	
	def map_voltage_to_duty_cycle(self, voltage: float):
		"""
		Maps approximately a given voltage to a little bit smaller duty cycle value based on the previously system limit finding

		Args:
			voltage (float): Voltage value to be mapped
		
		Returns:
			float: A smaller duty cycle value in percent
		"""
		for index, u in enumerate(self.duty_cycle_values_against_voltage_values[1]):
			if voltage < u:
				return float(self.duty_cycle_values_against_voltage_values[0][index-2])
			
		return 1.0  # If none of our list entries matches, we safely return 1% duty cycle


class PowerConverterDdpgAgent():
	"""
	Custom DDPG agent for continiuous controlling the power converter hardware environment

	This class builds the DDPG agent, creates the replay buffer, and implements methods for collecting data and training the agent
	"""
	def __init__(self, logger, hyperparameter, env, raw_env, skip_init_training_and_reload_agent):
		"""
		Initializes the DDPG agent with the given environments and logger

		Args:
			logger (logging.Logger): logger instance for logging messages
			hyperparameter (Hyperparameters): Instance where hyperparameters are stored
			env (tf_py_environment.TFPyEnvironment): TensorFlow environment for the agent
			raw_env (HardwareEnvironment): Raw hardware environment for interaction with the hardware and for setting references
			skip_init_training_and_reload_agent (bool): Flag to skip initial training and load old agent
		"""
		self.logger = logger
		self.logger.info(f'{__class__.__name__} - Initialization start')
		self.root_dir = 'DDPG'
		#self.summary_writer = tf.compat.v2.summary.create_file_writer(self.root_dir, flush_millis=10 * 1000)
		self.agent = self._build_agent(hyperparameter, env)
		self.agent_checkpointer = common.Checkpointer(ckpt_dir=os.path.join(self.root_dir, 'agent'), max_to_keep=1, agent=self.agent, global_step=self.agent.train_step_counter)
		self.min_frames_to_train = hyperparameter.replay_buffer_length_for_control
		self.replay_buffer, collect_driver, train_step_fn, time_step, policy_state = self._create_replay_buffer(hyperparameter, env)
		self._dataset = self.replay_buffer.as_dataset(num_parallel_calls=1, sample_batch_size=hyperparameter.batch_size, num_steps=2).prefetch(1)
		self._iterator = iter(self._dataset)
		#self.train_loss_threshold = 1
		#self.min_iterations_initial_training = 50000
		if skip_init_training_and_reload_agent:
			self.logger.info(f'{__class__.__name__} - Skip initial training and reload previous DDPG agent')
			self.agent_checkpointer.initialize_or_restore()
		else:		
			self.logger.info(f'{__class__.__name__} - Start initial collection and training with {hyperparameter.num_iterations} iterations')
			for it in range(hyperparameter.num_iterations):
				time_step, policy_state, train_loss = self.initial_train_step(hyperparameter, raw_env, collect_driver, train_step_fn, time_step, policy_state, it)
				#if not train_loss:
					#continue
				#train_loss_value = train_loss.loss.numpy()
				#if train_loss_value < self.train_loss_threshold and it > self.min_iterations_initial_training:
					#self.logger.info(f'{__class__.__name__} - Stop initial training at iteration {it} and a loss of {train_loss_value}')
					#break
		self.reduce_replay_buffer_length(hyperparameter, hyperparameter.replay_buffer_length_for_control, skip_init_training_and_reload_agent)  # Finally reduce replay buffer for real time control and react faster for system changes
		self.logger.info(f'{__class__.__name__} - Initialization done')
	
	def _build_agent(self, hyperparameter, env):
		"""
		Builds the DDPG agent with acording to the given environment specifications

		Args:
			hyperparameter (Hyperparameters): Instance where hyperparameters are stored
			env (tf_py_environment.TFPyEnvironment): TensorFlow environment for the agent
		
		Returns:
			ddpg.ddpg_agent.DdpgAgent: initialized DDPG agent
		"""
		time_step_spec = env.time_step_spec()
		observation_spec = time_step_spec.observation
		action_spec = env.action_spec()
		self.logger.info(f'{__class__.__name__} - Create DDPG agent with {observation_spec.shape[0]} obervation(s) and {action_spec.shape[0]} action(s)')
		
		actor_net = ddpg.actor_network.ActorNetwork(observation_spec, action_spec, fc_layer_params=(hyperparameter.actor_nodes_per_layer, hyperparameter.actor_nodes_per_layer))
		self.logger.info(f'{__class__.__name__} - Create actor network with {len(actor_net.layers)} layers and {hyperparameter.actor_nodes_per_layer} nodes')
		critic_net = ddpg.critic_network.CriticNetwork((observation_spec, action_spec), joint_fc_layer_params=(hyperparameter.critic_nodes_per_layer, hyperparameter.critic_nodes_per_layer))
		self.logger.info(f'{__class__.__name__} - Create critic network with {len(critic_net.layers)} layers and {hyperparameter.critic_nodes_per_layer} nodes')
		
		agent = ddpg.ddpg_agent.DdpgAgent(
			time_step_spec,
			action_spec,
			actor_network=actor_net,
			critic_network=critic_net,
			actor_optimizer=tf.keras.optimizers.Adam(learning_rate=hyperparameter.actor_learning_rate),
			critic_optimizer=tf.keras.optimizers.Adam(learning_rate=hyperparameter.critic_learning_rate),
			target_update_tau=hyperparameter.target_update_tau,
			# Temporal difference is the loss function to train critic network (err. between predicted Q and target Q)
			# 1. (default) Huber loss: Gentler with outliers and often more stable
			# 2. Mean squared error: Punishes large errors strongly, can be unstable if your reward/targets have big variance
			#td_errors_loss_fn=tf.compat.v1.losses.mean_squared_error,  # Default is Huber ()
			gamma=hyperparameter.gamma,
			train_step_counter=tf.Variable(0, dtype=tf.int64))
		agent.initialize()
		
		# Create checkpoint manager to save/restore the agent to disk
		#agent_checkpointer = common.Checkpointer(ckpt_dir=os.path.join(self.root_dir, 'agent'), max_to_keep=1, agent=agent, global_step=agent.train_step_counter)
		
		self.logger.info(f'{__class__.__name__} - DDPG agent + checkpointer created and initialized')
		return agent 
	
	def _create_replay_buffer(self, hyperparameter, env):
		"""
		Creates the replay buffer for the agent
		
		Args:
			hyperparameter (Hyperparameters): Instance where hyperparameters are stored
			env (tf_py_environment.TFPyEnvironment): TensorFlow environment for the agent
		
		Returns:
			Everything needed for training
		"""
		self.logger.info(f'{__class__.__name__} - Create replay buffer')
		replay_buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(data_spec=self.agent.collect_data_spec, batch_size=1, max_length=hyperparameter.replay_buffer_max_length)
		replay_observer = [replay_buffer.add_batch]
		collect_policy = self.agent.collect_policy

		# Create driver to collect experience
		collect_driver = dynamic_step_driver.DynamicStepDriver(env, collect_policy, observers=replay_observer, num_steps=hyperparameter.collect_steps_per_iteration)
		collect_driver.run = common.function(collect_driver.run)  # Wrap in tensorflow function for speed
		self.agent.train = common.function(self.agent.train)  # Wrap in tensorflow function for speed

		# Replay buffer as dataset to iterate over for training
		dataset = replay_buffer.as_dataset(num_parallel_calls=3, sample_batch_size=hyperparameter.batch_size, num_steps=2).prefetch(3)
		iterator = iter(dataset)
		
		# Wrap one training step
		def train_step_fn():
			experience, _ = next(iterator)  # Get experience batch from buffer/dataset
			training_loss = self.agent.train(experience)  # Train the agent and update actor and critic networks
			return training_loss
		
		time_step = None
		policy_state = collect_policy.get_initial_state(1)
		self.logger.info(f'{__class__.__name__} - Replay buffer setup done')
		return replay_buffer, collect_driver, train_step_fn, time_step, policy_state
	
	def reduce_replay_buffer_length(self, hyperparameter, new_length: int, skip_init_training_and_reload_agent: bool):
		"""
		Reduce replay buffer length and store most recent transitions from old buffer
		
		Args:
			hyperparameter (Hyperparameters): Instance where hyperparameters are stored
			new_length (int): The new length of the replay buffer
			skip_init_training_and_reload_agent (bool): Flag if initial training was skipped
		"""
		self.logger.info(f'{__class__.__name__} - Reduce replay buffer length to {new_length}')
		new_buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(data_spec=self.agent.collect_data_spec, batch_size=1, max_length=new_length)
		
		# If initial training was executed, store most recent transitions from old buffer
		if not skip_init_training_and_reload_agent:
			num_frames = int(self.replay_buffer.num_frames())
			old_dataset = self.replay_buffer.as_dataset(sample_batch_size=1, num_steps=1, single_deterministic_pass=True)
			num_to_copy = min(new_length, num_frames)
			recent_experiences = old_dataset.skip(max(0, num_frames - num_to_copy)).take(num_to_copy)
			copied = 0
			for exp, _ in recent_experiences:
				traj = tf.nest.map_structure(lambda t: tf.squeeze(t, axis=0), exp)
				new_buffer.add_batch(traj)
				copied += 1
			self.logger.info(f'{__class__.__name__} - Migrated {copied} recent experiences to new buffer')
		
		self.replay_buffer = new_buffer
		self._dataset = self.replay_buffer.as_dataset(num_parallel_calls=1, sample_batch_size=hyperparameter.batch_size, num_steps=2).prefetch(1)
		self._iterator = iter(self._dataset)
		self.logger.info(f'{__class__.__name__} - Replay buffer switched successfully to length {new_length}')
	
	def initial_train_step(self, hyperparameter, raw_env, collect_driver, train_step_fn, time_step, policy_state, it):
		"""
		One collect and train step for initial training of the agent
		"""
		raw_env.set_reference_and_accepted_control_error()  # Nothing passed so reference is choosen random and accepted error stays the same

		time_step, policy_state = collect_driver.run(time_step=time_step, policy_state=policy_state)
		frames = int(self.replay_buffer.num_frames())
		
		# Train only if buffer has enough samples
		train_loss = None
		if frames >= self.min_frames_to_train:
			for t in range(hyperparameter.train_steps_per_iteration):
				train_loss = train_step_fn()
		
		return time_step, policy_state, train_loss
	
	def online_train_step(self, steps: int = 1):
		"""
		Do online training updates from the replay buffer if it is sufficiently full
		
		Args:
			steps (int): Steps to take, default = 1 for training every cycle
		"""
		frames = int(self.replay_buffer.num_frames())
		if frames < self.min_frames_to_train:
			return None  # Skip training if the replay buffer has not enough samples
		
		for _ in range(steps):
			experience, _ = next(self._iterator)
			_ = self.agent.train(experience)

	def collect_step(self, environment):
		"""
		Collects a single step from the environment using the collect policy and adds it to the replay buffer

		Args:
			environment (tf_py_environment.TFPyEnvironment): TensorFlow environment from which to collect the step
		"""
		time_step = environment.current_time_step()
		action_step = self.agent.policy.action(time_step)  # Use greedy policy instead of collect policy for stability and no exploration
		next_time_step = environment.step(action_step.action)
		traj = trajectory.from_transition(time_step, action_step, next_time_step)
		self.replay_buffer.add_batch(traj)
		return time_step, action_step.action
	
	def save_agent(self):
		"""
		Save (checkpoint) current agent (actor, critic, target, optimizer, train step counter) to /DDPG/agent directory
		"""
		self.logger.info(f'{__class__.__name__} - Save agent in /{self.root_dir} to reload it later again ...')
		self.agent_checkpointer.save(global_step=self.agent.train_step_counter.numpy())
