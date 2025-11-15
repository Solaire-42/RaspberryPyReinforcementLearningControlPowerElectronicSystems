#!/usr/bin/env python
# -*- coding: utf-8 -*-
import time
import threading
import matplotlib.pyplot as plt
from enum import Enum, auto
from helpers import create_logger, LivePlot
from rl_control_util import HardwareEnvironment, PowerConverterDdpgAgent, wrap_environment


class ControllerState(Enum):
	"""
	Enum representing the states of the controller
	"""
	INIT = auto()
	READY = auto()
	STOP = auto()
	RUN = auto()
	ERROR = auto()


def control_loop_for_threading(controller, cycle_time_s: float, live=None):
	"""
	Control loop function to be run in a separate thread with fixed cycle time so user interaction in terminal is still possible
	
	Args:
		controller (Controller): Controller instance to execute control cyclic
		cycle_time_s (float): Time in seconds for each cycle
		live (live_plot.LivePlot, optional): For updating the live plot if enabled
	"""
	next_time = time.monotonic()  # Initial start time (for fixed cycle time calculation)
	
	while controller.state == ControllerState.RUN:
		reference_value, current_value = controller.cyclic()
		if live is not None:
			live.update_values(reference_value, current_value)  # Update live plot
		
		next_time += cycle_time_s  # Time what we have for current cycle
		now = time.monotonic()  # Time what it took for current cycle
		time_to_sleep = next_time - now  # Time to sleep to have fixed cycle time (or maybe cycle time overrun)
		
		if time_to_sleep > 0:
			time.sleep(time_to_sleep)  # Sleep for rest time to have fixed cycle time
		else:
			controller.log_cycle_time_overrun(time_to_sleep)
			next_time = now  # Overrun, resync to current time
		
		controller.log_values()  # Log at the end so log time matches the cycle time
			
	if live is not None:
		live.update_values(0, 0)  # Set to zero when controlling is stopped


class Controller:
	"""
	Controller class for orchestration of the control application with hardware and user interaction with state machine

	This class initializes the hardware environment, sets up the DDPG agent, and provides methods for runnnig control app with user interaction
	"""
	def __init__(self, logger, hardware_settings, hyperparameter):
		"""
		Initializes the controller with hardware settings and hyperparameter for DDPG agent

		Args:
			logger (logging.Logger): Logger instance for logging messages
			hardware_settings (HardwareSettings): Instance where pwm and adc settings are stored
			hyperparameter (Hyperparameters): Instance where hyperparameters are stored
		"""
		self.logger = logger
		self.logger.info(f'{__class__.__name__} - Initialization start')
		self.hardware_settings = hardware_settings
		self.hyperparameter = hyperparameter
		self.logger_for_control = create_logger('control')
		self.find_system_limits_in_savemode = True  # For initial syste limits determination with ramp input (if True, ramp stops at ref. value)
		self.setpoint = 0.0
		self.setpoint_temp = self.setpoint  # Temp setpoint to store value while ramp execution
		self.ramp_execution = False  # Reference ramp to see how good the controller performs
		self.ramp_increment = 1.0  # [V], how fast the reference ramp increases
		self.accepted_control_error = 1.0
		self.min_max_voltage_observed = [0.0, 1.0]
		self.state = ControllerState.INIT
		self.action = None
		self.observations = None
		self.reward = None
		self.logger.info(f'{__class__.__name__} - Initialization done')

	def get_ready(self, skip_init_training_and_reload_agent: bool=False):
		"""
		Prepares the controller for control application by initializing the hardware environment and DDPG agent (collecting and training)
		"""
		try:
			self.raw_env = HardwareEnvironment(self.logger, self.hardware_settings, self.hyperparameter, self.setpoint, self.find_system_limits_in_savemode)  # Raw hardware environment for interaction
			self.min_max_voltage_observed = [self.raw_env.observation_spec().minimum, self.raw_env.observation_spec().maximum]
			self.env = wrap_environment(self.raw_env)  # Wrapper for custom environment
			self.time_step = self.env.reset()
			self.raw_env.set_reference_and_accepted_control_error(self.setpoint, self.accepted_control_error)
			self.agent = PowerConverterDdpgAgent(self.logger, self.hyperparameter, self.env, self.raw_env, skip_init_training_and_reload_agent)
			self.raw_env.pwm.off()  # Turn of PWM after initializing
			self.state = ControllerState.READY
			self.logger.info(f'{__class__.__name__} - Controller is ready for control application')
		except Exception as e:
			self.logger.error(f'{__class__.__name__} - Error in Controller.get_ready(): {e}', exc_info=True)
			self.error()
	
	def stop(self):
		"""
		Stops the control application and PWM output
		"""
		try:
			self.raw_env.pwm.off()
			self.raw_env.set_controller_state_to_run(False)
			if self.ramp_execution:
				self.ramp_execution = False
				self.setpoint = self.setpoint_temp
			self.state = ControllerState.STOP
			self.logger.info(f'{__class__.__name__} - Stopping control')
		except Exception as e:
			self.logger.error(f'{__class__.__name__} - Error in Controller.stop(): {e}', exc_info=True)
			self.error()
	
	def start(self):
		"""
		Starts the control application and resetting the environment
		"""
		try:
			self.time_step = self.env.reset()
			self.raw_env.set_controller_state_to_run(True)
			self.state = ControllerState.RUN
			if self.ramp_execution:
				self.logger.info(f'{__class__.__name__} - Start ramp execution with reference value from {self.min_max_voltage_observed[0]}V to {self.min_max_voltage_observed[1]}V')
			else:
				self.logger.info(f'{__class__.__name__} - Start control (reference value = {self.setpoint}V)')
		except Exception as e:
			self.logger.error(f'{__class__.__name__} - Error in Controller.start(): {e}', exc_info=True)
			self.error()
	
	def error(self):
		"""
		Handles errors in the controller by stopping the PWM and setting the state to ERROR
		"""
		try:
			self.raw_env.pwm.off()
		except Exception as e:
			self.logger.error(f'{__class__.__name__} - Error in Controller.error() stopping PWM output: {e}', exc_info=True)
		self.state = ControllerState.ERROR
		self.logger.error(f'{__class__.__name__} - Oh no! Something went wrong!')

	def cyclic(self):
		"""
		Cyclic method for the control loop, called in a separate thread
		This method sets the reference voltage, gets the action from the agent, steps the environment, collects the step, trains the agent and logs values of interest
		"""
		if self.ramp_execution:
			self.setpoint = self.setpoint + self.ramp_increment
			if self.setpoint >= self.min_max_voltage_observed[1]:
				self.ramp_execution = False
				self.setpoint = self.setpoint_temp
		try:
			self.raw_env.set_reference_and_accepted_control_error(self.setpoint, self.accepted_control_error)
			
			self.time_step, action = self.agent.collect_step(self.env)
			self.agent.online_train_step(steps=1)
			
			self.action = action[0]
			self.observations = self.time_step.observation[0]
			self.reward = round(float(self.time_step.reward), 3)
			
			return self.observations[1], self.observations[0]  # ref. value, current value
			
		except Exception as e:
			self.logger.error(f'{__class__.__name__} - Error in Controller.cyclic(): {e}', exc_info=True)
			self.error()
	
	def execute_ramp(self):
		self.ramp_execution = True
		self.setpoint_temp = self.setpoint
		self.setpoint = self.min_max_voltage_observed[0]  # Start at lowest reference voltage
		voltage_range = abs(self.min_max_voltage_observed[1] - self.min_max_voltage_observed[0])
		
		if voltage_range <= 1:
			self.ramp_increment = 0.001
		elif voltage_range <= 10:
			self.ramp_increment = 0.01
		elif voltage_range <= 100:
			self.ramp_increment = 0.1
		elif voltage_range <= 1000:
			self.ramp_increment = 1
		
		self.logger.info(f'{__class__.__name__} - Starting reference ramp from {self.min_max_voltage_observed[0]}V up to {self.min_max_voltage_observed[1]}V by incrementing {self.ramp_increment}V')
	
	def log_cycle_time_overrun(self, t_overrun_s):
		self.logger_for_control.info(f'{__class__.__name__} - Cycle time overrun of {t_overrun_s:.3f}s')
	
	def log_values(self):
		self.logger_for_control.info(f'{__class__.__name__} - action={self.action[0]:.3f}, observations={self.observations}, reward={self.reward}')


class ControllerStateMachine():
	def __init__(self, logger, controller):
		self.logger = logger
		self.logger.info(f'{__class__.__name__} - Initialization start')
		self.controller = controller
		self.thread_for_control = None
		self.state = self.controller.state.name
		self.show_live_plot_window = False
		self.cycle_time_s_without_live_plot = 0.065
		self.cycle_time_s_with_live_plot = 0.2
		self.state_machine_running = True
		self.logger.info(f'{__class__.__name__} - Initialization done')
	
	def _init(self, user_input):
		if self.state == 'INIT':
			if user_input == 'train':
				self.controller.get_ready()
			elif user_input == 'restore':
				self.controller.get_ready(skip_init_training_and_reload_agent=True)
		elif self.state == 'STOP':
			plt.close('all') if plt.get_fignums() else None  # Close plot window if still open
			self.controller.state = ControllerState.INIT  # Set controller state back to init and start from beginning
	
	def _ref(self):
		self.logger.info(f'Current reference value is set to {self.controller.setpoint}V')
		input_ref = input('\nEnter new reference value in V or "ramp" for testing: ')
		if input_ref == 'ramp':
			self.controller.execute_ramp()
			if self.state == 'STOP':
				self._start()
		else:
			self.controller.setpoint = float(input_ref)
			self.logger.info(f'New reference value is set to {self.controller.setpoint}V')
	
	def _err(self):
		if self.state == 'STOP':
			self.logger.info(f'Current accepted control error is {self.controller.accepted_control_error}V')
			self.controller.accepted_control_error = float(input('Enter new accepted control error in V: '))
			self.logger.info(f'New accepted control error is set to {self.controller.accepted_control_error}V')

	def _stop(self):
		if self.state == 'READY' or self.state == 'STOP':
			self.controller.stop()
			self.logger.info(f'{__class__.__name__} - Stopping state machine')
			self.state_machine_running = False
		elif self.state == 'RUN':
			self.controller.state = ControllerState.STOP  # Little buggy: First change controller state directly to finish thread and exit while loop
			self.thread_for_control.join()  # Wait till thread has finishes
			self.controller.stop()  # Stop controller to turn off PWM after thread is done
	
	def _start(self):
		if self.state == 'READY' or self.state == 'STOP':
			# Show live plot if flag is set to True and no plot window exists yet
			if self.show_live_plot_window and not plt.get_fignums():
				self.live = LivePlot(x_length=200, y_range=[self.controller.min_max_voltage_observed[0], self.controller.min_max_voltage_observed[1]])
				plt.show(block=False)  # Show plot window but do not block the main thread so user interaction is still possible
			self.controller.start()
			if self.show_live_plot_window:
				self.thread_for_control = threading.Thread(target=control_loop_for_threading, args=(self.controller, self.cycle_time_s_with_live_plot, self.live))
			else:
				self.thread_for_control = threading.Thread(target=control_loop_for_threading, args=(self.controller, self.cycle_time_s_without_live_plot,))
			self.thread_for_control.start()
	
	def user_interaction(self):
		if self.state == 'INIT':
			self.logger.info(f'Plase enter reference value and accepted control error in V!')
			self.controller.setpoint = float(input('Reference value in V: '))
			self.controller.accepted_control_error = float(input('Accepted control error in V: '))
			self.logger.info(f'Reference value is set to {self.controller.setpoint}V and accepted control error is {self.controller.accepted_control_error}V\n')
			
			self.logger.info(f'For system limits a ramp input will be executed during initialization!')
			user_input = input(f'Enter "full" for full range or "save" for save mode (ramp stops at ref. value {self.controller.setpoint}V): ')
			if user_input == 'full':
				self.logger.info(f'System limits will be determined in full range mode!\n')
				self.controller.find_system_limits_in_savemode = False
			else:
				self.logger.info(f'System limits will be determined in save mode (stops at reference value {self.controller.setpoint}V)!\n')
				self.controller.find_system_limits_in_savemode = True
			
			user_input = input('Hit enter to continue or "live" to create a live plot window for control scope: ')
			if user_input == 'live':
				self.show_live_plot_window = True
			
			user_input = input('\nEnter "train" to initialize and train or "restore" to initialize and restore DDPG agent for control: ')
			
		elif self.state == 'READY':
			user_input = input('\nEnter "start" to start controlling or "stop" for abort: ')
			
		elif self.state == 'STOP':
			user_input = input('\nEnter "init" to initialize, "ref" to set reference, "err" to set accepted control error, "start" for controlling or "stop": ')
			
		elif self.state == 'RUN':
			user_input = input('\nEnter "stop" to stop controlling or "ref" to update reference value: ')
		
		self.logger.info(f'{user_input} command entered by user')
		return user_input
	
	def controller_action(self, user_input):
		if user_input == 'train' or user_input == 'restore' or user_input == 'init':
			self._init(user_input)
		elif user_input == 'ref':
			self._ref()
		elif user_input == 'err':
			self._err()
		elif user_input == 'stop':
			self._stop()
		elif user_input == 'start':
			self._start()
	
	def cyclic(self):
		# Get current controller state
		self.state = self.controller.state.name
		
		# Ask user what to do depending on current state
		user_input = self.user_interaction()
		
		# Then start action with controller
		self.controller_action(user_input)
