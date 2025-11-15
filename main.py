#!/usr/bin/env python
# -*- coding: utf-8 -*-
import sys
import signal
from helpers import suppress_warnings, create_logger, cleanup_and_exit
from rl_control_util import HardwareSettings, Hyperparameter
from controller_util import Controller, ControllerStateMachine

def voltage_sensing_function(voltage: float) -> float:
	"""Function how to calculate converter output voltage from voltage sensing circuit"""
	var1 = -5.311
	var2 = 9.063
	var3 = 0.8411  #²
	var4 = -0.1545  #³
	converters_output_voltage = var4*voltage**3 + var3*voltage**2 + var2*voltage + var1
	return round(converters_output_voltage, 1)

def main(args):
	suppress_warnings()  # Keep terminal free from framework warnings
	logger = create_logger('logger', print_to_console=True)  # Instantiate main logger for application which prints to console
	
	hardware_settings = HardwareSettings(
		pwm_sw_freqency = 20_000,
		pwm_out_channel = 1,
		adc_ref = 5.08,
		adc_samples = 38400,
		adc_mode = 0,
		adc_channel = 0,
		adc_moving_average_window = 1,
		voltage_sensing_function = voltage_sensing_function
		)
	
	hyperparameter = Hyperparameter(
		num_iterations = 25_000,
		train_steps_per_iteration = 10,
		collect_steps_per_iteration = 10,
		max_steps_per_episode = 200,
		replay_buffer_max_length = 10_000,
		batch_size = 64,
		replay_buffer_length_for_control = 64*2,  # Reduce to recognize system changes faster
		actor_learning_rate = 1e-4,  # Needs careful, slower updates because if actor changes too quickly, it can destabilize training
		critic_learning_rate = 1e-3,  # Can learn faster since it is directly supervised with TD-errors from rewards and bootstrapping
		target_update_tau = 0.005,  # Factor for soft update of the target networks, small means smoother more stable learning, but slower adaption
		gamma = 0.99,  # Discount factor for future rewards, close to 1 means future rewards are important
		critic_nodes_per_layer = 64,
		actor_nodes_per_layer = 64)
	
	controller = Controller(logger, hardware_settings, hyperparameter)  # Instantiate controller
	state_machine = ControllerStateMachine(logger, controller)  # Instantiate controller state machine
	thread_for_control = None
	logger.info('Use the command line to interact with the control application\n')
	
	def handler(*_):
		"""Register cleanup handler"""
		cleanup_and_exit(logger, controller, thread_for_control)
	
	signal.signal(signal.SIGINT, handler)
	signal.signal(signal.SIGTERM, handler)
	
	try:
		while(state_machine.state_machine_running):
			state_machine.cyclic()
	finally:
		controller.agent.save_agent()
		return 0

if __name__ == '__main__':
	sys.exit(main(sys.argv))
