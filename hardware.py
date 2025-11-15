#!/usr/bin/env python
# -*- coding: utf-8 -*-
import time
import ADS1263
from rpi_hardware_pwm import HardwarePWM


class Pwm:
	def __init__(self, switching_frequency: int, output_channel: int):
		"""
		Initializes the Raspberry Pi hardware PWM component
		
		Args:
			switching_frequency (int): PWMs switching frequency
			output_channel (int): Channel 0 = GPIO18, channel 1 = GPIO19
		"""
		# Super buggy error handling
		# There is a bug in HardwarePWM module with permissions and file writing what happens only at first start up
		# So if we go an error here, simply start a second try
		try:
			self.pwm = HardwarePWM(pwm_channel=output_channel, hz=switching_frequency)
		except:
			time.sleep(1)
			self.pwm = HardwarePWM(pwm_channel=output_channel, hz=switching_frequency)
		self.pwm.start(0)
	
	def off(self):
		self.pwm.change_duty_cycle(0)
	
	def set_duty_cycle(self, duty_cycle):
		self.pwm.change_duty_cycle(duty_cycle*100)
	
	def duty_cycle(self):
		return round(self.pwm._duty_cycle/100, 3)


class Adc():
	def __init__(self, ref: float, samples: float, mode: float, channel: float, moving_average_window: int=1):
		"""
		Initializes the ADS1263 ADC component with SPI communication
		
		Args:
			ref (float): According to actual voltage external AVDD and AVSS(Default), or internal 2.5V
			samples (int): Possible samples: 5, 10, 20, 50, 60, 100, 400, 1200, 2400, 4800, 7200, 14400, 19200, 38400
			mode (int): 0 is singleChannel, 1 is diffChannel
			channel (int): Input channel
			moving_average_window (int): For moving average filter, default = 1
		"""
		self.channel = channel
		self.ref = ref
		self.moving_average_window = moving_average_window
		
		self.adc = ADS1263.ADS1263()
		self.adc.ADS1263_init_ADC1(f'ADS1263_{samples}SPS')
		self.adc.ADS1263_SetMode(mode)
	
	def get(self):
		"""
		Getter method with moving average filter
		"""
		voltage_list = []
		for _ in range(self.moving_average_window):
			voltage = self.adc.ADS1263_GetChannalValue(self.channel) * self.ref / 0x7fffffff
			voltage_list.append(voltage)
		voltage = sum(voltage_list)/len(voltage_list)
		return round(voltage, 3)
