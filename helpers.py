#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import sys
import mat4py
import logging
import warnings
import datetime
import matplotlib.pyplot as plt
import matplotlib.animation as animation

def suppress_warnings():
	"""
	Suppress terminal warnings (tensorflow and live plotting) for control app
	"""
	warnings.filterwarnings("ignore", message=".*iCCP.*")  # Matplotlib libpng iCCP
	warnings.filterwarnings("ignore", message=".*wayland.*")  # Qt/Wayland
	os.environ["QT_LOGGING_RULES"] = "qt.qpa.*=false"  # Qt/Wayland
	os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Tensorflow/TF-agents
	
	# Qt permission warnings
	uid = os.getuid()  # Get current user
	gid = os.getgid()  #  Get current group
	os.chown('/run/user/1000', uid, gid)  # Ensure ownership
	os.chmod('/run/user/1000', 0o700)  # Set permissions to 0700

def create_logger(name: str, print_to_console: bool=False):
	"""
    Creates a logger that writes to /log directory with a timestamped in filename

    Args:
        name (str): name used to generate the filename
		print_to_console (bool): if True, also prints log messages to console

    Returns:
        logger (logging.Logger): configured logger instance
    """
	t = datetime.datetime.now()  # Get current date and time
	current_time = t.strftime('%Y-%m-%d_%H-%M-%S')  # String from time
	log_filename = f'log/{current_time}_{name}.log'  # Name of log file with current time
	
	if not os.path.exists('log'):
		os.makedirs('log')  # Create a /log directory if we do not have one yet
	
	logger = logging.getLogger(name)
	logger.setLevel(logging.DEBUG)  # DEBUG = lowest logging level (to capture all other levels too)
	
	file_handler = logging.FileHandler(log_filename)  # To create a log file
	file_handler.setLevel(logging.DEBUG)
	formatter = logging.Formatter('%(asctime)s %(levelname)-8s %(message)s')
	file_handler.setFormatter(formatter)
	logger.addHandler(file_handler)
	
	if print_to_console:
		console_handler = logging.StreamHandler()  # To make print statements with log messages
		console_handler.setLevel(logging.DEBUG)
		console_handler.setFormatter(formatter)
		logger.addHandler(console_handler)
	
	logger.propagate = False  # Do not propagate to root logger
	logger.info('Logger created')
	logger.info('Running script ' + __file__)
	
	return logger

def load_mat(name: str):
	"""
	Loads .mat file from the /data directory

	Args:
		name (str): name of the .mat file without extension
	
	Returns:
		data (dict): dictionary containing the data from the .mat file
	"""
	try: 
		file_path = os.path.abspath(os.path.join(__file__, '..', 'data'))
		data = mat4py.loadmat(os.path.join(file_path, name + '.mat'))
		return data
	except Exception as e:
			raise Exception(f'Error while loading {name}.mat file: {e}')

def save_mat(name: str, data: dict):
	"""
	Saves passed data to a .mat file in the /data directory

	Args:
		name (str): name of the .mat file without extension
		data (dict): dictionary containing the data to be saved
	"""
	try: 
		file_path = os.path.abspath(os.path.join(__file__, '..', 'data'))
		mat4py.savemat(os.path.join(file_path, name + '_saved_from_raspy.mat'), data)
	except Exception as e:
		raise Exception(f'Error while saving {name}.mat file: {e}')

def cleanup_and_exit(logger, controller, thread=None):
	"""
	Clean-up function to be called on exit signals (ctrl+c, SIGTERM)
	
	Args:
		logger (logging.Logger): Main logger instance
		controller (Controller): Controller instace for control application
	"""
	logger.info('Exit signal received. Stopping controller ...')
	
	if thread and thread.is_alive():
		thread.join(timeout=2)
	
	try:
		controller.stop()
	except Exception:
		logger.exception('Error while stopping controller')
	sys.exit(0)


class LivePlot():
	def __init__(self, x_length, y_range, update_interval_ms: int=50):
		"""
		Initializes a live plot window for reference value and current value
		
		Args:
			x_length (int): Number of data points shown in the plot
			y_range (list): Range of values represented on the y-axis + 10%
			update_interval_ms (int): Update interval in millisecons (50ms -> 20 frames per second), default=50
		"""
		self.update_interval_ms = update_interval_ms
		self.x_axis_length = x_length
		self.y_axis_length = y_range[1] - y_range[0]  #  Calculate length of y-axis
		self.y_axis_range = [y_range[0], y_range[1] + self.y_axis_length/10]  # Add 10 percent to y-axis range
		
		self.fig, self.ax = plt.subplots()
		
		self.xs = list(range(0, x_length))
		self.y_reference = [0] * self.x_axis_length
		self.y_current = [0] * self.x_axis_length
		self.ax.set_ylim(self.y_axis_range)
		
		self.line1, = self.ax.plot(self.xs, self.y_reference)
		self.line2, = self.ax.plot(self.xs, self.y_current)
		
		self.ax.set_title('Control scope')
		self.ax.set_xlabel('Samples')
		self.ax.set_ylabel('u/V')
		self.legend = self.ax.legend(['Reference value: _V', 'Current value: _V'])  # Store legend handle to update dynamically
		self.ax.grid(True)
		
		self.new_value1 = 0.0
		self.new_value2 = 0.0
		
		# This function is called periodically from FuncAnimation
		def animate(i, y_reference, y_current):
			self.legend.remove()  # Remove old legend
			self.legend = self.ax.legend([f'Reference value: {self.new_value1:.2f}V', f'Current value: {self.new_value2:.2f}V'])  # Create new legend with latest variables
			
			y_reference.append(self.new_value1)
			y_current.append(self.new_value2)
			del y_reference[:-self.x_axis_length]  # keep last N
			del y_current[:-self.x_axis_length]  # keep last N
			self.line1.set_ydata(y_reference)
			self.line2.set_ydata(y_current)
			
			return self.line1, self.line2
		
		self.ani = animation.FuncAnimation(self.fig, animate, fargs=(self.y_reference, self.y_current), interval=self.update_interval_ms, blit=False, cache_frame_data=False)
	
	def update_values(self, reference, current):
		"""
		Update reference and current value in the live plot
		
		Args:
			ref: Reference value
			current: Current value
		"""
		self.new_value1 = reference
		self.new_value2 = current
