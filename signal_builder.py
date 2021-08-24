import numpy as np

class SignalBuilder:
	signal_type: dict
	horizon: int

	def __init__(elf, signal_type: dict, horizon: int) -> None:
		self.signal_type = signal_type
		self.horizon = horizon
	
	def step_function(self, start, end, transition):
		signal = np.full(self.horizon, start)
		signal[transition:] = end
		return signal

	def ramp(self, start, stop):
		signal = np.linspace(start, stop, self.horizon)
		return signal

	def sinewave(self, amplitude, median):
		x = np.linspace(-np.pi, np.pi, self.horizon)
		return median + (amplitude * np.sin(x))