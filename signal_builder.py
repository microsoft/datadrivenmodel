import numpy as np


class SignalBuilder:
    def __init__(self, signal_type: dict, horizon: int, signal_params: dict) -> None:
        self.signal_type = signal_type
        self.horizon = horizon
        self.build_signal(signal_params)
        self.iteration = 0

    def step_function(self, start, stop, transition):
        signal = np.full(self.horizon + 1, start)
        signal[transition:] = stop
        return signal

    def ramp(self, start, stop):
        signal = np.linspace(start, stop, self.horizon + 1)
        return signal

    def sinewave(self, amplitude, median):
        x = np.linspace(-np.pi, np.pi, self.horizon + 1)
        return median + (amplitude * np.sin(x))

    def build_signal(self, signal_params):
        if self.signal_type == "step_function":
            self.signal = self.step_function(
                signal_params["start"],
                signal_params["stop"],
                signal_params["transition"],
            )
        elif self.signal_type == "ramp":
            self.signal = self.ramp(
                signal_params["start"],
                signal_params["stop"],
            )
        elif self.signal_type == "sinewave":
            self.signal = self.sinewave(
                signal_params["amplitude"],
                signal_params["median"],
            )
        else:
            print("Signal Type provided is not available.")

    def get_current_signal(self):
        signal = self.signal[self.iteration]
        self.iteration += 1
        return signal
