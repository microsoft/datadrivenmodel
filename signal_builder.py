import numpy as np
import random
import pdb


class SignalBuilder:
    def __init__(self, signal_type: dict, horizon: int, signal_params: dict) -> None:
        self.signal_type = signal_type
        self.horizon = horizon
        self.build_signal(signal_params)

    def step_function(self, start, stop, transition):
        try:
            start = random.uniform(start["min"], start["max"])
            stop = random.uniform(stop["min"], stop["max"])
            transition = int(random.uniform(transition["min"], transition["max"]))
        except:
            transition = int(transition)

        signal = np.full(self.horizon + 1, float(start))
        signal[transition:] = stop
        return iter(signal)

    def ramp(self, start, stop):
        try:
            start = random.uniform(start["min"], start["max"])
            stop = random.uniform(stop["min"], stop["max"])
        except:
            pass

        signal = np.linspace(start, stop, self.horizon + 1)
        return iter(signal)

    def sinewave(self, amplitude, median):
        try:
            amplitude = random.uniform(amplitude["min"], amplitude["max"])
            median = random.uniform(median["min"], median["max"])
        except:
            pass

        x = np.linspace(0, 2 * np.pi, self.horizon + 1)
        return iter(median + (amplitude * np.sin(x)))

    def constant(self, value):
        try:
            value = random.uniform(value["min"], value["max"])
        except:
            pass

        return iter(value * np.ones(self.horizon + 1))

    def piecewise(self, conditions, values):
        x = np.arange(self.horizon + 1)

        try:
            conditions = [
                random.uniform(conditions["min"][i], conditions["max"][i])
                for i in range(len(conditions["max"]))
            ]
            values = [
                random.uniform(values["min"][i], values["max"][i])
                for i in range(len(values["max"]))
            ]
        except:
            pass

        signal = np.piecewise(
            x,
            [
                (x >= conditions[i]) & (x <= conditions[i + 1])
                for i in range(len(conditions) - 1)
            ],
            values,
        )

        return iter(signal)

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
        elif self.signal_type == "constant":
            self.signal = self.constant(signal_params["value"])
        elif self.signal_type == "piecewise":
            self.signal = self.piecewise(
                signal_params["conditions"],
                signal_params["values"],
            )
        else:
            print("Signal Type provided is not available.")

    def get_current_signal(self):
        return next(self.signal)
