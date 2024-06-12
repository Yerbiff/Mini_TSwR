import numpy as np
from models.manipulator_model import ManipulatorModel
from .controller import Controller

class FeedbackLinearizationController(Controller):
    def __init__(self, Tp):
        self.model = ManipulatorModel(Tp, 0.1, 0.05)

        # Wzmocnienia K_p i K_d
        self.K_d = 0.5
        self.K_p = 1

    def calculate_control(self, x, q_r, q_r_dot, q_r_ddot):
        """
        Implementacja linearyzacji sprzężenia zwrotnego przy użyciu modelu manipulatora,
        stanu robota oraz żądanej kontroli v.
        """
        # Obliczanie wektora v
        v = q_r_ddot + self.K_d * (q_r_dot - x[2:]) + self.K_p * (q_r - x[:2])

        # Obliczanie tau
        tau = self.model.M(x) @ v + self.model.C(x) @ x[2:]

        return tau
