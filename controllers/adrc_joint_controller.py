import numpy as np
from observers.eso import ESO
from .controller import Controller
from models.manipulator_model import ManipulatorModel

class ADRCJointController(Controller):
    def __init__(self, b, kp, kd, p, q0, Tp):
        self.model = ManipulatorModel(Tp, 0.1, 0.05)
        self.b = b
        self.kp = kp
        self.kd = kd

        self.A = np.array([[0, 1, 0], [0, 0, 1], [0, 0, 0]])
        self.B = np.zeros((3, 1))
        self.B[1, 0] = self.b
        self.L = np.array([[3 * p], [3 * p ** 2], [p ** 3]])
        self.W = np.array([[1, 0, 0]])

        self.eso = ESO(self.A, self.B, self.W, self.L, q0, Tp)

    def set_b(self, b):
        """
        Metoda aktualizuje wartość parametru b oraz macierz B w obserwatorze ESO.
        """
        self.b = b
        self.B[1, 0] = self.b
        self.eso.set_B(self.B)

    def calculate_control(self, x, q_d, q_d_dot, q_d_ddot, i):
        """
        Metoda oblicza sterowanie przy użyciu algorytmu ADRC.
        """
        q, q_f = x
        z_h = self.eso.get_state()
        f_h = z_h[2]

        v = self.kp * (q_d - q) + self.kd * (q_d_dot - z_h[1]) + q_d_ddot
        u = (v - f_h) / self.b

        if i == 1:
            M_inv = np.linalg.inv(self.model.M([0.0, q, 0.0, q_f]))
            new_b = M_inv[i, i]
            self.set_b(new_b)

        self.eso.update(q, u)
        return u
