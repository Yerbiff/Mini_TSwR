import math
import matplotlib
matplotlib.use('TkAgg')
import numpy as np
from .controller import Controller
from models.manipulator_model import ManipulatorModel


class MMAController(Controller):
    def __init__(self, Tp):
        # TODO: Fill the list self.models with 3 models of 2DOF manipulators with different m3 and r3
        # I:   m3=0.1,  r3=0.05
        # II:  m3=0.01, r3=0.01
        # III: m3=1.0,  r3=0.3
        self.models = [ManipulatorModel(Tp, 0.1, 0.05),
                       ManipulatorModel(Tp, 0.01, 0.01),
                       ManipulatorModel(Tp, 1., 0.3)]
        self.i = 0
        self.Tp = Tp
        self.u = np.zeros((2, 2), dtype=np.float32)
        self.x = np.zeros((4, 1))
        self.x_dot = np.zeros((1, 2))
        self.xd = 0
        self.error = 100

    def choose_model(self, x, u, x_dot):
        # Inicjalizujemy zmienną przechowującą najlepszy błąd na bardzo wysoką wartość
        best_error = float('inf')
        best_model_index = 0

        # Dla każdego modelu w liście
        for i, model in enumerate(self.models):
            # Obliczamy błąd dopasowania dla aktualnego modelu
            M_inv = np.linalg.inv(model.M(x))
            invM_C = -M_inv @ model.C(x)

            A = np.array([[0,0,1,0],
                       [0,0,0,1],
                       [0,0,invM_C[0][0],invM_C[0][1]],
                       [0,0,invM_C[1][0],invM_C[1][1]]])

            B = np.array([[0,0],
                       [0,0],
                       [M_inv[0][0],M_inv[0][1]],
                       [M_inv[1][0],M_inv[1][1]]])

            x_m = x[:, np.newaxis] + self.Tp * (A @ x[:, np.newaxis] + B @ u)
            x_error = np.abs(x_m[2:4, 0] - x_dot)

            # Obliczamy błąd całkowity jako suma błędów w prędkościach dla obu zmiennych stanu
            total_error = np.sum(x_error)

            # Jeśli bieżący błąd jest mniejszy niż najlepszy do tej pory
            if total_error < best_error:
                # Zapisujemy indeks tego modelu jako najlepszego
                best_model_index = i
                best_error = total_error

        # Ustawiamy indeks najlepszego modelu
        self.i = best_model_index

    def calculate_control(self, x, q_r, q_r_dot, q_r_ddot):
        q = x[:2]
        q_dot = x[2:]

        if self.xd > 0:
            self.choose_model(self.x, self.u, q_dot)

        self.x_dot = q_dot

        K_d = 0.5
        K_p = 1

        v = q_r_ddot + (K_d*q_r_dot - K_d*q_dot) + (K_p*q_r - K_p*q)
        M = self.models[self.i].M(x)
        C = self.models[self.i].C(x)
        u = M @ v[:, np.newaxis] + C @ q_dot[:, np.newaxis]

        self.u = u
        self.x = x
        self.x_dot = q_dot
        self.xd = 1

        return u
