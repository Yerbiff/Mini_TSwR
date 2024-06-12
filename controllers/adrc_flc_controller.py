import numpy as np

# from models.free_model import FreeModel
from observers.eso import ESO
from .adrc_joint_controller import ADRCJointController
from .controller import Controller
# from models.ideal_model import IdealModel
from models.manipulator_model import ManipulatorModel

class ADRFLController(Controller):
    def __init__(self, Tp, q0, Kp, Kd, p):
        self.model = ManipulatorModel(Tp, 0.1, 0.05)
        self.Kp = np.array(Kp)
        self.Kd = np.array(Kd)
        self.L = np.array([[3 * p[0], 0],
                           [0, 3 * p[1]],
                           [3 * p[0] ** 2, 0],
                           [0, 3 * p[1] ** 2],
                           [p[0] ** 3, 0],
                           [0, p[1] ** 3]])

        self.A = self._initialize_A()
        self.B = self._initialize_B()
        self.W = self._initialize_W()

        self.eso = ESO(self.A, self.B, self.W, self.L, q0, Tp)
        self.update_params(q0[:2], q0[2:])

    def _initialize_A(self):
        A = np.zeros((6, 6))
        A[:4, 2:] = np.eye(4)
        return A

    def _initialize_B(self):
        return np.zeros((6, 2))

    def _initialize_W(self):
        W = np.zeros((2, 6))
        W[:2, :2] = np.eye(2)
        return W

    def update_params(self, q, q_dot):
        """
        Aktualizacja parametrów ESO
        """
        x = np.concatenate([q, q_dot])
        M_inv = np.linalg.inv(self.model.M(x))
        self.A[2:4, 2:4] = -M_inv @ self.model.C(x)
        self.B[2:4, :] = M_inv

        self.eso.A = self.A
        self.eso.B = self.B

    def calculate_control(self, x, q_d, q_d_dot, q_d_ddot):
        """
        Implementacja ADRFLC
        """
        q1, q2, q1_dot, q2_dot = x

        z_h = self.eso.get_state()

        q_h = z_h[:2]
        q_dot_h = z_h[2:4]

        f_h = z_h[4:6]

        q_error = np.array([q1, q2]) - q_h
        q_dot_error = np.array([q1_dot, q2_dot]) - q_dot_h

        Kp_error = np.dot(self.Kp, q_error)
        Kd_error = np.dot(self.Kd, q_dot_error)

        # Oblicz sterowanie
        v = q_d_ddot + Kd_error + Kp_error
        u = np.dot(self.model.M(z_h[:4]), (v - f_h)) + np.dot(self.model.C(z_h[:4]), q_dot_h)

        # Aktualizuj parametry w oparciu o bieżący stan
        self.update_params(q_h, q_dot_h)

        # Zaktualizuj ESO o aktualną pozycję i obliczone wejście sterujące u
        self.eso.update(np.array([[q1], [q2]]), u[:, np.newaxis])

        return u
