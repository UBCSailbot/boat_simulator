from abc import ABC, abstractmethod
import numpy as np


class ReferenceFrame(ABC):
    def __init__(self, origin=np.array([0, 0, 0]), axes={}, mass=1):
        self.origin = origin
        self.axes = axes
        self.mass = mass
        self.cartesian_position = np.array([0, 0, 0])
        self.cartesian_velocity = np.array([0, 0, 0])
        self.cartesian_acceleration = np.array([0, 0, 0])
        self.angular_position = np.array([0, 0, 0])  # Measured in rads
        self.angular_velocity = np.array([0, 0, 0])  # Measured in rads/s
        self.angular_acceleration = np.array([0, 0, 0])  # Measured in rads/s^2
        self.forces = []
        self.net_force = np.array([0, 0, 0])
        self.torques = []
        self.net_torque = np.array([0, 0, 0])

    def add_force(self, force):
        self.forces.append(force)

    def add_torque(self, torque):
        self.torques.append(torque)

    def compute_net_force(self):
        self.net_force = np.sum(self.forces, axis=0)

    def compute_net_torque(self):
        self.net_torque = np.sum(self.torques, axis=0)

    def update_kinematics(self, moment_of_inertia):
        self.cartesian_acceleration = self.net_force / self.mass
        self.cartesian_velocity += self.cartesian_acceleration
        self.cartesian_position += self.cartesian_velocity

        angular_acceleration = np.linalg.inv(moment_of_inertia) @ self.net_torque  # I * alpha = net_torque
        self.angular_velocity += angular_acceleration
        self.angular_position += self.angular_velocity

    @abstractmethod
    def convert_to(self, target_frame):
        pass
