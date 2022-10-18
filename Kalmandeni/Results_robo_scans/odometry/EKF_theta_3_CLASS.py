import math
import numpy as np


class EKFDeniTheta:
    def __init__(self):
        self.A_k_minus_1 = np.array([[1.]])  # Expresses how the state of the system [yaw] changes,identity matrix
        self.process_noise_v_k_minus_1 = np.array([0.01])  # noice by comands, big=big predict error
        self.Q_k = np.array([[1.0]])  # 2.0 #State model noise covariance matrix Q_k,   big believe in measurment
        self.H_k = np.array([[1.0]])  # predicted state estimate at time k, 0.91= is more, 1= is less(estimate positiion)
        self.R_k = np.array([[0.0001]])  # Sensor measurement noise covariance matrix R_k
        self.sensor_noise_w_k = np.array([0.01])

        self.state_estimate_k_minus_1 = np.array([math.radians(0)])  # [ radians] ,start position
        self.control_vector_k_minus_1 = np.array([math.radians(10)])  # [v, yaw_rate] [meters/second, radians/second]
        self.P_k_minus_1 = np.array([[0.3]]) # accuracy of the state estimate at time k
        self.end_position=0
    def getB(self,delta_theta):
        B = np.array([[math.radians(delta_theta)]])
        return B

    def ekf(self,z_k_observation_vector, state_estimate_k_minus_1,
            control_vector_k_minus_1, P_k_minus_1):
        # predict step
        state_estimate_k = self.A_k_minus_1 @ (state_estimate_k_minus_1) + (self.getB(state_estimate_k_minus_1[0])) @ (
            control_vector_k_minus_1) + (self.process_noise_v_k_minus_1)
        #print(f'Before EKF={math.degrees(state_estimate_k):.3f}')
        P_k = self.A_k_minus_1 @ P_k_minus_1 @ self.A_k_minus_1.T + (self.Q_k)
        # update
        measurement_residual_y_k = z_k_observation_vector - ((self.H_k @ state_estimate_k) + (self.sensor_noise_w_k))
        #print(f'Measurment={math.degrees(z_k_observation_vector):.3f}')
        # Calculate the measurement residual covariance
        S_k = self.H_k @ P_k @ self.H_k.T + self.R_k
        K_k = P_k @ self.H_k.T @ np.linalg.pinv(S_k)
        state_estimate_k = state_estimate_k + (K_k @ measurement_residual_y_k)

        # if z_k_observation_vector  > :
        #     state_estimate_k=

        P_k = P_k - (K_k @ self.H_k @ P_k)
        after = math.degrees(state_estimate_k)
        # if after<0:
        #     after=360+after
        #print(f'After EKF={after:.3f}')
        self.end_position=after
        return state_estimate_k, P_k

    def calculate_error_turning(self,z_k):
        """
        get the measured angle and return the corection of the angle with EKF
        :param z_k: np.array[[theta in radians],]
        :return: final position after calculation
        """
        for obs_vector_z_k in z_k:
            print('Class ekf')
            optimal_state_estimate_k, covariance_estimate_k = self.ekf(math.radians(obs_vector_z_k),
                                                                  self.state_estimate_k_minus_1, self.control_vector_k_minus_1,
                                                                  self.P_k_minus_1, )
            # update matrices
            self.state_estimate_k_minus_1 = optimal_state_estimate_k
            self.P_k_minus_1 = covariance_estimate_k

        return self.end_position
