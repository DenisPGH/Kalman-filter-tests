import math
import numpy as np

np.set_printoptions(precision=3, suppress=True)
A_k_minus_1 = np.array([[1.0]]) # Expresses how the state of the system [yaw] changes,identity matrix
process_noise_v_k_minus_1 = np.array([0.001]) # noice by comands
Q_k = np.array([[10]]) #State model noise covariance matrix Q_k,   big believe in measurment
H_k = np.array([[1.0]]) # predicted state estimate at time k
R_k = np.array([[1.0]]) # Sensor measurement noise covariance matrix R_k
sensor_noise_w_k = np.array([0.001])


def getB(delta_theta):
    B = np.array([[math.radians(delta_theta)]])
    return B


def ekf(z_k_observation_vector, state_estimate_k_minus_1,
        control_vector_k_minus_1, P_k_minus_1):
    state_estimate_k = A_k_minus_1 @ (state_estimate_k_minus_1) + (getB(state_estimate_k_minus_1[0])) @ (
                           control_vector_k_minus_1) + (process_noise_v_k_minus_1)
    print(f'Before EKF={math.degrees(state_estimate_k):.3f}')
    P_k = A_k_minus_1 @ P_k_minus_1 @ A_k_minus_1.T + (Q_k)
    measurement_residual_y_k = z_k_observation_vector - ((H_k @ state_estimate_k) + (sensor_noise_w_k))
    print(f'Measurment={math.degrees(z_k_observation_vector):.3f}')
    # Calculate the measurement residual covariance
    S_k = H_k @ P_k @ H_k.T + R_k
    K_k = P_k @ H_k.T @ np.linalg.pinv(S_k)
    state_estimate_k = state_estimate_k + (K_k @ measurement_residual_y_k)
    P_k = P_k - (K_k @ H_k @ P_k)
    print(f'After EKF={math.degrees(state_estimate_k):.3f}')
    return state_estimate_k, P_k


def main():
    k = 1
    z_k = np.array([[0], [20]])
    state_estimate_k_minus_1 = np.array([math.radians(0)]) # [ radians] ,start position
    control_vector_k_minus_1 = np.array([math.radians(10)])  # [v, yaw_rate] [meters/second, radians/second]
    P_k_minus_1 = np.array([[ 1.0]]) # accuracy of the state estimate at time k
    for k, obs_vector_z_k in enumerate(z_k, start=1):
        print(f'========= {k} =============')
        optimal_state_estimate_k, covariance_estimate_k = ekf(
            math.radians(obs_vector_z_k),  # Most recent sensor measurement
            state_estimate_k_minus_1,  # Our most recent estimate of the state
            control_vector_k_minus_1,  # Our most recent control input
            P_k_minus_1,  # Our most recent state covariance matrix
            )
        #update matrices
        state_estimate_k_minus_1 = optimal_state_estimate_k
        P_k_minus_1 = covariance_estimate_k



if __name__=='__main__':
    main()