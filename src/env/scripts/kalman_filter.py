# kalman_filter.py

import numpy as np

class KalmanFilter:
    def __init__(self, A, B, H, Q, R, X0, P0):
        self.A = A  # 狀態轉移矩陣
        self.B = B  # 控制輸入矩陣
        self.H = H  # 觀測矩陣
        self.Q = Q  # 過程噪聲協方差
        self.R = R  # 觀測噪聲協方差
        self.X = X0  # 初始狀態估計
        self.P = P0  # 初始估計誤差協方差

    def predict(self, U=None):
        if U is None:
            U = np.zeros((self.B.shape[1], 1))
        self.X = np.dot(self.A, self.X) + np.dot(self.B, U)
        self.P = np.dot(np.dot(self.A, self.P), self.A.T) + self.Q
        return self.X

    def update(self, Z):
        Y = Z - np.dot(self.H, self.X)
        S = np.dot(np.dot(self.H, self.P), self.H.T) + self.R
        K = np.dot(np.dot(self.P, self.H.T), np.linalg.inv(S))
        self.X = self.X + np.dot(K, Y)
        I = np.eye(self.X.shape[0])
        self.P = np.dot((I - np.dot(K, self.H)), self.P)
        return self.X, self.P

def kalman_filter(dt, nx, initial_state, initial_P, Q_scale, R_scale):
    # 計算系數
    A = np.array([
        [1, 0, 0, dt, 0, 0, 0.5*dt**2, 0, 0],
        [0, 1, 0, 0, dt, 0, 0, 0.5*dt**2, 0],
        [0, 0, 1, 0, 0, dt, 0, 0, 0.5*dt**2],
        [0, 0, 0, 1, 0, 0, dt, 0, 0],
        [0, 0, 0, 0, 1, 0, 0, dt, 0],
        [0, 0, 0, 0, 0, 1, 0, 0, dt],
        [0, 0, 0, 0, 0, 0, 1, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 1, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 1]
    ])
    B = np.zeros((nx, 3))
    H = np.eye(9)  # 現在我們觀測所有9個狀態

    # 初始狀態估計
    X0 = np.array(initial_state).reshape(nx, 1)

    # 初始狀態不確定度
    P0 = np.diag(initial_P)

    # 狀態遞推噪聲協方差
    Q = np.diag(Q_scale)

    # 觀測噪聲協方差
    R = np.diag(R_scale)

    return KalmanFilter(A, B, H, Q, R, X0, P0)