import numpy as np

class KF:
    """
    State (8x1):
        [x, y, w, h, vx, vy, vw, vh]^T

    Measurement (4x1):
        [x, y, w, h]^T
    """

    def __init__(self, x0, P0, Q, R, A, H):

        self.x = x0  

        self.P = P0  

        self.Q = Q    
        self.R = R    

        self.A = A   
        self.H = H    

        self.x_pred = None
        self.P_pred = None

    # PREDICT
    def predict(self):

        self.x_pred = self.A @ self.x

        self.P_pred = self.A @ self.P @ self.A.T + self.Q

        return self.x_pred, self.P_pred

    # KALMAN GAIN
    def kalman_gain(self):

        S = self.H @ self.P_pred @ self.H.T + self.R

        K = self.P_pred @ self.H.T @ np.linalg.inv(S)
        
        return K

    # UPDATE
    def update(self, z):
        """
        z is a (4x1) measurement column vector: [x, y, w, h]^T
        """
        K = self.kalman_gain()

        innovation = z - (self.H @ self.x_pred)

        self.x = self.x_pred + (K @ innovation)

        self.P = self.P_pred - (K @ self.H @ self.P_pred)

        return self.x, self.P
    
    
    @staticmethod
    def init_from_two_detections(det1, det2, dt=1.0):
        """
        det = [x, y, w, h]
        returns x0 as (8x1)
        """
        x1, y1, w1, h1 = det1
        x2, y2, w2, h2 = det2

        vx = (x2 - x1) / dt
        vy = (y2 - y1) / dt
        vw = (w2 - w1) / dt
        vh = (h2 - h1) / dt

        x0 = np.array([[x2], [y2], [w2], [h2], [vx], [vy], [vw], [vh]], dtype=float)
        return x0

