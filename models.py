import numpy as np
from scipy.optimize import minimize

class LogisticRegression:
    def __init__(self, lambda_reg=1e-3):
        self.lambda_reg = lambda_reg
        self.w = None
        
    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))
    
    def loss(self, w, X, y):
        """Regularized logistic loss."""
        n = len(y)
        logits = X @ w
        loss = np.mean(np.log(1 + np.exp(-y * logits)))
        reg = 0.5 * self.lambda_reg * np.sum(w**2)
        return loss + reg
    
    def gradient(self, w, X, y):
        """Gradient of logistic loss (fixed shape mismatch)."""
        n = len(y)
        logits = X @ w
        # Reshape y to (n, 1) to broadcast with X (n, d)
        y_reshaped = y.reshape(-1, 1)
        sigmoid_term = 1 - self.sigmoid(y_reshaped * logits.reshape(-1, 1))
        grad = -np.mean(y_reshaped * X * sigmoid_term, axis=0)
        reg_grad = self.lambda_reg * w
        return grad + reg_grad
    
    def hessian(self, w, X):
        """Hessian of logistic loss (for Newton)."""
        n = len(X)
        logits = X @ w
        s = self.sigmoid(logits) * (1 - self.sigmoid(logits))
        # Reshape s to (n, 1) to broadcast with X (n, d)
        s_reshaped = s.reshape(-1, 1)
        hess = (X.T @ (s_reshaped * X)) / n
        reg_hess = self.lambda_reg * np.eye(len(w))
        return hess + reg_hess
    
    def fit(self, X, y):
        """Train model with L-BFGS (baseline)."""
        w0 = np.zeros(X.shape[1])
        result = minimize(
            fun=self.loss,
            x0=w0,
            args=(X, y),
            jac=self.gradient,
            method='L-BFGS-B',
            options={'maxiter': 1000}
        )
        self.w = result.x
        return self
    
    def predict(self, X):
        return np.sign(X @ self.w)
    
    def accuracy(self, X, y):
        return np.mean(self.predict(X) == y)