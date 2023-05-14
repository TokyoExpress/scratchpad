# 1.1 Backpropagation

import numpy as np

class Sigmoid:
    def forward(self, x):
        self.out = 1 / (1 + np.exp(-x))
        return 1 / (1 + np.exp(-x))
    
    def backward(self, grad):
        # dx = σ(x) * (1 - σ(x))
        return grad * self.out * (np.ones(self.out.shape) - self.out)

class ReLU:
    def forward(self, x):
        self.out = np.maximum(0, x)
        return self.out
    
    def backward(self, grad):
        # dx = 1 if x > 0, 0 if x < 0
        # In this implementation, derivative is 0 if x == 0, in reality it is indifferentiable
        return grad * np.where(self.out > 0, 1, 0)

class Linear:
    def __init__(self, input_dim, output_dim):
        self.w = np.random.normal(0, 1, [input_dim, output_dim])
        self.b = np.random.normal(0, 1, [output_dim])

    def forward(self, x):
        self.x = x
        x = x @ self.w
        x = x + self.b
        return x

    def backward(self, grad):
        
        dL_dy = grad
        dL_dw = self.x.T @ dL_dy
        dL_db = np.sum(dL_dy)
        dL_dx = dL_dy @ self.w.T

        # gradient descent
        self.w = self.w - dL_dw * 0.01
        self.b = self.b - dL_db * 0.01

        return dL_dx

class Model:
    def __init__(self, layers):
        self.layers = layers

    def forward(self, X):
        for layer in self.layers:
          X = layer.forward(X)
        self.out = X
        return X

    def backward(self, y):
        
        # MSE loss = 0.5 * (yhat - y).T @ (yhat - y)
        # d(MSE) = y - yhat

        loss = 0.5 * (y - self.out).T @ (y - self.out)
        dL_dy = self.out - y

        grad = dL_dy

        for layer in reversed(range(len(self.layers))):
          grad = self.layers[layer].backward(grad)
        
        return loss.item()