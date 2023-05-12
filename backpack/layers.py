import numpy as np

class Sigmoid:
    def forward(self, x):
        self.out = 1 / (1 + np.exp(-x))
        return self.out

    def backward(self, grad):
        # dx = σ(x) * (1 - σ(x))
        return grad * Sigmoid(self.out) * (np.ones(Sigmoid(self.out).shape) - Sigmoid(self.out))

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
        # We need these two derivatives to update the parameters:
        # dL / dW = (dL / d(Wx+b)) * (d(Wx+b) / dW) = x^T @ (dL / d(Wx+b))
        # dL / db = (dL / d(Wx+b)) * (d(Wx+b) / db) = dL / d(Wx+b)
        # and because x is the output of the previous layers, we need to pass backward:
        # dL / dx = (dL / d(Wx+b)) * (d(Wx+b) / dx) = (dL / d(Wx+b)) @ W^T

        dL_dwxb = grad
        dL_dw = self.x.T @ dL_dwxb
        dL_dx = dL_dwxb @ self.w.T

        # here we have one gradient for each item in the batch, so we can average them to update the bias
        dL_db = np.mean(dL_dwxb)

        self.w = self.w - dL_dw * 0.1
        self.b = self.b - dL_db * 0.1

        return dL_dx

class Model:
    def __init__(self, *layers):
        self.layers = layers

    def forward(self, x):
        for layer in self.layers:
          x = layer.forward(x)
        self.out = x
        return x

    def backward(self, y):
        
        # MSE loss = 0.5 * (self.out - y).T @ (self.out - y)
        dL_dy = self.out - y
        grad = dL_dy

        for layer in reversed(range(len(self.layers))):
          grad = self.layers[layer].backward(grad)