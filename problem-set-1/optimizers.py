"""
Optimizer for NeuralNetwork class
"""
import numpy as np

class Optimizer:
    def __init__(self, gradient):
        self.gradient = gradient
        return

    def optimize(self, new_grad):
        # print(self.gradient)
        # print(new_grad)
        # print(self.gradient.shape)
        # print(new_grad.shape)
        self.gradient[:, 0:new_grad.shape[1]] = new_grad
        return self.gradient[:, 0:new_grad.shape[1]]

class GradientDescentWithMomentum(Optimizer):
    def __init__(self, gradient, beta):
        super().__init__(gradient)
        self.beta = beta

    def optimize(self, new_grad):
        # trim size for the last batch
        self.gradient[:, 0:new_grad.shape[1]] = self.beta * self.gradient[:, 0:new_grad.shape[1]] + (1-self.beta) * new_grad
        return self.gradient[:, 0:new_grad.shape[1]]

def main():
    pass
if __name__=="__main__":
    main()