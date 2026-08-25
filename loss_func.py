import numpy as np

def mse_loss(predicted,actual):
    return ((predicted-actual)**2)

def mse_loss_derivative(predicted,actual):
    return 2*(predicted-actual)