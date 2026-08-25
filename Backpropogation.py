import numpy as np
import sigmoid
import loss_func
import random

def backprop(a_output,weights,bias,inputs_list,z_list,y):

    dW=[None]*len(weights)
    dB=[None]*len(bias)
    delta=[None]*len(weights)

    L=len(weights)-1

    delta[L] = loss_func.mse_loss_derivative(a_output, y) * sigmoid.sigmoid_derivative(z_list[L])
    dW[L] = inputs_list[L].T @ delta[L]
    dB[L] = delta[L]

    for i in reversed(range(L)):
        delta[i] = (delta[i+1] @ weights[i+1].T) * sigmoid.sigmoid_derivative(z_list[i])
        dW[i] = inputs_list[i].T @ delta[i]
        dB[i] = delta[i]

    return dW, dB 

def updation(dW,dB,weights,bias):
    L=len(weights)
    for i in range(L):
        weights[i]=weights[i]-0.1*dW[i]
        bias[i]=bias[i]-0.1*dB[i]

    return weights,bias