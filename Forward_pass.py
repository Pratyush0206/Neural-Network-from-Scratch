import numpy as np
import sigmoid

def Forpass(layer_size,input_arr,weights,bias):
    inputs_list = []
    z_list = []

    for j in range(len(layer_size)-1):
        inputs_list.append(input_arr)                    # save what went IN
        z = input_arr @ weights[j] + bias[j]
        z_list.append(z)                                  # save pre-activation
        input_arr = sigmoid.sigmoid(z)                    # activate

    a_output = input_arr   

    return a_output,inputs_list,z_list










