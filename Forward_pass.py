import numpy as np
import sigmoid

def Forpass(layer_size,input_arr):
    weights=[]

    for i in range(len(layer_size)-1):
        w=np.random.uniform(-1,1 ,size=(layer_size[i],layer_size[i+1]))
        weights.append(w)

    bias=[]

    for i in range(len(layer_size)-1):
        b=np.random.uniform(-1,1,size=(layer_size[i+1],))
        bias.append(b)

    for j in range(len(layer_size)-1):
        z=input_arr @ weights[j] +bias[j]
        input_arr=sigmoid.sigmoid(z)

    print(input_arr)








