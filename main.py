import numpy as np
import Forward_pass

num_input=int(input())
layer_size=[num_input,5,5,5,1]

l=[]
for _ in range(num_input):
    l.append(float(input()))
    
arr=np.array(l)
input_arr=arr.reshape(1,num_input)

Forward_pass.Forpass(layer_size,input_arr)

