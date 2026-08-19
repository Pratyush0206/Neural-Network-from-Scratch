import sigmoid
import numpy as np

num_input=int(input())
num_neuron=int(input())

l=[]
for _ in range(num_input):
    l.append(float(input()))

arr=np.array(l)
input_arr=arr.reshape(1,num_input)

weight_list=np.random.uniform(-1,1 ,size=(num_input,num_neuron))

bias=np.random.uniform(-1,1,size=num_neuron)

z=input_arr @ weight_list +bias

print(z.shape)

print(sigmoid.sigmoid(z))








