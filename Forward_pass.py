import sigmoid
import numpy as np

num_input=int(input())
layer_size=[num_input,5,5,5,1]

l=[]
for _ in range(num_input):
    l.append(float(input()))
    
arr=np.array(l)
input_arr=arr.reshape(1,num_input)

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








