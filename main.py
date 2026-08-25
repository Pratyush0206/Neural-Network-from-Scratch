import numpy as np
import Forward_pass
import Backpropogation
import loss_func

num_input=int(input())
layer_size=[num_input,5,5,5,1]
y=1.0  #Just for example as of now

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

for epoch in range(1000):
    a_output, inputs_list, z_list = Forward_pass.Forpass(layer_size, input_arr, weights, bias)
    dW, dB = Backpropogation.backprop(a_output, weights, bias, inputs_list, z_list, y)
    weights, bias = Backpropogation.updation(dW, dB, weights, bias)

    if epoch % 100 == 0:
        loss = loss_func.mse_loss(a_output, y)
        print(f"Epoch {epoch}, loss: {loss}, prediction: {a_output}")

