
#COMP30230 Programming Assignment

from numpy import *
import numpy as np
from mlp import nn

#mlp trains a neural network to recognise the XoR model


#XOR inputs
inputs = np.array ([[0,0],[0,1],[1,0],[1,1]])
#results of the XOR
targets = np.array([[0],[1],[1],[0]])
#weight scaling
momentum = 0.9

neural_network = nn(inputs, targets hidden_layer, momentum)

neural_network.randomize()

#loops through each epoch going forward and backward through the neural 
network printing out the error
for each in range(epoch_num)
	neural_network.forward(neural_network.input)
	error = neural_network.backwards()
	neural_network.update_weights(learning_rate)
	if mod(each,100) == 0
		print("Error at  Epoch: ", num, ": ",error )

print("learning_rate:", learning_rate, "momentum:", momentum)
print("epoch_num:", epochs_num, "hidden units:",hidden_layer)

test_input = concatenate ((inputs, -ones((input_vectors, 1))), axis =1)

test_output = neural_network.forward(test_input)

# prints the results
print()
print("output: ")
for line in test_output:
print(line)