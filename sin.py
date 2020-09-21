
#COMP30230 Programming Assignment

from numpy import *
import numpy as np
from mlp import nn

#mlp trains a neural network to recognise the XoR model
input_layer = 4
output_layer = 1
hidden_layer = 10
learning_rate = 0.2
epochs_num = 1000
np.random.seed(3)
input_vectors = 50
#weight scaling
momentum = 0.9

inputs = 2*np.random.random((input_vectors,4)) -1
targets = []

for index in inputs

summation  = index[0] - index[1]+ index[2]- index[3]
targets.append([math.sin(summation)])

targets = np.array(targets)
#split the input to 40 for train and 10 for testing
training_data_range = (input_vectors / 5) * 4
testing_data_range = input_vectors / 5

#data to train
training_set = inputs [0:training_data_range, :]
training_data_output = targets[0: training_data_range, :]


#data to test
testing_set = inputs[training_data_range:input_vectors]
testing_outputs = targets[training_data_range: input_vectors]

neural_network = nn(training_set, training_data_output, hidden_layer, momentum)
neural_network.randomize()

for each in range(epochs_num)
neural_network.forward(neural_network,inputs)
error = neural_network.backwards()
neural_network.update_weights(learning_rate)

if mod(each,50) == 0:
print("error at epoch ", num, ": ", error)

print("learning_rate:", learning_rate, "momentum:", momentum)
print("epoch_num:", epochs_num, "hidden units:",hidden_layer)

#generates the test data
input_vectors = shape(testing_set)[0]
test_input = concatenate((testing_set, -ones((input_vectors,1))), axis = 1)
predicted_output = neural_network.forward(test_input)
error = 0.5 * sum((predicted_output - testing_outputs) ** 2)
print()
print("Error rate: ", error)