
#COMP30230 Programming Assignment

from numpy import *
import numpy as np
from mlp import nn

input_layer = 16
output_layer = 35
hidden_layer = 26
learning_rate = 0.15
epochs_num = 10000
np.random.seed(3)
input_vectors = 20000
momentum = 0.5

input_file = "letter-recognition.data.txt"
inputs = []
targets = []
letters = []

with open input_file as file:
	for line in file:
		letters.append([str(num) for num in line.strip().split(',')])
for each in letters:
	try:
		alphabet = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
		letter_to_int = dict((char,int) for int, char in enumerate(alphabet))
		number_letter_pairs = [letter_to_int[each[0]]]
		array = []
		for value in number_letter_pairs:
			letter = [0 for _ in range(len(alphabet))]
			letter[value] = 1
			array = letter
		targets.append(array)
		inputs.append([int(each[1]), int(each[2]), int(each[3]), int(each[4]), int(each[5]), int(each[6]),
                       int(each[7]), int(each[8]), int(each[9]), int(each[10]), int(each[11]), int(each[12]),
                       int(each[13]), int(each[14]), int(each[15]), int(each[16])])
	except IndexError:
		print("file error")

targets = np.array(targets)
training_data_range = int((input_vectors / 5) * 4)
testing_data_range = input_vectors / 5
training_set = inputs[:training_data_range]
training_set_output = targets[:training_data_range]

testing_set = inputs[training_data_range:input_vectors]
testing_set_output = targets[training_data_range:input_vectors]

neural_network = nn(training_set,training_set_output,input_layer,momentum)
neural_network.randomise()

for each in range(epochs_num)
	neural_network.forward(mlp.I)
	error = neural_network.backwards()
	neural_network.update_weights(learning_rate)
	
	if mod(each,100) == 0:
		print(""Error at epoch: ", each, ": ", error")
print("learning rate:", learning_rate, "Momentum:",momentum)
print("Epochs:",epochs_num, "Hidden units:", hidden_layer)

input_vectors = shape(testing_set)[0]
test_input = concatenate((testing_set,-ones((input_vectors,1))), axis=1)
predicted_output = neural_network.forward(test_input)
error = 0.5 * sum((predicted_output - testing_set_output) ** 2)

print()
print("Test set error: ", error)