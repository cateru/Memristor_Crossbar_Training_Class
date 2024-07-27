import numpy as np
from experimental_conductances import conductance_data
import Memristor_Crossbar

# Define training dataset
# 
# A 2D NumPy array where each row represents a training sample.
# Each sample is a binary vector used for training the model.
train_set = np.array([[0, 0, 0, 1],
                      [0, 0, 1, 0],
                      [0, 1, 1, 0],
                      [1, 0, 0, 1],
                      [1, 1, 0, 1],
                      [1, 1, 1, 0]])

# Define testing dataset
# 
# A 2D NumPy array where each row represents a test sample.
# Each sample is a binary vector used for evaluating the model's performance.
test_set = np.array([[0, 0, 0, 0],
                     [0, 0, 1, 1],
                     [0, 1, 0, 0],
                     [0, 1, 0, 1],
                     [0, 1, 1, 1],
                     [1, 0, 0, 0],
                     [1, 0, 1, 0],
                     [1, 0, 1, 1],
                     [1, 1, 0, 0],
                     [1, 1, 1, 1]])

# Define training outputs
# 
# A 2D NumPy array where each row corresponds to the output labels for the training samples.
# Each label is a binary vector representing the expected output for a given training sample.
train_outputs = np.array([[0, 1],
                          [1, 0],
                          [1, 0],
                          [0, 1],
                          [0, 1],
                          [1, 0]])

# Define testing outputs
# 
# A 2D NumPy array where each row corresponds to the output labels for the test samples.
# Each label is a binary vector representing the expected output for a given test sample.
test_outputs = np.array([[0, 0],
                         [0, 0],
                         [1, 0],
                         [0, 0],
                         [1, 0],
                         [0, 1],
                         [0, 0],
                         [0, 1],
                         [0, 0],
                         [0, 0]])

# Initialize Memristor Crossbar model
# 
# Creates an instance of the Memristor_Crossbar with specified parameters:
# - `beta`: a parameter affecting the model's behavior (20000 in this case).
# - `positive_target`: target value for positive output (0.75).
# - `negative_target`: target value for negative output (-0.75).
# - `range`: the range for the conductance values (0.001).
# - `multiplication_factor`: a factor for multiplication in the model (10).
model = Memristor_Crossbar(beta = 20000, positive_target = 0.75, negative_target = -0.75, range = 0.001, multiplication_factor = 10)

# Train the model
# 
# Fits the Memristor_Crossbar model to the training data. This involves training the model using
# the provided training samples (`train_set`) and their corresponding outputs (`train_outputs`).
# The `conductance_data` is used in the training process.
model.fit(train_set, train_outputs, conductance_data)

# Predict using the model
# 
# Uses the trained Memristor_Crossbar model to make predictions on the testing data.
# The predictions are compared against the provided outputs (`test_outputs`).
model.predict(test_set, test_outputs)