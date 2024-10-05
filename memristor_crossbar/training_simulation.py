import numpy as np
import logging
from experimental_conductances import conductance_data
from Memristor_Crossbar import Memristor_Crossbar

# Configures the logging module for the application.
# This setup defines the log level and format for log messages.
# 
# Args:
#     level (int): The logging level, set to `logging.INFO` in this case.
#                  This means that all log messages at this level and above 
#                  (i.e., INFO, WARNING, ERROR, and CRITICAL) will be output.
#     format (str): A string that specifies the format of the log messages.
#                   In this case, the format includes:
#                   - `%(asctime)s`: The timestamp when the log message is created.
#                   - `%(levelname)s`: The severity level of the log message (e.g., INFO, WARNING).
#                   - `%(message)s`: The actual log message.
# 
# Example:
#     logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
#     # This will log messages like:
#     # 2024-10-05 12:34:56,789 - INFO - This is an info message.
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


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
# - `positive_target`: target value for positive output (0.8).
# - `negative_target`: target value for negative output (-0.8).
# - `range`: the range for the conductance values (0.0001).
# - `multiplication_factor`: a factor for multiplication in the model (10).
model = Memristor_Crossbar(beta = 20000, positive_target = 0.8, negative_target = -0.8, range = 0.0001, multiplication_factor = 10)

# Train the model
# 
# Fits the Memristor_Crossbar model to the training data. This involves training the model using
# the provided training samples (`train_set`) and their corresponding outputs (`train_outputs`).
# The `conductance_data` is used in the training process.# 
# During training, key events are logged.# 
# After fitting the model, the results are plotted.
# The `plot_results` method visualizes the results of the training, using the epoch number,
# training samples, and their corresponding outputs.
epoch = model.fit(train_set, train_outputs, conductance_data)
converged = (epoch < model.epochs)
model.visualize_graphs(epoch, train_set, train_outputs, converged)


# Predict using the model
# 
# Uses the trained Memristor_Crossbar model to make predictions on the testing data.
# The predictions are compared against the provided outputs (`test_outputs`).
model.predict(test_set, test_outputs)

# Iterative training
# The model is trained multiple times (100 iterations) to refine its performance.
# Each training run fits the model using the training data and saves the results to files.
# 
# - `save_data`: Boolean indicating whether to save the training data (set to True here).
# - `filename`: Template for naming the output files (here, it includes the iteration number).
# 
# Key events during training are logged but the general ones are disabled.
logging.disable(logging.CRITICAL)
for i in range(100):
    model.fit(train_set, train_outputs, conductance_data, save_data = True, filename = f"test_10_{i}")