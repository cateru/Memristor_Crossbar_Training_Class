import numpy as np
import logging
from experimental_conductances import load_conductance_data
from Memristor_Crossbar import Memristor_Crossbar

def setup_logging(level=logging.INFO):
    """
    Configures the logging module for the application.

    Args:
        level (int): The logging level, set to `logging.INFO` by default.
                     This means that all log messages at this level and above 
                     (i.e., INFO, WARNING, ERROR, and CRITICAL) will be output.

    Example:
        >>> setup_logging()
    """
    logging.basicConfig(level=level, format='%(asctime)s - %(levelname)s - %(message)s')

def define_training_data():
    """
    Defines the training dataset.

    Returns:
        np.ndarray: A 2D NumPy array where each row represents a training sample.
    """
    return np.array([[0, 0, 0, 1],
                     [0, 0, 1, 0],
                     [0, 1, 1, 0],
                     [1, 0, 0, 1],
                     [1, 1, 0, 1],
                     [1, 1, 1, 0]])

def define_testing_data():
    """
    Defines the testing dataset.

    Returns:
        np.ndarray: A 2D NumPy array where each row represents a test sample.
    """
    return np.array([[0, 0, 0, 0],
                     [0, 0, 1, 1],
                     [0, 1, 0, 0],
                     [0, 1, 0, 1],
                     [0, 1, 1, 1],
                     [1, 0, 0, 0],
                     [1, 0, 1, 0],
                     [1, 0, 1, 1],
                     [1, 1, 0, 0],
                     [1, 1, 1, 1]])

def define_training_outputs():
    """
    Defines the training outputs.

    Returns:
        np.ndarray: A 2D NumPy array where each row corresponds to the output labels for the training samples.
    """
    return np.array([[0, 1],
                     [1, 0],
                     [1, 0],
                     [0, 1],
                     [0, 1],
                     [1, 0]])

def define_testing_outputs():
    """
    Defines the testing outputs.

    Returns:
        np.ndarray: A 2D NumPy array where each row corresponds to the output labels for the test samples.
    """
    return np.array([[0, 0],
                     [0, 0],
                     [1, 0],
                     [0, 0],
                     [1, 0],
                     [0, 1],
                     [0, 0],
                     [0, 1],
                     [0, 0],
                     [0, 0]])


if __name__ == "__main__":
    # Load conductance data
    conductance_data = load_conductance_data('datafile.csv')

    # Initialize Memristor Crossbar model
    model = Memristor_Crossbar(beta=20000, positive_target=0.8, negative_target=-0.8, multiplication_factor=10)

    # Train the model
    epoch = model.fit(define_training_data(), define_training_outputs(), conductance_data)
    converged = (epoch < model.epochs)
    model.visualize_graphs(epoch, define_training_data(), define_training_outputs(), converged)

    # Predict using the model
    model.predict(define_testing_data(), define_testing_outputs())

    # Iterative training
    logging.disable(logging.CRITICAL)
    for i in range(100):
        model.fit(define_training_data(), define_training_outputs(), conductance_data, save_data=True, filename=f"test_10_{i}")