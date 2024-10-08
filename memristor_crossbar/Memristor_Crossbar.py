import os
import csv
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from dataclasses import dataclass
from matplotlib.cm import get_cmap
from datetime import datetime
import logging


@dataclass
class Memristor_Crossbar:

    beta: float
    positive_target: float
    negative_target: float
    range: float
    multiplication_factor: int
    training_set_width: int = 6
    epochs: int = 48
    number_of_neurons: int = 2
    number_of_rows: int = 4
    number_of_columns: int = 4

    test_set_width: int = 16 - training_set_width
    conductance_data: np.ndarray = None
    shifts: np.ndarray = None
    logic_currents: np.ndarray = None
    all_delta_ij: np.ndarray = None
    conductances: np.ndarray = None
    all_conductances: np.ndarray = None
    saved_correct_conductances: np.ndarray = None
    errors: np.ndarray = None
    all_errors: np.ndarray = None
    result: np.ndarray = None
    predictions: np.ndarray = None

    def __post_init__(self):

        self.conductance_data = []
        self.shifts = np.empty([self.number_of_rows, self.number_of_columns])
        self.logic_currents = np.empty([self.number_of_neurons])
        self.all_delta_ij = np.empty([self.training_set_width, self.number_of_neurons, self.number_of_rows])
        self.conductances = np.empty([self.number_of_neurons, self.number_of_rows, self.number_of_columns])
        self.all_conductances = np.empty([self.epochs, self.number_of_rows, self.number_of_columns])
        self.saved_correct_conductances = np.empty([self.number_of_rows, self.number_of_columns])
        self.errors = np.empty([self.training_set_width, self.number_of_neurons])
        self.all_errors = np.empty([self.epochs])
        self.result = np.empty([self.epochs, self.number_of_neurons, self.training_set_width])
        self.predictions = np.empty([self.test_set_width, self.number_of_neurons])

    def experimental_data(self, conductance_data: np.ndarray):
        """
        Initializes the conductance data by normalizing it to the first value.

        Args:
            conductance_data (np.ndarray): The raw set of conductance data to be processed.
                Example: np.array([5.0, 6.0, 7.0, 8.0])

        Returns:
            None

        Sets:
            self.conductance_data (np.ndarray): The normalized conductance data,
                where the first value is subtracted from each element.
                Example: if conductance_data = np.array([5.0, 6.0, 7.0, 8.0]),
                then self.conductance_data = np.array([0.0, 1.0, 2.0, 3.0])
        """
        raw_conductance_data = conductance_data
        first_value = raw_conductance_data[0]
        self.conductance_data = raw_conductance_data - first_value

    def shift_exp(self) -> None:
        """
        Generates random shifts for the conductance values based on an exponential distribution
        and reshapes them into a 4x4 array.

        Returns:
            None

        Sets:
            self.shifts (np.ndarray): A 4x4 array of random shifts generated using the exponential
                                    distribution, centered around 0.
        """
        center_value = 0
        num_elements = 16
        lambda_param = 55845
        rnd_shifts = np.random.exponential(scale=1 / lambda_param, size=num_elements)
        rnd_shifts -= np.mean(rnd_shifts) - center_value
        self.shifts = np.reshape(rnd_shifts, (4, 4))

    def custom_shift(self, custom_shifts: np.ndarray) -> None:
        """
        Sets custom shifts for the conductance values based on a user-defined 4x4 array.

        Args:
            custom_shifts (np.ndarray): A 4x4 NumPy array containing user-defined shifts for the conductance values.

        Returns:
            None

        Sets:
            self.shifts (np.ndarray): The shifts attribute is updated with the provided custom shifts.
        """
        self.shifts = custom_shifts

    def conductance_init(self) -> None:
        """
        Initializes the conductance values with random shifts and a multiplication factor.

        This method computes the initial conductance values by applying random shifts to the 
        normalized conductance data and multiplying by a specified multiplication factor. 
        It also logs the initialized conductance values and the epoch number.

        Args:
            stamp (bool): If True, prints the initial conductance values and the epoch number.
                        This is useful for debugging or tracking the initialization process.
                        Default is True.

        Returns:
            None

        Sets:
            self.conductances (np.ndarray): An array where the first element is the product
                of the normalized conductance data and shifts, multiplied by the multiplication factor.
            self.all_conductances (np.ndarray): Similar to self.conductances,
                storing the same product in the first element.

        Example: if self.conductance_data[0] = np.array([0.0, 1.0, 2.0, 3.0]),
            self.shifts = np.array([[ 0.1, -0.5,  0.4, -0.3], [ 0.2, -0.1,  0.5, -0.4], [ 0.3, -0.2,  0.6, -0.5], [ 0.4, -0.3,  0.7, -0.6]])
            and self.multiplication_factor = 2, then:
            self.conductances[0] = np.array([[ 0.2, -1.0,  0.8, -0.6], [ 0.4, -0.2,  1.0, -0.8], [ 0.6, -0.4,  1.2, -1.0], [ 0.8, -0.6,  1.4, -1.2]])
            self.conductances[1] = 0

        Logging:
            This method logs the initialized conductances and the epoch number.
            For example:
            - "Initial Conductances: [0.2 -1.0 0.8 -0.6 ...]"
            - "Epoch: 0"
        """
        self.conductances[0] = (
            self.conductance_data[0] + self.shifts
        ) * self.multiplication_factor
        self.conductances[1] = 0
        self.all_conductances[0] = (
            self.conductance_data[0] + self.shifts
        ) * self.multiplication_factor
        logging.info(f"Initial Conductances: {self.all_conductances[0]}")
        logging.info("Epoch: 0")

    def voltage_array(self, pattern: np.ndarray, V0=-0.1, V1=0.1) -> np.ndarray:
        """
        Generates an array of voltages based on a given pattern.

        Args:
            pattern (np.ndarray): The input pattern determining the voltage values.
            V0 (float, optional): Voltage for pattern value 0. Defaults to -0.1.
            V1 (float, optional): Voltage for pattern value 1. Defaults to 0.1.

        Returns:
            np.ndarray: An array of voltages corresponding to the pattern.

        Example:
            Given pattern = np.array([0, 1, 0, 1]) and default V0 = -0.1 and V1 = 0.1,
            the resulting voltage array will be:
            voltages_j = np.array([-0.1,  0.1, -0.1,  0.1])
        """
        voltages_j = np.array([V0 if i == 0 else V1 for i in pattern])
        return voltages_j

    def calculate_hardware_currents(
        self, pattern: np.ndarray, conductances: np.ndarray
    ) -> np.ndarray:
        """
        Calculates the hardware currents as the vector by matrix product of the given pattern and conductances.

        Args:
            pattern (np.ndarray): The input pattern.
            conductances (np.ndarray): Array of conductances.

        Returns:
            np.ndarray: The calculated hardware currents.

        Example:
            Given pattern = np.array([1, 0, 1, 1])
            and conductances = np.array([[ 0.2, -1.0,  0.8, -0.6], [ 0.4, -0.2,  1.0, -0.8], [ 0.6, -0.4,  1.2, -1.0], [ 0.8, -0.6,  1.4, -1.2]]),
            the resulting hardware currents will be:
            hardware_currents = np.array([0.12, -0.18 , 0.24, -0.2])
        """
        applied_voltages = self.voltage_array(pattern)
        hardware_currents = applied_voltages.dot(conductances)
        return hardware_currents

    def calculate_logic_currents(
        self, pattern: np.ndarray, conductances: np.ndarray
    ) -> None:
        """
        Calculates the logic currents by subtracting alternate hardware currents.

        Args:
            pattern (np.ndarray): The input pattern determining the applied voltages.
            conductances (np.ndarray): The conductance values.

        Returns:
            None

        Sets:
            self.logic_currents (np.ndarray): The calculated logic currents.
        """
        currents_array = self.calculate_hardware_currents(pattern, conductances)
        self.logic_currents = currents_array[::2] - currents_array[1::2]

    def activation_function(self) -> np.ndarray:
        """
        Applies the activation function (hyperbolic tangent) to the logic currents.

        Args:
            None

        Returns:
            np.ndarray: The activation values.
        """
        activation = np.tanh(self.beta * self.logic_currents)
        return activation

    def activation_function_derivative(self) -> np.ndarray:
        """
        Calculates the derivative of the activation function.

        Args:
            None

        Returns:
            np.ndarray: The derivative of the activation values.
        """
        derivative = self.beta / (np.cosh(self.beta * self.logic_currents)) ** 2
        return derivative

    def calculate_delta_i(self, output: np.ndarray) -> np.ndarray:
        """
        Computes the delta values for a given output using the activation function and its derivative.

        This function uses `np.where` to assign the positive or negative target values 
        based on the given output, and calculates the delta values using the activation 
        function and its derivative.

        Args:
            output (numpy.ndarray): A 1D array of target outputs, where each element is 
                either 1 (positive target) or 0 (negative target).

        Returns:
            numpy.ndarray: A 1D array of delta values, calculated using the formula:
                delta = (target_value - activation) * activation_derivative.
        
        Example:
            Given an output array where 1 represents the positive target and 0 
            represents the negative target:

            >>> output = np.array([1, 0, 1, 0, 1])
            >>> calculate_delta_i(output)
            array([delta_value_1, delta_value_2, ..., delta_value_n])
        """
        activation = self.activation_function()
        activation_derivative = self.activation_function_derivative()
        target_values = np.where(output == 1, self.positive_target, self.negative_target)
        delta_i = (target_values - activation) * activation_derivative

        return delta_i

    def calculate_Delta_ij(self, output: np.ndarray, pattern: np.ndarray, i) -> None:
        """
        Calculates and stores the Delta_ij values as the outer product between the voltages_j and the delta_i based on the output and pattern.

        Args:
            output (np.ndarray): The target output values.
            pattern (np.ndarray): The input pattern determining the applied voltages.
            i (int): The index at which to store the calculated Delta_ij values.

        Returns:
            None

        Sets:
            self.all_delta_ij (np.ndarray): The Delta_ij values stored at index i.
        """
        Delta_ij = np.outer(self.calculate_delta_i(output), self.voltage_array(pattern))
        self.all_delta_ij[i] = Delta_ij

    def calculate_DeltaW_ij(self) -> np.ndarray:
        """
        Calculates the DeltaW_ij values by summing and transposing the Delta_ij values.
        It also logs the Delta_ij value.

        Args:
            stamp (bool): If True, prints the calculated DeltaW_ij values.
                        Useful for debugging or verifying the calculation process.
                        Default is True.

        Returns:
            np.ndarray: The transposed array of DeltaW_ij values.

        Notes:
            The DeltaW_ij values are calculated by taking the sign of the sum of all Delta_ij values and transposing the result.
            If stamp is True, the method prints the DeltaW_ij array.
        """
        deltaW_ij = np.sign(np.sum(self.all_delta_ij, axis=0))
        DeltaW_ij = np.transpose(deltaW_ij)
        logging.info(f"DeltaW_ij: {DeltaW_ij}")
        return DeltaW_ij

    def update_weights(self, epoch) -> None:
        """
        Updates the weights based on the DeltaW_ij values and stores the conductances for the given epoch.

        Args:
            epoch (int): The current epoch index.
            stamp (bool): If True, prints the DeltaW_ij values during the weight update process.
                        This is useful for monitoring the changes during each epoch.
                        Default is True.

        Returns:
            None

        Sets:
            self.conductances (np.ndarray): Updated conductance values.
            self.all_conductances (np.ndarray): Stores the conductances for each epoch.

        Notes:
            The function iterates through each element of DeltaW_ij and updates the conductance values accordingly:
            - If DeltaW_ij[i, j] > 0 or < 0, the conductance index and value are updated, applying shifts and a multiplication factor.
            - If DeltaW_ij[i, j] = 0, no changes are made.
            The conductances are stored for the given epoch.
        """
        DeltaW_ij = self.calculate_DeltaW_ij()
        index_value_pairs = np.array(
            [[index, value] for index, value in enumerate(self.conductance_data)]
        )
        index = np.array(index_value_pairs[:, 0], dtype=int)
        value = index_value_pairs[:, 1]
        rows, cols = DeltaW_ij.shape
        for j in range(cols):
            for i in range(rows):
                if DeltaW_ij[i, j] == 0:
                    continue
                adjustment = 0 if DeltaW_ij[i, j] > 0 else 1
                ind = self.conductances[1, i, j * 2 + adjustment].astype(int)
                new_index = index[ind + 1]
                new_conductance = value[ind + 1]
                self.conductances[1, i, j * 2 + adjustment] = new_index
                self.conductances[0, i, j * 2 + adjustment] = (
                    new_conductance + self.shifts[i, j * 2 + adjustment]
                ) * self.multiplication_factor
        self.all_conductances[epoch] = self.conductances[0]         

    def convergence_criterion(self, output: np.ndarray, i, epoch) -> bool:
        """
        Checks if the current model's activation values meet the convergence criterion and if not, stores the distance from the convergence in the error value.

        Args:
            output (np.ndarray): The target output values.
            i (int): The index of the current pattern.
            epoch (int): The current epoch index.

        Returns:
            bool: True if the model has converged, False otherwise.

        Sets:
            self.errors (np.ndarray): The calculated errors for the current pattern.
            self.result (np.ndarray): The activation values for the current epoch and pattern.
        """
        fi = self.activation_function()
        positive_diff = np.where((output == 1) & (fi <= self.positive_target), self.positive_target - fi, 0)
        negative_diff = np.where((output == 0) & (fi >= self.negative_target), fi - self.negative_target, 0)
        self.errors[i] = positive_diff + negative_diff
        self.result[epoch, :, i] = np.where(output == 1, 
                                            np.minimum(fi, self.positive_target), 
                                            np.maximum(fi, self.negative_target))
        found_difference = np.any(positive_diff > 0) or np.any(negative_diff > 0)
        return not found_difference

    def total_error(self, epoch) -> None:
        """
        Calculates and prints the total error for the given epoch.
        It also logs the Total error value.

        Args:
            epoch (int): The current epoch index.
            stamp (bool): If True, prints the total error for the current epoch.
                        This is useful for monitoring and debugging the error during training.
                        Default is True.

        Returns:
            None

        Sets:
            self.all_errors (np.ndarray): The total error for the current epoch.

        Notes:
            The total error is the sum of all individual errors. If stamp is True, the total error is printed for the current epoch.
        """
        total_error = np.sum(self.errors)
        self.all_errors[epoch] = total_error
        logging.info(f"Total error: {total_error}")
        

    def plot_final_weights(self):
        """
        Generates a 3D bar plot of the final conductance weights.

        Args:
            None

        Returns:
            tuple: A tuple containing the figure and axes objects of the plot.

        Notes:
            This method creates a 3D bar plot using matplotlib to visualize the final conductance weights.
            The x and y axes represent the neurons and inputs, respectively, while the z axis represents the conductance values.
        """
        categories = ["1", "2", "3", "4"]
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection="3d")
        xpos, ypos = np.meshgrid(np.arange(len(categories)), np.arange(len(categories)))
        xpos = xpos.flatten()
        ypos = ypos.flatten()
        zpos = np.zeros_like(xpos)
        dx = dy = 0.75
        dz = self.conductances[0].flatten()
        num_bars = len(dz)
        colors = plt.cm.tab20(np.linspace(0, 1, num_bars))
        ax.bar3d(xpos, ypos, zpos, dx, dy, dz, color=colors, shade=True)
        ax.set_xticks(np.arange(len(categories)))
        ax.set_yticks(np.arange(len(categories)))
        ax.set_xticklabels(categories)
        ax.set_yticklabels(categories)
        ax.set_xlabel("Neuron")
        ax.set_ylabel("Input")
        plt.show()
        return fig, ax

    def plot_conductances(self, epochs):
        """
        Generates a series of subplots showing the evolution of conductances over epochs.

        Args:
            epochs (int): The number of epochs to plot.

        Returns:
            tuple: A tuple containing the figure and axes objects of the plot.

        Notes:
            This method creates a grid of subplots using matplotlib, where each subplot shows the conductance values for a specific neuron and input pair over the given epochs.
        """
        rows, cols = self.conductances[0].shape
        cmap = get_cmap("tab20")
        num_plots = rows * cols
        fig, axes = plt.subplots(rows, cols, figsize=(15, 10), sharex=True)
        for j in range(cols):
            for i in range(rows):
                Wij = self.all_conductances[:epochs, i, j]
                pulses = np.arange(epochs)
                ax = axes[i, j]
                color = cmap((i * cols + j) % num_plots)
                ax.plot(
                    pulses,
                    Wij,
                    "o-",
                    color=color,
                    linewidth=2,
                    label=f"Row = {i+1}\nColumn = {j+1}",
                )
                ax.xaxis.set_major_formatter(ticker.ScalarFormatter(useMathText=True))
                ax.yaxis.set_major_formatter(ticker.ScalarFormatter(useMathText=True))
                ax.ticklabel_format(style="sci", axis="y", scilimits=(0, 0))
                ax.tick_params(labelsize=12)
                ax.legend(loc="lower right")
        fig.text(0.5, 0.04, "Epoch", ha="center", va="center", fontsize=20)
        fig.text(
            0.04,
            0.5,
            "Conductance (S)",
            ha="center",
            va="center",
            rotation="vertical",
            fontsize=20,
        )
        plt.subplots_adjust(
            left=0.1, right=0.95, top=0.95, bottom=0.1, wspace=0.3, hspace=0.3
        )
        plt.show()
        return fig, axes

    def plot_weights(self, epochs):
        """
        Generates a series of subplots showing the evolution of synaptic weights over epochs.

        Args:
            epochs (int): The number of epochs to plot.

        Returns:
            tuple: A tuple containing the figure and axes objects of the plot.

        Notes:
            This method creates a grid of subplots using matplotlib, where each subplot shows the difference between paired conductances (synaptic weights) for a specific neuron over the given epochs.
        """
        rows, cols = self.conductances[0].shape
        cmap = get_cmap("tab10")
        cols = 2
        num_plots = rows * cols
        fig, axes = plt.subplots(rows, cols, figsize=(12, 7), sharex=True)
        for j in range(cols):
            for i in range(rows):
                Wij = (
                    self.all_conductances[:epochs, i, j * 2]
                    - self.all_conductances[:epochs, i, j * 2 + 1]
                )
                pulses = np.arange(epochs)
                ax = axes[i, j]
                color = cmap((i * cols + j) % num_plots)
                ax.plot(
                    pulses,
                    Wij,
                    "o-",
                    color=color,
                    linewidth=3,
                    label=f"Synapse = {i+1}\nNeuron = {j+1}",
                )
                ax.xaxis.set_major_formatter(ticker.ScalarFormatter(useMathText=True))
                ax.yaxis.set_major_formatter(ticker.ScalarFormatter(useMathText=True))
                ax.ticklabel_format(style="sci", axis="y", scilimits=(0, 0))
                ax.tick_params(labelsize=12)
                ax.legend(loc="best", fontsize=10)
                ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
        fig.text(0.5, 0.04, "Epoch", ha="center", va="center", fontsize=17)
        fig.text(
            0.04,
            0.5,
            "Synaptic Weights (S)",
            ha="center",
            va="center",
            rotation="vertical",
            fontsize=17,
        )
        plt.subplots_adjust(
            left=0.08, right=0.95, top=0.95, bottom=0.08, wspace=0.2, hspace=0.2
        )
        plt.show()
        return fig, axes

    def plot_error(self, epochs):
        """
        Generates a plot of the total error over epochs.

        Args:
            epochs (int): The number of epochs to plot.

        Returns:
            tuple: A tuple containing the figure and axes objects of the plot.

        Notes:
            This method creates a line plot using matplotlib to visualize the evolution of the total error over the given epochs.
        """
        pulses = np.arange(epochs + 1)
        pulses = pulses[1:]
        errors = self.all_errors[1:epochs]
        errors = np.append(errors, 0)
        fig, ax = plt.subplots(1, 1, figsize=(10, 3))
        ax.plot(pulses, errors, "o-", linewidth=2)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Total error")
        ax.set_title("Error evolution")
        plt.show()
        return fig

    def plot_results(self, pattern: np.ndarray, output: np.ndarray, epochs):
        """
        Generates a series of subplots showing the activation results over epochs.

        Args:
            pattern (np.ndarray): The input pattern.
            output (np.ndarray): The target output values.
            epochs (int): The number of epochs to plot.

        Returns:
            tuple: A tuple containing the figure and axes objects of the plot.

        Notes:
            This method creates a grid of subplots using matplotlib, where each subplot shows the activation results for a specific neuron and input pair over the given epochs.
            Horizontal lines indicate the positive and negative targets.
        """
        pulses = np.arange(epochs)
        rows, cols = self.result[0].shape
        cmap = get_cmap("tab20")
        num_plots = rows * cols
        fig, axes = plt.subplots(rows, cols, figsize=(17, 8), sharex=True)
        for j in range(cols):
            for i in range(rows):
                results = self.result[:epochs, i, j]
                ax = axes[i, j]
                color = cmap((i * cols + j) % num_plots)
                patt = pattern[j]
                out = output[j]
                ax.plot(
                    pulses,
                    results,
                    "o-",
                    color=color,
                    linewidth=2,
                    label=f"Pattern: {patt}\nOutput: {out}\nLogic Neuron: {i + 1}",
                )
                if out[i] == 1:
                    ax.axhline(self.positive_target)
                else:
                    ax.axhline(self.negative_target)
                ax.xaxis.set_major_formatter(ticker.ScalarFormatter(useMathText=True))
                ax.yaxis.set_major_formatter(ticker.ScalarFormatter(useMathText=True))
                ax.ticklabel_format(style="sci", axis="y", scilimits=(0, 0))
                ax.tick_params(labelsize=12)
                ax.legend(loc="best")
        fig.text(0.5, 0.04, "Epoch", ha="center", va="center", fontsize=20)
        fig.text(
            0.04,
            0.5,
            "Activation Result",
            ha="center",
            va="center",
            rotation="vertical",
            fontsize=20,
        )
        plt.subplots_adjust(
            left=0.1, right=0.95, top=0.95, bottom=0.1, wspace=0.3, hspace=0.3
        )
        plt.show()
        return fig, axes

    def save_data(self, base_filename="simulation", converged=False):
        """
        Saves the simulation data to a CSV file.

        Args:
            base_filename (str, optional): The base name for the output file. Defaults to "simulation".
            converged (bool, optional): Indicates if the simulation converged. Defaults to False.

        Returns:
            None

        Notes:
            This method saves various parameters and results of the simulation to a CSV file.
            The file is stored in a directory named with the current date. Subdirectories for converged
            and non-converged simulations are created as needed.
        """
        current_date = datetime.now().strftime("%d-%m-%Y")

        os.makedirs(current_date)

        converged_dir = os.path.join(current_date, "converged")
        not_converged_dir = os.path.join(current_date, "not_converged")

        os.makedirs(converged_dir)
        os.makedirs(not_converged_dir)

        if converged:
            filename = f"{base_filename}_converged_data.csv"
            file_dir = converged_dir
        else:
            filename = f"{base_filename}_not_converged_data.csv"
            file_dir = not_converged_dir

        file_path = os.path.join(file_dir, filename)
        with open(file_path, mode="w", newline="") as file:
            writer = csv.writer(file)

            writer.writerow(["Simulation Results"])
            file.write(f"Beta = {self.beta}\n")
            file.write(f"Positive Target = {self.positive_target}\n")
            file.write(f"Negative Target = {self.negative_target}\n")
            file.write(f"Multiplication factor = {self.multiplication_factor}\n")
            file.write(f"Random distribution parameter = {self.range}\n")
            file.write(f"Shifts = {self.shifts}\n")
            file.write(f"C_(ij) where i = Input, j = Neuron\n")
            writer.writerow([])
            writer.writerow(
                ["Epoch", " Total Error"]
                + [f" C_({i}{j})" for i in range(1, 5) for j in range(1, 5)]
            )

            for epoch in range(self.epochs):
                conductances_flat = self.all_conductances[epoch].flatten().tolist()
                row = [epoch, self.all_errors[epoch]] + conductances_flat
                writer.writerow(row)

    def fit(
            self,
            patterns: np.ndarray,
            outputs: np.ndarray,
            conductance_data: np.ndarray = None,
            custom_shifts: np.ndarray = None,
            save_data: bool = False,
            filename: str = "simulation",
        ) -> None:
        """
        Trains the model using the provided patterns and outputs.
        This method iteratively updates the model weights based on the training patterns and target outputs until convergence is achieved. 
        It logs key events during the training process, including convergence status and conductance values.

        Args:
            patterns (np.ndarray): The input patterns for training, where each row is a sample and each column represents a feature.
            outputs (np.ndarray): The target outputs corresponding to the input patterns, structured similarly to the patterns array.
            conductance_data (np.ndarray, optional): Experimental conductance data to use during training. Defaults to None.
            custom_shifts (np.ndarray, optional): A 4x4 array of custom shifts to apply to the conductance values. If provided, these shifts will be used instead of randomly generated ones. Defaults to None.
            save_data (bool, optional): Whether to save the simulation data to a file. Defaults to False.
            filename (str, optional): The base filename for saving data. Defaults to "simulation".

        Returns:
            int: The epoch at which convergence was reached, or the total number of epochs if not converged.

        Notes:
            This method trains the model by iterating through the epochs and updating the weights until convergence is reached.
            If `save_data` is True, the simulation data is saved to a file.

        Logging:
            The method logs:
            - The epoch at which convergence is reached, along with the final conductance values.
            - The epoch number for each iteration.
            - A message indicating that the training did not converge if applicable.
        """
        self.experimental_data(conductance_data)
        if custom_shifts is not None:
            self.custom_shift(custom_shifts)
        else:
            self.shift_exp()
        self.conductance_init()
        for epoch in range(1, self.epochs):
            converged = True
            for i in range(patterns.shape[0]):
                self.calculate_logic_currents(patterns[i], self.conductances[0])
                self.calculate_Delta_ij(outputs[i], patterns[i], i)
                if not self.convergence_criterion(outputs[i], i, epoch):
                    converged = False            
            if converged:
                logging.info(f"\nConvergence reached at epoch {epoch}")
                logging.info(f"Conductances: {self.conductances[0]}")
                correct_conductances = (
                    self.conductances[0] + self.shifts
                ) * self.multiplication_factor
                self.saved_correct_conductances = correct_conductances                
                if save_data:
                    self.save_data(base_filename=filename, converged=converged)
                return epoch            
            self.update_weights(epoch)
            self.total_error(epoch)
            logging.info(f"Epoch: {epoch}")
        logging.info("Not converged")
        if save_data:
            self.save_data(base_filename=filename, converged=converged)
        return epoch 

    def visualize_graphs(self, epoch: int, patterns: np.ndarray, outputs: np.ndarray, converged: bool) -> None:
        """
        Generates plots for the results after fitting the model.

        Args:
            epoch (int): The epoch at which convergence was reached.
            patterns (np.ndarray): The input patterns used for training.
            outputs (np.ndarray): The target outputs corresponding to the input patterns.
            converged (bool): Indicates whether the simulation converged.

        Returns:
            None
        """
        self.plot_conductances(epoch)
        self.plot_weights(epoch)
        self.plot_error(epoch)
        self.plot_results(patterns, outputs, epoch)
        if converged:
            self.plot_final_weights()

    def check_convergence(self, i) -> bool:
        """
        Checks the convergence of the activation function for a given pattern.

        Args:
            i (int): The index of the pattern to check.

        Returns:
            None

        Notes:
            This method updates the predictions array based on the comparison of the activation function results with the positive and negative targets.
        """
        fi = self.activation_function()
        self.predictions[i] = np.select(
            [fi >= self.positive_target, fi <= self.negative_target],
            [1, 0],
            default=2
        )

    def predict(self, patterns, outputs):
        """
        Makes predictions on the output based on the saved correct conductances and the provided patterns.

        Args:
            patterns (np.ndarray): The input patterns.
            outputs (np.ndarray): The expected output values.

        Returns:
            None

        Notes:
            This method calculates the logic currents for each pattern using the saved correct conductances and then checks for convergence.
            It prints the pattern, prediction, and expected result for each input pattern.
        """
        for i in range(patterns.shape[0]):
            self.calculate_logic_currents(patterns[i], self.saved_correct_conductances)
            self.check_convergence(i)
            print(
                "Pattern:",
                patterns[i],
                "Prediction:",
                self.predictions[i],
                "Expected result:",
                outputs[i],
            )
