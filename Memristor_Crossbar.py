import os
import csv
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from dataclasses import dataclass
from matplotlib.cm import get_cmap
from datetime import datetime


@dataclass
class Memristor_Crossbar:

    beta: float
    positive_target: float
    negative_target: float
    range: float
    multiplication_factor: int
    training_set_width: int = 6
    epochs: int = 48

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
        self.shifts = np.empty([4, 4])
        self.logic_currents = np.empty([2])
        self.all_delta_ij = np.empty([self.training_set_width, 2, 4])
        self.conductances = np.empty([2, 4, 4])
        self.all_conductances = np.empty([self.epochs, 4, 4])
        self.saved_correct_conductances = np.empty([4, 4])
        self.errors = np.empty([self.training_set_width, 2])
        self.all_errors = np.empty([self.epochs])
        self.result = np.empty([self.epochs, 2, self.training_set_width])
        self.predictions = np.empty([self.test_set_width, 2])

    def experimental_data(self, conductance_data: np.ndarray):
        """
        Initializes the conductance data by normalizing it to the first value.

        Args:
            conductance_data (np.ndarray): The raw conductance data to be processed.

        Returns:
            None

        Sets:
            self.conductance_data (np.ndarray): The normalized conductance data, where the first value is subtracted from each element.
        """
        raw_conductance_data = conductance_data
        first_value = raw_conductance_data[0]
        self.conductance_data = raw_conductance_data - first_value

    def shift(self) -> None:
        """
        Generates random shifts for the conductance values and reshapes them into a 4x4 array.

        Args:
            None

        Returns:
            None

        Sets:
            self.shifts (np.ndarray): A 4x4 array of random shifts, uniformly distributed around the center value of 0 within the range specified by self.range.
        """
        center_value = 0
        num_elements = 16
        range_width = self.range
        rnd_shifts = np.random.uniform(
            low=center_value - range_width / 2,
            high=center_value + range_width / 2,
            size=num_elements,
        )
        self.shifts = np.reshape(rnd_shifts, (4, 4))

    def conductance_init_rnd(self) -> None:
        """
        Initializes the conductance values with random shifts and a multiplication factor, and prints the initial conductances and epoch.

        Args:
            None

        Returns:
            None

        Sets:
            self.conductances (np.ndarray): An array where the first element is the product of the normalized conductance data and shifts, multiplied by the multiplication factor. While the second element is the index zero, which indicates the initial set of conductances for the training.
            self.all_conductances (np.ndarray): Similar to self.conductances, storing the same product in the first element.

        Prints:
            The initial conductances and the epoch (0).
        """
        self.conductances[0] = (
            self.conductance_data[0] + self.shifts
        ) * self.multiplication_factor
        self.conductances[1] = 0
        self.all_conductances[0] = (
            self.conductance_data[0] + self.shifts
        ) * self.multiplication_factor
        print("Initial Conductances:", self.all_conductances[0])
        print("Epoch:", 0)

    def voltage_array(self, pattern: np.ndarray, V0=-0.1, V1=0.1) -> np.ndarray:
        """
        Generates an array of voltages based on a given pattern.

        Args:
            pattern (np.ndarray): The input pattern determining the voltage values.
            V0 (float, optional): Voltage for pattern value 0. Defaults to -0.1.
            V1 (float, optional): Voltage for pattern value 1. Defaults to 0.1.

        Returns:
            np.ndarray: An array of voltages corresponding to the pattern.
        """
        voltages_j = np.array([V0 if i == 0 else V1 for i in pattern])
        return voltages_j

    def calculate_hardware_currents(
        self, pattern: np.ndarray, conductances: np.ndarray
    ) -> np.ndarray:
        """
        Calculates the hardware currents as a vector by matrix product based on applied voltages and conductances.

        Args:
            pattern (np.ndarray): The input pattern determining the applied voltages.
            conductances (np.ndarray): The conductance values.

        Returns:
            np.ndarray: The calculated hardware currents.
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
        Calculates the delta values for each output based on the activation function and its derivative.

        Args:
            output (np.ndarray): The target output values.

        Returns:
            np.ndarray: An array containing the delta values for each output.

        Notes:
            The delta values are calculated as follows:
            - If the target output is 1, delta is calculated using the positive target.
            - If the target output is 0, delta is calculated using the negative target.
        """
        delta_i = np.empty([2])
        for index, target_output in enumerate(output):
            if target_output == 1:
                delta = (
                    self.positive_target - self.activation_function()[index]
                ) * self.activation_function_derivative()[index]
                delta_i[index] = delta
            elif target_output == 0:
                delta = (
                    self.negative_target - self.activation_function()[index]
                ) * self.activation_function_derivative()[index]
                delta_i[index] = delta
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

        Args:
            None

        Returns:
            np.ndarray: The transposed array of DeltaW_ij values.

        Notes:
            The DeltaW_ij values are calculated by taking the sign of the sum of all Delta_ij values and transposing the result.
        """
        deltaW_ij = np.sign(np.sum(self.all_delta_ij, axis=0))
        DeltaW_ij = np.transpose(deltaW_ij)
        return DeltaW_ij

    def update_weights(self, epoch) -> None:
        """
        Updates the weights based on the DeltaW_ij values and stores the conductances for the given epoch.

        Args:
            epoch (int): The current epoch index.

        Returns:
            None

        Sets:
            self.conductances (np.ndarray): Updated conductance values.
            self.all_conductances (np.ndarray): Stores the conductances for each epoch.

        Notes:
            The function iterates through each element of DeltaW_ij and updates the conductance values accordingly. If DeltaW_ij[i, j] > 0 or < 0,
            the conductance index and value are updated, applying shifts and a multiplication factor. If DeltaW_ij[i, j] == 0, no changes are made.
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
                if DeltaW_ij[i, j] > 0:
                    ind = self.conductances[1, i, j * 2].astype(int)
                    new_index = index[ind + 1]
                    new_conductance = value[ind + 1]
                    self.conductances[1, i, j * 2] = new_index
                    self.conductances[0, i, j * 2] = (
                        new_conductance + self.shifts[i, j * 2]
                    ) * self.multiplication_factor
                elif DeltaW_ij[i, j] < 0:
                    ind = self.conductances[1, i, j * 2 + 1].astype(int)
                    new_index = index[ind + 1]
                    new_conductance = value[ind + 1]
                    self.conductances[1, i, j * 2 + 1] = new_index
                    self.conductances[0, i, j * 2 + 1] = (
                        new_conductance + self.shifts[i, j * 2 + 1]
                    ) * self.multiplication_factor
                else:
                    continue
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

        Notes:
            The function checks if the activation values are within the target range.
            If not, it calculates the errors and updates the result array.
        """
        fi = self.activation_function()
        found_difference = False
        for index, element in enumerate(fi):
            if (output[index] == 1) and (element <= self.positive_target):
                difference = self.positive_target - element
                self.errors[i, index] = difference
                self.result[epoch, index, i] = element
                found_difference = True
            elif (output[index] == 0) and (element >= self.negative_target):
                difference = element - self.negative_target
                self.errors[i, index] = difference
                self.result[epoch, index, i] = element
                found_difference = True
            else:
                self.errors[i, index] = 0
                if output[index] == 0:
                    self.result[epoch, index, i] = self.negative_target
                else:
                    self.result[epoch, index, i] = self.positive_target
        if found_difference:
            return False
        return True

    def total_error(self, epoch) -> None:
        """
        Calculates and prints the total error for the given epoch.

        Args:
            epoch (int): The current epoch index.

        Returns:
            None

        Sets:
            self.all_errors (np.ndarray): The total error for the current epoch.

        Notes:
            The total error is the sum of all individual errors.
        """
        total_error = np.sum(self.errors)
        self.all_errors[epoch] = total_error
        print("Total error:", total_error)

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
        ax.ticklabel_format(style="sci", axis="z", scilimits=(0, 0))
        ax.set_xlabel("Neuron")
        ax.set_ylabel("Input")
        ax.set_zlabel("Conductances")
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
                ax.grid()
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
                ax.grid()
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
        errors = self.all_errors[1:epochs]
        errors = np.append(errors, 0)
        fig, ax = plt.subplots(1, 1, figsize=(10, 3))
        ax.plot(pulses, self.all_errors, "o-", linewidth=2)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Total error")
        ax.set_title("Error evolution")
        ax.grid()
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
                ax.grid()
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
        if not os.path.exists(current_date):
            os.makedirs(current_date)

        converged_dir = os.path.join(current_date, "converged")
        not_converged_dir = os.path.join(current_date, "not_converged")

        if not os.path.exists(converged_dir):
            os.makedirs(converged_dir)

        if not os.path.exists(not_converged_dir):
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
        plot: bool = True,
        conductance_data: np.ndarray = None,
        save_data: bool = False,
        filename: str = "simulation",
    ) -> None:
        """
        Trains the model using the provided patterns and outputs.

        Args:
            patterns (np.ndarray): The input patterns.
            outputs (np.ndarray): The target outputs.
            plot (bool, optional): Whether to generate plots of the results. Defaults to True.
            conductance_data (np.ndarray, optional): Experimental conductance data to use. Defaults to None.
            save_data (bool, optional): Whether to save the simulation data. Defaults to False.
            filename (str, optional): The base filename for saving data. Defaults to "simulation".

        Returns:
            int: The epoch at which convergence was reached, or the total number of epochs if not converged.

        Notes:
            This method trains the model by iterating through the epochs and updating the weights until convergence is reached.
            If `plot` is True, various plots of the results are generated.
            If `save_data` is True, the simulation data is saved to a file.
        """
        self.experimental_data(conductance_data)
        self.shift()
        self.conductance_init_rnd()
        for epoch in range(1, self.epochs):
            converged = True
            for i in range(patterns.shape[0]):
                self.calculate_logic_currents(patterns[i], self.conductances[0])
                self.calculate_Delta_ij(outputs[i], patterns[i], i)
                if not self.convergence_criterion(outputs[i], i, epoch):
                    converged = False
            if converged:
                print(f"\nConvergence reached at epoch {epoch}")
                print("Conductances:", self.conductances[0])
                correct_conductances = (
                    self.conductances[0] + self.shifts
                ) * self.multiplication_factor
                self.saved_correct_conductances = correct_conductances
                if save_data:
                    self.save_data(base_filename=filename, converged=converged)
                if plot:
                    self.plot_conductances(epoch)
                    self.plot_weights(epoch)
                    self.plot_error(epoch)
                    self.plot_results(patterns, outputs, epoch)
                    self.plot_final_weights()
                return epoch
            self.update_weights(epoch)
            self.total_error(epoch)
            print("Epoch:", epoch)
        print("Not converged")

        if save_data:
            self.save_data(base_filename=filename, converged=converged)

        if plot:
            self.plot_conductances(epoch)
            self.plot_weights(epoch)
            self.plot_error(epoch)
            self.plot_results(patterns, outputs, epoch)

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
        for index, element in enumerate(fi):
            if element >= self.positive_target:
                self.predictions[i, index] = 1
            elif element <= self.negative_target:
                self.predictions[i, index] = 0
            else:
                self.predictions[i, index] = 2

    def predict(self, patterns, outputs):
        """
        Makes predictions based on the saved correct conductances and the provided patterns.

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
