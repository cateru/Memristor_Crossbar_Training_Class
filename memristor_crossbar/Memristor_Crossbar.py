import os
import csv
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from dataclasses import dataclass
from matplotlib.cm import get_cmap
from datetime import datetime
import logging
from matplotlib.colors import LinearSegmentedColormap


@dataclass
class Memristor_Crossbar:

    beta: float
    positive_target: float
    negative_target: float
    multiplication_factor: int
    training_set_width: int = 6
    epochs: int = 51
    number_of_neurons: int = 2
    number_of_rows: int = 4
    number_of_columns: int = 4
    max_value : int = 10**-7
    min_value : int = 10**-8
    division_factor : int = 5       

    test_set_width: int = 16 - training_set_width
    num_elements: int = number_of_rows*number_of_columns
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

        Parameters
        ----------
        - conductance_data : np.ndarray
            The raw set of conductance data to be processed.

        Returns
        -------
        None

        Examples
        --------
        >>> conductance_data = np.array([5.0, 6.0, 7.0, 8.0])
        >>> self.conductance_data = np.array([0.0, 1.0, 2.0, 3.0])
        """
        raw_conductance_data = conductance_data
        first_value = raw_conductance_data[0]
        self.conductance_data = raw_conductance_data - first_value

    def shift_lognormal(self) -> None:
        """
        Generates random shifts for conductance values using a lognormal distribution,
        scales them to a specified range, and reshapes the result into a 4x4 array.

        Parameters
        ----------
        - self.min_value : float
            The lower bound of the scaled random shifts.
        - self.max_value : float
            The upper bound of the scaled random shifts.
        - self.division_factor : float
            Controls the spread (standard deviation) of the lognormal distribution.
        - self.num_elements : int
            Total number of elements to be generated (size of reshaped array).

        Returns
        -------
        None

        Examples
        --------
        >>> crossbar = Memristor_Crossbar()
        >>> crossbar.shift_lognormal()
        >>> print(crossbar.shifts)
        [[1.23e-08 1.56e-08 1.10e-08 1.45e-08]
        [1.34e-08 1.29e-08 1.60e-08 1.47e-08]
        [1.32e-08 1.50e-08 1.48e-08 1.28e-08]
        [1.40e-08 1.36e-08 1.41e-08 1.44e-08]]
        """
        mu = np.log(np.mean([self.min_value, self.max_value]))
        sigma = abs(np.log(self.max_value / self.min_value) / self.division_factor)
        rnd_shifts = np.random.lognormal(mean=mu, sigma=sigma, size=self.num_elements)
        scaled_shifts = self.min_value + (self.max_value - self.min_value) * (
            rnd_shifts - rnd_shifts.min()
        ) / (rnd_shifts.max() - rnd_shifts.min())
        self.shifts = np.reshape(scaled_shifts, (self.number_of_rows, self.number_of_columns))

    def custom_shift(self, custom_shifts: np.ndarray) -> None:
        """
        Sets custom shifts for the conductance values based on a user-defined 4x4 array.

        Parameters
        ----------
        custom_shifts : np.ndarray
            A 4x4 NumPy array containing user-defined shifts for the conductance values.

        Returns
        -------
        None
        """
        self.shifts = custom_shifts

    def conductance_init(self) -> None:
        """
        Initializes the conductance values with random shifts and a multiplication factor. The first element is the sum of the normalized conductance data and shifts, multiplied by the multiplication factor while the second element is the index zero.

        Returns
        -------
        None

        Examples
        --------
        >>> self.conductance_data[0] = np.array([0.0, 1.0, 2.0, 3.0])
        >>> self.shifts = np.array([[ 0.1, -0.5,  0.4, -0.3],
        ...                         [ 0.2, -0.1,  0.5, -0.4],
        ...                         [ 0.3, -0.2,  0.6, -0.5],
        ...                         [ 0.4, -0.3,  0.7, -0.6]])
        >>> self.multiplication_factor = 2
        >>> self.conductances[0] = np.array([[ 0.2, -1.0,  0.8, -0.6],
        ...                                  [ 0.4, -0.2,  1.0, -0.8],
        ...                                  [ 0.6, -0.4,  1.2, -1.0],
        ...                                  [ 0.8, -0.6,  1.4, -1.2]])
        >>> self.conductances[1] = 0


        Logs the initialized conductances and the epoch number:

        >>> "Initial Conductances: [0.2 -1.0 0.8 -0.6 ...]"
        >>> "Epoch: 0"
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

        Parameters
        ----------
        pattern : np.ndarray
            The input pattern determining the voltage values.
        V0 : float, optional
            Voltage for pattern value 0. Defaults to -0.1.
        V1 : float, optional
            Voltage for pattern value 1. Defaults to 0.1.

        Returns
        -------
        np.ndarray
            An array of voltages corresponding to the pattern.

        Examples
        --------
        >>> pattern = np.array([0, 1, 0, 1])
        >>> V0 = -0.1
        >>> V1 = 0.1
        >>> voltages_j = np.array([-0.1,  0.1, -0.1,  0.1])
        """
        voltages_j = np.array([V0 if i == 0 else V1 for i in pattern])
        return voltages_j
    
    def calculate_hardware_currents(
        self, pattern: np.ndarray, conductances: np.ndarray
    ) -> np.ndarray:
        """
        Calculate hardware currents as the vector-by-matrix product of the pattern and conductances.

        Parameters
        ----------
        pattern : np.ndarray
            Input pattern.
        conductances : np.ndarray
            Array of conductances.

        Returns
        -------
        np.ndarray
            Calculated hardware currents.

        Examples
        --------
        >>> pattern = np.array([1, 0, 1, 1])
        >>> conductances = np.array([[0.2, -1.0,  0.8, -0.6],
        ...                          [0.4, -0.2,  1.0, -0.8],
        ...                          [0.6, -0.4,  1.2, -1.0],
        ...                          [0.8, -0.6,  1.4, -1.2]])
        >>> hardware_currents = obj.calculate_hardware_currents(pattern, conductances)
        >>> hardware_currents
        array([0.12, -0.18, 0.24, -0.2])
        """
        applied_voltages = self.voltage_array(pattern)
        hardware_currents = applied_voltages.dot(conductances)
        return hardware_currents

    def calculate_logic_currents(
        self, pattern: np.ndarray, conductances: np.ndarray
    ) -> None:
        """
        Calculate logic currents by subtracting alternate hardware currents.

        Parameters
        ----------
        pattern : np.ndarray
            Input pattern determining applied voltages.
        conductances : np.ndarray
            Conductance values.

        Returns
        -------
        None
        """
        currents_array = self.calculate_hardware_currents(pattern, conductances)
        self.logic_currents = currents_array[::2] - currents_array[1::2]

    def activation_function(self) -> np.ndarray:
        """
        Apply the activation function (hyperbolic tangent) to the logic currents.

        Returns
        -------
        np.ndarray
            Activation values.
        """
        activation = np.tanh(self.beta * self.logic_currents)
        return activation

    def activation_function_derivative(self) -> np.ndarray:
        """
        Calculate the derivative of the activation function.

        Returns
        -------
        np.ndarray
            Derivative of the activation values.
        """
        derivative = self.beta / (np.cosh(self.beta * self.logic_currents)) ** 2
        return derivative

    def calculate_delta_i(self, output: np.ndarray) -> np.ndarray:
        """
        Compute delta values for a given output using activation function and its derivative.

        Parameters
        ----------
        output : np.ndarray
            Target output, where each element is either 1 (positive target) or 0 (negative target).

        Returns
        -------
        np.ndarray
            Delta values calculated as `(target_value - activation) * activation_derivative`.

        Examples
        --------
        >>> output = np.array([1, 0, 1, 0, 1])
        >>> delta_i = obj.calculate_delta_i(output)
        >>> delta_i
        array([delta_value_1, delta_value_2, ..., delta_value_n])
        """
        activation = self.activation_function()
        activation_derivative = self.activation_function_derivative()
        target_values = np.where(output == 1, self.positive_target, self.negative_target)
        delta_i = (target_values - activation) * activation_derivative
        return delta_i

    def calculate_Delta_ij(
        self, output: np.ndarray, pattern: np.ndarray, i: int
    ) -> None:
        """
        Calculate and store Delta_ij values as the outer product of voltages and delta_i.

        Parameters
        ----------
        output : np.ndarray
            Target output values.
        pattern : np.ndarray
            Input pattern determining applied voltages.
        i : int
            Index to store calculated Delta_ij values.

        Returns
        -------
        None
        """
        Delta_ij = np.outer(self.calculate_delta_i(output), self.voltage_array(pattern))
        self.all_delta_ij[i] = Delta_ij


    def calculate_DeltaW_ij(self) -> np.ndarray:
        """
        Calculate the DeltaW_ij values by summing and transposing the Delta_ij values.

        Returns
        -------
        np.ndarray
            The transposed array of DeltaW_ij values.

        Notes
        -----
        - DeltaW_ij values are calculated by taking the sign of the sum of all Delta_ij values and transposing the result.
        - Logs the DeltaW_ij values for debugging or verification.
        """
        deltaW_ij = np.sign(np.sum(self.all_delta_ij, axis=0))
        DeltaW_ij = np.transpose(deltaW_ij)
        logging.info(f"DeltaW_ij: {DeltaW_ij}")
        return DeltaW_ij

    def update_weights(self, epoch) -> None:
        """
        Update the weights based on the DeltaW_ij values and store the conductances for the given epoch.

        Parameters
        ----------
        epoch : int
            The current epoch index.

        Returns
        -------
        None
        
        Notes
        -----
        - Iterates through each element of DeltaW_ij to update conductance values.
        - If DeltaW_ij[i, j] > 0 or < 0, applies shifts and a multiplication factor to update the conductances.
        - If DeltaW_ij[i, j] == 0, no changes are made.
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

    def convergence_criterion(self, output: np.ndarray, i: int, epoch: int) -> bool:
        """
        Check if the model's activation values meet the convergence criterion.

        Parameters
        ----------
        output : np.ndarray
            The target output values.
        i : int
            Index of the current pattern.
        epoch : int
            The current epoch index.

        Returns
        -------
        bool
            True if the model has converged, False otherwise.
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

    def total_error(self, epoch: int) -> None:
        """
        Calculate and log the total error for the given epoch.

        Parameters
        ----------
        epoch : int
            The current epoch index.

        Returns
        -------
        None

        Notes
        -----
        - The total error is the sum of all individual errors.
        - Logs the total error for debugging or monitoring purposes.
        """
        total_error = np.sum(self.errors)
        self.all_errors[epoch] = total_error
        logging.info(f"Total error: {total_error}")


    def plot_final_weights(self):
        """
        Generates a 3D bar plot of the final conductance weights.

        Returns
        -------
        tuple
            A tuple containing the figure and axes objects of the plot.

        Notes
        -----
        This method creates a 3D bar plot using matplotlib to visualize the final conductance weights.
        The x and y axes represent the neurons and inputs, respectively, while the z-axis represents the conductance values.
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

        Parameters
        ----------
        epochs : int
            The number of epochs to plot.

        Returns
        -------
        tuple
            A tuple containing the figure and axes objects of the plot.

        Notes
        -----
        This method creates a grid of subplots using matplotlib, where each subplot shows the conductance values 
        for a specific neuron and input pair over the given epochs.
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

        Parameters
        ----------
        epochs : int
            The number of epochs to plot.

        Returns
        -------
        tuple
            A tuple containing the figure and axes objects of the plot.

        Notes
        -----
        This method creates a grid of subplots using matplotlib, where each subplot shows the difference between 
        paired conductances (synaptic weights) for a specific neuron over the given epochs.
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
                ax.set_xticks(pulses[::3])
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

        Parameters
        ----------
        epochs : int
            The number of epochs to plot.

        Returns
        -------
        tuple
            A tuple containing the figure and axes objects of the plot.

        Notes
        -----
        This method creates a line plot using matplotlib to visualize the evolution of the total error 
        over the given epochs.
        """
        pulses = np.arange(1, epochs+1)
        errors = self.all_errors[1:epochs+1] 
        fig, ax = plt.subplots(1, 1, figsize=(10, 3))
        ax.plot(pulses, errors, "o-", linewidth=2)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Total error")
        ax.set_title("Error evolution")
        ax.set_xticks(pulses[::3])
        ax.set_ylim(bottom=0, top=np.max(errors)+1)
        plt.show()
        return fig

    def plot_results(self, pattern: np.ndarray, output: np.ndarray, epochs):
        """
        Generates a series of subplots showing the activation results over epochs.

        Parameters
        ----------
        pattern : np.ndarray
            The input pattern.
        output : np.ndarray
            The target output values.
        epochs : int
            The number of epochs to plot.

        Returns
        -------
        tuple
            A tuple containing the figure and axes objects of the plot.

        Notes
        -----
        This method creates a grid of subplots using matplotlib, where each subplot shows the activation results 
        for a specific neuron and input pair over the given epochs. Horizontal lines indicate the positive 
        and negative targets.
        """
        pulses = np.arange(epochs+1)
        rows, cols = self.result[0].shape
        blue_palette= LinearSegmentedColormap.from_list("cyan_palette", ["#00B4D8", "#023E8A"])
        num_plots = rows * cols
        fig, axes = plt.subplots(rows, cols, figsize=(17, 8), sharex=True)
        for j in range(cols):
            for i in range(rows):
                results = self.result[:epochs+1, i, j]
                ax = axes[i, j]
                color = blue_palette((i * cols + j) / num_plots)
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
                    ax.axhline(self.positive_target, linewidth = 1.5, color = "red")
                else:
                    ax.axhline(self.negative_target, linewidth = 1.5, color = "red")
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

        Parameters
        ----------
        base_filename : str, optional
            The base name for the output file. Defaults to "simulation".
        converged : bool, optional
            Indicates if the simulation converged. Defaults to False.

        Returns
        -------
        None

        Notes
        -----
        This method saves various parameters and results of the simulation to a CSV file. The file is stored in a 
        directory named with the current date. Subdirectories for converged and non-converged simulations are 
        created as needed.
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
            file.write(f"Shifts = {self.shifts}\n")
            file.write(f"Shift range upper limit = {self.max_value}\n")
            file.write(f"Shift range lower limit = {self.min_value}\n")
            file.write(f"Division factor = {self.division_factor}\n")
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
        Trains the model using the provided patterns and outputs. This method iteratively updates the model weights based on the training patterns and target outputs until convergence is achieved. 
        It logs key events during the training process, including convergence status and conductance values.

        Parameters
        ----------
        patterns : np.ndarray
            The input patterns for training, where each row is a sample and each column represents a feature.
        outputs : np.ndarray
            The target outputs for training, corresponding to the input patterns.
        conductance_data : np.ndarray, optional
            Initial conductance data used for training. Defaults to None.
        custom_shifts : np.ndarray, optional
            Custom shift values for conductance updates. Defaults to None.
        save_data : bool, optional
            Whether to save the simulation data. Defaults to False.
        filename : str, optional
            The base filename for saving data. Defaults to "simulation".

        Returns
        -------
        None

        Notes
        -----
        This method iteratively updates the model weights based on the training patterns and target outputs until 
        convergence is achieved. It logs key events during the training process, including convergence status 
        and conductance values.
        """
        self.experimental_data(conductance_data)
        if custom_shifts is not None:
            self.custom_shift(custom_shifts)
        else:
            self.shift_lognormal()
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
        Generate plots for the results after fitting the model.

        Parameters
        ----------
        epoch : int
            The epoch at which convergence was reached.
        patterns : np.ndarray
            The input patterns used for training.
        outputs : np.ndarray
            The target outputs corresponding to the input patterns.
        converged : bool
            Indicates whether the simulation converged.

        Returns
        -------
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
        Check the convergence of the activation function for a given pattern.

        Parameters
        ----------
        i : int
            The index of the pattern to check.

        Returns
        -------
        bool
            Whether the activation function converged for the given pattern.

        Notes
        -----
        Updates the `predictions` array based on the comparison of the activation 
        function results with the positive and negative targets.
        """
        fi = self.activation_function()
        self.predictions[i] = np.select(
            [fi >= self.positive_target, fi <= self.negative_target],
            [1, 0],
            default=2
        )

    def predict(self, patterns, outputs):
        """
        Make predictions based on saved conductances and the provided patterns.

        Parameters
        ----------
        patterns : np.ndarray
            The input patterns.
        outputs : np.ndarray
            The expected output values.

        Returns
        -------
        None

        Notes
        -----
        Calculates the logic currents for each pattern using the saved correct conductances 
        and checks for convergence. Prints the pattern, prediction, and expected result 
        for each input pattern.
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
