import os
import csv
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from dataclasses import dataclass
from matplotlib.cm import get_cmap
from datetime import datetime


@dataclass(frozen=True)
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


    def experimental_data(self, conductance_data : np.ndarray):
        raw_conductance_data = conductance_data
        first_value = raw_conductance_data[0]
        self.conductance_data = raw_conductance_data - first_value


    def shift(self) -> None:
        center_value = 0
        num_elements = 16  
        range_width = self.range
        rnd_shifts = np.random.uniform(low = center_value - range_width / 2, high = center_value + range_width / 2, size = num_elements)
        self.shifts = np.reshape(rnd_shifts, (4, 4))


    def conductance_init_rnd(self) -> None:
        self.conductances[0] = (self.conductance_data[0] + self.shifts) * self.multiplication_factor
        self.conductances[1] = 0
        self.all_conductances[0] = (self.conductance_data[0] + self.shifts) * self.multiplication_factor
        print("Initial Conductances:", self.all_conductances[0])
        print("Epoch:", 0)


    def voltage_array(self, pattern : np.ndarray, V0 = -0.1, V1 = 0.1) -> np.ndarray:
        voltages_j = np.array([V0 if i == 0 else V1 for i in pattern])
        return voltages_j
    

    def calculate_hardware_currents(self, pattern : np.ndarray) -> np.ndarray:
        applied_voltages = self.voltage_array(pattern)
        hardware_currents = applied_voltages.dot(self.conductances[0])
        return hardware_currents
    

    def calculate_logic_currents(self, pattern : np.ndarray) -> None:
        currents_array = self.calculate_hardware_currents(pattern)
        self.logic_currents = currents_array[::2] - currents_array[1::2]


    def activation_function(self) -> np.ndarray:
        activation = np.tanh(self.beta * self.logic_currents)
        return activation
    

    def activation_function_derivative(self) -> np.ndarray:
        derivative = self.beta / (np.cosh(self.beta * self.logic_currents))**2
        return derivative
    

    def calculate_delta_i(self, output : np.ndarray) -> np.ndarray:
        delta_i = np.empty([2])
        for index, target_output in enumerate(output):
            if target_output == 1:  
                delta = (self.positive_target - self.activation_function()[index]) * self.activation_function_derivative()[index]
                delta_i[index] = delta
            elif target_output == 0:
                delta = (self.negative_target - self.activation_function()[index]) * self.activation_function_derivative()[index]
                delta_i[index] = delta        
        return delta_i
    
    
    def calculate_Delta_ij(self, output : np.ndarray, pattern : np.ndarray, i) -> None:
        Delta_ij = np.outer(self.calculate_delta_i(output), self.voltage_array(pattern))
        self.all_delta_ij[i] = Delta_ij 


    def calculate_DeltaW_ij(self) -> np.ndarray:
        deltaW_ij = np.sign(np.sum(self.all_delta_ij, axis = 0))
        DeltaW_ij = np.transpose(deltaW_ij)
        return DeltaW_ij
    

    def update_weights(self, epoch) -> None:
        DeltaW_ij = self.calculate_DeltaW_ij()
        index_value_pairs = np.array([[index, value] for index, value in enumerate(self.conductance_data)])
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
                    self.conductances[0, i, j * 2] = (new_conductance + self.shifts[i, j * 2]) * self.multiplication_factor
                elif DeltaW_ij[i, j] < 0:
                    ind = self.conductances[1, i, j * 2 + 1].astype(int)
                    new_index = index[ind + 1]
                    new_conductance = value[ind + 1]
                    self.conductances[1, i, j * 2 + 1] = new_index
                    self.conductances[0, i, j * 2 + 1] = (new_conductance + self.shifts[i, j * 2 + 1]) * self.multiplication_factor
        self.all_conductances[epoch] = self.conductances[0] 


    def convergence_criterion(self, output: np.ndarray, i, epoch) -> bool:
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
    

    def check_convergence(self, i) -> bool:
        fi = self.activation_function()
        for index, element in enumerate(fi):
            if element >= self.positive_target:
                self.predictions[i, index] = 1   
            elif element <= self.negative_target:
                self.predictions[i, index] = 0
            else:
                self.predictions[i, index] = 2


    def total_error(self, epoch) -> None:
        total_error = np.sum(self.errors)
        self.all_errors[epoch] = total_error
        print("Total error:", total_error)


    def plot_final_weights(self):
        categories = ['1', '2', '3', '4']
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
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
        ax.ticklabel_format(style='sci', axis='z', scilimits=(0,0)) 
        ax.set_xlabel('Neuron')
        ax.set_ylabel('Input')
        ax.set_zlabel('Conductances')
        plt.show()
        return fig, ax


    def plot_conductances(self):
        rows, cols = self.conductances[0].shape
        cmap = get_cmap('tab20')
        num_plots = rows * cols
        fig, axes = plt.subplots(rows, cols, figsize=(15, 10), sharex=True)
        for j in range(cols):
            for i in range(rows):
                Wij = self.all_conductances[:, i, j]
                pulses = np.arange(self.epochs)
                ax = axes[i, j]
                color = cmap((i * cols + j) % num_plots) 
                ax.plot(pulses, Wij, 'o-', color=color, linewidth=2, label = f'Row = {i+1}\nColumn = {j+1}')
                ax.xaxis.set_major_formatter(ticker.ScalarFormatter(useMathText=True))
                ax.yaxis.set_major_formatter(ticker.ScalarFormatter(useMathText=True))
                ax.ticklabel_format(style='sci', axis='y', scilimits=(0,0))   
                ax.tick_params(labelsize=12)     
                ax.legend(loc = 'lower right')
                ax.grid()
        fig.text(0.5, 0.04, 'Epoch', ha='center', va='center', fontsize=20)
        fig.text(0.04, 0.5, 'Conductance (S)', ha='center', va='center', rotation='vertical', fontsize=20)
        plt.subplots_adjust(left=0.1, right=0.95, top=0.95, bottom=0.1, wspace=0.3, hspace=0.3)
        plt.show()
        return fig, axes
    

    def plot_weights(self, epochs):
        rows, cols = self.conductances[0].shape
        cmap = get_cmap('tab10')
        cols = 2
        num_plots = rows * cols
        fig, axes = plt.subplots(rows, cols, figsize=(12, 7), sharex=True)
        for j in range(cols):
            for i in range(rows):
                Wij = self.all_conductances[:epochs, i, j*2] - self.all_conductances[:epochs, i, j*2 + 1] 
                pulses = np.arange(epochs)
                ax = axes[i, j]
                color = cmap((i * cols + j) % num_plots) 
                ax.plot(pulses, Wij, 'o-', color=color, linewidth=3, label = f'Synapse = {i+1}\nNeuron = {j+1}')
                ax.xaxis.set_major_formatter(ticker.ScalarFormatter(useMathText=True))
                ax.yaxis.set_major_formatter(ticker.ScalarFormatter(useMathText=True))
                ax.ticklabel_format(style='sci', axis='y', scilimits=(0,0))   
                ax.tick_params(labelsize=12)     
                ax.legend(loc = 'best', fontsize=10)
                ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
                ax.grid()
        fig.text(0.5, 0.04, 'Epoch', ha='center', va='center', fontsize=17)
        fig.text(0.04, 0.5, 'Synaptic Weights (S)', ha='center', va='center', rotation='vertical', fontsize=17)
        plt.subplots_adjust(left=0.08, right=0.95, top=0.95, bottom=0.08, wspace=0.2, hspace=0.2)
        plt.show()
        return fig, axes


    def plot_error(self):
        pulses = np.arange(self.epochs)
        fig, ax = plt.subplots(1, 1, figsize=(10, 3))
        ax.plot(pulses, self.all_errors, 'o-', linewidth=2)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Total error')
        ax.set_title('Error evolution')
        ax.grid()
        plt.show()
        return fig


    def plot_results(self, pattern : np.ndarray, output : np.ndarray):
        pulses = np.arange(self.epochs)
        rows, cols = self.result[0].shape
        cmap = get_cmap('tab20')
        num_plots = rows * cols
        fig, axes = plt.subplots(rows, cols, figsize=(17, 8), sharex=True)
        for j in range(cols):
            for i in range(rows):
                results = self.result[:, i, j]
                ax = axes[i, j]
                color = cmap((i * cols + j) % num_plots) 
                patt = pattern[j]
                out = output[j]
                ax.plot(pulses, results, 'o-', color=color, linewidth=2, label = f'Pattern: {patt}\nOutput: {out}\nLogic Neuron: {i + 1}')
                if out[i] == 1:
                    ax.axhline(self.positive_target)
                else:
                    ax.axhline(self.negative_target)
                ax.xaxis.set_major_formatter(ticker.ScalarFormatter(useMathText=True))
                ax.yaxis.set_major_formatter(ticker.ScalarFormatter(useMathText=True))
                ax.ticklabel_format(style='sci', axis='y', scilimits=(0,0))   
                ax.tick_params(labelsize=12)     
                ax.legend(loc = 'best')
                ax.grid()
        fig.text(0.5, 0.04, 'Epoch', ha='center', va='center', fontsize=20)
        fig.text(0.04, 0.5, 'Activation Result', ha='center', va='center', rotation='vertical', fontsize=20)
        plt.subplots_adjust(left=0.1, right=0.95, top=0.95, bottom=0.1, wspace=0.3, hspace=0.3)
        plt.show()
        return fig, axes    

    
    def save_data(self, base_filename="simulation", converged=False):
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
            writer.writerow(["Epoch", " Total Error"] + [f" C_({i}{j})" for i in range(1,5) for j in range(1,5)])

            for epoch in range(self.epochs):
                conductances_flat = self.all_conductances[epoch].flatten().tolist()
                row = [epoch, self.all_errors[epoch]] + conductances_flat
                writer.writerow(row)


    def fit(self, patterns: np.ndarray, outputs: np.ndarray, plot: bool = True, conductance_data: np.ndarray = None, save_data: bool = False, filename: str = "simulation") -> None:
        self.experimental_data(conductance_data)
        self.shift()
        self.conductance_init_rnd()
        for epoch in range(1, self.epochs):
            converged = True
            for i in range(patterns.shape[0]):
                self.calculate_logic_currents(patterns[i])
                self.calculate_Delta_ij(outputs[i], patterns[i], i)
                if not self.convergence_criterion(outputs[i], i, epoch):
                    converged = False
            if converged:
                print(f"\nConvergence reached at epoch {epoch}")
                print("Conductances:", self.conductances[0])
                self.save_data(base_filename = filename, converged = converged)
                if plot:
                    self.plot_conductances()
                    self.plot_weights()
                    self.plot_error()
                    self.plot_results(patterns, outputs)
                    self.plot_final_weights()
                return epoch
            self.update_weights(epoch)
            self.total_error(epoch)
            print("Epoch:", epoch)
        print("Not converged")

        if save_data:
            self.save_data(base_filename = filename)

        if plot:
            self.plot_conductances()
            self.plot_weights()
            self.plot_error()
            self.plot_results(patterns, outputs)


    def predict(self, patterns, outputs):
        for i in range(patterns.shape[0]):
            self.calculate_logic_currents(patterns[i], self.saved_correct_conductances)
            self.check_convergence(i)
            print("Pattern:", patterns[i], "Prediction:", self.predictions[i], "Expected result:", outputs[i])