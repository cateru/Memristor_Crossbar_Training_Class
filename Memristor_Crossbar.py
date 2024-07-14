import numpy as np
from dataclasses import dataclass


@dataclass(frozen=True)
class Memristor_Crossbar:

    beta: float 
    positive_target: float
    negative_target: float
    range: float
    training_set_width: int
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
        self.all_delta_ij = np.empty([16, 2, 4])
        self.conductances = np.empty([2, 4, 4])
        self.all_conductances = np.empty([self.epochs, 4, 4])
        self.saved_correct_conductances = np.empty([4, 4])
        self.errors = np.empty([self.training_set_width, 2])
        self.all_errors = np.empty([self.epochs])     
        self.result = np.empty([self.epochs, 2, self.training_set_width])
        self.predictions = np.empty([self.test_set_width, 2])


    def experimental_data(self, conductance_data : np.ndarray):
        self.conductance_data = conductance_data


    def shift(self) -> None:
        center_value = 0
        num_elements = 16  
        range_width = self.range
        rnd_shifts = np.random.uniform(low = center_value - range_width / 2, high = center_value + range_width / 2, size = num_elements)
        self.shifts = np.reshape(rnd_shifts, (4, 4))


    def conductance_init_rnd(self) -> None:
        self.conductances[0:] = self.conductance_data[0] + self.shifts 
        self.conductances[1:] = 0
        self.all_conductances[0:] = self.conductance_data[0] + self.shifts
        print("Initial Conductances:", self.conductances[0])
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
                    self.conductances[0, i, j * 2] = new_conductance + self.shifts[i, j * 2]
                elif DeltaW_ij[i, j] < 0:
                    ind = self.conductances[1, i, j * 2 + 1].astype(int)
                    new_index = index[ind + 1]
                    new_conductance = value[ind + 1]
                    self.conductances[1, i, j * 2 + 1] = new_index
                    self.conductances[0, i, j * 2 + 1] = new_conductance + self.shifts[i, j * 2 + 1]
        self.all_conductances[epoch] = self.conductances[0] 
        print("Updated Conductances:", self.conductances[0])


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


    def fit(self, patterns: np.ndarray, outputs: np.ndarray, conductance_data: np.ndarray = None) -> None:
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
                return epoch
            self.update_weights(epoch)
            self.total_error(epoch)
            print("Epoch:", epoch)
        print("Not converged")


    def predict(self, patterns, outputs):
        for i in range(patterns.shape[0]):
            self.calculate_logic_currents(patterns[i], self.saved_correct_conductances)
            self.check_convergence(i)
            print("Pattern:", patterns[i], "Prediction:", self.predictions[i], "Expected result:", outputs[i])

