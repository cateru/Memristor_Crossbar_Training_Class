import numpy as np
from dataclasses import dataclass


@dataclass(frozen=True)
class Memristor_Crossbar:

    beta: float 
    positive_target: float
    negative_target: float
    range: float
    epochs: int = 48
    
    conductance_data: np.ndarray = None
    shifts: np.ndarray = None
    logic_currents: np.ndarray = None
    all_delta_ij: np.ndarray = None
    conductances: np.ndarray = None
    all_conductances: np.ndarray = None

    def __post_init__(self):

        self.conductance_data = []
        self.shifts = np.empty([4, 4])
        self.logic_currents = np.empty([2])
        self.all_delta_ij = np.empty([16, 2, 4])
        self.conductances = np.empty([2, 4, 4])
        self.all_conductances = np.empty([self.epochs, 4, 4])


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