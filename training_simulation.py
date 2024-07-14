import numpy as np
import pandas as pd
import Memristor_Crossbar

experimental_data = pd.read_csv('SP291D1NR100.dat' , skiprows=11, header=None, sep=',', names=['Time stamp (s)', 'Voltage (V)', 'Current (A)', 'half_Voltage (V)', 'half_Current (A)', 'Resistance (Ohm)'], na_values=[''])
Voltage = experimental_data['Voltage (V)'].to_numpy()
Resistance = experimental_data['Resistance (Ohm)'].to_numpy()
Resistance = Resistance[(Voltage == 2) | (Voltage == -2)]
conductance_data = 1/Resistance


train_set = np.array([[0, 0, 0, 1],
                      [0, 0, 1, 0],
                      [0, 1, 1, 0],
                      [1, 0, 0, 1],
                      [1, 1, 0, 1],
                      [1, 1, 1, 0]])


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


train_outputs = np.array([[0, 1],
                          [1, 0],
                          [1, 0],
                          [0, 1],
                          [0, 1],
                          [1, 0]])


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


model = Memristor_Crossbar(beta = 20000, positive_target = 0.75, negative_target = -0.75, range = 0.001, multiplication_factor = 10)

model.fit(train_set, train_outputs, conductance_data)