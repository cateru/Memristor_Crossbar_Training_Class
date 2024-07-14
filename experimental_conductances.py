import pandas as pd

experimental_data = pd.read_csv('SP291D1NR100.dat' , skiprows=11, header=None, sep=',', names=['Time stamp (s)', 'Voltage (V)', 'Current (A)', 'half_Voltage (V)', 'half_Current (A)', 'Resistance (Ohm)'], na_values=[''])
Voltage = experimental_data['Voltage (V)'].to_numpy()
Resistance = experimental_data['Resistance (Ohm)'].to_numpy()
Resistance = Resistance[(Voltage == 2) | (Voltage == -2)]
conductance_data = 1/Resistance