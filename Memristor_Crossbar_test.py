import pytest
import numpy as np
import os
import copy
import shutil
from datetime import datetime
from unittest.mock import patch, MagicMock
from Memristor_Crossbar import Memristor_Crossbar

class Test_Memristor_Crossbar:

    def setup_method(self):
        """
        Set up the test method with initial parameters and create an instance of Memristor_Crossbar.
        """
        self.beta = 20000
        self.positive_target = 0.75
        self.negative_target = -0.75
        self.range = 0.001
        self.multiplication_factor = 10
        self.training_set_width = 6
        self.epochs = 5
        self.crossbar = Memristor_Crossbar(
            beta=self.beta,
            positive_target = self.positive_target,
            negative_target = self.negative_target,
            range=self.range,
            multiplication_factor = self.multiplication_factor,
            training_set_width = self.training_set_width,
            epochs = self.epochs
        )
        self.crossbar.predictions = np.zeros((self.crossbar.test_set_width, 2))

    def test_initialization(self):
        """
        Test the initialization of the Memristor_Crossbar instance.
        
        Asserts:
            The attributes of the crossbar are correctly set.
        """
        assert self.crossbar.beta == self.beta
        assert self.crossbar.positive_target == self.positive_target
        assert self.crossbar.negative_target == self.negative_target
        assert self.crossbar.range == self.range
        assert self.crossbar.multiplication_factor == self.multiplication_factor
        assert self.crossbar.training_set_width == self.training_set_width


    def test_experimental_data(self):
        """
        Test the experimental_data method of the Memristor_Crossbar instance.
        
        Asserts:
            The conductance_data is correctly transformed and stored.
        """
        conductance_data = np.array([1.0, 1.5, 2.0, 2.5, 3.0])
        expected_data = np.array([0.0, 0.5, 1.0, 1.5, 2.0])        
        self.crossbar.experimental_data(conductance_data)
        np.testing.assert_array_equal(self.crossbar.conductance_data, expected_data)


    def test_shift(self):
        """
        Test the shift method of the Memristor_Crossbar instance.
        
        Asserts:
            The shifts matrix has the correct shape and values within the expected range.
        """
        self.crossbar.shift()
        assert self.crossbar.shifts.shape == (4, 4)
        assert (self.crossbar.shifts >= -self.range / 2).all() and (self.crossbar.shifts <= self.range / 2).all()


    def test_conductance_init_rnd(self):
        """
        Test the conductance_init_rnd method of the Memristor_Crossbar instance.
        
        Asserts:
            The conductances are correctly initialized based on the experimental data and shifts.
        """
        conductance_data = np.array([1.0, 1.5, 2.0, 2.5, 3.0])
        shifts = np.array([
            [0., 0., 0., 0.],
            [0., 0., 0., 0.],
            [0., 0., 0., 0.],
            [0., 0., 0., 0.]
        ])
        expected_conductances = np.array([
            [0., 0., 0., 0.],
            [0., 0., 0., 0.],
            [0., 0., 0., 0.],
            [0., 0., 0., 0.]
        ])        
        self.crossbar.experimental_data(conductance_data)
        self.crossbar.shifts = shifts
        self.crossbar.conductance_init_rnd()
        np.testing.assert_array_almost_equal(self.crossbar.conductances[0], expected_conductances)


    def test_voltage_array(self):
        """
        Test the voltage_array method of the Memristor_Crossbar instance.
        
        Asserts:
            The voltage array is correctly calculated from the input pattern.
        """
        pattern = np.array([0, 1, 0, 1])
        expected_voltages = np.array([-0.1, 0.1, -0.1, 0.1])        
        voltages = self.crossbar.voltage_array(pattern)
        np.testing.assert_array_equal(voltages, expected_voltages)


    def test_calculate_hardware_currents(self):
        """
        Test the calculate_hardware_currents method of the Memristor_Crossbar instance.
        
        Asserts:
            The hardware currents are correctly calculated based on the input pattern and conductances.
        """
        pattern = np.array([0, 1, 0, 1])
        conductances = np.array([
            [10, 10, 10, 10],
            [10, 10, 10, 10],
            [10, 10, 10, 10],
            [10, 10, 10, 10]
        ])
        expected_currents = np.array([0., 0., 0., 0.])        
        self.crossbar.conductances[0] = conductances
        hardware_currents = self.crossbar.calculate_hardware_currents(pattern, conductances)
        np.testing.assert_array_almost_equal(hardware_currents, expected_currents)


    @pytest.mark.parametrize("pattern, conductances", [
        (np.array([0, 1, 0, 1]), np.random.rand(4, 4)),
        (np.array([0, 1, 0, 1]), np.array([[10, 10, 10, 10], [10, 10, 10, 10], [10, 10, 10, 10], [10, 10, 10, 10]])),
    ])
    def test_calculate_logic_currents(self, pattern, conductances):
        """
        Test the calculate_logic_currents method of the Memristor_Crossbar instance.
        
        Args:
            pattern (np.array): The input pattern.
            conductances (np.array): The conductances matrix.
        
        Asserts:
            The logic currents are correctly calculated from the input pattern and conductances.
        """
        self.crossbar.conductances[0] = conductances
        self.crossbar.calculate_logic_currents(pattern, conductances)
        expected_currents = self.crossbar.calculate_hardware_currents(pattern, conductances)
        expected_logic_currents = expected_currents[::2] - expected_currents[1::2]
        np.testing.assert_array_almost_equal(self.crossbar.logic_currents, expected_logic_currents)


    @pytest.mark.parametrize("logic_currents, expected_activation", [
        (np.array([5e-4, -5e-4]), np.array([1., -1.])),
        (np.array([5e-20, -5e-20]), np.array([0., -0.])),
    ])
    def test_activation_function(self, logic_currents, expected_activation):
        """
        Test the activation_function method of the Memristor_Crossbar instance.
        
        Args:
            logic_currents (np.array): The logic currents.
            expected_activation (np.array): The expected activation values.
        
        Asserts:
            The activation values are correctly calculated from the logic currents.
        """
        self.crossbar.logic_currents = logic_currents
        activation = self.crossbar.activation_function()
        np.testing.assert_array_almost_equal(activation, expected_activation)


    @pytest.mark.parametrize("logic_currents, expected_derivative", [
        (np.array([5e-4, -5e-4]), np.array([0.000165, 0.000165])),
        (np.array([5e-20, -5e-20]), np.array([20000, 20000])),
    ])
    def test_activation_function_derivative(self, logic_currents, expected_derivative):
        """
        Test the activation_function_derivative method of the Memristor_Crossbar instance.
        
        Args:
            logic_currents (np.array): The logic currents.
            expected_derivative (np.array): The expected derivatives.
        
        Asserts:
            The activation function derivatives are correctly calculated from the logic currents.
        """
        self.crossbar.logic_currents = logic_currents
        derivative = self.crossbar.activation_function_derivative()
        np.testing.assert_array_almost_equal(derivative, expected_derivative)


    def test_calculate_delta_i(self):
        """
        Test the calculate_delta_i method of the Memristor_Crossbar instance.
        
        Asserts:
            The delta_i values are correctly calculated based on the output and logic currents.
        """
        output = np.array([1, 0])
        logic_currents = np.array([5e-20, -5e-20])
        expected_delta_i = np.array([15000, -15000])    
        self.crossbar.logic_currents = logic_currents
        delta_i = self.crossbar.calculate_delta_i(output)
        np.testing.assert_array_almost_equal(delta_i, expected_delta_i)


    def test_calculate_Delta_ij(self):
        """
        Test the calculate_Delta_ij method of the Memristor_Crossbar instance.
        
        Asserts:
            The Delta_ij values are correctly calculated based on the output, pattern, and index.
        """
        pattern = np.array([0, 1, 0, 1])
        output = np.array([1, 0])
        logic_currents = np.array([5e-20, -5e-20])
        expected_Delta_ij = np.array([
            [-1500, 1500, -1500, 1500],
            [1500, -1500, 1500, -1500]
        ])
        
        i = 0
        self.crossbar.logic_currents = logic_currents
        self.crossbar.calculate_Delta_ij(output, pattern, i)
        np.testing.assert_array_almost_equal(self.crossbar.all_delta_ij[i], expected_Delta_ij)


    def test_calculate_DeltaW_ij(self):
        """
        Test the calculate_DeltaW_ij method of the Memristor_Crossbar instance.
        
        Asserts:
            The DeltaW_ij values are correctly calculated from all_delta_ij.
        """
        all_delta_ij = np.array([
            [
                [1500, 1500, 1500, -1500],
                [-1500, -1500, -1500, 1500]
            ],
            [
                [-1500, -1500, 1500, -1500],
                [1500, 1500, -1500, 1500]
            ],
            [
                [-1500, 1500, 1500, -1500],
                [1500, -1500, -1500, 1500]
            ],
            [
                [-1500, 1500, 1500, -1500],
                [1500, -1500, -1500, 1500]
            ],
            [
                [-1500, -1500, 1500, -1500],
                [1500, 1500, -1500, 1500]
            ],
            [
                [1500, 1500, 1500, -1500],
                [-1500, -1500, -1500, 1500]
            ]
        ])
        expected_deltaW_ij = np.array([
            [-1, +1, +1, -1],
            [+1, -1, -1, +1]
        ])        
        self.crossbar.all_delta_ij = all_delta_ij
        deltaW_ij = self.crossbar.calculate_DeltaW_ij()
        expected_DeltaW_ij = np.transpose(expected_deltaW_ij)
        np.testing.assert_array_almost_equal(deltaW_ij, expected_DeltaW_ij)


    def test_update_weights(self):
        """
        Test the update_weights method of the Memristor_Crossbar instance.
        
        Asserts:
            The conductances are correctly updated based on DeltaW_ij.
        """
        conductance_data = np.array([1.0, 1.5, 2.0, 2.5, 3.0])
        shifts = np.array([
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0]
        ])
        unchanged_conductances = np.array([
            [
                [0., 0., 0., 0.],
                [0., 0., 0., 0.],
                [0., 0., 0., 0.],
                [0., 0., 0., 0.]
            ],
            [
                [0, 0, 0, 0],
                [0, 0, 0, 0],
                [0, 0, 0, 0],
                [0, 0, 0, 0]
            ]
        ])
        DeltaW_ij = np.array([
            [+1, 0],
            [0, 0],
            [0, -1],
            [0, 0]
        ])        
        with patch.object(self.crossbar, 'calculate_DeltaW_ij', return_value=DeltaW_ij):
            self.crossbar.experimental_data(conductance_data)
            self.crossbar.shifts = shifts
            self.crossbar.conductances = copy.deepcopy(unchanged_conductances)
            epoch = 1
            self.crossbar.update_weights(epoch)
            for j in range(4):
                for i in range(4):
                    if (i == 0 and j == 0) or (i == 2 and j == 3):
                        assert unchanged_conductances[0, i, j] != self.crossbar.conductances[0, i, j]
                        assert self.crossbar.conductances[0, i, j] == 5
                        assert unchanged_conductances[1, i, j] != self.crossbar.conductances[1, i, j]
                        assert self.crossbar.conductances[1, i, j] == 1
                    else:
                        assert unchanged_conductances[0, i, j] == self.crossbar.conductances[0, i, j]
                        assert unchanged_conductances[1, i, j] == self.crossbar.conductances[1, i, j]


    def test_convergence_criterion(self):
        """
        Test the convergence_criterion method of the Memristor_Crossbar instance.
        
        Asserts:
            The convergence is correctly determined based on the logic currents and outputs.
        """
        output = np.array([1, 0])
        logic_currents = np.array([1e-4, -1e-4])        
        i = 0
        epoch = 1
        self.crossbar.logic_currents = logic_currents
        expected_convergence = True
        converged = self.crossbar.convergence_criterion(output, i, epoch)
        assert converged == expected_convergence


    def test_total_error(self):
        """
        Test the total error calculation of the Memristor_Crossbar instance.

        Asserts:
            The total error for the specified epoch matches the expected value.
        """
        errors = np.array([
            [0.5, 0, 0, 0, 0, 0],
            [0.5, 0, 0, 0, 0, 0]
        ])        
        epoch = 0
        expected_total_error = 1
        self.crossbar.errors = errors
        self.crossbar.total_error(epoch)
        total_error = self.crossbar.all_errors[epoch]
        assert total_error == expected_total_error

    def test_save_data(self):
        """
        Test the save_data method of the Memristor_Crossbar instance.

        Asserts:
            The file corresponding to the simulated data exists.
        """
        conductance_data = np.array([1.0, 1.5, 2.0, 2.5, 3.0])        
        self.crossbar.experimental_data(conductance_data)
        self.crossbar.shift()
        self.crossbar.conductance_init_rnd()
        self.crossbar.update_weights(1)
        self.crossbar.save_data(base_filename="test_simulation", converged=False)
        filename = "test_simulation_not_converged_data.csv"
        date_dir = datetime.now().strftime("%d-%m-%Y")
        full_path = os.path.join(date_dir, 'not_converged', filename)
        assert os.path.exists(full_path)
        shutil.rmtree(date_dir)


    @pytest.mark.parametrize("conductance_data, patterns, outputs, expected_converged", [
        (np.array([1.0, 1.5, 2.0, 2.5, 3.0]), 
        np.array([[0, 0, 0, 1], [0, 0, 1, 0], [0, 1, 1, 0], [1, 0, 0, 1], [1, 1, 0, 1], [1, 1, 1, 0]]), 
        np.array([[0, 1], [1, 0], [1, 0], [0, 1], [0, 1], [1, 0]]), 
        False),
        (np.array([1.0, 1.5, 2.0, 2.5, 3.0]), 
        np.array([[0, 0, 0, 1], [0, 0, 1, 0], [0, 1, 1, 0], [1, 0, 0, 1], [1, 1, 0, 1], [1, 1, 1, 0]]), 
        np.array([[0, 1], [1, 0], [1, 0], [0, 1], [0, 1], [1, 0]]), 
        True),
    ])
    def test_fit(self, conductance_data, patterns, outputs, expected_converged):
        """
        Test the fit method of the Memristor_Crossbar instance.

        Asserts:
            All mocked methods are called correctly based on the parameters.
        """
        self.crossbar.experimental_data = MagicMock()
        self.crossbar.shift = MagicMock()
        self.crossbar.conductance_init_rnd = MagicMock()
        self.crossbar.calculate_Delta_ij = MagicMock()
        self.crossbar.calculate_logic_currents = MagicMock()
        self.crossbar.save_data = MagicMock()
        self.crossbar.plot_conductances = MagicMock()
        self.crossbar.plot_weights = MagicMock()
        self.crossbar.plot_error = MagicMock()
        self.crossbar.plot_results = MagicMock()
        self.crossbar.plot_final_weights = MagicMock()
        self.crossbar.update_weights = MagicMock()
        self.crossbar.total_error = MagicMock()
        self.crossbar.convergence_criterion = MagicMock(return_value = expected_converged)
        self.crossbar.fit(patterns = patterns, outputs = outputs, plot = True, conductance_data = conductance_data)
        self.crossbar.experimental_data.assert_called_once_with(conductance_data)
        self.crossbar.shift.assert_called_once()
        self.crossbar.conductance_init_rnd.assert_called_once()
        self.crossbar.plot_conductances.assert_called_once()
        self.crossbar.plot_weights.assert_called_once()
        self.crossbar.plot_error.assert_called_once()
        self.crossbar.plot_results.assert_called_once()        
        self.crossbar.calculate_logic_currents.assert_called()
        self.crossbar.calculate_Delta_ij.assert_called()
        self.crossbar.convergence_criterion.assert_called()
        if expected_converged:            
            self.crossbar.plot_conductances.assert_called_once()
            self.crossbar.plot_final_weights.assert_called_once()
            self.crossbar.save_data.assert_not_called()
        else: 
            self.crossbar.update_weights.assert_called()
            self.crossbar.total_error.assert_called()
            self.crossbar.save_data.assert_not_called() 


    @pytest.mark.parametrize("logic_currents, expected_predictions", [
        (np.array([5e-4, -5e-4]),np.array([1, 0])),
        (np.array([-5e-20, 5e-6]),np.array([2, 2])),
    ])
    def test_check_convergence(self, logic_currents, expected_predictions):
        """
        Test the check_convergence method of the Memristor_Crossbar instance.

        Asserts:
            The output is calculated correctly from the logic currents and matches the expected one.
        """
        self.crossbar.logic_currents = logic_currents
        self.crossbar.check_convergence(0)
        np.testing.assert_array_equal(self.crossbar.predictions[0], expected_predictions)


    def test_predict(self):
        """
        Test the predict method of the Memristor_Crossbar instance.

        Asserts:
            The predictions made by the `predict` method match the expected output.
        """
        conductances = np.array([
            [0, 0, 0, 0], 
            [0, 0, 0, 0], 
            [0, 0, 0, 0], 
            [0, 0, 0, 0]
        ])
        pattern = np.array([
            [0, 0, 0, 0],
            [0, 0, 1, 1],
            [0, 1, 0, 0],
            [0, 1, 0, 1],
            [0, 1, 1, 1],
            [1, 0, 0, 0],
            [1, 0, 1, 0],
            [1, 0, 1, 1],
            [1, 1, 0, 0],
            [1, 1, 1, 1]
        ])
        output = np.array([
            [0, 0],
            [0, 0],
            [1, 0],
            [0, 0],
            [1, 0],
            [0, 1],
            [0, 0],
            [0, 1],
            [0, 0],
            [0, 0]
        ])
        expected_output = np.array([
            [2, 2], 
            [2, 2], 
            [2, 2], 
            [2, 2], 
            [2, 2], 
            [2, 2], 
            [2, 2], 
            [2, 2], 
            [2, 2], 
            [2, 2]
        ])        
        self.crossbar.saved_correct_conductances = conductances
        self.crossbar.predict(pattern, output)
        np.testing.assert_array_equal(self.crossbar.predictions, expected_output)


if __name__ == '__main__':
    pytest.main()
