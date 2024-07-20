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
        self.beta = 20000
        self.positive_target = 0.75
        self.negative_target = -0.75
        self.range = 0.001
        self.multiplication_factor = 10
        self.training_set_width = 6
        self.epochs = 48
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
        assert self.crossbar.beta == self.beta
        assert self.crossbar.positive_target == self.positive_target
        assert self.crossbar.negative_target == self.negative_target
        assert self.crossbar.range == self.range
        assert self.crossbar.multiplication_factor == self.multiplication_factor
        assert self.crossbar.training_set_width == self.training_set_width

    @pytest.mark.parametrize("conductance_data, expected_data", [
        (np.array([1.0, 1.5, 2.0, 2.5, 3.0]), np.array([0.0, 0.5, 1.0, 1.5, 2.0])),
    ])
    def test_experimental_data(self, conductance_data, expected_data):
        self.crossbar.experimental_data(conductance_data)
        np.testing.assert_array_equal(self.crossbar.conductance_data, expected_data)

    def test_shift(self):
        self.crossbar.shift()
        assert self.crossbar.shifts.shape == (4, 4)
        assert (self.crossbar.shifts >= -self.range / 2).all() and (self.crossbar.shifts <= self.range / 2).all()

    @pytest.mark.parametrize("conductance_data, shifts, expected_conductances", [(np.array([1.0, 1.5, 2.0, 2.5, 3.0]), 
            np.array([[0., 0., 0., 0.],
                      [0., 0., 0., 0.],
                      [0., 0., 0., 0.],
                      [0., 0., 0., 0.]]),
            np.array([[0., 0., 0., 0.],
                      [0., 0., 0., 0.],
                      [0., 0., 0., 0.],
                      [0., 0., 0., 0.]])
    )])
    def test_conductance_init_rnd(self, conductance_data, shifts, expected_conductances):
        self.crossbar.experimental_data(conductance_data)
        self.crossbar.shifts = shifts
        self.crossbar.conductance_init_rnd()
        expected_conductances = expected_conductances
        np.testing.assert_array_almost_equal(self.crossbar.conductances[0], expected_conductances)

    @pytest.mark.parametrize("pattern, expected_voltages", [
        (np.array([0, 1, 0, 1]), np.array([-0.1, 0.1, -0.1, 0.1])),
    ])
    def test_voltage_array(self, pattern, expected_voltages):
        voltages = self.crossbar.voltage_array(pattern)
        np.testing.assert_array_equal(voltages, expected_voltages)

    @pytest.mark.parametrize("pattern, conductances, expected_currents", [
        (np.array([0, 1, 0, 1]), np.array([[10, 10, 10, 10], [10, 10, 10, 10], [10, 10, 10, 10], [10, 10, 10, 10]]), np.array([0., 0., 0., 0.])),
    ])
    def test_calculate_hardware_currents(self, pattern, conductances, expected_currents):
        self.crossbar.conductances[0] = conductances
        hardware_currents = self.crossbar.calculate_hardware_currents(pattern)
        np.testing.assert_array_almost_equal(hardware_currents, expected_currents)

    @pytest.mark.parametrize("pattern, conductances", [
        (np.array([0, 1, 0, 1]), np.random.rand(4, 4)),
        (np.array([0, 1, 0, 1]), np.array([[10, 10, 10, 10], [10, 10, 10, 10], [10, 10, 10, 10], [10, 10, 10, 10]])),
    ])
    def test_calculate_logic_currents(self, pattern, conductances):
        self.crossbar.conductances[0] = conductances
        self.crossbar.calculate_logic_currents(pattern)
        expected_currents = self.crossbar.calculate_hardware_currents(pattern)
        expected_logic_currents = expected_currents[::2] - expected_currents[1::2]
        np.testing.assert_array_almost_equal(self.crossbar.logic_currents, expected_logic_currents)

    @pytest.mark.parametrize("logic_currents, expected_activation", [
        (np.array([5e-4, -5e-4]), np.array([1., -1.])),
        (np.array([5e-20, -5e-20]), np.array([0., -0.])),
    ])
    def test_activation_function(self, logic_currents, expected_activation):
        self.crossbar.logic_currents = logic_currents
        activation = self.crossbar.activation_function()
        np.testing.assert_array_almost_equal(activation, expected_activation)

    @pytest.mark.parametrize("logic_currents, expected_derivative", [
        (np.array([5e-4, -5e-4]), np.array([0.000165, 0.000165])),
        (np.array([5e-20, -5e-20]), np.array([20000, 20000])),
    ])
    def test_activation_function_derivative(self, logic_currents, expected_derivative):
        self.crossbar.logic_currents = logic_currents
        derivative = self.crossbar.activation_function_derivative()
        np.testing.assert_array_almost_equal(derivative, expected_derivative)

    @pytest.mark.parametrize("output, logic_currents, expected_delta_i", [(np.array([1, 0]), np.array([5e-20, -5e-20]), np.array([15000, -15000]))])
    def test_calculate_delta_i(self, output,  logic_currents, expected_delta_i):
        self.crossbar.logic_currents = logic_currents
        delta_i = self.crossbar.calculate_delta_i(output)
        np.testing.assert_array_almost_equal(delta_i, expected_delta_i)

    @pytest.mark.parametrize("pattern, output, logic_currents, expected_Delta_ij", [(np.array([0, 1, 0, 1]), np.array([1, 0]), np.array([5e-20, -5e-20]), np.array([[-1500, 1500, -1500, 1500], [1500, -1500, 1500, -1500]]))])
    def test_calculate_Delta_ij(self, pattern, output, logic_currents, expected_Delta_ij):
        i = 0
        self.crossbar.logic_currents = logic_currents
        self.crossbar.calculate_Delta_ij(output, pattern, i)
        np.testing.assert_array_almost_equal(self.crossbar.all_delta_ij[i], expected_Delta_ij)
 
    @pytest.mark.parametrize(
    "all_delta_ij, expected_deltaW_ij",
    [
        (
            np.array([
                [[1500, 1500, 1500, -1500],
                 [-1500, -1500, -1500, 1500]],
                [[-1500, -1500, 1500, -1500],
                 [1500, 1500, -1500, 1500]],
                [[-1500, 1500, 1500, -1500],
                 [1500, -1500, -1500, 1500]],
                [[-1500, 1500, 1500, -1500],
                 [1500, -1500, -1500, 1500]],
                [[-1500, -1500, 1500, -1500],
                 [1500, 1500, -1500, 1500]],
                [[1500, 1500, 1500, -1500],
                 [-1500, -1500, -1500, 1500]]
            ]),
            np.array([
                [-1, +1, +1, -1],
                [+1, -1, -1, +1]
            ])
        )
    ])
    def test_calculate_DeltaW_ij(self, all_delta_ij, expected_deltaW_ij):
        self.crossbar.all_delta_ij = all_delta_ij
        deltaW_ij = self.crossbar.calculate_DeltaW_ij()
        expected_DeltaW_ij = np.transpose(expected_deltaW_ij)
        np.testing.assert_array_almost_equal(deltaW_ij, expected_DeltaW_ij)

    @pytest.mark.parametrize(
    "conductance_data, shifts, unchanged_conductances, DeltaW_ij",
    [
        (
            np.array([1.0, 1.5, 2.0, 2.5, 3.0]),
            np.array([[0, 0, 0, 0],
                      [0, 0, 0, 0],
                      [0, 0, 0, 0],
                      [0, 0, 0, 0]]),
            np.array([
                [[0., 0., 0., 0.],
                 [0., 0., 0., 0.],
                 [0., 0., 0., 0.],
                 [0., 0., 0., 0.]],
                [[0, 0, 0, 0],
                 [0, 0, 0, 0],
                 [0, 0, 0, 0],
                 [0, 0, 0, 0]]
            ]),
            np.array([
                        [+1, 0],
                        [0, 0],
                        [0, -1],
                        [0, 0]
                    ])
        )
    ])
    def test_update_weights(self, conductance_data, shifts, unchanged_conductances, DeltaW_ij):
        with patch.object(self.crossbar, 'calculate_DeltaW_ij', return_value = DeltaW_ij):
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

    @pytest.mark.parametrize("output, logic_currents", [(np.array([1, 0]), np.array([1e-4, -1e-4]))])
    def test_convergence_criterion(self, output, logic_currents):
        i = 0
        epoch = 1
        self.crossbar.logic_currents = logic_currents
        expected_convergence = True
        converged = self.crossbar.convergence_criterion(output, i, epoch)
        assert converged == expected_convergence

    @pytest.mark.parametrize("errors", [np.array([[0.5, 0, 0, 0, 0, 0],[0.5, 0, 0, 0, 0, 0]])])
    def test_total_error(self, errors):
        epoch = 0
        expected_total_error = 1
        self.crossbar.errors = errors
        self.crossbar.total_error(epoch)
        total_error = self.crossbar.all_errors[epoch]
        assert total_error == expected_total_error

    @pytest.mark.parametrize("conductance_data", [np.array([1.0, 1.5, 2.0, 2.5, 3.0])])
    def test_save_data(self, conductance_data):
        self.crossbar.experimental_data(conductance_data)
        self.crossbar.shift()
        self.crossbar.conductance_init_rnd()
        self.crossbar.update_weights(1)
        self.crossbar.save_data(base_filename="test_simulation", converged = False)
        filename = "test_simulation_not_converged_data.csv"
        date_dir = datetime.now().strftime("%d-%m-%Y")
        full_path = os.path.join(date_dir, 'not_converged', filename)
        assert os.path.exists(full_path) == True
        shutil.rmtree(date_dir)

if __name__ == '__main__':
    pytest.main()
