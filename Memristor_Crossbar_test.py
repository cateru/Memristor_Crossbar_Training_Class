import pytest
import numpy as np
import os
from datetime import datetime
import Memristor_Crossbar

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
            positive_target=self.positive_target,
            negative_target=self.negative_target,
            range=self.range,
            multiplication_factor=self.multiplication_factor,
            training_set_width=self.training_set_width,
            epochs=self.epochs
        )

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

    @pytest.mark.parametrize("conductance_data, _", [
        (np.array([1.0, 1.5, 2.0, 2.5, 3.0]), None),
    ])
    def test_conductance_init_rnd(self, conductance_data, _):
        self.crossbar.experimental_data(conductance_data)
        self.crossbar.shift()
        self.crossbar.conductance_init_rnd()
        expected_conductances = (conductance_data[0] + self.crossbar.shifts) * self.multiplication_factor
        np.testing.assert_array_almost_equal(self.crossbar.conductances[0], expected_conductances)

    @pytest.mark.parametrize("pattern, expected_voltages", [
        (np.array([0, 1, 0, 1]), np.array([-0.1, 0.1, -0.1, 0.1])),
    ])
    def test_voltage_array(self, pattern, expected_voltages):
        voltages = self.crossbar.voltage_array(pattern)
        np.testing.assert_array_equal(voltages, expected_voltages)

    @pytest.mark.parametrize("pattern", [np.array([0, 1, 0, 1])])
    def test_calculate_hardware_currents(self, pattern):
        self.crossbar.conductances[0] = np.random.rand(4, 4)
        hardware_currents = self.crossbar.calculate_hardware_currents(pattern)
        expected_currents = self.crossbar.voltage_array(pattern).dot(self.crossbar.conductances[0])
        np.testing.assert_array_almost_equal(hardware_currents, expected_currents)

    @pytest.mark.parametrize("pattern, conductances, expected_currents", [
        (np.array([0, 1, 0, 1]), np.array([[10, 10, 10, 10], [10, 10, 10, 10], [10, 10, 10, 10], [10, 10, 10, 10]]), np.array([0., 0., 0., 0.])),
    ])
    def test_calculate_hardware_currents_2(self, pattern, conductances, expected_currents):
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
        (np.array([5e-4, -5e-4]), np.array([1, -1])),
        (np.array([5e-8, -5e-8]), np.array([0, 0])),
    ])
    def test_activation_function(self, logic_currents, expected_activation):
        self.crossbar.logic_currents = logic_currents
        activation = self.crossbar.activation_function()
        np.testing.assert_array_almost_equal(activation, expected_activation)

    @pytest.mark.parametrize("logic_currents, expected_derivative", [
        (np.array([5e-4, -5e-4]), np.array([0, 0])),
        (np.array([5e-8, -5e-8]), np.array([1, 1])),
    ])
    def test_activation_function_derivative(self, logic_currents, expected_derivative):
        self.crossbar.logic_currents = logic_currents
        derivative = self.crossbar.activation_function_derivative()
        np.testing.assert_array_almost_equal(derivative, expected_derivative)

    @pytest.mark.parametrize("logic_currents", [(np.array([5e-4, -5e-4]))])
    def calculate_delta_i_test(self, logic_currents):
        self.crossbar.logic_currents = logic_currents
        output = np.array([1, 0])
        delta_i = self.crossbar.calculate_delta_i(output)
        expected_delta_i = np.empty([2])
        for index, target_output in enumerate(output):
            if target_output == 1:
                delta = (self.positive_target - 0) * 1
                expected_delta_i[index] = delta
            elif target_output == 0:
                delta = (self.negative_target - 0) * 1
                expected_delta_i[index] = delta
        np.testing.assert_array_almost_equal(delta_i, expected_delta_i)

    @pytest.mark.parametrize("pattern, output, logic_currents, expected_Delta_ij", [
        (np.array([0, 1, 0, 1]), np.array([1, 0]), np.array([5e-8, -5e-8]), np.array([[-0.075, 0.075, -0.075, 0.075], [0.075, -0.075, 0.075, -0.075]]))])
    def calculate_Delta_ij_test(self, pattern, output, logic_currents, expected_Delta_ij):
        i = 0
        self.crossbar.logic_currents = logic_currents
        self.crossbar.calculate_Delta_ij(output, pattern, i)
        np.testing.assert_array_almost_equal(self.crossbar.all_delta_ij[i], expected_Delta_ij)
 
    @pytest.mark.parametrize(
    "all_delta_ij, expected_deltaW_ij",
    [
        (
            np.array([
                [[0.075, 0.075, 0.075, -0.075],
                 [-0.075, -0.075, -0.075, 0.075]],
                [[-0.075, -0.075, 0.075, -0.075],
                 [0.075, 0.075, -0.075, 0.075]],
                [[-0.075, 0.075, 0.075, -0.075],
                 [0.075, -0.075, -0.075, 0.075]],
                [[-0.075, 0.075, 0.075, -0.075],
                 [0.075, -0.075, -0.075, 0.075]],
                [[-0.075, -0.075, 0.075, -0.075],
                 [0.075, 0.075, -0.075, 0.075]],
                [[0.075, 0.075, 0.075, -0.075],
                 [-0.075, -0.075, -0.075, 0.075]]
            ]),
            np.array([
                [-1, +1, +1, -1],
                [+1, -1, -1, +1]
            ])
        )
    ])
    def calculate_DeltaW_ij_test(self, all_delta_ij, expected_deltaW_ij):
        self.crossbar.all_delta_ij = all_delta_ij
        deltaW_ij = self.crossbar.calculate_DeltaW_ij()
        expected_DeltaW_ij = np.transpose(expected_deltaW_ij)
        np.testing.assert_array_almost_equal(deltaW_ij, expected_DeltaW_ij)

    @pytest.mark.parametrize("conductance_data", [np.array([1.0, 1.5, 2.0, 2.5, 3.0])])
    def update_weights_test(self, conductance_data):
        self.crossbar.experimental_data(conductance_data)
        self.crossbar.shift()
        self.crossbar.conductance_init_rnd()
        epoch = 1
        self.crossbar.update_weights(epoch)
        self.assertTrue(self.crossbar.all_conductances[epoch].any())

    @pytest.mark.parametrize(
    "conductance_data, shifts, unchanged_conductances, all_delta_ij",
    [
        (
            np.array([1.0, 1.5, 2.0, 2.5, 3.0]),
            np.array([[0, 0, 0, 0],
                      [0, 0, 0, 0],
                      [0, 0, 0, 0],
                      [0, 0, 0, 0]]),
            np.array([
                [[10, 10, 10, 10],
                 [10, 10, 10, 10],
                 [10, 10, 10, 10],
                 [10, 10, 10, 10]],
                [[0, 0, 0, 0],
                 [0, 0, 0, 0],
                 [0, 0, 0, 0],
                 [0, 0, 0, 0]]
            ]),
            np.array([
                [[0, 0, 0, 0],
                 [0, 0, -1, 0]],
                [[0, 0, 0, 0],
                 [0, 0, 0, 0]],
                [[0, 0, 0, 0],
                 [0, 0, 0, 0]],
                [[0, 0, 0, 0],
                 [0, 0, 0, 0]],
                [[+1, 0, 0, 0],
                 [0, 0, 0, 0]],
                [[0, 0, 0, 0]],
                [[0, 0, 0, 0]]
            ])
        )
    ])
    def update_weights_test_2(self,conductance_data, shifts, unchanged_conductances, all_delta_ij):
        self.crossbar.experimental_data(conductance_data)
        self.crossbar.shifts = shifts
        self.crossbar.conductance_init_rnd()
        self.crossbar.all_delta_ij = all_delta_ij
        epoch = 1
        self.crossbar.update_weights(epoch)
        for i in 4:
            for j in 4:
                if [(i == 1) and (j == 1)] or [(i == 3) and (j == 3)]:
                    assert unchanged_conductances[0, i, j] != self.crossbar.conductances[0, i, j]
                    assert self.crossbar.conductances[0, i, j] == 50
                    assert unchanged_conductances[1, i, j] != self.crossbar.conductances[1, i, j]
                    assert self.crossbar.conductances[1, i, j] == 1
                else:
                    assert unchanged_conductances[0, i, j] == self.crossbar.conductances[0, i, j]
                    assert unchanged_conductances[1, i, j] == self.crossbar.conductances[1, i, j]

    @pytest.mark.parametrize("logic_currents", [(np.array([5e-4, -5e-4]))])
    def convergence_criterion_test(self, logic_currents):
        output = np.array([1, 0])
        i = 0
        epoch = 1
        self.crossbar.logic_currents = logic_currents
        converged = self.crossbar.convergence_criterion(output, i, epoch)
        self.assertIsInstance(converged, bool)

    @pytest.mark.parametrize("output, logic_currents", [(np.array([1, 0])), (np.array([1e-4, -2e-5]))])
    def convergence_criterion_test_2(self, output, logic_currents):
        i = 0
        epoch = 1
        self.crossbar.logic_currents = logic_currents
        expected_convergence = True
        converged = self.crossbar.convergence_criterion(output, i, epoch)
        self.assertEqual(converged, expected_convergence)

    @pytest.mark.parametrize("conductance_data", [np.array([1.0, 1.5, 2.0, 2.5, 3.0])])
    def save_data_test(self, conductance_data):
        self.crossbar.experimental_data(conductance_data)
        self.crossbar.shift()
        self.crossbar.conductance_init_rnd()
        self.crossbar.update_weights(1)
        self.crossbar.save_data(base_filename="test_simulation", converged=False)
        self.assertTrue(os.path.exists(datetime.now().strftime("%d_%m_%Y") + "_test_simulation.npy"))

if __name__ == '__main__':
    pytest.main()



