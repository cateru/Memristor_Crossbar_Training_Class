import pytest
import numpy as np
import os
import copy
import shutil
from datetime import datetime
from unittest.mock import patch, MagicMock
from memristor_crossbar.Memristor_Crossbar import Memristor_Crossbar


class Test_Memristor_Crossbar:

    def setup_method(self):
        """
        Configure the test environment and create an instance of `Memristor_Crossbar`.

        This method sets up the necessary parameters and data structures for testing.
        It defines key attributes such as `beta`, target values, and the multiplication factor.
        Additionally, it creates an instance of `Memristor_Crossbar` with these parameters
        and prepares an empty `predictions` array.

        Parameters
        ----------
        - `beta` : float
            The scaling factor for the activation function.
        - `positive_target` : float
            The target value for positive outputs.
        - `negative_target` : float
            The target value for negative outputs.
        - `multiplication_factor` : float
            A factor used for scaling conductances.
        - `predictions` : np.ndarray
            A zero-filled array of shape (`test_set_width`, 2) to store predictions.

        Returns
        -------
        None
        """
        self.beta = 1e6
        self.positive_target = 0.75
        self.negative_target = -0.75
        self.multiplication_factor = 10
        self.crossbar = Memristor_Crossbar(
            beta=self.beta,
            positive_target=self.positive_target,
            negative_target=self.negative_target,
            multiplication_factor=self.multiplication_factor,
        )

    def test_initialization(self):
        """
        Verify the attributes and structure of the `Memristor_Crossbar` instance.

        This test ensures that the manually assigned attributes and default parameters 
        of the `Memristor_Crossbar` class are correctly set. It also validates the shapes 
        and types of key attributes such as arrays and matrices.

        Returns
        -------
        None
        """
        assert self.crossbar.beta == self.beta
        assert self.crossbar.positive_target == self.positive_target
        assert self.crossbar.negative_target == self.negative_target
        assert self.crossbar.multiplication_factor == self.multiplication_factor
        assert self.crossbar.num_elements == self.crossbar.number_of_rows * self.crossbar.number_of_columns
        assert self.crossbar.max_value == 10**-7
        assert self.crossbar.min_value == 10**-8
        assert self.crossbar.division_factor == 5
        assert self.crossbar.training_set_width == 6
        assert self.crossbar.epochs == 51
        assert self.crossbar.number_of_neurons == 2
        assert self.crossbar.number_of_rows == 4
        assert self.crossbar.number_of_columns == 4
        assert self.crossbar.test_set_width == (16 - self.crossbar.training_set_width) 
        assert isinstance(self.crossbar.predictions, np.ndarray)
        assert self.crossbar.predictions.shape == (self.crossbar.test_set_width, self.crossbar.number_of_neurons)
        assert self.crossbar.shifts.shape == (self.crossbar.number_of_rows, self.crossbar.number_of_columns)
        assert self.crossbar.logic_currents.shape == (self.crossbar.number_of_neurons,)
        assert self.crossbar.all_delta_ij.shape == (self.crossbar.training_set_width, self.crossbar.number_of_neurons, self.crossbar.number_of_rows)
        assert self.crossbar.conductances.shape == (self.crossbar.number_of_neurons, self.crossbar.number_of_rows, self.crossbar.number_of_columns)
        assert self.crossbar.all_conductances.shape == (self.crossbar.epochs, self.crossbar.number_of_rows, self.crossbar.number_of_columns)
        assert self.crossbar.saved_correct_conductances.shape == (self.crossbar.number_of_rows, self.crossbar.number_of_columns)
        assert self.crossbar.errors.shape == (self.crossbar.training_set_width, self.crossbar.number_of_neurons)
        assert self.crossbar.all_errors.shape == (self.crossbar.epochs,)
        assert self.crossbar.result.shape == (self.crossbar.epochs, self.crossbar.number_of_neurons, self.crossbar.training_set_width)     

    def test_experimental_data(self):
        """
        Test the `experimental_data` method of the `Memristor_Crossbar` instance. Verifies that conductance data is correctly transformed and stored.

        Returns
        -------
        None
        """
        conductance_data = np.array([1.0, 1.5, 2.0, 2.5, 3.0])
        expected_data = np.array([0.0, 0.5, 1.0, 1.5, 2.0])
        self.crossbar.experimental_data(conductance_data)
        np.testing.assert_array_equal(self.crossbar.conductance_data, expected_data)

    def test_shift_lognormal(self):
        """
        Test the `shift_lognormal` method of the `Memristor_Crossbar` instance without a seed. Controls that the shifts matrix has the correct shape (4, 4) and that the mean of the shifts is approximately 0 (within tolerance).

        Returns
        -------
        None
        """
        self.crossbar.shift_lognormal()
        assert self.crossbar.shifts.shape == (4, 4)
        assert np.isclose(np.mean(self.crossbar.shifts), 0, atol=1e-6)

    def test_shift_lognormal_seed(self):
        """
        Test the `shift_lognormal` method of the `Memristor_Crossbar` instance with a seed. In particular it verifies that the shifts matrix has the correct shape (4, 4) and that the shifts match the expected predefined values.

        Returns
        -------
        None
        """
        seed = 5
        np.random.seed(seed) 
        self.crossbar.shift_lognormal() 
        assert self.crossbar.shifts.shape == (4, 4)
        expected_shifts = np.array(
            [
                [3.63488861e-08, 2.36501542e-08, 1.00000000e-07, 2.47489063e-08],
                [3.03383895e-08, 6.56920543e-08, 1.66960742e-08, 2.02848130e-08],
                [3.16705896e-08, 2.36638560e-08, 1.39068835e-08, 2.54268044e-08],
                [2.32696858e-08, 3.96412432e-08, 1.00000000e-08, 1.89988454e-08],
            ]
        )
        np.testing.assert_array_almost_equal(self.crossbar.shifts, expected_shifts)

    def test_custom_shift(self):
        """
        Test the `custom_shift` method of the `Memristor_Crossbar` instance checking that the custom shifts are correctly set in the `Memristor_Crossbar` instance.

        Returns
        -------
        None
        """
        custom_shifts = np.array(
            [
                [0.1, 0.2, 0.3, 0.4],
                [0.5, 0.6, 0.7, 0.8],
                [0.9, 1.0, 1.1, 1.2],
                [1.3, 1.4, 1.5, 1.6],
            ]
        )
        expected_shifts = custom_shifts
        self.crossbar.custom_shift(custom_shifts)
        np.testing.assert_array_equal(self.crossbar.shifts, expected_shifts)

    def test_conductance_init(self):
        """
        Test `conductance_init` with zero shifts checking that the conductances are set to zero, not matching expected values.

        Returns
        -------
        None
        """
        conductance_data = np.array([1.0, 1.5, 2.0, 2.5, 3.0])
        shifts = np.array(
            [
                [0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0],
            ]
        )
        expected_conductances = np.array(
            [
                [0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0],
            ]
        )
        self.crossbar.experimental_data(conductance_data)
        self.crossbar.shifts = shifts
        self.crossbar.conductance_init()
        np.testing.assert_array_almost_equal(
            self.crossbar.conductances[0], expected_conductances
        )

    def test_conductance_init_seed(self):
        """
        Test conductance_init with a fixed seed verifying that the the conductances match the expected values based on the set seed.

        Returns
        -------
        None
        """
        conductance_data = np.array([1.0, 1.5, 2.0, 2.5, 3.0])
        seed = 5
        np.random.seed(seed) 
        expected_conductances = np.array(
            [
                [3.63488861e-07, 2.36501542e-07, 1.00000000e-06, 2.47489063e-07],
                [3.03383895e-07, 6.56920543e-07, 1.66960742e-07, 2.02848130e-07],
                [3.16705896e-07, 2.36638560e-07, 1.39068835e-07, 2.54268044e-07],
                [2.32696858e-07, 3.96412432e-07, 1.00000000e-07, 1.89988454e-07],
            ]
        )
        self.crossbar.experimental_data(conductance_data)
        self.crossbar.shift_lognormal()
        self.crossbar.conductance_init()
        np.testing.assert_array_almost_equal(
            self.crossbar.conductances[0], expected_conductances
        )

    def test_voltage_array(self):
        """
        Test the voltage_array method of the Memristor_Crossbar instance controlling that the voltage array is correctly calculated from the input pattern.

        Returns
        -------
        None
        """
        pattern = np.array([0, 1, 0, 1])
        expected_voltages = np.array([-0.1, 0.1, -0.1, 0.1])
        voltages = self.crossbar.voltage_array(pattern)
        np.testing.assert_array_equal(voltages, expected_voltages)

    def test_calculate_hardware_currents(self):
        """
        Test the calculate_hardware_currents method of the Memristor_Crossbar instance verifying that the hardware currents are correctly calculated based on the voltages and conductances.

        Returns
        -------
        None
        """
        pattern = np.array([1, 0, 1, 1])
        conductances = np.array(
            [
                [0.2, -1.0, 0.8, -0.6],
                [0.4, -0.2, 1.0, -0.8],
                [0.6, -0.4, 1.2, -1.0],
                [0.8, -0.6, 1.4, -1.2],
            ]
        )
        expected_currents = np.array([0.12, -0.18, 0.24, -0.2])
        self.crossbar.conductances[0] = conductances
        hardware_currents = self.crossbar.calculate_hardware_currents(
            pattern, conductances
        )
        np.testing.assert_array_almost_equal(hardware_currents, expected_currents)

    def test_calculate_logic_currents(self):
        """
        Test the calculate_logic_currents method of the Memristor_Crossbar instance. This method verifies that the logic currents are accurately computed based on the input pattern and conductances, expected to be [1, 2].

        Args
        ----
        - pattern (np.array): The input pattern for testing.
        - conductances (np.array): The conductance matrix used in the test.

        Returns
        -------
        None
        """
        pattern = np.array([0, 0, 0, 1])
        conductances = np.array(
            [[10, 10, 10, 10], [10, 10, 10, 10], [10, 10, 10, 10], [20, 10, 30, 10]]
        )
        self.crossbar.conductances[0] = conductances
        self.crossbar.calculate_logic_currents(pattern, conductances)
        expected_logic_currents = np.array([1, 2])
        np.testing.assert_array_almost_equal(
            self.crossbar.logic_currents, expected_logic_currents
        )

    @pytest.mark.parametrize(
        "pattern, conductances",
        [
            (np.array([0, 1, 0, 1]), np.random.rand(4, 4)),
            (
                np.array([0, 1, 0, 1]),
                np.array(
                    [
                        [10, 10, 10, 10],
                        [10, 10, 10, 10],
                        [10, 10, 10, 10],
                        [10, 10, 10, 10],
                    ]
                ),
            ),
        ],
    )
    def test_calculate_logic_currents_rnd(self, pattern, conductances):
        """
        Test the calculate_logic_currents method with various input patterns and conductance matrices to verify that the logic currents are correctly calculated from the input pattern and conductances.

        Args
        ----
        - pattern (np.array): The input pattern used for testing.
        - conductances (np.array): The conductance matrix for the test.

        Returns
        -------
        None
        """
        self.crossbar.conductances[0] = conductances
        self.crossbar.calculate_logic_currents(pattern, conductances)
        expected_currents = self.crossbar.calculate_hardware_currents(
            pattern, conductances
        )
        expected_logic_currents = expected_currents[::2] - expected_currents[1::2]
        np.testing.assert_array_almost_equal(
            self.crossbar.logic_currents, expected_logic_currents
        )

    @pytest.mark.parametrize(
        "logic_currents, expected_activation",
        [
            (np.array([1e-5, -1e-5]), np.array([1.0, -1.0])),
            (np.array([1e-20, -1e-20]), np.array([0.0, -0.0])),
        ],
    )
    def test_activation_function(self, logic_currents, expected_activation):
        """
        Test the activation_function method of the Memristor_Crossbar instance ensuring that the activation values are correctly calculated from the logic currents.         

        Args
        ----
        - logic_currents (np.array): The logic currents.
        - expected_activation (np.array): The expected activation values.

        Returns
        -------
        None
        """
        self.crossbar.logic_currents = logic_currents
        activation = self.crossbar.activation_function()
        np.testing.assert_array_almost_equal(activation, expected_activation)

    @pytest.mark.parametrize(
        "logic_currents, expected_derivative",
        [
            (np.array([1e-5, -1e-5]), np.array([0.008245, 0.008245])),
            (np.array([1e-20, -1e-20]), np.array([1e6, 1e6])),
        ],
    )
    def test_activation_function_derivative(self, logic_currents, expected_derivative):
        """
        Test the activation_function_derivative method of the Memristor_Crossbar instance making sure that the activation function derivatives are correctly calculated from the logic currents.

        Args
        ----
        - logic_currents (np.array): The logic currents.
        - expected_derivative (np.array): The expected derivatives.

        Returns
        -------
        None
        """
        self.crossbar.logic_currents = logic_currents
        derivative = self.crossbar.activation_function_derivative()
        np.testing.assert_array_almost_equal(derivative, expected_derivative)

    def test_calculate_delta_i(self):
        """
        Test the calculate_delta_i method of the Memristor_Crossbar instance checking that the delta_i values are correctly calculated based on the output and logic currents.

        Returns
        -------
        None
        """
        output = np.array([1, 0])
        logic_currents = np.array([1e-20, -1e-20])
        expected_delta_i = np.array([750000, -750000])
        self.crossbar.logic_currents = logic_currents
        delta_i = self.crossbar.calculate_delta_i(output)
        np.testing.assert_array_almost_equal(delta_i, expected_delta_i)

    def test_calculate_Delta_ij(self):
        """
        Test the calculate_Delta_ij method of the Memristor_Crossbar instance checking that the Delta_ij values are correctly calculated based on the output, pattern, and index.

        Returns
        -------
        None
        """
        pattern = np.array([0, 1, 0, 1])
        output = np.array([1, 0])
        logic_currents = np.array([1e-20, -1e-20])
        expected_Delta_ij = np.array(
            [[-75000, 75000, -75000, 75000], [75000, -75000, 75000, -75000]]
        )
        i = 0
        self.crossbar.logic_currents = logic_currents
        self.crossbar.calculate_Delta_ij(output, pattern, i)
        np.testing.assert_array_almost_equal(
            self.crossbar.all_delta_ij[i], expected_Delta_ij
        )

    def test_calculate_DeltaW_ij(self):
        """
        Test the calculate_DeltaW_ij method of the Memristor_Crossbar instance verifying that the calculation of the DeltaW_ij values are correct.

        Returns
        -------
        None
        """
        all_delta_ij = np.array(
            [
                [[75000, 75000, 75000, -75000], [-75000, -75000, -75000, 75000]],
                [[-75000, -75000, 75000, -75000], [75000, 75000, -75000, 75000]],
                [[-75000, 75000, 75000, -75000], [75000, -75000, -75000, 75000]],
                [[-75000, 75000, 75000, -75000], [75000, -75000, -75000, 75000]],
                [[-75000, -75000, 75000, -75000], [75000, 75000, -75000, 75000]],
                [[75000, 75000, 75000, -75000], [-75000, -75000, -75000, 75000]],
            ]
        )
        expected_deltaW_ij = np.array([[-1, +1, +1, -1], [+1, -1, -1, +1]])
        self.crossbar.all_delta_ij = all_delta_ij
        deltaW_ij = self.crossbar.calculate_DeltaW_ij()
        expected_DeltaW_ij = np.transpose(expected_deltaW_ij)
        np.testing.assert_array_almost_equal(deltaW_ij, expected_DeltaW_ij)

    def test_update_weights(self):
        """
        Test the update_weights method of the Memristor_Crossbar instance checking that the conductances are correctly updated based on DeltaW_ij.

        Returns
        -------
        None
        """
        conductance_data = np.array([1.0, 1.5, 2.0, 2.5, 3.0])
        shifts = np.array([[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]])
        unchanged_conductances = np.array(
            [
                [
                    [0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0],
                ],
                [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
            ]
        )
        DeltaW_ij = np.array([[+1, 0], [0, 0], [0, -1], [0, 0]])
        with patch.object(self.crossbar, "calculate_DeltaW_ij", return_value=DeltaW_ij):
            self.crossbar.experimental_data(conductance_data)
            self.crossbar.shifts = shifts
            self.crossbar.conductances = copy.deepcopy(unchanged_conductances)
            epoch = 1
            self.crossbar.update_weights(epoch)
            for j in range(4):
                for i in range(4):
                    if (i == 0 and j == 0) or (i == 2 and j == 3):
                        assert (
                            unchanged_conductances[0, i, j]
                            != self.crossbar.conductances[0, i, j]
                        )
                        assert self.crossbar.conductances[0, i, j] == 5
                        assert (
                            unchanged_conductances[1, i, j]
                            != self.crossbar.conductances[1, i, j]
                        )
                        assert self.crossbar.conductances[1, i, j] == 1
                    else:
                        assert (
                            unchanged_conductances[0, i, j]
                            == self.crossbar.conductances[0, i, j]
                        )
                        assert (
                            unchanged_conductances[1, i, j]
                            == self.crossbar.conductances[1, i, j]
                        )

    def test_convergence_criterion(self):
        """
        Test the convergence_criterion method of the Memristor_Crossbar instance verifying that the convergence is correctly determined based on the logic currents and outputs.

        Returns
        -------
        None
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
        Test the total error calculation of the Memristor_Crossbar instance checking that the total error for the specified epoch does match the expected value.

        Returns
        -------
        None
        """
        errors = np.array([[0.5, 0, 0, 0, 0, 0], [0.5, 0, 0, 0, 0, 0]])
        epoch = 0
        expected_total_error = 1
        self.crossbar.errors = errors
        self.crossbar.total_error(epoch)
        total_error = self.crossbar.all_errors[epoch]
        assert total_error == expected_total_error

    def test_save_data(self):
        """
        Test the save_data method of the Memristor_Crossbar instance controlling that the file corresponding to the simulated data exists.

        Returns
        -------
        None
        """
        conductance_data = np.array([1.0, 1.5, 2.0, 2.5, 3.0])
        self.crossbar.experimental_data(conductance_data)
        self.crossbar.shift_lognormal()
        self.crossbar.conductance_init()
        self.crossbar.update_weights(1)
        self.crossbar.save_data(base_filename="test_simulation", converged=False)
        filename = "test_simulation_not_converged_data.csv"
        date_dir = datetime.now().strftime("%d-%m-%Y")
        full_path = os.path.join(date_dir, "not_converged", filename)
        assert os.path.exists(full_path)
        shutil.rmtree(date_dir)

    @pytest.mark.parametrize(
        "conductance_data, patterns, outputs, expected_converged, save_data",
        [
            (
                np.array([1.0, 1.5, 2.0, 2.5, 3.0]),
                np.array(
                    [
                        [0, 0, 0, 1],
                        [0, 0, 1, 0],
                        [0, 1, 1, 0],
                        [1, 0, 0, 1],
                        [1, 1, 0, 1],
                        [1, 1, 1, 0],
                    ]
                ),
                np.array([[0, 1], [1, 0], [1, 0], [0, 1], [0, 1], [1, 0]]),
                False,
                True,
            ),
            (
                np.array([1.0, 1.5, 2.0, 2.5, 3.0]),
                np.array(
                    [
                        [0, 0, 0, 1],
                        [0, 0, 1, 0],
                        [0, 1, 1, 0],
                        [1, 0, 0, 1],
                        [1, 1, 0, 1],
                        [1, 1, 1, 0],
                    ]
                ),
                np.array([[0, 1], [1, 0], [1, 0], [0, 1], [0, 1], [1, 0]]),
                True,
                False,
            ),
        ],
    )
    def test_fit(
        self, conductance_data, patterns, outputs, expected_converged, save_data
    ):
        """
        Test the fit method of the Memristor_Crossbar instance. In particular, it checks that all mocked methods are called correctly based on the parameters.

        Returns
        -------
        None
        """
        self.crossbar.experimental_data = MagicMock()
        self.crossbar.shift_lognormal = MagicMock()
        self.crossbar.conductance_init = MagicMock()
        self.crossbar.calculate_Delta_ij = MagicMock()
        self.crossbar.calculate_logic_currents = MagicMock()
        self.crossbar.save_data = MagicMock()
        self.crossbar.update_weights = MagicMock()
        self.crossbar.total_error = MagicMock()
        self.crossbar.convergence_criterion = MagicMock(return_value=expected_converged)
        self.crossbar.fit(
            patterns=patterns,
            outputs=outputs,
            conductance_data=conductance_data,
            save_data=save_data,
        )
        self.crossbar.experimental_data.assert_called_once_with(conductance_data)
        self.crossbar.shift_lognormal.assert_called_once()
        self.crossbar.conductance_init.assert_called_once()
        self.crossbar.calculate_logic_currents.assert_called()
        self.crossbar.calculate_Delta_ij.assert_called()
        self.crossbar.convergence_criterion.assert_called()
        if expected_converged:
            self.crossbar.save_data.assert_not_called()
            self.crossbar.total_error.assert_not_called()
        else:
            self.crossbar.update_weights.assert_called()
            self.crossbar.total_error.assert_called()
            self.crossbar.save_data.assert_called_once()

    @pytest.mark.parametrize(
        "epoch, patterns, outputs, converged",
        [
            (5, 
            np.array([[0, 0, 0, 1], 
                    [0, 1, 1, 0]]), 
            np.array([[1, 0], 
                    [0, 1]]), 
            True),
            (10, 
            np.array([[1, 0, 1, 0], 
                    [1, 1, 0, 1]]), 
            np.array([[0, 1], 
                    [1, 0]]), 
            False),
        ],
    )
    def test_visualize_graphs(self, epoch, patterns, outputs, converged):
        """
        Test the visualize_graphs method of the Memristor_Crossbar instance. Controls that the appropriate plotting methods are called based on the convergence status.

        Returns
        -------
        None
        """
        self.crossbar.plot_conductances = MagicMock()
        self.crossbar.plot_weights = MagicMock()
        self.crossbar.plot_error = MagicMock()
        self.crossbar.plot_results = MagicMock()
        self.crossbar.plot_final_weights = MagicMock()
        self.crossbar.visualize_graphs(epoch, patterns, outputs, converged)
        self.crossbar.plot_conductances.assert_called_once_with(epoch)
        self.crossbar.plot_weights.assert_called_once_with(epoch)
        self.crossbar.plot_error.assert_called_once_with(epoch)
        self.crossbar.plot_results.assert_called_once_with(patterns, outputs, epoch)
        if converged:
            self.crossbar.plot_final_weights.assert_called_once()
        else:
            self.crossbar.plot_final_weights.assert_not_called()

    @pytest.mark.parametrize(
        "logic_currents, expected_predictions",
        [
            (np.array([1e-6, -1e-6]), np.array([1, 0])),
            (np.array([-1e-20, 1e-20]), np.array([2, 2])),
        ],
    )
    def test_check_convergence(self, logic_currents, expected_predictions):
        """
        Test the check_convergence method of the Memristor_Crossbar instance verifying that the output is calculated correctly from the logic currents and don't match the expected one.

        Returns
        -------
        None
        """
        self.crossbar.logic_currents = logic_currents
        self.crossbar.check_convergence(0)
        np.testing.assert_array_equal(
            self.crossbar.predictions[0], expected_predictions
        )

    def test_predict(self):
        """
        Test the predict method of the Memristor_Crossbar instance verifying that the predictions made by the `predict` method match the expected output.

        Returns
        -------
        None
        """
        conductances = np.array(
            [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]
        )
        pattern = np.array(
            [
                [0, 0, 0, 0],
                [0, 0, 1, 1],
                [0, 1, 0, 0],
                [0, 1, 0, 1],
                [0, 1, 1, 1],
                [1, 0, 0, 0],
                [1, 0, 1, 0],
                [1, 0, 1, 1],
                [1, 1, 0, 0],
                [1, 1, 1, 1],
            ]
        )
        output = np.array(
            [
                [0, 0],
                [0, 0],
                [1, 0],
                [0, 0],
                [1, 0],
                [0, 1],
                [0, 0],
                [0, 1],
                [0, 0],
                [0, 0],
            ]
        )
        expected_output = np.array(
            [
                [2, 2],
                [2, 2],
                [2, 2],
                [2, 2],
                [2, 2],
                [2, 2],
                [2, 2],
                [2, 2],
                [2, 2],
                [2, 2],
            ]
        )
        self.crossbar.saved_correct_conductances = conductances
        self.crossbar.predict(pattern, output)
        np.testing.assert_array_equal(self.crossbar.predictions, expected_output)


if __name__ == "__main__":
    pytest.main()
