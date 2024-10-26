# Usage

To begin using the library, you can execute the `training_simulation.py` file, which provides a usage example. For instance:

```bash
$ python training_simulation.py
```

This file includes the Memristor Crossbar class and creates an instance. It also provides an example set of real conductance values, in case the user doesn’t have his own. For this specific example, a set of optimal parameters has been chosen for the dataset, and the fit method is applied using these parameters along with the provided experimental data. <br>
The training and testing datasets are defined as binary vectors:
-   **Training Set**: A 2D array where each row represents a sample used for training the model, with the corresponding output labels provided in the train_outputs array. The patterns were selected to maintain a Hamming distance of 1 from the ideal patterns (more in the [Appendix](#appendix) section). Users may also define their own training set if desired.
-  **Testing Set**: A separate 2D array, used to evaluate the model’s performance after training, contains all remaining 4-bit patterns. Expected outputs for these patterns are defined in the test_outputs array.

Note that each time the simulation starts, it relies on the random generation of shifts. <br>
There are two recommended ways to use the fit method, as demonstrated in the example:
1. **Single Simulation**: This approach performs a single simulation, generating plots and printing information that helps monitor the training process. 
2. **Multiple Simulations**: This method utilizes a for loop, disabling plots and prints while enabling data saving for the shifts. The data is automatically saved in two separate folders based on whether the simulation converged or not. 

By using this approach, if the simulation converges, you can take the saved shifts and utilize the custom shift option in the fit method. This allows you to check convergence again while plotting the data. <br>

```{note}
**Note:** Running the code directly creates a folder named `dd/mm/yy` containing the results.
To perform a single simulation, comment out the last two lines of the `training_simulation.py` code.
```

<div class="github-badge">
<a href="https://github.com/cateru/Memristor_Crossbar_Training_Class">
    <img src="https://img.shields.io/badge/View_on-GitHub-blue?logo=github&style=for-the-badge" alt="View on GitHub">
</a>
</div>