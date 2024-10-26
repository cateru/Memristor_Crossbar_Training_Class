# Memristor Crossbar Training Class

This library facilitates the creation and management of **4x4 memristor crossbars** tailored for *training* a **single-layer perceptron classifier** in neuromorphic computing applications. It provides the necessary tools for training simulations using binary patterns for both inputs and outputs, along with an experimental dataset that mimics how the physical crossbar device behaves. The library also includes features for logging, saving data, and plotting, making it easier to track and visualize the training process.


<center>
<div style="text-align: center;">
    <img src="pictures/IMG_0235.PNG" alt="Memristor Crossbar" width="300">
</div>
</center>


## Table of Contents

1. [Project Structure](#project_structure)
2. [Installation](#installation)
3. [Usage](#usage)
4. [Features](#features)
5. [Contributing](#contributing)
6. [Appendix](#appendix)


## Project Structure 

This project is organized into the following main components, located in the [memristor_crossbar](memristor_crossbar/) folder:

1. **[Memristor_Crossbar.py](memristor_crossbar/Memristor_Crossbar.py)**: Defines the Memristor Crossbar class, encompassing all methods necessary for training and inference.
2. **[Memristor_Crossbar_test.py](memristor_crossbar/Memristor_Crossbar_test.py)**: Includes testing functions designed specifically for the Memristor Crossbar class.
3. **[experimental_conductances.py](memristor_crossbar/experimental_conductances.py)**: Includes the load_conductance_data function, which loads and processes conductance data from a specified CSV or .dat file.
4. **[training_simulation.py](memristor_crossbar/training_simulation.py)**: Demonstrates how to conduct training simulations. It creates an instance of the memristor crossbar class, defines the training and expected output sets, and integrates the loading of conductance data for a complete training workflow.

## Installation

To install the library, clone the repository and install the necessary dependencies:

```bash
git clone https://github.com/cateru/Memristor_Crossbar_Training_Class.git
cd Memristor_Crossbar_Training_Class
pip install -r requirements.txt
```

## Usage

To begin using the library, you can execute the `training_simulation.py` file, which provides a usage example. For instance:

```bash
python training_simulation.py
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

<div style="background-color: #f8f9fa; padding: 10px; border-radius: 5px; border-left: 5px solid #007bff;">
<strong>Note:</strong> Running the code directly creates a folder named <code>dd/mm/yy</code> containing the results.<br>
To perform a single simulation, comment out the last two lines of the <code>training_simulation.py</code> code.
</div>


## Features 

Inside the `Memristor_Crossbar.py` module, there are methods that handle the mathematical processes of the training algorithm and perform weight updates. The module also includes methods for plotting key metrics that monitor the training progress, such as activation functions, total distance to convergence, weight updates, and individual activation responses. For users who want to explore the technical details of the algorithm and its code implementation, it's recommended to refer to the appendix.

## Contributing

Contributions are welcome! If you'd like to contribute, please follow these steps:

1. Fork the repository.
2. Create a new branch for your feature or bug fix.
3. Commit your changes.
4. Open a pull request and describe your changes.

Please ensure that all tests pass before submitting your pull request.


## Appendix

This appendix provides an overview of the physical principles that motivated the creation of this project and its relevance within the neuromorphic computing framework. It also includes a detailed description of the algorithm’s steps along with any modifications made to implement it in Python. Finally, example output graphs are presented to help users verify the correctness of their results.

### Neuromorphic Computing

<u>*Neuromorphic computing*</u> is an approach to designing *hardware* and *software* that **mimics the structure and function of the human brain's neural networks** to improve energy efficiency and performance in tasks like pattern recognition and learning. <br>
The key components of neuromorphic systems are <u>*synaptic-like*</u> devices, which emulate the behavior of biological synapses. These devices enable systems to replicate brain-like learning and adaptability by adjusting their "*synaptic weights*" based on experience, similar to how biological synapses **strengthen** or **weaken** connections through a process called *synaptic plasticity*.

### Metal-Oxide Memristors Crossbars

Metal-Oxide Memristors are Spintronic devices which present a synaptic behavior after the application of **identical Voltage pulses** with <u>sufficiently high amplitude</u>. In particular, the synaptic weight is represented by the **conductance** of the device and, by sending a train of voltage pulses, it can be increased in a **non-volatile way**, emulating in this way the synapse potentiation happening in the human brain. Also, after the conductance change, it's possibile to <u>return to the previous state</u> by applying a voltage pulse of opposite polarity, emulating so the synapse depression process [*Fig2.a*]. <br>
It is so possible to build **integrated circuits** formed by memristors arranged in a crossbar configuration which play the role of simple neural networks and can learn tasks for neuromorphic computing applications. <br>
In the present work, we take a 4x4 memristor crossbar circuit to which we associate a single-layer perceptron architechture with two neurons, capable of recognizing the set of patterns in *Fig2.b*. <br>
In the physical picture, the 4-bit patterns are represented by 2 distinct values of *voltage* which represent the "0" and "1", the synaptic weights are associated with the conductances of the memristors while the neural network outputs are connected with the output *currents*.

<center>
<div style="text-align: center; display: flex; justify-content: center;">
    <div style="margin: 10px;">
        <img src="pictures/volt-pulse.png" alt="Voltage Pulse" width="500">
        <p>Figure 2.a: Conductance change as a function of the number of voltage pulses and pulse duration [1].</p>
    </div>
    <div style="margin: 10px;">
        <img src="pictures/IMG_0234.PNG" alt="Patterns" width="400">
        <p>Figure 2.b: Training set of 4-bit patterns.</p>
    </div>
</div>
</center>

### Training Algorithm 

In the following section is described the mathematical algorithm used for the training of the neural network. 
As I introduced before, the memristive crossbar has been used to implement a single-layer perceptron with 4 inputs and 2 outputs.<br>
In particular, the outputs are calculated as <u>nonlinear activation functions</u>:

$$f_i = tanh(\beta I_i) \quad \textit{(eq. 1)}$$

of the vector-by-matrix product components:

$$I_i = \sum_{j=1}^{4} W_{ij}V_j \quad \textit{(eq. 2)}$$  

In order to have more degrees of freedom during the training, the actual network logic currents are calculated as the <u>differential output of two adjacent columns</u>, while the synaptic weights are represented by the <u>differential weigths of two adjacent columns</u>, thus forming 2 neurons containing 2 columns each:

$$I_i = I^+_i - I^-_i$$ 

$$W_{ij} = G^+_{ij} - G^-_{ij}$$

To update the weights, is used **backpropagation** with the **Manhattan update rule**:

$$\delta_i = [f^g_i(n) - f_i(n)]\frac{df}{dI}\Bigg|_{I = I_i(n)} \quad \textit{(eq. 3)}$$ 

$$\Delta_{ij}(n) = \delta_i(n)V_j(n) \quad \textit{(eq. 4)}$$ 

$$\Delta W_{ij} = sgn\sum_{n=1}^{N} \Delta_{ij}(n) \quad \textit{(eq. 5)}$$

At this point, if $\Delta W_{ij} > 0$, synapse $W_{ij}$ is **potentiated** sending a <u>positive voltage pulse to $G^+_{ij}$</u>, on the contrary, if $\Delta W_{ij} < 0$, synapse $W_{ij}$ is **depressed** sending a <u>positive voltage pulse to $G^-_{ij}$ </u>.<br> The parameters $\beta$ and $f^g_i(n)$ are numerical values that must be fine-tuned to achieve successful training. Specifically, $f^g_i(n)$ represents the **target value** for the activation function. If the pattern is to be recognized by the neuron as output '1', the activation function must **exceed** the target value $f^g_i(n)$. Conversely, if the pattern is to be classified as output '0', the activation function must **remain below** $f^g_i(n)$. 

<center>
<div style="text-align: center;">
    <img src="pictures/activation.png" alt="tanh" width="700">
</div>
</center>

The neural network is considered successfully trained when the activation functions for all patterns lie within their respective target ranges. Additionally, in this implementation, the target values for outputs '1' and '0' are assumed to be the same for the entire set of patterns. <br>
All these steps are summarized in the following picture [[2]](#references):

<center>
<div style="text-align: center;">
    <img src="pictures/tr_alg.png" alt="Algorithm" width="1000">
</div>
</center>


### Python Simulation

In this work, we present the implementation of a Python class that simulates the ***in-situ*** training process of a physical device. To replicate the role of synaptic depression and potentiation, an experimental dataset of conductances (similar to the one shown in Figure 2a, 2V, 50ms) has been used. Additionally, to achieve a sufficient dynamic range and account for device-to-device variation, the conductances are initialized and updated as follows:

$$G_{ij} = K \times [(G_{exp}(m) - G_{exp}(0)) + shift_{ij}]$$

Where $K$ is a multilication factor while $shift_{ij}$ was obtained using a random exponential distribution similar to that of the physical device.

### Examples 

Below are examples of output graphs from a simulation that successfully converged after 18 epochs:

<center>
<div style="text-align: center;">
    <img src="pictures/conds_18ep.png" alt="conductances" width="800">
</div>

<div style="text-align: center;">
    <img src="pictures/logw.png" alt="logic_weights" width="800">
</div>

<div style="text-align: center;">
    <img src="pictures/activ_18ep.png" alt="activations" width="800">
</div>

<div style="text-align: center;">
    <img src="pictures/error.png" alt="error" width="600">
</div>

<div style="text-align: center;">
    <img src="pictures/3d.png" alt="3d" width="400">
</div>
</center>

And here are examples from a simulation that did not converge after the default 48 epochs:

<center>
<div style="text-align: center;">
    <img src="pictures/nonconv_conds.png" alt="conductances_nonconv" width="800">
</div>

<div style="text-align: center;">
    <img src="pictures/nonconv_synoutput.png" alt="logic_weights_nonconv" width="800">
</div>

<div style="text-align: center;">
    <img src="pictures/nonconv_activ.png" alt="activations_nonconv" width="800">
</div>

<div style="text-align: center;">
    <img src="pictures/nonconv_error.png" alt="error_nonconv" width="600">
</div>
</center>



### References

[[1]](https://doi.org/10.1002/aelm.202300887) Shumilin, A., Neha, P., Benini, M., Rakshit, R., Singh, M., Graziosi, P., ... & Riminucci, A. (2024). Glassy Synaptic Time Dynamics in Molecular La0. 7Sr0. 3MnO3/Gaq3/AlOx/Co Spintronic Crossbar Devices. Advanced Electronic Materials, 2300887. <br>
[[2]](https://doi.org/10.1038/nature14441) Prezioso, M., Merrikh-Bayat, F., Hoskins, B. D., Adam, G. C., Likharev, K. K., & Strukov, D. B. (2015). Training and operation of an integrated neuromorphic network based on metal-oxide memristors. Nature, 521(7550), 61-64.