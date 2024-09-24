# Memristor Crossbar Training Class

This project aims to demonstrate the possibility to obtain a **successful training simulation** of a <u>*single-layer perceptron*</u> classifier represented by a **4x4 Memristor crossbar** for Neuromorphic Computing applications, utilizing an experimental dataset to emulate the device's physical behavior.

<center>
<div style="text-align: center;">
    <img src="pictures/IMG_0235.PNG" alt="Memristor Crossbar" width="300">
</div>
</center>

## Neuromorphic Computing

<u>*Neuromorphic computing*</u> is an approach to designing *hardware* and *software* that **mimics the structure and function of the human brain's neural networks** to improve energy efficiency and performance in tasks like pattern recognition and learning. <br>
The key components of neuromorphic systems are <u>*synaptic-like*</u> devices, which emulate the behavior of biological synapses. These devices enable systems to replicate brain-like learning and adaptability by adjusting their "*synaptic weights*" based on experience, similar to how biological synapses **strengthen** or **weaken** connections through a process called *synaptic plasticity*.

## Metal-Oxide Memristors Crossbars

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

## Training Algorithm 

In the following section is described the mathematical algorithm used for the training of the neural network. 
As I introduced before, the memristive crossbar has been used to implement a single-layer perceptron with 4 inputs and 2 outputs.<br>
In particular, the outputs are calculated as <u>nonlinear activation functions</u>:

$$f_i = tanh(\beta I_i) \quad \textit{(eq. 1)}$$

of the vector-by-matrix product components:

$$I_i = \sum_{j=1}^{4} W_{ij}V_j \quad \textit{(eq. 2)}$$  

In order to have more degrees of freedom during the training, the actual network logic currents are calculated as the <u>differential output of two adjacent columns</u>, while the synaptic weights are represented by the <u>differential weigths of two adjacent columns</u>, thus forming 2 neurons containing 2 columns each:

$$I_i = I^+_i - I^-_i$$
<br>
$$W_{ij} = G^+_{ij} - G^-_{ij} $$

To update the weights, is used **backpropagation** with the **Manhattan update rule**:

$$\delta_i = [f^g_i(n) - f_i(n)]\frac{df}{dI}\Bigg|_{I = I_i(n)} \quad \textit{(eq. 3)}$$
<br>
$$\Delta_{ij}(n) = \delta_i(n)V_j(n) \quad \textit{(eq. 4)}$$
<br>
$$\Delta W_{ij} = sgn\sum_{n=1}^{N} \Delta_{ij}(n) \quad \textit{(eq. 5)}$$

At this point, if $\Delta W_{ij} > 0$, synapse $W_{ij}$ is **potentiated** sending a <u>positive voltage pulse to $G^+_{ij}$</u>, on the contrary, if $\Delta W_{ij} < 0$, synapse $W_{ij}$ is **depressed** sending a <u>positive voltage pulse to $G^-_{ij}$</u>. <br>
The parameters $\beta$ and $f^g_i(n)$ are numerical values that must be fine-tuned to achieve successful training. Specifically, $f^g_i(n)$ represents the **target value** for the activation function. If the pattern is to be recognized by the neuron as output '1', the activation function must **exceed** the target value $f^g_i(n)$. Conversely, if the pattern is to be classified as output '0', the activation function must **remain below** $f^g_i(n)$. 

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


## Python Simulation

In this work, we present the implementation of a Python class that simulates the ***in-situ*** training process of a physical device. To replicate the role of synaptic depression and potentiation, an experimental dataset of conductances (similar to the one shown in Figure 2a, 2V, 50ms) has been used. Additionally, to achieve a sufficient dynamic range and account for device-to-device variation, the conductances are initialized and updated as follows:

$$G_{ij} = K \times [(G_{exp}(m) - G_{exp}(0)) + shift_{ij}]$$

Where $K$ is a multilication factor while $shift_{ij}$ was obtained using a random exponential distribution similar to that of the physical device.


## Project Structure 

This project is divided into the following blocks: <br>
1. The [experimental conductances](memristor_crossbar/experimental_conductances.py) file, which processes the 'SP291D1NR100.dat' file and generates a NumPy array containing the conductance values. 
2. The [Memristor Crossbar](memristor_crossbar/Memristor_Crossbar.py) file, which defines the Memristor Crossbar class, encompassing all methods necessary for training and inference.
3. The [Memristor Crossbar test](memristor_crossbar/Memristor_Crossbar_test.py) file, which includes testing functions specifically designed for the Memristor Crossbar class.
4. The [training simulation](memristor_crossbar/training_simulation.py) file, which imports the experimental conductances and the Memristor Crossbar class, creates an instance of the class, and utilizes the fit and predict methods.

To initiate a simulation, simply open the [training simulation](memristor_crossbar/training_simulation.py) file. In this file, the experimental conductances and the Memristor Crossbar class are already imported, and an instance of the class is created. <br>
In this specific case, a set of optimal parameters has been selected for this dataset. The fit method is implemented using these parameters along with the provided experimental data. <br>
To use different sets of conductances, simply select them when calling the fit method. Note that each time the simulation starts, it relies on the random generation of shifts. <br>
There are two recommended ways to use the fit method, as demonstrated in the example:
1. Single Simulation: This approach performs a single simulation, generating plots and printing information that helps monitor the training process. 
2. Multiple Simulations: This method utilizes a for loop, disabling plots and prints while enabling data saving for the shifts. The data is automatically saved in two separate folders based on whether the simulation converged or not. 

By using this approach, if the simulation converges, you can take the saved shifts and utilize the custom shift option in the fit method. This allows you to check convergence again while plotting the data. <br>
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



## References

[[1]](https://doi.org/10.1002/aelm.202300887) Shumilin, A., Neha, P., Benini, M., Rakshit, R., Singh, M., Graziosi, P., ... & Riminucci, A. (2024). Glassy Synaptic Time Dynamics in Molecular La0. 7Sr0. 3MnO3/Gaq3/AlOx/Co Spintronic Crossbar Devices. Advanced Electronic Materials, 2300887. <br>
[[2]](https://doi.org/10.1038/nature14441) Prezioso, M., Merrikh-Bayat, F., Hoskins, B. D., Adam, G. C., Likharev, K. K., & Strukov, D. B. (2015). Training and operation of an integrated neuromorphic network based on metal-oxide memristors. Nature, 521(7550), 61-64.