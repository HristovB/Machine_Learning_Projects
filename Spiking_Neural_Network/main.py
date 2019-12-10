import numpy as np
import pandas as pd
from scipy.sparse import random
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
from Spiking_Neural_Network.snn import SNN as snn
from Spiking_Neural_Network.snn import Layer, Neuron


np.random.seed(157)

def generate_input(amplitude, impulse, start, stop, step, sim_time, plot=False):

    inputs = np.zeros(int(simulation_time / impulse_count / time_step))

    for index in range(len(inputs)):
        if len(inputs) * start < index < len(inputs) * stop:
            inputs[index] = spike_amplitude

    inputs = np.tile(inputs, impulse_count)

    if plot is True:
        sim = np.arange(0, simulation_time, time_step)
        sns.lineplot(sim, inputs, color='navy')
        plt.show()

    return inputs


if __name__ == '__main__':

    a = 0.02
    b = 0.2
    c = -65
    d = 8

    time_step = 0.5
    simulation_time = 100

    simulation = np.arange(0, simulation_time, time_step)

    # spike_amplitude = 5
    # impulse_count = 1
    # impulse_start = 0
    # impulse_stop = 1

    # input_spikes = (generate_input(spike_amplitude, impulse_count, impulse_start, impulse_stop, time_step, simulation_time, plot=False))

    input_one = np.zeros((1, int(simulation_time / time_step))).flatten()
    input_one[::10] = 5

    input_zero = np.zeros((1, int(simulation_time / time_step))).flatten()

    input_00 = np.array([input_zero, input_zero])
    input_01 = np.array([input_zero, input_one])
    input_10 = np.array([input_one, input_zero])
    input_11 = np.array([input_one, input_one])

    output_one = np.zeros((1, int(simulation_time / time_step))).flatten()
    output_one[::11] = 5

    output_zero = np.zeros((1, int(simulation_time / time_step))).flatten()

    output_00 = np.reshape(output_zero, (1, len(output_zero)))
    output_01 = np.reshape(output_one, (1, len(output_one)))
    output_10 = np.reshape(output_one, (1, len(output_one)))
    output_11 = np.reshape(output_one, (1, len(output_one)))

    # input_spikes = np.random.randint(0, 2, int(simulation_time / impulse_count / time_step))
    # input_spikes = np.array([1 if element != 0 else 0 for element in input_spikes.A.flatten()])

    # n1 = snn.Neuron(a, b, c, d, time_step)
    #
    # output = []
    # membrane_potential = []
    # for spike in input_spikes:
    #     mV, mR = n1.calculate_diff_equations(spike)
    #
    #     membrane_potential.append(mV)
    #
    #     if n1.is_fired():
    #         output.append(1)
    #     else:
    #         output.append(0)
    #
    # fig, ax = plt.subplots(1, 3, figsize=(14, 7))
    # sns.lineplot(simulation, input_spikes, color='navy', ax=ax[0])
    # ax[0].set_title('Input Spikes')
    # ax[0].set_xlabel('Time (ms)')
    # ax[0].set_ylabel('Amplitude (mV)')
    #
    # sns.lineplot(simulation, membrane_potential, color='purple', ax=ax[1])
    # ax[1].set_title('Neuron membrane potential')
    # ax[1].set_xlabel('Time (ms)')
    # ax[1].set_ylabel('Potential (mV)')
    #
    # sns.lineplot(simulation, output, color='black', ax=ax[2])
    # ax[2].set_title('Output Spikes')
    # ax[2].set_xlabel('Time (ms)')
    # ax[2].set_ylabel('Output (binary)')
    #
    # plt.show()

    inputs = [input_00, input_01, input_10, input_11]
    outputs = [output_00, output_01, output_10, output_11]

    layer = Layer(num_neurons=2, a=a, b=b, c=c, d=d, step=time_step)
    nn = snn(input_layer=layer, input_dim=np.shape(inputs)[1], step=time_step)

    layer = Layer(num_neurons=1, a=a, b=b, c=c, d=d, step=time_step)
    nn.add(layer)

    weights = nn.get_weights()
    print(weights[0])
    print()

    nn.fit(snn_inputs=inputs, snn_outputs=outputs, num_iter=100)

    weights = nn.get_weights()
    print(weights[0])


    # fig, ax = plt.subplots(1, 3, figsize=(18, 7))
    # sns.lineplot(simulation, ins[0], color='navy', ax=ax[0])
    # ax[0].set_title('Input')
    # ax[0].set_xlabel('Time (ms)')
    # ax[0].set_ylabel('Input (binary)')
    # ax[0].set_ylim(-0.5, 5.5)
    #
    # sns.lineplot(simulation, ous[0], color='navy', ax=ax[1])
    # ax[1].set_title('Input')
    # ax[1].set_xlabel('Time (ms)')
    # ax[1].set_ylabel('Input (binary)')
    # ax[1].set_ylim(-0.5, 5.5)

    # sns.lineplot(simulation, out[0].flatten(), color='purple', ax=ax[2])
    # ax[2].set_title('SNN Output')
    # ax[2].set_xlabel('Time (ms)')
    # ax[2].set_ylabel('Output (binary)')
    #
    # plt.show()


