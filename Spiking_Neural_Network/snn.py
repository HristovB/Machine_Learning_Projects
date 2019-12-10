import numpy as np


class SNN(object):

    def __init__(self, input_layer, input_dim, step, input_weights=None):

        self.layers = []
        self.step = step
        self.input_layer = input_layer

        if input_weights is not None:
            self.input_layer.weights = input_weights
        else:
            self.input_layer.weights = np.random.random(size=(self.input_layer.num_neurons, input_dim))

        self.layers.append(input_layer)

    def get_weights(self):
        weights = []
        for layer in self.layers:
            weights.append(layer.weights)

        return weights

    def add(self, layer):
        """ Add layer to existing neural network"""

        if layer.weights is None:
            layer.weights = np.random.random(size=(layer.num_neurons, self.layers[-1].num_neurons))

        self.layers.append(layer)

    def compile(self, input_spikes=None, snn_outputs=None):

        self.layers[0].inputs = input_spikes

        layer_inputs = []
        neuron_output = []
        layer_outputs = []
        for i in range(len(self.layers)):

            if i == len(self.layers) - 1 and snn_outputs is not None:
                self.layers[i].inputs = snn_outputs

            elif i > 0 and i != len(self.layers) - 1:
                self.layers[i].inputs = neuron_output

            neuron_output = self.layers[i].calculate_output()

            layer_inputs.append(self.layers[i].inputs)
            layer_outputs.append(neuron_output)

        return layer_inputs, layer_outputs

    @staticmethod
    def stdp(x):
        if x > 0:
            return 1.0 * np.exp(-x/10.0)
        else:
            return -1.0 * np.exp(x/10.0)

    def update_weights(self, update):
        for l in range(len(self.layers)):
            self.layers[l].weights += update[l]

    def fit(self, snn_inputs, snn_outputs=None, num_iter=10):

        for iteration in range(num_iter):
            output_index = 0
            for sample in snn_inputs:
                inputs, outputs = self.compile(input_spikes=sample, snn_outputs=snn_outputs[output_index])

                dw_array = []
                for l in range(len(inputs)):
                    dw_layer = []
                    for i in range(len(inputs[l])):
                        for o in range(len(outputs[l])):
                            input_neuron = inputs[l][i]
                            output_neuron = outputs[l][o]

                            input_times = np.where(input_neuron != 0)
                            input_times = input_times[0]

                            output_times = np.where(output_neuron != 0)
                            output_times = output_times[0]

                            dt = []
                            for in_t in input_times:
                                for out_t in output_times:
                                    dt.append(in_t - out_t)

                            dw = []
                            for time in dt:
                                dw.append(self.stdp(time))

                            dw_layer.append(sum(dw))

                    dw_array.append(np.reshape(dw_layer, (len(outputs[l]), len(inputs[l]))))

                dw_array = np.array(dw_array)

                self.update_weights(dw_array)

                output_index += 1

    def predict(self, inputs):
        _, outputs = self.compile(input_spikes=inputs)

        return outputs

class Neuron:
        """ Izhikevich model of Spiking Neuron implemented in Python 3 """

        def __init__(self, a, b, c, d, step):
            """ Initialize Izhikevich neuron """

            self.a = a  # time-scale of the recovery variable
            self.b = b  # sensitivity of the recovery variable
            self.c = c  # after-spike reset value of the membrane potential
            self.d = d  # after-spike reset value of the recovery variable
            self.time_step = step

            self.v = self.c  # membrane potential (millivolts)

            self.u = self.b * self.v  # membrane recovery variable

            self.fired = False  # value to indicate if the neuron has fired or not

        def calculate_diff_equations(self, input_spike):
            """ Transform the Izhikevich differential equation model to a discrete model and calculate the membrane potential and recovery for a given input spike """

            self.v = self.v + self.time_step * (0.04 * self.v ** 2 + 5 * self.v + 140 - self.u + input_spike)
            self.u = self.u + self.time_step * self.a * (self.b * self.v - self.u)

            return self.v, self.u

        def is_fired(self):
            """ Boolean function to determine if the neuron has fired or not, thus generating the neuron's output """

            if self.v > 30:
                self.fired = True
                self.v = self.c
                self.u += self.d
            else:
                self.fired = False

            return self.fired


class Layer:
        """ Generate layer of spiking neurons and initialize random (or given) weights """

        def __init__(self, num_neurons, a, b, c, d, step, inputs=None, weights=None):
            """ Initialize SNN layer """

            self.num_neurons = num_neurons
            self.inputs = inputs

            self.a = a  # time-scale of the recovery variable
            self.b = b  # sensitivity of the recovery variable
            self.c = c  # after-spike reset value of the membrane potential
            self.d = d  # after-spike reset value of the recovery variable
            self.time_step = step
            self.weights = weights

        def generate_layer(self):
            """ Generate layer with given number of neurons """

            neurons = [Neuron(a=self.a, b=self.b, c=self.c, d=self.d, step=self.time_step)] * self.num_neurons
            neurons = np.array(neurons)

            return neurons

        def calculate_output(self):
            """ Calculate output of layer """

            neurons = self.generate_layer()
            layer_output = []
            neuron_index = 0

            for neuron in neurons:
                neuron_output = []

                spike_array = []

                for i in range(len(self.inputs)):
                    spike_array.append((self.inputs[i] * self.weights[neuron_index][i]))

                spike_array = sum(spike_array)

                for spike in spike_array:
                    neuron.calculate_diff_equations(spike)

                    if neuron.is_fired():
                        neuron_output.append(1)
                    else:
                        neuron_output.append(0)

                layer_output.append(neuron_output)
                neuron_index += 1

            return np.array(layer_output)
