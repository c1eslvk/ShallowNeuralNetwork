import streamlit as st
import numpy as np
import plotly.express as px
import pandas as pd

# Activation Functions
def heaviside(s):
    return np.where(s >= 0, 1, 0)

def sigmoid(s):
    return 1 / (1 + np.exp(-s))

def sigmoid_derivative(s):
    return sigmoid(s) * (1 - sigmoid(s))

def sin(s):
    return np.sin(s)

def sin_derivative(s):
    return np.cos(s)

def tanh(s):
    return np.tanh(s)

def tanh_derivative(s):
    return 1 - np.tanh(s)**2

def sign(s):
    return np.sign(s)

def relu(s):
    return np.where(s > 0, s, 0.0)

def relu_derivative(s):
    return np.where(s > 0, 1.0, 0.0)

def leaky_relu(s):
    return np.where(s > 0, s, 0.01 * s)

def leaky_relu_derivative(s):
    return np.where(s > 0, 1.0, 0.01)


# Neural Network
class Layer:
    def __init__(self, num_of_inputs, num_of_neurons, activation="sigmoid"):
        self.weights = np.random.randn(num_of_neurons, num_of_inputs)
        self.biases = np.random.randn(num_of_neurons, 1)
        self.activation = activation
        self.outputs = None
        self.inputs = None
        self.deltas = None

    def activate(self, s):
        if self.activation == 'heaviside':
            return heaviside(s)
        elif self.activation == 'sigmoid':
            return sigmoid(s)
        elif self.activation == 'sin':
            return sin(s)
        elif self.activation == 'tanh':
            return tanh(s)
        elif self.activation == 'sign':
            return sign(s)
        elif self.activation == 'relu':
            return relu(s)
        elif self.activation == 'leaky relu':
            return leaky_relu(s)
    
    def derivative(self, s):
        if self.activation == 'heaviside':
            return 1
        elif self.activation == 'sigmoid':
            return sigmoid_derivative(s)
        elif self.activation == 'sin':
            return sin_derivative(s)
        elif self.activation == 'tanh':
            return tanh_derivative(s)
        elif self.activation == 'sign':
            return 1
        elif self.activation == 'relu':
            return relu_derivative(s)
        elif self.activation == 'leaky relu':
            return leaky_relu_derivative(s)
        
    def forward(self, inputs):
        self.inputs = inputs
        z = np.dot(self.weights, inputs) + self.biases
        self.outputs = self.activate(z)
        return self.outputs

    def compute_deltas(self, delta_next_layer, weights_next_layer):
        z = np.dot(self.weights, self.inputs) + self.biases
        self.deltas = np.dot(weights_next_layer.T, delta_next_layer) * self.derivative(z)
        return self.deltas


class NeuralNetwork:
    def __init__(self, layer_sizes, activation="sigmoid", learning_rate=0.1):
        self.layers = []
        self.learning_rate = learning_rate
        for i in range(1, len(layer_sizes)):
            self.layers.append(Layer(layer_sizes[i-1], layer_sizes[i], activation))

    def forward(self, inputs):
        for layer in self.layers:
            inputs = layer.forward(inputs)
        return inputs

    def backpropagate(self, y_true):
        # Calculate delta for the output layer
        output_layer = self.layers[-1]
        output_error = output_layer.outputs - y_true
        output_layer.deltas = output_error * output_layer.derivative(output_layer.outputs)

        # Calculate delta for hidden layers
        for i in reversed(range(len(self.layers) - 1)):
            next_layer = self.layers[i + 1]
            self.layers[i].compute_deltas(next_layer.deltas, next_layer.weights)

        # Update weights and biases for all layers
        for layer in self.layers:
            layer.weights -= self.learning_rate * np.dot(layer.deltas, layer.inputs.T)
            layer.biases -= self.learning_rate * np.sum(layer.deltas, axis=1, keepdims=True)

    def train(self, X, y, epochs=100):
        for epoch in range(epochs):
            for x_i, y_i in zip(X, y):
                x_i = x_i.reshape(-1, 1)
                y_i = y_i.reshape(-1, 1)
                self.forward(x_i)
                self.backpropagate(y_i)

    def predict(self, X):
        predictions = []
        for x_i in X:
            x_i = x_i.reshape(-1, 1)
            output = self.forward(x_i)
            predictions.append(output.flatten())
        return np.array(predictions)

# Generate Gaussian Data
def generate_gaussian_mode(mean, variance, num_of_samples):
    cov_matrix = [[variance, 0], [0, variance]]
    samples = np.random.multivariate_normal(mean, cov_matrix, num_of_samples)
    return samples

def generate_class_data(num_of_modes, samples_per_mode):
    all_samples = []
    for _ in range(num_of_modes):
        mean = np.random.uniform(-1, 1, 2)  # center
        variance = np.random.uniform(0.05, 0.2)  # spread
        samples = generate_gaussian_mode(mean, variance, samples_per_mode)
        all_samples.append(samples)
    return np.vstack(all_samples)

# Main
def main():
    st.title("Shallow Neural Network")

    st.sidebar.header("Settings")
    num_of_modes_class_0 = st.sidebar.number_input("Number of Modes for Class 0", min_value=1, max_value=10, value=1)
    num_of_modes_class_1 = st.sidebar.number_input("Number of Modes for Class 1", min_value=1, max_value=10, value=1)
    samples_per_mode = st.sidebar.number_input("Number of Samples per Mode", min_value=10, max_value=1000, value=100)
    activation_function = st.sidebar.selectbox("Activation Function", ["heaviside", "sigmoid", "sin", "tanh", "sign", "relu", "leaky relu"])
    num_of_hidden_layers = st.sidebar.slider("Number of Hidden Layers", min_value=1, max_value=3, value=1)
    neurons_per_hidden_layer = st.sidebar.slider("Neurons per Hidden Layer", min_value=2, max_value=10, value=5)
    learning_rate = st.sidebar.slider("Learning Rate", min_value=0.01, max_value=1.0, value=0.1)
    
    if st.sidebar.button("Generate Data and Train Neural Network"):
        class_0_data = generate_class_data(num_of_modes_class_0, samples_per_mode)
        class_1_data = generate_class_data(num_of_modes_class_1, samples_per_mode)

        class_0_labels = np.zeros((len(class_0_data), 2))
        class_0_labels[:, 0] = 1
        class_1_labels = np.zeros((len(class_1_data), 2))
        class_1_labels[:, 1] = 1

        data = np.vstack([class_0_data, class_1_data])
        labels = np.vstack([class_0_labels, class_1_labels])

        input_size = 2
        output_size = 2
        layer_sizes = [input_size] + [neurons_per_hidden_layer] * num_of_hidden_layers + [output_size]
        neural_net = NeuralNetwork(layer_sizes, learning_rate=learning_rate, activation=activation_function)
        neural_net.train(data, labels)

        class_0_df = {"x": class_0_data[:, 0], "y": class_0_data[:, 1], "class": "Class 0"}
        class_1_df = {"x": class_1_data[:, 0], "y": class_1_data[:, 1], "class": "Class 1"}
        data_df = pd.DataFrame(class_0_df)._append(pd.DataFrame(class_1_df))

        x_min, x_max = data[:, 0].min() - 1, data[:, 0].max() + 1
        y_min, y_max = data[:, 1].min() - 1, data[:, 1].max() + 1
        xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100), np.linspace(y_min, y_max, 100))
        grid = np.c_[xx.ravel(), yy.ravel()]
        predictions = neural_net.predict(grid)
        zz = predictions[:, 1].reshape(xx.shape)

        fig = px.scatter(data_df, x="x", y="y", color="class", title="Neural Network Decision Boundary")
        fig.add_contour(x=xx[0], y=yy[:,0], z=zz, colorscale="Blues", opacity=0.3)

        st.plotly_chart(fig)

if __name__ == '__main__':
    main()
