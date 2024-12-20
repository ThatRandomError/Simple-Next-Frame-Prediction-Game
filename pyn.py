import pickle
import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return sigmoid(x) * (1 - sigmoid(x))

def relu(x):
    return np.maximum(0, x)
    
def relu_derivative(x):
    return np.where(x > 0, 1, 0)

class Layer:
    def __init__(self, size, activation = None, derivative = None):
        self.size = size
        if activation:
            self.activation = activation
        else:
            self.activation = self.none
        if derivative:
            self.derivative = derivative
        else:
            self.derivative = self.none_derivative

    @staticmethod
    def none(x):
        return 1 / (1 + np.exp(-x))
    
    @staticmethod
    def none_derivative(x):
        return x * (1 - x)

class NN:
    def __init__(self, architecture):
        self.architecture = architecture
        self.layers = []
        self.output = []
        self.dataset = []

        self.nn = []
        self.gradients = []

        self.activation_functions = []
        self.activation_derv_functions = []


        for index, i in enumerate(architecture):
            self.activation_functions.append(i.activation)
            self.activation_derv_functions.append(i.derivative)
            self.layers.append(i.size)

        for i in range(len(self.layers) - 1):
            self.gradients.append([
                np.zeros((self.layers[i], self.layers[i + 1])), 
                np.zeros(self.layers[i + 1])
            ])


    def sigmoid(x):
        print("Test")
        return 1 / (1 + np.exp(-x))

    def sigmoid_derivative(x):
        return x * (1 - x)

    def load_dataset(self, dataset):
        self.dataset = dataset

    def __cross_entropy_loss(self, output, target):
        # Avoid division by zero by adding a small epsilon value
        epsilon = 1e-9
        output = np.clip(output, epsilon, 1 - epsilon)  # To prevent log(0)
        
        # Binary Cross-Entropy (for binary classification)
        if len(target.shape) == 1 or target.shape[1] == 1:
            return -np.sum(target * np.log(output) + (1 - target) * np.log(1 - output))
        
        # Multi-Class Cross-Entropy (for multi-class classification)
        else:
            return -np.sum(target * np.log(output))


    def __mean_square_error(self, output, target):
        return np.sum((output - target) ** 2)

    def __apply_gradient(self, learnRate):
        for index, (weights, biases) in enumerate(self.nn):
            weights -= self.gradients[index][0] * learnRate
            biases -= self.gradients[index][1] * learnRate

    def __activations(self, inputs):
        activations = [np.array(inputs)]
        count = 0
        for weights, biases in self.nn:
            z = np.dot(activations[-1], weights) + biases
            activation = self.activation_functions[count](z)
            activations.append(activation)
            count += 1
        self.output = activations[-1]
        return activations
    
    def __backpropagate(self, activations, target):
        # Start with the error of the output layer
        delta = (activations[-1] - target) * self.activation_derv_functions[-1](activations[-1])
        self.gradients[-1][0] = np.outer(activations[-2], delta)
        self.gradients[-1][1] = delta

        # Backpropagate to the hidden layers (loop over all hidden layers in reverse)
        for l in range(2, len(self.layers)):  # Ensure it goes all the way to the first hidden layer
            # Calculate delta for hidden layers (error propagated back)
            sp = self.activation_derv_functions[-l](activations[-l])  # Derivative of activation
            delta = np.dot(delta, self.nn[-l + 1][0].T) * sp  # Propagate error to the current layer
            self.gradients[-l][0] = np.outer(activations[-l - 1], delta)
            self.gradients[-l][1] = delta


    def train(self, epochs, learnRate=0.01, batch_size=None, printdata=False, costfunction=None):
        # Choose the cost function
        if costfunction is None:
            self.__cost = self.__mean_square_error
        elif costfunction == "cross_entropy":
            self.__cost = self.__cross_entropy_loss
        else:
            self.__cost = costfunction

        for epoch in range(epochs):
            total_cost = 0
            
            # Shuffle the dataset for each epoch
            np.random.shuffle(self.dataset)
            
            # Mini-batching
            if batch_size:
                batches = [
                    self.dataset[i:i + batch_size] 
                    for i in range(0, len(self.dataset), batch_size)
                ]
            else:
                batches = [self.dataset]
            
            for batch in batches:
                for inputs, target in batch:
                    # Forward pass: Get activations
                    activations = self.__activations(inputs)

                    # Backpropagation
                    self.__backpropagate(activations, target)

                    # Apply gradients (update weights and biases)
                    self.__apply_gradient(learnRate)

                # Accumulate cost for the batch
                total_cost += self.__cost(activations[-1], target)
            
            # Print epoch data
            if printdata:
                print(f"Epoch {epoch + 1}/{epochs}, Cost: {total_cost / len(self.dataset)}")




    def forward(self, inputs):
        activations = np.array(inputs)
        count = 0
        for weights, biases in self.nn:
            activations = self.activation_functions[count](np.dot(activations, weights) + biases)
            count += 1
        self.output = activations
        return self.output

    def init_weights(self, mode = "random"):
        self.nn = []
        if mode == "random":
            for i in range(len(self.layers) - 1):
                self.nn.append([
                    np.random.randn(self.layers[i], self.layers[i + 1]),
                    np.random.randn(self.layers[i + 1])
                ])
        elif mode == "zeros":
            for i in range(len(self.layers) - 1):
                for i in range(len(self.layers) - 1):
                    self.nn.append([
                        np.zeros((self.layers[i], self.layers[i + 1])),
                        np.zeros(self.layers[i + 1])
                    ])

        elif mode == "ones":
            for i in range(len(self.layers) - 1):
                for i in range(len(self.layers) - 1):
                    self.nn.append([
                        np.ones((self.layers[i], self.layers[i + 1])),
                        np.ones(self.layers[i + 1])
                    ])

    def save(self, name):
        with open(name, 'wb') as f:
            pickle.dump(self.nn, f)
            pickle.dump(self.activation_functions, f)
            pickle.dump(self.activation_derv_functions, f)
            

    def load(self, name):
        with open(name, 'rb') as f:
            self.nn = pickle.load(f)
            self.activation_functions = pickle.load(f)
            self.activation_derv_functions = pickle.load(f)
