import pyn
import pickle

nn = pyn.NN([pyn.Layer(103, activation=pyn.relu, derivative=pyn.relu_derivative),
            pyn.Layer(64, activation=pyn.relu, derivative=pyn.relu_derivative),
            pyn.Layer(32, activation=pyn.relu, derivative=pyn.relu_derivative),
            pyn.Layer(16, activation=pyn.relu, derivative=pyn.relu_derivative),
            pyn.Layer(100, activation=pyn.sigmoid, derivative=pyn.sigmoid_derivative)])

nn.init_weights()

with open("dataset.pkl", 'rb') as f:
    dataset = pickle.load(f)

print(len(dataset))

nn.load_dataset(dataset)

nn.train(epochs=1000, learnRate=0.001, batch_size=32)

nn.save("model1.pyn")