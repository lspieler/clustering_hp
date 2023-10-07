import numpy as np

class FFNN:

    def __init__(self, input_size, output_size, hidden_sizes, environment=None):
        combined_sizes = [input_size] + hidden_sizes + [output_size]
        # Create random weight matrices for each layer. The +1 is for the bias.
        self.weights = [np.random.rand(combined_sizes[i] + 1, combined_sizes[i + 1]) for i in range(len(combined_sizes) - 1)]
    
    def train(self, X, Y, alpha, alpha_decay, epochs, verbose_freq = 0):
        self.alpha = alpha
        self.alpha_decay = alpha_decay
        for i in range(epochs):
            if verbose_freq != 0 and i % verbose_freq == 0:
                print("Epoch: ", i)
            pred = self._iterate(X)
            self._backprop(Y, pred)
            self.alpha *= self.alpha_decay
        return pred

    def test(self, X):
        X = np.concatenate((X, np.ones((X.shape[0], 1))*-1), axis=1)
        for i in range(len(self.weights)):
            X = np.dot(X, self.weights[i])
            # Add bias input
            X = np.concatenate((X, np.ones((X.shape[0], 1))*-1), axis=1)
            if i != len(self.weights) - 1:
                X = self._sigmoid(X)
        return X[:, :-1]

    def save_model(self, filename):
        np.save(filename, self.weights)
    
    def load_model(self, filename):
        self.weights = np.load(filename, allow_pickle=True)
    
    def _iterate(self, X):
        X = np.concatenate((X, np.ones((X.shape[0], 1))*-1), axis=1)
        self._weighted_sums = [X]
        self._activations = [X]
        for i in range(len(self.weights)):
            X = np.dot(X, self.weights[i])
            X = np.concatenate((X, np.ones((X.shape[0], 1))*-1), axis=1)
            self._weighted_sums.append(X)
            if i != len(self.weights) - 1:
                X = self._sigmoid(X)
            self._activations.append(X)
        return X[:, :-1]
    
    def _backprop(self, Y, pred):
        # Y = np.concatenate((Y, np.ones((Y.shape[0], 1))*-1), axis=1)
        deltas = [(Y - pred).T] # self._sigmoid(self._weighted_sums[-1], deriv=True)
        for i in range(len(self.weights) - 1, 0, -1):
            sig = self._sigmoid(self._weighted_sums[i], deriv=True)
            delta = np.dot(self.weights[i], deltas[-1]) * sig.T
            # Will be applied to previous weight matrix, so remove bias row (non-existant neuron)
            deltas.append(delta[:-1])
        deltas.reverse()
        for i in range(0, len(self.weights)):
            self.weights[i] += self.alpha * np.dot(deltas[i], self._activations[i]).T 

    def _sigmoid(self, a, deriv=False):
        if not deriv:
            return 1 / (1 + np.exp(-a))
        else:
            return np.exp(-a) / ((1 + np.exp(-a)) ** 2)
    
    def _fake_relu(self, a, deriv=False):
        if not deriv:
            return np.minimum(np.maximum(-0.5, a/4), 0.5)
        else:
            return np.where((a/4 >= -0.5) & (a/4 <= 0.5), 1/4, 0)