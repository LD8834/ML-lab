import numpy as np

# Activation function (step function)
def step_function(x):
    return np.where(x >= 0, 1, 0)

# Define the training data for the AND and OR functions
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y_and = np.array([[0], [0], [0], [1]])
y_or = np.array([[0], [1], [1], [1]])

# Define the Perceptron class
class Perceptron:
    def _init_(self, input_size, learning_rate=0.1, epochs=1000):
        self.weights = np.zeros((input_size, 1))
        self.bias = 0
        self.learning_rate = learning_rate
        self.epochs = epochs

    def train(self, X, y):
        for _ in range(self.epochs):
            for inputs, label in zip(X, y):
                inputs = inputs.reshape(-1, 1)
                prediction = step_function(np.dot(inputs.T, self.weights) + self.bias)
                error = label - prediction
                self.weights += self.learning_rate * error * inputs
                self.bias += self.learning_rate * error

    def predict(self, X):
        return step_function(np.dot(X, self.weights) + self.bias)

# Train and test the Perceptron for the AND function
perceptron_and = Perceptron(input_size=2)
perceptron_and.train(X, y_and)
print("AND Function Predictions:")
print(perceptron_and.predict(X))

# Train and test the Perceptron for the OR function
perceptron_or = Perceptron(input_size=2)
perceptron_or.train(X, y_or)
print("\nOR Function Predictions:")
print(perceptron_or.predict(X))

# Manually test specific input values
print("\nAND Function Prediction for input [1, 1]:")
print(perceptron_and.predict(np.array([[1, 1]])))

print("\nOR Function Prediction for input [0, 1]:")
print(perceptron_or.predict(np.array([[0, 1]])))
