# q_network.py
import numpy as np
import random

class QNetwork:
    def __init__(self, input_dim, hidden_dim, output_dim, alpha=0.001, gamma=0.99):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.alpha = alpha
        self.gamma = gamma

        self.W1 = np.random.randn(input_dim, hidden_dim) * np.sqrt(2. / input_dim)
        self.W2 = np.random.randn(hidden_dim, output_dim) * np.sqrt(2. / hidden_dim)
        self.W1_target = np.copy(self.W1)
        self.W2_target = np.copy(self.W2)

    def forward_pass(self, state):
        if state is None:
            print("State is None!")
            return None, None
        state_array = np.array(state)  # Convert the list to a NumPy array
        state_flattened = state_array.flatten()  # Now you can call .flatten()
        z1 = state_flattened.dot(self.W1)
        a1 = np.tanh(z1)
        z2 = a1.dot(self.W2)
        return z2, a1


    def train(self, state, action, reward, next_state):
        q_values, a1 = self.forward_pass(state)
        next_q_values, _ = self.forward_pass(next_state)
        
        target = q_values.copy()
        target[action] = reward + self.gamma * np.max(next_q_values)
        
        loss = np.sum(np.square(target - q_values))
        
        delta = target - q_values
        dW2 = a1.T.dot(delta)
        dW1 = state.T.dot((delta.dot(self.W2.T) * (a1 > 0)))  # Gradient through ReLU
    
        self.W1 += self.alpha * dW1
        self.W2 += self.alpha * dW2
        
        return loss

    def update(self, state, action, target, learning_rate=0.01):
        state = np.array(state).flatten()

        # Forward pass
        q_values, hidden_layer_output = self.forward_pass(state)
        
        # Compute the loss (Mean Squared Error)
        loss = (q_values[action] - target) ** 2
        
        # Compute the gradient (Backpropagation)
        delta_q = 2 * (q_values[action] - target)
        
        # Gradients for W2
        gradient_W2 = hidden_layer_output * delta_q
        
        # Gradients for W1
        delta_hidden = (1 - hidden_layer_output ** 2) * (self.W2[:, action] * delta_q)

        gradient_W1 = state[:, np.newaxis] * delta_hidden

        
        # Update weights
        self.W1 -= learning_rate * gradient_W1
        self.W2[:, action] -= learning_rate * gradient_W2


    def update_target_network(self):
        self.W1_target = np.copy(self.W1)
        self.W2_target = np.copy(self.W2)

    def epsilon_greedy(self, q_values, epsilon):
        if np.random.rand() < epsilon:
            return np.random.randint(self.output_dim)
        else:
            return np.argmax(q_values)
