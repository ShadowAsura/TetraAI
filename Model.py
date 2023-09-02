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
        z1 = state.dot(self.W1)
        a1 = np.maximum(z1, 0)
        q_values = a1.dot(self.W2)
        return q_values, a1

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

    def update_target_network(self):
        self.W1_target = np.copy(self.W1)
        self.W2_target = np.copy(self.W2)

    def epsilon_greedy(self, q_values, epsilon):
        if np.random.rand() < epsilon:
            return np.random.randint(self.output_dim)
        else:
            return np.argmax(q_values)
