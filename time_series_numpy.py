"""
Time Series RNN Implementation using NumPy
This implementation creates a vanilla RNN for time series prediction.
It includes both forward and backward propagation, and can be used
for various time series tasks like prediction and forecasting.
"""

import numpy as np

class TimeSeriesRNN:
    def __init__(self, input_size, hidden_size, output_size, sequence_length, learning_rate=0.01):
        """
        Initialize the RNN for time series prediction
        
        Args:
            input_size (int): Number of input features
            hidden_size (int): Number of neurons in hidden layer
            output_size (int): Number of output features
            sequence_length (int): Length of input sequence
            learning_rate (float): Learning rate for gradient descent
        """
        # Model architecture parameters
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.sequence_length = sequence_length
        self.learning_rate = learning_rate
        
        # Initialize weights with random values
        # Weight matrices for input->hidden, hidden->hidden, and hidden->output
        self.Wxh = np.random.randn(hidden_size, input_size) * 0.01
        self.Whh = np.random.randn(hidden_size, hidden_size) * 0.01
        self.Why = np.random.randn(output_size, hidden_size) * 0.01
        
        # Bias terms
        self.bh = np.zeros((hidden_size, 1))
        self.by = np.zeros((output_size, 1))
        
        # Memory variables for backpropagation
        self.reset_memory()

    def reset_memory(self):
        """Reset memory variables used during forward and backward passes"""
        self.x_steps = {}
        self.h_steps = {}
        self.y_steps = {}
        self.h_raw_steps = {}  # Pre-activation hidden states
        
    def forward(self, x_sequence):
        """
        Forward pass of the RNN
        
        Args:
            x_sequence: Input sequence of shape (sequence_length, input_size)
        
        Returns:
            List of outputs for each time step
        """
        self.reset_memory()
        batch_size = 1  # We'll process one sequence at a time
        
        # Initialize first hidden state with zeros
        self.h_steps[-1] = np.zeros((self.hidden_size, batch_size))
        
        # Forward pass through time steps
        for t in range(self.sequence_length):
            # Reshape input to (input_size, 1)
            self.x_steps[t] = x_sequence[t].reshape(-1, 1)
            
            # Compute hidden state
            # h_t = tanh(Wxh * x_t + Whh * h_(t-1) + bh)
            self.h_raw_steps[t] = np.dot(self.Wxh, self.x_steps[t]) + \
                                 np.dot(self.Whh, self.h_steps[t-1]) + self.bh
            self.h_steps[t] = np.tanh(self.h_raw_steps[t])
            
            # Compute output
            # y_t = Why * h_t + by
            self.y_steps[t] = np.dot(self.Why, self.h_steps[t]) + self.by
            
        return self.y_steps

    def backward(self, x_sequence, y_targets, y_pred):
        """
        Backward pass to compute gradients
        
        Args:
            x_sequence: Input sequence
            y_targets: Target values
            y_pred: Predicted values from forward pass
            
        Returns:
            Gradients for all weights and biases
        """
        # Initialize gradient accumulators
        dWxh = np.zeros_like(self.Wxh)
        dWhh = np.zeros_like(self.Whh)
        dWhy = np.zeros_like(self.Why)
        dbh = np.zeros_like(self.bh)
        dby = np.zeros_like(self.by)
        
        # Initialize gradient of next hidden state
        dh_next = np.zeros_like(self.h_steps[0])
        
        # Backward through time
        for t in reversed(range(self.sequence_length)):
            # Gradient of output
            dy = y_pred[t] - y_targets[t]
            
            # Gradient of Why and by
            dWhy += np.dot(dy, self.h_steps[t].T)
            dby += dy
            
            # Gradient of hidden state
            dh = np.dot(self.Why.T, dy) + dh_next
            
            # Gradient through tanh
            dh_raw = (1 - self.h_steps[t] ** 2) * dh
            
            # Gradient of Wxh, Whh, and bh
            dWxh += np.dot(dh_raw, self.x_steps[t].T)
            dWhh += np.dot(dh_raw, self.h_steps[t-1].T)
            dbh += dh_raw
            
            # Gradient for next iteration
            dh_next = np.dot(self.Whh.T, dh_raw)
        
        # Clip gradients to prevent exploding gradients
        for grad in [dWxh, dWhh, dWhy, dbh, dby]:
            np.clip(grad, -5, 5, out=grad)
            
        return dWxh, dWhh, dWhy, dbh, dby

    def train_step(self, x_sequence, y_targets):
        """
        Perform one training step
        
        Args:
            x_sequence: Input sequence
            y_targets: Target values
            
        Returns:
            Loss value
        """
        # Forward pass
        y_pred = self.forward(x_sequence)
        
        # Compute loss (MSE)
        loss = 0
        for t in range(self.sequence_length):
            loss += np.sum((y_pred[t] - y_targets[t]) ** 2)
        loss /= self.sequence_length
        
        # Backward pass
        gradients = self.backward(x_sequence, y_targets, y_pred)
        dWxh, dWhh, dWhy, dbh, dby = gradients
        
        # Update weights and biases
        self.Wxh -= self.learning_rate * dWxh
        self.Whh -= self.learning_rate * dWhh
        self.Why -= self.learning_rate * dWhy
        self.bh -= self.learning_rate * dbh
        self.by -= self.learning_rate * dby
        
        return loss

    def predict(self, x_sequence):
        """
        Make predictions for a sequence
        
        Args:
            x_sequence: Input sequence
            
        Returns:
            Predicted values as numpy array
        """
        # Forward pass
        predictions = self.forward(x_sequence)
        # Convert predictions dictionary to array
        pred_array = np.array([pred.flatten() for pred in predictions.values()])
        return pred_array

# Example usage
if __name__ == "__main__":
    # Generate sample time series data (sine wave)
    t = np.linspace(0, 20, 100)
    data = np.sin(t)
    
    # Prepare sequences
    def create_sequences(data, seq_length):
        sequences = []
        targets = []
        for i in range(len(data) - seq_length):
            seq = data[i:i+seq_length]
            target = data[i+1:i+seq_length+1]
            sequences.append(seq)
            targets.append(target)
        return np.array(sequences), np.array(targets)
    
    # Parameters
    sequence_length = 10
    input_size = 1
    hidden_size = 16
    output_size = 1
    
    # Create and reshape sequences
    X, y = create_sequences(data, sequence_length)
    # Reshape X to have shape (num_sequences, sequence_length, input_size)
    X = X.reshape(-1, sequence_length, input_size)
    # Reshape y to have shape (num_sequences, sequence_length, output_size)
    y = y.reshape(-1, sequence_length, output_size)
    
    # Create model
    rnn = TimeSeriesRNN(input_size, hidden_size, output_size, sequence_length)
    
    # Training loop
    print("Training the model...")
    for epoch in range(100):
        total_loss = 0
        for i in range(len(X)):
            loss = rnn.train_step(X[i], y[i])
            total_loss += loss
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch + 1}, Loss: {total_loss/len(X):.4f}")
    
    # Make predictions
    print("\nMaking predictions...")
    test_sequence = X[-1]  # Use last sequence for testing
    predictions = rnn.predict(test_sequence)
    
    # Print results
    print("\nExample prediction:")
    print("Input sequence:", test_sequence.flatten()[:5], "...")
    print("Predicted next values:", predictions[:5, 0], "...")
    
    # Optional: Print actual vs predicted values
    print("\nComparison of actual vs predicted values:")
    actual_next = y[-1].flatten()[:5]
    predicted_next = predictions[:5, 0]
    print("Actual values:    ", actual_next)
    print("Predicted values: ", predicted_next)
    print("Mean squared error:", np.mean((actual_next - predicted_next) ** 2))
