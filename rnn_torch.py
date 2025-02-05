"""
Simple RNN Implementation for Sequence Prediction
This code demonstrates how to implement a basic RNN using PyTorch to predict the next number in a sequence.
The model learns to predict the next number given a sequence of three consecutive numbers.

Example:
    Input sequence: [12, 13, 14]
    Expected output: 15 (as it follows the increment-by-1 pattern)
"""

import torch
import torch.nn as nn
import numpy as np

class SimpleRNN(nn.Module):
    """
    A simple RNN model that learns to predict the next number in a sequence.
    
    Architecture:
    1. RNN Layer: Processes the input sequence
    2. Fully Connected Layer: Maps RNN output to predicted value
    
    Args:
        input_size (int): Number of features in input (1 for single numbers)
        hidden_size (int): Number of features in hidden state
        output_size (int): Number of features in output (1 for single number prediction)
    """
    def __init__(self, input_size, hidden_size, output_size):
        super(SimpleRNN, self).__init__()
        self.hidden_size = hidden_size
        # RNN layer with batch_first=True means input shape is (batch, seq_len, features)
        self.rnn = nn.RNN(input_size, hidden_size, batch_first=True)
        # Linear layer to produce final output
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        """
        Forward pass of the model.
        
        Args:
            x: Input tensor of shape (batch_size, sequence_length, input_size)
               Example: For batch_size=1, sequence_length=3, input_size=1
                       x shape would be (1, 3, 1) containing [12, 13, 14]
        
        Returns:
            Predicted next value in sequence
        """
        # Initialize hidden state with zeros
        # Shape: (num_layers=1, batch_size, hidden_size)
        h0 = torch.zeros(1, x.size(0), self.hidden_size)
        
        # Forward propagate through RNN
        # out shape: (batch_size, sequence_length, hidden_size)
        out, _ = self.rnn(x, h0)
        
        # Take only the last time step output
        # Shape after indexing: (batch_size, hidden_size)
        out = self.fc(out[:, -1, :])
        return out

# Generate training sequence (0 to 14)
sequence = np.array([i for i in range(15)], dtype=np.float32)

def create_dataset(sequence, n_steps):
    """
    Create input-output pairs for training.
    
    Args:
        sequence: Array of numbers
        n_steps: Length of input sequence
    
    Example:
        sequence = [0,1,2,3,4]
        n_steps = 3
        Returns:
        X = [[0,1,2], [1,2,3]]  # Input sequences
        y = [3, 4]              # Target values
    """
    X, y = [], []
    for i in range(len(sequence) - n_steps):
        seq_x, seq_y = sequence[i:i+n_steps], sequence[i+n_steps]
        X.append(seq_x)
        y.append(seq_y)
    return np.array(X), np.array(y)

# Create sequences of 3 numbers as input
n_steps = 3
X, y = create_dataset(sequence, n_steps)

# Reshape data for PyTorch RNN
# X shape: (num_sequences, sequence_length, features)
# Example: If sequence=[0,1,2,3,4], then X will contain:
# [[[0],[1],[2]], [[1],[2],[3]]] and y will be [3, 4]
X = torch.from_numpy(X).unsqueeze(-1)  # Add feature dimension
y = torch.from_numpy(y)

# Model Configuration
input_size = 1    # Each input is a single number
hidden_size = 32  # Size of hidden state
output_size = 1   # Predict a single number
model = SimpleRNN(input_size, hidden_size, output_size)

# Loss function and optimizer
criterion = nn.MSELoss()  # Mean Squared Error Loss
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Training loop
num_epochs = 1000
for epoch in range(num_epochs):
    model.train()
    # Forward pass
    outputs = model(X)
    # Compute loss
    loss = criterion(outputs, y)
    # Backward pass and optimize
    optimizer.zero_grad()  # Clear previous gradients
    loss.backward()       # Compute gradients
    optimizer.step()      # Update weights
    
    # Print progress
    if (epoch+1) % 100 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

# Make predictions
model.eval()  # Set model to evaluation mode
with torch.no_grad():  # No gradient computation needed for predictions
    # Test sequence [12, 13, 14]
    x_input = torch.from_numpy(np.array([12, 13, 14], dtype=np.float32)).unsqueeze(0).unsqueeze(-1)
    print("\nInput sequence:", [12, 13, 14])
    yhat = model(x_input)
    print(f"Predicted next value: {yhat.item():.1f}")  # Should predict close to 15
