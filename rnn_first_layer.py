import numpy as np

class RNNFirstLayer:
    """
    First layer of a Recurrent Neural Network (RNN)
    
    Input Shape Explanation:
    ----------------------
    - batch_size: number of sequences processed in parallel
    - sequence_length: number of time steps in each sequence
    - input_size: dimension of input features at each time step
    
    Example: For a batch of 32 sequences, each containing 10 time steps,
    where each time step has 50 features, the input shape would be (32, 10, 50)
    """
    
    def __init__(self, input_size, hidden_size):
        """
        Initialize RNN first layer parameters
        
        Parameters:
        -----------
        input_size: int
            Size of input features at each time step
        hidden_size: int
            Size of hidden state vector
            
        Mathematical Components:
        ----------------------
        1. Wx: Weight matrix for input-to-hidden connections
        2. Wh: Weight matrix for hidden-to-hidden connections
        3. b: Bias vector
        4. tanh: Activation function
        
        The forward pass computes:
        h_t = tanh(Wx * x_t + Wh * h_(t-1) + b)
        
        where:
        - h_t is the hidden state at time t
        - x_t is the input at time t
        - h_(t-1) is the hidden state from previous time step
        """
        # Initialize weights with random values scaled by 0.01
        self.Wx = np.random.randn(input_size, hidden_size) * 0.01  # Input to hidden
        self.Wh = np.random.randn(hidden_size, hidden_size) * 0.01  # Hidden to hidden
        self.b = np.zeros((1, hidden_size))  # Bias
        
        self.hidden_size = hidden_size
        self.input_size = input_size
        
    def forward(self, x, h_prev=None):
        """
        Forward pass for the first RNN layer
        
        Parameters:
        -----------
        x : numpy array
            Input data of shape (batch_size, sequence_length, input_size)
        h_prev : numpy array, optional
            Initial hidden state of shape (batch_size, hidden_size)
            
        Returns:
        --------
        hidden_states : numpy array
            All hidden states for each time step
            Shape: (batch_size, sequence_length, hidden_size)
        """
        batch_size, sequence_length, _ = x.shape
        
        # Initialize hidden state if not provided
        if h_prev is None:
            h_prev = np.zeros((batch_size, self.hidden_size))
            
        # Store all hidden states
        hidden_states = np.zeros((batch_size, sequence_length, self.hidden_size))
        
        # Process each time step
        for t in range(sequence_length):
            # Current input: shape (batch_size, input_size)
            x_t = x[:, t, :]
            
            # 1. Input transformation: x_t @ Wx
            input_transform = np.dot(x_t, self.Wx)  # Shape: (batch_size, hidden_size)
            
            # 2. Hidden state transformation: h_prev @ Wh
            hidden_transform = np.dot(h_prev, self.Wh)  # Shape: (batch_size, hidden_size)
            
            # 3. Combine transformations and add bias
            # h_t = tanh(x_t @ Wx + h_prev @ Wh + b)
            h_prev = np.tanh(input_transform + hidden_transform + self.b)
            
            # Store current hidden state
            hidden_states[:, t, :] = h_prev
            
        return hidden_states

# Example usage
if __name__ == "__main__":
    # Example parameters
    batch_size = 32
    sequence_length = 10
    input_size = 50
    hidden_size = 64
    
    # Create sample input
    x = np.random.randn(batch_size, sequence_length, input_size)
    
    # Initialize RNN layer
    rnn_layer = RNNFirstLayer(input_size, hidden_size)
    
    # Forward pass
    hidden_states = rnn_layer.forward(x)
    
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {hidden_states.shape}")
    print("\nShape explanation:")
    print(f"- Batch size: {batch_size} sequences")
    print(f"- Sequence length: {sequence_length} time steps")
    print(f"- Input size: {input_size} features per time step")
    print(f"- Hidden size: {hidden_size} dimensions in hidden state")
