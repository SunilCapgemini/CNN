import torch
import torch.nn as nn
import torch.optim as optim
from cnn_model import CNN
import numpy as np

def explain_backward_pass():
    """
    Detailed explanation of backward pass through each layer of the CNN
    with mathematical notations and examples.
    """
    # Create a small batch of data for demonstration
    batch_size = 2
    input_data = torch.randn(batch_size, 1, 28, 28, requires_grad=True)
    target = torch.tensor([0, 1])  # Two examples with different target classes
    
    # Initialize model and optimizer
    model = CNN()
    optimizer = optim.SGD(model.parameters(), lr=0.01)
    criterion = nn.CrossEntropyLoss()
    
    """
    Forward Pass
    -----------
    We'll track intermediate values for backward pass explanation
    """
    # 1. Convolution Layer
    conv_output = model.conv1(input_data)
    conv_output.retain_grad()  # Retain gradient for conv_output
    conv_activated = torch.relu(conv_output)
    conv_activated.retain_grad()  # Retain gradient for conv_activated
    
    # 2. Average Pooling
    pooled = model.avg_pool(conv_activated)
    pooled.retain_grad()  # Retain gradient for pooled output
    
    # 3. Flatten
    flattened = pooled.view(pooled.size(0), -1)
    flattened.retain_grad()  # Retain gradient for flattened output
    
    # 4. Fully Connected + Softmax
    logits = model.fc(flattened)
    logits.retain_grad()  # Retain gradient for logits
    output = torch.softmax(logits, dim=1)
    
    # Calculate loss
    loss = criterion(logits, target)
    
    """
    Backward Pass Explanation
    ------------------------
    """
    # Clear any existing gradients
    optimizer.zero_grad()
    
    # Compute gradients
    loss.backward()
    
    print("Backward Pass Analysis:")
    print("======================")
    
    """
    1. Softmax and Cross-Entropy Gradient
    -----------------------------------
    For softmax followed by cross-entropy loss:
    dL/dz_i = p_i - y_i
    where:
    - p_i is the softmax probability for class i
    - y_i is 1 for correct class, 0 otherwise
    """
    print("\n1. Softmax and Cross-Entropy Gradient:")
    print(f"Output probabilities: \n{output.detach().numpy()}")
    print(f"Gradient at logits: \n{logits.grad}")
    
    """
    2. Fully Connected Layer Gradient
    -------------------------------
    dL/dW = dL/dy * x^T
    dL/db = dL/dy
    where:
    - x is the input (flattened features)
    - W is the weight matrix
    """
    print("\n2. Fully Connected Layer Gradient:")
    print(f"FC weights gradient shape: {model.fc.weight.grad.shape}")
    print(f"FC bias gradient shape: {model.fc.bias.grad.shape}")
    
    """
    3. Flatten Layer Gradient
    -----------------------
    Gradient simply gets reshaped back to original dimensions
    No parameters to update
    """
    print("\n3. Flatten Layer Gradient:")
    print(f"Gradient shape before unflatten: {flattened.grad.shape}")
    
    """
    4. Average Pooling Gradient
    -------------------------
    For 2x2 average pooling:
    - Gradient gets upsampled
    - Each gradient value gets distributed equally to all positions
    - dL/dx_{i,j} = (1/4) * dL/dy_{i//2,j//2}
    """
    print("\n4. Average Pooling Gradient:")
    print(f"Pooling input gradient shape: {conv_activated.grad.shape}")
    
    """
    5. ReLU Gradient
    --------------
    dL/dx_i = {
        dL/dy_i  if x_i > 0
        0        if x_i ≤ 0
    }
    """
    print("\n5. ReLU Gradient:")
    relu_mask = (conv_output > 0).float()
    print(f"Number of active ReLU units: {relu_mask.sum().item()}")
    
    """
    6. Convolution Layer Gradient
    ---------------------------
    For each filter f:
    dL/df = input * dL/doutput (convolution operation)
    where * denotes cross-correlation
    """
    print("\n6. Convolution Layer Gradient:")
    print(f"Conv weights gradient shape: {model.conv1.weight.grad.shape}")
    print(f"Conv bias gradient shape: {model.conv1.bias.grad.shape}")
    
    # Update weights
    optimizer.step()
    
    return {
        'loss': loss.item(),
        'output': output.detach().numpy(),
        'gradients': {
            'conv': model.conv1.weight.grad.norm().item(),
            'fc': model.fc.weight.grad.norm().item()
        }
    }

if __name__ == "__main__":
    num_epochs = 10
    for epoch in range(num_epochs):
        results = explain_backward_pass()
        print(f"\nEpoch {epoch + 1}/{num_epochs}")
        print(f"Loss: {results['loss']:.4f}")
        print(f"Conv Layer Gradient Norm: {results['gradients']['conv']:.4f}")
        print(f"FC Layer Gradient Norm: {results['gradients']['fc']:.4f}")
    print("\nTraining Results:")
    print("================")
    print(f"Loss: {results['loss']}")
    print(f"Conv Layer Gradient Norm: {results['gradients']['conv']}")
    print(f"FC Layer Gradient Norm: {results['gradients']['fc']}")
