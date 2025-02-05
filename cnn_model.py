import torch
import torch.nn as nn
import torch.nn.functional as F

class CNN(nn.Module):
    def __init__(self, num_classes=10):
        super(CNN, self).__init__()
        
        # Layer 1: Convolutional Layer
        # Input: (batch_size, 1, 28, 28) - Single channel 28x28 image
        # Conv2d parameters: (in_channels, out_channels, kernel_size)
        # Output: (batch_size, 15, 24, 24)
        # Example: If input is a digit '7', conv filters might detect edges, curves
        # Some filters might activate strongly on vertical lines (common in '7')
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=15, kernel_size=5)
        
        # Layer 2: Average Pooling Layer
        # Input: (batch_size, 15, 24, 24)
        # AvgPool2d parameters: kernel_size=2, stride=2
        # Output: (batch_size, 15, 12, 12)
        # Example: If a filter detected a vertical line, pooling will preserve this
        # feature while reducing spatial dimensions by taking average in 2x2 windows
        self.avg_pool = nn.AvgPool2d(kernel_size=2, stride=2)
        
        # Layer 3: Flatten Layer
        # Input: (batch_size, 15, 12, 12)
        # Output: (batch_size, 15 * 12 * 12) = (batch_size, 2160)
        # Example: Converting the 3D feature maps into a 1D vector
        # If we had activation showing a '7', it's now a long feature vector
        
        # Layer 4: Fully Connected Layer
        # Input: (batch_size, 2160)
        # Output: (batch_size, num_classes)
        # Example: Final layer that maps our features to class scores
        # High values in certain positions indicate likelihood of corresponding digit
        self.fc = nn.Linear(15 * 12 * 12, num_classes)

    def forward(self, x):
        # Print original input shape
        print(f"Input shape: {x.shape}")
        
        # Apply convolution and ReLU
        x = F.relu(self.conv1(x))
        print(f"After Conv1 + ReLU shape: {x.shape}")
        
        # Apply average pooling
        x = self.avg_pool(x)
        print(f"After Average Pooling shape: {x.shape}")
        
        # Flatten the output
        x = x.view(x.size(0), -1)
        print(f"After Flatten shape: {x.shape}")
        
        # Apply final fully connected layer
        x = self.fc(x)
        print(f"Output shape: {x.shape}")
        
        # Apply softmax
        x = F.softmax(x, dim=1)
        
        return x

# Example usage:
if __name__ == "__main__":
    # Create a random batch of 5 images
    batch_size = 5
    input_tensor = torch.randn(batch_size, 1, 28, 28)
    
    # Initialize the model
    model = CNN()
    
    # Forward pass
    output = model(input_tensor)
    
    # Example of using CrossEntropyLoss
    criterion = nn.CrossEntropyLoss()
    target = torch.randint(0, 10, (batch_size,))  # Random targets for demonstration
    loss = criterion(output, target)
    print(f"\nExample loss value: {loss.item()}")
