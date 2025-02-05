import numpy as np

def apply_convolution(input_batch, filters):
    """
    Apply convolution operation with multiple filters
    input_batch shape: (batch_size, height, width)
    filters shape: (num_filters, filter_height, filter_width)
    """
    batch_size = input_batch.shape[0]
    input_height, input_width = 28, 28
    num_filters, filter_height, filter_width = filters.shape
    
    # Calculate output dimensions
    output_height = input_height - filter_height + 1
    output_width = input_width - filter_width + 1
    
    # Initialize output
    output = np.zeros((batch_size, num_filters, output_height, output_width))
    
    # Perform convolution for each image in batch
    for b in range(batch_size):
        for f in range(num_filters):
            for i in range(output_height):
                for j in range(output_width):
                    output[b, f, i, j] = np.sum(
                        input_batch[b, i:i+filter_height, j:j+filter_width] * 
                        filters[f]
                    )
    
    print(f"Shape after convolution: {output.shape}")
    return output

def average_pooling(conv_output, pool_size=2):
    """
    Apply average pooling
    conv_output shape: (batch_size, num_filters, height, width)
    """
    batch_size, num_filters, height, width = conv_output.shape
    pooled_height = height // pool_size
    pooled_width = width // pool_size
    
    # Initialize output
    pooled = np.zeros((batch_size, num_filters, pooled_height, pooled_width))
    
    # Perform average pooling
    for b in range(batch_size):
        for f in range(num_filters):
            for i in range(pooled_height):
                for j in range(pooled_width):
                    pooled[b, f, i, j] = np.mean(
                        conv_output[b, f,
                                  i*pool_size:(i+1)*pool_size,
                                  j*pool_size:(j+1)*pool_size]
                    )
    
    print(f"Shape after average pooling: {pooled.shape}")
    return pooled

def flatten_layer(pooled_output):
    """
    Flatten the pooled output
    pooled_output shape: (batch_size, num_filters, height, width)
    """
    batch_size = pooled_output.shape[0]
    flattened = pooled_output.reshape(batch_size, -1)
    print(f"Shape after flattening: {flattened.shape}")
    return flattened

# Main execution
if __name__ == "__main__":
    # Generate random batch of 32 images (28x28)
    batch_size = 32
    input_images = np.random.randn(batch_size, 28, 28)
    print(f"Input shape: {input_images.shape}")
    
    # Create 15 random filters of size 5x5
    filters = np.random.randn(15, 5, 5)
    print(f"Filters shape: {filters.shape}")
    
    # Apply convolution
    conv_output = apply_convolution(input_images, filters)
    
    # Apply average pooling
    pooled_output = average_pooling(conv_output)
    
    # Flatten the output
    flattened_output = flatten_layer(pooled_output)
