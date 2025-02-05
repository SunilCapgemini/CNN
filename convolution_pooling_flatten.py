import numpy as np

# Define the input image of size 28x28
image = np.random.rand(28, 28)

# Layer 1: Convolution with 15 filters of size 5x5
num_filters = 15
filter_size = 5
conv_output_size = 28 - filter_size + 1

# Initialize convolution output
conv_output = np.zeros((num_filters, conv_output_size, conv_output_size))

# Apply convolution
for f in range(num_filters):
    filter = np.random.rand(filter_size, filter_size)
    for i in range(conv_output_size):
        for j in range(conv_output_size):
            conv_output[f, i, j] = np.sum(image[i:i+filter_size, j:j+filter_size] * filter)

print(f"Shape after Layer 1 (Convolution): {conv_output.shape}")

# Layer 2: Average Pooling with pool size 2x2 and stride 2
pool_size = 2
stride = 2
pool_output_size = conv_output_size // stride

# Initialize pooling output
pool_output = np.zeros((num_filters, pool_output_size, pool_output_size))

# Apply average pooling
for f in range(num_filters):
    for i in range(0, conv_output_size, stride):
        for j in range(0, conv_output_size, stride):
            pool_output[f, i//stride, j//stride] = np.mean(conv_output[f, i:i+pool_size, j:j+pool_size])

print(f"Shape after Layer 2 (Average Pooling): {pool_output.shape}")

# Layer 3: Flatten the data
flattened_output = pool_output.flatten()
print(f"Shape after Layer 3 (Flattening): {flattened_output}")

# Layer 4: Fully Connected Layer
num_units = 10
fc_output = np.zeros((num_units))
for i in range(flattened_output.shape[0]):
    for j in range(num_units):
        weights = np.random.rand(flattened_output.shape[0], num_units)
        biases = np.random.rand(num_units)
        fc_output[j] += flattened_output[i] * weights[i, j] + biases[j]
        
print(f"Shape after Layer 4 (Fully Connected): {fc_output.shape}")

# Layer 5: Softmax
softmax_output = np.exp(fc_output - np.max(fc_output)) / np.sum(np.exp(fc_output - np.max(fc_output)))
print(f"Shape after Layer 5 (Softmax): {softmax_output.shape}")

# Layer 6: Output
output = np.argmax(softmax_output)
print(f"Output after Layer 6 (Output): {output}")