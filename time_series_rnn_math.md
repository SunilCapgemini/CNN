# Mathematics Behind Time Series RNN Implementation

## 1. Architecture Overview

The Recurrent Neural Network (RNN) implemented for time series prediction consists of three main weight matrices and two bias vectors:

- `Wxh`: Input to hidden layer weights
- `Whh`: Hidden to hidden layer weights (recurrent connections)
- `Why`: Hidden to output layer weights
- `bh`: Hidden layer bias
- `by`: Output layer bias

## 2. Forward Propagation

### 2.1 Hidden State Computation

At each time step t, the hidden state is computed as:

```
h_t = tanh(Wxh * x_t + Whh * h_(t-1) + bh)
```

Where:
- `h_t`: Hidden state at time t
- `x_t`: Input at time t
- `h_(t-1)`: Hidden state from previous time step
- `tanh`: Hyperbolic tangent activation function

### 2.2 Output Computation

The output at each time step is computed as:

```
y_t = Why * h_t + by
```

## 3. Backpropagation Through Time (BPTT)

### 3.1 Loss Function

We use Mean Squared Error (MSE) loss:

```
L = (1/T) * Σ(y_pred - y_target)²
```

Where T is the sequence length.

### 3.2 Gradient Computation

The gradients are computed backwards through time:

1. **Output Layer Gradients**:
   ```
   dy = y_pred - y_target
   dWhy = dy * h_t^T
   dby = dy
   ```

2. **Hidden Layer Gradients**:
   ```
   dh = Why^T * dy + dh_next
   dh_raw = (1 - h_t^2) * dh  # derivative of tanh
   ```

3. **Weight Gradients**:
   ```
   dWxh = dh_raw * x_t^T
   dWhh = dh_raw * h_(t-1)^T
   dbh = dh_raw
   ```

### 3.3 Gradient Flow

The gradients flow backwards through the sequence:
```
dh_next = Whh^T * dh_raw
```

## 4. Weight Updates

The weights are updated using gradient descent:

```
Wxh = Wxh - learning_rate * dWxh
Whh = Whh - learning_rate * dWhh
Why = Why - learning_rate * dWhy
bh = bh - learning_rate * dbh
by = by - learning_rate * dby
```

## 5. Gradient Clipping

To prevent exploding gradients, we clip the gradients to a maximum absolute value:

```
clip(gradient, -5, 5)
```

## 6. Time Series Specific Considerations

### 6.1 Sequence Preparation

For a time series with values [x₁, x₂, x₃, x₄, ...], we create input-output pairs:
```
Input: [x₁, x₂, x₃] → Output: [x₂, x₃, x₄]
Input: [x₂, x₃, x₄] → Output: [x₃, x₄, x₅]
```

### 6.2 Prediction

For prediction, we:
1. Take the last known sequence
2. Feed it through the network
3. Get the next predicted value

## 7. Implementation Details

### 7.1 Initialization

Weights are initialized with small random values to break symmetry:
```
Wxh = randn(hidden_size, input_size) * 0.01
Whh = randn(hidden_size, hidden_size) * 0.01
Why = randn(output_size, hidden_size) * 0.01
```

### 7.2 Memory Management

The implementation maintains dictionaries to store:
- Input steps (x_steps)
- Hidden states (h_steps)
- Output values (y_steps)
- Raw hidden states before activation (h_raw_steps)

This allows for efficient backpropagation through time.

## 8. Hyperparameters

Key hyperparameters in the implementation:
- Learning rate: Controls the size of weight updates
- Hidden size: Number of neurons in hidden layer
- Sequence length: Number of time steps to consider
- Number of epochs: Number of training iterations

## 9. Activation Function

The implementation uses tanh activation function because:
1. Output range [-1, 1] helps with gradient flow
2. Centered around zero, helping with gradient-based learning
3. Smooth derivative suitable for backpropagation

The derivative of tanh used in backpropagation is:
```
tanh'(x) = 1 - tanh²(x)
```
