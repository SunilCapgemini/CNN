# Convolutional Neural Network: Mathematical Analysis

## 1. Architecture Overview
Our CNN consists of the following layers:
- Input Layer (28×28 grayscale image)
- Convolutional Layer (15 filters of 5×5)
- ReLU Activation
- Average Pooling Layer (2×2)
- Flatten Layer
- Fully Connected Layer
- Softmax Activation

## 2. Forward Pass Mathematical Analysis

### 2.1 Convolutional Layer
The convolution operation for a single filter is defined as:

$$(f * g)[n] = \sum_{m=-\infty}^{\infty} f[m]g[n-m]$$

For our 2D image case with filter w:

$$z_{i,j} = \sum_{k=0}^{4}\sum_{l=0}^{4} w_{k,l} \cdot x_{i+k,j+l} + b$$

where:
- $z_{i,j}$ is the output at position (i,j)
- $w_{k,l}$ is the weight at position (k,l) in the 5×5 filter
- $x_{i+k,j+l}$ is the input value at offset (k,l) from position (i,j)
- $b$ is the bias term

Output dimensions: $(28-5+1) × (28-5+1) = 24×24$

### 2.2 ReLU Activation
ReLU function is defined as:

$$f(x) = \max(0,x)$$

Applied element-wise to the convolution output.

### 2.3 Average Pooling Layer
For a 2×2 window at position (i,j):

$$p_{i,j} = \frac{1}{4}\sum_{k=0}^{1}\sum_{l=0}^{1} x_{2i+k,2j+l}$$

Output dimensions: $12×12$ (halved in both dimensions)

### 2.4 Flatten Layer
Transformation from 3D to 1D:

$$f_{k} = x_{i,j,c}$$
where $k = i \cdot (W \cdot C) + j \cdot C + c$
- W: width of feature map
- C: number of channels

### 2.5 Fully Connected Layer
Linear transformation:

$$y = Wx + b$$

where:
- $W \in \mathbb{R}^{n_{out} × n_{in}}$
- $x \in \mathbb{R}^{n_{in}}$
- $b \in \mathbb{R}^{n_{out}}$

### 2.6 Softmax Function
For input vector $z \in \mathbb{R}^n$:

$$\sigma(z)_i = \frac{e^{z_i}}{\sum_{j=1}^n e^{z_j}}$$

## 3. Backward Pass Derivations

### 3.1 Softmax Derivative (First Principles)
Let's derive $\frac{\partial \sigma_i}{\partial z_j}$ from first principles:

For i = j:
$$\begin{align}
\frac{\partial \sigma_i}{\partial z_i} &= \frac{\partial}{\partial z_i} \frac{e^{z_i}}{\sum_{k=1}^n e^{z_k}} \\
&= \frac{e^{z_i} \sum_{k=1}^n e^{z_k} - e^{z_i} e^{z_i}}{(\sum_{k=1}^n e^{z_k})^2} \\
&= \sigma_i(1 - \sigma_i)
\end{align}$$

For i ≠ j:
$$\begin{align}
\frac{\partial \sigma_i}{\partial z_j} &= \frac{\partial}{\partial z_j} \frac{e^{z_i}}{\sum_{k=1}^n e^{z_k}} \\
&= -\frac{e^{z_i} e^{z_j}}{(\sum_{k=1}^n e^{z_k})^2} \\
&= -\sigma_i\sigma_j
\end{align}$$

### 3.2 Cross-Entropy Loss Gradient
For cross-entropy loss $L = -\sum_i y_i \log(\sigma_i)$:

$$\frac{\partial L}{\partial z_i} = \sigma_i - y_i$$

### 3.3 Fully Connected Layer Gradient
Weight gradient:
$$\frac{\partial L}{\partial W_{ij}} = \frac{\partial L}{\partial y_i} \cdot \frac{\partial y_i}{\partial W_{ij}} = \delta_i x_j$$

where $\delta_i$ is the error term from the previous layer.

### 3.4 Average Pooling Gradient
For 2×2 window:
$$\frac{\partial L}{\partial x_{i,j}} = \frac{1}{4} \frac{\partial L}{\partial p_{\lfloor i/2 \rfloor,\lfloor j/2 \rfloor}}$$

### 3.5 ReLU Gradient
$$\frac{\partial L}{\partial x_i} = \begin{cases} 
\frac{\partial L}{\partial y_i} & \text{if } x_i > 0 \\
0 & \text{if } x_i \leq 0
\end{cases}$$

### 3.6 Convolution Layer Gradient
For filter weights:
$$\frac{\partial L}{\partial w_{k,l}} = \sum_{i,j} x_{i+k,j+l} \frac{\partial L}{\partial z_{i,j}}$$

For input:
$$\frac{\partial L}{\partial x_{i,j}} = \sum_{k,l} w_{k,l} \frac{\partial L}{\partial z_{i-k,j-l}}$$

## 4. Implementation Notes

The gradient calculations follow the chain rule of calculus:
$$\frac{\partial L}{\partial \theta} = \frac{\partial L}{\partial y} \cdot \frac{\partial y}{\partial \theta}$$

where $\theta$ represents any parameter in the network.

During backpropagation, these gradients are computed in reverse order, starting from the loss function and moving backward through the network, updating each parameter using:

$$\theta_{new} = \theta_{old} - \alpha \frac{\partial L}{\partial \theta}$$

where $\alpha$ is the learning rate.
