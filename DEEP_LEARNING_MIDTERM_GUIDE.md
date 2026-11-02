# Artificial Neural Networks & Deep Learning: Master Midterm Guide & Exam Archive
### Course: ANN & Deep Learning (Semester 5 | BS Artificial Intelligence, UET Peshawar)
### Author: Muhammad Arsalan (AI Engineer Intern @ FlyRank AI)
### Scope: Lectures 01 to 08, CNN Architectures & Fuzzy Logic Foundations (Midterm Scope)

---

## 1. Learning Philosophy & Cognitive Framework

### Philosophy: `FIRST LEARN → THEN UNDERSTAND → THEN APPLY → THEN BUILD`

Mastering deep learning requires rejecting passive memorization. Complex mathematical models become intuitive when grounded in:
1. **Production-Grade Analogies:** Mapping abstract tensor equations onto physical machinery.
2. **Outcome Prediction:** Calculating matrix dimensions, parameter counts, and loss directions *before* running computations.
3. **Productive Difficulty (The 15-Minute Struggle Rule):** Never immediately look up solutions or seek automated fixes. Work through the derivation or tensor mismatch manually for at least 15 minutes to cement synaptic learning.
4. **Line-by-Line Dissection:** Breaking down matrix orders ($m \times n$), chain rule branches, and adaptive momentum updates into atomic arithmetic steps.
5. **No Data Tampering:** Preserving raw experimental observations, logging real empirical curves (such as our Lab 3, 4, and 5 runs), and deriving conclusions strictly from observed data.

---

## 2. Consolidated Midterm Curriculum Blueprint

```
                      DEEP LEARNING MIDTERM ARCHIVE
                                    │
    ┌───────────────────────────────┼───────────────────────────────┐
    ▼                               ▼                               ▼
FOUNDATIONS & PIPELINE        NEURON MODELS & RULES         TRAINING & OPTIMIZATION
• AI vs ML vs DL Hierarchy    • McCulloch-Pitts Model       • Backpropagation & Chain Rule
• 4 Crucial ML/DL Axes        • Activation Functions        • Batch Regimes (Full, SGD, Mini)
• Linear Algebra & Tensors    • Adaline & Madaline          • Overfitting (Dropout, Early Stop)
• 9-Stage DS Pipeline         • 5 Classical Learning Rules  • Adaptive Optimizers (Adam, RMSprop)
                                    │
                                    ▼
                      ADVANCED VISION & INTEL SYSTEMS
                      • CNN Convolutions & Pooling
                      • Landmark Nets (VGG, ResNet)
                      • Fuzzy Logic, Mamdani & Centroid
```

---

### Lecture 1: Deep Learning Foundations & Linear Algebra

#### 1. The Intelligence Hierarchy
$$\text{Artificial Intelligence (AI)} \supset \text{Machine Learning (ML)} \supset \text{Neural Networks (ANN)} \supset \text{Deep Neural Networks (DNN)}$$
- **Deep Neural Network Definition:** A neural network architecture containing two or more hidden layers between the input and output layers ($\ge 2$ hidden layers).

#### 2. Traditional Machine Learning vs. Deep Learning (4 Crucial Axes)

| Axis | Traditional Machine Learning | Deep Learning |
|:---|:---|:---|
| **Data Volume Scaling** | Performance plateaus early; adding more data provides diminishing returns. | Performance scales monotonically with massive dataset sizes ($10^5$ to $10^9$ samples). |
| **Hardware Dependency** | Executes efficiently on standard CPUs. | Relies heavily on parallel tensor acceleration (GPUs, TPUs). |
| **Feature Engineering** | **Manual & Handcrafted:** Domain experts extract edge histograms, SIFT, or HOG descriptors. | **Automated & Hierarchical:** Learns representations end-to-end from raw data. |
| **Execution Profile** | Fast training (minutes to hours); fast inference. | Computationally intensive training (hours to weeks); optimized low-latency inference. |

#### 3. Progressive Feature Extraction
In deep vision pipelines, representation complexity grows depth-wise:
$$\text{Raw Pixels} \xrightarrow{\text{Layer 1}} \text{Edges / Gradients} \xrightarrow{\text{Layer 2}} \text{Textures / Corners} \xrightarrow{\text{Layer 3}} \text{Object Parts} \xrightarrow{\text{Output}} \text{Class Probabilities}$$

#### 4. Linear Algebra Foundations for Deep Learning
- **Tensors & Ranks:**
  - Rank 0: **Scalar** ($s \in \mathbb{R}$)
  - Rank 1: **Vector** ($\mathbf{v} \in \mathbb{R}^n$)
  - Rank 2: **Matrix** ($M \in \mathbb{R}^{m \times n}$)
  - Rank 3+: **Tensor** ($\mathcal{T} \in \mathbb{R}^{B \times H \times W \times C}$ for computer vision batches)
- **Matrix Multiplication Constraint:**
  Two matrices $A$ and $B$ can be multiplied if and only if the inner dimensions match:
  $$A_{m \times k} \times B_{k \times n} = C_{m \times n}, \quad C_{ij} = \sum_{p=1}^k A_{ip} B_{pj}$$
- **Artificial Neuron Mathematical Model:**
  $$z = \sum_{i=1}^n w_i x_i + b = W^T X + b, \quad \hat{y} = f(z)$$
  - **Weights ($W$):** Scale factor / slope controller that modulates feature importance.
  - **Bias ($b$):** Threshold shift controller independent of input features, allowing the decision boundary to shift away from the origin.

#### 5. Biological vs. Artificial Neuron Mapping

| Biological Feature | Artificial Component | Mathematical Function |
|:---|:---|:---|
| **Dendrites** | Input Lines ($x_1, x_2, \dots, x_n$) | Receives input signals/features from data or prior layers |
| **Synapses** | Weights ($w_1, w_2, \dots, w_n$) | Strength of connection; scales incoming signals |
| **Soma (Cell Body)** | Linear Combiner ($\sum w_i x_i + b$) | Sums weighted inputs and incorporates bias offset |
| **Axon Hillock** | Activation Function ($f(z)$) | Evaluates non-linear threshold/firing decision |
| **Axon** | Output Transmission ($\hat{y}$) | Propagates signal forward to subsequent neurons |

#### 6. Iteration Arithmetic
$$\text{Iterations (Weight Updates) per Epoch} = \left\lceil \frac{\text{Total Training Samples}}{\text{Batch Size}} \right\rceil$$

---

### Lecture 2: End-to-End Data Science & Modeling Pipeline

The systematic 9-stage engineering workflow:
1. **Problem Formulation:** Classify objective as classification, regression, clustering, or generation; define quantifiable success KPIs.
2. **Data Acquisition:** Secure raw data via database queries, web APIs, telemetry sensors, or public benchmarks.
3. **Exploratory Data Analysis (EDA):** Inspect distributions, detect skewness, evaluate correlations, identify outliers.
4. **Data Cleaning & Imputation:** Handle missing values (mean/median/KNN), remove invalid records, correct label inconsistencies.
5. **Feature Engineering & Transformation:** One-hot encoding, standard scaling ($z = \frac{x-\mu}{\sigma}$), min-max normalization, polynomial expansion.
6. **Model Architecture Selection:** Select baseline models (MLP, CNN, RNN) aligned with data dimensionality.
7. **Dataset Partitioning:** Strict separation into **Train** (parameter learning), **Validation** (hyperparameter tuning & early stopping), and **Held-out Test** (unbiased final evaluation).
8. **Quantitative Evaluation:**
   - *Classification:* Accuracy, Precision, Recall, F1-Score, ROC-AUC, Confusion Matrix.
   - *Regression:* Mean Absolute Error (MAE), Mean Squared Error (MSE), Root Mean Squared Error (RMSE), Coefficient of Determination ($R^2$).
9. **Deployment & Monitoring:** Containerization, model serving via REST/gRPC endpoints, drift detection, and automated retraining triggers.

---

### Lecture 3 & 4: Neuron Models, Activations & Learning Rules

#### 1. McCulloch-Pitts Neuron (1943)
- First computational abstraction of a biological neuron.
- Uses binary inputs ($x_i \in \{0, 1\}$) and produces binary output ($y \in \{0, 1\}$).
- **Two types of inputs:**
  - *Excitatory inputs:* Contribute $+1$ toward the firing threshold $\theta$.
  - *Inhibitory inputs:* Absolute veto power; if any inhibitory input is active ($1$), output is unconditionally $0$.
- **Firing Rule:**
  $$y = \begin{cases} 1 & \text{if } \sum_{i=1}^n x_i \ge \theta \text{ AND all inhibitory inputs } = 0 \\ 0 & \text{otherwise} \end{cases}$$
- **Boolean Implementations:**
  - **AND gate:** 2 excitatory inputs, $\theta = 2$.
  - **OR gate:** 2 excitatory inputs, $\theta = 1$.
  - **NOT gate:** 1 excitatory input with $\theta = 0$, 1 inhibitory input.

#### 2. Activation Functions Comparative Matrix

| Activation Function | Formula | Range | Key Strengths | Critical Weaknesses / Traps |
|:---|:---|:---:|:---|:---|
| **Sigmoid** | $\sigma(z) = \frac{1}{1 + e^{-z}}$ | $(0, 1)$ | Smooth probability output; historically used in binary classification output. | **Vanishing Gradient:** For $|z| > 4$, $\sigma'(z) \approx 0$; outputs are not zero-centered. |
| **Tanh** | $\tanh(z) = \frac{e^z - e^{-z}}{e^z + e^{-z}}$ | $(-1, 1)$ | Zero-centered; stronger gradient signals than sigmoid. | Still suffers from vanishing gradients at large positive and negative saturations. |
| **ReLU** | $f(z) = \max(0, z)$ | $[0, \infty)$ | Computationally fast (threshold at 0); avoids saturation for $z > 0$; promotes sparsity. | **Dying ReLU:** Neurons with negative activation for all inputs have gradient $0$ and never recover. |
| **Leaky ReLU** | $f(z) = \max(\alpha z, z), \; \alpha \approx 0.01$ | $(-\infty, \infty)$ | Solves the dying ReLU problem by ensuring non-zero gradient for $z < 0$. | Hyperparameter $\alpha$ must be preset or tuned. |
| **Softmax** | $\sigma(z)_i = \frac{e^{z_i}}{\sum_{j=1}^K e^{z_j}}$ | $(0, 1), \; \sum p_i = 1$ | Converts arbitrary real logits into a normalized probability distribution across $K$ mutually exclusive classes. | Sensitive to large outliers in logits before numerical stabilization (log-sum-exp trick). |

#### 3. Why Linear Stacking Fails (The Mathematical Necessity of Non-Linearity)
If activation functions were linear ($f(z) = c z$), stacking multiple layers collapses mathematically into a single linear transformation:
$$\hat{y} = W_2 (W_1 X + b_1) + b_2 = (W_2 W_1) X + (W_2 b_1 + b_2) = W' X + b'$$
A 100-layer linear network has zero additional expressive power over a single-layer perceptron and cannot solve non-linear problems like XOR.

#### 4. Adaline vs. Madaline

| Feature | Adaline (Adaptive Linear Neuron) | Madaline (Many Adaline) |
|:---|:---|:---|
| **Architecture** | Single processing unit with linear combiner and bipolar threshold. | Multi-layer array of Adalines feeding a fixed logic gate (AND, OR, Majority). |
| **Learning Rule** | **Widrow-Hoff Delta Rule (LMS):** Updates weights based on continuous error before quantization. | **Madaline Rule II (MRI):** Minimal disturbance principle; adjusts Adaline whose output is closest to switching. |
| **Linear Separability** | Restricted to linearly separable problems. | Capable of forming complex non-linear decision boundaries. |

#### 5. Classical Learning Rules
1. **Hebbian Learning Rule (1949):** *"Neurons that fire together, wire together."* Unsupervised: $\Delta w_i = \eta x_i y$.
2. **Perceptron Learning Rule (Rosenblatt, 1958):** Supervised error correction on quantized output: $\Delta w_i = \eta (y - \hat{y}) x_i$.
3. **Delta Rule / Widrow-Hoff LMS (1960):** Minimizes Mean Squared Error over continuous linear output: $\Delta w_i = \eta (y - W^T X) x_i$.
4. **Correlation Learning Rule:** Hebbian variant using supervised desired output: $\Delta w_i = \eta x_i y_{\text{target}}$.
5. **Outstar Learning Rule (Grossberg):** Used for pattern generation and memory recall; updates weights directed outward from a source neuron.

---

### Lecture 5 & 6: Training Dynamics, Backpropagation & Regularization

#### 1. Optimization & Loss Landscapes
- Loss function $J(W)$ maps parameter configurations to a scalar error penalty.
- **Convex Surface:** Single global minimum; guaranteed global convergence via gradient descent.
- **Non-Convex Surface:** Characterized by multiple local minima, saddle points (where gradient $\nabla J = 0$ but curvature has mixed signs), and flat plateaus.

#### 2. Backpropagation via the Multivariate Chain Rule
For a 2-layer network with MSE loss $J = \frac{1}{2}(\hat{y} - y)^2$:
$$\hat{y} = f(z_2), \quad z_2 = w_2 a_1 + b_2, \quad a_1 = f(z_1), \quad z_1 = w_1 x + b_1$$

The gradient with respect to the input-layer weight $w_1$ decomposes into an unbroken chain:
$$\frac{\partial J}{\partial w_1} = \underbrace{\frac{\partial J}{\partial \hat{y}}}_{(\hat{y} - y)} \cdot \underbrace{\frac{\partial \hat{y}}{\partial z_2}}_{f'(z_2)} \cdot \underbrace{\frac{\partial z_2}{\partial a_1}}_{w_2} \cdot \underbrace{\frac{\partial a_1}{\partial z_1}}_{f'(z_1)} \cdot \underbrace{\frac{\partial z_1}{\partial w_1}}_{x}$$

#### 3. Gradient Descent Batching Schemes

| Scheme | Batch Size ($B$) | Updates / Epoch | Trajectory Characteristic | Memory Demand | Vectorization Efficiency |
|:---|:---:|:---:|:---|:---:|:---:|
| **Full-Batch** | $B = N$ (Entire dataset) | $1$ | Smooth, deterministic, direct descent toward local minimum. | High ($O(N)$) | Moderate |
| **Stochastic (SGD)** | $B = 1$ (Single sample) | $N$ | Highly erratic, noisy, jagged trajectory; escapes shallow minima. | Minimal ($O(1)$) | None (cannot use matrix ops) |
| **Mini-Batch** | $1 < B < N$ (e.g., 32, 64) | $\lceil N/B \rceil$ | Balanced, steady descent with mild stochastic regularization. | Manageable ($O(B)$) | Optimal (maximizes GPU throughput) |

#### 4. Overfitting Countermeasures
- **Dropout (Srivastava et al., 2014):** During training, randomly zero-out each neuron's activation with probability $p$ (typically $p=0.5$). Forces network to learn redundant, robust feature representations without co-adaptation. At inference, all neurons are kept active and scaled by $(1-p)$.
- **Early Stopping:** Continuously track validation loss alongside training loss. When validation loss ceases to improve and begins rising for $k$ consecutive epochs (patience threshold), halt training and restore the model weights from the epoch with the lowest validation loss.

---

### Lecture 7 & 8: Adaptive Optimizers & Gradient Descent

#### 1. Mathematical Formulas & Update Dynamics

##### A. Classical Stochastic Gradient Descent (SGD)
$$w_{t+1} = w_t - \eta g_t$$
- Fixed learning rate $\eta$; treats all parameter dimensions identically regardless of gradient frequency or scale.

##### B. Adagrad (Adaptive Gradient Algorithm)
Accumulates squared gradients from the beginning of training:
$$G_t = G_{t-1} + g_t^2, \quad w_{t+1} = w_t - \frac{\eta}{\sqrt{G_t} + \epsilon} g_t$$
- *Strength:* Scales down updates for frequent features and scales up updates for infrequent features.
- *Critical Flaw:* $G_t$ increases monotonically ($G_t \ge G_{t-1}$), causing the effective learning rate $\frac{\eta}{\sqrt{G_t}+\epsilon}$ to shrink to near-zero, permanently halting learning before convergence.

##### C. RMSprop (Root Mean Square Propagation - Geoff Hinton)
Replaces unbounded accumulation with an exponentially decaying moving average of squared gradients:
$$v_t = \rho v_{t-1} + (1 - \rho) g_t^2, \quad w_{t+1} = w_t - \frac{\eta}{\sqrt{v_t} + \epsilon} g_t$$
- Typical decay rate $\rho = 0.9$. Solves Adagrad's premature stopping problem by discarding historical gradients from long ago.

##### D. Adadelta
Eliminates the arbitrary global learning rate parameter $\eta$ by tracking running averages of both squared gradients ($v_t$) and past squared parameter updates ($\Delta w$):
$$w_{t+1} = w_t - \frac{\sqrt{\Delta w_{t-1}^2 + \epsilon}}{\sqrt{v_t + \epsilon}} g_t$$

##### E. Adam (Adaptive Moment Estimation - Kingma & Ba, 2014)
Combines first-order momentum (running mean of gradients) and second-order momentum (uncentered variance of gradients) with initial bias correction:
$$m_t = \beta_1 m_{t-1} + (1 - \beta_1) g_t \quad (\text{1st Moment: Momentum})$$
$$v_t = \beta_2 v_{t-1} + (1 - \beta_2) g_t^2 \quad (\text{2nd Moment: Variance})$$
$$\hat{m}_t = \frac{m_t}{1 - \beta_1^t}, \quad \hat{v}_t = \frac{v_t}{1 - \beta_2^t} \quad (\text{Bias Correction for } t=1, 2, \dots)$$
$$w_{t+1} = w_t - \frac{\eta}{\sqrt{\hat{v}_t} + \epsilon} \hat{m}_t$$
- Standard default hyperparameters: $\eta = 0.001$, $\beta_1 = 0.9$, $\beta_2 = 0.999$, $\epsilon = 10^{-8}$.

---

### Advanced Vision & Intelligent Systems (CNNs & Fuzzy Logic)

#### 1. Convolutional Neural Networks (CNNs)
- **Convolutional Layer:** Operates via sliding spatial filters (kernels). Learns translation-invariant spatial patterns (edges, blobs).
  - **Output Spatial Dimension Formula:**
    $$O = \left\lfloor \frac{I - K + 2P}{S} \right\rfloor + 1$$
    where $I = \text{Input size}$, $K = \text{Kernel size}$, $P = \text{Padding}$, $S = \text{Stride}$.
- **Pooling Layer:** Non-parametric downsampling (Max Pooling, Average Pooling). Reduces spatial dimension, parameter count, and computational cost while providing minor spatial translation invariance.
- **Landmark Architectures:**
  - **AlexNet (2012):** 8 layers; pioneered GPU training, ReLU activations, and Dropout.
  - **VGGNet (2014):** Demonstrated that deep stacks of small $3 \times 3$ filters outperform large $7 \times 7$ filters with fewer parameters and greater non-linear capacity.
  - **ResNet (He et al., 2015):** Introduced **Residual Skip Connections**:
    $$H(x) = F(x) + x \implies F(x) = H(x) - x$$
    Allows gradient to flow directly back through the identity path ($\frac{\partial H}{\partial x} = \frac{\partial F}{\partial x} + 1$), enabling networks to scale to 152+ layers without vanishing gradients.

#### 2. Fuzzy Logic Systems
- **Crisp Set vs. Fuzzy Set:**
  - *Crisp Set:* Binary membership $\mu_A(x) \in \{0, 1\}$.
  - *Fuzzy Set:* Continuous degree of membership $\mu_A(x) \in [0, 1]$.
- **Core Architecture:**
  $$\text{Crisp Inputs} \xrightarrow{\text{Fuzzifier}} \text{Fuzzy Sets} \xrightarrow{\text{Inference Engine + Rule Base}} \text{Aggregated Fuzzy Output} \xrightarrow{\text{Defuzzifier}} \text{Crisp Output}$$
- **Fuzzy Inference Methods:**
  - **Mamdani:** Produces fuzzy output sets; intuitive, human-like reasoning.
  - **Sugeno (TSK):** Produces mathematical polynomial or constant outputs; computationally efficient.
- **Defuzzification (Centroid / Center of Gravity Method):**
  $$z^* = \frac{\int z \cdot \mu_C(z) \, dz}{\int \mu_C(z) \, dz} \quad \text{or for discrete points:} \quad z^* = \frac{\sum z_i \cdot \mu_C(z_i)}{\sum \mu_C(z_i)}$$

---

## 3. Midterm Examination Benchmark (Complete Solved Paper)

### Section A: Foundations, Linear Algebra & Neuron Models

#### Q1.1: Tensor Arithmetic & Reshaping
* **Problem:** Given a mini-batch of grayscale medical images of shape $(B, H, W) = (32, 28, 28)$:
  1. What is the rank of this tensor?
  2. How many total floating-point elements are contained in the tensor?
  3. If we flatten each image into a 1D feature vector for an MLP, what is the resulting tensor shape?
* **Solution:**
  1. **Rank:** $3$ (3 axes: Batch, Height, Width).
  2. **Total Elements:** $32 \times 28 \times 28 = 32 \times 784 = \mathbf{25{,}088}$ elements.
  3. **Flattened Shape:** $(32, 784)$ preserving total element count ($32 \times 784 = 25{,}088$).

#### Q1.2: Iterations per Epoch Calculation
* **Problem:** A dataset has $N = 3{,}000$ training images.
  1. How many weight updates occur in one epoch if using batch size $B = 64$?
  2. How many updates occur in 4 epochs?
* **Solution:**
  1. $\text{Updates per epoch} = \lceil 3000 / 64 \rceil = \lceil 46.875 \rceil = \mathbf{47}$ weight updates.
  2. In 4 epochs: $47 \times 4 = \mathbf{188}$ total weight updates.

#### Q1.3: McCulloch-Pitts Boolean Logic Realization
* **Problem:** Design a McCulloch-Pitts neuron to compute the Boolean function $Y = (X_1 \land \neg X_2)$.
* **Solution:**
  - $X_1$ is an **excitatory input** with weight $+1$.
  - $X_2$ is an **inhibitory input** (veto power).
  - Set threshold $\theta = 1$.
  - **Truth Table Verification:**
    - $(0, 0) \implies \text{Excitatory} = 0 < 1 \implies Y = 0$.
    - $(0, 1) \implies \text{Inhibited by } X_2 \implies Y = 0$.
    - $(1, 0) \implies \text{Excitatory} = 1 \ge 1 \text{ and not inhibited} \implies Y = 1$.
    - $(1, 1) \implies \text{Inhibited by } X_2 \implies Y = 0$.
  - Matches $X_1 \land \neg X_2$ identically.

---

### Section B: Backpropagation & Optimization

#### Q2.1: Backpropagation Step-by-Step Derivation
* **Problem:** For a single neuron $\hat{y} = \sigma(w \cdot x + b)$ trained with squared error $J = \frac{1}{2}(\hat{y} - y)^2$, derive the exact expression for $\frac{\partial J}{\partial w}$.
* **Solution:**
  - Let $z = w \cdot x + b$, so $\hat{y} = \sigma(z)$.
  - Derivative of Sigmoid: $\sigma'(z) = \sigma(z)(1 - \sigma(z)) = \hat{y}(1 - \hat{y})$.
  - Applying chain rule:
    $$\frac{\partial J}{\partial w} = \frac{\partial J}{\partial \hat{y}} \cdot \frac{\partial \hat{y}}{\partial z} \cdot \frac{\partial z}{\partial w}$$
  - Step 1: $\frac{\partial J}{\partial \hat{y}} = \hat{y} - y$
  - Step 2: $\frac{\partial \hat{y}}{\partial z} = \hat{y}(1 - \hat{y})$
  - Step 3: $\frac{\partial z}{\partial w} = x$
  - **Final Gradient:**
    $$\mathbf{\frac{\partial J}{\partial w} = (\hat{y} - y) \cdot \hat{y}(1 - \hat{y}) \cdot x}$$

#### Q2.2: Adagrad vs. RMSprop Comparative Analysis
* **Problem:** Explain why Adagrad fails on deep convex-loss networks and how RMSprop resolves this failure mathematically.
* **Solution:**
  - **Adagrad Failure:** Adagrad accumulates squared gradients monotonically: $G_t = G_{t-1} + g_t^2$. Because $g_t^2 \ge 0$, $G_t$ grows continuously. In the denominator of the update rule $\frac{\eta}{\sqrt{G_t}+\epsilon}$, as $G_t \to \infty$, the effective step size approaches $0$. The network stops learning prematurely before reaching the minimum.
  - **RMSprop Resolution:** RMSprop replaces the monotonic sum with an exponentially weighted moving average:
    $$v_t = \rho v_{t-1} + (1 - \rho) g_t^2$$
    Old gradients decay exponentially by factor $\rho$ (e.g., $0.9$). The denominator reflects only recent gradient magnitudes, ensuring steady progress throughout training.

---

### Section C: Convolutional Architectures & Intelligent Systems

#### Q3.1: CNN Feature Map Arithmetic
* **Problem:** An input image of size $28 \times 28 \times 1$ passes through a Conv2D layer with 16 filters of size $3 \times 3$, stride $S = 1$, and valid padding ($P = 0$), followed by a $2 \times 2$ Max Pooling layer with stride $S = 2$.
  1. What is the spatial output shape after Conv2D?
  2. How many trainable parameters are in the Conv2D layer?
  3. What is the final tensor shape after Max Pooling?
* **Solution:**
  1. **Conv2D Spatial Shape:**
     $$O_{\text{conv}} = \left\lfloor \frac{28 - 3 + 2(0)}{1} \right\rfloor + 1 = 26 \implies \mathbf{(26, 26, 16)}$$
  2. **Trainable Parameters:**
     Each filter has $3 \times 3 \times 1 = 9$ weights plus $1$ bias $= 10$ parameters.
     For 16 filters: $16 \times 10 = \mathbf{160}$ trainable parameters.
  3. **Max Pooling Shape:**
     $$O_{\text{pool}} = \left\lfloor \frac{26 - 2}{2} \right\rfloor + 1 = 13 \implies \mathbf{(13, 13, 16)}$$

#### Q3.2: ResNet Residual Connection Derivation
* **Problem:** Explain why the residual formulation $H(x) = F(x) + x$ prevents the vanishing gradient problem in deep architectures.
* **Solution:**
  - In a standard network, gradients must backpropagate through multiplicative layer weight matrices: $\frac{\partial J}{\partial x} = \frac{\partial J}{\partial H} \cdot \prod W_l$. If weights are small, gradients vanish exponentially.
  - In ResNet: $H(x) = F(x) + x$. Differentiating with respect to input $x$:
    $$\frac{\partial H(x)}{\partial x} = \frac{\partial F(x)}{\partial x} + \mathbf{1}$$
  - Even if the learned residual gradient $\frac{\partial F(x)}{\partial x} \approx 0$, the additive $+1$ term guarantees that gradients flow cleanly back to earlier layers without diminishing.

#### Q3.3: Fuzzy Logic Defuzzification Calculation
* **Problem:** Given an aggregated discrete fuzzy output set $C$ with points $z = \{2, 4, 6\}$ and membership values $\mu_C(2) = 0.3$, $\mu_C(4) = 0.8$, and $\mu_C(6) = 0.5$, calculate the crisp output $z^*$ using the Centroid method.
* **Solution:**
  $$z^* = \frac{\sum z_i \cdot \mu_C(z_i)}{\sum \mu_C(z_i)} = \frac{(2 \times 0.3) + (4 \times 0.8) + (6 \times 0.5)}{0.3 + 0.8 + 0.5}$$
  $$z^* = \frac{0.6 + 3.2 + 3.0}{1.6} = \frac{6.8}{1.6} = \mathbf{4.25}$$

---

<div align="center">

<sub>Curated with precision by <a href="https://github.com/24pwai0015-max">Muhammad Arsalan</a> · BS Artificial Intelligence · UET Peshawar</sub>

</div>
