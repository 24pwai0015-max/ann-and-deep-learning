# Deep Learning Quick Reference & Formula Cheat-Sheet (Midterms)
### Author: Muhammad Arsalan | Course: ANN & Deep Learning

---

## 1. Linear Algebra & Dimensions

| Concept | Mathematical Formula / Rule | Notes |
|:---|:---|:---|
| **Dot Product** | $C_{m \times n} = A_{m \times k} \times B_{k \times n}$ | Inner dimension $k$ must match |
| **Neuron Output** | $z = W^T X + b, \quad \hat{y} = f(z)$ | $W \in \mathbb{R}^n$, $b \in \mathbb{R}$ |
| **Iterations / Epoch** | $\text{Iterations} = \lceil N / B \rceil$ | $N = \text{Dataset size}$, $B = \text{Batch size}$ |
| **Flatten Size** | $\text{Features} = H \times W \times C$ | e.g. $28 \times 28 \times 1 = 784$ |

---

## 2. Activation Functions Summary

| Activation | Function $f(z)$ | Derivative $f'(z)$ | Range |
|:---|:---|:---|:---:|
| **Sigmoid** | $\frac{1}{1 + e^{-z}}$ | $f(z)(1 - f(z))$ | $(0, 1)$ |
| **Tanh** | $\frac{e^z - e^{-z}}{e^z + e^{-z}}$ | $1 - f(z)^2$ | $(-1, 1)$ |
| **ReLU** | $\max(0, z)$ | $1$ if $z > 0$ else $0$ | $[0, \infty)$ |
| **Leaky ReLU** | $\max(\alpha z, z)$ | $1$ if $z > 0$ else $\alpha$ | $(-\infty, \infty)$ |
| **Softmax** | $\frac{e^{z_i}}{\sum_{j=1}^K e^{z_j}}$ | $\sigma_i(\delta_{ij} - \sigma_j)$ | $(0, 1), \; \sum p_i = 1$ |

---

## 3. Classical Learning Rules Matrix

| Rule | Formula | Type | Key Characteristic |
|:---|:---|:---:|:---|
| **Hebbian** | $\Delta w_i = \eta x_i y$ | Unsupervised | Strengthens co-activated connections |
| **Perceptron** | $\Delta w_i = \eta (y - \hat{y}) x_i$ | Supervised | Discrete step error correction |
| **Delta (LMS)** | $\Delta w_i = \eta (y - W^T X) x_i$ | Supervised | Continuous linear error minimization |
| **Correlation** | $\Delta w_i = \eta x_i y_{\text{target}}$ | Supervised | Hebbian correlation with target |
| **Outstar** | $\Delta w_i = \eta (y_i - w_i) x$ | Supervised | Pattern generator weight adjustment |

---

## 4. Optimizer Formulas & Updates

| Optimizer | Accumulator / Moment | Update Formula |
|:---|:---|:---|
| **SGD** | None | $w_{t+1} = w_t - \eta g_t$ |
| **Adagrad** | $G_t = G_{t-1} + g_t^2$ | $w_{t+1} = w_t - \frac{\eta}{\sqrt{G_t} + \epsilon} g_t$ |
| **RMSprop** | $v_t = \rho v_{t-1} + (1-\rho) g_t^2$ | $w_{t+1} = w_t - \frac{\eta}{\sqrt{v_t} + \epsilon} g_t$ |
| **Adam** | $m_t = \beta_1 m_{t-1} + (1-\beta_1) g_t$ <br> $v_t = \beta_2 v_{t-1} + (1-\beta_2) g_t^2$ | $w_{t+1} = w_t - \frac{\eta}{\sqrt{\hat{v}_t} + \epsilon} \hat{m}_t$ <br> where $\hat{m}_t = \frac{m_t}{1-\beta_1^t}, \; \hat{v}_t = \frac{v_t}{1-\beta_2^t}$ |

---

## 5. CNN & Vision Formulas

- **Conv2D Spatial Output:**
  $$O = \left\lfloor \frac{I - K + 2P}{S} \right\rfloor + 1$$
- **Conv2D Parameter Count:**
  $$\text{Parameters} = (\text{Kernel}_H \times \text{Kernel}_W \times C_{\text{in}} + 1) \times C_{\text{out}}$$
- **ResNet Residual Block:**
  $$H(x) = F(x) + x \implies \frac{\partial H}{\partial x} = \frac{\partial F}{\partial x} + 1$$

---

## 6. Fuzzy Logic Centroid Defuzzification
$$z^* = \frac{\sum z_i \cdot \mu_C(z_i)}{\sum \mu_C(z_i)}$$
