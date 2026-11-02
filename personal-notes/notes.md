# Personal Study Notes & Worked Examples (Midterm Archive)
### Course: Artificial Neural Networks & Deep Learning | 5th Semester BS AI
### Author: Muhammad Arsalan

---

## 1. Deep Learning Foundations & Linear Algebra
- **Hierarchy of AI:**
  $$\text{AI} \supset \text{Machine Learning} \supset \text{Neural Networks} \supset \text{Deep Learning}$$
  - *Deep Neural Network (DNN):* Requires $\ge 2$ hidden layers. Extra depth allows progressive hierarchical abstraction.
- **The 4 Key Axes of Distinction:**
  1. *Data Volume:* Traditional ML plateaus; DL scales with large data.
  2. *Hardware:* ML runs on CPUs; DL demands parallel GPUs/TPUs.
  3. *Feature Engineering:* Manual in ML; automated end-to-end representation learning in DL.
  4. *Execution Time:* ML trains fast with fast inference; DL trains slowly with fast real-time inference.
- **Tensor Operations & Dimensionality:**
  - Scalar: 0D tensor ($x = 5$).
  - Vector: 1D tensor ($\mathbf{x} = [x_1, x_2, \dots]$).
  - Matrix: 2D tensor ($A \in \mathbb{R}^{m \times n}$).
  - Tensor: 3D+ tensor (e.g., $(B, H, W, C)$).
  - *Matrix Dot Product:*
    $$A_{m \times k} \times B_{k \times n} = C_{m \times n}$$
    *Inner dimensions must match!*
- **Artificial Neuron Mechanics:**
  $$z = W \cdot X + b = \sum_{i=1}^n w_i x_i + b, \quad \hat{y} = f(z)$$
  - Weights control direction and slope of the hyperplane.
  - Bias shifts the hyperplane away from the origin.

---

## 2. McCulloch-Pitts Model & Activation Functions
- **McCulloch-Pitts Firing Condition:**
  $$y = 1 \iff \left( \sum_{i \in \text{excitatory}} x_i \ge \theta \right) \land \left( \forall j \in \text{inhibitory}, \; x_j = 0 \right)$$
- **Why Non-Linearity is Essential (Linear Stacking Proof):**
  If activation is linear $f(z) = c z$:
  $$\hat{y} = W_2 (W_1 X + b_1) + b_2 = (W_2 W_1) X + (W_2 b_1 + b_2) = W_{\text{eff}} X + b_{\text{eff}}$$
  Stacking 100 linear layers collapses to a single affine transform. Non-linearity bends decision boundaries to separate non-linear data (like XOR).
- **Core Activations:**
  - *Sigmoid:* $\sigma(z) = \frac{1}{1 + e^{-z}} \in (0, 1)$, $\sigma'(z) = \sigma(z)(1 - \sigma(z))$. Vanishing gradient for $|z| > 4$.
  - *Tanh:* $\tanh(z) = \frac{e^z - e^{-z}}{e^z + e^{-z}} \in (-1, 1)$, $\tanh'(z) = 1 - \tanh^2(z)$. Zero-centered.
  - *ReLU:* $f(z) = \max(0, z)$. Fast, non-saturating for $z > 0$. Prone to "dying ReLU" if weights update into permanent negative territory.
  - *Leaky ReLU:* $f(z) = \max(\alpha z, z)$, with $\alpha \approx 0.01$. Prevents neuron death.
  - *Softmax:* $\sigma(z)_i = \frac{e^{z_i}}{\sum_j e^{z_j}}$. Converts logits into valid probability distribution ($\sum p_i = 1$).

---

## 3. Backpropagation & Optimization Dynamics
- **Chain Rule Backpropagation:**
  $$\frac{\partial J}{\partial w_1} = \frac{\partial J}{\partial \hat{y}} \cdot \frac{\partial \hat{y}}{\partial z_2} \cdot \frac{\partial z_2}{\partial a_1} \cdot \frac{\partial a_1}{\partial z_1} \cdot \frac{\partial z_1}{\partial w_1}$$
- **Batching Trade-offs:**
  - *Full Batch ($B=N$):* 1 update/epoch, exact gradient, high memory, slow per epoch.
  - *Stochastic ($B=1$):* $N$ updates/epoch, noisy path, escapes local minima, cannot vectorize.
  - *Mini-Batch ($1 < B < N$):* Balances GPU vectorization with regularizing noise.
- **Adaptive Optimizers Compared:**
  - *SGD:* Fixed rate $\eta$.
  - *Adagrad:* Accumulates $G_t = G_{t-1} + g_t^2$. Steps shrink monotonically to zero.
  - *RMSprop:* Exponential moving average $v_t = \rho v_{t-1} + (1-\rho) g_t^2$. Prevents premature stopping.
  - *Adadelta:* Tracks running averages of updates, eliminating global learning rate.
  - *Adam:* Combines 1st moment (momentum) and 2nd moment (RMSprop variance) with bias correction terms $\hat{m}_t$ and $\hat{v}_t$.

---

## 4. Convolutional Neural Networks & Intelligent Systems
- **Convolution Output Size:**
  $$O = \left\lfloor \frac{I - K + 2P}{S} \right\rfloor + 1$$
- **ResNet Skip Connection:**
  $$H(x) = F(x) + x \implies \frac{\partial H}{\partial x} = \frac{\partial F}{\partial x} + 1$$
  The identity $+1$ term ensures uninterrupted gradient backpropagation.
- **Fuzzy Logic Centroid Method:**
  $$z^* = \frac{\sum z_i \cdot \mu_C(z_i)}{\sum \mu_C(z_i)}$$
