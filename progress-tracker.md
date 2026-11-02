# Mids Academic Progress Tracker
### Course: Artificial Neural Networks & Deep Learning (Semester 5 | Fall 2026)
### Department: BS Artificial Intelligence, UET Peshawar
### Student: Muhammad Arsalan (Registration: 24PWAI0015)

---

## Midterm Academic Curriculum Mastery (Lectures 01 – 05)

| Lecture / Module | Theoretical Scope & Core Concepts | Academic Status | Mastery Level | Verification Artifact |
|:---:|:---|:---:|:---:|:---|
| **Lecture 01** | **Foundations & Linear Algebra:**<br>• AI $\supset$ ML $\supset$ ANN $\supset$ DNN ($\ge 2$ hidden layers)<br>• 4 Key ML vs. DL axes (Data volume, GPU/TPU hardware, Automated representations, Execution profile)<br>• Progressive feature hierarchy (edges $\to$ textures $\to$ object parts)<br>• Tensors (ranks 0–4), matrix dot products ($A_{m \times k} B_{k \times n} = C_{m \times n}$)<br>• Artificial neuron model ($z = W^T X + b$, $y = f(z)$)<br>• Biological mapping (Dendrite, Synapse, Soma, Axon)<br>• Iterations per epoch formula ($\lceil N / B \rceil$) | ✅ Completed | High (100%) | [`DEEP_LEARNING_MIDTERM_GUIDE.md`](DEEP_LEARNING_MIDTERM_GUIDE.md#lecture-1-deep-learning-foundations--linear-algebra) |
| **Lecture 02** | **End-to-End Data Science & Modeling Pipeline:**<br>• 9-Stage modeling workflow (Problem Definition $\to$ Deployment)<br>• Train / Validation / Test strict partitioning<br>• Classification metrics (Accuracy, Precision, Recall, F1, Confusion Matrix)<br>• Regression metrics (MAE, MSE, RMSE, $R^2$ score) | ✅ Completed | High (95%) | [`DEEP_LEARNING_MIDTERM_GUIDE.md`](DEEP_LEARNING_MIDTERM_GUIDE.md#lecture-2-end-to-end-data-science--modeling-pipeline) |
| **Lecture 03** | **Neuron Models & Non-Linearity:**<br>• McCulloch-Pitts neuron (excitatory $+1$, inhibitory veto)<br>• Realization of Boolean logic (AND, OR, NOT, NAND)<br>• Mathematical necessity of non-linear activations<br>• Proof: Multi-layer linear networks collapse to single affine map ($W_2(W_1 X + b_1) + b_2 = W' X + b'$) | ✅ Completed | High (100%) | [`personal-notes/notes.md`](personal-notes/notes.md#2-mcculloch-pitts-model--activation-functions) |
| **Lecture 04** | **Activation Functions & Classical Learning Rules:**<br>• Activation comparison (Sigmoid, Tanh, ReLU, Leaky ReLU, Softmax)<br>• Mathematical saturation & vanishing gradient analysis<br>• Dying ReLU diagnosis & leaky recovery<br>• Adaline vs. Madaline (LMS Widrow-Hoff vs. Minimal Disturbance MRI)<br>• Classical rules: Hebbian, Perceptron, Delta, Outstar | ✅ Completed | High (95%) | [`personal-notes/reference.md`](personal-notes/reference.md#2-activation-functions-summary) |
| **Lecture 05** | **Training Dynamics & Optimization:**<br>• Loss landscapes (convex vs. non-convex, saddle points)<br>• Backpropagation multivariate chain rule derivation<br>• Batching regimes: Full-batch ($B=N$), Stochastic ($B=1$), Mini-batch ($1<B<N$)<br>• Overfitting countermeasures: Dropout ($p=0.5$), Early stopping patience | ✅ Completed | High (100%) | [`labs/Lab5_Batch_vs_Stochastic_vs_MiniBatch.ipynb`](labs/Lab5_Batch_vs_Stochastic_vs_MiniBatch.ipynb) |

---

## Practical Labs Integration (Midterm Scope)

| Lab Module | Topic & Experimental Focus | Code (Jupyter / Colab) | Academic Report | Evaluation Status |
|:---:|:---|:---:|:---:|:---:|
| **Lab 01** | *Foundational ANN Implementation (Uploading)* | ⏳ Pending Upload | ⏳ Pending Upload | In Progress |
| **Lab 02** | *Multi-Layer Perceptron Forward/Backward (Uploading)* | ⏳ Pending Upload | ⏳ Pending Upload | In Progress |
| **Lab 03** | **Adaptive Optimizers for ANN & MLP (MNIST)**<br>Comparing SGD, Adam, RMSprop, Adadelta & Adagrad on 1-layer ANN vs 2-layer MLP | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/24pwai0015-max/ann-and-deep-learning/blob/main/labs/Lab3_Adaptive_Optimizers_ANN_MLP.ipynb) | [`Lab3_Adaptive_Optimizers_ANN_MLP.pdf`](labs/delivered-reports/Lab3_Adaptive_Optimizers_ANN_MLP.pdf) | ✅ Evaluated & Verified |
| **Lab 04** | **Adaptive Optimizers for CNN (Fashion-MNIST)**<br>Evaluating 5 optimizers on Conv2D layers with identical initial weights | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/24pwai0015-max/ann-and-deep-learning/blob/main/labs/Lab4_Adaptive_Optimizers_CNN.ipynb) | [`Lab4_Adaptive_Optimizers_CNN.pdf`](labs/delivered-reports/Lab4_Adaptive_Optimizers_CNN.pdf) | ✅ Evaluated & Verified |
| **Lab 05** | **Batch vs. Stochastic vs. Mini-Batch Gradient Descent**<br>Empirical comparison across ANN, MLP, and CNN architectures | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/24pwai0015-max/ann-and-deep-learning/blob/main/labs/Lab5_Batch_vs_Stochastic_vs_MiniBatch.ipynb) | [`Lab5_Batch_vs_Stochastic_vs_MiniBatch.pdf`](labs/delivered-reports/Lab5_Batch_vs_Stochastic_vs_MiniBatch.pdf) | ✅ Evaluated & Verified |
| **Case Study** | **Mask R-CNN for Instance Segmentation**<br>Car damage detection and segmentation study | — | [`Lab_Mask_RCNN_Car_Damage_Detection.pdf`](labs/delivered-reports/Lab_Mask_RCNN_Car_Damage_Detection.pdf) | ✅ Delivered |

---

## Academic Verification Summary

- **Midterm Lectures Completed:** 5 of 5 (100% of Midterm Syllabus)
- **Practical Labs Verified:** Labs 3, 4, 5 (Labs 1 & 2 arriving)
- **Mock Exam Preparedness:** Full solved benchmark ready in [`DEEP_LEARNING_MIDTERM_GUIDE.md`](DEEP_LEARNING_MIDTERM_GUIDE.md#3-midterm-examination-benchmark-complete-solved-paper)
