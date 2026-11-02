# Deep Learning Practical Labs

This directory contains the executable Jupyter notebooks, official lab task sheets, student handouts, and delivered academic reports for the course **Artificial Neural Networks & Deep Learning** (5th Semester, BS AI, UET Peshawar).

---

## Lab Directory Overview

| Lab | Topic | Executable Notebook | Open in Colab | Handout / Data | Delivered Report |
| :---: | :--- | :---: | :---: | :---: | :---: |
| **Lab 01** | Application of Mask R-CNN for Car Damage Detection & Instance Segmentation | [`Lab1_Car_Damage_Detection_Mask_RCNN.ipynb`](Lab1_Car_Damage_Detection_Mask_RCNN.ipynb) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/24pwai0015-max/ann-and-deep-learning/blob/main/labs/Lab1_Car_Damage_Detection_Mask_RCNN.ipynb) | [`dataset/`](dataset/) | [`Lab1_Car_Damage_Detection_Mask_RCNN.pdf`](delivered-reports/Lab1_Car_Damage_Detection_Mask_RCNN.pdf) |
| **Lab 02** | Binary Classification of Water Potability via ANN & MLP Architectures | [`Lab2_Water_Potability_Classification.ipynb`](Lab2_Water_Potability_Classification.ipynb) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/24pwai0015-max/ann-and-deep-learning/blob/main/labs/Lab2_Water_Potability_Classification.ipynb) | [`water_potability.csv`](water_potability.csv) | [`Lab2_Water_Potability_Classification.pdf`](delivered-reports/Lab2_Water_Potability_Classification.pdf) |
| **Lab 03** | Adaptive Optimizers for ANN & MLP (MNIST Benchmark) | [`Lab3_Adaptive_Optimizers_ANN_MLP.ipynb`](Lab3_Adaptive_Optimizers_ANN_MLP.ipynb) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/24pwai0015-max/ann-and-deep-learning/blob/main/labs/Lab3_Adaptive_Optimizers_ANN_MLP.ipynb) | [`Lab_Task_03.pdf`](handouts/Lab_Task_03.pdf) | [`Lab3_Adaptive_Optimizers_ANN_MLP.pdf`](delivered-reports/Lab3_Adaptive_Optimizers_ANN_MLP.pdf) |
| **Lab 04** | Adaptive Optimizers for CNN (Fashion-MNIST Benchmark) | [`Lab4_Adaptive_Optimizers_CNN.ipynb`](Lab4_Adaptive_Optimizers_CNN.ipynb) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/24pwai0015-max/ann-and-deep-learning/blob/main/labs/Lab4_Adaptive_Optimizers_CNN.ipynb) | [`Lab_Task_04.pdf`](handouts/Lab_Task_04.pdf) | [`Lab4_Adaptive_Optimizers_CNN.pdf`](delivered-reports/Lab4_Adaptive_Optimizers_CNN.pdf) |
| **Lab 05** | Batch vs. Stochastic vs. Mini-Batch Gradient Descent Dynamics | [`Lab5_Batch_vs_Stochastic_vs_MiniBatch.ipynb`](Lab5_Batch_vs_Stochastic_vs_MiniBatch.ipynb) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/24pwai0015-max/ann-and-deep-learning/blob/main/labs/Lab5_Batch_vs_Stochastic_vs_MiniBatch.ipynb) | [`Lab_Task_05.pdf`](handouts/Lab_Task_05.pdf) | [`Lab5_Batch_vs_Stochastic_vs_MiniBatch.pdf`](delivered-reports/Lab5_Batch_vs_Stochastic_vs_MiniBatch.pdf) |

---

## Folder Organization

```
labs/
├── Lab1_Car_Damage_Detection_Mask_RCNN.ipynb   ← Lab 1: Mask R-CNN Instance Segmentation
├── Lab2_Water_Potability_Classification.ipynb   ← Lab 2: Binary Water Potability MLP
├── Lab3_Adaptive_Optimizers_ANN_MLP.ipynb       ← Lab 3: Adaptive Optimizers in ANN/MLP
├── Lab4_Adaptive_Optimizers_CNN.ipynb           ← Lab 4: Adaptive Optimizers in CNN
├── Lab5_Batch_vs_Stochastic_vs_MiniBatch.ipynb  ← Lab 5: Batch vs Mini-batch vs SGD
├── custom.py                                    ← Custom Mask R-CNN Dataset & Config helpers
├── mrcnn/                                       ← Mask R-CNN core package architecture
├── dataset/                                     ← Train/val annotations & vehicle sample images
├── water_potability.csv                         ← 3,276 water quality parameter dataset
├── README.md                                    ← Directory index with Colab badges
├── handouts/
│   ├── Lab_Task_03.pdf
│   ├── Lab_Task_04.pdf
│   ├── Lab_Task_05.pdf
│   └── Student_Lab_Handout_03_04_05.pdf
└── delivered-reports/
    ├── Lab1_Car_Damage_Detection_Mask_RCNN.pdf  ← Evaluated submission with Colab link
    ├── Lab2_Water_Potability_Classification.pdf  ← Evaluated submission with Colab link
    ├── Lab3_Adaptive_Optimizers_ANN_MLP.pdf     ← Evaluated submission with Colab link
    ├── Lab4_Adaptive_Optimizers_CNN.pdf         ← Evaluated submission with Colab link
    └── Lab5_Batch_vs_Stochastic_vs_MiniBatch.pdf← Evaluated submission with Colab link
```
