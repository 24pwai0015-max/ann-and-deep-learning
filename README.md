<div align="center">

<picture>
  <source srcset="assets/hero.svg" media="(prefers-color-scheme: dark)">
  <source srcset="assets/hero.svg" media="(prefers-color-scheme: light)">
  <img src="assets/hero.svg" alt="Arsalan – ANN and Deep Learning – animated neural network with signal pulses" width="100%">
</picture>

<br/>

[![Python](https://img.shields.io/badge/Python-3.x-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![NumPy](https://img.shields.io/badge/NumPy-1.24+-013243?logo=numpy&logoColor=white)](https://numpy.org/)
[![Matplotlib](https://img.shields.io/badge/Matplotlib-3.7+-11557C)](https://matplotlib.org/)
[![Seaborn](https://img.shields.io/badge/Seaborn-0.12+-444876)](https://seaborn.pydata.org/)
&nbsp;&nbsp;`Semester 5 · ANN & Deep Learning`

</div>

---

## Mission

This is my semester-long lab notebook for Artificial Neural Networks & Deep Learning.
The goal is to understand every mathematical detail — from the perceptron to backpropagation — before touching any framework.
Nothing here is copied. If it is in this repo, I worked through it myself.

---

<div align="center">

<picture>
  <source srcset="assets/roadmap.svg" media="(prefers-color-scheme: dark)">
  <source srcset="assets/roadmap.svg" media="(prefers-color-scheme: light)">
  <img src="assets/roadmap.svg" alt="Course roadmap – 4 weeks, Week 1 pulsing as next" width="100%">
</picture>

</div>

| Week | Topic | Status |
|:----:|:------|:------:|
| **1** | Perceptron, linear separability & activation functions | 🔴 up next |
| **2** | Multi-Layer Perceptron (MLP) & forward propagation | ○ planned |
| **3** | Loss functions & gradient descent mathematics | ○ planned |
| **4** | The backpropagation algorithm from scratch | ○ planned |

> **Honesty note:** All four modules are marked "Not Learned" in my progress tracker.
> This repo is scaffolded and ready — the learning starts now.

---

## What's Already Here

| Folder | Contents |
|:-------|:---------|
| [`personal-notes/`](personal-notes/) | Intro to ANN — a handwritten/typed PDF from my first reading |
| [`connections-to-ai-robotics/`](connections-to-ai-robotics/) | Two bridge docs: how deep learning maps to robotics perception and agent policies |
| [`lecture-notes/`](lecture-notes/) | Placeholder — will hold derivations and proofs |
| [`assignments/`](assignments/) | Placeholder — semester problem sets |
| [`quizzes/`](quizzes/) | Placeholder — retrieval practice tests |
| [`projects/`](projects/) | Placeholder — from-scratch model implementations |
| [`presentations/`](presentations/) | Placeholder — slide decks |
| [`progress-tracker/`](progress-tracker/) | Weekly status and confidence log |

---

## Planned Skills

What I will be implementing from scratch over the semester.

| Foundations | Architectures | Optimization | Tooling |
|:------------|:--------------|:-------------|:--------|
| Perceptron | Multi-layer perceptron | Gradient descent | NumPy |
| Activation functions | Forward propagation | Backpropagation | Matplotlib |
| Linear separability | Loss functions | Chain rule derivations | Seaborn |

---

## Repository Map

```
ann-and-deep-learning/
├── assets/
│   ├── hero.svg                          ← animated neural network banner
│   └── roadmap.svg                       ← course timeline
├── personal-notes/
│   ├── intro to ann.pdf                  ← first reading notes
│   └── README.md
├── connections-to-ai-robotics/
│   ├── deep-learning-robotics-perception.md
│   └── neural-networks-agent-policies.md
├── lecture-notes/
│   └── README.md
├── assignments/
│   └── README.md
├── quizzes/
│   └── README.md
├── projects/
│   └── README.md
├── presentations/
│   └── README.md
├── progress-tracker/
│   └── README.md
├── progress-tracker.md                   ← weekly status table
├── .gitignore
├── requirements.txt
└── README.md                             ← you are here
```

---

## Quick Start (Windows PowerShell)

```powershell
# 1  Clone
git clone https://github.com/24pwai0015-max/ann-and-deep-learning.git
cd ann-and-deep-learning

# 2  Virtual environment
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# 3  Dependencies
pip install -r requirements.txt
```

---

## How I Learn

1. **Read** the math first — no code until the derivation makes sense on paper.
2. **Derive** every equation by hand in `personal-notes/`.
3. **Implement** from scratch using only NumPy — no frameworks allowed initially.
4. **Visualise** loss curves, decision boundaries, and gradient flow with Matplotlib.
5. **Document** what confused me and what clicked in `lecture-notes/`.

---

## Current Focus & Next Steps

- **Now:** reading perceptron theory and preparing Week 1 notes.
- **Next:** implement a single-layer perceptron from scratch in NumPy with a decision-boundary plot.
- **After that:** MLP forward pass, then loss functions, then the full backprop derivation.

---

## Bridge to AI & Robotics

This course connects directly to my longer-term interest in autonomous agents:

- [Neural Networks for Agent Policies](connections-to-ai-robotics/neural-networks-agent-policies.md) — DQN, Actor-Critic, latent embeddings.
- [Deep Learning in Robotics Perception](connections-to-ai-robotics/deep-learning-robotics-perception.md) — CNNs for vision, PointNet for LiDAR, edge inference.

---

<div align="center">

<sub>Made with patience by <a href="https://github.com/24pwai0015-max">Arsalan</a> · BS AI · UET Peshawar</sub>

</div>
