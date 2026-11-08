# 8-3-8 Neural Network with Backpropagation (from scratch)

Assignment 1 for **Advanced Concepts in Machine Learning (ACML)**,
Maastricht University.

**Authors:** Jessica Tambo Epoh, Vlad Stefan

---

## About

This project implements a small neural network **from scratch in NumPy**,
without deep learning libraries such as PyTorch, Keras or TensorFlow.
The forward pass, backpropagation and gradient descent updates are written by hand.

The network has three layers:

```
8 inputs  →  3 hidden neurons (+ bias)  →  8 outputs (+ bias)
```

It is trained on the 8 one-hot vectors (seven 0s and one 1), and the target is
the input itself. Because the hidden layer has only 3 neurons, the network has
to compress 8 different inputs into 3 numbers, and it learns its own
**binary-like code** to do so.

---

## Contents

| File | Description |
|---|---|
| `assignment1_Group17.ipynb` | Jupyter notebook with the code, experiments and report |
| `assignment1_Group17_report.pdf` | PDF version of the notebook |
| `assignment1_Group17_video.mp4` | Video (≈ 5 min) walking through the code and results |

---

## How to run

**Requirements:** Python 3, NumPy, Matplotlib

```bash
pip install numpy matplotlib jupyter
jupyter notebook assignment1_Group17.ipynb
```

Then choose **Kernel → Restart & Run All**.
The two experiment cells train 90 and 70 networks, so each takes about 1–2 minutes.

The training function can also be used on its own:

```python
converged_at, history, A1 = train_network(
    lr=2.0, seed=0, init_scale=0.5, max_epochs=20000, tol=0.01)
```

| Parameter | Meaning |
|---|---|
| `lr` | learning rate |
| `seed` | fixes the random starting weights (reproducibility) |
| `init_scale` | standard deviation of the random starting weights |
| `max_epochs` | maximum number of training epochs |
| `tol` | cost threshold used in the convergence check |

---

## Implementation

- **Activation:** sigmoid
- **Cost:** least squares, J = 1/(2m) · Σ ‖h(x) − y‖²
- **Training:** full-batch gradient descent with backpropagation
- **Biases:** stored as separate vectors
- **Initialisation:** small random weights (normal distribution), biases at 0

---

## Main results

**Learning rate** (10 random starts each):

| Learning rate | Converged | Avg. epochs |
|---|---|---|
| 0.1 | 0/10 | – |
| 0.5 | 10/10 | 13,319 |
| 2 | 10/10 | 3,330 |
| 20 | 10/10 | 376 |
| 50 | 10/10 | ~4,100 (overshooting) |
| 100 | 0/10 | – |

- Training is **reliable**: every learning rate between 0.5 and 50 converged for all 10 seeds.
- Starting weights must be **small and random**: large weights saturate the sigmoid,
  and all-zero weights create a symmetry problem between the hidden neurons.

**What the network learned:**
- The 3 hidden neurons act almost like on/off switches and use **all 8 possible
  3-bit patterns**, one per input.
- **W1 works as a code book** (the signs of each row give that input's code), and
  **W2 works as a decoder** (each output's weights match the code of its own input).

See the notebook or the PDF for the full analysis.

---

## Notes

See the "LLM Usage and Extra Reading" section in the notebook for the tools and
sources used. Besides the lecture slides, we used the "Backprop Help" document
provided by the course.