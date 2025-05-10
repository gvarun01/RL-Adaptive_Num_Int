# Reinforcement Learning for Adaptive Numerical Integration

This project presents a novel framework for adaptive numerical integration using **Reinforcement Learning (RL)**. Designed to efficiently approximate definite integrals in both **1D and 2D** domains, the system leverages a **PPO (Proximal Policy Optimization)** agent to intelligently guide interval refinement based on function complexity, oscillation, curvature, and error estimates.

---

## 🚀 Overview

Traditional adaptive integration techniques use fixed heuristics to determine how and where to split integration domains. These approaches struggle with functions exhibiting discontinuities, singularities, or high-frequency oscillations. Our approach reframes the integration task as a **Markov Decision Process (MDP)**, where an RL agent learns to allocate computational effort adaptively.

### Highlights

- ✅ PPO-trained agent using [Stable-Baselines3](https://github.com/DLR-RM/stable-baselines3)
- ✅ Supports both 1D and 2D integration domains
- ✅ Multiple numerical methods: Gauss-Legendre, Adaptive Simpson, Monte Carlo, PFEM
- ✅ Richardson extrapolation and statistical variance for error estimation
- ✅ Gym-based custom environments: `EnhancedAdaptiveIntegrationEnv`

---

## 📂 Project Structure


---

## 🔬 Methodology

### 1D Integration

- Environment state captures interval position, curvature, local error, and multiple quadrature estimates.
- Agent actions: interval selection, split ratio, and strategy (e.g., midpoint, error-based).
- Reward: error reduction per function evaluation, with terminal accuracy bonus.

### 2D Integration

- State includes geometric bounds, region-wise statistics (oscillation, skew, kurtosis), and error estimates.
- Actions: region selection, axis split, split ratio, and refinement method.
- Uses multiple integration techniques selected adaptively by the agent:
  - **Gauss-Legendre Quadrature (5-point)**
  - **Adaptive Simpson’s Rule (tensor-product)**
  - **Monte Carlo Sampling**
  - **Particle Finite Element Method (PFEM)**
 
Currently the model is only implemented to do the Integration upto 2D Functions but it could be easily extended to N-Dimensional Integration.

---

## 📊 Evaluation

The framework has been tested on a range of function classes:

- Smooth (Gaussian peaks)
- Highly Oscillatory (sinusoids, checkerboards)
- Singularities (1 / √(x² + y²))
- Discontinuous (step functions)

### Results

- Significantly lower integration error for a fixed budget of function evaluations compared to classical heuristics.
- Adaptive refinement concentrates effort in regions with high local error or structural complexity.
- The results could be seen in assets folder.

---

## 🛠️ Installation

```bash
# Clone the repo
git clone https://github.com/YOUR_USERNAME/adaptive-integration.git
cd adaptive-integration

# Create virtual environment
python -m venv venv
source venv/bin/activate  # or .\venv\Scripts\activate on Windows

# Install dependencies
pip install -r requirements.txt
```
