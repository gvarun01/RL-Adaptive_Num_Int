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

The project is organized as follows:

- **`.gitignore`**: Specifies intentionally untracked files that Git should ignore.
- **`LICENSE`**: Contains the license information for the project.
- **`Presentation.pdf`**: A PDF presentation document, likely summarizing the project.
- **`README.md`**: This file, providing an overview of the project.
- **`REPORT.pdf`**: A PDF report document, likely detailing the project's findings.
- **`results/`**: Contains the outputs and plots from the integration experiments.
    - **`results/README.md`**: A detailed summary of the benchmark results and comparisons.
    - **`results/1D/`**: Stores images and data related to 1D integration benchmarks.
    - **`results/2D/`**: Stores images and data related to 2D integration benchmarks.
---

## 🛠️ Installation

```bash
# Clone the repo
git clone https://github.com/username/repository-name.git
cd adaptive-integration

# Create virtual environment
python -m venv venv
source venv/bin/activate  # or .\venv\Scripts\activate on Windows

# Install dependencies
pip install -r requirements.txt
```

---

## 🚀 Usage

To run the project, navigate to the root directory of the cloned repository and execute the main script:
   ```bash
   python -m code.main
   ```
   This command will start the process, which typically involves training the reinforcement learning model using the 2D test functions defined in the project and then evaluating its performance. The training progress and evaluation results will be displayed in the console. Intermediate models and normalization statistics are saved in the `adaptive_integration_2d_models` directory, and the final model is saved as `adaptive_integration_2d_models/adaptive_integration_final.zip`.

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
 
Currently the model is only implemented to do the Integration up to 2D Functions but it could be easily extended to N-Dimensional Integration.

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
- Detailed results, including plots and summaries, can be found in the `results/` directory. See `results/README.md` for a comprehensive overview.

---

## 🤝 Contributing

Contributions are welcome! If you'd like to contribute to this project, please follow these general guidelines:

1.  **Fork the repository** to your own GitHub account.
2.  **Create a new branch** for your feature or bug fix: `git checkout -b feature/your-feature-name` or `bugfix/issue-description`.
3.  **Make your changes** and commit them with clear, descriptive messages.
4.  **Ensure your code lints and tests pass** (if applicable). Given the current state of the repository, this might involve adding tests if you are contributing code.
5.  **Push your branch** to your fork: `git push origin feature/your-feature-name`.
6.  **Submit a pull request** to the main repository, detailing the changes you've made.

We appreciate your help in improving this project!

---

## 📜 License

This project is licensed under the terms of the `LICENSE` file. Please see that file for more details.
