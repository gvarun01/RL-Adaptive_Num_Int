## 📊 Results Summary

The RL-based integration framework was benchmarked across a variety of function types, comparing performance against classical methods including **Monte Carlo**, **Gauss-Legendre Quadrature**,**Adaptive Sympson and Trapezoidal** and **QUADPACK**.

### 🔎 Function Categories Tested

- **Exponential**: Smooth Gaussian-like curves
- **Polynomial/Rational**: Nonlinear low-frequency functions
- **Oscillatory**: High-frequency peaks (e.g., sinusoids)
- **Trigonometric**: Periodic sawtooth or sine waves
- **Piecewise**: Checkerboard-style discontinuities
- **Discontinuous**: Step functions, sharp gradients
- **Other**: Custom or mixed-behavior functions

---

### 🧠 RL Model vs Traditional Methods

#### 🔻 Error, Bias & Time Comparison (by function class)

![Detailed Comparison](assets/detailed_function_examples.png)

> The RL agent achieves **lower relative error and bias** across all function categories, often with **reduced execution time**.

---

#### 📉 Average Relative Error by Function Class

![Error by Class](assets/error_comparison_by_class.png)

> The RL model consistently outperforms both Monte Carlo and Gauss-Legendre in terms of **log-scale error**, particularly for irregular, oscillatory, and discontinuous functions.

---

#### ⚡ Efficiency Breakdown

![Efficiency](assets/efficiency.jpeg)

> Efficiency normalized per category shows the RL model balancing **accuracy and cost**, outperforming in complex domains while maintaining strong generalization.

---

#### 🔁 Convergence Trends (Error vs. Function Evaluations)

![Convergence](assets/error_convergence.jpeg)

> The RL agent achieves **faster convergence** to lower errors with fewer evaluations, confirming its ability to dynamically prioritize difficult regions.

---

#### ⏱️ Error vs. Time (Log-Log Scale)

![Error vs Time](assets/error_vs_time_ind.png)

> When plotted against execution time, the RL model yields a **favorable tradeoff**, with lower error in less time compared to Monte Carlo and Gauss methods.

---

#### 🔄 Function Evaluations Across Methods

![Function Evaluations](assets/func_evaluations.jpeg)

> The RL model demonstrates **significant savings in function evaluations** while preserving or improving overall accuracy.

---

#### 🕸️ Overall Capability Comparison (Radar Chart)

![Model Capabilities](assets/stats.jpeg)

> A radar plot comparison highlights the RL model’s strength in **accuracy, adaptivity, dimensional scalability, and discontinuity handling**, making it a robust solution for complex numerical integration tasks.

---
