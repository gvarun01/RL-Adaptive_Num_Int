"""
This module provides sets of 1D and 2D test functions for evaluating
numerical integration algorithms.

The functions included cover a range of challenging scenarios, such as:
- Highly oscillatory behavior
- Functions with steep gradients or rapid changes
- Discontinuous functions
- Functions with singularities
- Combinations of these challenging features

These predefined sets are useful for benchmarking and testing the robustness
and adaptability of integration methods, particularly those developed for
reinforcement learning environments or adaptive quadrature.
"""
import numpy as np
import scipy.special as sp
from typing import Callable, Dict, Tuple


def define_test_functions() -> Dict[str, Tuple[Callable[[float], float], float, float]]:
    """
    Defines and returns a dictionary of 1D test functions for numerical integration.

    Each entry in the dictionary maps a function name (string) to a tuple.
    The tuple contains:
    1.  The callable function itself (Callable[[float], float]): Takes a float
        (x-coordinate) and returns a float.
    2.  The lower bound of the integration interval (float).
    3.  The upper bound of the integration interval (float).

    The functions cover various challenges:
    - Highly oscillatory functions (e.g., `sin_high_freq`, `cos_increasing_freq`).
    - Functions with rapid changes or steep gradients (e.g., `steep_sigmoid`, `runge`).
    - Discontinuous functions (e.g., `step`, `sawtooth`).
    - Functions with singularities at the edge or within the domain
      (e.g., `sqrt_singularity`, `log_singularity`, `inverse_singularity`).
      Note: For singularities, the interval might be slightly adjusted (e.g., 1e-6 offset)
      to avoid direct evaluation at the singular point during true value calculation,
      though the challenge for the integrator remains.
    - Combinations of these behaviors (e.g., `oscillating_with_peaks`).

    Returns:
        Dict[str, Tuple[Callable[[float], float], float, float]]:
            A dictionary of 1D test functions and their integration bounds.
    """
    functions: Dict[str, Tuple[Callable[[float], float], float, float]] = {}

    # 1. Highly oscillatory functions
    functions["sin_high_freq"] = (lambda x: np.sin(50 * x), 0.0, 2.0)
    functions["cos_increasing_freq"] = (lambda x: np.cos(x**2), 0.0, 10.0)
    functions["sin_exp"] = (lambda x: np.sin(np.exp(x)), 0.0, 3.0)

    # 2. Functions with rapid changes
    functions["steep_sigmoid"] = (lambda x: 1 / (1 + np.exp(-100 * (x - 0.5))), 0.0, 1.0)
    functions["runge"] = (lambda x: 1 / (1 + 25 * x**2), -1.0, 1.0)

    # 3. Discontinuous functions
    functions["step"] = (lambda x: 1.0 if x > 0.5 else 0.0, 0.0, 1.0)
    functions["sawtooth"] = (lambda x: x - np.floor(x), 0.0, 5.0)

    # 4. Functions with singularities
    # Note: Bounds are chosen to be near, but not at, the singularity for true value calculation.
    functions["sqrt_singularity"] = (lambda x: 1 / np.sqrt(x) if x > 0 else 0, 1e-6, 1.0)
    functions["log_singularity"] = (lambda x: np.log(x) if x > 0 else 0, 1e-6, 2.0)
    functions["inverse_singularity"] = (lambda x: 1 / (x - 0.5)**2 if abs(x - 0.5) > 1e-9 else 0, 0.0, 1.0)

    # 5. Combined challenging behaviors
    functions["oscillating_with_peaks"] = (lambda x: np.sin(10 * x) + 5 * np.exp(-100 * (x - 0.5)**2), 0.0, 1.0)
    functions["discontinuous_oscillatory"] = (lambda x: np.sin(20 * x) * (1 if x > 0.5 else 0.5), 0.0, 1.0)

    return functions


def define_2d_test_functions() -> Dict[str, Tuple[Callable[[float, float], float], float, float, float, float]]:
    """
    Defines and returns a dictionary of 2D test functions for numerical integration.

    Each entry in the dictionary maps a function name (string) to a tuple.
    The tuple contains:
    1.  The callable function itself (Callable[[float, float], float]): Takes two
        floats (x and y coordinates) and returns a float.
    2.  The lower x-bound of the integration domain (float).
    3.  The upper x-bound of the integration domain (float).
    4.  The lower y-bound of the integration domain (float).
    5.  The upper y-bound of the integration domain (float).

    The functions cover various categories:
    - Standard smooth functions (e.g., `gaussian_2d`, `sinc_2d`).
    - Highly oscillatory functions (e.g., `oscillatory_2d`, `bessel_2d`).
    - Functions with rapid changes or sharp peaks (e.g., `peaks_2d`, `gaussian_peaks`).
    - Discontinuous functions (e.g., `step_2d`, `checkerboard`).
    - Functions with singularities (e.g., `inverse_r`, `log_singularity_2d`).
    - Combinations of these challenging characteristics (e.g., `oscillating_peaks_2d`).

    Returns:
        Dict[str, Tuple[Callable[[float, float], float], float, float, float, float]]:
            A dictionary of 2D test functions and their integration bounds.
    """
    functions: Dict[str, Tuple[Callable[[float, float], float], float, float, float, float]] = {}

    # 1. Standard smooth functions
    functions["gaussian_2d"] = (
        lambda x, y: np.exp(-(x**2 + y**2)),
        -3.0, 3.0, -3.0, 3.0
    )
    functions["sinc_2d"] = (
        lambda x, y: np.sinc(x) * np.sinc(y), # np.sinc(x) = sin(pi*x)/(pi*x)
        -4.0, 4.0, -4.0, 4.0
    )
    functions["polynomial_2d"] = (
        lambda x, y: x**2 * y**3 - x*y + y**2,
        -2.0, 2.0, -2.0, 2.0
    )

    # 2. Highly oscillatory functions
    functions["oscillatory_2d"] = (
        lambda x, y: np.sin(50*x) * np.cos(50*y),
        0.0, 2.0, 0.0, 2.0
    )
    functions["bessel_2d"] = ( # Bessel function of first kind, order 0
        lambda x, y: sp.j0(np.sqrt(x**2 + y**2)),
        -10.0, 10.0, -10.0, 10.0
    )
    functions["frequency_modulated"] = (
        lambda x, y: np.sin(x * (1 + y**2)) * np.cos(y * (1 + x**2)),
        -2.0, 2.0, -2.0, 2.0
    )
    functions["wave_packet"] = (
        lambda x, y: np.exp(-(x**2 + y**2)) * np.sin(10*(x + y)),
        -3.0, 3.0, -3.0, 3.0
    )

    # 3. Functions with rapid changes
    # Franke's function (modified) - often used for 2D interpolation/approximation tests
    functions["peaks_2d"] = (
        lambda x, y: (0.75 * np.exp(-((9*x-2)**2 + (9*y-2)**2)/4) +
                      0.75 * np.exp(-((9*x+1)**2)/49 - (9*y+1)/10) +
                      0.5 * np.exp(-((9*x-7)**2 + (9*y-3)**2)/4) -
                      0.2 * np.exp(-((9*x-4)**2 + (9*y-7)**2))),
        0.0, 1.0, 0.0, 1.0 # Standard domain for Franke's
    )
    # A common "peaks" function from Matlab, different from Franke's
    # functions["peaks_matlab"] = (
    #     lambda x, y: 3*(1-x)**2 * np.exp(-x**2 - (y+1)**2) - \
    #                 10*(x/5 - x**3 - y**5) * np.exp(-x**2 - y**2) - \
    #                 1/3 * np.exp(-(x+1)**2 - y**2),
    #     -3.0, 3.0, -3.0, 3.0
    # )
    functions["gaussian_peaks"] = (
        lambda x, y: sum(np.exp(-((x-xi)**2 + (y-yi)**2)/0.05) # Sharper peaks
                        for xi, yi in [(-0.5,-0.5), (0.5,0.5), (-0.5,0.5), (0.5,-0.5)]),
        -1.0, 1.0, -1.0, 1.0
    )

    #  4. Discontinuous functions
    functions["step_2d"] = (
        lambda x, y: 1.0 if x > 0.5 and y > 0.5 else 0.0, # Centered step
        0.0, 1.0, 0.0, 1.0
    )
    functions["checkerboard"] = (
        lambda x, y: 1.0 if (int(3*x) + int(3*y)) % 2 == 0 else 0.0, # 3x3 checkerboard pattern
        0.0, 1.0, 0.0, 1.0
    )
    functions["circular_step"] = (
        lambda x, y: 1.0 if x**2 + y**2 < 0.5**2 else 0.0, # Circle of radius 0.5
        -1.0, 1.0, -1.0, 1.0
    )
    functions["sawtooth_2d"] = (
        lambda x, y: (2*(x - np.floor(x+0.5))) * (2*(y - np.floor(y+0.5))), # Triangle waves
        -1.0, 1.0, -1.0, 1.0
    )

    # 5. Functions with singularities
    # Singularity at (0,0)
    functions["inverse_r"] = (
        lambda x, y: 1.0 / (np.sqrt(x**2 + y**2) + 1e-9), # Epsilon to avoid /0 if evaluated at (0,0)
        -1.0, 1.0, -1.0, 1.0
    )
    # Singularity at (0,0)
    functions["log_singularity_2d"] = (
        lambda x, y: np.log(x**2 + y**2 + 1e-9),
        -1.0, 1.0, -1.0, 1.0
    )
    # Ring-like singularity
    functions["pole_singularity"] = (
        lambda x, y: 1.0 / (abs(x**2 + y**2 - 0.5**2) + 0.01), # Peak around radius 0.5
        -1.0, 1.0, -1.0, 1.0
    )

    # 6. Combined challenging behaviors
    functions["oscillating_peaks_2d"] = (
        lambda x, y: np.sin(20*x) * np.cos(20*y) * np.exp(-2*((x-0.5)**2 + (y-0.5)**2)),
        0.0, 1.0, 0.0, 1.0
    )
    functions["mixed_features"] = ( # Oscillatory with a sharp peak
        lambda x, y: (np.sin(10*x*y) / (0.1 + x**2 + y**2) +
                     2*np.exp(-50*((x-0.25)**2 + (y-0.75)**2))),
        -1.0, 1.0, -1.0, 1.0
    )
    functions["complex_oscillatory"] = ( # Oscillations on a varying background
        lambda x, y: np.sin(15*x) * np.cos(15*y) + np.exp(-0.5*(x**2 + y**2)) * np.cos(5*x*y),
        -1.5, 1.5, -1.5, 1.5
    )
    functions["hybrid_singularity_oscillation"] = ( # Oscillations around a singularity
        lambda x, y: np.cos(30 / (x**2 + y**2 + 0.1)) / (x**2 + y**2 + 0.1),
        -1.0, 1.0, -1.0, 1.0
    )

    return functions
