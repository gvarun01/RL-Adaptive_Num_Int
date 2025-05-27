"""
This module defines the custom Gymnasium environment for adaptive numerical
integration using reinforcement learning.

The `EnhancedAdaptiveIntegrationEnv` class provides an RL environment where an
agent learns to make decisions about how to partition an integration domain
to accurately estimate the integral of a given function with minimal
computational effort.
"""
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Callable, Dict, List, Tuple, Optional, Union
from scipy.stats import skew, kurtosis
from scipy.integrate import nquad
import matplotlib.pyplot as plt

# Assuming particle_filter.py is in the same directory 'code'
from .particle_filter import PFEMIntegrator
# Note: The Particle class itself is not directly used by EnhancedAdaptiveIntegrationEnv
# but PFEMIntegrator is, which in turn uses Particle.


class EnhancedAdaptiveIntegrationEnv(gym.Env):
    """
    An advanced Gymnasium environment for adaptive numerical integration of 2D functions
    using reinforcement learning.

    The agent learns to iteratively partition a 2D integration domain into smaller
    rectangular regions. The goal is to achieve a precise estimate of the definite
    integral of a given function over this domain, while minimizing the number of
    function evaluations (computational effort).

    The environment features:
    - Normalization of observations.
    - Enhanced error estimation techniques (e.g., Richardson extrapolation).
    - Adaptive splitting strategies for regions.
    - Complex reward structure balancing accuracy, efficiency, and handling of
      function complexity.
    - Optional use of Particle Filter Extended Finite Element Method (PFEM) for
      certain types of functions.

    Args:
        ax (float): The lower x-bound of the integration domain. Defaults to 0.0.
        bx (float): The upper x-bound of the integration domain. Defaults to 1.0.
        ay (float): The lower y-bound of the integration domain. Defaults to 0.0.
        by (float): The upper y-bound of the integration domain. Defaults to 1.0.
        max_intervals (int): The maximum number of rectangular regions the domain
                             can be partitioned into before an episode terminates.
                             Defaults to 20.
        function (Callable[[float, float], float]): The 2D function to be integrated.
                                                   Defaults to `lambda x, y: np.sin(x) * np.cos(y)`.
        function_params (Optional[Dict]): A dictionary of parameters that might be
                                          used by the `function` if it's designed
                                          to be configurable. Defaults to None.

    Action Space:
        A `gym.spaces.Box` of shape (4,) with `dtype=np.float32`. The components are:
        - `region_idx_normalized` (float in [0, 1]): Normalized index of the region to split.
                                                    This is mapped to an actual region index.
        - `split_ratio` (float in [0.1, 0.9]): The ratio at which to split the chosen
                                              dimension of the selected region.
        - `dimension` (float in [0, 1]): The dimension to split: 0 for x-axis, 1 for y-axis.
                                         Typically interpreted as int(round(value)).
        - `strategy` (float in [0, 1]): A parameter influencing the splitting strategy.
                                        For example, it might determine whether to use the
                                        `region_idx_normalized` or pick the highest-error
                                        region, or influence an adaptive splitting heuristic.

    Observation Space:
        A `gym.spaces.Box` with `dtype=np.float32`. The shape is
        `(max_intervals * 16 + 5 + n_params,)`, where `n_params` is the number of
        parameters in `function_params`.
        The observation includes:
        - For each of `max_intervals` possible regions (padded with zeros if fewer exist):
            - Normalized coordinates (x0, x1, y0, y1)
            - Normalized width and height
            - Area
            - Function value at center (cached)
            - Gauss-Legendre integral estimate
            - Variation estimates in x and y directions
            - Max and min of these variations
            - Ratio of variations
            - Richardson error estimate
            - Relative error contribution of this region
            (Total 16 features per region)
        - Global statistics:
            - Normalized count of current regions
            - Normalized current integral approximation
            - Normalized current absolute error from true value
            - Normalized total function evaluations
            - Normalized current step count
            (Total 5 global features)
        - `n_params` values from `function_params` (if any).
        All observation values are clipped to the range [-10.0, 10.0].
    """
    def __init__(self,
                 ax: float = 0.0,
                 bx: float = 1.0,
                 ay: float = 0.0,
                 by: float = 1.0,
                 max_intervals: int = 20,
                 function: Callable[[float, float], float] = lambda x, y: np.sin(x) * np.cos(y),
                 function_params: Optional[Dict] = None):
        super().__init__()

        self.ax, self.bx = ax, bx
        self.ay, self.by = ay, by
        self.x_width = bx - ax
        self.y_width = by - ay
        self.max_intervals = max_intervals

        self.f = function
        self.function_params = function_params if function_params is not None else {}
        self.param_values = list(self.function_params.values()) if self.function_params else []

        try:
            self.true_value, _ = nquad(self.f, [[ax, bx], [ay, by]])
        except Exception as e:
            # Fallback if nquad fails (e.g., for highly complex or singular functions)
            # This is a simple Monte Carlo as a fallback.
            # For production, a more robust method or error handling would be needed.
            print(f"Warning: nquad failed to compute true value: {e}. Using MC fallback.")
            mc_samples = 1000000
            x_samples = np.random.uniform(ax, bx, mc_samples)
            y_samples = np.random.uniform(ay, by, mc_samples)
            vals = np.array([self.f(x,y) for x,y in zip(x_samples, y_samples)])
            self.true_value = (self.x_width * self.y_width) * np.mean(vals)


        self.value_scale = 1.0

        self.action_space = spaces.Box(
            low=np.array([0, 0.1, 0, 0]),
            high=np.array([1, 0.9, 1, 1]),
            shape=(4,), dtype=np.float32
        )

        n_params = len(self.param_values)
        self.observation_space = spaces.Box(
            low=-10.0, high=10.0,
            shape=(max_intervals * 16 + 5 + n_params,), # 16 features per region now
            dtype=np.float32
        )

        self.regions: List[Tuple[float, float, float, float]] = []
        self.evals: Dict[Tuple[float, float], float] = {}
        self.center_cache: Dict[Tuple[float, float], float] = {}
        self.region_history: List[Union[Tuple[float, float, float, float], Tuple[float, float, float, float, float, int]]] = []
        
        self.mc_samples = 1000
        self.method_history: List[str] = []

        self.pfem_integrator = PFEMIntegrator(function, min_particles=20, max_particles=100)
        self.use_pfem = False
        self.steps = 0


    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None) -> Tuple[np.ndarray, Dict]:
        """
        Resets the environment to an initial state for a new episode.

        The integration domain is initialized as a single region. Function evaluation
        cache and other historical data are cleared. A value scale is computed
        based on initial samples of the function to aid normalization.

        Args:
            seed (Optional[int]): The seed for the random number generator.
            options (Optional[dict]): Additional options for resetting the environment.

        Returns:
            Tuple[np.ndarray, Dict]: A tuple containing:
                - observation (np.ndarray): The initial observation of the environment.
                - info (Dict): Auxiliary information (empty in this case).
        """
        super().reset(seed=seed)
        self.regions = [(self.ax, self.bx, self.ay, self.by)]
        self.evals = {}
        self.center_cache = {}
        self.steps = 0
        self.region_history = [(self.ax, self.bx, self.ay, self.by)] # Initial region
        self.method_history = []
        self.use_pfem = False # Reset PFEM usage flag

        # Sample points for value scale calculation
        x_points = np.linspace(self.ax, self.bx, 5)
        y_points = np.linspace(self.ay, self.by, 5)
        function_values = []
        for x_val in x_points:
            for y_val in y_points:
                val = self.f(x_val, y_val)
                self.evals[(x_val, y_val)] = val
                function_values.append(val)
        
        if function_values: # Ensure list is not empty
            value_range = max(abs(np.max(function_values) - np.min(function_values)), 1e-10)
            self.value_scale = max(1.0, value_range)
        else: # Fallback if function always returns non-finite or list is empty
            self.value_scale = 1.0


        obs = self._get_observation()
        return obs, {}

    def _eval(self, x: float, y: float) -> float:
        """
        Evaluates the function `self.f` at point (x, y) using a cache.

        Args:
            x (float): The x-coordinate.
            y (float): The y-coordinate.

        Returns:
            float: The function value f(x, y).
        """
        if (x, y) not in self.evals:
            self.evals[(x, y)] = self.f(x, y)
        return self.evals[(x, y)]

    def _gauss_legendre_2d(self, x0: float, x1: float, y0: float, y1: float) -> float:
        """
        Performs 2D Gauss-Legendre quadrature over a rectangular region.

        Uses 5-point Gauss-Legendre rule for each dimension.

        Args:
            x0, x1 (float): The x-bounds of the region.
            y0, y1 (float): The y-bounds of the region.

        Returns:
            float: The estimated integral value over the region.
        """
        weights = np.array([0.236926885056189, 0.478628670499366, 0.568888888888889,
                              0.478628670499366, 0.236926885056189])
        points = np.array([-0.906179845938664, -0.538469310105683, 0.0,
                             0.538469310105683, 0.906179845938664])

        x_points = 0.5 * (x1 - x0) * points + 0.5 * (x1 + x0)
        y_points = 0.5 * (y1 - y0) * points + 0.5 * (y1 + y0)

        result = 0.0
        for i, wx in enumerate(weights):
            for j, wy in enumerate(weights):
                result += wx * wy * self._eval(x_points[i], y_points[j])
        
        return result * 0.25 * (x1 - x0) * (y1 - y0)

    def _richardson_extrapolation(self, x0: float, x1: float, y0: float, y1: float) -> Tuple[float, float]:
        """
        Estimates the integration error for a region using Richardson extrapolation.

        Compares a coarse integral estimate (I1) with a finer estimate (I2) obtained
        by summing integrals over four sub-regions.

        Args:
            x0, x1 (float): The x-bounds of the region.
            y0, y1 (float): The y-bounds of the region.

        Returns:
            Tuple[float, float]: A tuple containing:
                - error_est (float): The estimated error.
                - improved_est (float): An improved integral estimate.
        """
        I1 = self._gauss_legendre_2d(x0, x1, y0, y1)
        xm, ym = (x0 + x1) / 2, (y0 + y1) / 2
        I2 = (self._gauss_legendre_2d(x0, xm, y0, ym) +
              self._gauss_legendre_2d(xm, x1, y0, ym) +
              self._gauss_legendre_2d(x0, xm, ym, y1) +
              self._gauss_legendre_2d(xm, x1, ym, y1))
        
        k = 10  # Order of convergence for 5-point Gauss-Legendre
        error_est = abs(I2 - I1) / (2**k - 1)
        improved_est = I2 + (I2 - I1) / (2**k - 1)
        return error_est, improved_est

    def _analyze_function_behavior(self, x0: float, x1: float, y0: float, y1: float, n_samples: int = 100) -> Dict[str, float]:
        """
        Analyzes statistical properties of the function within a region.

        Calculates oscillation, smoothness, skewness, kurtosis, mean, and std
        deviation of function values sampled within the region. These properties
        can inform the choice of integration method or splitting strategy.

        Args:
            x0, x1 (float): The x-bounds of the region.
            y0, y1 (float): The y-bounds of the region.
            n_samples (int): The number of random samples to take within the region.

        Returns:
            Dict[str, float]: A dictionary of statistical measures.
        """
        x_samples = np.random.uniform(x0, x1, n_samples)
        y_samples = np.random.uniform(y0, y1, n_samples)
        values = np.array([self.f(x, y) for x, y in zip(x_samples, y_samples) if np.isfinite(self.f(x,y))]) # ensure finite values

        if len(values) < 2: # Not enough valid samples for stats
            return {
            'oscillation': 0.0, 'smoothness': 0.0, 'skewness': 0.0, 
            'kurtosis': 0.0, 'mean': 0.0, 'std': 0.0
            }
        
        oscillation = np.std(np.diff(values)) if len(values) > 1 else 0.0
        smoothness = np.mean(np.abs(np.diff(values, 2))) if len(values) > 2 else 0.0
        value_skew = skew(values) if len(values) > 2 else 0.0 # skew needs >2 points
        value_kurt = kurtosis(values) if len(values) > 2 else 0.0 # kurtosis needs >2 points
        
        return {
            'oscillation': oscillation, 'smoothness': smoothness,
            'skewness': value_skew, 'kurtosis': value_kurt,
            'mean': np.mean(values), 'std': np.std(values)
        }

    def _monte_carlo_integrate(self, x0: float, x1: float, y0: float, y1: float, n_samples: Optional[int] = None) -> Tuple[float, float]:
        """
        Performs Monte Carlo integration over a region.

        Optionally adapts the number of samples based on function behavior (e.g.,
        higher oscillation or kurtosis).

        Args:
            x0, x1 (float): The x-bounds of the region.
            y0, y1 (float): The y-bounds of the region.
            n_samples (Optional[int]): Number of MC samples. If None, uses `self.mc_samples`.

        Returns:
            Tuple[float, float]: Estimated integral and error estimate.
        """
        num_samples = n_samples if n_samples is not None else self.mc_samples
        behavior = self._analyze_function_behavior(x0, x1, y0, y1)
        if behavior['oscillation'] > 1.0 or behavior['kurtosis'] > 3.0:
            num_samples *= 2
            
        x_s = np.random.uniform(x0, x1, num_samples)
        y_s = np.random.uniform(y0, y1, num_samples)
        
        values = np.array([self._eval(x, y) for x, y in zip(x_s, y_s)])
        area = (x1 - x0) * (y1 - y0)
        integral = area * np.mean(values)
        error_est = area * np.std(values) / np.sqrt(num_samples)
        return integral, error_est

    def _choose_integration_method(self, x0: float, x1: float, y0: float, y1: float) -> str:
        """
        Heuristically chooses an integration method ('monte_carlo' or 'gaussian')
        based on analyzed function behavior within the region.

        Args:
            x0, x1 (float): The x-bounds of the region.
            y0, y1 (float): The y-bounds of the region.

        Returns:
            str: The name of the chosen integration method.
        """
        behavior = self._analyze_function_behavior(x0, x1, y0, y1)
        use_monte_carlo = (
            behavior['oscillation'] > 2.0 or behavior['kurtosis'] > 5.0 or
            abs(behavior['skewness']) > 2.0 or behavior['smoothness'] > 1.0
        )
        return 'monte_carlo' if use_monte_carlo else 'gaussian'

    def _should_use_pfem(self, x0: float, x1: float, y0: float, y1: float) -> bool:
        """
        Determines if Particle Filter Extended Finite Element Method (PFEM) should
        be used for integrating the current region.

        The decision is based on function properties like oscillation, kurtosis,
        and gradient magnitude.

        Args:
            x0, x1 (float): The x-bounds of the region.
            y0, y1 (float): The y-bounds of the region.

        Returns:
            bool: True if PFEM should be used, False otherwise.
        """
        try:
            behavior = self._analyze_function_behavior(x0, x1, y0, y1)
            curvature = self._analyze_curvature(x0, x1, y0, y1)
            return (behavior['oscillation'] > 2.0 or
                    behavior['kurtosis'] > 5.0 or
                    curvature['gradient_mag'] > 10.0)
        except Exception: # Catch any error during analysis
            return False

    def _adaptive_integrate(self, x0: float, x1: float, y0: float, y1: float) -> Tuple[float, float]:
        """
        Adaptively chooses and applies an integration method for the region.

        May choose between Gauss-Legendre, Monte Carlo, or PFEM based on
        function characteristics.

        Args:
            x0, x1 (float): The x-bounds of the region.
            y0, y1 (float): The y-bounds of the region.

        Returns:
            Tuple[float, float]: Estimated integral and error estimate.
        """
        if self._should_use_pfem(x0, x1, y0, y1):
            self.use_pfem = True # Mark that PFEM was used at least once
            self.pfem_integrator.initialize_particles(x0, x1, y0, y1)
            integral, error = self.pfem_integrator.integrate(x0, x1, y0, y1, self.evals)
            self.pfem_integrator.adapt_particles(x0, x1, y0, y1)
            self.method_history.append('pfem')
        else:
            method = self._choose_integration_method(x0, x1, y0, y1)
            self.method_history.append(method)
            if method == 'monte_carlo':
                integral, error = self._monte_carlo_integrate(x0, x1, y0, y1)
            else: # gaussian
                integral = self._gauss_legendre_2d(x0, x1, y0, y1)
                _, error = self._richardson_extrapolation(x0, x1, y0, y1) # error from Richardson
        return integral, error

    def _get_region_features(self, x0: float, x1: float, y0: float, y1: float) -> np.ndarray:
        """
        Extracts a feature vector for a given 2D region.

        This vector includes geometric properties, function value at center,
        integral estimates, error estimates, and function behavior statistics.
        This is one part of the observation provided to the RL agent.

        Args:
            x0, x1 (float): The x-bounds of the region.
            y0, y1 (float): The y-bounds of the region.

        Returns:
            np.ndarray: A 1D NumPy array of features for the region.
                        The number of features is 16.
        """
        x_width, y_width = x1 - x0, y1 - y0
        area = x_width * y_width
        center_x, center_y = (x0 + x1) / 2, (y0 + y1) / 2
        
        if (center_x, center_y) not in self.center_cache:
             self.center_cache[(center_x, center_y)] = self._eval(center_x, center_y)
        f_center = self.center_cache[(center_x, center_y)]

        integral, error = self._adaptive_integrate(x0, x1, y0, y1)
        richardson_error, _ = self._richardson_extrapolation(x0, x1, y0, y1) # Use Richardson for one error feature
        behavior = self._analyze_function_behavior(x0, x1, y0, y1)
        
        x_variation = self._estimate_variation(x0, x1, center_y, is_y_dim=False)
        y_variation = self._estimate_variation(y0, y1, center_x, is_y_dim=True)

        features = np.array([
            x0, x1, y0, y1,
            x_width, y_width,
            area,
            f_center, # Using f_center from cache
            integral, # Using result from adaptive_integrate
            x_variation,
            y_variation,
            max(x_variation, y_variation) if x_variation is not None and y_variation is not None else 0,
            min(x_variation, y_variation) if x_variation is not None and y_variation is not None else 0,
            x_variation / (y_variation + 1e-10) if x_variation is not None and y_variation is not None and y_variation > 1e-10 else 1.0,
            richardson_error, # Using the more consistent Richardson error for this feature slot
            error / (abs(integral) + 1e-10) if integral != 0 else (error if error !=0 else 0.0) # Relative error from adaptive_integrate
        ], dtype=np.float32)
        return features


    def _estimate_variation(self, a: float, b: float, fixed_coord: float, is_y_dim: bool = False) -> float:
        """
        Estimates the variation of the function along one dimension at a fixed coordinate
        of the other dimension.

        Args:
            a, b (float): The bounds of the interval along which to estimate variation.
            fixed_coord (float): The fixed coordinate value for the other dimension.
            is_y_dim (bool): If True, variation is estimated along y-axis (a,b are y-bounds),
                             otherwise along x-axis.

        Returns:
            float: The estimated variation (max absolute difference / interval length).
        """
        points = np.linspace(a, b, 5)
        values = []
        for p_val in points:
            coord = (fixed_coord, p_val) if is_y_dim else (p_val, fixed_coord)
            values.append(self._eval(*coord))
        
        if len(values) < 2 or (b-a) == 0: # Need at least 2 points to compute diff
            return 0.0
        return max(abs(np.diff(values))) / (b - a)


    def _estimate_total_error(self, x0: float, x1: float, y0: float, y1: float) -> float:
        """
        Estimates the total error for a region by comparing a coarse integral
        estimate with a finer one (sum of integrals over four sub-regions).
        This is similar to the error estimation part of Richardson extrapolation.

        Args:
            x0, x1 (float): The x-bounds of the region.
            y0, y1 (float): The y-bounds of the region.

        Returns:
            float: The absolute difference between coarse and fine integral estimates.
        """
        coarse = self._gauss_legendre_2d(x0, x1, y0, y1)
        xm, ym = (x0 + x1)/2, (y0 + y1)/2
        fine = (self._gauss_legendre_2d(x0, xm, y0, ym) +
                self._gauss_legendre_2d(xm, x1, y0, ym) +
                self._gauss_legendre_2d(x0, xm, ym, y1) +
                self._gauss_legendre_2d(xm, x1, ym, y1))
        return abs(fine - coarse)

    def _get_observation(self) -> np.ndarray:
        """
        Constructs the observation vector for the current state of the environment.

        This involves gathering features for each current region, normalizing them,
        padding if there are fewer than `max_intervals` regions, and appending
        global statistics.

        Returns:
            np.ndarray: The complete, normalized observation vector.
        """
        all_features_list = []
        total_richardson_error_sum = 0
        max_richardson_error = 0

        # Collect raw features and sum Richardson errors for normalization
        # The _get_region_features now returns 16 features, index 14 is Richardson error.
        raw_region_features_list = []
        for x0, x1, y0, y1 in self.regions:
            region_f = self._get_region_features(x0, x1, y0, y1) # This is a (16,) array
            raw_region_features_list.append(region_f)
            error_val = region_f[14] # Richardson error is at index 14
            total_richardson_error_sum += error_val
            max_richardson_error = max(max_richardson_error, error_val)
            
        # Normalize and prepare features for each region
        for region_raw_features in raw_region_features_list:
            # Deconstruct the 16 features
            r_x0, r_x1, r_y0, r_y1, r_w, r_h, r_area, r_fcenter, \
            r_integral, r_xvar, r_yvar, r_maxvar, r_minvar, r_varratio, \
            r_rich_err, r_rel_err = region_raw_features

            norm_x0 = (r_x0 - self.ax) / self.x_width if self.x_width > 0 else 0
            norm_x1 = (r_x1 - self.ax) / self.x_width if self.x_width > 0 else 0
            norm_y0 = (r_y0 - self.ay) / self.y_width if self.y_width > 0 else 0
            norm_y1 = (r_y1 - self.ay) / self.y_width if self.y_width > 0 else 0
            norm_x_width = r_w / self.x_width if self.x_width > 0 else 0
            norm_y_width = r_h / self.y_width if self.y_width > 0 else 0
            
            scale_factor = max(r_area, 1e-10) * self.value_scale
            norm_integral = r_integral / scale_factor

            norm_richardson_error = r_rich_err / (max_richardson_error + 1e-10) # Normalize by max error in any region
            
            norm_x_variation = np.tanh(r_xvar / 10.0)
            norm_y_variation = np.tanh(r_yvar / 10.0)
            norm_max_variation = np.tanh(r_maxvar / 10.0)
            norm_min_variation = np.tanh(r_minvar / 10.0)
            norm_variation_ratio = np.tanh(r_varratio / 10.0) # Already a ratio, but tanh scales it

            rel_error_contribution = r_rich_err / (total_richardson_error_sum + 1e-10)

            normalized_feature_vector = np.array([
                norm_x0, norm_x1, norm_y0, norm_y1,
                norm_x_width, norm_y_width,
                r_area, # Area is not normalized in this version of features
                r_fcenter, # f_center is not normalized here
                norm_integral,
                norm_x_variation, norm_y_variation,
                norm_max_variation, norm_min_variation,
                norm_variation_ratio,
                norm_richardson_error,
                rel_error_contribution
            ], dtype=np.float32)
            all_features_list.append(normalized_feature_vector)

        # Sort regions by error contribution (descending) for consistent observation structure
        if all_features_list:
            # The 16th feature (index 15) is rel_error_contribution
            all_features_list.sort(key=lambda x: x[15], reverse=True) 
        
        # Pad if fewer than max_intervals regions
        # Each feature vector has 16 elements
        while len(all_features_list) < self.max_intervals:
            all_features_list.append(np.zeros(16, dtype=np.float32))

        current_approx = sum(self._gauss_legendre_2d(x0,x1,y0,y1) for x0,x1,y0,y1 in self.regions)
        current_error = abs(current_approx - self.true_value)

        norm_region_count = len(self.regions) / self.max_intervals
        norm_approx_val = current_approx / (self.value_scale * self.x_width * self.y_width + 1e-10)
        
        rel_error_val = min(current_error / (abs(self.true_value) + 1e-10), 1.0)
        norm_error_val = np.log1p(rel_error_val * 10) / np.log(11)

        norm_evals_count = len(self.evals) / (self.max_intervals * 25 + 1e-10) # Estimate 25 evals per region max
        norm_steps_count = self.steps / 50.0 # Assume typical max steps around 50

        global_stats_vec = np.array([
            norm_region_count, norm_approx_val, norm_error_val,
            norm_evals_count, norm_steps_count
        ] + self.param_values, dtype=np.float32) # Add function parameters if any

        final_obs = np.concatenate([np.concatenate(all_features_list), global_stats_vec])
        return np.clip(final_obs, -10.0, 10.0)


    def _analyze_curvature(self, x0: float, x1: float, y0: float, y1: float, n_points: int = 20) -> Dict[str, float]:
        """
        Analyzes function curvature, gradient magnitude, and oscillation within a region.

        Args:
            x0, x1 (float): The x-bounds of the region.
            y0, y1 (float): The y-bounds of the region.
            n_points (int): Number of points per dimension for creating a grid to estimate derivatives.

        Returns:
            Dict[str, float]: Dictionary with 'curvature', 'gradient_mag', 'oscillation'.
        """
        x_pts = np.linspace(x0, x1, n_points)
        y_pts = np.linspace(y0, y1, n_points)
        X, Y = np.meshgrid(x_pts, y_pts)
        Z = np.array([[self._eval(xi, yi) for xi in x_pts] for yi in y_pts])
        
        if Z.size < 4 : # Need at least 2x2 grid for gradient
             return {'curvature': 0.0, 'gradient_mag': 0.0, 'oscillation': 0.0}

        dx, dy = np.gradient(Z)
        dx2, _ = np.gradient(dx) # Second derivative w.r.t x
        _, dy2 = np.gradient(dy) # Second derivative w.r.t y
        
        curvature = np.mean(np.abs(dx2 + dy2)) if Z.size > 0 else 0.0
        gradient_mag = np.mean(np.sqrt(dx**2 + dy**2)) if Z.size > 0 else 0.0
        oscillation = np.std(np.diff(Z.flatten())) if Z.flatten().size > 1 else 0.0
        
        return {'curvature': curvature, 'gradient_mag': gradient_mag, 'oscillation': oscillation}

    def _calculate_enhanced_reward(self, prev_error: float, new_error: float, evals_used: int,
                                   old_region_features: np.ndarray,
                                   new_sub_region_features_list: List[np.ndarray]) -> float:
        """
        Calculates an enhanced reward signal based on various factors.

        These include error reduction, efficiency (error reduction per function
        evaluation), changes in function behavior indicators (oscillation,
        smoothness, curvature) between the parent region and its new sub-regions,
        and gradient improvements.

        Args:
            prev_error (float): Absolute error before the split.
            new_error (float): Absolute error after the split.
            evals_used (int): Number of new function evaluations for this step.
            old_region_features (np.ndarray): Feature vector of the region before split.
                                             Expected to be (16,) array from _get_region_features.
            new_sub_region_features_list (List[np.ndarray]): List of feature vectors for
                                                            the new sub-regions.

        Returns:
            float: The calculated reward, clipped between -10.0 and 10.0.
        """
        try:
            error_reduction = prev_error - new_error
            efficiency = error_reduction / max(np.sqrt(evals_used), 1.0) # Normalize by sqrt of evals
            base_reward = 10.0 * efficiency
            
            # old_region_features indices: x0=0, x1=1, y0=2, y1=3
            behavior_old = self._analyze_function_behavior(
                old_region_features[0], old_region_features[1], 
                old_region_features[2], old_region_features[3]
            )
            
            behavior_new_means = {'oscillation': [], 'smoothness': []}
            for features in new_sub_region_features_list:
                b_new = self._analyze_function_behavior(features[0], features[1], features[2], features[3])
                behavior_new_means['oscillation'].append(b_new['oscillation'])
                behavior_new_means['smoothness'].append(b_new['smoothness'])

            curv_old = self._analyze_curvature(old_region_features[0],old_region_features[1],old_region_features[2],old_region_features[3])
            curv_new_means = {'curvature': [], 'gradient_mag': []}
            for features in new_sub_region_features_list:
                c_new = self._analyze_curvature(features[0],features[1],features[2],features[3])
                curv_new_means['curvature'].append(c_new['curvature'])
                curv_new_means['gradient_mag'].append(c_new['gradient_mag'])

            # Avoid division by zero if new means are zero
            mean_new_osc = np.mean(behavior_new_means['oscillation']) if behavior_new_means['oscillation'] else 0.0
            mean_new_smooth = np.mean(behavior_new_means['smoothness']) if behavior_new_means['smoothness'] else 0.0
            mean_new_curv = np.mean(curv_new_means['curvature']) if curv_new_means['curvature'] else 0.0
            mean_new_grad = np.mean(curv_new_means['gradient_mag']) if curv_new_means['gradient_mag'] else 0.0

            oscillation_factor = max(1.0, behavior_old['oscillation'] / (mean_new_osc + 1e-10))
            smoothness_factor = max(1.0, behavior_old['smoothness'] / (mean_new_smooth + 1e-10))
            curvature_factor = max(1.0, curv_old['curvature'] / (mean_new_curv + 1e-10))
            
            complexity_bonus = 2.0 * (oscillation_factor + smoothness_factor + curvature_factor) / 3.0
            gradient_improvement = max(0, curv_old['gradient_mag'] - mean_new_grad)
            gradient_reward = 3.0 * gradient_improvement
            
            total_reward = base_reward + complexity_bonus + gradient_reward - 0.1 # Small penalty for action

            if behavior_old['oscillation'] > 2.0 or curv_old['curvature'] > 5.0:
                total_reward *= 1.5 # Bonus for tackling difficult regions

            if len(self.regions) >= self.max_intervals and new_error < prev_error: # Terminal bonus
                accuracy_ratio = min(self.max_intervals / len(self.regions), 1.0)
                difficulty_factor = max(1.0, np.mean([b * c for b,c in zip(behavior_new_means['oscillation'], curv_new_means['curvature'])] if behavior_new_means['oscillation'] and curv_new_means['curvature'] else [1.0]))
                total_reward += 5.0 * accuracy_ratio * difficulty_factor
            
            return np.clip(total_reward, -10.0, 10.0)
            
        except Exception as e:
            # print(f"Warning: Error calculating enhanced reward: {str(e)}") # Optional: for debugging
            return -1.0 # Default penalty on error

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """
        Executes one time step within the environment.

        The agent provides an action (which region to split, how, and where).
        The environment updates its state (splits the region, creating two new ones),
        calculates the new integral approximation and error, computes a reward,
        and returns the new observation.

        Args:
            action (np.ndarray): The action chosen by the agent. See Action Space specs.

        Returns:
            Tuple[np.ndarray, float, bool, bool, Dict]: A tuple containing:
                - observation (np.ndarray): The new observation.
                - reward (float): The reward for the taken action.
                - terminated (bool): True if the episode has ended (max_intervals reached).
                - truncated (bool): False, as truncation is not explicitly handled here beyond termination.
                - info (Dict): Auxiliary information (current error, approximation, evals, etc.).
        """
        self.steps += 1
        region_idx_normalized, split_ratio, dimension_float, strategy_float = action
        
        # Store previous state for reward calculation
        prev_evals_count = len(self.evals)
        prev_approx = sum(self._gauss_legendre_2d(x0,x1,y0,y1) for x0,x1,y0,y1 in self.regions)
        prev_error = abs(prev_approx - self.true_value)

        # Determine region to split
        # The _get_observation sorts regions by error contribution.
        # So, taking region_idx_normalized from this sorted list.
        # For strategy > 0.7, we pick the highest error region (which is index 0 after sorting).
        if strategy_float > 0.7 and hasattr(self, 'sorted_to_original_idx'):
            # This assumes sorted_to_original_idx maps sorted index (0 for highest error)
            # to the original index in self.regions *before* it was sorted for the observation.
            # However, self.regions is also sorted by position in step().
            # Simpler: if strategy > 0.7, pick region with highest error directly from current self.regions.
            # This requires re-calculating errors or using stored ones if available.
            # For now, let's use a simplified interpretation tied to the observation's sorting.
            # If observation sorts by error, index 0 *of the observation* is highest error.
            # We need to map this back to self.regions' current order.
            # The current self.regions is sorted by (x0,y0).
            # This makes direct mapping from observation's sorted order complex.
            # Fallback: if strategy > 0.7, pick the region with largest Richardson error from current self.regions
            if self.regions:
                 errors_with_indices = [ (self._get_region_features(*r)[14], i) for i, r in enumerate(self.regions)]
                 errors_with_indices.sort(key=lambda x: x[0], reverse=True)
                 region_idx = errors_with_indices[0][1] if errors_with_indices else 0
            else: region_idx = 0 # Should not happen if not done
        else:
            region_idx = int(region_idx_normalized * (len(self.regions) -1 + 1e-9)) # Map normalized to actual index

        if region_idx >= len(self.regions) or not self.regions:
            obs_fail = self._get_observation()
            return obs_fail, -10.0, True, False, {"error": prev_error, "approximation": prev_approx, "evals": len(self.evals), "regions": len(self.regions), "efficiency":0}


        x0, x1, y0, y1 = self.regions.pop(region_idx)
        old_region_f = self._get_region_features(x0,x1,y0,y1) # Get features before it's gone

        dimension = int(round(dimension_float)) # 0 for x, 1 for y
        new_regions_coords: List[Tuple[float,float,float,float]] = []

        if 0.3 < strategy_float <= 0.7: # Adaptive split
            num_samples = 5
            if dimension == 0: # X-split
                sample_pts = np.linspace(x0, x1, num_samples + 2)[1:-1]
                sample_vals = [self._eval(pt, (y0+y1)/2) for pt in sample_pts]
                changes = [abs(sample_vals[i+1] - sample_vals[i]) for i in range(len(sample_vals)-1)]
                split_pt = (sample_pts[np.argmax(changes)] + sample_pts[np.argmax(changes)+1])/2 if changes else (x0+x1)/2
                new_regions_coords = [(x0, split_pt, y0, y1), (split_pt, x1, y0, y1)]
            else: # Y-split
                sample_pts = np.linspace(y0, y1, num_samples + 2)[1:-1]
                sample_vals = [self._eval((x0+x1)/2, pt) for pt in sample_pts]
                changes = [abs(sample_vals[i+1] - sample_vals[i]) for i in range(len(sample_vals)-1)]
                split_pt = (sample_pts[np.argmax(changes)] + sample_pts[np.argmax(changes)+1])/2 if changes else (y0+y1)/2
                new_regions_coords = [(x0, x1, y0, split_pt), (x0, x1, split_pt, y1)]
        else: # Ratio-based split
            if dimension == 0:
                split_pt = x0 + split_ratio * (x1 - x0)
                new_regions_coords = [(x0, split_pt, y0, y1), (split_pt, x1, y0, y1)]
            else:
                split_pt = y0 + split_ratio * (y1 - y0)
                new_regions_coords = [(x0, x1, y0, split_pt), (x0, x1, split_pt, y1)]
        
        self.regions.extend(new_regions_coords)
        self.region_history.append((x0,x1,y0,y1, split_ratio if not (0.3 < strategy_float <=0.7) else -1, dimension)) # Log split

        new_sub_region_fs = [self._get_region_features(*r) for r in new_regions_coords]

        new_approx = sum(self._adaptive_integrate(r0,r1,s0,s1)[0] for r0,r1,s0,s1 in self.regions)
        new_error = abs(new_approx - self.true_value)
        evals_this_step = len(self.evals) - prev_evals_count

        reward = self._calculate_enhanced_reward(prev_error, new_error, evals_this_step, old_region_f, new_sub_region_fs)
        
        terminated = len(self.regions) >= self.max_intervals
        
        # Sort regions by position (x0, then y0) for consistent observation ordering if needed elsewhere
        self.regions.sort(key=lambda r_coords: (r_coords[0], r_coords[2]))

        obs = self._get_observation()
        info = {
            "error": new_error, "approximation": new_approx, "evals": len(self.evals),
            "regions": len(self.regions), "efficiency": (prev_error - new_error) / max(evals_this_step, 1)
        }
        return obs, reward, terminated, False, info


    def visualize_solution(self, num_points=500):
        """
        Visualizes the current state of the integration process.

        This includes:
        - A contour plot of the function `f(x,y)`.
        - The rectangular regions overlaid on the contour plot.
        - A scatter plot showing the distribution of error estimates at region centers.
        - If PFEM was used, a plot of PFEM particle distribution and their errors.
        - A plot of error convergence history over splits (if available).

        Args:
            num_points (int): The number of points per dimension for the contour plot.
        """
        plt.figure(figsize=(15, 10 if self.use_pfem else 5)) # Adjust figure size
        
        # Plot function and regions
        ax1 = plt.subplot(1, 2, 1) if not self.use_pfem else plt.subplot(2, 2, 1)
        x_grid = np.linspace(self.ax, self.bx, num_points)
        y_grid = np.linspace(self.ay, self.by, num_points)
        X, Y = np.meshgrid(x_grid, y_grid)
        Z = np.vectorize(self.f)(X, Y)
        
        ax1.contourf(X, Y, Z, cmap='viridis', levels=50) # More levels for smoother contour
        #plt.colorbar(label='f(x, y)', ax=ax1) # Correct way to add colorbar to subplot
        
        for x0, x1, y0, y1 in self.regions:
            ax1.plot([x0, x1, x1, x0, x0], [y0, y0, y1, y1, y0], 'r-', alpha=0.7, linewidth=1.5)
        ax1.set_title('Function and Integration Regions')
        ax1.set_xlabel('x')
        ax1.set_ylabel('y')
        
        # Plot error distribution
        ax2 = plt.subplot(1, 2, 2) if not self.use_pfem else plt.subplot(2, 2, 2)
        # Use Richardson error (index 14) from _get_region_features
        errors = [self._get_region_features(r0,r1,s0,s1)[14] for r0,r1,s0,s1 in self.regions]
        region_centers_x = [(r0+r1)/2 for r0,r1,s0,s1 in self.regions]
        region_centers_y = [(s0+s1)/2 for r0,r1,s0,s1 in self.regions]
        
        sc = ax2.scatter(region_centers_x, region_centers_y, c=errors, cmap='hot', s=100, edgecolors='k')
        #plt.colorbar(sc, label='Error Estimate', ax=ax2)
        ax2.set_title('Error Distribution (Richardson Estimate)')
        ax2.set_xlabel('x')
        ax2.set_ylabel('y')
        
        if self.use_pfem:
            ax3 = plt.subplot(2, 2, 3)
            if self.pfem_integrator.particles: # Check if particles exist
                particle_x = [p.x for p in self.pfem_integrator.particles]
                particle_y = [p.y for p in self.pfem_integrator.particles]
                particle_errors = [p.error_estimate for p in self.pfem_integrator.particles]
                
                sc_pfem = ax3.scatter(particle_x, particle_y, c=particle_errors, cmap='coolwarm', alpha=0.8, s=50, edgecolors='k')
                #plt.colorbar(sc_pfem, label='Particle Error', ax=ax3)
                
                # Plot particle connectivity (optional, can be noisy)
                # for p in self.pfem_integrator.particles:
                #     for n in p.neighbors:
                #         ax3.plot([p.x, n.x], [p.y, n.y], 'b-', alpha=0.1)
            ax3.set_title('PFEM Particle Distribution')
            ax3.set_xlabel('x')
            ax3.set_ylabel('y')
        
        # Plot convergence history (optional, needs history tracking)
        # This part requires self.true_value and a history of approximations
        # For example, if region_history stores (approx_val_at_split_k, ...)
        # This was not fully implemented in the provided snippet for error history.
        # Assuming region_history stores tuples where the first element might be an approximation or error
        # This part needs proper data from history.
        # if hasattr(self, 'region_history') and self.region_history:
        #     ax4 = plt.subplot(2,2,4) if self.use_pfem else plt.subplot(1,3,3) # Adjust subplotting
        #     # Example: if region_history stores (approx_value, num_evals) tuples at each step
        #     # steps = range(len(self.region_history))
        #     # errors_over_time = [abs(rh[0] - self.true_value) for rh in self.region_history if isinstance(rh, tuple) and len(rh)>0]
        #     # if errors_over_time:
        #     #    ax4.semilogy(steps[:len(errors_over_time)], errors_over_time, 'b-', label='Error')
        #     #    ax4.grid(True)
        #     #    ax4.set_title('Convergence History (Example)')
        #     #    ax4.set_xlabel('Split Number / Step')
        #     #    ax4.set_ylabel('Error (log scale)')


        plt.tight_layout()
        plt.show()
        
        # Print summary statistics
        current_approx = sum(self._adaptive_integrate(r0,r1,s0,s1)[0] for r0,r1,s0,s1 in self.regions)
        current_error = abs(current_approx - self.true_value)
        
        print("\nVisualization Summary Statistics:")
        print(f"True Value:           {self.true_value:.10e}")
        print(f"Current Approximation:{current_approx:.10e}")
        print(f"Current Absolute Error: {current_error:.10e}")
        if abs(self.true_value) > 1e-12: # Avoid division by zero for true value of 0
            print(f"Current Relative Error: {current_error/abs(self.true_value):.10e}")
        print(f"Function Evaluations: {len(self.evals)}")
        print(f"Number of Regions:    {len(self.regions)}")
        if self.use_pfem and self.pfem_integrator.particles:
            print(f"PFEM Particles:       {len(self.pfem_integrator.particles)}")
        elif self.use_pfem:
             print(f"PFEM Particles:       0 (PFEM was enabled but no particles)")
