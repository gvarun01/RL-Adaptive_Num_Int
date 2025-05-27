# !pip install stable_baselines3
import numpy as np
import matplotlib.pyplot as plt
import gymnasium as gym
from gymnasium import spaces
import torch.nn as nn
from typing import Callable, Dict, List, Tuple, Optional, Union
import time
from scipy.stats import skew, kurtosis
from scipy.interpolate import UnivariateSpline
from scipy.integrate import quad, nquad
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy
import scipy.special as sp
import os
from stable_baselines3.common.vec_env import VecNormalize



def linear_schedule(initial_value: float, final_value: float) -> Callable[[float], float]:

    """
    Linear learning rate schedule.
    :param initial_value: Initial learning rate.
    :param final_value: Final learning rate.
    :return: schedule that computes current learning rate depending on remaining progress
    """
    def func(progress_remaining: float) -> float:
        """
        Progress will decrease from 1 (beginning) to 0 (end)
        :param progress_remaining:
        :return: current learning rate
        """
        return final_value + progress_remaining * (initial_value - final_value)
    return func


class EarlyStopCallback(BaseCallback):
    """
    Enhanced callback for early stopping with improved monitoring and stopping criteria.
    Tracks both global and local error improvements.
    """
    def __init__(self, 
                 check_freq: int = 5000, 
                 min_improvement: float = 1e-6, 
                 min_local_improvement: float = 1e-7,
                 patience: int = 5, 
                 min_episodes: int = 20, 
                 verbose: int = 1):
        super().__init__(verbose)
        self.check_freq = check_freq
        self.min_improvement = min_improvement
        self.min_local_improvement = min_local_improvement
        self.patience = patience
        self.min_episodes = min_episodes
        self.verbose = verbose
        
        # Initialize tracking variables
        self.best_mean_reward = -float('inf')
        self.best_local_error = float('inf')
        self.no_improvement_count = 0
        self.episode_count = 0
        self.reward_history = []
        self.error_history = []
        self.local_error_history = []
        self.training_start = time.time()

    def _on_step(self) -> bool:
        if self.n_calls % self.check_freq == 0:
            # Get current metrics
            mean_reward = self.model.logger.name_to_value.get('rollout/ep_rew_mean', None)
            ep_count = self.model.logger.name_to_value.get('time/episodes', 0)
            mean_error = self.model.logger.name_to_value.get('rollout/ep_error_mean', float('inf'))
            local_error = self.model.logger.name_to_value.get('rollout/local_error_mean', float('inf'))
            
            if mean_reward is not None:
                self.reward_history.append(mean_reward)
                self.error_history.append(mean_error)
                self.local_error_history.append(local_error)
                self.episode_count = ep_count
                
                # Calculate improvements
                reward_improvement = mean_reward - self.best_mean_reward
                local_error_improvement = self.best_local_error - local_error
                
                # Check for significant improvement in either metric
                if (reward_improvement > self.min_improvement or 
                    local_error_improvement > self.min_local_improvement):
                    self.best_mean_reward = max(mean_reward, self.best_mean_reward)
                    self.best_local_error = min(local_error, self.best_local_error)
                    self.no_improvement_count = 0
                    if self.verbose > 0:
                        elapsed_time = time.time() - self.training_start
                        print(f"\nImprovement at episode {ep_count} ({elapsed_time:.1f}s):")
                        print(f"  Mean reward:     {mean_reward:.6f}")
                        print(f"  Local error:     {local_error:.6e}")
                else:
                    self.no_improvement_count += 1
                    if self.verbose > 0:
                        print(f"\nNo significant improvement: {self.no_improvement_count}/{self.patience}")
                        print(f"  Current reward:  {mean_reward:.6f}")
                        print(f"  Current local error: {local_error:.6e}")
                
                # Check stopping conditions
                if self.episode_count < self.min_episodes:
                    return True
                
                # Stop if no improvement for too long
                if self.no_improvement_count >= self.patience:
                    if self.verbose > 0:
                        print("\nEarly stopping triggered:")
                        print(f"  Episodes:        {self.episode_count}")
                        print(f"  Final reward:    {mean_reward:.6f}")
                        print(f"  Best reward:     {self.best_mean_reward:.6f}")
                        print(f"  Final local error: {local_error:.6e}")
                        print(f"  Training time:   {time.time() - self.training_start:.1f}s")
                    return False
                
                # Check for performance degradation
                if len(self.reward_history) > 5:
                    recent_reward_mean = np.mean(self.reward_history[-5:])
                    recent_error_mean = np.mean(self.local_error_history[-5:])
                    if (recent_reward_mean < self.best_mean_reward * 0.5 or 
                        recent_error_mean > self.best_local_error * 2.0):
                        if self.verbose > 0:
                            print("\nStopping due to performance degradation:")
                            print(f"  Recent reward mean: {recent_reward_mean:.6f}")
                            print(f"  Recent error mean:  {recent_error_mean:.6e}")
                        return False
        
        return True

    def get_training_summary(self) -> Dict:
        """Return summary of training progress"""
        return {
            'best_reward': self.best_mean_reward,
            'best_local_error': self.best_local_error,
            'episodes': self.episode_count,
            'training_time': time.time() - self.training_start,
            'reward_history': self.reward_history,
            'error_history': self.error_history,
            'local_error_history': self.local_error_history
        }


class Particle:
    """Represents a particle for PFEM integration"""
    def __init__(self, x: float, y: float, weight: float = 1.0):
        self.x = x
        self.y = y
        self.weight = weight
        self.value = None
        self.neighbors = []
        self.error_estimate = 0.0

class PFEMIntegrator:
    """Handles PFEM-based integration"""
    def __init__(self, function, min_particles=20, max_particles=100):
        self.function = function
        self.min_particles = min_particles
        self.max_particles = max_particles
        self.particles = []
        
    def initialize_particles(self, x0, x1, y0, y1, n_initial=20):
        """Initialize particles in the region with jittered grid distribution"""
        nx = ny = int(np.sqrt(n_initial))
        self.particles = []
        
        for i in range(nx):
            for j in range(ny):
                # Add jitter to avoid regular grid artifacts
                jitter_x = np.random.uniform(-0.1, 0.1) * (x1 - x0) / nx
                jitter_y = np.random.uniform(-0.1, 0.1) * (y1 - y0) / ny
                
                x = x0 + (i + 0.5) * (x1 - x0) / nx + jitter_x
                y = y0 + (j + 0.5) * (y1 - y0) / ny + jitter_y
                
                self.particles.append(Particle(x, y))

    def update_particle_values(self, eval_cache):
        """Update function values at particle locations using cache"""
        for p in self.particles:
            if (p.x, p.y) in eval_cache:
                p.value = eval_cache[(p.x, p.y)]
            else:
                p.value = self.function(p.x, p.y)
                eval_cache[(p.x, p.y)] = p.value

    def find_neighbors(self, max_dist):
        """Find neighbors for each particle within max_dist"""
        for p1 in self.particles:
            p1.neighbors = []
            for p2 in self.particles:
                if p1 != p2:
                    dist = np.sqrt((p1.x - p2.x)**2 + (p1.y - p2.y)**2)
                    if dist < max_dist:
                        p1.neighbors.append(p2)

    def estimate_local_error(self):
        """Estimate error for each particle based on neighbor value differences"""
        for p in self.particles:
            if p.neighbors:
                values = [n.value for n in p.neighbors]
                p.error_estimate = np.std(values)

    def adapt_particles(self, x0, x1, y0, y1):
        """Adapt particle distribution based on error estimates"""
        # Remove particles with low error estimates
        self.particles = [p for p in self.particles if p.error_estimate > np.median([p.error_estimate for p in self.particles])]
        
        # Add particles in high error regions
        new_particles = []
        for p in self.particles:
            if p.error_estimate > np.percentile([p.error_estimate for p in self.particles], 75):
                # Add new particles around high error particle
                for _ in range(2):
                    radius = 0.1 * min(x1 - x0, y1 - y0)
                    angle = np.random.uniform(0, 2 * np.pi)
                    new_x = np.clip(p.x + radius * np.cos(angle), x0, x1)
                    new_y = np.clip(p.y + radius * np.sin(angle), y0, y1)
                    new_particles.append(Particle(new_x, new_y))
        
        self.particles.extend(new_particles)
        
        # Limit total number of particles
        if len(self.particles) > self.max_particles:
            self.particles = sorted(self.particles, key=lambda p: p.error_estimate, reverse=True)[:self.max_particles]

    def integrate(self, x0, x1, y0, y1, eval_cache):
        """Perform PFEM integration"""
        area = (x1 - x0) * (y1 - y0)
        self.update_particle_values(eval_cache)
        max_dist = 0.2 * min(x1 - x0, y1 - y0)
        self.find_neighbors(max_dist)
        self.estimate_local_error()
        
        # Weighted sum of particle values
        total_weight = sum(p.weight for p in self.particles)
        integral = area * sum(p.value * p.weight for p in self.particles) / total_weight
        
        # Error estimate based on particle distribution
        error = np.mean([p.error_estimate for p in self.particles])
        
        return integral, error


class EnhancedAdaptiveIntegrationEnv(gym.Env):
    """
    Advanced environment for adaptive numerical integration using reinforcement learning.
    Includes normalization, enhanced error estimation, adaptive splitting, and advanced rewards.
    """
    def __init__(self,
                 ax: float = 0.0,
                 bx: float = 1.0,
                 ay: float = 0.0,
                 by: float = 1.0,
                 max_intervals: int = 20,
                 function: Callable[[float, float], float] = lambda x, y: np.sin(x) * np.cos(y),
                 function_params: Optional[Dict] = None):
        """
        Initialize the advanced adaptive integration environment for 2D.

        Args:
            ax, bx (float): Lower and upper bounds of x domain
            ay, by (float): Lower and upper bounds of y domain
            max_intervals (int): Maximum number of rectangular regions
            function (callable): 2D function to integrate
            function_params (dict, optional): Parameters of the function
        """
        super().__init__()

        # Domain boundaries
        self.ax, self.bx = ax, bx
        self.ay, self.by = ay, by
        self.x_width = bx - ax
        self.y_width = by - ay
        self.max_intervals = max_intervals

        # Function and parameters
        self.f = function
        self.function_params = function_params if function_params is not None else {}
        self.param_values = list(self.function_params.values()) if self.function_params else []

        # Calculate true value using high-precision integration
        self.true_value, _ = nquad(self.f, [[ax, bx], [ay, by]])

        # Normalization factors
        self.value_scale = 1.0

        # Action space: [region_idx, split_ratio, dimension, strategy]
        # dimension: 0 for x-split, 1 for y-split
        self.action_space = spaces.Box(
            low=np.array([0, 0.1, 0, 0]),
            high=np.array([1, 0.9, 1, 1]),
            shape=(4,), dtype=np.float32
        )

        # Enhanced observation space for 2D
        # Features per region: 15 base + 1 Richardson
        # Global stats: 5 base + function parameters
        n_params = len(self.param_values)
        self.observation_space = spaces.Box(
            low=-10.0, high=10.0,
            shape=(max_intervals * 16 + 5 + n_params,),
            dtype=np.float32
        )

        # Initialize storage for rectangular regions
        self.regions = []  # List of (x0, x1, y0, y1) tuples
        self.evals = {}
        self.center_cache = {}
        self.region_history = []
        
        # Add integration method parameters
        self.mc_samples = 1000  # Base number of Monte Carlo samples
        self.method_history = []  # Track which method was used for each region

        # Initialize PFEM integrator
        self.pfem_integrator = PFEMIntegrator(function)
        self.use_pfem = False

    def reset(self, seed=None, options=None):
        """
        Reset the environment for a new episode.

        Returns:
            tuple: (observation, info)
        """
        super().reset(seed=seed)
        # Initialize with single rectangle covering whole domain
        self.regions = [(self.ax, self.bx, self.ay, self.by)]
        self.evals = {}
        self.center_cache = {}
        self.steps = 0
        self.region_history = [(self.ax, self.bx, self.ay, self.by)]

        # Sample points for value scale calculation
        x_points = np.linspace(self.ax, self.bx, 5)
        y_points = np.linspace(self.ay, self.by, 5)
        function_values = []
        for x in x_points:
            for y in y_points:
                val = self.f(x, y)
                self.evals[(x, y)] = val
                function_values.append(val)

        value_range = max(abs(np.max(function_values) - np.min(function_values)), 1e-10)
        self.value_scale = max(1.0, value_range)

        obs = self._get_observation()
        return obs, {}

    def _eval(self, x, y):
        """
        Evaluate function with caching.

        Args:
            x, y (float): Points to evaluate

        Returns:
            float: Function value at (x, y)
        """
        if (x, y) not in self.evals:
            self.evals[(x, y)] = self.f(x, y)
        return self.evals[(x, y)]

    def _gauss_legendre_2d(self, x0, x1, y0, y1):
        """
        2D Gauss-Legendre quadrature.

        Args:
            x0, x1, y0, y1 (float): Region bounds

        Returns:
            float: Integral approximation
        """
        # 5-point GL weights and points (same as before)
        weights = np.array([0.236926885056189, 0.478628670499366, 0.568888888888889,
                          0.478628670499366, 0.236926885056189])
        points = np.array([-0.906179845938664, -0.538469310105683, 0.0,
                          0.538469310105683, 0.906179845938664])

        # Transform points to intervals
        x_points = 0.5 * (x1 - x0) * points + 0.5 * (x1 + x0)
        y_points = 0.5 * (y1 - y0) * points + 0.5 * (y1 + y0)

        # 2D integration
        result = 0.0
        for i, wx in enumerate(weights):
            for j, wy in enumerate(weights):
                if (x_points[i], y_points[j]) not in self.evals:
                    self.evals[(x_points[i], y_points[j])] = self.f(x_points[i], y_points[j])
                result += wx * wy * self.evals[(x_points[i], y_points[j])]

        return result * 0.25 * (x1 - x0) * (y1 - y0)

    def _richardson_extrapolation(self, x0, x1, y0, y1):
        """
        Estimate error using Richardson extrapolation.

        Args:
            x0, x1, y0, y1 (float): Region bounds

        Returns:
            tuple: (error_estimate, improved_estimate)
        """
        # Calculate first approximation (coarse)
        I1 = self._gauss_legendre_2d(x0, x1, y0, y1)

        # Calculate second approximation (finer, splitting region)
        xm, ym = (x0 + x1) / 2, (y0 + y1) / 2
        I2 = (self._gauss_legendre_2d(x0, xm, y0, ym) +
              self._gauss_legendre_2d(xm, x1, y0, ym) +
              self._gauss_legendre_2d(x0, xm, ym, y1) +
              self._gauss_legendre_2d(xm, x1, ym, y1))

        # Richardson extrapolation formula for error estimation
        # For 5-point Gauss-Legendre, error should decrease as O(h^10)
        k = 10  # Order of convergence
        error_est = abs(I2 - I1) / (2**k - 1)

        # Also return improved estimate
        improved_est = I2 + (I2 - I1) / (2**k - 1)

        return error_est, improved_est

    def _analyze_function_behavior(self, x0, x1, y0, y1, n_samples=100):
        """Analyze function behavior in a region to determine best integration method"""
        x_samples = np.random.uniform(x0, x1, n_samples)
        y_samples = np.random.uniform(y0, y1, n_samples)
        values = np.array([self.f(x, y) for x, y in zip(x_samples, y_samples)])
        
        # Calculate statistical measures
        oscillation = np.std(np.diff(values))
        smoothness = np.mean(np.abs(np.diff(values, 2)))
        value_skew = skew(values)
        value_kurt = kurtosis(values)
        
        return {
            'oscillation': oscillation,
            'smoothness': smoothness,
            'skewness': value_skew,
            'kurtosis': value_kurt,
            'mean': np.mean(values),
            'std': np.std(values)
        }

    def _monte_carlo_integrate(self, x0, x1, y0, y1, n_samples=None):
        """Monte Carlo integration with importance sampling"""
        if n_samples is None:
            n_samples = self.mc_samples
            
        # Generate samples with importance sampling near high-gradient regions
        behavior = self._analyze_function_behavior(x0, x1, y0, y1)
        if behavior['oscillation'] > 1.0 or behavior['kurtosis'] > 3.0:
            # Use more samples for challenging regions
            n_samples *= 2
            
        x_samples = np.random.uniform(x0, x1, n_samples)
        y_samples = np.random.uniform(y0, y1, n_samples)
        
        values = np.array([self._eval(x, y) for x, y in zip(x_samples, y_samples)])
        area = (x1 - x0) * (y1 - y0)
        integral = area * np.mean(values)
        error_est = area * np.std(values) / np.sqrt(n_samples)
        
        return integral, error_est

    def _choose_integration_method(self, x0, x1, y0, y1):
        """Choose between Monte Carlo and Gaussian quadrature based on function behavior"""
        behavior = self._analyze_function_behavior(x0, x1, y0, y1)
        
        # Decision criteria
        use_monte_carlo = (
            behavior['oscillation'] > 2.0 or  # Highly oscillatory
            behavior['kurtosis'] > 5.0 or     # Heavy-tailed distribution
            abs(behavior['skewness']) > 2.0 or # Highly skewed
            behavior['smoothness'] > 1.0       # Non-smooth function
        )
        
        return 'monte_carlo' if use_monte_carlo else 'gaussian'

    def _should_use_pfem(self, x0, x1, y0, y1):
        """Determine if PFEM should be used for this region"""
        try:
            # Sample points to analyze function behavior
            behavior = self._analyze_function_behavior(x0, x1, y0, y1)
            curvature = self._analyze_curvature(x0, x1, y0, y1)
            
            # Use PFEM if function is highly oscillatory or has sharp gradients
            return (behavior['oscillation'] > 2.0 or
                    behavior['kurtosis'] > 5.0 or
                    curvature['gradient_mag'] > 10.0)
        except:
            return False

    def _adaptive_integrate(self, x0, x1, y0, y1):
        """Adaptively choose and apply integration method"""
        if self._should_use_pfem(x0, x1, y0, y1):
            self.use_pfem = True
            self.pfem_integrator.initialize_particles(x0, x1, y0, y1)
            integral, error = self.pfem_integrator.integrate(x0, x1, y0, y1, self.evals)
            self.pfem_integrator.adapt_particles(x0, x1, y0, y1)
            self.method_history.append('pfem')
        else:
            method = self._choose_integration_method(x0, x1, y0, y1)
            self.method_history.append(method)
            
            if method == 'monte_carlo':
                integral, error = self._monte_carlo_integrate(x0, x1, y0, y1)
            else:
                integral = self._gauss_legendre_2d(x0, x1, y0, y1)
                _, error = self._richardson_extrapolation(x0, x1, y0, y1)
                
        return integral, error

    def _get_region_features(self, x0, x1, y0, y1):
        """
        Extract features from a 2D region.

        Args:
            x0, x1, y0, y1 (float): Region bounds

        Returns:
            ndarray: Feature vector
        """
        x_width = x1 - x0
        y_width = y1 - y0
        area = x_width * y_width

        # Basic evaluations
        center_x = (x0 + x1) / 2
        center_y = (y0 + y1) / 2
        
        if (center_x, center_y) not in self.center_cache:
            self.center_cache[(center_x, center_y)] = self.f(center_x, center_y)
        f_center = self.center_cache[(center_x, center_y)]

        # Integration estimates
        gauss_integral = self._gauss_legendre_2d(x0, x1, y0, y1)
        
        # Error estimates for both dimensions
        x_variation = self._estimate_variation(x0, x1, center_y)
        y_variation = self._estimate_variation(y0, y1, center_x, is_y_dim=True)

        # Combine features
        features = np.array([
            x0, x1, y0, y1,        # Region boundaries
            x_width, y_width,      # Dimensions
            area,                  # Area
            f_center,             # Center value
            gauss_integral,       # Integration estimate
            x_variation,          # X-direction variation
            y_variation,          # Y-direction variation
            max(x_variation, y_variation),  # Max variation
            min(x_variation, y_variation),  # Min variation
            x_variation/y_variation if y_variation > 1e-10 else 1.0,  # Variation ratio
            self._estimate_total_error(x0, x1, y0, y1)  # Total error estimate
        ], dtype=np.float32)
        
        # Add integration method results
        integral, error = self._adaptive_integrate(x0, x1, y0, y1)
        behavior = self._analyze_function_behavior(x0, x1, y0, y1)
        
        # Enhanced feature vector
        features = np.array([
            x0, x1, y0, y1,        # Region boundaries
            x_width, y_width,      # Dimensions
            area,                  # Area
            integral,              # Adaptive integration result
            error,                 # Error estimate
            behavior['oscillation'],
            behavior['smoothness'],
            behavior['skewness'],
            behavior['kurtosis'],
            1.0 if self.method_history[-1] == 'monte_carlo' else 0.0,  # Method indicator
            error / (abs(integral) + 1e-10)  # Relative error
        ], dtype=np.float32)
        
        return features

    def _estimate_variation(self, a, b, fixed_coord, is_y_dim=False):
        """
        Estimate variation along one dimension.

        Args:
            a, b (float): Interval bounds
            fixed_coord (float): Fixed coordinate
            is_y_dim (bool): Whether the dimension is y

        Returns:
            float: Variation estimate
        """
        points = np.linspace(a, b, 5)
        values = []
        for p in points:
            coord = (fixed_coord, p) if is_y_dim else (p, fixed_coord)
            if coord not in self.evals:
                self.evals[coord] = self.f(*coord)
            values.append(self.evals[coord])
        return max(abs(np.diff(values))) / (b - a)

    def _estimate_total_error(self, x0, x1, y0, y1):
        """
        Estimate total error for a region using multiple refinements.

        Args:
            x0, x1, y0, y1 (float): Region bounds

        Returns:
            float: Total error estimate
        """
        coarse = self._gauss_legendre_2d(x0, x1, y0, y1)
        
        # Split region into four and compare
        xm, ym = (x0 + x1)/2, (y0 + y1)/2
        fine = (self._gauss_legendre_2d(x0, xm, y0, ym) +
                self._gauss_legendre_2d(xm, x1, y0, ym) +
                self._gauss_legendre_2d(x0, xm, ym, y1) +
                self._gauss_legendre_2d(xm, x1, ym, y1))
        
        return abs(fine - coarse)

    def _get_observation(self):
        """
        Create normalized observation vector from current state.

        Returns:
            ndarray: Normalized observation vector
        """
        features = []

        # First pass: collect all raw features and calculate total error
        raw_features = []
        total_error = 0
        max_error = 0

        for x0, x1, y0, y1 in self.regions:
            # Get raw features for this region
            feature = self._get_region_features(x0, x1, y0, y1)
            raw_features.append(feature)

            # Track total and max error for normalization
            error = feature[14]  # Richardson error (more accurate)
            total_error += error
            max_error = max(max_error, error)

        # Second pass: normalize features and add relative error information
        for feature in raw_features:
            # Extract components for normalization
            x0, x1, y0, y1 = feature[0], feature[1], feature[2], feature[3]
            x_width, y_width = feature[4], feature[5]
            area = feature[6]
            f_center = feature[7]
            gauss_integral = feature[8]
            x_variation, y_variation = feature[9], feature[10]
            max_variation, min_variation = feature[11], feature[12]
            variation_ratio = feature[13]
            richardson_error = feature[14]

            # Normalize position to [0,1] within domain
            norm_x0 = (x0 - self.ax) / self.x_width
            norm_x1 = (x1 - self.ax) / self.x_width
            norm_y0 = (y0 - self.ay) / self.y_width
            norm_y1 = (y1 - self.ay) / self.y_width

            # Normalize dimensions relative to domain
            norm_x_width = x_width / self.x_width
            norm_y_width = y_width / self.y_width

            # Normalize integration values by value scale and area
            scale_factor = max(area, 1e-10) * self.value_scale
            norm_gauss_integral = gauss_integral / scale_factor

            # Normalize error estimates relative to max error
            if max_error > 1e-10:
                norm_richardson_error = richardson_error / max_error
            else:
                norm_richardson_error = 0.0

            # Normalize variations
            norm_x_variation = np.tanh(x_variation / 10.0)  # Tanh keeps in [-1, 1]
            norm_y_variation = np.tanh(y_variation / 10.0)
            norm_max_variation = np.tanh(max_variation / 10.0)
            norm_min_variation = np.tanh(min_variation / 10.0)
            norm_variation_ratio = np.tanh(variation_ratio / 10.0)

            # Calculate relative error contribution
            rel_error_contribution = richardson_error / (total_error + 1e-10)

            # Create normalized feature vector
            normalized_feature = np.array([
                norm_x0, norm_x1, norm_y0, norm_y1,
                norm_x_width, norm_y_width,
                area,
                f_center,
                norm_gauss_integral,
                norm_x_variation, norm_y_variation,
                norm_max_variation, norm_min_variation,
                norm_variation_ratio,
                norm_richardson_error,
                rel_error_contribution
            ], dtype=np.float32)

            features.append(normalized_feature)

        # Sort regions by error contribution (highest error first)
        if features:
            features.sort(key=lambda x: x[15], reverse=True)

            # Store mapping from sorted indices to original indices
            self.sorted_to_original_idx = {}
            for i, (x0, x1, y0, y1) in enumerate(self.regions):
                for j, feature in enumerate(features):
                    # Match original region to sorted feature using normalized positions
                    orig_x0_norm = (x0 - self.ax) / self.x_width
                    orig_x1_norm = (x1 - self.ax) / self.x_width
                    orig_y0_norm = (y0 - self.ay) / self.y_width
                    orig_y1_norm = (y1 - self.ay) / self.y_width
                    if (abs(feature[0] - orig_x0_norm) < 1e-6 and
                        abs(feature[1] - orig_x1_norm) < 1e-6 and
                        abs(feature[2] - orig_y0_norm) < 1e-6 and
                        abs(feature[3] - orig_y1_norm) < 1e-6):
                        self.sorted_to_original_idx[j] = i
                        break

        # Pad to max_intervals with zeros
        while len(features) < self.max_intervals:
            features.append(np.zeros(16, dtype=np.float32))

        # Calculate current approximation and error
        approx = sum(self._gauss_legendre_2d(x0, x1, y0, y1) for x0, x1, y0, y1 in self.regions)
        error = abs(approx - self.true_value)

        # Normalize global statistics
        norm_region_count = len(self.regions) / self.max_intervals
        norm_approx = approx / (self.value_scale * self.x_width * self.y_width)

        # Normalize error on log scale to handle wide range of errors
        if self.true_value != 0:
            rel_error = min(error / abs(self.true_value), 1.0)  # Cap at 100% error
        else:
            rel_error = min(error, 1.0)  # If true value is 0, use absolute error capped at 1
        norm_error = np.log1p(rel_error * 10) / np.log(11)  # Maps [0,1] to [0,1] with log scaling

        # Normalized evaluation count
        norm_evals = len(self.evals) / (self.max_intervals * 25)  # Assuming ~25 evals per region max

        # Normalized step count
        norm_steps = self.steps / 50.0  # Assuming max steps of 50

        # Global stats with normalization
        global_stats = np.array([
            norm_region_count,
            norm_approx,
            norm_error,
            norm_evals,
            norm_steps
        ] + self.param_values, dtype=np.float32)

        # Return flattened array of all features and global stats
        output = np.concatenate([np.concatenate(features), global_stats])
        return np.clip(output, -10.0, 10.0)  # Clip to observation space bounds

    def _analyze_curvature(self, x0, x1, y0, y1, n_points=20):
        """Analyze function curvature in the region"""
        x = np.linspace(x0, x1, n_points)
        y = np.linspace(y0, y1, n_points)
        X, Y = np.meshgrid(x, y)
        Z = np.array([[self._eval(xi, yi) for xi in x] for yi in y])
        
        # Calculate gradient and curvature
        dx, dy = np.gradient(Z)
        dx2, _ = np.gradient(dx)
        _, dy2 = np.gradient(dy)
        
        # Mean absolute curvature
        curvature = np.mean(np.abs(dx2 + dy2))
        # Gradient magnitude
        gradient_mag = np.mean(np.sqrt(dx**2 + dy**2))
        # Oscillation measure
        oscillation = np.std(np.diff(Z.flatten()))
        
        return {
            'curvature': curvature,
            'gradient_mag': gradient_mag,
            'oscillation': oscillation
        }

    def _calculate_enhanced_reward(self, prev_error, new_error, evals_used, old_features, new_features):
        """Calculate reward with enhanced metrics"""
        try:
            # Basic error reduction reward
            error_reduction = prev_error - new_error
            efficiency = error_reduction / max(np.sqrt(evals_used), 1.0)
            base_reward = 10.0 * efficiency
            
            # Function behavior rewards
            behavior_old = self._analyze_function_behavior(
                old_features[0], old_features[1], 
                old_features[2], old_features[3]
            )
            
            # Calculate average behavior for new regions
            behavior_new = []
            for features in new_features:
                b = self._analyze_function_behavior(
                    features[0], features[1], 
                    features[2], features[3]
                )
                behavior_new.append(b)
            
            # Curvature analysis for old and new regions
            curv_old = self._analyze_curvature(
                old_features[0], old_features[1], 
                old_features[2], old_features[3]
            )
            
            curv_new = [
                self._analyze_curvature(f[0], f[1], f[2], f[3])
                for f in new_features
            ]
            
            # Reward components based on function properties
            oscillation_factor = max(
                1.0,
                behavior_old['oscillation'] / (np.mean([b['oscillation'] for b in behavior_new]) + 1e-10)
            )
            
            smoothness_factor = max(
                1.0,
                behavior_old['smoothness'] / (np.mean([b['smoothness'] for b in behavior_new]) + 1e-10)
            )
            
            curvature_factor = max(
                1.0,
                curv_old['curvature'] / (np.mean([c['curvature'] for c in curv_new]) + 1e-10)
            )
            
            # Additional rewards for handling challenging regions well
            complexity_bonus = 2.0 * (
                oscillation_factor +
                smoothness_factor +
                curvature_factor
            ) / 3.0
            
            # Gradient-based reward
            gradient_improvement = max(
                0,
                curv_old['gradient_mag'] - np.mean([c['gradient_mag'] for c in curv_new])
            )
            gradient_reward = 3.0 * gradient_improvement
            
            # Combine all reward components
            total_reward = (
                base_reward +
                complexity_bonus +
                gradient_reward -
                0.1  # Small constant penalty
            )
            
            # Scale reward based on region properties
            if behavior_old['oscillation'] > 2.0 or curv_old['curvature'] > 5.0:
                total_reward *= 1.5  # Bonus for handling difficult regions
            
            # Terminal rewards
            if len(self.regions) >= self.max_intervals:
                if new_error < prev_error:
                    accuracy_ratio = min(self.max_intervals / len(self.regions), 1.0)
                    difficulty_factor = max(
                        1.0,
                        np.mean([b['oscillation'] * c['curvature'] 
                               for b, c in zip(behavior_new, curv_new)])
                    )
                    terminal_bonus = 5.0 * accuracy_ratio * difficulty_factor
                    total_reward += terminal_bonus
            
            return np.clip(total_reward, -10.0, 10.0)  # Clip reward for stability
            
        except Exception as e:
            print(f"Warning: Error calculating enhanced reward: {str(e)}")
            return -1.0

    def step(self, action):
        """
        Take a step by splitting a region with enhanced strategy.

        Args:
            action (ndarray): [region_idx_normalized, split_ratio, dimension, strategy]

        Returns:
            tuple: (observation, reward, terminated, truncated, info)
        """
        self.steps += 1

        # Extract action components with fallback for compatibility
        if len(action) == 4:
            region_idx_normalized, split_ratio, dimension, strategy = action
        else:
            region_idx_normalized, split_ratio, dimension = action
            strategy = 0.5  # Default strategy

        # Store previous state for reward calculation
        prev_evals_count = len(self.evals)
        prev_approx = sum(self._gauss_legendre_2d(x0, x1, y0, y1) for x0, x1, y0, y1 in self.regions)
        prev_error = abs(prev_approx - self.true_value)

        # Determine which region to split based on strategy
        if strategy > 0.7:  # Use highest-error region
            # Sort regions by estimated error
            sorted_regions = sorted(
                range(len(self.regions)),
                key=lambda i: self._get_region_features(*self.regions[i])[14],  # Richardson error
                reverse=True
            )
            region_idx = sorted_regions[0] if sorted_regions else 0
        else:  # Use selected region
            region_idx = int(region_idx_normalized * (len(self.regions) - 0.001))

            # Map through sorted_to_original_idx if available
            if hasattr(self, 'sorted_to_original_idx') and region_idx in self.sorted_to_original_idx:
                region_idx = self.sorted_to_original_idx[region_idx]

        # Handle out-of-bounds index
        if region_idx >= len(self.regions):
            return self._get_observation(), -1.0, True, False, {
                "error": prev_error,
                "approximation": prev_approx,
                "evals": len(self.evals)
            }

        # Extract the region to split
        x0, x1, y0, y1 = self.regions.pop(region_idx)
        x_width = x1 - x0
        y_width = y1 - y0

        # Get features of the region being split for reward calculation
        old_features = self._get_region_features(x0, x1, y0, y1)
        old_error_estimate = old_features[14]  # Richardson error estimate

        # Apply adaptive splitting based on strategy
        if 0.3 < strategy <= 0.7:
            # Adaptive split based on function behavior
            # Sample more points to find where function changes most
            if dimension == 0:  # x-split
                n_samples = 5
                sample_points = np.linspace(x0, x1, n_samples+2)[1:-1]  # Skip endpoints
                sample_values = [self._eval(x, (y0 + y1) / 2) for x in sample_points]

                # Find largest change in function values
                changes = [abs(sample_values[i+1] - sample_values[i])
                           for i in range(len(sample_values)-1)]
                if changes:
                    max_change_idx = np.argmax(changes)
                    split_point = (sample_points[max_change_idx] +
                                   sample_points[max_change_idx+1]) / 2
                else:
                    # Default to midpoint
                    split_point = (x0 + x1) / 2
                new_regions = [(x0, split_point, y0, y1), (split_point, x1, y0, y1)]
            else:  # y-split
                n_samples = 5
                sample_points = np.linspace(y0, y1, n_samples+2)[1:-1]  # Skip endpoints
                sample_values = [self._eval((x0 + x1) / 2, y) for y in sample_points]

                # Find largest change in function values
                changes = [abs(sample_values[i+1] - sample_values[i])
                           for i in range(len(sample_values)-1)]
                if changes:
                    max_change_idx = np.argmax(changes)
                    split_point = (sample_points[max_change_idx] +
                                   sample_points[max_change_idx+1]) / 2
                else:
                    # Default to midpoint
                    split_point = (y0 + y1) / 2
                new_regions = [(x0, x1, y0, split_point), (x0, x1, split_point, y1)]
        else:
            # User-specified split ratio
            if dimension == 0:  # x-split
                split_point = x0 + split_ratio * x_width
                new_regions = [(x0, split_point, y0, y1), (split_point, x1, y0, y1)]
            else:  # y-split
                split_point = y0 + split_ratio * y_width
                new_regions = [(x0, x1, y0, split_point), (x0, x1, split_point, y1)]

        # Add new regions
        self.regions.extend(new_regions)

        # Keep track of split history
        self.region_history.append((x0, x1, y0, y1, split_point, dimension))

        # Calculate new approximation and error
        new_approx = sum(self._adaptive_integrate(x0, x1, y0, y1)[0]
                         for x0, x1, y0, y1 in self.regions)
        new_error = abs(new_approx - self.true_value)

        # Calculate error reduction and evaluations used
        error_reduction = prev_error - new_error
        evals_used = len(self.evals) - prev_evals_count

        # Calculate efficiency metrics
        efficiency_factor = 1.0 / np.sqrt(max(evals_used, 1))
        error_reduction = prev_error - new_error
        total_error = sum(self._get_region_features(*region)[14] for region in self.regions)
        max_error = max(self._get_region_features(*region)[14] for region in self.regions)

        # Calculate immediate reward
        immediate_reward = 10.0 * error_reduction * efficiency_factor
        
        # Calculate local metrics
        local_errors = [self._get_region_features(*region)[14] for region in new_regions]
        local_error_improvement = (max(old_error_estimate, 1e-10) - max(local_errors, default=0)) / max(old_error_estimate, 1e-10)
        local_error_ratio = max(local_errors, default=0) / (total_error + 1e-10)
        
        # Calculate reward components
        local_reward = 5.0 * local_error_improvement * efficiency_factor
        smoothness_reward = 5.0 * (1.0 - np.std(local_errors) / (np.mean(local_errors) + 1e-10))
        exploration_reward = 2.5 * old_error_estimate * (1 + local_error_ratio)
        
        # Combine rewards with weights
        global_weight = 0.6
        local_weight = 0.4
        base_reward = (
            global_weight * (immediate_reward + exploration_reward) +
            local_weight * (local_reward + smoothness_reward)
        )

        # Apply urgency factor
        regions_left = max(0, self.max_intervals - len(self.regions))
        urgency_factor = np.exp(-regions_left / (self.max_intervals * 0.3))
        total_reward = base_reward * (1.0 + urgency_factor) - 0.1  # Small constant penalty

        done = len(self.regions) >= self.max_intervals

        # Terminal rewards with local error consideration
        if done:
            if new_error < prev_error:
                accuracy_ratio = min(self.max_intervals / max(len(self.regions), 1), 1.0)
                global_accuracy = 1.0 - min(new_error * 1e6, 1.0)
                local_accuracy = 1.0 - min(max(local_errors, default=0) * 1e6, 1.0)
                
                terminal_bonus = 10.0 * accuracy_ratio * (
                    global_weight * global_accuracy +
                    local_weight * local_accuracy
                )
                total_reward += terminal_bonus

        # Sort regions by position for more efficient lookup
        self.regions.sort(key=lambda x: (x[0], x[2]))

        old_features = self._get_region_features(x0, x1, y0, y1)
        new_features = [self._get_region_features(*region) for region in new_regions]
        
        reward = self._calculate_enhanced_reward(
            prev_error,
            new_error,
            evals_used,
            old_features,
            new_features
        )

        return self._get_observation(), reward, done, False, {
            "error": new_error,
            "approximation": new_approx,
            "evals": len(self.evals),
            "regions": len(self.regions),
            "efficiency": error_reduction / max(evals_used, 1) if evals_used > 0 else 0
        }

    def visualize_solution(self, num_points=500):
        """Enhanced visualization with PFEM particle distribution"""
        # Create figure with subplots
        plt.figure(figsize=(15, 10))
        
        # Plot function and regions
        plt.subplot(2, 2, 1)
        x = np.linspace(self.ax, self.bx, num_points)
        y = np.linspace(self.ay, self.by, num_points)
        X, Y = np.meshgrid(x, y)
        Z = np.vectorize(self.f)(X, Y)
        
        plt.contourf(X, Y, Z, cmap='viridis')
        plt.colorbar(label='f(x, y)')
        
        # Plot integration regions
        for x0, x1, y0, y1 in self.regions:
            plt.plot([x0, x1, x1, x0, x0], [y0, y0, y1, y1, y0], 'r-', alpha=0.5)
        
        plt.title('Function and Integration Regions')
        plt.xlabel('x')
        plt.ylabel('y')
        
        # Plot error distribution
        plt.subplot(2, 2, 2)
        errors = [self._get_region_features(x0, x1, y0, y1)[14] for x0, x1, y0, y1 in self.regions]
        region_centers_x = [(x0 + x1) / 2 for x0, x1, y0, y1 in self.regions]
        region_centers_y = [(y0 + y1) / 2 for x0, x1, y0, y1 in self.regions]
        
        plt.scatter(region_centers_x, region_centers_y, c=errors, cmap='hot', s=100)
        plt.colorbar(label='Error Estimate')
        plt.title('Error Distribution')
        plt.xlabel('x')
        plt.ylabel('y')
        
        # Plot PFEM particles if used
        if self.use_pfem:
            plt.subplot(2, 2, 3)
            particle_x = [p.x for p in self.pfem_integrator.particles]
            particle_y = [p.y for p in self.pfem_integrator.particles]
            particle_errors = [p.error_estimate for p in self.pfem_integrator.particles]
            
            plt.scatter(particle_x, particle_y, 
                       c=particle_errors, 
                       cmap='hot', 
                       alpha=0.8, 
                       s=50)
            plt.colorbar(label='Particle Error')
            plt.title('PFEM Particle Distribution')
            plt.xlabel('x')
            plt.ylabel('y')
            
            # Plot particle connectivity
            for p in self.pfem_integrator.particles:
                for n in p.neighbors:
                    plt.plot([p.x, n.x], [p.y, n.y], 'b-', alpha=0.2)
        
        # Plot convergence history
        plt.subplot(2, 2, 4)
        if hasattr(self, 'region_history'):
            history_x = range(len(self.region_history))
            errors = [abs(r[0] - self.true_value) for r in self.region_history]
            plt.semilogy(history_x, errors, 'b-', label='Error')
            plt.grid(True)
            plt.title('Convergence History')
            plt.xlabel('Split Number')
            plt.ylabel('Error (log scale)')
        
        plt.tight_layout()
        plt.show()
        
        # Print summary statistics
        approx = sum(self._adaptive_integrate(x0, x1, y0, y1)[0] for x0, x1, y0, y1 in self.regions)
        error = abs(approx - self.true_value)
        
        print("\nSummary Statistics:")
        print(f"True Value:           {self.true_value:.10e}")
        print(f"Approximation:        {approx:.10e}")
        print(f"Absolute Error:       {error:.10e}")
        print(f"Relative Error:       {error/abs(self.true_value):.10e}")
        print(f"Function Evaluations: {len(self.evals)}")
        print(f"Number of Regions:    {len(self.regions)}")
        if self.use_pfem:
            print(f"PFEM Particles:       {len(self.pfem_integrator.particles)}")



def linear_schedule(initial_value: float, final_value: float) -> Callable[[float], float]:
    """Linear learning rate schedule."""
    def func(progress_remaining: float) -> float:
        return final_value + (initial_value - final_value) * progress_remaining
    return func

# Function to create a custom environment factory for a specific function
def make_env_factory(func, ax, bx, ay, by, max_intervals=40):
    """Create factory function for environment creation"""
    def _init():
        env = EnhancedAdaptiveIntegrationEnv(
            ax=ax, bx=bx, ay=ay, by=by,
            max_intervals=max_intervals,
            function=func
        )
        return env
    return _init

# Define challenging integration functions
def define_test_functions() -> Dict[str, Tuple[Callable, float, float]]:
    """Define a variety of challenging functions with their integration bounds
    Returns a dictionary mapping function names to (function, lower_bound, upper_bound)"""

    functions = {}

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
    functions["sqrt_singularity"] = (lambda x: 1 / np.sqrt(x), 1e-6, 1.0)  # Singularity at x=0
    functions["log_singularity"] = (lambda x: np.log(x), 1e-6, 2.0)  # Singularity at x=0
    functions["inverse_singularity"] = (lambda x: 1 / (x - 0.5)**2 if abs(x - 0.5) > 1e-6 else 0, 0.0, 1.0)  # Singularity at x=0.5

    # 5. Combined challenging behaviors
    functions["oscillating_with_peaks"] = (lambda x: np.sin(10 * x) + 5 * np.exp(-100 * (x - 0.5)**2), 0.0, 1.0)
    functions["discontinuous_oscillatory"] = (lambda x: np.sin(20 * x) * (1 if x > 0.5 else 0.5), 0.0, 1.0)

    return functions


def define_2d_test_functions() -> Dict[str, Tuple[Callable, float, float, float, float]]:
    """Define comprehensive set of 2D test functions"""
    functions = {}
    
    # 1. Standard smooth functions
    functions["gaussian_2d"] = (
        lambda x, y: np.exp(-(x**2 + y**2)),
        -3.0, 3.0, -3.0, 3.0
    )
    functions["sinc_2d"] = (
        lambda x, y: np.sinc(x) * np.sinc(y),
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
    functions["bessel_2d"] = (
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
    functions["peaks_2d"] = (
        lambda x, y: 3*(1-x)**2 * np.exp(-x**2 - (y+1)**2) - 
                    10*(x/5 - x**3 - y**5) * np.exp(-x**2 - y**2) - 
                    1/3 * np.exp(-(x+1)**2 - y**2),
        -3.0, 3.0, -3.0, 3.0
    )
    functions["gaussian_peaks"] = (
        lambda x, y: sum(np.exp(-((x-xi)**2 + (y-yi)**2)/0.1) 
                        for xi, yi in [(-1,-1), (1,1), (-1,1), (1,-1)]),
        -2.0, 2.0, -2.0, 2.0
    )
    
    #  4. Discontinuous functions
    functions["step_2d"] = (
        lambda x, y: 1.0 if x > 0 and y > 0 else 0.0,
        -1.0, 1.0, -1.0, 1.0
    )
    functions["checkerboard"] = (
        lambda x, y: 1.0 if (int(2*x) + int(2*y)) % 2 == 0 else 0.0,
        0.0, 2.0, 0.0, 2.0
    )
    functions["circular_step"] = (
        lambda x, y: 1.0 if x**2 + y**2 < 1 else 0.0,
        -2.0, 2.0, -2.0, 2.0
    )
    functions["sawtooth_2d"] = (
        lambda x, y: (x - np.floor(x)) * (y - np.floor(y)),
        0.0, 3.0, 0.0, 3.0
    )
    
    # 5. Functions with singularities
    functions["inverse_r"] = (
        lambda x, y: 1.0 / (np.sqrt(x**2 + y**2) + 1e-10),
        -1.0, 1.0, -1.0, 1.0
    )
    functions["log_singularity_2d"] = (
        lambda x, y: np.log(x**2 + y**2 + 1e-10),
        -1.0, 1.0, -1.0, 1.0
    )
    functions["pole_singularity"] = (
        lambda x, y: 1.0 / ((x**2 + y**2 - 0.5**2)**2 + 0.1),
        -1.0, 1.0, -1.0, 1.0
    )
    
    # 6. Combined challenging behaviors
    functions["oscillating_peaks_2d"] = (
        lambda x, y: np.sin(10*x) * np.cos(10*y) * np.exp(-((x-0.5)**2 + (y-0.5)**2)),
        0.0, 2.0, 0.0, 2.0
    )
    functions["mixed_features"] = (
        lambda x, y: (np.sin(20*x*y) / (1 + x**2 + y**2) + 
                     np.exp(-((x-0.5)**2 + (y-0.5)**2) * 10)),
        -2.0, 2.0, -2.0, 2.0
    )
    functions["complex_oscillatory"] = (
        lambda x, y: np.sin(30*x) * np.cos(30*y) + np.exp(-((x-0.5)**2 + (y-0.5)**2) * 5),
        -1.0, 1.0, -1.0, 1.0
    )
    functions["hybrid_singularity"] = (
        lambda x, y: np.sin(10*x*y) / (0.1 + x**2 + y**2),
        -2.0, 2.0, -2.0, 2.0
    )
    
    return functions


# Training function
# Modified version of the train_model function to ensure training stops at 200000 steps
def train_model(functions, training_steps=200000, save_dir="models", evaluate=True):
    """Train a model on sequence of 2D functions with enhanced local error focus and error handling"""
    # Import necessary modules at the function level to ensure availability
    import os
    import traceback
    import warnings
    import torch
    import sys
    from tqdm import std
    from stable_baselines3.common.callbacks import CallbackList
    
    os.makedirs(save_dir, exist_ok=True)
    model = None
    
    # Set up warnings
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    warnings.filterwarnings("ignore", category=UserWarning)
    
    # Fix progress bar conflicts by forcing standard tqdm
    os.environ["TQDM_DISABLE"] = "0"
    os.environ["RICH_FORCE_TERMINAL"] = "0"
    # Monkey patch tqdm to use standard version
    sys.modules["tqdm.auto"] = std
    sys.modules["tqdm"].tqdm = std.tqdm
    
    # Function wrapper to handle invalid values and errors
    def safe_function_wrapper(original_func):
        """Wrapper that makes integration functions robust against invalid values"""
        def safe_func(*args):
            try:
                result = original_func(*args)
                # Ensure result is valid
                if not np.isfinite(result):
                    return 0.0
                return result
            except Exception as e:
                # Return safe value on any calculation error
                return 0.0
        return safe_func

    # Enhanced training configuration with adjusted parameters for longer training
    # Use a more gradually decreasing learning rate
    schedule = linear_schedule(5e-4, 1e-5)  # Adjusted for longer training
    early_stop = EarlyStopCallback(
        check_freq=5000,  # Increased frequency for longer training
        min_improvement=1e-5,  # More precise improvement threshold
        min_local_improvement=1e-6,  # More precise local improvement threshold
        patience=10,  # Increased patience for longer training
        min_episodes=50,  # Increased minimum episodes
        verbose=1
    )

    for i, (func_name, (func, ax, bx, ay, by)) in enumerate(functions.items()):
        print(f"\n{'-'*50}")
        print(f"Training on 2D function: {func_name}")
        
        # Make function safe against invalid values
        safe_func = safe_function_wrapper(func)
        
        # Test the function at boundary points to ensure it's stable
        test_points = [(ax, ay), (ax, by), (bx, ay), (bx, by), 
                      ((ax+bx)/2, (ay+by)/2)]
        
        print(f"Testing function stability at boundaries...")
        valid_function = True
        for x, y in test_points:
            try:
                val = safe_func(x, y)
                if not np.isfinite(val):
                    print(f"  Warning: Function returns non-finite value at ({x}, {y}): {val}")
                    valid_function = False
            except Exception as e:
                print(f"  Error evaluating function at ({x}, {y}): {str(e)}")
                valid_function = False
        
        if not valid_function:
            print(f"Skipping function {func_name} due to stability issues")
            continue

        # Create environment with enhanced local error handling
        def make_env():
            try:
                # Create environment with Box action space to match the model
                env = EnhancedAdaptiveIntegrationEnv(
                    ax=ax, bx=bx, ay=ay, by=by,
                    max_intervals=40,  # Increased max intervals
                    function=safe_func  # Use safe version of function
                )
                
                # Verify the action space is correct
                if not isinstance(env.action_space, spaces.Box) or env.action_space.shape != (4,):
                    print(f"WARNING: Environment action space is {env.action_space}, expected Box(4)")
                    print("Fixing action space...")
                    env.action_space = spaces.Box(
                        low=np.array([0, 0.1, 0, 0]),
                        high=np.array([1, 0.9, 1, 1]),
                        shape=(4,), dtype=np.float32
                    )
                
                # Add observation validation to prevent NaN values
                original_reset = env.reset
                original_step = env.step
                
                def safe_reset(*args, **kwargs):
                    obs, info = original_reset(*args, **kwargs)
                    # Replace any NaN/inf values with zeros
                    if np.any(~np.isfinite(obs)):
                        obs = np.nan_to_num(obs, nan=0.0, posinf=0.0, neginf=0.0)
                    return obs, info
                
                def safe_step(action):
                    obs, reward, done, truncated, info = original_step(action)
                    # Replace any NaN/inf values
                    if np.any(~np.isfinite(obs)):
                        obs = np.nan_to_num(obs, nan=0.0, posinf=0.0, neginf=0.0)
                    if not np.isfinite(reward):
                        reward = 0.0
                    return obs, reward, done, truncated, info
                
                env.reset = safe_reset
                env.step = safe_step
                
                return env
            except Exception as e:
                print(f"Error creating environment: {str(e)}")
                # Create a default environment that won't cause errors
                default_env = EnhancedAdaptiveIntegrationEnv()
                # Ensure the default environment has the correct action space
                default_env.action_space = spaces.Box(
                    low=np.array([0, 0.1, 0, 0]),
                    high=np.array([1, 0.9, 1, 1]),
                    shape=(4,), dtype=np.float32
                )
                return default_env

        # Create vectorized environment with enhanced normalization and error handling
        try:
            vec_env = SubprocVecEnv([make_env for _ in range(8)])
            vec_env = VecNormalize(
                vec_env,
                norm_obs=True,
                norm_reward=True,
                clip_obs=10.0,
                clip_reward=10.0,
                gamma=0.99  # Standard gamma
            )
        except Exception as e:
            print(f"Error creating environment: {str(e)}")
            print("Skipping this function")
            continue

        if model is None:
            try:
                model = PPO(
                    "MlpPolicy",
                    vec_env,
                    verbose=1,
                    learning_rate=schedule,
                    n_steps=2048,  # Larger steps for more stable updates in longer training
                    batch_size=256,  # Larger batches for more stable gradients
                    gamma=0.99,
                    tensorboard_log=f"{save_dir}/tensorboard/",
                    policy_kwargs={
                        'net_arch': [128, 64],  # Larger network for more capacity
                        'activation_fn': nn.Tanh,  # Use Tanh for better numerical stability
                        'ortho_init': True,
                        'log_std_init': -2.0,
                        # squash_output requires use_sde=True
                        'squash_output': True  # Add squashing for additional stability
                    },
                    max_grad_norm=0.5,  # Strong gradient clipping
                    use_sde=True,  # Enable SDE to allow squash_output
                    sde_sample_freq=8,  # Increased sampling frequency for SDE
                    clip_range=0.15,  # Slightly larger clip range for longer training
                    clip_range_vf=0.15,
                    ent_coef=0.01  # Slightly increased for better exploration
                )
            except Exception as e:
                print(f"Error creating model: {str(e)}")
                vec_env.close()
                continue
        else:
            # Reset normalization stats for new function
            try:
                vec_env = VecNormalize(
                    vec_env,
                    norm_obs=True,
                    norm_reward=True,
                    clip_obs=10.0,
                    clip_reward=10.0,
                    gamma=0.99
                )
                model.set_env(vec_env)
            except Exception as e:
                print(f"Error resetting environment: {str(e)}")
                vec_env.close()
                continue

        try:
            print(f"Training for {training_steps} steps...")
            start_time = time.time()

            # Custom callback for monitoring local errors with error handling
            class LocalErrorMonitor(BaseCallback):
                def __init__(self, verbose=0):
                    super().__init__(verbose)
                    self.local_errors = []

                def _on_step(self):
                    try:
                        if len(self.model.ep_info_buffer) > 0:
                            info = self.model.ep_info_buffer[-1]
                            if 'local_errors' in info:
                                # Validate value before adding
                                error_val = np.mean(info['local_errors'])
                                if np.isfinite(error_val):
                                    self.local_errors.append(error_val)
                    except Exception as e:
                        if self.verbose > 0:
                            print(f"Warning in error monitor: {str(e)}")
                    return True

            local_monitor = LocalErrorMonitor()
            
            # Enhanced callback to ensure hard stop at training_steps with error handling
            class StrictStepLimitCallback(BaseCallback):
                def __init__(self, total_steps: int, verbose: int = 0):
                    super().__init__(verbose)
                    self.total_steps = total_steps
                    self.training_start = time.time()
                    # More frequent progress reports for longer training
                    self.warning_intervals = [int(total_steps * x) for x in [0.1, 0.25, 0.5, 0.75, 0.9]]
                    self.warnings_issued = set()
                
                def _on_step(self) -> bool:
                    try:
                        # Progress notifications
                        current_steps = self.num_timesteps
                        for interval in self.warning_intervals:
                            if current_steps >= interval and interval not in self.warnings_issued:
                                elapsed = time.time() - self.training_start
                                if self.verbose > 0:
                                    print(f"Progress: {current_steps}/{self.total_steps} steps ({current_steps/self.total_steps*100:.0f}%) after {elapsed:.1f}s")
                                self.warnings_issued.add(interval)
                        
                        # Check if we've reached the desired number of steps
                        if current_steps >= self.total_steps:
                            elapsed = time.time() - self.training_start
                            if self.verbose > 0:
                                print(f"\nReached {current_steps}/{self.total_steps} steps after {elapsed:.1f}s")
                                print("Stopping training as requested.")
                            # Return False to stop training
                            return False
                    except Exception as e:
                        if self.verbose > 0:
                            print(f"Warning in step limit callback: {str(e)}")
                    return True

            # Create the step limit callback with the correct max steps
            step_limit = StrictStepLimitCallback(total_steps=training_steps, verbose=1)
            
            # Value monitoring callback to catch NaN/Inf values
            class ValueMonitorCallback(BaseCallback):
                def __init__(self, verbose=0):
                    super().__init__(verbose)
                    self.nan_detected = False
                    self.check_freq = 5  # Check very frequently
                
                def _on_step(self) -> bool:
                    if self.n_calls % self.check_freq == 0:
                        # Check model parameters for invalid values
                        for name, param in self.model.policy.named_parameters():
                            if torch.isnan(param).any() or torch.isinf(param).any():
                                if not self.nan_detected:
                                    print(f"WARNING: NaN/Inf detected in model parameter {name}!")
                                    self.nan_detected = True
                                # Reset parameters to small random values
                                param.data.copy_(torch.randn_like(param.data) * 0.01)
                    return True
            
            value_monitor = ValueMonitorCallback(verbose=1)
            
            # Use CallbackList to combine callbacks
            from stable_baselines3.common.callbacks import CallbackList
            callbacks = CallbackList([early_stop, local_monitor, step_limit, value_monitor])

            # Disable progress bar in model.learn to avoid conflict
            # Run training with error handling
            try:
                model.learn(
                    total_timesteps=training_steps + 100,  # Add a buffer to ensure our callback has control
                    callback=callbacks,
                    progress_bar=False,  # Disable the progress bar completely
                    reset_num_timesteps=True  # Start counting from 0 for each function
                )
            except Exception as e:
                print(f"Training error: {str(e)}")
                print("Attempting to continue with the next function")
                # Do not return here - continue with other functions
            
            # Save normalization statistics
            try:
                vec_env.save(f"{save_dir}/vec_normalize_{i}_{func_name}.pkl")
            except Exception as e:
                print(f"Error saving normalization stats: {str(e)}")

            print("\nTraining Summary:")
            # Enhanced training summary with error handling
            try:
                summary = early_stop.get_training_summary()
                print(f"Best reward: {summary['best_reward']:.6f}")
                print(f"Best local error: {summary['best_local_error']:.6e}")
                
                # Safely calculate mean of local errors
                if local_monitor.local_errors and len(local_monitor.local_errors) >= 10:
                    mean_local_error = np.mean(local_monitor.local_errors[-10:])
                    if np.isfinite(mean_local_error):  # Removed extra parenthesis
                        print(f"Mean local error (last 10): {mean_local_error:.6e}")
                    else:
                        print("Mean local error: Invalid values detected")
                else:
                    print("Mean local error: Insufficient data")
                    
                print(f"Episodes completed: {summary['episodes']}")
                print(f"Training time: {time.time() - start_time:.1f}s")
            except Exception as e:
                print(f"Error generating training summary: {str(e)}")

            # Save model with metadata and error handling
            try:
                model_path = f"{save_dir}/adaptive_integration_2d_{i}_{func_name}"
                model.save(model_path)
                print(f"Model saved to {model_path}")
            except Exception as e:
                print(f"Error saving model: {str(e)}")

            if evaluate:
                try:
                    # Use fewer episodes to avoid taking too long
                    mean_reward, std_reward = evaluate_policy(
                        model, vec_env, n_eval_episodes=5,
                        deterministic=True
                    )
                    print(f"\nEvaluation results:")
                    print(f"Mean reward: {mean_reward:.4f} ± {std_reward:.4f}")
                except Exception as e:
                    print(f"Error during evaluation: {str(e)}")
        except Exception as e:
            print(f"\nError during training: {str(e)}")
            traceback.print_exc()  # Added to show full error trace
        finally:
            # Ensure environment is always closed properly
            try:
                vec_env.close()
            except:
                pass
            
    # Save final model with error handling
    try:
        final_model_path = f"{save_dir}/adaptive_integration_final"
        if model is not None:
            model.save(final_model_path)
            print(f"Final model saved to {final_model_path}")
    except Exception as e:
        print(f"Error saving final model: {str(e)}")
    
    return model


# Visualization function to evaluate and visualize results
def evaluate_and_visualize(model_path, functions, max_intervals=20):
    """Evaluate the trained model"""
    model = PPO.load(model_path)
    results = {}

    for func_name, (func, ax, bx, ay, by) in functions.items():
        print(f"\nEvaluating on {func_name}...")

        # Create environment using EnhancedAdaptiveIntegrationEnv
        env = EnhancedAdaptiveIntegrationEnv(
            ax=ax, bx=bx, ay=ay, by=by,
            max_intervals=max_intervals,
            function=func
        )

        # Reset environment
        obs, _ = env.reset()
        total_reward = 0
        # Track progress
        done = False
        total_reward = 0
        while not done:
            # Get action from model
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, _, info = env.step(action)
            total_reward += reward

        # Store results
        results[func_name] = {
            'true_value': env.true_value,
            'approximation': info['approximation'],
            'error': info['error'],
            'num_intervals': len(env.intervals),
            'num_evaluations': len(env.evals)
        }
        print(f"  True value:     {env.true_value:.8f}")
        print(f"  Approximation:  {info['approximation']:.8f}")
        print(f"  Error:          {info['error']:.8e}")
        print(f"  Intervals used: {len(env.intervals)}")
        print(f"  Evaluations:    {len(env.evals)}")
    return results

def evaluate_and_visualize_2d(model_path, functions, vec_normalize_path=None, max_intervals=30):
    """Evaluate the trained model on 2D functions"""
    model = PPO.load(model_path)
    results = {}

    # Get the observation space dimension expected by the model
    expected_obs_dim = model.policy.observation_space.shape[0]
    print(f"Model expects observation shape: {expected_obs_dim}")

    for func_name, (func, ax, bx, ay, by) in functions.items():
        print(f"\nEvaluating on {func_name}...")
        
        # Special handling for circular_step since it has discontinuities
        is_circular_step = func_name == "circular_step"
        
        # Wrap the function to handle potential numerical issues
        def safe_func(x, y):
            try:
                result = func(x, y)
                return 0.0 if not np.isfinite(result) else result
            except:
                return 0.0

        # Create environment using EnhancedAdaptiveIntegrationEnv with safe function
        env = EnhancedAdaptiveIntegrationEnv(
            ax=ax, bx=bx, ay=ay, by=by,
            max_intervals=max_intervals,
            function=safe_func if is_circular_step else func
        )

        # Wrap in VecNormalize if path provided
        if vec_normalize_path:
            env = DummyVecEnv([lambda: env])
            env = VecNormalize.load(vec_normalize_path, env)
            env.training = False  # Don't update statistics during evaluation
            env.norm_reward = False  # Don't normalize rewards during evaluation

        # Reset environment
        obs, _ = env.reset()
        print(f"Environment observation shape: {obs.shape}")
        
        total_reward = 0
        # Track progress
        done = False
        step_count = 0
        error_history = []
        nan_encountered = False
        
        while not done:
            try:
                # Fix observation shape mismatch
                if len(obs.shape) == 1 and obs.shape[0] != expected_obs_dim:
                    # Adapt observation to expected dimension
                    if obs.shape[0] < expected_obs_dim:
                        # Pad with zeros if observation is smaller than expected
                        padded_obs = np.zeros(expected_obs_dim, dtype=obs.dtype)
                        padded_obs[:obs.shape[0]] = obs
                        obs = padded_obs
                    else:
                        # Truncate if observation is larger than expected
                        obs = obs[:expected_obs_dim]
                
                # Ensure observation has no NaN or Inf values
                if np.any(~np.isfinite(obs)):
                    print(f"Warning: Observation contains NaN or Inf values, replacing with zeros")
                    obs = np.nan_to_num(obs, nan=0.0, posinf=0.0, neginf=0.0)
                
                # Get action from model with error handling
                try:
                    action, _ = model.predict(obs, deterministic=True)
                except ValueError as e:
                    if "invalid values" in str(e) and not nan_encountered:
                        # If we encounter NaN in action prediction once, try a random action instead
                        print(f"Warning: NaN action encountered. Using random action instead.")
                        action = env.action_space.sample()
                        nan_encountered = True
                    else:
                        # If it happens repeatedly, use a safe default action
                        print(f"Error in action prediction: {e}")
                        # Safe fallback action: split region in middle across both dimensions
                        action = np.array([0.5, 0.5, 0.5, 0.5])
                
                # Ensure action has no NaN values
                if np.any(np.isnan(action)):
                    print(f"Warning: Action contains NaN values, using default action")
                    action = np.array([0.5, 0.5, 0.5, 0.5])
                
                # Step the environment
                obs, reward, done, _, info = env.step(action)
                total_reward += reward
                error_history.append(info['error'])
                step_count += 1
                
                # Break loop if too many steps (failsafe)
                if step_count > 100:
                    print(f"Warning: Maximum steps reached, breaking loop")
                    done = True
            
            except Exception as e:
                print(f"Error during evaluation step: {e}")
                # Try to recover and continue
                if not done and step_count < 100:
                    try:
                        # Reset environment and try again
                        obs, _ = env.reset()
                        step_count += 1
                    except:
                        # If reset fails, we have to stop
                        done = True
                else:
                    done = True

        # Store detailed results
        results[func_name] = {
            'true_value': env.true_value,
            'approximation': info['approximation'],
            'error': info['error'],
            'relative_error': info['error']/abs(env.true_value),
            'num_regions': len(env.regions),
            'num_evaluations': len(env.evals),
            'total_reward': total_reward,
            'steps': step_count,
            'error_history': error_history,
            'efficiency': info['efficiency']
        }
        print(f"\nResults for {func_name}:")
        print(f"  True value:      {env.true_value:.10e}")
        print(f"  Approximation:   {info['approximation']:.10e}")
        print(f"  Absolute Error:  {info['error']:.10e}")
        print(f"  Relative Error:  {info['error']/abs(env.true_value):.10e}")
        print(f"  Regions used:    {len(env.regions)}")
        print(f"  Evaluations:     {len(env.evals)}")
        print(f"  Total reward:    {total_reward:.4f}")

    return results

class EarlyStopCallback(BaseCallback):
    """
    Enhanced callback for early stopping with improved monitoring and stopping criteria.
    Tracks both global and local error improvements.
    """
    def __init__(self, check_freq: int = 5000, 
                 min_improvement: float = 1e-6, 
                 min_local_improvement: float = 1e-7,
                 patience: int = 5, 
                 min_episodes: int = 20, 
                 verbose: int = 1):
        super().__init__(verbose)
        self.check_freq = check_freq
        self.min_improvement = min_improvement
        self.min_local_improvement = min_local_improvement
        self.patience = patience
        self.min_episodes = min_episodes
        self.verbose = verbose

        # Initialize tracking variables
        self.best_mean_reward = -float('inf')
        self.best_local_error = float('inf')
        self.no_improvement_count = 0
        self.episode_count = 0
        self.reward_history = []
        self.error_history = []
        self.local_error_history = []
        self.training_start = time.time()

    def _on_step(self) -> bool:
        if self.n_calls % self.check_freq == 0:
            # Get current metrics
            mean_reward = self.model.logger.name_to_value.get('rollout/ep_rew_mean', None)
            ep_count = self.model.logger.name_to_value.get('time/episodes', 0)
            mean_error = self.model.logger.name_to_value.get('rollout/ep_error_mean', float('inf'))
            local_error = self.model.logger.name_to_value.get('rollout/local_error_mean', float('inf'))

            if mean_reward is not None:
                self.reward_history.append(mean_reward)
                self.error_history.append(mean_error)
                self.local_error_history.append(local_error)
                self.episode_count = ep_count

                # Calculate improvements
                reward_improvement = mean_reward - self.best_mean_reward
                local_error_improvement = self.best_local_error - local_error

                # Check for significant improvement in either metrics
                if (reward_improvement > self.min_improvement or 
                    local_error_improvement > self.min_local_improvement):
                    self.best_mean_reward = max(mean_reward, self.best_mean_reward)
                    self.best_local_error = min(local_error, self.best_local_error)
                    self.no_improvement_count = 0
                    if self.verbose > 0:
                        elapsed_time = time.time() - self.training_start
                        print(f"\nImprovement at episode {ep_count} ({elapsed_time:.1f}s):")
                        print(f"  Mean reward:     {mean_reward:.6f}")
                        print(f"  Local error:     {local_error:.6e}")
                else:
                    self.no_improvement_count += 1
                    if self.verbose > 0:
                        print(f"\nNo significant improvement: {self.no_improvement_count}/{self.patience}")
                        print(f"  Current reward:  {mean_reward:.6f}")
                        print(f"  Current local error: {local_error:.6e}")

                # Check stopping conditions
                if self.episode_count < self.min_episodes:
                    return True

                # Stop if no improvement for too long
                if self.no_improvement_count >= self.patience:
                    if self.verbose > 0:
                        print("\nEarly stopping triggered:")
                        print(f"  Episodes:        {self.episode_count}")
                        print(f"  Final reward:    {mean_reward:.6f}")
                        print(f"  Best reward:     {self.best_mean_reward:.6f}")
                        print(f"  Final local error: {local_error:.6e}")
                        print(f"  Training time:   {time.time() - self.training_start:.1f}s")
                    return False

                # Check for performance degradation
                if len(self.reward_history) > 5:
                    recent_reward_mean = np.mean(self.reward_history[-5:])
                    recent_error_mean = np.mean(self.local_error_history[-5:])
                    if (recent_reward_mean < self.best_mean_reward * 0.5 or 
                        recent_error_mean > self.best_local_error * 2.0):
                        if self.verbose > 0:
                            print("\nStopping due to performance degradation:")
                            print(f"  Recent reward mean: {recent_reward_mean:.6f}")
                            print(f"  Recent error mean:  {recent_error_mean:.6e}")
                        return False
        return True

    def get_training_summary(self) -> Dict:
        """Return summary of training progress"""
        return {
            'best_reward': self.best_mean_reward,
            'best_local_error': self.best_local_error,
            'episodes': self.episode_count,
            'training_time': time.time() - self.training_start,
            'reward_history': self.reward_history,
            'error_history': self.error_history,
            'local_error_history': self.local_error_history
        }

class StepLimitCallback(BaseCallback):
    """Callback to force training to stop after exactly 200000 steps"""
    def __init__(self, total_steps: int = None, verbose: int = 0):
        super().__init__(verbose)
        # Hardcode to exactly 200000 steps regardless of passed parameter
        self.total_steps = 200000  # Updated to 200000 steps
        self.step_count = 0
        self.verbose = verbose
        self.start_time = time.time()
        print(f"StepLimitCallback initialized with hardcoded limit of {self.total_steps} steps")

    def _on_step(self) -> bool:
        self.step_count += 1
        # Print progress every 20000 steps (10% of 200000)
        if self.verbose > 0 and self.step_count % 20000 == 0:
            elapsed = time.time() - self.start_time
            print(f"Training progress: {self.step_count}/{self.total_steps} steps ({self.step_count/self.total_steps*100:.1f}%) - {elapsed:.1f}s elapsed")
        
        # Force stop at exactly 200000 steps
        if self.step_count >= self.total_steps:
            if self.verbose > 0:
                print(f"\n>>> STOPPING: Reached exactly {self.total_steps} steps <<<")
                print(f"Total training time: {time.time() - self.start_time:.1f} seconds")
            
            # This will definitely stop the training
            self.training_env.reset()  # Reset environment to avoid potential errors
            return False  # Return False to stop training
            
        return True
    
    def on_training_end(self) -> None:
        print(f"Training ended at exactly {self.step_count} steps")

if __name__ == "__main__":
    # Fix TQDM/rich progress bar conflicts
    import os
    os.environ["RICH_FORCE_TERMINAL"] = "0"
    from tqdm import std
    import sys
    sys.modules["tqdm.auto"] = std
    # Force standard tqdm for entire script
    sys.modules["tqdm"].tqdm = std.tqdm

    # Get all 2D test functions
    all_functions = define_2d_test_functions()

    # Training functions with progressive difficulty
    training_functions = {
        k: all_functions[k] for k in [
            # Start with simpler functions
            "gaussian_2d"
            # Comment out other functions for initial testing
            , "sinc_2d", "polynomial_2d", 
            "wave_packet", "oscillatory_2d", "bessel_2d",
            "peaks_2d", "gaussian_peaks",
             "circular_step", 
            "inverse_r", "log_singularity_2d",
            "oscillating_peaks_2d", "complex_oscillatory",
            "mixed_features", "hybrid_singularity"
        ]
    }

    # Train with longer steps
    model = train_model(
        training_functions,
        training_steps=200000,  # Changed from 200 to 200000
        save_dir="adaptive_integration_2d_models"
    )

    # Test functions for evaluation - Fixed missing comma
    test_functions = {
        k: all_functions[k] for k in [
            "gaussian_2d",         # Added comma here
            "bessel_2d",            
            "gaussian_peaks",       
            "circular_step",         
            "pole_singularity",     
            "complex_oscillatory"    
        ]
    }

    # Evaluate and visualize results
    results = evaluate_and_visualize_2d(
        "adaptive_integration_2d_models/adaptive_integration_final",
        test_functions
    )

    # Print summary of all results
    print("\n" + "="*60)
    print("SUMMARY OF 2D INTEGRATION RESULTS")
    print("="*60)
    avg_rel_error = np.mean([r['relative_error'] for r in results.values()])
    avg_efficiency = np.mean([r['efficiency'] for r in results.values()])
    total_evals = sum(r['num_evaluations'] for r in results.values())

    for func_name, result in results.items():
        print(f"\n{func_name}:")
        print(f"  Relative error: {result['relative_error']:.2e}")
        print(f"  Efficiency:     {result['efficiency']:.2e}")
        print(f"  Regions/Evals:  {result['num_regions']}/{result['num_evaluations']}")

    print("\nAggregate Performance:")
    print(f"Average Relative Error: {avg_rel_error:.2e}")
    print(f"Average Efficiency:     {avg_efficiency:.2e}")
    print(f"Total Evaluations:      {total_evals}")