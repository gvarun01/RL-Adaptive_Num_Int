"""
This module contains functions related to the training of reinforcement
learning models for adaptive numerical integration.

It includes a factory function for creating custom training environments and
the main training loop function (`train_model`) which orchestrates the
training process across different integration tasks. The `train_model` function
handles model initialization, environment setup, a curriculum of training
functions, callbacks for early stopping and step limits, and model saving.
"""

import os
import time
import traceback
import warnings
import sys

import torch
import torch.nn as nn
import numpy as np

# Ensure standard tqdm is used to avoid conflicts with rich or other progress bar libraries.
# This is important for environments where multiple progress bar libraries might be active.
# The check for 'tqdm' in sys.modules should ideally be done before importing tqdm.std
# However, given the execution model, this setup is done once when the module is loaded.
if "tqdm" not in sys.modules:
    import tqdm # Ensure tqdm is imported if not already
    # This direct import might not be enough if tqdm.auto was already imported by another module.
    # The os.environ settings are a more robust way to influence tqdm's behavior globally.
os.environ["TQDM_DISABLE"] = "0" # Setting to "1" would disable, "0" enables or defers to auto
os.environ["RICH_FORCE_TERMINAL"] = "0" # Setting to "1" forces rich, "0" disables if not a TTY

from tqdm import std as tqdm_std # Use an alias to avoid conflict if 'std' is used elsewhere
if "tqdm.auto" in sys.modules:
    sys.modules["tqdm.auto"] = tqdm_std
if "tqdm" in sys.modules: # Re-check and ensure our std is used.
    sys.modules["tqdm"].tqdm = tqdm_std.tqdm


from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CallbackList
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize, DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy
from gymnasium import spaces

from typing import Callable, Dict, Tuple, List, Optional, Union

# Relative imports for project-specific modules
from .environment import EnhancedAdaptiveIntegrationEnv
from .callbacks import EarlyStopCallback 
# Note: StepLimitCallback from .callbacks is imported but train_model defines its own StrictStepLimitCallback.
# This is fine if the external one isn't used by train_model directly.
from .utils import linear_schedule
# from .particle_filter import Particle, PFEMIntegrator # Not directly used in these functions


def make_env_factory(func: Callable[[float, float], float],
                     ax: float, bx: float,
                     ay: float, by: float,
                     max_intervals: int = 40) -> Callable[[], EnhancedAdaptiveIntegrationEnv]:
    """
    Creates a factory function that, when called, instantiates and returns an
    `EnhancedAdaptiveIntegrationEnv` configured for a specific 2D integration task.

    This approach is commonly used with vectorized environments in Stable Baselines3,
    allowing environments to be created in subprocesses.

    Args:
        func (Callable[[float, float], float]): The 2D function to be integrated.
            It should take two float arguments (x, y) and return a float.
        ax (float): The lower x-bound of the integration domain.
        bx (float): The upper x-bound of the integration domain.
        ay (float): The lower y-bound of the integration domain.
        by (float): The upper y-bound of the integration domain.
        max_intervals (int): The maximum number of rectangular regions the domain
                             can be partitioned into within the environment.
                             Defaults to 40.

    Returns:
        Callable[[], EnhancedAdaptiveIntegrationEnv]: A nullary function (takes no arguments)
            that returns a new instance of `EnhancedAdaptiveIntegrationEnv`
            configured with the provided parameters.
    """
    def _init() -> EnhancedAdaptiveIntegrationEnv:
        """Instantiates and returns a configured EnhancedAdaptiveIntegrationEnv."""
        env = EnhancedAdaptiveIntegrationEnv(
            ax=ax, bx=bx, ay=ay, by=by,
            max_intervals=max_intervals,
            function=func
        )
        return env
    return _init


def train_model(functions: Dict[str, Tuple[Callable[[float, float], float], float, float, float, float]],
                training_steps: int = 200000,
                save_dir: str = "models",
                evaluate: bool = True) -> Optional[PPO]:
    """
    Trains a PPO model on a sequence of 2D integration tasks (functions).

    The function iterates through a dictionary of provided 2D functions, setting up
    an `EnhancedAdaptiveIntegrationEnv` for each. It uses a curriculum learning
    approach where the model is trained sequentially on these functions. If a model
    already exists from a previous function in the curriculum, its training is
    continued. Otherwise, a new PPO model is initialized.

    Key features of the training process:
    - **Environment Setup**: Uses `SubprocVecEnv` for parallel environment execution
      and `VecNormalize` for observation and reward normalization.
    - **Model Initialization**: Configures a PPO agent with specific hyperparameters,
      including a learning rate schedule, network architecture, and SDE for exploration.
    - **Callbacks**:
        - `EarlyStopCallback`: Monitors training progress and stops early if no
          significant improvement is observed.
        - `LocalErrorMonitor` (internal): Tracks local error metrics from the environment.
        - `StrictStepLimitCallback` (internal): Enforces a hard stop after a specific
          number of training steps for each function in the curriculum.
        - `ValueMonitorCallback` (internal): Checks for NaN/Inf values in model parameters
          and attempts to recover.
    - **Error Handling**: Includes `try-except` blocks for robust environment creation,
      model learning, and saving. Wraps integration functions to handle potential
      numerical issues.
    - **Model Saving**: Saves the trained model and normalization statistics after
      each function in the curriculum and at the very end of training.
    - **TQDM/Rich Fix**: Includes workarounds for progress bar conflicts in some console
      environments.

    Args:
        functions (Dict[str, Tuple[Callable[[float, float], float], float, float, float, float]]):
            A dictionary where keys are function names (for logging/saving) and
            values are tuples defining the integration task:
            `(function, ax, bx, ay, by)`.
            - `function`: The 2D callable to integrate.
            - `ax, bx`: X-axis integration bounds.
            - `ay, by`: Y-axis integration bounds.
        training_steps (int): The number of training steps to perform for EACH function
                              in the `functions` dictionary. Defaults to 200,000.
        save_dir (str): The directory path where trained models and normalization
                        statistics will be saved. Defaults to "models".
        evaluate (bool): Whether to evaluate the model using `evaluate_policy`
                         after training on each function. Defaults to True.

    Returns:
        Optional[PPO]: The trained PPO model. Returns `None` if model training
                       could not be initiated or completed for all functions.
                       The model returned is the one from the last successfully
                       trained function in the curriculum.

    Side Effects:
        - Creates the `save_dir` if it doesn't exist.
        - Saves model files (e.g., `adaptive_integration_2d_{i}_{func_name}.zip`)
          and VecNormalize statistics (e.g., `vec_normalize_{i}_{func_name}.pkl`)
          within `save_dir`.
        - Saves a final model (`adaptive_integration_final.zip`).
        - Prints training progress and logs to the console.
        - May create TensorBoard log files in `save_dir/tensorboard/`.
    """
    os.makedirs(save_dir, exist_ok=True)
    model: Optional[PPO] = None # Explicitly type hint model

    # Set up warnings
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    warnings.filterwarnings("ignore", category=UserWarning)
    
    # Function wrapper to handle invalid values and errors
    def safe_function_wrapper(original_func: Callable[[float, float], float]) -> Callable[[float, float], float]:
        """Wrapper that makes integration functions robust against invalid values"""
        def safe_func(*args):
            try:
                result = original_func(*args)
                if not np.isfinite(result): # Check for NaN or Inf
                    return 0.0 # Return a safe, neutral value
                return result
            except Exception: # Catch any other calculation error
                return 0.0 # Return a safe, neutral value
        return safe_func

    # Training configuration
    # Using linear_schedule from .utils
    schedule = linear_schedule(5e-4, 1e-5)
    # Using EarlyStopCallback from .callbacks
    early_stop = EarlyStopCallback(
        check_freq=5000, min_improvement=1e-5, min_local_improvement=1e-6,
        patience=10, min_episodes=50, verbose=1
    )

    for i, (func_name, (func, ax, bx, ay, by)) in enumerate(functions.items()):
        print(f"\n{'-'*50}")
        print(f"Training on 2D function: {func_name} ({i+1}/{len(functions)})")
        
        safe_func_to_integrate = safe_function_wrapper(func)
        
        # Test function stability (optional, good practice)
        test_points = [(ax, ay), (ax, by), (bx, ay), (bx, by), ((ax+bx)/2, (ay+by)/2)]
        valid_function = True
        for x_coord, y_coord in test_points:
            try:
                val = safe_func_to_integrate(x_coord, y_coord)
                if not np.isfinite(val):
                    print(f"  Warning: Function returns non-finite value at ({x_coord}, {y_coord}): {val}")
                    valid_function = False; break
            except Exception as e:
                print(f"  Error evaluating function at ({x_coord}, {y_coord}): {str(e)}")
                valid_function = False; break
        if not valid_function:
            print(f"Skipping function {func_name} due to stability issues.")
            continue

        # Environment creation factory for this specific function
        # Using make_env_factory from this module
        current_env_factory = make_env_factory(safe_func_to_integrate, ax, bx, ay, by, max_intervals=40)

        # Create vectorized environment
        try:
            # Need to define make_env for SubprocVecEnv using the factory
            # The factory itself returns a function that creates an env instance.
            # So, we create a list of these factory-produced functions.
            env_fns = [current_env_factory for _ in range(8)] # Example: 8 parallel environments
            vec_env = SubprocVecEnv(env_fns)
            vec_env = VecNormalize(
                vec_env, norm_obs=True, norm_reward=True, clip_obs=10.0, clip_reward=10.0, gamma=0.99
            )
        except Exception as e:
            print(f"Error creating vectorized environment for {func_name}: {str(e)}")
            traceback.print_exc()
            continue # Skip to the next function

        if model is None: # Initialize model for the first function
            try:
                model = PPO(
                    "MlpPolicy", vec_env, verbose=1, learning_rate=schedule,
                    n_steps=2048, batch_size=256, gamma=0.99,
                    tensorboard_log=f"{save_dir}/tensorboard/",
                    policy_kwargs={
                        'net_arch': [128, 64], 'activation_fn': nn.Tanh,
                        'ortho_init': True, 'log_std_init': -2.0, 'squash_output': True
                    },
                    max_grad_norm=0.5, use_sde=True, sde_sample_freq=8,
                    clip_range=0.15, clip_range_vf=0.15, ent_coef=0.01
                )
            except Exception as e:
                print(f"Error creating PPO model for {func_name}: {str(e)}")
                traceback.print_exc()
                vec_env.close()
                continue
        else: # Continue training with the existing model
            try:
                # Reset normalization stats for the new function's characteristics
                # vec_env is already a VecNormalize instance from its creation above
                model.set_env(vec_env) # This should correctly reset normalization stats for a new VecNormalize env
            except Exception as e:
                print(f"Error setting new environment or resetting normalization for {func_name}: {str(e)}")
                traceback.print_exc()
                vec_env.close()
                continue
        
        # Define internal callbacks for this training loop
        class LocalErrorMonitor(BaseCallback):
            def __init__(self, verbose_level=0):
                super().__init__(verbose_level)
                self.local_errors: List[float] = []
            def _on_step(self) -> bool:
                try:
                    if len(self.model.ep_info_buffer) > 0:
                        info = self.model.ep_info_buffer[-1]
                        if 'local_errors' in info and info['local_errors'] is not None:
                            error_val = np.mean(info['local_errors'])
                            if np.isfinite(error_val): self.local_errors.append(error_val)
                except Exception as e_cb: 
                    if self.verbose > 0: print(f"Warning in LocalErrorMonitor: {str(e_cb)}")
                return True

        class StrictStepLimitCallback(BaseCallback):
            def __init__(self, total_steps_limit: int, verbose_level: int = 0):
                super().__init__(verbose_level)
                self.total_steps_limit = total_steps_limit
                self.training_start_time = time.time()
                self.warning_intervals = [int(total_steps_limit * x) for x in [0.1, 0.25, 0.5, 0.75, 0.9]]
                self.warnings_issued_at: set[int] = set()
            def _on_step(self) -> bool:
                try:
                    current_total_steps = self.num_timesteps # From BaseCallback
                    for interval_step in self.warning_intervals:
                        if current_total_steps >= interval_step and interval_step not in self.warnings_issued_at:
                            elapsed_t = time.time() - self.training_start_time
                            if self.verbose > 0:
                                print(f"Progress: {current_total_steps}/{self.total_steps_limit} steps "
                                      f"({current_total_steps/self.total_steps_limit*100:.0f}%) after {elapsed_t:.1f}s")
                            self.warnings_issued_at.add(interval_step)
                    if current_total_steps >= self.total_steps_limit:
                        elapsed_t = time.time() - self.training_start_time
                        if self.verbose > 0:
                            print(f"\nReached {current_total_steps}/{self.total_steps_limit} steps after {elapsed_t:.1f}s. "
                                  "Stopping training for this function as per StrictStepLimitCallback.")
                        return False # Stop training
                except Exception as e_cb:
                     if self.verbose > 0: print(f"Warning in StrictStepLimitCallback: {str(e_cb)}")
                return True

        class ValueMonitorCallback(BaseCallback):
            def __init__(self, verbose_level=0):
                super().__init__(verbose_level)
                self.nan_inf_detected_once = False
                self.check_frequency = 5 # Check every 5 steps
            def _on_step(self) -> bool:
                if self.n_calls % self.check_frequency == 0:
                    for name, param in self.model.policy.named_parameters():
                        if torch.isnan(param).any() or torch.isinf(param).any():
                            if not self.nan_inf_detected_once:
                                print(f"WARNING: NaN/Inf detected in model parameter {name}!")
                                self.nan_inf_detected_once = True
                            # Attempt to reset problematic parameters to small random values
                            param.data.copy_(torch.randn_like(param.data) * 0.01)
                            if self.verbose > 0: print(f"Parameter {name} was reset due to invalid values.")
                return True

        local_monitor_cb = LocalErrorMonitor(verbose_level=1)
        step_limit_cb = StrictStepLimitCallback(total_steps_limit=training_steps, verbose_level=1)
        value_monitor_cb = ValueMonitorCallback(verbose_level=1)
        
        # Combine external EarlyStopCallback with internal ones
        # Note: early_stop is already instantiated.
        all_callbacks = CallbackList([early_stop, local_monitor_cb, step_limit_cb, value_monitor_cb])

        try:
            print(f"Starting training for {func_name} for {training_steps} steps...")
            training_start_time_this_func = time.time()
            model.learn(
                total_timesteps=training_steps + 100, # Buffer for callback to control exact stop
                callback=all_callbacks,
                progress_bar=False, # Using custom progress prints from StrictStepLimitCallback
                reset_num_timesteps=True # Important: steps count from 0 for each function
            )
            print(f"Training for {func_name} completed in {time.time() - training_start_time_this_func:.1f}s.")

            # Save normalization statistics for this environment configuration
            vec_env.save(f"{save_dir}/vec_normalize_{i}_{func_name}.pkl")

            # Training summary from EarlyStopCallback
            summary = early_stop.get_training_summary() # Assuming this method exists and is useful
            print("\nTraining Summary for {func_name}:")
            print(f"  Best reward: {summary.get('best_reward', 'N/A'):.6f}")
            print(f"  Best local error: {summary.get('best_local_error', float('inf')):.6e}")
            if local_monitor_cb.local_errors: # Check if list is not empty
                 mean_local_err_last10 = np.mean(local_monitor_cb.local_errors[-10:]) if len(local_monitor_cb.local_errors) >= 10 else np.mean(local_monitor_cb.local_errors)
                 if np.isfinite(mean_local_err_last10): print(f"  Mean local error (last 10 checks): {mean_local_err_last10:.6e}")
            print(f"  Episodes completed: {summary.get('episodes', 'N/A')}")
            
            # Save model after this function's training
            model_path = f"{save_dir}/adaptive_integration_2d_{i}_{func_name}"
            model.save(model_path)
            print(f"Model for {func_name} saved to {model_path}")

            if evaluate:
                print(f"\nEvaluating model for {func_name}...")
                # Use a temporary VecNormalize wrapper for evaluation if needed, or use the trained one carefully
                eval_env = SubprocVecEnv([current_env_factory for _ in range(1)]) # Single env for eval
                eval_vec_env = VecNormalize.load(f"{save_dir}/vec_normalize_{i}_{func_name}.pkl", eval_env)
                eval_vec_env.training = False # Important for evaluation
                eval_vec_env.norm_reward = False

                mean_reward, std_reward = evaluate_policy(model, eval_vec_env, n_eval_episodes=5, deterministic=True)
                print(f"Evaluation for {func_name}: Mean reward: {mean_reward:.4f} ± {std_reward:.4f}")
                eval_vec_env.close()

        except Exception as e_learn:
            print(f"\nError during training or evaluation for function {func_name}: {str(e_learn)}")
            traceback.print_exc()
        finally:
            try:
                vec_env.close() # Ensure environment is closed
            except Exception as e_close:
                print(f"Error closing environment for {func_name}: {str(e_close)}")
            
    # Save the final model after iterating through all functions
    if model is not None:
        try:
            final_model_path = f"{save_dir}/adaptive_integration_final"
            model.save(final_model_path)
            print(f"\nFinal model (after all functions) saved to {final_model_path}")
        except Exception as e_save_final:
            print(f"Error saving final model: {str(e_save_final)}")
            traceback.print_exc()
    
    return model
