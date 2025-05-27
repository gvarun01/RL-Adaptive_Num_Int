"""
This module provides functions for evaluating the performance of trained
reinforcement learning models on numerical integration tasks and visualizing
the results.

It includes functions to load a trained model, run it on specified test
functions (1D or 2D), print out performance metrics (e.g., error,
approximation, number of evaluations), and potentially generate plots
to visualize the integration process and outcomes.
"""
import numpy as np
import matplotlib.pyplot as plt # Should be imported if visualize_solution is called from here
from typing import Dict, Callable, Tuple, List, Any, Optional # Added Optional

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecNormalize, DummyVecEnv

# Relative import for the custom environment
from .environment import EnhancedAdaptiveIntegrationEnv


def evaluate_and_visualize(model_path: str,
                           functions: Dict[str, Tuple[Callable[[float, float], float], float, float, float, float]],
                           max_intervals: int = 20) -> Dict[str, Dict[str, Any]]:
    """
    Evaluates a trained PPO model on a set of 2D integration tasks and prints results.

    This function is designed for models trained with `EnhancedAdaptiveIntegrationEnv`.
    It iterates through a dictionary of test functions, creates an environment for each,
    runs the model to perform the integration, and prints key performance metrics.
    The original name suggests 1D, but the `functions` type hint and `env` instantiation
    are for 2D functions. Assuming this is intended for 2D evaluation without visualization.

    Args:
        model_path (str): Path to the saved PPO model file.
        functions (Dict[str, Tuple[Callable[[float, float], float], float, float, float, float]]):
            A dictionary where keys are function names and values are tuples defining
            the 2D integration task: `(function, ax, bx, ay, by)`.
        max_intervals (int): Maximum number of intervals/regions the agent can create.
                             Defaults to 20.

    Returns:
        Dict[str, Dict[str, Any]]: A dictionary where keys are function names.
            Each value is another dictionary containing performance metrics for that
            function, such as 'true_value', 'approximation', 'error',
            'num_intervals' (Note: 'num_intervals' was 'len(env.intervals)' which might be different from 'len(env.regions)'),
            and 'num_evaluations'.
    """
    model = PPO.load(model_path)
    results: Dict[str, Dict[str, Any]] = {}

    for func_name, (func, ax, bx, ay, by) in functions.items():
        print(f"\nEvaluating on {func_name} (2D function)...")

        # Create environment using EnhancedAdaptiveIntegrationEnv
        env = EnhancedAdaptiveIntegrationEnv(
            ax=ax, bx=bx, ay=ay, by=by,
            max_intervals=max_intervals,
            function=func
        )

        obs, _ = env.reset()
        done = False
        total_reward = 0.0 # Initialize total_reward
        info: Dict[str, Any] = {} # Initialize info to ensure it's available

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            done = terminated or truncated # Update done based on gym's new API

        # Ensure info is populated from the last step, or handle if episode ends prematurely
        # The 'approximation' and 'error' should ideally be calculated based on the final state.
        # The environment's step method should provide these in `info`.
        
        # If env.intervals was a typo and it should be env.regions:
        num_regions_final = len(env.regions)

        results[func_name] = {
            'true_value': env.true_value,
            'approximation': info.get('approximation', None), # Use .get for safety
            'error': info.get('error', None),
            'num_regions': num_regions_final, # Corrected from env.intervals
            'num_evaluations': len(env.evals),
            'total_reward': total_reward
        }
        print(f"  True value:     {env.true_value:.8f}")
        print(f"  Approximation:  {info.get('approximation', 'N/A')}")
        print(f"  Error:          {info.get('error', 'N/A'):.8e}")
        print(f"  Regions used:   {num_regions_final}")
        print(f"  Evaluations:    {len(env.evals)}")
        print(f"  Total reward:   {total_reward:.4f}")

        # Note: This function does not call env.visualize_solution()
        # If visualization is needed, evaluate_and_visualize_2d is more appropriate.

    return results


def evaluate_and_visualize_2d(model_path: str,
                              functions: Dict[str, Tuple[Callable[[float, float], float], float, float, float, float]],
                              vec_normalize_path: Optional[str] = None,
                              max_intervals: int = 30) -> Dict[str, Dict[str, Any]]:
    """
    Evaluates a trained PPO model on 2D integration tasks, prints results,
    and calls the environment's `visualize_solution` method for each task.

    This function loads a PPO model, and for each provided 2D test function,
    it sets up the `EnhancedAdaptiveIntegrationEnv`. It can optionally wrap the
    environment with `VecNormalize` if statistics were saved during training.
    The model then performs the integration task, and detailed performance
    metrics are collected and printed. Finally, `env.visualize_solution()` is called
    to generate plots.

    Args:
        model_path (str): Path to the saved PPO model file.
        functions (Dict[str, Tuple[Callable[[float, float], float], float, float, float, float]]):
            A dictionary of 2D test functions. Keys are function names, values are
            tuples: `(function, ax, bx, ay, by)`.
        vec_normalize_path (Optional[str]): Path to saved `VecNormalize` statistics.
                                            If provided, the environment will be wrapped.
                                            Defaults to None.
        max_intervals (int): Maximum number of intervals/regions for evaluation.
                             Defaults to 30.

    Returns:
        Dict[str, Dict[str, Any]]: A dictionary where keys are function names.
            Each value is a dictionary of detailed performance metrics, including
            'true_value', 'approximation', 'error', 'relative_error', 'num_regions',
            'num_evaluations', 'total_reward', 'steps', 'error_history', and 'efficiency'.
    """
    model = PPO.load(model_path)
    results: Dict[str, Dict[str, Any]] = {}

    expected_obs_dim = model.policy.observation_space.shape[0]
    print(f"Model expects observation shape: {expected_obs_dim}")

    for func_name, (func, ax, bx, ay, by) in functions.items():
        print(f"\nEvaluating and Visualizing {func_name}...")

        is_circular_step = func_name == "circular_step" # Example of special handling

        def safe_func(x: float, y: float) -> float:
            try:
                res = func(x, y)
                return 0.0 if not np.isfinite(res) else res
            except Exception:
                return 0.0

        env_instance = EnhancedAdaptiveIntegrationEnv(
            ax=ax, bx=bx, ay=ay, by=by,
            max_intervals=max_intervals,
            function=safe_func if is_circular_step else func
        )

        # Correctly type the env variable that will be used.
        # It could be EnhancedAdaptiveIntegrationEnv or VecNormalize.
        env: Union[EnhancedAdaptiveIntegrationEnv, VecNormalize]
        if vec_normalize_path:
            print(f"Loading VecNormalize stats from: {vec_normalize_path}")
            # Create a DummyVecEnv first as VecNormalize expects a VecEnv
            dummy_env = DummyVecEnv([lambda: env_instance])
            env = VecNormalize.load(vec_normalize_path, dummy_env)
            env.training = False
            env.norm_reward = False
        else:
            env = env_instance # Use the direct instance if no normalization path

        obs_tuple = env.reset() # VecEnv returns a tuple for reset, Env returns ndarray, info
        obs = obs_tuple[0] if isinstance(obs_tuple, tuple) else obs_tuple # Handle both cases for obs
        
        print(f"Initial observation shape from env: {obs.shape}")

        done = False
        total_reward = 0.0
        step_count = 0
        error_history: List[float] = []
        nan_action_encountered = False
        # Initialize info dict to ensure it's defined before the loop for the results
        # The actual useful info will come from the last step of the loop
        last_info: Dict[str, Any] = { 
            'approximation': None, 'error': None, 'efficiency': 0.0
        }


        while not done:
            try:
                # Ensure obs is correctly shaped for the model
                if obs.shape[0] != expected_obs_dim:
                    if obs.shape[0] < expected_obs_dim:
                        padded_obs = np.zeros(expected_obs_dim, dtype=obs.dtype)
                        padded_obs[:obs.shape[0]] = obs.flatten() # Ensure obs is 1D
                        obs = padded_obs
                    else:
                        obs = obs.flatten()[:expected_obs_dim]
                    print(f"Adjusted observation shape to: {obs.shape}")

                if np.any(~np.isfinite(obs)):
                    print("Warning: Observation contains NaN/Inf. Replacing with zeros.")
                    obs = np.nan_to_num(obs, nan=0.0, posinf=0.0, neginf=0.0)
                
                action, _ = model.predict(obs, deterministic=True)
                
                if np.any(np.isnan(action)):
                    print("Warning: Action contains NaN. Using a default safe action.")
                    action = env_instance.action_space.sample() # Get a sample from the original env action space

                obs_step, reward_step, terminated_step, truncated_step, info_step = env.step(action)
                obs = obs_step
                total_reward += reward_step
                if info_step and 'error' in info_step: # Assuming info_step is a dict or list of dicts
                    current_info = info_step[0] if isinstance(info_step, list) else info_step
                    error_history.append(current_info['error'])
                    last_info = current_info # Update last_info
                
                done = terminated_step or truncated_step
                step_count += 1
                if step_count > max_intervals + 10: # Failsafe break
                    print("Warning: Exceeded max evaluation steps. Breaking loop.")
                    break
            except Exception as e_step:
                print(f"Error during evaluation step for {func_name}: {e_step}")
                traceback.print_exc()
                done = True # Stop evaluation for this function on error

        # Prepare results dict
        final_num_regions = len(env_instance.regions) # From the underlying env instance
        final_num_evals = len(env_instance.evals)   # From the underlying env instance
        true_val = env_instance.true_value          # From the underlying env instance

        results[func_name] = {
            'true_value': true_val,
            'approximation': last_info['approximation'],
            'error': last_info['error'],
            'relative_error': abs(last_info['error'] / true_val) if true_val != 0 and last_info['error'] is not None else None,
            'num_regions': final_num_regions,
            'num_evaluations': final_num_evals,
            'total_reward': total_reward,
            'steps': step_count,
            'error_history': error_history,
            'efficiency': last_info['efficiency']
        }
        print(f"\nResults for {func_name}:")
        print(f"  True value:      {true_val:.10e}")
        print(f"  Approximation:   {last_info['approximation'] if last_info['approximation'] is not None else 'N/A'}")
        if last_info['error'] is not None:
             print(f"  Absolute Error:  {last_info['error']:.10e}")
             if true_val !=0 : print(f"  Relative Error:  {abs(last_info['error']/true_val):.10e}")
        else:
             print(f"  Absolute Error:  N/A")
        print(f"  Regions used:    {final_num_regions}")
        print(f"  Evaluations:     {final_num_evals}")
        print(f"  Total reward:    {total_reward:.4f}")

        # Call visualization from the original environment instance
        if hasattr(env_instance, 'visualize_solution') and callable(getattr(env_instance, 'visualize_solution')):
            print(f"Generating visualization for {func_name}...")
            env_instance.visualize_solution()
        else:
            print("Visualization method not found on the environment instance.")

        if isinstance(env, VecNormalize): # Close VecNormalize wrapper if used
            env.close()
        # No explicit close for DummyVecEnv if not wrapped, env_instance does not have close()

    return results
