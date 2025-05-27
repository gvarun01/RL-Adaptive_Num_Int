"""
This module provides custom callback classes for use with Stable Baselines3.

It includes callbacks for tasks such as early stopping based on performance
metrics and enforcing strict step limits during training. These callbacks
can be used to enhance and control the training process of reinforcement
learning agents.
"""
import time
import numpy as np
from typing import Dict
from stable_baselines3.common.callbacks import BaseCallback


class EarlyStopCallback(BaseCallback):
    """
    A custom callback for implementing early stopping criteria in Stable Baselines3.

    This callback monitors the training progress and stops training if the
    performance (e.g., mean reward or a custom error metric) does not improve
    significantly over a specified number of evaluation steps (patience).
    It can track both global reward improvements and local error improvements
    if the environment provides such metrics in the logger.

    Args:
        check_freq (int): How often (in terms of training steps) to check
                          for improvement.
        min_improvement (float): The minimum absolute improvement in the primary
                                 metric (e.g., mean reward) required to reset
                                 the patience counter.
        min_local_improvement (float): The minimum absolute improvement in a
                                       secondary, local error metric required
                                       to reset the patience counter. This allows
                                       for continued training if local sub-problems
                                       are still being refined.
        patience (int): The number of checks with no significant improvement
                        after which training will be stopped.
        min_episodes (int): The minimum number of episodes that must be completed
                            before early stopping can be triggered.
        verbose (int): Verbosity level: 0 for no output, 1 for info messages.
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
        """
        This method is called by the Trainer object after each training step.
        It checks if the early stopping conditions are met.

        Returns:
            bool: False if training should be stopped, True otherwise.
        """
        if self.n_calls % self.check_freq == 0:
            # Get current metrics from the logger
            mean_reward = self.model.logger.name_to_value.get('rollout/ep_rew_mean', None)
            ep_count = self.model.logger.name_to_value.get('time/episodes', 0)
            # Assuming 'rollout/ep_error_mean' and 'rollout/local_error_mean' might be logged
            mean_error = self.model.logger.name_to_value.get('rollout/ep_error_mean', float('inf'))
            local_error = self.model.logger.name_to_value.get('rollout/local_error_mean', float('inf'))

            if mean_reward is not None:
                self.reward_history.append(mean_reward)
                self.error_history.append(mean_error)
                self.local_error_history.append(local_error)
                self.episode_count = ep_count

                reward_improvement = mean_reward - self.best_mean_reward
                local_error_improvement = self.best_local_error - local_error # Lower error is better

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

                if self.episode_count < self.min_episodes:
                    return True # Continue training if min_episodes not reached

                if self.no_improvement_count >= self.patience:
                    if self.verbose > 0:
                        print("\nEarly stopping triggered:")
                        print(f"  Episodes:        {self.episode_count}")
                        print(f"  Final reward:    {mean_reward:.6f}")
                        print(f"  Best reward:     {self.best_mean_reward:.6f}")
                        print(f"  Final local error: {local_error:.6e}")
                        print(f"  Training time:   {time.time() - self.training_start:.1f}s")
                    return False # Stop training

                # Optional: Check for performance degradation
                if len(self.reward_history) > 5: # Check after at least 5 records
                    recent_reward_mean = np.mean(self.reward_history[-5:])
                    recent_error_mean = np.mean(self.local_error_history[-5:])
                    # Stop if reward significantly drops or error significantly increases
                    if (recent_reward_mean < self.best_mean_reward * 0.5 or # Reward dropped by 50%
                        (self.best_local_error != float('inf') and recent_error_mean > self.best_local_error * 2.0)): # Error doubled
                        if self.verbose > 0:
                            print("\nStopping due to performance degradation:")
                            print(f"  Recent reward mean: {recent_reward_mean:.6f}")
                            print(f"  Recent error mean:  {recent_error_mean:.6e}")
                        return False
        return True

    def get_training_summary(self) -> Dict[str, any]:
        """
        Returns a summary of the training progress monitored by this callback.

        Returns:
            Dict[str, any]: A dictionary containing training metrics such
                            as best reward, best local error, episode count,
                            training time, and history of rewards and errors.
        """
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
    """
    A custom callback to enforce a strict total number of training steps.

    This callback stops the training process once a predefined number of
    total steps (`num_timesteps` in Stable Baselines3) has been reached.
    It is useful for ensuring that training does not exceed a specific
    computational budget or for experiments requiring a fixed number of
    interactions with the environment.

    The `total_steps` parameter is hardcoded to 200,000 steps within this
    implementation, as per the specific requirements of the task.

    Args:
        total_steps (int, optional): The target total number of steps.
                                     Note: This parameter is currently ignored,
                                     and the limit is hardcoded to 200,000.
        verbose (int): Verbosity level: 0 for no output, 1 for info messages,
                       including progress updates.
    """
    def __init__(self, total_steps: int = None, verbose: int = 0):
        super().__init__(verbose)
        # Hardcode to exactly 200000 steps regardless of passed parameter
        self.total_steps = 200000
        self.step_count = 0 # Tracks steps within this callback's lifecycle for this training run
        self.verbose = verbose
        self.start_time = time.time()
        if self.verbose > 0:
            print(f"StepLimitCallback initialized with hardcoded limit of {self.total_steps} steps.")

    def _on_step(self) -> bool:
        """
        This method is called by the Trainer object after each training step.
        It checks if the total step limit has been reached.
        The `self.num_timesteps` attribute from `BaseCallback` tracks the
        total number of steps taken across the entire training process,
        even if `learn()` is called multiple times.

        Returns:
            bool: False if training should be stopped, True otherwise.
        """
        # self.num_timesteps is the total number of steps taken so far in training
        current_total_steps = self.num_timesteps

        # Print progress every 10% of total_steps
        if self.verbose > 0 and current_total_steps % (self.total_steps // 10) == 0 and current_total_steps > 0:
            elapsed = time.time() - self.start_time
            print(f"Training progress: {current_total_steps}/{self.total_steps} steps ({current_total_steps/self.total_steps*100:.1f}%) - {elapsed:.1f}s elapsed")

        # Force stop if current_total_steps reaches or exceeds the hardcoded limit
        if current_total_steps >= self.total_steps:
            if self.verbose > 0:
                print(f"\n>>> STOPPING: Reached {current_total_steps} steps (limit: {self.total_steps}) <<<")
                print(f"Total training time for this run: {time.time() - self.start_time:.1f} seconds")
            return False  # Return False to stop training
        return True

    def _on_training_end(self) -> None:
        """
        This method is called by the Trainer object when the training ends.
        """
        if self.verbose > 0:
            print(f"Training ended. Total steps reached: {self.num_timesteps}.")
            print(f"StepLimitCallback recorded {self.num_timesteps} total steps for this training session.")
