"""
This module provides utility functions for the project.

It includes functions for common tasks such as learning rate scheduling,
data preprocessing, and other helper functionalities that are used across
different parts of the project.
"""
from typing import Callable


def linear_schedule(initial_value: float, final_value: float) -> Callable[[float], float]:
    """
    Creates a linear learning rate schedule.

    This function returns another function that takes the remaining progress
    (a float from 1.0 at the beginning of training to 0.0 at the end)
    and returns the current learning rate. The learning rate will
    linearly interpolate between `initial_value` and `final_value`
    as the progress decreases.

    Args:
        initial_value (float): The learning rate at the beginning of training
                               (when progress_remaining is 1.0).
        final_value (float): The learning rate at the end of training
                             (when progress_remaining is 0.0).

    Returns:
        Callable[[float], float]: A function that takes the remaining progress
                                 (float) as input and returns the current
                                 learning rate (float).
    """
    def func(progress_remaining: float) -> float:
        """
        Calculates the current learning rate based on the remaining progress.

        Args:
            progress_remaining (float): The fraction of training progress
                                        remaining, from 1.0 (start) to 0.0 (end).

        Returns:
            float: The current learning rate.
        """
        return final_value + progress_remaining * (initial_value - final_value)
    return func
