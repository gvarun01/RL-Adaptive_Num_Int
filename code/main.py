"""
This script serves as the main entry point for running the adaptive numerical
integration reinforcement learning project.

It orchestrates the training of the PPO model using a defined set of 2D test
functions and then evaluates the trained model's performance, including
visualization of the 2D integration process.

To run the project:
1. Ensure all dependencies are installed.
2. Execute this script from the command line: `python -m code.main`
   (assuming you are in the parent directory of `code`).
"""
import os
import sys
import numpy as np

# TQDM and Rich progress bar conflict resolution.
# This setup is included here as it was part of the original __main__ block.
# It's also handled in code.train, so this might be redundant but ensures
# the environment is set up if this script directly invokes tqdm dependent code
# or if other modules imported by main also have conflicting tqdm usage.
os.environ["RICH_FORCE_TERMINAL"] = "0"
os.environ["TQDM_DISABLE"] = "0" # Ensure tqdm is not globally disabled by default

# Attempt to ensure standard tqdm is used if tqdm.auto or tqdm.tqdm were already # monkey-patched by another library.
if "tqdm" not in sys.modules:
    import tqdm # Import tqdm if not already imported.
    # If tqdm.std is the desired standard version.
    from tqdm import std as tqdm_std
    if "tqdm.auto" in sys.modules:
        sys.modules["tqdm.auto"] = tqdm_std
    sys.modules["tqdm"].tqdm = tqdm_std.tqdm
else: # tqdm is already imported
    from tqdm import std as tqdm_std
    if "tqdm.auto" in sys.modules:
         # If tqdm.auto was imported before this script ran, try to repatch it.
        sys.modules["tqdm.auto"] = tqdm_std
    # Attempt to repatch the main tqdm entry point.
    sys.modules["tqdm"].tqdm = tqdm_std.tqdm


from .test_functions import define_2d_test_functions
from .train import train_model # train_model also handles its own tqdm setup.
from .evaluate import evaluate_and_visualize_2d

if __name__ == "__main__":
    # Note: The tqdm setup above is from the original main block.
    # The train_model function in code/train.py also performs a similar tqdm setup.
    # This redundancy is kept to match the original structure but might be simplified
    # if only train_model directly uses and configures tqdm.

    # Get all 2D test functions
    all_functions = define_2d_test_functions()

    # Training functions with progressive difficulty
    # This selection determines the curriculum for training.
    training_functions_keys = [
        "gaussian_2d", "sinc_2d", "polynomial_2d", 
        "wave_packet", "oscillatory_2d", "bessel_2d",
        "peaks_2d", # This was Franke's function in the original, using the new peaks_2d
        "gaussian_peaks", "circular_step", 
        "inverse_r", "log_singularity_2d",
        "oscillating_peaks_2d", "complex_oscillatory",
        "mixed_features", "hybrid_singularity_oscillation" # Changed from hybrid_singularity
    ]
    training_functions = {
        k: all_functions[k] for k in training_functions_keys if k in all_functions
    }
    if len(training_functions) != len(training_functions_keys):
        print("Warning: Some training functions specified were not found in all_functions.")

    # Train the model
    # The train_model function will iterate through the 'training_functions',
    # save intermediate models, and normalization stats.
    trained_model = train_model(
        training_functions,
        training_steps=200000,  # Number of steps per function in the curriculum
        save_dir="adaptive_integration_2d_models" # Directory to save models and stats
    )

    # Define a subset of functions for final evaluation and visualization
    # These functions might be chosen to represent a diverse set of challenges.
    evaluation_function_keys = [
        "gaussian_2d",
        "bessel_2d",            
        "gaussian_peaks",       
        "circular_step",         
        "pole_singularity",     
        "complex_oscillatory"
    ]
    evaluation_functions = {
        k: all_functions[k] for k in evaluation_function_keys if k in all_functions
    }
    if len(evaluation_functions) != len(evaluation_function_keys):
        print("Warning: Some evaluation functions specified were not found in all_functions.")


    # Evaluate and visualize results using the final trained model
    if trained_model is not None:
        print("\n" + "="*60)
        print("STARTING FINAL EVALUATION AND VISUALIZATION")
        print("="*60)
        # The final model saved by train_model is at 'adaptive_integration_final.zip'
        final_model_path = "adaptive_integration_2d_models/adaptive_integration_final"
        
        # For evaluation, we might need to load the VecNormalize stats specific to the
        # *last* environment the model was trained on if not using a fresh VecNormalize.
        # However, evaluate_and_visualize_2d can handle loading VecNormalize stats
        # if a path is provided. If not, it creates a fresh, non-normalized environment.
        # For a general evaluation, it's often better to evaluate on non-normalized envs
        # or ensure normalization is consistently applied.
        # The current evaluate_and_visualize_2d doesn't require vec_normalize_path for the *final* model.
        
        results = evaluate_and_visualize_2d(
            model_path=final_model_path, # Path to the final saved model
            functions=evaluation_functions
            # vec_normalize_path can be specified if needed, e.g., for the last env's stats:
            # vec_normalize_path=f"adaptive_integration_2d_models/vec_normalize_{len(training_functions)-1}_{training_functions_keys[-1]}.pkl"
        )

        # Print summary of all evaluation results
        print("\n" + "="*60)
        print("SUMMARY OF 2D INTEGRATION EVALUATION RESULTS")
        print("="*60)
        
        # Filter out results where 'relative_error' might be None or NaN before calculating mean
        valid_rel_errors = [r['relative_error'] for r in results.values() if r['relative_error'] is not None and np.isfinite(r['relative_error'])]
        avg_rel_error = np.mean(valid_rel_errors) if valid_rel_errors else float('nan')
        
        valid_efficiencies = [r['efficiency'] for r in results.values() if r['efficiency'] is not None and np.isfinite(r['efficiency'])]
        avg_efficiency = np.mean(valid_efficiencies) if valid_efficiencies else float('nan')
        
        total_evals = sum(r['num_evaluations'] for r in results.values() if r['num_evaluations'] is not None)

        for func_name, result_data in results.items():
            print(f"\n{func_name}:")
            rel_err_display = f"{result_data['relative_error']:.2e}" if result_data['relative_error'] is not None and np.isfinite(result_data['relative_error']) else "N/A"
            eff_display = f"{result_data['efficiency']:.2e}" if result_data['efficiency'] is not None and np.isfinite(result_data['efficiency']) else "N/A"
            print(f"  Relative error: {rel_err_display}")
            print(f"  Efficiency:     {eff_display}")
            print(f"  Regions/Evals:  {result_data['num_regions']}/{result_data['num_evaluations']}")

        print("\nAggregate Performance on Evaluation Set:")
        print(f"Average Relative Error: {avg_rel_error:.2e}")
        print(f"Average Efficiency:     {avg_efficiency:.2e}")
        print(f"Total Evaluations:      {total_evals}")
    else:
        print("Model training did not complete successfully. Skipping final evaluation.")

    print("\nScript finished.")
