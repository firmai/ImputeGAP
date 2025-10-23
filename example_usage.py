#!/usr/bin/env python3
"""
Example usage of the simplified MISSNET imputation implementation.

This script demonstrates various ways to use MISSNET for time series imputation.
"""

import numpy as np
import matplotlib.pyplot as plt
from missnet_imputer import missnet_impute, MissNet


def example_basic_usage():
    """Basic example of MISSNET imputation."""
    print("=" * 60)
    print("EXAMPLE 1: Basic Usage")
    print("=" * 60)
    
    # Create synthetic time series data
    np.random.seed(42)
    n_timesteps, n_features = 100, 5
    t = np.linspace(0, 4*np.pi, n_timesteps)
    
    # Create correlated time series with different patterns
    data = np.zeros((n_timesteps, n_features))
    for i in range(n_features):
        if i % 2 == 0:
            data[:, i] = np.sin(t + i*0.5) + 0.1 * np.random.randn(n_timesteps)
        else:
            data[:, i] = np.cos(t + i*0.3) + 0.1 * np.random.randn(n_timesteps)
    
    # Introduce missing values (25% missing)
    mask = np.random.random(data.shape) < 0.25
    data_missing = data.copy()
    data_missing[mask] = np.nan
    
    print(f"Dataset shape: {data.shape}")
    print(f"Missing values: {np.isnan(data_missing).sum()}/{data_missing.size}")
    print(f"Missing percentage: {np.isnan(data_missing).sum()/data_missing.size*100:.1f}%")
    
    # Perform imputation
    imputed_data = missnet_impute(
        data_missing,
        alpha=0.6,
        beta=0.1,
        L=8,
        n_cl=1,
        max_iteration=15,
        verbose=True
    )
    
    # Calculate metrics
    mse = np.mean((data[mask] - imputed_data[mask]) ** 2)
    mae = np.mean(np.abs(data[mask] - imputed_data[mask]))
    
    print(f"\nImputation Quality:")
    print(f"MSE: {mse:.6f}")
    print(f"MAE: {mae:.6f}")
    print(f"All missing values imputed: {not np.any(np.isnan(imputed_data[mask]))}")
    
    return data, data_missing, imputed_data, mask


def example_class_interface():
    """Example using the MissNet class directly."""
    print("\n" + "=" * 60)
    print("EXAMPLE 2: Class Interface")
    print("=" * 60)
    
    # Create sample data
    np.random.seed(123)
    data = np.random.randn(80, 6)
    
    # Add some structure
    for i in range(6):
        data[:, i] += 0.3 * np.sin(np.linspace(0, 2*np.pi, 80) + i)
    
    # Create missing data with block pattern
    data_missing = data.copy()
    data_missing[20:30, 1:3] = np.nan  # Block missing
    data_missing[50:60, 3:5] = np.nan  # Another block
    data_missing[np.random.random(data.shape) < 0.1] = np.nan  # Random missing
    
    print(f"Dataset shape: {data_missing.shape}")
    print(f"Missing values: {np.isnan(data_missing).sum()}/{data_missing.size}")
    
    # Create and fit model
    model = MissNet(alpha=0.5, beta=0.1, L=8, n_cl=2)
    
    print("\nTraining MISSNET model...")
    history = model.fit(data_missing, max_iter=12, verbose=True)
    
    # Generate imputations
    imputed_data = model.imputation()
    
    # Show training history
    print(f"\nTraining History:")
    print(f"Final LLE: {history['lle'][-1]:.3f}")
    print(f"Total training time: {sum(history['time']):.3f} seconds")
    
    # Show model information
    print(f"\nModel Information:")
    print(f"Number of clusters: {model.n_cl}")
    print(f"Hidden dimension: {model.L}")
    print(f"Cluster assignments: {[np.sum(model.F == k) for k in range(model.n_cl)]}")
    
    # Calculate quality metrics
    mask_missing = np.isnan(data_missing)
    mse = np.mean((data[mask_missing] - imputed_data[mask_missing]) ** 2)
    mae = np.mean(np.abs(data[mask_missing] - imputed_data[mask_missing]))
    
    print(f"\nImputation Quality:")
    print(f"MSE: {mse:.6f}")
    print(f"MAE: {mae:.6f}")
    
    return data, data_missing, imputed_data, model


def example_parameter_comparison():
    """Example comparing different parameter settings."""
    print("\n" + "=" * 60)
    print("EXAMPLE 3: Parameter Comparison")
    print("=" * 60)
    
    # Create test data
    np.random.seed(456)
    data = np.random.randn(60, 8)
    
    # Add temporal structure
    t = np.linspace(0, 3*np.pi, 60)
    for i in range(8):
        data[:, i] += 0.4 * np.sin(t + i*0.2)
    
    # Create missing data
    data_missing = data.copy()
    mask = np.random.random(data.shape) < 0.2
    data_missing[mask] = np.nan
    
    print(f"Dataset shape: {data.shape}")
    print(f"Missing values: {mask.sum()}/{data.size}")
    
    # Test different parameter combinations
    parameter_sets = [
        {'alpha': 0.3, 'beta': 0.05, 'L': 5, 'n_cl': 1, 'name': 'Conservative'},
        {'alpha': 0.5, 'beta': 0.1, 'L': 10, 'n_cl': 1, 'name': 'Balanced'},
        {'alpha': 0.7, 'beta': 0.15, 'L': 12, 'n_cl': 2, 'name': 'Complex'},
    ]
    
    results = []
    
    for params in parameter_sets:
        print(f"\nTesting {params['name']} configuration...")
        print(f"  alpha={params['alpha']}, beta={params['beta']}, L={params['L']}, n_cl={params['n_cl']}")
        
        imputed = missnet_impute(
            data_missing,
            alpha=params['alpha'],
            beta=params['beta'],
            L=params['L'],
            n_cl=params['n_cl'],
            max_iteration=10,
            verbose=False
        )
        
        mse = np.mean((data[mask] - imputed[mask]) ** 2)
        mae = np.mean(np.abs(data[mask] - imputed[mask]))
        
        results.append({
            'name': params['name'],
            'mse': mse,
            'mae': mae,
            'params': params
        })
        
        print(f"  MSE: {mse:.6f}, MAE: {mae:.6f}")
    
    # Find best configuration
    best_result = min(results, key=lambda x: x['mse'])
    print(f"\nBest configuration: {best_result['name']}")
    print(f"Best MSE: {best_result['mse']:.6f}")
    print(f"Best MAE: {best_result['mae']:.6f}")
    
    return results


def example_visualization():
    """Example with visualization of imputation results."""
    print("\n" + "=" * 60)
    print("EXAMPLE 4: Visualization")
    print("=" * 60)
    
    try:
        import matplotlib.pyplot as plt
        
        # Create synthetic data with clear patterns
        np.random.seed(789)
        t = np.linspace(0, 6*np.pi, 150)
        
        # Create multiple time series with different patterns
        series1 = np.sin(t) + 0.1 * np.random.randn(150)
        series2 = 0.8 * np.cos(2*t) + 0.1 * np.random.randn(150)
        series3 = 0.5 * np.sin(3*t + np.pi/4) + 0.1 * np.random.randn(150)
        
        data = np.column_stack([series1, series2, series3])
        
        # Create missing data with different patterns
        data_missing = data.copy()
        
        # Block missing in first series
        data_missing[30:50, 0] = np.nan
        
        # Random missing in second series
        data_missing[np.random.random(150) < 0.15, 1] = np.nan
        
        # Block missing in third series
        data_missing[80:100, 2] = np.nan
        
        print(f"Dataset shape: {data.shape}")
        print(f"Missing values: {np.isnan(data_missing).sum()}/{data_missing.size}")
        
        # Perform imputation
        imputed_data = missnet_impute(
            data_missing,
            alpha=0.6,
            beta=0.08,
            L=10,
            n_cl=1,
            max_iteration=15,
            verbose=False
        )
        
        # Create visualization
        fig, axes = plt.subplots(3, 1, figsize=(12, 10))
        series_names = ['Series 1 (Sinusoidal)', 'Series 2 (Cosine)', 'Series 3 (Complex)']
        
        for i, ax in enumerate(axes):
            # Plot original data
            ax.plot(t, data[:, i], 'b-', label='Original', linewidth=2, alpha=0.7)
            
            # Plot imputed data
            ax.plot(t, imputed_data[:, i], 'r--', label='Imputed', linewidth=2, alpha=0.8)
            
            # Highlight missing points
            missing_mask = np.isnan(data_missing[:, i])
            ax.scatter(t[missing_mask], data[missing_mask, i], 
                      color='red', s=30, zorder=5, label='Missing Values')
            
            ax.set_title(series_names[i])
            ax.set_xlabel('Time')
            ax.set_ylabel('Value')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('missnet_imputation_example.png', dpi=300, bbox_inches='tight')
        print("Visualization saved as 'missnet_imputation_example.png'")
        
        # Calculate overall metrics
        mask_missing = np.isnan(data_missing)
        mse = np.mean((data[mask_missing] - imputed_data[mask_missing]) ** 2)
        mae = np.mean(np.abs(data[mask_missing] - imputed_data[mask_missing]))
        
        print(f"\nOverall Imputation Quality:")
        print(f"MSE: {mse:.6f}")
        print(f"MAE: {mae:.6f}")
        
        return data, data_missing, imputed_data
        
    except ImportError:
        print("Matplotlib not available. Skipping visualization example.")
        return None


def main():
    """Run all examples."""
    print("MISSNET Imputation Examples")
    print("=" * 60)
    print("This script demonstrates various usage patterns of the MISSNET imputation algorithm.")
    
    # Run examples
    example_basic_usage()
    example_class_interface()
    example_parameter_comparison()
    example_visualization()
    
    print("\n" + "=" * 60)
    print("All examples completed successfully!")
    print("=" * 60)
    
    print("\nKey Takeaways:")
    print("1. MISSNET can handle various missing data patterns effectively")
    print("2. Parameters can be tuned based on data characteristics")
    print("3. The algorithm works well with both random and block missing patterns")
    print("4. Multiple clusters can capture non-stationary patterns in time series")
    print("5. The class interface provides more control and access to learned parameters")


if __name__ == "__main__":
    main()
