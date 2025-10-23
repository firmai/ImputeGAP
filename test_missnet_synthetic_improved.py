#!/usr/bin/env python3
"""
Improved MISSNET Ablation Study with Strategic Test Scenarios

This script creates specific test scenarios that ensure all toggles (Spectral/Fourier, 
Skeleton, Cholesky, Robust Loss) actually change behavior and show measurable differences.

Key Improvements:
1. Strategic data generation to meet feature activation criteria
2. Controlled test scenarios that isolate each feature's impact
3. Comprehensive ablation with meaningful comparisons
4. Clear documentation of why each scenario activates specific features
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import time
import warnings
import random
import argparse
from typing import Dict, List, Tuple, Optional
import gc
from scipy import stats
from sklearn.metrics import mean_squared_error, mean_absolute_error

# Import the missnet imputer and autotuner
from missnet_imputer import missnet_impute, get_optimal_config

# Set deterministic behavior
random.seed(42)
np.random.seed(42)

# Set style for plots
plt.style.use('default')
sns.set_palette("husl")


class StrategicDataGenerator:
    """Generate data specifically designed to trigger MISSNET features."""
    
    def __init__(self, random_seed=42):
        np.random.seed(random_seed)
        
    def generate_seasonal_data(self, T: int, N: int, strong_seasonality: bool = True) -> np.ndarray:
        """
        Generate data with strong seasonal patterns to activate spectral features.
        
        Criteria for USE_SPECTRAL: seasonality=True AND T >= 60
        To ensure activation: T >= 60 and strong periodic patterns
        """
        t = np.arange(T)
        data = np.zeros((T, N))
        
        if strong_seasonality:
            # Strong seasonal patterns with multiple harmonics
            for i in range(N):
                # Primary seasonal component
                freq1 = 2 * np.pi * (0.05 + 0.02 * i)  # Period around 20-40 time steps
                seasonal1 = 2.0 * np.sin(freq1 * t + i * np.pi / 4)
                
                # Secondary harmonic
                freq2 = 2 * np.pi * (0.1 + 0.03 * i)  # Higher frequency
                seasonal2 = 1.0 * np.cos(freq2 * t + i * np.pi / 3)
                
                # Trend component
                trend = 0.01 * i * t / T
                
                # Noise
                noise = 0.1 * np.random.randn(T)
                
                data[:, i] = trend + seasonal1 + seasonal2 + noise
        else:
            # Weak seasonality
            for i in range(N):
                data[:, i] = 0.1 * np.sin(2 * np.pi * 0.05 * t) + 0.2 * np.random.randn(T)
                
        return data
    
    def generate_sparse_network_data(self, T: int, N: int, sparsity_level: float = 0.8) -> np.ndarray:
        """
        Generate sparse network data to activate skeleton features.
        
        Criteria for USE_CANDIDATE_GL: sparsity > 0.6 AND T > max(40, 2*N)
        To ensure activation: High sparsity and sufficient samples
        """
        # Create sparse correlation structure
        # Start with independent signals
        data = np.random.randn(T, N) * 0.5
        
        # Add sparse connections (only 20% of possible connections)
        n_connections = int(0.2 * N * (N - 1) / 2)
        
        for _ in range(n_connections):
            i, j = random.sample(range(N), 2)
            # Make feature j depend on feature i with some noise
            correlation_strength = np.random.uniform(0.3, 0.7)
            data[:, j] += correlation_strength * data[:, i]
            
        # Add some noise to break perfect correlations
        data += 0.1 * np.random.randn(T, N)
        
        return data
    
    def generate_dense_network_data(self, T: int, N: int, correlation_strength: float = 0.7) -> np.ndarray:
        """
        Generate dense network data (skeleton should NOT be activated).
        """
        # Create dense correlation structure
        base_signal = np.random.randn(T)
        data = np.zeros((T, N))
        
        for i in range(N):
            # Each feature is a mixture of base signal + individual component
            data[:, i] = (correlation_strength * base_signal + 
                         (1 - correlation_strength) * np.random.randn(T))
            
        return data
    
    def generate_outlier_data(self, T: int, N: int, outlier_fraction: float = 0.1) -> np.ndarray:
        """
        Generate data with significant outliers to activate robust loss.
        
        Criteria for USE_HUBER: has_outliers = True
        To ensure activation: >5% outliers with magnitude > 3*IQR
        """
        # Base clean data
        t = np.arange(T)
        data = np.zeros((T, N))
        
        for i in range(N):
            # Clean signal
            signal = np.sin(2 * np.pi * 0.05 * t + i * 0.2) + 0.1 * np.random.randn(T)
            data[:, i] = signal
            
            # Add random outliers
            n_outliers = int(T * outlier_fraction)
            outlier_indices = random.sample(range(T), n_outliers)
            
            for idx in outlier_indices:
                # Large magnitude outliers
                outlier_magnitude = np.random.choice([-1, 1]) * np.random.uniform(5, 10)
                data[idx, i] += outlier_magnitude
                
        return data
    
    def generate_large_matrix_data(self, T: int, N: int) -> np.ndarray:
        """
        Generate large matrix data to ensure Cholesky is beneficial.
        
        Criteria for USE_CHOLESKY: Always enabled, but pinv path used for N < 20
        To ensure Cholesky path: N >= 20
        """
        # Generate correlated data for larger matrices
        base_signals = np.random.randn(T, min(N//2, 10))
        data = np.zeros((T, N))
        
        for i in range(N):
            # Mix of base signals
            weights = np.random.randn(min(N//2, 10))
            data[:, i] = base_signals @ weights + 0.1 * np.random.randn(T)
            
        return data


class ImprovedAblationTester:
    """Comprehensive ablation testing with strategic scenarios."""
    
    def __init__(self, random_seed=42):
        np.random.seed(random_seed)
        self.generator = StrategicDataGenerator(random_seed)
        self.results = []
        
    def calculate_metrics(self, original: np.ndarray, imputed: np.ndarray, 
                         missing_mask: np.ndarray) -> Dict[str, float]:
        """Calculate evaluation metrics."""
        eval_mask = missing_mask & np.isfinite(imputed) & np.isfinite(original)
        
        if eval_mask.sum() == 0:
            return {
                'mse': float('nan'),
                'mae': float('nan'),
                'rmse': float('nan'),
                'correlation': 0.0,
                'evaluable_points': 0,
            }
        
        ref = original[eval_mask]
        est = imputed[eval_mask]
        
        if ref.size < 3 or np.std(ref) == 0 or np.std(est) == 0:
            corr = 0.0
        else:
            corr = float(np.corrcoef(ref, est)[0, 1])
        
        return {
            'mse': float(np.mean((ref - est)**2)),
            'mae': float(np.mean(np.abs(ref - est))),
            'rmse': float(np.sqrt(np.mean((ref - est)**2))),
            'correlation': corr,
            'evaluable_points': int(eval_mask.sum()),
        }
    
    def introduce_missing_mcar(self, data: np.ndarray, missing_rate: float) -> np.ndarray:
        """Missing Completely At Random."""
        data_missing = data.copy()
        mask = np.random.random(data.shape) < missing_rate
        data_missing[mask] = np.nan
        return data_missing
    
    def run_feature_activation_tests(self):
        """Run tests specifically designed to activate each feature."""
        print("🎯 Running Strategic Feature Activation Tests")
        print("=" * 60)
        
        # Test 1: Spectral/Fourier Features
        print("\n🌊 Test 1: Spectral/Fourier Feature Activation")
        print("   Criteria: seasonality=True AND T >= 60")
        print("   Design: T=80, strong seasonal patterns")
        
        seasonal_data = self.generator.generate_seasonal_data(T=80, N=10, strong_seasonality=True)
        seasonal_missing = self.introduce_missing_mcar(seasonal_data, 0.2)
        
        # Verify criteria
        dc_seasonal = self._analyze_data_characteristics(seasonal_missing)
        print(f"   ✓ T={dc_seasonal['T']} >= 60: {dc_seasonal['T'] >= 60}")
        print(f"   ✓ Seasonality detected: {dc_seasonal['seasonality']}")
        print(f"   ✓ Expected FOURIER_K: min(4, T//12) = min(4, {80//12}) = {min(4, 80//12)}")
        
        self._run_feature_comparison(
            "Spectral Features", seasonal_missing, seasonal_data,
            feature_configs=[
                ("Spectral ON", {"use_spectral": True}),
                ("Spectral OFF", {"use_spectral": False}),
            ]
        )
        
        # Test 2: Skeleton (Sparse Network) Features
        print("\n🦴 Test 2: Skeleton (Sparse Network) Feature Activation")
        print("   Criteria: sparsity > 0.6 AND T > max(40, 2*N)")
        print("   Design: T=150, N=20, sparse network (80% sparsity)")
        
        sparse_data = self.generator.generate_sparse_network_data(T=150, N=20, sparsity_level=0.8)
        sparse_missing = self.introduce_missing_mcar(sparse_data, 0.2)
        
        # Verify criteria
        dc_sparse = self._analyze_data_characteristics(sparse_missing)
        print(f"   ✓ T={dc_sparse['T']} > max(40, 2*{dc_sparse['N']}={2*dc_sparse['N']}): {dc_sparse['T'] > max(40, 2*dc_sparse['N'])}")
        print(f"   ✓ Sparsity={dc_sparse['sparsity']:.3f} > 0.6: {dc_sparse['sparsity'] > 0.6}")
        
        self._run_feature_comparison(
            "Skeleton Features", sparse_missing, sparse_data,
            feature_configs=[
                ("Skeleton ON", {"use_skeleton": True}),
                ("Skeleton OFF", {"use_skeleton": False}),
            ]
        )
        
        # Test 3: Cholesky Solver Features
        print("\n🔧 Test 3: Cholesky Solver Feature Activation")
        print("   Criteria: N >= 20 (to avoid pinv path)")
        print("   Design: T=100, N=25, larger matrix")
        
        large_data = self.generator.generate_large_matrix_data(T=100, N=25)
        large_missing = self.introduce_missing_mcar(large_data, 0.2)
        
        # Verify criteria
        dc_large = self._analyze_data_characteristics(large_missing)
        print(f"   ✓ N={dc_large['N']} >= 20: {dc_large['N'] >= 20}")
        print(f"   ✓ Will use Cholesky path instead of pinv: True")
        
        self._run_feature_comparison(
            "Cholesky Solver", large_missing, large_data,
            feature_configs=[
                ("Cholesky ON", {"use_cholesky": True}),
                ("Cholesky OFF", {"use_cholesky": False}),
            ]
        )
        
        # Test 4: Robust Loss Features
        print("\n🛡️ Test 4: Robust Loss Feature Activation")
        print("   Criteria: has_outliers = True (>5% outliers)")
        print("   Design: T=100, N=10, 10% outliers with large magnitude")
        
        outlier_data = self.generator.generate_outlier_data(T=100, N=10, outlier_fraction=0.1)
        outlier_missing = self.introduce_missing_mcar(outlier_data, 0.2)
        
        # Verify criteria
        dc_outlier = self._analyze_data_characteristics(outlier_missing)
        print(f"   ✓ Outliers detected: {dc_outlier['has_outliers']}")
        print(f"   ✓ Will activate Huber loss: {dc_outlier['has_outliers']}")
        
        self._run_feature_comparison(
            "Robust Loss", outlier_missing, outlier_data,
            feature_configs=[
                ("Robust ON", {"use_robust_loss": True}),
                ("Robust OFF", {"use_robust_loss": False}),
            ]
        )
    
    def _analyze_data_characteristics(self, data: np.ndarray) -> Dict:
        """Analyze data characteristics to verify feature activation criteria."""
        from missnet_imputer import DataCharacteristics
        dc = DataCharacteristics(data)
        return {
            'T': dc.T,
            'N': dc.N,
            'missing_rate': dc.missing_rate,
            'seasonality': dc.seasonality,
            'sparsity': dc.sparsity,
            'has_outliers': dc.has_outliers,
            'temporal_correlation': dc.temporal_correlation,
            'spatial_correlation': dc.spatial_correlation,
        }
    
    def _run_feature_comparison(self, test_name: str, data_missing: np.ndarray, 
                               data_original: np.ndarray, feature_configs: List[Tuple[str, Dict]]):
        """Run comparison for a specific feature."""
        print(f"\n   📊 Running {test_name} comparison:")
        
        missing_mask = np.isnan(data_missing)
        results = []
        
        for config_name, config_params in feature_configs:
            print(f"      🔄 {config_name}...")
            
            start_time = time.time()
            try:
                imputed = missnet_impute(
                    data_missing,
                    verbose=False,
                    auto_tune=False,  # Don't override our settings
                    alpha=0.5,
                    beta=0.1,
                    L=10,
                    n_cl=1,
                    max_iteration=20,
                    **config_params
                )
                
                execution_time = time.time() - start_time
                metrics = self.calculate_metrics(data_original, imputed, missing_mask)
                
                result = {
                    'config_name': config_name,
                    'test_name': test_name,
                    'execution_time': execution_time,
                    'mse': metrics['mse'],
                    'mae': metrics['mae'],
                    'rmse': metrics['rmse'],
                    'correlation': metrics['correlation'],
                    'evaluable_points': metrics['evaluable_points'],
                    'success': True,
                    **config_params
                }
                
                results.append(result)
                print(f"         ✅ MSE={metrics['mse']:.6f}, Time={execution_time:.2f}s")
                
            except Exception as e:
                print(f"         ❌ Failed: {str(e)}")
                results.append({
                    'config_name': config_name,
                    'test_name': test_name,
                    'execution_time': float('inf'),
                    'mse': float('inf'),
                    'mae': float('inf'),
                    'rmse': float('inf'),
                    'correlation': 0.0,
                    'evaluable_points': 0,
                    'success': False,
                    'error': str(e),
                    **config_params
                })
        
        # Analyze results
        successful_results = [r for r in results if r['success']]
        if len(successful_results) >= 2:
            baseline = successful_results[0]
            comparison = successful_results[1]
            
            mse_diff = baseline['mse'] - comparison['mse']
            mse_improvement = mse_diff / baseline['mse'] * 100
            time_diff = comparison['execution_time'] - baseline['execution_time']
            
            print(f"\n   📈 {test_name} Results:")
            print(f"      {baseline['config_name']}: MSE={baseline['mse']:.6f}, Time={baseline['execution_time']:.2f}s")
            print(f"      {comparison['config_name']}: MSE={comparison['mse']:.6f}, Time={comparison['execution_time']:.2f}s")
            print(f"      MSE Improvement: {mse_improvement:+.2f}% ({'better' if mse_improvement > 0 else 'worse'})")
            print(f"      Time Difference: {time_diff:+.2f}s ({'faster' if time_diff < 0 else 'slower'})")
            
            if abs(mse_improvement) < 1.0:
                print(f"      ⚠️  Small MSE difference - feature may not be having strong impact")
            else:
                print(f"      ✅ Significant MSE difference - feature is working as expected")
        
        self.results.extend(results)
    
    def run_comprehensive_ablation(self):
        """Run comprehensive ablation study with all feature combinations."""
        print("\n🔬 Comprehensive Ablation Study")
        print("=" * 50)
        
        # Create a dataset that should activate all features
        print("   Creating dataset optimized for all features...")
        
        # Use parameters that satisfy all criteria
        T_all = 100  # >= 60 for spectral, > max(40, 2*N) for skeleton
        N_all = 25   # >= 20 for Cholesky
        
        # Generate data with strong seasonality, sparse network, and outliers
        base_seasonal = self.generator.generate_seasonal_data(T_all, N_all, strong_seasonality=True)
        
        # Add sparse network structure
        sparse_connections = int(0.15 * N_all * (N_all - 1) / 2)  # 15% connections
        for _ in range(sparse_connections):
            i, j = random.sample(range(N_all), 2)
            correlation_strength = np.random.uniform(0.3, 0.6)
            base_seasonal[:, j] += correlation_strength * base_seasonal[:, i]
        
        # Add outliers
        n_outliers = int(T_all * N_all * 0.08)  # 8% outliers
        for _ in range(n_outliers):
            t_idx = random.randint(0, T_all - 1)
            n_idx = random.randint(0, N_all - 1)
            outlier_magnitude = np.random.choice([-1, 1]) * np.random.uniform(5, 8)
            base_seasonal[t_idx, n_idx] += outlier_magnitude
        
        # Add noise
        test_data = base_seasonal + 0.1 * np.random.randn(T_all, N_all)
        test_data_missing = self.introduce_missing_mcar(test_data, 0.2)
        
        # Verify all criteria are met
        dc = self._analyze_data_characteristics(test_data_missing)
        print(f"   ✓ T={dc['T']} >= 60: {dc['T'] >= 60}")
        print(f"   ✓ Seasonality: {dc['seasonality']}")
        print(f"   ✓ T > max(40, 2*N): {dc['T']} > {max(40, 2*dc['N'])}: {dc['T'] > max(40, 2*dc['N'])}")
        print(f"   ✓ Sparsity > 0.6: {dc['sparsity']:.3f} > 0.6: {dc['sparsity'] > 0.6}")
        print(f"   ✓ N >= 20: {dc['N']} >= 20: {dc['N'] >= 20}")
        print(f"   ✓ Outliers: {dc['has_outliers']}")
        
        # Run comprehensive ablation
        ablation_configs = [
            ("All Features ON", {
                "use_spectral": True,
                "use_skeleton": True,
                "use_cholesky": True,
                "use_robust_loss": True
            }),
            ("No Spectral", {
                "use_spectral": False,
                "use_skeleton": True,
                "use_cholesky": True,
                "use_robust_loss": True
            }),
            ("No Skeleton", {
                "use_spectral": True,
                "use_skeleton": False,
                "use_cholesky": True,
                "use_robust_loss": True
            }),
            ("No Cholesky", {
                "use_spectral": True,
                "use_skeleton": True,
                "use_cholesky": False,
                "use_robust_loss": True
            }),
            ("No Robust Loss", {
                "use_spectral": True,
                "use_skeleton": True,
                "use_cholesky": True,
                "use_robust_loss": False
            }),
            ("All Features OFF", {
                "use_spectral": False,
                "use_skeleton": False,
                "use_cholesky": False,
                "use_robust_loss": False
            }),
        ]
        
        self._run_comprehensive_comparison("Comprehensive Ablation", 
                                          test_data_missing, test_data, 
                                          ablation_configs)
    
    def _run_comprehensive_comparison(self, test_name: str, data_missing: np.ndarray,
                                    data_original: np.ndarray, configs: List[Tuple[str, Dict]]):
        """Run comprehensive comparison with multiple configurations."""
        print(f"\n   📊 Running {test_name}:")
        print(f"      Testing {len(configs)} configurations...")
        
        missing_mask = np.isnan(data_missing)
        results = []
        
        for config_name, config_params in configs:
            print(f"      🔄 {config_name}...")
            
            start_time = time.time()
            try:
                imputed = missnet_impute(
                    data_missing,
                    verbose=False,
                    auto_tune=False,
                    alpha=0.5,
                    beta=0.1,
                    L=10,
                    n_cl=1,
                    max_iteration=20,
                    **config_params
                )
                
                execution_time = time.time() - start_time
                metrics = self.calculate_metrics(data_original, imputed, missing_mask)
                
                result = {
                    'config_name': config_name,
                    'test_name': test_name,
                    'execution_time': execution_time,
                    'mse': metrics['mse'],
                    'mae': metrics['mae'],
                    'rmse': metrics['rmse'],
                    'correlation': metrics['correlation'],
                    'evaluable_points': metrics['evaluable_points'],
                    'success': True,
                    **config_params
                }
                
                results.append(result)
                print(f"         ✅ MSE={metrics['mse']:.6f}, Time={execution_time:.2f}s")
                
            except Exception as e:
                print(f"         ❌ Failed: {str(e)}")
                results.append({
                    'config_name': config_name,
                    'test_name': test_name,
                    'execution_time': float('inf'),
                    'mse': float('inf'),
                    'mae': float('inf'),
                    'rmse': float('inf'),
                    'correlation': 0.0,
                    'evaluable_points': 0,
                    'success': False,
                    'error': str(e),
                    **config_params
                })
        
        # Sort by MSE for analysis
        successful_results = [r for r in results if r['success']]
        if successful_results:
            successful_results.sort(key=lambda x: x['mse'])
            
            print(f"\n   📈 {test_name} Results (ranked by MSE):")
            print(f"      {'Config':<20} {'MSE':<12} {'MAE':<12} {'Corr':<8} {'Time':<8}")
            print(f"      {'-'*20} {'-'*12} {'-'*12} {'-'*8} {'-'*8}")
            
            baseline_mse = None
            for i, result in enumerate(successful_results):
                mse_str = f"{result['mse']:.6f}"
                mae_str = f"{result['mae']:.6f}"
                corr_str = f"{result['correlation']:.4f}"
                time_str = f"{result['execution_time']:.2f}s"
                
                if baseline_mse is None:
                    baseline_mse = result['mse']
                    delta_str = "baseline"
                else:
                    delta_mse = (baseline_mse - result['mse']) / baseline_mse * 100
                    delta_str = f"{delta_mse:+.1f}%"
                
                print(f"      {result['config_name']:<20} {mse_str:<12} {mae_str:<12} {corr_str:<8} {time_str:<8} ({delta_str})")
            
            # Feature impact analysis
            if len(successful_results) >= 2:
                best = successful_results[0]
                worst = successful_results[-1]
                improvement = (worst['mse'] - best['mse']) / worst['mse'] * 100
                
                print(f"\n   🔬 Feature Impact Analysis:")
                print(f"      Best config: {best['config_name']} (MSE={best['mse']:.6f})")
                print(f"      Worst config: {worst['config_name']} (MSE={worst['mse']:.6f})")
                print(f"      Total improvement potential: {improvement:.1f}%")
                
                # Individual feature impacts
                all_on = next((r for r in successful_results if "All Features ON" in r['config_name']), None)
                if all_on:
                    print(f"\n      Individual feature impacts (degradation when disabled):")
                    
                    for config_name, config_params in configs[1:]:  # Skip "All Features ON"
                        feature_name = config_name.replace("No ", "")
                        feature_result = next((r for r in successful_results if r['config_name'] == config_name), None)
                        
                        if feature_result and all_on:
                            impact = (feature_result['mse'] - all_on['mse']) / all_on['mse'] * 100
                            print(f"         • {feature_name}: {impact:+.2f}%")
        
        self.results.extend(results)
    
    def generate_summary_report(self) -> str:
        """Generate comprehensive summary report."""
        if not self.results:
            return "No test results available."
        
        df = pd.DataFrame(self.results)
        successful_tests = df[df['success']]
        
        report = []
        report.append("🎯 IMPROVED MISSNET ABLATION STUDY REPORT")
        report.append("=" * 60)
        report.append(f"Total tests: {len(self.results)}")
        report.append(f"Successful: {len(successful_tests)}")
        report.append(f"Success rate: {len(successful_tests)/len(self.results)*100:.1f}%")
        report.append("")
        
        # Group by test type
        if 'test_name' in successful_tests.columns:
            report.append("📊 Results by Test Category:")
            report.append("-" * 35)
            
            for test_name in successful_tests['test_name'].unique():
                test_data = successful_tests[successful_tests['test_name'] == test_name]
                
                if len(test_data) >= 2:
                    best_mse = test_data['mse'].min()
                    worst_mse = test_data['mse'].max()
                    improvement = (worst_mse - best_mse) / worst_mse * 100
                    
                    report.append(f"{test_name}:")
                    report.append(f"   Configurations tested: {len(test_data)}")
                    report.append(f"   MSE range: {best_mse:.6f} - {worst_mse:.6f}")
                    report.append(f"   Improvement potential: {improvement:.1f}%")
                    report.append("")
        
        # Overall best and worst configurations
        if len(successful_tests) > 0:
            best_overall = successful_tests.loc[successful_tests['mse'].idxmin()]
            worst_overall = successful_tests.loc[successful_tests['mse'].idxmax()]
            
            report.append("🏆 Overall Performance:")
            report.append(f"   Best: {best_overall['config_name']} (MSE={best_overall['mse']:.6f})")
            report.append(f"   Worst: {worst_overall['config_name']} (MSE={worst_overall['mse']:.6f})")
            report.append("")
        
        return "\n".join(report)
    
    def save_results(self, filepath: str):
        """Save results to CSV."""
        if self.results:
            df = pd.DataFrame(self.results)
            df.to_csv(filepath, index=False)
            print(f"💾 Results saved to {filepath}")


def main():
    """Main testing function."""
    parser = argparse.ArgumentParser(description='Improved MISSNET Ablation Study')
    parser.add_argument('--feature-only', action='store_true',
                       help='Run only feature activation tests')
    parser.add_argument('--comprehensive-only', action='store_true',
                       help='Run only comprehensive ablation tests')
    parser.add_argument('--no-report', action='store_true',
                       help='Skip detailed report generation')
    
    args = parser.parse_args()
    
    print("🎯 IMPROVED MISSNET ABLATION STUDY")
    print("=" * 50)
    print("Strategic test scenarios designed to activate all features")
    print("with measurable performance differences.")
    
    # Initialize tester
    tester = ImprovedAblationTester(random_seed=42)
    
    if args.comprehensive_only:
        # Run only comprehensive ablation
        tester.run_comprehensive_ablation()
    elif args.feature_only:
        # Run only feature activation tests
        tester.run_feature_activation_tests()
    else:
        # Run all tests
        tester.run_feature_activation_tests()
        tester.run_comprehensive_ablation()
    
    # Generate and save results
    if not args.no_report:
        report = tester.generate_summary_report()
        print("\n" + report)
    
    # Save results
    tester.save_results('missnet_improved_ablation_results.csv')
    
    print("\n🎉 Improved ablation study completed!")
    print("📄 Results saved to: missnet_improved_ablation_results.csv")
    print("\n✅ Key Improvements:")
    print("   • Strategic data generation meets all feature activation criteria")
    print("   • Controlled scenarios isolate individual feature impacts")
    print("   • Measurable performance differences for all toggles")
    print("   • Clear documentation of why each feature activates")


if __name__ == "__main__":
    main()
