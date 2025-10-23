#!/usr/bin/env python3
"""
MISSNET Skeleton Activation Debug Tool

This script helps diagnose why the skeleton method never activates
and provides solutions to enable skeleton functionality.
"""

import numpy as np
import warnings
import time
from typing import Dict, Tuple, Optional, List
from dataclasses import dataclass

# Import the main missnet functionality
from missnet_imputer import (
    MissNet, MissNetConfig, DataCharacteristics, 
    get_optimal_config, CFG, apply_cfg_overrides
)


class MissNetSkeletonDebug(MissNet):
    """
    Enhanced MISSNET with detailed skeleton activation diagnostics.
    """
    
    def __init__(self, alpha=0.5, beta=0.1, L=10, n_cl=1, 
                 use_robust_loss=True, use_skeleton=True, use_cholesky=True,
                 use_spectral=False, use_consistency_loss=False, debug_skeleton=True):
        super().__init__(alpha, beta, L, n_cl, use_robust_loss, use_skeleton, 
                        use_cholesky, use_spectral, use_consistency_loss)
        self.debug_skeleton = debug_skeleton
        self.skeleton_debug_info = []
    
    def diagnose_skeleton_activation(self, X, verbose=True):
        """
        Diagnose why skeleton might not be activating for your data.
        """
        if verbose:
            print("\n🔍 SKELETON ACTIVATION DIAGNOSIS")
            print("=" * 50)
        
        # Analyze data characteristics
        data_chars = DataCharacteristics(X)
        
        if verbose:
            print(f"📊 Data Characteristics:")
            print(f"   Shape: {data_chars.T} × {data_chars.N}")
            print(f"   Missing rate: {data_chars.missing_rate*100:.1f}%")
            print(f"   Estimated sparsity: {data_chars.sparsity:.3f}")
            print(f"   Temporal correlation: {data_chars.temporal_correlation:.3f}")
            print(f"   Spatial correlation: {data_chars.spatial_correlation:.3f}")
        
        # Check skeleton activation conditions
        conditions = {
            "use_skeleton enabled": self.use_skeleton,
            "CFG['USE_CANDIDATE_GL']": CFG.get('USE_CANDIDATE_GL', False),
            "sklearn available": True,  # Assuming available if import worked
            "data sparsity > 0.6": data_chars.sparsity > 0.6,
            "T > max(40, 2*N)": data_chars.T > max(40, 2 * data_chars.N),
        }
        
        if verbose:
            print(f"\n🔧 Skeleton Activation Conditions:")
            for condition, met in conditions.items():
                status = "✅ PASS" if met else "❌ FAIL"
                print(f"   {condition}: {status}")
                if not met:
                    if condition == "data sparsity > 0.6":
                        print(f"      → Current sparsity: {data_chars.sparsity:.3f} (need > 0.6)")
                    elif condition == "T > max(40, 2*N)":
                        threshold = max(40, 2 * data_chars.N)
                        print(f"      → Current T: {data_chars.T}, threshold: {threshold}")
        
        # Check per-regime conditions
        if verbose:
            print(f"\n📈 Per-Regime Analysis:")
            print(f"   Number of regimes (n_cl): {self.n_cl}")
            avg_samples_per_regime = data_chars.T / self.n_cl
            print(f"   Average samples per regime: {avg_samples_per_regime:.1f}")
            print(f"   Samples per regime > 10: {avg_samples_per_regime > 10}")
        
        # Overall assessment
        all_conditions_met = all(conditions.values()) and avg_samples_per_regime > 10
        
        if verbose:
            print(f"\n🎯 Overall Assessment:")
            if all_conditions_met:
                print("   ✅ All conditions met - SKELETON SHOULD ACTIVATE")
            else:
                print("   ❌ Some conditions failed - SKELETON WILL NOT ACTIVATE")
                
        return conditions, all_conditions_met, data_chars
    
    def update_networks_debug(self, X, W):
        """
        Enhanced update_networks with detailed skeleton diagnostics.
        """
        self.skeleton_debug_info = []
        
        for k in range(self.n_cl):
            F_k = np.where(self.F == k)[0]
            if len(F_k) == 0: 
                continue
            
            debug_info = {
                'regime': k,
                'samples': len(F_k),
                'skeleton_used': False,
                'reason': ''
            }
            
            # Step 1: Create imputed data
            Y = np.array([self.U[k] @ zt for zt in self.zt])
            X_impute = np.nan_to_num(X) + (1-W)*Y
            X_regime = X_impute[F_k]
            
            # Estimate sparsity to choose method
            est_sparsity = 0.5  # Default
            if hasattr(self, 'H') and k < len(self.H) and self.H[k] is not None:
                # Estimate from previous iteration
                H_prev = self.H[k]
                est_sparsity = (np.abs(H_prev) < 0.01).sum() / (self.N ** 2)
            
            debug_info['estimated_sparsity'] = est_sparsity
            
            # Check all skeleton conditions
            conditions = {
                'use_skeleton': self.use_skeleton,
                'USE_CANDIDATE_GL': CFG['USE_CANDIDATE_GL'],
                'Lasso_available': True,  # Simplified
                'sparsity_threshold': est_sparsity > 0.6,
                'enough_samples': len(F_k) > 10
            }
            
            debug_info['conditions'] = conditions
            
            # Adaptive method selection
            use_skeleton_method = all(conditions.values())
            
            if use_skeleton_method:
                debug_info['skeleton_used'] = True
                debug_info['reason'] = 'All conditions met'
                try:
                    # Fast path: nodewise skeleton with adaptive alpha
                    from missnet_imputer import _standardize, adaptive_lasso_alpha, nodewise_skeleton_and_precision
                    X_std, mu, sd = _standardize(X_regime)
                    alpha_adaptive = adaptive_lasso_alpha(X_std, base_alpha=CFG['LASSO_ALPHA'])
                    kmax = CFG['MAX_NEIGHBORS'] if CFG['MAX_NEIGHBORS'] is not None else None
                    
                    H_new, A_mask = nodewise_skeleton_and_precision(
                        X_std, alpha=alpha_adaptive, k_max=kmax)
                    self.H[k] = H_new
                    self.G[k] = np.nanmean(X_regime, axis=0)
                    self.S[k] = self.normalize_precision(H_new)
                    
                except Exception as e:
                    debug_info['skeleton_used'] = False
                    debug_info['reason'] = f'Exception: {str(e)}'
                    use_skeleton_method = False
            else:
                # Find which condition failed
                failed_conditions = [cond for cond, met in conditions.items() if not met]
                debug_info['reason'] = f'Failed conditions: {failed_conditions}'
            
            if not use_skeleton_method:
                # Dense/fallback path: TVGL
                from missnet_imputer import TVGL
                test = TVGL(alpha=self.beta, beta=0, max_iter=1000, 
                           psi='laplacian', assume_centered=False)
                test.fit(X_regime, np.zeros(X_regime.shape[0]))
                self.H[k] = test.precision_[0]
                self.G[k] = np.nanmean(X_regime, axis=0)
                self.S[k] = self.normalize_precision(test.precision_[0])
            
            self.skeleton_debug_info.append(debug_info)
    
    def print_skeleton_summary(self):
        """Print summary of skeleton usage across regimes."""
        print("\n📋 SKELETON USAGE SUMMARY")
        print("=" * 30)
        
        total_regimes = len(self.skeleton_debug_info)
        skeleton_regimes = sum(1 for info in self.skeleton_debug_info if info['skeleton_used'])
        
        print(f"Total regimes: {total_regimes}")
        print(f"Regimes using skeleton: {skeleton_regimes}")
        print(f"Skeleton usage rate: {skeleton_regimes/total_regimes*100:.1f}%")
        
        for info in self.skeleton_debug_info:
            status = "🦴 SKELETON" if info['skeleton_used'] else "📊 TVGL"
            print(f"   Regime {info['regime']} ({info['samples']} samples): {status}")
            if not info['skeleton_used']:
                print(f"      Reason: {info['reason']}")
                if 'estimated_sparsity' in info:
                    print(f"      Sparsity: {info['estimated_sparsity']:.3f}")
    
    def fit(self, X, random_init=False, max_iter=20, min_iter=3, tol=5, verbose=True, savedir='./temp'):
        """
        Override fit to use debug version of update_networks.
        """
        import os
        import time
        import shutil
        from missnet_imputer import make_dir
        
        make_dir(savedir, delete=True)

        W = ~np.isnan(X)
        if verbose:
            print('\nnumber of nan', np.count_nonzero(np.isnan(X)), 'percentage', 
                  np.round(np.count_nonzero(np.isnan(X))/X.size*100, decimals=1), '%\n')
        
        # Initialize model parameters first
        self.initialize(X, random_init)
        
        # Diagnose skeleton activation before training
        if self.debug_skeleton:
            self.diagnose_skeleton_activation(X, verbose)
        
        # De-seasonalize data if spectral features are enabled
        X_work = X
        if self.use_spectral and int(CFG.get("FOURIER_K", 0)) > 0:
            self.compute_seasonal_baseline(X, W)
            X_work = X - self.X_season  # de-seasonalize
            
            # Re-center to preserve mean level after de-seasonalization
            col_means = np.nanmean(X_work, axis=0)
            X_work -= col_means
            self.X_season += col_means  # add it back later during imputation
            
        history = {'lle': [], 'time': []}
        min_lle, tol_counter = np.inf, 0
        
        for iteration in range(1, max_iter + 1):
            tic = time.time()
            try:
                """ E-step """
                # infer F, Z
                lle = self.forward_viterbi(X_work, W)
                if verbose and self.n_cl > 1: 
                    print('\tcluster assignments', [np.count_nonzero(self.F == k) for k in range(self.n_cl)])
                self.backward()
                # infer V
                self.update_latent_context()

                """ M-step """
                # update parameters
                lle -= self.solve_model(X_work, W, return_loglikelihood=True)

                """ Update the missing values and networks"""
                self.update_networks_debug(X_work, W)  # Use debug version

                toc = time.time()
                history['time'].append(toc - tic)
                history['lle'].append(lle)
                if verbose: 
                    print(f'\titer= {iteration}, lle= {lle:.3f}, time= {toc-tic:.3f} [sec]')

                # if LLE no more reduce then convergence (tol=5)
                if iteration > min_iter:
                    if lle < min_lle:
                        min_lle = lle
                        self.save_pkl(savedir)  # save the best model
                        tol_counter = 0
                    else:
                        tol_counter += 1

                    if tol_counter >= tol:  # Early stopping (end of training)
                        self.load_pkl(savedir)  # load the best model
                        break
            except:
                if verbose: print("EM algorithm Error\n")
                self.load_pkl(savedir)  # load the best model
                break

        # Print skeleton summary
        if self.debug_skeleton:
            self.print_skeleton_summary()

        return history


def missnet_impute_debug(incomp_data, alpha=None, beta=None, L=None, n_cl=None, max_iteration=None, 
                         tol=5, random_init=False, verbose=True, use_robust_loss=None, 
                         use_skeleton=None, use_cholesky=None, use_spectral=None, 
                         use_consistency_loss=None, auto_tune=True, debug_skeleton=True, **kwargs):
    """
    Debug version of missnet_impute with skeleton diagnostics.
    """
    # Handle duplicate verbose parameter
    if 'verbose' in kwargs:
        kwargs.pop('verbose')
    
    # Filter out any other unexpected kwargs
    allowed_kwargs = ['alpha', 'beta', 'L', 'n_cl', 'max_iteration', 
                     'use_robust_loss', 'use_skeleton', 'use_cholesky',
                     'use_spectral', 'use_consistency_loss', 'auto_tune']
    filtered_kwargs = {k: v for k, v in kwargs.items() if k in allowed_kwargs}
    
    # Merge filtered kwargs with function parameters
    locals().update(filtered_kwargs)
    recov = np.copy(incomp_data)
    m_mask = np.isnan(incomp_data)
    
    # Auto-tune parameters if requested
    if auto_tune:
        if verbose and any(p is None for p in [alpha, beta, L, n_cl, max_iteration, 
                                               use_robust_loss, use_skeleton, use_cholesky,
                                               use_spectral, use_consistency_loss]):
            print("\n🔧 Auto-tuning parameters...")

        optimal_config = get_optimal_config(incomp_data, fast=True, verbose=verbose)

        # Apply CFG overrides
        cfg_overrides = optimal_config.get('cfg_overrides', {})
        if cfg_overrides and verbose:
            print("   • applying CFG overrides:", ", ".join(sorted(cfg_overrides.keys())))
        apply_cfg_overrides(CFG, cfg_overrides)

        # Respect explicit args over auto-tune
        if use_robust_loss is True:
            CFG["USE_HUBER"] = True
        elif use_robust_loss is False:
            CFG["USE_HUBER"] = False

        if use_spectral is True:
            CFG["USE_SPECTRAL"] = True
        elif use_spectral is False:
            CFG["USE_SPECTRAL"] = False

        # Use auto-tuned values only for fields the user left as None
        if alpha is None: alpha = optimal_config['alpha']
        if beta is None: beta = optimal_config['beta']
        if L is None: L = optimal_config['L']
        if n_cl is None: n_cl = optimal_config['n_cl']
        if max_iteration is None: max_iteration = optimal_config['max_iteration']
        if use_robust_loss is None: use_robust_loss = optimal_config['use_robust_loss']
        if use_skeleton is None: use_skeleton = optimal_config['use_skeleton']
        if use_cholesky is None: use_cholesky = optimal_config['use_cholesky']

        if use_spectral is None: use_spectral = bool(CFG.get('USE_SPECTRAL', False))
        if use_consistency_loss is None: use_consistency_loss = bool(CFG.get('USE_CONSISTENCY_LOSS', False))
    else:
        # Use fixed defaults for any None parameters
        if alpha is None: alpha = 0.5
        if beta is None: beta = 0.1
        if L is None: L = 10
        if n_cl is None: n_cl = 1
        if max_iteration is None: max_iteration = 20
        if use_robust_loss is None: use_robust_loss = False
        if use_skeleton is None: use_skeleton = False
        if use_cholesky is None: use_cholesky = False

    if verbose:
        print(f"\n(IMPUTATION) MISSNET++ DEBUG VERSION")
        print(f"\tMatrix Shape: {incomp_data.shape[0]} × {incomp_data.shape[1]}")
        print(f"\tMissing rate: {np.isnan(incomp_data).sum() / incomp_data.size * 100:.1f}%")
        print(f"\n\tParameters:")
        print(f"\t  alpha: {alpha:.3f}")
        print(f"\t  beta: {beta:.4f}")
        print(f"\t  L: {L}")
        print(f"\t  n_cl: {n_cl}")
        print(f"\t  max_iteration: {max_iteration}")
        print(f"\t  use_robust_loss: {use_robust_loss}")
        print(f"\t  use_skeleton: {use_skeleton}")
        print(f"\t  use_cholesky: {use_cholesky}")

    start_time = time.time()

    missnet_model = MissNetSkeletonDebug(alpha=alpha, beta=beta, L=L, n_cl=n_cl,
                                        use_robust_loss=use_robust_loss, 
                                        use_skeleton=use_skeleton, use_cholesky=use_cholesky,
                                        use_spectral=use_spectral, 
                                        use_consistency_loss=use_consistency_loss,
                                        debug_skeleton=debug_skeleton)
    missnet_model.fit(incomp_data, random_init=random_init, max_iter=max_iteration, 
                     tol=tol, verbose=verbose)
    recov_data = missnet_model.imputation()

    end_time = time.time()

    recov[m_mask] = recov_data[m_mask]

    if verbose:
        print(f"\n> logs: imputation miss_net - Execution Time: {(end_time - start_time):.4f} seconds\n")

    return recov


if __name__ == "__main__":
    print("MISSNET Skeleton Debug Tool")
    print("=" * 50)
    
    # Create test data with different sparsity levels
    np.random.seed(42)
    
    # Test Case 1: Dense data (skeleton should NOT activate)
    print("\n" + "="*50)
    print("TEST CASE 1: Dense Data (Skeleton Should NOT Activate)")
    print("="*50)
    
    n_timesteps, n_features = 100, 10
    data_dense = np.random.randn(n_timesteps, n_features)
    # Add some correlation to make it dense
    data_dense += 0.5 * np.random.randn(n_timesteps, 1)
    
    # Introduce missing values
    mask = np.random.random(data_dense.shape) < 0.2
    data_dense[mask] = np.nan
    
    print(f"Dense data shape: {data_dense.shape}")
    print(f"Missing values: {np.isnan(data_dense).sum()}/{data_dense.size}")
    
    imputed_dense = missnet_impute_debug(
        data_dense,
        max_iteration=10,
        verbose=True
    )
    
    # Test Case 2: Sparse data (skeleton SHOULD activate)
    print("\n" + "="*50)
    print("TEST CASE 2: Sparse Data (Skeleton SHOULD Activate)")
    print("="*50)
    
    # Create sparse data by making most correlations near zero
    data_sparse = np.random.randn(n_timesteps, n_features)
    # Add strong noise to reduce correlations
    data_sparse += 2.0 * np.random.randn(n_timesteps, n_features)
    
    # Introduce missing values
    mask = np.random.random(data_sparse.shape) < 0.2
    data_sparse[mask] = np.nan
    
    print(f"Sparse data shape: {data_sparse.shape}")
    print(f"Missing values: {np.isnan(data_sparse).sum()}/{data_sparse.size}")
    
    imputed_sparse = missnet_impute_debug(
        data_sparse,
        max_iteration=10,
        verbose=True
    )
    
    print("\n" + "="*50)
    print("DIAGNOSIS COMPLETE")
    print("="*50)
    print("Check the output above to see why skeleton is/ isn't activating!")
