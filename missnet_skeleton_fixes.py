#!/usr/bin/env python3
"""
MISSNET Skeleton Activation Fixes

This script provides practical solutions to enable skeleton functionality
when it doesn't activate due to strict conditions.
"""

import numpy as np
import warnings
from typing import Dict, Tuple, Optional, List
from dataclasses import dataclass

# Import the main missnet functionality
from missnet_imputer import (
    MissNet, MissNetConfig, DataCharacteristics, 
    get_optimal_config, CFG, apply_cfg_overrides, _cfg_overrides_from_chars
)


def get_skeleton_friendly_config(X: np.ndarray, verbose: bool = True) -> Dict:
    """
    Generate a configuration that is more likely to activate skeleton.
    
    This function relaxes the strict skeleton activation conditions while
    maintaining good performance.
    """
    data_chars = DataCharacteristics(X)
    
    if verbose:
        print("\n🔧 GENERATING SKELETON-FRIENDLY CONFIG")
        print("=" * 50)
        print(f"📊 Original Data Characteristics:")
        print(f"   Shape: {data_chars.T} × {data_chars.N}")
        print(f"   Missing rate: {data_chars.missing_rate*100:.1f}%")
        print(f"   Original sparsity: {data_chars.sparsity:.3f}")
    
    # Get original config
    original_config = get_optimal_config(X, fast=True, verbose=False)
    
    # Create skeleton-friendly overrides
    overrides = {}
    
    # FIX 1: Force enable candidate GL (skeleton)
    overrides["USE_CANDIDATE_GL"] = True
    
    # FIX 2: Lower sparsity threshold for skeleton activation
    # Original requires > 0.6, we'll use > 0.3
    if data_chars.sparsity <= 0.6:
        if verbose:
            print(f"   📉 Lowering sparsity threshold: {data_chars.sparsity:.3f} → 0.3")
        # Note: This requires code modification, but we can influence it
    
    # FIX 3: Ensure minimum samples per regime
    min_samples_needed = max(15, data_chars.T // 10)
    if data_chars.T < min_samples_needed:
        if verbose:
            print(f"   📈 Recommending more time points or fewer regimes")
    
    # FIX 4: Reduce minimum samples requirement from 10 to 5
    # This requires code modification in update_networks
    
    # FIX 5: Adjust Lasso alpha for better skeleton performance
    base_alpha = _cfg_overrides_from_chars(data_chars).get("LASSO_ALPHA", 0.02)
    # Make it more aggressive for sparser graphs
    if data_chars.sparsity > 0.4:
        overrides["LASSO_ALPHA"] = base_alpha * 0.5  # Less regularization
    else:
        overrides["LASSO_ALPHA"] = base_alpha * 1.5  # More regularization
    
    # FIX 6: Limit maximum neighbors to ensure sparsity
    if data_chars.N > 20:
        overrides["MAX_NEIGHBORS"] = max(3, int(np.sqrt(data_chars.N) // 2))
    else:
        overrides["MAX_NEIGHBORS"] = max(2, data_chars.N // 3)
    
    # Apply the overrides
    apply_cfg_overrides(CFG, overrides)
    
    # Update the original config
    skeleton_config = original_config.copy()
    skeleton_config['cfg_overrides'] = overrides
    
    # Force skeleton usage in the main config
    skeleton_config['use_skeleton'] = True
    
    if verbose:
        print(f"\n🎯 Skeleton-Friendly Overrides Applied:")
        for key, value in overrides.items():
            print(f"   {key}: {value}")
        print(f"\n💡 Expected skeleton activation: HIGH")
    
    return skeleton_config


class MissNetSkeletonFriendly(MissNet):
    """
    Modified MISSNET with relaxed skeleton activation conditions.
    """
    
    def __init__(self, alpha=0.5, beta=0.1, L=10, n_cl=1, 
                 use_robust_loss=True, use_skeleton=True, use_cholesky=True,
                 use_spectral=False, use_consistency_loss=False, 
                 skeleton_sparsity_threshold=0.3, skeleton_min_samples=5):
        super().__init__(alpha, beta, L, n_cl, use_robust_loss, use_skeleton, 
                        use_cholesky, use_spectral, use_consistency_loss)
        self.skeleton_sparsity_threshold = skeleton_sparsity_threshold
        self.skeleton_min_samples = skeleton_min_samples
    
    def update_networks(self, X, W):
        """
        Enhanced update_networks with relaxed skeleton conditions.
        """
        for k in range(self.n_cl):
            F_k = np.where(self.F == k)[0]
            if len(F_k) == 0: 
                continue
            
            # Step 1: Create imputed data
            Y = np.array([self.U[k] @ zt for zt in self.zt])
            X_impute = np.nan_to_num(X) + (1-W)*Y
            X_regime = X_impute[F_k]
            
            # Uncertainty gating (optional)
            if hasattr(self, "compute_cell_variance") and CFG['USE_UNCERTAINTY_GATE']:
                varM = self.compute_cell_variance()
                conf = (varM[F_k].mean() < np.percentile(varM, CFG['VAR_TOPQ']))
                if not conf:
                    # Skip unstable regime updates this iter
                    self.G[k] = np.nanmean(X_regime, axis=0)
                    continue
            
            # Estimate sparsity to choose method
            est_sparsity = 0.5  # Default
            if hasattr(self, 'H') and k < len(self.H) and self.H[k] is not None:
                # Estimate from previous iteration
                H_prev = self.H[k]
                est_sparsity = (np.abs(H_prev) < 0.01).sum() / (self.N ** 2)
            
            # RELAXED skeleton conditions
            use_skeleton_method = (self.use_skeleton and 
                                  CFG['USE_CANDIDATE_GL'] and 
                                  True and  # Assuming sklearn available
                                  est_sparsity > self.skeleton_sparsity_threshold and  # RELAXED: 0.3 instead of 0.6
                                  len(F_k) > self.skeleton_min_samples)  # RELAXED: 5 instead of 10
            
            if use_skeleton_method:
                # Fast path: nodewise skeleton with adaptive alpha
                try:
                    from missnet_imputer import _standardize, adaptive_lasso_alpha, nodewise_skeleton_and_precision
                    X_std, mu, sd = _standardize(X_regime)
                    alpha_adaptive = adaptive_lasso_alpha(X_std, base_alpha=CFG['LASSO_ALPHA'])
                    kmax = CFG['MAX_NEIGHBORS'] if CFG['MAX_NEIGHBORS'] is not None else None
                    
                    H_new, A_mask = nodewise_skeleton_and_precision(
                        X_std, alpha=alpha_adaptive, k_max=kmax)
                    self.H[k] = H_new
                    self.G[k] = np.nanmean(X_regime, axis=0)
                    self.S[k] = self.normalize_precision(H_new)
                    
                    print(f"   🦴 Regime {k}: SKELETON ACTIVATED (sparsity: {est_sparsity:.3f}, samples: {len(F_k)})")
                    
                except Exception as e:
                    print(f"   ⚠️  Regime {k}: Skeleton failed ({str(e)}), falling back to TVGL")
                    use_skeleton_method = False
            
            if not use_skeleton_method:
                # Dense/fallback path: TVGL
                from missnet_imputer import TVGL
                test = TVGL(alpha=self.beta, beta=0, max_iter=1000, 
                           psi='laplacian', assume_centered=False)
                test.fit(X_regime, np.zeros(X_regime.shape[0]))
                self.H[k] = test.precision_[0]
                self.G[k] = np.nanmean(X_regime, axis=0)
                self.S[k] = self.normalize_precision(test.precision_[0])
                
                print(f"   📊 Regime {k}: TVGL used (sparsity: {est_sparsity:.3f}, samples: {len(F_k)})")


def missnet_impute_skeleton_friendly(incomp_data, alpha=None, beta=None, L=None, n_cl=None, 
                                     max_iteration=None, tol=5, random_init=False, verbose=True, 
                                     use_robust_loss=None, use_skeleton=True, use_cholesky=None, 
                                     use_spectral=None, use_consistency_loss=None, auto_tune=True, 
                                     skeleton_sparsity_threshold=0.3, skeleton_min_samples=5, **kwargs):
    """
    Skeleton-friendly version of missnet_impute with relaxed activation conditions.
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
        if verbose:
            print("\n🔧 Auto-tuning with SKELETON-FRIENDLY settings...")

        optimal_config = get_skeleton_friendly_config(incomp_data, verbose=verbose)

        # Apply CFG overrides
        cfg_overrides = optimal_config.get('cfg_overrides', {})
        if cfg_overrides and verbose:
            print("   • applying skeleton-friendly CFG overrides:", ", ".join(sorted(cfg_overrides.keys())))
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
        if use_cholesky is None: use_cholesky = optimal_config['use_cholesky']

        # Force skeleton usage
        use_skeleton = True

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
        if use_cholesky is None: use_cholesky = False

    if verbose:
        print(f"\n(IMPUTATION) MISSNET++ SKELETON-FRIENDLY VERSION")
        print(f"\tMatrix Shape: {incomp_data.shape[0]} × {incomp_data.shape[1]}")
        print(f"\tMissing rate: {np.isnan(incomp_data).sum() / incomp_data.size * 100:.1f}%")
        print(f"\n\tParameters:")
        print(f"\t  alpha: {alpha:.3f}")
        print(f"\t  beta: {beta:.4f}")
        print(f"\t  L: {L}")
        print(f"\t  n_cl: {n_cl}")
        print(f"\t  max_iteration: {max_iteration}")
        print(f"\t  use_robust_loss: {use_robust_loss}")
        print(f"\t  use_skeleton: {use_skeleton} (FORCED)")
        print(f"\t  use_cholesky: {use_cholesky}")
        print(f"\t  skeleton_sparsity_threshold: {skeleton_sparsity_threshold}")
        print(f"\t  skeleton_min_samples: {skeleton_min_samples}")

    import time
    start_time = time.time()

    missnet_model = MissNetSkeletonFriendly(alpha=alpha, beta=beta, L=L, n_cl=n_cl,
                                           use_robust_loss=use_robust_loss, 
                                           use_skeleton=use_skeleton, use_cholesky=use_cholesky,
                                           use_spectral=use_spectral, 
                                           use_consistency_loss=use_consistency_loss,
                                           skeleton_sparsity_threshold=skeleton_sparsity_threshold,
                                           skeleton_min_samples=skeleton_min_samples)
    missnet_model.fit(incomp_data, random_init=random_init, max_iter=max_iteration, 
                     tol=tol, verbose=verbose)
    recov_data = missnet_model.imputation()

    end_time = time.time()

    recov[m_mask] = recov_data[m_mask]

    if verbose:
        print(f"\n> logs: imputation miss_net - Execution Time: {(end_time - start_time):.4f} seconds\n")

    return recov


def force_skeleton_activation_config():
    """
    Force skeleton activation by modifying CFG directly.
    This is the most aggressive approach.
    """
    print("\n🔥 FORCING SKELETON ACTIVATION")
    print("=" * 40)
    
    # Force all skeleton-related settings
    force_overrides = {
        "USE_CANDIDATE_GL": True,
        "LASSO_ALPHA": 0.005,  # Very low to encourage connections
        "MAX_NEIGHBORS": None,  # No limit on neighbors
        "USE_CHOLESKY": True,
        "USE_HUBER": False,  # Disable robust loss for simplicity
        "USE_UNCERTAINTY_GATE": False,  # Disable uncertainty gating
        "USE_SPECTRAL": False,  # Disable spectral to focus on skeleton
        "USE_CONSISTENCY_LOSS": False,  # Disable consistency loss
    }
    
    apply_cfg_overrides(CFG, force_overrides)
    
    print("🔧 Applied force overrides:")
    for key, value in force_overrides.items():
        print(f"   {key}: {value}")
    
    print("\n⚠️  WARNING: This is an aggressive approach that may affect performance!")
    print("   Use only for testing skeleton functionality.")
    
    return force_overrides


if __name__ == "__main__":
    print("MISSNET Skeleton Activation Fixes")
    print("=" * 50)
    
    # Create test data
    np.random.seed(42)
    n_timesteps, n_features = 80, 8  # Smaller to test edge cases
    
    # Create moderately sparse data
    data = np.random.randn(n_timesteps, n_features)
    # Add some structure but keep it relatively sparse
    data += 0.3 * np.sin(np.linspace(0, 4*np.pi, n_timesteps))[:, None]
    
    # Introduce missing values
    mask = np.random.random(data.shape) < 0.25
    data_with_missing = data.copy()
    data_with_missing[mask] = np.nan
    
    print(f"Test data shape: {data_with_missing.shape}")
    print(f"Missing values: {np.isnan(data_with_missing).sum()}/{data_with_missing.size}")
    
    # Test 1: Regular MISSNET (skeleton probably won't activate)
    print("\n" + "="*50)
    print("TEST 1: Regular MISSNET")
    print("="*50)
    
    try:
        from missnet_imputer import missnet_impute
        imputed_regular = missnet_impute(
            data_with_missing,
            max_iteration=10,
            verbose=False
        )
        print("✅ Regular MISSNET completed")
    except Exception as e:
        print(f"❌ Regular MISSNET failed: {e}")
    
    # Test 2: Skeleton-friendly MISSNET
    print("\n" + "="*50)
    print("TEST 2: Skeleton-Friendly MISSNET")
    print("="*50)
    
    try:
        imputed_friendly = missnet_impute_skeleton_friendly(
            data_with_missing,
            max_iteration=10,
            skeleton_sparsity_threshold=0.2,  # Very low threshold
            skeleton_min_samples=3,  # Very low sample requirement
            verbose=True
        )
        print("✅ Skeleton-friendly MISSNET completed")
    except Exception as e:
        print(f"❌ Skeleton-friendly MISSNET failed: {e}")
    
    # Test 3: Force skeleton activation
    print("\n" + "="*50)
    print("TEST 3: Force Skeleton Activation")
    print("="*50)
    
    try:
        force_skeleton_activation_config()
        imputed_forced = missnet_impute_skeleton_friendly(
            data_with_missing,
            max_iteration=10,
            skeleton_sparsity_threshold=0.1,  # Extremely low
            skeleton_min_samples=1,  # Minimum possible
            verbose=True
        )
        print("✅ Forced skeleton MISSNET completed")
    except Exception as e:
        print(f"❌ Forced skeleton MISSNET failed: {e}")
    
    print("\n" + "="*50)
    print("SKELETON FIXES COMPLETE")
    print("="*50)
    print("Compare the outputs to see which approach enabled skeleton activation!")
