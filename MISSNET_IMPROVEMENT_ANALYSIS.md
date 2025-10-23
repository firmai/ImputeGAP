# MISSNET++ Implementation Analysis & Improvement Plan

## Test Results Summary

### Current Performance Issues:

1. **Huber Loss**: Showing -4.4% to -27% improvements (WORSE)
   - Issue: Only partially integrated
   - Impact: Not being applied consistently across likelihood computations

2. **Cholesky Optimization**: Only 1.01-1.13× speedup
   - Issue: Wrapper overhead negates benefits on small matrices
   - Impact: Minimal speedup, sometimes slower

3. **Skeleton Graph**: Inconsistent results (0.89-1.02× speedup)
   - Issue: Alpha parameter too aggressive, poor scaling
   - Impact: Sometimes slower than TVGL

4. **Overall**: Combined enhancements show 1.16× speedup but -27% accuracy loss
   - Issue: Enhancements working against each other
   - Impact: Not ready for production

## Root Cause Analysis

### 1. Huber Loss Integration Problems

**Current Implementation:**
- Only in `update_observation_covariance` (line 1206-1230)
- NOT in `forward_viterbi` Kalman filter (line 529-592)
- NOT in `update_object_latent_matrix` (line 746-776)

**Why It's Failing:**
```python
# In forward_viterbi (line 569-572), still using squared loss:
delta = xt - Ht @ mu_tt1[i, j, t]
sigma = Ht @ psi_tt1[i, j, t] @ Ht.T + self.sgmX[i] * It
inv_sigma = np.linalg.pinv(sigma)
df = delta @ inv_sigma @ delta / 2  # <-- SQUARED LOSS
```

**Fix Required:**
- Replace squared loss with Huber in ALL likelihood computations
- Add IRLS reweighting in parameter updates
- Consistent delta computation across E-step and M-step

### 2. Cholesky Optimization Overhead

**Current Implementation:**
```python
def _chol_solve(A, B, jitter=1e-6):
    A = A + jitter * np.eye(A.shape[0])  # <-- OVERHEAD: Creating eye each call
    if spla is not None:
        L = np.linalg.cholesky(A)
        return spla.cho_solve((L, True), B)
    return np.linalg.lstsq(A, B, rcond=None)[0]
```

**Problems:**
1. Creates identity matrix every call (expensive for small matrices)
2. No size threshold (overhead > benefit for N < 20)
3. No caching of decompositions
4. Fallback to lstsq adds branching overhead

**Fix Required:**
- Add size threshold: only use Cholesky for N > 20
- Cache identity matrices
- Remove spla checking overhead
- Pre-compute decompositions where possible

### 3. Skeleton Graph Poor Integration

**Current Issues:**
1. `LASSO_ALPHA=0.02` too aggressive for most data
2. No warm-starting from previous iteration
3. Standardization happening twice (MB + normalize_precision)
4. No sparsity benefit because TVGL also uses beta=0

**Fix Required:**
- Adaptive alpha based on data sparsity: `alpha = 0.01 * sqrt(log(N) / T)`
- Keep precision matrix between iterations for warm-starting
- Skip skeleton if true sparsity > 0.3 (use TVGL directly)
- Add sparsity threshold parameter

### 4. Incomplete Vectorization

**Still Using Loops:**
```python
# Line 749-760: update_object_latent_matrix
for i in range(self.N):
    A1 = self.alpha / self.sgmS[k] * sum(...)  # <-- PYTHON LOOP
    A1 += (1 - self.alpha) / self.sgmX[k] * sum(...)  # <-- PYTHON LOOP
```

**Fix Required:**
- Vectorize network term: `A1_network = (alpha/sgmS) * (S @ v.T).T`
- Vectorize temporal term using broadcasting
- Batch solve for all features simultaneously

## Priority Fixes (High Impact, Low Risk)

### Fix 1: Add Size Threshold to Cholesky (5 min)

```python
def _chol_solve(A, B, jitter=1e-6, size_threshold=20):
    """Solve with automatic method selection."""
    # For small matrices, pinv is faster due to overhead
    if A.shape[0] < size_threshold:
        return np.linalg.pinv(A + jitter * np.eye(A.shape[0])) @ B
    
    # For large matrices, Cholesky is much faster
    try:
        A_reg = A + jitter * np.eye(A.shape[0])
        L = np.linalg.cholesky(A_reg)
        y = np.linalg.solve(L, B)
        x = np.linalg.solve(L.T, y)
        return x
    except np.linalg.LinAlgError:
        return np.linalg.pinv(A + jitter * np.eye(A.shape[0])) @ B
```

**Expected Impact:** 1.5-2× speedup on large matrices, no slowdown on small

### Fix 2: Adaptive Lasso Alpha (10 min)

```python
def adaptive_lasso_alpha(X, base_alpha=0.01):
    """Compute adaptive alpha based on data characteristics."""
    T, N = X.shape
    # Estimate sparsity from correlation matrix
    corr = np.corrcoef(X, rowvar=False)
    sparsity = (np.abs(corr) < 0.1).sum() / (N * N)
    
    # Adaptive formula
    alpha = base_alpha * np.sqrt(np.log(N) / T)
    
    # Adjust based on estimated sparsity
    if sparsity > 0.7:  # Very sparse
        alpha *= 0.5
    elif sparsity < 0.3:  # Dense
        alpha *= 2.0
    
    return alpha
```

**Expected Impact:** 30-50% better accuracy on varied data types

### Fix 3: Skip Skeleton for Dense Graphs (5 min)

```python
def update_networks(self, X, W):
    """Smart selection between skeleton and TVGL."""
    for k in range(self.n_cl):
        F_k = np.where(self.F == k)[0]
        if len(F_k) == 0: continue
        
        # Estimate sparsity
        if hasattr(self, 'H') and self.H[k] is not None:
            est_sparsity = (np.abs(self.H[k]) < 0.01).sum() / (self.N ** 2)
        else:
            est_sparsity = 0.5
        
        # Use skeleton only for sparse graphs
        if self.use_skeleton and est_sparsity > 0.6:
            # Fast path: nodewise skeleton
            X_std, mu, sd = _standardize(X_regime)
            alpha = adaptive_lasso_alpha(X_std)
            H_new, A_mask = nodewise_skeleton_and_precision(X_std, alpha=alpha)
            self.H[k] = H_new
        else:
            # Dense path: TVGL
            test = TVGL(alpha=self.beta, beta=0, max_iter=1000)
            test.fit(X_regime, np.zeros(X_regime.shape[0]))
            self.H[k] = test.precision_[0]
```

**Expected Impact:** 2-5× speedup on truly sparse data, no slowdown on dense

### Fix 4: Fix Huber Loss in Forward Pass (30 min)

This requires modifying the Kalman filter to use robust likelihoods. More complex but necessary.

## Recommended Implementation Order

1. **Week 1: Quick Wins (2-3 hours)**
   - [ ] Add size threshold to Cholesky (Fix 1)
   - [ ] Skip skeleton for dense graphs (Fix 3)
   - [ ] Add adaptive Lasso alpha (Fix 2)
   - [ ] Test on ablation suite

2. **Week 2: Huber Integration (4-6 hours)**
   - [ ] Add Huber to forward_viterbi
   - [ ] Add IRLS reweighting to M-step
   - [ ] Consistent robust likelihoods
   - [ ] Test on outlier data

3. **Week 3: Vectorization (4-6 hours)**
   - [ ] Vectorize update_object_latent_matrix
   - [ ] Batch operations in forward pass
   - [ ] Optimize memory allocation
   - [ ] Performance benchmarks

4. **Week 4: Advanced Features (8-10 hours)**
   - [ ] Spectral features (seasonal component)
   - [ ] Uncertainty quantification
   - [ ] Consistency regularization
   - [ ] Full integration testing

## Expected Results After Fixes

| Enhancement | Current | After Fixes | Target |
|------------|---------|-------------|--------|
| Robust Loss | -27% | +15-25% | +20% |
| Cholesky | 1.01× | 1.5-2× | 2× |
| Skeleton | 0.89× | 2-5× | 3× |
| Combined | -27% acc, 1.16× speed | +20% acc, 3× speed | +25% acc, 5× speed |

## Testing Strategy

1. **Unit Tests**: Each fix tested independently
2. **Ablation Study**: Compare all combinations
3. **Stress Tests**: Extreme cases (high missingness, outliers, sparse/dense)
4. **Benchmark Suite**: Standard datasets with known ground truth

## Conclusion

The current implementation has the RIGHT IDEAS but WRONG EXECUTION. The enhancements are:
- Partially implemented (Huber only in M-step)
- Not optimized for common cases (Cholesky overhead)
- Using fixed parameters (Lasso alpha too aggressive)
- Missing key optimizations (vectorization)

With these targeted fixes, we can achieve the promised 5× speedup and 25% accuracy improvement.
