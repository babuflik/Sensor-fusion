# Comprehensive Estimator Test Suite

## Overview

`test_all_estimators.cpp` provides a comprehensive test suite that evaluates all available estimators (LS, WLS, ML, CRLB, EKF, and UKF) using the same microphone configuration. This ensures consistent, fair comparison across all estimation methods.

## Test Coverage

### 1. **Least Squares (LS) Estimator**
- Tests basic LS estimation
- Verifies covariance computation
- Checks estimate accuracy

### 2. **Weighted Least Squares (WLS) Estimator**
- Tests WLS with measurement noise weighting
- Compares performance with LS

### 3. **Maximum Likelihood (ML) Estimator**
- Tests grid-based ML search
- Evaluates likelihood function computation
- Note: Uses coarse grid for computational efficiency

### 4. **Cramér-Rao Lower Bound (CRLB)**
- Computes theoretical minimum achievable variance
- Validates positive definiteness of covariance
- Provides performance baseline for comparison

### 5. **Extended Kalman Filter (EKF)**
- Tests recursive Bayesian estimation with linearization
- Evaluates performance with nonlinear measurement model
- Computes and validates posterior covariance

### 6. **Unscented Kalman Filter (UKF)**
- Tests sigma-point based nonlinear filtering
- Compares with EKF performance
- Validates covariance propagation

### 7. **Estimator Comparison**
- Runs all estimators on same data (including filters)
- Compares estimation errors across LS, WLS, ML, EKF, UKF
- Relates performance to CRLB

### 8. **Multi-Sample Time Series**
- Tests estimators with multiple measurements over time
- Demonstrates CRLB improvement with more samples
- Validates tracking of moving source

### 9. **Robustness to Initial Guess**
- Tests with various initial conditions
- Evaluates convergence properties
- Important for iterative methods (LS, WLS)

### 10. **Different Microphone Configurations**
- Linear array (poor 3D geometry)
- Square planar array (better geometry)
- Tetrahedron array (optimal 3D geometry)
- Demonstrates impact of sensor placement

### 11. **1D and 2D Likelihood Functions**
- Tests `lh1()` for 1D likelihood profiles
- Tests `lh2()` for 2D likelihood surfaces
- Validates maximum likelihood detection

### 12. **CRLB Grid Evaluation**
- Evaluates CRLB across spatial grid
- Tests different metrics (trace, RMSE, determinant, max eigenvalue)
- Identifies regions of best/worst estimation performance

## Microphone Configuration

The test uses a 3D tetrahedron microphone array:
```cpp
// 4 microphones in tetrahedron formation
mic 1: (0.00,  0.00,  0.00)  // origin
mic 2: (0.15,  0.00,  0.00)  // 15 cm on x-axis
mic 3: (0.075, 0.13,  0.00)  // forms triangle base
mic 4: (0.075, 0.065, 0.12)  // apex (12 cm up)
```

Ground truth source: `(1.2, 0.5, 0.3)` meters

## Sensor Model

- **State**: 3D position `(x, y, z)`
- **Measurements**: Time-of-arrival (TOA) at each microphone
- **Measurement noise**: 1 sample at 48 kHz (~20 μs)

## Building and Running

### Build the test:
```bash
cd build
cmake ..
make test_all_estimators
```

### Run the test:
```bash
./test_all_estimators
```

### Run via CTest:
```bash
ctest -R AllEstimatorsTest -V
```

## Expected Results

All tests should pass with the following typical performance:

| Estimator | RMSE (meters) | Notes |
|-----------|---------------|-------|
| LS        | ~0.012        | Fast, good with reasonable initial guess |
| WLS       | ~0.012        | Similar to LS with uniform noise |
| ML        | ~0.28         | Grid search - coarse for speed |
| EKF       | ~0.007        | Excellent for this nearly-linear problem |
| UKF       | ~0.023        | Good, slightly worse than EKF here |
| CRLB      | N/A           | Theoretical minimum variance bound |

### Key Observations

1. **EKF** achieves best performance (~0.007m) for this nearly-linear problem
2. **LS and WLS** perform very well (~0.012m) when initialized near the true position
3. **UKF** performs well (~0.023m), slightly worse than EKF for this case
4. **ML** performance depends on grid resolution (trade-off: accuracy vs. speed)
5. **CRLB** shows theoretical minimum is ~0.12m RMSE with this configuration
6. **Geometry matters**: tetrahedron > square > linear for 3D localization
7. **More samples** improve performance (CRLB trace decreases)

## Implementation Details

### Test Fixture
All tests inherit from `AllEstimatorsTest` which provides:
- Consistent microphone configuration
- Ground truth source position
- Synthetic measurements
- Helper functions for printing results

### Measurement Generation
Simulates perfect time-of-arrival measurements with small additive noise to represent realistic conditions.

### Comparison Metrics
- **RMSE**: Root mean square error in 3D position
- **Covariance trace**: Sum of variances (related to RMSE²)
- **Covariance determinant**: Volume of uncertainty ellipsoid
- **Max eigenvalue**: Maximum variance along any direction

## Future Enhancements

Potential additions to the test suite:
- [x] Extended Kalman Filter (EKF) testing
- [x] Unscented Kalman Filter (UKF) testing
- [ ] Particle filter comparison
- [ ] Different noise models
- [ ] Outlier/robustness testing
- [ ] Real audio data integration
- [ ] Computational performance benchmarking
