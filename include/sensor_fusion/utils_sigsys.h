#pragma once
#include <Eigen/Dense>
#include <functional>
#include <vector>
#include <cmath>

/**
 * @brief Numerical gradient computation
 * C++ implementation of MATLAB numgrad.m
 * 
 * @param f Function to differentiate
 * @param x Point at which to compute gradient
 * @param h Step size (default: sqrt(eps))
 * @return Gradient vector
 */
template<typename Func>
Eigen::VectorXd numgrad(Func f, const Eigen::VectorXd& x, double h = 0) {
    int n = x.size();
    
    if (h == 0) {
        h = std::sqrt(std::numeric_limits<double>::epsilon());
    }
    
    Eigen::VectorXd grad(n);
    Eigen::VectorXd x_plus = x;
    Eigen::VectorXd x_minus = x;
    
    for (int i = 0; i < n; ++i) {
        x_plus(i) = x(i) + h;
        x_minus(i) = x(i) - h;
        
        double f_plus = f(x_plus);
        double f_minus = f(x_minus);
        
        grad(i) = (f_plus - f_minus) / (2.0 * h);
        
        x_plus(i) = x(i);
        x_minus(i) = x(i);
    }
    
    return grad;
}

/**
 * @brief Numerical Jacobian computation (vector function)
 * Computes Jacobian matrix J where J(i,j) = ∂f_i/∂x_j
 * 
 * @param f Vector function f: R^n -> R^m
 * @param x Point at which to compute Jacobian
 * @param h Step size (default: sqrt(eps))
 * @return Jacobian matrix (m x n)
 */
template<typename Func>
Eigen::MatrixXd numjac(Func f, const Eigen::VectorXd& x, double h = 0) {
    int n = x.size();
    
    if (h == 0) {
        h = std::sqrt(std::numeric_limits<double>::epsilon());
    }
    
    // Evaluate at x to get output dimension
    Eigen::VectorXd f0 = f(x);
    int m = f0.size();
    
    Eigen::MatrixXd J(m, n);
    Eigen::VectorXd x_plus = x;
    Eigen::VectorXd x_minus = x;
    
    for (int j = 0; j < n; ++j) {
        x_plus(j) = x(j) + h;
        x_minus(j) = x(j) - h;
        
        Eigen::VectorXd f_plus = f(x_plus);
        Eigen::VectorXd f_minus = f(x_minus);
        
        J.col(j) = (f_plus - f_minus) / (2.0 * h);
        
        x_plus(j) = x(j);
        x_minus(j) = x(j);
    }
    
    return J;
}

/**
 * @brief Numerical Hessian computation
 * C++ implementation of MATLAB numhess.m
 * 
 * @param f Function to differentiate
 * @param x Point at which to compute Hessian
 * @param h Step size (default: eps^(1/3))
 * @return Hessian matrix
 */
template<typename Func>
Eigen::MatrixXd numhess(Func f, const Eigen::VectorXd& x, double h = 0) {
    int n = x.size();
    
    if (h == 0) {
        h = std::pow(std::numeric_limits<double>::epsilon(), 1.0 / 3.0);
    }
    
    Eigen::MatrixXd hess(n, n);
    Eigen::VectorXd x_temp = x;
    
    // Diagonal elements
    for (int i = 0; i < n; ++i) {
        x_temp(i) = x(i) + h;
        double f_plus = f(x_temp);
        
        x_temp(i) = x(i) - h;
        double f_minus = f(x_temp);
        
        double f_center = f(x);
        
        hess(i, i) = (f_plus - 2.0 * f_center + f_minus) / (h * h);
        
        x_temp(i) = x(i);
    }
    
    // Off-diagonal elements
    for (int i = 0; i < n; ++i) {
        for (int j = i + 1; j < n; ++j) {
            x_temp(i) = x(i) + h;
            x_temp(j) = x(j) + h;
            double f_pp = f(x_temp);
            
            x_temp(j) = x(j) - h;
            double f_pm = f(x_temp);
            
            x_temp(i) = x(i) - h;
            x_temp(j) = x(j) + h;
            double f_mp = f(x_temp);
            
            x_temp(j) = x(j) - h;
            double f_mm = f(x_temp);
            
            hess(i, j) = (f_pp - f_pm - f_mp + f_mm) / (4.0 * h * h);
            hess(j, i) = hess(i, j);
            
            x_temp(i) = x(i);
            x_temp(j) = x(j);
        }
    }
    
    return hess;
}

/**
 * @brief Compute square root of covariance matrix (Cholesky decomposition)
 * C++ implementation of MATLAB sqrtcov.m
 * 
 * @param P Covariance matrix
 * @return Lower triangular matrix L such that P = L * L^T
 */
inline Eigen::MatrixXd sqrtcov(const Eigen::MatrixXd& P) {
    if (P.rows() != P.cols()) {
        throw std::runtime_error("sqrtcov: P must be square");
    }
    
    Eigen::LLT<Eigen::MatrixXd> llt(P);
    if (llt.info() != Eigen::Success) {
        throw std::runtime_error("sqrtcov: P is not positive definite");
    }
    
    return llt.matrixL();
}

/**
 * @brief Enhanced covariance matrix validation
 * C++ implementation of MATLAB iscov.m
 * 
 * @param P Matrix to check
 * @return pair<bool, int> - (is_valid, error_code)
 *         error_code: 0=valid, 1=not real, 2=not square, 3=not symmetric, 4=not PSD
 */
inline std::pair<bool, int> iscov(const Eigen::MatrixXd& P) {
    // Check 1: Real matrix
    if (!P.array().isFinite().all()) {
        return {false, 1};
    }
    
    // Check 2: Square
    if (P.rows() != P.cols()) {
        return {false, 2};
    }
    
    // Check 3: Symmetric
    if (!P.isApprox(P.transpose(), 1e-10)) {
        return {false, 3};
    }
    
    // Check 4: Positive semi-definite
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es(P);
    if (es.eigenvalues().minCoeff() < -1e-10) {
        return {false, 4};
    }
    
    return {true, 0};
}

/**
 * @brief Compute confidence ellipse points
 * C++ implementation of MATLAB confellipse.m
 * 
 * @param P 2x2 covariance matrix
 * @param center Center point (2D)
 * @param conf Confidence level (0-1), default 0.95 for 95%
 * @param npoints Number of points on ellipse, default 100
 * @return Matrix of ellipse points (npoints x 2)
 */
inline Eigen::MatrixXd confellipse(const Eigen::Matrix2d& P, 
                                   const Eigen::Vector2d& center,
                                   double conf = 0.95,
                                   int npoints = 100) {
    if (P.rows() != 2 || P.cols() != 2) {
        throw std::runtime_error("confellipse: P must be 2x2");
    }
    
    // Chi-squared value for 2 DOF at confidence level
    // For 95%: chi2 ≈ 5.991
    double chi2_val = -2.0 * std::log(1.0 - conf);
    double scale = std::sqrt(chi2_val);
    
    // Eigen decomposition
    Eigen::SelfAdjointEigenSolver<Eigen::Matrix2d> es(P);
    Eigen::Vector2d eigenvals = es.eigenvalues();
    Eigen::Matrix2d eigenvecs = es.eigenvectors();
    
    // Generate ellipse in standard form (closed curve: first = last)
    Eigen::MatrixXd ellipse(npoints + 1, 2);
    for (int i = 0; i <= npoints; ++i) {
        double theta = 2.0 * M_PI * i / npoints;
        double x = scale * std::sqrt(eigenvals(0)) * std::cos(theta);
        double y = scale * std::sqrt(eigenvals(1)) * std::sin(theta);
        
        Eigen::Vector2d point(x, y);
        point = eigenvecs * point + center;
        
        ellipse.row(i) = point.transpose();
    }
    
    return ellipse;
}

/**
 * @brief Get standard filter by name
 * C++ implementation of MATLAB getfilter.m (simplified)
 * 
 * @param name Filter name ("butterworth", "chebyshev", "moving_average")
 * @param order Filter order
 * @param cutoff Cutoff frequency (normalized 0-1)
 * @return Filter coefficients (b, a)
 */
inline std::pair<Eigen::VectorXd, Eigen::VectorXd> getfilter(
    const std::string& name, int order, double cutoff) {
    
    Eigen::VectorXd b, a;
    
    if (name == "moving_average") {
        // Simple moving average filter
        b = Eigen::VectorXd::Constant(order, 1.0 / order);
        a = Eigen::VectorXd::Constant(1, 1.0);
    } else if (name == "low_pass") {
        // Simple low-pass filter (exponential smoothing)
        double alpha = cutoff;
        b = Eigen::VectorXd(1);
        b(0) = alpha;
        a = Eigen::VectorXd(2);
        a(0) = 1.0;
        a(1) = -(1.0 - alpha);
    } else {
        throw std::runtime_error("getfilter: Filter type not yet implemented");
    }
    
    return {b, a};
}

/**
 * @brief Get window function
 * C++ implementation of MATLAB getwindow.m
 * 
 * @param name Window name ("hamming", "hann", "blackman", "rectangular")
 * @param N Window length
 * @return Window coefficients
 */
inline Eigen::VectorXd getwindow(const std::string& name, int N) {
    Eigen::VectorXd window(N);
    
    if (name == "hamming") {
        for (int n = 0; n < N; ++n) {
            window(n) = 0.54 - 0.46 * std::cos(2.0 * M_PI * n / (N - 1));
        }
    } else if (name == "hann") {
        for (int n = 0; n < N; ++n) {
            window(n) = 0.5 * (1.0 - std::cos(2.0 * M_PI * n / (N - 1)));
        }
    } else if (name == "blackman") {
        for (int n = 0; n < N; ++n) {
            window(n) = 0.42 - 0.5 * std::cos(2.0 * M_PI * n / (N - 1)) +
                       0.08 * std::cos(4.0 * M_PI * n / (N - 1));
        }
    } else if (name == "rectangular") {
        window.setOnes();
    } else {
        throw std::runtime_error("getwindow: Window type not recognized");
    }
    
    return window;
}

/**
 * @brief Matrix condition number
 */
inline double condition(const Eigen::MatrixXd& A) {
    Eigen::JacobiSVD<Eigen::MatrixXd> svd(A);
    double cond = svd.singularValues()(0) / 
                  svd.singularValues()(svd.singularValues().size()-1);
    return cond;
}

/**
 * @brief Zero-phase digital filtering
 * C++ implementation of MATLAB filtfilt - simplified version
 * 
 * Applies forward and backward filtering to eliminate phase distortion.
 * This is a simplified implementation for IIR filters.
 * 
 * @param b Numerator coefficients (feedforward)
 * @param a Denominator coefficients (feedback)
 * @param x Input signal
 * @return Zero-phase filtered signal
 */
inline Eigen::VectorXd filtfilt(const Eigen::VectorXd& b, const Eigen::VectorXd& a, 
                                const Eigen::VectorXd& x) {
    int N = x.size();
    int nb = b.size();
    int na = a.size();
    
    if (N == 0) {
        return Eigen::VectorXd();
    }
    
    // Normalize by a(0)
    Eigen::VectorXd b_norm = b / a(0);
    Eigen::VectorXd a_norm = a / a(0);
    
    // Forward filtering
    Eigen::VectorXd y_forward = Eigen::VectorXd::Zero(N);
    
    for (int n = 0; n < N; ++n) {
        double sum = 0.0;
        
        // Feedforward (numerator)
        for (int k = 0; k < nb && k <= n; ++k) {
            sum += b_norm(k) * x(n - k);
        }
        
        // Feedback (denominator) - skip a(0) which is 1 after normalization
        for (int k = 1; k < na && k <= n; ++k) {
            sum -= a_norm(k) * y_forward(n - k);
        }
        
        y_forward(n) = sum;
    }
    
    // Reverse the forward filtered signal
    Eigen::VectorXd y_forward_rev(N);
    for (int i = 0; i < N; ++i) {
        y_forward_rev(i) = y_forward(N - 1 - i);
    }
    
    // Backward filtering (on reversed signal)
    Eigen::VectorXd y_backward = Eigen::VectorXd::Zero(N);
    
    for (int n = 0; n < N; ++n) {
        double sum = 0.0;
        
        // Feedforward
        for (int k = 0; k < nb && k <= n; ++k) {
            sum += b_norm(k) * y_forward_rev(n - k);
        }
        
        // Feedback
        for (int k = 1; k < na && k <= n; ++k) {
            sum -= a_norm(k) * y_backward(n - k);
        }
        
        y_backward(n) = sum;
    }
    
    // Reverse back to get final result
    Eigen::VectorXd result(N);
    for (int i = 0; i < N; ++i) {
        result(i) = y_backward(N - 1 - i);
    }
    
    return result;
}

/**
 * @brief Linear interpolation
 * C++ implementation of MATLAB interp - simplified version using linear interpolation
 * 
 * @param y1 Input signal values
 * @param t1 Input time points
 * @param t2 Output time points (where to interpolate)
 * @return Interpolated signal values at t2
 */
inline Eigen::VectorXd interp(const Eigen::VectorXd& y1, const Eigen::VectorXd& t1, 
                              const Eigen::VectorXd& t2) {
    int N1 = y1.size();
    int N2 = t2.size();
    
    if (N1 != t1.size()) {
        throw std::runtime_error("interp: y1 and t1 must have same length");
    }
    
    if (N1 < 2) {
        throw std::runtime_error("interp: need at least 2 points for interpolation");
    }
    
    Eigen::VectorXd y2(N2);
    
    for (int i = 0; i < N2; ++i) {
        double t = t2(i);
        
        // Find the interval [t1(j), t1(j+1)] containing t
        if (t <= t1(0)) {
            // Extrapolate or use first value
            y2(i) = y1(0);
        } else if (t >= t1(N1 - 1)) {
            // Extrapolate or use last value
            y2(i) = y1(N1 - 1);
        } else {
            // Find the right interval
            int j = 0;
            for (j = 0; j < N1 - 1; ++j) {
                if (t >= t1(j) && t <= t1(j + 1)) {
                    break;
                }
            }
            
            // Linear interpolation
            double alpha = (t - t1(j)) / (t1(j + 1) - t1(j));
            y2(i) = (1.0 - alpha) * y1(j) + alpha * y1(j + 1);
        }
    }
    
    return y2;
}

/**
 * @brief Signal resampling
 * Resamples signal from sampling rate fs1 to fs2
 * 
 * @param y Input signal
 * @param fs1 Original sampling frequency
 * @param fs2 Target sampling frequency
 * @return Resampled signal
 */
inline Eigen::VectorXd resample(const Eigen::VectorXd& y, double fs1, double fs2) {
    int N1 = y.size();
    
    if (fs1 <= 0 || fs2 <= 0) {
        throw std::runtime_error("resample: sampling frequencies must be positive");
    }
    
    if (N1 == 0) {
        return Eigen::VectorXd();
    }
    
    // Create original time vector
    Eigen::VectorXd t1(N1);
    for (int i = 0; i < N1; ++i) {
        t1(i) = i / fs1;
    }
    
    // Create new time vector with target sampling rate
    double duration = (N1 - 1) / fs1;
    int N2 = static_cast<int>(std::ceil(duration * fs2)) + 1;
    
    Eigen::VectorXd t2(N2);
    for (int i = 0; i < N2; ++i) {
        t2(i) = i / fs2;
    }
    
    // Use interpolation to resample
    return interp(y, t1, t2);
}

/**
 * @brief Downsample signal by integer factor
 * Simple downsampling by taking every M-th sample
 * 
 * @param y Input signal
 * @param M Downsampling factor
 * @return Downsampled signal
 */
inline Eigen::VectorXd downsample(const Eigen::VectorXd& y, int M) {
    if (M <= 0) {
        throw std::runtime_error("downsample: M must be positive");
    }
    
    if (M == 1) {
        return y;
    }
    
    int N = y.size();
    int N_new = (N + M - 1) / M;  // Ceiling division
    
    Eigen::VectorXd y_down(N_new);
    
    for (int i = 0; i < N_new; ++i) {
        int idx = i * M;
        if (idx < N) {
            y_down(i) = y(idx);
        }
    }
    
    return y_down;
}

/**
 * @brief Upsample signal by integer factor
 * Inserts zeros between samples (needs filtering for proper interpolation)
 * 
 * @param y Input signal
 * @param L Upsampling factor
 * @return Upsampled signal (with zeros inserted)
 */
inline Eigen::VectorXd upsample(const Eigen::VectorXd& y, int L) {
    if (L <= 0) {
        throw std::runtime_error("upsample: L must be positive");
    }
    
    if (L == 1) {
        return y;
    }
    
    int N = y.size();
    int N_new = N * L;
    
    Eigen::VectorXd y_up = Eigen::VectorXd::Zero(N_new);
    
    for (int i = 0; i < N; ++i) {
        y_up(i * L) = y(i);
    }
    
    return y_up;
}

