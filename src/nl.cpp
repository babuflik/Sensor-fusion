#include "sensor_fusion/nl.h"
#include <random>
#include <cmath>

/**
 * @brief Simulate the nonlinear system
 */
Sig NL::simulate(const Matrix& u, const Vector& t, int MC) const {
    int nx = nn(0), nu = nn(1), ny = nn(2);
    
    // Determine time vector
    Vector time_vec;
    int N;
    
    if (t.size() > 0) {
        time_vec = t;
        N = t.size();
    } else if (u.cols() > 0) {
        N = u.cols();
        if (std::isnan(fs)) {
            throw std::runtime_error("NL::simulate: Need either time vector or fs");
        }
        time_vec = Vector::LinSpaced(N, 0, (N - 1) / fs);
    } else {
        throw std::runtime_error("NL::simulate: Need either u or t");
    }
    
    // Check input dimensions
    if (u.rows() > 0 && u.rows() != nu) {
        throw std::runtime_error("NL::simulate: u has wrong dimension");
    }
    
    Matrix u_actual = (u.rows() == 0) ? Matrix::Zero(nu, N) : u;
    
    // Allocate output
    Matrix y_sim(ny, N);
    Matrix x_sim(nx, N);
    
    // Single realization for now (MC > 1 would need multiple runs)
    Vector x = x0;
    if (px0) {
        x = px0->rand(1).row(0);
    }
    
    for (int k = 0; k < N; ++k) {
        double tk = time_vec(k);
        Vector uk = u_actual.col(k);
        
        // Measurement
        Vector yk = h(tk, x, uk, th);
        if (pe) {
            yk += pe->rand(1).row(0);
        }
        y_sim.col(k) = yk;
        x_sim.col(k) = x;
        
        // State update (if not last timestep)
        if (k < N - 1) {
            Vector xnext = f(tk, x, uk, th);
            if (pv) {
                xnext += pv->rand(1).row(0);
            }
            x = xnext;
        }
    }
    
    // Create Sig object
    Sig z;
    z.y = y_sim;
    z.t = time_vec;
    z.u = u_actual;
    z.x = x_sim;
    z.fs = fs;
    z.ylabel = ylabel;
    z.ulabel = ulabel;
    z.xlabel = xlabel;
    
    return z;
}

/**
 * @brief Extended Kalman Filter implementation
 */
std::pair<NL::Matrix, std::vector<NL::Matrix>> NL::ekf(
    const Sig& z, 
    const Matrix& u,
    const Vector& x0_init,
    const Matrix& P0_init) const {
    
    int nx = nn(0), nu = nn(1), ny = nn(2);
    int N = z.y.cols();
    
    // Initialize state and covariance
    Vector x_hat = (x0_init.size() > 0) ? x0_init : x0;
    Matrix P_hat;
    
    if (P0_init.rows() > 0) {
        P_hat = P0_init;
    } else if (px0) {
        P_hat = std::dynamic_pointer_cast<NDist>(px0)->P;
    } else {
        P_hat = Matrix::Identity(nx, nx);
    }
    
    // Get noise covariances
    Matrix Q = Matrix::Zero(nx, nx);
    Matrix R = Matrix::Zero(ny, ny);
    
    if (pv) {
        Q = std::dynamic_pointer_cast<NDist>(pv)->P;
    }
    if (pe) {
        R = std::dynamic_pointer_cast<NDist>(pe)->P;
    }
    
    // Input handling
    Matrix u_actual = (u.rows() == 0) ? Matrix::Zero(nu, N) : u;
    
    // Allocate output
    Matrix x_estimates(nx, N);
    std::vector<Matrix> P_estimates;
    P_estimates.reserve(N);
    
    for (int k = 0; k < N; ++k) {
        double tk = (z.t.size() > 0) ? z.t(k) : k / fs;
        Vector uk = u_actual.col(k);
        Vector yk = z.y.col(k);
        
        // Prediction step
        Vector x_pred = f(tk, x_hat, uk, th);
        Matrix F = compute_dfdx(tk, x_hat, uk, th);
        Matrix P_pred = F * P_hat * F.transpose() + Q;
        
        // Update step
        Vector y_pred = h(tk, x_pred, uk, th);
        Matrix H_k = compute_dhdx(tk, x_pred, uk, th);
        
        Vector innovation = yk - y_pred;
        Matrix S = H_k * P_pred * H_k.transpose() + R;
        Matrix K = P_pred * H_k.transpose() * S.inverse();
        
        x_hat = x_pred + K * innovation;
        P_hat = (Matrix::Identity(nx, nx) - K * H_k) * P_pred;
        
        // Store results
        x_estimates.col(k) = x_hat;
        P_estimates.push_back(P_hat);
    }
    
    return {x_estimates, P_estimates};
}

/**
 * @brief Unscented Kalman Filter implementation
 */
std::pair<NL::Matrix, std::vector<NL::Matrix>> NL::ukf(
    const Sig& z,
    const Matrix& u,
    const Vector& x0_init,
    const Matrix& P0_init) const {
    
    int nx = nn(0), nu = nn(1), ny = nn(2);
    int N = z.y.cols();
    
    // UKF parameters
    double alpha = 1e-3;
    double beta = 2.0;
    double kappa = 0.0;
    double lambda = alpha * alpha * (nx + kappa) - nx;
    
    // Initialize
    Vector x_hat = (x0_init.size() > 0) ? x0_init : x0;
    Matrix P_hat = (P0_init.rows() > 0) ? P0_init : 
                   (px0 ? std::dynamic_pointer_cast<NDist>(px0)->P : Matrix::Identity(nx, nx));
    
    Matrix Q = pv ? std::dynamic_pointer_cast<NDist>(pv)->P : Matrix::Zero(nx, nx);
    Matrix R = pe ? std::dynamic_pointer_cast<NDist>(pe)->P : Matrix::Zero(ny, ny);
    
    Matrix u_actual = (u.rows() == 0) ? Matrix::Zero(nu, N) : u;
    
    Matrix x_estimates(nx, N);
    std::vector<Matrix> P_estimates;
    P_estimates.reserve(N);
    
    // Weights
    int n_sigma = 2 * nx + 1;
    Vector weights_m(n_sigma), weights_c(n_sigma);
    weights_m(0) = lambda / (nx + lambda);
    weights_c(0) = lambda / (nx + lambda) + (1 - alpha * alpha + beta);
    for (int i = 1; i < n_sigma; ++i) {
        weights_m(i) = 1.0 / (2.0 * (nx + lambda));
        weights_c(i) = weights_m(i);
    }
    
    for (int k = 0; k < N; ++k) {
        double tk = (z.t.size() > 0) ? z.t(k) : k / fs;
        Vector uk = u_actual.col(k);
        Vector yk = z.y.col(k);
        
        // Generate sigma points
        Matrix L = sqrtcov(P_hat);
        Matrix sigma_points(nx, n_sigma);
        sigma_points.col(0) = x_hat;
        
        double scale = std::sqrt(nx + lambda);
        for (int i = 0; i < nx; ++i) {
            sigma_points.col(i + 1) = x_hat + scale * L.col(i);
            sigma_points.col(i + 1 + nx) = x_hat - scale * L.col(i);
        }
        
        // Predict sigma points
        Matrix sigma_pred(nx, n_sigma);
        for (int i = 0; i < n_sigma; ++i) {
            sigma_pred.col(i) = f(tk, sigma_points.col(i), uk, th);
        }
        
        // Predicted mean and covariance
        Vector x_pred = Vector::Zero(nx);
        for (int i = 0; i < n_sigma; ++i) {
            x_pred += weights_m(i) * sigma_pred.col(i);
        }
        
        Matrix P_pred = Q;
        for (int i = 0; i < n_sigma; ++i) {
            Vector diff = sigma_pred.col(i) - x_pred;
            P_pred += weights_c(i) * diff * diff.transpose();
        }
        
        // Predict measurements
        Matrix y_sigma(ny, n_sigma);
        for (int i = 0; i < n_sigma; ++i) {
            y_sigma.col(i) = h(tk, sigma_pred.col(i), uk, th);
        }
        
        Vector y_pred = Vector::Zero(ny);
        for (int i = 0; i < n_sigma; ++i) {
            y_pred += weights_m(i) * y_sigma.col(i);
        }
        
        // Innovation covariance and cross-covariance
        Matrix Pyy = R;
        Matrix Pxy = Matrix::Zero(nx, ny);
        for (int i = 0; i < n_sigma; ++i) {
            Vector dy = y_sigma.col(i) - y_pred;
            Vector dx = sigma_pred.col(i) - x_pred;
            Pyy += weights_c(i) * dy * dy.transpose();
            Pxy += weights_c(i) * dx * dy.transpose();
        }
        
        // Update
        Matrix K = Pxy * Pyy.inverse();
        x_hat = x_pred + K * (yk - y_pred);
        P_hat = P_pred - K * Pyy * K.transpose();
        
        x_estimates.col(k) = x_hat;
        P_estimates.push_back(P_hat);
    }
    
    return {x_estimates, P_estimates};
}

/**
 * @brief Particle Filter implementation
 */
NL::Matrix NL::pf(const Sig& z, const Matrix& u, int N_particles) const {
    int nx = nn(0), nu = nn(1), ny = nn(2);
    int N = z.y.cols();
    
    // Initialize particles
    Matrix particles(nx, N_particles);
    Vector weights = Vector::Constant(N_particles, 1.0 / N_particles);
    
    // Sample initial particles
    if (px0) {
        particles = px0->rand(N_particles).transpose();
    } else {
        for (int i = 0; i < N_particles; ++i) {
            particles.col(i) = x0;
        }
    }
    
    Matrix u_actual = (u.rows() == 0) ? Matrix::Zero(nu, N) : u;
    Matrix x_estimates(nx, N);
    
    // Get noise distributions
    std::shared_ptr<NDist> pv_ndist = pv ? std::dynamic_pointer_cast<NDist>(pv) : nullptr;
    std::shared_ptr<NDist> pe_ndist = pe ? std::dynamic_pointer_cast<NDist>(pe) : nullptr;
    
    for (int k = 0; k < N; ++k) {
        double tk = (z.t.size() > 0) ? z.t(k) : k / fs;
        Vector uk = u_actual.col(k);
        Vector yk = z.y.col(k);
        
        // Propagate particles
        Matrix new_particles(nx, N_particles);
        for (int i = 0; i < N_particles; ++i) {
            Vector x_i = particles.col(i);
            Vector x_next = f(tk, x_i, uk, th);
            
            if (pv_ndist) {
                x_next += pv_ndist->rand(1).row(0);
            }
            new_particles.col(i) = x_next;
        }
        
        // Update weights based on measurement likelihood
        Vector new_weights(N_particles);
        for (int i = 0; i < N_particles; ++i) {
            Vector y_pred = h(tk, new_particles.col(i), uk, th);
            
            if (pe_ndist) {
                // Evaluate likelihood p(y | x)
                NDist y_dist(y_pred, pe_ndist->P);
                Matrix yk_mat(1, ny);
                yk_mat.row(0) = yk.transpose();
                new_weights(i) = weights(i) * y_dist.pdf(yk_mat)(0);
            } else {
                new_weights(i) = weights(i);
            }
        }
        
        // Normalize weights
        double sum_weights = new_weights.sum();
        if (sum_weights > 0) {
            new_weights /= sum_weights;
        } else {
            new_weights.setConstant(1.0 / N_particles);
        }
        
        // Estimate (weighted mean)
        Vector x_est = Vector::Zero(nx);
        for (int i = 0; i < N_particles; ++i) {
            x_est += new_weights(i) * new_particles.col(i);
        }
        x_estimates.col(k) = x_est;
        
        // Resample if effective sample size is too low
        double ess = 1.0 / new_weights.squaredNorm();
        if (ess < N_particles / 2.0) {
            // Systematic resampling
            Matrix resampled(nx, N_particles);
            Vector cumsum = Vector::Zero(N_particles);
            cumsum(0) = new_weights(0);
            for (int i = 1; i < N_particles; ++i) {
                cumsum(i) = cumsum(i - 1) + new_weights(i);
            }
            
            std::random_device rd;
            std::mt19937 gen(rd());
            std::uniform_real_distribution<> dis(0.0, 1.0 / N_particles);
            
            double u0 = dis(gen);
            int j = 0;
            for (int i = 0; i < N_particles; ++i) {
                double u_i = u0 + i / static_cast<double>(N_particles);
                while (j < N_particles - 1 && u_i > cumsum(j)) {
                    j++;
                }
                resampled.col(i) = new_particles.col(j);
            }
            
            particles = resampled;
            weights.setConstant(1.0 / N_particles);
        } else {
            particles = new_particles;
            weights = new_weights;
        }
    }
    
    return x_estimates;
}

/**
 * @brief Cramér-Rao Lower Bound computation
 */
Sig NL::crlb(const Sig& z,
             const Matrix& P0_init,
             const Matrix& Q_override,
             const Matrix& R_override,
             int pred_horizon) const {
    
    if (std::isnan(fs)) {
        throw std::runtime_error("NL::crlb: only discrete time models supported");
    }
    
    if (z.x.cols() == 0) {
        throw std::runtime_error("NL::crlb: input signal z must have state x");
    }
    
    if (pred_horizon != 0 && pred_horizon != 1) {
        throw std::runtime_error("NL::crlb: pred_horizon must be 0 or 1");
    }
    
    int nx = nn(0), nu = nn(1), ny = nn(2);
    int N = z.y.cols();
    
    // Get noise covariances
    Matrix Q;
    if (Q_override.rows() > 0) {
        Q = Q_override;
    } else if (pv) {
        Q = std::dynamic_pointer_cast<NDist>(pv)->P;
    } else {
        Q = Matrix::Zero(nx, nx);
    }
    
    Matrix R;
    if (R_override.rows() > 0) {
        R = R_override;
    } else if (pe) {
        R = std::dynamic_pointer_cast<NDist>(pe)->P;
    } else {
        throw std::runtime_error("NL::crlb: measurement noise covariance R must be defined");
    }
    
    if (R.diagonal().sum() == 0) {
        throw std::runtime_error("NL::crlb: R must be positive definite");
    }
    
    // Initialize P
    Matrix P;
    if (P0_init.rows() > 0) {
        P = P0_init;
    } else if (px0) {
        P = std::dynamic_pointer_cast<NDist>(px0)->P;
    } else {
        P = 1000.0 * Matrix::Identity(nx, nx);
    }
    
    // Storage for results
    Matrix x_filter(nx, N);
    Matrix x_pred(nx, N);
    std::vector<Matrix> P_filter, P_pred;
    Matrix y_filter(ny, N);
    Matrix y_pred(ny, N);
    std::vector<Matrix> Py_filter, Py_pred;
    
    Matrix I_nx = Matrix::Identity(nx, nx);
    
    for (int k = 0; k < N; ++k) {
        // Get input
        Vector u_k;
        if (z.u.cols() > k) {
            u_k = z.u.col(k);
        } else {
            u_k = Vector::Zero(nu);
        }
        double t_k = z.t(k);
        
        // Use true state from z.x for linearization
        Vector x_true = z.x.col(k);
        
        // Measurement update (using true state)
        Vector y_hat = h(t_k, x_true, u_k, th);
        
        // Compute measurement Jacobian C = dh/dx at true state
        Matrix C = compute_dhdx(t_k, x_true, u_k, th);
        
        // Kalman gain
        Matrix S = C * P * C.transpose() + R;
        Matrix K = P * C.transpose() * S.inverse();
        
        // Update covariance using Joseph form for numerical stability
        Matrix I_KC = I_nx - K * C;
        P = I_KC * P * I_KC.transpose() + K * R * K.transpose();
        P = 0.5 * (P + P.transpose());  // Ensure symmetry
        
        // Store filter results
        x_filter.col(k) = x_true;
        P_filter.push_back(P);
        y_filter.col(k) = y_hat;
        Py_filter.push_back(C * P * C.transpose() + R);
        
        // Time update (prediction for next step)
        Matrix A = compute_dfdx(t_k, x_true, u_k, th);
        P = A * P * A.transpose() + Q;
        
        // Store prediction results
        x_pred.col(k) = x_true;
        y_pred.col(k) = y_hat;
        P_pred.push_back(P);
        Py_pred.push_back(C * P * C.transpose() + R);
    }
    
    // Create output signal
    Sig result;
    result.t = z.t;
    result.u = z.u;
    result.fs = z.fs;
    result.xlabel = xlabel;
    result.ylabel = ylabel;
    result.ulabel = ulabel;
    result.name = name;
    
    if (pred_horizon == 0) {
        // Filter mode
        result.y = y_filter;
        result.x = x_filter;
        result.Py = Py_filter;
        result.Px = P_filter;
    } else {
        // Predictor mode
        result.y = y_pred;
        result.x = x_pred;
        result.Py = Py_pred;
        result.Px = P_pred;
    }
    
    return result;
}

