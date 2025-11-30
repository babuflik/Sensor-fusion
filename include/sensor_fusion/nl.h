#ifndef NL_H
#define NL_H

#include "sig.h"
#include "ndist.h"
#include "pdfclass.h"
#include "utils_sigsys.h"
#include <Eigen/Dense>
#include <functional>
#include <string>
#include <vector>
#include <memory>

/**
 * @brief NL - Nonlinear dynamic system model class
 * C++ implementation of MATLAB nl.m
 * 
 * Represents nonlinear time-varying systems:
 *   x+(t) = f(t, x(t), u(t), th) + v(t),  v(t) ~ pv
 *    y(t) = h(t, x(t), u(t), th) + e(t),  e(t) ~ pe
 *   x(0) ~ px0, E[x(0)] = x0
 * 
 * where x+(t) = x'(t) for continuous time or x(t+1) for discrete time
 */
class NL {
public:
    using Vector = Eigen::VectorXd;
    using Matrix = Eigen::MatrixXd;
    
    // Function types: f(t, x, u, th) -> x_next
    using StateFunction = std::function<Vector(double, const Vector&, const Vector&, const Vector&)>;
    using MeasurementFunction = std::function<Vector(double, const Vector&, const Vector&, const Vector&)>;
    
    // Model dimensions [nx, nu, ny, nth]
    Eigen::Vector4i nn;
    
    // Model functions
    StateFunction f;           // State dynamics
    MeasurementFunction h;     // Measurement function
    
    // Parameters and initial conditions
    Vector x0;                 // Initial state mean
    Vector th;                 // Parameter vector
    
    // Noise distributions
    std::shared_ptr<PDFClass> pv;   // Process noise v(t)
    std::shared_ptr<PDFClass> pe;   // Measurement noise e(t)
    std::shared_ptr<PDFClass> px0;  // Initial state distribution
    
    // Covariance matrices (alternative to distributions)
    Matrix P;                  // Parameter covariance
    Matrix I;                  // Information matrix (inverse of joint [x;th] cov)
    
    // Optional Jacobian functions (for EKF)
    std::function<Matrix(double, const Vector&, const Vector&, const Vector&)> dfdx;  // ∂f/∂x
    std::function<Matrix(double, const Vector&, const Vector&, const Vector&)> dhdx;  // ∂h/∂x
    
    // Sampling frequency
    double fs;
    
    // Labels and metadata
    std::vector<std::string> xlabel;
    std::vector<std::string> ylabel;
    std::vector<std::string> ulabel;
    std::vector<std::string> thlabel;
    std::string tlabel;
    std::string name;
    std::string desc;
    
    // Optional measurement matrix (for linear h)
    Matrix H;
    
    /**
     * @brief Default constructor
     */
    NL() : nn(0, 0, 0, 0), fs(NAN), tlabel("Time") {}
    
    /**
     * @brief Main constructor
     * @param f_func State dynamics function
     * @param h_func Measurement function
     * @param dimensions [nx, nu, ny, nth]
     * @param sampling_freq Sampling frequency (NaN for continuous)
     */
    NL(StateFunction f_func, MeasurementFunction h_func, 
       const Eigen::Vector4i& dimensions, double sampling_freq = NAN)
        : f(f_func), h(h_func), nn(dimensions), fs(sampling_freq), tlabel("Time") {
        
        int nx = nn(0), nu = nn(1), ny = nn(2), nth = nn(3);
        
        // Initialize state and parameters
        x0 = Vector::Zero(nx);
        th = Vector::Zero(nth);
        
        // Initialize labels
        for (int i = 0; i < nx; ++i) {
            xlabel.push_back("x" + std::to_string(i + 1));
        }
        for (int i = 0; i < nu; ++i) {
            ulabel.push_back("u" + std::to_string(i + 1));
        }
        for (int i = 0; i < ny; ++i) {
            ylabel.push_back("y" + std::to_string(i + 1));
        }
        for (int i = 0; i < nth; ++i) {
            thlabel.push_back("th" + std::to_string(i + 1));
        }
    }
    
    /**
     * @brief Set process noise covariance
     */
    void set_pv(const Matrix& Qv) {
        if (Qv.rows() != nn(0) || Qv.cols() != nn(0)) {
            throw std::invalid_argument("NL::set_pv: Qv must be nx x nx");
        }
        pv = std::make_shared<NDist>(Vector::Zero(nn(0)), Qv);
    }
    
    /**
     * @brief Set measurement noise covariance
     */
    void set_pe(const Matrix& Re) {
        if (Re.rows() != nn(2) || Re.cols() != nn(2)) {
            throw std::invalid_argument("NL::set_pe: Re must be ny x ny");
        }
        pe = std::make_shared<NDist>(Vector::Zero(nn(2)), Re);
    }
    
    /**
     * @brief Set initial state covariance
     */
    void set_px0(const Matrix& P0) {
        if (P0.rows() != nn(0) || P0.cols() != nn(0)) {
            throw std::invalid_argument("NL::set_px0: P0 must be nx x nx");
        }
        px0 = std::make_shared<NDist>(x0, P0);
    }
    
    /**
     * @brief Simulate the nonlinear system
     * @param u Input signal (nu x N matrix)
     * @param t Time vector (N x 1) or empty for discrete time
     * @param MC Number of Monte Carlo simulations
     * @return Sig object with simulated outputs
     */
    Sig simulate(const Matrix& u, const Vector& t = Vector(), int MC = 1) const;
    
    /**
     * @brief Extended Kalman Filter (EKF)
     * @param z Measurement signal
     * @param u Input signal (optional)
     * @param x0_init Initial state estimate (optional)
     * @param P0_init Initial covariance (optional)
     * @return Pair of (state estimates, covariances)
     */
    std::pair<Matrix, std::vector<Matrix>> ekf(
        const Sig& z, 
        const Matrix& u = Matrix(),
        const Vector& x0_init = Vector(),
        const Matrix& P0_init = Matrix()) const;
    
    /**
     * @brief Unscented Kalman Filter (UKF)
     * @param z Measurement signal
     * @param u Input signal (optional)
     * @param x0_init Initial state estimate (optional)
     * @param P0_init Initial covariance (optional)
     * @return Pair of (state estimates, covariances)
     */
    std::pair<Matrix, std::vector<Matrix>> ukf(
        const Sig& z,
        const Matrix& u = Matrix(),
        const Vector& x0_init = Vector(),
        const Matrix& P0_init = Matrix()) const;
    
    /**
     * @brief Particle Filter (PF)
     * @param z Measurement signal
     * @param u Input signal (optional)
     * @param N Number of particles
     * @return State estimates
     */
    Matrix pf(const Sig& z, const Matrix& u = Matrix(), int N = 1000) const;
    
    /**
     * @brief Cramér-Rao Lower Bound (CRLB) computation
     * @param z Measurement signal with true states in z.x
     * @param P0_init Initial covariance (optional)
     * @param Q_override Process noise covariance override (optional)
     * @param R_override Measurement noise covariance override (optional)
     * @param pred_horizon Prediction horizon: 0 for filter, 1 for predictor
     * @return Sig object with CRLB in Px and Py
     * 
     * Computes the parametric Cramér-Rao Lower Bound for state estimation.
     * The CRLB provides the theoretical lower bound on estimation variance
     * for any unbiased estimator. Uses the true states from z.x to linearize.
     */
    Sig crlb(const Sig& z,
             const Matrix& P0_init = Matrix(),
             const Matrix& Q_override = Matrix(),
             const Matrix& R_override = Matrix(),
             int pred_horizon = 0) const;
    
private:
    /**
     * @brief Compute numerical Jacobian of f with respect to x
     */
    Matrix compute_dfdx(double t, const Vector& x, const Vector& u, const Vector& th) const {
        if (dfdx) {
            return dfdx(t, x, u, th);
        }
        
        // Numerical differentiation
        auto f_func = [this, t, u, th](const Vector& x_var) {
            return this->f(t, x_var, u, th);
        };
            return numjac(f_func, x);
    }
    
    /**
     * @brief Compute numerical Jacobian of h with respect to x
     */
    Matrix compute_dhdx(double t, const Vector& x, const Vector& u, const Vector& th) const {
        if (dhdx) {
            return dhdx(t, x, u, th);
        }
        
        // Numerical differentiation
        auto h_func = [this, t, u, th](const Vector& x_var) {
            return this->h(t, x_var, u, th);
        };
            return numjac(h_func, x);
    }
};

#endif // NL_H
