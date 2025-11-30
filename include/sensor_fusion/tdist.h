#ifndef TDIST_H
#define TDIST_H

#include "pdfclass.h"
#include <Eigen/Dense>
#include <random>
#include <cmath>

/**
 * @brief Student's t-distribution
 * C++ implementation of MATLAB tdist.m
 * 
 * T(nu) where nu is degrees of freedom
 * For multivariate: T(mu, Sigma, nu)
 */
class TDist : public PDFClass {
public:
    using Vector = Eigen::VectorXd;
    using Matrix = Eigen::MatrixXd;
    
private:
    double nu_;                  // Degrees of freedom
    Eigen::VectorXd mu_;        // Mean (if nu > 1)
    Eigen::MatrixXd Sigma_;     // Scale matrix
    int dim_;                    // Dimension
    
public:
    // Scalar constructor: T(nu)
    TDist(double nu) 
        : nu_(nu), dim_(1) {
        mu_ = Eigen::VectorXd::Zero(1);
        Sigma_ = Eigen::MatrixXd::Identity(1, 1);
        state = generate_random_state();
    }
    
    // Multivariate constructor: T(mu, Sigma, nu)
    TDist(const Eigen::VectorXd& mu, const Eigen::MatrixXd& Sigma, double nu)
        : nu_(nu), mu_(mu), Sigma_(Sigma), dim_(mu.size()) {
        if (Sigma.rows() != dim_ || Sigma.cols() != dim_) {
            throw std::invalid_argument("TDist: Sigma dimensions must match mu");
        }
        state = generate_random_state();
    }
    
    // Generate random samples
    Matrix rand(int n = 1) override {
        // Student's t = Normal / sqrt(Chi2/nu)
        // For multivariate: x = mu + L*z / sqrt(w/nu)
        // where z ~ N(0,I), w ~ Chi2(nu), L*L^T = Sigma
        
        Matrix samples(n, dim_);
        std::mt19937& gen = get_generator();
        std::normal_distribution<double> normal(0.0, 1.0);
        std::chi_squared_distribution<double> chi2(nu_);
        
        // Cholesky decomposition for Sigma
        Eigen::LLT<Eigen::MatrixXd> llt(Sigma_);
        Eigen::MatrixXd L = llt.matrixL();
        
        for (int i = 0; i < n; ++i) {
            Eigen::VectorXd z(dim_);
            for (int j = 0; j < dim_; ++j) {
                z(j) = normal(gen);
            }
            
            double w = chi2(gen);
            double scale = std::sqrt(nu_ / w);
            
            samples.row(i) = (mu_ + scale * L * z).transpose();
        }
        
        return samples;
    }
    
    // Probability density function
    Vector pdf(const Matrix& x) const override {
        int N = x.rows();
        if (x.cols() != dim_) {
            throw std::invalid_argument("TDist::pdf: dimension mismatch");
        }
        
        Vector result(N);
        
        // Multivariate t-distribution PDF:
        // f(x) = Gamma((nu+d)/2) / (Gamma(nu/2) * (nu*pi)^(d/2) * |Sigma|^(1/2))
        //        * (1 + delta^2/nu)^(-(nu+d)/2)
        // where delta^2 = (x-mu)^T * Sigma^(-1) * (x-mu)
        
        // Compute delta^2 using LLT solver
        Eigen::LLT<Eigen::MatrixXd> llt(Sigma_);
        Eigen::MatrixXd L = llt.matrixL();
        
        // Log-space computation for numerical stability
        double log_num = std::lgamma((nu_ + dim_) / 2.0);
        double log_den = std::lgamma(nu_ / 2.0) 
                       + 0.5 * dim_ * std::log(nu_ * M_PI)
                       + L.diagonal().array().log().sum();  // log(det(Sigma)) = 2*sum(log(diag(L)))
        
        for (int i = 0; i < N; ++i) {
            Eigen::VectorXd diff = x.row(i).transpose() - mu_;
            Eigen::VectorXd alpha = L.triangularView<Eigen::Lower>().solve(diff);
            double delta2 = alpha.squaredNorm();
            
            double log_kernel = -(nu_ + dim_) / 2.0 * std::log(1.0 + delta2 / nu_);
            result(i) = std::exp(log_num - log_den + log_kernel);
        }
        
        return result;
    }
    
    // Cumulative distribution function (1D only)
    Vector cdf(const Vector& x) const override {
        if (dim_ != 1) {
            throw std::runtime_error("TDist::cdf: only implemented for 1D");
        }
        
        int N = x.size();
        Vector result(N);
        
        for (int i = 0; i < N; ++i) {
            double t = (x(i) - mu_(0)) / std::sqrt(Sigma_(0, 0));
            
            // Use regularized incomplete beta function
            // F(t) = 0.5 + t*Gamma((nu+1)/2) / (sqrt(nu*pi)*Gamma(nu/2)) * 2F1(...)
            // Simplified approximation using std::beta
            double x_beta = nu_ / (nu_ + t * t);
            double cdf_val = 0.5 + 0.5 * std::copysign(1.0, t) * 
                             (1.0 - std::beta(nu_ / 2.0, 0.5) * x_beta);
            
            result(i) = std::max(0.0, std::min(1.0, cdf_val));
        }
        
        return result;
    }
    
    // Mean (only defined for nu > 1)
    Vector mean() const override {
        if (nu_ <= 1.0) {
            throw std::runtime_error("TDist::mean: undefined for nu <= 1");
        }
        return mu_;
    }
    
    // Covariance (only defined for nu > 2)
    Matrix cov() const override {
        if (nu_ <= 2.0) {
            throw std::runtime_error("TDist::cov: undefined for nu <= 2");
        }
        return nu_ / (nu_ - 2.0) * Sigma_;
    }
    
    // Variance (for 1D distributions)
    double var() const override {
        if (dim_ != 1) {
            throw std::runtime_error("TDist::var: only defined for 1D distributions, use cov()");
        }
        if (nu_ <= 2.0) {
            throw std::runtime_error("TDist::var: undefined for nu <= 2");
        }
        return nu_ / (nu_ - 2.0) * Sigma_(0, 0);
    }
    
    // Skewness (for nu > 3)
    double skew() const override {
        if (dim_ != 1) {
            throw std::runtime_error("TDist::skew: only defined for 1D distributions");
        }
        if (nu_ <= 3.0) {
            throw std::runtime_error("TDist::skew: undefined for nu <= 3");
        }
        return 0.0;  // t-distribution is symmetric
    }
    
    // Kurtosis (for nu > 4)
    double kurt() const override {
        if (dim_ != 1) {
            throw std::runtime_error("TDist::kurt: only defined for 1D distributions");
        }
        if (nu_ <= 4.0) {
            throw std::runtime_error("TDist::kurt: undefined for nu <= 4");
        }
        double excess_kurt = 6.0 / (nu_ - 4.0);
        return 3.0 + excess_kurt;
    }
    
    // Distribution length (dimension)
    int length() const override {
        return dim_;
    }
    
    // Description string
    std::string desc() const override {
        return "Student's t-distribution T(nu=" + std::to_string(nu_) + 
               ", dim=" + std::to_string(dim_) + ")";
    }
    
    // Accessors
    double nu() const { return nu_; }
    Eigen::VectorXd mu() const { return mu_; }
    Eigen::MatrixXd Sigma() const { return Sigma_; }
};

#endif // TDIST_H
