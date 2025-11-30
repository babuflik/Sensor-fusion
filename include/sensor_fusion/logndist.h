#ifndef LOGNDIST_H
#define LOGNDIST_H

#include "pdfclass.h"
#include <Eigen/Dense>
#include <random>
#include <cmath>

/**
 * @brief Log-normal distribution
 * C++ implementation of MATLAB lognormal distribution
 * 
 * If Y ~ LogN(mu, sigma^2) then log(Y) ~ N(mu, sigma^2)
 */
class LogNDist : public PDFClass {
private:
    double mu_;      // Mean of the underlying normal
    double sigma_;   // Standard deviation of the underlying normal
    
public:
    // Constructor
    LogNDist(double mu = 0.0, double sigma = 1.0)
        : mu_(mu), sigma_(sigma) {
        if (sigma <= 0.0) {
            throw std::invalid_argument("LogNDist: sigma must be positive");
        }
        state = generate_random_state();
    }
    
    // Generate random samples
    Matrix rand(int n = 1) override {
        auto& gen = get_generator();
        std::normal_distribution<double> normal(mu_, sigma_);
        
        Matrix samples(n, 1);
        for (int i = 0; i < n; ++i) {
            double z = normal(gen);
            samples(i, 0) = std::exp(z);
        }
        
        return samples;
    }
    
    // Probability density function
    Vector pdf(const Matrix& x) const override {
        int n = x.rows();
        Vector result(n);
        
        for (int i = 0; i < n; ++i) {
            double val = x(i, 0);
            if (val <= 0.0) {
                result(i) = 0.0;
            } else {
                // f(x) = 1/(x*sigma*sqrt(2*pi)) * exp(-(ln(x)-mu)^2 / (2*sigma^2))
                double log_x = std::log(val);
                double z = (log_x - mu_) / sigma_;
                double log_pdf = -std::log(val) - std::log(sigma_) 
                               - 0.5 * std::log(2.0 * M_PI)
                               - 0.5 * z * z;
                result(i) = std::exp(log_pdf);
            }
        }
        
        return result;
    }
    
    // Cumulative distribution function
    Vector cdf(const Vector& x) const override {
        int n = x.size();
        Vector result(n);
        
        for (int i = 0; i < n; ++i) {
            double val = x(i);
            if (val <= 0.0) {
                result(i) = 0.0;
            } else {
                // F(x) = Phi((ln(x) - mu) / sigma)
                double z = (std::log(val) - mu_) / sigma_;
                result(i) = 0.5 * (1.0 + std::erf(z / std::sqrt(2.0)));
            }
        }
        
        return result;
    }
    
    // Mean: exp(mu + sigma^2/2)
    Vector mean() const override {
        Vector m(1);
        m(0) = std::exp(mu_ + sigma_ * sigma_ / 2.0);
        return m;
    }
    
    // Covariance (variance for 1D)
    Matrix cov() const override {
        Matrix C(1, 1);
        double exp_2mu_sigma2 = std::exp(2.0 * mu_ + sigma_ * sigma_);
        C(0, 0) = exp_2mu_sigma2 * (std::exp(sigma_ * sigma_) - 1.0);
        return C;
    }
    
    // Median: exp(mu)
    Vector median() const override {
        Vector m(1);
        m(0) = std::exp(mu_);
        return m;
    }
    
    // Mode: exp(mu - sigma^2)
    Vector mode() const override {
        Vector m(1);
        m(0) = std::exp(mu_ - sigma_ * sigma_);
        return m;
    }
    
    // Variance
    double var() const override {
        double exp_2mu_sigma2 = std::exp(2.0 * mu_ + sigma_ * sigma_);
        return exp_2mu_sigma2 * (std::exp(sigma_ * sigma_) - 1.0);
    }
    
    // Skewness
    double skew() const override {
        double exp_sigma2 = std::exp(sigma_ * sigma_);
        return (exp_sigma2 + 2.0) * std::sqrt(exp_sigma2 - 1.0);
    }
    
    // Kurtosis (excess kurtosis + 3)
    double kurt() const override {
        double exp_sigma2 = std::exp(sigma_ * sigma_);
        double exp_2sigma2 = exp_sigma2 * exp_sigma2;
        double exp_3sigma2 = exp_2sigma2 * exp_sigma2;
        double exp_4sigma2 = exp_2sigma2 * exp_2sigma2;
        return 3.0 + exp_4sigma2 + 2.0 * exp_3sigma2 + 3.0 * exp_2sigma2 - 6.0;
    }
    
    // Distribution length
    int length() const override {
        return 1;
    }
    
    // Description string
    std::string desc() const override {
        return "Log-normal distribution LogN(mu=" + std::to_string(mu_) + 
               ", sigma=" + std::to_string(sigma_) + ")";
    }
    
    // Accessors
    double mu() const { return mu_; }
    double sigma() const { return sigma_; }
};

#endif // LOGNDIST_H
