#ifndef BETADIST_H
#define BETADIST_H

#include "pdfclass.h"
#include <Eigen/Dense>
#include <random>
#include <cmath>

/**
 * @brief Beta distribution
 * C++ implementation of MATLAB betadist.m
 * 
 * Beta(alpha, beta) on [0, 1]
 */
class BetaDist : public PDFClass {
public:
    using Vector = Eigen::VectorXd;
    using Matrix = Eigen::MatrixXd;
    
private:
    double alpha_;
    double beta_;
    
public:
    // Constructor
    BetaDist(double alpha, double beta)
        : alpha_(alpha), beta_(beta) {
        if (alpha <= 0 || beta <= 0) {
            throw std::invalid_argument("BetaDist: alpha and beta must be positive");
        }
        state = generate_random_state();
    }
    
    // Generate random samples
    Matrix rand(int n = 1) override {
        // Use gamma distribution to generate beta samples
        // If X ~ Gamma(alpha, 1) and Y ~ Gamma(beta, 1)
        // then X/(X+Y) ~ Beta(alpha, beta)
        
        Matrix samples(n, 1);
        std::mt19937& gen = get_generator();
        std::gamma_distribution<double> gamma_a(alpha_, 1.0);
        std::gamma_distribution<double> gamma_b(beta_, 1.0);
        
        for (int i = 0; i < n; ++i) {
            double x = gamma_a(gen);
            double y = gamma_b(gen);
            samples(i, 0) = x / (x + y);
        }
        
        return samples;
    }
    
    // Probability density function
    Vector pdf(const Matrix& x) const override {
        int N = x.rows();
        if (x.cols() != 1) {
            throw std::invalid_argument("BetaDist::pdf: only 1D supported");
        }
        
        Vector result(N);
        
        for (int i = 0; i < N; ++i) {
            double val = x(i, 0);
            if (val < 0.0 || val > 1.0) {
                result(i) = 0.0;
            } else {
                // PDF: f(x) = x^(alpha-1) * (1-x)^(beta-1) / B(alpha, beta)
                // where B(alpha, beta) = Gamma(alpha)*Gamma(beta)/Gamma(alpha+beta)
                
                // Log-space computation
                double log_pdf = (alpha_ - 1.0) * std::log(val) 
                               + (beta_ - 1.0) * std::log(1.0 - val)
                               - std::lgamma(alpha_) - std::lgamma(beta_) 
                               + std::lgamma(alpha_ + beta_);
                
                result(i) = std::exp(log_pdf);
            }
        }
        
        return result;
    }
    
    // Cumulative distribution function
    Vector cdf(const Vector& x) const override {
        int N = x.size();
        Vector result(N);
        
        for (int i = 0; i < N; ++i) {
            double val = x(i);
            if (val <= 0.0) {
                result(i) = 0.0;
            } else if (val >= 1.0) {
                result(i) = 1.0;
            } else {
                // Use regularized incomplete beta function
                // I_x(a,b) = B_x(a,b) / B(a,b)
                result(i) = incomplete_beta(val, alpha_, beta_);
            }
        }
        
        return result;
    }
    
    // Mean
    Vector mean() const override {
        Vector m(1);
        m(0) = alpha_ / (alpha_ + beta_);
        return m;
    }
    
    // Covariance (variance for 1D)
    Matrix cov() const override {
        Matrix C(1, 1);
        double sum = alpha_ + beta_;
        C(0, 0) = (alpha_ * beta_) / (sum * sum * (sum + 1.0));
        return C;
    }
    
    // Variance
    double var() const override {
        double sum = alpha_ + beta_;
        return (alpha_ * beta_) / (sum * sum * (sum + 1.0));
    }
    
    // Mode
    Vector mode() const override {
        if (alpha_ > 1.0 && beta_ > 1.0) {
            Vector m(1);
            m(0) = (alpha_ - 1.0) / (alpha_ + beta_ - 2.0);
            return m;
        } else if (alpha_ < 1.0 && beta_ < 1.0) {
            throw std::runtime_error("BetaDist::mode: bimodal (0 and 1)");
        } else if (alpha_ <= 1.0) {
            return Vector::Zero(1);
        } else {
            return Vector::Ones(1);
        }
    }
    
    // Skewness
    double skew() const override {
        double sum = alpha_ + beta_;
        return 2.0 * (beta_ - alpha_) * std::sqrt(sum + 1.0) 
             / ((sum + 2.0) * std::sqrt(alpha_ * beta_));
    }
    
    // Kurtosis
    double kurt() const override {
        double sum = alpha_ + beta_;
        double num = 6.0 * ((alpha_ - beta_) * (alpha_ - beta_) * (sum + 1.0) 
                           - alpha_ * beta_ * (sum + 2.0));
        double den = alpha_ * beta_ * (sum + 2.0) * (sum + 3.0);
        return 3.0 + num / den;
    }
    
    // Distribution length
    int length() const override {
        return 1;
    }
    
    // Description string
    std::string desc() const override {
        return "Beta distribution Beta(alpha=" + std::to_string(alpha_) + 
               ", beta=" + std::to_string(beta_) + ")";
    }
    
    // Accessors
    double alpha() const { return alpha_; }
    double beta() const { return beta_; }
    
private:
    // Regularized incomplete beta function I_x(a,b)
    // Approximation using continued fractions
    double incomplete_beta(double x, double a, double b) const {
        if (x == 0.0) return 0.0;
        if (x == 1.0) return 1.0;
        
        // Use symmetry if needed
        if (x > (a + 1.0) / (a + b + 2.0)) {
            return 1.0 - incomplete_beta(1.0 - x, b, a);
        }
        
        // Compute using continued fractions (simplified)
        double log_beta = std::lgamma(a) + std::lgamma(b) - std::lgamma(a + b);
        double front = std::exp(a * std::log(x) + b * std::log(1.0 - x) - log_beta) / a;
        
        // Continued fraction (limited iterations)
        double f = 1.0;
        double c = 1.0;
        double d = 0.0;
        
        for (int m = 0; m < 200; ++m) {
            double aa = (m % 2 == 0) ? 
                        m * (b - m) * x / ((a + 2.0 * m - 1.0) * (a + 2.0 * m)) :
                        -(a + m) * (a + b + m) * x / ((a + 2.0 * m + 1.0) * (a + 2.0 * m));
            
            d = 1.0 + aa * d;
            if (std::abs(d) < 1e-30) d = 1e-30;
            c = 1.0 + aa / c;
            if (std::abs(c) < 1e-30) c = 1e-30;
            d = 1.0 / d;
            double delta = c * d;
            f *= delta;
            
            if (std::abs(delta - 1.0) < 1e-10) break;
        }
        
        return front * f;
    }
};

#endif // BETADIST_H
