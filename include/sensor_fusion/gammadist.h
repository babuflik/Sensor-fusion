#pragma once
#include "pdfclass.h"
#include <Eigen/Dense>
#include <random>
#include <cmath>

/**
 * @brief GammaDist - Gamma distribution Γ(k, θ)
 * C++ implementation of MATLAB gammadist.m
 * Shape parameter k, scale parameter θ
 */
class GammaDist : public PDFClass {
public:
    using Vector = Eigen::VectorXd;
    using Matrix = Eigen::MatrixXd;
    
    double k;     // Shape parameter
    double theta; // Scale parameter
    
    GammaDist() : PDFClass() {
        k = 1.0;
        theta = 1.0;
        state = generate_random_state();
        xlabel = {"x1"};
    }
    
    GammaDist(double k_, double theta_) : PDFClass() {
        if (k_ <= 0 || theta_ <= 0) {
            throw std::runtime_error("GammaDist: parameters must be positive");
        }
        k = k_;
        theta = theta_;
        state = generate_random_state();
        xlabel = {"x1"};
    }
    
    Matrix rand(int n = 1) override {
        Matrix samples(n, 1);
        std::mt19937& gen = get_generator();
        std::gamma_distribution<double> dist(k, theta);
        
        for (int i = 0; i < n; ++i) {
            samples(i, 0) = dist(gen);
        }
        return samples;
    }
    
    Vector pdf(const Matrix& x) const override {
        int N = x.rows();
        Vector result(N);
        
        double log_norm = k * std::log(theta) + std::lgamma(k);
        
        for (int i = 0; i < N; ++i) {
            if (x(i, 0) <= 0) {
                result(i) = 0.0;
            } else {
                double log_pdf = (k - 1.0) * std::log(x(i, 0)) - x(i, 0) / theta - log_norm;
                result(i) = std::exp(log_pdf);
            }
        }
        return result;
    }
    
    Vector cdf(const Vector& x) const override {
        // Simplified - would use incomplete gamma function
        throw std::runtime_error("GammaDist::cdf not fully implemented");
    }
    
    Vector mean() const override {
        Vector mu(1);
        mu(0) = k * theta;
        return mu;
    }
    
    Matrix cov() const override {
        Matrix P(1, 1);
        P(0, 0) = k * theta * theta;
        return P;
    }
    
    std::string desc() const override { return "Gamma distribution"; }
    int length() const override { return 1; }
    
    std::string symbolic() const override {
        return "Γ(" + std::to_string(k) + ", " + std::to_string(theta) + ")";
    }
    
    Vector mode() const override {
        Vector m(1);
        m(0) = (k >= 1.0) ? (k - 1.0) * theta : 0.0;
        return m;
    }
    
    double skew() const override {
        return 2.0 / std::sqrt(k);
    }
    
    double kurt() const override {
        return 6.0 / k;
    }
};
