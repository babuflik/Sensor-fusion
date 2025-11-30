#pragma once
#include "pdfclass.h"
#include <Eigen/Dense>
#include <random>
#include <cmath>

/**
 * @brief ExpDist - Exponential distribution Exp(lambda)
 * C++ implementation of MATLAB expdist.m
 */
class ExpDist : public PDFClass {
public:
    using Vector = Eigen::VectorXd;
    using Matrix = Eigen::MatrixXd;
    
    double lambda;  // Rate parameter
    
    // Constructors
    
    ExpDist() : PDFClass() {
        lambda = 1.0;
        state = generate_random_state();
        xlabel = {"x1"};
    }
    
    ExpDist(double lambda_) : PDFClass() {
        if (lambda_ <= 0) {
            throw std::runtime_error("ExpDist: lambda must be positive");
        }
        lambda = lambda_;
        state = generate_random_state();
        xlabel = {"x1"};
    }
    
    // PDFClass implementations
    
    Matrix rand(int n = 1) override {
        Matrix samples(n, 1);
        
        std::mt19937& gen = get_generator();
        std::exponential_distribution<double> dist(lambda);
        
        for (int i = 0; i < n; ++i) {
            samples(i, 0) = dist(gen);
        }
        
        return samples;
    }
    
    Vector pdf(const Matrix& x) const override {
        int N = x.rows();
        
        if (x.cols() != 1) {
            throw std::runtime_error("ExpDist::pdf: x must be 1D");
        }
        
        Vector result(N);
        
        for (int i = 0; i < N; ++i) {
            if (x(i, 0) < 0) {
                result(i) = 0.0;
            } else {
                result(i) = lambda * std::exp(-lambda * x(i, 0));
            }
        }
        
        return result;
    }
    
    Vector cdf(const Vector& x) const override {
        Vector result(x.size());
        
        for (int i = 0; i < x.size(); ++i) {
            if (x(i) < 0) {
                result(i) = 0.0;
            } else {
                result(i) = 1.0 - std::exp(-lambda * x(i));
            }
        }
        
        return result;
    }
    
    Vector mean() const override {
        Vector mu(1);
        mu(0) = 1.0 / lambda;
        return mu;
    }
    
    Matrix cov() const override {
        Matrix P(1, 1);
        P(0, 0) = 1.0 / (lambda * lambda);
        return P;
    }
    
    std::string desc() const override {
        return "Exponential distribution";
    }
    
    int length() const override {
        return 1;
    }
    
    std::string symbolic() const override {
        return "Exp(" + std::to_string(lambda) + ")";
    }
    
    Vector median() const override {
        Vector m(1);
        m(0) = std::log(2.0) / lambda;
        return m;
    }
    
    Vector mode() const override {
        Vector m(1);
        m(0) = 0.0;
        return m;
    }
    
    double skew() const override {
        return 2.0;
    }
    
    double kurt() const override {
        return 6.0;  // Excess kurtosis
    }
};
