#pragma once
#include "pdfclass.h"
#include <Eigen/Dense>
#include <random>
#include <cmath>

/**
 * @brief UDist - Uniform distribution U(a, b)
 * C++ implementation of MATLAB udist.m
 */
class UDist : public PDFClass {
public:
    using Vector = Eigen::VectorXd;
    using Matrix = Eigen::MatrixXd;
    
    Vector a;  // Lower bounds
    Vector b;  // Upper bounds
    
    // Constructors
    
    UDist() : PDFClass() {
        a = Vector();
        b = Vector();
        state = generate_random_state();
    }
    
    UDist(const Vector& a_, const Vector& b_) : PDFClass() {
        if (a_.size() != b_.size()) {
            throw std::runtime_error("UDist: a and b must have same dimension");
        }
        if ((a_.array() >= b_.array()).any()) {
            throw std::runtime_error("UDist: a must be less than b");
        }
        a = a_;
        b = b_;
        state = generate_random_state();
        
        xlabel.resize(a.size());
        for (int i = 0; i < a.size(); ++i) {
            xlabel[i] = "x" + std::to_string(i + 1);
        }
    }
    
    UDist(double a_scalar, double b_scalar) : PDFClass() {
        if (a_scalar >= b_scalar) {
            throw std::runtime_error("UDist: a must be less than b");
        }
        a = Vector::Constant(1, a_scalar);
        b = Vector::Constant(1, b_scalar);
        state = generate_random_state();
        xlabel = {"x1"};
    }
    
    // PDFClass implementations
    
    Matrix rand(int n = 1) override {
        if (a.size() == 0) {
            throw std::runtime_error("UDist::rand: Distribution not initialized");
        }
        
        int dim = a.size();
        Matrix samples(n, dim);
        
        std::mt19937& gen = get_generator();
        
        for (int j = 0; j < dim; ++j) {
            std::uniform_real_distribution<double> dist(a(j), b(j));
            for (int i = 0; i < n; ++i) {
                samples(i, j) = dist(gen);
            }
        }
        
        return samples;
    }
    
    Vector pdf(const Matrix& x) const override {
        int N = x.rows();
        int dim = a.size();
        
        if (x.cols() != dim) {
            throw std::runtime_error("UDist::pdf: x must have same dimension as a,b");
        }
        
        Vector result(N);
        double volume = (b - a).prod();
        double pdf_val = 1.0 / volume;
        
        for (int i = 0; i < N; ++i) {
            bool inside = true;
            for (int j = 0; j < dim; ++j) {
                if (x(i, j) < a(j) || x(i, j) > b(j)) {
                    inside = false;
                    break;
                }
            }
            result(i) = inside ? pdf_val : 0.0;
        }
        
        return result;
    }
    
    Vector cdf(const Vector& x) const override {
        if (a.size() != 1) {
            throw std::runtime_error("UDist::cdf: Only implemented for 1D distributions");
        }
        
        Vector result(x.size());
        double range = b(0) - a(0);
        
        for (int i = 0; i < x.size(); ++i) {
            if (x(i) <= a(0)) {
                result(i) = 0.0;
            } else if (x(i) >= b(0)) {
                result(i) = 1.0;
            } else {
                result(i) = (x(i) - a(0)) / range;
            }
        }
        
        return result;
    }
    
    Vector mean() const override {
        return 0.5 * (a + b);
    }
    
    Matrix cov() const override {
        int dim = a.size();
        Matrix P = Matrix::Zero(dim, dim);
        Vector range = b - a;
        
        for (int i = 0; i < dim; ++i) {
            P(i, i) = range(i) * range(i) / 12.0;
        }
        
        return P;
    }
    
    std::string desc() const override {
        return "Uniform distribution";
    }
    
    int length() const override {
        return a.size();
    }
    
    std::string symbolic() const override {
        if (a.size() == 1) {
            return "U(" + std::to_string(a(0)) + ", " + std::to_string(b(0)) + ")";
        }
        return "U([" + std::to_string(a.size()) + "D])";
    }
    
    Vector median() const override {
        return mean();
    }
    
    Vector mode() const override {
        return mean();  // Uniform has no unique mode
    }
    
    double skew() const override {
        if (a.size() != 1) {
            throw std::runtime_error("skew() not defined for multi-dimensional distributions");
        }
        return 0.0;  // Symmetric distribution
    }
    
    double kurt() const override {
        if (a.size() != 1) {
            throw std::runtime_error("kurt() not defined for multi-dimensional distributions");
        }
        return -1.2;  // Excess kurtosis for uniform = -6/5
    }
};
