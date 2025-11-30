#pragma once
#include "pdfclass.h"
#include <Eigen/Dense>
#include <random>
#include <cmath>

/**
 * @brief NCChi2Dist - Non-central chi-squared distribution χ²(n, λ)
 * C++ implementation of MATLAB ncchi2dist.m
 * 
 * The non-central chi-squared distribution with n degrees of freedom and
 * non-centrality parameter λ (denoted h in MATLAB).
 * 
 * Mean: n + λ
 * Variance: 2n + 4λ
 * 
 * When λ = 0, this reduces to the central chi-squared distribution.
 */
class NCChi2Dist : public PDFClass {
public:
    using Vector = Eigen::VectorXd;
    using Matrix = Eigen::MatrixXd;
    
    int n;          // Degrees of freedom
    double lambda;  // Non-centrality parameter (h in MATLAB)
    
    // Constructors
    
    NCChi2Dist() : PDFClass() {
        n = 1;
        lambda = 0.0;
        state = generate_random_state();
        xlabel = {"x1"};
    }
    
    NCChi2Dist(int n_) : PDFClass() {
        if (n_ <= 0) {
            throw std::runtime_error("NCChi2Dist: degrees of freedom must be positive");
        }
        n = n_;
        lambda = 0.0;
        state = generate_random_state();
        xlabel = {"x1"};
    }
    
    NCChi2Dist(int n_, double lambda_) : PDFClass() {
        if (n_ <= 0) {
            throw std::runtime_error("NCChi2Dist: degrees of freedom must be positive");
        }
        if (lambda_ < 0) {
            throw std::runtime_error("NCChi2Dist: non-centrality parameter must be non-negative");
        }
        n = n_;
        lambda = lambda_;
        state = generate_random_state();
        xlabel = {"x1"};
    }
    
    // PDFClass implementations
    
    Matrix rand(int num_samples = 1) override {
        Matrix samples(num_samples, 1);
        
        std::mt19937& gen = get_generator();
        
        // Non-central chi-squared is sum of:
        // 1. Central chi-squared with n degrees of freedom
        // 2. Non-centrality from normal distribution
        std::chi_squared_distribution<double> central_chi2(n);
        std::normal_distribution<double> normal(std::sqrt(lambda), 1.0);
        
        for (int i = 0; i < num_samples; ++i) {
            // One approach: sum of squared normals
            // More stable: use non-central chi-squared generation
            if (lambda == 0.0) {
                samples(i, 0) = central_chi2(gen);
            } else {
                // Generate as sum of (n-1) central chi-squared + squared normal with mean sqrt(lambda)
                double z = normal(gen);
                double central = (n > 1) ? std::chi_squared_distribution<double>(n - 1)(gen) : 0.0;
                samples(i, 0) = z * z + central;
            }
        }
        
        return samples;
    }
    
    Vector pdf(const Matrix& x) const override {
        int N = x.rows();
        
        if (x.cols() != 1) {
            throw std::runtime_error("NCChi2Dist::pdf: x must be 1D");
        }
        
        Vector result(N);
        double d = static_cast<double>(n);
        double h = lambda;
        if (h < 1e-10) h = 1e-10;  // Avoid division by zero
        
        // PDF: 0.5 * (x/h)^(d/4-0.5) * exp(-0.5*(x+h)) * I_{d/2-1}(sqrt(h*x))
        // where I_ν is the modified Bessel function of the first kind
        
        for (int i = 0; i < N; ++i) {
            if (x(i, 0) <= 0) {
                result(i) = 0.0;
            } else {
                double xi = x(i, 0);
                double power_term = std::pow(xi / h, d / 4.0 - 0.5);
                double exp_term = std::exp(-0.5 * (xi + h));
                double bessel_arg = std::sqrt(h * xi);
                double bessel_order = d / 2.0 - 1.0;
                
                // Compute modified Bessel function I_ν
                double bessel_val = modified_bessel_i(bessel_order, bessel_arg);
                
                result(i) = 0.5 * power_term * exp_term * bessel_val;
            }
        }
        
        return result;
    }
    
    Vector cdf(const Vector& x) const override {
        Vector result(x.size());
        
        // CDF of non-central chi-squared is complex to compute
        // For simplicity, we use numerical integration or approximation
        // Here we use a simple approximation based on the central chi-squared when lambda is small
        
        for (int i = 0; i < x.size(); ++i) {
            if (x(i) <= 0) {
                result(i) = 0.0;
            } else if (lambda < 1e-6) {
                // Use central chi-squared approximation
                if (n == 1) {
                    result(i) = std::erf(std::sqrt(x(i) / 2.0));
                } else if (n == 2) {
                    result(i) = 1.0 - std::exp(-x(i) / 2.0);
                } else {
                    // Incomplete gamma approximation
                    result(i) = incomplete_gamma_cdf(n / 2.0, x(i) / 2.0);
                }
            } else {
                // For non-central case, use numerical integration
                // This is a placeholder - could be improved with boost or other libraries
                result(i) = numerical_cdf(x(i));
            }
        }
        
        return result;
    }
    
    Vector mean() const override {
        Vector mu(1);
        mu(0) = static_cast<double>(n) + lambda;
        return mu;
    }
    
    Matrix cov() const override {
        Matrix P(1, 1);
        P(0, 0) = 2.0 * static_cast<double>(n) + 4.0 * lambda;
        return P;
    }
    
    std::string desc() const override {
        return "non-central chi-squared distribution";
    }
    
    int length() const override {
        return 1;
    }
    
    // Additional statistical moments
    
    double var() const override {
        return 2.0 * static_cast<double>(n) + 4.0 * lambda;
    }
    
    double std() const override {
        return std::sqrt(var());
    }
    
    double skew() const override {
        double d = static_cast<double>(n);
        double h = lambda;
        return (d + 3.0 * h) * std::pow(2.0 / (d + 2.0 * h), 1.5);
    }
    
    double kurt() const override {
        double d = static_cast<double>(n);
        double h = lambda;
        return 12.0 * (d + 4.0 * h) / std::pow(d + 2.0 * h, 2.0);
    }
    
    std::string symbolic() const override {
        return "ncchi2(" + std::to_string(n) + "," + std::to_string(lambda) + ")";
    }
    
private:
    // Modified Bessel function of the first kind I_ν(x)
    // Using series expansion for numerical computation
    double modified_bessel_i(double nu, double x) const {
        if (x == 0.0) {
            return (nu == 0.0) ? 1.0 : 0.0;
        }
        
        const int max_iterations = 1000;
        const double epsilon = 1e-12;
        
        // Series expansion: I_ν(x) = Σ (x/2)^(ν+2k) / (k! * Γ(ν+k+1))
        double half_x = x / 2.0;
        double sum = 0.0;
        double term = std::pow(half_x, nu) / std::tgamma(nu + 1.0);
        
        for (int k = 0; k < max_iterations; ++k) {
            sum += term;
            if (std::abs(term) < epsilon * std::abs(sum) && k > 0) break;
            
            term *= (half_x * half_x) / ((k + 1.0) * (nu + k + 1.0));
        }
        
        return sum;
    }
    
    // Helper function for incomplete gamma CDF (regularized)
    double incomplete_gamma_cdf(double a, double x) const {
        if (x <= 0) return 0.0;
        if (x > 100) return 1.0;
        
        // Use series expansion for small x, continued fraction for large x
        if (x < a + 1.0) {
            return gamma_series(a, x);
        } else {
            return 1.0 - gamma_continued_fraction(a, x);
        }
    }
    
    double gamma_series(double a, double x) const {
        const int max_iterations = 1000;
        const double epsilon = 1e-10;
        
        double sum = 1.0 / a;
        double term = 1.0 / a;
        
        for (int n = 1; n < max_iterations; ++n) {
            term *= x / (a + n);
            sum += term;
            if (std::abs(term) < epsilon * std::abs(sum)) break;
        }
        
        return sum * std::exp(-x + a * std::log(x) - std::lgamma(a));
    }
    
    double gamma_continued_fraction(double a, double x) const {
        const int max_iterations = 1000;
        const double epsilon = 1e-10;
        const double tiny = 1e-30;
        
        double b = x + 1.0 - a;
        double c = 1.0 / tiny;
        double d = 1.0 / b;
        double h = d;
        
        for (int i = 1; i <= max_iterations; ++i) {
            double an = -i * (i - a);
            b += 2.0;
            d = an * d + b;
            if (std::abs(d) < tiny) d = tiny;
            c = b + an / c;
            if (std::abs(c) < tiny) c = tiny;
            d = 1.0 / d;
            double delta = d * c;
            h *= delta;
            if (std::abs(delta - 1.0) < epsilon) break;
        }
        
        return h * std::exp(-x + a * std::log(x) - std::lgamma(a));
    }
    
    // Numerical CDF by integrating PDF
    double numerical_cdf(double x_val) const {
        const int num_points = 1000;
        double dx = x_val / num_points;
        double sum = 0.0;
        
        for (int i = 0; i < num_points; ++i) {
            double x = (i + 0.5) * dx;
            Matrix x_mat(1, 1);
            x_mat(0, 0) = x;
            sum += pdf(x_mat)(0) * dx;
        }
        
        return sum;
    }
};
