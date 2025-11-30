#pragma once
#include "pdfclass.h"
#include <Eigen/Dense>
#include <random>
#include <cmath>

/**
 * @brief Chi2Dist - Chi-squared distribution χ²(n)
 * C++ implementation of MATLAB chi2dist.m
 */
class Chi2Dist : public PDFClass {
public:
    using Vector = Eigen::VectorXd;
    using Matrix = Eigen::MatrixXd;
    
    int n;  // Degrees of freedom
    
    // Constructors
    
    Chi2Dist() : PDFClass() {
        n = 1;
        state = generate_random_state();
        xlabel = {"x1"};
    }
    
    Chi2Dist(int n_) : PDFClass() {
        if (n_ <= 0) {
            throw std::runtime_error("Chi2Dist: degrees of freedom must be positive");
        }
        n = n_;
        state = generate_random_state();
        xlabel = {"x1"};
    }
    
    // PDFClass implementations
    
    Matrix rand(int num_samples = 1) override {
        Matrix samples(num_samples, 1);
        
        std::mt19937& gen = get_generator();
        std::chi_squared_distribution<double> dist(n);
        
        for (int i = 0; i < num_samples; ++i) {
            samples(i, 0) = dist(gen);
        }
        
        return samples;
    }
    
    Vector pdf(const Matrix& x) const override {
        int N = x.rows();
        
        if (x.cols() != 1) {
            throw std::runtime_error("Chi2Dist::pdf: x must be 1D");
        }
        
        Vector result(N);
        double k = static_cast<double>(n);
        
        // PDF: (1 / (2^(k/2) * Γ(k/2))) * x^(k/2-1) * exp(-x/2)
        double log_norm = 0.5 * k * std::log(2.0) + std::lgamma(k / 2.0);
        
        for (int i = 0; i < N; ++i) {
            if (x(i, 0) <= 0) {
                result(i) = 0.0;
            } else {
                double log_pdf = (k / 2.0 - 1.0) * std::log(x(i, 0)) - x(i, 0) / 2.0 - log_norm;
                result(i) = std::exp(log_pdf);
            }
        }
        
        return result;
    }
    
    Vector cdf(const Vector& x) const override {
        Vector result(x.size());
        
        // Use incomplete gamma function: P(k/2, x/2)
        // This is a simplified implementation
        for (int i = 0; i < x.size(); ++i) {
            if (x(i) <= 0) {
                result(i) = 0.0;
            } else {
                // Approximate using error function for n=1,2
                if (n == 1) {
                    result(i) = std::erf(std::sqrt(x(i) / 2.0));
                } else if (n == 2) {
                    result(i) = 1.0 - std::exp(-x(i) / 2.0);
                } else {
                    // For general case, use numerical approximation
                    result(i) = incomplete_gamma_cdf(n / 2.0, x(i) / 2.0);
                }
            }
        }
        
        return result;
    }
    
    Vector mean() const override {
        Vector mu(1);
        mu(0) = static_cast<double>(n);
        return mu;
    }
    
    Matrix cov() const override {
        Matrix P(1, 1);
        P(0, 0) = 2.0 * n;
        return P;
    }
    
    std::string desc() const override {
        return "Chi-squared distribution";
    }
    
    int length() const override {
        return 1;
    }
    
    std::string symbolic() const override {
        return "χ²(" + std::to_string(n) + ")";
    }
    
    Vector mode() const override {
        Vector m(1);
        m(0) = std::max(0.0, static_cast<double>(n - 2));
        return m;
    }
    
    double skew() const override {
        return std::sqrt(8.0 / n);
    }
    
    double kurt() const override {
        return 12.0 / n;  // Excess kurtosis
    }
    
private:
    // Simplified incomplete gamma CDF approximation
    double incomplete_gamma_cdf(double a, double x) const {
        // Use series expansion for small x, continued fraction for large x
        if (x < a + 1.0) {
            return gamma_series(a, x);
        } else {
            return 1.0 - gamma_continued_fraction(a, x);
        }
    }
    
    double gamma_series(double a, double x) const {
        const int max_iter = 100;
        const double eps = 1e-10;
        
        double sum = 1.0 / a;
        double term = 1.0 / a;
        
        for (int n = 1; n < max_iter; ++n) {
            term *= x / (a + n);
            sum += term;
            if (std::abs(term) < eps * std::abs(sum)) break;
        }
        
        return sum * std::exp(-x + a * std::log(x) - std::lgamma(a));
    }
    
    double gamma_continued_fraction(double a, double x) const {
        const int max_iter = 100;
        const double eps = 1e-10;
        const double small = 1e-30;
        
        double b = x + 1.0 - a;
        double c = 1.0 / small;
        double d = 1.0 / b;
        double h = d;
        
        for (int i = 1; i < max_iter; ++i) {
            double an = -i * (i - a);
            b += 2.0;
            d = an * d + b;
            if (std::abs(d) < small) d = small;
            c = b + an / c;
            if (std::abs(c) < small) c = small;
            d = 1.0 / d;
            double delta = d * c;
            h *= delta;
            if (std::abs(delta - 1.0) < eps) break;
        }
        
        return h * std::exp(-x + a * std::log(x) - std::lgamma(a));
    }
};
