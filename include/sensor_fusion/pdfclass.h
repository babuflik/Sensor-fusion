#pragma once
#include <Eigen/Dense>
#include <vector>
#include <string>
#include <random>
#include <cmath>

/**
 * @brief PDFCLASS is the parent class for all probability distribution families
 * 
 * This is the C++ implementation of MATLAB's pdfclass.m
 * All derived distributions inherit these common methods.
 */
class PDFClass {
public:
    using Vector = Eigen::VectorXd;
    using Matrix = Eigen::MatrixXd;
    
    int MC = 1000;              // Number of Monte Carlo samples (default)
    long state;                 // Random state
    std::vector<std::string> xlabel; // Labels for variables
    
    // Virtual destructor for proper cleanup
    virtual ~PDFClass() = default;
    
    // Pure virtual methods that must be implemented by derived classes
    
    /**
     * @brief Generate random samples from the distribution
     * @param n Number of samples (default: 1)
     * @return Matrix of samples (n x dim)
     */
    virtual Matrix rand(int n = 1) = 0;
    
    /**
     * @brief Evaluate probability density function
     * @param x Points at which to evaluate PDF (N x dim)
     * @return Vector of PDF values (N x 1)
     */
    virtual Vector pdf(const Matrix& x) const = 0;
    
    /**
     * @brief Evaluate cumulative distribution function
     * @param x Points at which to evaluate CDF (N x dim or N x 1 for scalar)
     * @return Vector of CDF values (N x 1)
     */
    virtual Vector cdf(const Vector& x) const = 0;
    
    /**
     * @brief Compute expectation/mean
     * @return Mean vector
     */
    virtual Vector mean() const = 0;
    
    /**
     * @brief Compute covariance matrix
     * @return Covariance matrix
     */
    virtual Matrix cov() const = 0;
    
    /**
     * @brief Get description of the distribution
     * @return String description
     */
    virtual std::string desc() const = 0;
    
    /**
     * @brief Get dimension of the distribution
     * @return Dimension
     */
    virtual int length() const = 0;
    
    // Optional virtual methods with default implementations
    
    /**
     * @brief Compute median
     * @return Median vector (default: same as mean for symmetric distributions)
     */
    virtual Vector median() const {
        return mean();
    }
    
    /**
     * @brief Compute mode
     * @return Mode vector (default: same as mean for unimodal distributions)
     */
    virtual Vector mode() const {
        return mean();
    }
    
    /**
     * @brief Compute variance (for 1D distributions)
     * @return Variance (throws error for multivariate)
     */
    virtual double var() const {
        if (length() > 1) {
            throw std::runtime_error("var() not defined for multi-dimensional distributions, use cov()");
        }
        return cov()(0, 0);
    }
    
    /**
     * @brief Compute standard deviation (for 1D distributions)
     * @return Standard deviation
     */
    virtual double std() const {
        return std::sqrt(var());
    }
    
    /**
     * @brief Compute skewness (for 1D distributions)
     * @return Skewness (default: 0 for symmetric distributions)
     */
    virtual double skew() const {
        if (length() > 1) {
            throw std::runtime_error("skew() not defined for multi-dimensional distributions");
        }
        return 0.0;
    }
    
    /**
     * @brief Compute kurtosis (for 1D distributions)
     * @return Kurtosis (default: 0 for normal distributions, excess kurtosis)
     */
    virtual double kurt() const {
        if (length() > 1) {
            throw std::runtime_error("kurt() not defined for multi-dimensional distributions");
        }
        return 0.0;
    }
    
    /**
     * @brief Symbolic representation
     * @return String representation of distribution
     */
    virtual std::string symbolic() const {
        return desc();
    }
    
protected:
    /**
     * @brief Generate random state
     */
    long generate_random_state() const {
        static std::random_device rd;
        static std::mt19937 gen(rd());
        static std::uniform_real_distribution<> distrib(0.0, 1e6);
        return static_cast<long>(std::round(distrib(gen)));
    }
    
    /**
     * @brief Get random number generator with current state
     */
    std::mt19937& get_generator() {
        static std::mt19937 gen(state);
        return gen;
    }
};
