#ifndef EMPDIST_H
#define EMPDIST_H

#include "pdfclass.h"
#include <Eigen/Dense>
#include <random>
#include <algorithm>
#include <vector>

/**
 * @brief Empirical distribution from samples
 * C++ implementation of MATLAB empdist.m
 * 
 * Stores samples and provides resampling, empirical PDF/CDF
 */
class EmpDist : public PDFClass {
public:
    using Vector = Eigen::VectorXd;
    using Matrix = Eigen::MatrixXd;
    
private:
    std::vector<Eigen::VectorXd> samples_;
    int dim_;
    int n_samples_;
    
public:
    // Constructor from samples
    EmpDist(const std::vector<Eigen::VectorXd>& samples)
        : samples_(samples), n_samples_(samples.size()) {
        if (samples.empty()) {
            throw std::invalid_argument("EmpDist: samples cannot be empty");
        }
        dim_ = samples[0].size();
        
        // Check all samples have same dimension
        for (const auto& s : samples) {
            if (s.size() != dim_) {
                throw std::invalid_argument("EmpDist: all samples must have same dimension");
            }
        }
        state = generate_random_state();
    }
    
    // Constructor from matrix (each row is a sample)
    EmpDist(const Eigen::MatrixXd& data) {
        n_samples_ = data.rows();
        dim_ = data.cols();
        
        if (n_samples_ == 0) {
            throw std::invalid_argument("EmpDist: data cannot be empty");
        }
        
        samples_.reserve(n_samples_);
        for (int i = 0; i < n_samples_; ++i) {
            samples_.push_back(data.row(i).transpose());
        }
        state = generate_random_state();
    }
    
    // Generate random samples (bootstrap resampling)
    Matrix rand(int n = 1) override {
        Matrix result(n, dim_);
        std::mt19937& gen = get_generator();
        std::uniform_int_distribution<int> dist(0, n_samples_ - 1);
        
        for (int i = 0; i < n; ++i) {
            int idx = dist(gen);
            result.row(i) = samples_[idx].transpose();
        }
        
        return result;
    }
    
    // Probability density function (kernel density estimation)
    Vector pdf(const Matrix& x) const override {
        int N = x.rows();
        if (x.cols() != dim_) {
            throw std::invalid_argument("EmpDist::pdf: dimension mismatch");
        }
        
        Vector result(N);
        
        // Simple kernel density estimation with Gaussian kernel
        // Bandwidth selection: Silverman's rule of thumb
        double h = std::pow(n_samples_, -1.0 / (dim_ + 4.0));
        double norm = std::pow(2.0 * M_PI * h * h, dim_ / 2.0);
        
        for (int i = 0; i < N; ++i) {
            // Compute KDE for this point
            double sum = 0.0;
            Eigen::VectorXd x_i = x.row(i).transpose();
            
            for (const auto& sample : samples_) {
                Eigen::VectorXd diff = (x_i - sample) / h;
                double exponent = -0.5 * diff.squaredNorm();
                sum += std::exp(exponent);
            }
            
            result(i) = sum / (n_samples_ * norm);
        }
        
        return result;
    }
    
    // Cumulative distribution function (empirical CDF, 1D only)
    Vector cdf(const Vector& x) const override {
        if (dim_ != 1) {
            throw std::runtime_error("EmpDist::cdf: only implemented for 1D");
        }
        
        int N = x.size();
        Vector result(N);
        
        for (int i = 0; i < N; ++i) {
            double val = x(i);
            int count = 0;
            for (const auto& sample : samples_) {
                if (sample(0) <= val) {
                    count++;
                }
            }
            result(i) = static_cast<double>(count) / n_samples_;
        }
        
        return result;
    }
    
    // Sample mean
    Vector mean() const override {
        Vector m = Vector::Zero(dim_);
        for (const auto& sample : samples_) {
            m += sample;
        }
        return m / n_samples_;
    }
    
    // Sample covariance
    Matrix cov() const override {
        Vector m = mean();
        Matrix C = Matrix::Zero(dim_, dim_);
        
        for (const auto& sample : samples_) {
            Vector diff = sample - m;
            C += diff * diff.transpose();
        }
        
        return C / (n_samples_ - 1);
    }
    
    // Median (component-wise for multivariate)
    Vector median() const override {
        Vector med(dim_);
        
        for (int d = 0; d < dim_; ++d) {
            std::vector<double> values;
            values.reserve(n_samples_);
            for (const auto& sample : samples_) {
                values.push_back(sample(d));
            }
            
            std::nth_element(values.begin(), 
                           values.begin() + n_samples_ / 2, 
                           values.end());
            med(d) = values[n_samples_ / 2];
        }
        
        return med;
    }
    
    // Distribution length
    int length() const override {
        return dim_;
    }
    
    // Description string
    std::string desc() const override {
        return "Empirical distribution (n=" + std::to_string(n_samples_) + 
               ", dim=" + std::to_string(dim_) + ")";
    }
    
    // Accessors
    int n_samples() const { return n_samples_; }
    const std::vector<Eigen::VectorXd>& samples() const { return samples_; }
    
    // Get samples as matrix
    Eigen::MatrixXd get_samples_matrix() const {
        Eigen::MatrixXd data(n_samples_, dim_);
        for (int i = 0; i < n_samples_; ++i) {
            data.row(i) = samples_[i].transpose();
        }
        return data;
    }
};

#endif // EMPDIST_H
