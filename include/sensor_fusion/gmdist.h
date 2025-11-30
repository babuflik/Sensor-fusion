#pragma once
#include "pdfclass.h"
#include <Eigen/Dense>
#include <vector>
#include <random>
#include <cmath>

/**
 * @brief GMDist - Gaussian Mixture distribution
 * C++ implementation of MATLAB gmdist.m
 * Mixture of K Gaussian components
 */
class GMDist : public PDFClass {
public:
    using Vector = Eigen::VectorXd;
    using Matrix = Eigen::MatrixXd;
    
    std::vector<Vector> mu;        // Means of components
    std::vector<Matrix> P;         // Covariances of components
    Vector w;                      // Mixture weights
    int K;                         // Number of components
    
    GMDist() : PDFClass() {
        K = 0;
        state = generate_random_state();
    }
    
    GMDist(const std::vector<Vector>& mu_, const std::vector<Matrix>& P_, 
           const Vector& w_) : PDFClass() {
        if (mu_.size() != P_.size() || mu_.size() != w_.size()) {
            throw std::runtime_error("GMDist: mu, P, w must have same number of components");
        }
        if (std::abs(w_.sum() - 1.0) > 1e-10) {
            throw std::runtime_error("GMDist: weights must sum to 1");
        }
        
        mu = mu_;
        P = P_;
        w = w_;
        K = mu_.size();
        state = generate_random_state();
        
        if (K > 0 && mu[0].size() > 0) {
            xlabel.resize(mu[0].size());
            for (int i = 0; i < mu[0].size(); ++i) {
                xlabel[i] = "x" + std::to_string(i + 1);
            }
        }
    }
    
    Matrix rand(int n = 1) override {
        if (K == 0) {
            throw std::runtime_error("GMDist::rand: Distribution not initialized");
        }
        
        int dim = mu[0].size();
        Matrix samples(n, dim);
        
        std::mt19937& gen = get_generator();
        std::discrete_distribution<int> component_dist(w.data(), w.data() + K);
        std::normal_distribution<double> norm_dist(0.0, 1.0);
        
        for (int i = 0; i < n; ++i) {
            // Sample component
            int k = component_dist(gen);
            
            // Sample from k-th Gaussian
            Eigen::LLT<Matrix> llt(P[k]);
            Matrix L = llt.matrixL();
            
            Vector z(dim);
            for (int j = 0; j < dim; ++j) {
                z(j) = norm_dist(gen);
            }
            
            samples.row(i) = (mu[k] + L * z).transpose();
        }
        
        return samples;
    }
    
    Vector pdf(const Matrix& x) const override {
        if (K == 0) {
            throw std::runtime_error("GMDist::pdf: Distribution not initialized");
        }
        
        int N = x.rows();
        int dim = mu[0].size();
        Vector result = Vector::Zero(N);
        
        for (int k = 0; k < K; ++k) {
            double det_P = P[k].determinant();
            if (det_P <= 0) continue;
            
            double coeff = w(k) / std::sqrt(std::pow(2.0 * M_PI, dim) * det_P);
            Eigen::LLT<Matrix> llt(P[k]);
            
            for (int i = 0; i < N; ++i) {
                Vector diff = x.row(i).transpose() - mu[k];
                Vector P_inv_diff = llt.solve(diff);
                double exponent = -0.5 * diff.dot(P_inv_diff);
                result(i) += coeff * std::exp(exponent);
            }
        }
        
        return result;
    }
    
    Vector cdf(const Vector& x) const override {
        throw std::runtime_error("GMDist::cdf: Not implemented for mixture distributions");
    }
    
    Vector mean() const override {
        if (K == 0) {
            return Vector();
        }
        
        int dim = mu[0].size();
        Vector m = Vector::Zero(dim);
        
        for (int k = 0; k < K; ++k) {
            m += w(k) * mu[k];
        }
        
        return m;
    }
    
    Matrix cov() const override {
        if (K == 0) {
            return Matrix();
        }
        
        int dim = mu[0].size();
        Vector mean_vec = mean();
        Matrix C = Matrix::Zero(dim, dim);
        
        for (int k = 0; k < K; ++k) {
            Vector diff = mu[k] - mean_vec;
            C += w(k) * (P[k] + diff * diff.transpose());
        }
        
        return C;
    }
    
    std::string desc() const override {
        return "Gaussian Mixture distribution";
    }
    
    int length() const override {
        return (K > 0) ? mu[0].size() : 0;
    }
    
    std::string symbolic() const override {
        return "GM(K=" + std::to_string(K) + ")";
    }
};
