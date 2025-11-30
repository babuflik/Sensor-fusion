#pragma once
#include "pdfclass.h"
#include <Eigen/Dense>
#include <vector>
#include <cmath>
#include <stdexcept>
#include <string>
#include <random>
#include <sstream>
#include <iomanip>

// Simple check for covariance matrix validity
inline bool is_valid_covariance(const Eigen::MatrixXd& P) {
    if (P.rows() != P.cols()) return false;
    if (!P.isApprox(P.transpose())) return false;
    Eigen::LLT<Eigen::MatrixXd> llt(P);
    return llt.info() == Eigen::Success;
}

/**
 * @brief NDist defines the Gaussian/Normal distribution N(mu, P)
 * C++ implementation of MATLAB ndist.m
 */
class NDist : public PDFClass {
public:
    using Vector = Eigen::VectorXd;
    using Matrix = Eigen::MatrixXd;
    
    // MATLAB-like properties
    Vector mu;
    Matrix P;
    
    // Constructors
    
    /**
     * @brief Empty constructor
     */
    NDist() : PDFClass() {
        mu = Vector();
        P = Matrix();
        state = generate_random_state();
    }
    
    /**
     * @brief Full constructor N(mu, P)
     */
    NDist(const Vector& mu_, const Matrix& P_) : PDFClass() {
        initialize(mu_, P_);
        state = generate_random_state();
    }

    /**
     * @brief Copy constructor
     */
    NDist(const NDist& other) : PDFClass() {
        mu = other.mu;
        P = other.P;
        MC = other.MC;
        xlabel = other.xlabel;
        state = generate_random_state();
    }
    
    // PDFClass virtual method implementations
    
    /**
     * @brief Generate random samples from N(mu, P)
     */
    Matrix rand(int n = 1) override {
        if (mu.size() == 0) {
            throw std::runtime_error("NDist::rand: Distribution not initialized");
        }
        
        int dim = mu.size();
        Matrix samples(n, dim);
        
        Eigen::LLT<Matrix> llt(P);
        if (llt.info() != Eigen::Success) {
            throw std::runtime_error("NDist::rand: Covariance matrix not positive definite");
        }
        Matrix L = llt.matrixL();
        
        std::mt19937& gen = get_generator();
        std::normal_distribution<double> dist(0.0, 1.0);
        
        for (int i = 0; i < n; ++i) {
            Vector z(dim);
            for (int j = 0; j < dim; ++j) {
                z(j) = dist(gen);
            }
            samples.row(i) = (mu + L * z).transpose();
        }
        
        return samples;
    }
    
    /**
     * @brief Evaluate probability density function
     */
    Vector pdf(const Matrix& x) const override {
        int N = x.rows();
        int dim = mu.size();
        
        if (x.cols() != dim) {
            throw std::runtime_error("NDist::pdf: x must have same dimension as mu");
        }
        
        Vector result(N);
        double det_P = P.determinant();
        
        if (det_P <= 0) {
            result.setZero();
            return result;
        }
        
        double coeff = 1.0 / std::sqrt(std::pow(2.0 * M_PI, dim) * det_P);
        Eigen::LLT<Matrix> llt(P);
        
        for (int i = 0; i < N; ++i) {
            Vector diff = x.row(i).transpose() - mu;
            Vector P_inv_diff = llt.solve(diff);
            double exponent = -0.5 * diff.dot(P_inv_diff);
            result(i) = coeff * std::exp(exponent);
        }
        
        return result;
    }
    
    /**
     * @brief Evaluate cumulative distribution function (1D only)
     */
    Vector cdf(const Vector& x) const override {
        if (mu.size() != 1) {
            throw std::runtime_error("NDist::cdf: Only implemented for 1D distributions");
        }
        
        Vector result(x.size());
        double sigma = std::sqrt(P(0, 0));
        
        for (int i = 0; i < x.size(); ++i) {
            double z = (x(i) - mu(0)) / sigma;
            result(i) = 0.5 * (1.0 + std::erf(z / std::sqrt(2.0)));
        }
        
        return result;
    }
    
    Vector mean() const override { return mu; }
    Matrix cov() const override { return P; }
    std::string desc() const override { return "Normal distribution"; }
    int length() const override { return mu.size(); }
    
    std::string symbolic() const override {
        return "N(" + vector_to_string(mu) + ", " + matrix_to_string(P) + ")";
    }
    
    // Setters
    
    void set_mu(const Vector& new_mu) {
        if (mu.size() > 0 && new_mu.size() != mu.size()) {
            throw std::runtime_error("NDist::set_mu: Cannot change the dimension of mu");
        }
        mu = new_mu;
    }

    void set_P(const Matrix& new_P) {
        if (P.rows() > 0 && new_P.rows() != P.rows()) {
            throw std::runtime_error("NDist::set_P: Cannot change the dimension of P");
        }
        if (!is_valid_covariance(new_P)) {
            throw std::runtime_error("NDist::set_P: P is not a valid covariance");
        }
        P = new_P;
    }

    // Indexing (MATLAB-style subsref)
    
    NDist subsref(const std::vector<int>& indices) const {
        for (int i : indices) {
            if (i < 1 || i > mu.size()) {
                throw std::runtime_error("NDist::subsref: Index out of range");
            }
        }
        
        Vector new_mu(indices.size());
        Matrix new_P(indices.size(), indices.size());
        std::vector<std::string> new_xlabel(indices.size());

        for (size_t k = 0; k < indices.size(); ++k) {
            int i = indices[k] - 1;
            new_mu[k] = mu[i];
            new_xlabel[k] = xlabel[i];

            for (size_t j = 0; j < indices.size(); ++j) {
                int l = indices[j] - 1;
                new_P(k, j) = P(i, l);
            }
        }

        NDist Xi(new_mu, new_P);
        Xi.xlabel = new_xlabel;
        return Xi;
    }

private:
    void initialize(const Vector& mu_, const Matrix& P_) {
        int n = mu_.size();
        
        if (P_.rows() != n || P_.cols() != n) {
            throw std::runtime_error("NDIST: P must be square with same dimension as mu");
        }
        
        if (!P_.array().isFinite().all() || !mu_.array().isFinite().all()) {
            throw std::runtime_error("NDIST: mu and P must contain real, finite values");
        }
        
        if (!is_valid_covariance(P_)) {
            throw std::runtime_error("NDIST: P is not a valid covariance matrix");
        }
        
        mu = mu_;
        P = P_;
        
        xlabel.resize(n);
        for (int i = 0; i < n; ++i) {
            xlabel[i] = "x" + std::to_string(i + 1);
        }
    }
    
    std::string vector_to_string(const Vector& v) const {
        if (v.size() == 0) return "[]";
        if (v.size() == 1) return std::to_string(v(0));
        
        std::ostringstream oss;
        oss << "[";
        for (int i = 0; i < v.size(); ++i) {
            oss << v(i);
            if (i < v.size() - 1) oss << ", ";
        }
        oss << "]";
        return oss.str();
    }
    
    std::string matrix_to_string(const Matrix& m) const {
        if (m.rows() == 0) return "[]";
        if (m.rows() == 1 && m.cols() == 1) return std::to_string(m(0,0));
        
        std::ostringstream oss;
        oss << "[" << m.rows() << "x" << m.cols() << " matrix]";
        return oss.str();
    }
};
