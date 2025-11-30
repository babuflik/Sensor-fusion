#pragma once
#include "sig.h"
#include "utils_sigsys.h"
#include <Eigen/Dense>
#include <functional>
#include <random>
#include <vector>
#include <string>
#include <cmath> // For std::pow, std::sqrt, M_PI

class SensorMod {
public:
    using DynFunc = std::function<Eigen::VectorXd(double,
                                                  const Eigen::VectorXd&,
                                                  const Eigen::VectorXd&,
                                                  const Eigen::VectorXd&)>;

    // -----------------------------------------------------
    // Existing members
    // -----------------------------------------------------
    Eigen::Vector4i nn;      // [nx, nu, ny, nth]

    Eigen::VectorXd x0;
    Eigen::VectorXd th;
    Eigen::MatrixXd pv; // process noise
    Eigen::MatrixXd pe; // measurement noise (ny x ny)
    DynFunc h;          // sensorfunktion
    double fs = 0.0;

    std::string name;
    std::string xlabel, ylabel, ulabel, thlabel;

    // Constructor
    SensorMod(const DynFunc& hFunc, const Eigen::Vector4i& nn_)
        : h(hFunc), nn(nn_)
    {
        x0 = Eigen::VectorXd::Zero(nn[0]);
        th = Eigen::VectorXd::Zero(nn[3]);
        pv = Eigen::MatrixXd::Zero(nn[0], nn[0]);
        pe = Eigen::MatrixXd::Zero(nn[2], nn[2]);
    }
    
    // -----------------------------------------------------
    // 1. simulate (original function)
    // -----------------------------------------------------
    Sig simulate(const Eigen::VectorXd& t,
                 const Eigen::MatrixXd* xOverride = nullptr,
                 const Eigen::MatrixXd* uOverride = nullptr,
                 int MC = 0)
    {
        int N = t.size();
        int nx = nn[0];
        int nu = nn[1];
        int ny = nn[2];

        Eigen::MatrixXd x = xOverride ? *xOverride : Eigen::MatrixXd::Zero(nx, N).colwise() + x0;
        Eigen::MatrixXd u = uOverride ? *uOverride : Eigen::MatrixXd::Zero(nu, N);
        Eigen::MatrixXd y(ny, N);

        std::random_device rd;
        std::mt19937 gen(rd());
        // NOTE: This line uses only the diagonal (assumes uncorrelated noise)
        Eigen::VectorXd pe_std = pe.diagonal().cwiseSqrt(); 

        for (int k = 0; k < N; ++k) {
            Eigen::VectorXd yk = h(t[k], x.col(k), u.col(k), th);
            for (int i = 0; i < ny; ++i) {
                std::normal_distribution<> d(0.0, pe_std[i]);
                yk[i] += d(gen);
            }
            y.col(k) = yk;
        }

        Sig z;
        z.y = y;
        z.t = t;
        z.x = x;
        z.u = u;

        if (MC > 0) {
            std::normal_distribution<> d_th(0.0, 1.0);
            for (int i = 0; i < MC; ++i) {
                Eigen::VectorXd thMC = th;
                for (int j = 0; j < nn[3]; ++j) thMC[j] += d_th(gen) * std::sqrt(pv(j,j));
                Eigen::MatrixXd yMC_i(ny, N);
                for (int k = 0; k < N; ++k) {
                    yMC_i.col(k) = h(t[k], x.col(k), u.col(k), thMC);
                }
                z.yMC.push_back(yMC_i);
                z.xMC.push_back(x);
            }
        }

        return z;
    }

    // -----------------------------------------------------
    // 2. likelihood_function (new function, corresponds to lh1/lh2)
    // -----------------------------------------------------
    /**
    * @brief Compute the total likelihood L(x | y) over a grid of state points.
    * @param y Measurement data as a Sig object.
    * @param xGrid Matrix of state points to evaluate (N_grid x nx).
    * @param state_indices Not implemented, kept for compatibility with lh1/lh2.
    * @return Eigen::VectorXd Vector with likelihood values for each grid point.
     */
    Eigen::VectorXd likelihood_function(
        const Sig& y, 
        const Eigen::MatrixXd& xGrid,
        const std::vector<int>& state_indices = {} // {0} is the default in C++
    )
    {
        int N_grid = xGrid.rows();
        int N_time = y.t.size();
        int ny = nn[2];
        int nx = nn[0];
        
        if (xGrid.cols() != nx) {
            throw std::runtime_error("likelihood_function: xGrid must have nx columns.");
        }
        if (pe.rows() != ny || pe.cols() != ny) {
            throw std::runtime_error("likelihood_function: Measurement noise covariance Pe has incorrect dimensions.");
        }

        Eigen::VectorXd L = Eigen::VectorXd::Ones(N_grid); // Initialize likelihood to 1.0
        Eigen::VectorXd mu_v = Eigen::VectorXd::Zero(ny);  // Measurement noise mean is zero

        // Iterate over all measurement time points (k)
        for (int k = 0; k < N_time; ++k) {
            Eigen::MatrixXd epsi_grid(N_grid, ny); // Measurement error matrix (N_grid x ny)
            
            double t_k = y.t[k];
            // NOTE: y.u can be empty (nu=0). Check whether u should be used or ignored.
            Eigen::VectorXd u_k = (y.u.cols() > 0) ? y.u.col(k).eval() : Eigen::VectorXd::Zero(nn[1]);
            
            for (int i = 0; i < N_grid; ++i) {
                Eigen::VectorXd x_i = xGrid.row(i).transpose();
                Eigen::VectorXd y_hat_i = h(t_k, x_i, u_k, th);
                
                // Measurement y_k - prediction y_hat_i
                Eigen::VectorXd epsi_i = y.y.col(k) - y_hat_i;
                epsi_grid.row(i) = epsi_i.transpose();
            }

            // Compute PDF for the error p(epsi_k | Pe) = p(y_k | x)
            Eigen::VectorXd p_y_given_x = multivariate_normal_pdf(pe, mu_v, epsi_grid);
            
            // Multiplicera Likelihood: L_total = L_prev * p(y_k | x)
            L.array() *= p_y_given_x.array();
        }

        return L;
    }

    /**
     * @brief Least Squares estimation
     * @param y Measurement signal
     * @return Pair of (state estimate signal, updated sensor model)
     * 
     * Computes: argmin sum((y - H*x)^T * (y - H*x))
     * Uses numerical gradient to linearize h around x0
     */
    std::pair<Sig, SensorMod> ls(const Sig& y) {
        int N = y.t.size();
        int nx = nn[0];
        int ny = nn[2];
        
        if (pe.rows() == 0 || pe.cols() == 0) {
            throw std::runtime_error("sensormod.ls: pe must be defined");
        }
        
        Eigen::MatrixXd x_est(N, nx);
        Eigen::MatrixXd y_est(N, ny);
        std::vector<Eigen::MatrixXd> Px_vec, Py_vec;
        
        for (int k = 0; k < N; ++k) {
            double t_k = y.t[k];
            Eigen::VectorXd u_k;
            if (y.u.cols() > k) {
                u_k = y.u.col(k);
            } else {
                u_k = Eigen::VectorXd::Zero(nn[1]);
            }
            
            // Compute Jacobian H = dh/dx at x0
            auto h_func = [this, t_k, u_k](const Eigen::VectorXd& x) {
                return this->h(t_k, x, u_k, this->th);
            };
            Eigen::MatrixXd H = numjac(h_func, x0);
            
            // Predicted measurement
            Eigen::VectorXd yhat = h(t_k, x0, u_k, th);
            
            // Least squares solution: x = (H^T H)^{-1} H^T (y - yhat + H*x0)
            Eigen::MatrixXd HtH = H.transpose() * H;
            Eigen::MatrixXd PP = HtH.completeOrthogonalDecomposition().pseudoInverse();
            Eigen::VectorXd ff = H.transpose() * (y.y.col(k) - yhat + H * x0);
            
            x_est.row(k) = (PP * ff).transpose();
            y_est.row(k) = (yhat + H * (x_est.row(k).transpose() - x0)).transpose();
            
            // Covariance: P = (H^T H)^{-1} H^T Pe H (H^T H)^{-1}
            Eigen::MatrixXd Px_k = PP * (H.transpose() * pe * H) * PP;
            Px_vec.push_back(Px_k);
            
            Eigen::MatrixXd Py_k = H * Px_k * H.transpose();
            Py_vec.push_back(Py_k);
        }
        
        Sig result;
        result.y = y_est.transpose();
        result.t = y.t;
        result.u = y.u;
        result.x = x_est.transpose();
        result.Py = Py_vec;
        result.Px = Px_vec;
        
        SensorMod shat = *this;
        if (N == 1) {
            shat.x0 = x_est.row(0).transpose();
        }
        
        return {result, shat};
    }
    
    /**
     * @brief Weighted Least Squares estimation
     * @param y Measurement signal
     * @return Pair of (state estimate signal, updated sensor model)
     * 
     * Computes: argmin sum((y - H*x)^T * Pe^{-1} * (y - H*x))
     * Uses numerical gradient to linearize h around x0
     */
    std::pair<Sig, SensorMod> wls(const Sig& y) {
        int N = y.t.size();
        int nx = nn[0];
        int ny = nn[2];
        
        if (pe.rows() == 0 || pe.cols() == 0) {
            throw std::runtime_error("sensormod.wls: pe must be defined");
        }
        
        Eigen::MatrixXd Rinv = pe.completeOrthogonalDecomposition().pseudoInverse();
        Eigen::MatrixXd x_est(N, nx);
        Eigen::MatrixXd y_est(N, ny);
        std::vector<Eigen::MatrixXd> Px_vec, Py_vec;
        
        for (int k = 0; k < N; ++k) {
            double t_k = y.t[k];
            Eigen::VectorXd u_k;
            if (y.u.cols() > k) {
                u_k = y.u.col(k);
            } else {
                u_k = Eigen::VectorXd::Zero(nn[1]);
            }
            
            // Compute Jacobian H = dh/dx at x0
            auto h_func = [this, t_k, u_k](const Eigen::VectorXd& x) {
                return this->h(t_k, x, u_k, this->th);
            };
            Eigen::MatrixXd H = numjac(h_func, x0);
            
            // Predicted measurement
            Eigen::VectorXd yhat = h(t_k, x0, u_k, th);
            
            // Weighted least squares: x = (H^T R^{-1} H)^{-1} H^T R^{-1} (y - yhat + H*x0)
            Eigen::MatrixXd HtRinvH = H.transpose() * Rinv * H;
            Eigen::MatrixXd PP = HtRinvH.completeOrthogonalDecomposition().pseudoInverse();
            Eigen::VectorXd ff = H.transpose() * Rinv * (y.y.col(k) - yhat + H * x0);
            
            x_est.row(k) = (PP * ff).transpose();
            y_est.row(k) = (yhat + H * (x_est.row(k).transpose() - x0)).transpose();
            
            // Covariance for WLS: P = (H^T R^{-1} H)^{-1}
            Eigen::MatrixXd Px_k = 0.5 * (PP + PP.transpose());  // Symmetrize
            Px_vec.push_back(Px_k);
            
            Eigen::MatrixXd Py_k = H * Px_k * H.transpose();
            Py_vec.push_back(Py_k);
        }
        
        Sig result;
        result.y = y_est.transpose();
        result.t = y.t;
        result.u = y.u;
        result.x = x_est.transpose();
        result.Py = Py_vec;
        result.Px = Px_vec;
        
        SensorMod shat = *this;
        if (N == 1) {
            shat.x0 = x_est.row(0).transpose();
        }
        
        return {result, shat};
    }
    
    /**
     * @brief Cramér-Rao Lower Bound computation
     * @param y Measurement signal (uses y.x as state, or x0 if empty)
     * @return Sig object with CRLB in Px
     * 
     * Computes the Fisher Information Matrix and its inverse
     */
    Sig crlb(const Sig* y = nullptr) {
        int nx = nn[0];
        int ny = nn[2];
        
        Eigen::VectorXd x_eval = (y && y->x.cols() > 0) ? y->x.col(0).eval() : x0;
        Eigen::VectorXd u_eval = (y && y->u.cols() > 0) ? y->u.col(0).eval() : Eigen::VectorXd::Zero(nn[1]);
        double t_eval = (y && y->t.size() > 0) ? y->t[0] : 0.0;
        
        if (y && y->t.size() > 1) {
            throw std::runtime_error("CRLB can only be computed for one sample at a time");
        }
        
        // Compute Jacobian H = dh/dx
        auto h_func = [this, t_eval, u_eval](const Eigen::VectorXd& x) {
            return this->h(t_eval, x, u_eval, this->th);
        };
        Eigen::MatrixXd H = numjac(h_func, x_eval);
        
        // Fisher Information Matrix: I = H^T * Pe^{-1} * H
        Eigen::MatrixXd Rinv = pe.completeOrthogonalDecomposition().pseudoInverse();
        Eigen::MatrixXd FIM = H.transpose() * Rinv * H;
        
        // CRLB = I^{-1}
        Eigen::MatrixXd CRLB = FIM.completeOrthogonalDecomposition().pseudoInverse();
        
        // Create result signal
        Sig result;
        result.x = x_eval;
        result.t = Eigen::VectorXd::Constant(1, t_eval);
        result.u = u_eval;
        result.y = h(t_eval, x_eval, u_eval, th);
        result.Px = {CRLB};
        result.Py = {H * CRLB * H.transpose()};
        
        return result;
    }

private:
    // -----------------------------------------------------
    // 3. multivariate_normal_pdf (New internal helper function)
    // -----------------------------------------------------
    Eigen::VectorXd multivariate_normal_pdf(
        const Eigen::MatrixXd& P, 
        const Eigen::VectorXd& mu, 
        const Eigen::MatrixXd& x) const
    {
        int nx = mu.size();
        int N = x.rows();
        
        if (P.rows() != nx || P.cols() != nx || x.cols() != nx) {
            // This should be caught by the calling function, but is a last resort
            Eigen::VectorXd p(N);
            p.setZero();
            return p;
        }
        
        Eigen::VectorXd p(N);
        double detP = P.determinant();

        if (detP <= 0 || !P.isApprox(P.transpose())) {
            // Set probability to 0 if matrix is singular or non-symmetric
            p.setZero();
            return p;
        }

        // Konstant term: 1 / sqrt((2*pi)^nx * det(P))
        double constant = 1.0 / (std::pow(2.0 * M_PI, (double)nx / 2.0) * std::sqrt(detP));
        
        // Uses P.llt().solve() for numerically stable solution of P_inv * epsi
        Eigen::LLT<Eigen::MatrixXd> llt(P);
        if(llt.info() != Eigen::Success) {
            // Fallback om Cholesky-faktorisering misslyckas (icke-positivt definit)
            p.setZero();
            return p;
        }

        for (int i = 0; i < N; ++i) {
            Eigen::VectorXd epsi = x.row(i).transpose() - mu; 
            
            // Solution of temp = P_inv * epsi (numerically stable)
            Eigen::VectorXd temp = llt.solve(epsi);
            
            // Kvadratisk form: V = epsi^T * P_inv * epsi
            double V = epsi.dot(temp); 

            // PDF: p = const * exp(-0.5 * V)
            p[i] = constant * std::exp(-0.5 * V);
        }

        return p;
    }
};