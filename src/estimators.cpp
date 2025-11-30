#include "sensor_fusion/estimators.h"
#include <iostream>
#include <cmath>
#include <limits>

// ------------------------
// Least Squares
// ------------------------
std::tuple<Sig, SensorMod> ls(const SensorMod& s, const Sig& y) {
    if (s.pe.size() == 0) throw std::runtime_error("pe must be defined");

    int N = y.t.size();
    int nx = s.nn[0];
    int ny = s.nn[2];

    Eigen::MatrixXd x_est(nx, N);
    Eigen::MatrixXd y_est(ny, N);
    std::vector<Eigen::MatrixXd> Px_list, Py_list;

    for (int k=0; k<N; ++k) {
        Eigen::MatrixXd H(nx, nx); // numgrad approximation
        double eps = 1e-6;
        Eigen::VectorXd x0 = s.x0;
        Eigen::VectorXd f0 = s.h(y.t[k], x0, y.u.col(k), s.th);
        H.resize(ny, nx);
        for (int i=0; i<nx; ++i) {
            Eigen::VectorXd x1 = x0;
            x1[i] += eps;
            H.col(i) = (s.h(y.t[k], x1, y.u.col(k), s.th) - f0)/eps;
        }
        Eigen::MatrixXd PP = (H.transpose()*H).completeOrthogonalDecomposition().pseudoInverse();
        Eigen::VectorXd ff = H.transpose() * (y.y.col(k) - f0 + H*x0);
        x_est.col(k) = PP*ff;
        y_est.col(k) = f0;
        Px_list.push_back(PP*(H.transpose()*s.pe*H)*PP);
        Py_list.push_back(H*Px_list.back()*H.transpose());
    }

    Sig xhat;
    xhat.x = x_est;
    xhat.y = y_est;
    xhat.t = y.t;
    xhat.u = y.u;
    xhat.xMC = Px_list; // temporary storage of covariance

    SensorMod shat = s;
    if (N==1) {
        shat.x0 = x_est.col(0);
    }
    return {xhat, shat};
}

// ------------------------
// Weighted Least Squares
// ------------------------
std::tuple<Sig, SensorMod> wls(const SensorMod& s, const Sig& y) {
    if (s.pe.size() == 0) throw std::runtime_error("pe must be defined");

    int N = y.t.size();
    int nx = s.nn[0];
    int ny = s.nn[2];

    Eigen::MatrixXd x_est(nx, N);
    Eigen::MatrixXd y_est(ny, N);

    Eigen::MatrixXd R = s.pe;
    Eigen::MatrixXd Rinv = R.completeOrthogonalDecomposition().pseudoInverse();

    for (int k=0; k<N; ++k) {
        Eigen::MatrixXd H(nx, nx); 
        double eps = 1e-6;
        Eigen::VectorXd x0 = s.x0;
        Eigen::VectorXd f0 = s.h(y.t[k], x0, y.u.col(k), s.th);
        H.resize(ny, nx);
        for (int i=0; i<nx; ++i) {
            Eigen::VectorXd x1 = x0;
            x1[i] += eps;
            H.col(i) = (s.h(y.t[k], x1, y.u.col(k), s.th) - f0)/eps;
        }
        Eigen::MatrixXd PP = (H.transpose()*Rinv*H).completeOrthogonalDecomposition().pseudoInverse();
        Eigen::VectorXd ff = H.transpose()*Rinv*(y.y.col(k)-f0 + H*x0);
        x_est.col(k) = PP*ff;
        y_est.col(k) = f0;
    }

    Sig xhat;
    xhat.x = x_est;
    xhat.y = y_est;
    xhat.t = y.t;
    xhat.u = y.u;

    SensorMod shat = s;
    if (N==1) shat.x0 = x_est.col(0);
    return {xhat, shat};
}

// ------------------------
// LH1 (1D likelihood)
// ------------------------
std::tuple<Eigen::VectorXd, Eigen::VectorXd, Eigen::VectorXd> lh1(
    const SensorMod& s, const Sig& y, 
    const Eigen::VectorXd* x1, int ind)
{
    int nx = s.nn[0];
    if (ind >= nx) throw std::runtime_error("ind exceeds state dimension");
    Eigen::VectorXd grid = x1 ? *x1 : Eigen::VectorXd::LinSpaced(100, 0.0, 1.0);
    int N = grid.size();
    Eigen::VectorXd lh(N), px0 = Eigen::VectorXd::Ones(N), px(N);
    px = px0;

    for (int k=0; k<y.t.size(); ++k) {
        for (int n=0; n<N; ++n) {
            Eigen::VectorXd x = s.x0;
            x[ind] = grid[n];
            Eigen::VectorXd yhat = s.h(y.t[k], x, y.u.col(k), s.th);
            Eigen::VectorXd epsi = y.y.col(k) - yhat;
            double val = epsi.transpose()*s.pe.completeOrthogonalDecomposition().pseudoInverse()*epsi;
            lh[n] = std::exp(-0.5*val);
        }
        px = px.cwiseProduct(lh);
    }

    return {lh, grid, px};
}

// ------------------------
// LH2 (2D likelihood)
// ------------------------
std::tuple<Eigen::MatrixXd, Eigen::VectorXd, Eigen::VectorXd, Eigen::MatrixXd, Eigen::MatrixXd, Eigen::MatrixXd, Eigen::MatrixXd>
lh2(const SensorMod& s, const Sig& y,
    const Eigen::VectorXd* x1_ptr,
    const Eigen::VectorXd* x2_ptr,
    const std::vector<int>& ind)
{
    if (ind.size() != 2) throw std::runtime_error("ind must have size 2");
    int nx = s.nn[0];
    if (ind[0]>=nx || ind[1]>=nx) throw std::runtime_error("ind exceeds state dimension");

    Eigen::VectorXd x1 = x1_ptr ? *x1_ptr : Eigen::VectorXd::LinSpaced(30, s.x0[ind[0]]-0.5, s.x0[ind[0]]+0.5);
    Eigen::VectorXd x2 = x2_ptr ? *x2_ptr : Eigen::VectorXd::LinSpaced(30, s.x0[ind[1]]-0.5, s.x0[ind[1]]+0.5);

    Eigen::MatrixXd X1, X2;
    Eigen::MatrixXd lh(x1.size(), x2.size());
    Eigen::MatrixXd px(x1.size(), x2.size()), px0(px);

    for (int i=0; i<x1.size(); ++i) {
        for (int j=0; j<x2.size(); ++j) {
            Eigen::VectorXd xx = s.x0;
            xx[ind[0]] = x1[i];
            xx[ind[1]] = x2[j];
            double val = 0.0;
            for (int k=0; k<y.t.size(); ++k) {
                Eigen::VectorXd yhat = s.h(y.t[k], xx, y.u.col(k), s.th);
                Eigen::VectorXd epsi = y.y.col(k) - yhat;
                val += epsi.transpose()*s.pe.completeOrthogonalDecomposition().pseudoInverse()*epsi;
            }
            lh(i,j) = std::exp(-0.5*val);
            px(i,j) = lh(i,j); 
            px0(i,j) = 1.0;
        }
    }

    // Return meshgrid for plotting
    X1 = x1.replicate(1,x2.size());
    X2 = x2.transpose().replicate(x1.size(),1);

    return {lh, x1, x2, px, px0, X1, X2};
}

// Numerical jacobian of h(t, x, u, th) wrt x
// h: std::function<Eigen::VectorXd(double,const Eigen::VectorXd&,const Eigen::VectorXd&,const Eigen::VectorXd&)>
// returns matrix ny x nx
static Eigen::MatrixXd numerical_jacobian(
    const SensorMod::DynFunc& h,
    double t,
    const Eigen::VectorXd& x,
    const Eigen::VectorXd& u,
    const Eigen::VectorXd& th)
{
    const double eps = 1e-6;
    int nx = x.size();
    Eigen::VectorXd h0 = h(t, x, u, th);
    int ny = h0.size();
    Eigen::MatrixXd J(ny, nx);
    for (int i = 0; i < nx; ++i) {
        Eigen::VectorXd xp = x;
        Eigen::VectorXd xm = x;
        double delta = eps * std::max(1.0, std::abs(x[i]));
        xp[i] += delta;
        xm[i] -= delta;
        Eigen::VectorXd hp = h(t, xp, u, th);
        Eigen::VectorXd hm = h(t, xm, u, th);
        J.col(i) = (hp - hm) / (2.0 * delta);
    }
    return J;
}

// Safe inverse: attempts direct inverse, otherwise regularizes
static Eigen::MatrixXd safe_inverse(const Eigen::MatrixXd& M)
{
    const double eps = 1e-8;
    if (M.rows() != M.cols()) throw std::runtime_error("safe_inverse: M not square");
    Eigen::FullPivLU<Eigen::MatrixXd> lu(M);
    if (lu.isInvertible()) {
        return M.inverse();
    } else {
        // Regularize with small diagonal
        double lambda = eps * M.trace(); // scale regularization by trace
        Eigen::MatrixXd Mreg = M;
        if (lambda <= 0) lambda = eps;
        Mreg += lambda * Eigen::MatrixXd::Identity(M.rows(), M.cols());
        Eigen::FullPivLU<Eigen::MatrixXd> lu2(Mreg);
        if (lu2.isInvertible()) return Mreg.inverse();
        // As last resort, pseudo-inverse via SVD
        Eigen::JacobiSVD<Eigen::MatrixXd> svd(M, Eigen::ComputeThinU | Eigen::ComputeThinV);
        double tol = std::max(M.rows(), M.cols()) * svd.singularValues().array().abs().maxCoeff() * std::numeric_limits<double>::epsilon();
        Eigen::VectorXd invS = svd.singularValues();
        for (int i = 0; i < invS.size(); ++i) {
            if (std::abs(invS(i)) > tol) invS(i) = 1.0 / invS(i);
            else invS(i) = 0.0;
        }
        return svd.matrixV() * invS.asDiagonal() * svd.matrixU().transpose();
    }
}

// ------------------------
// Maximum Likelihood (ML) Estimator
// ------------------------
std::tuple<Sig, SensorMod, Eigen::MatrixXd> ml(const SensorMod& s, const Sig& y) {
    if (s.pe.size() == 0) throw std::runtime_error("pe must be defined");

    int nx = s.nn[0];
    int ny = s.nn[2];
    
    // Create a non-const copy for likelihood computation
    SensorMod s_copy = s;
    
    // Create a grid around the initial guess
    // For 3D: create a reasonable search grid
    double search_range = 2.0; // meters
    int grid_points_per_dim = 21;
    
    // Build grid based on dimensionality
    Eigen::MatrixXd grid;
    
    if (nx == 2) {
        // 2D grid
        int N_grid = grid_points_per_dim * grid_points_per_dim;
        grid.resize(N_grid, nx);
        int idx = 0;
        for (int i = 0; i < grid_points_per_dim; ++i) {
            for (int j = 0; j < grid_points_per_dim; ++j) {
                double x = s.x0[0] - search_range + 2.0 * search_range * i / (grid_points_per_dim - 1);
                double y_val = s.x0[1] - search_range + 2.0 * search_range * j / (grid_points_per_dim - 1);
                grid(idx, 0) = x;
                grid(idx, 1) = y_val;
                idx++;
            }
        }
    } else if (nx == 3) {
        // 3D grid - use coarser grid to keep computation reasonable
        int gp = 15; // coarser for 3D
        int N_grid = gp * gp * gp;
        grid.resize(N_grid, nx);
        int idx = 0;
        for (int i = 0; i < gp; ++i) {
            for (int j = 0; j < gp; ++j) {
                for (int k = 0; k < gp; ++k) {
                    double x = s.x0[0] - search_range + 2.0 * search_range * i / (gp - 1);
                    double y_val = s.x0[1] - search_range + 2.0 * search_range * j / (gp - 1);
                    double z = s.x0[2] - search_range + 2.0 * search_range * k / (gp - 1);
                    grid(idx, 0) = x;
                    grid(idx, 1) = y_val;
                    grid(idx, 2) = z;
                    idx++;
                }
            }
        }
    } else {
        throw std::runtime_error("ml: only supports nx=2 or nx=3");
    }
    
    // Compute likelihood over grid
    Eigen::VectorXd L = s_copy.likelihood_function(y, grid);
    
    // Find maximum likelihood
    int best_idx;
    L.maxCoeff(&best_idx);
    
    Eigen::VectorXd x_ml = grid.row(best_idx).transpose();
    
    // Prepare output
    Sig xhat;
    xhat.x.resize(nx, 1);
    xhat.x.col(0) = x_ml;
    
    // Compute predicted measurement at ML estimate
    int nu = s.nn[1];
    Eigen::VectorXd u_k;
    if (y.u.size() > 0 && y.u.cols() > 0) {
        u_k = y.u.col(0);
    } else {
        u_k = Eigen::VectorXd::Zero(nu);
    }
    double t_k = y.t.size() > 0 ? y.t[0] : 0.0;
    xhat.y.resize(ny, 1);
    xhat.y.col(0) = s.h(t_k, x_ml, u_k, s.th);
    
    xhat.t = y.t;
    xhat.u = y.u;
    
    SensorMod shat = s;
    shat.x0 = x_ml;
    
    // Return grid as likelihood matrix (reshape for visualization if needed)
    Eigen::MatrixXd L_matrix = L; // Keep as vector for now
    
    return {xhat, shat, L_matrix};
}

// Main CRLB implementation
Sig crlb(const SensorMod& s, const Sig* y)
{
    // Determine nx, ny, N_time
    int nx = s.nn[0];
    int ny = s.nn[2];

    // Choose state vector x0: if y provided and y->x has at least one column, use it; else s.x0
    Eigen::VectorXd x0;
    if (y != nullptr && y->x.size() > 0 && y->x.cols() > 0) {
        // assuming y->x is nx x N matrix
        x0 = y->x.col(0);
    } else {
        x0 = s.x0;
    }

    // Choose number of time samples and iterate
    int N_time = 1;
    if (y != nullptr && y->t.size() > 0) N_time = y->t.size();

    // Precompute Pe inverse
    if (s.pe.rows() != ny || s.pe.cols() != ny) {
        throw std::runtime_error("crlb: Pe dimensions mismatch");
    }
    Eigen::MatrixXd Pe = s.pe;
    // Try to invert Pe
    Eigen::MatrixXd Pe_inv = safe_inverse(Pe);

    // Build Fisher information J
    Eigen::MatrixXd J = Eigen::MatrixXd::Zero(nx, nx);

    for (int k = 0; k < N_time; ++k) {
        double t_k = (y != nullptr && y->t.size() > k) ? y->t[k] : (k * (1.0 / s.fs));
        // determine u_k: if y->u exists and has columns
        Eigen::VectorXd u_k;
        if (y != nullptr && y->u.size() > 0 && y->u.cols() > k) {
            u_k = y->u.col(k);
        } else {
            // if nu==0, u_k should be zero-length vector
            int nu = s.nn[1];
            u_k = Eigen::VectorXd::Zero(nu);
        }

        // compute Jacobian H (ny x nx)
        Eigen::MatrixXd H = numerical_jacobian(s.h, t_k, x0, u_k, s.th); // note: using x0 for all k like MATLAB crlb(s,y)
        // accumulate J
        J.noalias() += H.transpose() * Pe_inv * H;
    }

    // Invert J to get CRLB
    Eigen::MatrixXd Px;
    // If J is nearly zero or singular, safe_inverse will regularize
    Px = safe_inverse(J);

    // Prepare output Sig object
    Sig out;
    // ASSUMPTION: Sig has members .x (nx x ?), .Px (vector of nx x nx). Adjust if names differ.
    // set out.x to column vector x0
    out.x = Eigen::MatrixXd::Zero(nx, 1);
    out.x.col(0) = x0;
    out.Px.push_back(Px);  // Store single covariance matrix

    return out;
}

// crlb2_grid: compute scalar CRLB measure over 2D grid for indices ind
Eigen::VectorXd crlb2_grid(const SensorMod& s,
                           const Sig* y,
                           const Eigen::VectorXd& x1,
                           const Eigen::VectorXd& x2,
                           const std::array<int,2>& ind,
                           const std::string& type)
{
    // ind holds 0-based indices of states to vary
    if (ind[0] < 0 || ind[1] < 0) throw std::runtime_error("crlb2_grid: invalid indices");

    int N1 = x1.size();
    int N2 = x2.size();
    int N = N1 * N2;

    Eigen::VectorXd c(N);
    // Build a copy of y or a temporary Sig to pass to crlb
    Sig ylocal;
    if (y != nullptr) ylocal = *y;

    for (int i = 0; i < N1; ++i) {
        for (int j = 0; j < N2; ++j) {
            // create a state vector xgrid = s.x0 with selected indices replaced
            Eigen::VectorXd xgrid = s.x0;
            xgrid[ind[0]] = x1[i];
            xgrid[ind[1]] = x2[j];

            // plug into ylocal.x (single-sample)
            ylocal.x = Eigen::MatrixXd::Zero(s.nn[0], 1);
            ylocal.x.col(0) = xgrid;

            // call crlb
            Sig cr = crlb(s, &ylocal);
            Eigen::MatrixXd Px = cr.Px.empty() ? Eigen::MatrixXd::Zero(s.nn[0], s.nn[0]) : cr.Px[0];

            double scalar = 0.0;
            if (type == "trace") {
                scalar = Px.trace();
            } else if (type == "rmse") {
                scalar = std::sqrt(Px.trace());
            } else if (type == "det") {
                scalar = Px.determinant();
            } else if (type == "max") {
                Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es(Px);
                scalar = es.eigenvalues().maxCoeff();
            } else {
                throw std::runtime_error("crlb2_grid: unknown type");
            }
            c[j * N1 + i] = scalar; // column-major like meshgrid
        }
    }
    return c;
}
