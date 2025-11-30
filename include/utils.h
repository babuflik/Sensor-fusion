#ifndef UTILS_H
#define UTILS_H

#include <vector>
#include <complex>
#include <Eigen/Dense>

// Constants
const double PI = 3.14159265358979323846;
// Assumed structure for CRLB data
struct CRLBData {
    // Top-left 2x2 of the full covariance matrix Px (for x, y)
    Eigen::Matrix2d Px; 
};

// Utility functions
std::vector<std::complex<double>> performDFT(const std::vector<double>& input);
void logMessage(const std::string& message);
double calculateMagnitude(const std::complex<double>& complexNumber);
std::vector<double> calculatePhase(const std::vector<std::complex<double>>& frequencyData);

void save_data_to_plots_flexible(
    const Eigen::Vector2d& x_est,
    const Eigen::Vector2d& x_true,
    const Eigen::VectorXd& tdoa_est_vec, 
    const Eigen::VectorXd& tdoa_true_vec, 
    double c,
    const Eigen::MatrixXd& mic_pos, 
    const CRLBData& crlb_data,
    bool save_est_result,
    bool enable_save_hyperbolas,
    bool enable_save_crlb_ellipse
);

#endif // UTILS_H