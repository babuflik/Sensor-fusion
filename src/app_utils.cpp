#include "utils.h"
#include <iostream>
#include <vector>
#include <cmath>
#include <fstream>
#include <Eigen/Dense>
#include <filesystem>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// Function to log messages to the console
void logMessage(const std::string& message) {
    std::cout << "[LOG]: " << message << std::endl;
}

// Function to calculate the distance between two points
double calculateDistance(double x1, double y1, double x2, double y2) {
    return std::sqrt(std::pow(x2 - x1, 2) + std::pow(y2 - y1, 2));
}

// Function to normalize a vector
std::vector<double> normalize(const std::vector<double>& vec) {
    double magnitude = 0.0;
    for (double val : vec) {
        magnitude += val * val;
    }
    magnitude = std::sqrt(magnitude);
    
    std::vector<double> normalizedVec;
    for (double val : vec) {
        normalizedVec.push_back(val / magnitude);
    }
    return normalizedVec;
}

// Function to convert frequency to wavelength
double frequencyToWavelength(double frequency) {
    const double speedOfSound = 343.0; // Speed of sound in air in m/s
    return speedOfSound / frequency;
}


// ======================================================================
// 1. Internal: Save basic estimation results
// ======================================================================
void save_estimation_result(
    const Eigen::Vector2d& x_est,
    const Eigen::Vector2d& x_true,
    const Eigen::VectorXd& tdoa_est_vec,
    const Eigen::VectorXd& tdoa_true_vec
) {
    size_t N_mics = tdoa_true_vec.size();
    std::filesystem::create_directories("artifacts");
    std::ofstream fout("artifacts/estimation_result.txt");
    if (fout.is_open()) {
        fout << "Estimated position: " << x_est.transpose() << "\n";
        fout << "True position:      " << x_true.transpose() << "\n";
        
           if (N_mics > 0 && tdoa_est_vec.size() == N_mics) {
             Eigen::VectorXd residual = tdoa_est_vec - tdoa_true_vec;
             fout << "Residual TDOA (vec):\n" << residual.transpose() << "\n";
        } else {
               fout << "Residual TDOA (vec): data missing or does not match number of microphones.\n";
        }
        fout.close();
    } else {
           std::cerr << "Could not open artifacts/estimation_result.txt\n";
    }
}

// ======================================================================
// 2. Internal: Save hyperbolas for all microphone pairs
// ======================================================================
void save_hyperbolas(
    const Eigen::VectorXd& tdoa_true_vec, // Assumed to be TOA values
    double c,
    const Eigen::MatrixXd& mic_pos
) {
    size_t N_mics = mic_pos.rows();
    if (N_mics < 2 || tdoa_true_vec.size() != N_mics) return;

    std::filesystem::create_directories("artifacts");
    std::ofstream fout("artifacts/hyperbolas.csv");
    if (fout.is_open()) {
        fout << "pair,x,y\n";

        for (size_t i = 0; i < N_mics; ++i) {
            for (size_t j = i + 1; j < N_mics; ++j) {
                
                // Compute the true distance difference d = d_j - d_i
                double dt = tdoa_true_vec(j) - tdoa_true_vec(i); 
                double d = dt * c;    

                // Microphone positions
                double x1 = mic_pos(i, 0);
                double y1 = mic_pos(i, 1);
                double x2 = mic_pos(j, 0);
                double y2 = mic_pos(j, 1);
                
                // Grid sampling
                for (double x = -2.0; x <= 2.0; x += 0.01) {
                    for (double y = -2.0; y <= 2.0; y += 0.01) {
                        double d1 = std::hypot(x - x1, y - y1);
                        double d2 = std::hypot(x - x2, y - y2);

                        // Check hyperbola condition: |d_2 - d_1| = d (within tolerance)
                        if (std::abs((d2 - d1) - d) < 0.01) { 
                            fout << i << "_" << j << "," << x << "," << y << "\n";
                        }
                    }
                }
            }
        }
        fout.close();
    } else {
        std::cerr << "Could not open artifacts/hyperbolas.csv\n";
    }
}

// ======================================================================
// 3. Internal: Save CRLB ellipse
// ======================================================================
void save_crlb_ellipse(
    const Eigen::Vector2d& x_est,
    const CRLBData& crlb_data
) {
    std::filesystem::create_directories("artifacts");
    std::ofstream fout("artifacts/crlb_ellipse.csv");
    if (fout.is_open()) {
        fout << "x,y\n";

        // CRLBData.Px is still a single matrix, not a vector
        Eigen::Matrix2d P = crlb_data.Px;
        Eigen::SelfAdjointEigenSolver<Eigen::Matrix2d> es(P);

        Eigen::Vector2d evals = es.eigenvalues();
        Eigen::Matrix2d evecs = es.eigenvectors();
        
        // Set axes to standard deviation (1-sigma)
        double a = std::sqrt(std::abs(evals(0))); 
        double b = std::sqrt(std::abs(evals(1))); 

        for (double th = 0; th < 2 * M_PI; th += 0.01) {
            Eigen::Vector2d p(a * std::cos(th), b * std::sin(th));
            Eigen::Vector2d q = evecs * p + x_est.head<2>(); // Rotate and center
            fout << q(0) << "," << q(1) << "\n";
        }

    } else {
        std::cerr << "Could not open artifacts/crlb_ellipse.csv\n";
    }
}


// ======================================================================
// Huvudfunktions-"Wrapper" med flexibla flaggor
// ======================================================================

/**
 * @brief Save localization data to selected files for plotting.
 * @param x_est The estimated position (2x1).
 * @param x_true The true position (2x1).
 * @param tdoa_est_vec Vector with estimated TOA/TDOA for each mic.
 * @param tdoa_true_vec Vector with true TOA/TDOA for each mic.
 * @param c Speed of sound.
 * @param mic_pos Matrix with microphone positions (N_mics x 2).
 * @param crlb_data CRLB data (covariance matrix Px).
 * @param save_est_result Save 'estimation_result.txt'.
 * @param save_hyperbolas Save 'hyperbolas.csv'.
 * @param save_crlb_ellipse Save 'crlb_ellipse.csv'.
 */
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
)
{
    if (save_est_result) {
        save_estimation_result(x_est, x_true, tdoa_est_vec, tdoa_true_vec);
    }

    if (enable_save_hyperbolas) {
        save_hyperbolas(tdoa_true_vec, c, mic_pos);
    }

    if (enable_save_crlb_ellipse) {
        save_crlb_ellipse(x_est, crlb_data);
    }
}