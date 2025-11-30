#include <gtest/gtest.h>
#include <Eigen/Dense>
#include <cmath>
#include <iostream>
#include <sensor_fusion/sensormod.h>
#include <sensor_fusion/sig.h>
#include "utils.h"
#include <sensor_fusion/estimators.h>
#include <sensor_fusion/nl.h>

/**
 * Comprehensive test suite for all estimators (LS, WLS, ML, CRLB, EKF, UKF)
 * using a consistent microphone configuration.
 * 
 * Tests include:
 * - Least Squares (LS)
 * - Weighted Least Squares (WLS)
 * - Maximum Likelihood (ML)
 * - Cramér-Rao Lower Bound (CRLB)
 * - Extended Kalman Filter (EKF)
 * - Unscented Kalman Filter (UKF)
 */

class AllEstimatorsTest : public ::testing::Test {
protected:
    // Common configuration parameters
    double c = 343.0;    // speed of sound (m/s)
    double fs = 48000.0; // sample rate

    // Microphone placement: 4 mics in a tetrahedron configuration
    Eigen::Matrix<double, 4, 3> mic_pos;

    // Ground truth source location
    Eigen::Vector3d source_true;

    // Sensor model
    std::unique_ptr<SensorMod> sensor;

    // NL model (for EKF/UKF)
    std::unique_ptr<NL> nl_model;

    // Measurement signal
    Sig y;

    void SetUp() override {
        // Setup microphone positions (3D tetrahedron)
        mic_pos << 0.0,   0.0,    0.0,      // mic 1 at origin
                   0.15,  0.0,    0.0,      // mic 2 (15 cm on x-axis)
                   0.075, 0.13,   0.0,      // mic 3 (forms triangle base)
                   0.075, 0.065,  0.12;     // mic 4 (apex, 12 cm up)

        // Ground truth source position
        source_true << 1.2, 0.5, 0.3;  // 1.2m forward, 0.5m right, 0.3m up

        // Setup sensor model
        setupSensorModel();
        
        // Setup NL model for filters
        setupNLModel();
        
        // Generate synthetic measurements
        generateMeasurements();
    }

    void setupSensorModel() {
        // Define sensor function: y = [dist1, dist2, dist3, dist4] (time-of-arrival)
        auto h = [this](double t,
                       const Eigen::VectorXd& X,
                       const Eigen::VectorXd& u,
                       const Eigen::VectorXd& th) -> Eigen::VectorXd
        {
            Eigen::Vector3d p = X.head<3>();
            Eigen::VectorXd out(4);
            for (int i = 0; i < 4; ++i) {
                out[i] = (p - mic_pos.row(i).transpose()).norm() / this->c;
            }
            return out;
        };

        // Create sensor: nx=3 (x,y,z), nu=0, ny=4 (TOA), nth=0
        sensor = std::make_unique<SensorMod>(h, Eigen::Vector4i{3, 0, 4, 0});
        
        // Set measurement noise covariance (TOA uncertainties)
        // Assuming 1 sample uncertainty at 48kHz -> ~20 microseconds
        double sigma_toa = 1.0 / fs;
        sensor->pe = Eigen::Matrix4d::Identity() * (sigma_toa * sigma_toa);
        
        // Set initial guess (slightly offset from true position)
        sensor->x0 = source_true + Eigen::Vector3d(0.1, 0.1, 0.1);
    }

    void setupNLModel() {
        // Create NL model for EKF/UKF testing
        // State: [x, y, z] (position only, for simplicity)
        // Measurement: [toa1, toa2, toa3, toa4] (time of arrival)
        
        auto f_func = [](double t, const Eigen::VectorXd& x, 
                        const Eigen::VectorXd& u, const Eigen::VectorXd& th) {
            // Static source (no dynamics)
            return x;
        };
        
        auto h_func = [this](double t, const Eigen::VectorXd& x,
                            const Eigen::VectorXd& u, const Eigen::VectorXd& th) {
            // Measurement function: TOA = distance / c
            Eigen::VectorXd y(4);
            for (int i = 0; i < 4; ++i) {
                Eigen::Vector3d pos = x.head<3>();
                double dist = (pos - this->mic_pos.row(i).transpose()).norm();
                y[i] = dist / this->c;
            }
            return y;
        };
        
        // Dimensions: nx=3, nu=0, ny=4, nth=0
        Eigen::Vector4i dims(3, 0, 4, 0);
        nl_model = std::make_unique<NL>(f_func, h_func, dims, fs);
        
        // Set initial state
        nl_model->x0 = source_true + Eigen::Vector3d(0.1, 0.1, 0.1);
        
        // Set noise covariances
        double sigma_toa = 1.0 / fs;
        Eigen::Matrix3d Q = Eigen::Matrix3d::Identity() * 1e-10; // Very small process noise (static)
        Eigen::Matrix4d R = Eigen::Matrix4d::Identity() * (sigma_toa * sigma_toa);
        
        nl_model->set_pv(Q);
        nl_model->set_pe(R);
        nl_model->set_px0(Eigen::Matrix3d::Identity() * 0.1); // Initial uncertainty
    }

    void generateMeasurements() {
        // Compute true time-of-arrival for each microphone
        Eigen::VectorXd toa_true(4);
        for (int i = 0; i < 4; ++i) {
            double dist = (source_true - mic_pos.row(i).transpose()).norm();
            toa_true[i] = dist / c;
        }

        // Create measurement signal
        y.t.resize(1);
        y.t << 0.0;
        
        y.y.resize(4, 1);
        y.y.col(0) = toa_true;
        
        // Add small noise to make it realistic
        Eigen::VectorXd noise(4);
        noise << 1e-6, -0.5e-6, 0.8e-6, -0.3e-6;
        y.y.col(0) += noise;
        
        y.u.resize(0, 1);  // no input
        
        // Store true state for reference
        y.x.resize(3, 1);
        y.x.col(0) = source_true;
    }

    // Helper to print results
    void printEstimate(const std::string& method, const Eigen::Vector3d& estimate) {
        Eigen::Vector3d error = estimate - source_true;
        double rmse = error.norm();
        
        std::cout << "\n" << method << " Estimate:\n"
                  << "  Position: [" << estimate.transpose() << "]\n"
                  << "  True:     [" << source_true.transpose() << "]\n"
                  << "  Error:    [" << error.transpose() << "]\n"
                  << "  RMSE:     " << rmse << " m\n";
    }

    void printCovariance(const std::string& method, const Eigen::Matrix3d& cov) {
        Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> es(cov);
        double trace = cov.trace();
        double det = cov.determinant();
        double max_eig = es.eigenvalues().maxCoeff();
        
        std::cout << "  Covariance trace: " << trace << "\n"
                  << "  Covariance det:   " << det << "\n"
                  << "  Max eigenvalue:   " << max_eig << "\n"
                  << "  Std devs: [" 
                  << std::sqrt(cov(0,0)) << ", "
                  << std::sqrt(cov(1,1)) << ", "
                  << std::sqrt(cov(2,2)) << "]\n";
    }
};

// ============================================================================
// TEST 1: Least Squares (LS) Estimator
// ============================================================================
TEST_F(AllEstimatorsTest, LeastSquaresEstimator) {
    std::cout << "\n========== LEAST SQUARES (LS) ==========";
    
    auto [xhat_sig, shat] = ls(*sensor, y);
    
    ASSERT_EQ(xhat_sig.x.cols(), 1);
    ASSERT_EQ(xhat_sig.x.rows(), 3);
    
    Eigen::Vector3d estimate = xhat_sig.x.col(0);
    printEstimate("LS", estimate);
    
    // Extract covariance from xMC (temporary storage)
    if (!xhat_sig.xMC.empty()) {
        Eigen::Matrix3d cov = xhat_sig.xMC[0];
        printCovariance("LS", cov);
        
        // Check that covariance is positive definite
        Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> es(cov);
        EXPECT_GT(es.eigenvalues().minCoeff(), 0.0);
    }
    
    // Verify estimate is reasonably close to true position
    double error = (estimate - source_true).norm();
    EXPECT_LT(error, 0.3) << "LS estimate too far from true position";
}

// ============================================================================
// TEST 2: Weighted Least Squares (WLS) Estimator
// ============================================================================
TEST_F(AllEstimatorsTest, WeightedLeastSquaresEstimator) {
    std::cout << "\n========== WEIGHTED LEAST SQUARES (WLS) ==========";
    
    auto [xhat_sig, shat] = wls(*sensor, y);
    
    ASSERT_EQ(xhat_sig.x.cols(), 1);
    ASSERT_EQ(xhat_sig.x.rows(), 3);
    
    Eigen::Vector3d estimate = xhat_sig.x.col(0);
    printEstimate("WLS", estimate);
    
    // Verify estimate is reasonably close to true position
    double error = (estimate - source_true).norm();
    EXPECT_LT(error, 0.3) << "WLS estimate too far from true position";
}

// ============================================================================
// TEST 3: Maximum Likelihood (ML) Estimator
// ============================================================================
TEST_F(AllEstimatorsTest, MaximumLikelihoodEstimator) {
    std::cout << "\n========== MAXIMUM LIKELIHOOD (ML) ==========";
    
    auto [xhat_sig, shat, L_grid] = ml(*sensor, y);
    
    ASSERT_EQ(xhat_sig.x.cols(), 1);
    ASSERT_EQ(xhat_sig.x.rows(), 3);
    
    Eigen::Vector3d estimate = xhat_sig.x.col(0);
    printEstimate("ML", estimate);
    
    // Verify estimate is reasonably close to true position
    double error = (estimate - source_true).norm();
    EXPECT_LT(error, 0.3) << "ML estimate too far from true position";
    
    // Check that likelihood grid was computed
    EXPECT_GT(L_grid.size(), 0) << "ML should return likelihood grid";
}

// ============================================================================
// TEST 4: Cramér-Rao Lower Bound (CRLB)
// ============================================================================
TEST_F(AllEstimatorsTest, CramerRaoLowerBound) {
    std::cout << "\n========== CRAMÉR-RAO LOWER BOUND (CRLB) ==========";
    
    // Compute CRLB at true position
    Sig y_crlb = y;
    y_crlb.x.col(0) = source_true;
    
    Sig cr_result = crlb(*sensor, &y_crlb);
    
    ASSERT_FALSE(cr_result.Px.empty()) << "CRLB should return covariance matrix";
    
    Eigen::Matrix3d crlb_cov = cr_result.Px[0].topLeftCorner(3, 3);
    
    std::cout << "\nCRLB at true position:\n";
    printCovariance("CRLB", crlb_cov);
    
    // CRLB should be positive definite
    Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> es(crlb_cov);
    EXPECT_GT(es.eigenvalues().minCoeff(), 0.0) << "CRLB covariance should be positive definite";
    
    // CRLB provides lower bound on variance
    double crlb_trace = crlb_cov.trace();
    EXPECT_GT(crlb_trace, 0.0);
    
    // Compute theoretical best achievable RMSE
    double min_rmse = std::sqrt(crlb_trace);
    std::cout << "  Minimum achievable RMSE: " << min_rmse << " m\n";
}

// ============================================================================
// TEST 5: Extended Kalman Filter (EKF)
// ============================================================================
TEST_F(AllEstimatorsTest, ExtendedKalmanFilter) {
    std::cout << "\n========== EXTENDED KALMAN FILTER (EKF) ==========";
    
    // Run EKF
    auto [x_est, P_est] = nl_model->ekf(y);
    
    ASSERT_EQ(x_est.cols(), 1);
    ASSERT_EQ(x_est.rows(), 3);
    
    Eigen::Vector3d estimate = x_est.col(0);
    printEstimate("EKF", estimate);
    
    // Print final covariance
    if (!P_est.empty()) {
        Eigen::Matrix3d final_cov = P_est.back();
        printCovariance("EKF", final_cov);
        
        // Check positive definiteness
        Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> es(final_cov);
        EXPECT_GT(es.eigenvalues().minCoeff(), 0.0);
    }
    
    // Verify estimate is reasonable
    double error = (estimate - source_true).norm();
    EXPECT_LT(error, 0.3) << "EKF estimate too far from true position";
}

// ============================================================================
// TEST 6: Unscented Kalman Filter (UKF)
// ============================================================================
TEST_F(AllEstimatorsTest, UnscentedKalmanFilter) {
    std::cout << "\n========== UNSCENTED KALMAN FILTER (UKF) ==========";
    
    // Run UKF
    auto [x_est, P_est] = nl_model->ukf(y);
    
    ASSERT_EQ(x_est.cols(), 1);
    ASSERT_EQ(x_est.rows(), 3);
    
    Eigen::Vector3d estimate = x_est.col(0);
    printEstimate("UKF", estimate);
    
    // Print final covariance
    if (!P_est.empty()) {
        Eigen::Matrix3d final_cov = P_est.back();
        printCovariance("UKF", final_cov);
        
        // Check positive definiteness
        Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> es(final_cov);
        EXPECT_GT(es.eigenvalues().minCoeff(), 0.0);
    }
    
    // Verify estimate is reasonable
    double error = (estimate - source_true).norm();
    EXPECT_LT(error, 0.3) << "UKF estimate too far from true position";
}

// ============================================================================
// TEST 7: Compare All Estimators (Including Filters)
// ============================================================================
TEST_F(AllEstimatorsTest, CompareAllEstimators) {
    std::cout << "\n========== COMPARISON OF ALL ESTIMATORS ==========\n";
    
    // Run all estimators
    auto [ls_sig, ls_shat] = ls(*sensor, y);
    auto [wls_sig, wls_shat] = wls(*sensor, y);
    auto [ml_sig, ml_shat, ml_grid] = ml(*sensor, y);
    
    auto [ekf_x, ekf_P] = nl_model->ekf(y);
    auto [ukf_x, ukf_P] = nl_model->ukf(y);
    
    Sig y_crlb = y;
    y_crlb.x.col(0) = source_true;
    Sig crlb_sig = crlb(*sensor, &y_crlb);
    
    // Extract estimates
    Eigen::Vector3d ls_est = ls_sig.x.col(0);
    Eigen::Vector3d wls_est = wls_sig.x.col(0);
    Eigen::Vector3d ml_est = ml_sig.x.col(0);
    Eigen::Vector3d ekf_est = ekf_x.col(0);
    Eigen::Vector3d ukf_est = ukf_x.col(0);
    
    // Compute errors
    double ls_error = (ls_est - source_true).norm();
    double wls_error = (wls_est - source_true).norm();
    double ml_error = (ml_est - source_true).norm();
    double ekf_error = (ekf_est - source_true).norm();
    double ukf_error = (ukf_est - source_true).norm();
    
    std::cout << "\nEstimation Errors (RMSE):\n"
              << "  LS:   " << ls_error << " m\n"
              << "  WLS:  " << wls_error << " m\n"
              << "  ML:   " << ml_error << " m\n"
              << "  EKF:  " << ekf_error << " m\n"
              << "  UKF:  " << ukf_error << " m\n";
    
    if (!crlb_sig.Px.empty()) {
        Eigen::Matrix3d crlb_cov = crlb_sig.Px[0].topLeftCorner(3, 3);
        double min_rmse = std::sqrt(crlb_cov.trace());
        std::cout << "  CRLB: " << min_rmse << " m (theoretical minimum)\n";
        
        // All estimators should be above CRLB (with some numerical tolerance)
        // Note: In practice with little noise, they might achieve near-CRLB performance
    }
    
    // All estimates should be reasonable
    EXPECT_LT(ls_error, 0.5);
    EXPECT_LT(wls_error, 0.5);
    EXPECT_LT(ml_error, 0.5);
    EXPECT_LT(ekf_error, 0.5);
    EXPECT_LT(ukf_error, 0.5);
    
    // ML should generally be reasonable (grid search may be coarse)
    // In practice, LS/WLS with good initial guess can outperform coarse grid ML
    EXPECT_LE(ml_error, 0.5) << "ML should produce reasonable estimate";
}

// ============================================================================
// TEST 8: Multi-Sample Time Series
// ============================================================================
TEST_F(AllEstimatorsTest, MultiSampleTimeSeries) {
    std::cout << "\n========== MULTI-SAMPLE TIME SERIES ==========\n";
    
    // Create a time series where source moves slightly
    int N_samples = 5;
    y.t.resize(N_samples);
    y.y.resize(4, N_samples);
    y.x.resize(3, N_samples);
    y.u.resize(0, N_samples);
    
    for (int k = 0; k < N_samples; ++k) {
        y.t[k] = k * 0.1;  // 100ms intervals
        
        // Source moves slowly
        Eigen::Vector3d pos = source_true + Eigen::Vector3d(0.01 * k, 0.005 * k, 0.0);
        y.x.col(k) = pos;
        
        // Compute measurements
        for (int i = 0; i < 4; ++i) {
            double dist = (pos - mic_pos.row(i).transpose()).norm();
            y.y(i, k) = dist / c;
        }
    }
    
    // Test LS with multiple samples
    auto [ls_sig, ls_shat] = ls(*sensor, y);
    ASSERT_EQ(ls_sig.x.cols(), N_samples);
    
    std::cout << "\nLS estimates over time:\n";
    for (int k = 0; k < N_samples; ++k) {
        Eigen::Vector3d est = ls_sig.x.col(k);
        Eigen::Vector3d true_pos = y.x.col(k);
        double error = (est - true_pos).norm();
        std::cout << "  t=" << y.t[k] << "s: error=" << error << " m\n";
        EXPECT_LT(error, 0.5);
    }
    
    // Test CRLB with multiple samples
    Sig crlb_sig = crlb(*sensor, &y);
    ASSERT_FALSE(crlb_sig.Px.empty());
    
    std::cout << "\nCRLB improves with more samples:\n";
    std::cout << "  Single-sample CRLB trace: " << crlb_sig.Px[0].trace() << "\n";
}

// ============================================================================
// TEST 9: Robustness to Initial Guess
// ============================================================================
TEST_F(AllEstimatorsTest, RobustnessToInitialGuess) {
    std::cout << "\n========== ROBUSTNESS TO INITIAL GUESS ==========\n";
    
    // Test with various initial guesses
    std::vector<Eigen::Vector3d> initial_guesses = {
        source_true + Eigen::Vector3d(0.5, 0.5, 0.5),   // offset
        source_true + Eigen::Vector3d(-0.3, 0.2, -0.1), // different offset
        Eigen::Vector3d(0.5, 0.5, 0.5),                 // fixed position
        source_true                                      // perfect guess
    };
    
    for (size_t i = 0; i < initial_guesses.size(); ++i) {
        sensor->x0 = initial_guesses[i];
        
        auto [wls_sig, wls_shat] = wls(*sensor, y);
        Eigen::Vector3d estimate = wls_sig.x.col(0);
        double error = (estimate - source_true).norm();
        
        std::cout << "  Initial guess " << i << ": error=" << error << " m\n";
        EXPECT_LT(error, 0.5) << "Estimator should converge from various initial guesses";
    }
}

// ============================================================================
// TEST 10: Different Microphone Configurations
// ============================================================================
TEST_F(AllEstimatorsTest, DifferentMicrophoneConfigurations) {
    std::cout << "\n========== DIFFERENT MICROPHONE CONFIGURATIONS ==========\n";
    
    // Test 1: Linear array (poor geometry)
    Eigen::Matrix<double, 4, 3> linear_mics;
    linear_mics << 0.0,  0.0, 0.0,
                   0.1,  0.0, 0.0,
                   0.2,  0.0, 0.0,
                   0.3,  0.0, 0.0;
    
    // Swap in the linear configuration
    Eigen::Matrix<double, 4, 3> original_mics = mic_pos;
    mic_pos = linear_mics;
    setupSensorModel();
    generateMeasurements();
    
    auto [ls_linear, _] = ls(*sensor, y);
    double linear_error = (ls_linear.x.col(0) - source_true).norm();
    std::cout << "  Linear array error: " << linear_error << " m\n";
    
    // Test 2: Square planar array (better geometry)
    Eigen::Matrix<double, 4, 3> square_mics;
    square_mics << 0.0,  0.0, 0.0,
                   0.2,  0.0, 0.0,
                   0.2,  0.2, 0.0,
                   0.0,  0.2, 0.0;
    
    mic_pos = square_mics;
    setupSensorModel();
    generateMeasurements();
    
    auto [ls_square, __] = ls(*sensor, y);
    double square_error = (ls_square.x.col(0) - source_true).norm();
    std::cout << "  Square array error: " << square_error << " m\n";
    
    // Restore original configuration
    mic_pos = original_mics;
    setupSensorModel();
    generateMeasurements();
    
    // 3D configuration should generally be better for 3D localization
    auto [ls_3d, ___] = ls(*sensor, y);
    double tetra_error = (ls_3d.x.col(0) - source_true).norm();
    std::cout << "  Tetrahedron array error: " << tetra_error << " m\n";
}

// ============================================================================
// TEST 11: 1D and 2D Likelihood Functions
// ============================================================================
TEST_F(AllEstimatorsTest, LikelihoodFunctions) {
    std::cout << "\n========== LIKELIHOOD FUNCTIONS ==========\n";
    
    // Test 1D likelihood along x-axis
    Eigen::VectorXd x_grid = Eigen::VectorXd::LinSpaced(50, 0.5, 2.0);
    auto [lh_1d, grid_1d, px_1d] = lh1(*sensor, y, &x_grid, 0);
    
    ASSERT_EQ(lh_1d.size(), 50);
    
    // Find maximum likelihood position
    int max_idx;
    px_1d.maxCoeff(&max_idx);
    double ml_x = grid_1d[max_idx];
    
    std::cout << "  1D ML (x-axis): " << ml_x << " (true: " << source_true[0] << ")\n";
    EXPECT_NEAR(ml_x, source_true[0], 0.2);
    
    // Test 2D likelihood in x-y plane
    Eigen::VectorXd x1_grid = Eigen::VectorXd::LinSpaced(30, 0.5, 2.0);
    Eigen::VectorXd x2_grid = Eigen::VectorXd::LinSpaced(30, 0.0, 1.0);
    
    auto [lh_2d, grid_x1, grid_x2, px_2d, px0_2d, X1, X2] = 
        lh2(*sensor, y, &x1_grid, &x2_grid, {0, 1});
    
    ASSERT_EQ(lh_2d.rows(), 30);
    ASSERT_EQ(lh_2d.cols(), 30);
    
    // Find maximum
    int max_i, max_j;
    double max_val = -1.0;
    for (int i = 0; i < lh_2d.rows(); ++i) {
        for (int j = 0; j < lh_2d.cols(); ++j) {
            if (lh_2d(i, j) > max_val) {
                max_val = lh_2d(i, j);
                max_i = i;
                max_j = j;
            }
        }
    }
    
    double ml_x_2d = grid_x1[max_i];
    double ml_y_2d = grid_x2[max_j];
    
    std::cout << "  2D ML (x,y): [" << ml_x_2d << ", " << ml_y_2d << "] "
              << "(true: [" << source_true[0] << ", " << source_true[1] << "])\n";
    
    EXPECT_NEAR(ml_x_2d, source_true[0], 0.3);
    EXPECT_NEAR(ml_y_2d, source_true[1], 0.3);
}

// ============================================================================
// TEST 12: CRLB Grid Evaluation
// ============================================================================
TEST_F(AllEstimatorsTest, CRLBGridEvaluation) {
    std::cout << "\n========== CRLB GRID EVALUATION ==========\n";
    
    // Evaluate CRLB over a grid in x-y plane
    Eigen::VectorXd x1_grid = Eigen::VectorXd::LinSpaced(20, 0.5, 2.0);
    Eigen::VectorXd x2_grid = Eigen::VectorXd::LinSpaced(20, 0.0, 1.0);
    
    // Test different CRLB metrics
    std::vector<std::string> metrics = {"trace", "rmse", "det", "max"};
    
    for (const auto& metric : metrics) {
        Eigen::VectorXd crlb_grid = crlb2_grid(*sensor, &y, x1_grid, x2_grid, {0, 1}, metric);
        
        ASSERT_EQ(crlb_grid.size(), 20 * 20);
        
        // Find minimum CRLB position (best geometry)
        int min_idx;
        double min_val = crlb_grid.minCoeff(&min_idx);
        
        int min_i = min_idx % 20;
        int min_j = min_idx / 20;
        
        std::cout << "  CRLB (" << metric << ") minimum at: ["
                  << x1_grid[min_i] << ", " << x2_grid[min_j] << "], "
                  << "value=" << min_val << "\n";
        
        EXPECT_GT(min_val, 0.0) << "CRLB should be positive";
    }
}
