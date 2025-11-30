#include <gtest/gtest.h>
#include <sensor_fusion/nl.h>
#include <iostream>
#include <Eigen/Dense>
#include <cmath>

// Test simple linear system (should work with EKF)
TEST(NLTest, LinearSystem_EKF) {
    // Simple scalar system: x(t+1) = 0.9*x(t) + u(t), y(t) = x(t)
    auto f = [](double t, const Eigen::VectorXd& x, const Eigen::VectorXd& u, const Eigen::VectorXd& th) {
        Eigen::VectorXd xnext(1);
        xnext(0) = 0.9 * x(0) + u(0);
        return xnext;
    };
    
    auto h = [](double t, const Eigen::VectorXd& x, const Eigen::VectorXd& u, const Eigen::VectorXd& th) {
        return x;  // y = x
    };
    
    Eigen::Vector4i nn(1, 1, 1, 0);  // nx=1, nu=1, ny=1, nth=0
    NL model(f, h, nn, 1.0);  // fs = 1 Hz
    
    model.x0 << 0.0;
    
    // Set noise covariances
    Eigen::MatrixXd Q(1, 1);
    Q << 0.01;
    model.set_pv(Q);
    
    Eigen::MatrixXd R(1, 1);
    R << 0.1;
    model.set_pe(R);
    
    // Create input signal
    int N = 50;
    Eigen::MatrixXd u = Eigen::MatrixXd::Ones(1, N);
    
    // Simulate
    Sig z = model.simulate(u);
    
    EXPECT_EQ(z.y.rows(), 1);
    EXPECT_EQ(z.y.cols(), N);
    EXPECT_EQ(z.x.rows(), 1);
    EXPECT_EQ(z.x.cols(), N);
    
    // Run EKF
    Eigen::VectorXd x0_init(1);
    x0_init << 0.0;
    Eigen::MatrixXd P0_init = Eigen::MatrixXd::Identity(1, 1);
    
    auto [x_est, P_est] = model.ekf(z, u, x0_init, P0_init);
    
    EXPECT_EQ(x_est.rows(), 1);
    EXPECT_EQ(x_est.cols(), N);
    EXPECT_EQ(P_est.size(), N);
    
    // Check that estimates are reasonable
    EXPECT_TRUE(std::isfinite(x_est(0, N - 1)));
    EXPECT_TRUE(std::isfinite(P_est[N - 1](0, 0)));
    EXPECT_GT(P_est[N - 1](0, 0), 0.0);
    
    std::cout << "EKF test passed. Final estimate: " << x_est(0, N - 1) << std::endl;
}

// Test nonlinear system
TEST(NLTest, NonlinearSystem_Simulate) {
    // Nonlinear system: x(t+1) = x(t) + 0.1*sin(x(t)), y(t) = x(t)^2
    auto f = [](double t, const Eigen::VectorXd& x, const Eigen::VectorXd& u, const Eigen::VectorXd& th) {
        Eigen::VectorXd xnext(1);
        xnext(0) = x(0) + 0.1 * std::sin(x(0));
        return xnext;
    };
    
    auto h = [](double t, const Eigen::VectorXd& x, const Eigen::VectorXd& u, const Eigen::VectorXd& th) {
        Eigen::VectorXd y(1);
        y(0) = x(0) * x(0);
        return y;
    };
    
    Eigen::Vector4i nn(1, 0, 1, 0);  // nx=1, nu=0, ny=1, nth=0
    NL model(f, h, nn, 10.0);
    
    model.x0 << 1.0;
    
    // Simulate without noise
    Eigen::MatrixXd u_empty;
    Sig z = model.simulate(u_empty, Eigen::VectorXd::LinSpaced(100, 0, 9.9));
    
    EXPECT_EQ(z.y.rows(), 1);
    EXPECT_EQ(z.y.cols(), 100);
    
    // Check that output is indeed x^2
    double x_final = z.x(0, 99);
    double y_final = z.y(0, 99);
    EXPECT_NEAR(y_final, x_final * x_final, 1e-6);
    
    std::cout << "Nonlinear simulation test passed." << std::endl;
}

// Test UKF on nonlinear system
TEST(NLTest, UKF_Basic) {
    // Simple nonlinear: x(t+1) = 0.5*x(t) + x(t)^3/10, y(t) = x(t)
    auto f = [](double t, const Eigen::VectorXd& x, const Eigen::VectorXd& u, const Eigen::VectorXd& th) {
        Eigen::VectorXd xnext(1);
        xnext(0) = 0.5 * x(0) + x(0) * x(0) * x(0) / 10.0;
        return xnext;
    };
    
    auto h = [](double t, const Eigen::VectorXd& x, const Eigen::VectorXd& u, const Eigen::VectorXd& th) {
        return x;
    };
    
    Eigen::Vector4i nn(1, 0, 1, 0);
    NL model(f, h, nn, 1.0);
    
    model.x0 << 0.5;
    
    Eigen::MatrixXd Q(1, 1);
    Q << 0.001;
    model.set_pv(Q);
    
    Eigen::MatrixXd R(1, 1);
    R << 0.01;
    model.set_pe(R);
    
    // Simulate
    Eigen::MatrixXd u_empty;
    Sig z = model.simulate(u_empty, Eigen::VectorXd::LinSpaced(30, 0, 29));
    
    // Run UKF
    Eigen::VectorXd x0_init(1);
    x0_init << 0.0;  // Wrong initial guess
    Eigen::MatrixXd P0_init = Eigen::MatrixXd::Identity(1, 1);
    
    auto [x_est, P_est] = model.ukf(z, u_empty, x0_init, P0_init);
    
    EXPECT_EQ(x_est.rows(), 1);
    EXPECT_EQ(x_est.cols(), 30);
    EXPECT_TRUE(std::isfinite(x_est(0, 29)));
    
    std::cout << "UKF test passed. Final estimate: " << x_est(0, 29) 
              << ", True: " << z.x(0, 29) << std::endl;
}

// Test Particle Filter
TEST(NLTest, PF_Basic) {
    // Linear system for PF test
    auto f = [](double t, const Eigen::VectorXd& x, const Eigen::VectorXd& u, const Eigen::VectorXd& th) {
        Eigen::VectorXd xnext(1);
        xnext(0) = 0.95 * x(0);
        return xnext;
    };
    
    auto h = [](double t, const Eigen::VectorXd& x, const Eigen::VectorXd& u, const Eigen::VectorXd& th) {
        return x;
    };
    
    Eigen::Vector4i nn(1, 0, 1, 0);
    NL model(f, h, nn, 1.0);
    
    model.x0 << 5.0;
    
    Eigen::MatrixXd Q(1, 1);
    Q << 0.1;
    model.set_pv(Q);
    
    Eigen::MatrixXd R(1, 1);
    R << 0.5;
    model.set_pe(R);
    
    // Simulate
    Sig z = model.simulate(Eigen::MatrixXd(), Eigen::VectorXd::LinSpaced(20, 0, 19));
    
    // Run PF
    Eigen::MatrixXd x_est = model.pf(z, Eigen::MatrixXd(), 500);
    
    EXPECT_EQ(x_est.rows(), 1);
    EXPECT_EQ(x_est.cols(), 20);
    EXPECT_TRUE(std::isfinite(x_est(0, 19)));
    
    std::cout << "PF test passed. Final estimate: " << x_est(0, 19)
              << ", True: " << z.x(0, 19) << std::endl;
}

// Test 2D system
TEST(NLTest, TwoDimensional_EKF) {
    // 2D system: position and velocity
    // x1(t+1) = x1(t) + x2(t)*dt
    // x2(t+1) = 0.9*x2(t)
    // y(t) = x1(t)
    
    double dt = 0.1;
    auto f = [dt](double t, const Eigen::VectorXd& x, const Eigen::VectorXd& u, const Eigen::VectorXd& th) {
        Eigen::VectorXd xnext(2);
        xnext(0) = x(0) + x(1) * dt;
        xnext(1) = 0.9 * x(1);
        return xnext;
    };
    
    auto h = [](double t, const Eigen::VectorXd& x, const Eigen::VectorXd& u, const Eigen::VectorXd& th) {
        Eigen::VectorXd y(1);
        y(0) = x(0);
        return y;
    };
    
    Eigen::Vector4i nn(2, 0, 1, 0);
    NL model(f, h, nn, 10.0);
    
    model.x0 << 0.0, 1.0;
    
    Eigen::MatrixXd Q = 0.01 * Eigen::MatrixXd::Identity(2, 2);
    model.set_pv(Q);
    
    Eigen::MatrixXd R(1, 1);
    R << 0.1;
    model.set_pe(R);
    
    // Simulate
    Sig z = model.simulate(Eigen::MatrixXd(), Eigen::VectorXd::LinSpaced(50, 0, 4.9));
    
    // EKF
    auto [x_est, P_est] = model.ekf(z);
    
    EXPECT_EQ(x_est.rows(), 2);
    EXPECT_EQ(x_est.cols(), 50);
    
    std::cout << "2D EKF test passed. Final state: [" 
              << x_est(0, 49) << ", " << x_est(1, 49) << "]" << std::endl;
}

// Test CRLB (Cramér-Rao Lower Bound)
TEST(NLTest, CRLB_Basic) {
    // Simple linear system: x(t+1) = 0.9*x(t) + u(t), y(t) = x(t)
    auto f = [](double t, const Eigen::VectorXd& x, const Eigen::VectorXd& u, const Eigen::VectorXd& th) {
        Eigen::VectorXd xnext(1);
        xnext(0) = 0.9 * x(0) + u(0);
        return xnext;
    };
    
    auto h = [](double t, const Eigen::VectorXd& x, const Eigen::VectorXd& u, const Eigen::VectorXd& th) {
        return x;  // y = x
    };
    
    Eigen::Vector4i nn(1, 1, 1, 0);  // nx=1, nu=1, ny=1, nth=0
    NL model(f, h, nn, 1.0);  // fs = 1 Hz
    
    model.x0 << 0.0;
    
    // Set noise covariances
    Eigen::MatrixXd Q(1, 1);
    Q << 0.01;
    model.set_pv(Q);
    
    Eigen::MatrixXd R(1, 1);
    R << 0.1;
    model.set_pe(R);
    
    // Create input signal
    int N = 50;
    Eigen::MatrixXd u = Eigen::MatrixXd::Ones(1, N);
    
    // Simulate to get true state trajectory
    Sig z = model.simulate(u);
    
    EXPECT_EQ(z.y.rows(), 1);
    EXPECT_EQ(z.y.cols(), N);
    EXPECT_EQ(z.x.rows(), 1);
    EXPECT_EQ(z.x.cols(), N);
    
    // Compute CRLB
    Eigen::VectorXd x0_init(1);
    x0_init << 0.0;
    Eigen::MatrixXd P0_init = Eigen::MatrixXd::Identity(1, 1);
    
    // CRLB signature: crlb(z, P0_init, Q_override, R_override, pred_horizon)
    // pred_horizon=0 means return filter results (default)
    Sig crlb_sig = model.crlb(z, P0_init);
    
    // Check dimensions
    EXPECT_EQ(crlb_sig.Px.size(), N);
    
    // CRLB should be positive definite
    for (int k = 0; k < N; ++k) {
        EXPECT_GT(crlb_sig.Px[k](0, 0), 0.0);
        EXPECT_TRUE(std::isfinite(crlb_sig.Px[k](0, 0)));
    }
    
    // For linear system with known dynamics, CRLB should match Kalman filter covariance
    // Run EKF for comparison
    auto [x_est, P_est] = model.ekf(z, u, x0_init, P0_init);
    
    // CRLB should be similar to steady-state Kalman covariance (may differ initially)
    double crlb_final = crlb_sig.Px[N - 1](0, 0);
    double kf_final = P_est[N - 1](0, 0);
    
    // They should be close for a linear system
    EXPECT_NEAR(crlb_final, kf_final, 0.1);
    
    std::cout << "CRLB test passed. Final CRLB: " << crlb_final 
              << ", KF covariance: " << kf_final << std::endl;
}
