#include <gtest/gtest.h>
#include "sensor_fusion/ndist.h"
#include "sensor_fusion/udist.h"
#include "sensor_fusion/expdist.h"
#include "sensor_fusion/chi2dist.h"
#include "sensor_fusion/gammadist.h"
#include "sensor_fusion/gmdist.h"
#include "sensor_fusion/tdist.h"
#include "sensor_fusion/betadist.h"
#include "sensor_fusion/empdist.h"
#include "sensor_fusion/logndist.h"
#include "sensor_fusion/ncchi2dist.h"
#include "sensor_fusion/utils_sigsys.h"
#include <iostream>
#include <Eigen/Dense>

// Test NDist (Gaussian distribution)
TEST(DistributionTest, NDist_Basic) {
    Eigen::Vector2d mu(1.0, 2.0);
    Eigen::Matrix2d P;
    P << 1.0, 0.3,
         0.3, 2.0;
    
    NDist X(mu, P);
    
    // Test properties
    EXPECT_EQ(X.length(), 2);
    EXPECT_TRUE(X.mean().isApprox(mu));
    EXPECT_TRUE(X.cov().isApprox(P));
    
    // Test sampling
    Eigen::MatrixXd samples = X.rand(100);
    EXPECT_EQ(samples.rows(), 100);
    EXPECT_EQ(samples.cols(), 2);
    
    // Test PDF
    Eigen::MatrixXd test_points(1, 2);
    test_points << 1.0, 2.0;
    Eigen::VectorXd pdf_vals = X.pdf(test_points);
    EXPECT_GT(pdf_vals(0), 0.0);
    
    std::cout << "NDist test passed. Mean: " << X.mean().transpose() << std::endl;
}

// Test UDist (Uniform distribution)
TEST(DistributionTest, UDist_Basic) {
    UDist U(0.0, 1.0);
    
    EXPECT_EQ(U.length(), 1);
    EXPECT_NEAR(U.mean()(0), 0.5, 1e-10);
    EXPECT_NEAR(U.var(), 1.0/12.0, 1e-10);
    
    // Test sampling
    Eigen::MatrixXd samples = U.rand(1000);
    EXPECT_EQ(samples.rows(), 1000);
    
    // All samples should be in [0, 1]
    for (int i = 0; i < samples.rows(); ++i) {
        EXPECT_GE(samples(i, 0), 0.0);
        EXPECT_LE(samples(i, 0), 1.0);
    }
    
    // Test PDF
    Eigen::MatrixXd test(2, 1);
    test << 0.5, 1.5;
    Eigen::VectorXd pdf = U.pdf(test);
    EXPECT_NEAR(pdf(0), 1.0, 1e-10);  // Inside [0,1]
    EXPECT_NEAR(pdf(1), 0.0, 1e-10);  // Outside [0,1]
    
    std::cout << "UDist test passed." << std::endl;
}

// Test ExpDist (Exponential distribution)
TEST(DistributionTest, ExpDist_Basic) {
    double lambda = 2.0;
    ExpDist E(lambda);
    
    EXPECT_EQ(E.length(), 1);
    EXPECT_NEAR(E.mean()(0), 1.0/lambda, 1e-10);
    EXPECT_NEAR(E.var(), 1.0/(lambda*lambda), 1e-10);
    
    // Test sampling
    Eigen::MatrixXd samples = E.rand(100);
    EXPECT_EQ(samples.rows(), 100);
    
    // All samples should be non-negative
    for (int i = 0; i < samples.rows(); ++i) {
        EXPECT_GE(samples(i, 0), 0.0);
    }
    
    std::cout << "ExpDist test passed." << std::endl;
}

// Test Chi2Dist
TEST(DistributionTest, Chi2Dist_Basic) {
    int df = 5;
    Chi2Dist C(df);
    
    EXPECT_EQ(C.length(), 1);
    EXPECT_NEAR(C.mean()(0), df, 1e-10);
    EXPECT_NEAR(C.var(), 2.0*df, 1e-10);
    
    // Test sampling
    Eigen::MatrixXd samples = C.rand(100);
    EXPECT_EQ(samples.rows(), 100);
    
    std::cout << "Chi2Dist test passed." << std::endl;
}

// Test GammaDist
TEST(DistributionTest, GammaDist_Basic) {
    double k = 2.0;
    double theta = 3.0;
    GammaDist G(k, theta);
    
    EXPECT_EQ(G.length(), 1);
    EXPECT_NEAR(G.mean()(0), k*theta, 1e-10);
    EXPECT_NEAR(G.var(), k*theta*theta, 1e-10);
    
    // Test sampling
    Eigen::MatrixXd samples = G.rand(100);
    EXPECT_EQ(samples.rows(), 100);
    
    std::cout << "GammaDist test passed." << std::endl;
}

// Test GMDist (Gaussian Mixture)
TEST(DistributionTest, GMDist_Basic) {
    // Create 2-component mixture
    std::vector<Eigen::VectorXd> mu(2);
    std::vector<Eigen::MatrixXd> P(2);
    Eigen::Vector2d w;
    
    mu[0] = Eigen::Vector2d(0.0, 0.0);
    mu[1] = Eigen::Vector2d(3.0, 3.0);
    
    P[0] = Eigen::Matrix2d::Identity();
    P[1] = 2.0 * Eigen::Matrix2d::Identity();
    
    w << 0.6, 0.4;
    
    GMDist GM(mu, P, w);
    
    EXPECT_EQ(GM.length(), 2);
    
    // Test sampling
    Eigen::MatrixXd samples = GM.rand(100);
    EXPECT_EQ(samples.rows(), 100);
    EXPECT_EQ(samples.cols(), 2);
    
    // Test mean
    Eigen::Vector2d expected_mean = 0.6 * mu[0] + 0.4 * mu[1];
    EXPECT_TRUE(GM.mean().isApprox(expected_mean, 1e-10));
    
    std::cout << "GMDist test passed. Mean: " << GM.mean().transpose() << std::endl;
}

TEST(DistributionTest, TDist_Basic) {
    // Test Student's t-distribution
    TDist t(5.0);  // 5 degrees of freedom
    
    EXPECT_EQ(t.length(), 1);
    
    // Test sampling
    Eigen::MatrixXd samples = t.rand(100);
    EXPECT_EQ(samples.rows(), 100);
    EXPECT_EQ(samples.cols(), 1);
    
    // Mean should be 0 for centered t-dist
    EXPECT_NEAR(t.mean()(0), 0.0, 1e-10);
    
    // Variance should be nu/(nu-2) for nu > 2
    EXPECT_NEAR(t.var(), 5.0 / 3.0, 1e-10);
    
    std::cout << "TDist test passed." << std::endl;
}

TEST(DistributionTest, BetaDist_Basic) {
    // Test Beta distribution
    BetaDist beta(2.0, 5.0);
    
    EXPECT_EQ(beta.length(), 1);
    
    // Test sampling
    Eigen::MatrixXd samples = beta.rand(100);
    EXPECT_EQ(samples.rows(), 100);
    EXPECT_EQ(samples.cols(), 1);
    
    // All samples should be in [0, 1]
    EXPECT_TRUE((samples.array() >= 0.0).all());
    EXPECT_TRUE((samples.array() <= 1.0).all());
    
    // Mean should be alpha/(alpha+beta)
    EXPECT_NEAR(beta.mean()(0), 2.0 / 7.0, 1e-10);
    
    std::cout << "BetaDist test passed." << std::endl;
}

TEST(DistributionTest, EmpDist_Basic) {
    // Test empirical distribution
    std::vector<Eigen::VectorXd> samples;
    for (int i = 0; i < 100; ++i) {
        Eigen::VectorXd s(2);
        s << i * 0.1, i * 0.2;
        samples.push_back(s);
    }
    
    EmpDist emp(samples);
    
    EXPECT_EQ(emp.length(), 2);
    EXPECT_EQ(emp.n_samples(), 100);
    
    // Test resampling
    Eigen::MatrixXd resamples = emp.rand(50);
    EXPECT_EQ(resamples.rows(), 50);
    EXPECT_EQ(resamples.cols(), 2);
    
    std::cout << "EmpDist test passed." << std::endl;
}

TEST(DistributionTest, LogNDist_Basic) {
    // Test log-normal distribution
    LogNDist logn(0.0, 1.0);
    
    EXPECT_EQ(logn.length(), 1);
    
    // Test sampling
    Eigen::MatrixXd samples = logn.rand(100);
    EXPECT_EQ(samples.rows(), 100);
    EXPECT_EQ(samples.cols(), 1);
    
    // All samples should be positive
    EXPECT_TRUE((samples.array() > 0.0).all());
    
    // Mean should be exp(mu + sigma^2/2) = exp(0.5)
    EXPECT_NEAR(logn.mean()(0), std::exp(0.5), 1e-10);
    
    std::cout << "LogNDist test passed." << std::endl;
}

// Test NCChi2Dist (Non-central chi-squared distribution)
TEST(DistributionTest, NCChi2Dist_Basic) {
    // Test non-central chi-squared distribution
    NCChi2Dist nc(3, 2.0);
    
    EXPECT_EQ(nc.length(), 1);
    
    // Mean should be n + lambda = 3 + 2 = 5
    EXPECT_NEAR(nc.mean()(0), 5.0, 1e-10);
    
    // Variance should be 2*n + 4*lambda = 6 + 8 = 14
    EXPECT_NEAR(nc.var(), 14.0, 1e-10);
    
    // Test sampling
    Eigen::MatrixXd samples = nc.rand(100);
    EXPECT_EQ(samples.rows(), 100);
    EXPECT_EQ(samples.cols(), 1);
    
    // All samples should be positive
    EXPECT_TRUE((samples.array() > 0.0).all());
    
    // Test PDF
    Eigen::MatrixXd test_points(1, 1);
    test_points << 5.0;
    Eigen::VectorXd pdf_vals = nc.pdf(test_points);
    EXPECT_GT(pdf_vals(0), 0.0);
    
    // Test central case (lambda = 0)
    NCChi2Dist central(3, 0.0);
    EXPECT_NEAR(central.mean()(0), 3.0, 1e-10);
    EXPECT_NEAR(central.var(), 6.0, 1e-10);
    
    std::cout << "NCChi2Dist test passed. Mean: " << nc.mean()(0) << ", Var: " << nc.var() << std::endl;
}

// Test numerical utilities
TEST(UtilsTest, NumGrad) {
    // Test on a simple quadratic function
    auto f = [](const Eigen::VectorXd& x) {
        return x.dot(x);  // f(x) = x^T x
    };
    
    Eigen::Vector2d x(1.0, 2.0);
    Eigen::VectorXd grad = numgrad(f, x);
    
    // Gradient should be 2*x
    Eigen::Vector2d expected(2.0, 4.0);
    EXPECT_TRUE(grad.isApprox(expected, 1e-6));
    
    std::cout << "NumGrad test passed. Grad: " << grad.transpose() << std::endl;
}

TEST(UtilsTest, NumHess) {
    // Test on a simple quadratic function
    auto f = [](const Eigen::VectorXd& x) {
        return x.dot(x);  // f(x) = x^T x
    };
    
    Eigen::Vector2d x(1.0, 2.0);
    Eigen::MatrixXd hess = numhess(f, x);
    
    // Hessian should be 2*I
    Eigen::Matrix2d expected = 2.0 * Eigen::Matrix2d::Identity();
    EXPECT_TRUE(hess.isApprox(expected, 1e-4));
    
    std::cout << "NumHess test passed." << std::endl;
}

TEST(UtilsTest, SqrtCov) {
    Eigen::Matrix2d P;
    P << 4.0, 1.2,
         1.2, 9.0;
    
    Eigen::MatrixXd L = sqrtcov(P);
    Eigen::Matrix2d reconstructed = L * L.transpose();
    
    EXPECT_TRUE(reconstructed.isApprox(P, 1e-10));
    
    std::cout << "SqrtCov test passed." << std::endl;
}

TEST(UtilsTest, IsCov) {
    Eigen::Matrix2d P_valid;
    P_valid << 1.0, 0.3,
               0.3, 2.0;
    
    auto [valid1, code1] = iscov(P_valid);
    EXPECT_TRUE(valid1);
    EXPECT_EQ(code1, 0);
    
    // Test non-symmetric matrix
    Eigen::Matrix2d P_nonsym;
    P_nonsym << 1.0, 0.5,
                0.3, 2.0;
    
    auto [valid2, code2] = iscov(P_nonsym);
    EXPECT_FALSE(valid2);
    EXPECT_EQ(code2, 3);  // Not symmetric
    
    std::cout << "IsCov test passed." << std::endl;
}

TEST(UtilsTest, ConfEllipse) {
    Eigen::Matrix2d P;
    P << 1.0, 0.3,
         0.3, 0.5;
    
    Eigen::Vector2d center(1.0, 2.0);
    
    Eigen::MatrixXd ellipse = confellipse(P, center, 0.95, 50);
    
    // Confellipse returns npoints+1 to close the curve
    EXPECT_EQ(ellipse.rows(), 51);
    EXPECT_EQ(ellipse.cols(), 2);
    
    // Check that ellipse closes (first ≈ last point)
    EXPECT_TRUE(ellipse.row(0).isApprox(ellipse.row(ellipse.rows()-1), 0.01));
    
    std::cout << "ConfEllipse test passed." << std::endl;
}

TEST(UtilsTest, GetWindow) {
    int N = 64;
    
    Eigen::VectorXd hamming = getwindow("hamming", N);
    EXPECT_EQ(hamming.size(), N);
    EXPECT_NEAR(hamming(0), hamming(N-1), 1e-10);  // Symmetric
    EXPECT_LT(hamming(0), hamming(N/2));  // Edge < center
    
    Eigen::VectorXd rect = getwindow("rectangular", N);
    EXPECT_TRUE(rect.isOnes());
    
    std::cout << "GetWindow test passed." << std::endl;
}

TEST(UtilsTest, Filtfilt) {
    // Create a simple signal with a step
    int N = 100;
    Eigen::VectorXd x(N);
    for (int i = 0; i < N; ++i) {
        x(i) = (i < N/2) ? 0.0 : 1.0;
    }
    
    // Simple moving average filter (FIR)
    Eigen::VectorXd b(3);
    b << 1.0/3.0, 1.0/3.0, 1.0/3.0;
    Eigen::VectorXd a(1);
    a << 1.0;
    
    Eigen::VectorXd y = filtfilt(b, a, x);
    
    EXPECT_EQ(y.size(), N);
    // Zero-phase filter should not introduce phase shift
    // Check that filtered signal is smoother
    double variance_x = (x.array() - x.mean()).square().mean();
    double variance_y = (y.array() - y.mean()).square().mean();
    EXPECT_LT(variance_y, variance_x * 1.1);  // Some smoothing occurred
    
    std::cout << "Filtfilt test passed." << std::endl;
}

TEST(UtilsTest, Interp) {
    // Create simple linear signal
    Eigen::VectorXd t1(5);
    t1 << 0.0, 1.0, 2.0, 3.0, 4.0;
    Eigen::VectorXd y1(5);
    y1 << 0.0, 2.0, 4.0, 6.0, 8.0;  // y = 2*t
    
    // Interpolate at intermediate points
    Eigen::VectorXd t2(3);
    t2 << 0.5, 1.5, 2.5;
    
    Eigen::VectorXd y2 = interp(y1, t1, t2);
    
    EXPECT_EQ(y2.size(), 3);
    EXPECT_NEAR(y2(0), 1.0, 1e-10);  // 2*0.5
    EXPECT_NEAR(y2(1), 3.0, 1e-10);  // 2*1.5
    EXPECT_NEAR(y2(2), 5.0, 1e-10);  // 2*2.5
    
    std::cout << "Interp test passed." << std::endl;
}

TEST(UtilsTest, Resample) {
    // Create signal at 100 Hz
    double fs1 = 100.0;
    int N1 = 100;
    Eigen::VectorXd y1(N1);
    for (int i = 0; i < N1; ++i) {
        y1(i) = std::sin(2.0 * M_PI * 5.0 * i / fs1);  // 5 Hz sine wave
    }
    
    // Resample to 50 Hz (downsample by 2)
    double fs2 = 50.0;
    Eigen::VectorXd y2 = resample(y1, fs1, fs2);
    
    // Should have approximately half the samples
    EXPECT_GT(y2.size(), N1/2 - 5);
    EXPECT_LT(y2.size(), N1/2 + 5);
    
    std::cout << "Resample test passed. N1=" << N1 << ", N2=" << y2.size() << std::endl;
}

TEST(UtilsTest, DownsampleUpsample) {
    Eigen::VectorXd x(10);
    for (int i = 0; i < 10; ++i) {
        x(i) = i;
    }
    
    // Downsample by 2
    Eigen::VectorXd y_down = downsample(x, 2);
    EXPECT_EQ(y_down.size(), 5);
    EXPECT_NEAR(y_down(0), 0.0, 1e-10);
    EXPECT_NEAR(y_down(1), 2.0, 1e-10);
    EXPECT_NEAR(y_down(2), 4.0, 1e-10);
    
    // Upsample by 2
    Eigen::VectorXd y_up = upsample(x, 2);
    EXPECT_EQ(y_up.size(), 20);
    EXPECT_NEAR(y_up(0), 0.0, 1e-10);
    EXPECT_NEAR(y_up(1), 0.0, 1e-10);  // Zero inserted
    EXPECT_NEAR(y_up(2), 1.0, 1e-10);
    
    std::cout << "Downsample/Upsample test passed." << std::endl;
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}

