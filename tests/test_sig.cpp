#include <gtest/gtest.h>
#include "sensor_fusion/sig.h"
#include <iostream>
#include <Eigen/Dense>

TEST(SigTest, BasicConstruction) {
    // Test construction with y and fs
    Eigen::MatrixXd y = Eigen::MatrixXd::Random(100, 2);
    Sig s1(y, 10.0);
    
    EXPECT_EQ(s1.length(), 100);
    EXPECT_EQ(s1.size(2), 2);  // ny = 2
    EXPECT_DOUBLE_EQ(s1.fs, 10.0);
    
    std::cout << "Basic construction test passed." << std::endl;
}

TEST(SigTest, StatisticalMethods) {
    // Create signal with known statistics
    Eigen::MatrixXd y(100, 2);
    y.col(0).setConstant(5.0);
    y.col(1).setLinSpaced(100, 0.0, 99.0);
    
    Sig s(y, 1.0);
    
    // Test mean
    Eigen::VectorXd m = s.mean_y();
    EXPECT_NEAR(m(0), 5.0, 1e-10);
    EXPECT_NEAR(m(1), 49.5, 1e-10);
    
    // Test variance
    Eigen::VectorXd v = s.var_y();
    EXPECT_NEAR(v(0), 0.0, 1e-10);  // Constant signal has zero variance
    EXPECT_GT(v(1), 0.0);  // Linear signal has non-zero variance
    
    // Test covariance
    Eigen::MatrixXd C = s.cov_y();
    EXPECT_EQ(C.rows(), 2);
    EXPECT_EQ(C.cols(), 2);
    EXPECT_NEAR(C(0, 0), v(0), 1e-10);
    EXPECT_NEAR(C(1, 1), v(1), 1e-10);
    
    std::cout << "Statistical methods test passed." << std::endl;
}

TEST(SigTest, ExtractSubset) {
    // Create signal
    Eigen::MatrixXd y = Eigen::MatrixXd::Random(100, 1);
    Eigen::VectorXd t = Eigen::VectorXd::LinSpaced(100, 0.0, 9.9);
    
    Sig s(y, t);
    
    // Extract middle portion
    Sig subset = s.extract(3.0, 7.0);
    
    EXPECT_LT(subset.length(), s.length());
    EXPECT_GE(subset.t(0), 3.0);
    EXPECT_LE(subset.t(subset.length() - 1), 7.0);
    
    std::cout << "Extract subset test passed. Original: " << s.length() 
              << " samples, subset: " << subset.length() << " samples" << std::endl;
}

TEST(SigTest, Downsample) {
    // Create signal
    Eigen::MatrixXd y = Eigen::MatrixXd::Random(100, 1);
    Sig s(y, 10.0);
    
    // Downsample by factor of 2
    Sig s_down = s.downsample(2);
    
    EXPECT_EQ(s_down.length(), 50);
    EXPECT_DOUBLE_EQ(s_down.fs, 5.0);
    
    // Check that downsampled signal has correct samples
    EXPECT_TRUE(s_down.y.row(0).isApprox(s.y.row(0)));
    EXPECT_TRUE(s_down.y.row(1).isApprox(s.y.row(2)));
    
    std::cout << "Downsample test passed. " << s.length() 
              << " -> " << s_down.length() << " samples" << std::endl;
}

TEST(SigTest, MultipleSignals) {
    // Test with multiple measurements and inputs
    Eigen::MatrixXd y = Eigen::MatrixXd::Random(50, 3);
    Eigen::VectorXd t = Eigen::VectorXd::LinSpaced(50, 0.0, 4.9);
    Eigen::MatrixXd u = Eigen::MatrixXd::Random(50, 2);
    
    Sig s(y, t, u);
    
    EXPECT_EQ(s.length(), 50);
    EXPECT_EQ(s.size(2), 3);  // ny
    EXPECT_EQ(s.size(3), 2);  // nu
    
    EXPECT_EQ(s.ylabel.size(), 3);
    EXPECT_EQ(s.ulabel.size(), 2);
    
    std::cout << "Multiple signals test passed." << std::endl;
}
