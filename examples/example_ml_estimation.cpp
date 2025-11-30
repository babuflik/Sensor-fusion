/**
 * @file example_ml_estimation.cpp
 * @brief Maximum Likelihood estimation example
 * 
 * Demonstrates using ML estimator for parameter estimation
 */

#include <sensor_fusion/sensor_fusion.h>
#include <iostream>

int main() {
    std::cout << "=== Sensor Fusion Toolbox - ML Estimation Example ===" << std::endl;
    std::cout << std::endl;
    
    // Problem: Estimate position of a sound source from TDOA measurements
    // State: [x, y, z] - source position
    // Measurements: Time differences of arrival at 4 microphones
    
    // Microphone positions
    Eigen::MatrixXd mic_pos(4, 3);
    mic_pos << 0.0, 0.0, 0.0,   // Mic 1
               0.2, 0.0, 0.0,   // Mic 2  
               0.2, 0.2, 0.0,   // Mic 3
               0.0, 0.2, 0.0;   // Mic 4
    
    const double SOUND_SPEED = 343.0;  // m/s
    
    // TDOA measurement function
    auto h = [mic_pos, SOUND_SPEED](double t, const Eigen::VectorXd& x, 
                                     const Eigen::VectorXd& u, const Eigen::VectorXd& th) {
        Eigen::VectorXd tdoa(6);  // 6 TDOA pairs
        
        // Calculate distances from source to each mic
        Eigen::VectorXd dist(4);
        for (int i = 0; i < 4; ++i) {
            Eigen::Vector3d diff = x - mic_pos.row(i).transpose();
            dist(i) = diff.norm();
        }
        
        // Calculate TDOAs (relative to mic 1)
        int idx = 0;
        for (int i = 0; i < 4; ++i) {
            for (int j = i + 1; j < 4; ++j) {
                tdoa(idx++) = (dist(i) - dist(j)) / SOUND_SPEED;
            }
        }
        
        return tdoa;
    };
    
    Eigen::Vector4i nn(3, 0, 6, 0);  // nx=3 (position), nu=0, ny=6 (TDOAs), nth=0
    SensorMod sensor(h, nn);
    
    // Set measurement noise (TDOA estimation uncertainty)
    sensor.pe = Eigen::MatrixXd::Identity(6, 6) * 1e-6;  // Very low noise for this example
    
    // True source position
    Eigen::MatrixXd x_true(3, 1);
    x_true << 0.15, 0.10, 0.05;  // True position
    
    std::cout << "True source position: (" 
              << x_true(0, 0) << ", "
              << x_true(1, 0) << ", "
              << x_true(2, 0) << ") m" << std::endl;
    
    // Simulate measurements
    Eigen::VectorXd t(1);
    t(0) = 0.0;
    
    Sig measurements = sensor.simulate(t, &x_true);
    
    std::cout << "TDOA measurements (μs):" << std::endl;
    for (int i = 0; i < 6; ++i) {
        std::cout << "  TDOA[" << i << "]: " << measurements.y(i, 0) * 1e6 << std::endl;
    }
    std::cout << std::endl;
    
    // Run ML estimation
    std::cout << "Running ML estimation..." << std::endl;
    auto [x_est, s_est, cov] = ml(sensor, measurements);
    
    std::cout << "Estimated position: ("
              << x_est.x(0, 0) << ", "
              << x_est.x(1, 0) << ", "
              << x_est.x(2, 0) << ") m" << std::endl;
    
    Eigen::Vector3d error = x_est.x - x_true;
    std::cout << "Estimation error: " << error.norm() << " m" << std::endl;
    std::cout << "Uncertainty (trace of cov): " << cov.trace() << std::endl;
    std::cout << std::endl;
    
    std::cout << "ML estimation example complete!" << std::endl;
    std::cout << "This demonstrates acoustic source localization from TDOA!" << std::endl;
    
    return 0;
}
