/**
 * @file example_sensor_model.cpp
 * @brief Demonstrates sensor modeling with the Sensor Fusion Toolbox
 * 
 * Shows how to create a sensor model and use ML estimation
 */

#include <sensor_fusion/sensor_fusion.h>
#include <iostream>

int main() {
    std::cout << "=== Sensor Fusion Toolbox - Sensor Model Example ===" << std::endl;
    std::cout << std::endl;
    
    // Define a simple range-bearing sensor
    // State: [x, y] - target position
    // Measurement: [range, bearing] from sensor at origin
    
    auto h = [](double t, const Eigen::VectorXd& x, const Eigen::VectorXd& u, const Eigen::VectorXd& th) {
        Eigen::VectorXd y(2);
        double range = std::sqrt(x(0)*x(0) + x(1)*x(1));
        double bearing = std::atan2(x(1), x(0));
        y(0) = range;
        y(1) = bearing;
        return y;
    };
    
    Eigen::Vector4i nn(2, 0, 2, 0);  // nx=2, nu=0, ny=2, nth=0
    SensorMod sensor(h, nn);
    
    // Set measurement noise
    sensor.pe = Eigen::MatrixXd::Identity(2, 2);
    sensor.pe(0, 0) = 0.1 * 0.1;  // Range noise: 10cm std dev
    sensor.pe(1, 1) = 0.05 * 0.05;  // Bearing noise: 0.05 rad std dev
    
    // Simulate measurements from target at (5, 3)
    Eigen::VectorXd t(1);
    t(0) = 0.0;
    
    Eigen::MatrixXd x_true(2, 1);
    x_true << 5.0, 3.0;
    
    std::cout << "True target position: (" << x_true(0, 0) << ", " << x_true(1, 0) << ")" << std::endl;
    
    Sig measurements = sensor.simulate(t, &x_true);
    
    std::cout << "Measurement:" << std::endl;
    std::cout << "  Range: " << measurements.y(0, 0) << " m" << std::endl;
    std::cout << "  Bearing: " << measurements.y(1, 0) << " rad ("
              << measurements.y(1, 0) * 180.0 / M_PI << " deg)" << std::endl;
    std::cout << std::endl;
    
    // Use ML estimation to estimate position
    std::cout << "Running ML estimation..." << std::endl;
    auto [x_est, s_est, cov] = ml(sensor, measurements);
    
    std::cout << "Estimated position: (" << x_est.x(0, 0) << ", " << x_est.x(1, 0) << ")" << std::endl;
    std::cout << "Estimation error: (" 
              << (x_est.x(0, 0) - x_true(0, 0)) << ", "
              << (x_est.x(1, 0) - x_true(1, 0)) << ")" << std::endl;
    std::cout << std::endl;
    
    std::cout << "Sensor model example complete!" << std::endl;
    
    return 0;
}
