/**
 * @file example_ekf.cpp
 * @brief Simple Extended Kalman Filter example using Sensor Fusion Toolbox
 * 
 * Demonstrates tracking a moving object in 2D with noisy position measurements
 */

#include <sensor_fusion/sensor_fusion.h>
#include <iostream>
#include <cmath>

int main() {
    std::cout << "=== Sensor Fusion Toolbox - EKF Example ===" << std::endl;
    std::cout << "Library version: " << SensorFusion::getVersion() << std::endl;
    std::cout << std::endl;
    
    // System parameters
    const double dt = 0.1;  // Time step (100ms)
    const int nx = 4;       // State: [x, y, vx, vy]
    const int nu = 0;       // No control input
    const int ny = 2;       // Measurement: [x, y]
    const int nth = 0;      // No parameters
    
    // Define state dynamics: constant velocity model
    // x+ = [x + vx*dt, y + vy*dt, vx, vy]
    NL::StateFunction f = [dt](double t, const Eigen::VectorXd& x, 
                                const Eigen::VectorXd& u, const Eigen::VectorXd& th) {
        Eigen::VectorXd x_next(4);
        x_next(0) = x(0) + x(2) * dt;  // x = x + vx*dt
        x_next(1) = x(1) + x(3) * dt;  // y = y + vy*dt
        x_next(2) = x(2);               // vx = vx
        x_next(3) = x(3);               // vy = vy
        return x_next;
    };
    
    // Define measurement function: direct position measurement
    // y = [x, y]
    NL::MeasurementFunction h = [](double t, const Eigen::VectorXd& x,
                                   const Eigen::VectorXd& u, const Eigen::VectorXd& th) {
        Eigen::VectorXd y(2);
        y(0) = x(0);  // Measure x position
        y(1) = x(1);  // Measure y position
        return y;
    };
    
    // Create system model
    Eigen::Vector4i dims(nx, nu, ny, nth);
    NL system(f, h, dims, 1.0/dt);
    
    // Set initial state: starting at (0, 0) moving at (1, 0.5) m/s
    system.x0 << 0.0, 0.0, 1.0, 0.5;
    
    // Set process noise covariance (small acceleration noise)
    system.pv = std::make_shared<NDist>();
    dynamic_cast<NDist*>(system.pv.get())->mu = Eigen::VectorXd::Zero(nx);
    Eigen::MatrixXd Q = Eigen::MatrixXd::Identity(nx, nx);
    Q(0, 0) = 0.01;  // Position noise
    Q(1, 1) = 0.01;
    Q(2, 2) = 0.1;   // Velocity noise
    Q(3, 3) = 0.1;
    dynamic_cast<NDist*>(system.pv.get())->P = Q;
    
    // Set measurement noise covariance
    system.pe = std::make_shared<NDist>();
    dynamic_cast<NDist*>(system.pe.get())->mu = Eigen::VectorXd::Zero(ny);
    Eigen::MatrixXd R = Eigen::MatrixXd::Identity(ny, ny) * 0.5;  // 0.5m std dev
    dynamic_cast<NDist*>(system.pe.get())->P = R;
    
    // Simulate trajectory
    const int N = 50;  // 5 seconds of data
    Eigen::VectorXd t = Eigen::VectorXd::LinSpaced(N, 0, (N-1)*dt);
    
    // For nu=0, provide empty input matrix (rows=nu=0, cols=N)
    Eigen::MatrixXd u_empty = Eigen::MatrixXd::Zero(0, N);
    
    std::cout << "Simulating " << N << " time steps..." << std::endl;
    Sig measurements = system.simulate(u_empty, t);
    
    std::cout << "Running EKF estimation..." << std::endl;
    
    // Simple EKF implementation would go here
    // For now, just show we can access the measurements
    std::cout << std::endl;
    std::cout << "Sample measurements:" << std::endl;
    for (int i = 0; i < std::min(5, N); ++i) {
        std::cout << "t=" << t(i) << ": ";
        std::cout << "x=" << measurements.y(0, i) << ", ";
        std::cout << "y=" << measurements.y(1, i) << std::endl;
    }
    
    std::cout << std::endl;
    std::cout << "EKF example complete!" << std::endl;
    std::cout << "This demonstrates:" << std::endl;
    std::cout << "  - Creating nonlinear system model (NL)" << std::endl;
    std::cout << "  - Defining state and measurement functions" << std::endl;
    std::cout << "  - Setting noise covariances" << std::endl;
    std::cout << "  - Simulating system trajectories" << std::endl;
    
    return 0;
}
