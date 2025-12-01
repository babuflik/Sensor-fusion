#include <sensor_fusion/sensor_fusion.h>
#include <sensor_fusion/lab1_constants.h>
#include <iostream>
#include <vector>

// Task 2: Signal Modeling
// Goal: Define the measurement function h(x) and the sensor model.

int main() {
    std::cout << "=== Task 2: Signal Modeling ===" << std::endl;

    // 1. Define Microphone Positions
    // We can use the helper from lab1_constants.h or define them manually.
    Eigen::MatrixXd mics = SensorFusion::Lab1::getMicPositions();
    const double C = SensorFusion::Lab1::SOUND_SPEED;

    std::cout << "Microphone positions:\n" << mics << std::endl;

    // 2. Define Measurement Function h(x)
    // The state x is [x_pos, y_pos] (2D position).
    // The measurement y is the TDOA (Time Difference of Arrival) relative to Mic 1.
    // y = [dist(x, m2)-dist(x, m1), ..., dist(x, m8)-dist(x, m1)] / C
    
    // TODO: Implement the lambda function for h
    auto h = [mics, C](double t, const Eigen::VectorXd& x, 
                       const Eigen::VectorXd& u, const Eigen::VectorXd& th) {
        // Output vector y has size 7 (8 mics - 1 reference)
        Eigen::VectorXd y(7);
        
        // TODO: Calculate distance from state x to Mic 1 (Reference)
        Eigen::MatrixXd d1 = mics.row(0);
        
        // TODO: Loop through Mics 2 to 8 (indices 1 to 7 in 0-based indexing)
        // for (int i = 0; i < 7; ++i) {
            // Calculate distance to Mic i+1
            // Calculate TDOA: (d_i - d_1) / C
            // Store in y(i)
        // }
        
        return y;
    };

    // 3. Create SensorMod Object
    // Dimensions: nx=2 (x,y), nu=0, ny=7 (7 TDOAs), nth=0
    Eigen::Vector4i dims(2, 0, 7, 0);
    SensorMod sensor(h, dims);

    // 4. Set Measurement Noise Covariance (R or pe)
    // Use the sigma you found in Task 1!
    // sigma approx 0.00022 s
    double sigma = 0.00022; 
    
    // TODO: Set sensor.pe to a diagonal matrix with sigma^2 on the diagonal
    // sensor.pe = ...

    std::cout << "Sensor model initialized." << std::endl;
    std::cout << "Measurement noise covariance (pe):\n" << sensor.pe << std::endl;

    // 5. Test the Model (Optional but recommended)
    // Pick a test point, e.g., (0, 0)
    Eigen::VectorXd x_test(2);
    x_test << 0.0, 0.0;
    
    // Calculate expected measurements
    // Eigen::VectorXd y_pred = h(0.0, x_test, Eigen::VectorXd(), Eigen::VectorXd());
    // std::cout << "Predicted TDOAs at (0,0):\n" << y_pred << std::endl;

    return 0;
}
