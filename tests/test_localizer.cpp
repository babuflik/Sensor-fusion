#include <sensor_fusion/sensormod.h>
#include <sensor_fusion/estimators.h> // ls, wls, ml, crlb
#include <gtest/gtest.h>
#include <Eigen/Dense>
#include <iostream>

// Helper: create synthetic signal for SensorMod
Sig generateSyntheticSignal(SensorMod& sensor,
                            const Eigen::VectorXd& t,
                            const Eigen::VectorXd& x0,
                            int MC = 0)
{
    Eigen::MatrixXd x = x0.replicate(1, t.size());
    return sensor.simulate(t, &x, nullptr, MC);
}

// --- TEST ---
TEST(SensorModTest, SyntheticSourceEstimators) {
    Eigen::Vector4i nn{3, 0, 3, 0}; // [nx, nu, ny, nth]
    auto hFunc = [](double t, const Eigen::VectorXd& x,
                    const Eigen::VectorXd& u,
                    const Eigen::VectorXd& th) {
        return x; // enkel identitetsmodell
    };

    SensorMod sensor(hFunc, nn);

    Eigen::VectorXd t(5);
    t << 0,1,2,3,4;

    // Skapa syntetisk signal
    Sig y = generateSyntheticSignal(sensor, t, sensor.x0);

    // --- CRLB ---
    Sig crlb_sig = crlb(sensor, &y);
    if (!crlb_sig.Px.empty()) {
        std::cout << "CRLB (Px):\n" << crlb_sig.Px[0] << "\n";
    }

    // Here you can continue with LS/WLS/ML estimators
}

