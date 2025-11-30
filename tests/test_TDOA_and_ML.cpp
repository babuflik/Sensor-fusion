#include <gtest/gtest.h>
#include <Eigen/Dense>
#include <cmath>
#include <sensor_fusion/sensormod.h>
#include <sensor_fusion/sig.h>
#include "utils.h"
#include <sensor_fusion/estimators.h>

// -------------------------------------------------------
// Simple GCC-PHAT (without FFTW, uses direct cross-correlation)
// -------------------------------------------------------
Eigen::VectorXd gcc_phat(const Eigen::VectorXd& x,
                         const Eigen::VectorXd& y)
{
    int N = x.size();
    Eigen::VectorXd corr = Eigen::VectorXd::Zero(2*N - 1);

    for(int lag = -(N-1); lag <= (N-1); ++lag) {
        double sum = 0;
        for(int n = 0; n < N; ++n) {
            int m = n + lag;  // Positive lag means y is delayed relative to x
            if (m >= 0 && m < N)
                sum += x[n] * y[m];
        }
        corr(lag + N - 1) = sum;
    }

    // Find maximum correlation
    int idx;
    corr.maxCoeff(&idx);

    return Eigen::VectorXd::Constant(1, idx - (N-1));  // return lag
}

// -------------------------------------------------------
// Generate point source / impulse in microphone array
// -------------------------------------------------------
Eigen::VectorXd simulateArrival(double fs,
                                double distance,
                                double c,
                                double phi)
{
    int N = 1024;
    Eigen::VectorXd sig = Eigen::VectorXd::Zero(N);

    double t0 = (distance + phi) / c; // time to microphone

    int sample = int(t0 * fs);
    if (sample >= 0 && sample < N)
        sig(sample) = 1.0;    // impulse

    return sig;
}

// -------------------------------------------------------
// TEST
// -------------------------------------------------------
TEST(LocalizerTest, MicrophoneLocalizationEndToEnd)
{
    double c = 343.0;    // speed of sound (m/s)
    double fs = 48000;

    // ---------------------------------------------------
    // Microphone placement: 3 mics in a triangle
    // ---------------------------------------------------
    Eigen::Matrix<double,3,2> mic_pos;
    mic_pos << 0.0,  0.0,      // mic 1 at origin
               0.1,  0.0,      // mic 2 (10 cm on x-axis)
               0.05, 0.0866;   // mic 3 (5 cm x, ~8.66 cm y, forms equilateral triangle)

    // ---------------------------------------------------
    // Ground truth source
    // ---------------------------------------------------
    Eigen::Vector2d source;
    source << 1.0, 0.4;   // 1 m forward, 0.4 m up

    // ---------------------------------------------------
    // Expected distances
    // ---------------------------------------------------
    double d1 = (source - mic_pos.row(0).transpose()).norm();
    double d2 = (source - mic_pos.row(1).transpose()).norm();
    double d3 = (source - mic_pos.row(2).transpose()).norm();
    
    double tdoa_12 = (d2 - d1) / c;  // TDOA between mic 1 and 2
    double tdoa_13 = (d3 - d1) / c;  // TDOA between mic 1 and 3

    // ---------------------------------------------------
    // INITIALIZE TRUE VALUES (TOA vector)
    // ---------------------------------------------------
    Eigen::VectorXd tdoa_true_vec(3);
    tdoa_true_vec << d1 / c, 
                     d2 / c,
                     d3 / c;

    // ---------------------------------------------------
    // 1) Simulate microphone signals
    // ---------------------------------------------------
    Eigen::VectorXd x1 = simulateArrival(fs, d1, c, 0);
    Eigen::VectorXd x2 = simulateArrival(fs, d2, c, 0);
    Eigen::VectorXd x3 = simulateArrival(fs, d3, c, 0);

    // ---------------------------------------------------
    // 2) TDOA via GCC-PHAT
    // ---------------------------------------------------
    Eigen::VectorXd lag12 = gcc_phat(x1, x2);
    Eigen::VectorXd lag13 = gcc_phat(x1, x3);
    double tdoa_est_12 = lag12[0] / fs;
    double tdoa_est_13 = lag13[0] / fs;

    // ---------------------------------------------------
    // INITIALIZE ESTIMATED VALUES (TDOA vector, relative to mic 1)
    // ---------------------------------------------------
    Eigen::VectorXd tdoa_est_vec(3);
    tdoa_est_vec << 0.0,           // reference mic
                    tdoa_est_12,   // TDOA to mic 2
                    tdoa_est_13;   // TDOA to mic 3

    ASSERT_NEAR(tdoa_est_12, tdoa_12, 1.0/fs);
    ASSERT_NEAR(tdoa_est_13, tdoa_13, 1.0/fs);

    // ---------------------------------------------------
    // 3) ML localization via SensorMod::likelihood_function
    // ---------------------------------------------------

    // Sensor function: y = [dist1, dist2, dist3]
    auto h = [&](double t,
                 const Eigen::VectorXd& X,
                 const Eigen::VectorXd& u,
                 const Eigen::VectorXd& th)
                 -> Eigen::VectorXd
    {
        Eigen::Vector2d p = X.head<2>();
        Eigen::VectorXd out(3);
        out[0] = (p - mic_pos.row(0).transpose()).norm();
        out[1] = (p - mic_pos.row(1).transpose()).norm();
        out[2] = (p - mic_pos.row(2).transpose()).norm();
        return out;
    };

    SensorMod sensor(h, Eigen::Vector4i{2,0,3,0});
    sensor.pe = Eigen::Matrix3d::Identity() * 1e-4;

    // ---------------------------------------------------
    // Build synthetic measurements
    // ---------------------------------------------------
    Eigen::VectorXd t(1);
    t << 0.0;

    Eigen::MatrixXd X(2,1);
    X.col(0) = source;

    Sig y = sensor.simulate(t, &X);

    // ---------------------------------------------------
    // Explore a grid in the XY plane
    // ---------------------------------------------------
    int N = 41;
    Eigen::MatrixXd grid(N*N, 2);
    int idx = 0;
    for(int i=0; i<N; ++i)
        for(int j=0; j<N; ++j)
        {
            grid(idx,0) = -1.0 + 2.0*i/(N-1);  // x ∈ [-1, +1]
            grid(idx,1) = -1.0 + 2.0*j/(N-1);  // y ∈ [-1, +1]
            idx++;
        }

    Eigen::VectorXd L = sensor.likelihood_function(y, grid);
    int best;
    L.maxCoeff(&best);

    Eigen::Vector2d est = grid.row(best);

    // ---------------------------------------------------
    // Verify ML estimate
    // ---------------------------------------------------
    EXPECT_NEAR(est[0], source[0], 0.1);
    EXPECT_NEAR(est[1], source[1], 0.1);
    
    
    // ---------------------------------------------------
    // 4) Compute CRLB (using the existing crlb function)
    // ---------------------------------------------------

    // We use the SensorMod configuration (sensor) and measurement input (y)
    // that were already created in the ML localization step.
    
    // 1. Create a temporary Sig object (y_crlb) based on the simulated y.
    // We use 'source' as the "true" position to center the CRLB calculation
    // (the Jacobian is evaluated at x0 = source).
    Sig y_crlb = y; 
    
    // Set the true position (source) into Sig.x.col(0).W
    // This is important because the crlb function uses y->x.col(0)
    // as the reference point (x0) if present.
    Eigen::MatrixXd X_true(2, 1);
    X_true.col(0) = source;
    y_crlb.x = X_true;
    
    // 2. Call the main CRLB function
    Sig cr_result = crlb(sensor, &y_crlb); 

    // 3. Populate CRLBData
    CRLBData crlb_data;
    if (!cr_result.Px.empty()) {
        crlb_data.Px = cr_result.Px[0].topLeftCorner(2, 2);
    } else {
        crlb_data.Px = Eigen::Matrix2d::Identity();
    }
    // NOTE: Assumes CRLB function returns Px with correct size,
    // but we take the top-left 2x2 block to be safe, as in the original code.

    // ---------------------------------------------------
    // 5) Save data
    // ---------------------------------------------------
    // ---------------------------------------------------
    // 5) Save data
    // ---------------------------------------------------
    save_data_to_plots_flexible(
        est, 
        source, 
        tdoa_est_vec,   // now a vector
        tdoa_true_vec,  // now a vector
        c, 
        mic_pos, 
        crlb_data,      // assuming CRLBData has been computed/definedW
        false, 
        false, 
        true
    );
}

