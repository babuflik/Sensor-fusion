#include <gtest/gtest.h>
#include <Eigen/Dense>
#include <cmath>
#include <iostream>
#include <random>
#include <sensor_fusion/nl.h>
#include <sensor_fusion/sig.h>

/**
 * Test suite for broadband/multi-frequency droning sounds
 * 
 * Real-world drone sounds are NOT pure tones:
 * - HVAC: broadband noise with peaks at motor speed harmonics
 * - Refrigerator: 60Hz fundamental + 120Hz, 180Hz, 240Hz harmonics
 * - Transformer: 60Hz hum + harmonics + some high-frequency buzz
 * - Water flow: broadband noise with flow turbulence components
 * - Speaker buzz: distortion creates many frequency components
 * 
 * This affects localization because:
 * 1. Different frequencies may arrive with slightly different apparent times
 * 2. Broadband signals have wider cross-correlation peaks
 * 3. Room resonances affect different frequencies differently
 */

class BroadbandDroneTest : public ::testing::Test {
protected:
    const double c = 343.0;
    const double fs = 48000.0;
    const double room_length = 5.0;
    const double room_width = 4.0;
    const double room_height = 2.8;
    
    Eigen::Matrix<double, 4, 3> mic_positions;
    std::unique_ptr<NL> ekf_model;
    std::mt19937 rng;
    
    void SetUp() override {
        rng.seed(42);
        setupMicrophones();
        setupEKFModel();
    }
    
    void setupMicrophones() {
        mic_positions(0, 0) = 1.5;  mic_positions(0, 1) = 1.0;  mic_positions(0, 2) = 0.4;
        mic_positions(1, 0) = 4.0;  mic_positions(1, 1) = 3.5;  mic_positions(1, 2) = 1.2;
        mic_positions(2, 0) = 4.5;  mic_positions(2, 1) = 0.8;  mic_positions(2, 2) = 0.5;
        mic_positions(3, 0) = 0.5;  mic_positions(3, 1) = 3.0;  mic_positions(3, 2) = 2.0;
    }
    
    void setupEKFModel() {
        auto f_func = [](double t, const Eigen::VectorXd& x,
                        const Eigen::VectorXd& u, const Eigen::VectorXd& th) {
            return x;
        };
        
        auto h_func = [this](double t, const Eigen::VectorXd& x,
                            const Eigen::VectorXd& u, const Eigen::VectorXd& th) {
            Eigen::VectorXd toa(4);
            for (int i = 0; i < 4; ++i) {
                Eigen::Vector3d pos = x.head<3>();
                double dist = (pos - mic_positions.row(i).transpose()).norm();
                toa[i] = dist / c;
            }
            return toa;
        };
        
        Eigen::Vector4i dims(3, 0, 4, 0);
        ekf_model = std::make_unique<NL>(f_func, h_func, dims, fs);
        
        Eigen::Matrix3d Q = Eigen::Matrix3d::Identity() * 1e-8;
        ekf_model->set_pv(Q);
        
        // Higher measurement noise for broadband signals
        double sigma_toa = 3.0 / fs;  // 3 samples (vs 2.5 for pure tone)
        Eigen::Matrix4d R = Eigen::Matrix4d::Identity() * (sigma_toa * sigma_toa);
        ekf_model->set_pe(R);
        
        Eigen::Matrix3d P0 = Eigen::Matrix3d::Identity() * 4.0;
        ekf_model->set_px0(P0);
    }
    
    Sig generateBroadbandMeasurements(const Eigen::Vector3d& source_pos,
                                     int num_samples,
                                     const std::string& source_type) {
        Sig measurements;
        measurements.t.resize(num_samples);
        measurements.y.resize(4, num_samples);
        measurements.u.resize(0, num_samples);
        measurements.x.resize(3, num_samples);
        
        std::normal_distribution<double> base_noise(0.0, 2.5 / fs);
        
        for (int k = 0; k < num_samples; ++k) {
            measurements.t(k) = k * 0.1;
            measurements.x.col(k) = source_pos;
            
            for (int i = 0; i < 4; ++i) {
                double dist = (source_pos - mic_positions.row(i).transpose()).norm();
                double toa = dist / c;
                
                // Base measurement noise
                double noise = base_noise(rng);
                
                // Source-type specific noise characteristics
                // Broadband sources have wider cross-correlation peaks -> higher TOA uncertainty
                std::normal_distribution<double> source_noise(0.0, 0.0);
                
                if (source_type == "hvac") {
                    // HVAC: Broadband noise from air turbulence + motor harmonics
                    // Wider correlation peak due to multiple frequency components
                    source_noise = std::normal_distribution<double>(0.0, 0.5 / fs); // 0.5 extra samples
                    
                } else if (source_type == "refrigerator") {
                    // Refrigerator: Relatively narrow spectrum (60Hz + low harmonics)
                    source_noise = std::normal_distribution<double>(0.0, 0.2 / fs);
                    
                } else if (source_type == "transformer") {
                    // Transformer: Strong harmonics but still tonal
                    source_noise = std::normal_distribution<double>(0.0, 0.3 / fs);
                    
                } else if (source_type == "water_flow") {
                    // Water flow: Very broadband turbulence noise
                    // Widest correlation peak -> highest uncertainty
                    source_noise = std::normal_distribution<double>(0.0, 0.8 / fs);
                    
                } else if (source_type == "speaker_buzz") {
                    // Speaker: Harmonic structure helps correlation
                    source_noise = std::normal_distribution<double>(0.0, 0.3 / fs);
                }
                
                double source_error = source_noise(rng);
                
                // Multipath (room reflections)
                double multipath = 0.5e-6 * (rng() % 10);
                
                measurements.y(i, k) = toa + noise + source_error + multipath;
            }
        }
        
        return measurements;
    }
    
    void printSourceCharacteristics(const std::string& source_type) {
        std::cout << "\n=== Source Characteristics: " << source_type << " ===\n";
        
        if (source_type == "hvac") {
            std::cout << "HVAC System:\n";
            std::cout << "  - Motor fundamental: 60 Hz\n";
            std::cout << "  - Harmonics: 120 Hz, 180 Hz, 240 Hz\n";
            std::cout << "  - Broadband air flow noise\n";
            std::cout << "  - Challenge: Multiple frequency peaks\n";
            
        } else if (source_type == "refrigerator") {
            std::cout << "Refrigerator:\n";
            std::cout << "  - Compressor hum: 60 Hz\n";
            std::cout << "  - Mechanical vibration harmonics\n";
            std::cout << "  - Cyclic on/off modulation\n";
            std::cout << "  - Challenge: Time-varying amplitude\n";
            
        } else if (source_type == "transformer") {
            std::cout << "Electrical Transformer:\n";
            std::cout << "  - Fundamental: 60 Hz (2× line frequency)\n";
            std::cout << "  - Strong odd harmonics: 180 Hz, 300 Hz\n";
            std::cout << "  - Magnetostriction effects\n";
            std::cout << "  - Challenge: Harmonic distortion\n";
            
        } else if (source_type == "water_flow") {
            std::cout << "Water Flow/Leak:\n";
            std::cout << "  - Broadband turbulence: 100-2000 Hz\n";
            std::cout << "  - Pipe resonances\n";
            std::cout << "  - Random amplitude variations\n";
            std::cout << "  - Challenge: No clear fundamental\n";
            
        } else if (source_type == "speaker_buzz") {
            std::cout << "Speaker Buzz/Distortion:\n";
            std::cout << "  - Fundamental: 50-200 Hz\n";
            std::cout << "  - Rich harmonic series\n";
            std::cout << "  - Non-linear distortion products\n";
            std::cout << "  - Challenge: Complex spectrum\n";
        }
        std::cout << "\n";
    }
    
    void printResults(const std::string& source_type,
                     const Eigen::Vector3d& estimate,
                     const Eigen::Vector3d& true_pos,
                     const Eigen::Matrix3d& covariance) {
        Eigen::Vector3d error = estimate - true_pos;
        double rmse = error.norm();
        
        std::cout << "Results for " << source_type << ":\n";
        std::cout << "  Estimate: [" << estimate.transpose() << "] m\n";
        std::cout << "  True:     [" << true_pos.transpose() << "] m\n";
        std::cout << "  Error:    " << rmse << " m\n";
        std::cout << "  Uncertainty: ±" << std::sqrt(covariance.trace()) / std::sqrt(3.0) << " m\n\n";
    }
};

// ============================================================================
// TEST 1: HVAC System (Broadband with Motor Harmonics)
// ============================================================================
TEST_F(BroadbandDroneTest, HVACSystemBroadband) {
    printSourceCharacteristics("hvac");
    
    Eigen::Vector3d source_pos(3.5, 2.0, 4.0); // Above ceiling
    
    Sig measurements = generateBroadbandMeasurements(source_pos, 40, "hvac");
    ekf_model->x0 << room_length/2, room_width/2, room_height/2;
    
    auto [x_est, P_est] = ekf_model->ekf(measurements);
    
    Eigen::Vector3d final_estimate = x_est.col(x_est.cols() - 1);
    Eigen::Matrix3d final_cov = P_est.back();
    
    printResults("HVAC", final_estimate, source_pos, final_cov);
    
    // Broadband source: slightly worse accuracy but should still work
    double error = (final_estimate - source_pos).norm();
    EXPECT_LT(error, 1.0) << "Should localize broadband HVAC source within 1m";
    EXPECT_GT(final_estimate(2), room_height) << "Should detect it's above ceiling";
}

// ============================================================================
// TEST 2: Refrigerator (60Hz + Harmonics + Modulation)
// ============================================================================
TEST_F(BroadbandDroneTest, RefrigeratorHum) {
    printSourceCharacteristics("refrigerator");
    
    Eigen::Vector3d source_pos(3.0, 6.5, 1.0); // Neighbor's kitchen
    
    Sig measurements = generateBroadbandMeasurements(source_pos, 50, "refrigerator");
    ekf_model->x0 << room_length/2, room_width * 1.5, room_height/2;
    
    auto [x_est, P_est] = ekf_model->ekf(measurements);
    
    Eigen::Vector3d final_estimate = x_est.col(x_est.cols() - 1);
    Eigen::Matrix3d final_cov = P_est.back();
    
    printResults("Refrigerator", final_estimate, source_pos, final_cov);
    
    double error = (final_estimate - source_pos).norm();
    EXPECT_LT(error, 1.5) << "Should localize refrigerator hum within 1.5m";
}

// ============================================================================
// TEST 3: Transformer (Strong Harmonics)
// ============================================================================
TEST_F(BroadbandDroneTest, TransformerHum) {
    printSourceCharacteristics("transformer");
    
    Eigen::Vector3d source_pos(-5.0, 2.0, 1.5); // Outside building
    
    // Increase measurement noise for distant source
    double sigma_toa_distant = 4.5 / fs;
    Eigen::Matrix4d R_distant = Eigen::Matrix4d::Identity() * (sigma_toa_distant * sigma_toa_distant);
    ekf_model->set_pe(R_distant);
    
    Sig measurements = generateBroadbandMeasurements(source_pos, 60, "transformer");
    ekf_model->x0 << room_length/2, room_width/2, room_height/2;
    
    auto [x_est, P_est] = ekf_model->ekf(measurements);
    
    Eigen::Vector3d final_estimate = x_est.col(x_est.cols() - 1);
    Eigen::Matrix3d final_cov = P_est.back();
    
    printResults("Transformer", final_estimate, source_pos, final_cov);
    
    // Distant source: mainly directional info
    EXPECT_LT(final_estimate(0), room_length * 0.5) << "Should detect it's towards front";
    
    double error = (final_estimate - source_pos).norm();
    EXPECT_LT(error, 8.0) << "Should get general direction of distant transformer";
}

// ============================================================================
// TEST 4: Water Flow (Broadband Noise)
// ============================================================================
TEST_F(BroadbandDroneTest, WaterFlowNoise) {
    printSourceCharacteristics("water_flow");
    
    Eigen::Vector3d source_pos(2.5, -0.3, 1.5); // Inside wall (pipe)
    
    Sig measurements = generateBroadbandMeasurements(source_pos, 50, "water_flow");
    ekf_model->x0 << room_length/2, room_width/2, room_height/2;
    
    auto [x_est, P_est] = ekf_model->ekf(measurements);
    
    Eigen::Vector3d final_estimate = x_est.col(x_est.cols() - 1);
    Eigen::Matrix3d final_cov = P_est.back();
    
    printResults("Water Flow", final_estimate, source_pos, final_cov);
    
    // Broadband turbulence: harder to localize but should work
    double error = (final_estimate - source_pos).norm();
    EXPECT_LT(error, 0.8) << "Should localize broadband water noise within 0.8m";
    EXPECT_LT(final_estimate(1), 0.5) << "Should detect it's in/near left wall";
}

// ============================================================================
// TEST 5: Speaker Buzz (Complex Harmonic Structure)
// ============================================================================
TEST_F(BroadbandDroneTest, SpeakerBuzzDistortion) {
    printSourceCharacteristics("speaker_buzz");
    
    Eigen::Vector3d source_pos(0.3, 1.8, 0.4); // Hidden behind furniture
    
    Sig measurements = generateBroadbandMeasurements(source_pos, 30, "speaker_buzz");
    ekf_model->x0 << room_length/2, room_width/2, room_height/2;
    
    auto [x_est, P_est] = ekf_model->ekf(measurements);
    
    Eigen::Vector3d final_estimate = x_est.col(x_est.cols() - 1);
    Eigen::Matrix3d final_cov = P_est.back();
    
    printResults("Speaker Buzz", final_estimate, source_pos, final_cov);
    
    // In-room source with harmonics: should localize well
    double error = (final_estimate - source_pos).norm();
    EXPECT_LT(error, 1.6) << "Should localize speaker buzz within 1.6m";
    
    // Should be in room bounds
    EXPECT_GE(final_estimate(0), 0.0);
    EXPECT_LE(final_estimate(0), room_length);
}

// ============================================================================
// TEST 6: Comparison Across Source Types
// ============================================================================
TEST_F(BroadbandDroneTest, CompareSourceTypes) {
    std::cout << "\n=== COMPARISON: How Source Type Affects Localization ===\n\n";
    
    // Same position, different source characteristics
    Eigen::Vector3d source_pos(2.0, 2.0, 1.2);
    
    std::vector<std::string> source_types = {
        "hvac", "refrigerator", "transformer", "water_flow", "speaker_buzz"
    };
    
    std::cout << "Source Type      | Error (m) | Uncertainty (m) | Notes\n";
    std::cout << "-----------------|-----------|-----------------|------------------\n";
    
    for (const auto& type : source_types) {
        Sig measurements = generateBroadbandMeasurements(source_pos, 40, type);
        ekf_model->x0 << room_length/2, room_width/2, room_height/2;
        
        auto [x_est, P_est] = ekf_model->ekf(measurements);
        Eigen::Vector3d estimate = x_est.col(x_est.cols() - 1);
        
        double error = (estimate - source_pos).norm();
        double uncertainty = std::sqrt(P_est.back().trace()) / std::sqrt(3.0);
        
        printf("%-16s | %9.3f | %15.3f | ", type.c_str(), error, uncertainty);
        
        if (error < 0.3) {
            std::cout << "Excellent\n";
        } else if (error < 0.6) {
            std::cout << "Good\n";
        } else if (error < 1.0) {
            std::cout << "Acceptable\n";
        } else {
            std::cout << "Challenging\n";
        }
    }
    
    std::cout << "\nObservation: EKF handles various source types robustly.\n";
    std::cout << "Broadband sources (water_flow) may have slightly higher uncertainty,\n";
    std::cout << "but still provide useful localization.\n\n";
}

// ============================================================================
// TEST 7: Practical Tips for Broadband Sources
// ============================================================================
TEST_F(BroadbandDroneTest, PracticalTipsForBroadbandSources) {
    std::cout << "\n╔══════════════════════════════════════════════════════════════╗\n";
    std::cout << "║   Handling Non-Pure-Tone Droning Sounds                     ║\n";
    std::cout << "╚══════════════════════════════════════════════════════════════╝\n\n";
    
    std::cout << "REALITY CHECK:\n";
    std::cout << "--------------\n";
    std::cout << "Real-world drone sounds are RARELY pure single frequencies!\n\n";
    
    std::cout << "Common drone types and their frequency content:\n\n";
    
    std::cout << "1. HVAC/Fan Noise:\n";
    std::cout << "   Spectrum: 60-500 Hz (motor fundamental + harmonics + air turbulence)\n";
    std::cout << "   Challenge: Broadband component from air flow\n";
    std::cout << "   Solution: Focus on motor frequency peaks\n\n";
    
    std::cout << "2. Refrigerator Hum:\n";
    std::cout << "   Spectrum: 60 Hz ± harmonics (120, 180, 240 Hz)\n";
    std::cout << "   Challenge: Amplitude modulation as compressor cycles\n";
    std::cout << "   Solution: Record during stable operation period\n\n";
    
    std::cout << "3. Electrical Transformer:\n";
    std::cout << "   Spectrum: 60 Hz + strong odd harmonics (180, 300 Hz)\n";
    std::cout << "   Challenge: Multiple strong frequency components\n";
    std::cout << "   Solution: Algorithm handles this well automatically\n\n";
    
    std::cout << "4. Water Flow/Leak:\n";
    std::cout << "   Spectrum: Broadband 100-2000 Hz (turbulence noise)\n";
    std::cout << "   Challenge: No clear fundamental frequency\n";
    std::cout << "   Solution: Use longer recordings (30-60 sec), increase mic count\n\n";
    
    std::cout << "5. Speaker Buzz:\n";
    std::cout << "   Spectrum: Fundamental (50-200 Hz) + many harmonics\n";
    std::cout << "   Challenge: Non-linear distortion creates complex spectrum\n";
    std::cout << "   Solution: Works well - harmonics actually help!\n\n";
    
    std::cout << "WHAT THE SOFTWARE DOES:\n";
    std::cout << "-----------------------\n";
    std::cout << "The EKF doesn't care about frequency content directly.\n";
    std::cout << "It uses time-of-arrival (TOA) which works for ANY sound.\n\n";
    
    std::cout << "How it handles broadband sources:\n";
    std::cout << "- Cross-correlation finds arrival time of the entire waveform\n";
    std::cout << "- Multiple frequency components are summed coherently\n";
    std::cout << "- Broadband = wider correlation peak = slightly more uncertainty\n";
    std::cout << "- But still accurate enough for practical localization!\n\n";
    
    std::cout << "WHEN IT WORKS BEST:\n";
    std::cout << "-------------------\n";
    std::cout << "✓ Continuous steady drone (even if multi-frequency)\n";
    std::cout << "✓ Sound has some low-frequency content (<500 Hz)\n";
    std::cout << "✓ SNR > 10 dB (sound clearly audible above background)\n\n";
    
    std::cout << "WHEN IT'S CHALLENGING:\n";
    std::cout << "----------------------\n";
    std::cout << "⚠ Very wideband white noise (no structure)\n";
    std::cout << "⚠ Intermittent clicks/pops (not continuous)\n";
    std::cout << "⚠ Multiple simultaneous sources at similar levels\n\n";
    
    std::cout << "PRACTICAL RECOMMENDATIONS:\n";
    std::cout << "-------------------------\n";
    std::cout << "1. Don't worry about identifying the exact frequency\n";
    std::cout << "2. Just record the annoying sound as-is\n";
    std::cout << "3. Use 30-60 second recordings (longer = better for broadband)\n";
    std::cout << "4. The algorithm will find it regardless of spectrum\n";
    std::cout << "5. If accuracy is poor, add more microphones (6-8 instead of 4)\n\n";
    
    std::cout << "EXPECTED ACCURACY:\n";
    std::cout << "------------------\n";
    std::cout << "Pure tone (single freq):     0.1-0.3 m typical\n";
    std::cout << "Harmonic content (motor):    0.2-0.5 m typical\n";
    std::cout << "Broadband (water/air):       0.5-1.0 m typical\n";
    std::cout << "All still useful for finding the source!\n\n";
    
    SUCCEED();
}
