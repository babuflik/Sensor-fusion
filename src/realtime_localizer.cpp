// src/realtime_localizer.cpp
// Real-time humming localization using audio callbacks

#include "localizer.h"
#include "tdoa_calculator.h"
#include "microphone.h"
#include <array>
#include <iostream>
#include <vector>
#include <chrono>
#include <thread>
#include <atomic>
#include <mutex>

// Simulated real-time audio processing
// In production: Replace with PortAudio, ALSA, or PulseAudio callbacks

class RealtimeLocalizer {
public:
    RealtimeLocalizer(const std::array<Microphone, MIC_COUNT>& mics, 
                      double dt, 
                      double sound_speed,
                      FilterType filter_type = FilterType::EKF1)
        : localizer_(mics, dt, sound_speed, filter_type),
          running_(false)
    {
    }

    // Start real-time processing
    void start() {
        running_ = true;
        process_thread_ = std::thread(&RealtimeLocalizer::processingLoop, this);
    }

    // Stop real-time processing
    void stop() {
        running_ = false;
        if (process_thread_.joinable()) {
            process_thread_.join();
        }
    }

    // Get latest position estimate (thread-safe)
    std::array<double, 3> getLatestPosition() {
        std::lock_guard<std::mutex> lock(mutex_);
        return latest_position_;
    }

    // Get latest covariance (uncertainty)
    Eigen::Matrix<double, 6, 6> getLatestCovariance() {
        std::lock_guard<std::mutex> lock(mutex_);
        return latest_covariance_;
    }

private:
    Localizer localizer_;
    std::thread process_thread_;
    std::atomic<bool> running_;
    std::mutex mutex_;
    
    std::array<double, 3> latest_position_{0.0, 0.0, 0.0};
    Eigen::Matrix<double, 6, 6> latest_covariance_;

    // Main processing loop - runs continuously
    void processingLoop() {
        const double buffer_duration = Microphone::BUFFER_SIZE / SAMPLE_RATE; // ~23ms for 1024 samples
        const auto sleep_duration = std::chrono::duration<double>(buffer_duration);

        while (running_) {
            // 1. Capture audio from all microphones
            std::array<std::vector<double>, MIC_COUNT> audioBuffers;
            captureAudioBuffers(audioBuffers);

            // 2. Run localization (EKF/UKF update)
            auto position = localizer_.locateSource(audioBuffers);

            // 3. Update shared state (thread-safe)
            {
                std::lock_guard<std::mutex> lock(mutex_);
                latest_position_ = position;
                latest_covariance_ = localizer_.getCovariance();
            }

            // 4. Print real-time update
            std::cout << "\r[REALTIME] Position: (" 
                      << position[0] << ", " 
                      << position[1] << ", " 
                      << position[2] << ")   " << std::flush;

            // 5. Sleep until next buffer is ready
            std::this_thread::sleep_for(sleep_duration);
        }
        std::cout << std::endl; // Newline after final update
    }

    // Capture audio from all microphones
    // TODO: Replace with actual audio capture (PortAudio, ALSA, etc.)
    void captureAudioBuffers(std::array<std::vector<double>, MIC_COUNT>& buffers) {
        // PLACEHOLDER: Generate simulated audio with slight TDOA
        // In production, this would read from actual microphones
        
        for (size_t i = 0; i < MIC_COUNT; ++i) {
            buffers[i].resize(Microphone::BUFFER_SIZE);
            
            // Simulate humming sound at ~200 Hz with noise
            for (size_t j = 0; j < Microphone::BUFFER_SIZE; ++j) {
                double t = j / SAMPLE_RATE;
                
                // Simple sinusoid with small delay per mic
                double delay = i * 0.0001; // 0.1ms delay between mics
                double signal = 0.5 * std::sin(2.0 * M_PI * 200.0 * (t - delay));
                
                // Add noise
                double noise = 0.01 * (rand() / double(RAND_MAX) - 0.5);
                
                buffers[i][j] = signal + noise;
            }
        }
    }
};

// Example usage
int main() {
    std::cout << "=== Real-Time Humming Localization ===" << std::endl;

    // Setup microphone array (square configuration)
    std::array<Microphone, MIC_COUNT> microphones = {
        Microphone(0.0, 0.0, 0.0),      // M1 (origin)
        Microphone(0.20, 0.0, 0.0),     // M2 (20 cm right)
        Microphone(0.20, 0.20, 0.0),    // M3 (20 cm diagonal)
        Microphone(0.0, 0.20, 0.0)      // M4 (20 cm up)
    };

    // Create real-time localizer
    RealtimeLocalizer rtLocalizer(microphones, 0.01, 343.0, FilterType::EKF1);

    // Start processing
    std::cout << "Starting real-time localization..." << std::endl;
    rtLocalizer.start();

    // Run for 10 seconds
    std::this_thread::sleep_for(std::chrono::seconds(10));

    // Stop processing
    std::cout << "\nStopping..." << std::endl;
    rtLocalizer.stop();

    // Final position
    auto final_pos = rtLocalizer.getLatestPosition();
    std::cout << "\nFinal estimated position: (" 
              << final_pos[0] << ", " 
              << final_pos[1] << ", " 
              << final_pos[2] << ")" << std::endl;

    return 0;
}
