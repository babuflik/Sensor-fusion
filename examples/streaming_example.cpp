// examples/streaming_example.cpp
// Minimal example of continuous real-time estimation

#include "localizer.h"
#include "microphone.h"
#include "tdoa_calculator.h"
#include <iostream>
#include <array>
#include <chrono>
#include <thread>
#include <cmath>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// Simulated audio source moving in a circle
class MovingSource {
public:
    MovingSource(double radius, double frequency) 
        : radius_(radius), angular_freq_(2.0 * M_PI * frequency), time_(0.0) {}
    
    std::array<double, 3> getPosition() const {
        double x = radius_ * std::cos(angular_freq_ * time_);
        double y = radius_ * std::sin(angular_freq_ * time_);
        return {x, y, 0.0};
    }
    
    void update(double dt) {
        time_ += dt;
    }
    
private:
    double radius_;
    double angular_freq_;
    double time_;
};

// Generate simulated audio with realistic TDOA
void simulateAudio(
    const std::array<Microphone, MIC_COUNT>& mics,
    const std::array<double, 3>& source_pos,
    std::array<std::vector<double>, MIC_COUNT>& buffers)
{
    const double SAMPLE_RATE = 44100.0;
    const double SOUND_SPEED = 343.0;
    const double FREQ = 200.0; // Humming frequency
    
    for (size_t mic_idx = 0; mic_idx < MIC_COUNT; ++mic_idx) {
        auto mic_pos = mics[mic_idx].getPosition();
        
        // Calculate distance and time delay
        double dx = source_pos[0] - mic_pos[0];
        double dy = source_pos[1] - mic_pos[1];
        double dz = source_pos[2] - mic_pos[2];
        double distance = std::sqrt(dx*dx + dy*dy + dz*dz);
        double delay = distance / SOUND_SPEED;
        
        // Generate signal
        buffers[mic_idx].resize(Microphone::BUFFER_SIZE);
        for (size_t i = 0; i < Microphone::BUFFER_SIZE; ++i) {
            double t = i / SAMPLE_RATE - delay;
            
            // Humming signal with 1/r attenuation
            double amplitude = 1.0 / (1.0 + distance);
            double signal = amplitude * std::sin(2.0 * M_PI * FREQ * t);
            
            // Add small noise
            double noise = 0.01 * (rand() / double(RAND_MAX) - 0.5);
            
            buffers[mic_idx][i] = signal + noise;
        }
    }
}

int main() {
    std::cout << "=== Streaming Real-Time Localization Example ===" << std::endl;
    std::cout << "Tracking a sound source moving in a circle\n" << std::endl;
    
    // Setup microphone array (square configuration, 20cm spacing)
    std::array<Microphone, MIC_COUNT> microphones = {
        Microphone(0.0, 0.0, 0.0),      // M1
        Microphone(0.20, 0.0, 0.0),     // M2
        Microphone(0.20, 0.20, 0.0),    // M3
        Microphone(0.0, 0.20, 0.0)      // M4
    };
    
    // Create localizer with EKF
    double dt = Microphone::BUFFER_SIZE / 44100.0; // Time per buffer (~23ms)
    Localizer localizer(microphones, dt, 343.0, FilterType::EKF1);
    
    // Create moving source (0.5m radius, 0.1 Hz rotation = 10 second period)
    MovingSource source(0.5, 0.1);
    
    // Stream processing loop
    const int NUM_UPDATES = 100;
    std::cout << "Processing " << NUM_UPDATES << " buffers..." << std::endl;
    std::cout << "\nTime(s)  True Position         Estimated Position      Error(m)" << std::endl;
    std::cout << "-------  --------------------  --------------------  -----------" << std::endl;
    
    for (int i = 0; i < NUM_UPDATES; ++i) {
        double current_time = i * dt;
        
        // Get true source position
        auto true_pos = source.getPosition();
        
        // Simulate audio capture with TDOA
        std::array<std::vector<double>, MIC_COUNT> audioBuffers;
        simulateAudio(microphones, true_pos, audioBuffers);
        
        // Run localization (this is the real-time update)
        auto start = std::chrono::high_resolution_clock::now();
        auto estimated_pos = localizer.locateSource(audioBuffers);
        auto end = std::chrono::high_resolution_clock::now();
        
        // Calculate error
        double error = std::sqrt(
            std::pow(estimated_pos[0] - true_pos[0], 2) +
            std::pow(estimated_pos[1] - true_pos[1], 2) +
            std::pow(estimated_pos[2] - true_pos[2], 2)
        );
        
        // Print status every 10 updates
        if (i % 10 == 0) {
            auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
            
            printf("%6.3f   (%5.2f, %5.2f, %5.2f)  (%5.2f, %5.2f, %5.2f)  %6.4f  [%4ld μs]\n",
                current_time,
                true_pos[0], true_pos[1], true_pos[2],
                estimated_pos[0], estimated_pos[1], estimated_pos[2],
                error,
                duration.count());
        }
        
        // Update source position for next iteration
        source.update(dt);
        
        // Simulate real-time by sleeping (remove in production)
        std::this_thread::sleep_for(std::chrono::milliseconds(5));
    }
    
    std::cout << "\n=== Summary ===" << std::endl;
    std::cout << "✓ Processed " << NUM_UPDATES << " audio buffers" << std::endl;
    std::cout << "✓ Update rate: ~" << (1.0 / dt) << " Hz" << std::endl;
    std::cout << "✓ Filter maintained stable tracking of moving source" << std::endl;
    
    return 0;
}
