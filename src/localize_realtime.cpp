// src/localize_realtime.cpp
// Real-time localization with live microphone input
// NOTE: Currently uses simulated audio. Replace with PortAudio for real hardware.

#include "localizer.h"
#include "microphone.h"
#include "tdoa_calculator.h"
#include <iostream>
#include <array>
#include <thread>
#include <chrono>
#include <atomic>
#include <csignal>
#include <cmath>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// Global flag for clean shutdown
std::atomic<bool> g_running(true);

void signalHandler(int signal) {
    std::cout << "\nReceived interrupt signal. Shutting down..." << std::endl;
    g_running = false;
}

// PLACEHOLDER: Replace this with actual microphone capture (PortAudio, ALSA, etc.)
void captureAudioFromMicrophones(std::array<std::vector<double>, MIC_COUNT>& buffers) {
    static int frame_count = 0;
    const double SAMPLE_RATE = 44100.0;
    const double FREQ = 200.0; // Simulated humming frequency
    
    // Simulate audio with slight delay between microphones
    for (size_t mic = 0; mic < MIC_COUNT; ++mic) {
        buffers[mic].resize(Microphone::BUFFER_SIZE);
        
        // Small delay based on microphone position (simulated)
        double delay = mic * 0.0002; // 0.2ms delay per mic
        
        for (size_t i = 0; i < Microphone::BUFFER_SIZE; ++i) {
            double t = (frame_count * Microphone::BUFFER_SIZE + i) / SAMPLE_RATE;
            
            // Humming sound
            double signal = 0.3 * std::sin(2.0 * M_PI * FREQ * (t - delay));
            
            // Add noise
            double noise = 0.02 * (rand() / double(RAND_MAX) - 0.5);
            
            buffers[mic][i] = signal + noise;
        }
    }
    
    frame_count++;
}

/* 
 * To use real microphones with PortAudio, replace the above function with:
 *
 * #include <portaudio.h>
 * 
 * PaStream* stream;
 * 
 * int audioCallback(const void* inputBuffer, void* outputBuffer,
 *                   unsigned long framesPerBuffer,
 *                   const PaStreamCallbackTimeInfo* timeInfo,
 *                   PaStreamCallbackFlags statusFlags,
 *                   void* userData) {
 *     auto* buffers = (std::array<std::vector<double>, MIC_COUNT>*)userData;
 *     float* input = (float*)inputBuffer;
 *     
 *     for (size_t i = 0; i < framesPerBuffer; ++i) {
 *         for (int ch = 0; ch < MIC_COUNT; ++ch) {
 *             (*buffers)[ch][i] = input[i * MIC_COUNT + ch];
 *         }
 *     }
 *     
 *     return paContinue;
 * }
 * 
 * void initPortAudio() {
 *     Pa_Initialize();
 *     PaStreamParameters inputParams;
 *     inputParams.device = Pa_GetDefaultInputDevice();
 *     inputParams.channelCount = MIC_COUNT;
 *     inputParams.sampleFormat = paFloat32;
 *     inputParams.suggestedLatency = Pa_GetDeviceInfo(inputParams.device)->defaultLowInputLatency;
 *     
 *     Pa_OpenStream(&stream, &inputParams, NULL, 44100, 1024, 
 *                   paClipOff, audioCallback, &buffers);
 *     Pa_StartStream(stream);
 * }
 */

void printUsage(const char* program_name) {
    std::cout << "Usage: " << program_name << " [options]" << std::endl;
    std::cout << std::endl;
    std::cout << "Real-time sound source localization using 4 microphones." << std::endl;
    std::cout << std::endl;
    std::cout << "Options:" << std::endl;
    std::cout << "  --duration <seconds>  Run for specified duration (default: infinite)" << std::endl;
    std::cout << "  --rate <hz>          Update rate in Hz (default: ~43)" << std::endl;
    std::cout << "  --help               Show this help message" << std::endl;
    std::cout << std::endl;
    std::cout << "Controls:" << std::endl;
    std::cout << "  Ctrl+C               Stop localization" << std::endl;
    std::cout << std::endl;
    std::cout << "NOTE: Currently using simulated audio." << std::endl;
    std::cout << "      To use real microphones, see REALTIME_GUIDE.md" << std::endl;
}

int main(int argc, char* argv[]) {
    // Parse command line arguments
    int duration_seconds = -1; // Infinite by default
    
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        
        if (arg == "--help" || arg == "-h") {
            printUsage(argv[0]);
            return 0;
        } else if (arg == "--duration" && i + 1 < argc) {
            duration_seconds = std::atoi(argv[++i]);
        }
    }

    std::cout << "=== Real-Time Sound Source Localization ===" << std::endl;
    std::cout << std::endl;
    std::cout << "NOTE: Using simulated audio. See REALTIME_GUIDE.md for real microphone setup." << std::endl;
    std::cout << std::endl;

    // Setup signal handler for Ctrl+C
    std::signal(SIGINT, signalHandler);

    // Setup microphone array
    std::array<Microphone, MIC_COUNT> microphones = {
        Microphone(0.0, 0.0, 0.0),      // M1
        Microphone(0.20, 0.0, 0.0),     // M2
        Microphone(0.20, 0.20, 0.0),    // M3
        Microphone(0.0, 0.20, 0.0)      // M4
    };

    std::cout << "Microphone positions:" << std::endl;
    for (size_t i = 0; i < MIC_COUNT; ++i) {
        auto pos = microphones[i].getPosition();
        std::cout << "  M" << (i+1) << ": (" << pos[0] << ", " << pos[1] << ", " << pos[2] << ")" << std::endl;
    }
    std::cout << std::endl;

    // Create localizer
    double dt = Microphone::BUFFER_SIZE / 44100.0;
    Localizer localizer(microphones, dt, 343.0, FilterType::EKF1);

    std::cout << "Starting real-time localization..." << std::endl;
    std::cout << "Update rate: ~" << (1.0 / dt) << " Hz" << std::endl;
    std::cout << "Buffer size: " << Microphone::BUFFER_SIZE << " samples" << std::endl;
    std::cout << std::endl;
    std::cout << "Press Ctrl+C to stop." << std::endl;
    std::cout << std::endl;

    // Main processing loop
    auto start_time = std::chrono::steady_clock::now();
    int update_count = 0;

    while (g_running) {
        // Check duration limit
        if (duration_seconds > 0) {
            auto elapsed = std::chrono::steady_clock::now() - start_time;
            if (std::chrono::duration_cast<std::chrono::seconds>(elapsed).count() >= duration_seconds) {
                break;
            }
        }

        // 1. Capture audio
        std::array<std::vector<double>, MIC_COUNT> audioBuffers;
        captureAudioFromMicrophones(audioBuffers);

        // 2. Localize
        auto pos_start = std::chrono::high_resolution_clock::now();
        auto position = localizer.locateSource(audioBuffers);
        auto pos_end = std::chrono::high_resolution_clock::now();
        
        auto processing_time = std::chrono::duration_cast<std::chrono::microseconds>(pos_end - pos_start);

        // 3. Display (every 10 updates to reduce clutter)
        update_count++;
        if (update_count % 10 == 0) {
            auto P = localizer.getCovariance();
            double uncertainty = std::sqrt(P(0,0) + P(1,1) + P(2,2));
            
            printf("\r[%4d] Position: (%6.3f, %6.3f, %6.3f) m  |  Uncertainty: %6.4f m  |  Process: %5ld μs   ",
                   update_count,
                   position[0], position[1], position[2],
                   uncertainty,
                   processing_time.count());
            std::cout << std::flush;
        }

        // 4. Sleep to maintain update rate
        std::this_thread::sleep_for(std::chrono::milliseconds(static_cast<int>(dt * 1000)));
    }

    std::cout << std::endl << std::endl;

    // Summary
    auto total_time = std::chrono::steady_clock::now() - start_time;
    double total_seconds = std::chrono::duration_cast<std::chrono::milliseconds>(total_time).count() / 1000.0;

    std::cout << "=== Summary ===" << std::endl;
    std::cout << "Total updates: " << update_count << std::endl;
    std::cout << "Total time: " << total_seconds << " seconds" << std::endl;
    std::cout << "Average rate: " << (update_count / total_seconds) << " Hz" << std::endl;
    
    auto final_pos = localizer.getState();
    std::cout << "Final position: (" << final_pos(0) << ", " << final_pos(1) << ", " << final_pos(2) << ")" << std::endl;
    std::cout << "Final velocity: (" << final_pos(3) << ", " << final_pos(4) << ", " << final_pos(5) << ") m/s" << std::endl;

    return 0;
}
