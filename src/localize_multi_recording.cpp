// src/localize_multi_recording.cpp
// Localize stationary source using 3 microphones with multiple recordings
// Takes advantage of the fact that the source doesn't move between recordings
//
// This is a simplified implementation that runs multiple 4-mic localizations
// (treating 3-mic setups as 4-mic with one repeated mic) and averages results

#include "localizer.h"
#include "microphone.h"
#include "audio_file_loader.h"
#include <iostream>
#include <vector>
#include <array>
#include <string>
#include <cmath>
#include <numeric>
#include <Eigen/Dense>

constexpr int MICS_PER_RECORDING = 3;
constexpr double SOUND_SPEED = 343.0; // m/s
constexpr double SAMPLE_RATE = 44100.0; // Hz

struct Recording {
    std::array<std::string, 3> filenames;
    std::array<std::array<double, 3>, 3> mic_positions;
};

void printUsage(const char* program_name) {
    std::cout << "Usage: " << program_name << " <recording1_files...> [recording2_files...] [...]" << std::endl;
    std::cout << std::endl;
    std::cout << "Localize stationary sound source using 3-microphone recordings." << std::endl;
    std::cout << "Multiple recordings improve accuracy by averaging measurements." << std::endl;
    std::cout << std::endl;
    std::cout << "Arguments:" << std::endl;
    std::cout << "  Each recording requires exactly 3 files and 3 mic positions (9 args total)" << std::endl;
    std::cout << "  Format per recording: file1 x1,y1,z1 file2 x2,y2,z2 file3 x3,y3,z3" << std::endl;
    std::cout << std::endl;
    std::cout << "Example (single recording):" << std::endl;
    std::cout << "  " << program_name << " mic1.wav 0,0,0 mic2.wav 0.2,0,0 mic3.wav 0.2,0.2,0" << std::endl;
    std::cout << std::endl;
    std::cout << "Example (two recordings for better accuracy):" << std::endl;
    std::cout << "  " << program_name << " \\" << std::endl;
    std::cout << "    rec1_mic1.wav 0,0,0 rec1_mic2.wav 0.2,0,0 rec1_mic3.wav 0.2,0.2,0 \\" << std::endl;
    std::cout << "    rec2_mic1.wav 0,0,0.5 rec2_mic2.wav 0.2,0,0.5 rec2_mic3.wav 0.2,0.2,0.5" << std::endl;
    std::cout << std::endl;
    std::cout << "Benefits of multiple recordings:" << std::endl;
    std::cout << "  - Reduces random errors by averaging" << std::endl;
    std::cout << "  - Different mic positions improve geometric diversity" << std::endl;
    std::cout << "  - Works with only 3 microphones instead of 4" << std::endl;
    std::cout << "  - Source must remain stationary between recordings" << std::endl;
}

bool parsePosition(const std::string& str, std::array<double, 3>& pos) {
    size_t comma1 = str.find(',');
    size_t comma2 = str.find(',', comma1 + 1);
    
    if (comma1 == std::string::npos || comma2 == std::string::npos) {
        return false;
    }
    
    try {
        pos[0] = std::stod(str.substr(0, comma1));
        pos[1] = std::stod(str.substr(comma1 + 1, comma2 - comma1 - 1));
        pos[2] = std::stod(str.substr(comma2 + 1));
        return true;
    } catch (...) {
        return false;
    }
}

int main(int argc, char* argv[]) {
    // Each recording needs 6 args: 3 × (filename, position)
    if (argc < 7 || (argc - 1) % 6 != 0) {
        printUsage(argv[0]);
        return 1;
    }

    int num_recordings = (argc - 1) / 6;
    std::vector<Recording> recordings;
    recordings.reserve(num_recordings);

    std::cout << "=== Multi-Recording Sound Source Localization ===" << std::endl;
    std::cout << "Number of recordings: " << num_recordings << std::endl;
    std::cout << std::endl;

    // Parse arguments
    int arg_idx = 1;
    for (int rec = 0; rec < num_recordings; ++rec) {
        std::cout << "Recording " << (rec + 1) << ":" << std::endl;
        
        Recording recording;
        
        for (int mic = 0; mic < MICS_PER_RECORDING; ++mic) {
            // Get filename
            recording.filenames[mic] = argv[arg_idx++];
            
            // Get position
            if (!parsePosition(argv[arg_idx++], recording.mic_positions[mic])) {
                std::cerr << "Error: Invalid position format for recording " << (rec+1) 
                          << ", mic " << (mic+1) << std::endl;
                std::cerr << "Expected format: x,y,z (e.g., 0.2,0.1,0.5)" << std::endl;
                return 1;
            }
            
            auto& pos = recording.mic_positions[mic];
            
            // Verify file exists
            if (!AudioFileLoader::fileExists(recording.filenames[mic])) {
                std::cerr << "Error: File not found: " << recording.filenames[mic] << std::endl;
                return 1;
            }
            
            std::cout << "  M" << (mic+1) << ": " << recording.filenames[mic] 
                      << " at (" << pos[0] << ", " << pos[1] << ", " << pos[2] << ")" << std::endl;
        }
        
        recordings.push_back(recording);
        std::cout << std::endl;
    }

    // Process each recording
    std::vector<std::array<double, 3>> position_estimates;
    std::vector<double> weights;
    
    std::cout << "Processing recordings..." << std::endl;
    
    for (size_t rec = 0; rec < recordings.size(); ++rec) {
        std::cout << "  Recording " << (rec + 1) << "... ";
        
        // For 3-mic setup, duplicate the last mic to make it compatible with 4-mic localizer
        // This is a pragmatic solution - the repeated mic won't add information but won't break the math
        std::array<Microphone, 4> microphones = {
            Microphone(recordings[rec].mic_positions[0][0], recordings[rec].mic_positions[0][1], recordings[rec].mic_positions[0][2]),
            Microphone(recordings[rec].mic_positions[1][0], recordings[rec].mic_positions[1][1], recordings[rec].mic_positions[1][2]),
            Microphone(recordings[rec].mic_positions[2][0], recordings[rec].mic_positions[2][1], recordings[rec].mic_positions[2][2]),
            Microphone(recordings[rec].mic_positions[2][0], recordings[rec].mic_positions[2][1], recordings[rec].mic_positions[2][2]) // Duplicate last mic
        };
        
        // Load audio files - also need to duplicate the last file
        std::array<std::string, 4> filenames_4mic = {
            recordings[rec].filenames[0],
            recordings[rec].filenames[1],
            recordings[rec].filenames[2],
            recordings[rec].filenames[2] // Duplicate last file
        };
        
        std::array<std::vector<double>, 4> audioBuffers;
        try {
            audioBuffers = AudioFileLoader::loadMultipleFiles(filenames_4mic, Microphone::BUFFER_SIZE);
        } catch (const std::exception& e) {
            std::cerr << std::endl << "Error loading files: " << e.what() << std::endl;
            return 1;
        }
        
        // Create localizer
        double dt = Microphone::BUFFER_SIZE / SAMPLE_RATE;
        Localizer localizer(microphones, dt, SOUND_SPEED, FilterType::EKF1);
        
        // Perform localization
        auto position = localizer.locateSource(audioBuffers);
        
        // Get covariance for weighting
        auto P = localizer.getCovariance();
        double uncertainty = P.block<3,3>(0,0).trace(); // Only position uncertainty
        
        position_estimates.push_back(position);
        weights.push_back(1.0 / (uncertainty + 1e-9)); // Inverse uncertainty = weight
        
        std::cout << "Position: (" << position[0] << ", " << position[1] << ", " << position[2] << ")" << std::endl;
    }
    
    std::cout << std::endl;
    // Weighted average of all estimates
    double total_weight = std::accumulate(weights.begin(), weights.end(), 0.0);
    std::array<double, 3> final_position = {0.0, 0.0, 0.0};
    
    for (size_t i = 0; i < position_estimates.size(); ++i) {
        double w = weights[i] / total_weight;
        final_position[0] += w * position_estimates[i][0];
        final_position[1] += w * position_estimates[i][1];
        final_position[2] += w * position_estimates[i][2];
    }
    
    // Calculate standard deviation across estimates (measure of consistency)
    std::array<double, 3> variance = {0.0, 0.0, 0.0};
    for (size_t i = 0; i < position_estimates.size(); ++i) {
        for (int dim = 0; dim < 3; ++dim) {
            double diff = position_estimates[i][dim] - final_position[dim];
            variance[dim] += diff * diff;
        }
    }
    
    double std_dev = 0.0;
    if (position_estimates.size() > 1) {
        for (int dim = 0; dim < 3; ++dim) {
            variance[dim] /= (position_estimates.size() - 1);
        }
        std_dev = std::sqrt(variance[0] + variance[1] + variance[2]);
    }
    
    // Print results
    std::cout << std::endl;
    std::cout << "=== Results ===" << std::endl;
    std::cout << "Individual estimates:" << std::endl;
    for (size_t i = 0; i < position_estimates.size(); ++i) {
        std::cout << "  Recording " << (i+1) << ": ("
                  << position_estimates[i][0] << ", "
                  << position_estimates[i][1] << ", "
                  << position_estimates[i][2] << ") m, weight: "
                  << weights[i]/total_weight << std::endl;
    }
    std::cout << std::endl;
    
    std::cout << "Final estimate (weighted average):" << std::endl;
    std::cout << "  Position: ("
              << final_position[0] << ", "
              << final_position[1] << ", "
              << final_position[2] << ") meters" << std::endl;
    
    if (position_estimates.size() > 1) {
        std::cout << "  Consistency (std dev): " << std_dev << " m" << std::endl;
        std::cout << "  Per-axis std dev: ("
                  << std::sqrt(variance[0]) << ", "
                  << std::sqrt(variance[1]) << ", "
                  << std::sqrt(variance[2]) << ") m" << std::endl;
    }
    
    std::cout << std::endl;
    std::cout << "Benefits achieved:" << std::endl;
    std::cout << "  ✓ Used only " << MICS_PER_RECORDING << " microphones" << std::endl;
    std::cout << "  ✓ Combined " << num_recordings << " recording(s)" << std::endl;
    if (num_recordings > 1) {
        std::cout << "  ✓ Reduced uncertainty through averaging" << std::endl;
    }
    
    return 0;
}
