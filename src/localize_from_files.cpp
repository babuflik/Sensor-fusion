// src/localize_from_files.cpp
// Process audio from file inputs (.wav or .csv)

#include "localizer.h"
#include "microphone.h"
#include "audio_file_loader.h"
#include "tdoa_calculator.h"
#include <iostream>
#include <array>
#include <string>
#include <cmath>

void printUsage(const char* program_name) {
    std::cout << "Usage: " << program_name << " <mic1_file> <mic2_file> <mic3_file> <mic4_file>" << std::endl;
    std::cout << std::endl;
    std::cout << "Localize sound source from pre-recorded audio files." << std::endl;
    std::cout << std::endl;
    std::cout << "Arguments:" << std::endl;
    std::cout << "  mic1_file  Audio file for microphone 1 (.wav or .csv)" << std::endl;
    std::cout << "  mic2_file  Audio file for microphone 2 (.wav or .csv)" << std::endl;
    std::cout << "  mic3_file  Audio file for microphone 3 (.wav or .csv)" << std::endl;
    std::cout << "  mic4_file  Audio file for microphone 4 (.wav or .csv)" << std::endl;
    std::cout << std::endl;
    std::cout << "Example:" << std::endl;
    std::cout << "  " << program_name << " mic1.wav mic2.wav mic3.wav mic4.wav" << std::endl;
    std::cout << "  " << program_name << " mic1.csv mic2.csv mic3.csv mic4.csv" << std::endl;
    std::cout << std::endl;
    std::cout << "File formats:" << std::endl;
    std::cout << "  WAV: Standard PCM WAV files (8/16/32 bit)" << std::endl;
    std::cout << "  CSV: One sample per line (plain text)" << std::endl;
}

int main(int argc, char* argv[]) {
    if (argc != 5) {
        printUsage(argv[0]);
        return 1;
    }

    std::array<std::string, 4> filenames = {
        argv[1], argv[2], argv[3], argv[4]
    };

    std::cout << "=== Sound Source Localization from Files ===" << std::endl;
    std::cout << std::endl;

    // Verify files exist
    for (size_t i = 0; i < 4; ++i) {
        if (!AudioFileLoader::fileExists(filenames[i])) {
            std::cerr << "Error: File not found: " << filenames[i] << std::endl;
            return 1;
        }
    }

    // Setup microphone positions (square array, 20cm spacing)
    std::array<Microphone, MIC_COUNT> microphones = {
        Microphone(0.0, 0.0, 0.0),      // M1 (origin)
        Microphone(0.20, 0.0, 0.0),     // M2 (20 cm right)
        Microphone(0.20, 0.20, 0.0),    // M3 (20 cm diagonal)
        Microphone(0.0, 0.20, 0.0)      // M4 (20 cm up)
    };

    std::cout << "Microphone positions:" << std::endl;
    for (size_t i = 0; i < MIC_COUNT; ++i) {
        auto pos = microphones[i].getPosition();
        std::cout << "  M" << (i+1) << ": (" << pos[0] << ", " << pos[1] << ", " << pos[2] << ")" << std::endl;
    }
    std::cout << std::endl;

    // Load audio files
    std::cout << "Loading audio files..." << std::endl;
    std::array<std::vector<double>, 4> audioBuffers;
    
    try {
        audioBuffers = AudioFileLoader::loadMultipleFiles(filenames, Microphone::BUFFER_SIZE);
    } catch (const std::exception& e) {
        std::cerr << "Error loading files: " << e.what() << std::endl;
        return 1;
    }

    std::cout << std::endl;

    // Create localizer
    double dt = Microphone::BUFFER_SIZE / 44100.0; // Assuming 44.1 kHz sample rate
    Localizer localizer(microphones, dt, 343.0, FilterType::EKF1);

    std::cout << "Running localization..." << std::endl;

    // Perform localization
    auto position = localizer.locateSource(audioBuffers);

    std::cout << std::endl;
    std::cout << "=== Results ===" << std::endl;
    std::cout << "Estimated position: (" 
              << position[0] << ", " 
              << position[1] << ", " 
              << position[2] << ") meters" << std::endl;

    // Get uncertainty
    auto P = localizer.getCovariance();
    double position_uncertainty = std::sqrt(P(0,0) + P(1,1) + P(2,2));
    std::cout << "Position uncertainty: " << position_uncertainty << " m (RMS)" << std::endl;

    std::cout << std::endl;
    std::cout << "Localization complete!" << std::endl;

    return 0;
}
