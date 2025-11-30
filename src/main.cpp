// src/main.cpp
#include "localizer.h"
#include "tdoa_calculator.h"
#include "microphone.h"
#include <array>
#include <iostream>
#include <vector>

int main() {
    // NOTE: MIC_COUNT is 4 in your header - create 4 microphones
    std::array<Microphone, MIC_COUNT> microphones = {
        Microphone(0.0, 0.0, 0.0),     // M1
        Microphone(0.20, 0.0, 0.0),    // M2 (20 cm to the right)
        Microphone(0.10, 0.1732, 0.0), // M3 (triangle)
        Microphone(-0.20, 0.0, 0.0)    // M4 (additional microphone)
    };

    // Create Localizer (use the same dt as desired, e.g. 0.01s)
    Localizer localizer(microphones, 0.01, 343.0, FilterType::EKF1);

    // Build test audio buffers: each buffer must be Microphone::BUFFER_SIZE long
    std::array<std::vector<double>, MIC_COUNT> audioBuffers;
    for (size_t i = 0; i < MIC_COUNT; ++i) {
        audioBuffers[(uint32_t)i].assign(Microphone::BUFFER_SIZE, 0.0); // fill with zeros (dummy)
        // For testing you can insert an impulse in the middle:
        audioBuffers[(uint32_t)i][Microphone::BUFFER_SIZE / 2] = (i == 0 ? 1.0 : 0.9); // small offset
    }

    // Run the localizer (uses your dummy TDOA if you haven't replaced it)
    auto sourceLocation = localizer.locateSource(audioBuffers);

    // Skriv ut x, y, z
    std::cout << "Source located at: (" << sourceLocation[0] << ", " << sourceLocation[1] << ", " << sourceLocation[2]
              << ")" << std::endl;

    return 0;
}
