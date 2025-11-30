#ifndef TDOA_CALCULATOR_H
#define TDOA_CALCULATOR_H

#include <vector>
#include <complex>
#include <array>
#include <Eigen/Dense>
#include <fftw3.h>
#include "microphone.h"

// TDOA configuration constants
inline constexpr int MIC_COUNT = 4;
inline constexpr int MEASUREMENT_DIM_TDOA = 6;
inline constexpr double SAMPLE_RATE = 44100.0; // Standard sampling rate (Hz)
inline constexpr size_t FFT_SIZE = Microphone::BUFFER_SIZE; // Use same size as Microphone buffer

// Type definitions to simplify the code
using ComplexVector = std::vector<std::complex<double>>;
using RealVector = std::vector<double>;
using TdoaVector = Eigen::Matrix<double, MEASUREMENT_DIM_TDOA, 1>;

class TdoaCalculator {
public:
    TdoaCalculator();

    // Main method to compute TDOA vector from raw buffers
    TdoaVector calculate(
        const std::array<RealVector, MIC_COUNT>& audioBuffers
    );

private:
    // --- Signal processing steps (SHOULD BE IMPLEMENTED USING FFT LIBRARY) ---
    
    void recursiveFft(ComplexVector& a, bool inverse);

    // Apply a Butterworth bandpass to isolate 100-300 Hz
    RealVector bandpassFilter(const RealVector& signal);

    // Perform FFT. In a real system use FFTW or similar.
    ComplexVector fft(const RealVector& signal);
    
    // Perform inverse FFT. In a real system use FFTW or similar.
    RealVector ifft(const ComplexVector& spectrum);
    
    // Compute TDOA in samples between a pair of signals
    double calculateGccPhat(const RealVector& signal1, const RealVector& signal2);
};

#endif // TDOA_CALCULATOR_H