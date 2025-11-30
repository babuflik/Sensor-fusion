#include "tdoa_calculator.h"
#include <algorithm>
#include <cmath>
#include <numeric>
#include <iostream>

// Standard C++ header for PI (sometimes missing in certain compilers)
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

TdoaCalculator::TdoaCalculator() {
    // Check that FFT_SIZE is a power of 2 (Required for the recursive FFT)
    if ((FFT_SIZE > 0) && ((FFT_SIZE & (FFT_SIZE - 1)) != 0)) {
        std::cerr << "WARNING: FFT_SIZE should be a power of two for the current FFT implementation!" << std::endl;
    }
}

// -----------------------------------------------------------------
// COOLEY-TUKEY REKURSIV FFT/IFFT IMPLEMENTATION
// -----------------------------------------------------------------

// Private recursive function for FFT/IFFT
void TdoaCalculator::recursiveFft(ComplexVector& a, bool inverse) {
    int n = a.size();
    if (n <= 1) return;

    // Split into even and odd parts
    ComplexVector a0(n / 2), a1(n / 2);
    for (int i = 0; i < n / 2; i++) {
        a0[i] = a[2 * i];
        a1[i] = a[2 * i + 1];
    }

    // Rekursiva anrop
    recursiveFft(a0, inverse);
    recursiveFft(a1, inverse);

    // Kombinera resultaten
    double angle_sign = inverse ? 2 * M_PI : -2 * M_PI;
    std::complex<double> w(1), wn = std::exp(std::complex<double>(0, angle_sign / n));

    for (int k = 0; k < n / 2; k++) {
        a[k] = a0[k] + w * a1[k];
        a[k + n / 2] = a0[k] - w * a1[k];
        w *= wn;
    }
}

// Perform FFT
ComplexVector TdoaCalculator::fft(const RealVector& signal) {
    if (signal.size() != FFT_SIZE) {
        std::cerr << "ERROR: Signal size (" << signal.size() << ") does not match FFT_SIZE (" << FFT_SIZE << ")." << std::endl;
        return ComplexVector(FFT_SIZE, 0.0);
    }
    
    ComplexVector result(signal.begin(), signal.end());
    recursiveFft(result, false);
    return result;
}

// Perform IFFT
RealVector TdoaCalculator::ifft(const ComplexVector& spectrum) {
    if (spectrum.size() != FFT_SIZE) {
        std::cerr << "ERROR: Spectrum size (" << spectrum.size() << ") does not match FFT_SIZE (" << FFT_SIZE << ")." << std::endl;
        return RealVector(FFT_SIZE, 0.0);
    }

    ComplexVector complex_result = spectrum;
    
    // 1. Inverse FFT (use reversed sign)
    recursiveFft(complex_result, true);
    
    // 2. Normalisering
    RealVector real_result(FFT_SIZE);
    for (size_t i = 0; i < FFT_SIZE; ++i) {
        // Dela med N och ta realdelen (FFT/IFFT skalan)
        real_result[i] = complex_result[i].real() / FFT_SIZE;
    }

    return real_result;
}


// -----------------------------------------------------------------
// BANDPASS FILTER IMPLEMENTATION (Frequency Domain)
// -----------------------------------------------------------------

/*
 * Implements a bandpass filter in the frequency domain.
 * Isolates the desired frequency range 100 Hz to 300 Hz.
 * * F_res = SAMPLE_RATE / FFT_SIZE = 44100 / 1024 = 43.066 Hz/bin
 * * Cutoff low: 100 Hz -> Bin 3 (129.19 Hz)
 * Cutoff high: 300 Hz -> Bin 7 (301.46 Hz)
 * * Range: [Bin 3, Bin 7] and its mirrored counterpart [N - 7, N - 3].
 */
RealVector TdoaCalculator::bandpassFilter(const RealVector& signal) {
    if (signal.size() != FFT_SIZE) {
        // Should not happen if call comes from calculateGccPhat, but is a safety check
        return signal;
    }

    // FFT
    ComplexVector spectrum = fft(signal);

    // Frequency limits based on 43.066 Hz/bin
    const int K_LOW = 3;
    const int K_HIGH = 7;
    const size_t N = FFT_SIZE;
    
    // 2. Zero all frequencies outside the band
    for (size_t k = 0; k < N; ++k) {
        // Zero for k < K_LOW (lowpass part)
        // Zero for k > K_HIGH AND k < N - K_HIGH (region between positive and negative spectrum)
        // Zero for k > N - K_LOW (highpass part in the negative spectrum)
        
        if ((k < K_LOW) || (k > K_HIGH && k < N - K_HIGH) || (k > N - K_LOW)) {
            // Zero if outside band [K_LOW, K_HIGH] U [N - K_HIGH, N - K_LOW]
            spectrum[k] = 0.0;
        }
    }

    // 3. IFFT
    return ifft(spectrum);
}


// Calculate TDOA in samples for a pair
double TdoaCalculator::calculateGccPhat(const RealVector& signal1, const RealVector& signal2) {
    
    // Steg 1: Filtrering 
    RealVector filtered1 = bandpassFilter(signal1);
    RealVector filtered2 = bandpassFilter(signal2);

    // Steg 2: FFT
    ComplexVector spectrum1 = fft(filtered1);
    ComplexVector spectrum2 = fft(filtered2);

    // Step 3: Calculate PHAT weight and cross-correlation spectrum (R_ij^PHAT)
    ComplexVector cross_spectrum_phat;
    cross_spectrum_phat.reserve(FFT_SIZE);

    for (size_t k = 0; k < FFT_SIZE; ++k) {
        // Y_i[k] * Y_j^*[k]
        std::complex<double> product = spectrum1[k] * std::conj(spectrum2[k]);
        
        double magnitude = std::abs(product);
        if (magnitude > 1e-10) {
            // R_ij^PHAT[k] = (Y_i[k] * Y_j^*[k]) / |Y_i[k] * Y_j^*[k]|
            cross_spectrum_phat.push_back(product / magnitude);
        } else {
            cross_spectrum_phat.push_back(0.0);
        }
    }

    // Steg 4: IFFT
    RealVector correlation = ifft(cross_spectrum_phat);

    // Step 5: Peak detection (Find index with highest magnitude)
    double max_corr = 0.0;
    int max_index = 0;

    // Vi letar efter toppen i korrelationsvektorn
    for (size_t n = 0; n < FFT_SIZE; ++n) {
        if (correlation[n] > max_corr) {
            max_corr = correlation[n];
            max_index = n;
        }
    }

    // Handle circular shift (wrap-around) of IFFT result
    // If index is greater than N/2, it means a negative delay
    int shift_samples = max_index;
    if (max_index > FFT_SIZE / 2) {
        shift_samples = max_index - (int)FFT_SIZE;
    }

    // Konvertera samples till tid (sekunder)
    return (double)shift_samples / SAMPLE_RATE;
}

// Huvudmetod
TdoaVector TdoaCalculator::calculate(
    const std::array<RealVector, MIC_COUNT>& audioBuffers
) {
    TdoaVector result;

    // Index mapping for the 6 TDOA pairs: (i, j)
    std::array<std::pair<int, int>, MEASUREMENT_DIM_TDOA> mic_pairs = {{
        {0, 1}, {0, 2}, {0, 3}, {1, 2}, {1, 3}, {2, 3}
    }};

    for (int k = 0; k < MEASUREMENT_DIM_TDOA; ++k) {
        int i = mic_pairs[k].first;
        int j = mic_pairs[k].second;

        if (audioBuffers[i].size() != FFT_SIZE || audioBuffers[j].size() != FFT_SIZE) {
            std::cerr << "ERROR: Buffer size (" << audioBuffers[i].size() << ") does not match FFT_SIZE (" << FFT_SIZE << ")." << std::endl;
            // Returns zero; in a real application this should be handled more robustly
            result(k) = 0.0; 
            continue;
        }
        
        // Calculate TDOA for pair i and j
        result(k) = calculateGccPhat(audioBuffers[i], audioBuffers[j]);
    }
    
    return result;
}