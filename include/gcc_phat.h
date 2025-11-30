#ifndef GCC_PHAT_H
#define GCC_PHAT_H

#include <vector>
#include <complex>
#include <cmath>
#include <Eigen/Dense>

// Type aliases
using RealVector = std::vector<double>;
using ComplexVector = std::vector<std::complex<double>>;

// -------------------------------------------------------------
// GCC-PHAT: Computes TDOA between two signals
// -------------------------------------------------------------
class GccPhat {
public:
    explicit GccPhat(size_t fft_size, double sample_rate);

    // Compute TDOA in seconds between signal1 and signal2
    double calculateTDOA(const RealVector& signal1, const RealVector& signal2);

    // Apply a Hann window
    void applyHannWindow(RealVector& signal);

    // Bandpass in frequency domain (optional)
    void bandpassFilter(ComplexVector& spectrum, double low_hz, double high_hz);

private:
    size_t fft_size_;
    double sample_rate_;

    // FFT / IFFT
    ComplexVector fft(const RealVector& signal);
    RealVector ifft(const ComplexVector& spectrum);

    // Recursive Cooley-Tukey FFT
    void recursiveFft(ComplexVector& a, bool inverse);
};

#endif // GCC_PHAT_H
