#include "gcc_phat.h"
#include <algorithm>
#include <iostream>
#include <complex>
#include <cassert>

GccPhat::GccPhat(size_t fft_size, double sample_rate)
    : fft_size_(fft_size), sample_rate_(sample_rate) {}

// -------------------------------------------------------------
// Hann-window
// -------------------------------------------------------------
void GccPhat::applyHannWindow(RealVector& signal) {
    for (size_t n = 0; n < signal.size(); ++n) {
        signal[n] *= 0.5 * (1.0 - cos(2.0 * M_PI * n / (signal.size() - 1)));
    }
}

// -------------------------------------------------------------
// FFT / IFFT
// -------------------------------------------------------------
ComplexVector GccPhat::fft(const RealVector& signal) {
    ComplexVector c(signal.begin(), signal.end());
    recursiveFft(c, false);
    return c;
}

RealVector GccPhat::ifft(const ComplexVector& spectrum) {
    ComplexVector c = spectrum;
    recursiveFft(c, true);
    RealVector result(c.size());
    for (size_t i = 0; i < c.size(); ++i) {
        result[i] = c[i].real() / fft_size_;
    }
    return result;
}

void GccPhat::recursiveFft(ComplexVector& a, bool inverse) {
    size_t n = a.size();
    if (n <= 1) return;

    ComplexVector a0(n/2), a1(n/2);
    for (size_t i = 0; i < n/2; ++i) {
        a0[i] = a[i*2];
        a1[i] = a[i*2+1];
    }

    recursiveFft(a0, inverse);
    recursiveFft(a1, inverse);

    double angle = (inverse ? 2.0 : -2.0) * M_PI / n;
    std::complex<double> w(1.0), wn = std::exp(std::complex<double>(0, angle));
    for (size_t k = 0; k < n/2; ++k) {
        a[k] = a0[k] + w * a1[k];
        a[k + n/2] = a0[k] - w * a1[k];
        if (inverse) w /= wn;
        else w *= wn;
    }
}

// -------------------------------------------------------------
// Simple bandpass: zeros frequencies outside [low_hz, high_hz]
// -------------------------------------------------------------
void GccPhat::bandpassFilter(ComplexVector& spectrum, double low_hz, double high_hz) {
    double freq_res = sample_rate_ / fft_size_;
    for (size_t k = 0; k < spectrum.size(); ++k) {
        double f = k * freq_res;
        if (f < low_hz || f > high_hz) spectrum[k] = 0.0;
    }
}

// -------------------------------------------------------------
// Calculate TDOA between two signals
// -------------------------------------------------------------
double GccPhat::calculateTDOA(const RealVector& sig1, const RealVector& sig2) {
    assert(sig1.size() == fft_size_ && sig2.size() == fft_size_);

    RealVector s1 = sig1;
    RealVector s2 = sig2;

    // 1. Hann-window
    applyHannWindow(s1);
    applyHannWindow(s2);

    // 2. FFT
    ComplexVector S1 = fft(s1);
    ComplexVector S2 = fft(s2);

    // 3. Korsspektrum med PHAT vikt
    ComplexVector R(S1.size());
    for (size_t k = 0; k < S1.size(); ++k) {
        std::complex<double> prod = S1[k] * std::conj(S2[k]);
        double mag = std::abs(prod);
        if (mag > 1e-10) R[k] = prod / mag;
        else R[k] = 0.0;
    }

    // 4. IFFT to get cross-correlation
    RealVector corr = ifft(R);

    // 5. Hitta peak
    size_t max_index = 0;
    double max_val = corr[0];
    for (size_t i = 1; i < corr.size(); ++i) {
        if (corr[i] > max_val) {
            max_val = corr[i];
            max_index = i;
        }
    }

    // 6. Hantera wrap-around
    int shift_samples = static_cast<int>(max_index);
    if (shift_samples > (int)fft_size_/2) shift_samples -= fft_size_;

    return static_cast<double>(shift_samples) / sample_rate_;
}

// -------------------------------------------------------------
// Testfall: syntetisk signal
// -------------------------------------------------------------
#ifdef GCC_PHAT_TEST

#include <iostream>

int main() {
    const size_t N = 1024;
    const double fs = 44100;
    const double delay_s = 0.001; // 1 ms delay
    const size_t delay_samples = static_cast<size_t>(fs * delay_s);

    RealVector sig1(N, 0.0), sig2(N, 0.0);

    // Syntetisk impulssignal
    sig1[0] = 1.0;
    if (delay_samples < N) sig2[delay_samples] = 1.0;

    GccPhat gcc(N, fs);
    double tdoa = gcc.calculateTDOA(sig1, sig2);

    std::cout << "Expected TDOA: " << delay_s << " s\n";
    std::cout << "Calculated TDOA: " << tdoa << " s\n";

    return 0;
}

#endif
