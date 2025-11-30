#include "microphone.h"
#include <algorithm> // For std::fill

// Konstruktor
Microphone::Microphone(double x, double y, double z) 
    : position_{x, y, z} {
    
    // Allocate memory for the entire buffer once at startup
    samples_.resize(BUFFER_SIZE);
    
    // Zero out the buffer
    std::fill(samples_.begin(), samples_.end(), 0.0);
}

// Position getter
std::array<double, 3> Microphone::getPosition() const {
    return position_;
}

void Microphone::setPosition(const std::array<double,3>& pos) {
    position_ = pos;
}

// Add a new audio sample and manage the ring buffer
void Microphone::addSample(double sample) {
    // Insert sample at the current write index
    samples_[write_index_] = sample;
    
    // Increment write index and wrap to 0 if we reach the end
    write_index_ = (write_index_ + 1) % BUFFER_SIZE;
    
    // Count how many samples have been filled. Increases until it reaches max size.
    if (samples_filled_count_ < BUFFER_SIZE) {
        samples_filled_count_++;
    }
}

// Returnerar hela den aktuella bufferten
std::vector<double> Microphone::getSamples() const {
    
    // If buffer is not full, return only the collected samples
    if (samples_filled_count_ < BUFFER_SIZE) {
        return samples_;
    }

    // --- Ring buffer read logic ---
    
    // If buffer is full: The oldest samples are at write_index_.
    // We must read the buffer in CORRECT TIME ORDER: 
    // [Oldest samples] followed by [Newest samples].
    
    std::vector<double> ordered_samples;
    ordered_samples.reserve(BUFFER_SIZE);
    
    // 1. Copy the oldest samples (from write_index_ to end)
    if (write_index_ < BUFFER_SIZE) {
        ordered_samples.insert(
            ordered_samples.end(),
            samples_.begin() + write_index_,
            samples_.end()
        );
    }
    
    // 2. Copy the newest samples (from beginning to write_index_)
    if (write_index_ > 0) {
        ordered_samples.insert(
            ordered_samples.end(),
            samples_.begin(),
            samples_.begin() + write_index_
        );
    }

    // Note: This returns a Copy. For real-time optimization, consider 
    // returning a reference or a specialized iterator.
    return ordered_samples;
}