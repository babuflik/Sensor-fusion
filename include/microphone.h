#ifndef MICROPHONE_H
#define MICROPHONE_H

#include <vector>
#include <array>
#include <cstddef>
#include <iostream>

class Microphone {
public:
    // Size of the data block the microphone buffers.
    // e.g. 1024 samples. This determines the TDOA window size.
    static constexpr size_t BUFFER_SIZE = 1024; 

    // Constructor: position x, y, z
    Microphone(double x, double y, double z);

    // Position getter
    std::array<double, 3> getPosition() const;

    void setPosition(const std::array<double,3>& pos);
    
    // Add a new audio sample to the buffer (append and overwrite oldest when full)
    void addSample(double sample);
    
    // Return the current buffer (BUFFER_SIZE samples).
    // NOTE: returns a COPY of the buffer for thread-safety and cleanliness.
    std::vector<double> getSamples() const;
    
    // Returns whether the buffer is full and ready to be processed
    bool isBufferFull() const { return samples_filled_count_ == BUFFER_SIZE; }

private:
    // Physical 3D position
    std::array<double, 3> position_; 

    // Ring buffer for audio samples. std::vector is used but logic treats it as fixed-size.
    std::vector<double> samples_; 
    
    // Index where next sample will be written
    size_t write_index_ = 0; 
    
    // Number of samples currently in the buffer (to check fullness)
    size_t samples_filled_count_ = 0; 
};

#endif // MICROPHONE_H