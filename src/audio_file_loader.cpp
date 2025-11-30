#include "audio_file_loader.h"
#include <fstream>
#include <iostream>
#include <algorithm>
#include <cstring>
#include <stdexcept>
#include <sys/stat.h>
#include <cstdint>

namespace AudioFileLoader {

// Simple WAV header structure
struct WAVHeader {
    char riff[4];           // "RIFF"
    uint32_t file_size;
    char wave[4];           // "WAVE"
    char fmt[4];            // "fmt "
    uint32_t fmt_size;
    uint16_t audio_format;  // 1 = PCM
    uint16_t num_channels;
    uint32_t sample_rate;
    uint32_t byte_rate;
    uint16_t block_align;
    uint16_t bits_per_sample;
    char data[4];           // "data"
    uint32_t data_size;
};

bool fileExists(const std::string& filename) {
    struct stat buffer;
    return (stat(filename.c_str(), &buffer) == 0);
}

std::vector<double> loadWAV(const std::string& filename) {
    std::ifstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        throw std::runtime_error("Cannot open WAV file: " + filename);
    }

    WAVHeader header;
    file.read(reinterpret_cast<char*>(&header), sizeof(WAVHeader));

    // Validate WAV file
    if (std::strncmp(header.riff, "RIFF", 4) != 0 ||
        std::strncmp(header.wave, "WAVE", 4) != 0) {
        throw std::runtime_error("Invalid WAV file format: " + filename);
    }

    std::cout << "Loading WAV: " << filename << std::endl;
    std::cout << "  Channels: " << header.num_channels << std::endl;
    std::cout << "  Sample rate: " << header.sample_rate << " Hz" << std::endl;
    std::cout << "  Bits per sample: " << header.bits_per_sample << std::endl;

    // Calculate number of samples
    size_t num_samples = header.data_size / (header.bits_per_sample / 8) / header.num_channels;
    std::vector<double> samples;
    samples.reserve(num_samples);

    // Read samples and convert to double [-1, 1]
    if (header.bits_per_sample == 16) {
        std::vector<int16_t> raw_samples(num_samples * header.num_channels);
        file.read(reinterpret_cast<char*>(raw_samples.data()), header.data_size);

        // Convert to mono if stereo (take left channel or average)
        for (size_t i = 0; i < num_samples; ++i) {
            if (header.num_channels == 1) {
                samples.push_back(raw_samples[i] / 32768.0);
            } else {
                // Average all channels
                double sum = 0.0;
                for (int ch = 0; ch < header.num_channels; ++ch) {
                    sum += raw_samples[i * header.num_channels + ch];
                }
                samples.push_back(sum / (header.num_channels * 32768.0));
            }
        }
    } else if (header.bits_per_sample == 8) {
        std::vector<uint8_t> raw_samples(num_samples * header.num_channels);
        file.read(reinterpret_cast<char*>(raw_samples.data()), header.data_size);

        for (size_t i = 0; i < num_samples; ++i) {
            if (header.num_channels == 1) {
                samples.push_back((raw_samples[i] - 128) / 128.0);
            } else {
                double sum = 0.0;
                for (int ch = 0; ch < header.num_channels; ++ch) {
                    sum += (raw_samples[i * header.num_channels + ch] - 128);
                }
                samples.push_back(sum / (header.num_channels * 128.0));
            }
        }
    } else if (header.bits_per_sample == 32) {
        std::vector<int32_t> raw_samples(num_samples * header.num_channels);
        file.read(reinterpret_cast<char*>(raw_samples.data()), header.data_size);

        for (size_t i = 0; i < num_samples; ++i) {
            if (header.num_channels == 1) {
                samples.push_back(raw_samples[i] / 2147483648.0);
            } else {
                double sum = 0.0;
                for (int ch = 0; ch < header.num_channels; ++ch) {
                    sum += raw_samples[i * header.num_channels + ch];
                }
                samples.push_back(sum / (header.num_channels * 2147483648.0));
            }
        }
    } else {
        throw std::runtime_error("Unsupported bits per sample: " + 
                                std::to_string(header.bits_per_sample));
    }

    std::cout << "  Loaded " << samples.size() << " samples" << std::endl;
    return samples;
}

std::vector<double> loadCSV(const std::string& filename) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        throw std::runtime_error("Cannot open CSV file: " + filename);
    }

    std::vector<double> samples;
    std::string line;
    
    std::cout << "Loading CSV: " << filename << std::endl;

    while (std::getline(file, line)) {
        // Skip empty lines and comments
        if (line.empty() || line[0] == '#') {
            continue;
        }

        try {
            double value = std::stod(line);
            samples.push_back(value);
        } catch (const std::exception& e) {
            std::cerr << "Warning: Skipping invalid line: " << line << std::endl;
        }
    }

    std::cout << "  Loaded " << samples.size() << " samples" << std::endl;
    return samples;
}

std::vector<double> loadAudioFile(const std::string& filename) {
    if (!fileExists(filename)) {
        throw std::runtime_error("File not found: " + filename);
    }

    // Detect file type by extension
    size_t dot_pos = filename.find_last_of('.');
    if (dot_pos == std::string::npos) {
        throw std::runtime_error("Cannot determine file type (no extension): " + filename);
    }

    std::string ext = filename.substr(dot_pos + 1);
    std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);

    if (ext == "wav") {
        return loadWAV(filename);
    } else if (ext == "csv") {
        return loadCSV(filename);
    } else {
        throw std::runtime_error("Unsupported file type: " + ext);
    }
}

std::array<std::vector<double>, 4> loadMultipleFiles(
    const std::array<std::string, 4>& filenames,
    size_t buffer_size)
{
    std::array<std::vector<double>, 4> buffers;

    for (size_t i = 0; i < 4; ++i) {
        std::vector<double> samples = loadAudioFile(filenames[i]);

        if (samples.empty()) {
            throw std::runtime_error("No samples loaded from: " + filenames[i]);
        }

        // Take first buffer_size samples or pad with zeros if too short
        buffers[i].resize(buffer_size, 0.0);
        size_t copy_size = std::min(buffer_size, samples.size());
        std::copy(samples.begin(), samples.begin() + copy_size, buffers[i].begin());

        if (samples.size() < buffer_size) {
            std::cout << "Warning: File " << filenames[i] << " has only " 
                      << samples.size() << " samples (padded to " << buffer_size << ")" << std::endl;
        }
    }

    return buffers;
}

} // namespace AudioFileLoader
