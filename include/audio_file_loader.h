#ifndef AUDIO_FILE_LOADER_H
#define AUDIO_FILE_LOADER_H

#include <vector>
#include <string>
#include <array>

namespace AudioFileLoader {

/**
 * Load audio samples from a WAV file
 * @param filename Path to WAV file
 * @return Vector of audio samples (mono, converted to double in range [-1, 1])
 */
std::vector<double> loadWAV(const std::string& filename);

/**
 * Load audio samples from a CSV file
 * @param filename Path to CSV file (one sample per line)
 * @return Vector of audio samples
 */
std::vector<double> loadCSV(const std::string& filename);

/**
 * Automatically detect file type and load
 * @param filename Path to file (.wav or .csv)
 * @return Vector of audio samples
 */
std::vector<double> loadAudioFile(const std::string& filename);

/**
 * Load audio files for all 4 microphones
 * @param filenames Array of 4 filenames
 * @param buffer_size Number of samples to load (default: 1024)
 * @return Array of 4 audio buffers
 */
std::array<std::vector<double>, 4> loadMultipleFiles(
    const std::array<std::string, 4>& filenames,
    size_t buffer_size = 1024
);

/**
 * Check if file exists
 */
bool fileExists(const std::string& filename);

} // namespace AudioFileLoader

#endif // AUDIO_FILE_LOADER_H
