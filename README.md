# Hum-Finder

Acoustic source localization for finding annoying low-frequency drone sounds in apartments using Time Difference of Arrival (TDOA) analysis.

## Quick Start

```bash
# Build
mkdir build && cd build
cmake .. && make

# Run tests
ctest --output-on-failure

# Localize a sound source (4 microphones)
./localize_from_files mic1.wav mic2.wav mic3.wav mic4.wav

# Alternative: 3 mics with multiple recordings (for stationary sources)
./localize_multi_recording \
  rec1_mic1.wav 0,0,0 rec1_mic2.wav 0.2,0,0 rec1_mic3.wav 0.2,0.2,0 \
  rec2_mic1.wav 0,0,0.5 rec2_mic2.wav 0.2,0,0.5 rec2_mic3.wav 0.2,0.2,0.5
```

## What It Does

Pinpoints the 3D location of continuous drone sounds using microphone arrays:
- ✅ HVAC/fan noise
- ✅ Refrigerator hum
- ✅ Electrical transformers
- ✅ Water leaks in pipes
- ✅ Any persistent low-frequency sound

**Accuracy**: 0.1-1.0m typical, depending on source type and mic placement.

## Components

### Main Project (Acoustic Localization)
- **Executables**: `localize_from_files`, `localize_multi_recording`, `localize_realtime`
- **Algorithms**: LS, WLS, ML, EKF with TDOA measurements
- **Test Suite**: 60+ tests covering all estimators and real-world scenarios

### Sensor Fusion Library
Standalone C++ library (`sensor-fusion/`) with:
- Extended Kalman Filter (EKF) and Unscented Kalman Filter (UKF)
- Maximum Likelihood (ML) estimation
- Sensor modeling and CRLB analysis
- **Can be used in other projects** - see `sensor-fusion/README.md`

## Documentation

| Document | Purpose |
|----------|---------|
| **[PRACTICAL_DRONE_FINDING_GUIDE.md](PRACTICAL_DRONE_FINDING_GUIDE.md)** | Step-by-step for end users |
| **[FILE_INPUT_GUIDE.md](FILE_INPUT_GUIDE.md)** | Using recorded audio files |
| **[REALTIME_GUIDE.md](REALTIME_GUIDE.md)** | Live microphone streaming |
| **[MULTI_RECORDING_GUIDE.md](MULTI_RECORDING_GUIDE.md)** | 3-mic multi-recording method |
| **[sensor-fusion/README.md](sensor-fusion/README.md)** | Sensor fusion library API |
| **[DOCS.md](DOCS.md)** | Complete technical documentation |

## Building and Testing

```bash
# Standard build
mkdir build && cd build
cmake .. && make

# Run all tests
./run_tests          # Core tests
./test_nl            # Kalman filters
./test_all_estimators # LS, WLS, ML, CRLB, EKF, UKF
./test_broadband_drone # Real-world sound types
./test_real_world_drone # Apartment scenarios

# Or run all at once
ctest --output-on-failure
```

## Dependencies

- C++17
- Eigen3 (required)
- FFTW3 (for main project only)
- Google Test (for tests)

## Microphone Configurations

**Option 1: 4 Microphones, 1 Recording** (standard)
- Best for moving or unknown sources
- Instant results
- Requires 4 simultaneous recording devices

**Option 2: 3 Microphones, N Recordings** (cost-effective)
- For stationary sources only
- Better accuracy through averaging
- Saves equipment cost (only 3 mics needed)

See [MULTI_RECORDING_GUIDE.md](MULTI_RECORDING_GUIDE.md) for details.

## License

MIT License - see LICENSE file
  
- ✅ Use only 3 microphones instead of 4
- ✅ Improved accuracy through averaging
- ✅ Lower hardware cost
- ⚠️ Requires stationary source
- ⚠️ Takes multiple recording sessions

See `MULTI_RECORDING_GUIDE.md` for detailed instructions.

## Requirements

- CMake 3.14+
- C++17
- Eigen3
- FFTW3
- Google Test (optional, for tests)

## License

See LICENSE file.
