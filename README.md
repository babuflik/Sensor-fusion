# Sensor Fusion Toolbox

A C++ library for nonlinear state estimation and sensor fusion, extracted from the Hum-Finder acoustic localization project.

## Features

- **Nonlinear System Modeling** (`NL` class)
  - Continuous and discrete-time systems
  - Extended Kalman Filter (EKF)
  - Unscented Kalman Filter (UKF)
  - Monte Carlo simulations

- **Sensor Modeling** (`SensorMod` class)
  - Flexible measurement models
  - Likelihood computation
  - CRLB (Cramér-Rao Lower Bound) analysis

- **Estimation Algorithms**
  - Least Squares (LS)
  - Weighted Least Squares (WLS)
  - Maximum Likelihood (ML)
  - CRLB evaluation

- **Signal/Data Containers** (`Sig` class)
  - Time-series data with measurements, states, inputs
  - Covariance tracking
  - Monte Carlo sample storage

- **Probability Distributions**
  - Gaussian (multivariate normal)
  - Beta, Chi-squared, Exponential
  - Gamma, Log-normal, Student's t
  - Uniform, Empirical, Gaussian Mixture

## Requirements

- CMake 3.14+
- C++17 compiler
- Eigen3

## Installation

### Build and Install

```bash
cd sensor-fusion
mkdir build && cd build
cmake ..
make
sudo make install
```

### Install to Custom Location

```bash
cmake .. -DCMAKE_INSTALL_PREFIX=/path/to/install
make
make install
```

### Build Options

```bash
cmake .. \
  -DSENSOR_FUSION_BUILD_EXAMPLES=ON \   # Build example programs
  -DSENSOR_FUSION_BUILD_SHARED=OFF       # Build static library (default)
```

## Usage in Your Project

### CMake Integration (Recommended)

After installing the library:

```cmake
cmake_minimum_required(VERSION 3.14)
project(MyProject)

# Find the installed library
find_package(SensorFusion REQUIRED)

add_executable(my_app main.cpp)
target_link_libraries(my_app PRIVATE SensorFusion::sensor_fusion)
```

### CMake Subdirectory (Alternative)

Add as a subdirectory without installing:

```cmake
add_subdirectory(path/to/sensor-fusion)
target_link_libraries(my_app PRIVATE SensorFusion::sensor_fusion)
```

### Manual Integration

If not using CMake, link against:
- Library: `libsensor_fusion.a` (or `.so` if shared)
- Include path: `/path/to/install/include`
- Dependencies: Eigen3

## Quick Start Examples

### Example 1: EKF for 2D Tracking

```cpp
#include <sensor_fusion/sensor_fusion.h>

int main() {
    // Define constant velocity model: x+ = [x+vx*dt, y+vy*dt, vx, vy]
    const double dt = 0.1;
    NL::StateFunction f = [dt](double t, const Eigen::VectorXd& x,
                                const Eigen::VectorXd& u, const Eigen::VectorXd& th) {
        Eigen::VectorXd x_next(4);
        x_next << x(0) + x(2)*dt, x(1) + x(3)*dt, x(2), x(3);
        return x_next;
    };
    
    // Direct position measurement: y = [x, y]
    NL::MeasurementFunction h = [](double t, const Eigen::VectorXd& x,
                                   const Eigen::VectorXd& u, const Eigen::VectorXd& th) {
        return x.head(2);  // Measure only position
    };
    
    // Create system model
    Eigen::Vector4i dims(4, 0, 2, 0);  // nx=4, nu=0, ny=2, nth=0
    NL system(f, h, dims, 1.0/dt);
    
    // Set initial state, noise, etc.
    system.x0 << 0, 0, 1, 0.5;  // Start at origin, moving at (1, 0.5) m/s
    
    // Run simulation and estimation...
}
```

### Example 2: ML Estimation with Sensor Model

```cpp
#include <sensor_fusion/sensor_fusion.h>

int main() {
    // Define range-bearing sensor model
    auto h = [](double t, const Eigen::VectorXd& x, const Eigen::VectorXd& u, const Eigen::VectorXd& th) {
        Eigen::VectorXd y(2);
        y(0) = x.norm();              // Range
        y(1) = std::atan2(x(1), x(0)); // Bearing
        return y;
    };
    
    Eigen::Vector4i nn(2, 0, 2, 0);
    SensorMod sensor(h, nn);
    sensor.pe = Eigen::MatrixXd::Identity(2, 2) * 0.01;  // Measurement noise
    
    // Create synthetic measurements
    Eigen::VectorXd t(1); t << 0.0;
    Eigen::MatrixXd x_true(2, 1); x_true << 5.0, 3.0;
    Sig measurements = sensor.simulate(t, &x_true);
    
    // Run ML estimation
    auto [x_est, s_est, cov] = ml(sensor, measurements);
    
    std::cout << "Estimated position: " << x_est.x.transpose() << std::endl;
}
```

### Example 3: Acoustic Source Localization (TDOA)

```cpp
#include <sensor_fusion/sensor_fusion.h>

int main() {
    // Microphone positions
    Eigen::MatrixXd mics(4, 3);
    mics << 0.0, 0.0, 0.0,
            0.2, 0.0, 0.0,
            0.2, 0.2, 0.0,
            0.0, 0.2, 0.0;
    
    const double C = 343.0;  // Speed of sound
    
    // TDOA measurement model
    auto h_tdoa = [mics, C](double t, const Eigen::VectorXd& x,
                            const Eigen::VectorXd& u, const Eigen::VectorXd& th) {
        Eigen::VectorXd tdoa(6);
        Eigen::VectorXd dist(4);
        
        for (int i = 0; i < 4; ++i) {
            dist(i) = (x - mics.row(i).transpose()).norm();
        }
        
        int idx = 0;
        for (int i = 0; i < 4; ++i) {
            for (int j = i+1; j < 4; ++j) {
                tdoa(idx++) = (dist(i) - dist(j)) / C;
            }
        }
        return tdoa;
    };
    
    Eigen::Vector4i nn(3, 0, 6, 0);  // 3D position, 6 TDOAs
    SensorMod sensor(h_tdoa, nn);
    sensor.pe = Eigen::MatrixXd::Identity(6, 6) * 1e-6;
    
    // Estimate from measurements
    // auto [x_est, s_est, cov] = ml(sensor, measurements);
}
```

## API Reference

### Core Classes

#### `NL` - Nonlinear System Model

```cpp
class NL {
public:
    using StateFunction = std::function<Vector(double, const Vector&, const Vector&, const Vector&)>;
    using MeasurementFunction = std::function<Vector(double, const Vector&, const Vector&, const Vector&)>;
    
    Eigen::Vector4i nn;  // [nx, nu, ny, nth]
    StateFunction f;      // State dynamics
    MeasurementFunction h; // Measurement function
    
    NL(StateFunction f_func, MeasurementFunction h_func, 
       const Eigen::Vector4i& dimensions, double sampling_freq = NAN);
    
    Sig simulate(const Eigen::VectorXd& t, ...);
    Sig ekf(const Sig& measurements, ...);
    Sig ukf(const Sig& measurements, ...);
};
```

####  `SensorMod` - Sensor Model

```cpp
class SensorMod {
public:
    using DynFunc = std::function<Eigen::VectorXd(double, const Eigen::VectorXd&,
                                                  const Eigen::VectorXd&, const Eigen::VectorXd&)>;
    
    Eigen::Vector4i nn;  // [nx, nu, ny, nth]
    DynFunc h;          // Measurement function
    Eigen::MatrixXd pe; // Measurement noise covariance
    
    SensorMod(const DynFunc& hFunc, const Eigen::Vector4i& nn_);
    
    Sig simulate(const Eigen::VectorXd& t, ...);
    Eigen::VectorXd likelihood_function(...);
};
```

#### `Sig` - Signal/Data Container

```cpp
struct Sig {
    Eigen::MatrixXd y;  // Measurements (N x ny)
    Eigen::VectorXd t;  // Time vector (N x 1)
    Eigen::MatrixXd x;  // States (N x nx)
    Eigen::MatrixXd u;  // Inputs (N x nu)
    double fs;          // Sampling frequency
    
    // Covariance and Monte Carlo data
    std::vector<Eigen::MatrixXd> Py;  // Measurement covariances
    std::vector<Eigen::MatrixXd> yMC; // MC samples
    // ...
};
```

### Estimation Functions

```cpp
// Least Squares
std::tuple<Sig, SensorMod> ls(const SensorMod& s, const Sig& y);

// Weighted Least Squares
std::tuple<Sig, SensorMod> wls(const SensorMod& s, const Sig& y);

// Maximum Likelihood
std::tuple<Sig, SensorMod, Eigen::MatrixXd> ml(const SensorMod& s, const Sig& y);

// Cramér-Rao Lower Bound
Sig crlb(const SensorMod& s, const Sig* y);

// 1D/2D Likelihood
std::tuple<...> lh1(const SensorMod& s, const Sig& y, ...);
std::tuple<...> lh2(const SensorMod& s, const Sig& y, ...);
```

## Examples

Build and run the included examples:

```bash
cd build
make

# Run examples
./examples/example_ekf
./examples/example_sensor_model
./examples/example_ml_estimation
```

## Applications

This library is being used in:

- **Hum-Finder**: Acoustic source localization for finding drone sounds in apartments
- **[Your project here]**: Submit a PR to add your application!

## Integration with Hum-Finder

The Hum-Finder project uses this library for all sensor fusion operations:

```cmake
# In hum-finder's CMakeLists.txt
add_subdirectory(sensor-fusion)
target_link_libraries(humming_core PUBLIC SensorFusion::sensor_fusion)
```

## Architecture

```
sensor-fusion/
├── include/sensor_fusion/
│   ├── sensor_fusion.h      # Main include (all-in-one)
│   ├── nl.h                 # Nonlinear system class
│   ├── sensormod.h          # Sensor model class
│   ├── sig.h                # Signal container
│   ├── estimators.h         # LS, WLS, ML, CRLB
│   ├── pdfclass.h           # Base PDF class
│   ├── *dist.h              # Probability distributions
│   └── utils_sigsys.h       # Utilities
├── src/
│   ├── nl.cpp
│   ├── estimators.cpp
│   └── utils.cpp
├── examples/
│   ├── example_ekf.cpp
│   ├── example_sensor_model.cpp
│   └── example_ml_estimation.cpp
└── CMakeLists.txt
```

## Documentation

- **API Reference**: See header files in `include/sensor_fusion/`
- **Examples**: See `examples/` directory
- **Theory**: Based on sensor fusion and nonlinear estimation theory

## Performance

- Lightweight: Header-only probability distributions, compiled core
- Efficient: Uses Eigen3 for optimized linear algebra
- Flexible: Template-based design allows custom function types

## License

[Same as Hum-Finder project]

## Contributing

Contributions welcome! This library was extracted from the Hum-Finder project to make the sensor fusion capabilities available to other projects.

### Roadmap

- [ ] Add Python bindings (pybind11)
- [ ] Particle filter implementation
- [ ] Additional probability distributions
- [ ] Benchmark suite
- [ ] Comprehensive unit tests
- [ ] Doxygen documentation generation

## Citation

If you use this library in academic work, please cite:

```bibtex
@software{sensor_fusion_toolbox,
  title={Sensor Fusion Toolbox},
  author={Hum-Finder Contributors},
  year={2025},
  url={https://github.com/babuflik/hum-finder}
}
```

## Support

- **Issues**: GitHub issue tracker
- **Discussions**: GitHub discussions
- **Contact**: [Your contact method]

---

**Note**: This is a standalone library extracted from Hum-Finder. It can be used independently for any sensor fusion or nonlinear estimation application!
