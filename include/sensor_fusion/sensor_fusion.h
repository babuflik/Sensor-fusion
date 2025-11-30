/**
 * @file sensor_fusion.h
 * @brief Main include file for the Sensor Fusion Toolbox
 * 
 * This toolbox provides nonlinear estimation and sensor fusion capabilities:
 * - Nonlinear system modeling (NL class)
 * - Sensor modeling (SensorMod class)
 * - Signal/data containers (Sig class)
 * - Estimation algorithms (LS, WLS, ML, CRLB)
 * - Extended/Unscented Kalman Filters
 * - Probability distributions
 * 
 * @version 1.0.0
 * @author Sensor Fusion Toolbox Contributors
 * 
 * Example usage:
 * @code
 * #include <sensor_fusion/sensor_fusion.h>
 * 
 * // Create nonlinear system
 * NL::StateFunction f = [](double t, const auto& x, const auto& u, const auto& th) {
 *     return x + u * dt; // Simple integrator
 * };
 * 
 * NL::MeasurementFunction h = [](double t, const auto& x, const auto& u, const auto& th) {
 *     return x; // Direct measurement
 * };
 * 
 * Eigen::Vector4i dims(2, 1, 2, 0); // nx=2, nu=1, ny=2, nth=0
 * NL system(f, h, dims);
 * @endcode
 */

#ifndef SENSOR_FUSION_H
#define SENSOR_FUSION_H

// Core classes
#include "sensor_fusion/sig.h"
#include "sensor_fusion/nl.h"
#include "sensor_fusion/sensormod.h"

// Estimation algorithms
#include "sensor_fusion/estimators.h"

// Probability distributions
#include "sensor_fusion/pdfclass.h"
#include "sensor_fusion/ndist.h"
#include "sensor_fusion/betadist.h"
#include "sensor_fusion/chi2dist.h"
#include "sensor_fusion/empdist.h"
#include "sensor_fusion/expdist.h"
#include "sensor_fusion/gammadist.h"
#include "sensor_fusion/gmdist.h"
#include "sensor_fusion/logndist.h"
#include "sensor_fusion/ncchi2dist.h"
#include "sensor_fusion/tdist.h"
#include "sensor_fusion/udist.h"

// Utilities
#include "sensor_fusion/utils_sigsys.h"

/**
 * @namespace SensorFusion
 * @brief Main namespace for sensor fusion functionality
 */
namespace SensorFusion {
    /**
     * @brief Library version
     */
    constexpr const char* VERSION = "1.0.0";
    
    /**
     * @brief Get library version string
     * @return Version string in format "major.minor.patch"
     */
    inline const char* getVersion() {
        return VERSION;
    }
}

#endif // SENSOR_FUSION_H
