#pragma once
#include "gcc_phat.h"
#include "microphone.h"
#include <Eigen/Dense>
#include <array>
#include <vector>
#include <string>

enum class FilterType { EKF1, EKF2, UKF };

class Localizer {
public:
    // ---------------- Constants ----------------
    static constexpr int STATE_DIM = 6;        // x,y,z,vx,vy,vz
    static constexpr int MEASUREMENT_DIM = 6;  // 6 TDOA pairs

    Localizer(
        const std::array<Microphone, 4>& microphones,
        double dt,
        double sound_speed,
        FilterType type = FilterType::EKF1
    );

    std::array<double,3> locateSource(
        const std::array<std::vector<double>, 4>& audioBuffers
    );

    // Get current covariance matrix (uncertainty)
    Eigen::Matrix<double, STATE_DIM, STATE_DIM> getCovariance() const {
        return P_;
    }

    // Get current state estimate
    Eigen::Matrix<double, STATE_DIM, 1> getState() const {
        return x_;
    }

private:
    // ---------------- Config ----------------
    FilterType filter_type_;
    double dt_;
    double C_;
    double sample_rate_;

    // ---------------- State ----------------
    Eigen::Matrix<double, STATE_DIM,1> x_;
    Eigen::Matrix<double, STATE_DIM, STATE_DIM> P_;
    Eigen::Matrix<double, STATE_DIM, STATE_DIM> Q_;
    Eigen::Matrix<double, MEASUREMENT_DIM, MEASUREMENT_DIM> R_;
    Eigen::Matrix<double, STATE_DIM, STATE_DIM> F_;
    Eigen::Matrix<double,3,1> mic_positions_[4];

    std::array<Microphone,4> microphones_;

    // ---------------- Helper functions ----------------
    Eigen::Matrix<double, MEASUREMENT_DIM,1> calculateTDOA(
        const std::array<std::vector<double>, 4>& audioBuffers
    );

    Eigen::Matrix<double, MEASUREMENT_DIM,1> calculateExpectedTDOA(
        const Eigen::Matrix<double, STATE_DIM,1>& x_pred
    );

    Eigen::Matrix<double, MEASUREMENT_DIM, STATE_DIM> calculateJacobianH();
};
