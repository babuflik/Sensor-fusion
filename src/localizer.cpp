#include "localizer.h"
#include <iostream>

constexpr double SAMPLE_RATE = 44100.0; // Hz
constexpr size_t FFT_SIZE = 4096;

Localizer::Localizer(
    const std::array<Microphone, 4>& microphones,
    double dt,
    double sound_speed,
    FilterType type
) : microphones_(microphones), dt_(dt), C_(sound_speed), filter_type_(type), sample_rate_(SAMPLE_RATE)
{
    for (int i=0;i<4;++i) {
        auto pos = microphones[i].getPosition();
        mic_positions_[i] << pos[0], pos[1], pos[2];
    }

    x_.setConstant(0.5);
    P_.setIdentity(); P_.block<3,3>(0,0) *= 1.0; P_.block<3,3>(3,3) *= 0.01;

    Q_.setIdentity(); Q_.block<3,3>(0,0) *= 0.01; Q_.block<3,3>(3,3) *= 0.05;

    R_.setIdentity(); R_ *= 1e-6;

    F_.setIdentity();
    F_.block<3,3>(0,3) = Eigen::Matrix3d::Identity()*dt_;
}

// ---------------- Main filter step ----------------
std::array<double,3> Localizer::locateSource(
    const std::array<std::vector<double>, 4>& audioBuffers
)
{
    Eigen::Matrix<double, MEASUREMENT_DIM,1> z_meas = calculateTDOA(audioBuffers);

    Eigen::Matrix<double, MEASUREMENT_DIM,1> z_pred;
    Eigen::Matrix<double, MEASUREMENT_DIM, STATE_DIM> H;
    Eigen::Matrix<double, STATE_DIM, STATE_DIM> P_pred = P_;

    if (filter_type_ == FilterType::EKF1 || filter_type_ == FilterType::EKF2) {
        // --- EKF ---
        z_pred = calculateExpectedTDOA(x_);
        H = calculateJacobianH();

        Eigen::Matrix<double, STATE_DIM, MEASUREMENT_DIM> K =
            P_ * H.transpose() * (H*P_*H.transpose() + R_).inverse();

        Eigen::VectorXd y = z_meas - z_pred;
        x_ = x_ + K*y;
        P_ = P_ - K*H*P_;
        P_ = 0.5*(P_+P_.transpose());
    }
    else if (filter_type_ == FilterType::UKF) {
        // --- UKF ---
        // Simple unscented transform implementation (1D sigma points)
        int n = STATE_DIM;
        double lambda = 1.0 - n;
        std::vector<Eigen::VectorXd> sigma_pts(2*n+1);
        Eigen::MatrixXd Psqrt = P_.llt().matrixL();

        sigma_pts[0] = x_;
        for (int i=0;i<n;++i) {
            sigma_pts[i+1]   = x_ + std::sqrt(n+lambda)*Psqrt.col(i);
            sigma_pts[i+1+n] = x_ - std::sqrt(n+lambda)*Psqrt.col(i);
        }

        std::vector<Eigen::VectorXd> z_sigma(2*n+1);
        for (int i=0;i<2*n+1;++i) z_sigma[i] = calculateExpectedTDOA(sigma_pts[i]);

        // Compute mean z_pred
        z_pred.setZero();
        z_pred += 0.5/(n+lambda) * z_sigma[0]; // central weight
        for (int i=1;i<2*n+1;++i) z_pred += 1.0/(2*(n+lambda))*z_sigma[i];

        // Compute covariances
        Eigen::MatrixXd Pz = Eigen::MatrixXd::Zero(MEASUREMENT_DIM, MEASUREMENT_DIM);
        Eigen::MatrixXd Pxz = Eigen::MatrixXd::Zero(STATE_DIM, MEASUREMENT_DIM);
        for (int i=0;i<2*n+1;++i) {
            Eigen::VectorXd dz = z_sigma[i]-z_pred;
            Eigen::VectorXd dx = sigma_pts[i]-x_;
            Pz  += dz*dz.transpose()/(2*(n+lambda));
            Pxz += dx*dz.transpose()/(2*(n+lambda));
        }
        Eigen::MatrixXd K = Pxz*Pz.inverse();
        x_ = x_ + K*(z_meas - z_pred);
        P_ = P_ - K*Pz*K.transpose();
    }

    // --- Time update ---
    x_ = F_*x_;
    P_ = F_*P_*F_.transpose() + Q_;

    return {x_(0), x_(1), x_(2)};
}

// ---------------- TDOA ----------------
Eigen::Matrix<double, Localizer::MEASUREMENT_DIM,1> Localizer::calculateTDOA(
    const std::array<std::vector<double>, 4>& audioBuffers
) {
    Eigen::Matrix<double, MEASUREMENT_DIM,1> z;
    GccPhat gcc(FFT_SIZE, SAMPLE_RATE);

    std::array<std::pair<int,int>, MEASUREMENT_DIM> pairs = {{
        {0,1},{0,2},{0,3},{1,2},{1,3},{2,3}
    }};

    for (int k=0;k<MEASUREMENT_DIM;++k) {
        int i=pairs[k].first;
        int j=pairs[k].second;
        std::vector<double> s1(audioBuffers[i].begin(), audioBuffers[i].begin()+FFT_SIZE);
        std::vector<double> s2(audioBuffers[j].begin(), audioBuffers[j].begin()+FFT_SIZE);
        z(k) = gcc.calculateTDOA(s1,s2);
    }
    return z;
}

// ---------------- Expected TDOA ----------------
Eigen::Matrix<double, Localizer::MEASUREMENT_DIM,1> Localizer::calculateExpectedTDOA(
    const Eigen::Matrix<double, STATE_DIM,1>& x_pred
) {
    Eigen::Vector3d pos = x_pred.block<3,1>(0,0);
    Eigen::Matrix<double, MEASUREMENT_DIM,1> h;
    auto dist=[&](int i){return (mic_positions_[i]-pos).norm();};
    h(0)=(dist(0)-dist(1))/C_;
    h(1)=(dist(0)-dist(2))/C_;
    h(2)=(dist(0)-dist(3))/C_;
    h(3)=(dist(1)-dist(2))/C_;
    h(4)=(dist(1)-dist(3))/C_;
    h(5)=(dist(2)-dist(3))/C_;
    return h;
}

// ---------------- Jacobian ----------------
Eigen::Matrix<double, Localizer::MEASUREMENT_DIM, Localizer::STATE_DIM> Localizer::calculateJacobianH() {
    Eigen::Matrix<double, MEASUREMENT_DIM, STATE_DIM> H = Eigen::MatrixXd::Zero(MEASUREMENT_DIM, STATE_DIM);
    Eigen::Vector3d pos = x_.block<3,1>(0,0);
    auto dR_dx = [&](int i){
        const Eigen::Vector3d& m = mic_positions_[i];
        double R = (pos-m).norm();
        Eigen::RowVector3d v; v=(pos-m)/R; return v;
    };
    std::array<std::pair<int,int>, MEASUREMENT_DIM> pairs = {{
        {0,1},{0,2},{0,3},{1,2},{1,3},{2,3}
    }};
    for(int k=0;k<MEASUREMENT_DIM;++k){
        int i=pairs[k].first; int j=pairs[k].second;
        H.block<1,3>(k,0) = (dR_dx(i)-dR_dx(j))/C_;
    }
    return H;
}
