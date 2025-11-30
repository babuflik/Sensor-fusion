#pragma once
#include "sensormod.h"
#include <Eigen/Dense>
#include <vector>
#include <tuple>

// Least Squares
std::tuple<Sig, SensorMod> ls(const SensorMod& s, const Sig& y);

// Weighted Least Squares
std::tuple<Sig, SensorMod> wls(const SensorMod& s, const Sig& y);

// Maximum Likelihood
std::tuple<Sig, SensorMod, Eigen::MatrixXd> ml(const SensorMod& s, const Sig& y);

// Cramer-Rao Lower Bound
Sig crlb(const SensorMod& s, const Sig* y);

// 2D CRLB evaluation
Eigen::VectorXd crlb2_grid(const SensorMod& s,
                           const Sig* y,
                           const Eigen::VectorXd& x1,
                           const Eigen::VectorXd& x2,
                           const std::array<int,2>& ind = {0,1},
                           const std::string& type = "trace");

// 1D likelihood
std::tuple<Eigen::VectorXd, Eigen::VectorXd, Eigen::VectorXd> lh1(
    const SensorMod& s, const Sig& y, 
    const Eigen::VectorXd* x1 = nullptr, int ind = 0);

// 2D likelihood
std::tuple<Eigen::MatrixXd, Eigen::VectorXd, Eigen::VectorXd, Eigen::MatrixXd, Eigen::MatrixXd, Eigen::MatrixXd, Eigen::MatrixXd>
lh2(const SensorMod& s, const Sig& y,
    const Eigen::VectorXd* x1 = nullptr,
    const Eigen::VectorXd* x2 = nullptr,
    const std::vector<int>& ind = {0,1});
