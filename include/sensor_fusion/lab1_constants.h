#ifndef LAB1_CONSTANTS_H
#define LAB1_CONSTANTS_H

#include <Eigen/Dense>

namespace SensorFusion {
    namespace Lab1 {
        // Microphone locations (x, y)
        // 8 microphones
        inline Eigen::MatrixXd getMicPositions() {
            Eigen::MatrixXd mics(8, 2);
            mics << 2.3512, 2.7546,   // Mic 1
                   -2.1742, 2.9252,   // Mic 2
                   -2.3378, -2.6652,  // Mic 3
                    2.4453, -2.7963,  // Mic 4
                   -1.3636, 3.2423,   // Mic 5
                   -1.7625, 3.1592,   // Mic 6
                   -2.5087, 2.4628,   // Mic 7
                   -2.6870, 2.0316;   // Mic 8
            return mics;
        }

        // Speed of sound (approximate)
        const double SOUND_SPEED = 343.0;
    }
}

#endif // LAB1_CONSTANTS_H
