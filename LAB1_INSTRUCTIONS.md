# Lab 1: Localization Using a Microphone Network

**Course:** TSRT14 - Sensor Fusion  
**Original Document:** `assets/SFlab1_data/datadescription.pdf` & `assets/localization.pdf`

## 1. Introduction
This lab focuses on **Acoustic Source Localization** using a network of microphones. You will track an autonomous RC car emitting sound pulses.

### The Problem
*   **Target:** An RC car moving in a 2D plane.
*   **Signal:** Short "chirp" sound pulses (0.1s duration, ~0.5s interval).
*   **Sensors:** 8 microphones (Behringer UMC1820 sound card).
*   **Goal:** Estimate the car's position $(x, y)$ over time.

### Key Concepts
*   **TOA (Time of Arrival):** Absolute time a signal reaches a sensor. Requires known emission time $t_0$.
*   **TDOA (Time Difference of Arrival):** Relative time difference between sensors. Eliminates unknown $t_0$.
    $$ \text{TDOA}_{i,1} = t_i - t_1 = \frac{||x - m_i|| - ||x - m_1||}{c} $$

---

## 2. Data Description

### Files
All data is located in `assets/SFlab1_data/`.

| File | Description | Key Variables |
| :--- | :--- | :--- |
| `calibration-data.mat` | Stationary data for noise estimation. | `tphat` (TOA), `fs` |
| `data.mat` | Moving car experiment. | `tphat`, `mic_locations`, `position` (GT) |
| `mic_locations` | 2x8 Matrix. Row 1: $x$, Row 2: $y$. | |
| `tphat` | 8xN Matrix. Estimated TOA for each pulse. | |
| `position` | 3xN Matrix. Ground Truth (Time, x, y). | |

### Setup
*   **Microphones 1-4:** "First" setup.
*   **Microphones 5-8:** "Second" setup.
*   **Ground Truth:** Provided by Qualisys motion capture system (mm precision).

---

## 3. Tasks (Step-by-Step)

### Task 1: Sensor Calibration
**Goal:** Quantify measurement noise.
1.  Use `calibration-data.mat`.
2.  Since mics are equidistant, true TDOA should be 0.
3.  Calculate errors: $e = t_{measured} - t_{mean}$.
4.  Compute **Bias** and **Standard Deviation** ($\sigma$).
5.  Plot histograms to verify Gaussian distribution.

### Task 2: Signal Modeling
**Goal:** Define the mathematical model.
*   Define measurement equation: $y = h(x) + e$.
*   Implement this as a function (or `SensorMod` in C++).

### Task 3: Configuration Analysis
**Goal:** Understand geometry effects (GDOP).
*   Compute **CRLB (Cramér-Rao Lower Bound)** map over the area.
*   Visualize where precision is high vs. low.

### Task 4: Localization Algorithms
**Goal:** Estimate position from measurements.
Compare methods:
1.  **NLS (Non-linear Least Squares) Grid Search:** Visualize the cost function.
2.  **Gauss-Newton / ML:** Iterative optimization (fast).
3.  **TDOA Approach:** Use pairwise differences to eliminate $t_0$.

### Task 5: Tracking
**Goal:** Smooth the trajectory using motion models.
1.  **Kalman Filter (KF/EKF):** Use localization results as inputs.
2.  **Direct Filtering:** Use raw TDOA measurements in an EKF/UKF/Particle Filter.

---

## 4. Report Requirements
*   **Length:** Max 6 pages.
*   **Content:** Data gathering, Task solutions, Conclusions.
*   **Appendix:** Code (does not count towards page limit).
