cmake_minimum_required(VERSION 3.14)

# Example 1: Simple EKF
add_executable(example_ekf example_ekf.cpp)
target_link_libraries(example_ekf PRIVATE sensor_fusion)

# Example 2: Sensor modeling
add_executable(example_sensor_model example_sensor_model.cpp)
target_link_libraries(example_sensor_model PRIVATE sensor_fusion)

# Example 3: ML estimation
add_executable(example_ml_estimation example_ml_estimation.cpp)
target_link_libraries(example_ml_estimation PRIVATE sensor_fusion)

# Lab 1: Acoustic Localization
# add_executable(lab1_main lab1_main.cpp)
# target_link_libraries(lab1_main PRIVATE sensor_fusion)

# Task 2: Modeling
add_executable(task2_modeling task2_modeling.cpp)
target_link_libraries(task2_modeling PRIVATE sensor_fusion)
