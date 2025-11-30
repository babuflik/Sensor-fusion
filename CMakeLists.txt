cmake_minimum_required(VERSION 3.10)
project(PhoneSensorFusion)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Add the main executable
add_executable(my_app main.cpp)

# Try to find the SensorFusion package
find_package(SensorFusion QUIET)

if(SensorFusion_FOUND)
    # Link the SensorFusion library if found
    target_link_libraries(my_app PRIVATE SensorFusion::sensor_fusion)
    message(STATUS "SensorFusion library found and linked")
else()
    message(STATUS "SensorFusion not found with find_package, will rely on system paths")
    # Fallback: link with basic sensor fusion if available
    target_link_libraries(my_app PRIVATE sensor_fusion)
endif()

# Add include directories
target_include_directories(my_app PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

# Ensure we have inet_pton available (cross-platform support)
if(WIN32)
    target_link_libraries(my_app PRIVATE ws2_32)
    target_compile_definitions(my_app PRIVATE _WIN32_WINNT=0x0601)
endif()
