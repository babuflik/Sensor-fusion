cmake_minimum_required(VERSION 3.14)
project(SensorFusion VERSION 1.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Options
option(SENSOR_FUSION_BUILD_EXAMPLES "Build example programs" ON)
option(SENSOR_FUSION_BUILD_SHARED "Build shared library" OFF)

# Dependencies
find_package(Eigen3 REQUIRED)

# Library sources
set(SENSOR_FUSION_SOURCES
    src/nl.cpp
    src/estimators.cpp
    src/utils.cpp
)

# Library headers
set(SENSOR_FUSION_HEADERS
    include/sensor_fusion/nl.h
    include/sensor_fusion/sensormod.h
    include/sensor_fusion/estimators.h
    include/sensor_fusion/sig.h
    include/sensor_fusion/pdfclass.h
    include/sensor_fusion/utils_sigsys.h
    include/sensor_fusion/ndist.h
    include/sensor_fusion/betadist.h
    include/sensor_fusion/chi2dist.h
    include/sensor_fusion/empdist.h
    include/sensor_fusion/expdist.h
    include/sensor_fusion/gammadist.h
    include/sensor_fusion/gmdist.h
    include/sensor_fusion/logndist.h
    include/sensor_fusion/ncchi2dist.h
    include/sensor_fusion/tdist.h
    include/sensor_fusion/udist.h
    include/sensor_fusion/utils.h
)

# Create library
if(SENSOR_FUSION_BUILD_SHARED)
    add_library(sensor_fusion SHARED ${SENSOR_FUSION_SOURCES})
else()
    add_library(sensor_fusion STATIC ${SENSOR_FUSION_SOURCES})
endif()

# Set library properties
set_target_properties(sensor_fusion PROPERTIES
    VERSION ${PROJECT_VERSION}
    SOVERSION 1
    PUBLIC_HEADER "${SENSOR_FUSION_HEADERS}"
)

# Include directories
target_include_directories(sensor_fusion
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
)

# Link dependencies
target_link_libraries(sensor_fusion
    PUBLIC
        Eigen3::Eigen
)

# Alias for subdirectory usage
add_library(SensorFusion::sensor_fusion ALIAS sensor_fusion)

# Examples
if(SENSOR_FUSION_BUILD_EXAMPLES)
    add_subdirectory(examples)
endif()

# Installation
include(GNUInstallDirs)

install(TARGETS sensor_fusion
    EXPORT SensorFusionTargets
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
    PUBLIC_HEADER DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/sensor_fusion
)

# Install export
install(EXPORT SensorFusionTargets
    FILE SensorFusionTargets.cmake
    NAMESPACE SensorFusion::
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/SensorFusion
)

# Create config file
include(CMakePackageConfigHelpers)

configure_package_config_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/cmake/SensorFusionConfig.cmake.in
    ${CMAKE_CURRENT_BINARY_DIR}/SensorFusionConfig.cmake
    INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/SensorFusion
)

write_basic_package_version_file(
    ${CMAKE_CURRENT_BINARY_DIR}/SensorFusionConfigVersion.cmake
    VERSION ${PROJECT_VERSION}
    COMPATIBILITY SameMajorVersion
)

install(FILES
    ${CMAKE_CURRENT_BINARY_DIR}/SensorFusionConfig.cmake
    ${CMAKE_CURRENT_BINARY_DIR}/SensorFusionConfigVersion.cmake
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/SensorFusion
)

# Export from build tree
export(EXPORT SensorFusionTargets
    FILE ${CMAKE_CURRENT_BINARY_DIR}/SensorFusionTargets.cmake
    NAMESPACE SensorFusion::
)

# Print summary
message(STATUS "SensorFusion ${PROJECT_VERSION}")
message(STATUS "  Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "  Library type: ${SENSOR_FUSION_BUILD_SHARED}")
message(STATUS "  Build examples: ${SENSOR_FUSION_BUILD_EXAMPLES}")
message(STATUS "  Install prefix: ${CMAKE_INSTALL_PREFIX}")

# -------------------------------------------------------------------
# Hum-Finder Application
# -------------------------------------------------------------------

# CMake modules path (for FindFFTW3.cmake)
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake/modules")

find_package(FFTW3 REQUIRED)
find_package(GTest REQUIRED)

# Library: humming_core
add_library(humming_core STATIC
    src/microphone.cpp
    src/app_utils.cpp
    src/tdoa_calculator.cpp
    src/localizer.cpp
    src/gcc_phat.cpp
    src/audio_file_loader.cpp
)

target_include_directories(humming_core PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_link_libraries(humming_core PUBLIC
    Eigen3::Eigen
    FFTW3::fftw3
    SensorFusion::sensor_fusion
)

# Main executable
add_executable(humming_localization
    src/main.cpp
)

target_link_libraries(humming_localization PRIVATE
    humming_core
)

# Real-time localization executable
add_executable(realtime_localizer
    src/realtime_localizer.cpp
)

target_link_libraries(realtime_localizer PRIVATE
    humming_core
    pthread
)

# Streaming example
add_executable(streaming_example
    examples/streaming_example.cpp
)

target_link_libraries(streaming_example PRIVATE
    humming_core
    pthread
)

# File-based localization
add_executable(localize_from_files
    src/localize_from_files.cpp
)

target_link_libraries(localize_from_files PRIVATE
    humming_core
)

# Multi-recording localization
add_executable(localize_multi_recording
    src/localize_multi_recording.cpp
)

target_link_libraries(localize_multi_recording PRIVATE
    humming_core
)

# Real-time microphone localization
add_executable(localize_realtime
    src/localize_realtime.cpp
)

target_link_libraries(localize_realtime PRIVATE
    humming_core
    pthread
)

# Tests
enable_testing()

add_executable(run_tests
    tests/test_localizer.cpp
    tests/test_TDOA_and_ML.cpp
)

target_link_libraries(run_tests
    PRIVATE
        humming_core
        GTest::GTest
        GTest::Main
        pthread
)

add_test(NAME LocalizerTest COMMAND run_tests)

# Toolbox test executable
add_executable(test_toolbox tests/test_toolbox.cpp)
target_link_libraries(test_toolbox
    PRIVATE
        humming_core
        GTest::GTest
        GTest::Main
        pthread
)
add_test(NAME ToolboxTest COMMAND test_toolbox)

# NL class test executable
add_executable(test_nl tests/test_nl.cpp)
target_link_libraries(test_nl
    PRIVATE
        humming_core
        GTest::GTest
        GTest::Main
        pthread
)
add_test(NAME NLTest COMMAND test_nl)

# Sig class test executable
add_executable(test_sig tests/test_sig.cpp)
target_link_libraries(test_sig
    PRIVATE
        humming_core
        GTest::GTest
        GTest::Main
        pthread
)
add_test(NAME SigTest COMMAND test_sig)

# All estimators test executable
add_executable(test_all_estimators tests/test_all_estimators.cpp)
target_link_libraries(test_all_estimators
    PRIVATE
        humming_core
        GTest::GTest
        GTest::Main
        pthread
)
add_test(NAME AllEstimatorsTest COMMAND test_all_estimators)

# Real-world drone localization test
add_executable(test_real_world_drone tests/test_real_world_drone.cpp)
target_link_libraries(test_real_world_drone
    PRIVATE
        humming_core
        GTest::GTest
        GTest::Main
        pthread
)
add_test(NAME RealWorldDroneTest COMMAND test_real_world_drone)

# Broadband/multi-frequency drone test
add_executable(test_broadband_drone tests/test_broadband_drone.cpp)
target_link_libraries(test_broadband_drone
    PRIVATE
        humming_core
        GTest::GTest
        GTest::Main
        pthread
)
add_test(NAME BroadbandDroneTest COMMAND test_broadband_drone)
