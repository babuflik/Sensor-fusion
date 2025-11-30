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
