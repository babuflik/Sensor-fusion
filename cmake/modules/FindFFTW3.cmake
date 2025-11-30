# Try to locate FFTW3 library manually

find_path(FFTW3_INCLUDE_DIR fftw3.h
    /usr/include
    /usr/local/include
    /opt/homebrew/include
)

find_library(FFTW3_LIBRARY NAMES fftw3
    PATHS
    /usr/lib
    /usr/local/lib
    /opt/homebrew/lib
)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(FFTW3 REQUIRED_VARS FFTW3_INCLUDE_DIR FFTW3_LIBRARY)

if(FFTW3_FOUND)
    set(FFTW3_LIBRARIES ${FFTW3_LIBRARY})
    set(FFTW3_INCLUDE_DIRS ${FFTW3_INCLUDE_DIR})

    add_library(FFTW3::fftw3 UNKNOWN IMPORTED)
    set_target_properties(FFTW3::fftw3 PROPERTIES
        IMPORTED_LOCATION ${FFTW3_LIBRARY}
        INTERFACE_INCLUDE_DIRECTORIES ${FFTW3_INCLUDE_DIR}
    )
endif()
