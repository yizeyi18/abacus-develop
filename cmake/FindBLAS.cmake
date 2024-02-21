find_library(BLAS_LIBRARY
    NAMES openblas blas blis
    HINTS ${BLAS_DIR} ${LAPACK_DIR}
    PATH_SUFFIXES "lib"
)

# Handle the QUIET and REQUIRED arguments.
include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(BLAS DEFAULT_MSG BLAS_LIBRARY)

# Copy the results to the output variables and target.
if(BLAS_FOUND)
    set(BLAS_LIBRARIES ${BLAS_LIBRARY})

    if(NOT TARGET BLAS::BLAS)
        add_library(BLAS::BLAS UNKNOWN IMPORTED)
        set_target_properties(BLAS::BLAS PROPERTIES
            IMPORTED_LINK_INTERFACE_LANGUAGES "C"
            IMPORTED_LOCATION "${BLAS_LIBRARY}")
    endif()
endif()

mark_as_advanced(BLAS_LIBRARY)
