#----------------------------------------------------------------
# Generated CMake target import file for configuration "Release".
#----------------------------------------------------------------

# Commands may need to know the format version.
set(CMAKE_IMPORT_FILE_VERSION 1)

# Import target "svs::svs_runtime" for configuration "Release"
set_property(TARGET svs::svs_runtime APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(svs::svs_runtime PROPERTIES
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/lib/libsvs_runtime.so.0.3.0"
  IMPORTED_SONAME_RELEASE "libsvs_runtime.so.0"
  )

list(APPEND _cmake_import_check_targets svs::svs_runtime )
list(APPEND _cmake_import_check_files_for_svs::svs_runtime "${_IMPORT_PREFIX}/lib/libsvs_runtime.so.0.3.0" )

# Commands beyond this point should not need to know the version.
set(CMAKE_IMPORT_FILE_VERSION)
