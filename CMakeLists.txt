cmake_minimum_required(VERSION 3.18.2)

project(G-KWAY VERSION 1.0)

# set everything up for c++ 17 features
set(CMAKE_CXX_STANDARD 17)

# test that filesystem header actually is there and works

add_executable(g-kway main/main.cu gkway/graph.cpp gkway/graph.hpp)
set_target_properties(g-kway PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/exec")
#------------------------------------------------------------------------------
# default release build
#------------------------------------------------------------------------------

# set compilation flags
set(CMAKE_CXX_FLAGS_RELEASE "-O3")
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  message(STATUS "Setting build type to Release ")
  set(
    CMAKE_BUILD_TYPE Release
    CACHE
    STRING "Choose the type of build."
    FORCE
  )
  # Set the possible values of build type for cmake-gui
  set_property(
    CACHE
    CMAKE_BUILD_TYPE
    PROPERTY STRINGS
    "Debug" "Release" "MinSizeRel" "RelWithDebInfo"
  )

endif()

# build options
option(TF_BUILD_CUDA "Enables builds of cuda code" ON)

# Turn on the verbose
set(CMAKE_VERBOSE_MAKEFILE ON)
# Include additional language check
include(CheckLanguage)
# Library-specific variable
set(GKWAY_3RD_PARTY_DIR ${PROJECT_SOURCE_DIR}/3rd-party)


# -----------------------------------------------------------------------------
# find nvcc
# https://cliutils.gitlab.io/modern-cmake/chapters/packages/CUDA.html
# -----------------------------------------------------------------------------
if(TF_BUILD_CUDA)
  message(STATUS "Configuring CUDA ...")
  check_language(CUDA)
  if(NOT CMAKE_CUDA_COMPILER)
    message(FATAL_ERROR "\nNo CUDA compiler found")
  endif()
  enable_language(CUDA)
  if(CMAKE_CUDA_COMPILER_VERSION VERSION_LESS "11")
    message(STATUS "CMAKE_CUDA_COMPILER_VERSION: ${CMAKE_CUDA_COMPILER_VERSION}")
    message(FATAL_ERROR "\nTaskflow requires CUDA at least v11")
  endif()
endif()  
message("PROJECT_NAME:" ${PROJECT_NAME})
message("PROJECT_SOURCE_DIR:" ${PROJECT_SOURCE_DIR})
message("CMAKE_BUILD_TYPE:" ${CMAKE_BUILD_TYPE})
# -----------------------------------------------------------------------------
# must-have package include
# -----------------------------------------------------------------------------

# Enable test
include(CTest)

# -----------------------------------------------------------------------------
# include sub cmake list
# -----------------------------------------------------------------------------

# add sub directories
add_subdirectory(gkway)
add_subdirectory(main)
#add_subdirectory(unittests)

