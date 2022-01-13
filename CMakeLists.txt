cmake_minimum_required(VERSION 2.8.12)
project(PyCeresFactors)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++17")

find_package(Eigen3 REQUIRED)
find_package(pybind11 REQUIRED)
find_package(manif-geom-cpp REQUIRED)
find_package(Ceres REQUIRED)
find_package(ceres-factors REQUIRED)

include_directories(
  ${EIGEN3_INCLUDE_DIRS}
)

pybind11_add_module(PyCeresFactors python_module.cpp)
target_link_libraries(PyCeresFactors
    PRIVATE
    ceres-factors
    INTERFACE
    manif-geom-cpp
)
