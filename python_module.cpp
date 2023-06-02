#include <SE3.h>
#include <SO3.h>
#include <ceres-factors/Factors.h>
#include <ceres-factors/Parameterizations.h>
#include <pybind11/eigen.h>
#include <pybind11/operators.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <sstream>

using namespace Eigen;
namespace py = pybind11;

PYBIND11_MODULE(PyCeresFactors, m)
{
    m.doc() = "Python binding module for custom Ceres factors.";

    // SO2 Factors
    m.def("SO2Parameterization", &SO2Parameterization::Create);

    // SE2 Factors
    m.def("SE2Parameterization", &SE2Parameterization::Create);

    // SO3 Factors
    m.def("SO3Parameterization", &SO3Parameterization::Create);
    m.def("SO3Factor", &SO3Factor::Create);

    // SE3 Factors
    m.def("SE3Parameterization", &SE3Parameterization::Create);
    m.def("RelSE3Factor", &RelSE3Factor::Create);

    // Specific Sensor Factors
    m.def("RangeFactor", &RangeFactor::Create);
    m.def("AltFactor", &AltFactor::Create);
    m.def("RangeBearing2DFactor", &RangeBearing2DFactor::Create);

    // Calibration Factors
    m.def("TimeSyncAttFactor", &TimeSyncAttFactor::Create);
    m.def("SO3OffsetFactor", &SO3OffsetFactor::Create);
    m.def("SE3OffsetFactor", &SE3OffsetFactor::Create);
}
