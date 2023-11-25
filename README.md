# pyceres_factors

![example workflow](https://github.com/goromal/pyceres_factors/actions/workflows/test.yml/badge.svg)

Python wrappers for custom Ceres Solver factors in the [ceres-factors](https://github.com/goromal/ceres-factors) library.

Some explanations and illustrative examples are available in my public-facing subset of [notes on optimization libraries](https://notes.andrewtorgesen.com/doku.php?id=public:autonomy:implementation:opt-libs).

## Building / Installing

This library is built with CMake. Most recently tested with the following dependencies:

- Pybind11
- Eigen 3.4.0
- ceres-solver 2.0.0
- [manif-geom-cpp](https://github.com/goromal/manif-geom-cpp)
- [ceres-factors](https://github.com/goromal/ceres-factors)

```bash
mkdir build
cd build
cmake ..
make # or make install
```

Pytest unit tests are present in the `tests/` folder.
