# 3d-reconstruction-from-accidental-motion

#### Building Ceres Solver
- `mkdir ceres-bin`
- `cd ceres-bin`
- `cmake ../ceres-solver-1.11.0/`
- `make -j4`

#### Building DenseCRF package
- `cd src`
- `chmod +x src/pydensecrf_setup.sh`
- `./pydensecrf_setup.sh`
**Note**: Make sure `Cython` is installed before running the above commands.

#### Dataset Location
- Download the dataset from [here](https://umich.box.com/shared/static/bnqgx0an4v1b0ioq80sejb7rfiuku8iy.zip) and store it in ./datasets

#### TODO
- [x] KLT
- [x] Filtering Good Points
- [x] Bundle Adjustment
- [ ] CRF Energy Minimization
