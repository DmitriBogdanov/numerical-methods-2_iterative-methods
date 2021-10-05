# Numerical methods 2 / Iterative Methods

Contains implementations of following iterative methods for solving linear systems:

* Richardson iteration
* Jacobi method
* Gauss–Seidel method
* Successive over-relaxation for tridiagonal matrices

## Compilation

* Recommended compiler: MSVC v142
* Requires C++17 support

## Usage

Input is a .dat file, containing floating-point matrix that represents our linear system. To configure input file, output path and other parameters, place config file of the
following format into the same folder as executable:

* Line 1: [input relative path]
* Line 2: [output relative path]
* Line 3: [target precision]
* Line 4: [max iterations]

## Version history

* 00.02
    * Implemented Richardson iteration method

* 00.02
    * Bugfixes in Gauss–Seidel method
    * Implementation of successive over-relaxation for tridiagonal matrices
    * Now exceptions are caught properly

* 00.01
    * Implemented contiguous matrix class 'CMatrix<>'
    * Implementation of Jacobi method
    * Implementation of Gauss–Seidel method
    * Timing and output formatting

## License

This project is licensed under the MIT License - see the LICENSE.md file for details