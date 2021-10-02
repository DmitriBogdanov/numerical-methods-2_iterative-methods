#pragma once

#include <vector> // internal storage uses vector
#include <algorithm> // std::max during norm calculation
#include <type_traits> // std::is_arithmetic<T> is used to ensure matrix elements are of a numeric type
#include <iostream> // printing to console
#include <iomanip> // print formatting



// Config
constexpr size_t CMATRIX_MAX_PRINT_SIZE = 8;
constexpr std::streamsize CMATRIX_PRINT_ALIGNMENT = 14;
constexpr auto PRINT_INDENT = "   ";


// # CMatrix #
// Simplistic matrix implementation
// - stores data in a contiguous fashion
// - [i][j] style of indexation 
template<typename T>
class CMatrix {
	// Template compiles only if T is numeric
	static_assert(std::is_arithmetic<T>::value, "T must be numeric");

	size_t _rows;
	size_t _cols;
	std::vector<T> _data;

public:
	CMatrix() :
		_rows(0),
		_cols(0)
	{}

	CMatrix(size_t rows, size_t cols) :
		_rows(rows),
		_cols(cols),
		_data(rows * cols)
	{}

	// Direct indexation (k)
	T& operator() (size_t k) { return _data[k]; }
	const T& operator() (size_t k) const { return _data[k]; }

	// Conventional indexation [i][j]
	T* operator[] (size_t i) { return _data.data() + i * _cols; } // allows [i][j] style of indexation
	const T* operator[] (size_t i) const { return _data.data() + i * _cols; }

	// Getters
	size_t rows() const { return _rows; };
	size_t cols() const { return _cols; };
	size_t size() const { return _rows * _cols; }

	void print() const {
		// If any of matrix dimensions is larger than that console output is supressed
		if (_rows > CMATRIX_MAX_PRINT_SIZE || _cols > CMATRIX_MAX_PRINT_SIZE) {
			std::cout << " [ matrix output supressed due to large size ]\n";
			return;
		}

		for (size_t i = 0; i < _rows; ++i) {
			std::cout << PRINT_INDENT << "[ ";

			const auto IxC = i * _cols;
			for (size_t j = 0; j < _cols; ++j) std::cout << std::setw(CMATRIX_PRINT_ALIGNMENT) << _data[IxC + j] << " ";

			std::cout << " ]\n";
		}
	}

	// Norms
	T norm_cubic() const {
		// Cubic norm => max_i { SUM_j |matrix[i][j]|}
		T maxSum(0);

		// Go over each row calculating SUM_j |matrix[i][j]|, select max_i
		for (size_t i = 0; i < _rows; ++i) {
			T sum(0);

			const auto IxC = i * _cols;
			for (size_t j = 0; j < _cols; ++j) sum += std::abs(_data[IxC + j]);

			maxSum = std::max(maxSum, sum);
		}

		return maxSum;
	}

	T norm_octahedral() const {
		// Octahedral norm => max_i { SUM_j |matrix[i][j]|}
		T maxSum(0);

		// Go over each row calculating SUM_j |matrix[i][j]|, select max_i
		for (size_t j = 0; j < _cols; ++j) {
			T sum(0);

			for (size_t i = 0; i < _rows; ++i) sum += std::abs(_data[i * _cols + j]);

			maxSum = std::max(maxSum, sum);
		}

		return maxSum;
	}
};



// Template shortcuts
using DMatrix = CMatrix<double>;
using FMatrix = CMatrix<float>;



// # Matrix operations #
// - to avoid unnecessary allocation, all matrix operations are implemented as functions 
//   that store results into externally allocated matrices passed by reference &dest
template<typename T>
void fill(CMatrix<T> &dest, T value) {
	for (size_t k = 0; k < dest.size(); ++k) dest(k) = value;
}

template<typename T>
void fill_diagonal(CMatrix<T> &dest, T value) {
	for (size_t k = 0; k < dest.rows(); ++k) dest[k][k] = value;
}

// matrix + matrix
template<typename T>
void add(CMatrix<T> &dest, const CMatrix<T> &src1, const CMatrix<T> &src2) {
	for (size_t k = 0; k < dest.size(); ++k) dest(k) = src1(k) + src2(k);
}

// matrix - matrix
template<typename T>
void substract(CMatrix<T> &dest, const CMatrix<T> &src1, const CMatrix<T> &src2) {
	for (size_t k = 0; k < dest.size(); ++k) dest(k) = src1(k) - src2(k);
}

// matrix * matrix
template<typename T>
void multiply(CMatrix<T> &dest, const CMatrix<T> &src1, const CMatrix<T> &src2) {
	fill(dest, static_cast<T>(0));

	for (size_t i = 0; i < src1.rows(); ++i)
		for (size_t k = 0; k < src1._cols(); ++k)
			for (size_t j = 0; j < src2._cols(); ++j)
				dest[i][j] += src1[i][k] * src2[k][j];
				// note that naive loop order would be [i]->[j]->[k], swapping [k] and [j]
				// loops reduces the number of cache misses since we access contiguously
				// stored elements in the inner-most loop
}

// matrix * scalar
template<typename T>
void multiply(CMatrix<T> &dest, const CMatrix<T> &src, T value) {
	for (size_t k = 0; k < dest.size(); ++k) dest(k) = src(k) * value;
}