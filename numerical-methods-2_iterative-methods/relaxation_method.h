#pragma once

#include <tuple> // returning multiple variables

#include "cmatrix.hpp"
#include "math_helpers.hpp"

#ifdef _DEBUG
#define RELAXATION_DEBUG // print more info to console when defined
#endif



// @return 1 => aproximate solution
// @return 2 => error
// @return 3 => number of iterations
std::tuple<DMatrix, double, unsigned int> seidel_method(const DMatrix &A, const DMatrix &b, double epsilon, unsigned int maxIterations) {
	const double w = 0.5; // relaxation parameter
	
	const auto N = A.rows();

	DMatrix X(N, 1); // current X estimate
	DMatrix X0(N, 1); // previous X estimate
	double differenceNorm = INF; // ||X - X0||

	fill(X0, 0.); // first estimate is zero-vector

	size_t iterations = 0;
	do {
		++iterations;

		// Compute new X
		for (size_t i = 0; i < N; ++i) {
			// Compute X[i] = w / A[i][i] ( b[i] - SUM_0<=j<i { A[i][j] X[j] } - SUM_i+1<=j<N { A[i][j] X0[j] } ) + (1-w) X0[i]
			double sum1 = 0.;
			for (size_t j = 0; j < i; ++j) sum1 -= A[i][j] * X(j);

			double sum2 = 0.;
			for (size_t j = i + 1; j < N; ++j) sum2 -= A[i][j] * X0(j);

			X(i) = (b(i) + sum1 + sum2) * w / A[i][i] + (1. - w) * X0(i);
		}

		// Find ||X - X0||
		// Cubic norm => max_i { SUM_j |matrix[i][j]|}
		differenceNorm = 0.;
		for (size_t i = 0; i < N; ++i) differenceNorm = std::max(differenceNorm, std::abs(X(i) - X0(i)));

		// Now X becomes X0
		X0 = X;

		#ifdef RELAXATION_DEBUG
		std::cout
			<< ">>> Iteration [" << iterations << "]\n"
			<< PRINT_INDENT << "||X - X0|| =\n"
			<< PRINT_INDENT << differenceNorm << "\n"
			<< PRINT_INDENT << "X =\n";
		X.print();
		#endif
	} while (differenceNorm > epsilon && iterations < maxIterations);

	return { X, differenceNorm, iterations };
}
