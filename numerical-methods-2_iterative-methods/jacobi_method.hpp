#pragma once

#include <tuple> // returning multiple variables

#include "cmatrix.hpp"
#include "math_helpers.hpp"

#ifdef _DEBUG
#define JACOBI_DEBUG // print more info to console when defined
#endif



// @return 1 => aproximate solution
// @return 2 => error
// @return 3 => number of iterations
inline std::tuple<DMatrix, double, unsigned int> jacobi_method(const DMatrix &A, const DMatrix &b, double epsilon, unsigned int maxIterations) {
	const auto N = A.rows();

	// Find ||C|| through C[i][j] = -A[i][j] / A[i][i] and C[i][i] = 0
	double normC = 0.;
	for (size_t i = 0; i < N; ++i) {
		double sum = 0.;
		for (size_t j = 0; j < i; ++j) sum += std::abs(A[i][j]);
		// skip i=j since C[i][i]=0
		for (size_t j = i + 1; j < N; ++j) sum += std::abs(A[i][j]);

		normC = std::max(normC, sum / A[i][i]);
	}

	// Find trueEpsilon = epsilon (1 - ||C||) / ||C|| that will be used for iteration
	const double trueEpsilon = epsilon * (1. - normC) / normC;
	
	// Finally, iteration
	DMatrix X(N, 1); // current X estimate
	DMatrix X0(N, 1); // previous X estimate
	double differenceNorm = INF; // ||X - X0||

	size_t iterations = 0;
	fill(X0, 0.); // first estimate is zero-vector

	do {
		++iterations;

		// Compute new X
		for (size_t i = 0; i < N; ++i) {
			// Compute X[i] = 1 / A[i][i] ( SUM_i!=j { -A[i][j] X0[j] } - b[i] )
			double sum = 0.;
			for (size_t j = 0; j < N; ++j) if (i != j) sum -= A[i][j] * X0(j);

			X(i) = (b(i) + sum) / A[i][i];
		}

		// Find ||X - X0||
		// Cubic norm => max_i { SUM_j |matrix[i][j]|}
		differenceNorm = 0.;
		for (size_t i = 0; i < N; ++i) differenceNorm = std::max( differenceNorm, std::abs(X(i) - X0(i)) );

		// Now X becomes X0
		X0 = X;

		#ifdef JACOBI_DEBUG
		std::cout
			<< ">>> Iteration [" << iterations << "]\n"
			<< PRINT_INDENT << "||X - X0|| =\n"
			<< PRINT_INDENT << differenceNorm << "\n"
			<< PRINT_INDENT << "X =\n";
		X.print();
		#endif
	} while (differenceNorm > trueEpsilon && iterations < maxIterations);

	return { X, differenceNorm, iterations };
}