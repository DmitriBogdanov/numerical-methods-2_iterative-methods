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
inline std::tuple<DMatrix, double, unsigned int> relaxation_method(const DMatrix &Diagonals, const DMatrix &b, double epsilon, unsigned int maxIterations) {
	const auto N = Diagonals.rows();

	const double w = 0.5; // relaxation parameter

	// Find ||C|| and ||CU|| through C[i][j] = -A[i][j] / A[i][i] and C[i][i] = 0
	double normC = 0.;
	double normCU = 0.;
	for (size_t i = 0; i < N; ++i) {
		// Abuse the fact that matrix is tridiagonal
		double sumC = std::abs(Diagonals[i][0]) + std::abs(Diagonals[i][2]);
		double sumCU = std::abs(Diagonals[i][2]);

		normC = std::max(normC, sumC / Diagonals[i][1]);
		normCU = std::max(normCU, sumCU / Diagonals[i][1]);
	}

	// Find trueEpsilon = epsilon (1 - ||C||) / ||CU|| that will be used for iteration
	const double trueEpsilon = epsilon * (1 - normC) / normCU;

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
			// Compute X[i] = (1 - w) X0[i] + w / A[i][i] ( b[i] - SUM_0<=j<i { A[i][j] X[j] } - SUM_i+1<=j<N { A[i][j] X0[j] } )
			double sum1 = (i > 0) ? -Diagonals[i][0] * X(i - 1) : 0.;
			double sum2 = (i < N - 1) ? -Diagonals[i][2] * X0(i + 1) : 0.;
				// checks 'i' so we don't go out of bounds
			
			X(i) = (1. - w) * X0(i) + (b(i) + sum1 + sum2) * w / Diagonals[i][1];
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
	} while (differenceNorm > trueEpsilon && iterations < maxIterations);

	return { X, differenceNorm, iterations };
}
