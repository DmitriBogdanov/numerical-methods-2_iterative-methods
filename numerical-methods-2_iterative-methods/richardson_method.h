#pragma once

#include <tuple> // returning multiple variables

#include "cmatrix.hpp"
#include "math_helpers.hpp"

#ifdef _DEBUG
#define ITERATION_DEBUG // print more info to console when defined
#endif



// @return 1 => aproximate solution
// @return 2 => error
// @return 3 => number of iterations
std::tuple<DMatrix, double, unsigned int> richardson_method(const DMatrix &A, const DMatrix &b, double epsilon, unsigned int maxIterations) {
	const auto N = A.rows();

	DMatrix X(N, 1); // current X estimate
	DMatrix X0(N, 1); // previous X estimate
	double differenceNorm = INF; // ||X - X0||

	fill(X0, 0.); // first estimate is zero-vector

	// Find iteration parameter Tau such that ||C|| < 1
	// C = E - Tau A
	// Analythycaly was derived that Tau < min_i |1 / SUM_i!=j |A[i][j]|| <= 1 / max_i SUM_j |A[i][j]| satisfy us
	double maxRowSum = 0.;
	for (size_t i = 0; i < N; ++i) {
		double sum = 0.;
		for (size_t j = 0; j < N; ++j) sum += std::abs(A[i][j]);

		maxRowSum = std::max(maxRowSum, sum);
	}

	const double Tau = -1e-3 / maxRowSum; /// TEST

	// Find ||C|| = ||E - Tau A||
	double normC = 0.;
	for (size_t i = 0; i < N; ++i) {
		double sum = 0.;
		for (size_t j = 0; j < i; ++j) sum += std::abs(Tau * A[i][j]);
		sum += std::abs(1. - Tau * A[i][i]);
		for (size_t j = i + 1; j < N; ++j) sum += std::abs(Tau * A[i][j]);

		normC = std::max(normC, sum);
	}

	#ifdef ITERATION_DEBUG
	std::cout
		<< ">>> Iteration parameter (Tau) =\n"
		<< PRINT_INDENT << Tau << "\n>>> ||C|| =\n"
		<< PRINT_INDENT << normC << '\n';
	#endif

	// Find trueEpsilon = epsilon (1 - ||C||) / ||C|| that will be used for iteration
	const double trueEpsilon = epsilon * (1 - normC) / normC;

	// Finally, iteration
	size_t iterations = 0;
	do {
		++iterations;

		// Compute new X
		for (size_t i = 0; i < N; ++i) {
			// From X = C * X0 + Tau b we derive following formula
			// Compute X[i] = SUM_j C[i][j] X0[j] - Tau b[i] = SUM_j (E[i][j] - Tau A[i][j]) X0[j] - Tau b[i]
			double sum = 0.;
			for (size_t j = 0; j < i; ++j) sum -= Tau * A[i][j] * X0(j);
			sum += 1. - Tau * A[i][i] * X0(i);
			for (size_t j = i + 1; j < N; ++j) sum -= Tau * A[i][j] * X0(j);

			X(i) = sum - Tau * b(i);
		}

		// Find ||X - X0||
		// Cubic norm => max_i { SUM_j |matrix[i][j]|}
		differenceNorm = 0.;
		for (size_t i = 0; i < N; ++i) differenceNorm = std::max(differenceNorm, std::abs(X(i) - X0(i)));

		// Now X becomes X0
		X0 = X;

		#ifdef ITERATION_DEBUG
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