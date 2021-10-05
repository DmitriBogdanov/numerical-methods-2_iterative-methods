#pragma once

#include <tuple> // returning multiple variables

#include "cmatrix.hpp"
#include "math_helpers.hpp"

#ifdef _DEBUG
#define RICHARDSON_DEBUG // print more info to console when defined
#endif



//   Richardson method requires all diagonal elements of diagonally-dominant matrix to be positive,
// during preprocessing we multiply some of the equations by -1 to ensure that condition without
// changing the solutions. Also, by multiplying the whole system by Tau, we can avoid excessive
// multiplications by Tau during the iteration and use formulas as if Tau was equal to 1.
inline void richardson_preprocess(DMatrix &A, DMatrix &b) {
	const auto N = A.rows();

	// Ensure positive diagonal, finding Tau in the process
	double maxDiagonalElement = 0.;
	for (size_t i = 0; i < N; ++i) {
		if (A[i][i] < 0.) {
			for (size_t j = 0; j < N; ++j) A[i][j] *= -1.;
			b(i) *= -1.;
		}
		
		maxDiagonalElement = std::max(maxDiagonalElement, A[i][i]);
	}

	const double Tau = 1. / maxDiagonalElement;

	// Multiply whole system by Tau
	multiply(A, A, Tau);
	multiply(b, b, Tau);

#ifdef RICHARDSON_DEBUG
	std::cout
		<< ">>> Tau =\n"
		<< PRINT_INDENT << Tau << '\n';
#endif
}


// @return 1 => aproximate solution
// @return 2 => error
// @return 3 => number of iterations
// Note that richardson method changes matrices A, b as it requires certain preprocessing of the system
inline std::tuple<DMatrix, double, unsigned int> richardson_method(DMatrix &A, DMatrix &b, double epsilon, unsigned int maxIterations) {
	const auto N = A.rows();

	// Preprocess system in a way that makes it suitable for the method while also ensuring that
	// ||C|| < 1, afterwards we can use formulas as if Tau was 1
	richardson_preprocess(A, b);

	// Find ||C|| = ||E - Tau A|| = ||E - A||
	double normC = 0.;
	for (size_t i = 0; i < N; ++i) {
		double sum = 0.;
		for (size_t j = 0; j < i; ++j) sum += std::abs(A[i][j]);
		sum += std::abs(1. - A[i][i]);
		for (size_t j = i + 1; j < N; ++j) sum += std::abs(A[i][j]);

		normC = std::max(normC, sum);
	}

	#ifdef RICHARDSON_DEBUG
	std::cout
		<< ">>> ||C|| =\n"
		<< PRINT_INDENT << normC << '\n';
	#endif

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
			// From X = X0 + b - A X0 (since Tau == 1) we derive following formula
			// X[i] = X0[i] + b[i] - SUM_j A[i][j] X0[j]
			double sumI = 0.;
			for (size_t j = 0; j < N; ++j) sumI += A[i][j] * X0(j);

			X(i) = X0(i) + b(i) - sumI;
		}

		// Find ||X - X0||
		// Cubic norm => max_i { SUM_j |matrix[i][j]|}
		differenceNorm = 0.;
		for (size_t i = 0; i < N; ++i) differenceNorm = std::max(differenceNorm, std::abs(X(i) - X0(i)));

		// Now X becomes X0
		X0 = X;

		#ifdef RICHARDSON_DEBUG
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