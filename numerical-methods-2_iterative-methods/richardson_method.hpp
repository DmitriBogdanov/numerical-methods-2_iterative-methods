#pragma once

#include <tuple> // returning multiple variables

#include "cmatrix.hpp"
#include "stop_conditions.hpp"
#include "math_helpers.hpp"



// Richardson method requires all diagonal elements of diagonally-dominant matrix to be positive,
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
	//const double Tau = 0.0096429; // optimal Tau for variant 2

	// Multiply whole system by Tau
	multiply(A, A, Tau);
	multiply(b, b, Tau);

	std::cout << ">>> Tau = " << Tau << '\n';
}



// @return 1 => aproximate solution
// @return 2 => number of iterations
// -> stops when ||X_k+1 - X_k|| < (1-||C||)/||C||
inline std::tuple<DMatrix, unsigned int> richardson_method(const DMatrix &A, const DMatrix &b, StopCondition stopCond) {
	const auto N = A.rows();

	// Find ||C|| = ||E - Tau A|| = ||E - A||
	double normC = 0.;
	for (size_t i = 0; i < N; ++i) {
		double sum = 0.;
		for (size_t j = 0; j < i; ++j) sum += std::abs(A[i][j]);
		sum += std::abs(1. - A[i][i]);
		for (size_t j = i + 1; j < N; ++j) sum += std::abs(A[i][j]);

		normC = std::max(normC, sum);
	}

	std::cout << ">>> ||C|| = " << normC << "\n";

	// Finally, iteration
	DMatrix X(N, 1); // current X estimate
	DMatrix X0(N, 1); // previous X estimate

	size_t iterations = 0;
	fill(X0, 0.); // first estimate is zero-vector

	DMatrix buffer(N, 1);
	bool flag = false; // true => stop iteration

	// Handle stop condition
	switch (stopCond.type) {
	case StopConditionType::MAX_ESTIMATE:
		stopCond.max_iterations = std::ceil(max_iteration_estimate(stopCond.epsilon, normC, vector_difference_norm(X0, *stopCond.precise_solution)));
		break;

	case StopConditionType::MIN_ESTIMATE:
		break;

	case StopConditionType::DEFAULT:
		stopCond.epsilon = stopCond.epsilon * (1. - normC) / normC;
		break;

	case StopConditionType::ALTERNATIVE_1:
		stopCond.epsilon_0 = 1e-8;
		break;

	case StopConditionType::ALTERNATIVE_2:
		break;

	default:
		throw std::runtime_error("ERROR: Unknown stop condition");
	}

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

		// Handle stop condition
		switch (stopCond.type) {
		case StopConditionType::MAX_ESTIMATE:
			flag = false;
			break;

		case StopConditionType::MIN_ESTIMATE:
			flag = (vector_difference_norm(X, *stopCond.precise_solution) < stopCond.epsilon);
			break;

		case StopConditionType::DEFAULT:
			flag = (vector_difference_norm(X, X0) < stopCond.epsilon);
			break;

		case StopConditionType::ALTERNATIVE_1:
			flag = (vector_difference_norm(X, X0) / (stopCond.epsilon_0 + X0.norm()) < stopCond.epsilon);
			break;

		case StopConditionType::ALTERNATIVE_2:
			multiply(buffer, A, X);
			flag = (vector_difference_norm(buffer, b) < stopCond.epsilon);
			break;

		default:
			throw std::runtime_error("ERROR: Unknown stop condition");
		}
		
		// Now X becomes X0
		X0 = X;

	} while (!flag && iterations < stopCond.max_iterations);

	return { X, iterations };
}