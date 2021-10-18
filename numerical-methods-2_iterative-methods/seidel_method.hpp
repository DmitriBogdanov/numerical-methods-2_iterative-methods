#pragma once

#include <tuple> // returning multiple variables

#include "cmatrix.hpp"
#include "math_helpers.hpp"



// @return 1 => aproximate solution
// @return 2 => number of iterations
inline std::tuple<DMatrix, unsigned int> seidel_method(const DMatrix &A, const DMatrix &b, StopCondition stopCond) {
	const auto N = A.rows();
	
	/// Is that requred?
	// Since Gauss-Seidel method only guarantees convergence for diagonally-dominant (or symmetrical + positive) matrices
	// We need to do pivotisation (partial) to make the equations diagonally dominant
	/*for (size_t i = 0; i < N; ++i)                    
		for (size_t k = i + 1; k < N; ++k)
			if (std::abs(A[i][i]) < std::abs(A[k][i]))
				for (size_t j = 0; j < N + 1; ++j)
					std::swap(A[i][j], A[k][j]);*/

	// Find ||C|| and ||CU|| through C[i][j] = -A[i][j] / A[i][i] and C[i][i] = 0

	double normC = 0.;
	double normCU = 0.;
	for (size_t i = 0; i < N; ++i) {
		double sumC = 0.;
		double sumCU = 0.;
		for (size_t j = 0; j < i; ++j) sumC += std::abs(A[i][j]);
		// skip i=j since C[i][i]=0
		for (size_t j = i + 1; j < N; ++j) sumC += std::abs(A[i][j]), sumCU += std::abs(A[i][j]);

		normC = std::max(normC, sumC / A[i][i]);
		normCU = std::max(normCU, sumCU / A[i][i]);
	}

	std::cout << ">>> ||C|| = " << normC << "]\n";
	std::cout << ">>> ||CU|| = " << normCU << "]\n";

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
		stopCond.epsilon = stopCond.epsilon * (1. - normC) / normCU;
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
			// Compute X[i] = 1 / A[i][i] ( b[i] - SUM_0<=j<i { A[i][j] X[j] } - SUM_i+1<=j<N { A[i][j] X0[j] } )
			double sum1 = 0.;
			for (size_t j = 0; j < i; ++j) sum1 += A[i][j] * X(j);

			double sum2 = 0.;
			for (size_t j = i + 1; j < N; ++j) sum2 += A[i][j] * X0(j);

			X(i) = (b(i) - sum1 - sum2) / A[i][i];
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