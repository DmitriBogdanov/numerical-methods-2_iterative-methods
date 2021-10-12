#pragma once

#include <tuple> // returning multiple variables

#include "cmatrix.hpp"
#include "math_helpers.hpp"



// @return 1 => aproximate solution
// @return 2 => number of iterations
inline std::tuple<DMatrix, unsigned int> relaxation_method(const DMatrix &Diagonals, const DMatrix &b, double w, StopCondition stopCond) {
	const auto N = Diagonals.rows();

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

	std::cout << ">>> ||C|| = " << normC << "]\n";
	std::cout << ">>> ||CU|| = " << normCU << "]\n";

	// Finally, iteration
	DMatrix X(N, 1); // current X estimate
	DMatrix X0(N, 1); // previous X estimate

	size_t iterations = 0;
	fill(X0, 0.); // first estimate is zero-vector

	DMatrix buffer(N, 1);
	bool flag = false; // true => stop iteration

	//double minerr = INF;

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

	default:
		throw std::runtime_error("ERROR: Unknown stop condition");
	}
	
	do {
		++iterations;

		// Compute new X
		for (size_t i = 0; i < N; ++i) {
			// Compute X[i] = (1 - w) X0[i] + w / A[i][i] ( b[i] - SUM_0<=j<i { A[i][j] X[j] } - SUM_i+1<=j<N { A[i][j] X0[j] } )
			double sum1 = (i > 0) ? Diagonals[i][0] * X(i - 1) : 0.;
			double sum2 = (i < N - 1) ? Diagonals[i][2] * X0(i + 1) : 0.;
				// checks 'i' so we don't go out of bounds
			
			X(i) = (1. - w) * X0(i) + (b(i) - sum1 - sum2) * w / Diagonals[i][1];
		}

		// Handle stop condition
		switch (stopCond.type) {
		case StopConditionType::MAX_ESTIMATE:
			flag = false;
			break;

		case StopConditionType::MIN_ESTIMATE:
			flag = (vector_difference_norm(X, *stopCond.precise_solution) < stopCond.epsilon);
			/*if (vector_difference_norm(X, X0) < minerr) {
				minerr = vector_difference_norm(X, X0);
				std::cout << "[" << iterations << "] -> " << vector_difference_norm(X, *stopCond.precise_solution) << '\n';
			}*/
			break;

		case StopConditionType::DEFAULT:
			flag = (vector_difference_norm(X, X0) < stopCond.epsilon);
			break;

		default:
			throw std::runtime_error("ERROR: Unknown stop condition");
		}

		// Now X becomes X0
		X0 = X;

	} while (!flag && iterations < stopCond.max_iterations);

	return { X, iterations };
}
