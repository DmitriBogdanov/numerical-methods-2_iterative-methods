#pragma once

#include <cmath>
#include <iostream>

#include "cmatrix.hpp"



enum class StopConditionType {
	MAX_ESTIMATE = -2,
	MIN_ESTIMATE = -1,
	DEFAULT = 0, // -> stop when ||X_k - X_k-1|| < (1-||C||) / ||C||
	ALTERNATIVE_1 = 1, // -> stop when ||X_k - X_k-1|| / (||X_k-1|| + epsilon0) < epsilon
	ALTERNATIVE_2 = 2 // -> stop when ||A X_k - b|| /  < epsilon
};


struct StopCondition {
	StopConditionType type;

	double epsilon;
	size_t max_iterations;

	union {
		const DMatrix* precise_solution;
		double epsilon_0;
	};
};


std::ostream& operator<<(std::ostream& out, StopConditionType type) {
	switch (type) {
	case StopConditionType::MAX_ESTIMATE: return out << "MAX ESTIMATE";
	case StopConditionType::MIN_ESTIMATE: return out << "MIN ESTIMATE";
	case StopConditionType::DEFAULT: return out << "DEFAULT";
	case StopConditionType::ALTERNATIVE_1: return out << "ALTERNATIVE_1";
	case StopConditionType::ALTERNATIVE_2: return out << "ALTERNATIVE_2";
	default: return out << "<UNKNOWN>";
	}
}


double max_iteration_estimate(double epsilon, double q, double rho0) {
	return std::log(epsilon * (1. - q) / rho0) / std::log(q);;
}