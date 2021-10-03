#include <exception>

#include <fstream> // parsing files
#include <string> // filepaths
#include <tuple> // returning multiple variables

#include "cmatrix.hpp"
#include "static_timer.hpp"

#include "richardson_method.h"
#include "jacobi_method.h"
#include "seidel_method.h"
#include "relaxation_method.h"



constexpr auto CONFIG_PATH = "config.txt";

// Parse config and return input/output filepaths
// @return 1 => input filepath
// @return 2 => output filepath
// @return 3 => epsilon
// @return 4 => max iterations
std::tuple<std::string, std::string, double, unsigned int> parse_config() {
	std::ifstream inConfig(CONFIG_PATH);

	if (!inConfig.is_open()) throw std::runtime_error("Could not open config file.");

	std::string inputFilepath;
	std::string outputFilepath;
	double precision;
	unsigned int maxIterations;

	std::getline(inConfig, inputFilepath);
	std::getline(inConfig, outputFilepath);
	inConfig >> precision;
	inConfig >> maxIterations;

	return { inputFilepath, outputFilepath, precision, maxIterations };
}


// Parse system Ax=b from file
// @return 1 => A
// @return 2 => b
std::tuple<DMatrix, DMatrix> parse_system(const std::string &filepath) {
	// Open file
	std::ifstream inFile(filepath);

	// Throw if unsuccesfull
	if (!inFile.is_open()) throw std::runtime_error("Could not open file <" + filepath + ">");

	// Parse matrix size and allocate matrices
	size_t N;
	inFile >> N;
	
	DMatrix A(N, N);
	DMatrix b(N, 1);

	// Parse matrix elements
	for (size_t i = 0; i < N; ++i) {
		// Fill row [i] for matrix 'A'
		for (size_t j = 0; j < N; ++j) inFile >> A[i][j];

		// Fill row [i] for column 'b'
		inFile >> b(i);
	}

	return { A, b };
}


// Method 1
void solve_richardson(const DMatrix &A, const DMatrix &b, double epsilon, unsigned int maxIterations) {
	std::cout << "\n##### Method -> Richardson iteration\n##### Norm   -> Cubic\n>>> Solving...\n";

	// Compute
	StaticTimer::start();
	const auto [solution, error, iterations] = richardson_method(A, b, epsilon, maxIterations);
	const auto elapsed = StaticTimer::elapsed();

	// Display
	std::cout << ">>> Solved in " << elapsed << " ms\n>>> Solution:\n";
	solution.print();

	std::cout << ">>> Error:\n" << PRINT_INDENT << error << "\n>>> Number of iterations:\n" << PRINT_INDENT <<
		iterations << '\n';
}


// Method 2
void solve_jacobi(const DMatrix &A, const DMatrix &b, double epsilon, unsigned int maxIterations) {
	std::cout << "\n##### Method -> Jacobi method\n##### Norm   -> Cubic\n>>> Solving...\n";

	// Compute
	StaticTimer::start();
	const auto [solution, error, iterations] = jacobi_method(A, b, epsilon, maxIterations);
	const auto elapsed = StaticTimer::elapsed();

	// Display
	std::cout << ">>> Solved in " << elapsed << " ms\n>>> Solution:\n";
	solution.print();

	std::cout << ">>> Error:\n" << PRINT_INDENT << error << "\n>>> Number of iterations:\n" << PRINT_INDENT <<
		iterations << '\n';
}


// Method 3
void solve_seidel(const DMatrix &A, const DMatrix &b, double epsilon, unsigned int maxIterations) {
	std::cout << "\n##### Method -> Gauss-Seidel method\n##### Norm   -> Cubic\n>>> Solving...\n";

	// Compute
	StaticTimer::start();
	const auto [solution, error, iterations] = seidel_method(A, b, epsilon, maxIterations);
	const auto elapsed = StaticTimer::elapsed();

	// Display
	std::cout << ">>> Solved in " << elapsed << " ms\n>>> Solution:\n";
	solution.print();

	std::cout << ">>> Error:\n" << PRINT_INDENT << error << "\n>>> Number of iterations:\n" << PRINT_INDENT <<
		iterations << '\n';
}


// Method 4
// Generates tridiagonal matrix used for testing successive over-relaxation method
// @return 1 => A
// @return 2 => b
std::tuple<DMatrix, DMatrix> generate_tridiagonal_system() {
	constexpr double ai = 6, bi = 8, ci = 1;
	constexpr size_t N = 4;

	DMatrix Diagonals(N, 3);
		// 'Diagonals' is de facto not an actual NxN matrix of the system
		// since it's tridiagonal we only allocate the necessary space
	DMatrix b(N, 1);

	for (size_t i = 0; i < N; ++i) {
		Diagonals[i][0] = ai;
		Diagonals[i][1] = bi;
		Diagonals[i][2] = ci;
		b(i) = i + 1;
	}

	return { Diagonals, b };
}

void solve_relaxation(const DMatrix &Diagonals, const DMatrix &b, double epsilon, unsigned int maxIterations) {
	std::cout << "\n##### Method -> Successive over-relaxation\n##### Norm   -> Cubic\n>>> Solving...\n";

	// Compute
	StaticTimer::start();
	const auto [solution, error, iterations] = relaxation_method(Diagonals, b, epsilon, maxIterations);
	const auto elapsed = StaticTimer::elapsed();

	// Display
	std::cout << ">>> Solved in " << elapsed << " ms\n>>> Solution:\n";
	solution.print();

	std::cout << ">>> Error:\n" << PRINT_INDENT << error << "\n>>> Number of iterations:\n" << PRINT_INDENT <<
		iterations << '\n';
}


int main(int argc, char** argv) {
	try {
		// Parse config
		const auto [inputFilepath, outputFilepath, precision, maxIterations] = parse_config(); // legal since C++17

		const std::string outputPath = outputFilepath.substr(0, outputFilepath.find_last_of("."));
		const std::string outputExtension = outputFilepath.substr(outputFilepath.find_last_of("."));

		// Fill matrices from input file
		std::cout << ">>> Parsing matrices...\n";

		StaticTimer::start();
		const auto [A, b] = parse_system(inputFilepath);
		const auto elapsed = StaticTimer::elapsed();

		// Print parsed matrices
		std::cout << ">>> Parsed in " << elapsed << " ms\n";

		std::cout << ">>> A(" << A.rows() << ", " << A.cols() << "):\n";
		A.print();
		std::cout << ">>> b(" << b.rows() << ", " << b.cols() << "):\n";
		b.print();
		std::cout << '\n';

		// Method 2
		///solve_richardson(A, b, precision, maxIterations);

		// Method 2
		solve_jacobi(A, b, precision, maxIterations);

		// Method 3
		solve_seidel(A, b, precision, maxIterations);

		// Method 4
		const auto [TridiagonalA, TridiagonalB] = generate_tridiagonal_system();
		TridiagonalA.print();
		TridiagonalB.print();
		solve_relaxation(TridiagonalA, TridiagonalB, precision, maxIterations);

	}
	// If caught any errors, show error message
	catch (const std::runtime_error& err) {
		std::cerr << "RUNTIME EXCEPTION -> " << err.what() << std::endl;
	}
	catch (...) {
		std::cerr << "CAUGHT UNKNOWN EXCEPTION" << std::endl;
	}

	return 0;
}