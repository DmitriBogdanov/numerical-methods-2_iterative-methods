#include <exception>

#include <fstream> // parsing files
#include <string> // filepaths
#include <tuple> // returning multiple variables

#include "cmatrix.hpp"
#include "static_timer.hpp"

#include "richardson_method.hpp"
#include "jacobi_method.hpp"
#include "seidel_method.hpp"
#include "relaxation_method.hpp"



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
// @return 3 => precise solution
std::tuple<DMatrix, DMatrix, DMatrix> parse_system(const std::string &filepath) {
	// Open file
	std::ifstream inFile(filepath);

	// Throw if unsuccesfull
	if (!inFile.is_open()) throw std::runtime_error("Could not open file <" + filepath + ">");

	// Parse matrix size and allocate matrices
	size_t N;
	inFile >> N;
	
	DMatrix A(N, N);
	DMatrix b(N, 1);
	DMatrix sol(N, 1);

	// Parse matrix elements
	for (size_t i = 0; i < N; ++i) {
		// Fill row [i] for matrix 'A'
		for (size_t j = 0; j < N; ++j) inFile >> A[i][j];

		// Fill row [i] for column 'b'
		inFile >> b(i);

		// Fill row [i] for column 'sol'
		inFile >> sol(i);
	}

	return { A, b, sol };
}


// Method 1
void solve_richardson(const DMatrix &A, const DMatrix &b, const DMatrix &preciseSolution, StopCondition stopCond, const std::string &outputFilepath) {
	std::cout << "\n##### Method    -> Richardson iteration\n##### Norm      -> Cubic\n##### Condition -> " <<
		stopCond.type << "\n>>> Solving...\n";

	// Preprocess system in a way that makes it suitable for the method while also ensuring that
	// ||C|| < 1, afterwards we can use formulas as if Tau was 1
	auto copyOfA = A;
	auto copyOfB = b;
	richardson_preprocess(copyOfA, copyOfB);

	// Compute
	StaticTimer::start();
	const auto [solution, iterations] = richardson_method(copyOfA, copyOfB, stopCond);
	const auto elapsed = StaticTimer::elapsed();

	// error = ||preciseSolution - solution||
	const double error = vector_difference_norm(solution, preciseSolution);

	// Display
	std::cout << ">>> Solved in " << elapsed << " ms\n>>> Solution:\n";
	solution.print();

	std::cout << ">>> Error:\n" << PRINT_INDENT << error << "\n>>> Number of iterations:\n" << PRINT_INDENT <<
		iterations << '\n';

	// Save results to file
	std::ofstream outFile(outputFilepath);

	outFile << "Iterations:\n" << iterations << "\n\nError:\n" << error << "\n\nSolution:\n";
	for (size_t i = 0; i < solution.rows(); ++i) outFile << solution(i) << '\n';

	outFile.close();
}


// Method 2
void solve_jacobi(const DMatrix &A, const DMatrix &b, const DMatrix &preciseSolution, StopCondition stopCond, const std::string &outputFilepath) {
	std::cout << "\n##### Method -> Jacobi method\n##### Norm   -> Cubic\n##### Condition -> " <<
		stopCond.type << "\n>>> Solving...\n";

	// Compute
	StaticTimer::start();
	const auto [solution, iterations] = jacobi_method(A, b, stopCond);
	const auto elapsed = StaticTimer::elapsed();

	// error = ||preciseSolution - solution||
	const double error = vector_difference_norm(solution, preciseSolution);

	// Display
	std::cout << ">>> Solved in " << elapsed << " ms\n>>> Solution:\n";
	solution.print();

	std::cout << ">>> Error:\n" << PRINT_INDENT << error << "\n>>> Number of iterations:\n" << PRINT_INDENT <<
		iterations << '\n';

	// Save results to file
	std::ofstream outFile(outputFilepath);

	outFile << "Iterations:\n" << iterations << "\n\nError:\n" << error << "\n\nSolution:\n";
	for (size_t i = 0; i < solution.rows(); ++i) outFile << solution(i) << '\n';

	outFile.close();
}


// Method 3
void solve_seidel(const DMatrix &A, const DMatrix &b, const DMatrix &preciseSolution, StopCondition stopCond, const std::string &outputFilepath) {
	std::cout << "\n##### Method -> Gauss-Seidel method\n##### Norm   -> Cubic\n##### Condition -> " <<
		stopCond.type << "\n>>> Solving...\n";

	// Compute
	StaticTimer::start();
	const auto [solution, iterations] = seidel_method(A, b, stopCond);
	const auto elapsed = StaticTimer::elapsed();

	// error = ||preciseSolution - solution||
	const double error = vector_difference_norm(solution, preciseSolution);

	// Display
	std::cout << ">>> Solved in " << elapsed << " ms\n>>> Solution:\n";
	solution.print();

	std::cout << ">>> Error:\n" << PRINT_INDENT << error << "\n>>> Number of iterations:\n" << PRINT_INDENT <<
		iterations << '\n';

	// Save results to file
	std::ofstream outFile(outputFilepath);

	outFile << "Iterations:\n" << iterations << "\n\nError:\n" << error << "\n\nSolution:\n";
	for (size_t i = 0; i < solution.rows(); ++i) outFile << solution(i) << '\n';

	outFile.close();
}


// Method 4
// Generates tridiagonal matrix used for testing successive over-relaxation method
// @return 1 => A
// @return 2 => b
// @return 3 => precise solution
std::tuple<DMatrix, DMatrix, DMatrix> generate_tridiagonal_system() {
	constexpr double ai = 2, bi = 8, ci = 6;
	constexpr size_t N = 200;

	DMatrix Diagonals(N, 3);
		// 'Diagonals' is de facto not an actual NxN matrix of the system,
		// since it's tridiagonal we only allocate the necessary space
	DMatrix b(N, 1);
	fill(b, 0.);

	// Create the solution we want
	DMatrix sol(N, 1);
	for (size_t i = 0; i < N; ++i) sol(i) = (i % 10);

	for (size_t i = 0; i < N; ++i) {
		Diagonals[i][0] = ai;
		Diagonals[i][1] = bi;
		Diagonals[i][2] = ci;

		if (i > 0) b(i) += ai * sol(i - 1);
		b(i) += bi * sol(i);
		if (i < N - 1) b(i) += ci * sol(i + 1);
	}

	return { Diagonals, b, sol };
}

void solve_relaxation(const DMatrix &Diagonals, const DMatrix &b, const DMatrix &preciseSolution, double w, StopCondition stopCond, const std::string &outputFilepath) {
	std::cout << "\n##### Method -> Successive over-relaxation\n##### Norm   -> Cubic\n##### W      -> " << w
		<< "\n##### Condition -> " << stopCond.type << "\n>>> Solving...\n";

	// Compute
	StaticTimer::start();
	const auto [solution, iterations] = relaxation_method(Diagonals, b, w, stopCond);
	const auto elapsed = StaticTimer::elapsed();

	// error = ||preciseSolution - solution||
	const double error = vector_difference_norm(solution, preciseSolution);

	// Display
	std::cout << ">>> Solved in " << elapsed << " ms\n>>> Solution:\n";
	solution.print();

	std::cout << ">>> Error:\n" << PRINT_INDENT << error << "\n>>> Number of iterations:\n" << PRINT_INDENT <<
		iterations << '\n';

	// Save results to file
	std::ofstream outFile(outputFilepath);

	outFile << "Iterations:\n" << iterations << "\n\nError:\n" << error << "\n\nSolution:\n";
	for (size_t i = 0; i < solution.rows(); ++i) outFile << solution(i) << '\n';

	outFile.close();
}


int main(int argc, char** argv) {
	try {
		// Parse config
		const auto [inputFilepath, outputFilepath, precision, maxIterations] = parse_config(); // legal since C++17

		const std::string outputPath = outputFilepath.substr(0, outputFilepath.find_last_of("."));
		const std::string outputExtension = outputFilepath.substr(outputFilepath.find_last_of("."));

		std::cout << ">>> Input filepath = " << inputFilepath << "\n>>> Output filepath = " << outputFilepath
			<< "\n>>> Precision = " << precision << "\n>>> Max iteraitons = " << maxIterations << "\n\n";

		// Fill matrices from input file
		std::cout << ">>> Parsing matrices...\n";

		StaticTimer::start();
		const auto [A, b, preciseSolution] = parse_system(inputFilepath);
		const auto elapsed = StaticTimer::elapsed();

		// Print parsed matrices
		std::cout << ">>> Parsed in " << elapsed << " ms\n";

		std::cout << ">>> A(" << A.rows() << ", " << A.cols() << "):\n";
		A.print();
		std::cout << ">>> b(" << b.rows() << ", " << b.cols() << "):\n";
		b.print();
		std::cout << '\n';

		// Stop conditions
		StopCondition stopCond_max_estimate = { StopConditionType::MAX_ESTIMATE, precision, maxIterations, &preciseSolution };
		StopCondition stopCond_min_estimate = { StopConditionType::MIN_ESTIMATE, precision, maxIterations, &preciseSolution };
		StopCondition stopCond_default = { StopConditionType::DEFAULT, precision, maxIterations, NULL };
		StopCondition stopCond_alternative_1 = { StopConditionType::ALTERNATIVE_1, precision, maxIterations, NULL };
		StopCondition stopCond_alternative_2 = { StopConditionType::ALTERNATIVE_2, precision, maxIterations, NULL };

		// Method 1
		///solve_richardson(A, b, preciseSolution, stopCond_max_estimate, outputPath + "[richardson][max_estimate]" + outputExtension);
		///solve_richardson(A, b, preciseSolution, stopCond_min_estimate, outputPath + "[richardson][min_estimate]" + outputExtension);
		solve_richardson(A, b, preciseSolution, stopCond_default, outputPath + "[richardson][default]" + outputExtension);
		///solve_richardson(A, b, preciseSolution, stopCond_alternative_1, outputPath + "[richardson][alternative_1]" + outputExtension);
		///solve_richardson(A, b, preciseSolution, stopCond_alternative_2, outputPath + "[richardson][alternative_2]" + outputExtension);
		  
		// Method 2
		///solve_jacobi(A, b, preciseSolution, stopCond_max_estimate, outputPath + "[jacobi][max_estimate]" + outputExtension);
		///solve_jacobi(A, b, preciseSolution, stopCond_min_estimate, outputPath + "[jacobi][min_estimate]" + outputExtension);
		solve_jacobi(A, b, preciseSolution, stopCond_default, outputPath + "[jacobi][default]" + outputExtension);

		// Method 3
		///solve_seidel(A, b, preciseSolution, stopCond_max_estimate, outputPath + "[seidel][max_estimate]" + outputExtension);
		///solve_seidel(A, b, preciseSolution, stopCond_min_estimate, outputPath + "[seidel][min_estimate]" + outputExtension);
		solve_seidel(A, b, preciseSolution, stopCond_default, outputPath + "[seidel][default]" + outputExtension);

		// Method 4
		const auto [TridiagonalA, TridiagonalB, TridiagonalPreciseSol] = generate_tridiagonal_system();
		TridiagonalB.print();
		const StopCondition relaxationStopCond = { StopConditionType::MIN_ESTIMATE, precision, maxIterations, &TridiagonalPreciseSol };
		///solve_relaxation(TridiagonalA, TridiagonalB, TridiagonalPreciseSol, 0.5, relaxationStopCond, outputPath + "[relaxation][w=0.5]" + outputExtension);
		///solve_relaxation(TridiagonalA, TridiagonalB, TridiagonalPreciseSol, 1.0, relaxationStopCond, outputPath + "[relaxation][w=1.0]" + outputExtension);
		///solve_relaxation(TridiagonalA, TridiagonalB, TridiagonalPreciseSol, 1.5, relaxationStopCond, outputPath + "[relaxation][w=1.5]" + outputExtension);
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