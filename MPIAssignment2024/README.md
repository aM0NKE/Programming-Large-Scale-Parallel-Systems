# MPI Assignment 2024

This assignment involves implementing and analyzing a parallel Successive Over-Relaxation (SOR) algorithm using Julia and MPI. The project structure is designed to help you compare sequential and parallel implementations and analyze their performance.

## Project Structure

- **sor_seq.jl**: Contains the sequential implementation of the SOR algorithm.
- **sor_par.jl**: Contains the parallel implementation of the SOR algorithm (to be implemented by you).
- **sor_test.jl**: A testing suite to verify if your parallel SOR implementation works as expected.
- **job_test_laptop.jl**: Contains tests which are to be run on laptop.
- **job_test.sh**: Script to run job_test_laptop.jl on DAS-5.
- **sor_analysis.jl**: Used for performance analysis - outputs .json files with runtime information 
- **job_analysis.sh**: Script to run performance analysis on DAS-5, outputs .json files for postprocessing

## SORTEST

The SORTEST module defined in the `sor_test.jl` file, compares the results of the sequential and parallel implementations to ensure the implementation is correct. Test cases are defined using the SORTEST module in the `job_test_laptop.jl` file. 

### Usage - Laptop 

1. Ensure sor_seq.jl and sor_par.jl are in the same directory as sor_test.jl.
2. Run job_test_laptop.jl to test the implementation on the Julia REPL:
```
include("job_test_laptop.jl")
```

### Usage - DAS-5
Once you have the Assignment folder on DAS-5 you can follow the steps below to run it:

Open the MPI Assignment folder:
```
$ cd  MPIAssignment2024
```

Load the Julia and openmpi modules using the configure script:
```
$ source configure.sh
```

Run the test module :
```
$ sbatch job_test.sh
```

Run the analysis module:
```
$ sbatch job_analysis.sh
```

Note: You have to modify job_analysis.sh the script to run different configurations you desire

### Test Matrices

The SORTEST module provides a number of test matrices. 

- **test_matrix_1(N)**: A matrix with boundary values set to 1.
- **test_matrix_2(N)**: A matrix with all values set to 1.
- **test_matrix_3(N)**: A matrix with boundary values set to `(i + j) / N`.
- **test_matrix_4(N)**: A matrix where all values are set to `(i + j) / N`.

### Tests 
The `job_test_laptop.jl` file defines 10 tests which are used to compare the results of the sequential and parallel (yours) implementation. These tests cover different matrix sizes, convergence thresholds, and processor counts.

Each test uses the following parameters:
- **N**: The size of the matrix.
- **P**: The number of ranks used in the test.
- **maxiters**: The maximum number of iterations allowed for the SOR algorithm to run.
- **stopdiff**: The convergence criterion — the test stops when the difference between iterations becomes smaller than this value.
- **matrix_index**: Indicates which predefined test matrix to use.

## SORANALYSIS

The `SORANALYSIS` module is responsible for running and timing the sequential and parallel implementations of the SOR algorithm. The results are stored in `.json` files, which include performance metrics such as execution time, convergence information, and number of iterations.

### Usage

1. Ensure the `sor_seq.jl`, `sor_par.jl`, and `sor_test.jl` files are present in the same directory as `sor_analysis.jl`.
2. Run the performance analysis and generate the .json files by executing the `job_analysis.sh` script with:

```bash
sbatch job_analysis.sh
```

### Arguments

The key parameters for the performance analysis are:

- **N_values**: Array of matrix sizes to be tested.
- **P_values**: Array of processor counts to be tested. The script will execute sequential (P=1) and parallel (P>1) runs for each matrix size in `N_values`.
- **stopdiff**: Convergence threshold for stopping execution. The algorithm will terminate if the difference between iterations is smaller than this value.
- **maxiters**: Maximum number of iterations. The algorithm will stop after this many iterations if convergence is not reached.
- **matrix_id**: Index of the test matrix to be used for the experiment. The available matrix indices are the same as described in the **SORTEST** section.

### Results

For each test, a `.json` file will be generated containing the following information:

- **N**: The size of the matrix.
- **P**: The number of ranks used in the test.
- **timings**: The execution time for each run.
- **min_timings**: The minimum execution time across all runs.
- **actual_iters**: The number of iterations taken to converge.
- **actual_diff**: The difference between the last two iterations at the end of the run.

The results are saved with filenames in the format:

- **Sequential runs**: `hash(matrix_id, N, stopdiff, maxiters)_seq.json`
- **Parallel runs**: `hash(matrix_id, N, stopdiff, maxiters)_npP.json` where `P` is the number of processors used.

## Contact
If you have any questions or need further assistance with the MPI Assignment 2024, please contact Mikhail Cassar - **m.cassar@vu.nl**. 


