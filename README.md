# CUDA Lab (mycuda)

A C++ repository demonstrating various CUDA features, profiling, and benchmarking, utilizing `doctest` for test runner verification and `nanobench` for microbenchmarks.

## 🚀 Features Demonstrated
* **Vector Addition:** Custom GPU vector addition with grid/block configurations and CPU comparisons.
* **Reduction:** Timed block-reduction utilizing shared memory and device clock counters.
* **CUDA Dynamic Parallelism (CDP):** Recursive parent/child kernel launches demonstrating recursive quicksort and print operations directly on the GPU.
* **Error Handling & Device Queries:** Querying hardware properties (compute capability, device count, SM/cores details) and matching targeted architectures dynamically.

---

## 🛠️ Requirements & Setup

### 1. Submodules
Before compiling, ensure the submodules are cloned to resolve testing/benchmarking headers:
```bash
git submodule update --init --recursive
```

### 2. CUDA Toolkit
Ensure a version of the CUDA toolkit is installed (e.g., CUDA 10.2, 11.x, 12.x, or 13+). The build script will automatically detect the paths and set the compiler version variables.

---

## ⚙️ Compilation & Execution

This project supports release (optimized) and debug build targets.

### Standard Build (Release)
Optimized with `-O3` compilation flags for exact profiling and benchmarking:
```bash
make
```

### Debug Build
Compiles with debug symbols (`-g -G`) for host/device code debugging (disables optimizations):
```bash
make debug
```

### Run All Tests
```bash
make run
```
or for debug mode:
```bash
make run-debug
```

### Clean Up Build Output
```bash
make clean
```

---

## 🔎 Filtering Test Cases
Since the project utilizes the `doctest` framework, you can pass arguments to filter specific tests:

* **List all available test cases:**
  ```bash
  make run ARGS="-lt"
  ```
* **Run a specific test case (e.g. Quicksort):**
  ```bash
  make run ARGS="-tc=cdpSimpleQuicksort"
  ```
* **Filter tests using wildcards:**
  ```bash
  make run ARGS="-tc=*vectoradd*"
  ```

---

## 📂 Repository Structure
* [mycuda.cu](file:///home/zliu/test/mycuda/mycuda.cu): Primary CUDA implementation containing all tests and benchmarks.
* [Makefile](file:///home/zliu/test/mycuda/Makefile): Build configuration with compiler path auto-detection and version-aware architecture setup.
* `third_party/`: Submodules containing testing (`doctest`) and benchmarking (`nanobench`) libraries.
