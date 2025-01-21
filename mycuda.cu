#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest.h"
#define ANKERL_NANOBENCH_IMPLEMENT
#include "nanobench.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <iostream>
#include <chrono>
#include <sys/utsname.h>


class Timer {
public:
    void start() {
        start_time_ = std::chrono::high_resolution_clock::now();
    }

    void stop() {
        end_time_ = std::chrono::high_resolution_clock::now();
    }

    double elapsed_seconds() const {
        return std::chrono::duration<double>(end_time_ - start_time_).count();
    }
private:
    std::chrono::high_resolution_clock::time_point start_time_;
    std::chrono::high_resolution_clock::time_point end_time_;
};

#define N 10000000
#define MAX_ERR 1e-6
#define MAX_DEPTH 16
#define INSERTION_SORT 32

#define check_cuda_errors(err) __check_cuda_errors(err, __FILE__, __LINE__)

inline void __check_cuda_errors(cudaError err, const char *file, const int line)
{
    if (cudaSuccess != err) {
        fprintf(stderr, "%s(%i): CUDA Runtime API error %d: %s.\n",
                file, line, (int)err, cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
}

inline int convert_smver2_cores(int major, int minor) {
    typedef struct {
        int SM; 
        int Cores;
    } sSMtoCores;
  sSMtoCores nGpuArchCoresPerSM[] = {
      {0x30, 192},
      {0x32, 192},
      {0x35, 192},
      {0x37, 192},
      {0x50, 128},
      {0x52, 128},
      {0x53, 128},
      {0x60,  64},
      {0x61, 128},
      {0x62, 128},
      {0x70,  64},
      {0x72,  64},
      {0x75,  64},
      {-1, -1}};

    int index = 0;
    while (nGpuArchCoresPerSM[index].SM != -1) {
        if (nGpuArchCoresPerSM[index].SM == ((major << 4) + minor)) {
            return nGpuArchCoresPerSM[index].Cores;
        }
        ++index;
    }
    printf("MapSMtoCores for SM %d.%d is undefined."
           " Default to use %d Cors/SM\n",
           major, minor, nGpuArchCoresPerSM[index - 1].Cores);
    return nGpuArchCoresPerSM[index - 1].Cores;
}

void cpu_vector_add(float *out, float *a, float *b, int n) {
    for (int i = 0; i < n; i++) {
        out[i] = a[i] + b[i];
    }
}

__global__ void cuda_vector_add(float *out, float *a, float *b, int n) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int i = index; i < n; i += stride) {
        out[index] = a[index] + b[index];
    }
}

TEST_CASE("cpu_vector_add") {
    float *a, *b, *out;

    a = (float*) malloc(sizeof(float) * N);
    b = (float*) malloc(sizeof(float) * N);
    out = (float*) malloc(sizeof(float) * N);

    for (int i = 0; i < N; i++) {
        a[i] = 1.0f;
        b[i] = 2.0f;
    }
    Timer timer;
    timer.start();
    cpu_vector_add(out, a, b, N);
    timer.stop();
    printf("cpu_vector_add N %d %f seconds\n", N, timer.elapsed_seconds());
    free(a);
    free(b);
    free(out);
}

void do_cuda_vector_add(int n_block, int n_thread) {
    float *a, *b, *out;
    float *d_a, *d_b, *d_out;
    cudaError_t e;

    a = (float*) malloc(sizeof(float) * N);
    b = (float*) malloc(sizeof(float) * N);
    out = (float*) malloc(sizeof(float) * N);

    for (int i = 0; i < N; i++) {
        a[i] = 1.0f;
        b[i] = 2.0f;
    }

    e = cudaMalloc((void**)&d_a, sizeof(float) * N);
    REQUIRE(e == cudaSuccess);
    e = cudaMalloc((void**)&d_b, sizeof(float) * N);
    REQUIRE(e == cudaSuccess);
    e = cudaMalloc((void**)&d_out, sizeof(float) * N);
    REQUIRE(e == cudaSuccess);

    cudaMemcpy(d_a, a, sizeof(float) * N, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, sizeof(float) * N, cudaMemcpyHostToDevice);
    Timer timer;
    timer.start();
    cuda_vector_add<<<n_block, n_thread>>>(d_out, d_a, d_b, N);
    cudaDeviceSynchronize();
    timer.stop();
    printf("cuda_vector_add N: %d <<<%d, %d>>> %f: seconds\n", N,
            n_block, n_thread, timer.elapsed_seconds());

    cudaMemcpy(out, d_out, sizeof(float) * N, cudaMemcpyDeviceToHost);
    
    /* for (int i = 0; i < N; i++) { */
    /*     INFO("i = ", i, " out=", out[i], " a=", a[i], " b=", b[i]); */
    /*     REQUIRE(fabs(out[i] - a[i] - b[i]) < MAX_ERR); */
    /* } */
    //printf("out[0] = %f\n", out[0]);
    //printf("PASSED\n");

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_out);

    free(a);
    free(b);
    free(out);
}

TEST_CASE("cuda_vector_add_1_1") {
    do_cuda_vector_add(1, 1);
}

TEST_CASE("cuda_vector_add_1_256") {
    do_cuda_vector_add(1, 256);
}

TEST_CASE("cuda_vector_add_2_256") {
    do_cuda_vector_add(2, 256);
}

TEST_CASE("cuda_vector_add_256_256") {
    do_cuda_vector_add(256, 256);
}

TEST_CASE("cuda_vector_add_256_1024") {
    do_cuda_vector_add(256, 1024);
}

TEST_CASE("bench") {
    double d = 1.0;
    ankerl::nanobench::Bench().run("some double ops", [&] {
        d += 1.0/d;
        if (d > 5.0) {
            d-= 5.0;
        }
        ankerl::nanobench::doNotOptimizeAway(d);
    });
}

__global__ void cuda_hello()
{
    printf("Hello world from GPU\n");
}

TEST_CASE("hello") {
    cuda_hello<<<1,1>>>();
}

TEST_CASE("device_count") {
    int count = 0;
    cudaError_t e = cudaGetDeviceCount(&count);
    REQUIRE(e == cudaSuccess);
    REQUIRE(count == 1);
    std::cout << "device count: " << count << "\n";
}

TEST_CASE("compute_mode") {
    int compute_mode = -1;
    int curr_dev = 0;
    
    cudaError_t e = cudaDeviceGetAttribute(&compute_mode,
            cudaDevAttrComputeMode, curr_dev);
    REQUIRE(e == cudaSuccess);
    std::cout << "compute mode: " << compute_mode << "\n";
}

const char* get_arch_name(int major, int minor) {
    typedef struct {
        int sm;
        const char* name;
    } ArchName;

    ArchName arch_names[] = {
      {0x30, "Kepler"},
      {0x32, "Kepler"},
      {0x35, "Kepler"},
      {0x37, "Kepler"},
      {0x50, "Maxwell"},
      {0x52, "Maxwell"},
      {0x53, "Maxwell"},
      {0x60, "Pascal"},
      {0x61, "Pascal"},
      {0x62, "Pascal"},
      {0x70, "Volta"},
      {0x72, "Xavier"},
      {0x75, "Turing"},
      {-1, "Graphics Device"}};
        
    int i;
    for (i = 0; arch_names[i].sm != -1; ++i) {
        if (arch_names[i].sm == ((major << 4) + minor)) {
            return arch_names[i].name;
        }
    }

    printf("dev %d.%d is undefined\n", major, minor);
    return arch_names[i].name;
}

TEST_CASE("major_minor") {
    int major = 0;
    int minor = 0;
    int curr_dev = 0;
    
    cudaError_t e = cudaDeviceGetAttribute(&major,
            cudaDevAttrComputeCapabilityMajor, curr_dev);
    REQUIRE(e == cudaSuccess);
    std::cout << "major: " << major << "\n";

    e = cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor,
            curr_dev);
    REQUIRE(e == cudaSuccess);
    std::cout << "minor: " << minor << "\n";
    
    const char* name = get_arch_name(major, minor);
    std::cout << name << "\n";

    auto err = cudaGetLastError();
    std::cout << "last cuda error: " << err << "\n";
}

TEST_CASE("getdevice") {
    int dev=0;
    
    cudaError_t e = cudaGetDevice(&dev);
    REQUIRE(e == cudaSuccess);
    std::cout << "dev " << dev << "\n";
}

void print_devprop(const cudaDeviceProp& devprop) {
    std::cout << "name: " << devprop.name << "\n"
        << "ECCEnabled: " << devprop.ECCEnabled << "\n"
        << "clockRate: " << devprop.clockRate << "\n"
        << "l2CacheSize: " << devprop.l2CacheSize << "\n";
}

TEST_CASE("devprop") {
    int dev = 0;
    cudaDeviceProp devprop;

    cudaError_t e = cudaGetDeviceProperties(&devprop, dev);
    REQUIRE(e == cudaSuccess);
    print_devprop(devprop);
}

__global__ void increment_kernel(int *g_data, int inc_value)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    g_data[idx] = g_data[idx] + inc_value;
}

TEST_CASE("asyncAPI") {
    int n = 16 * 1024 * 1024;
    int nbytes = n * sizeof(int);
    int *a = 0;

    cudaError_t e = cudaMallocHost((void**)&a, nbytes);
    REQUIRE(e == cudaSuccess);
    memset(a, 0, nbytes);

    int *d_a = 0;
    e = cudaMalloc((void**)&d_a, nbytes);
    REQUIRE(e == cudaSuccess);
    e = cudaMemset(d_a, 255, nbytes);
    REQUIRE(e == cudaSuccess);

    // event
    cudaEvent_t start, stop;
    e = cudaEventCreate(&start);
    REQUIRE(e == cudaSuccess);
    e = cudaEventCreate(&stop);
    REQUIRE(e == cudaSuccess);

    e = cudaDeviceSynchronize();
    REQUIRE(e == cudaSuccess);
    
    cudaMemcpyAsync(d_a, a, nbytes, cudaMemcpyHostToDevice, 0);

    dim3 threads = dim3(512, 1);
    dim3 blocks = dim3(n / threads.x, 1);
    int value = 26;
    
    increment_kernel<<<blocks, threads, 0, 0>>>(d_a, value);
    cudaMemcpyAsync(a, d_a, nbytes, cudaMemcpyDeviceToHost, 0);

    e = cudaFreeHost(a);
    REQUIRE(e == cudaSuccess);
    e = cudaFree(d_a);
    REQUIRE(e == cudaSuccess);
}

__device__ double atomic_add(double* address, double val)
{
    unsigned long long int* address_as_ull =
        (unsigned long long int*) address;
    unsigned long long int old = *address_as_ull, assumed;

    do {
        assumed = old;
        old = atomicCAS(address_as_ull, assumed,
                        __double_as_longlong(val +
                        __longlong_as_double(assumed)));
    } while (assumed != old);

    return __longlong_as_double(old);
}

__device__ void print_info(int depth, int thread, int uid, int parent_uid)
{
    if (threadIdx.x == 0) {
        if (depth == 0) {
            printf("BLOCK %d launched by the host\n", uid);
        } else {
            char buffer[32];

            for (int i = 0; i < depth; ++i) {
                buffer[3*i+0] = '|';
                buffer[3*i+1] = ' ';
                buffer[3*i+2] = ' ';
            }
            buffer[3*depth] = '\0';
            printf("%sBLOCK %d launched by thread %d of block %d\n", buffer,
                    uid, thread, parent_uid);
        }
    }
    __syncthreads();
}


__device__ int g_uids = 0;

__global__ void cdp_kernel(int max_depth, int depth, int thread,
        int parent_uid)
{
    __shared__ int s_uid;

    if (threadIdx.x == 0) {
        s_uid = atomic_add((double*)&g_uids, 1);
    }

    __syncthreads();
    print_info(depth, thread, s_uid, parent_uid);

    if (++depth >= max_depth) {
        return;
    }
    cdp_kernel<<<gridDim.x, blockDim.x>>>(max_depth, depth, threadIdx.x, s_uid);
}

TEST_CASE("cdpSimplePrint") {
    int num_blocks = 2;
    int sum = 2;
    int max_depth = 2;

    for (int i = 1; i < max_depth; ++i) {
        num_blocks *= 4;
        printf("+%d", num_blocks);
        sum += num_blocks;
    }
    printf("=%d blocks are launched(%d from the GPU)\n", sum, sum-2);

    cudaDeviceSetLimit(cudaLimitDevRuntimeSyncDepth, max_depth);
    cdp_kernel<<<2,2>>>(max_depth, 0, 0, -1);
    cudaError_t e = cudaGetLastError();
    REQUIRE(e == cudaSuccess);
    cudaDeviceSynchronize();
}


__device__ int g_int = 2;

__global__ void run_atomicCAS(int *d_myint)
{
    atomicCAS(&g_int, 2, 3);
    __syncthreads();
    *d_myint = g_int;
    printf("hello %d %d\n", g_int, *d_myint);
}

TEST_CASE("atomicCAS") {
    int* d_myint;
    int myint;
    
    cudaMalloc(&d_myint, sizeof(int));
    run_atomicCAS<<<1,1>>>(d_myint);
    cudaDeviceSynchronize();
    
    cudaMemcpy(&myint, d_myint, sizeof(int), cudaMemcpyDeviceToHost);
    printf("%d\n", myint);
}

__device__ void selection_sort(unsigned int *data, int left, int right)
{
    for (int i = left; i <= right; ++i) {
        unsigned min_val = data[i];
        int min_idx = i;

        for (int j = i+1; j <= right; ++j) {
            unsigned val_j = data[j];

            if (val_j < min_val) {
                min_idx = j;
                min_val = val_j;
            }
        }

        if (i != min_idx) {
            data[min_idx] = data[i];
            data[i] = min_val;
        }
    }
}

__global__ void cdp_simple_quicksort(unsigned int *data, int left, int right,
        int depth)
{
    if (depth >= MAX_DEPTH || right -left <= INSERTION_SORT) {
        selection_sort(data, left, right);
        return;
    }

    unsigned int *lptr = data + left;
    unsigned int *rptr = data + right;
    unsigned int pivot = data[(left+right)/2];

    while (lptr <= rptr) {

    }
}

void initialize_data(unsigned int *dst, unsigned int nitems)
{
    srand(2047);

    for (unsigned int i = 0; i < nitems; i++) {
        dst[i] = rand() % nitems;
    }
}

void run_qsort(unsigned int *data, unsigned int nitems)
{
    cudaDeviceSetLimit(cudaLimitDevRuntimeSyncDepth, MAX_DEPTH);
    int left = 0;
    int right = nitems - 1;

}

TEST_CASE("cdpSimpleQuicksort") {
    cudaError_t e;
    int device_count = 0;
    
    e = cudaGetDeviceCount(&device_count);
    REQUIRE(e == cudaSuccess);
    printf("device_count %d\n", device_count);

    cudaDeviceProp devprop;
    int dev = 0;
    cudaGetDeviceProperties(&devprop, dev);
    printf("major %d minor %d\n", devprop.major, devprop.minor);

    int num_items= 128;
    unsigned int *h_data = 0;
    h_data = (unsigned int*) malloc(num_items * sizeof(unsigned int));
    initialize_data(h_data, num_items);

    unsigned int *d_data = 0;
    e = cudaMalloc((void**)&d_data, num_items * sizeof(unsigned int));
    REQUIRE(e == cudaSuccess);
    e = cudaMemcpy(d_data, h_data, num_items * sizeof(unsigned int),
            cudaMemcpyHostToDevice);
    REQUIRE(e == cudaSuccess);
    cudaFree(d_data);
    free(h_data);
}

TEST_CASE("vectoradd") {
    int num_elements = 50000;
    size_t size = num_elements * sizeof(float);
    printf("vector addition of %d elements\n", num_elements);

    float *h_a = (float*) malloc(size);
    float *h_b = (float*) malloc(size);
    float *h_c = (float*) malloc(size);

    for (int i = 0; i < num_elements; ++i) {
        h_a[i] = rand() / (float) RAND_MAX;
        h_b[i] = rand() / (float) RAND_MAX;
    }

    float *d_a = NULL;
    cudaError_t e = cudaSuccess;

    e = cudaMalloc((void**)&d_a, size);
    if (e != cudaSuccess) {
        printf("failed to allocate device mem: %s\n", cudaGetErrorString(e));
    }
    REQUIRE(e == cudaSuccess);
}

__global__ static void timed_reduction(const float *input, float *output,
        clock_t *timer) {
    extern __shared__ float shared[];
    const int tid = threadIdx.x;
    const int bid = blockIdx.x;

    if (tid == 0) timer[bid] = clock();

    shared[tid] = input[tid];
    shared[tid + blockDim.x] = input[tid + blockDim.x];

    for (int d = blockDim.x; d > 0; d /= 2) {
        __syncthreads();

        if (tid < d) {
            float f0 = shared[tid];
            float f1 = shared[tid + d];

            if (f1 < f0) {
                shared[tid] = f1;
            }
        }
    }

    if (tid == 0) output[bid] = shared[0];

    __syncthreads();

    if (tid == 0) timer[bid + gridDim.x] = clock();
}

TEST_CASE("clock") {
    int device_count = 0;

    cudaGetDeviceCount(&device_count);
    printf("Device count: %d\n", device_count);

    int dev_id = 0;
    int compute_mode = 0;
    int major = 0;
    int minor = 0;
    cudaSetDevice(dev_id);

    cudaDeviceGetAttribute(&compute_mode, cudaDevAttrComputeMode, dev_id);
    cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, dev_id);
    cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, dev_id);
    printf("compute_mode %d major %d minor %d\n", compute_mode, major, minor);

    int num_cores = convert_smver2_cores(major, minor);
    printf("num cores: %d\n", num_cores);

    int mprocessor_count = 0;
    int clockrate = 0;
    cudaDeviceGetAttribute(&mprocessor_count, cudaDevAttrMultiProcessorCount,
            dev_id);
    cudaDeviceGetAttribute(&clockrate, cudaDevAttrClockRate, dev_id);
    printf("mprocessor count: %d\n", mprocessor_count);
    printf("clockrate: %d\n", clockrate);

    uint64_t compute_ref = (uint64_t) mprocessor_count * num_cores * clockrate;

    const int NUM_THREADS = 256;
    const int NUM_BLOCKS = 64;
    
    clock_t timer[NUM_BLOCKS * 2];
    float input[NUM_THREADS * 2];
    clock_t *dtimer = NULL;
    float *dinput = NULL;
    float *doutput = NULL;

    for (int i = 0; i < NUM_THREADS * 2; i++) {
        input[i] = (float)i;
    }

    // alloc mem
    cudaMalloc((void **)&dinput, sizeof(float) * NUM_THREADS * 2);
    cudaMalloc((void **)&doutput, sizeof(float) * NUM_BLOCKS);
    cudaMalloc((void **)&dtimer, sizeof(clock_t) * NUM_BLOCKS * 2);
    
    cudaMemcpy(dinput, input, sizeof(float) * NUM_THREADS * 2, 
               cudaMemcpyHostToDevice);
    timed_reduction<<<NUM_BLOCKS, NUM_THREADS, sizeof(float) * 2 * NUM_THREADS>>> (dinput, doutput, dtimer);

    cudaMemcpy(timer, dtimer, sizeof(clock_t) * NUM_BLOCKS * 2,
            cudaMemcpyDeviceToHost);

    cudaFree(dinput);
    cudaFree(doutput);
    cudaFree(dtimer);

    long double avg_elpased_clocks = 0;
    for (int i = 0; i < NUM_BLOCKS; i++) {
        avg_elpased_clocks += (long double) (timer[i + NUM_BLOCKS] - timer[i]);
    }

    avg_elpased_clocks = avg_elpased_clocks / NUM_BLOCKS;
    printf("Average clocks/block = %Lf\n", avg_elpased_clocks);
}

TEST_CASE("s_vectorAdd") {
    cudaError_t err = cudaSuccess;
    int num_elements = 50000;
    size_t size = num_elements * sizeof(float);
    float *h_A = (float *)malloc(size);
    float *h_B = (float *)malloc(size);
    float *h_C = (float *)malloc(size);

    if (h_A == NULL || h_B == NULL || h_C == NULL) {
        fprintf(stderr, "Failed to allocate host vectors!\n");
        exit(EXIT_FAILURE);
    }

    for (int i = 0; i < num_elements; ++i) {
        h_A[i] = rand()/(float)RAND_MAX;
        h_B[i] = rand()/(float)RAND_MAX;
    }

    float *d_A = NULL;
    err = cudaMalloc((void **)&d_A, size);
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to allocate device vector A "
                "(error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    float *d_B = NULL;
    err = cudaMalloc((void **)&d_B, size);
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to allocate device vector B "
                "(error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    float *d_C = NULL;
    err = cudaMalloc((void **)&d_C, size);
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to allocate device vector C "
                "(error code %s)!]n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    printf("copy input data from host memory to CUDA device\n");
    err = cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed top copy vector A from host to device "
                "(error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    err = cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);
}

TEST_CASE("max_n_block_thread") {
    cudaDeviceProp prop;
    int device;

    cudaGetDevice(&device);
    cudaGetDeviceProperties(&prop, device);
    std::cout << "Device Name: " << prop.name << std::endl;
    std::cout << "Max Threads per Block: " << prop.maxThreadsPerBlock << std::endl;
    std::cout << "Max Threads Dimension (x, y, z): "
        << prop.maxThreadsDim[0] << ", "
        << prop.maxThreadsDim[1] << ", "
        << prop.maxThreadsDim[2] << std::endl;
    std::cout << "Max Grid Size (x, y, z):"
        << prop.maxGridSize[0] << ", "
        << prop.maxGridSize[1] << ", "
        << prop.maxGridSize[2] << std::endl;
}

__global__ void test_kernel(int num) {
    int gtid = blockIdx.x * blockDim.x + threadIdx.x;
    assert(gtid < num);
}

TEST_CASE("simple_assert") {
    utsname os_type;
    uname(&os_type);

    printf("os type release=%s version %s\n", os_type.release, os_type.version);
    
    int n_blocks = 2;
    int n_threads = 32;
    dim3 dim_grid(n_blocks);
    dim3 dim_block(n_threads);

    test_kernel<<<dim_grid, dim_block>>>(60);
    printf("\nBegin assert\n");
    cudaError_t error = cudaDeviceSynchronize();
    printf("\nEnd assert\n");

    if (error == cudaErrorAssert) {
        printf("Device assert failed as expected, "
               "CUDA error message: %s\n", cudaGetErrorString(error));
    }
}
