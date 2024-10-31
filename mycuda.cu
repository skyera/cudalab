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

#define N 10000000
#define MAX_ERR 1e-6
#define MAX_DEPTH 16
#define INSERTION_SORT 32

void vector_add(float *out, float *a, float *b, int n) {
    for (int i = 0; i < n; i++) {
        out[i] = a[i] + b[i];
    }
}

__global__ void cuda_vector_add(float *out, float *a, float *b, int n) {
    for (int i = 0; i < n; i++) {
        out[i] = a[i] + b[i];
    }
}

__global__ void cuda_vector_add2(float *out, float *a, float *b, int n) {
    int index = 0;
    int stride = 1;

    for (int i = index; i < n; i += stride) {
        out[i] = a[i] + b[i];
    }
}

__global__ void cuda_vector_add3(float *out, float *a, float *b, int n) {
    int index = threadIdx.x;
    int stride = blockDim.x;

    for (int i = index; i < n; i += stride) {
        out[i] = a[i] + b[i];
    }
}

__global__ void cuda_vector_add_grid(float *out, float *a, float *b, int n) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < n) {
        out[tid] = a[tid] + b[tid];
    }
}

void add(int n, float *x, float *y) {
    for (int i = 0; i < n; i++) {
        y[i] = x[i] + y[i];
    }
}

__global__
void cuda_add(int n, float *x, float *y) {
    for (int i = 0; i < n; ++i) {
        y[i] = x[i] + y[i];
    }
}

__global__
void cuda_add_thread(int n, float *x, float *y) {
    int index = threadIdx.x;
    int stride = blockDim.x;

    for (int i = index; i < n; i+= stride) {
        y[i] = x[i] + y[i];
    }
}

TEST_CASE("vector_add") {
    float *a, *b, *out;

    a = (float*) malloc(sizeof(float) * N);
    b = (float*) malloc(sizeof(float) * N);
    out = (float*) malloc(sizeof(float) * N);

    for (int i = 0; i < N; i++) {
        a[i] = 1.0f;
        b[i] = 2.0f;
    }
    vector_add(out, a, b, N);
    free(a);
    free(b);
    free(out);
}

TEST_CASE("cuda_vector_add") {
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
    cuda_vector_add<<<1,1>>>(d_out, d_a, d_b, N);

    cudaMemcpy(out, d_out, sizeof(float) * N, cudaMemcpyDeviceToHost);
    
    /* for (int i = 0; i < N; i++) { */
    /*     INFO("i = ", i, " out=", out[i], " a=", a[i], " b=", b[i]); */
    /*     REQUIRE(fabs(out[i] - a[i] - b[i]) < MAX_ERR); */
    /* } */
    printf("out[0] = %f\n", out[0]);
    printf("PASSED\n");

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_out);

    free(a);
    free(b);
    free(out);
}

TEST_CASE("cuda_vector_addi_mthreads") {
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
    cuda_vector_add3<<<1,256>>>(d_out, d_a, d_b, N);

    cudaMemcpy(out, d_out, sizeof(float) * N, cudaMemcpyDeviceToHost);
    
    for (int i = 0; i < N; i++) {
        INFO("i = ", i, " out=", out[i], " a=", a[i], " b=", b[i]);
        REQUIRE(fabs(out[i] - a[i] - b[i]) < MAX_ERR);
    }
    printf("out[0] = %f\n", out[0]);
    printf("PASSED\n");

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_out);

    free(a);
    free(b);
    free(out);
}

void run_cuda_add_grid()
{
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
    
    /* int block_size = 256; */
    /* int grid_size = (N + block_size) / block_size; */
    int grid_size = 2;
    int block_size = 32;
    cuda_vector_add_grid<<<grid_size,block_size>>>(d_out, d_a, d_b, N);

    cudaMemcpy(out, d_out, sizeof(float) * N, cudaMemcpyDeviceToHost);
    
    for (int i = 0; i < N; i++) {
        INFO("i = ", i, " out=", out[i], " a=", a[i], " b=", b[i]);
        // XXX
        //REQUIRE(fabs(out[i] - a[i] - b[i]) < MAX_ERR);
    }
    printf("out[0] = %f\n", out[0]);
    printf("PASSED\n");

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_out);

    free(a);
    free(b);
    free(out);
}

TEST_CASE("cuda_vector_add_grid") {
    ankerl::nanobench::Bench bench;

    bench.run("add_grid", [&] {
        run_cuda_add_grid();
    });    
}

void run_add() {
#undef N
    int N = 1 << 20;

    float *x = new float[N];
    float *y = new float[N];

    for (int i = 0; i < N; i++) {
        x[i] = 1.0f;
        y[i] = 2.0f;
    }
    add(N, x, y);

    float max_error = 0.0f;
    for (int i = 0; i < N; i++) {
        max_error = fmax(max_error, fabs(y[i]-3.0f));
    }
    //std::cout << "max error: " << max_error << std::endl;  
    
    delete [] x;
    delete [] y;
}

TEST_CASE("add") {
    ankerl::nanobench::Bench bench;

    bench.run("add", [&] {
        run_add();
    });    
}

void run_cuda_add()
{
#undef N
    int N = 1 << 20;

    float *x, *y;
    cudaMallocManaged(&x, N*sizeof(float));
    cudaMallocManaged(&y, N*sizeof(float));

    for (int i = 0; i < N; i++) {
        x[i] = 1.0f;
        y[i] = 2.0f;
    }
    cuda_add<<<1,1>>>(N, x, y);
    cudaDeviceSynchronize();

    float max_error = 0.0f;
    for (int i = 0; i < N; i++) {
        max_error = fmax(max_error, fabs(y[i]-3.0f));
    }
    //std::cout << "max error: " << max_error << std::endl;  
    
    cudaFree(x);
    cudaFree(y);
}

TEST_CASE("cuda_add") {
    ankerl::nanobench::Bench bench;
    
    bench.run("yyy", [&] {
        run_add();
    });
}

void run_add_thread()
{
#undef N
    int N = 1 << 20;

    float *x, *y;
    cudaMallocManaged(&x, N*sizeof(float));
    cudaMallocManaged(&y, N*sizeof(float));

    for (int i = 0; i < N; i++) {
        x[i] = 1.0f;
        y[i] = 2.0f;
    }
    cuda_add_thread<<<1,256>>>(N, x, y);
    cudaDeviceSynchronize();

    float max_error = 0.0f;
    for (int i = 0; i < N; i++) {
        max_error = fmax(max_error, fabs(y[i]-3.0f));
    }
    //std::cout << "max error: " << max_error << std::endl;  
    
    cudaFree(x);
    cudaFree(y);
}

TEST_CASE("cuda_add_thread") {
    ankerl::nanobench::Bench bench;
    
    bench.run("xxx", [&] {
        run_add_thread();
    });
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

TEST_CASE("clock") {
    const int NUM_THREADS = 256;
    float input[NUM_THREADS * 2];

    for (int i = 0; i < NUM_THREADS; i++) {
        input[i] = (float)i;
    }

    float *dinput = NULL;

    cudaMalloc((void **)&dinput, sizeof(float) * NUM_THREADS * 2);
}
