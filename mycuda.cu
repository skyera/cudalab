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
#include <vector>
#include <memory>
#include <stdexcept>

template <typename T>
struct DevicePtr {
    T* ptr = nullptr;
    explicit DevicePtr(size_t count) {
        cudaError_t err = cudaMalloc(&ptr, count * sizeof(T));
        REQUIRE(err == cudaSuccess);
    }
    ~DevicePtr() {
        if (ptr) {
            cudaFree(ptr);
        }
    }
    T* get() const { return ptr; }
    DevicePtr(const DevicePtr&) = delete;
    DevicePtr& operator=(const DevicePtr&) = delete;
};

template <typename T>
struct HostPinnedPtr {
    T* ptr = nullptr;
    explicit HostPinnedPtr(size_t count) {
        cudaError_t err = cudaMallocHost((void**)&ptr, count * sizeof(T));
        REQUIRE(err == cudaSuccess);
    }
    ~HostPinnedPtr() {
        if (ptr) {
            cudaFreeHost(ptr);
        }
    }
    T* get() const { return ptr; }
    HostPinnedPtr(const HostPinnedPtr&) = delete;
    HostPinnedPtr& operator=(const HostPinnedPtr&) = delete;
};

struct CudaEvent {
    cudaEvent_t event;
    CudaEvent() {
        cudaError_t err = cudaEventCreate(&event);
        REQUIRE(err == cudaSuccess);
    }
    ~CudaEvent() {
        cudaEventDestroy(event);
    }
    cudaEvent_t get() const { return event; }
    operator cudaEvent_t() const { return event; }
    CudaEvent(const CudaEvent&) = delete;
    CudaEvent& operator=(const CudaEvent&) = delete;
};

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
      {0x80,  64},
      {0x86, 128},
      {0x87, 128},
      {0x89, 128},
      {0x90, 128},
      {0xa0, 128},
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
        out[i] = a[i] + b[i];
    }
}

TEST_CASE("cpu_vector_add") {
    std::vector<float> a(N, 1.0f);
    std::vector<float> b(N, 2.0f);
    std::vector<float> out(N, 0.0f);

    Timer timer;
    timer.start();
    cpu_vector_add(out.data(), a.data(), b.data(), N);
    timer.stop();
    printf("cpu_vector_add N %d %f seconds\n", N, timer.elapsed_seconds());
}

void do_cuda_vector_add(int n_block, int n_thread) {
    cudaError_t e;

    int num_elements = N;
    // Scale down the size for 1 block / 1 thread configuration. 
    // In debug builds (-G), 10 million elements sequentially on a single thread exceeds the OS watchdog timer.
    if (n_block == 1 && n_thread == 1) {
        num_elements = 100000;
    }

    std::vector<float> a(num_elements, 1.0f);
    std::vector<float> b(num_elements, 2.0f);
    std::vector<float> out(num_elements, 0.0f);

    DevicePtr<float> d_a(num_elements);
    DevicePtr<float> d_b(num_elements);
    DevicePtr<float> d_out(num_elements);

    e = cudaMemcpy(d_a.get(), a.data(), sizeof(float) * num_elements, cudaMemcpyHostToDevice);
    REQUIRE(e == cudaSuccess);
    e = cudaMemcpy(d_b.get(), b.data(), sizeof(float) * num_elements, cudaMemcpyHostToDevice);
    REQUIRE(e == cudaSuccess);

    Timer timer;
    timer.start();
    cuda_vector_add<<<n_block, n_thread>>>(d_out.get(), d_a.get(), d_b.get(), num_elements);
    e = cudaGetLastError();
    REQUIRE(e == cudaSuccess);
    
    e = cudaDeviceSynchronize();
    REQUIRE(e == cudaSuccess);
    timer.stop();
    printf("cuda_vector_add N: %d <<<%d, %d>>> %f: seconds\n", num_elements,
            n_block, n_thread, timer.elapsed_seconds());

    e = cudaMemcpy(out.data(), d_out.get(), sizeof(float) * num_elements, cudaMemcpyDeviceToHost);
    REQUIRE(e == cudaSuccess);
    
    for (int i = 0; i < num_elements; i++) {
        INFO("i = ", i, " out=", out[i], " a=", a[i], " b=", b[i]);
        REQUIRE(fabs(out[i] - a[i] - b[i]) < MAX_ERR);
    }
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
    cudaDeviceSynchronize();
    REQUIRE(cudaGetLastError() == cudaSuccess);
}

TEST_CASE("device_count") {
    int count = 0;
    cudaError_t e = cudaGetDeviceCount(&count);
    REQUIRE(e == cudaSuccess);
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
      {0x80, "Ampere"},
      {0x86, "Ampere"},
      {0x87, "Ampere"},
      {0x89, "Ada Lovelace"},
      {0x90, "Hopper"},
      {0xa0, "Blackwell"},
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
        << "ECCEnabled: " << devprop.ECCEnabled << "\n";
    int clockRate = 0;
    int dev_id = 0;
    if (cudaGetDevice(&dev_id) == cudaSuccess) {
        cudaDeviceGetAttribute(&clockRate, cudaDevAttrClockRate, dev_id);
    }
    std::cout << "clockRate: " << clockRate << "\n";
    std::cout << "l2CacheSize: " << devprop.l2CacheSize << "\n";
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

    HostPinnedPtr<int> a(n);
    memset(a.get(), 0, nbytes);

    DevicePtr<int> d_a(n);
    cudaError_t e = cudaMemset(d_a.get(), 255, nbytes);
    REQUIRE(e == cudaSuccess);

    // event
    CudaEvent start, stop;

    e = cudaDeviceSynchronize();
    REQUIRE(e == cudaSuccess);
    
    e = cudaEventRecord(start, 0);
    REQUIRE(e == cudaSuccess);
    
    e = cudaMemcpyAsync(d_a.get(), a.get(), nbytes, cudaMemcpyHostToDevice, 0);
    REQUIRE(e == cudaSuccess);

    dim3 threads = dim3(512, 1);
    dim3 blocks = dim3(n / threads.x, 1);
    int value = 26;
    
    increment_kernel<<<blocks, threads, 0, 0>>>(d_a.get(), value);
    e = cudaGetLastError();
    REQUIRE(e == cudaSuccess);

    e = cudaMemcpyAsync(a.get(), d_a.get(), nbytes, cudaMemcpyDeviceToHost, 0);
    REQUIRE(e == cudaSuccess);
    
    e = cudaEventRecord(stop, 0);
    REQUIRE(e == cudaSuccess);
    e = cudaEventSynchronize(stop);
    REQUIRE(e == cudaSuccess);

    float elapsed_time = 0.0f;
    e = cudaEventElapsedTime(&elapsed_time, start, stop);
    REQUIRE(e == cudaSuccess);
    printf("asyncAPI GPU elapsed time: %f ms\n", elapsed_time);
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
        s_uid = atomicAdd(&g_uids, 1);
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

    cudaError_t e = cudaDeviceSetLimit(cudaLimitDevRuntimeSyncDepth, max_depth);
    REQUIRE(e == cudaSuccess);
    cdp_kernel<<<2,2>>>(max_depth, 0, 0, -1);
    e = cudaGetLastError();
    REQUIRE(e == cudaSuccess);
    e = cudaDeviceSynchronize();
    REQUIRE(e == cudaSuccess);
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
    DevicePtr<int> d_myint(1);
    int myint = 0;
    
    run_atomicCAS<<<1,1>>>(d_myint.get());
    cudaError_t e = cudaDeviceSynchronize();
    REQUIRE(e == cudaSuccess);
    
    e = cudaMemcpy(&myint, d_myint.get(), sizeof(int), cudaMemcpyDeviceToHost);
    REQUIRE(e == cudaSuccess);
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
    if (depth >= MAX_DEPTH || right - left <= INSERTION_SORT) {
        selection_sort(data, left, right);
        return;
    }

    unsigned int *lptr = data + left;
    unsigned int *rptr = data + right;
    unsigned int pivot = data[(left + right) / 2];

    while (lptr <= rptr) {
        while (*lptr < pivot) {
            lptr++;
        }
        while (*rptr > pivot) {
            rptr--;
        }

        if (lptr <= rptr) {
            unsigned int temp = *lptr;
            *lptr = *rptr;
            *rptr = temp;
            lptr++;
            rptr--;
        }
    }

    int nright = rptr - data;
    int nleft = lptr - data;

    if (left < nright) {
        cudaStream_t s;
        cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking);
        cdp_simple_quicksort<<<1, 1, 0, s>>>(data, left, nright, depth + 1);
        cudaStreamDestroy(s);
    }
    if (nleft < right) {
        cudaStream_t s;
        cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking);
        cdp_simple_quicksort<<<1, 1, 0, s>>>(data, nleft, right, depth + 1);
        cudaStreamDestroy(s);
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
    cudaError_t e = cudaDeviceSetLimit(cudaLimitDevRuntimeSyncDepth, MAX_DEPTH);
    REQUIRE(e == cudaSuccess);
    int left = 0;
    int right = nitems - 1;
    cdp_simple_quicksort<<<1, 1>>>(data, left, right, 0);
}

TEST_CASE("cdpSimpleQuicksort") {
    cudaError_t e;
    int device_count = 0;
    
    e = cudaGetDeviceCount(&device_count);
    REQUIRE(e == cudaSuccess);
    printf("device_count %d\n", device_count);

    cudaDeviceProp devprop;
    int dev = 0;
    e = cudaGetDeviceProperties(&devprop, dev);
    REQUIRE(e == cudaSuccess);
    printf("major %d minor %d\n", devprop.major, devprop.minor);

    int num_items = 128;
    std::vector<unsigned int> h_data(num_items);
    initialize_data(h_data.data(), num_items);

    DevicePtr<unsigned int> d_data(num_items);
    e = cudaMemcpy(d_data.get(), h_data.data(), num_items * sizeof(unsigned int),
            cudaMemcpyHostToDevice);
    REQUIRE(e == cudaSuccess);

    run_qsort(d_data.get(), num_items);
    e = cudaDeviceSynchronize();
    REQUIRE(e == cudaSuccess);
    e = cudaGetLastError();
    REQUIRE(e == cudaSuccess);

    e = cudaMemcpy(h_data.data(), d_data.get(), num_items * sizeof(unsigned int),
            cudaMemcpyDeviceToHost);
    REQUIRE(e == cudaSuccess);

    for (int i = 1; i < num_items; i++) {
        REQUIRE(h_data[i - 1] <= h_data[i]);
    }
}

TEST_CASE("vectoradd") {
    int num_elements = 50000;
    size_t size = num_elements * sizeof(float);
    printf("vector addition of %d elements\n", num_elements);

    std::vector<float> h_a(num_elements);
    std::vector<float> h_b(num_elements);
    std::vector<float> h_c(num_elements);

    for (int i = 0; i < num_elements; ++i) {
        h_a[i] = rand() / (float) RAND_MAX;
        h_b[i] = rand() / (float) RAND_MAX;
    }

    DevicePtr<float> d_a(num_elements);
    DevicePtr<float> d_b(num_elements);
    DevicePtr<float> d_c(num_elements);

    cudaError_t e = cudaMemcpy(d_a.get(), h_a.data(), size, cudaMemcpyHostToDevice);
    REQUIRE(e == cudaSuccess);
    e = cudaMemcpy(d_b.get(), h_b.data(), size, cudaMemcpyHostToDevice);
    REQUIRE(e == cudaSuccess);

    int threadsPerBlock = 256;
    int blocksPerGrid = (num_elements + threadsPerBlock - 1) / threadsPerBlock;
    cuda_vector_add<<<blocksPerGrid, threadsPerBlock>>>(d_c.get(), d_a.get(), d_b.get(), num_elements);
    e = cudaGetLastError();
    REQUIRE(e == cudaSuccess);
    
    e = cudaDeviceSynchronize();
    REQUIRE(e == cudaSuccess);

    e = cudaMemcpy(h_c.data(), d_c.get(), size, cudaMemcpyDeviceToHost);
    REQUIRE(e == cudaSuccess);

    for (int i = 0; i < num_elements; ++i) {
        REQUIRE(fabs(h_c[i] - (h_a[i] + h_b[i])) < MAX_ERR);
    }
}

__global__ static void timed_reduction(const float *input, float *output,
        long long *timer) {
    extern __shared__ float shared[];
    const int tid = threadIdx.x;
    const int bid = blockIdx.x;

    if (tid == 0) timer[bid] = clock64();

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

    if (tid == 0) timer[bid + gridDim.x] = clock64();
}

TEST_CASE("clock") {
    int device_count = 0;

    cudaError_t e = cudaGetDeviceCount(&device_count);
    REQUIRE(e == cudaSuccess);
    printf("Device count: %d\n", device_count);

    int dev_id = 0;
    int compute_mode = 0;
    int major = 0;
    int minor = 0;
    e = cudaSetDevice(dev_id);
    REQUIRE(e == cudaSuccess);

    e = cudaDeviceGetAttribute(&compute_mode, cudaDevAttrComputeMode, dev_id);
    REQUIRE(e == cudaSuccess);
    e = cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, dev_id);
    REQUIRE(e == cudaSuccess);
    e = cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, dev_id);
    REQUIRE(e == cudaSuccess);
    printf("compute_mode %d major %d minor %d\n", compute_mode, major, minor);

    int num_cores = convert_smver2_cores(major, minor);
    printf("num cores: %d\n", num_cores);

    int mprocessor_count = 0;
    int clockrate = 0;
    e = cudaDeviceGetAttribute(&mprocessor_count, cudaDevAttrMultiProcessorCount,
            dev_id);
    REQUIRE(e == cudaSuccess);
    e = cudaDeviceGetAttribute(&clockrate, cudaDevAttrClockRate, dev_id);
    REQUIRE(e == cudaSuccess);
    printf("mprocessor count: %d\n", mprocessor_count);
    printf("clockrate: %d\n", clockrate);

    uint64_t compute_ref = (uint64_t) mprocessor_count * num_cores * clockrate;
    printf("compute_ref: %llu\n", (unsigned long long)compute_ref);

    const int NUM_THREADS = 256;
    const int NUM_BLOCKS = 64;
    
    long long timer[NUM_BLOCKS * 2];
    std::vector<float> input(NUM_THREADS * 2);

    for (int i = 0; i < NUM_THREADS * 2; i++) {
        input[i] = (float)i;
    }

    // alloc mem
    DevicePtr<float> dinput(NUM_THREADS * 2);
    DevicePtr<float> doutput(NUM_BLOCKS);
    DevicePtr<long long> dtimer(NUM_BLOCKS * 2);
    
    e = cudaMemcpy(dinput.get(), input.data(), sizeof(float) * NUM_THREADS * 2, 
               cudaMemcpyHostToDevice);
    REQUIRE(e == cudaSuccess);

    timed_reduction<<<NUM_BLOCKS, NUM_THREADS, sizeof(float) * 2 * NUM_THREADS>>> (dinput.get(), doutput.get(), dtimer.get());
    e = cudaGetLastError();
    REQUIRE(e == cudaSuccess);

    e = cudaMemcpy(timer, dtimer.get(), sizeof(long long) * NUM_BLOCKS * 2,
            cudaMemcpyDeviceToHost);
    REQUIRE(e == cudaSuccess);

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
    
    std::vector<float> h_A(num_elements);
    std::vector<float> h_B(num_elements);
    std::vector<float> h_C(num_elements);

    for (int i = 0; i < num_elements; ++i) {
        h_A[i] = rand()/(float)RAND_MAX;
        h_B[i] = rand()/(float)RAND_MAX;
    }

    DevicePtr<float> d_A(num_elements);
    DevicePtr<float> d_B(num_elements);
    DevicePtr<float> d_C(num_elements);

    printf("copy input data from host memory to CUDA device\n");
    err = cudaMemcpy(d_A.get(), h_A.data(), size, cudaMemcpyHostToDevice);
    REQUIRE(err == cudaSuccess);

    err = cudaMemcpy(d_B.get(), h_B.data(), size, cudaMemcpyHostToDevice);
    REQUIRE(err == cudaSuccess);

    int threadsPerBlock = 256;
    int blocksPerGrid = (num_elements + threadsPerBlock - 1) / threadsPerBlock;
    cuda_vector_add<<<blocksPerGrid, threadsPerBlock>>>(d_C.get(), d_A.get(), d_B.get(), num_elements);
    err = cudaGetLastError();
    REQUIRE(err == cudaSuccess);
    
    err = cudaDeviceSynchronize();
    REQUIRE(err == cudaSuccess);

    err = cudaMemcpy(h_C.data(), d_C.get(), size, cudaMemcpyDeviceToHost);
    REQUIRE(err == cudaSuccess);

    for (int i = 0; i < num_elements; ++i) {
        REQUIRE(fabs(h_C[i] - (h_A[i] + h_B[i])) < MAX_ERR);
    }
}

TEST_CASE("max_n_block_thread") {
    cudaDeviceProp prop;
    int device = 0;

    cudaError_t e = cudaGetDevice(&device);
    REQUIRE(e == cudaSuccess);
    e = cudaGetDeviceProperties(&prop, device);
    REQUIRE(e == cudaSuccess);
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
        printf("Device assert failed as expected, CUDA error message: %s\n", cudaGetErrorString(error));
        // Reset device to clean up the assertion state for subsequent runs/tests
        cudaDeviceReset();
    }
}

TEST_CASE("device") {
    int device_count = 0;
    cudaError_t e = cudaGetDeviceCount(&device_count);
    REQUIRE(e == cudaSuccess);
    printf("Number of CUDA devices: %d\n\n", device_count);

    for (int i = 0; i < device_count; ++i) {
        cudaDeviceProp prop;
        e = cudaGetDeviceProperties(&prop, i);
        REQUIRE(e == cudaSuccess);

        int clockRate = 0;
        cudaDeviceGetAttribute(&clockRate, cudaDevAttrClockRate, i);

        printf("Device %d: \"%s\"\n", i, prop.name);
        printf("  Compute Capability:          %d.%d\n", prop.major, prop.minor);
        printf("  Total Global Memory:         %.2f GB (%llu bytes)\n", 
               (double)prop.totalGlobalMem / (1024.0 * 1024.0 * 1024.0), 
               (unsigned long long)prop.totalGlobalMem);
        printf("  Multiprocessors (SMs):       %d\n", prop.multiProcessorCount);
        printf("  GPU Max Clock Rate:          %.0f MHz (%d kHz)\n", 
               (double)clockRate / 1000.0, clockRate);
        printf("  L2 Cache Size:               %d bytes\n", prop.l2CacheSize);
        printf("  Total Constant Memory:       %zu bytes\n", prop.totalConstMem);
        printf("  Shared Memory per Block:     %zu bytes\n", prop.sharedMemPerBlock);
        printf("  Registers per Block:         %d\n", prop.regsPerBlock);
        printf("  Warp Size:                   %d\n", prop.warpSize);
        printf("  Max Threads per Block:       %d\n", prop.maxThreadsPerBlock);
        printf("  Max Block Dimensions:        [%d, %d, %d]\n", 
               prop.maxThreadsDim[0], prop.maxThreadsDim[1], prop.maxThreadsDim[2]);
        printf("  Max Grid Dimensions:         [%d, %d, %d]\n", 
               prop.maxGridSize[0], prop.maxGridSize[1], prop.maxGridSize[2]);
        printf("\n");
    }
}

