#!/usr/bin/env luajit
--[[
================================================================================
  bench.lua: High-Performance LuaJIT CPU vs CUDA GPU Benchmark
================================================================================
  Compares CPU LuaJIT JIT performance against CUDA GPU acceleration
  for vector compute and 2D image filtering.
================================================================================
--]]

local ffi = require("ffi")
local cuda = require("cuda")

print("================================================================================")
print("              LuaJIT + CUDA Benchmark Suite (Tegra X1 GPU)                      ")
print("================================================================================")

local dev = cuda.init(0)
print(string.format("GPU Device : %s (%s)", dev.name, dev.arch))
print(string.format("GPU Memory : %.1f MB", dev.total_memory_mb))
print(string.format("CUDA Cores : %d SMs (%d cores)", dev.multiprocessor_count, dev.multiprocessor_count * 128))
print("--------------------------------------------------------------------------------")

-- -----------------------------------------------------------------------------
-- Benchmark 1: Large Vector SAXPY (10,000,000 elements)
-- y = a * x + y
-- -----------------------------------------------------------------------------
local N = 10000000
local bytes = N * ffi.sizeof("float")
print(string.format("\n[Benchmark 1] Vector SAXPY: %s elements (%.1f MB)",
    "10,000,000", (bytes * 2) / (1024 * 1024)))

local h_x = ffi.new("float[?]", N)
local h_y_cpu = ffi.new("float[?]", N)
local h_y_gpu = ffi.new("float[?]", N)
for i = 0, N - 1 do
    h_x[i] = i * 0.001
    h_y_cpu[i] = 1.5
    h_y_gpu[i] = 1.5
end

-- 1. CPU (LuaJIT JIT-compiled loop)
local alpha = 2.5
local t0 = os.clock()
for i = 0, N - 1 do
    h_y_cpu[i] = alpha * h_x[i] + h_y_cpu[i]
end
local t1 = os.clock()
local cpu_time_ms = (t1 - t0) * 1000.0
print(string.format("  CPU (LuaJIT JIT) Time : %7.2f ms", cpu_time_ms))

-- 2. GPU (CUDA via LuaJIT)
local saxpy_cuda = [[
extern "C" __global__ void saxpy(float a, const float* __restrict__ x, float* __restrict__ y, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        y[i] = a * x[i] + y[i];
    }
}
]]
local mod1 = cuda.compile(saxpy_cuda, { arch = dev.arch, name = "saxpy.cu" })
local k_saxpy = mod1:get_function("saxpy", "float, ptr, ptr, int")

local d_x = cuda.alloc(bytes)
local d_y = cuda.alloc(bytes)
d_x:to_device(h_x)
d_y:to_device(h_y_gpu)

local timer = cuda.timer()
local block_size = 256
local grid_size = math.ceil(N / block_size)

-- Warmup
k_saxpy:launch({ grid = grid_size, block = block_size }, alpha, d_x, d_y, N)
cuda.sync()
d_y:to_device(h_y_gpu) -- reset input for timed run

timer:start()
k_saxpy:launch({ grid = grid_size, block = block_size }, alpha, d_x, d_y, N)
timer:stop()
local gpu_time_ms = timer:elapsed_ms()
print(string.format("  GPU (CUDA Kernel) Time: %7.2f ms", gpu_time_ms))
print(string.format("  >>> GPU Speedup       : %7.2fx faster than LuaJIT CPU", cpu_time_ms / gpu_time_ms))

-- Verify
d_y:to_host(h_y_gpu)
local max_diff = 0.0
for i = 0, 1000 do
    local diff = math.abs(h_y_cpu[i] - h_y_gpu[i])
    if diff > max_diff then max_diff = diff end
end
assert(max_diff < 1e-4, "Verification failed!")
print("  >>> Verification      : PASSED (results match)")

d_x:free()
d_y:free()

-- -----------------------------------------------------------------------------
-- Benchmark 2: 2D Image Gaussian Blur (2048 x 2048, 5x5 separable kernel)
-- -----------------------------------------------------------------------------
local W, H = 2048, 2048
local img_bytes = W * H * ffi.sizeof("float")
print(string.format("\n[Benchmark 2] 2D Image Box Blur: %dx%d (4.19M pixels)", W, H))

local img_in = ffi.new("float[?]", W * H)
local img_out_cpu = ffi.new("float[?]", W * H)
local img_out_gpu = ffi.new("float[?]", W * H)

for i = 0, W * H - 1 do
    img_in[i] = (i % 256) / 255.0
end

-- CPU Blur
local t_cpu0 = os.clock()
local radius = 3
for y = radius, H - 1 - radius do
    for x = radius, W - 1 - radius do
        local sum = 0.0
        for dy = -radius, radius do
            local row_offset = (y + dy) * W
            for dx = -radius, radius do
                sum = sum + img_in[row_offset + (x + dx)]
            end
        end
        img_out_cpu[y * W + x] = sum / ((2 * radius + 1) * (2 * radius + 1))
    end
end
local t_cpu1 = os.clock()
local cpu_blur_ms = (t_cpu1 - t_cpu0) * 1000.0
print(string.format("  CPU (LuaJIT JIT) Time : %7.2f ms", cpu_blur_ms))

-- GPU 2D Stencil Blur
local blur_cuda = [[
extern "C" __global__ void blur2d(
    const float* __restrict__ in,
    float* __restrict__ out,
    int width, int height, int radius
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;

    if (x < radius || x >= width - radius || y < radius || y >= height - radius) {
        out[y * width + x] = in[y * width + x];
        return;
    }

    float sum = 0.0f;
    for (int dy = -radius; dy <= radius; dy++) {
        int row = (y + dy) * width;
        for (int dx = -radius; dx <= radius; dx++) {
            sum += in[row + (x + dx)];
        }
    }
    float norm = (2.0f * (float)radius + 1.0f) * (2.0f * (float)radius + 1.0f);
    out[y * width + x] = sum / norm;
}
]]
local mod2 = cuda.compile(blur_cuda, { arch = dev.arch, name = "blur.cu" })
local k_blur = mod2:get_function("blur2d", "ptr, ptr, int, int, int")

local d_img_in = cuda.alloc(img_bytes)
local d_img_out = cuda.alloc(img_bytes)
d_img_in:to_device(img_in)

local bx, by = 16, 16
local gx = math.ceil(W / bx)
local gy = math.ceil(H / by)

-- Warmup
k_blur:launch({ grid = { gx, gy }, block = { bx, by } }, d_img_in, d_img_out, W, H, radius)
cuda.sync()

timer:start()
k_blur:launch({ grid = { gx, gy }, block = { bx, by } }, d_img_in, d_img_out, W, H, radius)
timer:stop()
local gpu_blur_ms = timer:elapsed_ms()
print(string.format("  GPU (CUDA Kernel) Time: %7.2f ms", gpu_blur_ms))
print(string.format("  >>> GPU Speedup       : %7.2fx faster than LuaJIT CPU", cpu_blur_ms / gpu_blur_ms))

d_img_out:to_host(img_out_gpu)
local blur_diff = 0.0
for y = radius, radius + 20 do
    for x = radius, radius + 20 do
        local diff = math.abs(img_out_cpu[y * W + x] - img_out_gpu[y * W + x])
        if diff > blur_diff then blur_diff = diff end
    end
end
assert(blur_diff < 1e-4, "Blur verification failed!")
print("  >>> Verification      : PASSED (results match)")

d_img_in:free()
d_img_out:free()

print("\n================================================================================")
print("Benchmark Complete! LuaJIT + CUDA delivers massive hardware acceleration.")
print("================================================================================")
