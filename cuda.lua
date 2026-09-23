--[[
================================================================================
  cuda.lua: High-Performance CUDA Driver API + NVRTC Wrapper for LuaJIT
================================================================================
  Features:
  - Zero C wrapper needed: pure LuaJIT FFI binding to libcuda.so and libnvrtc.so
  - Runtime JIT compilation of raw CUDA C/C++ strings via NVRTC
  - Automatic compute capability detection and arch flag configuration
  - Zero-allocation typed kernel launch wrapper
  - Clean memory management (cuda.alloc / Buffer) with automatic GC finalizers
  - High-precision GPU event profiling timers
================================================================================
--]]

local ffi = require("ffi")

-- -----------------------------------------------------------------------------
-- 1. C Declarations for CUDA Driver API and NVRTC
-- -----------------------------------------------------------------------------
ffi.cdef[[
typedef int CUresult;
typedef int CUdevice;
typedef struct CUctx_st* CUcontext;
typedef struct CUmod_st* CUmodule;
typedef struct CUfunc_st* CUfunction;
typedef struct CUevent_st* CUevent;
typedef struct CUstream_st* CUstream;
typedef uintptr_t CUdeviceptr;

CUresult cuInit(unsigned int Flags);
CUresult cuDeviceGet(CUdevice *device, int ordinal);
CUresult cuDeviceGetName(char *name, int len, CUdevice dev);
CUresult cuDeviceGetAttribute(int *pi, int attrib, CUdevice dev);
CUresult cuDeviceTotalMem(size_t *bytes, CUdevice dev);
CUresult cuCtxCreate(CUcontext *pctx, unsigned int flags, CUdevice dev);
CUresult cuCtxDestroy(CUcontext ctx);
CUresult cuCtxSynchronize(void);

CUresult cuMemAlloc(CUdeviceptr *dptr, size_t bytesize);
CUresult cuMemFree(CUdeviceptr dptr);
CUresult cuMemcpyHtoD(CUdeviceptr dstDevice, const void *srcHost, size_t ByteCount);
CUresult cuMemcpyDtoH(void *dstHost, CUdeviceptr srcDevice, size_t ByteCount);
CUresult cuMemsetD8(CUdeviceptr dstDevice, unsigned char uc, size_t N);
CUresult cuMemsetD32(CUdeviceptr dstDevice, unsigned int ui, size_t N);

CUresult cuModuleLoadData(CUmodule *module, const void *image);
CUresult cuModuleUnload(CUmodule hmod);
CUresult cuModuleGetFunction(CUfunction *hfunc, CUmodule hmod, const char *name);

CUresult cuLaunchKernel(CUfunction f,
                        unsigned int gridDimX, unsigned int gridDimY, unsigned int gridDimZ,
                        unsigned int blockDimX, unsigned int blockDimY, unsigned int blockDimZ,
                        unsigned int sharedMemBytes, CUstream hStream, void **kernelParams, void **extra);

CUresult cuEventCreate(CUevent *phEvent, unsigned int Flags);
CUresult cuEventRecord(CUevent hEvent, CUstream hStream);
CUresult cuEventSynchronize(CUevent hEvent);
CUresult cuEventElapsedTime(float *pMilliseconds, CUevent hStart, CUevent hEnd);
CUresult cuEventDestroy(CUevent hEvent);

CUresult cuGetErrorName(CUresult error, const char **pStr);
CUresult cuGetErrorString(CUresult error, const char **pStr);

typedef int nvrtcResult;
typedef struct _nvrtcProgram *nvrtcProgram;

const char *nvrtcGetErrorString(nvrtcResult result);
nvrtcResult nvrtcCreateProgram(nvrtcProgram *prog, const char *src, const char *name, int numHeaders, const char * const *headers, const char * const *includeNames);
nvrtcResult nvrtcDestroyProgram(nvrtcProgram *prog);
nvrtcResult nvrtcCompileProgram(nvrtcProgram prog, int numOptions, const char * const *options);
nvrtcResult nvrtcGetPTXSize(nvrtcProgram prog, size_t *ptxSizeRet);
nvrtcResult nvrtcGetPTX(nvrtcProgram prog, char *ptx);
nvrtcResult nvrtcGetProgramLogSize(nvrtcProgram prog, size_t *logSizeRet);
nvrtcResult nvrtcGetProgramLog(nvrtcProgram prog, char *log);
]]

-- -----------------------------------------------------------------------------
-- 2. Library Loading
-- -----------------------------------------------------------------------------
local function try_load(names)
    for _, name in ipairs(names) do
        local ok, lib = pcall(ffi.load, name)
        if ok then return lib end
    end
    return nil
end

local libcuda = try_load({
    "cuda",
    "libcuda.so.1",
    "/usr/lib/aarch64-linux-gnu/tegra/libcuda.so.1",
    "/usr/lib/aarch64-linux-gnu/libcuda.so",
    "/usr/lib/x86_64-linux-gnu/libcuda.so",
})
if not libcuda then
    error("Failed to load CUDA driver library (libcuda.so)!")
end

local libnvrtc = try_load({
    "nvrtc",
    "libnvrtc.so.10.2",
    "libnvrtc.so.11.0",
    "libnvrtc.so.12",
    "/usr/local/cuda-10.2/targets/aarch64-linux/lib/libnvrtc.so",
    "/usr/local/cuda/lib64/libnvrtc.so",
    "/usr/local/cuda/lib/libnvrtc.so",
})
if not libnvrtc then
    error("Failed to load NVRTC library (libnvrtc.so)!")
end

-- -----------------------------------------------------------------------------
-- 3. Error Handling Helpers
-- -----------------------------------------------------------------------------
local function check_cu(err, msg)
    if err ~= 0 then
        local err_str = ffi.new("const char*[1]")
        libcuda.cuGetErrorString(err, err_str)
        local err_name = ffi.new("const char*[1]")
        libcuda.cuGetErrorName(err, err_name)
        local s = err_str[0] ~= nil and ffi.string(err_str[0]) or "unknown"
        local n = err_name[0] ~= nil and ffi.string(err_name[0]) or "unknown"
        error(string.format("CUDA error [%s]: %s (%s, code %d)", msg or "call", s, n, err), 2)
    end
end

local function check_nvrtc(err, msg, prog)
    if err ~= 0 then
        local log_msg = ""
        if prog then
            local sz = ffi.new("size_t[1]")
            libnvrtc.nvrtcGetProgramLogSize(prog, sz)
            if sz[0] > 1 then
                local log = ffi.new("char[?]", sz[0])
                libnvrtc.nvrtcGetProgramLog(prog, log)
                log_msg = "\n--- NVRTC Compiler Log ---\n" .. ffi.string(log) .. "\n-------------------------"
            end
        end
        local err_str = libnvrtc.nvrtcGetErrorString(err)
        local s = err_str ~= nil and ffi.string(err_str) or "unknown"
        error(string.format("NVRTC error [%s]: %s%s", msg or "call", s, log_msg), 2)
    end
end

-- -----------------------------------------------------------------------------
-- 4. CUDA Device & Context Management
-- -----------------------------------------------------------------------------
local cuda = {}
local context = nil
local active_dev = nil
local device_info = nil

-- CUDA Device attribute constants
local CU_DEVICE_ATTRIBUTE = {
    MAX_THREADS_PER_BLOCK = 1,
    MAX_BLOCK_DIM_X = 2,
    MAX_BLOCK_DIM_Y = 3,
    MAX_BLOCK_DIM_Z = 4,
    MAX_GRID_DIM_X = 5,
    MAX_GRID_DIM_Y = 6,
    MAX_GRID_DIM_Z = 7,
    TOTAL_CONSTANT_MEMORY = 9,
    WARP_SIZE = 10,
    CLOCK_RATE = 13,
    MULTIPROCESSOR_COUNT = 16,
    COMPUTE_CAPABILITY_MAJOR = 75,
    COMPUTE_CAPABILITY_MINOR = 76,
}

local function query_attr(dev, attr_id)
    local val = ffi.new("int[1]")
    check_cu(libcuda.cuDeviceGetAttribute(val, attr_id, dev), "cuDeviceGetAttribute")
    return val[0]
end

function cuda.init(dev_id)
    dev_id = dev_id or 0
    check_cu(libcuda.cuInit(0), "cuInit")

    local dev = ffi.new("CUdevice[1]")
    check_cu(libcuda.cuDeviceGet(dev, dev_id), "cuDeviceGet")
    active_dev = dev[0]

    local name_buf = ffi.new("char[256]")
    check_cu(libcuda.cuDeviceGetName(name_buf, 256, active_dev), "cuDeviceGetName")

    local total_bytes = ffi.new("size_t[1]")
    check_cu(libcuda.cuDeviceTotalMem(total_bytes, active_dev), "cuDeviceTotalMem")

    local major = query_attr(active_dev, CU_DEVICE_ATTRIBUTE.COMPUTE_CAPABILITY_MAJOR)
    local minor = query_attr(active_dev, CU_DEVICE_ATTRIBUTE.COMPUTE_CAPABILITY_MINOR)

    device_info = {
        id = dev_id,
        name = ffi.string(name_buf),
        major = major,
        minor = minor,
        arch = string.format("compute_%d%d", major, minor),
        total_memory_bytes = tonumber(total_bytes[0]),
        total_memory_mb = tonumber(total_bytes[0]) / (1024 * 1024),
        multiprocessor_count = query_attr(active_dev, CU_DEVICE_ATTRIBUTE.MULTIPROCESSOR_COUNT),
        warp_size = query_attr(active_dev, CU_DEVICE_ATTRIBUTE.WARP_SIZE),
        max_threads_per_block = query_attr(active_dev, CU_DEVICE_ATTRIBUTE.MAX_THREADS_PER_BLOCK),
        clock_rate_khz = query_attr(active_dev, CU_DEVICE_ATTRIBUTE.CLOCK_RATE),
    }

    local pctx = ffi.new("CUcontext[1]")
    check_cu(libcuda.cuCtxCreate(pctx, 0, active_dev), "cuCtxCreate")
    context = pctx[0]

    return device_info
end

function cuda.get_device_info()
    if not device_info then
        cuda.init(0)
    end
    return device_info
end

function cuda.sync()
    check_cu(libcuda.cuCtxSynchronize(), "cuCtxSynchronize")
end

-- -----------------------------------------------------------------------------
-- 5. Memory Management (Buffer Object)
-- -----------------------------------------------------------------------------
local Buffer = {}
Buffer.__index = Buffer

function cuda.alloc(bytes_or_type, count)
    if not context then cuda.init(0) end

    local bytes
    local elem_type = nil
    if type(bytes_or_type) == "string" then
        elem_type = bytes_or_type
        bytes = ffi.sizeof(elem_type) * (count or 1)
    else
        bytes = tonumber(bytes_or_type)
    end

    local dptr = ffi.new("CUdeviceptr[1]")
    check_cu(libcuda.cuMemAlloc(dptr, bytes), "cuMemAlloc")

    local buf = setmetatable({
        dptr = dptr[0],
        size = bytes,
        elem_type = elem_type,
        count = count or 1,
        freed = false,
    }, Buffer)

    -- Register GC finalizer on a dummy anchor
    local anchor = ffi.new("CUdeviceptr[1]", dptr[0])
    ffi.gc(anchor, function(p)
        if not buf.freed then
            libcuda.cuMemFree(p[0])
            buf.freed = true
        end
    end)
    buf._anchor = anchor

    return buf
end

function Buffer:to_device(host_data, bytes)
    assert(not self.freed, "Buffer already freed")
    local size = bytes or self.size
    local host_ptr

    if type(host_data) == "table" and self.elem_type then
        -- Convert Lua table to C array
        local c_arr = ffi.new(self.elem_type .. "[?]", #host_data)
        for i = 1, #host_data do
            c_arr[i - 1] = host_data[i]
        end
        host_ptr = c_arr
    elseif type(host_data) == "cdata" then
        host_ptr = host_data
    elseif type(host_data) == "string" then
        host_ptr = host_data
        size = math.min(size, #host_data)
    else
        error("Unsupported host_data type: " .. type(host_data))
    end

    check_cu(libcuda.cuMemcpyHtoD(self.dptr, host_ptr, size), "cuMemcpyHtoD")
    return self
end

function Buffer:to_host(dst_ptr, bytes)
    assert(not self.freed, "Buffer already freed")
    local size = bytes or self.size
    assert(type(dst_ptr) == "cdata", "dst_ptr must be a cdata pointer/array")
    check_cu(libcuda.cuMemcpyDtoH(dst_ptr, self.dptr, size), "cuMemcpyDtoH")
    return dst_ptr
end

function Buffer:free()
    if not self.freed then
        check_cu(libcuda.cuMemFree(self.dptr), "cuMemFree")
        self.freed = true
        ffi.gc(self._anchor, nil)
    end
end

function Buffer:device_ptr()
    return self.dptr
end

-- -----------------------------------------------------------------------------
-- 6. Kernel Object & Parameter Packing
-- -----------------------------------------------------------------------------
local TYPE_MAPPINGS = {
    ptr       = "CUdeviceptr[1]",
    pointer   = "CUdeviceptr[1]",
    deviceptr = "CUdeviceptr[1]",
    int       = "int32_t[1]",
    int32     = "int32_t[1]",
    uint      = "uint32_t[1]",
    uint32    = "uint32_t[1]",
    float     = "float[1]",
    double    = "double[1]",
    int64     = "int64_t[1]",
    uint64    = "uint64_t[1]",
    short     = "int16_t[1]",
    int16     = "int16_t[1]",
    byte      = "uint8_t[1]",
    uchar     = "uint8_t[1]",
    int8      = "int8_t[1]",
    size_t    = "size_t[1]",
}

local Kernel = {}
Kernel.__index = Kernel

local function parse_signature(sig)
    if not sig then return nil end
    local types = {}
    if type(sig) == "string" then
        for t in sig:gmatch("[%w_%*]+") do
            table.insert(types, t:lower())
        end
    elseif type(sig) == "table" then
        for _, t in ipairs(sig) do
            table.insert(types, tostring(t):lower())
        end
    end
    return types
end

local function make_kernel(hfunc, name, signature)
    local param_types = parse_signature(signature)
    local k = setmetatable({
        hfunc = hfunc,
        name = name,
        param_types = param_types,
    }, Kernel)

    if param_types then
        local n = #param_types
        k.param_count = n
        k.holders = {}
        k.arg_ptrs = ffi.new("void*[" .. n .. "]")
        for i = 1, n do
            local ptype = param_types[i]
            local ctype = TYPE_MAPPINGS[ptype] or (ptype .. "[1]")
            local holder = ffi.new(ctype)
            k.holders[i] = holder
            k.arg_ptrs[i - 1] = holder
        end
    end

    return k
end

function Kernel:launch(config, ...)
    local args = { ... }
    local grid = config.grid or { 1, 1, 1 }
    local block = config.block or { 1, 1, 1 }
    local shared = config.shared or 0
    local stream = config.stream or nil

    local gx = type(grid) == "table" and (grid[1] or grid.x or 1) or grid
    local gy = type(grid) == "table" and (grid[2] or grid.y or 1) or 1
    local gz = type(grid) == "table" and (grid[3] or grid.z or 1) or 1

    local bx = type(block) == "table" and (block[1] or block.x or 1) or block
    local by = type(block) == "table" and (block[2] or block.y or 1) or 1
    local bz = type(block) == "table" and (block[3] or block.z or 1) or 1

    local arg_ptrs
    if self.param_types then
        assert(#args == self.param_count,
            string.format("Kernel %s expects %d arguments, got %d", self.name, self.param_count, #args))
        for i = 1, self.param_count do
            local val = args[i]
            local ptype = self.param_types[i]
            local holder = self.holders[i]

            if ptype == "ptr" or ptype == "pointer" or ptype == "deviceptr" then
                if type(val) == "table" and val.dptr then
                    holder[0] = val.dptr
                else
                    holder[0] = ffi.cast("CUdeviceptr", val)
                end
            else
                holder[0] = val
            end
        end
        arg_ptrs = self.arg_ptrs
    else
        -- Raw launch without pre-declared signature: args must be pointer holders
        local n = #args
        arg_ptrs = ffi.new("void*[" .. n .. "]")
        for i = 1, n do
            arg_ptrs[i - 1] = args[i]
        end
    end

    check_cu(libcuda.cuLaunchKernel(
        self.hfunc,
        gx, gy, gz,
        bx, by, bz,
        shared, stream,
        arg_ptrs, nil
    ), "cuLaunchKernel (" .. self.name .. ")")
end

-- -----------------------------------------------------------------------------
-- 7. Module & Runtime Compilation (NVRTC)
-- -----------------------------------------------------------------------------
local Module = {}
Module.__index = Module

function Module:get_function(name, signature)
    local hfunc = ffi.new("CUfunction[1]")
    check_cu(libcuda.cuModuleGetFunction(hfunc, self.hmod, name), "cuModuleGetFunction (" .. name .. ")")
    return make_kernel(hfunc[0], name, signature)
end

function Module:unload()
    if self.hmod then
        check_cu(libcuda.cuModuleUnload(self.hmod), "cuModuleUnload")
        self.hmod = nil
    end
end

local function find_cuda_includes()
    local candidates = {
        "/usr/local/cuda/include",
        "/usr/local/cuda-10.2/include",
        "/usr/local/cuda-11.0/include",
        "/usr/local/cuda-12/include",
    }
    for _, dir in ipairs(candidates) do
        local f = io.open(dir .. "/cuda.h", "r")
        if f then
            f:close()
            return dir
        end
    end
    return nil
end

function cuda.compile(cuda_source, opts)
    if not context then cuda.init(0) end
    opts = opts or {}

    local prog_name = opts.name or "kernel.cu"
    local prog = ffi.new("nvrtcProgram[1]")
    check_nvrtc(libnvrtc.nvrtcCreateProgram(prog, cuda_source, prog_name, 0, nil, nil), "nvrtcCreateProgram")

    local arch = opts.arch or device_info.arch or "compute_53"
    local compile_opts = {
        "--gpu-architecture=" .. arch,
    }

    local cuda_inc = opts.include_dir or find_cuda_includes()
    if cuda_inc then
        table.insert(compile_opts, "-I" .. cuda_inc)
    end

    if opts.fast_math ~= false then
        table.insert(compile_opts, "--use_fast_math")
    end

    if opts.options then
        for _, opt in ipairs(opts.options) do
            table.insert(compile_opts, opt)
        end
    end

    local c_opts = ffi.new("const char*[" .. #compile_opts .. "]")
    for i, opt in ipairs(compile_opts) do
        c_opts[i - 1] = opt
    end

    local res = libnvrtc.nvrtcCompileProgram(prog[0], #compile_opts, c_opts)
    if res ~= 0 then
        check_nvrtc(res, "nvrtcCompileProgram", prog[0])
    end

    local ptx_sz = ffi.new("size_t[1]")
    check_nvrtc(libnvrtc.nvrtcGetPTXSize(prog[0], ptx_sz), "nvrtcGetPTXSize")
    local ptx = ffi.new("char[?]", ptx_sz[0])
    check_nvrtc(libnvrtc.nvrtcGetPTX(prog[0], ptx), "nvrtcGetPTX")

    libnvrtc.nvrtcDestroyProgram(prog)

    local hmod = ffi.new("CUmodule[1]")
    check_cu(libcuda.cuModuleLoadData(hmod, ptx), "cuModuleLoadData")

    return setmetatable({
        hmod = hmod[0],
        ptx = ptx,
        ptx_size = ptx_sz[0],
    }, Module)
end

-- -----------------------------------------------------------------------------
-- 8. High-Precision GPU Event Timer
-- -----------------------------------------------------------------------------
local Timer = {}
Timer.__index = Timer

function cuda.timer()
    if not context then cuda.init(0) end

    local e_start = ffi.new("CUevent[1]")
    local e_stop = ffi.new("CUevent[1]")
    check_cu(libcuda.cuEventCreate(e_start, 0), "cuEventCreate start")
    check_cu(libcuda.cuEventCreate(e_stop, 0), "cuEventCreate stop")

    return setmetatable({
        start_event = e_start[0],
        stop_event = e_stop[0],
    }, Timer)
end

function Timer:start(stream)
    check_cu(libcuda.cuEventRecord(self.start_event, stream or nil), "cuEventRecord start")
end

function Timer:stop(stream)
    check_cu(libcuda.cuEventRecord(self.stop_event, stream or nil), "cuEventRecord stop")
end

function Timer:elapsed_ms()
    check_cu(libcuda.cuEventSynchronize(self.stop_event), "cuEventSynchronize")
    local ms = ffi.new("float[1]")
    check_cu(libcuda.cuEventElapsedTime(ms, self.start_event, self.stop_event), "cuEventElapsedTime")
    return ms[0]
end

function Timer:destroy()
    if self.start_event then
        libcuda.cuEventDestroy(self.start_event)
        self.start_event = nil
    end
    if self.stop_event then
        libcuda.cuEventDestroy(self.stop_event)
        self.stop_event = nil
    end
end

return cuda
