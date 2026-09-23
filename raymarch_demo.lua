#!/usr/bin/env luajit
--[[
================================================================================
  raymarch_demo.lua: Real-Time 3D Raymarching Engine via LuaJIT + CUDA
================================================================================
  Renders a 3D signed distance field (SDF) scene entirely on the GPU,
  streaming true 24-bit RGB ANSI graphics directly to your terminal
  using half-block pixels (▀) at 2x vertical resolution.

  Usage:
    luajit raymarch_demo.lua [options]

  Options:
    --fps <N>          Target frame rate cap (default: 60, 0 for uncapped)
    --frames <N>       Render N frames and exit (default: infinite)
    --size <WxH>       Render resolution (e.g. 100x60, default: auto-fit terminal)
    --save <file.ppm>  Render a single frame snapshot to a high-res PPM image file
    --time <seconds>   Time offset in seconds for snapshot (default: 2.0)
    --help             Show this help message
================================================================================
--]]

local ffi = require("ffi")
local cuda = require("cuda")

ffi.cdef[[
struct winsize {
    unsigned short ws_row;
    unsigned short ws_col;
    unsigned short ws_xpixel;
    unsigned short ws_ypixel;
};
int ioctl(int fd, unsigned long request, ...);
int usleep(unsigned int usec);
]]

-- -----------------------------------------------------------------------------
-- Parse Command Line Arguments
-- -----------------------------------------------------------------------------
local args = { ... }
local opt_fps = 60
local opt_frames = nil
local opt_width = nil
local opt_height = nil
local opt_save = nil
local opt_time = 2.0

local i = 1
while i <= #args do
    local a = args[i]
    if a == "--fps" then
        i = i + 1; opt_fps = tonumber(args[i]) or 60
    elseif a == "--frames" then
        i = i + 1; opt_frames = tonumber(args[i])
    elseif a == "--size" then
        i = i + 1
        local w, h = (args[i] or ""):match("(%d+)x(%d+)")
        if w and h then opt_width = tonumber(w); opt_height = tonumber(h) end
    elseif a == "--save" then
        i = i + 1; opt_save = args[i]
    elseif a == "--time" then
        i = i + 1; opt_time = tonumber(args[i]) or 2.0
    elseif a == "--help" or a == "-h" then
        print([[
Real-Time 3D Raymarching Engine via LuaJIT + CUDA
Usage: luajit raymarch_demo.lua [options]

Options:
  --fps <N>          Target FPS (default: 60, 0 for uncapped)
  --frames <N>       Stop after N frames (default: run continuously)
  --size <WxH>       Resolution (e.g. 120x80, default: auto-fit terminal)
  --save <file.ppm>  Save a high-res snapshot to PPM image file
  --time <sec>       Time offset for snapshot (default: 2.0)
  --help             Show help
]])
        os.exit(0)
    end
    i = i + 1
end

-- Detect terminal size
local function get_terminal_size()
    local ws = ffi.new("struct winsize")
    if ffi.C.ioctl(1, 0x5413, ws) == 0 and ws.ws_col > 10 and ws.ws_row > 10 then
        -- Terminal character cells: each row has 2 vertical pixels using half-block ▀
        -- Reserve 3 rows for HUD and status
        local w = ws.ws_col
        local h = (ws.ws_row - 3) * 2
        return w, math.max(16, h)
    end
    return 100, 60
end

local width = opt_width
local height = opt_height
if not width or not height then
    if opt_save then
        width, height = 1280, 720 -- Default high-res for saving
    else
        width, height = get_terminal_size()
    end
end

-- Make sure width & height are even
width = math.floor(width / 2) * 2
height = math.floor(height / 2) * 2

-- -----------------------------------------------------------------------------
-- CUDA Raymarching Kernel
-- -----------------------------------------------------------------------------
local cuda_source = [[
#define PI 3.14159265358979323846f

struct Vec3 {
    float x, y, z;
    __device__ __host__ Vec3() : x(0), y(0), z(0) {}
    __device__ __host__ Vec3(float x, float y, float z) : x(x), y(y), z(z) {}
    __device__ Vec3 operator+(const Vec3& b) const { return Vec3(x + b.x, y + b.y, z + b.z); }
    __device__ Vec3 operator-(const Vec3& b) const { return Vec3(x - b.x, y - b.y, z - b.z); }
    __device__ Vec3 operator*(float s) const { return Vec3(x * s, y * s, z * s); }
    __device__ Vec3 operator*(const Vec3& b) const { return Vec3(x * b.x, y * b.y, z * b.z); }
};

__device__ inline float dot(const Vec3& a, const Vec3& b) { return a.x*b.x + a.y*b.y + a.z*b.z; }
__device__ inline Vec3 cross(const Vec3& a, const Vec3& b) {
    return Vec3(a.y*b.z - a.z*b.y, a.z*b.x - a.x*b.z, a.x*b.y - a.y*b.x);
}
__device__ inline float length(const Vec3& v) { return sqrtf(dot(v, v)); }
__device__ inline Vec3 normalize(const Vec3& v) {
    float len = length(v);
    return len > 1e-6f ? v * (1.0f / len) : Vec3(0, 0, 0);
}
__device__ inline float clampf(float v, float minVal, float maxVal) {
    return fminf(fmaxf(v, minVal), maxVal);
}

// Rotations
__device__ Vec3 rotateY(Vec3 p, float a) {
    float c = cosf(a), s = sinf(a);
    return Vec3(c*p.x + s*p.z, p.y, -s*p.x + c*p.z);
}
__device__ Vec3 rotateX(Vec3 p, float a) {
    float c = cosf(a), s = sinf(a);
    return Vec3(p.x, c*p.y - s*p.z, s*p.y + c*p.z);
}
__device__ Vec3 rotateZ(Vec3 p, float a) {
    float c = cosf(a), s = sinf(a);
    return Vec3(c*p.x - s*p.y, s*p.x + c*p.y, p.z);
}

// Distance Primitives
__device__ inline float sdSphere(Vec3 p, float r) { return length(p) - r; }

__device__ inline float sdTorus(Vec3 p, float R, float r) {
    float qx = sqrtf(p.x*p.x + p.z*p.z) - R;
    return sqrtf(qx*qx + p.y*p.y) - r;
}

// Smooth Minimum for organic blending
__device__ inline float smin(float a, float b, float k) {
    float h = clampf(0.5f + 0.5f * (b - a) / k, 0.0f, 1.0f);
    return (1.0f - h) * b + h * a - k * h * (1.0f - h);
}

// Scene Signed Distance Function
__device__ float sceneSDF(Vec3 p, float time, int& matId) {
    // Ground plane
    float dFloor = p.y + 1.25f;

    // Torus 1 (tilted & rotating)
    Vec3 tp1 = rotateY(p, time * 0.9f);
    tp1 = rotateX(tp1, time * 0.5f + 0.5f);
    float dTorus1 = sdTorus(tp1, 1.25f, 0.28f);

    // Torus 2 (counter-rotating inside)
    Vec3 tp2 = rotateY(p, -time * 0.7f + 1.2f);
    tp2 = rotateZ(tp2, time * 0.6f);
    float dTorus2 = sdTorus(tp2, 0.85f, 0.16f);

    // Pulsing central plasma sphere
    float sphereR = 0.50f + 0.10f * sinf(time * 3.5f);
    float dSphere = sdSphere(p, sphereR);

    // Floating satellite spheres
    Vec3 satP = p - Vec3(sinf(time * 1.8f) * 1.8f, sinf(time * 2.5f) * 0.4f, cosf(time * 1.8f) * 1.8f);
    float dSat = sdSphere(satP, 0.20f);

    // Material determination
    float d = dFloor;
    matId = 0; // floor

    if (dTorus1 < d) { d = dTorus1; matId = 1; } // Torus 1 (gold)
    if (dTorus2 < d) { d = dTorus2; matId = 2; } // Torus 2 (copper)
    if (dSphere < d) { d = dSphere; matId = 3; } // Core Sphere (cyberpunk neon cyan)
    if (dSat < d)    { d = dSat;    matId = 4; } // Orbiting satellite (magenta)

    return d;
}

// Surface Normal Estimation via Finite Differences
__device__ Vec3 calcNormal(Vec3 p, float time) {
    int dummy;
    const float eps = 0.001f;
    float d = sceneSDF(p, time, dummy);
    return normalize(Vec3(
        sceneSDF(p + Vec3(eps, 0, 0), time, dummy) - d,
        sceneSDF(p + Vec3(0, eps, 0), time, dummy) - d,
        sceneSDF(p + Vec3(0, 0, eps), time, dummy) - d
    ));
}

// Soft Shadow Raymarching
__device__ float softShadow(Vec3 ro, Vec3 rd, float mint, float maxt, float k, float time) {
    int dummy;
    float res = 1.0f;
    float t = mint;
    for (int i = 0; i < 24 && t < maxt; i++) {
        float h = sceneSDF(ro + rd * t, time, dummy);
        if (h < 0.001f) return 0.0f;
        res = fminf(res, k * h / t);
        t += clampf(h, 0.02f, 0.25f);
    }
    return clampf(res, 0.0f, 1.0f);
}

// Ambient Occlusion Estimation
__device__ float calcAO(Vec3 p, Vec3 n, float time) {
    int dummy;
    float occ = 0.0f;
    float sca = 1.0f;
    for (int i = 1; i <= 5; i++) {
        float hr = 0.05f * (float)i;
        float dd = sceneSDF(p + n * hr, time, dummy);
        occ += -(dd - hr) * sca;
        sca *= 0.75f;
    }
    return clampf(1.0f - 2.0f * occ, 0.0f, 1.0f);
}

// Main Raymarching Kernel
extern "C" __global__ void renderRaymarch(unsigned char* out, int width, int height, float time) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;

    // Normalized screen coordinates [-1, 1]
    float aspect = (float)width / (float)height;
    float u = ((float)x + 0.5f) / (float)width * 2.0f - 1.0f;
    float v = 1.0f - ((float)y + 0.5f) / (float)height * 2.0f; // y-up
    u *= aspect;

    // Smooth Orbiting Camera
    float camDist = 3.8f;
    float camAngle = time * 0.35f;
    Vec3 ro(sinf(camAngle) * camDist, 1.35f + 0.35f * sinf(time * 0.5f), cosf(camAngle) * camDist);
    Vec3 ta(0.0f, -0.05f, 0.0f); // Look target

    Vec3 ww = normalize(ta - ro);
    Vec3 uu = normalize(cross(ww, Vec3(0, 1, 0)));
    Vec3 vv = cross(uu, ww);
    Vec3 rd = normalize(uu * u + vv * v + ww * 1.85f); // Field of view

    // Sphere Tracing loop
    float t = 0.02f;
    int matId = -1;
    bool hit = false;
    for (int step = 0; step < 80; step++) {
        Vec3 p = ro + rd * t;
        float d = sceneSDF(p, time, matId);
        if (d < 0.001f) {
            hit = true;
            break;
        }
        t += d;
        if (t > 22.0f) break;
    }

    // Sky Background gradient
    float skyGrad = clampf(0.5f * (rd.y + 1.0f), 0.0f, 1.0f);
    Vec3 col = Vec3(0.04f, 0.05f, 0.10f) * (1.0f - skyGrad) + Vec3(0.12f, 0.16f, 0.28f) * skyGrad;

    if (hit) {
        Vec3 p = ro + rd * t;
        Vec3 n = calcNormal(p, time);
        Vec3 viewDir = normalize(ro - p);

        // Orbiting Key Light
        Vec3 lightPos(sinf(time * 0.6f) * 4.5f, 5.0f, cosf(time * 0.6f) * 4.5f);
        Vec3 lightDir = normalize(lightPos - p);
        float diff = clampf(dot(n, lightDir), 0.0f, 1.0f);

        // Blinn-Phong Specular
        Vec3 halfDir = normalize(lightDir + viewDir);
        float spec = powf(clampf(dot(n, halfDir), 0.0f, 1.0f), 40.0f);

        // Ambient Occlusion & Soft Shadow
        float ao = calcAO(p, n, time);
        float shadow = softShadow(p + n * 0.01f, lightDir, 0.02f, 8.0f, 12.0f, time);

        // Material Coloring
        Vec3 baseCol(0.8f, 0.8f, 0.8f);
        float shine = 0.6f;
        float emission = 0.0f;

        if (matId == 0) { // Checkerboard floor
            float check = fmodf(floorf(p.x * 1.5f) + floorf(p.z * 1.5f), 2.0f);
            baseCol = (fabsf(check) < 0.5f) ? Vec3(0.16f, 0.18f, 0.24f) : Vec3(0.42f, 0.44f, 0.52f);
            shine = 0.35f;
        } else if (matId == 1) { // Outer Torus (Rich Gold)
            baseCol = Vec3(0.98f, 0.72f, 0.22f);
            shine = 1.0f;
        } else if (matId == 2) { // Inner Torus (Burnished Copper / Rose Gold)
            baseCol = Vec3(0.92f, 0.42f, 0.28f);
            shine = 0.8f;
        } else if (matId == 3) { // Core Pulsing Sphere (Cyberpunk Neon Cyan)
            baseCol = Vec3(0.10f, 0.95f, 0.98f);
            emission = 0.75f;
            shine = 0.5f;
        } else if (matId == 4) { // Satellite (Neon Magenta)
            baseCol = Vec3(1.0f, 0.20f, 0.70f);
            emission = 0.6f;
        }

        // Lighting model
        Vec3 ambient = baseCol * (0.12f * ao + emission);
        Vec3 diffuse = baseCol * (diff * shadow * 0.85f);
        Vec3 specular = Vec3(1.0f, 0.98f, 0.90f) * (spec * shadow * shine);

        col = ambient + diffuse + specular;

        // Distance Fog
        float fog = clampf((t - 2.0f) / 18.0f, 0.0f, 1.0f);
        Vec3 fogCol(0.06f, 0.08f, 0.15f);
        col = col * (1.0f - fog) + fogCol * fog;
    }

    // Gamma correction and Tonemapping
    col.x = sqrtf(clampf(col.x, 0.0f, 1.0f));
    col.y = sqrtf(clampf(col.y, 0.0f, 1.0f));
    col.z = sqrtf(clampf(col.z, 0.0f, 1.0f));

    // Write RGB output buffer
    int idx = (y * width + x) * 3;
    out[idx + 0] = (unsigned char)(col.x * 255.0f);
    out[idx + 1] = (unsigned char)(col.y * 255.0f);
    out[idx + 2] = (unsigned char)(col.z * 255.0f);
}
]]

-- -----------------------------------------------------------------------------
-- Initialize GPU & Compile Kernel
-- -----------------------------------------------------------------------------
io.write("Initializing CUDA Device...\n")
local dev_info = cuda.init(0)
print(string.format("  GPU: %s | Compute: %s | VRAM: %.1f MB",
    dev_info.name, dev_info.arch, dev_info.total_memory_mb))

io.write("Compiling 3D Raymarching Kernel via NVRTC...\n")
local t_comp0 = os.clock()
local mod = cuda.compile(cuda_source, {
    arch = dev_info.arch,
    fast_math = true,
    name = "raymarch.cu",
})
local t_comp1 = os.clock()
print(string.format("  Compiled successfully in %.2f seconds!\n", t_comp1 - t_comp0))

local kernel = mod:get_function("renderRaymarch", "ptr, int, int, float")

-- Allocate GPU buffer & Host buffer
local num_pixels = width * height
local buf_size = num_pixels * 3
local d_out = cuda.alloc(buf_size)
local h_out = ffi.new("uint8_t[?]", buf_size)
local gpu_timer = cuda.timer()

-- -----------------------------------------------------------------------------
-- Save Snapshot Mode (PPM image output)
-- -----------------------------------------------------------------------------
local function save_ppm(filename, w, h, data)
    local f = assert(io.open(filename, "wb"))
    f:write(string.format("P6\n%d %d\n255\n", w, h))
    f:write(ffi.string(data, w * h * 3))
    f:close()
end

if opt_save then
    print(string.format("Rendering snapshot (%dx%d) at t = %.2fs to %s...", width, height, opt_time, opt_save))
    local bx, by = 16, 16
    local gx = math.ceil(width / bx)
    local gy = math.ceil(height / by)

    gpu_timer:start()
    kernel:launch({ grid = { gx, gy }, block = { bx, by } }, d_out, width, height, opt_time)
    gpu_timer:stop()
    local ms = gpu_timer:elapsed_ms()

    d_out:to_host(h_out)
    save_ppm(opt_save, width, height, h_out)
    print(string.format("Saved %s successfully! GPU Render Time: %.2f ms", opt_save, ms))
    os.exit(0)
end

-- -----------------------------------------------------------------------------
-- Terminal Display Renderer (Half-Block ▀ with 24-bit TrueColor)
-- -----------------------------------------------------------------------------
local function format_terminal_frame(w, h, data)
    local lines = {}
    local rows = math.floor(h / 2)

    for cy = 0, rows - 1 do
        local top_y = cy * 2
        local bot_y = top_y + 1
        local line = {}
        local prev_tr, prev_tg, prev_tb = -1, -1, -1
        local prev_br, prev_bg, prev_bb = -1, -1, -1

        for x = 0, w - 1 do
            local top_idx = (top_y * w + x) * 3
            local bot_idx = (bot_y * w + x) * 3

            local tr = data[top_idx + 0]
            local tg = data[top_idx + 1]
            local tb = data[top_idx + 2]

            local br = data[bot_idx + 0]
            local bg = data[bot_idx + 1]
            local bb = data[bot_idx + 2]

            -- ANSI color escape sequence
            if tr ~= prev_tr or tg ~= prev_tg or tb ~= prev_tb or
               br ~= prev_br or bg ~= prev_bg or bb ~= prev_bg then
                line[#line + 1] = string.format("\27[38;2;%d;%d;%dm\27[48;2;%d;%d;%dm▀", tr, tg, tb, br, bg, bb)
                prev_tr, prev_tg, prev_tb = tr, tg, tb
                prev_br, prev_bg, prev_bb = br, bg, bb
            else
                line[#line + 1] = "▀"
            end
        end
        lines[#lines + 1] = table.concat(line)
    end
    lines[#lines + 1] = "\27[0m"
    return table.concat(lines, "\n")
end

-- -----------------------------------------------------------------------------
-- Real-Time Animation Loop
-- -----------------------------------------------------------------------------
local running = true

local function cleanup()
    io.write("\27[?25h\27[0m\n") -- Show cursor and reset colors
    io.flush()
end

-- Hide cursor and clear screen
io.write("\27[?25l\27[2J")
io.flush()

local bx, by = 16, 16
local gx = math.ceil(width / bx)
local gy = math.ceil(height / by)

local frame_count = 0
local t_start = os.clock()
local last_time = t_start
local fps = 0.0
local gpu_ms = 0.0

local ok, err = pcall(function()
    while running do
        local now = os.clock()
        local sim_time = now - t_start

        -- Launch GPU Kernel
        gpu_timer:start()
        kernel:launch({ grid = { gx, gy }, block = { bx, by } }, d_out, width, height, sim_time)
        gpu_timer:stop()
        gpu_ms = gpu_timer:elapsed_ms()

        -- Copy pixels back to host
        d_out:to_host(h_out)

        -- Convert to ANSI string
        local frame_str = format_terminal_frame(width, height, h_out)

        -- Calculate FPS
        frame_count = frame_count + 1
        local dt = now - last_time
        if dt >= 0.5 then
            fps = frame_count / (now - t_start)
            last_time = now
        end

        -- Header HUD
        local hud = string.format(
            "\27[H\27[1;37;44m [CUDA 10.2 | %s]  FPS: %5.1f  |  GPU: %5.2f ms  |  Res: %dx%d  |  Frame: %d  \27[0m\n",
            dev_info.name, fps > 0 and fps or (1.0 / (gpu_ms * 1e-3)), gpu_ms, width, height, frame_count
        )

        -- Output single buffer to terminal to eliminate flicker
        io.write(hud .. frame_str .. "\n\27[2;37m Press Ctrl+C to exit \27[0m")
        io.flush()

        if opt_frames and frame_count >= opt_frames then
            break
        end

        -- Frame rate throttling
        if opt_fps > 0 then
            local elapsed = (os.clock() - now)
            local target_frame_time = 1.0 / opt_fps
            if elapsed < target_frame_time then
                local sleep_us = math.floor((target_frame_time - elapsed) * 1e6)
                if sleep_us > 100 then
                    ffi.C.usleep(sleep_us)
                end
            end
        end
    end
end)

cleanup()

if not ok and err and not err:match("interrupted") then
    print("\nError: " .. tostring(err))
else
    print(string.format("\nFinished! Rendered %d frames at an average of %.1f FPS.",
        frame_count, frame_count / math.max(1e-4, os.clock() - t_start)))
end
