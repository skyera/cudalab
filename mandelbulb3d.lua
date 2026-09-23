#!/usr/bin/env luajit
--[[
================================================================================
  mandelbulb3d.lua: 3D Mandelbulb Fractal Raymarcher via LuaJIT + CUDA
================================================================================
  Renders the legendary 3D Mandelbulb fractal entirely on the GPU,
  streaming 24-bit RGB TrueColor graphics directly to your terminal.

  Usage:
    luajit mandelbulb3d.lua [options]

  Options:
    --fps <N>          Target frame rate (default: 60)
    --frames <N>       Stop after N frames
    --size <WxH>       Resolution (default: auto-fit terminal)
    --power <N>        Fixed power N (e.g. 8.0, default: dynamic 4 to 8)
    --save <file.ppm>  Export high-res snapshot (e.g. 1280x720)
    --time <seconds>   Time offset for snapshot (default: 3.5)
    --help             Show help
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

local args = { ... }
local opt_fps = 60
local opt_frames = nil
local opt_width = nil
local opt_height = nil
local opt_power = nil
local opt_save = nil
local opt_time = 3.5

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
    elseif a == "--power" then
        i = i + 1; opt_power = tonumber(args[i])
    elseif a == "--save" then
        i = i + 1; opt_save = args[i]
    elseif a == "--time" then
        i = i + 1; opt_time = tonumber(args[i]) or 3.5
    elseif a == "--help" or a == "-h" then
        print([[
3D Mandelbulb Fractal Raymarcher via LuaJIT + CUDA
Usage: luajit mandelbulb3d.lua [options]

Options:
  --fps <N>          Target FPS (default: 60)
  --frames <N>       Stop after N frames
  --size <WxH>       Resolution (e.g. 100x50, default: auto-fit terminal)
  --power <N>        Fixed power N (e.g. 8.0, default: dynamic morph)
  --save <file.ppm>  Export high-res snapshot
  --time <sec>       Time offset for snapshot
  --help             Show help
]])
        os.exit(0)
    end
    i = i + 1
end

local function get_terminal_size()
    local ws = ffi.new("struct winsize")
    if ffi.C.ioctl(1, 0x5413, ws) == 0 and ws.ws_col > 10 and ws.ws_row > 10 then
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
        width, height = 1280, 720
    else
        width, height = get_terminal_size()
    end
end

width = math.floor(width / 2) * 2
height = math.floor(height / 2) * 2

local cuda_source = [[
#define PI 3.14159265358979323846f

struct Vec3 {
    float x, y, z;
    __device__ __host__ Vec3() : x(0), y(0), z(0) {}
    __device__ __host__ Vec3(float x, float y, float z) : x(x), y(y), z(z) {}
    __device__ Vec3 operator-() const { return Vec3(-x, -y, -z); }
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

// 3D Mandelbulb Distance Estimator (White & Nylander formula)
__device__ float mandelbulbSDF(Vec3 pos, float power, float& trap) {
    Vec3 z = pos;
    float dr = 1.0f;
    float r = 0.0f;
    trap = 1e10f;

    for (int i = 0; i < 5; i++) {
        r = length(z);
        if (r > 2.2f) break;
        trap = fminf(trap, r);

        // Convert to spherical coordinates
        float theta = acosf(clampf(z.z / r, -1.0f, 1.0f));
        float phi = atan2f(z.y, z.x);
        dr = powf(r, power - 1.0f) * power * dr + 1.0f;

        // Scale and rotate
        float zr = powf(r, power);
        theta = theta * power;
        phi = phi * power;

        // Convert back to cartesian coordinates
        z = Vec3(sinf(theta) * cosf(phi), sinf(theta) * sinf(phi), cosf(theta)) * zr + pos;
    }
    return 0.5f * logf(r) * r / dr;
}

// Finite-difference normal estimation
__device__ Vec3 calcNormal(Vec3 p, float power) {
    float dummy;
    const float eps = 0.0015f;
    float d = mandelbulbSDF(p, power, dummy);
    return normalize(Vec3(
        mandelbulbSDF(p + Vec3(eps, 0, 0), power, dummy) - d,
        mandelbulbSDF(p + Vec3(0, eps, 0), power, dummy) - d,
        mandelbulbSDF(p + Vec3(0, 0, eps), power, dummy) - d
    ));
}

// Ambient Occlusion
__device__ float calcAO(Vec3 p, Vec3 n, float power) {
    float dummy;
    float occ = 0.0f;
    float sca = 1.0f;
    for (int i = 1; i <= 4; i++) {
        float hr = 0.03f * (float)i;
        float dd = mandelbulbSDF(p + n * hr, power, dummy);
        occ += -(dd - hr) * sca;
        sca *= 0.7f;
    }
    return clampf(1.0f - 2.5f * occ, 0.0f, 1.0f);
}

// Inigo Quilez cosine gradient palette
__device__ Vec3 bulbPalette(float t) {
    // Electric alien crystal palette (gold, magenta, cyan, deep violet)
    return Vec3(
        clampf(0.5f + 0.5f * cosf(2.0f * PI * (t + 0.00f)), 0.0f, 1.0f),
        clampf(0.5f + 0.5f * cosf(2.0f * PI * (t + 0.33f)), 0.0f, 1.0f),
        clampf(0.5f + 0.5f * cosf(2.0f * PI * (t + 0.67f)), 0.0f, 1.0f)
    );
}

extern "C" __global__ void renderMandelbulb(
    unsigned char* out,
    int width, int height,
    float time, float fixedPower
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;

    float aspect = (float)width / (float)height;
    float u = ((float)x + 0.5f) / (float)width * 2.0f - 1.0f;
    float v = 1.0f - ((float)y + 0.5f) / (float)height * 2.0f;
    u *= aspect;

    // Camera Orbit
    float camDist = 2.45f;
    float camAngle = time * 0.25f;
    Vec3 ro(sinf(camAngle) * camDist, 0.65f * sinf(time * 0.18f) + 0.2f, cosf(camAngle) * camDist);
    Vec3 ta(0.0f, 0.0f, 0.0f);

    Vec3 ww = normalize(ta - ro);
    Vec3 uu = normalize(cross(ww, Vec3(0, 1, 0)));
    Vec3 vv = cross(uu, ww);
    Vec3 rd = normalize(uu * u + vv * v + ww * 1.6f);

    // Power modulation (e.g. morphing 4 -> 8 or fixed)
    float power = (fixedPower > 0.0f) ? fixedPower : (5.5f + 2.5f * sinf(time * 0.2f));

    // Sphere tracing
    float t = 0.02f;
    float trap = 0.0f;
    bool hit = false;
    int stepsTaken = 0;

    for (int step = 0; step < 64; step++) {
        Vec3 p = ro + rd * t;
        float d = mandelbulbSDF(p, power, trap);
        if (d < 0.0018f) {
            hit = true;
            stepsTaken = step;
            break;
        }
        t += d * 0.85f; // slight under-relaxation for fractal detail
        if (t > 5.0f) break;
    }

    // Cosmic background with subtle stars
    Vec3 bgCol = Vec3(0.02f, 0.025f, 0.05f) + Vec3(0.03f, 0.02f, 0.06f) * (v * 0.5f + 0.5f);
    Vec3 col = bgCol;

    if (hit) {
        Vec3 p = ro + rd * t;
        Vec3 n = calcNormal(p, power);
        Vec3 viewDir = normalize(ro - p);

        // Key light
        Vec3 lightDir = normalize(Vec3(2.0f, 3.5f, 1.5f));
        float diff = clampf(dot(n, lightDir), 0.0f, 1.0f);

        // Rim / Back light (gives beautiful electric silhouette)
        Vec3 rimDir = normalize(-ro + Vec3(0, 1, 0));
        float rim = powf(clampf(1.0f - dot(n, viewDir), 0.0f, 1.0f), 3.0f);

        // Specular highlight
        Vec3 halfDir = normalize(lightDir + viewDir);
        float spec = powf(clampf(dot(n, halfDir), 0.0f, 1.0f), 32.0f);

        // Ambient Occlusion
        float ao = calcAO(p, n, power);

        // Orbit trap based coloring
        float colorParam = trap * 2.5f + (float)stepsTaken * 0.02f;
        Vec3 baseCol = bulbPalette(colorParam);

        // Shading composition
        Vec3 ambient = baseCol * (0.15f * ao);
        Vec3 diffuse = baseCol * (diff * 0.85f * ao);
        Vec3 specular = Vec3(1.0f, 0.95f, 0.85f) * (spec * 0.7f);
        Vec3 rimCol = Vec3(0.3f, 0.7f, 1.0f) * (rim * 0.6f);

        col = ambient + diffuse + specular + rimCol;

        // Depth fog
        float fog = clampf((t - 1.5f) / 3.5f, 0.0f, 1.0f);
        col = col * (1.0f - fog) + bgCol * fog;
    }

    // Tone-mapping and Gamma correction
    col.x = sqrtf(clampf(col.x, 0.0f, 1.0f));
    col.y = sqrtf(clampf(col.y, 0.0f, 1.0f));
    col.z = sqrtf(clampf(col.z, 0.0f, 1.0f));

    int idx = (y * width + x) * 3;
    out[idx + 0] = (unsigned char)(col.x * 255.0f);
    out[idx + 1] = (unsigned char)(col.y * 255.0f);
    out[idx + 2] = (unsigned char)(col.z * 255.0f);
}
]]

io.write("Initializing CUDA Device...\n")
local dev_info = cuda.init(0)
io.write("Compiling 3D Mandelbulb Kernel via NVRTC...\n")
local mod = cuda.compile(cuda_source, { arch = dev_info.arch, fast_math = true, name = "mandelbulb.cu" })
local kernel = mod:get_function("renderMandelbulb", "ptr, int, int, float, float")

local num_pixels = width * height
local buf_size = num_pixels * 3
local d_out = cuda.alloc(buf_size)
local h_out = ffi.new("uint8_t[?]", buf_size)
local gpu_timer = cuda.timer()
local fixed_power_val = opt_power or -1.0

local function save_ppm(filename, w, h, data)
    local f = assert(io.open(filename, "wb"))
    f:write(string.format("P6\n%d %d\n255\n", w, h))
    f:write(ffi.string(data, w * h * 3))
    f:close()
end

if opt_save then
    print(string.format("Rendering 3D Mandelbulb (%dx%d) to %s...", width, height, opt_save))
    local bx, by = 16, 16
    local gx = math.ceil(width / bx)
    local gy = math.ceil(height / by)

    gpu_timer:start()
    kernel:launch({ grid = { gx, gy }, block = { bx, by } }, d_out, width, height, opt_time, fixed_power_val)
    gpu_timer:stop()
    local ms = gpu_timer:elapsed_ms()

    d_out:to_host(h_out)
    save_ppm(opt_save, width, height, h_out)
    print(string.format("Saved %s successfully! Render Time: %.2f ms", opt_save, ms))
    os.exit(0)
end

-- Terminal Renderer
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

local function cleanup()
    io.write("\27[?25h\27[0m\n")
    io.flush()
end

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
    while true do
        local now = os.clock()
        local sim_time = now - t_start

        gpu_timer:start()
        kernel:launch({ grid = { gx, gy }, block = { bx, by } },
            d_out, width, height, sim_time, fixed_power_val)
        gpu_timer:stop()
        gpu_ms = gpu_timer:elapsed_ms()

        d_out:to_host(h_out)
        local frame_str = format_terminal_frame(width, height, h_out)

        frame_count = frame_count + 1
        local dt = now - last_time
        if dt >= 0.5 then
            fps = frame_count / (now - t_start)
            last_time = now
        end

        local cur_power = (opt_power or (5.5 + 2.5 * math.sin(sim_time * 0.2)))
        local hud = string.format(
            "\27[H\27[1;37;41m [3D Mandelbulb | %s]  FPS: %5.1f  |  GPU: %5.2f ms  |  Power: %4.1f  |  Frame: %d  \27[0m\n",
            dev_info.name, fps > 0 and fps or (1.0 / (gpu_ms * 1e-3)), gpu_ms, cur_power, frame_count
        )

        io.write(hud .. frame_str .. "\n\27[2;37m Press Ctrl+C to exit \27[0m")
        io.flush()

        if opt_frames and frame_count >= opt_frames then break end

        if opt_fps > 0 then
            local elapsed = (os.clock() - now)
            local target_frame_time = 1.0 / opt_fps
            if elapsed < target_frame_time then
                local sleep_us = math.floor((target_frame_time - elapsed) * 1e6)
                if sleep_us > 100 then ffi.C.usleep(sleep_us) end
            end
        end
    end
end)

cleanup()

if not ok and err and not err:match("interrupted") then
    print("\nError: " .. tostring(err))
else
    print(string.format("\nFinished! Rendered %d frames at %.1f FPS.",
        frame_count, frame_count / math.max(1e-4, os.clock() - t_start)))
end
