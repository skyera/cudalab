#!/usr/bin/env luajit
--[[
================================================================================
  galaxy3d.lua: 100,000+ Star 3D Spiral Galaxy & Nebula Simulator via LuaJIT + CUDA
================================================================================
  Simulates a realistic 3D spiral galaxy with 100,000+ stars on the GPU:
  - Realistic Vera Rubin flat rotation curve with dark matter halo dynamics
  - Galactic core/bulge, logarithmic spiral arms, star-forming nebulae, and stellar halo
  - Bilinear sub-pixel splatting with floating-point atomic accumulation
  - HDR filmic tonemapping with core bloom
  - 3D orbiting camera with dynamic disk tilt and perspective projection
  - Streams 24-bit RGB TrueColor ANSI graphics to your terminal at 60+ FPS

  Usage:
    luajit galaxy3d.lua [options]

  Options:
    --stars <N>        Number of stars (default: 120,000)
    --arms <N>         Number of spiral arms (default: 2)
    --fps <N>          Target frame rate (default: 60)
    --frames <N>       Stop after N frames
    --size <WxH>       Resolution (default: auto-fit terminal)
    --tilt <deg>       Fixed camera tilt angle in degrees (default: dynamic)
    --save <file.ppm>  Export high-res wallpaper (e.g. 1920x1080)
    --time <seconds>   Time offset for snapshot (default: 1.5)
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
local opt_stars = 120000
local opt_arms = 2
local opt_fps = 60
local opt_frames = nil
local opt_width = nil
local opt_height = nil
local opt_tilt = nil
local opt_save = nil
local opt_time = 1.5

local i = 1
while i <= #args do
    local a = args[i]
    if a == "--stars" then
        i = i + 1; opt_stars = tonumber(args[i]) or 120000
    elseif a == "--arms" then
        i = i + 1; opt_arms = tonumber(args[i]) or 2
    elseif a == "--fps" then
        i = i + 1; opt_fps = tonumber(args[i]) or 60
    elseif a == "--frames" then
        i = i + 1; opt_frames = tonumber(args[i])
    elseif a == "--size" then
        i = i + 1
        local w, h = (args[i] or ""):match("(%d+)x(%d+)")
        if w and h then opt_width = tonumber(w); opt_height = tonumber(h) end
    elseif a == "--tilt" then
        i = i + 1; opt_tilt = tonumber(args[i])
    elseif a == "--save" then
        i = i + 1; opt_save = args[i]
    elseif a == "--time" then
        i = i + 1; opt_time = tonumber(args[i]) or 1.5
    elseif a == "--help" or a == "-h" then
        print([[
100,000+ Star 3D Spiral Galaxy Simulator via LuaJIT + CUDA
Usage: luajit galaxy3d.lua [options]

Options:
  --stars <N>        Number of stars (default: 120,000)
  --arms <N>         Number of spiral arms (default: 2)
  --fps <N>          Target FPS (default: 60)
  --frames <N>       Stop after N frames
  --size <WxH>       Resolution (e.g. 120x80, default: auto-fit terminal)
  --tilt <deg>       Camera elevation tilt angle (default: dynamic orbit)
  --save <file.ppm>  Export high-res wallpaper
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
        width, height = 1920, 1080
    else
        width, height = get_terminal_size()
    end
end

width = math.floor(width / 2) * 2
height = math.floor(height / 2) * 2

-- -----------------------------------------------------------------------------
-- CUDA Kernels: Splatting & Tonemapping
-- -----------------------------------------------------------------------------
local cuda_source = [[
#define PI 3.14159265358979323846f

struct Star {
    float r;          // Orbital radius
    float theta;      // Initial orbital phase angle
    float z;          // Vertical offset from galactic plane
    float cr, cg, cb; // Color components
    float bright;     // Intrinsic stellar luminosity
};

__device__ inline float clampf(float v, float minVal, float maxVal) {
    return fminf(fmaxf(v, minVal), maxVal);
}

// -----------------------------------------------------------------------------
// 1. Clear HDR Accumulation Buffer
// -----------------------------------------------------------------------------
extern "C" __global__ void clearBuffer(float* ar, float* ag, float* ab, int total_pixels) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < total_pixels) {
        ar[idx] = 0.0f;
        ag[idx] = 0.0f;
        ab[idx] = 0.0f;
    }
}

// -----------------------------------------------------------------------------
// 2. Project & Splat Stars into 2D HDR Buffer (with Bilinear Filtering)
// -----------------------------------------------------------------------------
extern "C" __global__ void renderStars(
    const Star* stars, int num_stars,
    float* accum_r, float* accum_g, float* accum_b,
    int width, int height,
    float time, float camElev, float camAzim, float camDist
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_stars) return;

    Star s = stars[i];

    // Flat galactic rotation curve (Vera Rubin dark matter halo profile)
    // Core has solid-body rotation; disk has constant linear orbital speed
    float v0 = 1.65f;
    float rc = 0.40f;
    float omega = v0 / sqrtf(s.r * s.r + rc * rc);
    float angle = s.theta + omega * time;

    // 3D coordinates in galaxy frame (disk lies in X-Z plane, Y is vertical)
    float px = s.r * cosf(angle);
    float pz = s.r * sinf(angle);
    float py = s.z;

    // 3D Camera transformation: Azimuth rotation (around Y)
    float ca = cosf(camAzim), sa = sinf(camAzim);
    float rx = px * ca - pz * sa;
    float rz = px * sa + pz * ca;

    // Elevation rotation (around X)
    float ce = cosf(camElev), se = sinf(camElev);
    float ry = py * ce - rz * se;
    float cz = py * se + rz * ce + camDist;

    // Clip stars behind camera near-plane
    if (cz <= 0.15f) return;

    // Perspective projection
    float fov = 1.35f * (float)height;
    float screen_x = (rx / cz) * fov + (float)width * 0.5f;
    float screen_y = (-ry / cz) * fov + (float)height * 0.5f;

    int x0 = (int)floorf(screen_x);
    int y0 = (int)floorf(screen_y);

    if (x0 < 0 || x0 >= width - 1 || y0 < 0 || y0 >= height - 1) return;

    // Sub-pixel bilinear fractions for anti-aliasing
    float fx = screen_x - (float)x0;
    float fy = screen_y - (float)y0;

    float dist_att = 8.5f / (cz * 0.25f + 0.75f);
    float b = s.bright * dist_att;

    float w00 = (1.0f - fx) * (1.0f - fy) * b;
    float w10 = fx * (1.0f - fy) * b;
    float w01 = (1.0f - fx) * fy * b;
    float w11 = fx * fy * b;

    int idx00 = y0 * width + x0;
    int idx10 = y0 * width + (x0 + 1);
    int idx01 = (y0 + 1) * width + x0;
    int idx11 = (y0 + 1) * width + (x0 + 1);

    // Atomically accumulate star photons into HDR buffer
    atomicAdd(&accum_r[idx00], s.cr * w00);
    atomicAdd(&accum_g[idx00], s.cg * w00);
    atomicAdd(&accum_b[idx00], s.cb * w00);

    atomicAdd(&accum_r[idx10], s.cr * w10);
    atomicAdd(&accum_g[idx10], s.cg * w10);
    atomicAdd(&accum_b[idx10], s.cb * w10);

    atomicAdd(&accum_r[idx01], s.cr * w01);
    atomicAdd(&accum_g[idx01], s.cg * w01);
    atomicAdd(&accum_b[idx01], s.cb * w01);

    atomicAdd(&accum_r[idx11], s.cr * w11);
    atomicAdd(&accum_g[idx11], s.cg * w11);
    atomicAdd(&accum_b[idx11], s.cb * w11);

    // Subtle cross-star diffraction spike for bright giants
    if (b > 18.0f) {
        float glow = b * 0.12f;
        if (x0 > 1 && x0 < width - 2 && y0 > 1 && y0 < height - 2) {
            atomicAdd(&accum_r[idx00 - 1], s.cr * glow);
            atomicAdd(&accum_g[idx00 - 1], s.cg * glow);
            atomicAdd(&accum_b[idx00 - 1], s.cb * glow);

            atomicAdd(&accum_r[idx00 + 1], s.cr * glow);
            atomicAdd(&accum_g[idx00 + 1], s.cg * glow);
            atomicAdd(&accum_b[idx00 + 1], s.cb * glow);

            atomicAdd(&accum_r[idx00 - width], s.cr * glow);
            atomicAdd(&accum_g[idx00 - width], s.cg * glow);
            atomicAdd(&accum_b[idx00 - width], s.cb * glow);

            atomicAdd(&accum_r[idx00 + width], s.cr * glow);
            atomicAdd(&accum_g[idx00 + width], s.cg * glow);
            atomicAdd(&accum_b[idx00 + width], s.cb * glow);
        }
    }
}

// -----------------------------------------------------------------------------
// 3. HDR Filmic Tonemapping & Bloom Post-Processing
// -----------------------------------------------------------------------------
extern "C" __global__ void tonemapBuffer(
    const float* accum_r, const float* accum_g, const float* accum_b,
    unsigned char* out,
    int width, int height
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;

    int idx = y * width + x;
    float r = accum_r[idx];
    float g = accum_g[idx];
    float b = accum_b[idx];

    // Luminance-based HDR tonemapping to preserve vivid stellar chromaticity
    float lum = 0.2126f * r + 0.7152f * g + 0.0722f * b;
    if (lum > 1e-5f) {
        float toned_lum = 1.0f - expf(-lum * 0.35f);
        float s = toned_lum / lum;
        r = r * s;
        g = g * s;
        b = b * s;
    }

    // Cosmic void dark navy background
    r += 0.010f;
    g += 0.008f;
    b += 0.020f;

    // Gamma correction
    r = sqrtf(clampf(r, 0.0f, 1.0f));
    g = sqrtf(clampf(g, 0.0f, 1.0f));
    b = sqrtf(clampf(b, 0.0f, 1.0f));

    int out_idx = idx * 3;
    out[out_idx + 0] = (unsigned char)(r * 255.0f);
    out[out_idx + 1] = (unsigned char)(g * 255.0f);
    out[out_idx + 2] = (unsigned char)(b * 255.0f);
}
]]

-- -----------------------------------------------------------------------------
-- Initialize GPU & Compile Kernels
-- -----------------------------------------------------------------------------
io.write("Initializing CUDA Device...\n")
local dev_info = cuda.init(0)
print(string.format("  GPU: %s | Compute: %s | VRAM: %.1f MB",
    dev_info.name, dev_info.arch, dev_info.total_memory_mb))

io.write("Compiling 3D Galaxy Simulation Kernels via NVRTC...\n")
local mod = cuda.compile(cuda_source, { arch = dev_info.arch, fast_math = true, name = "galaxy.cu" })
local k_clear   = mod:get_function("clearBuffer",   "ptr, ptr, ptr, int")
local k_render  = mod:get_function("renderStars",   "ptr, int, ptr, ptr, ptr, int, int, float, float, float, float")
local k_tonemap = mod:get_function("tonemapBuffer", "ptr, ptr, ptr, ptr, int, int")

-- -----------------------------------------------------------------------------
-- Galaxy Particle Generation (Procedural Astrophysics Model)
-- -----------------------------------------------------------------------------
print(string.format("Generating %s stars (%d spiral arms)...",
    string.format("%d", opt_stars):reverse():gsub("(%d%d%d)", "%1,"):reverse():gsub("^,", ""), opt_arms))

ffi.cdef[[
typedef struct {
    float r;
    float theta;
    float z;
    float cr, cg, cb;
    float bright;
} StarData;
]]

local stars_host = ffi.new("StarData[?]", opt_stars)
math.randomseed(42)

local function randf(min_val, max_val)
    return min_val + math.random() * (max_val - min_val)
end

local function rand_gaussian(std)
    local u1 = math.max(1e-7, math.random())
    local u2 = math.random()
    return math.sqrt(-2.0 * math.log(u1)) * math.cos(2.0 * math.pi * u2) * std
end

local num_arms = opt_arms
local arm_winding = 2.1 -- Logarithmic spiral winding
local max_galaxy_radius = 4.0

for i = 0, opt_stars - 1 do
    local r, theta, z, cr, cg, cb, bright
    local star_type = math.random()

    if star_type < 0.32 then
        -- 1. Galactic Bulge / Core (Spikes exponentially at r=0, white-hot to warm gold)
        -- Exponential radial distribution: high central concentration
        local u = math.random()
        r = 0.95 * math.pow(u, 2.2)
        theta = randf(0, 2.0 * math.pi)
        z = rand_gaussian(0.12 * math.exp(-r * 2.0) + 0.02)

        local temp = randf(0.0, 1.0)
        cr = 1.0
        cg = 0.85 + 0.15 * temp
        cb = 0.55 + 0.45 * temp
        bright = (2.5 / (r * 1.5 + 0.25)) * randf(0.8, 1.4)

    elseif star_type < 0.90 then
        -- 2. Logarithmic Spiral Arms (Electric Cyan, Blue-White, Magenta Nebulae)
        local arm_id = math.random(1, num_arms)
        local arm_base_angle = (arm_id - 1) * (2.0 * math.pi / num_arms)

        -- Continuous distribution from inner core to outer rim
        local u = math.random()
        r = 0.35 + math.pow(u, 0.75) * (max_galaxy_radius - 0.35)

        -- Logarithmic spiral equation + density wave scatter
        local spiral_angle = arm_base_angle + arm_winding * math.log(r / 0.35 + 0.2)
        local arm_scatter = rand_gaussian(0.14 / (r * 0.4 + 0.6))
        theta = spiral_angle + arm_scatter

        -- Thin disk with natural flare at outer edges
        z = rand_gaussian(0.045 + 0.020 * r)

        local sub_type = math.random()
        if sub_type < 0.60 then
            -- Brilliant young O/B blue-white giant stars
            cr = 0.45 + 0.25 * math.random()
            cg = 0.75 + 0.25 * math.random()
            cb = 1.0
            bright = randf(1.8, 3.8)
        elseif sub_type < 0.85 then
            -- Starburst H II ionized hydrogen nebulae (Hot Magenta / Neon Pink)
            cr = 1.0
            cg = 0.20 + 0.25 * math.random()
            cb = 0.75 + 0.25 * math.random()
            bright = randf(2.2, 4.5)
        else
            -- Golden / yellow main sequence stars
            cr = 1.0
            cg = 0.95
            cb = 0.75
            bright = randf(1.2, 2.2)
        end

    else
        -- 3. Outer Spherical Halo & Globular Clusters (Reddish-orange dwarfs)
        r = randf(0.5, max_galaxy_radius * 1.3)
        theta = randf(0, 2.0 * math.pi)
        z = rand_gaussian(0.35)

        cr = 1.0
        cg = 0.55 + 0.30 * math.random()
        cb = 0.25 + 0.25 * math.random()
        bright = randf(0.8, 1.8)
    end

    local s = stars_host[i]
    s.r = r
    s.theta = theta
    s.z = z
    s.cr = cr
    s.cg = cg
    s.cb = cb
    s.bright = bright
end

-- Allocate GPU memory
local d_stars = cuda.alloc(opt_stars * ffi.sizeof("StarData"))
d_stars:to_device(stars_host)

local total_pixels = width * height
local d_accum_r = cuda.alloc("float", total_pixels)
local d_accum_g = cuda.alloc("float", total_pixels)
local d_accum_b = cuda.alloc("float", total_pixels)

local d_out = cuda.alloc(total_pixels * 3)
local h_out = ffi.new("uint8_t[?]", total_pixels * 3)
local gpu_timer = cuda.timer()

-- -----------------------------------------------------------------------------
-- Save Wallpaper Mode (PPM image output)
-- -----------------------------------------------------------------------------
local function save_ppm(filename, w, h, data)
    local f = assert(io.open(filename, "wb"))
    f:write(string.format("P6\n%d %d\n255\n", w, h))
    f:write(ffi.string(data, w * h * 3))
    f:close()
end

if opt_save then
    print(string.format("Rendering 3D Galaxy Wallpaper (%dx%d) to %s...", width, height, opt_save))
    local cam_elev = opt_tilt and math.rad(opt_tilt) or math.rad(48.0)
    local cam_azim = math.rad(32.0)
    local cam_dist = 5.2

    gpu_timer:start()
    -- Clear accumulation buffer
    k_clear:launch({ grid = math.ceil(total_pixels / 256), block = 256 },
        d_accum_r, d_accum_g, d_accum_b, total_pixels)

    -- Render stars
    k_render:launch({ grid = math.ceil(opt_stars / 256), block = 256 },
        d_stars, opt_stars, d_accum_r, d_accum_g, d_accum_b,
        width, height, opt_time, cam_elev, cam_azim, cam_dist)

    -- Tonemap & post-process
    local bx, by = 16, 16
    local gx = math.ceil(width / bx)
    local gy = math.ceil(height / by)
    k_tonemap:launch({ grid = { gx, gy }, block = { bx, by } },
        d_accum_r, d_accum_g, d_accum_b, d_out, width, height)

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
local function cleanup()
    io.write("\27[?25h\27[0m\n")
    io.flush()
end

io.write("\27[?25l\27[2J")
io.flush()

local frame_count = 0
local t_start = os.clock()
local last_time = t_start
local fps = 0.0
local gpu_ms = 0.0

local clear_grid = math.ceil(total_pixels / 256)
local star_grid  = math.ceil(opt_stars / 256)
local bx, by = 16, 16
local gx = math.ceil(width / bx)
local gy = math.ceil(height / by)

local ok, err = pcall(function()
    while true do
        local now = os.clock()
        local sim_time = now - t_start

        -- Smooth 3D orbiting camera with gentle elevation bobbing
        local cam_azim = sim_time * 0.18
        local cam_elev = opt_tilt and math.rad(opt_tilt)
            or (math.rad(46.0) + math.sin(sim_time * 0.25) * math.rad(24.0))
        local cam_dist = 4.8 + math.sin(sim_time * 0.15) * 0.6

        gpu_timer:start()

        -- 1. Clear accumulation buffer
        k_clear:launch({ grid = clear_grid, block = 256 },
            d_accum_r, d_accum_g, d_accum_b, total_pixels)

        -- 2. Render all 100,000+ stars with sub-pixel splatting
        k_render:launch({ grid = star_grid, block = 256 },
            d_stars, opt_stars, d_accum_r, d_accum_g, d_accum_b,
            width, height, sim_time, cam_elev, cam_azim, cam_dist)

        -- 3. Tonemap to 24-bit RGB
        k_tonemap:launch({ grid = { gx, gy }, block = { bx, by } },
            d_accum_r, d_accum_g, d_accum_b, d_out, width, height)

        gpu_timer:stop()
        gpu_ms = gpu_timer:elapsed_ms()

        -- Transfer to host and format terminal frame
        d_out:to_host(h_out)
        local frame_str = format_terminal_frame(width, height, h_out)

        frame_count = frame_count + 1
        local dt = now - last_time
        if dt >= 0.5 then
            fps = frame_count / (now - t_start)
            last_time = now
        end

        local hud = string.format(
            "\27[H\27[1;37;44m [3D Galaxy | %s]  Stars: %s  |  FPS: %5.1f  |  GPU: %5.2f ms  |  Tilt: %2.0f°  |  Frame: %d  \27[0m\n",
            dev_info.name,
            string.format("%d", opt_stars):reverse():gsub("(%d%d%d)", "%1,"):reverse():gsub("^,", ""),
            fps > 0 and fps or (1.0 / (gpu_ms * 1e-3)), gpu_ms, math.deg(cam_elev), frame_count
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
    print(string.format("\nFinished! Simulated %d frames at an average of %.1f FPS.",
        frame_count, frame_count / math.max(1e-4, os.clock() - t_start)))
end
