#!/usr/bin/env luajit
--[[
================================================================================
  synthwave3d.lua: 80s Cyberpunk / Outrun 3D Flyover via LuaJIT + CUDA
================================================================================
  Renders an iconic 80s Synthwave / Retrowave landscape in real-time on the GPU:
  - Infinite glowing neon wireframe highway speeding into the horizon
  - Rolling procedural mountain ranges on both sides
  - Glowing retro sun with horizontal blind cuts
  - Starry cyberpunk sky and atmospheric neon haze
  - Streams 24-bit RGB TrueColor directly to terminal or exports wallpapers

  Usage:
    luajit synthwave3d.lua [options]

  Options:
    --fps <N>          Target frame rate (default: 60)
    --frames <N>       Stop after N frames
    --size <WxH>       Resolution (default: auto-fit terminal)
    --speed <N>        Flyover speed multiplier (default: 1.0)
    --save <file.ppm>  Export high-res wallpaper (e.g. 1920x1080)
    --time <seconds>   Time offset for snapshot (default: 2.0)
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
local opt_speed = 1.0
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
    elseif a == "--speed" then
        i = i + 1; opt_speed = tonumber(args[i]) or 1.0
    elseif a == "--save" then
        i = i + 1; opt_save = args[i]
    elseif a == "--time" then
        i = i + 1; opt_time = tonumber(args[i]) or 2.0
    elseif a == "--help" or a == "-h" then
        print([[
80s Synthwave / Outrun 3D Landscape via LuaJIT + CUDA
Usage: luajit synthwave3d.lua [options]

Options:
  --fps <N>          Target FPS (default: 60)
  --frames <N>       Stop after N frames
  --size <WxH>       Resolution (e.g. 100x50, default: auto-fit terminal)
  --speed <N>        Flight speed multiplier (default: 1.0)
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
__device__ inline float length(const Vec3& v) { return sqrtf(dot(v, v)); }
__device__ inline Vec3 normalize(const Vec3& v) {
    float len = length(v);
    return len > 1e-6f ? v * (1.0f / len) : Vec3(0, 0, 0);
}
__device__ inline float clampf(float v, float minVal, float maxVal) {
    return fminf(fmaxf(v, minVal), maxVal);
}
__device__ inline float smoothstepf(float edge0, float edge1, float x) {
    float t = clampf((x - edge0) / (edge1 - edge0), 0.0f, 1.0f);
    return t * t * (3.0f - 2.0f * t);
}

// Procedural Terrain Height
__device__ float terrainHeight(float x, float z) {
    // Flat central highway corridor
    float roadDist = fabsf(x);
    float mountainFactor = smoothstepf(1.8f, 7.5f, roadDist);

    // Multi-octave rolling synthwave ridges
    float h = sinf(x * 0.4f + z * 0.15f) * 1.5f
            + sinf(x * 0.8f - z * 0.35f) * 0.9f
            + sinf(x * 1.6f + z * 0.70f) * 0.4f;

    h = fabsf(h); // Sharpen mountain ridges (voronoi-like folds)
    return h * mountainFactor * 2.2f - 0.2f;
}

// Signed Distance Field to terrain surface
__device__ float terrainSDF(Vec3 p) {
    return p.y - terrainHeight(p.x, p.z);
}

// Pseudo-random starfield
__device__ float hash21(float x, float y) {
    float n = sinf(x * 12.9898f + y * 78.233f) * 43758.5453f;
    return n - floorf(n);
}

extern "C" __global__ void renderSynthwave(
    unsigned char* out,
    int width, int height,
    float time, float speed
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;

    float aspect = (float)width / (float)height;
    float u = ((float)x + 0.5f) / (float)width * 2.0f - 1.0f;
    float v = 1.0f - ((float)y + 0.5f) / (float)height * 2.0f;
    u *= aspect;

    // Camera setup (forward flight over highway)
    float camZ = time * 7.5f * speed;
    Vec3 ro(0.0f, 1.35f, camZ);
    Vec3 rd = normalize(Vec3(u, v - 0.08f, 1.35f));

    Vec3 col(0.04f, 0.01f, 0.08f); // Deep night sky base

    // -------------------------------------------------------------------------
    // 1. Sky & Sun Rendering
    // -------------------------------------------------------------------------
    if (rd.y > -0.05f) {
        // Deep dusk horizon gradient (magenta -> violet -> dark navy)
        float horizonFade = clampf(rd.y * 3.5f, 0.0f, 1.0f);
        Vec3 skyDusk = Vec3(0.95f, 0.15f, 0.55f);  // Hot pink / magenta
        Vec3 skyZenith = Vec3(0.05f, 0.02f, 0.15f); // Deep violet
        col = skyDusk * (1.0f - horizonFade) + skyZenith * horizonFade;

        // Starfield in upper sky
        if (rd.y > 0.12f) {
            float starU = floorf((u + 50.0f) * 65.0f);
            float starV = floorf((v + 50.0f) * 65.0f);
            float starVal = hash21(starU, starV);
            if (starVal > 0.985f) {
                float twinkle = 0.6f + 0.4f * sinf(time * 3.0f + starVal * 20.0f);
                col = col + Vec3(0.9f, 0.85f, 1.0f) * twinkle;
            }
        }

        // Giant Retrowave Sun on horizon
        Vec3 sunCenter(0.0f, 0.18f, 1.0f);
        float sunDist = length(Vec3(rd.x, rd.y, 1.0f) - sunCenter);
        float sunRadius = 0.42f;

        if (sunDist < sunRadius) {
            float sunNormY = (rd.y - (sunCenter.y - sunRadius)) / (sunRadius * 2.0f);

            // Horizontal blind slats in the lower half of the sun
            bool blind = false;
            if (sunNormY < 0.65f) {
                float stripeFreq = 26.0f;
                float stripePhase = fmodf((1.0f - sunNormY) * stripeFreq, 1.0f);
                float slatThickness = 0.15f + (0.65f - sunNormY) * 0.95f;
                if (stripePhase < slatThickness) blind = true;
            }

            if (!blind) {
                // Sun vertical gradient (yellow-orange at top to deep neon pink at bottom)
                Vec3 sunTop(1.0f, 0.90f, 0.15f);
                Vec3 sunBottom(1.0f, 0.10f, 0.50f);
                Vec3 sunCol = sunTop * sunNormY + sunBottom * (1.0f - sunNormY);

                // Edge glow
                float sunGlow = smoothstepf(sunRadius, sunRadius - 0.04f, sunDist);
                col = sunCol * sunGlow + col * (1.0f - sunGlow);
            }
        } else if (sunDist < sunRadius * 1.6f) {
            // Corona glow around sun
            float corona = 1.0f - (sunDist - sunRadius) / (sunRadius * 0.6f);
            col = col + Vec3(1.0f, 0.35f, 0.6f) * (corona * corona * 0.45f);
        }
    }

    // -------------------------------------------------------------------------
    // 2. 3D Terrain Raymarching
    // -------------------------------------------------------------------------
    if (rd.y < 0.3f) {
        float t = 0.5f;
        bool hit = false;
        Vec3 p;

        for (int i = 0; i < 75; i++) {
            p = ro + rd * t;
            float d = terrainSDF(p);
            if (d < 0.005f * t) {
                hit = true;
                break;
            }
            t += d * 0.55f;
            if (t > 45.0f) break;
        }

        if (hit) {
            // Grid lines computation
            float gridScale = 0.8f;
            float gx = fabsf(fmodf(p.x, gridScale));
            float gz = fabsf(fmodf(p.z, gridScale));

            float lineWidth = 0.035f + t * 0.002f; // Anti-aliasing with distance
            bool onGridX = (gx < lineWidth) || (gx > gridScale - lineWidth);
            bool onGridZ = (gz < lineWidth) || (gz > gridScale - lineWidth);

            // Ground base color: dark reflective purple
            Vec3 groundCol(0.06f, 0.02f, 0.10f);

            // Center highway glow vs mountain glow
            Vec3 wireCol;
            if (fabsf(p.x) < 1.7f) {
                // Highway grid: glowing electric cyan
                wireCol = Vec3(0.05f, 0.95f, 1.0f);
                // Center dashed road markings
                if (fabsf(p.x) < 0.08f) {
                    if (fmodf(p.z * 0.5f, 1.0f) > 0.4f) {
                        wireCol = Vec3(1.0f, 0.95f, 0.2f); // yellow road line
                        onGridZ = true;
                    }
                }
            } else {
                // Mountain grid: hot neon magenta / neon violet
                wireCol = Vec3(1.0f, 0.12f, 0.65f);
            }

            // Combine grid wireframe and terrain body
            if (onGridX || onGridZ) {
                float intensity = 1.0f + (onGridX && onGridZ ? 0.6f : 0.0f);
                col = wireCol * intensity;
            } else {
                col = groundCol;
            }

            // Horizon neon haze / distance fog
            float fog = clampf((t - 1.0f) / 38.0f, 0.0f, 1.0f);
            Vec3 fogCol = Vec3(0.85f, 0.12f, 0.45f); // Neon pink haze
            col = col * (1.0f - fog) + fogCol * (fog * 0.95f);
        }
    }

    // Post-processing: Vignette and slight CRT bloom tone
    float vig = 1.0f - length(Vec3(u * 0.4f, v * 0.4f, 0.0f));
    col = col * clampf(vig, 0.0f, 1.0f);

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
io.write("Compiling Synthwave 3D Landscape Kernel via NVRTC...\n")
local mod = cuda.compile(cuda_source, { arch = dev_info.arch, fast_math = true, name = "synthwave.cu" })
local kernel = mod:get_function("renderSynthwave", "ptr, int, int, float, float")

local num_pixels = width * height
local buf_size = num_pixels * 3
local d_out = cuda.alloc(buf_size)
local h_out = ffi.new("uint8_t[?]", buf_size)
local gpu_timer = cuda.timer()

local function save_ppm(filename, w, h, data)
    local f = assert(io.open(filename, "wb"))
    f:write(string.format("P6\n%d %d\n255\n", w, h))
    f:write(ffi.string(data, w * h * 3))
    f:close()
end

if opt_save then
    print(string.format("Rendering Synthwave Wallpaper (%dx%d) to %s...", width, height, opt_save))
    local bx, by = 16, 16
    local gx = math.ceil(width / bx)
    local gy = math.ceil(height / by)

    gpu_timer:start()
    kernel:launch({ grid = { gx, gy }, block = { bx, by } }, d_out, width, height, opt_time, opt_speed)
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
            d_out, width, height, sim_time, opt_speed)
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

        local hud = string.format(
            "\27[H\27[1;37;45m [Synthwave 80s | %s]  FPS: %5.1f  |  GPU: %5.2f ms  |  Speed: %.1fx  |  Frame: %d  \27[0m\n",
            dev_info.name, fps > 0 and fps or (1.0 / (gpu_ms * 1e-3)), gpu_ms, opt_speed, frame_count
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
