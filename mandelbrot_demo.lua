#!/usr/bin/env luajit
--[[
================================================================================
  mandelbrot_demo.lua: GPU Smooth Fractal Explorer via LuaJIT + CUDA
================================================================================
  Renders continuous-potential Mandelbrot & Julia fractals in real-time on GPU,
  streaming true 24-bit RGB ANSI graphics to the terminal, or exporting
  high-res desktop wallpapers.

  Usage:
    luajit mandelbrot_demo.lua [options]

  Options:
    --mode <mandel|julia> Fractal mode (default: mandel)
    --zoom                 Enable continuous deep zoom animation
    --fps <N>             Target FPS (default: 60)
    --frames <N>          Render N frames and exit
    --size <WxH>          Resolution (default: auto-fit terminal)
    --save <file.ppm>     Save high-res snapshot (e.g. 1920x1080)
    --help                Show this help message
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
local opt_mode = "mandel"
local opt_zoom = false
local opt_fps = 60
local opt_frames = nil
local opt_width = nil
local opt_height = nil
local opt_save = nil

local i = 1
while i <= #args do
    local a = args[i]
    if a == "--mode" then
        i = i + 1; opt_mode = args[i] or "mandel"
    elseif a == "--zoom" then
        opt_zoom = true
    elseif a == "--fps" then
        i = i + 1; opt_fps = tonumber(args[i]) or 60
    elseif a == "--frames" then
        i = i + 1; opt_frames = tonumber(args[i])
    elseif a == "--size" then
        i = i + 1
        local w, h = (args[i] or ""):match("(%d+)x(%d+)")
        if w and h then opt_width = tonumber(w); opt_height = tonumber(h) end
    elseif a == "--save" then
        i = i + 1; opt_save = args[i]
    elseif a == "--help" or a == "-h" then
        print([[
GPU Smooth Fractal Explorer via LuaJIT + CUDA
Usage: luajit mandelbrot_demo.lua [options]

Options:
  --mode <mandel|julia> Fractal mode (default: mandel)
  --zoom                 Animate deep zoom
  --fps <N>              Target FPS (default: 60)
  --frames <N>           Stop after N frames
  --size <WxH>           Resolution (default: auto-fit terminal)
  --save <file.ppm>      Export high-res wallpaper
  --help                 Show help
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

__device__ inline float clampf(float v, float minVal, float maxVal) {
    return fminf(fmaxf(v, minVal), maxVal);
}

// Cosine gradient palette generator (Inigo Quilez technique)
__device__ void palette(float t, float& r, float& g, float& b) {
    float a_r = 0.5f, a_g = 0.5f, a_b = 0.5f;
    float b_r = 0.5f, b_g = 0.5f, b_b = 0.5f;
    float c_r = 1.0f, c_g = 1.0f, c_b = 1.0f;
    float d_r = 0.0f, d_g = 0.33f, d_b = 0.67f;

    r = clampf(a_r + b_r * cosf(2.0f * PI * (c_r * t + d_r)), 0.0f, 1.0f);
    g = clampf(a_g + b_g * cosf(2.0f * PI * (c_g * t + d_g)), 0.0f, 1.0f);
    b = clampf(a_b + b_b * cosf(2.0f * PI * (c_b * t + d_b)), 0.0f, 1.0f);
}

extern "C" __global__ void renderFractal(
    unsigned char* out,
    int width, int height,
    double centerX, double centerY, double scale,
    float time, int mode, int max_iter
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;

    double aspect = (double)width / (double)height;
    double u = ((double)x + 0.5) / (double)width * 2.0 - 1.0;
    double v = ((double)y + 0.5) / (double)height * 2.0 - 1.0;
    u *= aspect;

    double zx, zy, cx, cy;

    if (mode == 0) { // Mandelbrot
        zx = 0.0;
        zy = 0.0;
        cx = centerX + u * scale;
        cy = centerY + v * scale;
    } else { // Morphing Julia Set
        zx = centerX + u * scale;
        zy = centerY + v * scale;
        cx = 0.7885 * cosf(time * 0.3f);
        cy = 0.7885 * sinf(time * 0.3f);
    }

    int iter = 0;
    double zx2 = zx * zx;
    double zy2 = zy * zy;

    while (zx2 + zy2 <= 4.0 && iter < max_iter) {
        zy = 2.0 * zx * zy + cy;
        zx = zx2 - zy2 + cx;
        zx2 = zx * zx;
        zy2 = zy * zy;
        iter++;
    }

    float r = 0.0f, g = 0.0f, b = 0.0f;

    if (iter < max_iter) {
        // Continuous smooth coloring formula
        double mag2 = zx2 + zy2;
        float nu = logf((float)(log(mag2) / 2.0 / log(2.0))) / logf(2.0f);
        float continuous_iter = (float)iter + 1.0f - nu;
        float color_t = continuous_iter * 0.035f + time * 0.05f;

        palette(color_t, r, g, b);
    } else {
        // Interior color: deep black/dark navy
        r = 0.01f; g = 0.01f; b = 0.03f;
    }

    int idx = (y * width + x) * 3;
    out[idx + 0] = (unsigned char)(r * 255.0f);
    out[idx + 1] = (unsigned char)(g * 255.0f);
    out[idx + 2] = (unsigned char)(b * 255.0f);
}
]]

io.write("Initializing CUDA...\n")
local dev_info = cuda.init(0)
local mod = cuda.compile(cuda_source, { arch = dev_info.arch, name = "fractal.cu" })
local kernel = mod:get_function("renderFractal", "ptr, int, int, double, double, double, float, int, int")

local num_pixels = width * height
local buf_size = num_pixels * 3
local d_out = cuda.alloc(buf_size)
local h_out = ffi.new("uint8_t[?]", buf_size)
local gpu_timer = cuda.timer()

local is_julia = (opt_mode:lower() == "julia")
local mode_val = is_julia and 1 or 0

-- Seahorse Valley coordinates for Mandelbrot
local center_x = is_julia and 0.0 or -0.743643887037158704752191506114774
local center_y = is_julia and 0.0 or  0.131825904205311970493132056385139
local base_scale = is_julia and 1.3 or 1.2
local max_iterations = 200

local function save_ppm(filename, w, h, data)
    local f = assert(io.open(filename, "wb"))
    f:write(string.format("P6\n%d %d\n255\n", w, h))
    f:write(ffi.string(data, w * h * 3))
    f:close()
end

if opt_save then
    print(string.format("Rendering high-res fractal (%dx%d) to %s...", width, height, opt_save))
    local bx, by = 16, 16
    local gx = math.ceil(width / bx)
    local gy = math.ceil(height / by)

    local scale = base_scale
    if opt_zoom then scale = base_scale * 0.005 end

    gpu_timer:start()
    kernel:launch({ grid = { gx, gy }, block = { bx, by } },
        d_out, width, height, center_x, center_y, scale, 0.0, mode_val, max_iterations)
    gpu_timer:stop()
    local ms = gpu_timer:elapsed_ms()

    d_out:to_host(h_out)
    save_ppm(opt_save, width, height, h_out)
    print(string.format("Saved %s successfully! Render Time: %.2f ms", opt_save, ms))
    os.exit(0)
end

-- Terminal rendering
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

        -- Compute scale
        local scale = base_scale
        if opt_zoom then
            scale = base_scale * math.exp(-sim_time * 0.45)
        end

        gpu_timer:start()
        kernel:launch({ grid = { gx, gy }, block = { bx, by } },
            d_out, width, height, center_x, center_y, scale, sim_time, mode_val, max_iterations)
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
            "\27[H\27[1;37;45m [%s %s]  FPS: %5.1f  |  GPU: %5.2f ms  |  Zoom: %.2e  |  Frame: %d  \27[0m\n",
            is_julia and "Julia" or "Mandelbrot", dev_info.name,
            fps > 0 and fps or (1.0 / (gpu_ms * 1e-3)), gpu_ms, 1.0 / scale, frame_count
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
