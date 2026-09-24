#!/usr/bin/env luajit
--[[
================================================================================
  fluid_demo.lua: Real-Time GPU Navier-Stokes Fluid & Fire Dynamics via LuaJIT + CUDA
================================================================================
  A high-performance Eulerian grid fluid and fire dynamics simulator running
  entirely on CUDA cores with ANSI 24-bit TrueColor half-block terminal rendering.

  Physics & GPU Features:
  - Full Incompressible Navier-Stokes Equations:
      ∂u/∂t + (u · ∇)u = -(1/ρ)∇p + ν∇²u + f_buoyancy + f_vorticity
      ∇ · u = 0 (Mass Conservation / Incompressibility)
  - Semi-Lagrangian Bilinear Advection (Unconditionally Stable)
  - Jacobi Iterative Pressure Poisson Solver (Divergence-Free Projection)
  - Fedkiw Vorticity Confinement (Preserves Swirling Micro-Turbulences)
  - Thermal Buoyancy & Smoke Density Mass Gravity (Boussinesq Approximation)
  - Real-Time Thermal Blackbody Radiation & Volumetric Soot Rendering
  - Multi-Channel RGB Dye Mixing for Neon Cyberpunk & Bio-Plasma Aesthetics

  Interactive Controls (Live at 60 FPS):
    [W/A/S/D or ↑/↓/←/→]  Steer interactive hot flame/smoke emitter
    [m or Space]         Cycle Color Palette (Fire, Neon Cyberpunk, Toxic Plasma, Vorticity)
    [e]                  Cycle Emitter Preset (Twin Jets, Bonfire, Whirlpool, Volcano)
    [v]                  Toggle Vorticity Confinement (Turbulence ON/OFF)
    [c]                  Clear fluid grid
    [q or ESC]           Quit to terminal

  Command Line Options:
    --fps <N>          Target frame rate cap (default: 60, 0 for uncapped)
    --frames <N>       Render N frames and exit (default: infinite)
    --size <WxH>       Simulation grid size (e.g. 100x60, default: auto-fit terminal)
    --preset <1-4>     Starting emitter preset (1: Twin Jets, 2: Bonfire, 3: Whirlpool, 4: Volcano)
    --mode <0-3>       Starting color mode (0: Fire, 1: Neon, 2: Toxic, 3: Vorticity)
    --save <file.ppm>  Render a snapshot to a PPM image file and exit
    --time <seconds>   Time offset in seconds for snapshot (default: 3.0)
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
int read(int fd, void *buf, size_t count);
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
local opt_preset = 1
local opt_mode = 0
local opt_save = nil
local opt_time = 3.0

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
    elseif a == "--preset" then
        i = i + 1; opt_preset = tonumber(args[i]) or 1
    elseif a == "--mode" then
        i = i + 1; opt_mode = tonumber(args[i]) or 0
    elseif a == "--save" then
        i = i + 1; opt_save = args[i]
    elseif a == "--time" then
        i = i + 1; opt_time = tonumber(args[i]) or 3.0
    elseif a == "--help" or a == "-h" then
        print([[
Usage: ./fluid_demo.lua [options]

Options:
  --fps <N>          Target frame rate cap (default: 60)
  --frames <N>       Render N frames and exit
  --size <WxH>       Simulation grid size (e.g. 100x60, default: auto-fit terminal)
  --preset <1-4>     Emitter preset (1: Twin Jets, 2: Bonfire, 3: Whirlpool, 4: Volcano)
  --mode <0-3>       Color mode (0: Fire & Smoke, 1: Neon Cyberpunk, 2: Toxic Plasma, 3: Vorticity)
  --save <file.ppm>  Render snapshot image to PPM file
  --time <sec>       Time offset for snapshot (default: 3.0)
  --help             Show this help message

Interactive Live Controls:
  W/A/S/D or Arrows : Steer interactive emitter cursor
  m or Space        : Cycle color palettes
  e                 : Cycle emitter presets
  v                 : Toggle vorticity confinement
  c                 : Clear fluid
  q or ESC          : Quit
]])
        os.exit(0)
    end
    i = i + 1
end

-- -----------------------------------------------------------------------------
-- Terminal Helpers
-- -----------------------------------------------------------------------------
local function get_term_size()
    local ws = ffi.new("struct winsize")
    if ffi.C.ioctl(1, 0x5413, ws) == 0 and ws.ws_col > 20 and ws.ws_row > 10 then
        return ws.ws_col, ws.ws_row
    end
    local handle = io.popen("stty size 2>/dev/null", "r")
    if handle then
        local out = handle:read("*a")
        handle:close()
        local r, c = out:match("(%d+)%s+(%d+)")
        if r and c then
            return tonumber(c), tonumber(r)
        end
    end
    return 80, 24
end

local function raw_mode_on()
    os.execute("stty -icanon -echo opost onlcr min 1 time 0 2>/dev/null")
    io.write("\27[?25l") -- Hide cursor
    io.flush()
end

local function raw_mode_off()
    os.execute("stty sane 2>/dev/null")
    io.write("\27[?25h\27[0m\r\n") -- Show cursor and reset styling
    io.flush()
end

local read_buf = ffi.new("char[32]")
local function poll_key()
    local bytes_avail = ffi.new("int[1]")
    ffi.C.ioctl(0, 0x541B, bytes_avail) -- FIONREAD
    if bytes_avail[0] <= 0 then return nil end

    local n = ffi.C.read(0, read_buf, math.min(bytes_avail[0], 31))
    if n <= 0 then return nil end

    if n == 1 then
        local b = read_buf[0]
        if b == 113 or b == 81 or b == 3 or b == 27 then return "quit" end -- q/Q/Ctrl+C/ESC
        if b == 109 or b == 77 or b == 32 then return "mode" end           -- m/M/Space
        if b == 101 or b == 69 then return "preset" end                   -- e/E
        if b == 118 or b == 86 then return "vorticity" end                -- v/V
        if b == 99  or b == 67 then return "clear" end                    -- c/C
        if b == 119 or b == 87 or b == 107 or b == 75 then return "up" end    -- w/W/k/K
        if b == 115 or b == 83 or b == 106 or b == 74 then return "down" end  -- s/S/j/J
        if b == 97  or b == 65 or b == 104 or b == 72 then return "left" end  -- a/A/h/H
        if b == 100 or b == 68 or b == 108 or b == 76 then return "right" end -- d/D/l/L
    elseif n >= 3 and read_buf[0] == 27 then
        local prefix = read_buf[1]
        local code = read_buf[2]
        if prefix == 91 or prefix == 79 then -- '[' or 'O'
            if code == 65 then return "up" end
            if code == 66 then return "down" end
            if code == 68 then return "left" end
            if code == 67 then return "right" end
        end
    end
    return nil
end

-- -----------------------------------------------------------------------------
-- Determine Resolution
-- -----------------------------------------------------------------------------
local term_cols, term_rows = get_term_size()
local hud_lines = 2
local term_render_rows = math.max(10, term_rows - hud_lines)

-- Width is strictly capped at term_cols - 2 to prevent terminal auto-wrap line gaps
local width = opt_width or math.max(40, term_cols - 2)
local height = opt_height or (term_render_rows * 2)

-- Ensure even width and height for half-block rendering
if width % 2 ~= 0 then width = width - 1 end
if height % 2 ~= 0 then height = height + 1 end

-- -----------------------------------------------------------------------------
-- Initialize CUDA Device
-- -----------------------------------------------------------------------------
local dev_info = cuda.init(0)

print(string.format("\27[1;36m=== CUDA GPU Navier-Stokes Fluid & Fire Dynamics ===\27[0m"))
print(string.format("  Device     : \27[1;32m%s\27[0m (%s)", dev_info.name, dev_info.arch))
print(string.format("  Grid Size  : \27[1;33m%dx%d\27[0m (%d cells)", width, height, width * height))
print(string.format("  VRAM Total : %.0f MB\n", dev_info.total_memory_mb))

-- -----------------------------------------------------------------------------
-- CUDA C Navier-Stokes Solver Kernel
-- -----------------------------------------------------------------------------
local cuda_source = [[
#include <cuda_runtime.h>

typedef unsigned char uint8_t;

#define CLAMP(v, minv, maxv) ((v) < (minv) ? (minv) : ((v) > (maxv) ? (maxv) : (v)))

__device__ inline float sample_bilinear(const float* __restrict__ field, float x, float y, int w, int h) {
    x = fmaxf(0.0f, fminf(x, (float)(w - 1)));
    y = fmaxf(0.0f, fminf(y, (float)(h - 1)));
    int x0 = (int)x;
    int y0 = (int)y;
    int x1 = x0 < w - 1 ? x0 + 1 : x0;
    int y1 = y0 < h - 1 ? y0 + 1 : y0;
    float sx = x - (float)x0;
    float sy = y - (float)y0;

    float v00 = field[y0 * w + x0];
    float v10 = field[y0 * w + x1];
    float v01 = field[y1 * w + x0];
    float v11 = field[y1 * w + x1];

    float top = (1.0f - sx) * v00 + sx * v10;
    float bot = (1.0f - sx) * v01 + sx * v11;
    return (1.0f - sy) * top + sy * bot;
}

// -----------------------------------------------------------------------------
// 1. Clear Fields
// -----------------------------------------------------------------------------
extern "C" __global__ void k_clear_fields(
    float* __restrict__ vx, float* __restrict__ vy,
    float* __restrict__ dens, float* __restrict__ temp,
    float* __restrict__ dye_r, float* __restrict__ dye_g, float* __restrict__ dye_b,
    int n
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        vx[i] = 0.0f;
        vy[i] = 0.0f;
        dens[i] = 0.0f;
        temp[i] = 0.0f;
        dye_r[i] = 0.0f;
        dye_g[i] = 0.0f;
        dye_b[i] = 0.0f;
    }
}

// -----------------------------------------------------------------------------
// 2. Add Sources (Emitters & Interactive Cursor)
// -----------------------------------------------------------------------------
extern "C" __global__ void k_add_sources(
    float* __restrict__ vx, float* __restrict__ vy,
    float* __restrict__ dens, float* __restrict__ temp,
    float* __restrict__ dye_r, float* __restrict__ dye_g, float* __restrict__ dye_b,
    int w, int h, float time, int preset,
    float user_x, float user_y, float user_vx, float user_vy, int user_active
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= w || y >= h) return;
    int idx = y * w + x;

    if (preset == 1) {
        // Preset 1: Twin Colliding Swirling Flame Jets (Magenta & Cyan)
        float flutter = sinf(time * 3.8f + (float)y * 0.15f) * 4.0f;
        float y_base = (float)h * 0.92f;

        float x1 = (float)w * (0.33f + 0.05f * sinf(time * 1.8f));
        float x2 = (float)w * (0.67f - 0.05f * sinf(time * 1.8f));

        // Distance from left and right nozzles
        float d1 = ((float)x - x1) * ((float)x - x1) + ((float)y - y_base) * ((float)y - y_base);
        float d2 = ((float)x - x2) * ((float)x - x2) + ((float)y - y_base) * ((float)y - y_base);

        float f1 = expf(-d1 * 0.10f);
        float f2 = expf(-d2 * 0.10f);

        // Smooth continuous burner bridge connecting both nozzles (eliminates stagnation gaps)
        float x_center = (float)w * 0.5f;
        float half_span = (x2 - x1) * 0.5f;
        float rel_x = ((float)x - x_center) / (half_span + 1e-3f);
        float dy_base = (float)y - y_base;
        float bridge_envelope = expf(-rel_x * rel_x * 1.5f) * expf(-dy_base * dy_base * 0.12f);
        float f_bridge = (x >= x1 - 3.0f && x <= x2 + 3.0f && fabsf(dy_base) < 6.0f) ? bridge_envelope * 0.85f : 0.0f;

        // Left Nozzle: Always aims inwards (+vx towards center) with powerful upward thrust
        if (f1 > 0.01f) {
            vy[idx] -= 42.0f * f1;
            vx[idx] += (22.0f + 6.0f * sinf(time * 2.3f) + flutter) * f1;
            dens[idx] = fmaxf(dens[idx], 1.0f * f1);
            temp[idx] = fmaxf(temp[idx], 1.30f * f1);
            dye_r[idx] = fmaxf(dye_r[idx], 1.0f * f1);
            dye_g[idx] = fmaxf(dye_g[idx], 0.15f * f1);
            dye_b[idx] = fmaxf(dye_b[idx], 0.70f * f1);
        }

        // Right Nozzle: Always aims inwards (-vx towards center) with powerful upward thrust
        if (f2 > 0.01f) {
            vy[idx] -= 42.0f * f2;
            vx[idx] -= (22.0f + 6.0f * sinf(time * 2.3f) - flutter) * f2;
            dens[idx] = fmaxf(dens[idx], 1.0f * f2);
            temp[idx] = fmaxf(temp[idx], 1.30f * f2);
            dye_r[idx] = fmaxf(dye_r[idx], 0.10f * f2);
            dye_g[idx] = fmaxf(dye_g[idx], 0.85f * f2);
            dye_b[idx] = fmaxf(dye_b[idx], 1.00f * f2);
        }

        // Burner Bridge: Smooth upward combustion uniting the two flames into one roaring fire
        if (f_bridge > 0.01f) {
            vy[idx] -= (30.0f + 8.0f * f_bridge) * f_bridge;
            vx[idx] += sinf(time * 3.2f + (float)y * 0.2f) * 5.0f * f_bridge;
            dens[idx] = fmaxf(dens[idx], 0.95f * f_bridge);
            temp[idx] = fmaxf(temp[idx], 1.20f * f_bridge);
            dye_r[idx] = fmaxf(dye_r[idx], (0.55f - 0.45f * rel_x) * f_bridge);
            dye_g[idx] = fmaxf(dye_g[idx], 0.50f * f_bridge);
            dye_b[idx] = fmaxf(dye_b[idx], (0.55f + 0.45f * rel_x) * f_bridge);
        }
    } else if (preset == 2) {
        // Preset 2: Infernal Roaring Bonfire
        if (y >= (int)((float)h * 0.90f) && x >= (int)((float)w * 0.22f) && x <= (int)((float)w * 0.78f)) {
            float rel_x = ((float)x - (float)w * 0.5f) / ((float)w * 0.28f);
            float base_shape = expf(-rel_x * rel_x * 2.0f);
            float turbulent_flicker = sinf((float)x * 0.3f + time * 5.0f) * 0.25f;
            float strength = fmaxf(0.0f, base_shape + turbulent_flicker);

            vy[idx] -= (22.0f + 14.0f * strength) * 0.8f;
            vx[idx] += sinf((float)x * 0.2f + time * 2.5f) * 4.0f;
            dens[idx] = fmaxf(dens[idx], 0.90f * strength);
            temp[idx] = fmaxf(temp[idx], 1.15f * strength);
            dye_r[idx] = fmaxf(dye_r[idx], 1.00f * strength);
            dye_g[idx] = fmaxf(dye_g[idx], 0.45f * strength);
            dye_b[idx] = fmaxf(dye_b[idx], 0.05f * strength);
        }
    } else if (preset == 3) {
        // Preset 3: Triple Galactic Whirlpool
        float cx = (float)w * 0.5f;
        float cy = (float)h * 0.5f;
        float radius = fminf((float)w, (float)h) * 0.24f;
        float omega = 1.9f;

        for (int k = 0; k < 3; k++) {
            float ang = time * omega + (float)k * 2.0943951f; // 2*pi/3
            float ex = cx + radius * cosf(ang);
            float ey = cy + radius * sinf(ang);
            float ed = ((float)x - ex) * ((float)x - ex) + ((float)y - ey) * ((float)y - ey);
            if (ed < 30.0f) {
                float f = expf(-ed * 0.15f);
                float tang_x = -sinf(ang) * 32.0f;
                float tang_y =  cosf(ang) * 32.0f;
                vx[idx] += tang_x * f;
                vy[idx] += tang_y * f;
                dens[idx] = fmaxf(dens[idx], 1.0f * f);
                temp[idx] = fmaxf(temp[idx], 1.1f * f);

                if (k == 0) {
                    dye_r[idx] = fmaxf(dye_r[idx], 1.0f * f);
                    dye_g[idx] = fmaxf(dye_g[idx], 0.1f * f);
                    dye_b[idx] = fmaxf(dye_b[idx], 0.2f * f);
                } else if (k == 1) {
                    dye_r[idx] = fmaxf(dye_r[idx], 0.1f * f);
                    dye_g[idx] = fmaxf(dye_g[idx], 1.0f * f);
                    dye_b[idx] = fmaxf(dye_b[idx], 0.3f * f);
                } else {
                    dye_r[idx] = fmaxf(dye_r[idx], 0.2f * f);
                    dye_g[idx] = fmaxf(dye_g[idx], 0.4f * f);
                    dye_b[idx] = fmaxf(dye_b[idx], 1.0f * f);
                }
            }
        }
    } else if (preset == 4) {
        // Preset 4: Explosive Volcano / Solar Flare
        float vx_center = (float)w * 0.5f;
        float vy_center = (float)h * 0.92f;
        float pulse = powf(fmaxf(0.0f, sinf(time * 3.5f)), 3.0f);
        float d = ((float)x - vx_center) * ((float)x - vx_center) + ((float)y - vy_center) * ((float)y - vy_center);
        if (d < 45.0f) {
            float f = expf(-d * 0.10f) * (0.4f + 1.2f * pulse);
            vy[idx] -= (35.0f + 40.0f * pulse) * f;
            vx[idx] += sinf(time * 5.0f) * 16.0f * f;
            dens[idx] = fmaxf(dens[idx], 1.2f * f);
            temp[idx] = fmaxf(temp[idx], 1.4f * f);
            dye_r[idx] = fmaxf(dye_r[idx], 1.0f * f);
            dye_g[idx] = fmaxf(dye_g[idx], 0.8f * f);
            dye_b[idx] = fmaxf(dye_b[idx], 0.1f * f);
        }
    }

    // Interactive user emitter cursor
    if (user_active) {
        float udx = (float)x - user_x;
        float udy = (float)y - user_y;
        float udist = udx * udx + udy * udy;
        if (udist < 28.0f) {
            float uf = expf(-udist * 0.16f);
            vx[idx] += user_vx * 25.0f * uf;
            vy[idx] += user_vy * 25.0f * uf - 18.0f * uf;
            dens[idx] = fmaxf(dens[idx], 1.2f * uf);
            temp[idx] = fmaxf(temp[idx], 1.3f * uf);
            dye_r[idx] = fmaxf(dye_r[idx], 1.0f * uf);
            dye_g[idx] = fmaxf(dye_g[idx], 0.9f * uf);
            dye_b[idx] = fmaxf(dye_b[idx], 1.0f * uf);
        }
    }
}

// -----------------------------------------------------------------------------
// 3. Thermal Buoyancy Force
// -----------------------------------------------------------------------------
extern "C" __global__ void k_buoyancy(
    float* __restrict__ vy,
    const float* __restrict__ temp,
    const float* __restrict__ dens,
    int w, int h, float dt,
    float alpha, float beta
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= w || y >= h) return;
    int idx = y * w + x;
    // Note: y=0 is top, y=h-1 is bottom. Upward force is negative y!
    vy[idx] += (-beta * temp[idx] + alpha * dens[idx]) * dt;
}

// -----------------------------------------------------------------------------
// 4. Vorticity Confinement (Fedkiw et al.)
// -----------------------------------------------------------------------------
extern "C" __global__ void k_vorticity_calc(
    const float* __restrict__ vx, const float* __restrict__ vy,
    float* __restrict__ vort, int w, int h
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= w || y >= h) return;
    int xL = x > 0 ? x - 1 : 0;
    int xR = x < w - 1 ? x + 1 : w - 1;
    int yT = y > 0 ? y - 1 : 0;
    int yB = y < h - 1 ? y + 1 : h - 1;
    vort[y * w + x] = (vy[y * w + xR] - vy[y * w + xL] - (vx[yB * w + x] - vx[yT * w + x])) * 0.5f;
}

extern "C" __global__ void k_vorticity_apply(
    float* __restrict__ vx, float* __restrict__ vy,
    const float* __restrict__ vort, int w, int h, float dt, float eps
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= w || y >= h) return;
    int xL = x > 0 ? x - 1 : 0;
    int xR = x < w - 1 ? x + 1 : w - 1;
    int yT = y > 0 ? y - 1 : 0;
    int yB = y < h - 1 ? y + 1 : h - 1;

    float eta_x = (fabsf(vort[y * w + xR]) - fabsf(vort[y * w + xL])) * 0.5f;
    float eta_y = (fabsf(vort[yB * w + x]) - fabsf(vort[yT * w + x])) * 0.5f;
    float len = sqrtf(eta_x * eta_x + eta_y * eta_y) + 1e-2f;
    float w_val = vort[y * w + x];
    vx[y * w + x] += (eps * (eta_y / len) * w_val) * dt;
    vy[y * w + x] += (-eps * (eta_x / len) * w_val) * dt;
}

// -----------------------------------------------------------------------------
// 5. Semi-Lagrangian Advection
// -----------------------------------------------------------------------------
extern "C" __global__ void k_advect_vel(
    float* __restrict__ vx_out, float* __restrict__ vy_out,
    const float* __restrict__ vx_in, const float* __restrict__ vy_in,
    int w, int h, float dt
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= w || y >= h) return;
    int idx = y * w + x;
    float u = vx_in[idx], v = vy_in[idx];
    float new_u = sample_bilinear(vx_in, (float)x - u * dt, (float)y - v * dt, w, h);
    float new_v = sample_bilinear(vy_in, (float)x - u * dt, (float)y - v * dt, w, h);

    // 5-point Laplacian smoothing to suppress odd-even checkerboard decoupling
    int xL = x > 0 ? x - 1 : 0;
    int xR = x < w - 1 ? x + 1 : w - 1;
    int yT = y > 0 ? y - 1 : 0;
    int yB = y < h - 1 ? y + 1 : h - 1;
    float lap_u = (vx_in[y * w + xL] + vx_in[y * w + xR] + vx_in[yT * w + x] + vx_in[yB * w + x] - 4.0f * u) * 0.015f;
    float lap_v = (vy_in[y * w + xL] + vy_in[y * w + xR] + vy_in[yT * w + x] + vy_in[yB * w + x] - 4.0f * v) * 0.015f;
    new_u += lap_u;
    new_v += lap_v;

    // Clamp velocity
    new_u = fmaxf(-35.0f, fminf(new_u, 35.0f));
    new_v = fmaxf(-35.0f, fminf(new_v, 35.0f));

    // Free-slip side and bottom boundaries
    if (x == 0 && new_u < 0.0f) new_u = 0.0f;
    if (x == w - 1 && new_u > 0.0f) new_u = 0.0f;
    if (y == h - 1 && new_v > 0.0f) new_v = 0.0f;
    // Open ceiling at top (smoke vents out)
    if (y == 0 && new_v > 0.0f) new_v = 0.0f;

    vx_out[idx] = new_u * 0.998f;
    vy_out[idx] = new_v * 0.998f;
}

extern "C" __global__ void k_advect_scalars(
    float* __restrict__ dens_out, float* __restrict__ temp_out,
    float* __restrict__ dye_r_out, float* __restrict__ dye_g_out, float* __restrict__ dye_b_out,
    const float* __restrict__ dens_in, const float* __restrict__ temp_in,
    const float* __restrict__ dye_r_in, const float* __restrict__ dye_g_in, const float* __restrict__ dye_b_in,
    const float* __restrict__ vx, const float* __restrict__ vy,
    int w, int h, float dt, float dens_dissip, float temp_cooling
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= w || y >= h) return;
    int idx = y * w + x;
    float u = vx[idx], v = vy[idx];
    float src_x = (float)x - u * dt;
    float src_y = (float)y - v * dt;

    dens_out[idx]  = fmaxf(0.0f, sample_bilinear(dens_in,  src_x, src_y, w, h) * (1.0f - dens_dissip * dt));
    temp_out[idx]  = fmaxf(0.0f, sample_bilinear(temp_in,  src_x, src_y, w, h) * (1.0f - temp_cooling * dt));
    dye_r_out[idx] = fmaxf(0.0f, sample_bilinear(dye_r_in, src_x, src_y, w, h) * (1.0f - dens_dissip * dt));
    dye_g_out[idx] = fmaxf(0.0f, sample_bilinear(dye_g_in, src_x, src_y, w, h) * (1.0f - dens_dissip * dt));
    dye_b_out[idx] = fmaxf(0.0f, sample_bilinear(dye_b_in, src_x, src_y, w, h) * (1.0f - dens_dissip * dt));
}

// -----------------------------------------------------------------------------
// 6. Divergence & Poisson Pressure Solver (Incompressibility Projection)
// -----------------------------------------------------------------------------
extern "C" __global__ void k_divergence(
    const float* __restrict__ vx, const float* __restrict__ vy,
    float* __restrict__ div, float* __restrict__ p, int w, int h
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= w || y >= h) return;
    int idx = y * w + x;
    p[idx] = 0.0f;
    int xL = x > 0 ? x - 1 : 0;
    int xR = x < w - 1 ? x + 1 : w - 1;
    int yT = y > 0 ? y - 1 : 0;
    int yB = y < h - 1 ? y + 1 : h - 1;
    div[idx] = 0.5f * ((vx[y * w + xR] - vx[y * w + xL]) + (vy[yB * w + x] - vy[yT * w + x]));
}

extern "C" __global__ void k_jacobi(
    float* __restrict__ p_out, const float* __restrict__ p_in,
    const float* __restrict__ div, int w, int h
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= w || y >= h) return;
    int xL = x > 0 ? x - 1 : x;
    int xR = x < w - 1 ? x + 1 : x;
    int yB = y < h - 1 ? y + 1 : y;
    float pT = y > 0 ? p_in[(y - 1) * w + x] : 0.0f; // Open top ceiling p=0
    p_out[y * w + x] = 0.25f * (p_in[y * w + xL] + p_in[y * w + xR] + pT + p_in[yB * w + x] - div[y * w + x]);
}

extern "C" __global__ void k_subtract(
    float* __restrict__ vx, float* __restrict__ vy,
    const float* __restrict__ p, int w, int h
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= w || y >= h) return;
    int xL = x > 0 ? x - 1 : 0;
    int xR = x < w - 1 ? x + 1 : w - 1;
    int yT = y > 0 ? y - 1 : 0;
    int yB = y < h - 1 ? y + 1 : h - 1;

    float new_u = vx[y * w + x] - (p[y * w + xR] - p[y * w + xL]) * 0.5f;
    float new_v = vy[y * w + x] - (p[yB * w + x] - p[yT * w + x]) * 0.5f;

    if (x == 0 && new_u < 0.0f) new_u = 0.0f;
    if (x == w - 1 && new_u > 0.0f) new_u = 0.0f;
    if (y == h - 1 && new_v > 0.0f) new_v = 0.0f;

    vx[y * w + x] = new_u;
    vy[y * w + x] = new_v;
}

// -----------------------------------------------------------------------------
// 7. Thermal Blackbody Color & Volumetric Shading
// -----------------------------------------------------------------------------
__device__ inline float3 get_fire_color(float t, float d) {
    float3 flame;
    if (t < 0.12f) {
        float k = t / 0.12f;
        flame = make_float3(k * 0.45f, k * 0.03f, k * 0.01f);
    } else if (t < 0.38f) {
        float k = (t - 0.12f) / 0.26f;
        flame = make_float3(0.45f + 0.55f * k, 0.03f + 0.38f * k, 0.01f);
    } else if (t < 0.68f) {
        float k = (t - 0.38f) / 0.30f;
        flame = make_float3(1.0f, 0.41f + 0.49f * k, 0.01f + 0.15f * k);
    } else if (t < 0.92f) {
        float k = (t - 0.68f) / 0.24f;
        flame = make_float3(1.0f, 0.90f + 0.10f * k, 0.16f + 0.64f * k);
    } else {
        float k = fminf((t - 0.92f) / 0.28f, 1.0f);
        flame = make_float3(1.0f, 1.0f, 0.80f + 0.20f * k);
    }

    float smoke_val = fminf(d * 0.70f, 0.90f);
    float3 smoke = make_float3(smoke_val * 0.32f, smoke_val * 0.33f, smoke_val * 0.38f);
    float flame_alpha = fminf(t * 3.8f, 1.0f);
    float3 col;
    col.x = flame.x * flame_alpha + smoke.x * (1.0f - flame_alpha);
    col.y = flame.y * flame_alpha + smoke.y * (1.0f - flame_alpha);
    col.z = flame.z * flame_alpha + smoke.z * (1.0f - flame_alpha);
    return col;
}

extern "C" __global__ void k_render(
    const float* __restrict__ dens, const float* __restrict__ temp,
    const float* __restrict__ dye_r, const float* __restrict__ dye_g, const float* __restrict__ dye_b,
    const float* __restrict__ vx, const float* __restrict__ vy, const float* __restrict__ vort,
    uint8_t* __restrict__ out_rgb, int w, int h, int color_mode,
    float cur_x, float cur_y, int cur_active
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= w || y >= h) return;
    int idx = y * w + x;
    float d = dens[idx], t = temp[idx];
    float cr = dye_r[idx], cg = dye_g[idx], cb = dye_b[idx];
    float u = vx[idx], v = vy[idx];
    float speed = sqrtf(u * u + v * v);

    float3 col = make_float3(0.0f, 0.0f, 0.0f);
    if (color_mode == 0) {
        // Mode 0: Infernal Fire & Volcanic Smoke
        col = get_fire_color(t, d);
    } else if (color_mode == 1) {
        // Mode 1: Cyberpunk Neon Dye Mixing
        float intense = fminf(d * 1.5f, 1.8f);
        float glow = fminf(speed * 0.005f, 0.25f);
        col.x = cr * intense + glow;
        col.y = cg * intense + glow;
        col.z = cb * intense + glow;
    } else if (color_mode == 2) {
        // Mode 2: Toxic Bio-Plasma
        float val = fminf(d * 1.6f, 1.5f);
        col.x = val * 0.10f + t * 0.40f;
        col.y = val * 0.95f + cr * 0.15f;
        col.z = val * 0.30f + cb * 0.60f;
    } else {
        // Mode 3: Scientific Vorticity Heatmap
        float w_val = vort[idx] * 0.15f;
        if (w_val > 0.0f) {
            col.x = fminf(w_val, 1.0f);
            col.y = fminf(w_val * 0.4f, 1.0f);
            col.z = 0.08f;
        } else {
            col.x = 0.08f;
            col.y = fminf(-w_val * 0.4f, 1.0f);
            col.z = fminf(-w_val, 1.0f);
        }
        col.x += d * 0.2f; col.y += d * 0.2f; col.z += d * 0.2f;
    }

    // Interactive cursor ring
    if (cur_active) {
        float dx = (float)x - cur_x;
        float dy = (float)y - cur_y;
        float dist = sqrtf(dx * dx + dy * dy);
        if (dist >= 1.5f && dist <= 2.8f) {
            col.x = 1.0f; col.y = 1.0f; col.z = 1.0f;
        }
    }

    col.x = fminf(fmaxf(col.x, 0.0f), 1.0f);
    col.y = fminf(fmaxf(col.y, 0.0f), 1.0f);
    col.z = fminf(fmaxf(col.z, 0.0f), 1.0f);

    int out_idx = idx * 3;
    out_rgb[out_idx + 0] = (uint8_t)(col.x * 255.0f);
    out_rgb[out_idx + 1] = (uint8_t)(col.y * 255.0f);
    out_rgb[out_idx + 2] = (uint8_t)(col.z * 255.0f);
}
]]

-- Compile CUDA kernels
io.write("  Compiling Navier-Stokes CUDA kernels via NVRTC... ")
io.flush()
local t_comp0 = os.clock()
local mod = cuda.compile(cuda_source, {
    arch = dev_info.arch,
    fast_math = true,
    name = "fluid_solver.cu",
})
local t_comp1 = os.clock()
print(string.format("\27[1;32mDone in %.2f s!\27[0m\n", t_comp1 - t_comp0))

-- Retrieve kernel handles
local k_clear = mod:get_function("k_clear_fields", "ptr, ptr, ptr, ptr, ptr, ptr, ptr, int")
local k_add = mod:get_function("k_add_sources", "ptr, ptr, ptr, ptr, ptr, ptr, ptr, int, int, float, int, float, float, float, float, int")
local k_buoy = mod:get_function("k_buoyancy", "ptr, ptr, ptr, int, int, float, float, float")
local k_vort_calc = mod:get_function("k_vorticity_calc", "ptr, ptr, ptr, int, int")
local k_vort_app = mod:get_function("k_vorticity_apply", "ptr, ptr, ptr, int, int, float, float")
local k_adv_vel = mod:get_function("k_advect_vel", "ptr, ptr, ptr, ptr, int, int, float")
local k_adv_scal = mod:get_function("k_advect_scalars", "ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, int, int, float, float, float")
local k_div = mod:get_function("k_divergence", "ptr, ptr, ptr, ptr, int, int")
local k_jacobi = mod:get_function("k_jacobi", "ptr, ptr, ptr, int, int")
local k_sub = mod:get_function("k_subtract", "ptr, ptr, ptr, int, int")
local k_render = mod:get_function("k_render", "ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, int, int, int, float, float, int")

-- -----------------------------------------------------------------------------
-- Allocate GPU Simulation Buffers
-- -----------------------------------------------------------------------------
local total_cells = width * height
local d_vx0 = cuda.alloc("float", total_cells)
local d_vx1 = cuda.alloc("float", total_cells)
local d_vy0 = cuda.alloc("float", total_cells)
local d_vy1 = cuda.alloc("float", total_cells)

local d_dens0 = cuda.alloc("float", total_cells)
local d_dens1 = cuda.alloc("float", total_cells)
local d_temp0 = cuda.alloc("float", total_cells)
local d_temp1 = cuda.alloc("float", total_cells)

local d_dye_r0 = cuda.alloc("float", total_cells)
local d_dye_r1 = cuda.alloc("float", total_cells)
local d_dye_g0 = cuda.alloc("float", total_cells)
local d_dye_g1 = cuda.alloc("float", total_cells)
local d_dye_b0 = cuda.alloc("float", total_cells)
local d_dye_b1 = cuda.alloc("float", total_cells)

local d_p0 = cuda.alloc("float", total_cells)
local d_p1 = cuda.alloc("float", total_cells)
local d_div = cuda.alloc("float", total_cells)
local d_vort = cuda.alloc("float", total_cells)

local d_out = cuda.alloc("uint8_t", total_cells * 3)
local h_out = ffi.new("uint8_t[?]", total_cells * 3)

local gpu_timer = cuda.timer()

-- -----------------------------------------------------------------------------
-- Save Snapshot PPM Image Mode
-- -----------------------------------------------------------------------------
local function save_ppm(filename, w, h, data)
    local f = assert(io.open(filename, "wb"))
    f:write(string.format("P6\n%d %d\n255\n", w, h))
    f:write(ffi.string(data, w * h * 3))
    f:close()
end

if opt_save then
    print(string.format("Simulating fluid up to t=%.2f s and saving snapshot to %s...", opt_time, opt_save))
    local bx, by = 16, 16
    local gx = math.ceil(width / bx)
    local gy = math.ceil(height / by)
    local grid2d = { gx, gy }
    local block2d = { bx, by }

    local dt = 0.033
    local total_steps = math.ceil(opt_time / dt)

    for step = 1, total_steps do
        local sim_t = step * dt
        k_add:launch({ grid = grid2d, block = block2d },
            d_vx0, d_vy0, d_dens0, d_temp0, d_dye_r0, d_dye_g0, d_dye_b0,
            width, height, sim_t, opt_preset, 0, 0, 0, 0, 0)
        k_buoy:launch({ grid = grid2d, block = block2d }, d_vy0, d_temp0, d_dens0, width, height, dt, 1.0, 16.0)
        k_vort_calc:launch({ grid = grid2d, block = block2d }, d_vx0, d_vy0, d_vort, width, height)
        k_vort_app:launch({ grid = grid2d, block = block2d }, d_vx0, d_vy0, d_vort, width, height, dt, 10.0)
        k_adv_vel:launch({ grid = grid2d, block = block2d }, d_vx1, d_vy1, d_vx0, d_vy0, width, height, dt)
        k_adv_scal:launch({ grid = grid2d, block = block2d },
            d_dens1, d_temp1, d_dye_r1, d_dye_g1, d_dye_b1,
            d_dens0, d_temp0, d_dye_r0, d_dye_g0, d_dye_b0,
            d_vx0, d_vy0, width, height, dt, 0.08, 0.22)
        k_div:launch({ grid = grid2d, block = block2d }, d_vx1, d_vy1, d_div, d_p0, width, height)

        local pin, pout = d_p0, d_p1
        for _ = 1, 24 do
            k_jacobi:launch({ grid = grid2d, block = block2d }, pout, pin, d_div, width, height)
            pin, pout = pout, pin
        end
        k_sub:launch({ grid = grid2d, block = block2d }, d_vx1, d_vy1, pin, width, height)

        d_vx0, d_vx1 = d_vx1, d_vx0
        d_vy0, d_vy1 = d_vy1, d_vy0
        d_dens0, d_dens1 = d_dens1, d_dens0
        d_temp0, d_temp1 = d_temp1, d_temp0
        d_dye_r0, d_dye_r1 = d_dye_r1, d_dye_r0
        d_dye_g0, d_dye_g1 = d_dye_g1, d_dye_g0
        d_dye_b0, d_dye_b1 = d_dye_b1, d_dye_b0
    end

    k_render:launch({ grid = grid2d, block = block2d },
        d_dens0, d_temp0, d_dye_r0, d_dye_g0, d_dye_b0,
        d_vx0, d_vy0, d_vort, d_out, width, height, opt_mode, 0, 0, 0)
    d_out:to_host(h_out)
    save_ppm(opt_save, width, height, h_out)
    print(string.format("Snapshot saved successfully to %s (%dx%d)!", opt_save, width, height))
    os.exit(0)
end

-- -----------------------------------------------------------------------------
-- ANSI Half-Block Frame Builder
-- -----------------------------------------------------------------------------
local function build_ansi_frame(data, w, h)
    local lines = {}
    local half_h = math.floor(h / 2)

    for y = 0, half_h - 1 do
        local top_y = y * 2
        local bot_y = top_y + 1
        local line = {}
        local cur_fg = -1
        local cur_bg = -1

        for x = 0, w - 1 do
            local top_idx = (top_y * w + x) * 3
            local bot_idx = (bot_y * w + x) * 3

            local tr = data[top_idx + 0]
            local tg = data[top_idx + 1]
            local tb = data[top_idx + 2]

            local br = data[bot_idx + 0]
            local bg = data[bot_idx + 1]
            local bb = data[bot_idx + 2]

            local top_black = (tr <= 2 and tg <= 2 and tb <= 2)
            local bot_black = (br <= 2 and bg <= 2 and bb <= 2)

            if top_black and bot_black then
                -- Ambient black empty space: seamless background fill with zero glyph rendering
                if cur_bg ~= 0 then
                    line[#line + 1] = "\27[48;2;0;0;0m"
                    cur_bg = 0
                end
                line[#line + 1] = " "
            elseif top_black then
                -- Top is black, bottom is colored: lower half block (prevents bright background bleeding into top)
                if cur_bg ~= 0 then
                    line[#line + 1] = "\27[48;2;0;0;0m"
                    cur_bg = 0
                end
                local b_code = br * 65536 + bg * 256 + bb
                if cur_fg ~= b_code then
                    line[#line + 1] = string.format("\27[38;2;%d;%d;%dm", br, bg, bb)
                    cur_fg = b_code
                end
                line[#line + 1] = "▄"
            elseif bot_black then
                -- Top is colored, bottom is black: upper half block (prevents bright background bleeding into bottom)
                if cur_bg ~= 0 then
                    line[#line + 1] = "\27[48;2;0;0;0m"
                    cur_bg = 0
                end
                local t_code = tr * 65536 + tg * 256 + tb
                if cur_fg ~= t_code then
                    line[#line + 1] = string.format("\27[38;2;%d;%d;%dm", tr, tg, tb)
                    cur_fg = t_code
                end
                line[#line + 1] = "▀"
            elseif tr == br and tg == bg and tb == bb then
                -- Both colored and identical: background-colored space provides 100% seamless fill
                local b_code = br * 65536 + bg * 256 + bb
                if cur_bg ~= b_code then
                    line[#line + 1] = string.format("\27[48;2;%d;%d;%dm", br, bg, bb)
                    cur_bg = b_code
                end
                line[#line + 1] = " "
            else
                -- Both colored and different
                local t_code = tr * 65536 + tg * 256 + tb
                local b_code = br * 65536 + bg * 256 + bb
                if cur_fg ~= t_code then
                    line[#line + 1] = string.format("\27[38;2;%d;%d;%dm", tr, tg, tb)
                    cur_fg = t_code
                end
                if cur_bg ~= b_code then
                    line[#line + 1] = string.format("\27[48;2;%d;%d;%dm", br, bg, bb)
                    cur_bg = b_code
                end
                line[#line + 1] = "▀"
            end
        end
        lines[#lines + 1] = "\r\27[2K" .. table.concat(line) .. "\27[0m"
    end
    return table.concat(lines, "\r\n")
end

-- -----------------------------------------------------------------------------
-- Real-Time Interactive Simulation Loop
-- -----------------------------------------------------------------------------
local preset_names = {
    [1] = "Twin Jets",
    [2] = "Bonfire",
    [3] = "Whirlpool",
    [4] = "Volcano",
}

local mode_names = {
    [0] = "Fire & Smoke",
    [1] = "Neon Dye",
    [2] = "Bio-Plasma",
    [3] = "Vorticity",
}

local cur_preset = opt_preset
local cur_mode = opt_mode
local vorticity_enabled = true

-- Interactive user emitter cursor
local user_x = width * 0.5
local user_y = height * 0.6
local user_vx = 0.0
local user_vy = 0.0
local user_active = 0
local user_decay = 0

local running = true
local frame_count = 0
local t_start = os.clock()
local last_time = t_start
local fps = 0.0
local gpu_ms = 0.0

local bx, by = 16, 16
local gx = math.ceil(width / bx)
local gy = math.ceil(height / by)
local grid2d = { gx, gy }
local block2d = { bx, by }

raw_mode_on()
io.write("\27[2J\27[H")
io.flush()

local ok, err = pcall(function()
    while running do
        local now = os.clock()
        local sim_time = now - t_start
        local dt = 0.033

        -- Poll interactive keys
        local key = poll_key()
        if key == "quit" then
            running = false
            break
        elseif key == "mode" then
            cur_mode = (cur_mode + 1) % 4
        elseif key == "preset" then
            cur_preset = (cur_preset % 4) + 1
        elseif key == "vorticity" then
            vorticity_enabled = not vorticity_enabled
        elseif key == "clear" then
            k_clear:launch({ grid = math.ceil(total_cells / 256), block = 256 },
                d_vx0, d_vy0, d_dens0, d_temp0, d_dye_r0, d_dye_g0, d_dye_b0, total_cells)
        elseif key == "up" then
            user_y = math.max(3, user_y - 3)
            user_vy = -1.5; user_vx = 0.0; user_active = 1; user_decay = 15
        elseif key == "down" then
            user_y = math.min(height - 4, user_y + 3)
            user_vy = 1.5; user_vx = 0.0; user_active = 1; user_decay = 15
        elseif key == "left" then
            user_x = math.max(3, user_x - 3)
            user_vx = -1.5; user_vy = -0.5; user_active = 1; user_decay = 15
        elseif key == "right" then
            user_x = math.min(width - 4, user_x + 3)
            user_vx = 1.5; user_vy = -0.5; user_active = 1; user_decay = 15
        end

        if user_decay > 0 then
            user_decay = user_decay - 1
            if user_decay == 0 then user_active = 0 end
        end

        -- ---------------------------------------------------------------------
        -- Run GPU Navier-Stokes Pipeline
        -- ---------------------------------------------------------------------
        gpu_timer:start()

        -- 1. Add sources (Preset emitters + User interactive cursor)
        k_add:launch({ grid = grid2d, block = block2d },
            d_vx0, d_vy0, d_dens0, d_temp0, d_dye_r0, d_dye_g0, d_dye_b0,
            width, height, sim_time, cur_preset,
            user_x, user_y, user_vx, user_vy, user_active)

        -- 2. Thermal buoyancy force
        k_buoy:launch({ grid = grid2d, block = block2d }, d_vy0, d_temp0, d_dens0, width, height, dt, 1.0, 18.0)

        -- 3. Vorticity confinement
        k_vort_calc:launch({ grid = grid2d, block = block2d }, d_vx0, d_vy0, d_vort, width, height)
        if vorticity_enabled then
            k_vort_app:launch({ grid = grid2d, block = block2d }, d_vx0, d_vy0, d_vort, width, height, dt, 10.0)
        end

        -- 4. Semi-Lagrangian Advection
        k_adv_vel:launch({ grid = grid2d, block = block2d }, d_vx1, d_vy1, d_vx0, d_vy0, width, height, dt)
        k_adv_scal:launch({ grid = grid2d, block = block2d },
            d_dens1, d_temp1, d_dye_r1, d_dye_g1, d_dye_b1,
            d_dens0, d_temp0, d_dye_r0, d_dye_g0, d_dye_b0,
            d_vx0, d_vy0, width, height, dt, 0.08, 0.22)

        -- 5. Divergence calculation
        k_div:launch({ grid = grid2d, block = block2d }, d_vx1, d_vy1, d_div, d_p0, width, height)

        -- 6. Jacobi Pressure Poisson Solver (Incompressibility)
        local pin, pout = d_p0, d_p1
        for _ = 1, 24 do
            k_jacobi:launch({ grid = grid2d, block = block2d }, pout, pin, d_div, width, height)
            pin, pout = pout, pin
        end

        -- 7. Pressure gradient subtraction (Enforce ∇ · u = 0)
        k_sub:launch({ grid = grid2d, block = block2d }, d_vx1, d_vy1, pin, width, height)

        -- Ping-pong buffer swap
        d_vx0, d_vx1 = d_vx1, d_vx0
        d_vy0, d_vy1 = d_vy1, d_vy0
        d_dens0, d_dens1 = d_dens1, d_dens0
        d_temp0, d_temp1 = d_temp1, d_temp0
        d_dye_r0, d_dye_r1 = d_dye_r1, d_dye_r0
        d_dye_g0, d_dye_g1 = d_dye_g1, d_dye_g0
        d_dye_b0, d_dye_b1 = d_dye_b1, d_dye_b0

        -- 8. Render to RGB byte buffer
        k_render:launch({ grid = grid2d, block = block2d },
            d_dens0, d_temp0, d_dye_r0, d_dye_g0, d_dye_b0,
            d_vx0, d_vy0, d_vort, d_out, width, height, cur_mode,
            user_x, user_y, user_active)

        gpu_timer:stop()
        gpu_ms = gpu_timer:elapsed_ms()

        -- Copy pixels to host
        d_out:to_host(h_out)

        -- Construct ANSI frame
        local ansi_frame = build_ansi_frame(h_out, width, height)

        -- Construct Header HUD
        frame_count = frame_count + 1
        if frame_count % 10 == 0 then
            local dt_wall = now - last_time
            if dt_wall > 0 then fps = 10.0 / dt_wall end
            last_time = now
        end

        local vort_str = vorticity_enabled and "\27[1;32mON\27[0m" or "\27[1;31mOFF\27[0m"
        local top_hud = string.format(
            "  \27[1;37mCUDA FLUID\27[0m \27[2;37m•\27[0m \27[1;32m%4.1f FPS\27[0m \27[2;37m(%4.1f ms)\27[0m \27[2;37m•\27[0m \27[1;36m%s\27[0m \27[2;37m•\27[0m \27[1;33m%s\27[0m \27[2;37m•\27[0m Vort: %s",
            fps, gpu_ms, mode_names[cur_mode], preset_names[cur_preset], vort_str
        )
        local bottom_hud = "  \27[2;37m[W/A/S/D] Steer  [m] Palette  [e] Preset  [v] Vorticity  [c] Clear  [q] Quit\27[0m"

        -- Blit single frame buffer with top status and bottom controls to eliminate duplication
        io.write("\27[H\r\27[2K" .. top_hud .. "\27[K\r\n" .. ansi_frame .. "\r\n\r\27[2K" .. bottom_hud .. "\27[K")
        io.flush()

        -- Check frame limit
        if opt_frames and frame_count >= opt_frames then
            running = false
            break
        end

        -- Frame rate cap
        if opt_fps > 0 then
            local elapsed = os.clock() - now
            local target_frame_time = 1.0 / opt_fps
            if elapsed < target_frame_time then
                ffi.C.usleep(math.floor((target_frame_time - elapsed) * 1e6))
            end
        end
    end
end)

raw_mode_off()

if not ok and err and not err:match("interrupted") then
    print("\nError during fluid simulation: " .. tostring(err))
else
    print("\n[Exited fluid simulation cleanly. Terminal restored.]\n")
end
