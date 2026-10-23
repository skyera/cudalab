#!/usr/bin/env luajit
--[[
================================================================================
  life_demo.lua: GPU Toroidal Conway's Game of Life via LuaJIT + CUDA
================================================================================
  A massive-scale, high-performance cellular automata simulation running on
  NVIDIA CUDA cores with sub-pixel Unicode Braille terminal rendering.

  Features:
  - Massive Toroidal Grid: 1024x1024, 2048x2048, or 4096x4096 (16.7M cells)
  - Sub-Pixel Unicode Braille: 8 dots per cell (2x4 grid) for crisp fidelity
  - Multi-Scale Dynamic LOD Zoom: Zoom from 1:1 single-cell view out to 32x
  - Dynamic Cell Longevity & Phosphor Trails: Cell age tracking & CRT ghosting
  - 5 Cybernetic Palettes: Matrix Phosphor, Cyberpunk, Amber CRT, Ocean, Magma
  - Iconic Pattern Stamper: Gosper Glider Gun, Simkin Gun, Acorn, Pulsar, etc.
  - Multi-Speed Simulation: 1 to 20 GPU generations per rendered frame

  Interactive Controls (Live at 60 FPS):
    [W/A/S/D or ↑/↓/←/→ or H/J/K/L]  Pan Viewport across the universe
    [+ / - or z / x]                Zoom In / Zoom Out (1x to 32x)
    [Space]                         Pause / Resume simulation
    [n or .]                        Single-step 1 generation (when paused)
    [ [ / ] ]                       Adjust simulation speed (1x to 20x steps/frame)
    [p / P]                         Stamp iconic pattern at viewport center
    [r]                             Randomize universe (primordial soup)
    [c]                             Clear universe (all dead)
    [m]                             Cycle color palette
    [t]                             Reset viewport camera to center (0, 0)
    [q or ESC]                      Quit to terminal

  Command Line Options:
    --fps <N>          Target frame rate cap (default: 60)
    --grid <N>         Universe dimension N x N: 1024, 2048, or 4096 (default: 2048)
    --zoom <N>         Initial zoom factor (1, 2, 4, 8, 16, 32; default: 1)
    --palette <0-4>    Initial color palette (0: Matrix, 1: Cyberpunk, 2: Amber, 3: Ocean, 4: Magma)
    --speed <N>        Initial simulation steps per frame (default: 1)
    --preset <name>    Initial pattern stamped at center (gosper, acorn, pulsar, etc.)
    --save <file.ppm>  Render a snapshot image of current view to PPM and exit
    --help             Show help message
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
local opt_grid = 2048
local opt_zoom = 1
local opt_palette = 0
local opt_speed = 1
local opt_preset = "gosper"
local opt_save = nil
local opt_frames = nil

local i = 1
while i <= #args do
    local a = args[i]
    if a == "--fps" then
        i = i + 1; opt_fps = tonumber(args[i]) or 60
    elseif a == "--grid" then
        i = i + 1; opt_grid = tonumber(args[i]) or 2048
    elseif a == "--zoom" then
        i = i + 1; opt_zoom = tonumber(args[i]) or 1
    elseif a == "--palette" then
        i = i + 1; opt_palette = tonumber(args[i]) or 0
    elseif a == "--speed" then
        i = i + 1; opt_speed = tonumber(args[i]) or 1
    elseif a == "--preset" then
        i = i + 1; opt_preset = tostring(args[i] or "gosper"):lower()
    elseif a == "--frames" then
        i = i + 1; opt_frames = tonumber(args[i])
    elseif a == "--save" then
        i = i + 1; opt_save = args[i]
    elseif a == "--help" or a == "-h" then
        print([[
Usage: ./life_demo.lua [options]

Options:
  --fps <N>          Target frame rate cap (default: 60)
  --grid <N>         Universe dimension: 1024, 2048, or 4096 (default: 2048)
  --zoom <N>         Initial zoom factor: 1, 2, 4, 8, 16, 32 (default: 1)
  --palette <0-4>    Color palette (0: Matrix, 1: Cyberpunk, 2: Amber, 3: Ocean, 4: Magma)
  --speed <N>        Simulation steps per frame (default: 1)
  --preset <name>    Initial pattern (gosper, simkin, acorn, rpentomino, pulsar, spaceship, soup)
  --frames <N>       Exit after N frames
  --save <file.ppm>  Render snapshot image to PPM file and exit
  --help             Show this help message
]])
        os.exit(0)
    end
    i = i + 1
end

-- Validate grid size (must be power of 2 for fast bitwise toroidal masking)
if opt_grid ~= 1024 and opt_grid ~= 2048 and opt_grid ~= 4096 then
    if opt_grid <= 1024 then opt_grid = 1024
    elseif opt_grid <= 2048 then opt_grid = 2048
    else opt_grid = 4096 end
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
    io.write("\27[?1049h\27[?25l") -- Switch to alternate screen & hide cursor
    io.flush()
end

local function raw_mode_off()
    os.execute("stty sane 2>/dev/null")
    io.write("\27[?1049l\27[?25h\27[0m\r\n") -- Restore main screen & show cursor
    io.flush()
end

local read_buf = ffi.new("char[64]")
local function poll_key()
    local bytes_avail = ffi.new("int[1]")
    ffi.C.ioctl(0, 0x541B, bytes_avail) -- FIONREAD
    if bytes_avail[0] <= 0 then return nil end

    local n = ffi.C.read(0, read_buf, math.min(bytes_avail[0], 63))
    if n <= 0 then return nil end

    if n == 1 then
        local b = read_buf[0]
        if b == 113 or b == 81 or b == 3 or b == 27 then return "quit" end       -- q/Q/Ctrl+C/ESC
        if b == 32 then return "pause" end                                      -- Space
        if b == 110 or b == 78 or b == 46 then return "step" end                -- n/N/.
        if b == 112 or b == 80 then return "pattern" end                        -- p/P
        if b == 114 or b == 82 then return "random" end                         -- r/R
        if b == 99  or b == 67 then return "clear" end                          -- c/C
        if b == 109 or b == 77 then return "mode" end                           -- m/M
        if b == 116 or b == 84 then return "center" end                         -- t/T
        if b == 61  or b == 43 or b == 122 or b == 90 then return "zoom_in" end  -- =/+ / z/Z
        if b == 45  or b == 95 or b == 120 or b == 88 then return "zoom_out" end -- -/_ / x/X
        if b == 91 then return "speed_down" end                                 -- [
        if b == 93 then return "speed_up" end                                   -- ]
        if b == 119 or b == 87 or b == 107 or b == 75 then return "up" end     -- w/W/k/K
        if b == 115 or b == 83 or b == 106 or b == 74 then return "down" end   -- s/S/j/J
        if b == 97  or b == 65 or b == 104 or b == 72 then return "left" end   -- a/A/h/H
        if b == 100 or b == 68 or b == 108 or b == 76 then return "right" end  -- d/D/l/L
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
-- Precompute Braille UTF-8 Lookup Table (0x2800 .. 0x28FF)
-- -----------------------------------------------------------------------------
local BRAILLE_CHARS = {}
for mask = 0, 255 do
    if mask == 0 then
        BRAILLE_CHARS[mask] = " "
    else
        local b1 = 0xE2
        local b2 = 0xA0 + bit.band(bit.rshift(mask, 6), 0x03)
        local b3 = 0x80 + bit.band(mask, 0x3F)
        BRAILLE_CHARS[mask] = string.char(b1, b2, b3)
    end
end

-- -----------------------------------------------------------------------------
-- Iconic Conway Life Patterns Catalog
-- -----------------------------------------------------------------------------
local function parse_ascii_pattern(str)
    local coords = {}
    local y = 0
    local min_x, max_x = 9999, -9999
    local min_y, max_y = 9999, -9999
    for line in str:gmatch("[^\r\n]+") do
        for x = 1, #line do
            local c = line:sub(x, x)
            if c == "O" or c == "o" or c == "#" or c == "*" then
                coords[#coords + 1] = { x - 1, y }
                if (x - 1) < min_x then min_x = x - 1 end
                if (x - 1) > max_x then max_x = x - 1 end
                if y < min_y then min_y = y end
                if y > max_y then max_y = y end
            end
        end
        y = y + 1
    end
    -- Center pattern around (0, 0)
    local cx = math.floor((min_x + max_x) / 2)
    local cy = math.floor((min_y + max_y) / 2)
    local centered = {}
    for _, pt in ipairs(coords) do
        centered[#centered + 1] = { pt[1] - cx, pt[2] - cy }
    end
    return centered
end

local PATTERNS = {
    {
        name = "Gosper Glider Gun",
        desc = "First infinite glider generator (Period 30)",
        cells = parse_ascii_pattern([[
........................O...........
......................O.O...........
............OO......OO............OO
...........O...O....OO............OO
OO........O.....O...OO..............
OO........O...O.OO....O.O...........
..........O.....O.......O...........
...........O...O....................
............OO......................
]])
    },
    {
        name = "Simkin Glider Gun",
        desc = "Modern compact glider gun (Period 120)",
        cells = parse_ascii_pattern([[
OO.....OO........................
OO.....OO........................
.................................
....OO...........................
....OO...........................
.................................
.................................
.................................
.................................
.................................
.......................OO.OO.....
......................O.....O....
......................O......O...
......................OOO...O....
............................OO...
]])
    },
    {
        name = "Acorn (Methuselah)",
        desc = "7 cells exploding into 5,206 generations of chaos",
        cells = parse_ascii_pattern([[
.O.....
...O...
OO..OOO
]])
    },
    {
        name = "R-pentomino",
        desc = "5 cells that take 1,103 generations to stabilize",
        cells = parse_ascii_pattern([[
.OO
OO.
.O.
]])
    },
    {
        name = "Diehard",
        desc = "7 cells that vanish completely at generation 130",
        cells = parse_ascii_pattern([[
......O.
OO......
.O...OOO
]])
    },
    {
        name = "Pulsar Oscillator",
        desc = "Large period-3 starburst oscillator (48 cells)",
        cells = parse_ascii_pattern([[
..OOO...OOO..
.............
O....O.O....O
O....O.O....O
O....O.O....O
..OOO...OOO..
.............
..OOO...OOO..
O....O.O....O
O....O.O....O
O....O.O....O
.............
..OOO...OOO..
]])
    },
    {
        name = "Pentadecathlon",
        desc = "High-efficiency period-15 oscillator",
        cells = parse_ascii_pattern([[
OOO
.O.
.O.
OOO
OOO
.O.
.O.
OOO
]])
    },
    {
        name = "Spaceship Squadron",
        desc = "Armada of Light, Medium, and Heavyweight spaceships",
        cells = parse_ascii_pattern([[
.O..O.................
....O.................
O...O.................
.OOOO.................
......................
.......O..............
.....O...O............
.........O............
.....O...O............
......OOOO............
......................
............OO........
..........O....O......
...............O......
..........O....O......
...........OOOO.......
]])
    },
    {
        name = "Glider Stream",
        desc = "Formation of gliders flying diagonally across space",
        cells = parse_ascii_pattern([[
.O...............
..O..............
OOO..............
........O........
.........O.......
.......OOO.......
...............O.
................O
..............OOO
]])
    },
}

local pattern_idx = 1
for idx, p in ipairs(PATTERNS) do
    if p.name:lower():find(opt_preset) then
        pattern_idx = idx
        break
    end
end

-- -----------------------------------------------------------------------------
-- Initialize CUDA Device
-- -----------------------------------------------------------------------------
local dev_info = cuda.init(0)
local grid_size = opt_grid
local total_cells = grid_size * grid_size

-- -----------------------------------------------------------------------------
-- CUDA C++ Kernels (Runtime Compiled via NVRTC)
-- -----------------------------------------------------------------------------
local cuda_source = [[
extern "C" {

// Step Conway's Game of Life on Toroidal Grid
__global__ void stepLife(
    const unsigned char* __restrict__ g_in,
    unsigned char* __restrict__ g_out,
    unsigned char* __restrict__ age_grid,
    int N
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= N || y >= N) return;
    int mask = N - 1;

    int xm = (x - 1) & mask;
    int xp = (x + 1) & mask;
    int ym = (y - 1) & mask;
    int yp = (y + 1) & mask;

    int neighbors =
        g_in[ym * N + xm] + g_in[ym * N + x] + g_in[ym * N + xp] +
        g_in[y  * N + xm]                    + g_in[y  * N + xp] +
        g_in[yp * N + xm] + g_in[yp * N + x] + g_in[yp * N + xp];

    int idx = y * N + x;
    unsigned char cur = g_in[idx];
    unsigned char next_st = 0;
    unsigned char a = age_grid[idx];

    if (cur == 1) {
        if (neighbors == 2 || neighbors == 3) {
            next_st = 1;
            a = (a < 255) ? (a + 1) : 255;
        } else {
            next_st = 0;
            // Ghost trail decay
            a = (a > 6) ? (a - 6) : 0;
        }
    } else {
        if (neighbors == 3) {
            next_st = 1;
            a = 1;
        } else {
            next_st = 0;
            a = (a > 6) ? (a - 6) : 0;
        }
    }

    g_out[idx] = next_st;
    age_grid[idx] = a;
}

// Fast Pseudo-Random Soup Generator (LCG / Xorshift)
__global__ void initRandomSoup(
    unsigned char* __restrict__ g_out,
    unsigned char* __restrict__ age_grid,
    int N,
    unsigned int seed,
    int fill_percent
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= N || y >= N) return;

    unsigned int h = (unsigned int)(y * N + x) ^ seed;
    h ^= h >> 16;
    h *= 0x85ebca6b;
    h ^= h >> 13;
    h *= 0xc2b2ae35;
    h ^= h >> 16;

    int roll = (int)(h % 100);
    int idx = y * N + x;
    if (roll < fill_percent) {
        g_out[idx] = 1;
        age_grid[idx] = 1 + (h % 30);
    } else {
        g_out[idx] = 0;
        age_grid[idx] = 0;
    }
}

// Clear Entire Grid
__global__ void clearGrid(
    unsigned char* __restrict__ g_out,
    unsigned char* __restrict__ age_grid,
    int total_cells
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < total_cells) {
        g_out[idx] = 0;
        age_grid[idx] = 0;
    }
}

// Stamp Pattern Coordinates
__global__ void stampPattern(
    unsigned char* __restrict__ g_out,
    unsigned char* __restrict__ age_grid,
    int N,
    int center_x,
    int center_y,
    const int* __restrict__ coords_x,
    const int* __restrict__ coords_y,
    int num_points
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_points) {
        int mask = N - 1;
        int gx = (center_x + coords_x[idx]) & mask;
        int gy = (center_y + coords_y[idx]) & mask;
        int cell_idx = gy * N + gx;
        g_out[cell_idx] = 1;
        age_grid[cell_idx] = 1;
    }
}

// -----------------------------------------------------------------------------
// Cybernetic Color Palette Function
// -----------------------------------------------------------------------------
__device__ void getPaletteColor(
    int mode,
    int age,
    unsigned char& r,
    unsigned char& g,
    unsigned char& b
) {
    float t = fminf(1.0f, (float)age / 60.0f);

    if (mode == 0) { // Matrix Phosphor Green
        if (t < 0.2f) {
            float u = t / 0.2f;
            r = (unsigned char)(40 + u * 60);
            g = (unsigned char)(200 + u * 55);
            b = (unsigned char)(80 + u * 40);
        } else if (t < 0.7f) {
            float u = (t - 0.2f) / 0.5f;
            r = (unsigned char)(100 - u * 80);
            g = (unsigned char)(255 - u * 35);
            b = (unsigned char)(120 - u * 50);
        } else {
            float u = (t - 0.7f) / 0.3f;
            r = (unsigned char)(20 + u * 190);
            g = (unsigned char)(220 + u * 35);
            b = (unsigned char)(70 + u * 160);
        }
    } else if (mode == 1) { // Cyberpunk Neon (Magenta -> Violet -> Laser Cyan)
        if (t < 0.4f) {
            float u = t / 0.4f;
            r = (unsigned char)(255 - u * 80);
            g = (unsigned char)(40 + u * 20);
            b = (unsigned char)(160 + u * 95);
        } else {
            float u = (t - 0.4f) / 0.6f;
            r = (unsigned char)(175 - u * 175);
            g = (unsigned char)(60 + u * 185);
            b = 255;
        }
    } else if (mode == 2) { // 1980s Amber CRT
        if (t < 0.5f) {
            float u = t / 0.5f;
            r = 255;
            g = (unsigned char)(120 + u * 60);
            b = (unsigned char)(u * 20);
        } else {
            float u = (t - 0.5f) / 0.5f;
            r = 255;
            g = (unsigned char)(180 + u * 55);
            b = (unsigned char)(20 + u * 160);
        }
    } else if (mode == 3) { // Bioluminescent Aqua Ocean
        if (t < 0.5f) {
            float u = t / 0.5f;
            r = (unsigned char)(20 + u * 40);
            g = (unsigned char)(150 + u * 105);
            b = (unsigned char)(220 + u * 35);
        } else {
            float u = (t - 0.5f) / 0.5f;
            r = (unsigned char)(60 + u * 180);
            g = 255;
            b = 255;
        }
    } else { // Magma / Solar Inferno
        if (t < 0.3f) {
            float u = t / 0.3f;
            r = (unsigned char)(180 + u * 75);
            g = (unsigned char)(30 + u * 60);
            b = 10;
        } else if (t < 0.7f) {
            float u = (t - 0.3f) / 0.4f;
            r = 255;
            g = (unsigned char)(90 + u * 130);
            b = (unsigned char)(10 + u * 40);
        } else {
            float u = (t - 0.7f) / 0.3f;
            r = 255;
            g = (unsigned char)(220 + u * 35);
            b = (unsigned char)(50 + u * 205);
        }
    }
}

// -----------------------------------------------------------------------------
// Sub-Pixel Braille Viewport Renderer Kernel
// -----------------------------------------------------------------------------
struct CellOutput {
    unsigned char mask;
    unsigned char r;
    unsigned char g;
    unsigned char b;
};

__global__ void renderBrailleViewport(
    const unsigned char* __restrict__ g_in,
    const unsigned char* __restrict__ age_grid,
    CellOutput* __restrict__ out_buf,
    int N,
    int view_cols,
    int view_rows,
    int cam_x,
    int cam_y,
    int zoom,
    int palette_mode
) {
    int cx = blockIdx.x * blockDim.x + threadIdx.x;
    int cy = blockIdx.y * blockDim.y + threadIdx.y;
    if (cx >= view_cols || cy >= view_rows) return;

    int mask = N - 1;

    // Unicode Braille 2x4 dot offsets:
    // Dot 1 (0x01): (0,0), Dot 2 (0x02): (0,1), Dot 3 (0x04): (0,2), Dot 7 (0x40): (0,3)
    // Dot 4 (0x08): (1,0), Dot 5 (0x10): (1,1), Dot 6 (0x20): (1,2), Dot 8 (0x80): (1,3)
    const int dot_dx[8] = { 0, 0, 0, 1, 1, 1, 0, 1 };
    const int dot_dy[8] = { 0, 1, 2, 0, 1, 2, 3, 3 };
    const unsigned char dot_bits[8] = { 0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80 };

    unsigned char braille_mask = 0;
    int live_dots = 0;
    int total_age = 0;

    if (zoom <= 1) {
        // Direct 1:1 single-cell sampling
        for (int k = 0; k < 8; ++k) {
            int wx = (cam_x + cx * 2 + dot_dx[k]) & mask;
            int wy = (cam_y + cy * 4 + dot_dy[k]) & mask;
            int idx = wy * N + wx;
            if (g_in[idx]) {
                braille_mask |= dot_bits[k];
                live_dots++;
                total_age += age_grid[idx];
            }
        }
    } else {
        // Multi-scale box sampling for zoomed out overview
        int step = (zoom > 8) ? (zoom / 4) : 1;
        for (int k = 0; k < 8; ++k) {
            int wx = (cam_x + (cx * 2 + dot_dx[k]) * zoom) & mask;
            int wy = (cam_y + (cy * 4 + dot_dy[k]) * zoom) & mask;

            int d_live = 0;
            int d_age = 0;
            for (int sy = 0; sy < zoom; sy += step) {
                for (int sx = 0; sx < zoom; sx += step) {
                    int gx = (wx + sx) & mask;
                    int gy = (wy + sy) & mask;
                    int idx = gy * N + gx;
                    if (g_in[idx]) {
                        d_live++;
                        d_age += age_grid[idx];
                    }
                }
            }
            if (d_live > 0) {
                braille_mask |= dot_bits[k];
                live_dots++;
                total_age += (d_age / d_live);
            }
        }
    }

    int out_idx = cy * view_cols + cx;
    out_buf[out_idx].mask = braille_mask;

    if (live_dots > 0) {
        int avg_age = total_age / live_dots;
        unsigned char r, g, b;
        getPaletteColor(palette_mode, avg_age, r, g, b);
        out_buf[out_idx].r = r;
        out_buf[out_idx].g = g;
        out_buf[out_idx].b = b;
    } else {
        out_buf[out_idx].r = 0;
        out_buf[out_idx].g = 0;
        out_buf[out_idx].b = 0;
    }
}

// High-Res Full Snapshot Render (RGB 3-channel image)
__global__ void renderFullSnapshot(
    const unsigned char* __restrict__ g_in,
    const unsigned char* __restrict__ age_grid,
    unsigned char* __restrict__ rgb_out,
    int N,
    int out_w,
    int out_h,
    int cam_x,
    int cam_y,
    int zoom,
    int palette_mode
) {
    int px = blockIdx.x * blockDim.x + threadIdx.x;
    int py = blockIdx.y * blockDim.y + threadIdx.y;
    if (px >= out_w || py >= out_h) return;

    int mask = N - 1;
    int wx = (cam_x + px * zoom) & mask;
    int wy = (cam_y + py * zoom) & mask;
    int idx = wy * N + wx;

    unsigned char r = 0, g = 0, b = 0;
    if (g_in[idx]) {
        getPaletteColor(palette_mode, age_grid[idx], r, g, b);
    } else if (age_grid[idx] > 0) {
        // Faint phosphor decay glow
        getPaletteColor(palette_mode, 1, r, g, b);
        float decay = (float)age_grid[idx] / 255.0f * 0.25f;
        r = (unsigned char)(r * decay);
        g = (unsigned char)(g * decay);
        b = (unsigned char)(b * decay);
    }

    int out_idx = (py * out_w + px) * 3;
    rgb_out[out_idx + 0] = r;
    rgb_out[out_idx + 1] = g;
    rgb_out[out_idx + 2] = b;
}

} // extern "C"
]]

local module = cuda.compile(cuda_source, { name = "life_kernels.cu" })
local k_step = module:get_function("stepLife", "ptr, ptr, ptr, int")
local k_random = module:get_function("initRandomSoup", "ptr, ptr, int, uint, int")
local k_clear = module:get_function("clearGrid", "ptr, ptr, int")
local k_stamp = module:get_function("stampPattern", "ptr, ptr, int, int, int, ptr, ptr, int")
local k_render_view = module:get_function("renderBrailleViewport", "ptr, ptr, ptr, int, int, int, int, int, int, int")
local k_render_snap = module:get_function("renderFullSnapshot", "ptr, ptr, ptr, int, int, int, int, int, int, int")

-- -----------------------------------------------------------------------------
-- GPU Buffers Allocation
-- -----------------------------------------------------------------------------
local d_grid0 = cuda.alloc("uint8_t", total_cells)
local d_grid1 = cuda.alloc("uint8_t", total_cells)
local d_age   = cuda.alloc("uint8_t", total_cells)

-- Terminal resolution and viewport setup
local term_cols, term_rows = get_term_size()
local hud_lines = 4
local view_rows = math.max(10, term_rows - hud_lines)
local view_cols = math.max(20, term_cols)

local out_cells_count = view_cols * view_rows
local d_out_cells = cuda.alloc(out_cells_count * 4) -- 4 bytes per CellOutput (mask, r, g, b)

ffi.cdef[[
typedef struct {
    uint8_t mask;
    uint8_t r;
    uint8_t g;
    uint8_t b;
} CellOut;
]]
local h_out_cells = ffi.new("CellOut[?]", out_cells_count)

-- Pattern Stamping Helper
local function stamp_pattern_at(p_idx, cx, cy)
    local pat = PATTERNS[p_idx]
    if not pat or #pat.cells == 0 then return end
    local n = #pat.cells
    local h_x = ffi.new("int[?]", n)
    local h_y = ffi.new("int[?]", n)
    for idx, pt in ipairs(pat.cells) do
        h_x[idx - 1] = pt[1]
        h_y[idx - 1] = pt[2]
    end
    local d_px = cuda.alloc("int", n)
    local d_py = cuda.alloc("int", n)
    d_px:to_device(h_x)
    d_py:to_device(h_y)

    local threads = 256
    local blocks = math.ceil(n / threads)
    k_stamp:launch({ grid = blocks, block = threads },
        d_grid0, d_age, grid_size, cx, cy, d_px, d_py, n)
    cuda.sync()
    d_px:free()
    d_py:free()
end

-- Initialize universe
local function init_universe()
    local threads_2d = { 16, 16 }
    local blocks_2d = { math.ceil(grid_size / 16), math.ceil(grid_size / 16) }

    -- Clear first
    local total_threads = 256
    local total_blocks = math.ceil(total_cells / total_threads)
    k_clear:launch({ grid = total_blocks, block = total_threads }, d_grid0, d_age, total_cells)
    k_clear:launch({ grid = total_blocks, block = total_threads }, d_grid1, d_age, total_cells)
    cuda.sync()

    if opt_preset == "soup" then
        k_random:launch({ grid = blocks_2d, block = threads_2d },
            d_grid0, d_age, grid_size, os.time(), 20)
    else
        stamp_pattern_at(pattern_idx, math.floor(grid_size / 2), math.floor(grid_size / 2))
    end
    cuda.sync()
end

init_universe()

-- -----------------------------------------------------------------------------
-- Snapshot Mode (--save file.ppm)
-- -----------------------------------------------------------------------------
local function save_ppm(filename, w, h, data)
    local f = io.open(filename, "wb")
    if not f then error("Cannot open " .. filename .. " for writing") end
    f:write(string.format("P6\n%d %d\n255\n", w, h))
    f:write(ffi.string(data, w * h * 3))
    f:close()
end

if opt_save then
    local snap_w = 1920
    local snap_h = 1080
    local snap_zoom = math.max(1, math.floor(grid_size / snap_w))
    local cam_x = math.floor(grid_size / 2) - math.floor((snap_w * snap_zoom) / 2)
    local cam_y = math.floor(grid_size / 2) - math.floor((snap_h * snap_zoom) / 2)

    local d_snap_out = cuda.alloc(snap_w * snap_h * 3)
    local h_snap_out = ffi.new("uint8_t[?]", snap_w * snap_h * 3)

    local grid2d = { math.ceil(snap_w / 16), math.ceil(snap_h / 16) }
    local block2d = { 16, 16 }

    print(string.format("Rendering high-res Life snapshot (%dx%d, zoom=%d)...", snap_w, snap_h, snap_zoom))
    k_render_snap:launch({ grid = grid2d, block = block2d },
        d_grid0, d_age, d_snap_out, grid_size, snap_w, snap_h, cam_x, cam_y, snap_zoom, opt_palette)
    cuda.sync()

    d_snap_out:to_host(h_snap_out)
    save_ppm(opt_save, snap_w, snap_h, h_snap_out)
    d_snap_out:free()
    print(string.format("Snapshot saved successfully to %s!", opt_save))
    os.exit(0)
end

-- -----------------------------------------------------------------------------
-- ANSI String Frame Builder
-- -----------------------------------------------------------------------------
local function build_ansi_frame(cells, cols, rows)
    local lines = {}
    local cur_fg = -1

    for y = 0, rows - 1 do
        local line = {}
        local row_offset = y * cols
        for x = 0, cols - 1 do
            local c = cells[row_offset + x]
            local mask = c.mask
            if mask == 0 then
                line[#line + 1] = " "
            else
                local code = bit.bor(bit.lshift(c.r, 16), bit.lshift(c.g, 8), c.b)
                if cur_fg ~= code then
                    line[#line + 1] = string.format("\27[38;2;%d;%d;%dm", c.r, c.g, c.b)
                    cur_fg = code
                end
                line[#line + 1] = BRAILLE_CHARS[mask]
            end
        end
        lines[#lines + 1] = "\r\27[2K" .. table.concat(line)
    end
    return table.concat(lines, "\r\n") .. "\27[0m"
end

-- -----------------------------------------------------------------------------
-- Main Interactive Simulation Loop
-- -----------------------------------------------------------------------------
local PALETTE_NAMES = {
    [0] = "Matrix Phosphor",
    [1] = "Cyberpunk Neon",
    [2] = "1980s Amber CRT",
    [3] = "Bioluminescent Abyss",
    [4] = "Solar Magma",
}

local cur_palette = opt_palette
local cur_zoom = opt_zoom
local cur_speed = opt_speed
local paused = false

-- Camera position (in universe grid coordinates)
local cam_x = math.floor(grid_size / 2) - math.floor((view_cols * 2 * cur_zoom) / 2)
local cam_y = math.floor(grid_size / 2) - math.floor((view_rows * 4 * cur_zoom) / 2)

local generation = 0
local frame_count = 0
local running = true
local t_start = os.clock()
local last_time = t_start
local actual_fps = 0.0
local gpu_step_ms = 0.0
local gpu_timer = cuda.timer()

local grid2d_life = { math.ceil(grid_size / 16), math.ceil(grid_size / 16) }
local block2d_life = { 16, 16 }

raw_mode_on()
io.write("\27[2J\27[H")
io.flush()

local ok, err = pcall(function()
    while running do
        local now = os.clock()

        -- Dynamic terminal resizing
        local new_cols, new_rows = get_term_size()
        local new_vrows = math.max(10, new_rows - hud_lines)
        local new_vcols = math.max(20, new_cols)
        if new_vcols ~= view_cols or new_vrows ~= view_rows then
            view_cols = new_vcols
            view_rows = new_vrows
            out_cells_count = view_cols * view_rows
            d_out_cells = cuda.alloc(out_cells_count * 4)
            h_out_cells = ffi.new("CellOut[?]", out_cells_count)
            io.write("\27[2J\27[H")
            io.flush()
        end

        -- Poll interactive keys
        local key = poll_key()
        if key == "quit" then
            running = false
            break
        elseif key == "pause" then
            paused = not paused
        elseif key == "step" then
            paused = true
            -- Advance single generation
            k_step:launch({ grid = grid2d_life, block = block2d_life },
                d_grid0, d_grid1, d_age, grid_size)
            d_grid0, d_grid1 = d_grid1, d_grid0
            generation = generation + 1
        elseif key == "up" then
            local pan_step = math.max(2, 4 * cur_zoom)
            cam_y = (cam_y - pan_step) % grid_size
        elseif key == "down" then
            local pan_step = math.max(2, 4 * cur_zoom)
            cam_y = (cam_y + pan_step) % grid_size
        elseif key == "left" then
            local pan_step = math.max(2, 4 * cur_zoom)
            cam_x = (cam_x - pan_step) % grid_size
        elseif key == "right" then
            local pan_step = math.max(2, 4 * cur_zoom)
            cam_x = (cam_x + pan_step) % grid_size
        elseif key == "zoom_in" then
            if cur_zoom > 1 then
                local old_zoom = cur_zoom
                cur_zoom = math.max(1, math.floor(cur_zoom / 2))
                -- Zoom in towards screen center
                local center_x = cam_x + math.floor((view_cols * 2 * old_zoom) / 2)
                local center_y = cam_y + math.floor((view_rows * 4 * old_zoom) / 2)
                cam_x = (center_x - math.floor((view_cols * 2 * cur_zoom) / 2)) % grid_size
                cam_y = (center_y - math.floor((view_rows * 4 * cur_zoom) / 2)) % grid_size
            end
        elseif key == "zoom_out" then
            if cur_zoom < 32 then
                local old_zoom = cur_zoom
                cur_zoom = math.min(32, cur_zoom * 2)
                -- Zoom out preserving screen center
                local center_x = cam_x + math.floor((view_cols * 2 * old_zoom) / 2)
                local center_y = cam_y + math.floor((view_rows * 4 * old_zoom) / 2)
                cam_x = (center_x - math.floor((view_cols * 2 * cur_zoom) / 2)) % grid_size
                cam_y = (center_y - math.floor((view_rows * 4 * cur_zoom) / 2)) % grid_size
            end
        elseif key == "speed_up" then
            cur_speed = math.min(20, cur_speed + 1)
        elseif key == "speed_down" then
            cur_speed = math.max(1, cur_speed - 1)
        elseif key == "mode" then
            cur_palette = (cur_palette + 1) % 5
        elseif key == "center" then
            cam_x = math.floor(grid_size / 2) - math.floor((view_cols * 2 * cur_zoom) / 2)
            cam_y = math.floor(grid_size / 2) - math.floor((view_rows * 4 * cur_zoom) / 2)
        elseif key == "clear" then
            local total_threads = 256
            local total_blocks = math.ceil(total_cells / total_threads)
            k_clear:launch({ grid = total_blocks, block = total_threads }, d_grid0, d_age, total_cells)
            k_clear:launch({ grid = total_blocks, block = total_threads }, d_grid1, d_age, total_cells)
            cuda.sync()
            generation = 0
        elseif key == "random" then
            local threads_2d = { 16, 16 }
            local blocks_2d = { math.ceil(grid_size / 16), math.ceil(grid_size / 16) }
            k_random:launch({ grid = blocks_2d, block = threads_2d },
                d_grid0, d_age, grid_size, math.floor(os.clock() * 1000) + frame_count, 22)
            cuda.sync()
            generation = 0
        elseif key == "pattern" then
            -- Stamp next pattern at current viewport center
            pattern_idx = (pattern_idx % #PATTERNS) + 1
            local center_x = (cam_x + math.floor((view_cols * 2 * cur_zoom) / 2)) % grid_size
            local center_y = (cam_y + math.floor((view_rows * 4 * cur_zoom) / 2)) % grid_size
            stamp_pattern_at(pattern_idx, center_x, center_y)
        end

        -- Step Simulation on GPU
        if not paused then
            gpu_timer:start()
            for s = 1, cur_speed do
                k_step:launch({ grid = grid2d_life, block = block2d_life },
                    d_grid0, d_grid1, d_age, grid_size)
                d_grid0, d_grid1 = d_grid1, d_grid0
                generation = generation + 1
            end
            gpu_timer:stop()
            gpu_step_ms = gpu_timer:elapsed_ms() / cur_speed
        end

        -- Render Viewport to Braille Character Buffer
        local grid2d_render = { math.ceil(view_cols / 16), math.ceil(view_rows / 16) }
        local block2d_render = { 16, 16 }
        k_render_view:launch({ grid = grid2d_render, block = block2d_render },
            d_grid0, d_age, d_out_cells, grid_size, view_cols, view_rows,
            cam_x, cam_y, cur_zoom, cur_palette)
        cuda.sync()

        -- Copy output cells back to host and build ANSI string
        d_out_cells:to_host(h_out_cells)
        local frame_ansi = build_ansi_frame(h_out_cells, view_cols, view_rows)

        -- Top HUD Bar
        local status_str = paused and "\27[1;33m[PAUSED]\27[0m" or "\27[1;32m[RUNNING]\27[0m"
        local hud_top = string.format(
            "\r\27[H\27[2K\27[1;36m▶ CUDA Conway's Life\27[0m %s │ \27[1;37mGen:\27[0m %d │ \27[1;37mGrid:\27[0m %dx%d │ \27[1;37mZoom:\27[0m %dx │ \27[1;37mCam:\27[0m (%d,%d) │ \27[1;37mPalette:\27[0m %s",
            status_str, generation, grid_size, grid_size, cur_zoom, cam_x, cam_y, PALETTE_NAMES[cur_palette]
        )

        -- Bottom HUD Bar & Controls
        local hud_bottom = string.format(
            "\r\n\27[2K\27[1;35mRate:\27[0m %.0f FPS (%d steps/f) │ \27[1;35mGPU Step:\27[0m %.2f ms │ \27[1;35mPattern [P]:\27[0m %s\r\n\27[2K\27[90m[W/A/S/D] Pan │ [+/-] Zoom │ [Space] Pause │ [N] Step │ [R] Soup │ [P] Stamp │ [M] Palette │ [Q] Quit\27[0m",
            actual_fps, cur_speed, gpu_step_ms, PATTERNS[pattern_idx].name
        )

        io.write(hud_top .. "\r\n" .. frame_ansi .. hud_bottom)
        io.flush()

        frame_count = frame_count + 1
        if frame_count % 10 == 0 then
            local dt = now - last_time
            if dt > 0 then actual_fps = 10 / dt end
            last_time = now
        end

        if opt_frames and frame_count >= opt_frames then
            running = false
            break
        end

        -- Frame rate throttling
        if opt_fps > 0 then
            local elapsed = os.clock() - now
            local target_frame = 1.0 / opt_fps
            if elapsed < target_frame then
                ffi.C.usleep(math.floor((target_frame - elapsed) * 1e6))
            end
        end
    end
end)

raw_mode_off()

if not ok and err then
    io.stderr:write("Error in simulation loop: " .. tostring(err) .. "\n")
end
