#!/usr/bin/env luajit
--[[
================================================================================
  menu.lua: Interactive TUI Launcher for LuaJIT + CUDA Demos
================================================================================
  A clean, responsive terminal dashboard to browse and launch all CUDA demos:
  - Arrow keys (↑ / ↓) or (j / k) to navigate
  - Enter or number keys (1-6) to launch selected demo
  - Automatically adapts to any terminal width
  - Detailed description, GPU techniques, and recommended flags
  - Returns cleanly back to menu when any demo exits (Ctrl+C)
  - 'q' to quit
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
-- Demo Registry
-- -----------------------------------------------------------------------------
local demos = {
    {
        id = 1,
        file = "raymarch_demo.lua",
        title = "3D Raymarching Engine",
        tag = "Real-Time 3D",
        tag_color = "\27[1;34m", -- Bright Blue
        desc = "Renders an animated 3D Signed Distance Field (SDF) scene with concentric rotating gold/copper toruses, a pulsing neon cyan plasma core, a checkered reflective ground plane, soft shadows, and ambient occlusion.",
        techniques = "Sphere Tracing SDF • Blinn-Phong Shading • Soft Shadows • Ambient Occlusion",
        suggested = "./raymarch_demo.lua --fps 60",
    },
    {
        id = 2,
        file = "mandelbulb3d.lua",
        title = "3D Mandelbulb Fractal",
        tag = "Fractal DE",
        tag_color = "\27[1;35m", -- Bright Magenta
        desc = "The legendary White & Nylander 3D Mandelbulb fractal (v^N + c). Features orbit-trap iridescent crystal coloring (gold, cyan, amethyst), dynamic power morphing from N=4 to 8, rim lighting, and specular highlights.",
        techniques = "3D Spherical Coordinate DE • Orbit Traps • Rim Lighting • Ambient Occlusion",
        suggested = "./mandelbulb3d.lua --power 8.0",
    },
    {
        id = 3,
        file = "synthwave3d.lua",
        title = "80s Synthwave / Outrun Landscape",
        tag = "Retro 3D",
        tag_color = "\27[1;31m", -- Bright Red/Pink
        desc = "Infinite high-speed flight across an 80s Retrowave wireframe highway with rolling procedural mountain ranges on both sides, a giant glowing sunset with horizontal blinds, and a starry night sky.",
        techniques = "Procedural Ridge Heightfield • Horizon Sun Blind Shader • Starfield (<1 ms GPU)",
        suggested = "./synthwave3d.lua --speed 1.5",
    },
    {
        id = 4,
        file = "galaxy3d.lua",
        title = "100,000+ Star 3D Spiral Galaxy",
        tag = "Astrophysics",
        tag_color = "\27[1;36m", -- Bright Cyan
        desc = "Simulates a realistic 3D spiral galaxy with 120,000+ stars using the Vera Rubin dark matter flat rotation curve, an exponential golden core, logarithmic spiral arms with starburst nebulae, and HDR tonemapping.",
        techniques = "N-Body Flat Rotation Curve • Bilinear Atomic Splatting • Filmic HDR Tonemapping",
        suggested = "./galaxy3d.lua --arms 4 --stars 150000",
    },
    {
        id = 5,
        file = "mandelbrot_demo.lua",
        title = "GPU Smooth Fractal Explorer",
        tag = "2D Fractals",
        tag_color = "\27[1;32m", -- Bright Green
        desc = "Continuous-potential Mandelbrot deep zoomer and morphing Julia sets (c(t) = 0.7885 e^it). Uses a renormalized iteration algorithm to completely eliminate color banding.",
        techniques = "Continuous Potential Distance • Smooth Palette Mapping • Deep Zoom",
        suggested = "./mandelbrot_demo.lua --zoom",
    },
    {
        id = 6,
        file = "bench.lua",
        title = "LuaJIT CPU vs CUDA GPU Benchmark",
        tag = "Benchmark",
        tag_color = "\27[1;33m", -- Bright Yellow
        desc = "Direct hardware performance benchmark measuring execution speedups between LuaJIT JIT CPU code and CUDA GPU acceleration for 10M-element vector math and 2048x2048 2D image stencil convolution.",
        techniques = "SAXPY (10M floats) • 2D Stencil Blur • Hardware Event Profiling",
        suggested = "./bench.lua",
    },
    {
        id = 7,
        file = "fluid_demo.lua",
        title = "Navier-Stokes Fluid & Fire Sim",
        tag = "CFD Physics",
        tag_color = "\27[1;31m", -- Bright Red/Orange
        desc = "Real-time Eulerian grid fluid solver on CUDA cores. Features full Navier-Stokes equations, Jacobi pressure projection, Fedkiw vorticity confinement, thermal buoyancy, and blackbody fire & neon dye palettes.",
        techniques = "Navier-Stokes • Pressure Poisson • Vorticity Confinement • Thermal Buoyancy",
        suggested = "./fluid_demo.lua --fps 60",
    },
}

-- -----------------------------------------------------------------------------
-- Terminal Helpers
-- -----------------------------------------------------------------------------
local function get_term_size()
    local ws = ffi.new("struct winsize")
    if ffi.C.ioctl(1, 0x5413, ws) == 0 and ws.ws_col > 20 and ws.ws_row > 10 then
        return ws.ws_col, ws.ws_row
    end
    return 80, 24
end

local function raw_mode_on()
    -- Set non-canonical mode, no echo, while strictly maintaining output post-processing (opost, onlcr)
    os.execute("stty -icanon -echo opost onlcr min 1 time 0 2>/dev/null")
    io.write("\27[?25l") -- Hide cursor
    io.flush()
end

local function raw_mode_off()
    os.execute("stty sane 2>/dev/null")
    io.write("\27[?25h\27[0m\r\n") -- Show cursor and reset styling
    io.flush()
end

local read_buf = ffi.new("char[16]")
local function read_key()
    local n = ffi.C.read(0, read_buf, 16)
    if n <= 0 then return "eof" end

    -- Check if it's ESC starting an escape sequence
    if n == 1 and read_buf[0] == 27 then
        local bytes_avail = ffi.new("int[1]")
        ffi.C.ioctl(0, 0x541B, bytes_avail) -- FIONREAD
        if bytes_avail[0] == 0 then
            ffi.C.usleep(20000)
            ffi.C.ioctl(0, 0x541B, bytes_avail)
        end
        if bytes_avail[0] > 0 then
            local n2 = ffi.C.read(0, read_buf + 1, math.min(bytes_avail[0], 15))
            if n2 > 0 then n = n + n2 end
        else
            return "quit" -- Lone Escape key pressed
        end
    end

    if n == 1 then
        local b = read_buf[0]
        if b == 13 or b == 10 then return "enter" end
        if b == 113 or b == 81 then return "quit" end -- q/Q
        if b == 107 or b == 75 then return "up" end   -- k/K
        if b == 106 or b == 74 then return "down" end -- j/J
        if b == 3   then return "quit" end            -- Ctrl+C
        if b == 27  then return "quit" end            -- ESC
        if b >= 49 and b <= 55 then
            return "num_" .. (b - 48)
        end
    elseif n >= 3 and read_buf[0] == 27 then
        local prefix = read_buf[1]
        local code = read_buf[2]
        if prefix == 91 or prefix == 79 then -- '[' or 'O'
            if code == 65 then return "up" end        -- Up arrow
            if code == 66 then return "down" end      -- Down arrow
            if code == 67 then return "enter" end     -- Right arrow
            if code == 53 then return "up" end        -- PgUp
            if code == 54 then return "down" end      -- PgDn
        end
    end
    return nil
end

-- -----------------------------------------------------------------------------
-- Query Hardware Info
-- -----------------------------------------------------------------------------
local dev_info = nil
local ok, err = pcall(function()
    dev_info = cuda.init(0)
end)
if not ok or not dev_info then
    dev_info = {
        name = "NVIDIA CUDA GPU",
        arch = "compute_53",
        total_memory_mb = 2048,
    }
end

-- -----------------------------------------------------------------------------
-- Responsive Drawing Functions
-- -----------------------------------------------------------------------------
local selected = 1

local function draw_menu()
    local cols, rows = get_term_size()
    local max_w = math.max(50, math.min(cols - 1, 80))

    local function hr(char, color)
        return (color or "\27[1;36m") .. string.rep(char or "─", max_w) .. "\27[0m"
    end

    local lines = {}

    -- Header Banner
    lines[#lines + 1] = hr("=")
    lines[#lines + 1] = "  \27[1;37mCUDA LAB\27[0m  \27[2;37m•  High-Performance GPU Graphics in LuaJIT\27[0m"
    lines[#lines + 1] = string.format("  \27[2;36m%s (%s)  •  %.0f MB VRAM\27[0m",
        dev_info.name, dev_info.arch, dev_info.total_memory_mb)
    lines[#lines + 1] = hr("=")
    lines[#lines + 1] = ""

    -- Menu Items
    for i, item in ipairs(demos) do
        local is_sel = (i == selected)
        local pointer = is_sel and "\27[1;33m▶\27[0m " or "  "
        local num = string.format("[%d]", item.id)
        local tag = "[" .. item.tag .. "]"

        if is_sel then
            lines[#lines + 1] = string.format(
                "  %s\27[1;37;42m %s \27[0m \27[1;37m%-35s\27[0m  %s%s\27[0m",
                pointer, num, item.title, item.tag_color, tag
            )
        else
            lines[#lines + 1] = string.format(
                "  %s\27[2;37m %s \27[0m \27[0;37m%-35s\27[0m  %s%s\27[0m",
                pointer, num, item.title, item.tag_color, tag
            )
        end
    end

    lines[#lines + 1] = ""
    lines[#lines + 1] = hr("─")

    -- Detail Card for Selected Item
    local cur = demos[selected]
    lines[#lines + 1] = string.format("  \27[1;33mScript   :\27[0m ./%s", cur.file)

    -- Word-wrap helper function
    local function add_wrapped(label, text, label_color)
        local wrap_w = max_w - 14
        local words = {}
        for w in text:gmatch("%S+") do table.insert(words, w) end

        local line_acc = ""
        local is_first = true
        for _, w in ipairs(words) do
            if #line_acc + #w + 1 <= wrap_w then
                line_acc = (line_acc == "") and w or (line_acc .. " " .. w)
            else
                if is_first then
                    lines[#lines + 1] = string.format("  %s%-9s:\27[0m %s", label_color or "\27[1;37m", label, line_acc)
                    is_first = false
                else
                    lines[#lines + 1] = "             " .. line_acc
                end
                line_acc = w
            end
        end
        if line_acc ~= "" then
            if is_first then
                lines[#lines + 1] = string.format("  %s%-9s:\27[0m %s", label_color or "\27[1;37m", label, line_acc)
            else
                lines[#lines + 1] = "             " .. line_acc
            end
        end
    end

    add_wrapped("Overview", cur.desc, "\27[1;37m")
    add_wrapped("Tech", cur.techniques, "\27[2;36m")

    lines[#lines + 1] = string.format("  \27[2;37mCommand  :\27[0m \27[0;37m%s\27[0m", cur.suggested)
    lines[#lines + 1] = hr("─")

    -- Navigation Footer
    if max_w < 76 then
        lines[#lines + 1] = "  \27[1;37;44m [↑/↓] Navigate   [Enter] Run   [1-7] Select   [q] Quit \27[0m"
    else
        lines[#lines + 1] = "  \27[1;37;44m [↑/↓ or j/k] Navigate   [Enter] Run Demo   [1-7] Quick Select   [q] Quit \27[0m"
    end
    lines[#lines + 1] = hr("=")

    -- Render with explicit Carriage Return and Line Clear on every line (\r\27[2K ... \r\n)
    -- This 100% eliminates the terminal staircase effect under all terminal configurations
    local out = "\27[H\27[2J"
    for _, line in ipairs(lines) do
        out = out .. "\r\27[2K" .. line .. "\r\n"
    end
    io.write(out)
    io.flush()
end

-- -----------------------------------------------------------------------------
-- Launch Demo Execution
-- -----------------------------------------------------------------------------
local function launch_demo(demo)
    raw_mode_off()

    io.write("\27[2J\27[H\r")
    io.flush()
    print(string.format("\27[1;32m=== Launching %s (%s) ===\27[0m", demo.title, demo.file))
    print("\27[2;37mPress Ctrl+C at any time during execution to return to the menu.\27[0m\n")

    -- Run script directly
    local cmd = string.format("./%s", demo.file)
    os.execute(cmd)

    io.write("\r\n\27[1;33m[Program finished. Press Enter to return to menu...]\27[0m\r\n")
    io.flush()
    io.read()

    raw_mode_on()
end

-- -----------------------------------------------------------------------------
-- Main Event Loop
-- -----------------------------------------------------------------------------
local function main()
    raw_mode_on()

    while true do
        draw_menu()
        local key = read_key()

        if key == "up" then
            selected = selected - 1
            if selected < 1 then selected = #demos end
        elseif key == "down" then
            selected = selected + 1
            if selected > #demos then selected = 1 end
        elseif key == "enter" then
            launch_demo(demos[selected])
        elseif key and key:match("^num_(%d)") then
            local num = tonumber(key:match("^num_(%d)"))
            if num >= 1 and num <= #demos then
                selected = num
                launch_demo(demos[selected])
            end
        elseif key == "quit" or key == "eof" then
            break
        end

        ffi.C.usleep(15000) -- ~60 Hz poll
    end

    raw_mode_off()
    print("\nGoodbye! Thanks for exploring CUDA Lab.\n")
end

-- Run protected with clean terminal restoration
local ok_run, err_run = pcall(main)
raw_mode_off()
if not ok_run and err_run and not err_run:match("interrupted") then
    print("\nError: " .. tostring(err_run))
end
