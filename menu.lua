#!/usr/bin/env luajit
--[[
================================================================================
  menu.lua: Interactive TUI Launcher for LuaJIT + CUDA Demos
================================================================================
  A terminal user interface that showcases and launches all CUDA GPU demos:
  - Up/Down or j/k to navigate
  - Enter or number keys (1-6) to launch selected demo
  - Detailed description, GPU techniques, and recommended flags for each script
  - Seamlessly returns to menu after each demo exits
  - 'q' or Ctrl+C to quit
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
        badge_color = "\27[1;37;44m", -- Blue badge
        desc = "Renders an animated 3D Signed Distance Field (SDF) scene with concentric rotating gold/copper toruses, a pulsing neon cyan plasma core, a checkered reflective ground plane, soft shadows, and ambient occlusion.",
        techniques = "Sphere Tracing SDF • Blinn-Phong Shading • Soft Shadows • Ambient Occlusion",
        hotkeys = "Ctrl+C to return",
        suggested = "--fps 60",
    },
    {
        id = 2,
        file = "mandelbulb3d.lua",
        title = "3D Mandelbulb Fractal",
        tag = "Fractal DE",
        badge_color = "\27[1;37;45m", -- Magenta badge
        desc = "The legendary White & Nylander 3D Mandelbulb fractal (v^N + c). Features orbit-trap iridescent crystal coloring (gold, cyan, amethyst), dynamic power morphing from N=4 to 8, rim lighting, and specular highlights.",
        techniques = "3D Spherical Coordinate DE • Orbit Traps • Rim Lighting • AO",
        hotkeys = "Ctrl+C to return",
        suggested = "--power 8.0",
    },
    {
        id = 3,
        file = "synthwave3d.lua",
        title = "80s Cyberpunk / Outrun Landscape",
        tag = "Retro 3D",
        badge_color = "\27[1;37;41m", -- Red/Pink badge
        desc = "Infinite high-speed flight across an 80s Retrowave wireframe highway with rolling procedural mountain ranges on both sides, a giant glowing sunset with horizontal blinds, and a starry night sky.",
        techniques = "Procedural Ridge Heightfield • Horizon Sun Blind Shader • Starfield (<1 ms GPU)",
        hotkeys = "Ctrl+C to return",
        suggested = "--speed 1.5",
    },
    {
        id = 4,
        file = "galaxy3d.lua",
        title = "100,000+ Star 3D Spiral Galaxy",
        tag = "Astrophysics",
        badge_color = "\27[1;37;46m", -- Cyan badge
        desc = "Simulates a realistic 3D spiral galaxy with 120,000+ stars using the Vera Rubin dark matter flat rotation curve, an exponential golden core, logarithmic spiral arms with starburst nebulae, and HDR tonemapping.",
        techniques = "N-Body Flat Rotation Curve • Bilinear Atomic Splatting • Filmic HDR Tonemapping",
        hotkeys = "Ctrl+C to return",
        suggested = "--arms 4 --stars 150000",
    },
    {
        id = 5,
        file = "mandelbrot_demo.lua",
        title = "GPU Smooth Fractal Explorer",
        tag = "2D Fractals",
        badge_color = "\27[1;37;42m", -- Green badge
        desc = "Continuous-potential Mandelbrot deep zoomer and morphing Julia sets (c(t) = 0.7885 e^it). Uses a renormalized iteration algorithm to completely eliminate color banding.",
        techniques = "Continuous Potential Distance • Smooth Palette Mapping • Deep Zoom",
        hotkeys = "Ctrl+C to return",
        suggested = "--zoom  OR  --mode julia",
    },
    {
        id = 6,
        file = "bench.lua",
        title = "LuaJIT CPU vs CUDA GPU Benchmark",
        tag = "Benchmark",
        badge_color = "\27[1;37;43m", -- Yellow badge
        desc = "Direct hardware performance benchmark measuring execution speedups between LuaJIT JIT CPU code and CUDA GPU acceleration for 10M-element vector math and 2048x2048 2D image stencil convolution.",
        techniques = "SAXPY (10M floats) • 2D Stencil Blur • Hardware Event Profiling",
        hotkeys = "Runs benchmark to completion",
        suggested = "(runs both benchmarks)",
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
    os.execute("stty raw -echo 2>/dev/null")
    io.write("\27[?25l") -- Hide cursor
    io.flush()
end

local function raw_mode_off()
    os.execute("stty sane 2>/dev/null")
    io.write("\27[?25h\27[0m") -- Show cursor and reset styling
    io.flush()
end

local read_buf = ffi.new("char[16]")
local function read_key()
    local n = ffi.C.read(0, read_buf, 16)
    if n <= 0 then return "eof" end

    if n == 1 then
        local b = read_buf[0]
        if b == 13 or b == 10 then return "enter" end
        if b == 113 or b == 81 then return "quit" end
        if b == 107 or b == 75 then return "up" end
        if b == 106 or b == 74 then return "down" end
        if b == 3   then return "quit" end -- Ctrl+C
        if b >= 49 and b <= 54 then
            return "num_" .. (b - 48)
        end
    elseif n >= 3 and read_buf[0] == 27 and read_buf[1] == 91 then
        if read_buf[2] == 65 then return "up" end
        if read_buf[2] == 66 then return "down" end
        if read_buf[2] == 67 then return "enter" end -- Right arrow = launch
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
-- TUI Drawing
-- -----------------------------------------------------------------------------
local selected = 1

local function visual_len(s)
    local clean = s:gsub("\27%[[%d;]*%a", "")
    local wide_count = 0
    clean = clean:gsub("🚀", function() wide_count = wide_count + 1; return "" end)
    clean = clean:gsub("[\194-\244][\128-\191]*", "X")
    return #clean + wide_count
end

local function box_line(pad, content, target_w)
    local vlen = visual_len(content)
    local fill = string.rep(" ", math.max(0, target_w - vlen))
    return pad .. "\27[1;36m│\27[0m" .. content .. fill .. "\27[1;36m│\27[0m"
end

local function draw_menu()
    local cols, rows = get_term_size()
    local box_w = math.min(cols - 2, 78)
    local inner_w = box_w - 2
    local pad = string.rep(" ", math.max(0, math.floor((cols - box_w) / 2)))

    local lines = {}
    lines[#lines + 1] = "\27[H\27[2J" -- Home & clear screen

    -- Title Banner
    lines[#lines + 1] = pad .. "\27[1;36m┌" .. string.rep("─", inner_w) .. "┐\27[0m"
    lines[#lines + 1] = box_line(pad, "  \27[1;37m🚀  CUDA LAB — High-Performance GPU Graphics in LuaJIT\27[0m", inner_w)

    local sub = string.format("  \27[2;37mGPU: %s (%s)  •  VRAM: %.0f MB\27[0m",
        dev_info.name, dev_info.arch, dev_info.total_memory_mb)
    lines[#lines + 1] = box_line(pad, sub, inner_w)
    lines[#lines + 1] = pad .. "\27[1;36m├" .. string.rep("─", inner_w) .. "┤\27[0m"

    -- Menu Items
    for i, item in ipairs(demos) do
        local is_sel = (i == selected)
        local pointer = is_sel and "\27[1;33m ▶ \27[0m" or "    "
        local num_badge = string.format("[%d]", item.id)
        local item_title = item.title
        local tag_str = " " .. item.tag .. " "

        local line_content
        if is_sel then
            line_content = string.format(
                "%s\27[1;37;42m %s \27[0m \27[1;37m%-32s\27[0m %s%s\27[0m",
                pointer, num_badge, item_title, item.badge_color, tag_str
            )
        else
            line_content = string.format(
                "%s\27[1;30m%s\27[0m  \27[0;37m%-32s\27[0m %s%s\27[0m",
                pointer, num_badge, item_title, item.badge_color, tag_str
            )
        end
        lines[#lines + 1] = box_line(pad, line_content, inner_w)
    end

    lines[#lines + 1] = pad .. "\27[1;36m├" .. string.rep("─", inner_w) .. "┤\27[0m"

    -- Detail Card for Selected Item
    local cur = demos[selected]
    local file_str = "  \27[1;33mScript: ./" .. cur.file .. "\27[0m"
    lines[#lines + 1] = box_line(pad, file_str, inner_w)

    -- Word wrap description to inner_w - 4
    local max_desc_w = inner_w - 4
    local words = {}
    for word in cur.desc:gmatch("%S+") do table.insert(words, word) end

    local cur_line = ""
    for _, word in ipairs(words) do
        if #cur_line + #word + 1 <= max_desc_w then
            cur_line = (cur_line == "") and word or (cur_line .. " " .. word)
        else
            lines[#lines + 1] = box_line(pad, "  \27[0;37m" .. cur_line .. "\27[0m", inner_w)
            cur_line = word
        end
    end
    if cur_line ~= "" then
        lines[#lines + 1] = box_line(pad, "  \27[0;37m" .. cur_line .. "\27[0m", inner_w)
    end

    lines[#lines + 1] = box_line(pad, "", inner_w)
    local tech_str = "  \27[2;36mTech: " .. cur.techniques .. "\27[0m"
    if visual_len(tech_str) > inner_w - 2 then
        while visual_len(tech_str) > inner_w - 5 do
            tech_str = tech_str:sub(1, #tech_str - 1)
        end
        tech_str = tech_str .. "\27[2;36m...\27[0m"
    end
    lines[#lines + 1] = box_line(pad, tech_str, inner_w)

    lines[#lines + 1] = pad .. "\27[1;36m└" .. string.rep("─", inner_w) .. "┘\27[0m"

    -- Footer Navigation Bar
    local nav = "\27[1;37;44m [↑/↓ or j/k] Navigate  •  [Enter] Run  •  [1-6] Quick Select  •  [q] Quit \27[0m"
    lines[#lines + 1] = pad .. nav

    io.write(table.concat(lines, "\n") .. "\n")
    io.flush()
end

-- -----------------------------------------------------------------------------
-- Launch Demo Execution
-- -----------------------------------------------------------------------------
local function launch_demo(demo)
    raw_mode_off()

    io.write("\27[2J\27[H")
    print(string.format("\27[1;32m=== Launching %s (%s) ===\27[0m", demo.title, demo.file))
    print("\27[2;37mPress Ctrl+C at any time during execution to return to the menu.\27[0m\n")

    -- Run the script
    local cmd = string.format("./%s", demo.file)
    os.execute(cmd)

    print("\n\27[1;33m[Program finished. Press Enter to return to menu...]\27[0m")
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
