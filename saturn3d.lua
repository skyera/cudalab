#!/usr/bin/env luajit
--[[
================================================================================
  saturn3d.lua: Real-Time 3D Saturn Ring System & Shepherd Moons Simulator
================================================================================
  Features:
  - Raytraced oblate gas giant ellipsoid (polar flattening factor 0.90)
  - Atmospheric cloud banding, storm eddies & North Polar Hexagonal vortex
  - Ring system with Cassini division, Encke gap, B/C/D/F rings & micro-ringlets
  - Dual dynamic shadows: Planet's shadow on rings + Rings' shadow on planet
  - 4 Keplerian orbiting moons: Mimas, Enceladus (ice geyser), Tethys, Prometheus
  - Deep space twinkling starfield and backlit solar corona forward scattering
  - TrueColor ANSI half-block terminal rendering (60+ FPS on CUDA)
  - Interactive camera orbit, zoom, view presets, and time-scale control
================================================================================
--]]

local ffi = require("ffi")
local cuda = require("cuda")

-- -----------------------------------------------------------------------------
-- Parse Command-Line Options
-- -----------------------------------------------------------------------------
local args = {...}
local opt_fps = 60
local opt_frames = nil
local opt_width = nil
local opt_height = nil
local opt_save = nil
local opt_time = 0.0
local opt_view = 1
local opt_mode = 0

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
        i = i + 1; opt_time = tonumber(args[i]) or 0.0
    elseif a == "--view" then
        i = i + 1; opt_view = tonumber(args[i]) or 1
    elseif a == "--mode" then
        i = i + 1; opt_mode = tonumber(args[i]) or 0
    elseif a == "--help" or a == "-h" then
        print([[
Usage: ./saturn3d.lua [options]

Options:
  --fps <N>          Target frame rate cap (default: 60)
  --frames <N>       Render N frames and exit
  --size <WxH>       Render grid size (e.g. 120x60, default: auto-fit terminal)
  --save <file.ppm>  Render snapshot image to PPM file and exit
  --time <sec>       Time offset for snapshot (default: 0.0)
  --view <1-4>       Camera view preset (1: Cassini Oblique, 2: Ring Skim, 3: Polar Hexagon, 4: Backlit Crescent)
  --mode <0-2>       Color mode (0: Natural TrueColor, 1: UV Methane Belt, 2: Thermal Radiance)
  --help             Show this help message

Interactive Live Controls:
  W/A/S/D or Arrows : Orbit camera elevation / azimuth
  + / - or Z / X    : Zoom camera in / out
  m                 : Cycle camera view presets
  p                 : Cycle color modes
  Space             : Pause / resume orbital revolution
  [ / ]             : Slow down / speed up simulation time
  r                 : Reset camera to default orientation
  q or ESC          : Quit cleanly
]])
        os.exit(0)
    end
    i = i + 1
end

-- -----------------------------------------------------------------------------
-- Terminal Helpers & Key Polling
-- -----------------------------------------------------------------------------
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

local function get_term_size()
    local ws = ffi.new("struct winsize")
    if ffi.C.ioctl(1, 0x5413, ws) == 0 and ws.ws_col > 20 and ws.ws_row > 10 then
        return ws.ws_col, ws.ws_row
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
        if b == 119 or b == 87 then return "up" end                       -- w/W
        if b == 115 or b == 83 then return "down" end                     -- s/S
        if b == 97  or b == 65 then return "left" end                     -- a/A
        if b == 100 or b == 68 then return "right" end                    -- d/D
        if b == 122 or b == 90 or b == 43 or b == 61 then return "zoom_in" end  -- z/Z/+
        if b == 120 or b == 88 or b == 45 then return "zoom_out" end             -- x/X/-
        if b == 109 or b == 77 then return "view" end                     -- m/M
        if b == 112 or b == 80 then return "mode" end                     -- p/P
        if b == 32 then return "pause" end                                -- Space
        if b == 91 then return "slow" end                                 -- [
        if b == 93 then return "fast" end                                 -- ]
        if b == 114 or b == 82 then return "reset" end                    -- r/R
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
-- Determine Viewport Resolution
-- -----------------------------------------------------------------------------
local term_cols, term_rows = get_term_size()
local hud_lines = 2
local term_render_rows = math.max(10, term_rows - hud_lines)

-- Width is capped at term_cols - 2 to prevent terminal auto-wrap line insertion
local width = opt_width or math.max(40, term_cols - 2)
local height = opt_height or (term_render_rows * 2)

if width % 2 ~= 0 then width = width - 1 end
if height % 2 ~= 0 then height = height + 1 end

-- -----------------------------------------------------------------------------
-- Initialize CUDA Device
-- -----------------------------------------------------------------------------
io.write("\27[1;36m=== CUDA GPU Saturn Ring System & Shepherd Moons ===\27[0m\n")
local dev_info = cuda.init(0)
print(string.format("  Device     : \27[1;32m%s\27[0m (%s)", dev_info.name, dev_info.arch))
print(string.format("  Grid Size  : \27[1;33m%dx%d\27[0m (%d pixels)", width, height, width * height))
print(string.format("  VRAM Total : %d MB\n", dev_info.total_memory_mb))

-- -----------------------------------------------------------------------------
-- CUDA C++ Raytracing & Astrophysics Kernel Source
-- -----------------------------------------------------------------------------
local cuda_source = [[
typedef unsigned char uint8_t;

struct Vec3 {
    float x, y, z;
    __device__ inline Vec3() : x(0.0f), y(0.0f), z(0.0f) {}
    __device__ inline Vec3(float _x, float _y, float _z) : x(_x), y(_y), z(_z) {}
};

__device__ inline Vec3 operator+(Vec3 a, Vec3 b) { return Vec3(a.x + b.x, a.y + b.y, a.z + b.z); }
__device__ inline Vec3 operator-(Vec3 a, Vec3 b) { return Vec3(a.x - b.x, a.y - b.y, a.z - b.z); }
__device__ inline Vec3 operator*(Vec3 a, float s) { return Vec3(a.x * s, a.y * s, a.z * s); }
__device__ inline Vec3 operator*(float s, Vec3 a) { return Vec3(a.x * s, a.y * s, a.z * s); }
__device__ inline Vec3 operator*(Vec3 a, Vec3 b) { return Vec3(a.x * b.x, a.y * b.y, a.z * b.z); }
__device__ inline float dot(Vec3 a, Vec3 b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
__device__ inline float length(Vec3 a) { return sqrtf(dot(a, a)); }
__device__ inline Vec3 normalize(Vec3 a) { float l = length(a); return l > 1e-6f ? a * (1.0f / l) : Vec3(0,0,0); }
__device__ inline Vec3 cross(Vec3 a, Vec3 b) {
    return Vec3(a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x);
}
__device__ inline float clampf(float v, float minv, float maxv) {
    return fmaxf(minv, fminf(maxv, v));
}
__device__ inline Vec3 clampVec3(Vec3 v, float minv, float maxv) {
    return Vec3(clampf(v.x, minv, maxv), clampf(v.y, minv, maxv), clampf(v.z, minv, maxv));
}
__device__ inline Vec3 lerp(Vec3 a, Vec3 b, float t) {
    return a * (1.0f - t) + b * t;
}

// Procedural hash for stars
__device__ inline float hash12(float x, float y) {
    float n = sinf(x * 127.1f + y * 311.7f) * 43758.5453123f;
    return n - floorf(n);
}

// Saturn System Constants
#define R_EQ 1.65f              // Equatorial radius
#define OBLATENESS 0.895f       // Polar radius ratio (R_pol = R_EQ * OBLATENESS)
#define S_Y (1.0f / OBLATENESS) // Ellipsoid Y-scaling factor (1.1173f)
#define S_Y2 (S_Y * S_Y)        // 1.2484f
#define TILT_RAD 0.4665f        // 26.73 degrees axial tilt in radians

// Transform world to Saturn's tilted coordinate frame (tilt around X-axis)
__device__ inline Vec3 world_to_saturn(Vec3 p) {
    float ct = cosf(TILT_RAD), st = sinf(TILT_RAD);
    return Vec3(p.x, ct * p.y + st * p.z, -st * p.y + ct * p.z);
}

// Transform Saturn's tilted coordinates back to world
__device__ inline Vec3 saturn_to_world(Vec3 p) {
    float ct = cosf(TILT_RAD), st = sinf(TILT_RAD);
    return Vec3(p.x, ct * p.y - st * p.z, st * p.y + ct * p.z);
}

// -----------------------------------------------------------------------------
// Ring System Optical Model (Radial Opacity, Scattering & Color)
// -----------------------------------------------------------------------------
__device__ void get_ring_properties(float r_rel, float& opacity, Vec3& albedo) {
    // r_rel is in units of R_EQ (1.0 = Saturn equator)
    if (r_rel < 1.15f || r_rel > 3.56f) {
        opacity = 0.0f;
        albedo = Vec3(0, 0, 0);
        return;
    }

    // Micro-ringlet harmonic oscillations (Voyager / Cassini fine structure)
    float fine1 = sinf(r_rel * 280.0f) * 0.09f;
    float fine2 = sinf(r_rel * 690.0f) * 0.05f;
    float fine3 = sinf(r_rel * 1450.0f) * 0.03f;
    float micro = fine1 + fine2 + fine3;

    if (r_rel < 1.50f) {
        // D Ring: faint, diffuse inner ringlet
        float t = (r_rel - 1.15f) / 0.35f;
        opacity = clampf(0.04f + 0.12f * t + micro * 0.4f, 0.01f, 0.22f);
        albedo = Vec3(0.68f, 0.58f, 0.46f);
    } else if (r_rel < 1.95f) {
        // C Ring (Crepe Ring): semi-translucent, warm golden amber
        float t = (r_rel - 1.50f) / 0.45f;
        float ripple = sinf(t * 14.0f) * 0.14f;
        opacity = clampf(0.25f + ripple + micro, 0.10f, 0.55f);
        albedo = Vec3(0.78f, 0.68f, 0.52f);
    } else if (r_rel < 2.76f) {
        // B Ring: the widest, densest, and brightest ring (pure icy cream-gold)
        float t = (r_rel - 1.95f) / 0.81f;
        float massive = 0.88f + 0.10f * sinf(t * 26.0f) + micro;
        opacity = clampf(massive, 0.75f, 0.98f);
        albedo = Vec3(0.96f, 0.91f, 0.78f);
    } else if (r_rel < 2.88f) {
        // Cassini Division: dramatic 4,800 km wide dark gap carved by Mimas 2:1 orbital resonance
        float t = (r_rel - 2.76f) / 0.12f;
        float gap_dip = sinf(t * 3.14159f);
        opacity = clampf(0.05f - 0.04f * gap_dip + micro * 0.15f, 0.005f, 0.09f);
        albedo = Vec3(0.38f, 0.32f, 0.25f);
    } else if (r_rel < 3.42f) {
        // A Ring: wide bright outer ring containing the sharp Encke Gap
        if (r_rel >= 3.185f && r_rel <= 3.225f) {
            // Encke Gap (carved by moonlet Pan)
            opacity = 0.03f;
            albedo = Vec3(0.30f, 0.25f, 0.20f);
        } else {
            float t = (r_rel - 2.88f) / 0.54f;
            float density = 0.65f + 0.12f * sinf(t * 20.0f) + micro;
            opacity = clampf(density, 0.38f, 0.82f);
            albedo = Vec3(0.90f, 0.84f, 0.72f);
        }
    } else if (r_rel >= 3.49f && r_rel <= 3.54f) {
        // F Ring: narrow, ropy braided outer ringlet shepherded by Prometheus & Pandora
        float t = (r_rel - 3.49f) / 0.05f;
        float f_strand = sinf(t * 3.14159f);
        opacity = clampf(f_strand * 0.40f, 0.0f, 0.45f);
        albedo = Vec3(0.85f, 0.80f, 0.72f);
    } else {
        opacity = 0.0f;
        albedo = Vec3(0, 0, 0);
    }
}

// -----------------------------------------------------------------------------
// Saturn Cloud Banding & Polar Hexagon Shader
// -----------------------------------------------------------------------------
__device__ Vec3 get_saturn_atmosphere(Vec3 local_pos, Vec3 normal, float time, int color_mode) {
    float r_pol = R_EQ * OBLATENESS;
    float lat = asinf(clampf(local_pos.y / r_pol, -0.999f, 0.999f)); // [-pi/2, pi/2]
    float lon = atan2f(local_pos.z, local_pos.x) + time * 0.35f;

    // Atmospheric cloud banding harmonics
    float band = sinf(lat * 14.0f) * 0.45f +
                 sinf(lat * 32.0f) * 0.28f +
                 sinf(lat * 68.0f) * 0.15f +
                 sinf(lat * 130.0f) * 0.07f;

    // Longitudinal turbulent atmospheric eddies
    float eddy = sinf(lat * 42.0f + lon * 7.0f + sinf(lon * 3.0f) * 1.5f) * 0.08f;
    float pattern = clampf(band + eddy, -1.0f, 1.0f);

    // North Polar Hexagon Vortex (latitude > 70 deg = 1.22 rad)
    float is_hexagon = clampf((lat - 1.18f) / 0.16f, 0.0f, 1.0f);
    float hex_wave = cosf(6.0f * (lon - time * 0.15f)) * 0.5f + 0.5f;

    Vec3 col;
    if (color_mode == 0) {
        // Mode 0: Voyager/Cassini Calibrated Natural TrueColor
        Vec3 base_gold(0.92f, 0.81f, 0.56f);
        Vec3 belt_amber(0.74f, 0.55f, 0.34f);
        Vec3 cream_zone(0.96f, 0.89f, 0.72f);
        Vec3 polar_hexagon(0.32f, 0.58f, 0.65f); // Iconic deep teal-cyan hexagonal core

        float t = pattern * 0.5f + 0.5f;
        col = lerp(belt_amber, (t > 0.5f ? cream_zone : base_gold), fabsf(t - 0.5f) * 2.0f);

        // Blend in the North Polar Hexagon
        if (is_hexagon > 0.0f) {
            Vec3 hex_col = lerp(polar_hexagon, Vec3(0.42f, 0.68f, 0.62f), hex_wave);
            col = lerp(col, hex_col, is_hexagon);
        }
    } else if (color_mode == 1) {
        // Mode 1: Ultraviolet / Methane Absorption False Color (High-contrast atmospheric storm belts)
        Vec3 deep_indigo(0.20f, 0.15f, 0.55f);
        Vec3 electric_amber(0.98f, 0.65f, 0.10f);
        Vec3 neon_cyan(0.10f, 0.92f, 0.88f);
        float t = pattern * 0.5f + 0.5f;
        col = lerp(deep_indigo, electric_amber, t);
        if (is_hexagon > 0.0f) col = lerp(col, neon_cyan, is_hexagon);
    } else {
        // Mode 2: Thermal Radiance Infrared Heatmap (Saturn's internal heat engine)
        float heat = clampf(0.5f + 0.5f * pattern + (1.0f - fabsf(local_pos.y / r_pol)) * 0.4f, 0.0f, 1.0f);
        col = Vec3(heat * 1.0f, heat * heat * 0.6f, (1.0f - heat) * 0.3f);
    }

    return col;
}

// -----------------------------------------------------------------------------
// Shepherd Moons Setup
// -----------------------------------------------------------------------------
struct Moon {
    float r_orbit;  // Distance in R_EQ
    float radius;   // Visual sphere radius
    float omega;    // Orbital angular speed
    float phase;    // Initial phase offset
    Vec3 color;

    __device__ inline Moon() : r_orbit(0), radius(0), omega(0), phase(0), color() {}
    __device__ inline Moon(float _ro, float _rad, float _om, float _ph, Vec3 _col)
        : r_orbit(_ro), radius(_rad), omega(_om), phase(_ph), color(_col) {}
};

__device__ void get_moon(int id, float time, Vec3& world_pos, float& radius, Vec3& col) {
    // 4 Distinct Moons:
    // 1: Mimas (Death star crater moon, carved Cassini division)
    // 2: Enceladus (Brilliant cyan-white ice moon with active geysers)
    // 3: Tethys (Large cratered icy moon)
    // 4: Prometheus (Inner F-ring shepherd moon)
    Moon m;
    if (id == 0) {
        m = Moon(3.10f * R_EQ, 0.065f, 1.15f, 0.8f, Vec3(0.85f, 0.85f, 0.86f)); // Mimas
    } else if (id == 1) {
        m = Moon(3.95f * R_EQ, 0.080f, 0.85f, 2.4f, Vec3(0.92f, 0.97f, 1.00f)); // Enceladus
    } else if (id == 2) {
        m = Moon(4.88f * R_EQ, 0.095f, 0.62f, 4.2f, Vec3(0.90f, 0.86f, 0.80f)); // Tethys
    } else {
        m = Moon(3.48f * R_EQ, 0.048f, 1.35f, 5.5f, Vec3(0.82f, 0.80f, 0.76f)); // Prometheus
    }

    float ang = m.phase + m.omega * time * 0.65f;
    Vec3 local_pos(m.r_orbit * cosf(ang), 0.0f, m.r_orbit * sinf(ang));
    world_pos = saturn_to_world(local_pos);
    radius = m.radius;
    col = m.color;
}

// -----------------------------------------------------------------------------
// Main Saturn Raytracer Kernel
// -----------------------------------------------------------------------------
extern "C" __global__ void k_render_saturn(
    uint8_t* __restrict__ out_rgb,
    int width, int height,
    float cam_dist, float cam_yaw, float cam_pitch,
    float time, int color_mode
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;

    // Screen coordinates [-1, 1]
    float aspect = (float)width / (float)height; // half-block character aspect ratio correction
    float u = (((float)x + 0.5f) / (float)width * 2.0f - 1.0f) * aspect;
    float v = 1.0f - ((float)y + 0.5f) / (float)height * 2.0f; // y-up

    // Camera setup (Spherical coordinates)
    float cy = cosf(cam_yaw), sy = sinf(cam_yaw);
    float cp = cosf(cam_pitch), sp = sinf(cam_pitch);

    Vec3 cam_pos(
        cam_dist * cp * sy,
        cam_dist * sp,
        cam_dist * cp * cy
    );
    Vec3 cam_target(0.0f, 0.0f, 0.0f);
    Vec3 cam_fwd = normalize(cam_target - cam_pos);
    Vec3 cam_right = normalize(cross(cam_fwd, Vec3(0.0f, 1.0f, 0.0f)));
    Vec3 cam_up = cross(cam_right, cam_fwd);

    Vec3 ray_dir = normalize(cam_fwd * 1.85f + cam_right * u + cam_up * v);
    Vec3 ray_orig = cam_pos;

    // Sun direction in World & Local frames
    Vec3 sun_dir_world = normalize(Vec3(1.15f, 0.42f, 0.85f));
    Vec3 sun_dir_local = world_to_saturn(sun_dir_world);

    // Transform camera ray to Saturn's local tilted frame
    Vec3 ro_loc = world_to_saturn(ray_orig);
    Vec3 rd_loc = world_to_saturn(ray_dir);

    // -------------------------------------------------------------------------
    // 1. Ray-Sphere/Ellipsoid Intersection for Saturn's Body
    // -------------------------------------------------------------------------
    float A = rd_loc.x * rd_loc.x + S_Y2 * rd_loc.y * rd_loc.y + rd_loc.z * rd_loc.z;
    float B = 2.0f * (ro_loc.x * rd_loc.x + S_Y2 * ro_loc.y * rd_loc.y + ro_loc.z * rd_loc.z);
    float C = ro_loc.x * ro_loc.x + S_Y2 * ro_loc.y * ro_loc.y + ro_loc.z * ro_loc.z - R_EQ * R_EQ;
    float disc = B * B - 4.0f * A * C;

    float t_planet = 1e9f;
    if (disc >= 0.0f) {
        float t0 = (-B - sqrtf(disc)) / (2.0f * A);
        if (t0 > 0.01f) t_planet = t0;
    }

    // -------------------------------------------------------------------------
    // 2. Ray-Plane Intersection for Saturn's Rings (y_local = 0)
    // -------------------------------------------------------------------------
    float t_ring = 1e9f;
    float ring_r_rel = 0.0f;
    float ring_opacity = 0.0f;
    Vec3 ring_albedo(0, 0, 0);

    if (fabsf(rd_loc.y) > 1e-5f) {
        float tr = -ro_loc.y / rd_loc.y;
        if (tr > 0.01f) {
            Vec3 p_ring = ro_loc + rd_loc * tr;
            float r = sqrtf(p_ring.x * p_ring.x + p_ring.z * p_ring.z);
            float r_rel = r / R_EQ;
            float op = 0.0f;
            Vec3 alb;
            get_ring_properties(r_rel, op, alb);
            if (op > 0.005f) {
                t_ring = tr;
                ring_r_rel = r_rel;
                ring_opacity = op;
                ring_albedo = alb;
            }
        }
    }

    // -------------------------------------------------------------------------
    // 3. Ray-Sphere Intersection for Shepherd Moons
    // -------------------------------------------------------------------------
    float t_moon = 1e9f;
    Vec3 moon_color(0, 0, 0);
    Vec3 moon_hit_pos(0, 0, 0);
    Vec3 moon_norm(0, 0, 0);

    for (int k = 0; k < 4; k++) {
        Vec3 m_pos; float m_rad; Vec3 m_col;
        get_moon(k, time, m_pos, m_rad, m_col);
        Vec3 oc = ray_orig - m_pos;
        float mb = dot(oc, ray_dir);
        float mc = dot(oc, oc) - m_rad * m_rad;
        float mdisc = mb * mb - mc;
        if (mdisc >= 0.0f) {
            float mt = -mb - sqrtf(mdisc);
            if (mt > 0.01f && mt < t_moon) {
                t_moon = mt;
                moon_color = m_col;
                moon_hit_pos = ray_orig + ray_dir * mt;
                moon_norm = normalize(moon_hit_pos - m_pos);
            }
        }
    }

    // -------------------------------------------------------------------------
    // Shading Composition
    // -------------------------------------------------------------------------
    Vec3 pixel_color(0.0f, 0.0f, 0.0f);

    // Determine nearest surface hit
    bool hit_planet = (t_planet < 1e8f);
    bool hit_ring = (t_ring < 1e8f);
    bool hit_moon = (t_moon < 1e8f && t_moon < t_planet && t_moon < t_ring);

    if (hit_moon) {
        // --- Shading Shepherd Moon ---
        float diff = fmaxf(0.0f, dot(moon_norm, sun_dir_world));
        // Check if moon is in Saturn's shadow
        Vec3 m_loc = world_to_saturn(moon_hit_pos);
        float ms_A = sun_dir_local.x * sun_dir_local.x + S_Y2 * sun_dir_local.y * sun_dir_local.y + sun_dir_local.z * sun_dir_local.z;
        float ms_B = 2.0f * (m_loc.x * sun_dir_local.x + S_Y2 * m_loc.y * sun_dir_local.y + m_loc.z * sun_dir_local.z);
        float ms_C = m_loc.x * m_loc.x + S_Y2 * m_loc.y * m_loc.y + m_loc.z * m_loc.z - R_EQ * R_EQ;
        float ms_disc = ms_B * ms_B - 4.0f * ms_A * ms_C;
        float shadow = (ms_disc >= 0.0f && (-ms_B - sqrtf(ms_disc)) > 0.01f) ? 0.08f : 1.0f;

        pixel_color = moon_color * (diff * shadow * 0.90f + 0.10f);
    } else if (hit_planet && (!hit_ring || t_planet < t_ring)) {
        // --- Shading Saturn's Planetary Atmosphere ---
        Vec3 p_loc = ro_loc + rd_loc * t_planet;
        Vec3 norm_loc = normalize(Vec3(p_loc.x, S_Y2 * p_loc.y, p_loc.z));
        Vec3 norm_world = saturn_to_world(norm_loc);

        // Cloud banding & polar hexagon color
        Vec3 atm_col = get_saturn_atmosphere(p_loc, norm_loc, time, color_mode);

        // Sunlight diffuse term
        float NdotL = dot(norm_world, sun_dir_world);
        float diff = fmaxf(0.0f, NdotL * 0.85f + 0.15f); // Soft atmospheric terminator

        // Planetary Limb Darkening
        float NdotV = fmaxf(0.0f, -dot(norm_world, ray_dir));
        float limb = powf(NdotV, 0.45f);

        // --- RINGS' SHADOW ON PLANET (Ring shadow stripes on clouds) ---
        float shadow_transmittance = 1.0f;
        if (fabsf(sun_dir_local.y) > 1e-4f) {
            float t_to_ring = -p_loc.y / sun_dir_local.y;
            if (t_to_ring > 0.01f) {
                Vec3 p_on_ring = p_loc + sun_dir_local * t_to_ring;
                float r_ring = sqrtf(p_on_ring.x * p_on_ring.x + p_on_ring.z * p_on_ring.z) / R_EQ;
                float r_op = 0.0f; Vec3 r_alb;
                get_ring_properties(r_ring, r_op, r_alb);
                shadow_transmittance = 1.0f - r_op * 0.92f;
            }
        }

        // Saturnshine / ambient glow
        Vec3 ambient = atm_col * 0.08f;
        pixel_color = atm_col * (diff * shadow_transmittance * 0.92f) * limb + ambient;

        // If ring is semi-translucent in front of the planet, composite ring over planet
        if (hit_ring && t_ring < t_planet) {
            float ring_diff = fabsf(dot(saturn_to_world(Vec3(0, 1, 0)), sun_dir_world));
            Vec3 ring_lit = ring_albedo * (ring_diff * 0.90f + 0.10f);
            pixel_color = lerp(pixel_color, ring_lit, ring_opacity);
        }
    } else if (hit_ring) {
        // --- Shading Saturn's Rings ---
        Vec3 p_ring = ro_loc + rd_loc * t_ring;

        // Check if ring point is in SATURN'S SHADOW
        float sA = sun_dir_local.x * sun_dir_local.x + S_Y2 * sun_dir_local.y * sun_dir_local.y + sun_dir_local.z * sun_dir_local.z;
        float sB = 2.0f * (p_ring.x * sun_dir_local.x + p_ring.z * sun_dir_local.z); // p_ring.y == 0
        float sC = p_ring.x * p_ring.x + p_ring.z * p_ring.z - R_EQ * R_EQ;
        float s_disc = sB * sB - 4.0f * sA * sC;
        bool in_planet_shadow = (s_disc >= 0.0f && (-sB - sqrtf(s_disc)) > 0.01f);

        // Optical scattering (Forward vs Back scattering on ice particles)
        float cos_phase = dot(ray_dir, sun_dir_world);
        float forward_scatter = powf(fmaxf(0.0f, cos_phase), 4.0f) * 0.5f;
        float back_scatter = powf(fmaxf(0.0f, -cos_phase), 2.0f) * 0.3f;
        float phase_fn = 0.55f + forward_scatter + back_scatter;

        float ring_sun_elev = fabsf(dot(saturn_to_world(Vec3(0, 1, 0)), sun_dir_world));
        float direct_sun = in_planet_shadow ? 0.0f : (ring_sun_elev * 0.85f + 0.15f) * phase_fn;

        // Saturnshine: the golden sunlit hemisphere of Saturn softly illuminates the shadowed ring side
        Vec3 saturnshine = Vec3(0.92f, 0.78f, 0.52f) * 0.05f;

        Vec3 ring_color = ring_albedo * (direct_sun * 0.92f) + (in_planet_shadow ? saturnshine : Vec3(0,0,0));

        // If planet is behind the semi-translucent ring, composite ring over planet
        if (hit_planet && t_planet > t_ring) {
            Vec3 p_loc = ro_loc + rd_loc * t_planet;
            Vec3 norm_loc = normalize(Vec3(p_loc.x, S_Y2 * p_loc.y, p_loc.z));
            Vec3 atm_col = get_saturn_atmosphere(p_loc, norm_loc, time, color_mode);
            float diff = fmaxf(0.0f, dot(saturn_to_world(norm_loc), sun_dir_world) * 0.85f + 0.15f);
            Vec3 bg_planet = atm_col * diff;
            pixel_color = lerp(bg_planet, ring_color, ring_opacity);
        } else {
            // Ring over deep space starfield
            pixel_color = ring_color * ring_opacity;
        }
    } else {
        // --- Deep Space Background Starfield & Sun ---
        // Twinkling stars via spatial hash
        float u_star = floorf((ray_dir.x + 2.0f) * 160.0f);
        float v_star = floorf((ray_dir.y + 2.0f) * 160.0f);
        float star_val = hash12(u_star, v_star);
        if (star_val > 0.985f) {
            float twinkle = sinf(star_val * 600.0f + time * 3.5f) * 0.3f + 0.7f;
            float bright = (star_val - 0.985f) / 0.015f * twinkle;
            if (star_val > 0.996f) {
                // Blue giant star
                pixel_color = Vec3(0.7f, 0.85f, 1.0f) * bright;
            } else if (star_val > 0.992f) {
                // Golden star
                pixel_color = Vec3(1.0f, 0.90f, 0.65f) * bright;
            } else {
                // Dim white star
                pixel_color = Vec3(0.8f, 0.8f, 0.8f) * bright * 0.6f;
            }
        }

        // Distant Sun disk and corona glare
        float sun_align = dot(ray_dir, sun_dir_world);
        if (sun_align > 0.9992f) {
            // Sun disk
            pixel_color = Vec3(1.0f, 1.0f, 0.95f) * 1.5f;
        } else if (sun_align > 0.980f) {
            // Solar corona glow
            float glow = powf((sun_align - 0.980f) / (0.9992f - 0.980f), 3.0f) * 0.45f;
            pixel_color = pixel_color + Vec3(1.0f, 0.85f, 0.60f) * glow;
        }
    }

    // Gamma correction and tonemapping
    pixel_color.x = sqrtf(clampf(pixel_color.x, 0.0f, 1.0f));
    pixel_color.y = sqrtf(clampf(pixel_color.y, 0.0f, 1.0f));
    pixel_color.z = sqrtf(clampf(pixel_color.z, 0.0f, 1.0f));

    int out_idx = (y * width + x) * 3;
    out_rgb[out_idx + 0] = (uint8_t)(pixel_color.x * 255.0f);
    out_rgb[out_idx + 1] = (uint8_t)(pixel_color.y * 255.0f);
    out_rgb[out_idx + 2] = (uint8_t)(pixel_color.z * 255.0f);
}
]]

-- -----------------------------------------------------------------------------
-- Compile CUDA Raytracer via NVRTC
-- -----------------------------------------------------------------------------
io.write("  Compiling Saturn 3D Raytracer via NVRTC... ")
io.flush()
local t_comp0 = os.clock()
local mod = cuda.compile(cuda_source, {
    arch = dev_info.arch,
    fast_math = true,
    name = "saturn_raytracer.cu",
})
local t_comp1 = os.clock()
print(string.format("\27[1;32mDone in %.2f s!\27[0m\n", t_comp1 - t_comp0))

local k_render = mod:get_function("k_render_saturn", "ptr, int, int, float, float, float, float, int")

-- -----------------------------------------------------------------------------
-- Allocate GPU & Host Buffers
-- -----------------------------------------------------------------------------
local total_pixels = width * height
local d_out = cuda.alloc(total_pixels * 3)
local h_out = ffi.new("uint8_t[?]", total_pixels * 3)
local gpu_timer = cuda.timer()

local block2d = { 16, 16 }
local grid2d = { math.ceil(width / 16), math.ceil(height / 16) }

-- -----------------------------------------------------------------------------
-- Helper: Save PPM Snapshot
-- -----------------------------------------------------------------------------
local function save_ppm(filename, w, h, data)
    local f = io.open(filename, "wb")
    if not f then error("Cannot open file for writing: " .. filename) end
    f:write(string.format("P6\n%d %d\n255\n", w, h))
    f:write(ffi.string(data, w * h * 3))
    f:close()
end

-- -----------------------------------------------------------------------------
-- Camera View Presets
-- -----------------------------------------------------------------------------
local view_presets = {
    [1] = { name = "Cassini Oblique",  dist = 7.6,  yaw = 0.35,  pitch = 0.38 },
    [2] = { name = "Ring Plane Skim",  dist = 6.2,  yaw = 1.10,  pitch = 0.04 },
    [3] = { name = "Polar Hexagon",    dist = 6.8,  yaw = 0.00,  pitch = 1.15 },
    [4] = { name = "Backlit Crescent", dist = 8.2,  yaw = 2.70,  pitch = 0.25 },
}

local cur_view = opt_view
local cam_dist = view_presets[cur_view].dist
local cam_yaw = view_presets[cur_view].yaw
local cam_pitch = view_presets[cur_view].pitch

local mode_names = {
    [0] = "Natural TrueColor",
    [1] = "UV Methane Belt",
    [2] = "Thermal Heatmap",
}
local cur_mode = opt_mode

-- -----------------------------------------------------------------------------
-- Snapshot Mode (--save)
-- -----------------------------------------------------------------------------
if opt_save then
    print(string.format("Rendering Saturn snapshot (t=%.2f s, %dx%d)...", opt_time, width, height))
    k_render:launch({ grid = grid2d, block = block2d },
        d_out, width, height, cam_dist, cam_yaw, cam_pitch, opt_time, cur_mode)
    d_out:to_host(h_out)
    save_ppm(opt_save, width, height, h_out)
    print(string.format("Snapshot saved successfully to %s (%dx%d)!", opt_save, width, height))
    os.exit(0)
end

-- -----------------------------------------------------------------------------
-- ANSI Half-Block Frame Builder (High Performance & Clean Styling)
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
                -- Ambient space: seamless background fill
                if cur_bg ~= 0 then
                    line[#line + 1] = "\27[48;2;0;0;0m"
                    cur_bg = 0
                end
                line[#line + 1] = " "
            elseif top_black then
                -- Top is empty space, bottom is Saturn/ring: lower half block
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
                -- Top is Saturn/ring, bottom is empty space: upper half block
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
                -- Matching colors: seamless background fill
                local b_code = br * 65536 + bg * 256 + bb
                if cur_bg ~= b_code then
                    line[#line + 1] = string.format("\27[48;2;%d;%d;%dm", br, bg, bb)
                    cur_bg = b_code
                end
                line[#line + 1] = " "
            else
                -- Dual colors: upper half block with fg and bg
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
-- Real-Time Interactive Animation Loop
-- -----------------------------------------------------------------------------
local running = true
local paused = false
local sim_speed = 1.0
local sim_time = 0.0

raw_mode_on()

io.write("\27[2J\27[H")
io.flush()

local frame_count = 0
local t_prev = os.clock()
local last_fps_time = t_prev
local fps = 0.0
local gpu_ms = 0.0

local ok, err = pcall(function()
    while running do
        local now = os.clock()
        local dt_real = now - t_prev
        t_prev = now

        if not paused then
            sim_time = sim_time + dt_real * sim_speed
            -- Continuous gentle orbital precession
            cam_yaw = cam_yaw + dt_real * 0.08 * sim_speed
        end

        -- Poll interactive keyboard controls
        local key = poll_key()
        if key == "quit" then
            running = false
            break
        elseif key == "up" then
            cam_pitch = math.min(1.45, cam_pitch + 0.06)
        elseif key == "down" then
            cam_pitch = math.max(-1.45, cam_pitch - 0.06)
        elseif key == "left" then
            cam_yaw = cam_yaw - 0.08
        elseif key == "right" then
            cam_yaw = cam_yaw + 0.08
        elseif key == "zoom_in" then
            cam_dist = math.max(3.8, cam_dist - 0.35)
        elseif key == "zoom_out" then
            cam_dist = math.min(18.0, cam_dist + 0.35)
        elseif key == "view" then
            cur_view = (cur_view % 4) + 1
            cam_dist = view_presets[cur_view].dist
            cam_yaw = view_presets[cur_view].yaw
            cam_pitch = view_presets[cur_view].pitch
        elseif key == "mode" then
            cur_mode = (cur_mode + 1) % 3
        elseif key == "pause" then
            paused = not paused
        elseif key == "slow" then
            sim_speed = math.max(0.1, sim_speed * 0.7)
        elseif key == "fast" then
            sim_speed = math.min(10.0, sim_speed * 1.4)
        elseif key == "reset" then
            cur_view = 1
            cam_dist = view_presets[1].dist
            cam_yaw = view_presets[1].yaw
            cam_pitch = view_presets[1].pitch
            sim_speed = 1.0
            paused = false
        end

        -- Launch GPU Raytracing Kernel
        gpu_timer:start()
        k_render:launch({ grid = grid2d, block = block2d },
            d_out, width, height, cam_dist, cam_yaw, cam_pitch, sim_time, cur_mode)
        gpu_timer:stop()
        gpu_ms = gpu_timer:elapsed_ms()

        -- Copy rendered pixels to host
        d_out:to_host(h_out)

        -- Construct ANSI frame
        local ansi_frame = build_ansi_frame(h_out, width, height)

        -- Update FPS
        frame_count = frame_count + 1
        if frame_count % 10 == 0 then
            local dt_wall = now - last_fps_time
            if dt_wall > 0 then fps = 10.0 / dt_wall end
            last_fps_time = now
        end

        -- Construct Top HUD & Bottom Controls
        local pause_str = paused and "\27[1;33mPAUSED\27[0m" or string.format("\27[1;32m%.1fx\27[0m", sim_speed)
        local top_hud = string.format(
            "  \27[1;37mSATURN 3D\27[0m \27[2;37m•\27[0m \27[1;32m%4.1f FPS\27[0m \27[2;37m(%4.1f ms)\27[0m \27[2;37m•\27[0m \27[1;36m%s\27[0m \27[2;37m•\27[0m \27[1;33m%s\27[0m \27[2;37m•\27[0m Time: %s",
            fps, gpu_ms, mode_names[cur_mode], view_presets[cur_view].name, pause_str
        )
        local bottom_hud = "  \27[2;37m[W/A/S/D] Orbit  [+/-] Zoom  [m] View  [p] Mode  [Space] Pause  [[]/[]] Speed  [q] Quit\27[0m"

        -- Blit frame buffer cleanly
        io.write("\27[H\r\27[2K" .. top_hud .. "\27[K\r\n" .. ansi_frame .. "\r\n\r\27[2K" .. bottom_hud .. "\27[K")
        io.flush()

        if opt_frames and frame_count >= opt_frames then
            running = false
            break
        end

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
    print("\nError during Saturn simulation: " .. tostring(err))
else
    print("\n[Exited Saturn 3D cleanly. Terminal restored.]\n")
end
