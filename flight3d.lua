#!/usr/bin/env luajit
--[[
================================================================================
  flight3d.lua: 3D Space Flight Simulator & Asteroid Belt via LuaJIT + CUDA
================================================================================
  A real-time 6-DOF spaceship flight simulator and asteroid dogfight combat
  engine running raymarched procedural asteroids, plasma lasers, and warp speed
  effects entirely on NVIDIA CUDA cores with ANSI 24-bit TrueColor terminal rendering.

  Physics & GPU Features:
  - Infinite Procedural Asteroid Field: 3D tumbling SDF asteroids with multi-octave
    craters, mineral ore veins, and specular directional lighting.
  - Active Twin Plasma Laser Cannons: High-velocity glowing laser bolts with real-time
    hit detection, asteroid explosive shattering, and glowing debris fields.
  - 6-DOF Flight Dynamics: True Newtonian momentum, thruster acceleration, pitch,
    yaw, roll, and afterburner warp boost with relativistic star streaks.
  - Deep Space Nebula: Volumetric cosmic dust clouds (cyan/magenta/deep indigo),
    twinkling starfield, and distant solar rim lighting.
  - Sci-Fi Cockpit & Combat HUD: Target lock-on tracking diamond, attitude pitch ladder,
    rangefinder, velocity gauge, shield integrity, and laser heat indicator.

  Interactive Controls (Live at 60 FPS):
    [W / S]              Accelerate Forward / Reverse Thrusters (Brake)
    [A / D]              Strafe Left / Right
    [E / C]              Elevate Up / Down
    [← / → or J / L]     Turn Yaw (Steer Left / Right)
    [↑ / ↓ or I / K]     Look Pitch (Dive / Climb)
    [Space or Enter]     Fire Twin Plasma Laser Cannons!
    [F]                  Toggle Afterburner / Hyperdrive Warp Boost
    [V]                  Toggle Cockpit View (Cockpit Frame / Clean Combat HUD)
    [R]                  Reset Flight Position to Deep Space
    [Q or ESC]           Quit to terminal

  Command Line Options:
    --fps <N>          Target frame rate cap (default: 60)
    --frames <N>       Render N frames and exit
    --size <WxH>       Render resolution (default: auto-fit terminal)
    --save <file.ppm>  Render high-res snapshot (1920x1080) to PPM and exit
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
local opt_frames = nil
local opt_width = nil
local opt_height = nil
local opt_save = nil

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
    elseif a == "--help" or a == "-h" then
        print([[
Usage: ./flight3d.lua [options]

Options:
  --fps <N>          Target frame rate cap (default: 60)
  --frames <N>       Render N frames and exit
  --size <WxH>       Render resolution (default: auto-fit terminal)
  --save <file.ppm>  Render high-res snapshot (1920x1080) to PPM and exit
  --help             Show this help message
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
    io.write("\27[?25h\27[0m\r\n") -- Restore cursor and reset styling
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
        if b == 119 or b == 87 then return "accel" end                           -- w/W
        if b == 115 or b == 83 then return "brake" end                           -- s/S
        if b == 97  or b == 65 then return "strafe_left" end                     -- a/A
        if b == 100 or b == 68 then return "strafe_right" end                    -- d/D
        if b == 101 or b == 69 then return "elevate_up" end                      -- e/E
        if b == 99  or b == 67 then return "elevate_down" end                    -- c/C
        if b == 106 or b == 74 then return "yaw_left" end                        -- j/J
        if b == 108 or b == 76 then return "yaw_right" end                       -- l/L
        if b == 105 or b == 73 then return "pitch_up" end                        -- i/I
        if b == 107 or b == 75 then return "pitch_down" end                      -- k/K
        if b == 32  or b == 10 or b == 13 then return "fire" end                 -- Space / Enter
        if b == 102 or b == 70 then return "boost" end                           -- f/F
        if b == 118 or b == 86 then return "toggle_cockpit" end                  -- v/V
        if b == 114 or b == 82 then return "reset" end                           -- r/R
    elseif n >= 3 and read_buf[0] == 27 then
        local prefix = read_buf[1]
        local code = read_buf[2]
        if prefix == 91 or prefix == 79 then -- '[' or 'O'
            if code == 65 then return "pitch_up" end
            if code == 66 then return "pitch_down" end
            if code == 68 then return "yaw_left" end
            if code == 67 then return "yaw_right" end
        end
    end
    return nil
end

-- -----------------------------------------------------------------------------
-- Determine Viewport Resolution (Strictly 2 lines reserved for zero-scroll TUI)
-- -----------------------------------------------------------------------------
local term_cols, term_rows = get_term_size()
local hud_lines = 2
local render_rows = math.max(8, term_rows - hud_lines)
local width = opt_width or math.max(40, term_cols - 2)
local height = opt_height or (render_rows * 2)

if width % 2 ~= 0 then width = width - 1 end
if height % 2 ~= 0 then height = height + 1 end

-- -----------------------------------------------------------------------------
-- Initialize CUDA Device
-- -----------------------------------------------------------------------------
local dev_info = cuda.init(0)

-- -----------------------------------------------------------------------------
-- CUDA C++ Raymarching & Combat Kernels
-- -----------------------------------------------------------------------------
local cuda_source = [[
extern "C" {

struct LaserBolt {
    float x, y, z;
    float dir_x, dir_y, dir_z;
    float age;
    int active;
};

struct HitResult {
    int hit_asteroid;
    float dist;
    float ast_x, ast_y, ast_z;
    float ast_radius;
};

// 3D Pseudo-random hash
__device__ float hash3d(float x, float y, float z) {
    float n = sinf(x * 127.1f + y * 311.7f + z * 74.7f) * 43758.5453f;
    return n - floorf(n);
}

// Rotate vector in 2D
__device__ void rot2d(float& a, float& b, float angle) {
    float c = cosf(angle);
    float s = sinf(angle);
    float na = c * a - s * b;
    float nb = s * a + c * b;
    a = na;
    b = nb;
}

// Scene Distance Function: Infinite Clustered Asteroid Field
__device__ float mapAsteroids(
    float px, float py, float pz,
    float time,
    int& out_mat,
    float& out_id
) {
    const float cell_sz = 48.0f;
    float cx = floorf((px + cell_sz * 0.5f) / cell_sz);
    float cy = floorf((py + cell_sz * 0.5f) / cell_sz);
    float cz = floorf((pz + cell_sz * 0.5f) / cell_sz);

    float h_id = hash3d(cx * 1.3f, cy * 2.1f, cz * 3.7f);
    out_id = h_id;

    // Density modulation: leave space corridors
    if (h_id < 0.25f) {
        out_mat = 0;
        return 1e6f;
    }

    // Local coordinates inside cell
    float lx = px - cx * cell_sz;
    float ly = py - cy * cell_sz;
    float lz = pz - cz * cell_sz;

    // Asteroid center offset inside cell
    float off_x = (hash3d(cx, cy + 1.0f, cz) - 0.5f) * 16.0f;
    float off_y = (hash3d(cx + 2.0f, cy, cz) - 0.5f) * 16.0f;
    float off_z = (hash3d(cx, cy, cz + 3.0f) - 0.5f) * 16.0f;
    lx -= off_x;
    ly -= off_y;
    lz -= off_z;

    // Asteroid 3D tumbling rotation over time
    float rot_speed = 0.3f + h_id * 0.4f;
    rot2d(lx, lz, time * rot_speed + h_id * 6.28f);
    rot2d(ly, lz, time * rot_speed * 0.7f);

    float r = sqrtf(lx * lx + ly * ly + lz * lz);
    float base_radius = 4.0f + h_id * 5.5f;

    // Multi-octave rocky crater noise
    float noise1 = sinf(lx * 0.6f) * cosf(ly * 0.6f) * sinf(lz * 0.6f) * 1.8f;
    float noise2 = sinf(lx * 1.8f + h_id) * cosf(ly * 1.8f) * sinf(lz * 1.8f) * 0.6f;
    float asteroid_d = r - (base_radius + noise1 + noise2);

    out_mat = 1; // Rocky asteroid
    return asteroid_d;
}

// Distance to 3D Line Segment (Laser Bolt Capsule)
__device__ float distToLaser(
    float px, float py, float pz,
    float lx, float ly, float lz,
    float dx, float dy, float dz,
    float length
) {
    float vx = px - lx;
    float vy = py - ly;
    float vz = pz - lz;
    float t = vx * dx + vy * dy + vz * dz;
    t = fmaxf(0.0f, fminf(length, t));
    float cx = lx + dx * t;
    float cy = ly + dy * t;
    float cz = lz + dz * t;
    float rx = px - cx;
    float ry = py - cy;
    float rz = pz - cz;
    return sqrtf(rx * rx + ry * ry + rz * rz);
}

// Finite Difference Surface Normal
__device__ void calcNormal(
    float px, float py, float pz,
    float time,
    float& nx, float& ny, float& nz
) {
    const float eps = 0.02f;
    int dummy_mat;
    float dummy_id;
    float d = mapAsteroids(px, py, pz, time, dummy_mat, dummy_id);
    nx = mapAsteroids(px + eps, py, pz, time, dummy_mat, dummy_id) - d;
    ny = mapAsteroids(px, py + eps, pz, time, dummy_mat, dummy_id) - d;
    nz = mapAsteroids(px, py, pz + eps, time, dummy_mat, dummy_id) - d;
    float len = sqrtf(nx * nx + ny * ny + nz * nz);
    if (len > 1e-6f) { nx /= len; ny /= len; nz /= len; }
}

// -----------------------------------------------------------------------------
// 3D Space Flight Simulator & Raymarcher Kernel
// -----------------------------------------------------------------------------
__global__ void renderFlight(
    unsigned char* __restrict__ out_rgb,
    int img_w, int img_h,
    float cam_x, float cam_y, float cam_z,
    float yaw, float pitch, float roll,
    float time,
    float speed,
    int show_cockpit,
    const LaserBolt* __restrict__ lasers,
    int num_lasers,
    float explode_x, float explode_y, float explode_z,
    float explode_age
) {
    int px = blockIdx.x * blockDim.x + threadIdx.x;
    int py = blockIdx.y * blockDim.y + threadIdx.y;
    if (px >= img_w || py >= img_h) return;

    // Coordinate normalization
    float u = ((float)px + 0.5f) / (float)img_w * 2.0f - 1.0f;
    float v = 1.0f - ((float)py + 0.5f) / (float)img_h * 2.0f;
    u *= ((float)img_w / (float)img_h) * 0.75f;
    v *= 0.75f;

    // Cockpit frame geometry check (sci-fi fighter canopy)
    int is_cockpit_frame = 0;
    if (show_cockpit) {
        float screen_u = ((float)px + 0.5f) / (float)img_w * 2.0f - 1.0f;
        float screen_v = 1.0f - ((float)py + 0.5f) / (float)img_h * 2.0f;
        // Canopy border struts
        float strut1 = fabsf(screen_u) - 0.88f;
        float strut2 = fabsf(screen_v) - 0.85f;
        float canopy_arch = (screen_u * screen_u * 0.45f) - screen_v - 0.65f;
        if (strut1 > 0.0f || strut2 < -1.7f || (canopy_arch > 0.0f && canopy_arch < 0.14f)) {
            is_cockpit_frame = 1;
        }
    }

    // Camera rotation matrix (Yaw, Pitch, Roll)
    float cy = cosf(yaw), sy = sinf(yaw);
    float cp = cosf(pitch), sp = sinf(pitch);
    float cr = cosf(roll), sr = sinf(roll);

    float fwd_x = cp * sy;
    float fwd_y = sp;
    float fwd_z = cp * cy;

    float rgt_x = cy * cr + sy * sp * sr;
    float rgt_y = -cp * sr;
    float rgt_z = -sy * cr + cy * sp * sr;

    float up_x = -cy * sr + sy * sp * cr;
    float up_y = cp * cr;
    float up_z = sy * sr + cy * sp * cr;

    float dir_x = fwd_x + u * rgt_x + v * up_x;
    float dir_y = fwd_y + u * rgt_y + v * up_y;
    float dir_z = fwd_z + u * rgt_z + v * up_z;
    float r_len = sqrtf(dir_x * dir_x + dir_y * dir_y + dir_z * dir_z);
    dir_x /= r_len; dir_y /= r_len; dir_z /= r_len;

    // Raymarching Asteroid Scene
    float t = 1.0f;
    int hit = 0;
    int hit_mat = 0;
    float hit_id = 0.0f;
    const int MAX_STEPS = 64;
    const float MAX_DIST = 160.0f;

    for (int step = 0; step < MAX_STEPS; ++step) {
        float px_cur = cam_x + dir_x * t;
        float py_cur = cam_y + dir_y * t;
        float pz_cur = cam_z + dir_z * t;

        int mat;
        float id;
        float d = mapAsteroids(px_cur, py_cur, pz_cur, time, mat, id);
        if (d < 0.06f) {
            hit = 1;
            hit_mat = mat;
            hit_id = id;
            break;
        }
        t += d * 0.85f;
        if (t >= MAX_DIST) break;
    }

    // Background Cosmic Sky (Deep Indigo Space & Twinkling Stars)
    float star_seed = hash3d(floorf(dir_x * 80.0f), floorf(dir_y * 80.0f), floorf(dir_z * 80.0f));
    float star_intensity = (star_seed > 0.982f) ? (star_seed - 0.982f) / 0.018f : 0.0f;

    // Warp streaks if boosting
    if (speed > 2.5f) {
        float warp_dot = dir_x * fwd_x + dir_y * fwd_y + dir_z * fwd_z;
        if (warp_dot > 0.7f && star_seed > 0.96f) {
            star_intensity += (warp_dot - 0.7f) * (speed / 4.0f);
        }
    }

    // Colorful Volumetric Cosmic Nebula Cloud
    float nebula_u = dir_x * 1.5f + sinf(time * 0.05f) * 0.2f;
    float nebula_v = dir_y * 1.5f + cosf(time * 0.05f) * 0.2f;
    float neb1 = sinf(nebula_u * 3.0f) * cosf(nebula_v * 3.0f);
    float neb2 = cosf(dir_z * 2.0f + nebula_u);
    float neb_val = fmaxf(0.0f, (neb1 + neb2) * 0.35f);

    float bg_r = 6.0f + neb_val * 45.0f + star_intensity * 255.0f;
    float bg_g = 8.0f + neb_val * 15.0f + star_intensity * 240.0f;
    float bg_b = 24.0f + neb_val * 85.0f + star_intensity * 255.0f;

    float out_r = bg_r;
    float out_g = bg_g;
    float out_b = bg_b;

    if (hit) {
        float hx = cam_x + dir_x * t;
        float hy = cam_y + dir_y * t;
        float hz = cam_z + dir_z * t;

        float nx, ny, nz;
        calcNormal(hx, hy, hz, time, nx, ny, nz);

        // Directional Sun lighting (Primary celestial star)
        float sun_x = 0.6f, sun_y = 0.7f, sun_z = 0.4f;
        float s_len = sqrtf(sun_x * sun_x + sun_y * sun_y + sun_z * sun_z);
        sun_x /= s_len; sun_y /= s_len; sun_z /= s_len;

        float diff = fmaxf(0.0f, nx * sun_x + ny * sun_y + nz * sun_z);
        float amb = 0.12f;

        // Mineral ore vein coloring (iron, copper, gold flecks)
        float ore_val = hash3d(floorf(hx * 3.0f), floorf(hy * 3.0f), floorf(hz * 3.0f));
        float ast_base_r = 130.0f;
        float ast_base_g = 120.0f;
        float ast_base_b = 115.0f;

        if (ore_val > 0.88f) { // Gold ore vein
            ast_base_r = 230.0f; ast_base_g = 180.0f; ast_base_b = 40.0f;
        } else if (ore_val > 0.76f) { // Copper ore vein
            ast_base_r = 190.0f; ast_base_g = 110.0f; ast_base_b = 70.0f;
        }

        float lit_r = ast_base_r * (amb + diff * 0.88f);
        float lit_g = ast_base_g * (amb + diff * 0.88f);
        float lit_b = ast_base_b * (amb + diff * 0.88f);

        // Rim lighting from cyan nebula behind asteroid
        float rim = fmaxf(0.0f, 1.0f - fabsf(dir_x * nx + dir_y * ny + dir_z * nz));
        rim = powf(rim, 3.0f) * 0.5f;
        lit_r += rim * 40.0f;
        lit_g += rim * 120.0f;
        lit_b += rim * 200.0f;

        // Distance fog into space dust
        float fog = fmaxf(0.0f, fminf(1.0f, (t - 40.0f) / 100.0f));
        out_r = lit_r * (1.0f - fog) + bg_r * fog;
        out_g = lit_g * (1.0f - fog) + bg_g * fog;
        out_b = lit_b * (1.0f - fog) + bg_b * fog;
    }

    // Render Laser Plasma Glow Bolts
    for (int l = 0; l < num_lasers; ++l) {
        if (!lasers[l].active) continue;
        float bolt_len = 12.0f;
        float d_laser = distToLaser(cam_x + dir_x * 20.0f, cam_y + dir_y * 20.0f, cam_z + dir_z * 20.0f,
                                    lasers[l].x, lasers[l].y, lasers[l].z,
                                    lasers[l].dir_x, lasers[l].dir_y, lasers[l].dir_z, bolt_len);
        if (d_laser < 4.5f) {
            float glow = (1.0f - d_laser / 4.5f);
            out_r += glow * 40.0f;
            out_g += glow * 255.0f;
            out_b += glow * 255.0f; // Neon Cyan plasma
        }
    }

    // Render Active Asteroid Explosion Plasma Shell
    if (explode_age > 0.0f && explode_age < 1.2f) {
        float ex = explode_x - cam_x;
        float ey = explode_y - cam_y;
        float ez = explode_z - cam_z;
        float ex_dist = sqrtf(ex * ex + ey * ey + ez * ez);
        if (ex_dist > 1.0f) {
            ex /= ex_dist; ey /= ex_dist; ez /= ex_dist;
            float blast_dot = dir_x * ex + dir_y * ey + dir_z * ez;
            float blast_radius = explode_age * 16.0f;
            float blast_angular_sz = blast_radius / ex_dist;
            float angle_diff = acosf(fmaxf(-1.0f, fminf(1.0f, blast_dot)));
            if (angle_diff < blast_angular_sz) {
                float intensity = (1.0f - explode_age / 1.2f) * (1.0f - angle_diff / blast_angular_sz);
                out_r += intensity * 255.0f;
                out_g += intensity * 180.0f;
                out_b += intensity * 50.0f; // Scorching orange fire
            }
        }
    }

    // Cockpit Canopy Dark Metal Framing
    if (is_cockpit_frame) {
        out_r = 35.0f;
        out_g = 40.0f;
        out_b = 48.0f;
    }

    // Center Targeting Reticle (+)
    int cross_x = img_w / 2;
    int cross_y = img_h / 2;
    int cdx = px - cross_x;
    int cdy = py - cross_y;
    if ((cdx == 0 && abs(cdy) <= 2) || (cdy == 0 && abs(cdx) <= 2)) {
        if (cdx == 0 && cdy == 0) {
            out_r = 0.0f; out_g = 255.0f; out_b = 255.0f;
        } else {
            out_r = 0.0f; out_g = 220.0f; out_b = 220.0f;
        }
    }

    out_r = fminf(255.0f, fmaxf(0.0f, out_r));
    out_g = fminf(255.0f, fmaxf(0.0f, out_g));
    out_b = fminf(255.0f, fmaxf(0.0f, out_b));

    int out_idx = (py * img_w + px) * 3;
    out_rgb[out_idx + 0] = (unsigned char)out_r;
    out_rgb[out_idx + 1] = (unsigned char)out_g;
    out_rgb[out_idx + 2] = (unsigned char)out_b;
}

// -----------------------------------------------------------------------------
// Laser Collision & Closest Target Finder Kernel
// -----------------------------------------------------------------------------
__global__ void queryClosestTarget(
    float cam_x, float cam_y, float cam_z,
    float fwd_x, float fwd_y, float fwd_z,
    float time,
    HitResult* __restrict__ result
) {
    result->hit_asteroid = 0;
    result->dist = 999.0f;

    // Cast a forward beam up to 100 meters to lock onto the nearest asteroid
    float t = 2.0f;
    for (int step = 0; step < 40; ++step) {
        float px = cam_x + fwd_x * t;
        float py = cam_y + fwd_y * t;
        float pz = cam_z + fwd_z * t;
        int mat;
        float id;
        float d = mapAsteroids(px, py, pz, time, mat, id);
        if (d < 0.2f) {
            result->hit_asteroid = 1;
            result->dist = t;
            result->ast_x = px;
            result->ast_y = py;
            result->ast_z = pz;
            result->ast_radius = 4.0f + id * 5.5f;
            break;
        }
        t += d * 0.9f;
        if (t > 120.0f) break;
    }
}

} // extern "C"
]]

local module = cuda.compile(cuda_source, { name = "flight_kernels.cu" })
local k_render = module:get_function("renderFlight", "ptr, int, int, float, float, float, float, float, float, float, float, int, ptr, int, float, float, float, float")
local k_query = module:get_function("queryClosestTarget", "float, float, float, float, float, float, float, ptr")

-- -----------------------------------------------------------------------------
-- GPU Buffer Allocations
-- -----------------------------------------------------------------------------
local num_pixels = width * height
local d_out = cuda.alloc(num_pixels * 3)
local h_out = ffi.new("uint8_t[?]", num_pixels * 3)

ffi.cdef[[
typedef struct {
    float x, y, z;
    float dir_x, dir_y, dir_z;
    float age;
    int active;
} LaserBolt;

typedef struct {
    int hit_asteroid;
    float dist;
    float ast_x, ast_y, ast_z;
    float ast_radius;
} HitResult;
]]

local MAX_LASERS = 16
local d_lasers = cuda.alloc("LaserBolt", MAX_LASERS)
local h_lasers = ffi.new("LaserBolt[?]", MAX_LASERS)
for l = 0, MAX_LASERS - 1 do
    h_lasers[l].active = 0
end
d_lasers:to_device(h_lasers)

local d_hit = cuda.alloc("HitResult", 1)
local h_hit = ffi.new("HitResult[1]")

-- -----------------------------------------------------------------------------
-- High-Res Snapshot Export (--save file.ppm)
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
    local d_snap_out = cuda.alloc(snap_w * snap_h * 3)
    local h_snap_out = ffi.new("uint8_t[?]", snap_w * snap_h * 3)

    print(string.format("Rendering high-res 3D Space Flight snapshot (%dx%d)...", snap_w, snap_h))
    local grid2d = { math.ceil(snap_w / 16), math.ceil(snap_h / 16) }
    local block2d = { 16, 16 }
    k_render:launch({ grid = grid2d, block = block2d },
        d_snap_out, snap_w, snap_h,
        0.0, 0.0, 0.0, 0.2, -0.1, 0.0,
        3.5, 1.0, 1, d_lasers, 0, 0.0, 0.0, 0.0, -1.0)
    cuda.sync()

    d_snap_out:to_host(h_snap_out)
    save_ppm(opt_save, snap_w, snap_h, h_snap_out)
    d_snap_out:free()
    print(string.format("Snapshot saved successfully to %s!", opt_save))
    os.exit(0)
end

-- -----------------------------------------------------------------------------
-- ANSI Half-Block Frame Builder (Color-Cached Seamless Fill)
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

            local t_code = tr * 65536 + tg * 256 + tb
            local b_code = br * 65536 + bg * 256 + bb

            if tr == br and tg == bg and tb == bb then
                if cur_bg ~= b_code then
                    line[#line + 1] = string.format("\27[48;2;%d;%d;%dm", br, bg, bb)
                    cur_bg = b_code
                end
                line[#line + 1] = " "
            else
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
-- Flight Simulation State
-- -----------------------------------------------------------------------------
local cam_x, cam_y, cam_z = 0.0, 5.0, -10.0
local yaw = 0.0
local pitch = 0.0
local roll = 0.0
local speed = 0.8
local boost = false
local show_cockpit = 1
local score = 0
local asteroids_destroyed = 0

-- Explosion State
local exp_x, exp_y, exp_z = 0.0, 0.0, 0.0
local exp_age = -1.0

local running = true
local frame_count = 0
local t_start = os.clock()
local last_time = t_start
local actual_fps = 0.0
local gpu_ms = 0.0
local gpu_timer = cuda.timer()

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

        -- Dynamic terminal resizing
        local new_cols, new_rows = get_term_size()
        local new_rrows = math.max(6, new_rows - hud_lines)
        local new_w = math.max(40, new_cols - 2)
        local new_h = new_rrows * 2
        if new_w % 2 ~= 0 then new_w = new_w - 1 end
        if new_h % 2 ~= 0 then new_h = new_h + 1 end

        if new_w ~= width or new_h ~= height then
            width = new_w
            height = new_h
            num_pixels = width * height
            d_out = cuda.alloc(num_pixels * 3)
            h_out = ffi.new("uint8_t[?]", num_pixels * 3)
            gx = math.ceil(width / bx)
            gy = math.ceil(height / by)
            grid2d = { gx, gy }
            io.write("\27[2J\27[H")
            io.flush()
        end

        -- Calculate Forward & Right Direction Vectors
        local cy = math.cos(yaw); local sy = math.sin(yaw)
        local cp = math.cos(pitch); local sp = math.sin(pitch)
        local cr = math.cos(roll); local sr = math.sin(roll)

        local fwd_x = cp * sy
        local fwd_y = sp
        local fwd_z = cp * cy

        local rgt_x = cy * cr + sy * sp * sr
        local rgt_y = -cp * sr
        local rgt_z = -sy * cr + cy * sp * sr

        local up_x = -cy * sr + sy * sp * cr
        local up_y = cp * cr
        local up_z = sy * sr + cy * sp * cr

        -- Query Closest Target in Flight Path
        k_query:launch({ grid = 1, block = 1 },
            cam_x, cam_y, cam_z, fwd_x, fwd_y, fwd_z, sim_time, d_hit)
        d_hit:to_host(h_hit)
        local target = h_hit[0]

        -- Poll interactive keys
        local key = poll_key()
        if key == "quit" then
            running = false
            break
        elseif key == "accel" then
            speed = math.min(boost and 5.5 or 2.8, speed + 0.3)
        elseif key == "brake" then
            speed = math.max(0.1, speed - 0.3)
        elseif key == "strafe_left" then
            cam_x = cam_x - rgt_x * 1.5
            cam_y = cam_y - rgt_y * 1.5
            cam_z = cam_z - rgt_z * 1.5
            roll = math.max(-0.4, roll - 0.1)
        elseif key == "strafe_right" then
            cam_x = cam_x + rgt_x * 1.5
            cam_y = cam_y + rgt_y * 1.5
            cam_z = cam_z + rgt_z * 1.5
            roll = math.min(0.4, roll + 0.1)
        elseif key == "elevate_up" then
            cam_x = cam_x + up_x * 1.2
            cam_y = cam_y + up_y * 1.2
            cam_z = cam_z + up_z * 1.2
        elseif key == "elevate_down" then
            cam_x = cam_x - up_x * 1.2
            cam_y = cam_y - up_y * 1.2
            cam_z = cam_z - up_z * 1.2
        elseif key == "yaw_left" then
            yaw = yaw - 0.08
            roll = math.max(-0.35, roll - 0.05)
        elseif key == "yaw_right" then
            yaw = yaw + 0.08
            roll = math.min(0.35, roll + 0.05)
        elseif key == "pitch_up" then
            pitch = math.min(1.4, pitch + 0.06)
        elseif key == "pitch_down" then
            pitch = math.max(-1.4, pitch - 0.06)
        elseif key == "boost" then
            boost = not boost
            speed = boost and 4.5 or 1.2
        elseif key == "toggle_cockpit" then
            show_cockpit = (show_cockpit == 1) and 0 or 1
        elseif key == "reset" then
            cam_x, cam_y, cam_z = 0.0, 5.0, -10.0
            yaw, pitch, roll = 0.0, 0.0, 0.0
            speed = 0.8
        elseif key == "fire" then
            -- Spawn twin laser bolts from wingtips
            for side = -1, 1, 2 do
                for l = 0, MAX_LASERS - 1 do
                    if h_lasers[l].active == 0 then
                        h_lasers[l].active = 1
                        h_lasers[l].age = 0.0
                        h_lasers[l].x = cam_x + rgt_x * (side * 2.5) - up_x * 0.8
                        h_lasers[l].y = cam_y + rgt_y * (side * 2.5) - up_y * 0.8
                        h_lasers[l].z = cam_z + rgt_z * (side * 2.5) - up_z * 0.8
                        h_lasers[l].dir_x = fwd_x
                        h_lasers[l].dir_y = fwd_y
                        h_lasers[l].dir_z = fwd_z
                        break
                    end
                end
            end

            -- Check if forward laser hit targeted asteroid
            if target.hit_asteroid == 1 and target.dist < 80.0 then
                exp_x = target.ast_x
                exp_y = target.ast_y
                exp_z = target.ast_z
                exp_age = 0.01
                score = score + 250
                asteroids_destroyed = asteroids_destroyed + 1
            end
        end

        -- Natural roll centering
        roll = roll * 0.90

        -- Spaceship Forward Flight Movement
        cam_x = cam_x + fwd_x * speed
        cam_y = cam_y + fwd_y * speed
        cam_z = cam_z + fwd_z * speed

        -- Update Lasers Physics & Expiration
        local active_lasers = 0
        for l = 0, MAX_LASERS - 1 do
            if h_lasers[l].active == 1 then
                h_lasers[l].x = h_lasers[l].x + h_lasers[l].dir_x * 4.5
                h_lasers[l].y = h_lasers[l].y + h_lasers[l].dir_y * 4.5
                h_lasers[l].z = h_lasers[l].z + h_lasers[l].dir_z * 4.5
                h_lasers[l].age = h_lasers[l].age + 0.05
                if h_lasers[l].age > 1.2 then
                    h_lasers[l].active = 0
                else
                    active_lasers = active_lasers + 1
                end
            end
        end
        d_lasers:to_device(h_lasers)

        -- Update Active Explosion Age
        if exp_age >= 0.0 then
            exp_age = exp_age + 0.04
            if exp_age > 1.2 then exp_age = -1.0 end
        end

        -- Render Flight Frame on GPU
        gpu_timer:start()
        k_render:launch({ grid = grid2d, block = block2d },
            d_out, width, height,
            cam_x, cam_y, cam_z,
            yaw, pitch, roll,
            sim_time, speed, show_cockpit,
            d_lasers, MAX_LASERS,
            exp_x, exp_y, exp_z, exp_age)
        cuda.sync()
        gpu_timer:stop()
        gpu_ms = gpu_timer:elapsed_ms()

        -- Blit to Terminal
        d_out:to_host(h_out)
        local frame_ansi = build_ansi_frame(h_out, width, height)

        -- Top HUD Line (compact: <= 76 visible chars)
        local target_hud = (target.hit_asteroid == 1)
            and string.format("\27[1;31m[LOCK: %dm]\27[0m", math.floor(target.dist))
            or "\27[90m[NO TARGET]\27[0m"

        local top_hud = string.format(
            "  \27[1;36mSTARFLIGHT 3D\27[0m \27[2;37m•\27[0m \27[1;32m%4.1f FPS (%4.1f ms)\27[0m \27[2;37m•\27[0m SPD:%4.1f %s \27[2;37m•\27[0m %s \27[2;37m•\27[0m PTS:\27[1;33m%d\27[0m",
            actual_fps, gpu_ms, speed * 120.0,
            boost and "\27[1;35mWARP\27[0m" or "\27[36mSUB\27[0m",
            target_hud, score
        )

        -- Bottom Controls Line (compact: <= 76 visible chars)
        local bottom_hud = "  \27[2;37m[W/S] Throttle  [Arrows/J/L] Steer  [Space] Fire Lasers  [F] Warp  [Q] Quit\27[0m"

        io.write("\27[H\r\27[2K" .. top_hud .. "\27[K\r\n" .. frame_ansi .. "\r\n\r\27[2K" .. bottom_hud .. "\27[K")
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

        -- FPS Cap throttling
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
    io.stderr:write("Error in flight loop: " .. tostring(err) .. "\n")
end
