"""GLSL sources for the GPU renderer (OpenGL 3.3 core)."""

NOISE = """
float hash12(vec2 p) {
    vec3 p3 = fract(vec3(p.xyx) * .1031);
    p3 += dot(p3, p3.yzx + 33.33);
    return fract((p3.x + p3.y) * p3.z);
}
float hash13(vec3 p3) {
    p3 = fract(p3 * .1031);
    p3 += dot(p3, p3.zyx + 31.32);
    return fract((p3.x + p3.y) * p3.z);
}
float vnoise(vec2 p) {
    vec2 i = floor(p), f = fract(p);
    vec2 u = f * f * (3.0 - 2.0 * f);
    return mix(mix(hash12(i), hash12(i + vec2(1, 0)), u.x),
               mix(hash12(i + vec2(0, 1)), hash12(i + vec2(1, 1)), u.x), u.y);
}
float vnoise3(vec3 p) {
    vec3 i = floor(p), f = fract(p);
    vec3 u = f * f * (3.0 - 2.0 * f);
    float a = mix(mix(hash13(i), hash13(i + vec3(1, 0, 0)), u.x),
                  mix(hash13(i + vec3(0, 1, 0)), hash13(i + vec3(1, 1, 0)), u.x), u.y);
    float b = mix(mix(hash13(i + vec3(0, 0, 1)), hash13(i + vec3(1, 0, 1)), u.x),
                  mix(hash13(i + vec3(0, 1, 1)), hash13(i + vec3(1, 1, 1)), u.x), u.y);
    return mix(a, b, u.z);
}
float fbm(vec2 p, int oct) {
    float a = 0.5, s = 0.0;
    for (int k = 0; k < 8; k++) {
        if (k >= oct) break;
        s += a * vnoise(p);
        p = p * 2.03 + vec2(17.1, 9.2);
        a *= 0.5;
    }
    return s;
}
float fbm3(vec3 p, int oct) {
    float a = 0.5, s = 0.0;
    for (int k = 0; k < 6; k++) {
        if (k >= oct) break;
        s += a * vnoise3(p);
        p = p * 2.07 + vec3(11.3, 5.7, 3.1);
        a *= 0.5;
    }
    return s;
}
"""

LIGHTING = """
uniform vec3 u_cam_pos;
uniform vec3 u_sun_dir;
uniform vec3 u_sun_col;
uniform vec3 u_amb_sky;
uniform vec3 u_amb_gnd;
uniform vec3 u_fog_col;
uniform float u_fog_dist;
uniform mat4 u_light_vp;
uniform sampler2DShadow u_shadow;
uniform float u_shadow_texel;

float shadow_at(vec3 wp, vec3 n) {
    vec4 lp = u_light_vp * vec4(wp + n * 0.04, 1.0);
    vec3 s = lp.xyz / lp.w * 0.5 + 0.5;
    if (s.x <= 0.0 || s.x >= 1.0 || s.y <= 0.0 || s.y >= 1.0 || s.z >= 1.0)
        return 1.0;
    float bias = 0.0008;
    float acc = 0.0;
    for (int dx = -1; dx <= 1; dx++)
        for (int dy = -1; dy <= 1; dy++)
            acc += texture(u_shadow, vec3(s.xy + vec2(dx, dy) * u_shadow_texel, s.z - bias));
    return acc / 9.0;
}

vec3 shade(vec3 albedo, vec3 n, vec3 wp, float wrap, float spec) {
    float sh = shadow_at(wp, n);
    float ndl = max((dot(n, u_sun_dir) + wrap) / (1.0 + wrap), 0.0);
    vec3 amb = mix(u_amb_gnd, u_amb_sky, 0.5 + 0.5 * n.y);
    vec3 c = albedo * (u_sun_col * ndl * sh + amb);
    if (spec > 0.0) {
        vec3 v = normalize(u_cam_pos - wp);
        vec3 h = normalize(v + u_sun_dir);
        c += u_sun_col * sh * spec * pow(max(dot(n, h), 0.0), 40.0);
    }
    return c;
}

vec3 fogged(vec3 c, vec3 wp) {
    float d = length(wp - u_cam_pos);
    float f = 1.0 - exp(-d / u_fog_dist);
    return mix(c, u_fog_col, clamp(f, 0.0, 1.0));
}
"""

HEIGHT = """
uniform sampler2D u_hmap;
uniform sampler2D u_smap;
uniform float u_extent;
uniform float u_res;
uniform float u_n;

float bil(sampler2D t, vec2 xz) {
    vec2 g = clamp((xz + u_extent) / u_res, vec2(0.0), vec2(u_n - 1.001));
    ivec2 i0 = ivec2(floor(g));
    vec2 f = g - vec2(i0);
    float a = texelFetch(t, i0, 0).r;
    float b = texelFetch(t, i0 + ivec2(1, 0), 0).r;
    float c = texelFetch(t, i0 + ivec2(0, 1), 0).r;
    float d = texelFetch(t, i0 + ivec2(1, 1), 0).r;
    return mix(mix(a, b, f.x), mix(c, d, f.x), f.y);
}
float hgt(vec2 xz) { return bil(u_hmap, xz); }
float sand(vec2 xz) { return clamp(bil(u_smap, xz), 0.0, 1.0); }
"""

# ---------------------------------------------------------------- terrain
TERRAIN_VS = """
#version 330
in vec2 in_xz;
uniform vec2 u_center;
uniform mat4 u_vp;
""" + HEIGHT + """
out vec3 v_wp;
void main() {
    vec2 xz = u_center + in_xz;
    v_wp = vec3(xz.x, hgt(xz), xz.y);
    gl_Position = u_vp * vec4(v_wp, 1.0);
}
"""

TERRAIN_FS = """
#version 330
in vec3 v_wp;
uniform mat4 u_view;
uniform vec3 u_ga;
uniform vec3 u_gb;
uniform vec3 u_gc;
uniform vec3 u_sandc;
uniform vec3 u_grassc;
uniform float u_grass_amt;
uniform float u_grass_thr;
uniform float u_rough;
uniform vec2 u_seed;
layout(location = 0) out vec4 o_color;
layout(location = 1) out float o_depth;
layout(location = 2) out float o_cls;
""" + NOISE + LIGHTING + HEIGHT + """
void main() {
    vec2 xz = v_wp.xz;
    float e = 0.5;
    vec3 n = normalize(vec3(-(hgt(xz + vec2(e, 0)) - hgt(xz - vec2(e, 0))) / (2.0 * e), 1.0,
                            -(hgt(xz + vec2(0, e)) - hgt(xz - vec2(0, e))) / (2.0 * e)));
    // pixel footprint (m) -> per-band fade, so fine detail never aliases
    float px = max(length(fwidth(xz)), 1e-4);
    float f_mid = clamp(1.5 - px * 1.2, 0.0, 1.0);     // ~0.3-1 m features
    float f_sm = clamp(1.5 - px * 8.0, 0.0, 1.0);      // ~5-20 cm
    float f_gr = clamp(1.5 - px * 40.0, 0.0, 1.0);     // ~1-3 cm grain
    vec2 p = xz + u_seed;
    float s = sand(xz);
    float nl = fbm(p * 0.025, 4);
    float nm = fbm(p * 0.21, 4);
    float ns = fbm(p * 1.7, 3);
    vec3 alb = mix(u_ga, u_gb, smoothstep(0.32, 0.68, nl));
    alb = mix(alb, u_gc, 0.55 * smoothstep(0.55, 0.8, fbm(p * 0.07 + 3.3, 3)));
    alb *= 0.84 + 0.32 * nm;
    alb *= mix(1.0, 0.78 + 0.44 * ns, f_mid);
    // grain: two octaves of fine speckle (strong near the lens, like real dirt)
    float g1 = vnoise(p * 23.0), g2 = vnoise(p * 61.0 + 4.7), g3 = vnoise(p * 140.0 + 1.3);
    alb *= mix(1.0, 0.72 + 0.56 * g1, f_sm * (1.0 - 0.6 * s));
    alb *= mix(1.0, 0.76 + 0.48 * g2, f_gr * (1.0 - 0.5 * s));
    alb *= mix(1.0, 0.85 + 0.3 * g3, f_gr);
    // pebbles: sparse blobs with their own tint, lit by the micro-bump below
    float pb = vnoise(p * 17.0 + 2.1);
    float peb = smoothstep(0.84, 0.9, pb) * f_sm * (1.0 - s);
    vec3 pcol = mix(alb, u_gc, 0.5) * (0.55 + 0.8 * vnoise(floor(p * 17.0) + 0.5));
    alb = mix(alb, pcol, peb * 0.85);
    // cracks in dry mud / hardpan (thin ridged lines), only on some palettes
    float cr = 1.0 - abs(2.0 * vnoise(p * 2.2 + 8.0) - 1.0);
    alb *= 1.0 - 0.35 * smoothstep(0.93, 0.985, cr) * f_mid * u_rough * 0.6 * (1.0 - s);
    // dry grass / scrub coverage
    float g = smoothstep(u_grass_thr, u_grass_thr + 0.18, fbm(p * 0.06 + 7.1, 4)) * u_grass_amt;
    g *= 0.55 + 0.45 * mix(0.5, vnoise(p * 4.0), f_mid);
    alb = mix(alb, u_grassc * (0.65 + 0.7 * ns) * mix(1.0, 0.7 + 0.6 * g2, f_gr), g);
    // sand: lighter, smoother, faint ripples
    float rip = 0.5 + 0.5 * sin(dot(p, vec2(5.3, 2.1)) + 3.0 * vnoise(p * 0.7));
    alb = mix(alb, u_sandc * (0.9 + 0.15 * nm) * (0.96 + 0.08 * rip * f_mid), 0.88 * s);
    // micro-relief: bump from the pebble field + mid relief (fades with range)
    float d = 0.012;
    float h0 = 0.6 * smoothstep(0.82, 0.92, pb) + 0.2 * vnoise(p * 23.0);
    float hx = 0.6 * smoothstep(0.82, 0.92, vnoise((p + vec2(d, 0)) * 17.0 + 2.1)) + 0.2 * vnoise((p + vec2(d, 0)) * 23.0);
    float hz = 0.6 * smoothstep(0.82, 0.92, vnoise((p + vec2(0, d)) * 17.0 + 2.1)) + 0.2 * vnoise((p + vec2(0, d)) * 23.0);
    vec3 bn = vec3(-(hx - h0) / d, 0.0, -(hz - h0) / d) * 0.02 * u_rough * f_sm * (1.0 - 0.8 * s);
    float m0 = fbm(p * 0.9, 3);
    float mx = fbm((p + vec2(0.05, 0)) * 0.9, 3), mz = fbm((p + vec2(0, 0.05)) * 0.9, 3);
    bn += vec3(-(mx - m0) / 0.05, 0.0, -(mz - m0) / 0.05) * 0.06 * u_rough * f_mid * (1.0 - 0.7 * s);
    n = normalize(n + bn);
    vec3 c = shade(alb, n, v_wp, 0.0, 0.0);
    o_color = vec4(fogged(c, v_wp), 1.0);
    o_depth = -(u_view * vec4(v_wp, 1.0)).z;
    o_cls = 0.0;
}
"""

# ---------------------------------------------------------------- objects
OBJ_VS = """
#version 330
in vec3 in_pos;
in vec3 in_nrm;
in float in_part;
in mat4 i_model;
in vec3 i_c0;
in vec3 i_c1;
in vec3 i_c2;
in vec4 i_mat;
uniform mat4 u_vp;
out vec3 v_wp;
out vec3 v_n;
out vec3 v_col;
out vec3 v_lp;
out float v_part;
flat out vec2 v_mat;
void main() {
    vec4 wp = i_model * vec4(in_pos, 1.0);
    v_wp = wp.xyz;
    v_n = normalize(transpose(inverse(mat3(i_model))) * in_nrm);
    v_col = in_part < 0.5 ? i_c0 : (in_part < 1.5 ? i_c1 : i_c2);
    v_lp = in_pos;
    v_part = in_part;
    float m = in_part < 0.5 ? i_mat.x : (in_part < 1.5 ? i_mat.y : i_mat.z);
    v_mat = vec2(m, i_mat.w);
    gl_Position = u_vp * wp;
}
"""

OBJ_FS = """
#version 330
in vec3 v_wp;
in vec3 v_n;
in vec3 v_col;
in vec3 v_lp;
in float v_part;
flat in vec2 v_mat;
uniform mat4 u_view;
layout(location = 0) out vec4 o_color;
layout(location = 1) out float o_depth;
layout(location = 2) out float o_cls;
""" + NOISE + LIGHTING + """
void main() {
    vec3 n = normalize(v_n);
    int mat = int(v_mat.x + 0.5);
    // thin double-sided geometry (grass blades): light the side we see
    if (mat == 6 && dot(n, u_cam_pos - v_wp) < 0.0) n = -n;
    vec3 alb = v_col;
    float wrap = 0.0, spec = 0.0;
    vec3 q = v_wp * 2.3;
    if (mat == 0) {                       // rock
        float a = fbm3(q, 5);
        float b = fbm3(q * 3.1 + 1.7, 3);
        alb *= 0.7 + 0.55 * a;
        alb *= 0.85 + 0.3 * b;
        alb = mix(alb, alb * 0.55, smoothstep(0.62, 0.7, fbm3(q * 1.4 + 9.0, 3)) * 0.6);
        float e = 0.05;
        vec3 gq = vec3(fbm3(q + vec3(e, 0, 0), 4), fbm3(q + vec3(0, e, 0), 4), fbm3(q + vec3(0, 0, e), 4)) - fbm3(q, 4);
        n = normalize(n - gq / e * 0.35);
        // dust settles on top faces
        alb = mix(alb, alb * 0.5 + vec3(0.25, 0.2, 0.15) * 0.5, smoothstep(0.6, 0.95, n.y) * 0.35);
    } else if (mat == 1) {                // foliage
        float a = fbm3(v_wp * 9.0, 3);
        alb *= 0.45 + 0.9 * a;
        float e = 0.03;
        vec3 w = v_wp * 9.0;
        vec3 gq = vec3(vnoise3(w + vec3(e * 9.0, 0, 0)), vnoise3(w + vec3(0, e * 9.0, 0)), vnoise3(w + vec3(0, 0, e * 9.0))) - vnoise3(w);
        n = normalize(n - gq * 1.2);
        wrap = 0.5;
        alb *= mix(0.55, 1.0, clamp(v_lp.y * 1.2, 0.0, 1.0));
    } else if (mat == 2) {                // bark / poles / dull
        alb *= 0.75 + 0.5 * vnoise3(vec3(v_wp.x * 30.0, v_wp.y * 3.0, v_wp.z * 30.0));
    } else if (mat == 3) {                // paint / plastic (cones, vehicles)
        spec = 0.35;
        alb *= 0.92 + 0.16 * fbm3(q * 0.5, 2);
    } else if (mat == 4) {                // cloth / people
        alb *= 0.8 + 0.4 * fbm3(v_wp * 12.0, 2);
        wrap = 0.2;
    } else if (mat == 5) {                // metal siding / buildings
        float rib = 0.5 + 0.5 * sin((v_wp.x + v_wp.z) * 18.0);
        alb *= 0.85 + 0.2 * rib;
        spec = 0.25;
        alb *= 0.9 + 0.2 * fbm3(q * 0.3, 2);
    } else {                              // grass tuft
        alb *= mix(0.55, 1.15, clamp(v_lp.y, 0.0, 1.0));
        alb *= 0.8 + 0.4 * hash13(floor(v_wp * 40.0));
        wrap = 0.6;
    }
    vec3 c = shade(alb, n, v_wp, wrap, spec);
    o_color = vec4(fogged(c, v_wp), 1.0);
    o_depth = -(u_view * vec4(v_wp, 1.0)).z;
    o_cls = v_mat.y / 255.0;
}
"""

SHADOW_VS_TERRAIN = """
#version 330
in vec2 in_xz;
uniform vec2 u_center;
uniform mat4 u_vp;
""" + HEIGHT + """
void main() {
    vec2 xz = u_center + in_xz;
    gl_Position = u_vp * vec4(xz.x, hgt(xz), xz.y, 1.0);
}
"""

SHADOW_VS_OBJ = """
#version 330
in vec3 in_pos;
in mat4 i_model;
uniform mat4 u_vp;
void main() { gl_Position = u_vp * i_model * vec4(in_pos, 1.0); }
"""

SHADOW_FS = """
#version 330
void main() {}
"""

# ---------------------------------------------------------------- sky
FULLSCREEN_VS = """
#version 330
out vec2 v_uv;
void main() {
    vec2 p = vec2((gl_VertexID << 1) & 2, gl_VertexID & 2);
    v_uv = p;
    gl_Position = vec4(p * 2.0 - 1.0, 0.0, 1.0);
}
"""

SKY_FS = """
#version 330
in vec2 v_uv;
uniform mat4 u_inv_vp;
uniform vec3 u_cam_pos;
uniform vec3 u_sun_dir;
uniform vec3 u_sun_col;
uniform vec3 u_zenith;
uniform vec3 u_horizon;
uniform vec3 u_fog_col;
uniform float u_cloud_cover;
uniform float u_cloud_scale;
uniform vec2 u_seed;
uniform float u_sun_vis;
uniform vec3 u_mtn_col;
uniform float u_mtn_amp;
uniform float u_mtn_freq;
layout(location = 0) out vec4 o_color;
layout(location = 1) out float o_depth;
layout(location = 2) out float o_cls;
""" + NOISE + """
void main() {
    vec4 a = u_inv_vp * vec4(v_uv * 2.0 - 1.0, 1.0, 1.0);
    vec3 dir = normalize(a.xyz / a.w - u_cam_pos);
    float el = dir.y;
    vec3 sky = mix(u_horizon, u_zenith, pow(clamp(el, 0.0, 1.0), 0.45));
    float cs = max(dot(dir, u_sun_dir), 0.0);
    // clouds on a plane at 1.5 km
    if (el > 0.005) {
        vec2 cp = dir.xz / el * 1.5 * u_cloud_scale + u_seed;
        float c = fbm(cp * 1.3, 6);
        float cov = smoothstep(1.0 - u_cloud_cover, 1.0 - u_cloud_cover + 0.35, c);
        vec3 ccol = mix(vec3(0.75), vec3(1.0), smoothstep(0.4, 0.9, fbm(cp * 2.6 + 3.0, 4)));
        ccol *= 0.75 + 0.5 * cs;
        sky = mix(sky, ccol * (u_sun_col * 0.35 + 0.65), cov * smoothstep(0.0, 0.08, el) * 0.95);
    }
    sky += u_sun_col * u_sun_vis * (pow(cs, 900.0) * 30.0 + pow(cs, 60.0) * 0.6 + pow(cs, 6.0) * 0.12);
    // distant mesas / hills on the horizon (elevation-angle profile)
    float az = atan(dir.x, dir.z);
    float prof = u_mtn_amp * (0.35 * fbm(vec2(az * u_mtn_freq, 1.3), 5) +
                              0.65 * pow(fbm(vec2(az * u_mtn_freq * 0.37, 7.7), 3), 2.0)) - 0.004;
    float cls = 3.0;
    if (el < prof) {
        float t = clamp(el / max(prof, 1e-4), 0.0, 1.0);
        vec3 m = u_mtn_col * (0.75 + 0.35 * fbm(vec2(az * 60.0, el * 400.0), 3));
        m = mix(m, u_fog_col, 0.55 + 0.25 * (1.0 - t));
        sky = m;
    }
    if (el < -0.02) sky = u_fog_col;
    o_color = vec4(sky, 1.0);
    o_depth = 0.0;
    o_cls = cls / 255.0;
}
"""

# ------------------------------------------------- supersample resolve + post
RESOLVE_FS = """
#version 330
in vec2 v_uv;
uniform sampler2D u_hdr;
uniform sampler2D u_lin;
uniform sampler2D u_cls;
uniform int u_ss;
uniform vec3 u_exposure;
uniform float u_vignette;
uniform vec2 u_size;
uniform float u_noise;
uniform float u_frame;
layout(location = 0) out vec4 o_color;
layout(location = 1) out float o_depth;
layout(location = 2) out float o_cls;
vec3 aces(vec3 x) {
    return clamp((x * (2.51 * x + 0.03)) / (x * (2.43 * x + 0.59) + 0.14), 0.0, 1.0);
}
void main() {
    ivec2 base = ivec2(gl_FragCoord.xy) * u_ss;
    vec3 acc = vec3(0.0);
    float dmin = 1e9;
    float cbest = 3.0 / 255.0;
    for (int dy = 0; dy < 4; dy++) {
        if (dy >= u_ss) break;
        for (int dx = 0; dx < 4; dx++) {
            if (dx >= u_ss) break;
            ivec2 p = base + ivec2(dx, dy);
            acc += texelFetch(u_hdr, p, 0).rgb;
            float d = texelFetch(u_lin, p, 0).r;
            if (d > 0.0 && d < dmin) { dmin = d; cbest = texelFetch(u_cls, p, 0).r; }
        }
    }
    vec3 c = acc / float(u_ss * u_ss);
    c *= u_exposure;
    vec2 q = gl_FragCoord.xy / u_size - 0.5;
    c *= 1.0 - u_vignette * dot(q, q) * 2.0;
    c = aces(c);
    c = mix(vec3(dot(c, vec3(0.2126, 0.7152, 0.0722))), c, 0.88);   // ACES oversaturates
    c = pow(c, vec3(1.0 / 2.2));
    // sensor noise: shot noise (~sqrt of signal) + read noise, per channel
    vec3 r3 = vec3(fract(sin(dot(gl_FragCoord.xy + u_frame * vec2(7.13, 3.71), vec2(12.9898, 78.233))) * 43758.5453),
                   fract(sin(dot(gl_FragCoord.xy + u_frame * vec2(1.37, 9.17), vec2(39.346, 11.135))) * 24634.6345),
                   fract(sin(dot(gl_FragCoord.xy + u_frame * vec2(5.51, 2.23), vec2(73.156, 52.235))) * 15731.7432));
    vec3 r4 = fract(r3.yzx * 91.7 + vec3(0.31, 0.17, 0.53));
    vec3 g = sqrt(-2.0 * log(max(r3, 1e-6))) * cos(6.2831853 * r4);
    c += g * u_noise * sqrt(max(c, 0.03)) * 1.6;
    o_color = vec4(clamp(c, 0.0, 1.0), 1.0);
    o_depth = dmin < 1e8 ? dmin : 0.0;
    o_cls = cbest;
}
"""
