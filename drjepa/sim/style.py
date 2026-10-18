"""Per-episode appearance randomization for the GPU renderer.

`sample_look(rng)` draws one episode's lighting, sky, ground materials,
haze and camera response. Colours are LINEAR RGB (the renderer tone-maps
and gamma-encodes at the end). Ranges are set from real field footage
(URC desert sites, dirt lots, scrubland) plus deliberate extra spread so
the perception net cannot key on any one palette.
"""

import math

import numpy as np


def _lin(rgb255):
    """sRGB 0..255 -> linear 0..1."""
    return (np.asarray(rgb255, float) / 255.0) ** 2.2


# (sRGB) ground palettes: base A, base B, accent patches
GROUND = [
    ((176, 140, 104), (150, 112, 82), (120, 92, 70)),     # tan dirt
    ((168, 108, 74), (140, 86, 60), (178, 128, 94)),      # red desert
    ((150, 144, 134), (124, 118, 110), (170, 160, 146)),  # grey gravel / shale
    ((190, 170, 140), (166, 146, 118), (140, 120, 98)),   # pale silt (Hanksville)
    ((128, 104, 80), (104, 84, 64), (150, 128, 96)),      # dark loam
    ((182, 150, 118), (158, 126, 96), (130, 120, 90)),    # dusty lot
]
GRASS = [(150, 146, 86), (124, 126, 70), (170, 150, 96), (96, 112, 60), (138, 118, 76)]
SAND = [(214, 190, 150), (226, 204, 168), (200, 170, 130), (230, 214, 186)]
ROCK = [(120, 104, 92), (96, 88, 82), (150, 128, 108), (132, 96, 74), (82, 76, 72),
        (170, 156, 140)]
FOLIAGE = [(70, 92, 48), (96, 108, 58), (60, 76, 44), (118, 120, 70), (84, 96, 70)]
BARK = [(80, 64, 50), (96, 82, 68), (62, 52, 44)]


def sample_look(rng: np.random.Generator, augment=False):
    """One episode's renderer parameters (dict consumed by GLRenderer)."""
    overcast = rng.random() < 0.2
    el = math.radians(rng.uniform(12, 75))
    az = rng.uniform(0, 2 * math.pi)
    sun_dir = np.array([math.cos(el) * math.sin(az), math.sin(el),
                        math.cos(el) * math.cos(az)])
    warm = rng.uniform(0.0, 1.0) * (1.0 - min(el / math.radians(40), 1.0))
    sun_col = np.array([1.0, 0.95 - 0.12 * warm, 0.88 - 0.3 * warm])
    sun_int = rng.uniform(2.6, 3.8) * (0.35 if overcast else 1.0)
    sky_int = rng.uniform(0.45, 0.75) * (1.6 if overcast else 1.0)

    zen = _lin(rng.choice([(60, 110, 200), (90, 140, 210), (110, 150, 200), (70, 120, 180)]))
    hor = _lin(rng.choice([(190, 210, 235), (210, 220, 230), (200, 200, 205), (225, 215, 200)]))
    if overcast:
        g = rng.uniform(0.45, 0.65)
        zen = zen * 0.3 + g * 0.7
        hor = hor * 0.3 + (g + 0.1) * 0.7
    zen = zen * rng.uniform(0.85, 1.15)
    hor = hor * rng.uniform(0.9, 1.1)

    gp = GROUND[rng.integers(len(GROUND))]
    jit = lambda c, s=0.08: _lin(np.clip(np.asarray(c) * rng.uniform(1 - s, 1 + s, 3), 0, 255))
    ga, gb, gc = jit(gp[0]), jit(gp[1]), jit(gp[2])
    look = {
        "sun_dir": sun_dir,
        "sun_col": sun_col * sun_int,
        "sun_vis": 0.0 if overcast else 1.0,
        "amb_sky": (hor * 0.6 + zen * 0.4) * sky_int,
        "amb_gnd": (ga * 0.5 + 0.05) * sky_int * 0.6,
        "fog_col": hor * rng.uniform(0.9, 1.05),
        "fog_dist": float(np.exp(rng.uniform(np.log(180), np.log(1500)))),
        "zenith": zen, "horizon": hor,
        "cloud_cover": float(rng.uniform(0.85, 1.0) if overcast else rng.uniform(0.0, 0.65)),
        "cloud_scale": float(rng.uniform(0.6, 1.6)),
        "seed": rng.uniform(-500, 500, 2),
        "mtn_col": jit(rng.choice(GROUND)[0], 0.15) * rng.uniform(0.6, 1.0),
        "mtn_amp": float(rng.uniform(0.0, 0.07)),
        "mtn_freq": float(rng.uniform(1.5, 6.0)),
        "ga": ga, "gb": gb, "gc": gc,
        "sandc": jit(SAND[rng.integers(len(SAND))]),
        "grassc": jit(GRASS[rng.integers(len(GRASS))], 0.12),
        "grass_amt": float(rng.uniform(0.0, 0.9)),
        "grass_thr": float(rng.uniform(0.35, 0.6)),
        "rough": float(rng.uniform(0.4, 1.4)),
        "exposure": rng.uniform(0.92, 1.08, 3) * rng.uniform(0.45, 0.8),
        "vignette": float(rng.uniform(0.0, 0.45)),
        # prop palettes (linear)
        "rock": [jit(c, 0.1) for c in ROCK],
        "foliage": [jit(c, 0.12) for c in FOLIAGE],
        "bark": [jit(c) for c in BARK],
        # camera / sensor (applied on the CPU after readback)
        "noise_sigma": float(rng.uniform(0.8, 4.0)),
        "motion_blur": float(rng.uniform(0.0, 0.35)),
        "jpeg_q": int(rng.integers(55, 96)),
        "tuft_density": float(rng.uniform(0.02, 0.6)),
        "bump_amp_deg": float(rng.uniform(0.2, 0.9)),
    }
    if augment:
        # colour is not a reliable class cue across sites: on some episodes
        # push every palette through a random hue rotation / desaturation
        look["hue_shift"] = float(rng.uniform(-20, 20))
        r = rng.random()
        look["sat_scale"] = 0.0 if r < 0.12 else (1.0 if r < 0.5 else rng.uniform(0.4, 1.3))
        look["gamma"] = float(rng.uniform(0.8, 1.25))
    else:
        look["hue_shift"], look["sat_scale"], look["gamma"] = 0.0, 1.0, 1.0
    return look


def camera_fx(rgb, look, rng, prev=None, speed_frac=0.5):
    """Sensor effects on a rendered uint8 RGB frame: motion blur, colour
    jitter (augment only) and JPEG compression. (Shot / read noise is added
    on the GPU in the resolve pass.)"""
    import cv2

    img = rgb
    if prev is not None and look["motion_blur"] > 0:
        a = min(look["motion_blur"] * (0.4 + speed_frac), 0.6)
        img = cv2.addWeighted(rgb, 1 - a, prev, a, 0.0)
    out_prev = img
    if look["gamma"] != 1.0:
        lut = (np.clip(np.arange(256) / 255.0, 0, 1) ** look["gamma"] * 255.0).astype(np.uint8)
        img = cv2.LUT(img, lut)
    if look["hue_shift"] != 0.0 or look["sat_scale"] != 1.0:
        hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV).astype(np.float32)
        hsv[..., 0] = (hsv[..., 0] + look["hue_shift"]) % 180.0
        hsv[..., 1] = np.clip(hsv[..., 1] * look["sat_scale"], 0, 255)
        img = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)
    q = look["jpeg_q"]
    if q < 95:
        ok, enc = cv2.imencode(".jpg", img[:, :, ::-1], [cv2.IMWRITE_JPEG_QUALITY, q])
        img = cv2.imdecode(enc, cv2.IMREAD_COLOR)[:, :, ::-1]
    return np.ascontiguousarray(img), out_prev
