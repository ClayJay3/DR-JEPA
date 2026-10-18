"""Shared visualization: belief-map view + HUD composition (RGB images)."""

import math

import cv2
import numpy as np


def map_view(nav, size=360, span_m=40.0):
    """North-up RGB view of the belief map around the robot.

    unknown = dark grey, observed free = light, obstacles = red (by
    probability), believed steep ground = orange, route = cyan,
    robot = green arrow, goal = blue."""
    bm, r = nav.map, nav.map.cfg.res
    if bm.corner is None or nav.pose is None:
        return np.zeros((size, size, 3), np.uint8)
    c = int(span_m / r)
    ci, cj = bm.index(nav.pose[0], nav.pose[1])
    i0, j0 = max(0, ci - c // 2), max(0, cj - c // 2)
    sl = (slice(i0, i0 + c), slice(j0, j0 + c))
    prob = bm.prob(sl)
    known = bm.observed(sl)
    img = np.full((*prob.shape, 3), 55, np.uint8)
    img[known] = (215, 215, 210)
    G = np.where(bm.GW[sl] > 0.3, bm.G[sl], 0.0)
    rb = nav.robot
    steep = known & (G > rb.grade_drive)
    a = np.clip((G[steep] - rb.grade_drive) / max(rb.tip_roll - rb.grade_drive, 1e-3), 0, 1)[:, None]
    img[steep] = ((1 - a) * img[steep] + a * np.array((255, 150, 30))).astype(np.uint8)
    occ = known & (prob > 0.5)
    v = ((prob[occ] - 0.5) * 2)[:, None]
    img[occ] = ((1 - v) * np.array((230, 140, 140)) + v * np.array((210, 20, 30))).astype(np.uint8)

    def mark(p, col, rad=1):
        i, j = bm.index(p[0], p[1])
        i, j = i - i0, j - j0
        if 0 <= i < img.shape[0] and 0 <= j < img.shape[1]:
            img[max(0, i - rad):i + rad + 1, max(0, j - rad):j + rad + 1] = col
    path = nav.planner.path
    if path is not None:
        for p in path:
            mark(p, (0, 200, 230), 0)
    mark(nav.goal, (40, 90, 255), 2)
    img = np.ascontiguousarray(np.flipud(img.transpose(1, 0, 2)))
    img = cv2.resize(img, (size, size), interpolation=cv2.INTER_NEAREST)
    # robot arrow (after resize, in pixel space; north-up)
    s = size / c
    px = (ci - i0 + 0.5) * s
    py = size - (cj - j0 + 0.5) * s
    h = math.radians(nav.heading)
    tip = (int(px + 12 * math.sin(h)), int(py - 12 * math.cos(h)))
    cv2.arrowedLine(img, (int(px), int(py)), tip, (20, 220, 60), 2, tipLength=0.4)
    return img


def compose_frame(rgb, nav, sim=None, throttle=0.0, steer=0.0, map_size=360):
    """Camera image (left) + belief map (right) + telemetry text."""
    H, W = rgb.shape[:2]
    out = np.zeros((max(H, map_size), W + map_size, 3), np.uint8)
    out[:H, :W] = rgb
    out[:map_size, W:] = map_view(nav, map_size)
    txt = [f"mode {nav.controller.mode}", f"thr {throttle:+.2f} steer {steer:+.2f}"]
    if sim is not None:
        txt += [f"{sim.scenario}  goal {sim.goal_dist_true():.0f} m", f"contacts {sim.collision_count}"]
    for k, t in enumerate(txt):
        cv2.putText(out, t, (8, 20 + 18 * k), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    return out
