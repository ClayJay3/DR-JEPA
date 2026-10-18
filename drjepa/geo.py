"""Small geodesy helpers: local tangent-plane metres <-> GPS degrees.

Local frame convention used throughout DR-JEPA: x = east, z = north
(metres), heading in compass degrees (0 = north, clockwise positive).
"""

import math

M_PER_DEG = 111139.0      # metres per degree of latitude (spherical approx.)


def meters_to_latlon(x, z, origin_lat, origin_lon):
    """Local metres (x east, z north) -> (lat, lon) degrees."""
    lat = origin_lat + z / M_PER_DEG
    lon = origin_lon + x / (M_PER_DEG * math.cos(math.radians(origin_lat)))
    return lat, lon


def latlon_to_meters(lat, lon, origin_lat, origin_lon):
    """(lat, lon) degrees -> local metres (x east, z north) about an origin."""
    z = (lat - origin_lat) * M_PER_DEG
    x = (lon - origin_lon) * M_PER_DEG * math.cos(math.radians(origin_lat))
    return x, z


def world_to_robot(dx, dz, yaw_deg):
    """World offset (east, north) -> robot frame (right, forward)."""
    c, s = math.cos(math.radians(yaw_deg)), math.sin(math.radians(yaw_deg))
    return dx * c - dz * s, dx * s + dz * c


def robot_to_world(xr, zr, yaw_deg):
    """Robot frame (right, forward) -> world offset (east, north)."""
    c, s = math.cos(math.radians(yaw_deg)), math.sin(math.radians(yaw_deg))
    return xr * c + zr * s, -xr * s + zr * c
