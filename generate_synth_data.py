import cv2
import numpy as np
import pandas as pd
import math
import os
from tqdm import tqdm
from multiprocessing import Pool, cpu_count

# ==========================================
# CONFIGURATION
# ==========================================
OUTPUT_DIR = "data_synthetic_unicycle"
NUM_EPISODES = 1000
FRAMES_PER_EP = 800
IMG_W, IMG_H = 224, 224
FOV = 90.0

# Base Origin for GPS Conversion
ORIGIN_LAT = 35.000000
ORIGIN_LON = -120.000000

# Colors (BGR)
C_SKY = (230, 200, 150)
C_GROUND = (60, 50, 40)
C_ROCK = (50, 50, 200)   
C_TREE = (50, 200, 50)   
C_GOAL = (0, 255, 255)   

# ==========================================
# HELPERS
# ==========================================
def meters_to_latlon(x, z, origin_lat, origin_lon):
    """Converts local Cartesian meters to GPS coordinates."""
    # Z represents North/South (Latitude). 1 degree lat is approx 111,139 meters.
    lat = origin_lat + (z / 111139.0)
    
    # X represents East/West (Longitude). Scales based on current latitude.
    lon = origin_lon + (x / (111139.0 * math.cos(math.radians(origin_lat))))
    
    return lat, lon

# ==========================================
# 3D RENDER ENGINE
# ==========================================
class RenderEngine:
    def __init__(self, width, height, fov):
        self.W, self.H = width, height
        self.cx, self.cy = width / 2, height / 2
        self.f = (width / 2) / math.tan(math.radians(fov / 2))
        
    def world_to_cam(self, points, cam_x, cam_y, cam_z, cam_yaw):
        pts_trans = points - np.array([cam_x, cam_y, cam_z])
        theta = math.radians(cam_yaw)
        c, s = math.cos(theta), math.sin(theta)
        x = pts_trans[:, 0] * c - pts_trans[:, 2] * s
        y = pts_trans[:, 1]
        z = pts_trans[:, 0] * s + pts_trans[:, 2] * c
        return np.stack([x, y, z], axis=1)

    def project(self, points_cam):
        z = np.maximum(points_cam[:, 2], 0.1)
        u = (points_cam[:, 0] / z) * self.f + self.cx
        v = self.cy - (points_cam[:, 1] / z) * self.f 
        return np.stack([u, v], axis=1)

class WorldObject:
    def __init__(self, x, z, width, height, depth, color, type_id):
        self.x, self.z = x, z
        self.c = color
        self.type = type_id 
        w, h, d = width/2, height, depth/2
        self.local_verts = np.array([
            [-w, 0, -d], [w, 0, -d], [w, 0, d], [-w, 0, d],  
            [-w, h, -d], [w, h, -d], [w, h, d], [-w, h, d]   
        ], dtype=float)
        self.faces = [
            ([0, 4, 5, 1], 0.7), ([1, 5, 6, 2], 0.8), 
            ([2, 6, 7, 3], 1.0), ([3, 7, 4, 0], 0.8), ([4, 7, 6, 5], 1.2)
        ]

    def get_world_verts(self):
        return self.local_verts + np.array([self.x, 0, self.z])

# ==========================================
# SIMULATION
# ==========================================
class RoverSim:
    def __init__(self, spawn_type='normal'):
        self.renderer = RenderEngine(IMG_W, IMG_H, FOV)
        self.cam_height = 1.5
        self.cam_offset_forward = 0.5 
        self.x = 0.0
        self.z = 0.0
        self.yaw = 0.0 
        self.objects = []
        self._init_world(spawn_type)
        
        if spawn_type == 'recovery':
            obstacles = [obj for obj in self.objects if obj.type != 'goal']
            if obstacles:
                target = np.random.choice(obstacles)
                spawn_angle = np.random.uniform(0, 360)
                distance_from_obj = 3.0 
                
                self.x = target.x + math.sin(math.radians(spawn_angle)) * distance_from_obj
                self.z = target.z + math.cos(math.radians(spawn_angle)) * distance_from_obj
                
                dx = target.x - self.x
                dz = target.z - self.z
                self.yaw = math.degrees(math.atan2(dx, dz))
                
        elif spawn_type == 'uturn':
            dx = self.goal.x - self.x
            dz = self.goal.z - self.z
            goal_yaw = math.degrees(math.atan2(dx, dz))
            self.yaw = goal_yaw + np.random.uniform(135, 225)

    def _init_world(self, spawn_type):
        dist = 50 if np.random.rand() < 0.2 else 400
        angle = np.random.uniform(-45, 45)
        gx = math.sin(math.radians(angle)) * dist
        gz = math.cos(math.radians(angle)) * dist
        self.goal = WorldObject(gx, gz, 4, 40, 4, C_GOAL, 'goal')
        self.objects.append(self.goal)
        
        if spawn_type == 'wall':
            wall_z = np.random.uniform(30, 80)
            gap_center = np.random.uniform(-15, 15)
            # Spawn a wall with a guaranteed gap so the reactive planner doesn't get stuck
            for ox in range(-30, 31, 4): 
                if abs(ox - gap_center) < 6.0: 
                    continue # Create a doorway
                # Add noise to break perfect force cancellation
                noisy_ox = ox + np.random.uniform(-0.5, 0.5)
                self.objects.append(WorldObject(noisy_ox, wall_z, 4.0, 1.5, 4.0, C_ROCK, 'rock'))
                
        elif spawn_type == 'dense':
            for _ in range(200):
                ox = np.random.uniform(-30, 30)
                oz = np.random.uniform(20, 100)
                if abs(ox) < 3 and abs(oz) < 10: continue
                self.objects.append(WorldObject(ox, oz, 1.0, 10.0, 1.0, C_TREE, 'tree'))
                
        else:
            for _ in range(350):
                ox = np.random.uniform(-250, 250) 
                oz = np.random.uniform(10, 300)   
                if abs(ox) < 5 and abs(oz) < 10: continue 
                
                if np.random.rand() > 0.5:
                    self.objects.append(WorldObject(ox, oz, 4.0, 1.5, 4.0, C_ROCK, 'rock'))
                else:
                    self.objects.append(WorldObject(ox, oz, 1.0, 10.0, 1.0, C_TREE, 'tree'))

    def step(self, angular_velocity, linear_velocity):
        dt = 0.1
        self.yaw += angular_velocity * dt
        rad = math.radians(self.yaw)
        self.x += math.sin(rad) * linear_velocity * dt
        self.z += math.cos(rad) * linear_velocity * dt

    def render(self):
        canvas = np.zeros((IMG_H, IMG_W, 3), dtype=np.uint8)
        canvas[:] = C_SKY
        
        c_rad = math.radians(self.yaw)
        cx = self.x + math.sin(c_rad) * self.cam_offset_forward
        cz = self.z + math.cos(c_rad) * self.cam_offset_forward
        
        lines_to_draw = []
        grid_sz = 100
        step = 10
        bx = int(self.x // step) * step
        bz = int(self.z // step) * step
        
        for i in range(-grid_sz, grid_sz+1, step):
            p1 = np.array([bx + i, 0, bz - grid_sz])
            p2 = np.array([bx + i, 0, bz + grid_sz])
            lines_to_draw.append((p1, p2))
            p3 = np.array([bx - grid_sz, 0, bz + i])
            p4 = np.array([bx + grid_sz, 0, bz + i])
            lines_to_draw.append((p3, p4))
            
        for p1, p2 in lines_to_draw:
            pts = np.vstack([p1, p2])
            c_pts = self.renderer.world_to_cam(pts, cx, self.cam_height, cz, self.yaw)
            if c_pts[0,2] < 1.0 and c_pts[1,2] < 1.0: continue
            if c_pts[0,2] < 1.0:
                t = (1.0 - c_pts[0,2]) / (c_pts[1,2] - c_pts[0,2])
                c_pts[0] += t * (c_pts[1] - c_pts[0])
            elif c_pts[1,2] < 1.0:
                t = (1.0 - c_pts[1,2]) / (c_pts[0,2] - c_pts[1,2])
                c_pts[1] += t * (c_pts[0] - c_pts[1])
            s_pts = self.renderer.project(c_pts)
            cv2.line(canvas, tuple(s_pts[0].astype(int)), tuple(s_pts[1].astype(int)), C_GROUND, 1)

        render_queue = []
        for obj in self.objects:
            if obj.type == 'goal':
                continue
            w_verts = obj.get_world_verts()
            c_verts = self.renderer.world_to_cam(w_verts, cx, self.cam_height, cz, self.yaw)
            if np.all(c_verts[:, 2] < 1.0): continue 
            s_verts = self.renderer.project(c_verts)
            for indices, shade in obj.faces:
                f_3d = c_verts[indices]
                f_2d = s_verts[indices]
                if np.min(f_3d[:, 2]) < 0.5: continue
                v1 = f_2d[1] - f_2d[0]
                v2 = f_2d[2] - f_2d[1]
                cross = v1[0]*v2[1] - v1[1]*v2[0]
                if cross > 0:
                    depth = np.mean(f_3d[:, 2])
                    color = tuple([int(c * shade) for c in obj.c])
                    color = tuple([min(255, max(0, x)) for x in color])
                    render_queue.append((depth, f_2d.astype(int), color))
        
        render_queue.sort(key=lambda x: x[0], reverse=True)
        for _, pts, color in render_queue:
            cv2.fillPoly(canvas, [pts], color)
            cv2.polylines(canvas, [pts], True, (0,0,0), 1)
        return canvas

# ==========================================
# WORKER FUNCTION (One Episode)
# ==========================================
def generate_episode(ep_id):
    attempt = 0
    while True:
        attempt += 1
        np.random.seed(ep_id + attempt * 10000) 
        cv2.setNumThreads(0)
        
        rand_val = np.random.rand()
        if rand_val < 0.20:
            spawn_type = 'uturn'
        elif rand_val < 0.40:
            spawn_type = 'recovery'
        elif rand_val < 0.60:
            spawn_type = 'wall'
        elif rand_val < 0.80:
            spawn_type = 'dense'
        else:
            spawn_type = 'normal'
            
        sim = RoverSim(spawn_type=spawn_type)
        
        base_name = f"{spawn_type}_{ep_id}"
        vid_path = os.path.join(OUTPUT_DIR, f"{base_name}.mp4")
        csv_path = os.path.join(OUTPUT_DIR, f"{base_name}.csv")
        
        out = cv2.VideoWriter(vid_path, cv2.VideoWriter_fourcc(*'mp4v'), 30.0, (IMG_W, IMG_H))
        
        data = []
        DT = 0.1
        recovery_frames = 30 if spawn_type == 'recovery' else 0
        
        current_steer = 0.0 
        collision_detected = False
        
        for f in range(FRAMES_PER_EP):
            dx = sim.goal.x - sim.x
            dz = sim.goal.z - sim.z
            dist_to_goal = math.sqrt(dx**2 + dz**2)
            
            target_steer = 0.0
            
            if recovery_frames > 0:
                throttle = -1.0 
                speed = 10.0 * throttle
                target_steer = 1.0 if (ep_id % 2 == 0) else -1.0 
                recovery_frames -= 1
            else:
                goal_yaw = math.degrees(math.atan2(dx, dz))
                bearing = (goal_yaw - sim.yaw + 180) % 360 - 180
                target_steer = np.clip(bearing / 30.0, -1.0, 1.0)
                
                c_rad = math.radians(sim.yaw)
                avoidance_force = 0.0
                total_repulsion = 0.0 # Track total danger regardless of steering cancellation
                
                for obj in sim.objects:
                    if obj.type == 'goal': continue
                    ox, oz = obj.x - sim.x, obj.z - sim.z
                    lx = ox * math.cos(c_rad) - oz * math.sin(c_rad) 
                    lz = ox * math.sin(c_rad) + oz * math.cos(c_rad) 
                    
                    if 0 < lz < 25 and abs(lx) < 8.0:
                        z_factor = (25.0 - lz) / 25.0 
                        x_factor = (8.0 - abs(lx)) / 8.0
                        
                        repulsion = (z_factor ** 1.5) * x_factor * 3.0
                        total_repulsion += repulsion
                        
                        direction = -1.0 if lx > 0 else 1.0
                        avoidance_force += direction * repulsion
                
                target_steer += avoidance_force
                target_steer = np.clip(target_steer, -1.0, 1.0)

                if dist_to_goal < 30.0:
                    throttle = max(0.2, dist_to_goal / 30.0)
                else:
                    throttle = 1.0
                    
                # Braking: Use total_repulsion so the rover stops even if forces perfectly cancel
                if total_repulsion > 0.5:
                    throttle *= 0.5 
                    
                speed = 10.0 * throttle
            
            current_steer += 0.2 * (target_steer - current_steer)
            
            # --- UPDATED COLLISION DETECTION ---
            closest_danger = 999.0
            for obj in sim.objects:
                if obj.type == 'goal': continue
                g_dist = math.sqrt((obj.x - sim.x)**2 + (obj.z - sim.z)**2)
                
                # Account for physical bounding box size (Rocks: ~2.4m radius, Trees: ~1.0m radius)
                col_radius = 2.4 if obj.type == 'rock' else 1.0
                
                # If we get inside the bounding box, trigger collision
                if g_dist < col_radius and recovery_frames == 0:
                    collision_detected = True
                    break
                    
                # Calculate edge distance for trav_score
                eff_dist = max(0.0, g_dist - col_radius)
                if eff_dist < closest_danger:
                    closest_danger = eff_dist
            
            if collision_detected:
                break
                
            safe_dist = 8.0
            danger_dist = 0.0
            trav_score = np.clip((closest_danger - danger_dist) / (safe_dist - danger_dist), 0.0, 1.0)
            
            final_steer = current_steer + np.random.normal(0, 0.01)
            final_steer = np.clip(final_steer, -1.0, 1.0)
            
            sim.step(final_steer * 30.0, speed)
            frame = sim.render()
            out.write(frame)
            
            # --- CONVERT METERS TO LAT/LON BEFORE LOGGING ---
            current_lat, current_lon = meters_to_latlon(sim.x, sim.z, ORIGIN_LAT, ORIGIN_LON)
            goal_lat, goal_lon = meters_to_latlon(sim.goal.x, sim.goal.z, ORIGIN_LAT, ORIGIN_LON)
            
            row = {
                'timestamp_ms': int(f * DT * 1000),
                'lat': current_lat, 'lon': current_lon,
                'goal_lat': goal_lat, 'goal_lon': goal_lon,
                'throttle': float(throttle), 'steer': float(final_steer),
                'heading': sim.yaw, 'speed': float(speed),
                'altitude': 0.0, 'trav_score': float(trav_score)
            }
            data.append(row)
            
            if dist_to_goal < 5.0: break
                
        out.release()
        
        if collision_detected:
            if os.path.exists(vid_path):
                os.remove(vid_path)
            continue
            
        df = pd.DataFrame(data)
        cols = ['timestamp_ms', 'lat', 'lon', 'goal_lat', 'goal_lon', 
                'throttle', 'steer', 'heading', 'speed', 'altitude', 'trav_score']
        df = df[cols]
        df.to_csv(csv_path, index=False)
        return ep_id

# ==========================================
# MAIN (Multiprocessing Setup)
# ==========================================
def main():
    if not os.path.exists(OUTPUT_DIR): os.makedirs(OUTPUT_DIR)
    
    num_workers = cpu_count()
    print(f"Starting generation of {NUM_EPISODES} episodes on {num_workers} cores...")
    
    with Pool(processes=num_workers) as pool:
        results = list(tqdm(pool.imap_unordered(generate_episode, range(NUM_EPISODES)), total=NUM_EPISODES))
    
    print("Done.")

if __name__ == "__main__":
    main()