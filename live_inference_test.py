import argparse
import os
import cv2
import numpy as np
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.v2 as v2 
import torchvision.models as models

# ==========================================
# CONFIGURATION
# ==========================================
CONFIG = {
    # Model dimensions
    'seq_len': 20,          
    'action_horizon': 10,    
    'img_size': 224,        
    'embed_dim': 384,       
    'hidden_dim': 512,      
    'proj_dim': 64,         
    'n_heads': 8,            
    'n_layers': 4,          
    'fusion_dropout_p': 0.2,  
    'num_experts': 3,
    'invert_heading': False,  
    
    # World Generation
    'img_w': 224,
    'img_h': 224,
    'fov': 90.0,
    'c_sky': (230, 200, 150),
    'c_ground': (60, 50, 40),
    'c_rock': (50, 50, 200),
    'c_tree': (50, 200, 50),
    'c_goal': (0, 255, 255)
}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

gpu_val_transform = torch.nn.Sequential(
    v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
).to(device)

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
        self.width = width
        self.height = height
        self.depth = depth
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
# INFINITE SIMULATOR WITH DYNAMIC BIOMES
# ==========================================
class InfiniteRoverSim:
    def __init__(self):
        self.renderer = RenderEngine(CONFIG['img_w'], CONFIG['img_h'], CONFIG['fov'])
        self.cam_height = 1.5
        self.cam_offset_forward = 0.5 
        self.x = 0.0
        self.z = 0.0
        self.yaw = 0.0 
        self.last_gen_x = 0.0
        self.last_gen_z = 0.0
        self.objects = []
        
        self._spawn_goal(400)
        self._generate_landscape()

    def _spawn_goal(self, distance):
        angle = self.yaw + np.random.uniform(-45, 45)
        gx = self.x + math.sin(math.radians(angle)) * distance
        gz = self.z + math.cos(math.radians(angle)) * distance
        self.goal = WorldObject(gx, gz, 4, 40, 4, CONFIG['c_goal'], 'goal')
        
        # Ensure goal replaces the old one and remains in objects list
        self.objects = [obj for obj in self.objects if obj.type != 'goal']
        self.objects.insert(0, self.goal)

    def _generate_landscape(self):
        """Generates dynamic biomes mimicking the training data scenarios."""
        self.last_gen_x = self.x
        self.last_gen_z = self.z
        
        # Cull old objects to prevent memory bloat, but keep very close ones to avoid visual popping
        alive_objects = [self.goal]
        for obj in self.objects:
            if obj.type == 'goal': continue
            if math.hypot(obj.x - self.x, obj.z - self.z) < 30.0:
                alive_objects.append(obj)
        self.objects = alive_objects

        # Randomly select the biome (Landscape Type)
        rand_val = np.random.rand()
        if rand_val < 0.33:
            spawn_type = 'wall'
        elif rand_val < 0.66:
            spawn_type = 'dense'
        else:
            spawn_type = 'normal'

        print(f"\n>>> Generating New Chunk Biome: {spawn_type.upper()} <<<")

        c_rad = math.radians(self.yaw)
        cos_y = math.cos(c_rad)
        sin_y = math.sin(c_rad)

        def add_obj(local_x, local_z, w, h, d, color, otype):
            # Prevent spawning directly on top of the rover
            if local_z < 15 and abs(local_x) < 5:
                return 
            # Transform local to world coordinates
            wx = self.x + local_x * cos_y + local_z * sin_y
            wz = self.z - local_x * sin_y + local_z * cos_y
            self.objects.append(WorldObject(wx, wz, w, h, d, color, otype))

        if spawn_type == 'wall':
            # Spawn multiple layers of walls with gaps along the chunk path
            for wall_z in [50, 150, 250, 350]:
                gap_center = np.random.uniform(-15, 15)
                for ox in range(-40, 41, 4):
                    if abs(ox - gap_center) < 8.0: 
                        continue # Leave a gap for the rover to pass
                    noisy_ox = ox + np.random.uniform(-0.5, 0.5)
                    add_obj(noisy_ox, wall_z + np.random.uniform(-1, 1), 4.0, 1.5, 4.0, CONFIG['c_rock'], 'rock')
            
            # Scatter a few standard rocks between the walls
            for _ in range(150):
                ox = np.random.uniform(-100, 100)
                oz = np.random.uniform(10, 400)
                add_obj(ox, oz, 4.0, 1.5, 4.0, CONFIG['c_rock'], 'rock')

        elif spawn_type == 'dense':
            # Tight corridor of dense trees
            for _ in range(500):
                ox = np.random.uniform(-40, 40)
                oz = np.random.uniform(20, 400)
                add_obj(ox, oz, 1.0, 10.0, 1.0, CONFIG['c_tree'], 'tree')
                
            # Throw in some rocks into the dense forest
            for _ in range(50):
                ox = np.random.uniform(-40, 40)
                oz = np.random.uniform(20, 400)
                add_obj(ox, oz, 4.0, 1.5, 4.0, CONFIG['c_rock'], 'rock')

        else: # normal
            # Standard wide dispersal of mixed obstacles
            for _ in range(400):
                ox = np.random.uniform(-150, 150)
                oz = np.random.uniform(10, 400)
                if np.random.rand() > 0.5:
                    add_obj(ox, oz, 4.0, 1.5, 4.0, CONFIG['c_rock'], 'rock')
                else:
                    add_obj(ox, oz, 1.0, 10.0, 1.0, CONFIG['c_tree'], 'tree')

    def manage_world_chunks(self):
        dist_from_last_gen = math.hypot(self.x - self.last_gen_x, self.z - self.last_gen_z)
        dist_to_goal = math.hypot(self.goal.x - self.x, self.goal.z - self.z)
        
        # Trigger next chunk if goal is reached OR if the rover drifted far out of the populated zone
        if dist_to_goal < 15.0:
            print(">>> GOAL REACHED! Spawning new objective... <<<")
            self._spawn_goal(400)
            self._generate_landscape()
        elif dist_from_last_gen > 300.0:
            print(">>> ROVER LEFT CHUNK! Regenerating environment... <<<")
            self._generate_landscape()
    
    # Collision assumes rover is a square, obstacles are rectangles
    def is_colliding(self, length=1):
        for obj in self.objects:
            if obj.type == 'goal':
                continue
                
            relative_x = abs(obj.x - self.x)
            relative_y = abs(obj.z - self.z)
            if relative_x <= (obj.width / 2) + length / 2 and relative_y <= (obj.depth / 2) + length / 2:
                return True
            
        return False

    def step(self, angular_velocity, linear_velocity):
        dt = 0.1
        self.yaw += angular_velocity * dt
        rad = math.radians(self.yaw)
        self.x += math.sin(rad) * linear_velocity * dt
        self.z += math.cos(rad) * linear_velocity * dt

    def render(self):
        canvas = np.zeros((CONFIG['img_h'], CONFIG['img_w'], 3), dtype=np.uint8)
        canvas[:] = CONFIG['c_sky']
        
        c_rad = math.radians(self.yaw)
        cx = self.x + math.sin(c_rad) * self.cam_offset_forward
        cz = self.z + math.cos(c_rad) * self.cam_offset_forward
        
        lines_to_draw = []
        grid_sz = 100
        step = 10
        bx = int(self.x // step) * step
        bz = int(self.z // step) * step
        
        for i in range(-grid_sz, grid_sz+1, step):
            p1 = np.array([bx + i, 0, bz - grid_sz]); p2 = np.array([bx + i, 0, bz + grid_sz])
            p3 = np.array([bx - grid_sz, 0, bz + i]); p4 = np.array([bx + grid_sz, 0, bz + i])
            lines_to_draw.extend([(p1, p2), (p3, p4)])
            
        for p1, p2 in lines_to_draw:
            pts = np.vstack([p1, p2])
            c_pts = self.renderer.world_to_cam(pts, cx, self.cam_height, cz, self.yaw)
            if c_pts[0,2] < 1.0 and c_pts[1,2] < 1.0: continue
            if c_pts[0,2] < 1.0: c_pts[0] += ((1.0 - c_pts[0,2]) / (c_pts[1,2] - c_pts[0,2])) * (c_pts[1] - c_pts[0])
            elif c_pts[1,2] < 1.0: c_pts[1] += ((1.0 - c_pts[1,2]) / (c_pts[0,2] - c_pts[1,2])) * (c_pts[0] - c_pts[1])
            s_pts = self.renderer.project(c_pts)
            cv2.line(canvas, tuple(s_pts[0].astype(int)), tuple(s_pts[1].astype(int)), CONFIG['c_ground'], 1)

        render_queue = []
        for obj in self.objects:
            if obj.type == 'goal':
                continue
            w_verts = obj.get_world_verts()
            c_verts = self.renderer.world_to_cam(w_verts, cx, self.cam_height, cz, self.yaw)
            if np.all(c_verts[:, 2] < 1.0): continue 
            s_verts = self.renderer.project(c_verts)
            for indices, shade in obj.faces:
                f_3d, f_2d = c_verts[indices], s_verts[indices]
                if np.min(f_3d[:, 2]) < 0.5: continue
                v1, v2 = f_2d[1] - f_2d[0], f_2d[2] - f_2d[1]
                if (v1[0]*v2[1] - v1[1]*v2[0]) > 0:
                    depth = np.mean(f_3d[:, 2])
                    color = tuple([min(255, max(0, int(c * shade))) for c in obj.c])
                    render_queue.append((depth, f_2d.astype(int), color))
        
        render_queue.sort(key=lambda x: x[0], reverse=True)
        for _, pts, color in render_queue:
            cv2.fillPoly(canvas, [pts], color)
            cv2.polylines(canvas, [pts], True, (0,0,0), 1)
        return canvas

    def get_reference_autopilot(self):
        dx = self.goal.x - self.x
        dz = self.goal.z - self.z
        
        goal_yaw = math.degrees(math.atan2(dx, dz))
        bearing = (goal_yaw - self.yaw + 180) % 360 - 180
        target_steer = np.clip(bearing / 30.0, -1.0, 1.0)
        
        c_rad = math.radians(self.yaw)
        avoidance_force = 0.0
        
        for obj in self.objects:
            if obj.type == 'goal': continue
            ox, oz = obj.x - self.x, obj.z - self.z
            lx = ox * math.cos(c_rad) - oz * math.sin(c_rad) 
            lz = ox * math.sin(c_rad) + oz * math.cos(c_rad) 
            
            if 0 < lz < 25 and abs(lx) < 8.0:
                z_factor = (25.0 - lz) / 25.0 
                x_factor = (8.0 - abs(lx)) / 8.0
                repulsion = (z_factor ** 1.5) * x_factor * 3.0
                direction = -1.0 if lx > 0 else 1.0
                avoidance_force += direction * repulsion
        
        target_steer = np.clip(target_steer + avoidance_force, -1.0, 1.0)
        return 1.0, target_steer # Throttle, Steer

class RunBasedRoverSim:
    def __init__(self):
        self.renderer = RenderEngine(CONFIG['img_w'], CONFIG['img_h'], CONFIG['fov'])
        self.cam_height = 1.5
        self.cam_offset_forward = 0.5 
        self.x = 0.0
        self.z = 0.0
        self.yaw = 0.0 
        self.run_number = 0
        self.run_steps = 0
        self.goal_distance = 175
        self.objects = []
        self.metrics = {
            "total_runs": 0,
            "reached_goal": 0,
            "total_goal_distance": 0,
            "distance_traveled": 0,
            "total_steps": 0,
            "steps_spent_colliding": 0,
            "total_throttle": 0,
            "mean_throttle": 0,
            "total_danger": 0,
            "mean_danger": 0,
            "total_jitter": 0,
            "mean_jitter": 0,
            "total_alignment": 0,
            "mean_alignment": 0,
            "total_path_efficiency": 0,
            "mean_path_efficiency": 0
        }
    
    def reset_run(self):
        self.cam_height = 1.5
        self.cam_offset_forward = 0.5 
        self.x = 0.0
        self.z = 0.0
        self.yaw = 0.0 
        self.last_gen_x = 0.0
        self.last_gen_z = 0.0
        self.run_steps = 0
        self.objects = []
        
        self.run_number += 1
        self.metrics["total_runs"] += 1
        self._spawn_goal(self.goal_distance)
        self.metrics["total_goal_distance"] += self.goal_distance
        self._generate_landscape()


    def _spawn_goal(self, distance):
        angle = self.yaw + np.random.uniform(-45, 45)
        gx = self.x + math.sin(math.radians(angle)) * distance
        gz = self.z + math.cos(math.radians(angle)) * distance
        self.goal = WorldObject(gx, gz, 4, 40, 4, CONFIG['c_goal'], 'goal')
        
        # Ensure goal replaces the old one and remains in objects list
        self.objects = [obj for obj in self.objects if obj.type != 'goal']
        self.objects.insert(0, self.goal)

    def _generate_landscape(self):
        """Generates dynamic biomes mimicking the training data scenarios."""
        self.last_gen_x = self.x
        self.last_gen_z = self.z
        
        # Cull old objects to prevent memory bloat, but keep very close ones to avoid visual popping
        alive_objects = [self.goal]
        for obj in self.objects:
            if obj.type == 'goal': continue
            if math.hypot(obj.x - self.x, obj.z - self.z) < 30.0:
                alive_objects.append(obj)
        self.objects = alive_objects

        # Randomly select the biome (Landscape Type)
        rand_val = np.random.rand()
        if rand_val < 0.33:
            spawn_type = 'wall'
        elif rand_val < 0.66:
            spawn_type = 'dense'
        else:
            spawn_type = 'normal'

        print(f"\n>>> Generating New Chunk Biome: {spawn_type.upper()} <<<")

        c_rad = math.radians(self.yaw)
        cos_y = math.cos(c_rad)
        sin_y = math.sin(c_rad)

        def add_obj(local_x, local_z, w, h, d, color, otype):
            # Prevent spawning directly on top of the rover
            if local_z < 15 and abs(local_x) < 5:
                return 
            # Transform local to world coordinates
            wx = self.x + local_x * cos_y + local_z * sin_y
            wz = self.z - local_x * sin_y + local_z * cos_y
            self.objects.append(WorldObject(wx, wz, w, h, d, color, otype))

        if spawn_type == 'wall':
            # Spawn multiple layers of walls with gaps along the chunk path
            for wall_z in [50, 150, 250, 350]:
                gap_center = np.random.uniform(-15, 15)
                for ox in range(-40, 41, 4):
                    if abs(ox - gap_center) < 8.0: 
                        continue # Leave a gap for the rover to pass
                    noisy_ox = ox + np.random.uniform(-0.5, 0.5)
                    add_obj(noisy_ox, wall_z + np.random.uniform(-1, 1), 4.0, 1.5, 4.0, CONFIG['c_rock'], 'rock')
            
            # Scatter a few standard rocks between the walls
            for _ in range(150):
                ox = np.random.uniform(-100, 100)
                oz = np.random.uniform(10, 400)
                add_obj(ox, oz, 4.0, 1.5, 4.0, CONFIG['c_rock'], 'rock')

        elif spawn_type == 'dense':
            # Tight corridor of dense trees
            for _ in range(500):
                ox = np.random.uniform(-40, 40)
                oz = np.random.uniform(20, 400)
                add_obj(ox, oz, 1.0, 10.0, 1.0, CONFIG['c_tree'], 'tree')
                
            # Throw in some rocks into the dense forest
            for _ in range(50):
                ox = np.random.uniform(-40, 40)
                oz = np.random.uniform(20, 400)
                add_obj(ox, oz, 4.0, 1.5, 4.0, CONFIG['c_rock'], 'rock')

        else: # normal
            # Standard wide dispersal of mixed obstacles
            for _ in range(400):
                ox = np.random.uniform(-150, 150)
                oz = np.random.uniform(10, 400)
                if np.random.rand() > 0.5:
                    add_obj(ox, oz, 4.0, 1.5, 4.0, CONFIG['c_rock'], 'rock')
                else:
                    add_obj(ox, oz, 1.0, 10.0, 1.0, CONFIG['c_tree'], 'tree')
    
    def manage_world_chunks(self):
        dist_from_last_gen = math.hypot(self.x - self.last_gen_x, self.z - self.last_gen_z)
        if dist_from_last_gen > 275.0:
            print(">>> ROVER LEFT CHUNK! Regenerating environment... <<<")
            self._generate_landscape()

    # Returns True when run should be terminated
    def check_termination_conditions(self):
        dist_to_goal = math.hypot(self.goal.x - self.x, self.goal.z - self.z)
        if self.run_steps > 500:
            print(">>> TOTAL TIME EXCEEDED. Starting new run...<<<")
            return True
        elif dist_to_goal < 15.0:
            self.metrics["reached_goal"] += 1
            # measure how efficient the path was (full efficiency may not be good if obstacles block path)
            shortest_path_distance = math.sqrt(self.goal.x ** 2 + self.goal.z ** 2)

            path_efficiency = shortest_path_distance / self.metrics["distance_traveled"]
            self.metrics["total_path_efficiency"] += path_efficiency

            # Mean of completed runs, not total
            self.metrics["mean_path_efficiency"] += self.metrics["total_path_erfficiency"] / self.metrics["reached_goal"] if self.metrics["reached_goal"] > 0 else 0
            print(">>> GOAL REACHED!  Starting new run... <<<")
            return True
        return False
    
    # Collision assumes rover is a square, obstacles are rectangles
    def is_colliding(self, length=1):
        for obj in self.objects:
            if obj.type == 'goal':
                continue
                
            relative_x = abs(obj.x - self.x)
            relative_y = abs(obj.z - self.z)
            if relative_x <= (obj.width / 2) + length / 2 and relative_y <= (obj.depth / 2) + length / 2:
                return True
            
        return False

    def step(self, angular_velocity, linear_velocity):
        dt = 0.1
        self.yaw += angular_velocity * dt
        rad = math.radians(self.yaw)
        dx = math.sin(rad) * linear_velocity * dt
        dz = math.cos(rad) * linear_velocity * dt
        self.x += dx
        self.z += dz

        distance_traveled = math.sqrt(dx**2 + dz**2)
        self.run_steps += 1
        self.metrics["total_steps"] += 1
        self.metrics["distance_traveled"] += distance_traveled

        # measure alignment
        vec_to_goal_x = self.goal.x - self.x
        vec_to_goal_z = self.goal.z - self.z
        dist_to_goal = math.sqrt(vec_to_goal_x**2 + vec_to_goal_z**2)
        if dist_to_goal > 0 and distance_traveled > 0:
            alignment = (dx * vec_to_goal_x + dz * vec_to_goal_z) / (distance_traveled * dist_to_goal)

            self.metrics["total_alignment"] += alignment
            self.metrics["mean_alignment"] = self.metrics["total_alignment"] / self.metrics["total_steps"]
        
        


    def render(self):
        canvas = np.zeros((CONFIG['img_h'], CONFIG['img_w'], 3), dtype=np.uint8)
        canvas[:] = CONFIG['c_sky']
        
        c_rad = math.radians(self.yaw)
        cx = self.x + math.sin(c_rad) * self.cam_offset_forward
        cz = self.z + math.cos(c_rad) * self.cam_offset_forward
        
        lines_to_draw = []
        grid_sz = 100
        step = 10
        bx = int(self.x // step) * step
        bz = int(self.z // step) * step
        
        for i in range(-grid_sz, grid_sz+1, step):
            p1 = np.array([bx + i, 0, bz - grid_sz]); p2 = np.array([bx + i, 0, bz + grid_sz])
            p3 = np.array([bx - grid_sz, 0, bz + i]); p4 = np.array([bx + grid_sz, 0, bz + i])
            lines_to_draw.extend([(p1, p2), (p3, p4)])
            
        for p1, p2 in lines_to_draw:
            pts = np.vstack([p1, p2])
            c_pts = self.renderer.world_to_cam(pts, cx, self.cam_height, cz, self.yaw)
            if c_pts[0,2] < 1.0 and c_pts[1,2] < 1.0: continue
            if c_pts[0,2] < 1.0: c_pts[0] += ((1.0 - c_pts[0,2]) / (c_pts[1,2] - c_pts[0,2])) * (c_pts[1] - c_pts[0])
            elif c_pts[1,2] < 1.0: c_pts[1] += ((1.0 - c_pts[1,2]) / (c_pts[0,2] - c_pts[1,2])) * (c_pts[0] - c_pts[1])
            s_pts = self.renderer.project(c_pts)
            cv2.line(canvas, tuple(s_pts[0].astype(int)), tuple(s_pts[1].astype(int)), CONFIG['c_ground'], 1)

        render_queue = []
        for obj in self.objects:
            if obj.type == 'goal':
                continue
            w_verts = obj.get_world_verts()
            c_verts = self.renderer.world_to_cam(w_verts, cx, self.cam_height, cz, self.yaw)
            if np.all(c_verts[:, 2] < 1.0): continue 
            s_verts = self.renderer.project(c_verts)
            for indices, shade in obj.faces:
                f_3d, f_2d = c_verts[indices], s_verts[indices]
                if np.min(f_3d[:, 2]) < 0.5: continue
                v1, v2 = f_2d[1] - f_2d[0], f_2d[2] - f_2d[1]
                if (v1[0]*v2[1] - v1[1]*v2[0]) > 0:
                    depth = np.mean(f_3d[:, 2])
                    color = tuple([min(255, max(0, int(c * shade))) for c in obj.c])
                    render_queue.append((depth, f_2d.astype(int), color))
        
        render_queue.sort(key=lambda x: x[0], reverse=True)
        for _, pts, color in render_queue:
            cv2.fillPoly(canvas, [pts], color)
            cv2.polylines(canvas, [pts], True, (0,0,0), 1)
        return canvas

    def get_reference_autopilot(self):
        dx = self.goal.x - self.x
        dz = self.goal.z - self.z
        
        goal_yaw = math.degrees(math.atan2(dx, dz))
        bearing = (goal_yaw - self.yaw + 180) % 360 - 180
        target_steer = np.clip(bearing / 30.0, -1.0, 1.0)
        
        c_rad = math.radians(self.yaw)
        avoidance_force = 0.0
        
        for obj in self.objects:
            if obj.type == 'goal': continue
            ox, oz = obj.x - self.x, obj.z - self.z
            lx = ox * math.cos(c_rad) - oz * math.sin(c_rad) 
            lz = ox * math.sin(c_rad) + oz * math.cos(c_rad) 
            
            if 0 < lz < 25 and abs(lx) < 8.0:
                z_factor = (25.0 - lz) / 25.0 
                x_factor = (8.0 - abs(lx)) / 8.0
                repulsion = (z_factor ** 1.5) * x_factor * 3.0
                direction = -1.0 if lx > 0 else 1.0
                avoidance_force += direction * repulsion
        
        target_steer = np.clip(target_steer + avoidance_force, -1.0, 1.0)
        return 1.0, target_steer # Throttle, Steer

# ==========================================
# MODEL ARCHITECTURE (Fully Synced with Training)
# ==========================================
class RoverJEPA_v2_Transformer(nn.Module):
    def __init__(self):
        super().__init__()
        
        print("Loading DINOv2 (ViT-S/14)...")
        self.backbone = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14', source='github')
        
        for param in self.backbone.parameters():
            param.requires_grad = False
            
        for param in self.backbone.blocks[-1].parameters():
            param.requires_grad = True
        for param in self.backbone.norm.parameters():
            param.requires_grad = True
            
        self.embed_dim = CONFIG['embed_dim']
        self.hidden_dim = CONFIG['hidden_dim']
        
        self.input_proj = nn.Linear(self.embed_dim, self.hidden_dim)
        self.pos_embed = nn.Parameter(torch.zeros(1, CONFIG['seq_len'], self.hidden_dim))
        
        self.mask_token = nn.Parameter(torch.randn(1, 1, self.hidden_dim) * 0.02)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.hidden_dim, 
            nhead=CONFIG['n_heads'], 
            dim_feedforward=self.hidden_dim*4, 
            dropout=0.1, 
            batch_first=True,
            norm_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=CONFIG['n_layers'])
        
        self.predictor = nn.Sequential(
            nn.Linear(self.hidden_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, CONFIG['proj_dim'])
        )
        
        self.target_projector = nn.Sequential(
            nn.Linear(self.embed_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, CONFIG['proj_dim'])
        )

        self.fusion_dropout = nn.Dropout(p=CONFIG['fusion_dropout_p'])

        self.num_experts = CONFIG['num_experts']
        fusion_dim = self.hidden_dim + 3
        
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(fusion_dim, 256),
                nn.ReLU(),
                nn.Linear(256, CONFIG['action_horizon'] * 2)
            ) for _ in range(self.num_experts)
        ])
        
        self.router = nn.Sequential(
            nn.Linear(fusion_dim, 128),
            nn.ReLU(),
            nn.Linear(128, self.num_experts)
        )
        
        self.critic = nn.Sequential(
            nn.Linear(self.hidden_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def encode(self, x):
        return self.backbone(x)

    def forward_sequence(self, images):
        B, S, C, H, W = images.shape
        flat_imgs = images.view(B*S, C, H, W)
        feats = self.backbone(flat_imgs)
        feats = feats.view(B, S, -1)
            
        x = self.input_proj(feats) 
        x = x + self.pos_embed[:, :S, :]
        
        if self.training:
            mask = torch.rand(B, S, 1, device=x.device) < 0.15
            mask[:, 0, :] = False 
            x = torch.where(mask, self.mask_token, x)
        
        attn_mask = nn.Transformer.generate_square_subsequent_mask(S, device=x.device)
        latent_seq = self.transformer(x, mask=attn_mask, is_causal=True)
        return latent_seq, feats

    def get_jepa_projections(self, latents_pred, feats_target):
        pred_proj = self.predictor(latents_pred)
        target_proj = self.target_projector(feats_target)
        return pred_proj, target_proj

    def get_action_heads(self, latents, context_sequence):
        safety_logits = self.critic(latents)
        danger_prob = torch.sigmoid(safety_logits)
        
        policy_input = latents
        danger_input = danger_prob.detach() 
        
        fusion_input = torch.cat([self.fusion_dropout(policy_input), context_sequence, danger_input], dim=1)
        
        routing_logits = self.router(fusion_input)
        
        if self.training:
            routing_logits = routing_logits + torch.randn_like(routing_logits) * 0.1
            
        routing_weights = F.softmax(routing_logits, dim=-1)
        
        expert_outputs = []
        for expert in self.experts:
            expert_outputs.append(expert(fusion_input).view(latents.shape[0], CONFIG['action_horizon'], 2))
            
        expert_outputs = torch.stack(expert_outputs, dim=1) 
        
        routing_weights_exp = routing_weights.view(latents.shape[0], self.num_experts, 1, 1)
        action_chunks = torch.sum(expert_outputs * routing_weights_exp, dim=1) 
        
        return action_chunks, safety_logits, routing_weights

class RoverJEPA_v2_LSTM(nn.Module):
    """
    Main Model Architecture combining DINOv2 (Vision), an LSTM (Temporal),
    and a Mixture of Experts (Action/Policy routing).

    This version preserves the original class structure as much as possible,
    replacing only the Transformer temporal encoder with an LSTM.
    """
    def __init__(self):
        super().__init__()
        
        print("Loading DINOv2 (ViT-S/14)...")
        self.backbone = torch.hub.load(
            'facebookresearch/dinov2',
            'dinov2_vits14',
            source='github'
        )
        
        # Freeze the majority of the DINOv2 backbone to prevent catastrophic forgetting
        for param in self.backbone.parameters():
            param.requires_grad = False
            
        # Unfreeze only the last attention block and normalization layers for fine-tuning
        for param in self.backbone.blocks[-1].parameters():
            param.requires_grad = True
        for param in self.backbone.norm.parameters():
            param.requires_grad = True
            
        self.embed_dim = CONFIG['embed_dim']
        self.hidden_dim = CONFIG['hidden_dim']
        
        # Temporal Modeling Components
        self.input_proj = nn.Linear(self.embed_dim, self.hidden_dim)
        self.pos_embed = nn.Parameter(torch.zeros(1, CONFIG['seq_len'], self.hidden_dim))
        
        # JEPA Mask Token (replaces masked frames during training)
        self.mask_token = nn.Parameter(torch.randn(1, 1, self.hidden_dim) * 0.02)
        
        # Replace Transformer with LSTM
        # batch_first=True keeps input/output shape as (B, S, hidden_dim)
        self.lstm = nn.LSTM(
            input_size=self.hidden_dim,
            hidden_size=self.hidden_dim,
            num_layers=CONFIG['n_layers'],
            batch_first=True,
            dropout=0.1 if CONFIG['n_layers'] > 1 else 0.0
        )
        
        # JEPA Projections (Maps latents and targets to the same space for VICReg loss)
        self.predictor = nn.Sequential(
            nn.Linear(self.hidden_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, CONFIG['proj_dim'])
        )
        self.target_projector = nn.Sequential(
            nn.Linear(self.embed_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, CONFIG['proj_dim'])
        )

        # Policy & Routing Components
        self.fusion_dropout = nn.Dropout(p=CONFIG['fusion_dropout_p'])
        self.num_experts = CONFIG['num_experts']
        
        # Fusion dimensionality: Latent state (hidden_dim) + Context (2) + Danger signal (1)
        fusion_dim = self.hidden_dim + 3
        
        # Multiple action experts
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(fusion_dim, 256),
                nn.ReLU(),
                nn.Linear(256, CONFIG['action_horizon'] * 2)
            ) for _ in range(self.num_experts)
        ])
        
        # The router decides which expert gets to act based on the current state
        self.router = nn.Sequential(
            nn.Linear(fusion_dim, 128),
            nn.ReLU(),
            nn.Linear(128, self.num_experts)
        )
        
        # Critic network to predict if the rover is currently "stuck" or in "danger"
        self.critic = nn.Sequential(
            nn.Linear(self.hidden_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def encode(self, x):
        return self.backbone(x)
    
    def forward_from_features(self, feats):
        """Processes pre-extracted features through the temporal and policy layers."""
        # feats shape: (B, S, embed_dim)
        x = self.input_proj(feats)
        x = x + self.pos_embed[:, :feats.size(1), :]
        
        # LSTM or Transformer processing
        latent_seq, _ = self.lstm(x) 
        return latent_seq

    def forward_sequence(self, images):
        """
        Processes a sequence of images through the backbone and temporal LSTM.
        Applies random masking during training for the JEPA objective.
        """
        B, S, C, H, W = images.shape
        flat_imgs = images.view(B * S, C, H, W)
        
        # Extract features using DINOv2
        feats = self.backbone(flat_imgs)
        feats = feats.view(B, S, -1)
            
        x = self.input_proj(feats)
        x = x + self.pos_embed[:, :S, :]
        
        # Self-supervised objective: Mask random frames (except the current frame)
        if self.training:
            mask = torch.rand(B, S, 1, device=x.device) < 0.15
            mask[:, 0, :] = False
            x = torch.where(mask, self.mask_token.expand(B, S, -1), x)
        
        # LSTM processes sequence causally by construction
        latent_seq, _ = self.lstm(x)
        
        return latent_seq, feats

    def get_jepa_projections(self, latents_pred, feats_target):
        """Projects LSTM latents and target backbone features for self-supervised loss."""
        pred_proj = self.predictor(latents_pred)
        target_proj = self.target_projector(feats_target)
        return pred_proj, target_proj

    def get_action_heads(self, latents, context_sequence):
        """
        Executes the Mixture of Experts (MoE) action head.
        Fuses vision latents with telemetry context and safety predictions.
        """
        # 1. Predict Danger / Stuck state
        safety_logits = self.critic(latents)
        danger_prob = torch.sigmoid(safety_logits)
        
        policy_input = latents
        
        # Detach danger input so action loss doesn't falsely train the safety critic
        danger_input = danger_prob.detach()
        
        # 2. Fuse all information
        fusion_input = torch.cat(
            [self.fusion_dropout(policy_input), context_sequence, danger_input],
            dim=1
        )
        
        # 3. Route to Experts
        routing_logits = self.router(fusion_input)
        
        # Add Gumbel-like noise during training to encourage expert exploration
        if self.training:
            routing_logits = routing_logits + torch.randn_like(routing_logits) * 0.1
            
        routing_weights = F.softmax(routing_logits, dim=-1)
        
        # Calculate actions from all experts
        expert_outputs = []
        for expert in self.experts:
            expert_outputs.append(
                expert(fusion_input).view(latents.shape[0], CONFIG['action_horizon'], 2)
            )
            
        expert_outputs = torch.stack(expert_outputs, dim=1)
        
        # Combine expert predictions weighted by the router's decision
        rw_expanded = routing_weights.view(latents.shape[0], self.num_experts, 1, 1)
        action_chunks = torch.sum(expert_outputs * rw_expanded, dim=1)
        
        return action_chunks, safety_logits, routing_weights

def evaluate_model(model, checkpoint_path, output_video_path, runs=5):
    print(f"Loading Model from {checkpoint_path}...")

    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    clean_state_dict = {k.replace('_orig_mod.', ''): v for k, v in checkpoint.items()}
        
    model.load_state_dict(clean_state_dict, strict=True)
    model.eval()

    print("Evaluating model metrics:")
    sim = RunBasedRoverSim()
    
    # fourcc = cv2.VideoWriter_fourcc(*'VP80')
    # out_writer = cv2.VideoWriter(output_video_path, fourcc, 30.0, (500, 500))
    # print(f"Recording video to {output_video_path}...")
    
    initial_frame = sim.render()
    initial_rgb = cv2.cvtColor(initial_frame, cv2.COLOR_BGR2RGB)
    img_t = torch.from_numpy(initial_rgb).permute(2, 0, 1).float().div(255.0).unsqueeze(0).to(device)
    img_t = gpu_val_transform(img_t)
    
    seq_buffer_feats = torch.zeros((1, CONFIG['seq_len'], CONFIG['embed_dim']), device=device)
    actual_steer = 0.0 
    for i in range(runs):
        prev_pred_str = 0
        sim.reset_run()
        while True:
            # Handles spawning new biomes/chunks continuously based on player position
            
            run_terminated = sim.check_termination_conditions()
            if run_terminated:
                break
            sim.manage_world_chunks()
            dx = sim.goal.x - sim.x
            dz = sim.goal.z - sim.z
            dist = math.sqrt(dx**2 + dz**2)
            
            target_bearing = math.degrees(math.atan2(dx, dz))
            rel_bearing = (target_bearing - sim.yaw + 180) % 360 - 180
            
            norm_dist = min(dist / 50.0, 1.0)
            norm_head = rel_bearing / 180.0
            
            # Apply Inverse Heading constraint correctly
            if CONFIG.get('invert_heading', False):
                norm_head *= -1.0
                
            ctx_now = torch.tensor([[norm_dist, norm_head]], dtype=torch.float32).to(device)

            ref_thr, ref_str = sim.get_reference_autopilot()

            # We still need the frame for the model to process
            frame = sim.render()
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img_t = torch.from_numpy(frame_rgb).permute(2, 0, 1).float().div(255.0).unsqueeze(0).to(device)
            img_t = gpu_val_transform(img_t)
            
            with torch.no_grad(), torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                new_feat = model.encode(img_t).unsqueeze(1) # Shape: (1, 1, embed_dim)

                seq_buffer_feats = torch.cat([seq_buffer_feats[:, 1:, :], new_feat], dim=1)

                latents = model.forward_from_features(seq_buffer_feats)
                last_latent = latents[:, -1, :]
                
                best_chunk, safe_logits, routing_weights = model.get_action_heads(last_latent, ctx_now)
                
                danger_tensor = torch.sigmoid(safe_logits)
                danger_prob = danger_tensor.item() 
                sim.metrics["total_danger"] += danger_prob
                sim.metrics["mean_danger"] = sim.metrics["total_danger"] / sim.metrics["total_steps"] if sim.metrics["total_steps"] > 0 else 0
                
                pred_thr = float(best_chunk[0, 0, 0])
                pred_str = float(best_chunk[0, 0, 1])
                
                sim.metrics["total_jitter"] += abs(pred_str - prev_pred_str)
                sim.metrics["mean_jitter"] = sim.metrics["total_jitter"] / sim.metrics["total_steps"] if sim.metrics["total_steps"] > 0 else 0
                sim.metrics["total_throttle"] += pred_thr
                sim.metrics["mean_throttle"] = sim.metrics["total_throttle"] / sim.metrics["total_steps"] if sim.metrics["total_steps"] > 0 else 0
                # Extract out the inverse steering to match training coordinates
                if CONFIG.get('invert_heading', False):
                    pred_str *= -1.0

                # dominant_expert = np.argmax(routing_weights[0].cpu().numpy())

            actual_steer += 0.2 * (pred_str - actual_steer)

            sim.step(angular_velocity=actual_steer * 30.0, linear_velocity=10.0 * pred_thr)
            

            if sim.is_colliding():
                sim.metrics["steps_spent_colliding"] += 1

            # ==========================================
            # OVERLAY HUD
            # ==========================================
            # display_frame = frame.copy()
            # cv2.rectangle(display_frame, (20, 20), (220, 50), (50, 50, 50), -1) 
            # bar_w = int(danger_prob * 200)
            # color = (0, 255, 0) if danger_prob < 0.5 else (0, 0, 255)
            # cv2.rectangle(display_frame, (20, 20), (20 + bar_w, 50), color, -1)
            # ... (All text/HUD logic skipped for speed)
            
            # out_writer.write(display_frame)
            # cv2.imshow("RoverJEPA - Infinite Live Deployment", display_frame)
            # if cv2.waitKey(1) == ord('q'): 
            #     break
                
        # out_writer.release()
        # cv2.destroyAllWindows()
    
    
    
    print(sim.metrics)
    return sim.metrics

# ==========================================
# MAIN LIVE DEPLOYMENT LOOP
# ==========================================
def deploy_live(checkpoint_path, output_video_path):
    print(f"Loading Model from {checkpoint_path}...")
    model = RoverJEPA_v2_LSTM().to(device)
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    clean_state_dict = {k.replace('_orig_mod.', ''): v for k, v in checkpoint.items()}
        
    model.load_state_dict(clean_state_dict, strict=True)
    model.eval()

    print("Starting Infinite Live Environment...")
    sim = InfiniteRoverSim()
    
    fourcc = cv2.VideoWriter_fourcc(*'VP80')
    out_writer = cv2.VideoWriter(output_video_path, fourcc, 30.0, (500, 500))
    print(f"Recording video to {output_video_path}...")
    
    initial_frame = sim.render()
    initial_rgb = cv2.cvtColor(initial_frame, cv2.COLOR_BGR2RGB)
    img_t = torch.from_numpy(initial_rgb).permute(2, 0, 1).float().div(255.0).unsqueeze(0).to(device)
    img_t = gpu_val_transform(img_t)
    
    seq_buffer_imgs = [img_t for _ in range(CONFIG['seq_len'])]
    actual_steer = 0.0 
    
    while True:
        # Handles spawning new biomes/chunks continuously based on player position
        sim.manage_world_chunks()
        
        dx = sim.goal.x - sim.x
        dz = sim.goal.z - sim.z
        dist = math.sqrt(dx**2 + dz**2)
        
        target_bearing = math.degrees(math.atan2(dx, dz))
        rel_bearing = (target_bearing - sim.yaw + 180) % 360 - 180
        
        norm_dist = min(dist / 50.0, 1.0)
        norm_head = rel_bearing / 180.0
        
        # Apply Inverse Heading constraint correctly
        if CONFIG.get('invert_heading', False):
            norm_head *= -1.0
            
        ctx_now = torch.tensor([[norm_dist, norm_head]], dtype=torch.float32).to(device)

        ref_thr, ref_str = sim.get_reference_autopilot()

        frame = sim.render()
        
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img_t = torch.from_numpy(frame_rgb).permute(2, 0, 1).float().div(255.0).unsqueeze(0).to(device)
        img_t = gpu_val_transform(img_t)
        
        seq_buffer_imgs.append(img_t)
        seq_buffer_imgs.pop(0)

        input_imgs = torch.stack(seq_buffer_imgs, dim=1)
        
        with torch.no_grad(), torch.autocast(device_type='cuda', dtype=torch.bfloat16):
            latents, _ = model.forward_sequence(input_imgs)
            last_latent = latents[:, -1, :]
            
            best_chunk, safe_logits, routing_weights = model.get_action_heads(last_latent, ctx_now)
            
            danger_tensor = torch.sigmoid(safe_logits)
            danger_prob = danger_tensor.item() 
            
            pred_thr = float(best_chunk[0, 0, 0])
            pred_str = float(best_chunk[0, 0, 1])
            
            # Extract out the inverse steering to match training coordinates
            if CONFIG.get('invert_heading', False):
                pred_str *= -1.0

            dominant_expert = np.argmax(routing_weights[0].cpu().numpy())

        actual_steer += 0.2 * (pred_str - actual_steer)

        sim.step(angular_velocity=actual_steer * 30.0, linear_velocity=10.0 * pred_thr)

        # ==========================================
        # OVERLAY HUD
        # ==========================================
        display_frame = frame.copy()
        
        cv2.rectangle(display_frame, (20, 20), (220, 50), (50, 50, 50), -1) 
        bar_w = int(danger_prob * 200)
        color = (0, 255, 0) if danger_prob < 0.5 else (0, 0, 255)
        cv2.rectangle(display_frame, (20, 20), (20 + bar_w, 50), color, -1)
        cv2.putText(display_frame, f"DANGER: {danger_prob:.2f}", (30, 45), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        cv2.putText(display_frame, "LIVE: ENDLESS RUNNER", (20, 80), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        cv2.putText(display_frame, f"Dist to Goal: {dist:.1f}m", (20, 105), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(display_frame, f"Model Thr: {pred_thr:.2f}", (20, 130), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        cv2.putText(display_frame, f"Ref Thr: {ref_thr:.2f}", (20, 150), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
        cv2.putText(display_frame, f"Active Expert: {dominant_expert}", (20, 175), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 150, 0), 2)
        
        cx_hud, cy_hud = display_frame.shape[1] // 2, display_frame.shape[0] - 80
        radius = 50
        cv2.circle(display_frame, (cx_hud, cy_hud), radius, (100, 100, 100), 2) 
        
        h_angle = ref_str * 1.5 
        hx_end = int(cx_hud + radius * math.sin(h_angle))
        hy_end = int(cy_hud - radius * math.cos(h_angle))
        cv2.line(display_frame, (cx_hud, cy_hud), (hx_end, hy_end), (255, 0, 0), 2)
        
        m_angle = pred_str * 1.5
        mx_end = int(cx_hud + radius * math.sin(m_angle))
        my_end = int(cy_hud - radius * math.cos(m_angle))
        cv2.line(display_frame, (cx_hud, cy_hud), (mx_end, my_end), (0, 255, 0), 4)
        
        display_frame = cv2.resize(display_frame, (500, 500))
        out_writer.write(display_frame)
        
        cv2.imshow("RoverJEPA - Infinite Live Deployment", display_frame)
        if cv2.waitKey(1) == ord('q'): 
            print("Exiting Live Deployment.")
            break
            
    out_writer.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Live deployment of RoverJEPA in an infinite synthetic world.")
    parser.add_argument('--checkpoint', required=True, help="Path to the trained .pth model file.")
    parser.add_argument('--output_video', default='live_run.webm', help="Output path for the video recording.")
    args = parser.parse_args()
    
    # deploy_live(args.checkpoint, args.output_video)
    evaluate_model(RoverJEPA_v2_LSTM().to(device), args.checkpoint, args.output_video)