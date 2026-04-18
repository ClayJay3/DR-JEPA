import argparse
import os
import glob
import math
import warnings
import sys
import shutil
import gc
import csv

import cv2
import pandas as pd
import numpy as np
from tqdm import tqdm
from multiprocessing import Pool, cpu_count

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler, Subset
import torchvision.transforms.v2 as v2 
import torchvision.models as models
import torchvision

from live_inference_test import evaluate_model

# ==========================================
# SYSTEM OPTIMIZATIONS
# ==========================================
# Enables PyTorch's expandable memory segments to reduce memory fragmentation
os.environ["expandable_segments"] = "True" 
# Suppress noisy OpenCV logs
os.environ["OPENCV_LOG_LEVEL"] = "SILENT"
warnings.filterwarnings("ignore")
# Use TensorFloat32 (TF32) for faster matrix multiplications on Ampere+ GPUs
torch.set_float32_matmul_precision('medium')

# ==========================================
# CONFIGURATION
# ==========================================
CONFIG = {
    # --- Training Hyperparameters ---
    'epochs': 100,            
    'warmup_epochs': 5,      
    'batch_size': 32,
    'lr': 1e-4,             
    'weight_decay': 1e-4,   
    'ema_decay': 0.996,     
    'seq_len': 20,           # Number of historical frames to look at
    'action_horizon': 10,    # Number of future actions to predict
    'jepa_offset': 3,        # Temporal offset for JEPA target prediction
    'patience': 25,          # Early stopping patience
    
    # --- Architecture Dimensions ---
    'img_size': 224,         # Input image resolution (expected by DINOv2)
    'embed_dim': 384,        # DINOv2 ViT-S embedding dimension
    'hidden_dim': 512,       # Internal Transformer hidden dimension
    'proj_dim': 64,          # VICReg projection dimension
    'n_heads': 8,            # Transformer attention heads
    'n_layers': 4,           # Transformer encoder layers
    
    # --- Preprocessing Logic ---
    'stop_threshold': 0.05,  # Threshold to define "stopped" state
    'keep_idle_prob': 0.1,   # Probability of keeping stationary frames during training
    
    # --- Loss Function Weights ---
    'w_phys': 1.0,           # Weight for VICReg Invariance + Variance loss
    'w_cov': 1.0,            # Weight for VICReg Covariance loss
    'w_safe': 2.0,           # Weight for safety/critic loss
    'w_act': 10.0,           # Weight for action (throttle/steer) prediction loss
    
    # --- Safety Heuristics ---
    'stuck_throttle': 0.5, 
    'stuck_speed': 0.05, 
    'stuck_window': 10,
    
    # --- MoE & Mitigation Hyperparameters ---
    'backbone_lr_scale': 0.1,  # Learning rate multiplier for DINOv2 backbone
    'fusion_dropout_p': 0.2,   # Dropout probability before the expert router
    'num_experts': 3,          # Number of experts in the Mixture of Experts head
    'invert_heading': False    # Flips heading coordinate system to match physical datasets
}

# Automatically select GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# ==========================================
# GPU DATA AUGMENTATION
# ==========================================
# We apply transforms on the GPU to save CPU bottlenecks
gpu_train_transform = torch.nn.Sequential(
    v2.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
    v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
).to(device)

gpu_val_transform = torch.nn.Sequential(
    v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
).to(device)

# ==========================================
# PART 1: EFFICIENT JPEG PACKING (SHARDED)
# ==========================================
def _calc_nav_vector(row, heading):
    """
    Calculates the normalized distance and relative bearing to the goal.
    Converts lat/lon differences into meters (approx 111,139 meters per degree).
    """
    dy = (row['goal_lat'] - row['lat']) * 111139
    dx = (row['goal_lon'] - row['lon']) * 111139 * math.cos(math.radians(row['lat']))
    
    dist = math.sqrt(dx**2 + dy**2)
    target_bearing = math.degrees(math.atan2(dx, dy))
    
    # Calculate shortest angular distance (-180 to 180)
    rel_bearing = (target_bearing - heading + 180) % 360 - 180
    
    # Normalize: distance maxes out at 50m (value=1.0), bearing divided by 180
    return min(dist / 50.0, 1.0), rel_bearing / 180.0

def _process_video_jpg(job_args):
    """
    Worker function for multiprocessing. 
    Reads a single video, matches frames with CSV telemetry, compresses to JPEG,
    and returns a list of bytes and metadata.
    """
    vid_path, data_dir, config, vid_id = job_args
    # Disable OpenCV multithreading inside the worker to prevent thread deadlock
    cv2.setNumThreads(0) 
    
    base_name = os.path.splitext(os.path.basename(vid_path))[0]
    csv_path = os.path.join(data_dir, base_name + ".csv")
    if not os.path.exists(csv_path): 
        return None

    try:
        df = pd.read_csv(csv_path)
        cap = cv2.VideoCapture(vid_path)
        total_rows = len(df)
        
        valid_indices = np.arange(total_rows)
        required = config['seq_len'] + config['action_horizon']
        
        # Skip videos that are too short to form a single sequence
        if len(valid_indices) <= required:
            cap.release()
            return None

        # Extract telemetry
        headings = df['heading'].values if 'heading' in df.columns else np.zeros(total_rows)
        is_stuck = np.zeros(total_rows)
        
        # If traversal score exists, low score means "stuck"
        if 'trav_score' in df.columns:
            is_stuck = 1.0 - df['trav_score'].values.clip(0, 1)

        jpeg_payloads = [] 
        meta_rows = []     
        
        target_set = set(valid_indices)
        max_idx = valid_indices[-1]
        frame_idx = 0
        
        # Extract and compress frames
        while True:
            ret, frame = cap.read()
            if not ret or frame_idx > max_idx: 
                break
            
            if frame_idx in target_set:
                row = df.iloc[frame_idx]
                # Downsample immediately to save memory
                frame = cv2.resize(frame, (config['img_size'], config['img_size']))
                
                # Compress to JPEG format in memory
                success, encoded_img = cv2.imencode('.jpg', frame)
                
                if success:
                    jpg_bytes = encoded_img.tobytes()
                    dist, head = _calc_nav_vector(row, headings[frame_idx])
                    
                    jpeg_payloads.append(jpg_bytes)
                    # Metadata format: [distance, heading, throttle, steer, is_stuck, video_id]
                    meta_rows.append([dist, head, row['throttle'], row['steer'], is_stuck[frame_idx], vid_id])
                
            frame_idx += 1
            
        cap.release()
        return (jpeg_payloads, meta_rows)
        
    except Exception as e:
        print(f"Error processing {vid_path}: {e}")
        return None

def process_and_pack(data_dir, output_dir):
    """
    Orchestrates the conversion of a directory of raw videos/csvs into a single 
    contiguous binary file of JPEGs (`images.bin`). This prevents file-system inode 
    exhaustion and allows highly efficient sequential disk reads during training.
    """
    if os.path.exists(output_dir): 
        shutil.rmtree(output_dir)
    os.makedirs(output_dir)
    
    video_files = sorted(glob.glob(os.path.join(data_dir, "*.mp4")) + glob.glob(os.path.join(data_dir, "*.avi")))
    if not video_files: 
        print("No videos found.")
        return

    print(f"--- Processing {len(video_files)} videos (JPEG Compression) ---")
    
    # Prepare arguments for multiprocessing pool
    worker_args = [(v, data_dir, CONFIG, float(i+1)) for i, v in enumerate(video_files)]
    bin_path = os.path.join(output_dir, 'images.bin')
    
    all_offsets = [] 
    all_meta = []
    current_offset = 0
    
    # Write everything into a single binary blob
    with open(bin_path, 'wb') as f_bin:
        with Pool(cpu_count()) as pool:
        # with Pool(processes=4) as pool:
            for result in tqdm(pool.imap(_process_video_jpg, worker_args), total=len(video_files)):
                if result:
                    jpg_bytes_list, meta_list = result
                    
                    for jpg_data in jpg_bytes_list:
                        length = len(jpg_data)
                        f_bin.write(jpg_data)
                        # Keep track of where each frame starts and its length
                        all_offsets.append([current_offset, length])
                        current_offset += length
                    
                    all_meta.extend(meta_list)
    
    print(f"Saving indices for {len(all_offsets)} frames...")
    
    # Save the lookup tables
    np.save(os.path.join(output_dir, 'offsets.npy'), np.array(all_offsets, dtype=np.int64))
    np.save(os.path.join(output_dir, 'meta.npy'), np.array(all_meta, dtype=np.float32))
    np.savez(os.path.join(output_dir, 'info.npz'), count=len(all_offsets))
    
    print(f"Done. Compressed Size: {current_offset / (1024**3):.2f} GB")

# ==========================================
# PART 2: JPEG DATASET (On-the-fly Decode)
# ==========================================
class JPGDataset(Dataset):
    """
    Custom PyTorch Dataset that reads sequences of compressed JPEGs from the binary file.
    Uses Memory Mapping (mmap_mode='r') for offset arrays so they don't consume RAM.
    """
    def __init__(self, data_dir, seq_len=20, is_val=False, val_split=0.2):
        self.seq_len = seq_len
        self.chunk_size = CONFIG['action_horizon']
        self.data_dir = data_dir
        self.total_len = seq_len + self.chunk_size
        
        # Load indices via memory mapping to support massive datasets
        self.offsets = np.load(os.path.join(data_dir, 'offsets.npy'), mmap_mode='r')
        self.meta = np.load(os.path.join(data_dir, 'meta.npy'), mmap_mode='r')
        self.bin_path = os.path.join(data_dir, 'images.bin')
        
        self.bin_file = None
        
        # Group frames by their source video ID
        raw_ids = self.meta[:, 5]
        log_ids = raw_ids.astype(int)
        unique_logs = np.unique(log_ids)
        
        valid_logs = []
        for log_id in unique_logs:
            (idxs,) = np.where(log_ids == log_id)
            if len(idxs) > self.total_len:
                valid_logs.append(log_id)
        
        valid_logs = np.array(valid_logs)
        
        # Local RNG to ensure train/val split is deterministic and prevents data leakage
        rng = np.random.default_rng(42)
        shuffled_logs = rng.permutation(valid_logs)
        
        # Split videos into train and validation sets
        n_val = max(1, int(len(shuffled_logs) * val_split))
        split_pt = len(shuffled_logs) - n_val
        target_logs = shuffled_logs[split_pt:] if is_val else shuffled_logs[:split_pt]
            
        self.indices = []
        for log_id in target_logs:
            (log_indices,) = np.where(log_ids == log_id)
            # Create sliding windows separated by 5 frames (stride = 5)
            for i in range(0, len(log_indices) - self.total_len + 1, 5): 
                self.indices.append(log_indices[i])

        print(f" {('Val' if is_val else 'Train')} Sequences: {len(self.indices)}")

        # Create sampling weights to balance active driving vs. idling
        self.weights = []
        if not is_val and len(self.indices) > 0:
            for idx in self.indices:
                # Check throttle at the center of the sequence
                thr = abs(self.meta[idx + seq_len//2, 2])
                # Downweight sequences where the rover is practically stopped
                self.weights.append(0.1 if thr < 0.05 else 1.0)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        # Open file lazily per worker to prevent multi-processing file descriptor errors
        if self.bin_file is None:
            self.bin_file = open(self.bin_path, 'rb')
            
        start_idx = self.indices[idx]
        read_len = self.seq_len + self.chunk_size
        
        # Retrieve metadata for the sequence
        meta_seq = self.meta[start_idx : start_idx + read_len]
        
        img_tensors = []
        offset_seq = self.offsets[start_idx : start_idx + self.seq_len] 
        
        # Read and decode JPEGs sequentially
        for i in range(self.seq_len):
            off, length = offset_seq[i]
            
            self.bin_file.seek(off)
            jpg_bytes = self.bin_file.read(length)
            
            # Hardware-accelerated (or highly optimized) JPEG decoding via TorchVision
            byte_tensor = torch.frombuffer(jpg_bytes, dtype=torch.uint8)
            img_t = torchvision.io.decode_jpeg(byte_tensor)
            img_tensors.append(img_t)
            
        # Shape: (Sequence, Channels, Height, Width)
        img_tensor = torch.stack(img_tensors) 
        
        # Context vectors (distance, heading) and safety ground truth
        context = torch.from_numpy(meta_seq[:self.seq_len, 0:2])
        stuck = torch.from_numpy(meta_seq[:self.seq_len, 4:5])
        
        # Formulate action chunks (Future prediction horizon)
        action_chunks = []
        full_actions = meta_seq[:, 2:4] 
        for t in range(self.seq_len):
            chunk = full_actions[t : t + self.chunk_size]
            action_chunks.append(chunk)
            
        # Shape: (Sequence, Action Horizon, 2 [throttle, steer])
        action_chunks = torch.from_numpy(np.stack(action_chunks))
        
        return img_tensor, action_chunks, context, stuck

# ==========================================
# PART 3: ARCHITECTURE (LATE FUSION)
# ==========================================
class RoverJEPA_v2(nn.Module):
    """
    Main Model Architecture combining DINOv2 (Vision), a Transformer Encoder (Temporal),
    and a Mixture of Experts (Action/Policy routing).
    """
    def __init__(self):
        super().__init__()
        
        print("Loading DINOv2 (ViT-S/14)...")
        self.backbone = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14', source='github')
        
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
        
        # Transformer processes the sequence of frames
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.hidden_dim, 
            nhead=CONFIG['n_heads'], 
            dim_feedforward=self.hidden_dim*4, 
            dropout=0.1, 
            batch_first=True,
            norm_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=CONFIG['n_layers'])
        
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
        
        # Multiple action experts (e.g., one might learn to drive straight, another to turn sharply)
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
        latent_seq, _ = self.transformer(x) 
        return latent_seq

    def forward_sequence(self, images):
        """
        Processes a sequence of images through the backbone and temporal transformer.
        Applies random masking during training for the JEPA objective.
        """
        B, S, C, H, W = images.shape
        flat_imgs = images.view(B*S, C, H, W)
        
        # Extract features using DINOv2
        feats = self.backbone(flat_imgs)
        feats = feats.view(B, S, -1)
            
        x = self.input_proj(feats) 
        x = x + self.pos_embed[:, :S, :]
        
        # Self-supervised objective: Mask random frames (except the current frame)
        if self.training:
            mask = torch.rand(B, S, 1, device=x.device) < 0.15
            mask[:, 0, :] = False 
            x = torch.where(mask, self.mask_token, x)
        
        # Prevent transformer from looking into the future
        attn_mask = nn.Transformer.generate_square_subsequent_mask(S, device=x.device)
        latent_seq = self.transformer(x, mask=attn_mask, is_causal=True)
        
        return latent_seq, feats

    def get_jepa_projections(self, latents_pred, feats_target):
        """Projects transformer latents and target backbone features for self-supervised loss."""
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
        fusion_input = torch.cat([self.fusion_dropout(policy_input), context_sequence, danger_input], dim=1)
        
        # 3. Route to Experts
        routing_logits = self.router(fusion_input)
        
        # Add Gumbel-like noise during training to encourage expert exploration
        if self.training:
            routing_logits = routing_logits + torch.randn_like(routing_logits) * 0.1
            
        routing_weights = F.softmax(routing_logits, dim=-1)
        
        # Calculate actions from all experts
        expert_outputs = []
        for expert in self.experts:
            expert_outputs.append(expert(fusion_input).view(latents.shape[0], CONFIG['action_horizon'], 2))
            
        expert_outputs = torch.stack(expert_outputs, dim=1) 
        
        # Combine expert predictions weighted by the router's decision
        rw_expanded = routing_weights.view(latents.shape[0], self.num_experts, 1, 1)
        action_chunks = torch.sum(expert_outputs * rw_expanded, dim=1) 
        
        return action_chunks, safety_logits, routing_weights

# ==========================================
# PART 4: LOSS FUNCTIONS
# ==========================================
def robust_vicreg_loss(x, y):
    """
    Variance-Invariance-Covariance Regularization.
    Forces representations to be similar (Invariance), prevents collapse (Variance),
    and decorrelates dimensions (Covariance).
    """
    # 1. Invariance (MSE between representations)
    loss_inv = F.mse_loss(x, y)
    
    # 2. Variance (Hinge loss on standard deviation)
    std_x = torch.sqrt(x.var(dim=0) + 1e-04)
    std_y = torch.sqrt(y.var(dim=0) + 1e-04)
    loss_var = torch.mean(F.relu(1 - std_x)) + torch.mean(F.relu(1 - std_y))
    
    # 3. Covariance (Decorrelate dimensions)
    N, D = x.shape
    x = x - x.mean(dim=0)
    y = y - y.mean(dim=0)
    cov_x = (x.T @ x) / (N - 1)
    cov_y = (y.T @ y) / (N - 1)
    
    # Ignore the diagonal (variances) and sum the off-diagonals (covariances)
    mask = ~torch.eye(D, dtype=torch.bool, device=x.device)
    loss_cov = cov_x[mask].pow(2).sum() / D + cov_y[mask].pow(2).sum() / D
    
    return loss_inv, loss_var, loss_cov

def weighted_action_loss(pred_chunks, target_chunks):
    """
    Custom Action Loss:
    - Standard smooth L1 for throttle.
    - Steering loss is multiplied heavily when target throttle is high (turning is more important at speed).
    - Adds a temporal 'jerk' penalty to encourage smooth, continuous actions over the horizon.
    """
    pred_thr = pred_chunks[:, :, 0]
    pred_str = pred_chunks[:, :, 1]
    target_thr = target_chunks[:, :, 0]
    target_str = target_chunks[:, :, 1]
    
    loss_thr = F.smooth_l1_loss(pred_thr, target_thr)
    
    # Weights for steering based on the magnitude of the target steering/speed
    weights = 1.0 + (torch.abs(target_str) * 2.0)
    loss_str = (weights * F.smooth_l1_loss(pred_str, target_str, reduction='none')).mean()
    
    # Jerk penalty: Penalize large sudden jumps between consecutive predictions in the action chunk
    action_jerk = torch.mean((pred_chunks[:, 1:, :] - pred_chunks[:, :-1, :]) ** 2)
    
    return loss_thr + loss_str + (0.5 * action_jerk)

# ==========================================
# PART 5: TRAINING LOOP
# ==========================================
def train_model(args):
    """Orchestrates the entire training process."""
    os.makedirs(args.save_dir, exist_ok=True)
    print(f"Loading JPG Dataset from {args.dataset}...")
    torch.cuda.empty_cache() 
    
    # Initialize datasets
    train_dataset = JPGDataset(args.dataset, seq_len=CONFIG['seq_len'], is_val=False)
    val_dataset = JPGDataset(args.dataset, seq_len=CONFIG['seq_len'], is_val=True)
    
    if len(train_dataset) == 0: 
        print("Dataset is empty. Exiting.")
        return

    # Weighted sampler to address class/driving-behavior imbalance
    sampler = WeightedRandomSampler(train_dataset.weights, len(train_dataset))
    batch_size = CONFIG['batch_size']
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        sampler=sampler, 
        num_workers=8,            
        pin_memory=True, 
        drop_last=True,
        prefetch_factor=3,           
        persistent_workers=True      
    )
    

    CONFIG['val_check_ratio'] = 0.10 

    val_loader = None
    if len(val_dataset) > 0:
        # num_train = len(train_dataset)
        # dynamic_val_size = int(num_train * CONFIG['val_check_ratio'])
        # # target_size = max(100, min(dynamic_val_size, 2000))
        # target_size = 100
        # if len(val_dataset) > target_size:
        #     indices = np.random.RandomState(42).choice(
        #         len(val_dataset), target_size, replace=False
        #     )
        #     val_dataset = Subset(val_dataset, indices)
        
        # We use a larger batch size for validation because there's no autograd overhead
        # val_batch_size = CONFIG['batch_size'] * 2 
        val_batch_size = CONFIG['batch_size']
        
        val_loader = DataLoader(
            val_dataset, 
            batch_size=val_batch_size, 
            shuffle=False, 
            num_workers=8,               # Match training workers for fast JPEG decoding
            pin_memory=True, 
            drop_last=True,
            prefetch_factor=2,           # Keep the CPU ahead of the GPU
            persistent_workers=True      
        )

    model = RoverJEPA_v2().to(device)
    print(f"Model Parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    
    # Separate parameter groups for differential learning rates
    bb_params = list(model.backbone.blocks[-1].parameters()) + list(model.backbone.norm.parameters())
    bb_param_ids = set(id(p) for p in bb_params)
    base_params = [p for p in filter(lambda p: p.requires_grad, model.parameters()) if id(p) not in bb_param_ids]
    
    # We start with the backbone frozen (lr: 0.0) during warmup
    optimizer = optim.AdamW([
        {'params': base_params},
        {'params': bb_params, 'lr': 0.0} 
    ], lr=CONFIG['lr'], weight_decay=CONFIG['weight_decay'], fused=True)

    # Use PyTorch 2.0+ Compiler for massive speed boosts
    model = torch.compile(model)
    print("Model Compiled with torch.compile()")
    
    best_val_loss = float('inf')
    patience_counter = 0 
    backbone_unfrozen = False
    
    print(f"--- TRAINING v2.1 (Efficient JPG Packing + Trap Mitigations) ---")
    
    for epoch in range(args.epochs):
        # Unfreeze backbone after warmup epochs
        if epoch >= CONFIG['warmup_epochs']:
            if not backbone_unfrozen: 
                bb_lr = CONFIG['lr'] * CONFIG['backbone_lr_scale']
                optimizer.param_groups[1]['lr'] = bb_lr
                backbone_unfrozen = True
                print("-> DINOv2 Last Block LR Increased (Unfrozen)")
                
        gc.collect() # Periodically clean up memory
        model.train()
        loop = tqdm(train_loader, desc=f"Ep {epoch+1}/{args.epochs}")
        
        metrics = {'phys': 0, 'cov': 0, 'safe': 0, 'act': 0}
        
        for i, (imgs, action_chunks, contexts, stucks) in enumerate(loop):
            imgs = imgs.to(device, non_blocking=True)
            contexts = contexts.to(device, non_blocking=True)
            action_targets = action_chunks.to(device, non_blocking=True)
            stucks = stucks.to(device, non_blocking=True)
            
            optimizer.zero_grad()
            
            # Use Automatic Mixed Precision (AMP) to save memory and speed up compute
            with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                B, S, C, H, W = imgs.shape
                
                # Spatial augmentation: Random horizontal flipping
                flip_mask = torch.rand(B, device=device) > 0.5
                imgs[flip_mask] = torch.flip(imgs[flip_mask], dims=[-1])
                # Invert target steering and relative heading if flipped
                action_targets[flip_mask, ..., 1] *= -1.0 
                contexts[flip_mask, ..., 1] *= -1.0

                # Prepare and run image transforms
                imgs_float = imgs.float() / 255.0
                imgs_strip = imgs_float.permute(0, 2, 1, 3, 4).reshape(B, C, S * H, W)
                imgs_aug_strip = gpu_train_transform(imgs_strip)
                imgs_aug = imgs_aug_strip.view(B, C, S, H, W).permute(0, 2, 1, 3, 4).contiguous()
                
                # Forward Pass
                latents, feats = model.forward_sequence(imgs_aug)
                
                offset = CONFIG.get('jepa_offset', 1)
                
                # Flatten sequences for loss calculations
                latents_flat = latents.reshape(-1, model.hidden_dim)
                contexts_flat = contexts.reshape(-1, 2)
                action_targets_flat = action_targets.reshape(-1, CONFIG['action_horizon'], 2)
                stucks_flat = stucks.reshape(-1, 1)
                
                # Heads Evaluation
                pred_chunks, safe_logits, routing_weights = model.get_action_heads(latents_flat, contexts_flat)
                loss_act = weighted_action_loss(pred_chunks, action_targets_flat)
                loss_safe = F.binary_cross_entropy_with_logits(safe_logits, stucks_flat)
                
                # Load balancing loss for MoE (forces router to use all experts)
                expert_usage = routing_weights.mean(dim=0)
                cv_loss = expert_usage.var()
                
                # JEPA Evaluation (Predict feature offsets `offset` steps into the future)
                latents_pred = latents[:, :-offset].reshape(-1, model.hidden_dim)
                feats_target = feats[:, offset:].reshape(-1, model.embed_dim)
                p_pred, p_target = model.get_jepa_projections(latents_pred, feats_target)
                
                l_inv, l_var, l_cov = robust_vicreg_loss(p_pred, p_target)
                loss_phys = l_inv + l_var
                
                # Total Combined Loss
                loss_total = (CONFIG['w_phys'] * loss_phys) + \
                             (CONFIG['w_cov'] * l_cov) + \
                             (CONFIG['w_safe'] * loss_safe) + \
                             (CONFIG['w_act'] * loss_act) + \
                             (0.1 * cv_loss)
            
            # Backpropagation
            loss_total.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            # Logging
            metrics['phys'] += loss_phys.item()
            metrics['cov'] += l_cov.item()
            metrics['act'] += loss_act.item()
            loop.set_postfix({
                'Tot': f"{loss_total.item():.2f}",   
                'Phy': f"{loss_phys.item():.2f}",    
                'Cov': f"{l_cov.item():.2f}",        
                'Act': f"{loss_act.item():.2f}",      
                'Safe': f"{loss_safe.item():.3f}"    
            })   

        # ==================== VALIDATION ====================
        if val_loader:
            model.eval()
            val_metrics = {'tot': 0.0, 'phys': 0.0, 'cov': 0.0, 'act': 0.0, 'safe': 0.0}
            val_loop = tqdm(val_loader, desc="Validation", leave=False)
            
            with torch.inference_mode(), torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                for i, (imgs, act_chunks, ctx, stuck) in enumerate(val_loop):
                    imgs = imgs.to(device)
                    ctx = ctx.to(device)
                    act_chunks = act_chunks.to(device)
                    stuck = stuck.to(device)
                    
                    B, S, C, H, W = imgs.shape
                    imgs_float = imgs.view(B*S, C, H, W).float() / 255.0
                    imgs_aug = gpu_val_transform(imgs_float).view(B, S, C, H, W)
                    
                    latents, feats = model.forward_sequence(imgs_aug)
                    
                    offset = CONFIG.get('jepa_offset', 1)
                    latents_flat = latents.reshape(-1, model.hidden_dim)
                    contexts_flat = ctx.reshape(-1, 2)
                    action_targets_flat = act_chunks.reshape(-1, CONFIG['action_horizon'], 2)
                    stucks_flat = stuck.reshape(-1, 1)
                    
                    pred_chunks, safe_logits, routing_weights = model.get_action_heads(latents_flat, contexts_flat)
                    loss_act = weighted_action_loss(pred_chunks, action_targets_flat)
                    loss_safe = F.binary_cross_entropy_with_logits(safe_logits, stucks_flat)

                    expert_usage = routing_weights.mean(dim=0)
                    cv_loss = expert_usage.var()

                    latents_pred = latents[:, :-offset].reshape(-1, model.hidden_dim)
                    feats_target = feats[:, offset:].reshape(-1, model.embed_dim)
                    p_pred, p_target = model.get_jepa_projections(latents_pred, feats_target)
                    
                    li, lv, lc = robust_vicreg_loss(p_pred, p_target)
                    loss_phys = li + lv
                    
                    loss_total = (CONFIG['w_phys'] * loss_phys) + \
                                 (CONFIG['w_cov'] * lc) + \
                                 (CONFIG['w_safe'] * loss_safe) + \
                                 (CONFIG['w_act'] * loss_act) + \
                                 (0.1 * cv_loss)
                    
                    val_metrics['tot'] += loss_total.item()
                    val_metrics['phys'] += loss_phys.item()
                    val_metrics['cov'] += lc.item()
                    val_metrics['act'] += loss_act.item()
                    val_metrics['safe'] += loss_safe.item()
                    
                    val_loop.set_postfix({
                        'Tot': f"{val_metrics['tot'] / (i+1):.2f}",
                        'Phy': f"{val_metrics['phys'] / (i+1):.2f}",
                        'Cov': f"{val_metrics['cov'] / (i+1):.2f}",
                        'Act': f"{val_metrics['act'] / (i+1):.2f}",
                        'Safe': f"{val_metrics['safe'] / (i+1):.3f}"
                    })
            
            avg_val_tot = val_metrics['tot'] / len(val_loader)
            avg_val_phys = val_metrics['phys'] / len(val_loader)
            avg_val_act = val_metrics['act'] / len(val_loader)
            avg_val_safe = val_metrics['safe'] / len(val_loader)
            
            print(f"  -> Val Loss | Tot: {avg_val_tot:.2f} | Phy: {avg_val_phys:.2f} | Act: {avg_val_act:.2f} | Safe: {avg_val_safe:.3f}")
            
            # Calculate a functional score that ignores the self-supervised penalties for checkpointing
            functional_val_loss = avg_val_act + avg_val_safe

            # Early Stopping and Checkpointing Check
            if functional_val_loss < best_val_loss:
                best_val_loss = functional_val_loss
                patience_counter = 0 
                torch.save(model.state_dict(), os.path.join(args.save_dir, "best_jepa_v2.pth"))
                metrics = evaluate_model(RoverJEPA_v2().to(device), "runs/best_jepa_v2.pth", "/", runs=5)
                training_metrics_path = os.path.join(args.save_dir, "training_metrics.csv")
                row = {'epoch': epoch}
                row.update(metrics)
                
                file_exists = os.path.isfile(training_metrics_path)
                
                # We open in 'a' (append) mode
                with open(training_metrics_path, mode='a', newline='') as f:
                    writer = csv.DictWriter(f, fieldnames=row.keys())
                    
                    # If the file is new, write the header first
                    if not file_exists:
                        writer.writeheader()
                        
                    writer.writerow(row)
                print(f"     [Saved Best Model (Action + Safe)]")
            else:
                patience_counter += 1
                print(f"     [No Improve] Patience: {patience_counter}/{CONFIG['patience']}")
                
            if patience_counter >= CONFIG['patience']:
                print(f"Early stopping triggered after {epoch+1} epochs.")
                break
                
        # Always save the latest model as well
        torch.save(model.state_dict(), os.path.join(args.save_dir, "latest_jepa_v2.pth"))

# ==========================================
# PART 6: VISUALIZATION (OPEN LOOP + HUD)
# ==========================================
def visualize(args):
    """
    Visualizes the model's predictions over a given video.
    Reads a video, runs the model on rolling sequences of frames, and 
    overlays a HUD comparing Ground Truth (Human) vs Model predictions.
    """
    model = RoverJEPA_v2().to(device)
    
    # Load checkpoint, strip `_orig_mod.` prefixes which are added by `torch.compile`
    checkpoint = torch.load(args.checkpoint, map_location=device)
    clean_state_dict = {k.replace('_orig_mod.', ''): v for k, v in checkpoint.items()}
    model.load_state_dict(clean_state_dict, strict=True)
    
    model.eval()
    
    # Load associated telemetry 
    csv_path = args.video.replace(".mp4", ".csv").replace(".avi", ".csv")
    if not os.path.exists(csv_path):
        print(f"Error: CSV file {csv_path} not found.")
        return
    df = pd.read_csv(csv_path)
    
    goal_lat = df.iloc[0]['goal_lat']
    goal_lon = df.iloc[0]['goal_lon']
    
    cap = cv2.VideoCapture(args.video)
    seq_buffer_imgs = []
    
    frame_idx = 0
    print("Running OPEN LOOP Visualization with Deterministic Decoding...")
    
    while True:
        ret, frame = cap.read()
        if not ret: 
            break
        
        # Calculate current context (Distance/Heading)
        if frame_idx < len(df):
            row = df.iloc[frame_idx]
            dy = (goal_lat - row['lat']) * 111139
            dx = (goal_lon - row['lon']) * 111139 * math.cos(math.radians(row['lat']))
            dist = math.sqrt(dx**2 + dy**2)
            target_bearing = math.degrees(math.atan2(dx, dy))
            rel_bearing = (target_bearing - row['heading'] + 180) % 360 - 180
            
            norm_dist = min(dist / 50.0, 1.0)
            norm_head = rel_bearing / 180.0
            
            if CONFIG.get('invert_heading', False):
                norm_head *= -1.0
                
            ctx_now = torch.tensor([[norm_dist, norm_head]], dtype=torch.float32).to(device)
            gt_thr, gt_str = row['throttle'], row['steer']
        else:
            gt_thr, gt_str = 0.0, 0.0
            ctx_now = torch.zeros(1, 2).to(device)
        
        # Preprocess Frame
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img_small = cv2.resize(frame_rgb, (CONFIG['img_size'], CONFIG['img_size']))
        img_t = torch.from_numpy(img_small).permute(2, 0, 1).float().div(255.0).unsqueeze(0).to(device)
        img_t = gpu_val_transform(img_t)
        
        # Maintain rolling buffer of frames
        if len(seq_buffer_imgs) == 0:
            seq_buffer_imgs = [img_t] * CONFIG['seq_len']
        else:
            seq_buffer_imgs.append(img_t)
            if len(seq_buffer_imgs) > CONFIG['seq_len']: 
                seq_buffer_imgs.pop(0)
            
        pred_thr, pred_str, danger_prob = 0.0, 0.0, 0.0
        
        # Run inference once the buffer is full
        if len(seq_buffer_imgs) == CONFIG['seq_len']:
            input_imgs = torch.stack(seq_buffer_imgs, dim=1)
            with torch.no_grad(), torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                latents, _ = model.forward_sequence(input_imgs)
                last_latent = latents[:, -1, :] # Take the latent corresponding to the current (last) frame
                
                best_chunk, safe_logits, routing_weights = model.get_action_heads(last_latent, ctx_now)
                
                danger_tensor = torch.sigmoid(safe_logits)
                danger_prob = danger_tensor.item() 
                
                # Extract first action from the predicted action horizon chunk
                pred_thr = float(best_chunk[0, 0, 0])
                pred_str = float(best_chunk[0, 0, 1])
                
                clean_weights_np = routing_weights[0].cpu().numpy()
                dominant_expert = np.argmax(clean_weights_np)
                
                print(f"Frame {frame_idx:04d} | Danger: {danger_prob:.2f} | "
                      f"Weights: [E0: {clean_weights_np[0]:.2f}, E1: {clean_weights_np[1]:.2f}, E2: {clean_weights_np[2]:.2f}] | "
                      f"Active: Expert {dominant_expert}")

        # --- Draw HUD (Heads Up Display) ---
        # Danger Bar
        cv2.rectangle(frame, (20, 20), (220, 50), (50, 50, 50), -1) 
        bar_w = int(danger_prob * 200)
        color = (0, 255, 0) if danger_prob < 0.5 else (0, 0, 255)
        cv2.rectangle(frame, (20, 20), (20 + bar_w, 50), color, -1)
        cv2.putText(frame, f"DANGER: {danger_prob:.2f}", (30, 45), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Text Metrics
        cv2.putText(frame, "DETERMINISTIC PLANNER", (20, 80), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.putText(frame, f"Model Thr: {pred_thr:.2f}", (20, 110), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.putText(frame, f"Human Thr: {gt_thr:.2f}", (20, 135), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
        cv2.putText(frame, f"Active Expert: {dominant_expert}", (20, 160), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 150, 0), 2)
        
        # Steering Wheel Visualization
        cx_hud, cy_hud = frame.shape[1] // 2, frame.shape[0] - 80
        radius = 60
        cv2.circle(frame, (cx_hud, cy_hud), radius, (100, 100, 100), 2) 
        
        # Human Steering (Blue Line)
        h_angle = gt_str * 1.5 
        hx_end = int(cx_hud + radius * math.sin(h_angle))
        hy_end = int(cy_hud - radius * math.cos(h_angle))
        cv2.line(frame, (cx_hud, cy_hud), (hx_end, hy_end), (255, 0, 0), 3)
        
        # Model Steering (Green Line)
        m_angle = pred_str * 1.5
        mx_end = int(cx_hud + radius * math.sin(m_angle))
        my_end = int(cy_hud - radius * math.cos(m_angle))
        cv2.line(frame, (cx_hud, cy_hud), (mx_end, my_end), (0, 255, 0), 4)
        
        cv2.putText(frame, "GREEN: Model", (cx_hud - 80, cy_hud + 70), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        cv2.putText(frame, "BLUE: Human", (cx_hud + 20, cy_hud + 70), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
        
        # Display the frame
        cv2.imshow("RoverJEPA Deterministic", frame)
        if cv2.waitKey(1) == ord('q'): 
            break
            
        frame_idx += 1
        
    cap.release()
    cv2.destroyAllWindows()

# ==========================================
# CLI ENTRY POINT
# ==========================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RoverJEPA v2 - Training and Inference Pipeline")
    sub = parser.add_subparsers(dest='mode', required=True)
    
    # Mode 1: Preprocess
    p_pre = sub.add_parser('preprocess', help="Extracts frames and compresses them into a binary file.")
    p_pre.add_argument('--data_dir', required=True, help="Directory containing raw .mp4 and .csv files")
    p_pre.add_argument('--output', required=True, help="Directory to save the packed binary dataset")
    
    # Mode 2: Train
    p_train = sub.add_parser('train', help="Trains the RoverJEPA model.")
    p_train.add_argument('--dataset', required=True, help="Directory containing the packed dataset")
    p_train.add_argument('--save_dir', default='runs', help="Directory to save checkpoints")
    p_train.add_argument('--epochs', type=int, default=CONFIG['epochs'], help="Number of epochs to train")
    
    # Mode 3: Visualize
    p_viz = sub.add_parser('viz', help="Visualizes the model predictions on a single video.")
    p_viz.add_argument('--video', required=True, help="Path to the video file to visualize (.mp4/.avi)")
    p_viz.add_argument('--checkpoint', required=True, help="Path to the trained model weights (.pth)")
    
    args = parser.parse_args()
    
    if args.mode == 'preprocess': 
        process_and_pack(args.data_dir, args.output)
    elif args.mode == 'train': 
        train_model(args)
    elif args.mode == 'viz': 
        visualize(args)