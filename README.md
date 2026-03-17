# DR-JEPAv6: Off-Road Navigation 

DR-JEPAv6 (Joint-Embedding Predictive Architecture) is an off-road autonomous navigation model. It combines self-supervised learning for vision with a Mixture of Experts (MoE) action policy to predict continuous driving trajectories from a rolling buffer of camera frames and GPS telemetry.

## Architecture Overview

*(Note: Download and open `DR-JEPAV6_ARCH.html` in your browser for the full interactive architecture diagram!)*

The model pipeline is split into a Training Phase and an Inference Phase, utilizing the following core components:

* **Vision Backbone (DINOv2):** A ViT-S/14 model that processes a 20-frame rolling buffer of images to extract spatial features (384 dimensions).
* **Temporal State Estimator:** A 4-layer Causal Transformer Encoder that aggregates the history of features into a 512-dimensional latent belief state.
* **JEPA Predictor (Training Only):** A self-supervised head that learns the environment's dynamics by predicting target DINOv2 features 3 steps into the future, optimized via a robust VICReg loss.
* **Safety Critic:** A binary classifier that monitors the latent state to predict the probability of the rover getting stuck or facing danger.
* **Action Policy (MoE):** A Mixture of Experts router that fuses the latent state, GPS context, and danger signal. It selects between 3 experts to predict a 10-step future action chunk (throttle and steering) with a temporal jerk penalty.

## Environment Setup

This project uses `pipenv` for dependency management. Ensure you have Python installed, then set up the environment:

1. Install pipenv if you haven't already:
   ```bash
   pip install pipenv
   ```
2. Install the project dependencies from the `Pipfile`:
   ```bash
   pipenv install
   ```
3. Activate the virtual environment:
   ```bash
   pipenv shell
   ```

## Usage & Example Commands

Below are the core commands to generate data, preprocess it, train the model, and run inferences.

### 1. Data Generation
Generate synthetic driving data and telemetry to test the pipeline:
```bash
python generate_synth_data.py
```

### 2. Preprocessing
Pack raw video (`.mp4`/`.avi`) and telemetry (`.csv`) files into an efficient, memory-mapped binary JPEG format for high-speed training:
```bash
python DR-JEPA6.py preprocess --data_dir /path/to/raw_data --output /path/to/packed_dataset
```

### 3. Training
Train the JEPA model using the packed binary dataset. The script automatically handles DINOv2 freezing/unfreezing, loss weighting, and early stopping.
```bash
python DR-JEPA6.py train --dataset /path/to/packed_dataset --save_dir runs/
```

### 4. Visualization (Open Loop)
Run deterministic open-loop visualization over a specific video, rendering a Heads-Up Display (HUD) showing ground truth vs. model predictions and active MoE experts.
```bash
python DR-JEPA6.py viz --video /path/to/video.mp4 --checkpoint runs/best_jepa_v2.pth
```

### 5. Live Inference
Test the model's closed-loop real-time capabilities:
```bash
python live_inference_test.py
```
