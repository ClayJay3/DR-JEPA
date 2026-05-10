# Comparative Evaluation of Predictive World-Modeling and Recurrent Reactive Policies for Reliable Navigation Under Shared Visual and Control Inputs

Compares two temporal modeling strategies for the same autonomous rover task: a predictive world-modeling approach based on JEPA and a recurrent reactive baseline based on LSTM. Both models use the same DINOv2 vision backbone, the same packed video/telemetry dataset, and the same continuous control outputs, which keeps the comparison focused on temporal reasoning rather than perception or data differences.

## Architecture Overview

_(Note: Download and open `ARCH_DIAGRAM.html` in your browser for the full interactive architecture diagram!)_

Both pipelines share the same front end:

- **Vision Backbone (DINOv2):** A ViT-S/14 model that processes a 20-frame rolling buffer of images to extract 384-dimensional visual embeddings.
- **Telemetry Context:** GPS-derived distance and heading signals that condition the policy.
- **Safety Critic:** A binary classifier that estimates whether the rover is stuck or in danger.
- **Action Policy (MoE):** A mixture-of-experts head that predicts a 10-step future action chunk for throttle and steering.

The temporal core differs by model:

- **JEPA model:** Uses a causal Transformer encoder plus a self-supervised predictor trained with a JEPA/VICReg objective to learn predictive latent dynamics.
- **LSTM model:** Uses a bidirectional recurrent encoder to model the same temporal feature sequence as a reactive baseline.

The project evaluates these models on success rate, path efficiency, average collisions per run, collisions per 1000 timesteps, average collision speed, and average speed near obstacles.

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

When evaluating the two models, the preprocessing step is shared. Each each model has its own training and visualization entrypoint.

### 1. Data Generation

Generate synthetic driving data and telemetry to test the pipeline:

```bash
python generate_synth_data.py
```

Use this only for the initial data creation step; the rest of the workflow consumes the generated raw files or packed dataset.

### 2. Preprocessing

Pack raw video (`.mp4`/`.avi`) and telemetry (`.csv`) files into an efficient, memory-mapped binary JPEG format for high-speed training:

```bash
python DR-JEPA6.py preprocess --data_dir /path/to/raw_data --output /path/to/packed_dataset
```

### 3. Train the JEPA model

Train the predictive world-model baseline using the packed binary dataset. The script handles DINOv2 freezing/unfreezing, JEPA masking, loss weighting, and early stopping. NOTE: This process can take several hours or days on low-end hardware, consider only running a few epochs using `--epochs <int>` for testing. The best and most recent model checkpoints will be saved to the specified `--save_dir` for later evaluation and visualization. Useful training metrics will be saved in `training_metrics.csv` to help understand how the model is learning over time.

```bash
python DR-JEPA6.py train --dataset /path/to/packed_dataset --save_dir runs/
```

### 4. Train the LSTM baseline

Train the recurrent reactive baseline with the matching packed dataset. NOTE: This process can take several hours or days on low-end hardware, consider only running a few epochs using `--epochs <int>` for testing. The best and most recent model checkpoints will be saved to the specified `--save_dir` for later evaluation and visualization. Useful training metrics will be saved in `training_metrics.csv` to help understand how the model is learning over time.

```bash
python DR-LSTM.py train --dataset /path/to/packed_dataset --save_dir runs_lstm/
```

### 5. Visualization (Open Loop)

Run deterministic open-loop visualization over a specific video, rendering a Heads-Up Display (HUD) showing ground truth vs. model predictions and the active expert routing.

```bash
python DR-JEPA6.py viz --video /path/to/video.mp4 --checkpoint runs/best_jepa_v2.pth
```

For the LSTM baseline, use:

```bash
python DR-LSTM.py viz --video /path/to/video.mp4 --checkpoint runs_lstm/best_lstm_v2.pth
```

### 6. Live Inference

Test the closed-loop real-time deployment path for either model using the shared live simulation script:

```bash
python jepa_live_inference_test.py --checkpoint runs/best_jepa_v2.pth --model jepa
python jepa_live_inference_test.py --checkpoint runs_lstm/best_lstm_v2.pth --model lstm
```
