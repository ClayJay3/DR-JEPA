"""DR-JEPA: camera + goal-vector autonomous rover navigation.

Package layout:
    config     -- all hyperparameters (data, model, training, simulation)
    simulator  -- domain-randomized 3D world, rover physics, noisy sensors
    expert     -- arc-sampling local planner that generates demonstrations
    model      -- RoverJEPA: frozen DINOv2 + temporal encoder + JEPA world
                  model + policy/safety heads
    dataset    -- feature preprocessing (packing) and the training Dataset
    pilot      -- streaming closed-loop inference wrapper (one frame per step)
"""
