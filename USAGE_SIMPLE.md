# Simple Usage - Feedback Action Freezing

## How to Control Training Stages

### 1. Toggle the `tune_feedback` switch

In your training script or model initialization, add:

```python
# STAGE 1: Freeze feedback_action
model.feedback_action.set_trainable(False)

# STAGE 2: Train only feedback_action  
model.feedback_action.set_trainable(True)
```

### 2. What the debug output shows

The debug print runs every 100 forward passes and shows:

```
[FeedbackAction] Status: trainable=False, params with grad=0/XX, training_mode=True
```

Meaning:
- **trainable**: Whether the module should be trainable (from `tune_feedback` flag)
- **params with grad**: How many parameters have `requires_grad=True` / total parameters
- **training_mode**: Whether the model is in training mode (from PyTorch)

### 3. Expected Outputs

**Stage 1 (Frozen):**
```
FeedbackAction trainable: False
[FeedbackAction] Status: trainable=False, params with grad=0/XX, training_mode=True
```

**Stage 2 (Training):**
```
FeedbackAction trainable: True
[FeedbackAction] Status: trainable=True, params with grad=XX/XX, training_mode=True
```

### 4. Manual Control Example

```python
from gr00t.model.gr00t_n1 import GR00T_N1_5

model = GR00T_N1_5.from_pretrained("nvidia/GR00T-N1.5-3B", ...)

# Stage 1: Comment out this line to train feedback_action
model.feedback_action.set_trainable(False)

# Stage 2: Uncomment these lines to freeze everything else
# model.backbone.set_trainable_parameters(tune_visual=False, tune_llm=False)
# model.action_head.set_trainable_parameters(tune_projector=False, tune_diffusion_model=False)
# model.feedback_action.set_trainable(True)

# Train...
```

## That's It!

Just toggle `model.feedback_action.set_trainable(True/False)` to control training!

