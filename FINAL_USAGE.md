# Final Usage - Feedback Action Freezing

## What Changed

1. **Removed `set_frozen_modules_to_eval_mode()`** - It was causing NaN issues
2. **Added automatic freezing** - `feedback_action` is now frozen by default when model loads
3. **Simple control** - Just use `set_trainable()` to freeze/unfreeze

## Default Behavior

When you load the model, **feedback_action is frozen by default**:

```python
model = GR00T_N1_5.from_pretrained("nvidia/GR00T-N1.5-3B", ...)
# feedback_action is already frozen!
```

Output:
```
FeedbackAction trainable: False
```

## Stage 1: Train Everything Except Feedback (Default)

```python
# Load model - feedback_action is already frozen
model = GR00T_N1_5.from_pretrained("nvidia/GR00T-N1.5-3B", ...)

# Train normally - feedback_action won't update
# ... your training loop ...
```

## Stage 2: Train Only Feedback Action

```python
# Load checkpoint from Stage 1
model = GR00T_N1_5.from_pretrained("./checkpoints/stage1/checkpoint-40000", ...)

# Freeze everything else
model.backbone.set_trainable_parameters(tune_visual=False, tune_llm=False)
model.action_head.set_trainable_parameters(tune_projector=False, tune_diffusion_model=False)

# Unfreeze feedback_action
model.feedback_action.set_trainable(True)

# Train - only feedback_action will update
# ... your training loop ...
```

## Manual Override (If Needed)

If you want to train feedback_action from the start:

```python
model = GR00T_N1_5.from_pretrained("nvidia/GR00T-N1.5-3B", ...)
model.feedback_action.set_trainable(True)  # Unfreeze it
```

## Summary

- **Default**: `feedback_action` is **frozen** when model loads
- **Stage 1**: Train everything except feedback_action (default behavior)
- **Stage 2**: Unfreeze feedback_action with `model.feedback_action.set_trainable(True)`
- **No NaN issues**: Removed the eval mode switching that was causing problems

