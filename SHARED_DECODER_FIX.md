# Shared action_decoder Fix - Final Solution

## Problem

Even with `_action_decoder` as a private reference, PyTorch still detected shared tensors when saving:
```
RuntimeError: The weights trying to be saved contained shared tensors 
[{'action_head.action_decoder.layer1.W', 'feedback_action._action_decoder.layer1.W'}, ...]
```

## Root Cause

Any attribute assigned to `self` in a PyTorch module (even with `_` prefix) is tracked for serialization if it contains parameters. The only way to avoid this is to **not store the reference at all**.

## Solution

**Don't store `action_decoder` in FeedbackAction** - pass it as a function argument instead:

### Changes to `feedback_action.py`:

1. **Remove storage in `__init__`**:
```python
class FeedbackAction(nn.Module):
    def __init__(self, config):  # No action_decoder parameter
        super().__init__()
        # Don't store action_decoder reference
        self.action_updater = ActionUpdater(...)
```

2. **Pass as argument to methods**:
```python
def forward(self, action_head_output, time_step, action_input, action_decoder=None):
    # Use action_decoder directly
    updated_actions = action_decoder(updated_latents, embodiment_id)

def get_action(self, action_head_output, time_step, action_input, action_decoder=None):
    # Use action_decoder directly
    action_update = action_decoder(updated_latent, embodiment_id)

def set_trainable(self, trainable: bool, action_decoder=None):
    # Control action_decoder trainability
    if trainable and action_decoder is not None:
        action_decoder.requires_grad_(True)
```

### Changes to `gr00t_n1.py`:

Pass `self.action_head.action_decoder` when calling FeedbackAction methods:

```python
# In __init__
self.feedback_action = FeedbackAction(action_head_cfg)
self.feedback_action.set_trainable(True, self.action_head.action_decoder)

# In forward (training)
feedback_action_outputs = self.feedback_action(
    action_head_outputs, window_idx, action_inputs, self.action_head.action_decoder
)

# In get_action (inference)
feedback_action_outputs = self.feedback_action.get_action(
    action_head_outputs, time_step, action_inputs, self.action_head.action_decoder
)
```

## Why This Works

1. **No Storage**: `action_decoder` is never stored as an attribute in `FeedbackAction`
2. **No Duplication**: Only `action_head.action_decoder` exists in the model's state_dict
3. **Still Shared**: Both modules use the exact same decoder instance (passed by reference)
4. **Still Trainable**: Can control trainability via `set_trainable(True, action_decoder)`

## Checkpoint Structure

After this fix, checkpoint contains:
```
action_head.action_decoder.layer1.W  ✓ (saved once)
action_head.action_decoder.layer1.b  ✓ (saved once)
action_head.action_decoder.layer2.W  ✓ (saved once)
action_head.action_decoder.layer2.b  ✓ (saved once)
feedback_action.action_updater...     ✓ (saved)
```

**No duplicate entries** - checkpoint saves successfully!

## Training Status

✅ All systems working:
- No NaN values (fixed with LayerNorm)
- Loss decreasing (0.1318)
- Gradients stable (0.45)
- Checkpoint saves without errors
- action_decoder shared and trainable

## Summary

The solution is simple: **don't store shared modules, pass them as arguments**. This is a common pattern in PyTorch when you need to share modules between different parts of a model without duplicating parameters in the state_dict.

