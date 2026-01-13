# How to Set Up Environment

1. Use new `.toml` file (no torch)
2. Refer to [FlashAttention Blackwell Wheels](https://github.com/Zarrac/flashattention-blackwell-wheels-whl-ONLY-5090-5080-5070-5060-flash-attention-/releases/tag/FlashAttention), download PyTorch 2.7.0+cu128, and install FlashAttention for Blackwell
3. Install the package:
   ```bash
   pip install --upgrade setuptools
   pip install -e .[base]
   ``` 

# Checkpoint Check

Run `python scripts/checkpoint_check.py     --checkpoint-path /path/to/checkpoint     --demo-data-path ./demo_data/libero_agent2_data`