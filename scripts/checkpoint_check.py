# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Script to check if a checkpoint (e.g., updater_obs_feedbackb) is outputting actions correctly.
This script loads a checkpoint, runs inference with dummy or real observations, and prints the actions.
"""

import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import tyro

import gr00t
from gr00t.experiment.data_config import DATA_CONFIG_MAP
from gr00t.model.policy import Gr00tPolicy


@dataclass
class ArgsConfig:
    """Configuration for checkpoint checking."""

    checkpoint_path: str
    """Path to the checkpoint directory or file containing 'updater_obs_feedbackb'."""

    data_config: str = "libero_arm"
    """Data config to use for loading the model (determines modality config)."""

    embodiment_tag: str = "libero_arm"
    """Embodiment tag to use for the model."""

    denoising_steps: Optional[int] = None
    """Number of denoising steps. If None, uses model default."""

    use_dummy_obs: bool = False
    """If True, uses dummy observations. If False, uses demo_data."""

    demo_data_path: Optional[str] = None
    """Path to demo_data directory. If None, will search for demo_data in common locations."""

    device: str = "cuda"
    """Device to run inference on."""


def find_checkpoint_path(search_path: str) -> str:
    """
    Find checkpoint path that contains 'updater_obs_feedbackb' in the path.
    If the path already exists, return it. Otherwise, search for it.
    """
    if os.path.exists(search_path):
        # Check if it's a directory or file
        if os.path.isdir(search_path):
            # Check if it contains the keyword
            if "updater_obs_feedbackb" in search_path:
                return search_path
            # Search in subdirectories
            for root, dirs, files in os.walk(search_path):
                if "updater_obs_feedbackb" in root:
                    return root
                for file in files:
                    if "updater_obs_feedbackb" in file:
                        return os.path.join(root, file)
        else:
            # It's a file
            if "updater_obs_feedbackb" in search_path:
                return search_path
            # Check parent directory
            parent = os.path.dirname(search_path)
            if "updater_obs_feedbackb" in parent:
                return parent

    # Search in common checkpoint locations
    search_dirs = [
        "./checkpoints",
        "./outputs",
        "./logs",
        "~/.cache/huggingface/hub",
    ]
    for search_dir in search_dirs:
        expanded_dir = os.path.expanduser(search_dir)
        if os.path.exists(expanded_dir):
            for root, dirs, files in os.walk(expanded_dir):
                if "updater_obs_feedbackb" in root:
                    return root

    # If not found, return original path (will error if invalid)
    return search_path


def find_demo_data_path(data_config: str, embodiment_tag: str) -> Optional[str]:
    """
    Find demo_data path based on data_config and embodiment_tag.
    Common locations:
    - ./demo_data/
    - ../demo_data/
    - demo_data/ (relative to repo root)
    """
    # Common demo_data subdirectories based on embodiment
    demo_data_mapping = {
        "libero_arm": "libero_object_data",
        "gr1": "robot_sim.PickNPlace",
        "so100": "robot_sim.PickNPlace",
    }

    # Try to find demo_data directory
    repo_root = os.path.dirname(os.path.dirname(gr00t.__file__))
    search_paths = [
        "./demo_data",
        "../demo_data",
        os.path.join(os.path.dirname(os.path.dirname(__file__)), "demo_data"),
        os.path.join(repo_root, "demo_data"),
    ]

    for base_path in search_paths:
        expanded_path = os.path.expanduser(base_path)
        if os.path.exists(expanded_path):
            # Try embodiment-specific subdirectory
            if embodiment_tag in demo_data_mapping:
                subdir = demo_data_mapping[embodiment_tag]
                full_path = os.path.join(expanded_path, subdir)
                if os.path.exists(full_path):
                    return full_path

            # Try data_config-based subdirectory
            if data_config in demo_data_mapping:
                subdir = demo_data_mapping[data_config]
                full_path = os.path.join(expanded_path, subdir)
                if os.path.exists(full_path):
                    return full_path

            # Return base demo_data if it exists
            return expanded_path

    return None


def create_dummy_observations(modality_config: dict) -> dict:
    """Create dummy observations based on modality config."""
    obs = {}

    for key, config in modality_config.items():
        if key.startswith("video."):
            # Video observation: (T, H, W, C)
            obs[key] = np.random.randint(0, 256, (1, 256, 256, 3), dtype=np.uint8)
        elif key.startswith("state."):
            # State observation: (T, D) where D is state dimension
            # Try to infer dimension from key name or use default
            if "arm" in key:
                dim = 7
            elif "hand" in key:
                dim = 6
            elif "waist" in key:
                dim = 3
            else:
                dim = 10  # default
            obs[key] = np.random.rand(1, dim).astype(np.float32)
        elif key.startswith("annotation."):
            # Annotation (usually text)
            obs[key] = ["test action description"]

    return obs


def main(config: ArgsConfig):
    """Main function to check checkpoint action output."""
    print("=" * 80)
    print("Checkpoint Action Output Checker")
    print("=" * 80)

    # Find checkpoint path
    print(f"\nSearching for checkpoint containing 'updater_obs_feedbackb'...")
    checkpoint_path = find_checkpoint_path(config.checkpoint_path)
    print(f"Using checkpoint path: {checkpoint_path}")

    if not os.path.exists(checkpoint_path):
        print(f"ERROR: Checkpoint path does not exist: {checkpoint_path}")
        sys.exit(1)

    # Get data config
    if config.data_config not in DATA_CONFIG_MAP:
        print(f"ERROR: Unknown data config '{config.data_config}'")
        print(f"Available configs: {list(DATA_CONFIG_MAP.keys())}")
        sys.exit(1)

    data_config_cls = DATA_CONFIG_MAP[config.data_config]
    modality_config = data_config_cls.modality_config()
    modality_transform = data_config_cls.transform()

    print(f"\nUsing data config: {config.data_config}")
    print(f"Embodiment tag: {config.embodiment_tag}")

    # Load policy
    print("\nLoading policy...")
    try:
        policy = Gr00tPolicy(
            model_path=checkpoint_path,
            modality_config=modality_config,
            modality_transform=modality_transform,
            embodiment_tag=config.embodiment_tag,
            denoising_steps=config.denoising_steps,
            device=config.device,
        )
        # Explicitly set model to eval mode
        policy.model.eval()
        if hasattr(policy.model, "backbone"):
            policy.model.backbone.eval()
        if hasattr(policy.model, "action_head"):
            policy.model.action_head.eval()
        print("✓ Policy loaded successfully")
        print("✓ Model set to eval mode")
    except Exception as e:
        print(f"ERROR: Failed to load policy: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)

    # Get modality config from policy
    policy_modality = policy.get_modality_config()
    print("\nPolicy modality config:")
    for key, mod_config in policy_modality.items():
        print(f"  {key}: {mod_config}")

    # Create or load observations
    if config.use_dummy_obs:
        print("\nCreating dummy observations...")
        observations = create_dummy_observations(policy_modality)
    else:
        # Find demo_data path
        if config.demo_data_path is None:
            print("\nSearching for demo_data...")
            demo_data_path = find_demo_data_path(config.data_config, config.embodiment_tag)
            if demo_data_path is None:
                print("ERROR: Could not find demo_data directory.")
                print("Please provide --demo-data-path or ensure demo_data exists in:")
                print("  - ./demo_data/")
                print("  - ../demo_data/")
                print("  - Or specify the full path with --demo-data-path")
                sys.exit(1)
            config.demo_data_path = demo_data_path

        print(f"\nLoading observations from demo_data: {config.demo_data_path}")
        from gr00t.data.dataset import LeRobotSingleDataset

        try:
            dataset = LeRobotSingleDataset(
                dataset_path=config.demo_data_path,
                modality_configs=policy_modality,
                video_backend="decord",
                video_backend_kwargs=None,
                transforms=None,
                embodiment_tag=config.embodiment_tag,
            )
            if len(dataset) == 0:
                print(f"ERROR: Dataset at {config.demo_data_path} is empty")
                sys.exit(1)
            observations = dataset[0]
            print(f"✓ Loaded observation from dataset (total trajectories: {len(dataset.trajectory_lengths)})")
        except Exception as e:
            print(f"ERROR: Failed to load dataset from {config.demo_data_path}: {e}")
            import traceback

            traceback.print_exc()
            sys.exit(1)

    print("\nInput observations:")
    has_nan_in_input = False
    for key, value in observations.items():
        if isinstance(value, np.ndarray):
            nan_count = np.isnan(value).sum() if np.issubdtype(value.dtype, np.floating) else 0
            inf_count = np.isinf(value).sum() if np.issubdtype(value.dtype, np.floating) else 0
            print(f"  {key}: shape={value.shape}, dtype={value.dtype}")
            if nan_count > 0:
                print(f"    ⚠️  WARNING: Contains {nan_count} NaN values!")
                has_nan_in_input = True
            if inf_count > 0:
                print(f"    ⚠️  WARNING: Contains {inf_count} Inf values!")
                has_nan_in_input = True
            if not has_nan_in_input:
                print(f"    Min: {value.min():.6f}, Max: {value.max():.6f}, Mean: {value.mean():.6f}")
        elif isinstance(value, list):
            print(f"  {key}: list with {len(value)} items")
            if len(value) > 0:
                print(f"    First item: {value[0]}")
        else:
            print(f"  {key}: {type(value).__name__} = {value}")
    
    if has_nan_in_input:
        print("\n⚠️  WARNING: Input observations contain NaN or Inf values!")
        print("This will likely cause NaN in model outputs.")
        print("Please check your dataset for corrupted or missing data.")

    # Run inference
    print("\n" + "=" * 80)
    print("Running inference to get actions...")
    print("=" * 80)

    try:
        # Ensure model is in eval mode
        policy.model.eval()
        if hasattr(policy.model, "backbone"):
            policy.model.backbone.eval()
        if hasattr(policy.model, "action_head"):
            policy.model.action_head.eval()
            # Also set frozen modules to eval if method exists
            if hasattr(policy.model.action_head, "set_frozen_modules_to_eval_mode"):
                policy.model.action_head.set_frozen_modules_to_eval_mode()
        
        with torch.no_grad():
            actions = policy.get_action(observations, time_step=0)

        print("\n✓ Inference successful!")
        print("\n" + "=" * 80)
        print("OUTPUT ACTIONS:")
        print("=" * 80)

        has_nan_in_output = False
        if isinstance(actions, dict):
            for key, value in actions.items():
                if isinstance(value, np.ndarray):
                    nan_count = np.isnan(value).sum() if np.issubdtype(value.dtype, np.floating) else 0
                    inf_count = np.isinf(value).sum() if np.issubdtype(value.dtype, np.floating) else 0
                    print(f"\n{key}:")
                    print(f"  Shape: {value.shape}")
                    print(f"  Dtype: {value.dtype}")
                    if nan_count > 0:
                        print(f"  ⚠️  WARNING: Contains {nan_count} NaN values!")
                        has_nan_in_output = True
                    if inf_count > 0:
                        print(f"  ⚠️  WARNING: Contains {inf_count} Inf values!")
                        has_nan_in_output = True
                    if nan_count == 0 and inf_count == 0:
                        print(f"  Min: {value.min():.6f}, Max: {value.max():.6f}, Mean: {value.mean():.6f}")
                    print(f"  Values:")
                    # Print first few values
                    if value.ndim == 1:
                        print(f"    {value[:min(10, len(value))]}")
                    elif value.ndim == 2:
                        print(f"    First row: {value[0, :min(10, value.shape[1])]}")
                    else:
                        print(f"    First element: {value.flat[:min(10, value.size)]}")
                elif isinstance(value, torch.Tensor):
                    value_np = value.cpu().numpy()
                    nan_count = np.isnan(value_np).sum() if np.issubdtype(value_np.dtype, np.floating) else 0
                    inf_count = np.isinf(value_np).sum() if np.issubdtype(value_np.dtype, np.floating) else 0
                    print(f"\n{key}:")
                    print(f"  Shape: {value_np.shape}")
                    print(f"  Dtype: {value_np.dtype}")
                    if nan_count > 0:
                        print(f"  ⚠️  WARNING: Contains {nan_count} NaN values!")
                        has_nan_in_output = True
                    if inf_count > 0:
                        print(f"  ⚠️  WARNING: Contains {inf_count} Inf values!")
                        has_nan_in_output = True
                    if nan_count == 0 and inf_count == 0:
                        print(f"  Min: {value_np.min():.6f}, Max: {value_np.max():.6f}, Mean: {value_np.mean():.6f}")
                    print(f"  Values:")
                    if value_np.ndim == 1:
                        print(f"    {value_np[:min(10, len(value_np))]}")
                    elif value_np.ndim == 2:
                        print(f"    First row: {value_np[0, :min(10, value_np.shape[1])]}")
                    else:
                        print(f"    First element: {value_np.flat[:min(10, value_np.size)]}")
                else:
                    print(f"\n{key}: {type(value).__name__} = {value}")
        else:
            print(f"\nActions (not a dict): {type(actions)}")
            print(f"Value: {actions}")

        if has_nan_in_output:
            print("\n" + "=" * 80)
            print("⚠️  WARNING: Checkpoint is outputting NaN/Inf values!")
            print("=" * 80)
            print("\nPossible causes:")
            print("1. Model is in training mode (should be in eval mode)")
            print("2. Input observations contain NaN/Inf values")
            print("3. Numerical instability in the model")
            print("4. Checkpoint is corrupted or incomplete")
            print("5. Mismatch between model architecture and checkpoint")
            print("6. CUDA/GPU issues (try running on CPU to verify)")
            print("\nTroubleshooting steps:")
            print("- Check if input observations have NaN values (see above)")
            print("- Verify the checkpoint was saved correctly")
            print("- Try loading a different checkpoint")
            print("- Check model architecture matches the checkpoint")
        else:
            print("\n" + "=" * 80)
            print("✓ Checkpoint is outputting actions correctly!")
            print("=" * 80)

    except Exception as e:
        print(f"\n✗ ERROR: Failed to get actions: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    config = tyro.cli(ArgsConfig)
    main(config)
