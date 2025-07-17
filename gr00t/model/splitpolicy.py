## combine gr00t_n1.py and policy.py for 2 brain inference

import json
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, Optional, Union

import numpy as np
import torch
from huggingface_hub import snapshot_download
from huggingface_hub.errors import HFValidationError, RepositoryNotFoundError

from gr00t.data.dataset import ModalityConfig
from gr00t.data.embodiment_tags import EmbodimentTag
from gr00t.data.schema import DatasetMetadata
from gr00t.data.transform.base import ComposedModalityTransform

COMPUTE_DTYPE = torch.bfloat16



from dataclasses import dataclass, field
from typing import Tuple

import numpy as np
import torch
import tree
from huggingface_hub import snapshot_download
from huggingface_hub.errors import HFValidationError, RepositoryNotFoundError
from transformers import AutoConfig, AutoModel, PretrainedConfig, PreTrainedModel
from transformers.feature_extraction_utils import BatchFeature

from .action_head.flow_matching_action_head import (
    FlowmatchingActionHead,
    FlowmatchingActionHeadConfig,
)
from .backbone import EagleBackbone

#######################################################################################################

BACKBONE_FEATURE_KEY = "backbone_features"
ACTION_KEY = "action_pred"
LOSS_KEY = "loss"
ERROR_MSG = "Error: unexpected input/output"
N_COLOR_CHANNELS = 3


# config
@dataclass
class GR00T_N1Config(PretrainedConfig):
    model_type = "gr00t_n1"
    backbone_cfg: dict = field(init=False, metadata={"help": "Backbone configuration."})

    action_head_cfg: dict = field(init=False, metadata={"help": "Action head configuration."})

    action_horizon: int = field(init=False, metadata={"help": "Action horizon."})

    action_dim: int = field(init=False, metadata={"help": "Action dimension."})
    compute_dtype: str = field(default="float32", metadata={"help": "Compute dtype."})

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        for key, value in kwargs.items():
            setattr(self, key, value)


# real model
class GR00T_N1(PreTrainedModel):
    supports_gradient_checkpointing = True
    config_class = GR00T_N1Config
    """
    we expect the backbone output to have a key 'backbone_features' with shape (batch_size, n, hidden_size)
    here n is variable and can be e.g. time, 1 or user specified
    we expect the action head output to have a key 'action_pred' with shape (batch_size, time, action_dim) during inference time
    we expect these to have type BatchFeature, and they can of course have many other user specified keys too
    """

    def __init__(
        self,
        config: GR00T_N1Config,
        local_model_path: str,
    ):
        assert isinstance(config.backbone_cfg, dict)
        assert isinstance(config.action_head_cfg, dict)

        super().__init__(config)
        self.local_model_path = local_model_path

        self.backbone = EagleBackbone(**config.backbone_cfg)
        action_head_cfg = FlowmatchingActionHeadConfig(**config.action_head_cfg)
        self.action_head = FlowmatchingActionHead(action_head_cfg)

        self.action_horizon = config.action_horizon
        self.action_dim = config.action_dim
        self.compute_dtype = config.compute_dtype

    def validate_inputs(self, inputs):
        # NOTE -- this should be handled internally by the model
        # however, doing that will likely be breaking changes -- so we'll need to do it after the deadline

        detected_error = False
        error_msg = ERROR_MSG
        if "action" in inputs:
            action = inputs["action"]
            type_ok = isinstance(action, torch.Tensor)
            shape_ok = (
                len(action.shape) == 3
                and action.shape[1] == self.action_horizon
                and action.shape[2] == self.action_dim
            )
            if not type_ok:
                error_msg += f"\n{action.dtype=}"
                detected_error = True
            if not shape_ok:
                error_msg += f"\n{action.shape=}"
                detected_error = True

        if "video" in inputs:
            video = inputs["video"]
            type_ok = isinstance(video, np.ndarray)
            dtype_ok = video.dtype == np.uint8
            shape_ok = len(video.shape) == 6 and video.shape[3] == N_COLOR_CHANNELS
            if not type_ok:
                error_msg += f"\n{type(video)=}"
                detected_error = True
            if not dtype_ok:
                error_msg += f"\n{video.dtype=}"
                detected_error = True
            if not shape_ok:
                error_msg += f"\n{video.shape=}"
                detected_error = True

        if detected_error:
            raise ValueError(error_msg)

    def validate_data(self, action_head_outputs, backbone_outputs, is_training):
        fail_backbone = (
            not isinstance(backbone_outputs, BatchFeature)
            or BACKBONE_FEATURE_KEY not in backbone_outputs
        )

        if fail_backbone:
            error_msg = ERROR_MSG
            error_msg += f"\n{isinstance(backbone_outputs, BatchFeature)=}"
            error_msg += f"\n{BACKBONE_FEATURE_KEY in backbone_outputs=}"
            error_msg += f"\n{backbone_outputs[BACKBONE_FEATURE_KEY].shape=}"
            raise ValueError(error_msg)

        fail_action_head = (not isinstance(action_head_outputs, BatchFeature)) or not (
            (
                LOSS_KEY in action_head_outputs and is_training
            )  # there might not be an action prediction during training
            or (
                ACTION_KEY in action_head_outputs
                and action_head_outputs[ACTION_KEY].shape[1] == self.action_horizon
                and action_head_outputs[ACTION_KEY].shape[2] == self.action_dim
            )
        )

        if fail_action_head:
            error_msg = ERROR_MSG
            error_msg += f"\n{isinstance(action_head_outputs, BatchFeature)=}"
            error_msg += f"\n{LOSS_KEY in action_head_outputs=}"
            error_msg += f"\n{action_head_outputs[ACTION_KEY].shape=}"
            error_msg += f"\n{self.action_horizon=}"
            error_msg += f"\n{self.action_dim=}"
            raise ValueError(error_msg)

    def forward(
        self,
        inputs: dict,
    ) -> BatchFeature:
        backbone_inputs, action_inputs = self.prepare_input(inputs)
        backbone_outputs = self.backbone(backbone_inputs)
        action_head_outputs = self.action_head(backbone_outputs, action_inputs)
        self.validate_data(action_head_outputs, backbone_outputs, is_training=True)
        return action_head_outputs

    def get_action(
        self,
        inputs: dict,
    ) -> BatchFeature:
        backbone_inputs, action_inputs = self.prepare_input(inputs)
        print("Run Original VLM+DiT Model")
        # Time the backbone (VLM) inference with proper GPU synchronization
        torch.cuda.synchronize()
        backbone_start_time = time.time()
        backbone_outputs = self.backbone(backbone_inputs)
        torch.cuda.synchronize()
        backbone_end_time = time.time()
        backbone_inference_time = backbone_end_time - backbone_start_time
        
        # Time the action head (DiT) inference with proper GPU synchronization
        torch.cuda.synchronize()
        action_head_start_time = time.time()
        action_head_outputs = self.action_head.get_action(backbone_outputs, action_inputs)
        torch.cuda.synchronize()
        action_head_end_time = time.time()
        action_head_inference_time = action_head_end_time - action_head_start_time
        
        # Store timing information in the outputs
        action_head_outputs["backbone_inference_time"] = backbone_inference_time
        action_head_outputs["action_head_inference_time"] = action_head_inference_time
        action_head_outputs["total_inference_time"] = backbone_inference_time + action_head_inference_time
        
        self.validate_data(action_head_outputs, backbone_outputs, is_training=False)
        return action_head_outputs
    

    def get_action_fast(
        self,
        inputs: dict,
        time_step: int,
    ) -> BatchFeature:
        backbone_inputs, action_inputs = self.prepare_input(inputs)
        print("Run Fast Model")
        if time_step % 16 == 0:
            # Run both backbone and action_head
            print(f"hey, im at {time_step}")
            print("im at 320")
            torch.cuda.synchronize()
            backbone_start_time = time.time()
            backbone_outputs = self.backbone(backbone_inputs)
            torch.cuda.synchronize()
            backbone_end_time = time.time()
            backbone_inference_time = backbone_end_time - backbone_start_time

            self._cached_backbone_outputs = backbone_outputs

            torch.cuda.synchronize()
            action_head_start_time = time.time()
            action_head_outputs = self.action_head.get_action(backbone_outputs, action_inputs)
            torch.cuda.synchronize()
            action_head_end_time = time.time()
            action_head_inference_time = action_head_end_time - action_head_start_time

            action_head_outputs["backbone_inference_time"] = backbone_inference_time
            action_head_outputs["action_head_inference_time"] = action_head_inference_time
            action_head_outputs["total_inference_time"] = backbone_inference_time + action_head_inference_time

            self.validate_data(action_head_outputs, backbone_outputs, is_training=False)
            return action_head_outputs
        else:
            print(f"hey, im at {time_step}") 
            # Reuse cached backbone outputs, only run action_head
            if not hasattr(self, '_cached_backbone_outputs'):
                raise ValueError(f"No cached backbone outputs available at timestep {time_step}")
            
            torch.cuda.synchronize()
            action_head_start_time = time.time()
            action_head_outputs = self.action_head.get_action(self._cached_backbone_outputs, action_inputs)
            torch.cuda.synchronize()
            action_head_end_time = time.time()
            action_head_inference_time = action_head_end_time - action_head_start_time

            action_head_outputs["backbone_inference_time"] = 0.0
            action_head_outputs["action_head_inference_time"] = action_head_inference_time
            action_head_outputs["total_inference_time"] = action_head_inference_time

            self.validate_data(action_head_outputs, self._cached_backbone_outputs, is_training=False)
            return action_head_outputs
            
            



    def prepare_input(self, inputs) -> Tuple[BatchFeature, BatchFeature]:
        self.validate_inputs(inputs)
        backbone_inputs = self.backbone.prepare_input(inputs)
        action_inputs = self.action_head.prepare_input(inputs)

        def to_device_with_maybe_dtype(x):
            # Only cast to self.compute_dtype if the tensor is floating
            if torch.is_floating_point(x):
                return x.to(self.device, dtype=self.action_head.dtype)
            else:
                # Keep original dtype
                return x.to(self.device)

        backbone_inputs = tree.map_structure(to_device_with_maybe_dtype, backbone_inputs)
        action_inputs = tree.map_structure(to_device_with_maybe_dtype, action_inputs)
        return backbone_inputs, action_inputs

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: str, **kwargs):
        tune_visual = kwargs.pop("tune_visual", True)
        tune_llm = kwargs.pop("tune_llm", False)
        tune_projector = kwargs.pop("tune_projector", True)
        tune_diffusion_model = kwargs.pop("tune_diffusion_model", True)

        print(f"Loading pretrained dual brain from {pretrained_model_name_or_path}")
        print(f"Tune backbone vision tower: {tune_visual}")
        print(f"Tune backbone LLM: {tune_llm}")
        print(f"Tune action head projector: {tune_projector}")
        print(f"Tune action head DiT: {tune_diffusion_model}")

        # get the current model path being downloaded
        try:
            # NOTE(YL) This downloads the model to the local cache and returns the local path to the model
            # saved in ~/.cache/huggingface/hub/
            local_model_path = snapshot_download(pretrained_model_name_or_path, repo_type="model")
            # HFValidationError, RepositoryNotFoundError
        except (HFValidationError, RepositoryNotFoundError):
            print(
                f"Model not found or avail in the huggingface hub. Loading from local path: {pretrained_model_name_or_path}"
            )
            local_model_path = pretrained_model_name_or_path

        pretrained_model = super().from_pretrained(
            local_model_path, local_model_path=local_model_path, **kwargs
        )

        pretrained_model.backbone.set_trainable_parameters(
            tune_visual=tune_visual, tune_llm=tune_llm
        )
        pretrained_model.action_head.set_trainable_parameters(
            tune_projector=tune_projector, tune_diffusion_model=tune_diffusion_model
        )
        return pretrained_model


# register
AutoConfig.register("gr00t_n1", GR00T_N1Config)
AutoModel.register(GR00T_N1Config, GR00T_N1)

#######################################################################################################


class BasePolicy(ABC):
    @abstractmethod
    def get_action(self, observations: Dict[str, Any]) -> Dict[str, Any]:
        """
        Abstract method to get the action for a given state.

        Args:
            observations: The observations from the environment.

        Returns:
            The action to take in the environment in dictionary format.
        """
        raise NotImplementedError

    @abstractmethod
    def get_modality_config(self) -> Dict[str, ModalityConfig]:
        """
        Return the modality config of the policy.
        """
        raise NotImplementedError


class SplitPolicy(BasePolicy):
    """
    A wrapper for Gr00t model checkpoints that handles loading the model, applying transforms,
    making predictions, and unapplying transforms. This loads some custom configs, stats
    and metadata related to the model checkpoints used
    in the Gr00t model.
    """

    def __init__(
        self,
        model_path: str,
        embodiment_tag: Union[str, EmbodimentTag],
        modality_config: Dict[str, ModalityConfig],
        modality_transform: ComposedModalityTransform,
        denoising_steps: Optional[int] = None,
        device: Union[int, str] = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        """
        Initialize the SplitPolicy.

        Args:
            model_path (str): Path to the model checkpoint directory or the huggingface hub id.
            modality_config (Dict[str, ModalityConfig]): The modality config for the model.
            modality_transform (ComposedModalityTransform): The modality transform for the model.
            embodiment_tag (Union[str, EmbodimentTag]): The embodiment tag for the model.
            denoising_steps: Number of denoising steps to use for the action head.
            device (Union[int, str]): Device to run the model on.
        """
        try:
            # NOTE(YL) this returns the local path to the model which is normally
            # saved in ~/.cache/huggingface/hub/
            model_path = snapshot_download(model_path, repo_type="model")
            # HFValidationError, RepositoryNotFoundError
        except (HFValidationError, RepositoryNotFoundError):
            print(
                f"Model not found or avail in the huggingface hub. Loading from local path: {model_path}"
            )

        self._modality_config = modality_config
        self._modality_transform = modality_transform
        self._modality_transform.eval()  # set this to eval mode
        self.model_path = Path(model_path)
        self.device = device

        # Convert string embodiment tag to EmbodimentTag enum if needed
        if isinstance(embodiment_tag, str):
            self.embodiment_tag = EmbodimentTag(embodiment_tag)
        else:
            self.embodiment_tag = embodiment_tag

        # Load model
        self._load_model(model_path)
        # Load transforms
        self._load_metadata(self.model_path / "experiment_cfg")
        # Load horizons
        self._load_horizons()

        if denoising_steps is not None:
            if hasattr(self.model, "action_head") and hasattr(
                self.model.action_head, "num_inference_timesteps"
            ):
                self.model.action_head.num_inference_timesteps = denoising_steps
                print(f"Set action denoising steps to {denoising_steps}")

    def apply_transforms(self, obs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Apply transforms to the observation.

        Args:
            obs (Dict[str, Any]): The observation to transform.

        Returns:
            Dict[str, Any]: The transformed observation.
        """
        # Ensure correct dimensions before applying transforms
        return self._modality_transform(obs)

    def unapply_transforms(self, action: Dict[str, Any]) -> Dict[str, Any]:
        """
        Unapply transforms to the action.

        Args:
            action (Dict[str, Any]): The action to unapply transforms to.

        Returns:
            Dict[str, Any]: The untransformed action.
        """
        return self._modality_transform.unapply(action)

    def get_action(self, observations: Dict[str, Any], time_step: int, select_model_fast: bool = False, print_timing: bool = False) -> Dict[str, Any]:
        """
        Make a prediction with the model.
        Args:
            obs (Dict[str, Any]): The observation to make a prediction for.
            print_timing (bool): Whether to print timing statistics for this inference call.

        e.g. obs = {
            "video.<>": np.ndarray,  # (T, H, W, C)
            "state.<>": np.ndarray, # (T, D)
        }

        or with batched input:
        e.g. obs = {
            "video.<>": np.ndarray,, # (B, T, H, W, C)
            "state.<>": np.ndarray, # (B, T, D)
        }

        Returns:
            Dict[str, Any]: The predicted action.
        """
        # let the get_action handles both batch and single input
        is_batch = self._check_state_is_batched(observations)
        if not is_batch:
            observations = unsqueeze_dict_values(observations)

        # Apply transforms
        normalized_input = self.apply_transforms(observations)

        normalized_action = self._get_action_from_normalized_input(normalized_input, time_step, select_model_fast)
        unnormalized_action = self._get_unnormalized_action(normalized_action)

        if not is_batch:
            unnormalized_action = squeeze_dict_values(unnormalized_action)
            
        # Print timing if requested
        if print_timing:
            self.print_timing_stats()
            
        return unnormalized_action

    def _get_action_from_normalized_input(self, normalized_input: Dict[str, Any], time_step: int, select_model_fast: bool = False) -> torch.Tensor:
        # Set up autocast context if needed
        if select_model_fast:
            with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=COMPUTE_DTYPE):
                model_pred = self.model.get_action_fast(normalized_input, time_step)
        else:
            with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=COMPUTE_DTYPE):
                model_pred = self.model.get_action(normalized_input)

        # Store timing information for later access
        self._last_model_output = model_pred
        
        normalized_action = model_pred["action_pred"].float()
        return normalized_action

    def _get_unnormalized_action(self, normalized_action: torch.Tensor) -> Dict[str, Any]:
        return self.unapply_transforms({"action": normalized_action.cpu()})

    def get_modality_config(self) -> Dict[str, ModalityConfig]:
        """
        Get the modality config for the model, overrides the base class method
        """
        return self._modality_config

    @property
    def modality_config(self) -> Dict[str, ModalityConfig]:
        return self._modality_config

    @property
    def modality_transform(self) -> ComposedModalityTransform:
        return self._modality_transform

    @property
    def video_delta_indices(self) -> np.ndarray:
        """Get the video delta indices."""
        return self._video_delta_indices

    @property
    def state_delta_indices(self) -> np.ndarray | None:
        """Get the state delta indices."""
        return self._state_delta_indices

    @property
    def denoising_steps(self) -> int:
        """Get the number of denoising steps."""
        return self.model.action_head.num_inference_timesteps

    @denoising_steps.setter
    def denoising_steps(self, value: int):
        """Set the number of denoising steps."""
        self.model.action_head.num_inference_timesteps = value

    def _check_state_is_batched(self, obs: Dict[str, Any]) -> bool:
        for k, v in obs.items():
            if "state" in k and len(v.shape) < 3:  # (B, Time, Dim)
                return False
        return True

    def _load_model(self, model_path):
        """Load the GR00T_N1 model."""
        model = GR00T_N1.from_pretrained(model_path, torch_dtype=COMPUTE_DTYPE)
        model.eval()  # Set model to eval mode
        model.to(device=self.device)  # type: ignore

        self.model = model

    def _load_metadata(self, exp_cfg_dir: Path):
        """Load the transforms for the model."""
        # Load metadata for normalization stats
        metadata_path = exp_cfg_dir / "metadata.json"
        with open(metadata_path, "r") as f:
            metadatas = json.load(f)

        # Get metadata for the specific embodiment
        metadata_dict = metadatas.get(self.embodiment_tag.value)
        if metadata_dict is None:
            raise ValueError(
                f"No metadata found for embodiment tag: {self.embodiment_tag.value}",
                f"make sure the metadata.json file is present at {metadata_path}",
            )

        metadata = DatasetMetadata.model_validate(metadata_dict)
        print("path ", self.model_path)

        self._modality_transform.set_metadata(metadata)
        self.metadata = metadata

    def _load_horizons(self):
        """Load the horizons needed for the model."""
        # Get modality configs
        # Video horizons
        self._video_delta_indices = np.array(self._modality_config["video"].delta_indices)
        self._assert_delta_indices(self._video_delta_indices)
        self._video_horizon = len(self._video_delta_indices)
        # State horizons (if used)
        if "state" in self._modality_config:
            self._state_delta_indices = np.array(self._modality_config["state"].delta_indices)
            self._assert_delta_indices(self._state_delta_indices)
            self._state_horizon = len(self._state_delta_indices)
        else:
            self._state_horizon = None
            self._state_delta_indices = None

    def _assert_delta_indices(self, delta_indices: np.ndarray):
        """Assert that the delta indices are valid."""
        # All delta indices should be non-positive because there's no way to get the future observations
        assert np.all(delta_indices <= 0), f"{delta_indices=}"
        # The last delta index should be 0 because it doesn't make sense to not use the latest observation
        assert delta_indices[-1] == 0, f"{delta_indices=}"
        if len(delta_indices) > 1:
            # The step is consistent
            assert np.all(
                np.diff(delta_indices) == delta_indices[1] - delta_indices[0]
            ), f"{delta_indices=}"
            # And the step is positive
            assert (delta_indices[1] - delta_indices[0]) > 0, f"{delta_indices=}"

    def get_timing_stats(self, model_output: Dict[str, Any]) -> Dict[str, float]:
        """
        Extract timing from model output.

        Args:
            model_output (Dict[str, Any]): The output from the model.

        Returns:
            Dict[str, float]: The timing statistics.
        """
        return {
            "backbone_inference_time": model_output.get("backbone_inference_time", 0),
            "action_head_inference_time": model_output.get("action_head_inference_time", 0),
            "total_inference_time": model_output.get("total_inference_time", 0),
        }

    def get_last_timing_stats(self) -> Dict[str, float]:
        """
        Get timing from the last inference call.

        Returns:
            Dict[str, float]: The timing statistics.
        """
        return self.get_timing_stats(self._last_model_output)

    def print_timing_stats(self):
        """Print timing statistics for the last inference call."""
        timing_stats = self.get_last_timing_stats()
        print("Timing statistics for the last inference call:")
        for stat, value in timing_stats.items():
            print(f"{stat}: {value:.4f} seconds")

    def get_timing_stats_string(self) -> str:
        """Get timing statistics as a formatted string for the last inference call."""
        timing_stats = self.get_last_timing_stats()
        lines = ["Timing statistics for the last inference call:"]
        for stat, value in timing_stats.items():
            lines.append(f"{stat}: {value:.4f} seconds")
        return "\n".join(lines)


#######################################################################################################


# Helper functions
def unsqueeze_dict_values(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Unsqueeze the values of a dictionary.
    This converts the data to be batched of size 1.
    """
    unsqueezed_data = {}
    for k, v in data.items():
        if isinstance(v, np.ndarray):
            unsqueezed_data[k] = np.expand_dims(v, axis=0)
        elif isinstance(v, torch.Tensor):
            unsqueezed_data[k] = v.unsqueeze(0)
        else:
            unsqueezed_data[k] = v
    return unsqueezed_data


def squeeze_dict_values(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Squeeze the values of a dictionary. This removes the batch dimension.
    """
    squeezed_data = {}
    for k, v in data.items():
        if isinstance(v, np.ndarray):
            squeezed_data[k] = np.squeeze(v)
        elif isinstance(v, torch.Tensor):
            squeezed_data[k] = v.squeeze()
        else:
            squeezed_data[k] = v
    return squeezed_data