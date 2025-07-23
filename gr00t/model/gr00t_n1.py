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

from dataclasses import dataclass, field
from typing import Tuple, Any
import copy
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

BACKBONE_FEATURE_KEY = "backbone_features"
ACTION_KEY = "action_pred"
LOSS_KEY = "loss"
ERROR_MSG = "Error: unexpected input/output"
N_COLOR_CHANNELS = 3


# config
@dataclass
class GR00T_N1_5_Config(PretrainedConfig):
    model_type = "gr00t_n1_5"
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
class GR00T_N1_5(PreTrainedModel):
    supports_gradient_checkpointing = True
    config_class = GR00T_N1_5_Config
    """
    we expect the backbone output to have a key 'backbone_features' with shape (batch_size, n, hidden_size)
    here n is variable and can be e.g. time, 1 or user specified
    we expect the action head output to have a key 'action_pred' with shape (batch_size, time, action_dim) during inference time
    we expect these to have type BatchFeature, and they can of course have many other user specified keys too
    """

    def __init__(
        self,
        config: GR00T_N1_5_Config,
        local_model_path: str,
    ):
        assert isinstance(config.backbone_cfg, dict)
        assert isinstance(config.action_head_cfg, dict)

        super().__init__(config)
        self.local_model_path = local_model_path

        self.backbone = EagleBackbone(**config.backbone_cfg)
        action_head_cfg = FlowmatchingActionHeadConfig(**config.action_head_cfg)
        # # try overriding some data
        # action_head_cfg.max_num_embodiments = 1
        # action_head_cfg.action_horizon = 4
        # print(f"overriding max_num_embodiments: {action_head_cfg.max_num_embodiments}")
        # print(f"overriding action_horizon: {action_head_cfg.action_horizon}")
        
        self.action_head = FlowmatchingActionHead(action_head_cfg)

        self.action_horizon = config.action_horizon
        self.action_dim = config.action_dim
        self.compute_dtype = config.compute_dtype

    def validate_inputs(self, inputs):
        # NOTE -- this should be handled internally by the model
        # however, doing that will likely be breaking changes -- so we'll need to do it after the deadline
        
        # print(f"action_horizon: {self.action_horizon}")
        # print(f"action_dim: {self.action_dim}")

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
    
    # TODO: when reuse vlm values, it's passed into backward progress again. Try to Detach

    # def forward(
    #     self,
    #     inputs: dict,
    #     window_idx: int = None,
    # ) -> BatchFeature:
    #     backbone_inputs, action_inputs = self.prepare_input(inputs)
    #     backbone_outputs = self.backbone(backbone_inputs)
    #     action_head_outputs = self.action_head(backbone_outputs, action_inputs)
    #     self.validate_data(action_head_outputs, backbone_outputs, is_training=True)
    #     return action_head_outputs

    # def forward(
    #     self,
    #     inputs: dict,
    #     window_idx: int = None,
    # ) -> BatchFeature:
    #     """
    #     For training that matches the fast inference architecture.
    #     Only run the VLM backbone every 4th window (window_idx % 4 == 0),
    #     and cache/reuse the VLM output for intermediate windows.
    #     If window_idx is None, always run the backbone (backward compatible).
    #     """
    #     backbone_inputs, action_inputs = self.prepare_input(inputs)
        
    #     if window_idx is None:
    #         # Always run backbone if no window index is provided
    #         print("Training: No window_idx provided, running VLM backbone")
    #         backbone_outputs = self.backbone(backbone_inputs)
    #         action_head_outputs = self.action_head(backbone_outputs, action_inputs)
    #         self.validate_data(action_head_outputs, backbone_outputs, is_training=True)
    #         return action_head_outputs
        
    #     if window_idx % 4 == 0:
    #         for name, tensor in backbone_inputs.items():
    #             print(f"{name}: grad_fn = {tensor.grad_fn}, requires_grad = {tensor.requires_grad}")

    #         # Run VLM backbone and cache output
    #         print(f"Training: window_idx={window_idx} (VLM RUNNING)")
    #         backbone_outputs_fresh = self.backbone(backbone_inputs)
    #         # Create completely independent copies of the outputs
    #         detached_data = {}
    #         for k, v in backbone_outputs_fresh.items():
    #             if torch.is_tensor(v):
    #                 detached_data[k] = v.clone().detach()
    #             else:
    #                 detached_data[k] = v
    #         self._cached_backbone_outputs = type(backbone_outputs_fresh)(data=detached_data)
    #         backbone_outputs = self._cached_backbone_outputs
    #     else:
    #         print(f"Training: window_idx={window_idx} (USING CACHED VLM)")
    #         backbone_outputs = self._cached_backbone_outputs
        
    #     action_head_outputs = self.action_head(backbone_outputs, action_inputs)
    #     self.validate_data(action_head_outputs, backbone_outputs, is_training=True)
    #     return action_head_outputs


    def forward(self, inputs: dict, window_idx: int = None) -> BatchFeature:
        backbone_inputs, action_inputs = self.prepare_input(inputs)
        if window_idx is None or window_idx % 4 == 0:
            # print(f"Training: window_idx={window_idx} (VLM RUNNING)")
            fresh = self.backbone(backbone_inputs)
            self._cached_backbone_outputs = self._detach_batchfeature(fresh)
            backbone_outputs = self._cached_backbone_outputs
            backbone_outputs["backbone_features"] = backbone_outputs["backbone_features"].clone().detach()
        else:
            # print(f"Training: window_idx={window_idx} (USING CACHED VLM)")
            backbone_outputs = self._cached_backbone_outputs
            backbone_outputs["backbone_features"] = backbone_outputs["backbone_features"].clone().detach()
            
            # # Debug: check detachment
            # for name, tensor in self._cached_backbone_outputs.items():
            #     if torch.is_tensor(tensor):
            #         print(
            #             f"[DETACH DEBUG backbone] {name}: requires_grad={tensor.requires_grad}, grad_fn={tensor.grad_fn}"
            #         )

        action_head_outputs = self.action_head(backbone_outputs, action_inputs)
        self.validate_data(action_head_outputs, backbone_outputs, is_training=True)
        return action_head_outputs

    def _detach_batchfeature(self, bf):
        """
        Return a copy of bf where *every* tensor in bf.data
        has been cloned, detached, and set requires_grad=False.
        We do this by shallow-copying bf and then replacing its .data.
        """
        # 1) Build a new dict of detached tensors
        detached_data = {}
        for k, v in bf.data.items():
            if torch.is_tensor(v):
                # clone so we don’t share storage, detach from any graph,
                # and prevent any future grad requests
                detached_data[k] = v.clone().detach().requires_grad_(False)
            else:
                detached_data[k] = v

        # 2) Shallow-copy the original BatchFeature object
        new_bf = copy.copy(bf)
        # 3) Overwrite its .data with our detached copy
        new_bf.data = detached_data
        return new_bf



    def get_action(
        self,
        inputs: dict,
    ) -> BatchFeature:
        backbone_inputs, action_inputs = self.prepare_input(inputs)
        # Because the behavior of backbones remains the same for training and inference, we can use `forward` for backbones.
        backbone_outputs = self.backbone(backbone_inputs)
        action_head_outputs = self.action_head.get_action(backbone_outputs, action_inputs)
        self.validate_data(action_head_outputs, backbone_outputs, is_training=False)
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
AutoConfig.register("gr00t_n1_5", GR00T_N1_5_Config)
AutoModel.register(GR00T_N1_5_Config, GR00T_N1_5)
