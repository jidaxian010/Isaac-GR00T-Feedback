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


import os
from typing import Optional

import torch
import transformers
from torch.utils.data import Dataset, Sampler, DataLoader
from transformers.trainer import (
    ALL_LAYERNORM_LAYERS,
    TRAINER_STATE_NAME,
    TrainerState,
    get_last_checkpoint,
    get_parameter_names,
    is_sagemaker_mp_enabled,
)


class BaseSampler(Sampler):
    """Sampler for dataset, which enables `set_epoch` for Dataset.
    `set_epoch` will be called by huggingface Trainer at the end of each epoch.
    `shuffle` is also supported for training set shuffling
    """

    def __init__(self, data_source: Dataset, shuffle: bool = False, seed: int = 0):
        self.data_source = data_source
        self.shuffle = shuffle
        self.seed = seed
        self.epoch = 0

    def __iter__(self):
        if self.shuffle:
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch)
            # must not add rank here, or randomization will be different for each rank
            return iter(torch.randperm(len(self.data_source), generator=g).tolist())
        return iter(range(len(self.data_source)))

    def set_epoch(self, epoch):
        self.epoch = epoch
        if hasattr(self.data_source, "set_epoch"):
            # this is important for dataset
            self.data_source.set_epoch(epoch)

    def __len__(self):
        return len(self.data_source)


class DualBrainTrainer(transformers.Trainer):
    def __init__(self, **kwargs):
        self.compute_dtype = kwargs.pop("compute_dtype")
        super().__init__(**kwargs)
        
        # Add custom logging callback
        from transformers import TrainerCallback
        
        class CustomLoggingCallback(TrainerCallback):
            def on_log(self, args, state, control, logs=None, **kwargs):
                if logs is not None:
                    # Add custom loss components to logs
                    if "action_loss" in logs:
                        logs["action_loss"] = logs["action_loss"]
                    if "vlm_loss" in logs:
                        logs["vlm_loss"] = logs["vlm_loss"]
                    if "loss_breakdown" in logs:
                        logs.update(logs["loss_breakdown"])
        
        self.add_callback(CustomLoggingCallback())

    def _get_eval_sampler(self, eval_dataset):
        return BaseSampler(eval_dataset, shuffle=False)

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):

        # # original compute loss
        # outputs = model(inputs)

        if not hasattr(self, '_window_idx'):
            self._window_idx = 0
            print(f"Starting new epoch, resetting window_idx from {self._window_idx} to 0")
        
        # Get trajectory information for this batch
        if not hasattr(self, '_current_batch_indices'):
            self._current_batch_indices = []
        current_indices = self._get_current_batch_indices()
        trajectory_info = self._get_trajectory_ids_for_batch(current_indices)
        
        # Format trajectory info for display: "traj_id(timestep)"
        trajectory_display = []
        trajectory_ids = []
        for traj_id, timestep in trajectory_info:
            if traj_id is not None:
                trajectory_display.append(f"{traj_id}({timestep})")
                trajectory_ids.append(traj_id)
            else:
                trajectory_display.append("None")
                trajectory_ids.append(None)
        
        # Print trajectory info with timesteps
        print(f"trajectory_info: {trajectory_display}, window_idx: {self._window_idx}")
        
        # # Print every 100 batches to monitor training progress
        # if self._window_idx % 100 == 0:
        #     print(f"Training batch {self._window_idx}: window_idx={self._window_idx}, VLM will run: {self._window_idx % 4 == 0}")
        outputs = model(inputs, window_idx=self._window_idx, trajectory_ids=trajectory_ids)
        self._window_idx += 1
        # print(f"window_idx: {self._window_idx}")

        loss = outputs["loss"]
        
        # Print custom loss components
        if "action_loss" in outputs:
            print(f"Loss breakdown - Action: {outputs['action_loss'].item():.4f}, VLM: {outputs['vlm_loss'].item():.4f}, Total: {outputs['loss'].item():.4f}")
        
        return (loss, outputs) if return_outputs else loss

    def _get_current_batch_indices(self):
        """Helper to get current batch indices"""
        if not hasattr(self, '_batch_counter'):
            self._batch_counter = 0
        batch_size = self.args.per_device_train_batch_size
        start_idx = self._batch_counter * batch_size
        end_idx = start_idx + batch_size
        indices = list(range(start_idx, end_idx))
        self._batch_counter += 1
        return indices
    
    def _get_trajectory_ids_for_batch(self, indices):
        """Helper to get trajectory IDs and timesteps(base_index) for the current batch from the dataset"""
        trajectory_info = []
        for idx in indices:
            if idx < len(self.train_dataset):
                if hasattr(self.train_dataset, 'sample_step'):
                    dataset, trajectory_id, base_index = self.train_dataset.sample_step(idx)
                    trajectory_info.append((trajectory_id, base_index))
                else:
                    # Use concurrent batching for proper recursive VLM updates
                    trajectory_id, base_index = self.train_dataset.all_steps[idx]
                    # trajectory_id, base_index = self.train_dataset.all_steps_concurrent[idx]
                    trajectory_info.append((trajectory_id, base_index))
            else:
                trajectory_info.append((None, None))
        return trajectory_info

    def _get_train_sampler(self):
        # Reset window_idx at the start of each epoch
        if hasattr(self, '_window_idx'):
            print(f"Starting new epoch, resetting window_idx from {self._window_idx} to 0")
            self._window_idx = 0
        if hasattr(self, '_batch_counter'):
            print(f"Starting new epoch, resetting batch_counter from {self._batch_counter} to 0")
            self._batch_counter = 0
        return BaseSampler(self.train_dataset, shuffle=True, seed=self.args.seed)

    def create_optimizer(self):
        """
        Setup the optimizer.

        We provide a reasonable default that works well. If you want to use something else, you can pass a tuple in the
        Trainer's init through `optimizers`, or subclass and override this method in a subclass.
        """
        if is_sagemaker_mp_enabled():
            return super().create_optimizer()

        opt_model = self.model

        if self.optimizer is None:
            decay_parameters = get_parameter_names(opt_model, ALL_LAYERNORM_LAYERS)
            decay_parameters = [name for name in decay_parameters if "bias" not in name]
            optimizer_grouped_parameters = [
                {
                    "params": [
                        p
                        for n, p in opt_model.named_parameters()
                        if (n in decay_parameters and p.requires_grad)
                    ],
                    "weight_decay": self.args.weight_decay,
                },
                {
                    "params": [
                        p
                        for n, p in opt_model.named_parameters()
                        if (n not in decay_parameters and p.requires_grad)
                    ],
                    "weight_decay": 0.0,
                },
            ]

            optimizer_cls, optimizer_kwargs = transformers.Trainer.get_optimizer_cls_and_kwargs(
                self.args
            )
            self.optimizer = optimizer_cls(optimizer_grouped_parameters, **optimizer_kwargs)

        return self.optimizer

    def save_model(self, output_dir: Optional[str], _internal_call: bool):
        ## save tuned model separately
        if self.is_deepspeed_enabled:
            state_dict = self.accelerator.get_state_dict(self.deepspeed)
        else:
            state_dict = self.model.state_dict()

        if self.args.should_save:
            return self.model.save_pretrained(output_dir, state_dict=state_dict)

    def get_train_dataloader(self):
        """
        Returns the training DataLoader with drop_last=True.
        """
        train_dataset = self.train_dataset
        data_collator = self.data_collator
        train_sampler = self._get_train_sampler()
        return DataLoader(
            train_dataset,
            batch_size=self.args.train_batch_size,
            sampler=train_sampler,
            collate_fn=data_collator,
            drop_last=True,  # Ensures the last small batch is dropped
            num_workers=self.args.dataloader_num_workers,
            pin_memory=self.args.dataloader_pin_memory,
        )

    def train(
        self,
        resume_from_checkpoint=None,
        trial=None,
        ignore_keys_for_eval=None,
        **kwargs,
    ):
        """Correctly set self.state from checkpoint so get_train_dataloader can read from it."""
        if resume_from_checkpoint is False:
            resume_from_checkpoint = None

        if isinstance(resume_from_checkpoint, bool) and resume_from_checkpoint:
            resume_from_checkpoint = get_last_checkpoint(self.args.output_dir)
            if resume_from_checkpoint is None:
                raise ValueError(
                    f"No valid checkpoint found in output directory ({self.args.output_dir})"
                )

        if resume_from_checkpoint is not None:
            # In case of repeating the find_executable_batch_size, set `self._train_batch_size` properly
            self.state = TrainerState.load_from_json(
                os.path.join(resume_from_checkpoint, TRAINER_STATE_NAME)
            )
        return super().train(resume_from_checkpoint, trial, ignore_keys_for_eval, **kwargs)
