import os
import torch
import gr00t
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import time

from gr00t.data.dataset import LeRobotSingleDataset
from gr00t.model.policy import Gr00tPolicy
from gr00t.model.splitpolicy import SplitPolicy
from gr00t.experiment.data_config import DATA_CONFIG_MAP

class PolicyEvaluator:
    def __init__(
        self,
        model_path: str,
        dataset_path: str,
        embodiment_tag: str,
        data_config_name: str,
        video_backend: str,
        device: str = None,
        num_episodes: int = None,
        action_horizon: int = None,
        denoising_steps: int = None
    ):
        """
        Initialize the policy evaluator.

        Args:
            model_path (str): Path to the model checkpoint directory
            dataset_path (str): Path to the dataset directory
            embodiment_tag (str): Embodiment tag for the model
            data_config_name (str): Name of the data configuration from DATA_CONFIG_MAP
            video_backend (str): Backend to use for video loading
            device (str): Device to run the model on (default: "cuda" if available, else "cpu")
            num_episodes (int): Number of episodes to use (default: None, which means all episodes)
            action_horizon (int): Action horizon to use (default: 16)
        """
        self.repo_path = os.path.dirname(os.path.dirname(gr00t.__file__))
        self.model_path = model_path
        self.dataset_path = dataset_path
        self.embodiment_tag = embodiment_tag
        self.video_backend = video_backend
        self.device = device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu")
        self.num_episodes = num_episodes
        self.action_horizon = action_horizon
        self.denoising_steps = denoising_steps

        # Load data configuration
        self.data_config = DATA_CONFIG_MAP["libero_arm"]
        self.modality_config = self.data_config.modality_config()
        self.modality_transform = self.data_config.transform()

        # Initialize policy and dataset
        self._init_policy()
        self._init_dataset()

    def _init_policy(self):
        """Initialize the policy model."""
        # self.policy = Gr00tPolicy(
        #     model_path=self.model_path,
        #     embodiment_tag=self.embodiment_tag,
        #     modality_config=self.modality_config,
        #     modality_transform=self.modality_transform,
        #     device=self.device,
        # )

        self.policy = SplitPolicy(
            model_path=self.model_path,
            embodiment_tag=self.embodiment_tag,
            modality_config=self.modality_config,
            modality_transform=self.modality_transform,
            device=self.device,
            denoising_steps=self.denoising_steps # original is 16
        )

    def _init_dataset(self):
        """Initialize the dataset."""
        self.dataset = LeRobotSingleDataset(
            dataset_path=self.dataset_path,
            modality_configs=self.modality_config,
            video_backend=self.video_backend,
            video_backend_kwargs=None,
            transforms=None,  # We'll handle transforms separately through the policy
            embodiment_tag=self.embodiment_tag,
        )

    def print_configuration(self):
        """Print the configuration details."""
        print("\n=== Policy Configuration ===")
        print("Model:", self.policy.model)
        print("\nModality Config Keys:", self.modality_config.keys())

        print("\n=== Modality Config Details ===")
        for key, value in self.modality_config.items():
            if isinstance(value, np.ndarray):
                print(f"{key}: shape={value.shape}")
            else:
                print(f"{key}: {value}")

    def print_step_data(self, step_idx: int = 0):
        """
        Print details about a specific step from the dataset.

        Args:
            step_idx (int): Index of the step to print (default: 0)
        """
        step_data = self.dataset[step_idx]
        print("\n=== Step Data Contents ===")
        print("total number of steps",len(self.dataset))
        print("Keys:", step_data.keys())
        for key, value in step_data.items():
            if isinstance(value, np.ndarray):
                print(f"\n{key}:")
                print(f"  Shape: {value.shape}")
                print(f"  Type: {value.dtype}")
                print(f"  Range: [{value.min():.3f}, {value.max():.3f}]")
                if value.size < 10:  # Only print small arrays
                    print(f"  Values: {value}")
            else:
                print(f"\n{key}:")
                print(f"  Type: {type(value)}")
                if isinstance(value, list) and len(value) < 10:
                    print(f"  Values: {value}")
    
    def compare_results_so100(self):
        """
        Compare the predicted action with the ground truth action.
        Creates subplots for each joint (5 arm joints + 1 gripper).
        """
        gt_arm_list = []
        gt_gripper_list = []
        predicted_arm_list = []
        predicted_gripper_list = []
        

        for i in range(self.num_episodes):
            step_data = self.dataset[i]
            gt_arm = step_data["action.single_arm"]  # dim = 5
            gt_gripper = step_data["action.gripper"]  # dim = 1
            predicted_action = self.policy.get_action(step_data)
            predicted_arm = predicted_action["action.single_arm"]  # dim = 5
            predicted_gripper = predicted_action["action.gripper"]  # dim = 1
            
            gt_arm_list.append(gt_arm[0])
            gt_gripper_list.append(gt_gripper[0])
            predicted_arm_list.append(predicted_arm[0])
            predicted_gripper_list.append(predicted_gripper[0])
        
        gt_arm_array = np.array(gt_arm_list)
        gt_gripper_array = np.array(gt_gripper_list)
        predicted_arm_array = np.array(predicted_arm_list)
        predicted_gripper_array = np.array(predicted_gripper_list)

        # print
        print("its the shape",gt_arm_array.shape)
        
        # Create subplots for each joint
        joint_names = ['Joint 1', 'Joint 2', 'Joint 3', 'Joint 4', 'Joint 5', 'Gripper']
        fig, axs = plt.subplots(3, 2, figsize=(15, 15))
        axs = axs.flatten()

        # Plot arm joints
        for i in range(5):
            axs[i].plot(gt_arm_array[:, i], label='Ground Truth', color='blue')
            axs[i].plot(predicted_arm_array[:, i], label='Predicted', color='red', linestyle='--')
            axs[i].set_title(f'{joint_names[i]}')
            axs[i].set_xlabel('Time Step')
            axs[i].set_ylabel('Joint Value')
            axs[i].legend()
            axs[i].grid(True)

        # Plot gripper
        axs[5].plot(gt_gripper_array, label='Ground Truth', color='blue')
        axs[5].plot(predicted_gripper_array, label='Predicted', color='red', linestyle='--')
        axs[5].set_title('Gripper')
        axs[5].set_xlabel('Time Step')
        axs[5].set_ylabel('Gripper Value')
        axs[5].legend()
        axs[5].grid(True)

        plt.tight_layout()
        plt.show()

    def compare_results_libero(self):
        """
        Compare the predicted action with the ground truth action.
        Creates subplots for each joint (5 arm joints + 1 gripper).
        """
        action_horizon = self.action_horizon
        gt_all_actions_list = []
        # predicted_all_actions_list = []
        predicted_action_chunk_list = []
        joint_states_list = []
        
        # Lists to store timing data
        backbone_times = []
        action_head_times = []
        total_times = []
        
        print(f"\n=== Running inference for {self.num_episodes} episodes ===")
        
        # # run at every steps (wrong, just for test):
        # for i in range(self.num_episodes):
        #     step_data = self.dataset[i]
        #     gt_all_actions = step_data["action.all_actions"]
        #     predicted_all_actions = self.policy.get_action(step_data)["action.all_actions"]

        #     gt_all_actions_list.append(gt_all_actions[0])
        #     predicted_action_chunk_list.append(predicted_all_actions[0])
        #     joint_states_list.append(step_data["state.joint_states"][0])

        for time_step in range(self.num_episodes):   # 132 is the number of steps in the dataset for the first task
            # Inference at each timestep
            step_data = self.dataset[time_step]
            gt_all_actions = step_data["action.all_actions"]

            gt_all_actions_list.append(gt_all_actions[0])
            # predicted_all_actions_list.append(predicted_all_actions[0])
            joint_states_list.append(step_data["state.joint_states"][0])
            
            
            # Inference at every 16 steps
            if time_step % action_horizon == 0:
                print(f"\n--- Inference at step: {time_step} ---")
                action_chunk = self.policy.get_action(step_data, time_step, select_model_fast=False, print_timing=True)["action.all_actions"]
                print("ACTION CHUNK SHAPE",action_chunk.shape)
                
                if action_chunk.shape[0] != action_horizon:
                    raise ValueError(f"Action chunk shape {action_chunk.shape[0]} does not match expected action horizon {action_horizon}")

                # Get timing stats for this inference
                timing_stats = self.policy.get_last_timing_stats()
                backbone_times.append(timing_stats.get('backbone_inference_time', 0))
                action_head_times.append(timing_stats.get('action_head_inference_time', 0))
                total_times.append(timing_stats.get('total_inference_time', 0))
                
                for j in range(action_horizon):
                    predicted_action = action_chunk[j]
                    predicted_action_chunk_list.append(predicted_action)

        # Print timing summary
        print(f"\n{'='*50}")
        print(f"TIMING SUMMARY")
        print(f"{'='*50}")
        print(f"Number of inference calls: {len(backbone_times)}")
        print(f"Denoising steps: {self.policy.denoising_steps}")
        print(f"Action horizon: {self.action_horizon}")
        print(f"Average VLM (backbone) time: {np.mean(backbone_times):.4f} ± {np.std(backbone_times):.4f} seconds")
        print(f"Average DiT (action head) time: {np.mean(action_head_times):.4f} ± {np.std(action_head_times):.4f} seconds")
        print(f"Average total inference time: {np.mean(total_times):.4f} ± {np.std(total_times):.4f} seconds")
        print(f"Min total time: {np.min(total_times):.4f} seconds")
        print(f"Max total time: {np.max(total_times):.4f} seconds")
        print(f"{'='*50}")

        # Convert lists to numpy arrays
        gt_all_actions_array = np.array(gt_all_actions_list)
        # predicted_all_actions_array = np.array(predicted_all_actions_list)
        joint_states_array = np.array(joint_states_list)
        predicted_action_chunk_array = np.array(predicted_action_chunk_list)
        # print
        print("its the action shape",gt_all_actions_array.shape)
        print("its the state shape",joint_states_array.shape)
        print("its the predicted action chunk shape",predicted_action_chunk_array.shape)
        # Create subplots for each joint
        joint_names = ['Joint 1', 'Joint 2', 'Joint 3', 'Joint 4', 'Joint 5', 'Joint 6', 'Joint 7']
        fig, axs = plt.subplots(4, 2, figsize=(15, 15))
        axs = axs.flatten()

        # Plot arm joints
        for i in range(7):
            axs[i].plot(gt_all_actions_array[:, i], label='Ground Truth', color='blue')
            # axs[i].plot(predicted_all_actions_array[:, i], label='Predicted', color='red', linestyle='--')
            axs[i].plot(joint_states_array[:, i], label='Joint States', color='green')
            axs[i].plot(predicted_action_chunk_array[:, i], label='Predicted Action Chunk', color='black', linestyle='--')
            axs[i].set_title(f'{joint_names[i]}')
            axs[i].set_xlabel('Time Step')
            axs[i].set_ylabel('Joint Value')
            axs[i].legend()
            axs[i].grid(True)

        plt.tight_layout()
        plt.show()

    def compare_results_libero_fast(self):
        """
        Compare the predicted action with the ground truth action using fast inference.
        Only runs backbone at timesteps 0, 16, 32, etc., reuses backbone outputs for other timesteps.
        """
        action_horizon = self.action_horizon
        gt_all_actions_list = []
        predicted_action_chunk_list = []
        joint_states_list = []
        
        # Lists to store timing data
        backbone_times = []
        action_head_times = []
        total_times = []
        
        print(f"\n=== Running fast inference for {self.num_episodes} episodes ===")
        
        for time_step in range(self.num_episodes):
            # Get ground truth for comparison
            step_data = self.dataset[time_step]
            gt_all_actions = step_data["action.all_actions"]
            gt_all_actions_list.append(gt_all_actions[0])
            joint_states_list.append(step_data["state.joint_states"][0])
            
            # Run fast inference (backbone only at action_horizon steps)
            action_chunk = self.policy.get_action(step_data, time_step, select_model_fast=True, print_timing=True)["action.all_actions"]
            
            if action_chunk.shape[0] != action_horizon:
                raise ValueError(f"Action chunk shape {action_chunk.shape[0]} does not match expected action horizon {action_horizon}")
            
            predicted_action_chunk_list.append(action_chunk[0])
            
            # Get timing stats if this was a full inference (backbone + action_head)
            if time_step % action_horizon == 0:
                timing_stats = self.policy.get_last_timing_stats()
                backbone_times.append(timing_stats.get('backbone_inference_time', 0))
                action_head_times.append(timing_stats.get('action_head_inference_time', 0))
                total_times.append(timing_stats.get('total_inference_time', 0))



   
                
                
        # Print timing summary
        print(f"\n{'='*50}")
        print(f"TIMING SUMMARY")
        print(f"{'='*50}")
        print(f"Number of inference calls: {len(backbone_times)}")
        print(f"Denoising steps: {self.policy.denoising_steps}")
        print(f"Action horizon: {self.action_horizon}")
        print(f"Average VLM (backbone) time: {np.mean(backbone_times):.4f} ± {np.std(backbone_times):.4f} seconds")
        print(f"Average DiT (action head) time: {np.mean(action_head_times):.4f} ± {np.std(action_head_times):.4f} seconds")
        print(f"Average total inference time: {np.mean(total_times):.4f} ± {np.std(total_times):.4f} seconds")
        print(f"Min total time: {np.min(total_times):.4f} seconds")
        print(f"Max total time: {np.max(total_times):.4f} seconds")
        print(f"{'='*50}")

        # Convert lists to numpy arrays
        gt_all_actions_array = np.array(gt_all_actions_list)
        # predicted_all_actions_array = np.array(predicted_all_actions_list)
        joint_states_array = np.array(joint_states_list)
        predicted_action_chunk_array = np.array(predicted_action_chunk_list)
        # print
        print("its the action shape",gt_all_actions_array.shape)
        print("its the state shape",joint_states_array.shape)
        print("its the predicted action chunk shape",predicted_action_chunk_array.shape)
        # Create subplots for each joint
        joint_names = ['Joint 1', 'Joint 2', 'Joint 3', 'Joint 4', 'Joint 5', 'Joint 6', 'Joint 7']
        fig, axs = plt.subplots(4, 2, figsize=(15, 15))
        axs = axs.flatten()

        # Plot arm joints
        for i in range(7):
            axs[i].plot(gt_all_actions_array[:, i], label='Ground Truth', color='blue')
            # axs[i].plot(predicted_all_actions_array[:, i], label='Predicted', color='red', linestyle='--')
            axs[i].plot(joint_states_array[:, i], label='Joint States', color='green')
            axs[i].plot(predicted_action_chunk_array[:, i], label='Predicted Action Chunk', color='black', linestyle='--')
            axs[i].set_title(f'{joint_names[i]}')
            axs[i].set_xlabel('Time Step')
            axs[i].set_ylabel('Joint Value')
            axs[i].legend()
            axs[i].grid(True)

        plt.tight_layout()
        plt.show()

    

    # def get_action(self, step_idx: int = 0):
    #     """
    #     For timing evaluation only

    #     Args:
    #         step_idx (int): Index of the step to get action for (default: 0)

    #     Returns:
    #         dict: The predicted action
    #     """
    #     step_data = self.dataset[step_idx]
    #     predicted_action = self.policy.get_action(step_data, print_timing=True)  # Enable timing output
        
    #     print("\n=== Predicted Action ===")
    #     for key, value in predicted_action.items():
    #         if isinstance(value, np.ndarray):
    #             print(f"{key}: shape={value.shape}, range=[{value.min():.3f}, {value.max():.3f}]")
    #         else:
    #             print(f"{key}: {type(value)}")
        
    #     return predicted_action

    def save_predicted_actions_to_parquet(self):
        """
        Save all predicted actions for all timesteps to a parquet file.
        The file will contain:
        - timestep: The timestep index
        - episode_id: The episode ID
        - predicted_actions: The predicted actions from the model
        - ground_truth_actions: The ground truth actions from the dataset
        - joint_states: The joint states from the dataset
        """
        # Lists to store data
        timesteps = []
        episode_ids = []
        predicted_actions = []
        
        
        # Get data for all timesteps
        for i in range(self.num_episodes):
            step_data = self.dataset[i]
            trajectory_id, base_index = self.dataset.all_steps[i]
            
            # Get predicted action
            action = self.policy.get_action(step_data)
            predicted_action = action["action.all_actions"][0]  # Get first element from batch
            
            
            
            # Store data
            timesteps.append(i)
            episode_ids.append(trajectory_id)
            predicted_actions.append(predicted_action)
           
        
        # Create DataFrame
        df = pd.DataFrame({
            'timestep': timesteps,
            'episode_id': episode_ids,
            'predicted_actions': predicted_actions,
        })
        
        # Save to parquet file
        timestamp = int(time.time())
        output_file = f'predicted_actions_{timestamp}.parquet'
        df.to_parquet(output_file)
        print(f"Successfully saved predicted actions to {output_file}")
        
        return df


def main():
    # libero
    evaluator = PolicyEvaluator(
        # model_path="./checkpoints/checkpoint-first",
        model_path="./checkpoints/checkpoint-38000",
        dataset_path=os.path.join(os.path.dirname(os.path.dirname(gr00t.__file__)), "demo_data/libero_object_data"),
        embodiment_tag="libero_arm",
        data_config_name="custom_panda_hand",
        video_backend="torchvision_av",
        num_episodes=132,
        action_horizon=16,
        denoising_steps=16,
    )

    # Print configurations and requirements
    evaluator.print_configuration()
    evaluator.print_step_data()
    
    # # Get action with timing output (this will print timing for a single inference)
    # print("\n=== Single Inference with Timing ===")
    # evaluator.get_action()
    

    
    # evaluator.compare_results_libero()  # This will show timing for each inference call
    evaluator.compare_results_libero_fast()  # This will reuse backbone outputs
    
    
    # evaluator.compare_results_so100()
    # evaluator.save_predicted_actions_to_parquet()

if __name__ == "__main__":
    main()