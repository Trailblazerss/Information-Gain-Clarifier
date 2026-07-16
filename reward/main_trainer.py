#!/usr/bin/env python3
"""
Main DAPO Training Script with Log Probability QA Reward

Key Features:
1. Rollout length for question extraction
2. Optimizes first N tokens only
3. History prompt truncation
4. Batch filtering based on question extraction rate
"""

import os
import sys
import socket

# Add paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, SCRIPT_DIR)

# VERL_DIR should be set via environment variable or defaults to the sibling checkout
VERL_DIR = os.environ.get("VERL_DIR", os.path.abspath(os.path.join(PROJECT_ROOT, "..", "verl")))
sys.path.insert(0, VERL_DIR)

import ray
import hydra
from omegaconf import OmegaConf

from verl.trainer.constants_ppo import get_ppo_ray_runtime_env
from verl.trainer.ppo.reward import load_reward_manager
from verl.utils.device import auto_set_ascend_device_name, is_cuda_available


def get_config_path():
    """Get the path to the config directory"""
    # Try to find config in verl installation
    verl_config_path = os.path.join(VERL_DIR, "recipe/dapo/config")
    if os.path.exists(verl_config_path):
        return verl_config_path
    
    # Fallback to local config
    local_config_path = os.path.join(PROJECT_ROOT, "verl_recipe/config")
    if os.path.exists(local_config_path):
        return local_config_path
    
    raise FileNotFoundError(f"Config directory not found. Tried: {verl_config_path}, {local_config_path}")


@hydra.main(config_path=get_config_path(), config_name="dapo_trainer", version_base=None)
def main(config):
    """Main function with log probability QA reward injection"""
    
    print("\n" + "="*70)
    print("DAPO Training with Log Probability QA Reward")
    print("   - Rollout: for question extraction")
    print("   - Optimize: first N tokens only")
    print("   - History: max chars truncation")
    print("   - Batch filter: question rate requirement")
    print("="*70)
    print(f"Script Dir: {SCRIPT_DIR}")
    print("="*70 + "\n")
    
    # Automatically set `config.trainer.device = npu` when running on Ascend NPU.
    auto_set_ascend_device_name(config)
    
    run_ppo_with_log_prob_reward(config)


def run_ppo_with_log_prob_reward(config) -> None:
    if not ray.is_initialized():
        # this is for local ray cluster
        default_runtime_env = get_ppo_ray_runtime_env()
        ray_init_kwargs = config.ray_kwargs.get("ray_init", {})
        runtime_env_kwargs = ray_init_kwargs.get("runtime_env", {})
        runtime_env = OmegaConf.merge(default_runtime_env, runtime_env_kwargs)
        ray_init_kwargs = OmegaConf.create({**ray_init_kwargs, "runtime_env": runtime_env})
        print(f"ray init kwargs: {ray_init_kwargs}")
        ray.init(**OmegaConf.to_container(ray_init_kwargs))

    try:
        if (
            is_cuda_available
            and config.global_profiler.tool == "nsys"
            and OmegaConf.select(config.global_profiler, "steps") is not None
            and len(OmegaConf.select(config.global_profiler, "steps")) > 0
        ):
            nsight_options = OmegaConf.to_container(
                config.global_profiler.global_tool_config.nsys.controller_nsight_options
            )
            runner = LogProbRewardTaskRunner.options(runtime_env={"nsight": nsight_options}).remote()
        else:
            runner = LogProbRewardTaskRunner.remote()
        ray.get(runner.run.remote(config))
    finally:
        if ray.is_initialized():
            ray.shutdown()


@ray.remote(num_cpus=1)
class LogProbRewardTaskRunner:
    """
    Custom TaskRunner that patches RayDAPOTrainer in run()
    """
    
    def run(self, config):
        from pprint import pprint
        from omegaconf import OmegaConf
        from verl.utils.fs import copy_to_local
        
        print(f"LogProbRewardTaskRunner hostname: {socket.gethostname()}, PID: {os.getpid()}")
        
        pprint(OmegaConf.to_container(config, resolve=True))
        OmegaConf.resolve(config)
        
        # ============ Key: Apply patch here ============
        print("\n" + "="*70)
        print("Applying Log Probability QA Reward Patch...")
        print("="*70 + "\n")
        
        import sys
        sys.path.insert(0, SCRIPT_DIR)
        
        from trainer_patch import patch_compute_kl_related_metrics
        patch_compute_kl_related_metrics()
        
        # ============ Original training logic ============
        
        # download the checkpoint from hdfs
        local_path = copy_to_local(config.actor_rollout_ref.model.path)

        # instantiate tokenizer
        from verl.utils import hf_processor, hf_tokenizer

        trust_remote_code = config.data.get("trust_remote_code", False)
        tokenizer = hf_tokenizer(local_path, trust_remote_code=trust_remote_code)
        processor = hf_processor(local_path, trust_remote_code=trust_remote_code, use_fast=True)
        
        # Check if passthrough chat template is needed (for base models)
        if os.environ.get("USE_PASSTHROUGH_TEMPLATE", "").lower() == "true":
            PASSTHROUGH_TEMPLATE = "{% for message in messages %}{{ message['content'] }}{% endfor %}"
            tokenizer.chat_template = PASSTHROUGH_TEMPLATE
            if processor is not None and hasattr(processor, 'chat_template'):
                processor.chat_template = PASSTHROUGH_TEMPLATE
            print("Set passthrough chat template (Base model mode)")

        from verl.single_controller.ray import RayWorkerGroup

        # define worker classes
        if config.actor_rollout_ref.actor.strategy in {"fsdp", "fsdp2"}:
            assert config.critic.strategy in {"fsdp", "fsdp2"}

            from verl.workers.fsdp_workers import AsyncActorRolloutRefWorker, CriticWorker

            ray_worker_group_cls = RayWorkerGroup

        elif config.actor_rollout_ref.actor.strategy == "megatron":
            assert config.actor_rollout_ref.actor.strategy == config.critic.strategy
            from verl.workers.megatron_workers import AsyncActorRolloutRefWorker, CriticWorker

            ray_worker_group_cls = RayWorkerGroup

        else:
            raise NotImplementedError

        from verl.trainer.ppo.ray_trainer import ResourcePoolManager, Role

        role_worker_mapping = {
            Role.ActorRollout: ray.remote(AsyncActorRolloutRefWorker),
            Role.Critic: ray.remote(CriticWorker),
        }

        global_pool_id = "global_pool"
        resource_pool_spec = {
            global_pool_id: [config.trainer.n_gpus_per_node] * config.trainer.nnodes,
        }
        mapping = {
            Role.ActorRollout: global_pool_id,
            Role.Critic: global_pool_id,
        }

        if config.reward_model.enable:
            if config.reward_model.strategy in {"fsdp", "fsdp2"}:
                from verl.workers.fsdp_workers import RewardModelWorker
            elif config.reward_model.strategy == "megatron":
                from verl.workers.megatron_workers import RewardModelWorker
            else:
                raise NotImplementedError
            role_worker_mapping[Role.RewardModel] = ray.remote(RewardModelWorker)
            if config.reward_model.enable_resource_pool:
                resource_pool_spec[config.reward_model.resource_pool_id] = (
                    [config.reward_model.resource_pool_spec.n_gpus_per_node] * config.reward_model.resource_pool_spec.nnodes
                )
                mapping[Role.RewardModel] = config.reward_model.resource_pool_id
            else:
                mapping[Role.RewardModel] = global_pool_id

        # reference model
        if config.algorithm.use_kl_in_reward or config.actor_rollout_ref.actor.use_kl_loss:
            role_worker_mapping[Role.RefPolicy] = ray.remote(AsyncActorRolloutRefWorker)
            mapping[Role.RefPolicy] = global_pool_id

        reward_fn = load_reward_manager(
            config,
            tokenizer,
            0,
            max_resp_len=config.data.max_response_length,
            overlong_buffer_cfg=config.reward_model.overlong_buffer,
        )

        val_reward_fn = load_reward_manager(
            config,
            tokenizer,
            1,
            max_resp_len=config.data.max_response_length,
            overlong_buffer_cfg=config.reward_model.overlong_buffer,
        )
        
        resource_pool_manager = ResourcePoolManager(
            resource_pool_spec=resource_pool_spec, mapping=mapping
        )

        # Import from local verl_recipe (derived from verl project)
        import sys
        sys.path.insert(0, PROJECT_ROOT)
        from verl_recipe.dapo_ray_trainer import RayDAPOTrainer

        trainer = RayDAPOTrainer(
            config=config,
            tokenizer=tokenizer,
            processor=processor,
            role_worker_mapping=role_worker_mapping,
            resource_pool_manager=resource_pool_manager,
            ray_worker_group_cls=ray_worker_group_cls,
            reward_fn=reward_fn,
            val_reward_fn=val_reward_fn,
        )
        trainer.init_workers()
        trainer.fit()


if __name__ == "__main__":
    main()
