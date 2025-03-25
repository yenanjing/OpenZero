set -x
#ray stop --force
export CUDA_VISIBLE_DEVICES=0,1
export NCCL_DEBUG=INFO

export VLLM_ATTENTION_BACKEND=XFORMERS
DATA_DIR=data/open_industry/v3
MODEL_PATH=/apdcephfs_cq11/share_2973545/wenweiwwli/projects/OpenZero/checkpoints/GRPO_OPEN_INDUSTRY/1node-Qwen-7B-bf16-v3-mb2-tp1-sp1-roll4-2h20-t0.7/actor/global_step_330
ROLLOUT_TP_SIZE=1
n_gpus_per_node=2
nnodes=1
experiment_name='1node-resume-stage1step330-v3-mb2-tp1-sp1-roll4-2h20-t1.2'

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files=$DATA_DIR/train.parquet \
    data.val_files=$DATA_DIR/test.parquet\
    data.train_batch_size=8 \
    data.val_batch_size=8 \
    data.max_prompt_length=16384 \
    data.max_response_length=1024 \
    actor_rollout_ref.model.path=$MODEL_PATH\
    actor_rollout_ref.actor.optim.lr=3e-7 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=1 \
    actor_rollout_ref.actor.ppo_micro_batch_size=1 \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=2 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    +actor_rollout_ref.actor.dtype=bfloat16 \
    +actor_rollout_ref.actor.fsdp_config.dtype=bfloat16 \
    actor_rollout_ref.rollout.dtype=bfloat16 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.grad_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.rollout.log_prob_micro_batch_size=8 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=$ROLLOUT_TP_SIZE \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.2 \
    actor_rollout_ref.rollout.temperature=1.2 \
    actor_rollout_ref.rollout.n=4 \
    actor_rollout_ref.rollout.max_num_batched_tokens=null \
    actor_rollout_ref.ref.log_prob_micro_batch_size=8 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.kl_ctrl.kl_coef=0.001 \
    +trainer.val_before_train=False \
    trainer.critic_warmup=0 \
    trainer.logger=['wandb'] \
    trainer.project_name='GRPO_OPEN_INDUSTRY' \
    trainer.experiment_name=$experiment_name \
    trainer.n_gpus_per_node=$n_gpus_per_node \
    trainer.nnodes=$nnodes \
    trainer.default_hdfs_dir=null \
    trainer.save_freq=30 \
    trainer.test_freq=10 \
    trainer.total_epochs=5 $@ 2>&1 | tee $experiment_name.log
