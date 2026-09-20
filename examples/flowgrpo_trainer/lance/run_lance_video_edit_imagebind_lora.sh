# Lance LoRA RL (video editing), vllm_omni rollout (FlowGRPO) with an ImageBind reward
#
# Mirrors run_lance_image_edit_pickscore_lora.sh, which is the image-editing
# recipe, on the video checkpoint.  What differs:
#
#   * the dataset carries a reference video per sample, bound to the sample by
#     the '<video>' marker the dataset layer substitutes (see
#     data_process/lance_video_edit.py).  The rollout transport knows only media
#     modalities, so the video arrives as multi_modal_data['video'] - which is
#     the key the video-edit node reads - and the Lance agent loop is what puts
#     it there, because the Lance bundle ships no processor for the dataset
#     layer to expand media with.
#   * the reference is a path, not bytes: the video-edit node decodes it with
#     decord (OpenCV as a fallback) so that upstream Lance's own bucket resize
#     and frame sampler run, and it reads the source frame rate from the file.
#   * the generated video takes the reference's bucketed shape, so the
#     num_frames / height / width knobs do not drive this recipe.
#   * imagebind replaces pickscore, scoring the edited video against the edit
#     instruction as the text-to-video recipe scores against the caption.
#
# Prerequisite: the Lance bundle and a video-edit dataset, for example::
#
#     python examples/flowgrpo_trainer/data_process/lance_video_edit.py \
#         --model_path ~/models/bytedance-research/Lance \
#         --input_dir ~/data/lance_video_edit --output_dir ~/data/lance_video_edit/lance

set -x

# Set WORKSPACE to any writable directory; defaults to $HOME
WORKSPACE=${WORKSPACE:-$HOME}

pickscore_train_path=$WORKSPACE/data/lance_video_edit/lance/train.parquet
pickscore_test_path=$WORKSPACE/data/lance_video_edit/lance/test.parquet

LANCE_DEPLOY_CONFIG=${LANCE_DEPLOY_CONFIG:-"$(dirname "$0")/lance_deploy_config.yaml"}

# The video checkpoint directory, not the bundle root: the bundle resolves to the
# image checkpoint first.
model_name=$WORKSPACE/models/bytedance-research/Lance/Lance_3B_Video
tokenizer_name=$WORKSPACE/data/pickscore/lance/tokenizer
custom_chat_template='{% for message in messages %}{% if message['\''role'\''] == '\''user'\'' %}<|im_start|>user
{{ message['\''content'\''] }}<|im_end|>
{% endif %}{% endfor %}'

IMGBIND_MODEL_PATH=${IMGBIND_MODEL_PATH:-.checkpoints/imagebind_huge.pth}
repo_root=$(cd "$(dirname "$0")/../../.." && pwd)

# Denoising schedule.  The generated video's shape is the reference's own
# bucketed shape, so these shape knobs are dead for this recipe and are set to
# the values the recipe was run with.
VIDEO_HEIGHT=${VIDEO_HEIGHT:-480}
VIDEO_WIDTH=${VIDEO_WIDTH:-768}
NUM_FRAMES=${NUM_FRAMES:-25}
INFER_STEPS=${INFER_STEPS:-30}

NUM_GPUS_ACTOR_ROLLOUT_REWARD=4
ROLLOUT_TP=1

ENGINE=vllm_omni

# enable reward model on 0'th gpu, it is a temporary workaround
export RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO=0

python3 -m verl_omni.trainer.main_diffusion \
    data.train_files=$pickscore_train_path \
    data.val_files=$pickscore_test_path \
    data.train_batch_size=48 \
    data.max_prompt_length=256 \
    data.filter_overlong_prompts=True \
    +data.apply_chat_template_kwargs.chat_template="\"$custom_chat_template\"" \
    data.trust_remote_code=True \
    algorithm.global_std=False \
    actor_rollout_ref.model.path=$model_name \
    actor_rollout_ref.model.tokenizer_path=$tokenizer_name \
    actor_rollout_ref.model.custom_chat_template="\"$custom_chat_template\"" \
    +actor_rollout_ref.model.architecture=OmniLanceForConditionalGeneration \
    actor_rollout_ref.model.algorithm=flow_grpo \
    actor_rollout_ref.model.trust_remote_code=True \
    actor_rollout_ref.model.lora_rank=64 \
    actor_rollout_ref.model.lora_alpha=128 \
    actor_rollout_ref.model.lora_dtype=float32 \
    actor_rollout_ref.model.lora.merge=True \
    actor_rollout_ref.model.target_modules="['q_proj_moe_gen','k_proj_moe_gen','v_proj_moe_gen','o_proj_moe_gen','mlp_moe_gen.gate_proj','mlp_moe_gen.up_proj','mlp_moe_gen.down_proj']" \
    actor_rollout_ref.model.fsdp_layer_prefixes="['layers.']" \
    actor_rollout_ref.actor.optim.lr=1e-4 \
    actor_rollout_ref.actor.optim.weight_decay=0.0001 \
    actor_rollout_ref.actor.ppo_mini_batch_size=24 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=12 \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.actor.fsdp_config.model_dtype=bfloat16 \
    actor_rollout_ref.actor.diffusion_loss.clip_ratio=1e-5 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=12 \
    actor_rollout_ref.rollout.calculate_log_probs=True \
    actor_rollout_ref.rollout.tensor_model_parallel_size=$ROLLOUT_TP \
    actor_rollout_ref.rollout.name=$ENGINE \
    actor_rollout_ref.rollout.agent.default_agent_loop=lance_diffusion_single_turn_agent \
    actor_rollout_ref.rollout.n=16 \
    actor_rollout_ref.rollout.agent.num_workers=$((NUM_GPUS_ACTOR_ROLLOUT_REWARD / ROLLOUT_TP)) \
    actor_rollout_ref.rollout.load_format=safetensors \
    actor_rollout_ref.rollout.layered_summon=True \
    actor_rollout_ref.rollout.pipeline.height=$VIDEO_HEIGHT \
    actor_rollout_ref.rollout.pipeline.width=$VIDEO_WIDTH \
    actor_rollout_ref.rollout.pipeline.num_frames=$NUM_FRAMES \
    actor_rollout_ref.rollout.pipeline.num_inference_steps=$INFER_STEPS \
    actor_rollout_ref.rollout.pipeline.max_sequence_length=256 \
    actor_rollout_ref.rollout.algo.noise_level=1.3 \
    actor_rollout_ref.rollout.algo.sde_type="sde" \
    actor_rollout_ref.rollout.algo.sde_window_size=2 \
    actor_rollout_ref.rollout.algo.sde_window_range="[0,7]" \
    actor_rollout_ref.rollout.val_kwargs.pipeline.height=$VIDEO_HEIGHT \
    actor_rollout_ref.rollout.val_kwargs.pipeline.width=$VIDEO_WIDTH \
    actor_rollout_ref.rollout.val_kwargs.pipeline.num_frames=$NUM_FRAMES \
    actor_rollout_ref.rollout.val_kwargs.pipeline.num_inference_steps=50 \
    actor_rollout_ref.rollout.val_kwargs.algo.noise_level=0.0 \
    +actor_rollout_ref.rollout.engine_kwargs.vllm_omni.deploy_config=$LANCE_DEPLOY_CONFIG \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=12 \
    reward.num_workers=1 \
    reward.reward_model.enable=False \
    reward.custom_reward_function.path=pkg://verl_omni.reward_loop.reward_manager.multi \
    reward.custom_reward_function.name=_multi_reward_placeholder \
    reward.reward_manager.name=MultiVisualRewardManager \
    reward.reward_manager.module.path=pkg://verl_omni.reward_loop.reward_manager \
    "+reward.reward_functions.imagebind.path=$repo_root/verl_omni/utils/reward_score/imagebind.py" \
    '+reward.reward_functions.imagebind.name=compute_score' \
    '+reward.reward_functions.imagebind.weight=1.0' \
    '+reward.reward_functions.imagebind.required=true' \
    "+reward.reward_functions.imagebind.device=cuda:0" \
    "+reward.reward_functions.imagebind.model_name_or_path=$IMGBIND_MODEL_PATH" \
    '+reward.reward_functions.imagebind.mode=text_video' \
    reward.aggregation=weighted_sum \
    trainer.logger='["console", "wandb"]' \
    trainer.project_name=flow_grpo \
    trainer.experiment_name=lance_video_edit_imagebind_lora \
    trainer.log_val_generations=8 \
    trainer.val_before_train=False \
    trainer.n_gpus_per_node=$NUM_GPUS_ACTOR_ROLLOUT_REWARD \
    trainer.nnodes=1 \
    trainer.save_freq=30 \
    trainer.test_freq=30 \
    trainer.total_epochs=15 \
    trainer.total_training_steps=300 "$@"
