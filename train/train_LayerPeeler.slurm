#!/bin/bash
#SBATCH --job-name=LayerPeeler
#SBATCH --partition=special_cs
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=8
#SBATCH --mem=800GB
#SBATCH --time=1-00:00:00
#SBATCH --account=pa_cs_department
#SBATCH --output=logs/LayerPeeler_rank256%j.out
#SBATCH --error=logs/LayerPeeler_rank256%j.err

export EXPERIMENT_NAME="LayerPeeler_rank256"
export OUTPUT_DIR="outputs/$EXPERIMENT_NAME"
export LOG_PATH="$OUTPUT_DIR/log"
export MODEL_DIR="../PhotoDoodle_Pretrain"
export CONFIG="./default_config.yaml"
export TRAIN_DATA="data/LayerPeeler/train.jsonl"

# Go to working directory
cd $SLURM_SUBMIT_DIR

# Run training
accelerate launch --config_file $CONFIG train_lora_flux_pe.py \
    --pretrained_model_name_or_path $MODEL_DIR \
    --width 512 \
    --height 512 \
    --source_column="source" \
    --target_column="target" \
    --caption_column="text" \
    --output_dir=$OUTPUT_DIR \
    --logging_dir=$LOG_PATH \
    --mixed_precision="bf16" \
    --train_data_dir=$TRAIN_DATA \
    --rank=256 \
    --learning_rate=1e-4 \
    --lr_scheduler "cosine" \
    --train_batch_size=1 \
    --gradient_accumulation_steps=8 \
    --num_validation_images=1 \
    --report_to "tensorboard" \
    --validation_image \
        "data/LayerPeeler/valid/303734_original.png" \
        "data/LayerPeeler/valid/303734_step_1.png" \
        "data/LayerPeeler/valid/303734_step_2.png" \
        "data/LayerPeeler/valid/303734_step_3.png" \
        "data/LayerPeeler/valid/317068_original.png" \
        "data/LayerPeeler/valid/317068_step_1.png" \
        "data/LayerPeeler/valid/317068_step_2.png" \
        "data/LayerPeeler/valid/317068_step_3.png" \
        "data/LayerPeeler/valid/368346_original.png" \
        "data/LayerPeeler/valid/368346_step_1.png" \
        "data/LayerPeeler/valid/368346_step_2.png" \
        "data/LayerPeeler/valid/404754_original.png" \
        "data/LayerPeeler/valid/404754_step_1.png" \
        "data/LayerPeeler/valid/404754_step_2.png" \
        "data/LayerPeeler/valid/404754_step_3.png" \
    --validation_prompt \
        "sksremovelayer, remove the shadow below the bottle top, the white highlight, four purple circles." \
        "sksremovelayer, remove pink top part of the bottle, curved purple potion liquid." \
        "sksremovelayer, remove the pink potion bottle." \
        "sksremovelayer, remove the peach colored shape." \
        "sksremovelayer, remove two bubbles, fish's eye, fish tail" \
        "sksremovelayer, remove the orange fish" \
        "sksremovelayer, remove the blue liquid inside the container" \
        "sksremovelayer, remove the rounded, light blue semi-circle with a flat base." \
        "sksremovelayer, remove rim of the glasses, arc of the glasses, highlight of the glasses, and the mouth." \
        "sksremovelayer, remove the left eye outline, the right eye outline." \
        "sksremovelayer, remove left teal eye, right teal eye" \
        "sksremovelayer, remove the light-blue halo, two brown oval eyes" \
        "sksremovelayer, remove the brown hair, the curved mouth." \
        "sksremovelayer, remove the brown circle" \
        "sksremovelayer, remove three light blue cloud shapes." \
    --max_train_steps=50000 \
    --validation_steps=2500 \
    --checkpointing_steps=2500
