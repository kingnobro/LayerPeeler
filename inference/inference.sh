EXPERIMENT_NAME="testset"

# FLUX Settings
MODEL_DIR="../PhotoDoodle_Pretrain"
LORA_PATH="kingno/LayerPeeler"
LORA_NAME="LayerPeeler_rank256_step20000"
NUM_INFERENCE_STEPS=40
GUIDANCE_SCALE=4.5

# VLM Settings
VLM_MODEL_NAME="gemini-2.5-pro"

# Image settings
VLM_RESOLUTION=512
FLUX_WIDTH=512
FLUX_HEIGHT=512

# Input/Output settings
INPUT_FOLDER="data/testset"
OUTPUT_FOLDER="outputs/${EXPERIMENT_NAME}"
MAX_IMAGES=9999  # Number of images to process; for testing

# Run inference
CUDA_VISIBLE_DEVICES=0 python inference.py \
    --pretrained_model_name_or_path "${MODEL_DIR}" \
    --lora_path "${LORA_PATH}" \
    --lora_name "${LORA_NAME}" \
    --num_inference_steps "${NUM_INFERENCE_STEPS}" \
    --guidance_scale "${GUIDANCE_SCALE}" \
    --vlm_model_name "${VLM_MODEL_NAME}" \
    --flux_width "${FLUX_WIDTH}" \
    --flux_height "${FLUX_HEIGHT}" \
    --vlm_resolution "${VLM_RESOLUTION}" \
    --output_folder "${OUTPUT_FOLDER}" \
    --input_folder "${INPUT_FOLDER}" \
    --max_images "${MAX_IMAGES}" \
    --use_layer_graph_reasoning
