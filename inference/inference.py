import argparse
import os
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple
import logging

import yaml
import torch
import numpy as np
from PIL import Image
from transformers import set_seed

from src.pipeline_pe_clone import FluxPipeline
from utils.util import load_image, load_json, load_or_generate, save_image, save_json, setup_logging, visualize_layer_graph
from utils.vlm_util import detect_top_layer, extract_tag_content, detect_mask, parse_segmentation_masks, plot_segmentation_masks, parse_json, merge_bbox_as_mask, SegmentationMask, MASK_PROMPT
from utils.image_util import is_pure_white, pad_image, unpad_image


@dataclass
class Config:
    seed: int
    pretrained_model_name_or_path: str
    lora_path: str
    lora_name: str
    num_inference_steps: int
    guidance_scale: float
    vlm_model_name: str
    enable_mask_detection: bool
    bbox_expansion: int
    mask_detection_temperature: float
    use_layer_graph_reasoning: bool
    max_steps: int
    flux_width: int
    flux_height: int
    vlm_resolution: int
    output_folder: str
    input_folder: str
    max_images: int


def load_model(model_path: str, lora_path: str, lora_name: str) -> FluxPipeline:
    pipeline = FluxPipeline.from_pretrained(model_path, torch_dtype=torch.bfloat16).to("cuda")
    pipeline.load_lora_weights(lora_path, weight_name=f"{lora_name}.safetensors")
    return pipeline


def load_image_paths(input_folder: str) -> List[Tuple[str, str]]:
    input_path = Path(input_folder)
    return [(path.stem, str(path)) for path in input_path.glob("*.png")]


def create_output_subfolders(output_folder: str, target: str) -> Dict[str, str]:
    folders = {
        "target": os.path.join(output_folder, target),
        "png": os.path.join(output_folder, target, "layer_png"),
        "svg": os.path.join(output_folder, target, "layer_svg"),
        "vlm": os.path.join(output_folder, target, "layer_vlm"),
        "mask": os.path.join(output_folder, target, "layer_mask"),
    }

    for folder in folders.values():
        os.makedirs(folder, exist_ok=True)

    return folders


def process_single_target(
    pipeline: FluxPipeline,
    image_path: str,
    target: str,
    cfg: Config,
    logger: logging.Logger,
) -> float:
    try:
        logger.info(f"Processing target: {target}")
        output_folders = create_output_subfolders(cfg.output_folder, target)

        start_time = time.time()
        create_layered_svg(pipeline, image_path, cfg, output_folders, logger)
        processing_time = time.time() - start_time

        logger.info(f"Finished processing target: {target}. Time: {processing_time:.2f}s")
        return processing_time
    except Exception as e:
        logger.error(f"Error processing target {target}: {e}")
        return 0.0


def warp_caption(caption: str) -> str:
    # "The red apple" -> "sksremovelayer, remove the red apple"
    caption = caption[0].lower() + caption[1:]
    caption = "sksremovelayer, remove " + caption
    return caption


def _get_vlm_response(image_vlm: Image.Image, step: int, cfg: Config, output_folders: Dict[str, str], logger: logging.Logger) -> Dict:
    """Gets the VLM response for a step, using cache if available."""
    vlm_response_path = os.path.join(output_folders["vlm"], f"vlm_response_{step}.json")

    def generate():
        raw_response = detect_top_layer(image_vlm, cfg.vlm_model_name, cfg.use_layer_graph_reasoning, logger=logger)
        if cfg.use_layer_graph_reasoning:
            response_dict = {
                "image_description": extract_tag_content("image_description", raw_response),
                "layer_graph_reasoning": extract_tag_content("layer_graph_reasoning", raw_response),
                "layer_graph": json.loads(parse_json(extract_tag_content("layer_graph", raw_response))),
                "non_occluded_analysis": extract_tag_content("non_occluded_analysis", raw_response),
                "caption": extract_tag_content("caption", raw_response)
            }
            layer_graph_path = os.path.join(output_folders["vlm"], f"layer_graph_{step}")
            visualize_layer_graph(response_dict["layer_graph"], logger, output_filename=layer_graph_path, output_format="svg")
        else:
            response_dict = {
                "description": extract_tag_content("description", raw_response),
                "think": extract_tag_content("think", raw_response),
                "caption": extract_tag_content("caption", raw_response)
            }
        return response_dict

    response_dict = load_or_generate(
        file_path=vlm_response_path,
        generate_func=generate,
        save_func=save_json,
        load_func=load_json,
        logger=logger,
        description=f"VLM response for step {step}"
    )

    return response_dict.get("caption") if response_dict else None


def _get_mask_response(
    image_vlm: Image.Image,
    caption: str,
    step: int,
    cfg: Config,
    output_folders: Dict[str, str],
    logger: logging.Logger
) -> List[SegmentationMask]:
    """Gets segmentation masks based on the caption, using cache if available."""
    mask_response_path = os.path.join(output_folders["vlm"], f"mask_response_{step}.json")
    mask_vis_path = os.path.join(output_folders["mask"], f"mask_{step}.png")
    bbox_vis_path = os.path.join(output_folders["mask"], f"bbox_{step}.png")

    def generate_mask_raw():
        raw_response = detect_mask(
            image_vlm,
            cfg.vlm_model_name,
            MASK_PROMPT.format(layers=caption),
            temperature=cfg.mask_detection_temperature,
            logger=logger
        )
        return json.loads(parse_json(raw_response))

    response_dict = load_or_generate(
        file_path=mask_response_path,
        generate_func=generate_mask_raw,
        save_func=save_json,
        load_func=load_json,
        logger=logger,
        description=f"Raw mask VLM response for step {step}"
    )

    if not response_dict:
        logger.error(f"Step {step}: Failed to get raw mask response from VLM.")
        return []

    try:
        segmentation_masks = parse_segmentation_masks(
            response_dict,
            img_height=image_vlm.size[1],
            img_width=image_vlm.size[0]
        )
        logger.info(f"Step {step}: Parsed {len(segmentation_masks)} masks.")
    except Exception as e:
        logger.error(f"Step {step}: Failed to parse segmentation masks: {e}")
        logger.error(f"Raw response was: {response_dict}")
        return []

    # sometimes VLM returns masks smaller than elements
    segmentation_masks = [mask.expand_bbox(cfg.bbox_expansion) for mask in segmentation_masks]

    unpadded_mask: np.ndarray = merge_bbox_as_mask(segmentation_masks, image_vlm.size[0], image_vlm.size[1])
    unpadded_mask_img = Image.fromarray(unpadded_mask)
    padded_mask_img = pad_image(unpadded_mask_img, cfg.flux_width, cfg.flux_height, padding_color=(0, 0, 0), logger=logger)

    # mask visualization
    img_with_masks: Image.Image = plot_segmentation_masks(image_vlm, segmentation_masks)
    img_with_masks.save(mask_vis_path)
    unpadded_mask_img.save(bbox_vis_path)

    return padded_mask_img


def _generate_next_image(
    pipeline: FluxPipeline,
    image_flux: Image.Image,
    mask_image: Image.Image,
    caption: str,
    step: int,
    cfg: Config,
    output_folders: Dict[str, str],
    logger: logging.Logger
) -> Image.Image:
    """Generates the next image in the sequence, using cache if available."""
    output_image_path = os.path.join(output_folders["png"], f"layer_{step}.png")

    def generate():
        generated_image = pipeline(
            prompt=caption,
            condition_image=image_flux,
            mask_image=mask_image,
            height=cfg.flux_height,
            width=cfg.flux_width,
            guidance_scale=cfg.guidance_scale,
            num_inference_steps=cfg.num_inference_steps,
            num_images_per_prompt=1,
            max_sequence_length=512,
        ).images[0]
        return unpad_image(generated_image, cfg.vlm_resolution, cfg.vlm_resolution, logger=logger)

    return load_or_generate(
        file_path=output_image_path,
        generate_func=generate,
        save_func=save_image,
        load_func=load_image,
        logger=logger,
        description=f"output image for step {step}"
    )


def create_layered_svg(pipeline: FluxPipeline, image_path: str, cfg: Config, output_folders: Dict[str, str], logger: logging.Logger) -> None:
    """Generates layered images iteratively until a pure white image is produced."""
    step = 0
    image = None
    try:
        logger.info("")
        logger.info(f"Step {step}: Loading initial image {image_path}...")
        image = load_image(image_path)
        save_image(image, os.path.join(output_folders["png"], f"layer_{step}.png"))

        while not is_pure_white(image):
            logger.info(f"Step {step}: Processing...")

            # Prepare images for VLM and Flux
            image_vlm = image.resize((cfg.vlm_resolution, cfg.vlm_resolution))
            image_flux = pad_image(image_vlm, cfg.flux_width, cfg.flux_height, logger=logger)

            # Get VLM response (cached)
            caption = _get_vlm_response(image_vlm, step, cfg, output_folders, logger)
            if not caption:
                logger.error(f"Step {step}: Could not extract caption from VLM response. Stopping.")
                break

            # Get segmentation masks if enabled (cached)
            mask_image = None
            if cfg.enable_mask_detection:
                mask_image = _get_mask_response(image_vlm, caption, step, cfg, output_folders, logger)

            # Prepare caption for image generation
            caption = warp_caption(caption)
            logger.info(f"Caption: {caption}")

            # Generate next image (cached)
            step += 1
            output_image = _generate_next_image(pipeline, image_flux, mask_image, caption, step, cfg, output_folders, logger)
            if not output_image:
                logger.error(f"Step {step}: Failed to generate image. Stopping.")
                break

            image = output_image

            if step >= cfg.max_steps:
                logger.info(f"Step {step}: Reached max steps. Stopping.")
                break

        if image is None:
             logger.info(f"Target '{os.path.basename(image_path)}' could not be loaded.")
        elif is_pure_white(image):
             logger.info(f"Target '{os.path.basename(image_path)}' finished after {step -1} generation steps.")
        else:
             logger.info(f"Target '{os.path.basename(image_path)}' stopped processing at step {step}.")

    except Exception as e:
        logger.error(f"An error occurred while processing '{os.path.basename(image_path)}' at step {step}: {e}")


def parse_arguments() -> Config:
    parser = argparse.ArgumentParser(description="Inference script for LayerSVG")

    # Flux Settings
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--pretrained_model_name_or_path", type=str, required=True)
    parser.add_argument("--lora_path", type=str, default="kingno/LayerPeeler")
    parser.add_argument("--lora_name", type=str, default="LayerPeeler_rank256_step20000")
    parser.add_argument("--num_inference_steps", type=int, default=30)
    parser.add_argument("--guidance_scale", type=float, default=4.5)

    # VLM Settings
    parser.add_argument("--vlm_model_name", type=str, default="gemini-2.5-pro")
    parser.add_argument("--enable_mask_detection", action="store_true")
    parser.add_argument("--bbox_expansion", type=int, default=15)
    parser.add_argument("--mask_detection_temperature", type=float, default=0.5)
    parser.add_argument("--use_layer_graph_reasoning", action="store_true")

    parser.add_argument("--max_steps", type=int, default=10)

    # Image Settings
    parser.add_argument("--flux_width", type=int, default=512)
    parser.add_argument("--flux_height", type=int, default=512)
    parser.add_argument("--vlm_resolution", type=int, default=512)

    # Input/Output Settings
    parser.add_argument("--output_folder", type=str, required=True)
    parser.add_argument("--input_folder", type=str, required=True)
    parser.add_argument("--max_images", type=int, default=5, help="Maximum number of images to process; for testing")

    args = parser.parse_args()
    os.makedirs(args.output_folder, exist_ok=True)

    # Save config
    config_path = Path(args.output_folder) / "config.yaml"
    config_dict = vars(args)
    config_path.write_text(yaml.dump(config_dict))

    return Config(**config_dict)


def main():
    cfg = parse_arguments()
    logger = setup_logging(cfg.output_folder)
    set_seed(cfg.seed)

    logger.info("===== Loading Model and LoRA weights =====")
    pipeline = load_model(cfg.pretrained_model_name_or_path, cfg.lora_path, cfg.lora_name)

    logger.info("===== Start Generation =====")
    # image_paths = load_image_paths(cfg.input_folder)[:cfg.max_images]
    # image_paths = load_image_paths(cfg.input_folder)[cfg.max_images:]
    image_paths = sorted(load_image_paths(cfg.input_folder))

    successful_targets = 0
    failed_targets = 0
    total_runtime = 0.0

    for target, image_path in image_paths:
        processing_time = process_single_target(pipeline, image_path, target, cfg, logger)
        total_runtime += processing_time
        if processing_time > 0:
            successful_targets += 1
        else:
            failed_targets += 1

    logger.info("===== End Generation =====")
    logger.info(f"[SUMMARY] Total runtime: {total_runtime:.2f} seconds")
    logger.info(f"[SUMMARY] Successfully processed: {successful_targets} targets")
    logger.info(f"[SUMMARY] Average runtime per target: {total_runtime / successful_targets:.2f} seconds")
    logger.info(f"[SUMMARY] Failed to process: {failed_targets} targets")


if __name__ == "__main__":
    main()
