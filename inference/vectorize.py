import os
import cv2
import numpy as np
import base64
import re
import xml.etree.ElementTree as ET
from openai import OpenAI
from PIL import Image
import argparse
import logging
from dotenv import load_dotenv


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

load_dotenv(dotenv_path=".env")
RECRAFT_API_KEY = os.getenv("RECRAFT_API_KEY")

client = OpenAI(base_url='https://external.api.recraft.ai/v1', api_key=RECRAFT_API_KEY)


def vectorize_png(png_path):
    """Vectorizes a PNG file using the Recraft API."""
    logger.info(f"Vectorizing {png_path}...")
    try:
        with open(png_path, 'rb') as f:
            response = client.post(
                path='/images/vectorize',
                cast_to=object,
                options={'headers': {'Content-Type': 'multipart/form-data'}},
                body={'response_format': 'b64_json'},
                files={'file': f},
            )

        # Check response structure for the expected base64 encoded SVG
        if response.get('image') and response['image'].get('b64_json'):
            b64_svg = response['image']['b64_json']
            svg_bytes = base64.b64decode(b64_svg)
            logger.info(f"Successfully vectorized {png_path}")
            return svg_bytes
        else:
            logger.error(f"Error vectorizing {png_path}: Invalid API response format. Response: {response}")
            return None
    except Exception as e:
        logger.error(f"Error during API call for {png_path}: {e}", exc_info=True)
        return None


def save_transparent_png(image_data, path):
    """Saves a NumPy array as a transparent PNG, ensuring BGRA format."""
    try:
        # Ensure the input image has 4 channels (BGRA) for transparency
        if len(image_data.shape) == 2: # Grayscale
            image_bgra = cv2.cvtColor(image_data, cv2.COLOR_GRAY2BGRA)
        elif image_data.shape[2] == 3: # BGR
            image_bgra = cv2.cvtColor(image_data, cv2.COLOR_BGR2BGRA)
        elif image_data.shape[2] == 4: # Assume BGRA
             image_bgra = image_data
        else:
            logger.error(f"Unsupported image shape for saving: {image_data.shape}")
            return False

        # Convert OpenCV's BGRA to Pillow's RGBA for saving with PIL
        image_rgba = cv2.cvtColor(image_bgra, cv2.COLOR_BGRA2RGBA)
        img = Image.fromarray(image_rgba, 'RGBA')
        img.save(path, "PNG")
        logger.debug(f"Saved transparent PNG: {path}")
        return True
    except Exception as e:
        logger.error(f"Error saving transparent PNG {path} with Pillow: {e}", exc_info=True)
        return False


def merge_svgs(svg_files, output_path):
    """Merges multiple SVG files into one, preserving dimensions from the first valid SVG.
       Content from input SVGs is appended sequentially. Files are processed in reverse,
       so the first file in the input list appears visually on top in the final SVG."""
    if not svg_files:
        logger.warning("No SVG files provided for merging.")
        return False

    logger.info(f"Merging {len(svg_files)} SVGs sequentially into {output_path}...")
    base_root = None
    width = None
    height = None
    viewBox = None

    for svg_file in svg_files:
        try:
            tree = ET.parse(svg_file)
            root = tree.getroot()
            if root.tag.lower().endswith('svg'):
                base_root = root
                width = base_root.get('width')
                height = base_root.get('height')
                viewBox = base_root.get('viewBox')
                logger.debug(f"Using dimensions from {svg_file}: w={width}, h={height}, vb={viewBox}")
                break
            else:
                 logger.warning(f"Root element in {svg_file} is not <svg> ({root.tag}). Skipping for dimension check.")
        except (ET.ParseError, FileNotFoundError, IsADirectoryError) as e:
            logger.warning(f"Skipping invalid/missing SVG {svg_file} during dimension check: {e}")
        except Exception as e:
             logger.error(f"Unexpected error reading {svg_file} for dimensions: {e}", exc_info=True)


    if base_root is None:
        logger.error("Could not find a valid base SVG to determine dimensions. Cannot merge.")
        return False

    # Create a new root SVG element with the determined dimensions
    svg_ns = "http://www.w3.org/2000/svg"
    svg_attrs = {}
    if width: svg_attrs['width'] = width
    if height: svg_attrs['height'] = height
    if viewBox: svg_attrs['viewBox'] = viewBox

    new_root = ET.Element('svg', svg_attrs)

    # Process files in REVERSE order. Elements from later files are added first (bottom),
    # elements from the first file are added last (top).
    elements_added_count = 0
    for svg_file in reversed(svg_files): # Iterate in reverse order
        try:
            tree = ET.parse(svg_file)
            root = tree.getroot()
            if not root.tag.lower().endswith('svg'):
                 logger.warning(f"Skipping {svg_file}: Root element is not <svg> ({root.tag}).")
                 continue

            # Add all children of the original SVG's root directly to the new root,
            # skipping metadata or nested svg tags.
            skipped_tags = {ET.QName(svg_ns, 'svg').text, ET.QName(svg_ns, 'metadata').text, "metadata", "svg"}
            file_elements_added = 0
            for element in root:
                if element.tag in skipped_tags:
                    continue
                new_root.append(element) # Append directly to new_root
                file_elements_added += 1

            if file_elements_added > 0:
                 logger.debug(f"Added {file_elements_added} elements from {os.path.basename(svg_file)} to merged SVG.")
                 elements_added_count += file_elements_added
            else:
                 logger.warning(f"No content elements found in {svg_file} to add.")

        except (ET.ParseError, FileNotFoundError, IsADirectoryError) as e:
             logger.error(f"Error processing {svg_file} during merge: {e}")
        except Exception as e:
             logger.error(f"Unexpected error processing {svg_file} during merge: {e}", exc_info=True)


    # Write the merged SVG file
    if elements_added_count == 0:
        logger.warning("No elements were added to the final SVG. Output file might be empty or invalid.")

    try:
        new_tree = ET.ElementTree(new_root)
        ET.register_namespace('', svg_ns) # Register the default SVG namespace
        new_tree.write(output_path, encoding='utf-8', xml_declaration=True)
        logger.info(f"Successfully merged {elements_added_count} elements from {len(svg_files)} SVGs to {output_path}")
        return True
    except Exception as e:
        logger.error(f"Error writing merged SVG {output_path}: {e}", exc_info=True)
        return False


def _calculate_and_save_diff_png(img_i_path, img_i_plus_1_path, output_path, diff_threshold, morph_kernel_size):
    """Reads two images, calculates the difference, refines the mask, extracts the differing region,
       and saves the extracted region as a transparent PNG."""
    try:
        img_i = cv2.imread(img_i_path, cv2.IMREAD_UNCHANGED)
        img_i_plus_1 = cv2.imread(img_i_plus_1_path, cv2.IMREAD_UNCHANGED)

        if img_i is None:
            logger.error(f"Error reading {img_i_path}")
            return False
        if img_i_plus_1 is None:
            logger.error(f"Error reading {img_i_plus_1_path}")
            return False

        # Ensure BGRA format
        img_i_bgra = img_i
        if len(img_i.shape) == 2: img_i_bgra = cv2.cvtColor(img_i, cv2.COLOR_GRAY2BGRA)
        elif img_i.shape[2] == 3: img_i_bgra = cv2.cvtColor(img_i, cv2.COLOR_BGR2BGRA)

        img_i_plus_1_bgra = img_i_plus_1
        if len(img_i_plus_1.shape) == 2: img_i_plus_1_bgra = cv2.cvtColor(img_i_plus_1, cv2.COLOR_GRAY2BGRA)
        elif img_i_plus_1.shape[2] == 3: img_i_plus_1_bgra = cv2.cvtColor(img_i_plus_1, cv2.COLOR_BGR2BGRA)

        # Calculate absolute difference on color channels only
        diff = cv2.absdiff(img_i_bgra[:,:,:3], img_i_plus_1_bgra[:,:,:3])

        # Create a binary mask from the difference
        gray_diff = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(gray_diff, diff_threshold, 255, cv2.THRESH_BINARY)

        # Apply morphological opening to remove small noise
        if morph_kernel_size > 0:
            kernel = np.ones((morph_kernel_size, morph_kernel_size), np.uint8)
            closed_mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
            refined_mask = cv2.morphologyEx(closed_mask, cv2.MORPH_OPEN, kernel)
        else:
            refined_mask = mask

        # Create a transparent background
        result_bgra = np.zeros_like(img_i_bgra)

        # Copy pixels from the *first* image (img_i) where the mask is non-zero
        result_bgra[refined_mask == 255] = img_i_bgra[refined_mask == 255]

        return save_transparent_png(result_bgra, output_path)

    except Exception as e:
        logger.error(f"Error calculating difference between {os.path.basename(img_i_path)} and {os.path.basename(img_i_plus_1_path)}: {e}", exc_info=True)
        return False


def process_id_folder(id_folder_path, diff_threshold, morph_kernel_size):
    """Processes layers within a single ID folder: finds PNGs, calculates differences,
       vectorizes differences to SVGs, merges SVGs, and keeps temporary difference PNGs."""
    logger.info(f"Processing folder: {id_folder_path}")
    layer_png_dir = os.path.join(id_folder_path, "layer_png")
    layer_svg_dir = os.path.join(id_folder_path, "layer_svg")
    temp_dir = os.path.join(id_folder_path, "temp_diff_pngs")

    if not os.path.isdir(layer_png_dir):
        logger.error(f"Layer PNG directory not found: {layer_png_dir}. Skipping folder.")
        return

    try:
        os.makedirs(layer_svg_dir, exist_ok=True)
        os.makedirs(temp_dir, exist_ok=True)
    except OSError as e:
        logger.error(f"Error creating output directories in {id_folder_path}: {e}. Skipping folder.")
        return


    # List and sort PNG files numerically
    try:
        png_files = [f for f in os.listdir(layer_png_dir) if f.startswith("layer_") and f.endswith(".png")]
        png_files.sort(key=lambda x: int(re.search(r'layer_(\d+)\.png$', x).group(1)))
    except Exception as e:
        logger.error(f"Error listing or sorting PNG files in {layer_png_dir}: {e}. Skipping folder.", exc_info=True)
        return

    if len(png_files) < 2:
        logger.warning(f"Less than 2 layer PNGs found in {layer_png_dir}. Cannot calculate differences. Skipping folder.")
        return

    logger.info(f"Found {len(png_files)} layer PNGs. Processing {len(png_files) - 1} difference pairs.")
    generated_svg_paths = []

    # Process Layer Differences (layer_i vs layer_{i+1})
    for i in range(len(png_files) - 1):
        layer_i_name = png_files[i]
        layer_i_plus_1_name = png_files[i+1]
        layer_i_path = os.path.join(layer_png_dir, layer_i_name)
        layer_i_plus_1_path = os.path.join(layer_png_dir, layer_i_plus_1_name)

        layer_svg_name = layer_i_name.replace(".png", ".svg")
        layer_svg_path = os.path.join(layer_svg_dir, layer_svg_name)
        temp_png_path = os.path.join(temp_dir, f"diff_{layer_i_name}")

        logger.info(f"Calculating difference: {layer_i_name} vs {layer_i_plus_1_name}")

        # Calculate diff and save the temporary difference PNG
        diff_success = _calculate_and_save_diff_png(
            layer_i_path, layer_i_plus_1_path, temp_png_path,
            diff_threshold, morph_kernel_size
        )

        if not diff_success:
            logger.warning(f"Failed to create difference PNG for {layer_i_name}. Skipping vectorization for this layer.")
            continue

        # Vectorize the temporary difference PNG
        svg_bytes = vectorize_png(temp_png_path)
        if svg_bytes:
            try:
                with open(layer_svg_path, "wb") as f:
                    f.write(svg_bytes)
                generated_svg_paths.append(layer_svg_path)
                logger.info(f"Saved SVG: {layer_svg_path}")
            except IOError as e:
                 logger.error(f"Error writing SVG file {layer_svg_path}: {e}")
        # Vectorization errors are logged within vectorize_png

    # Merge Generated SVGs
    if generated_svg_paths:
        # Sort SVGs by layer index for correct visual stacking in the merged file
        generated_svg_paths.sort(key=lambda x: int(re.search(r'layer_(\d+)\.svg$', os.path.basename(x)).group(1)))

        current_id = os.path.basename(id_folder_path)
        final_svg_path = os.path.join(layer_svg_dir, f"final_{current_id}.svg")
        merge_success = merge_svgs(generated_svg_paths, final_svg_path)
        if not merge_success:
             logger.error(f"Failed to merge SVGs for {id_folder_path}")
    else:
        logger.warning(f"No SVGs were generated for {id_folder_path}. Skipping merge.")

    logger.info(f"Temporary directory kept: {temp_dir}")


def main():
    # If this list is not empty, only these folders (relative to base_dir or absolute)
    # will be processed *unless* folders are provided via command line.
    SPECIFIC_FOLDERS_TO_PROCESS = [] # Default: empty list means process all unless specified by args

    parser = argparse.ArgumentParser(description="Vectorize differences between layered PNGs and merge into a final SVG.")
    parser.add_argument("--base_dir", default="outputs/testset",
                        help="Base directory containing ID folders (default: outputs)")
    parser.add_argument("--diff_threshold", type=int, default=20,
                        help="Difference threshold for pixel comparison (0-255)")
    parser.add_argument("--morph_kernel_size", type=int, default=3,
                        help="Kernel size for morphological opening (e.g., 3). Use 0 to disable. (default: 3)")

    args = parser.parse_args()

    logger.info(f"Starting processing...")
    logger.info(f"Base Directory: {args.base_dir}")
    logger.info(f"Parameters: Diff Threshold={args.diff_threshold}, Morph Kernel Size={args.morph_kernel_size}")

    folders_to_process = []
    if SPECIFIC_FOLDERS_TO_PROCESS:
        logger.info(f"Processing specific folders defined in script: {SPECIFIC_FOLDERS_TO_PROCESS}")
        folders_to_process = SPECIFIC_FOLDERS_TO_PROCESS
    else:
        logger.info(f"No specific folders provided. Scanning base directory: {args.base_dir}")
        try:
            if not os.path.isdir(args.base_dir):
                 logger.error(f"Base directory '{args.base_dir}' not found or is not a directory.")
                 exit(1)
            folders_to_process = [f.name for f in os.scandir(args.base_dir) if f.is_dir()]
            if not folders_to_process:
                 logger.warning(f"No subdirectories found in {args.base_dir}.")
            else:
                 logger.info(f"Found {len(folders_to_process)} subdirectories to process.")
        except OSError as e:
            logger.error(f"Error scanning base directory {args.base_dir}: {e}", exc_info=True)
            exit(1)

    processed_count = 0
    error_count = 0
    skipped_count = 0

    # Process Each Determined Folder
    if not folders_to_process:
        logger.warning("No folders selected for processing.")
    else:
        logger.info(f"Beginning processing of {len(folders_to_process)} selected folder(s)...")

    folders_to_process = sorted(folders_to_process)
    for folder_ref in folders_to_process:
        # Construct full, normalized path
        target_path = folder_ref if os.path.isabs(folder_ref) else os.path.join(args.base_dir, folder_ref)
        target_path = os.path.normpath(target_path)

        if not os.path.isdir(target_path):
            logger.warning(f"Target folder '{target_path}' (from ref '{folder_ref}') not found or is not a directory. Skipping.")
            skipped_count += 1
            continue

        try:
            final_svg_path = os.path.join(target_path, "layer_svg", f"final_{os.path.basename(target_path)}.svg")
            if os.path.exists(final_svg_path):
                logger.info(f"Skipping {target_path} because final SVG already exists.")
                continue
            process_id_folder(
                target_path,
                args.diff_threshold,
                args.morph_kernel_size
            )
            processed_count += 1
        except Exception as e:
            logger.error(f"Unhandled exception processing folder {target_path}: {e}", exc_info=True)
            error_count += 1
    # End Folder Processing Loop


    # Final Summary
    logger.info(f"\nProcessing complete.")
    logger.info(f"Successfully processed {processed_count} folder(s).")
    if skipped_count > 0:
        logger.warning(f"Skipped {skipped_count} folder reference(s) due to path issues.")
    if error_count > 0:
        logger.warning(f"Encountered errors during processing in {error_count} folder(s). Check logs above.")
    # End Final Summary


if __name__ == "__main__":
    main()
