from PIL import Image
import logging


def pad_image(img, target_width=512, target_height=768, padding_color=(255, 255, 255), logger: logging.Logger = None):
    """
    Pads the given PIL Image to the target width and height with a white background, centering the image.
    Returns a new PIL Image object.
    """
    if logger:
        logger.info(f"Padding image {img.size} to ({target_width}, {target_height})")
    if img.mode != "RGB":
        img = img.convert("RGB")
    new_img = Image.new("RGB", (target_width, target_height), padding_color)
    paste_x = (target_width - img.width) // 2
    paste_y = (target_height - img.height) // 2
    new_img.paste(img, (paste_x, paste_y))
    return new_img


def unpad_image(img, target_width=512, target_height=512, logger: logging.Logger = None):
    # Calculate crop coordinates
    if logger:
        logger.info(f"Unpadding image {img.size} to ({target_width}, {target_height})")
    left = (img.width - target_width) // 2
    upper = (img.height - target_height) // 2
    right = left + target_width
    lower = upper + target_height
    
    return img.crop((left, upper, right, lower))


def is_pure_white(img):
    """
    Check if an image is effectively pure white by checking if all RGB channels
    have values very close to 255 (allowing for small variations).
    """
    extrema = img.getextrema()
    return all(min_val >= 240 for min_val, _ in extrema)
