import os
import io
import re
import time
import json
import base64
import requests
import logging
import dataclasses
import numpy as np
from PIL import Image, ImageColor, ImageDraw
from dotenv import load_dotenv
from typing import Dict, Any, Optional, Tuple


class VLMConfigError(Exception):
    """Custom exception for VLM configuration errors"""
    pass

class VLMAPIError(Exception):
    """Custom exception for VLM API related errors"""
    pass


# API Configuration
BASE_URL = "https://us.vveai.com/v1/chat/completions"

DEFAULT_PROMPT = '''Analyze the provided flat-color, cartoon-style image. Based purely on the visual presentation, infer the visual stacking and occlusion of elements. Your goal is to identify **only** the elements that are visually positioned at the very top, meaning they are **not covered by any other element** within the image. Provide a concise caption describing these non-occluded elements.

# Core Task:
Identify *all* visual elements that are completely unobstructed by *any other element* in the image, based on visual analysis of overlaps and occlusion. These are the non-occluded visual elements.

# Definition of Non-Occluded Element:
An element is considered non-occluded if and only if no other element in the image is visually positioned above *any part* of it. This must be determined by analyzing how elements visually overlap and obscure each other in the rendered image.

# Instructions for Description and Caption:
1.  **Semantic Description:** If a non-occluded element has a clear real-world meaning (e.g., "eye", "hat", "wheel", "liquid"), describe it using its **meaning** and key visual attributes (e.g., "the green left eye," "the tall blue hat", "the black outline of the computer screen", "the light blue texture over the bottle body").
2.  **Geometric Description:** If a non-occluded element's meaning is abstract or unclear, describe it using its primary **geometric shape, color, and relative position** (e.g., "the large red circle in the center", "the thin black outline curving upwards on the right").
3.  The white background filling the entire canvas is not considered a specific element. Ignore it.
4.  White shapes *inside* other shapes (e.g., liquid, highlights, inner details) are separate elements and *should* be considered if they are non-occluded and visible.
5.  **Object Count Limit:** If there are too many non-occluded elements (e.g., >= 5), consider grouping the elements to describe them collectively (e.g., "the black left eye, the black right eye" → "the black eyes"), or return only a subset of them based on similarity (i.e., position, color, shape, etc.).

# Output Format:
*   **Description:** In `<description>...</description>` tags, provide a concise description of the overall image content or key visible components. This sets the context.
*   **Think Process:** Enclose your reasoning within `<think>...</think>` tags. **Describe your visual analysis process.** Identify the distinct visual elements and **analyze how they overlap and obscure each other** to infer the visual stacking order. Based on this visual stacking determined from occlusion, identify the element(s) that are not covered by *any* other element. Explain why elements that are partially or fully covered by others are excluded. For example, if element A is visually covering element B, B is not non-occluded. If B is covering C, C is not non-occluded. Only elements with *no* part visually obscured by *any other element* are non-occluded.
*   **Caption:** Enclose the final description of *all* identified non-occluded element(s) within `<caption>...</caption>` tags. This should be a list or combined description of the non-occluded elements found. The caption should be LESS than 40 words. To reduce word count, you may need to group similar elements or describe them collectively.

# Example 0 (Semantic Element):
###
<description>
The image shows a simple cartoon face. The face has two black circular eyes positioned symmetrically, a curved red line forming a smile, and a yellow star-shaped decoration placed on top of the head. Two hands with darker color curves cover portions of the face.
</description>
<think>
I am analyzing the visual image for elements and how they overlap. I see distinct shapes: the face, eyes, mouth, star, and hands. I observe that the hands cover parts of the face. However, the eyes, mouth, and star are fully visible and no part of them is covered by the hands or any other element in the image. The hands themselves are not single-color continuous shapes, they have darker color curves inside them. So hands are not non-occluded.
</think>
<caption>The black left eye, the black right eye, the red mouth, and the yellow star on the hair</caption>
###

# Example 1 (Semantic Element - Partial Covering):
###
<description>
The image contains a pink peach with two green leaves and a dark-red curve on the peach body. The right leaf is partially covered by the peach body.
</description>
<think>
I am analyzing the visual image for overlaps. I see the peach, the dark-red curve on the peach, and two leaves. The dark-red curve is fully visible on the peach. The left leaf is fully visible. The right leaf has a portion of its shape obscured by the peach body. Therefore, the dark-red curve and the left leaf are the elements not covered by any other element.
</think>
<caption>The dark-red curve on the peach body and the left leaf</caption>
###

# Example 2 (Nested Covering - Abstract Elements):
###
<description>
The image contains a black circle visually positioned over a green rectangle, which is visually positioned over a yellow triangle.
</description>
<think>
I am analyzing the visual image for overlaps. I see a black circle, a green rectangle, and a yellow triangle. The black circle visually covers the green rectangle. The green rectangle visually covers the yellow triangle. This establishes a visual stacking order: Circle > Rectangle > Triangle. The black circle is not visually covered by any other element. The green rectangle is covered by the black circle, and the yellow triangle is covered by the green rectangle. Therefore, only the black circle is non-occluded.
</think>
<caption>The black circle in the center</caption>
###

# Example 3 (Multiple Unrelated Non-Occluded Elements):
###
<description>
The image shows a small blue square in the top-left corner and a large yellow triangle dominating the center. These two elements are visually separate and do not overlap.
</description>
<think>
I am analyzing the visual image for overlaps. I see a small blue square and a large yellow triangle. These two elements do not visually overlap at all. The yellow triangle is fully visible and not covered by any other element. The small blue square is also fully visible and not covered by any other element. Both meet the criteria for being non-occluded elements as nothing obscures them.
</think>
<caption>The large yellow triangle and the small blue square in the top-left</caption>
###

Proceed with the analysis of the provided image.
'''

LAYER_GRAPH_PROMPT = '''You are an advanced Vision-Language Model tasked with analyzing cartoon-style, flat-color images to construct a simplified 2D layer graph and identify non-occluded visual elements. Your analysis will form the basis for a layered image decomposition process, where layers correspond to distinct color regions.

**Core Task:**
1.  **Image Analysis:** Analyze the provided flat-color, cartoon-style image.
2.  **Layer Graph Construction:** Infer and describe a 2D layer graph.
    * **Nodes:** Identify distinct, continuous flat-color regions within the image. **Each such visually separable color region constitutes a node.** These regions are the fundamental building blocks for the layered structure.
    * **Node Attributes:** For each node (color region), describe:
        * Its dominant color (e.g., "red," "light blue").
        * A semantic label if the region clearly corresponds to a recognizable object part (e.g., "the character's skin," "the hat's brim," "the left eye's pupil," "a highlight on the sphere").
        * If no clear semantic label applies, use a geometric description (e.g., "large circular red area," "thin black outline").
        * `part_of_object(N, O)`: Color-region Node N is recognized as a component of a larger semantic object O (e.g., a 'blue hat body' `part_of_object` 'hat'; a 'highlight in the eye' `part_of_object` 'eye'). This helps group related color regions conceptually.
    * **Relationships (Edges):** Define the spatial and occlusion relationships between these color-region nodes. Key relationship types are:
        * `occludes(A, B)`: Color-region Node A visually covers (partially or fully) Color-region Node B.
        * `interrupted_shape(A, B)`: Color-region Node A and Color-region Node B share the same color and are conceptually part of the same larger shape, but are visually disconnected because another region occludes the space and splits the shape into two or more parts. If the occluding region(s) were removed, A and B would form a single continuous visual entity. If A and B are interrupted shapes, then it means there is another color region that occludes the space between them.
3.  **Non-Occluded Element Identification:** Based *solely* on the `occludes` relationships in your constructed layer graph, identify all color-region nodes that are not occluded by any other node.
4.  **Caption Generation:** Provide a concise caption describing these non-occluded color-region nodes, using their semantic or geometric descriptions.

**Instructions & Definitions:**

* **Node Granularity:** **Each continuous patch of flat color must be treated as a distinct node.** If an apparent 'object' (e.g., a 'hat') is composed of multiple color areas (e.g., 'blue hat body,' 'yellow hat band,' 'white highlight on hat'), each of these color areas must be a separate node in the graph. Do not merge distinct color regions into a single 'object' node at the graph construction stage.
* **Non-Occluded Element Definition (Graph-based):** A color-region node in the layer graph is considered non-occluded if and only if there are no `occludes(X, Node)` relationships where `Node` is the element in question (i.e., no other color-region node X occludes it).
* **Background Exclusion:** The global white background of the canvas is not considered a color-region node for the layer graph.
* **Internal Details:** White or other colored shapes *inside* what might be perceived as a larger object (e.g., highlights, pupils within eyes, patterns on clothes) are themselves distinct color-region nodes. Their relationship to surrounding/underlying color-region nodes must be captured.
* **Concise Captioning:**
    * If multiple, similar non-occluded color-region nodes exist and belong to the same semantic object (e.g., "the white highlight on the left eye," "the white highlight on the right eye"), group them in the caption (e.g., "the white highlights on the eyes").
    * Describe the elements using their semantic labels or geometric descriptions. **Avoid using the word 'region' in the final caption** (e.g., "the blue hat band," not "the blue hat band region").
    * The caption should be LESS than 40 words.

**Output Format:**

* **Image Description:** `<image_description>...</image_description>`
    Provide a brief overall description of the image content.

* **Layer Graph Construction Reasoning:** `<layer_graph_reasoning>...</layer_graph_reasoning>`
    Describe your thought process for segmenting the image into flat color-region nodes and the visual cues used to determine their relationships, especially occlusion. Explain how you decided on the chosen nodes and edges.

* **Layer Graph (Simplified JSON-like Representation):** `<layer_graph>...</layer_graph>`
    Represent the graph with nodes and edges. Use unique IDs for nodes.
    Example (reflecting color-region nodes for a hat with a band and star):
    ```json
    {
      "nodes": [
        {"id": "N1", "description": "character's head skin", "color": "peach"},
        {"id": "N2", "description": "upper hat body", "color": "blue", "part_of_object": "hat"},
        {"id": "N3", "description": "hat band", "color": "yellow", "part_of_object": "hat"},
        {"id": "N4", "description": "star decoration on hat", "color": "gold", "part_of_object": "hat"},
        {"id": "N5", "description": "left eye pupil", "color": "black", "part_of_object": "left eye"},
        {"id": "N6", "description": "highlight on left eye pupil", "color": "white", "part_of_object": "left eye"},
        {"id": "N7", "description": "lower hat body", "color": "blue", "part_of_object": "hat"}
      ],
      "edges": [
        // Assuming hat body covers head, band covers hat body and divides the hat into upper and lower parts, star covers hat body (or band)
        {"source": "N7", "target": "N1", "relationship": "occludes"},
        {"source": "N3", "target": "N2", "relationship": "occludes"},
        {"source": "N3", "target": "N7", "relationship": "occludes"},
        {"source": "N4", "target": "N2", "relationship": "occludes"}, // Star occludes main hat body
        {"source": "N6", "target": "N5", "relationship": "occludes"}, // Highlight occludes pupil
        {"source": "N7", "target": "N2", "relationship": "interrupted_shape"} // Lower hat body and upper hat body are interrupted shapes
      ]
    }
    ```

* **Non-Occluded Element Analysis from Graph:** `<non_occluded_analysis>...</non_occluded_analysis>`
    Explain which color-region nodes are identified as non-occluded by analyzing the `occludes` relationships in the `<layer_graph>`. Specifically, list the nodes that are not targets in any `occludes(X, Node)` relationship.

* **Caption:** `<caption>...</caption>`
    Provide the final concise caption of *all* identified non-occluded color-region nodes.

**Example:**

---
**Input Image:** (Imagine a cartoon cat. The cat wears a blue hat. A wide red band is on the hat, and a prominent yellow feather is attached, clearly covering the red band. One of the cat's grey ears is visible, but the other is tucked under and occluded by the blue hat. The cat is holding a large green fish, which covers most of its grey body and one grey paw. The cat's face shows a pink nose and black mouth. Its eyes have white sclera, black pupils, and tiny white highlights on the pupils.)

**Output:**

<image_description>
A cartoon cat is shown wearing a blue hat with a red band and a yellow feather. One ear is visible, the other is under the hat. The cat holds a large green fish occluding its body and a paw.
</image_description>

<layer_graph_reasoning>
I segmented the image into distinct flat-color areas.
- Cat: grey body, grey head, grey visible ear, pink nose, black mouth, white sclera (for two eyes), black pupils (for two eyes), small white highlights on pupils (for two eyes), grey visible paw (small part maybe).
- Hat: blue main hat, red band, yellow feather.
- Held item: green fish.
Occlusion analysis was performed as follows:
- The blue hat occludes part of the grey head and one (unseen) grey ear.
- The red band occludes part of the blue hat.
- The yellow feather occludes part of the red band.
- The green fish occludes most of the grey body and one grey paw.
- Each white highlight occludes its respective black pupil.
- Each black pupil occludes its respective white sclera.
</layer_graph_reasoning>

<layer_graph>
```json
{
  "nodes": [
    {"id": "N1", "description": "cat body", "color": "grey", "part_of_object": "cat"},
    {"id": "N2", "description": "cat head", "color": "grey", "part_of_object": "cat"},
    {"id": "N3", "description": "visible cat ear", "color": "grey", "part_of_object":"head"},
    {"id": "N4", "description": "occluded cat ear", "color": "grey", "part_of_object":"head"},
    {"id": "N5", "description": "left eye sclera", "color": "white", "part_of_object": "left eye"},
    {"id": "N6", "description": "right eye sclera", "color": "white", "part_of_object": "right eye"},
    {"id": "N7", "description": "left pupil", "color": "black", "part_of_object": "left eye"},
    {"id": "N8", "description": "right pupil", "color": "black", "part_of_object": "right eye"},
    {"id": "N9", "description": "highlight on left pupil", "color": "white", "part_of_object": "left eye"},
    {"id": "N10", "description": "highlight on right pupil", "color": "white", "part_of_object": "right eye"},
    {"id": "N11", "description": "nose", "color": "pink", "part_of_object": "head"},
    {"id": "N12", "description": "mouth", "color": "black", "part_of_object": "head"},
    {"id": "N13", "description": "main hat body", "color": "blue", "part_of_object": "hat"},
    {"id": "N14", "description": "hat band", "color": "red", "part_of_object": "hat"},
    {"id": "N15", "description": "feather on hat", "color": "yellow", "part_of_object": "hat"},
    {"id": "N16", "description": "held green fish", "color": "green", "part_of_object": "held item"},
    {"id": "N17", "description": "visible cat paw", "color": "grey", "part_of_object": "cat"},
    {"id": "N18", "description": "occluded cat paw", "color": "grey", "part_of_object": "cat"}
  ],
  "edges": [
    {"source": "N13", "target": "N2", "relationship": "occludes"}, // Hat occludes head
    {"source": "N13", "target": "N4", "relationship": "occludes"}, // Hat occludes ear N4
    {"source": "N14", "target": "N13", "relationship": "occludes"},// Band occludes hat body
    {"source": "N15", "target": "N14", "relationship": "occludes"},// Feather occludes band
    {"source": "N16", "target": "N1", "relationship": "occludes"},  // Fish occludes body
    {"source": "N16", "target": "N18", "relationship": "occludes"},// Fish occludes paw N18
    {"source": "N7", "target": "N5", "relationship": "occludes"},   // Pupil occludes sclera
    {"source": "N8", "target": "N6", "relationship": "occludes"},   // Pupil occludes sclera
    {"source": "N9", "target": "N7", "relationship": "occludes"},   // Highlight occludes pupil
    {"source": "N10", "target": "N8", "relationship": "occludes"}  // Highlight occludes pupil
  ]
}
</layer_graph>

<non_occluded_analysis>
Based on the layer graph, the following color-region nodes are not targets in any occludes(X, Node) relationship:
N3 (visible cat ear), N9 (highlight on left pupil), N10 (highlight on right pupil), N11 (nose), N12 (mouth), N15 (feather on hat), N16 (held green fish), and N17 (visible cat paw).
The 'occluded cat ear' (N4) is occluded by N13. 'Left/Right eye sclera' (N5, N6) are occluded by pupils. 'Left/Right pupil' (N7, N8) are occluded by highlights. 'Main hat body' (N13) is occluded by band. 'Hat band' (N14) is occluded by feather. 'Cat body' (N1) is occluded by fish. 'Occluded cat paw' (N18) is occluded by fish.
Therefore, the non-occluded nodes are: N3, N9, N10, N11, N12, N15, N16, N17.
</non_occluded_analysis>
<caption>
The grey visible ear, white highlights on pupils, pink nose, black mouth, yellow feather, green fish, and grey visible paw.
</caption>
'''

MAINTAIN_LAYER_GRAPH_PROMPT = '''You are an advanced Vision-Language Model tasked with analyzing cartoon-style, flat-color images. You will be given the current image and the layer graph that was constructed for a *previous* version of this image (before an editing operation was applied). Your goal is to:

1.  **Verify, Correct, and Update Layer Graph:** Compare the provided *previous layer graph* with the *current image*.
    *   If the previous layer graph, after any necessary corrections, accurately represents the current image (i.e., the layer removal was successful and the corrected graph reflects the visible elements and their occlusions correctly), you can use this corrected version as a basis.
    *   If the previous layer graph is **no longer accurate** due to changes in the image (e.g., layers were not removed as expected, new artifacts appeared, occlusion relationships have changed), OR if you identified errors in the previous graph itself, you **must update or reconstruct the layer graph** to accurately reflect the *current image*. This might involve adding, removing, or modifying nodes (color regions) and their attributes or relationships (occlusion, interrupted shapes).
    *   Focus on creating a graph that represents the **current visual state** of the image with the highest possible accuracy.

2.  **Layer Graph Construction (if updating/correcting):** If updating or correcting, follow these rules:
    *   **Nodes:** Identify distinct, continuous flat-color regions within the current image. Each such visually separable color region constitutes a node.
    *   **Node Attributes:** For each node (color region), describe:
        *   Its dominant color.
        *   A semantic label if the region clearly corresponds to a recognizable object part.
        *   If no clear semantic label applies, use a geometric description.
        *   `part_of_object(N, O)`: Color-region Node N is part of a larger semantic object O.
    *   **Relationships (Edges):** Define spatial and occlusion relationships:
        *   `occludes(A, B)`: Color-region Node A visually covers Color-region Node B in the current image.
        *   `interrupted_shape(A, B)`: Color-region Node A and B share the same color and are conceptually part of the same larger shape but are visually disconnected in the current image.

3.  **Non-Occluded Element Identification (from updated/corrected graph):** Based *solely* on the `occludes` relationships in the current (potentially updated/corrected) layer graph, identify all color-region nodes that are not occluded by any other node in the *current image*.

4.  **Caption Generation:** Provide a concise caption describing these non-occluded color-region nodes, using their semantic or geometric descriptions.

**Instructions & Definitions:**

*   **Node Granularity:** Each continuous patch of flat color must be a distinct node.
*   **Non-Occluded Element Definition (Graph-based):** A color-region node is non-occluded if no `occludes(X, Node)` relationship exists for it in the *current graph*.
*   **Background Exclusion:** The global white background is not a node.
*   **Internal Details:** White or other colored shapes *inside* objects are distinct color-region nodes.
*   **Concise Captioning:** Group similar non-occluded elements if they belong to the same semantic object. Describe elements using semantic/geometric labels. Avoid 'region' in the caption. Caption < 40 words.

**Input Context (User will provide):**
*   The current image.
*   The layer graph from the *previous* step, formatted as JSON. This graph needs to be checked against the current image, corrected for any existing errors, and updated if necessary.

**Output Format (Strictly Adhere):**

*   **Image Description:** `<image_description>...</image_description>`
    Brief overall description of the *current* image content.

*   **Layer Graph Update Reasoning:** `<layer_graph_reasoning>...</layer_graph_reasoning>`
    Describe your thought process. **Critically, explain if the previous graph was suitable, if and why it needed updates to match the current image, and importantly, if any corrections were made to the previous graph's structure or interpretation based on the current visual evidence, irrespective of the editing operation.** Detail changes made to nodes or edges if any. If no changes were needed beyond validating the previous graph, state that.

*   **Layer Graph (JSON-like Representation):** `<layer_graph>...</layer_graph>`
    Represent the **final, accurate layer graph for the current image**. This will be the validated, corrected, or newly updated one.
    Example structure:
    ```json
    {
      "nodes": [
        {"id": "N1", "description": "...", "color": "...", "part_of_object": "..."}, ...
      ],
      "edges": [
        {"source": "...", "target": "...", "relationship": "..."}, ...
      ]
    }
    ```

*   **Non-Occluded Element Analysis from Graph:** `<non_occluded_analysis>...</non_occluded_analysis>`
    Explain which color-region nodes are identified as non-occluded by analyzing the `occludes` relationships in the `<layer_graph>` you are outputting (the one for the current image).

*   **Caption:** `<caption>...</caption>`
    Provide the final concise caption of *all* identified non-occluded color-region nodes from the *current image*.

**Example Scenario:**
Suppose the previous image showed a "blue square occluding a red circle". The previous graph would be:
Nodes: N1 (blue square), N2 (red circle). Edges: occludes(N1, N2). Non-occluded: N1 (blue square). Caption: "The blue square".
Now, an editing operation attempted to remove the "blue square".
- **If successful:** The current image shows only "red circle".
  Your reasoning: "The blue square was successfully removed. The previous graph is no longer valid. Updated graph contains only the red circle."
  Updated graph: Nodes: N2 (red circle). Edges: []. Non-occluded: N2 (red circle). Caption: "The red circle".
- **If failed:** The current image still shows "blue square occluding red circle".
  Your reasoning: "The editing operation to remove the blue square failed. The previous graph still accurately represents the current image."
  Graph: (Same as previous). Non-occluded: N1 (blue square). Caption: "The blue square".
- **If partially failed / new artifact:** Current image shows "a smaller blue square fragment and the red circle".
  Your reasoning: "The blue square was partially removed. The previous graph needs update to reflect the smaller blue fragment."
  Updated graph: Nodes: N1_frag (blue square fragment), N2 (red circle). Edges: occludes(N1_frag, N2). Non-occluded: N1_frag. Caption: "The blue square fragment".

Proceed with the analysis of the provided current image, considering the provided previous layer graph for verification, correction, and updates.
'''

MASK_PROMPT = '''Give the segmentation masks for the "{layers}". Output a JSON list of segmentation masks where each entry contains the 2D bounding box in the key "box_2d", the segmentation mask in key "mask", and the text label in the key "label". Use descriptive labels.'''

additional_colors = [colorname for (colorname, colorcode) in ImageColor.colormap.items()]


def _initialize_api():
    load_dotenv(dotenv_path=".env")
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise VLMConfigError("API_KEY not found in environment variables")
    
    return {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }


try:
    API_HEADERS = _initialize_api()
except Exception as e:
    raise VLMConfigError(f"Failed to initialize API: {str(e)}")


@dataclasses.dataclass(frozen=True)
class SegmentationMask:
    # bounding box pixel coordinates (not normalized)
    y0: int  # in [0..height - 1]
    x0: int  # in [0..width - 1]
    y1: int  # in [0..height - 1]
    x1: int  # in [0..width - 1]
    mask: np.array  # [img_height, img_width] with values 0..255
    label: str

    def expand_bbox(self, expansion_value: int) -> 'SegmentationMask':
        new_y0 = self.y0 - expansion_value
        new_x0 = self.x0 - expansion_value
        new_y1 = self.y1 + expansion_value
        new_x1 = self.x1 + expansion_value

        return SegmentationMask(
            y0=new_y0,
            x0=new_x0,
            y1=new_y1,
            x1=new_x1,
            mask=self.mask,
            label=self.label
        )


def merge_bbox_as_mask(masks: list[SegmentationMask], width: int, height: int) -> np.array:
    merged_mask = np.zeros((height, width), dtype=np.uint8)
    for mask in masks:
        merged_mask[mask.y0:mask.y1, mask.x0:mask.x1] = 255
    return merged_mask


def parse_segmentation_masks(
    items: list[Dict[str, Any]], *, img_height: int, img_width: int
) -> list[SegmentationMask]:
    masks = []
    for item in items:
        raw_box = item["box_2d"]
        abs_y0 = int(item["box_2d"][0] / 1000 * img_height)
        abs_x0 = int(item["box_2d"][1] / 1000 * img_width)
        abs_y1 = int(item["box_2d"][2] / 1000 * img_height)
        abs_x1 = int(item["box_2d"][3] / 1000 * img_width)
        if abs_y0 >= abs_y1 or abs_x0 >= abs_x1:
            print("Invalid bounding box", item["box_2d"])
            continue
        label = item["label"]
        png_str = item["mask"]
        if not png_str.startswith("data:image/png;base64,"):
            print("Invalid mask")
            continue
        png_str = png_str.removeprefix("data:image/png;base64,")
        png_str = base64.b64decode(png_str)
        mask = Image.open(io.BytesIO(png_str))
        bbox_height = abs_y1 - abs_y0
        bbox_width = abs_x1 - abs_x0
        if bbox_height < 1 or bbox_width < 1:
            print("Invalid bounding box")
            continue
        mask = mask.resize(
            (bbox_width, bbox_height), resample=Image.Resampling.BILINEAR
        )
        np_mask = np.zeros((img_height, img_width), dtype=np.uint8)
        np_mask[abs_y0:abs_y1, abs_x0:abs_x1] = mask
        masks.append(SegmentationMask(abs_y0, abs_x0, abs_y1, abs_x1, np_mask, label))
    return masks


def overlay_mask_on_img(
    img: Image, mask: np.ndarray, color: str, alpha: float = 0.7
) -> Image.Image:
    if not (0.0 <= alpha <= 1.0):
        raise ValueError("Alpha must be between 0.0 and 1.0")

    # Convert the color name string to an RGB tuple
    try:
        color_rgb: Tuple[int, int, int] = ImageColor.getrgb(color)
    except ValueError as e:
        raise ValueError(
            f"Invalid color name '{color}'. Supported names are typically HTML/CSS color names. Error: {e}"
        )

    # Prepare the base image for alpha compositing
    img_rgba = img.convert("RGBA")
    width, height = img_rgba.size

    # Create the colored overlay layer
    # Calculate the RGBA tuple for the overlay color
    alpha_int = int(alpha * 255)
    overlay_color_rgba = color_rgb + (alpha_int,)

    # Create an RGBA layer (all zeros = transparent black)
    colored_mask_layer_np = np.zeros((height, width, 4), dtype=np.uint8)

    # Mask has values between 0 and 255, threshold at 127 to get binary mask.
    mask_np_logical = mask > 127

    # Apply the overlay color RGBA tuple where the mask is True
    colored_mask_layer_np[mask_np_logical] = overlay_color_rgba

    # Convert the NumPy layer back to a PIL Image
    colored_mask_layer_pil = Image.fromarray(colored_mask_layer_np, "RGBA")

    # Composite the colored mask layer onto the base image
    result_img = Image.alpha_composite(img_rgba, colored_mask_layer_pil)

    return result_img


def plot_segmentation_masks(img: Image, segmentation_masks: list[SegmentationMask]):
    # Define a list of colors
    colors = [
        "red",
        "green",
        "blue",
        "yellow",
        "orange",
        "pink",
        "purple",
        "brown",
        "gray",
        "beige",
        "turquoise",
        "cyan",
        "magenta",
        "lime",
        "navy",
        "maroon",
        "teal",
        "olive",
        "coral",
        "lavender",
        "violet",
        "gold",
        "silver",
    ] + additional_colors

    # Overlay the mask
    for i, mask in enumerate(segmentation_masks):
        color = colors[i % len(colors)]
        img = overlay_mask_on_img(img, mask.mask, color)

    # Create a drawing object
    draw = ImageDraw.Draw(img)

    # Draw the bounding boxes
    for i, mask in enumerate(segmentation_masks):
        color = colors[i % len(colors)]
        draw.rectangle(((mask.x0, mask.y0), (mask.x1, mask.y1)), outline=color, width=4)

    # Draw the text labels
    for i, mask in enumerate(segmentation_masks):
        color = colors[i % len(colors)]
        if mask.label != "":
            draw.text((mask.x0 + 8, mask.y0 - 20), mask.label, fill=color)
    return img


def image_to_base64_from_pil(image: Image.Image, format: str = "PNG") -> str:
    buffered = io.BytesIO()
    image.save(buffered, format=format)
    img_str = base64.b64encode(buffered.getvalue()).decode('utf-8')
    return img_str


def create_payload(image_base64: str, model_name: str, prompt: str, temperature: float = None, existing_layer_graph_json: Optional[str] = None) -> str:
    messages = [
        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_base64}"}},
        {"type": "text", "text": prompt}
    ]
    if existing_layer_graph_json:
        messages.append({"type": "text", "text": "Here is the layer graph from the previous step:\n```json\n" + existing_layer_graph_json + "\n```"})

    payload_content = {
        "model": model_name,
        "messages": [{
            "role": "user",
            "content": messages
        }]
    }
    if temperature is not None:
        payload_content["temperature"] = temperature

    return json.dumps(payload_content)


def get_response(payload: str) -> Dict[str, Any]:
    try:
        response = requests.post(BASE_URL, headers=API_HEADERS, data=payload)
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"]
    except requests.exceptions.RequestException as e:
        raise VLMAPIError(f"API request failed: {str(e)}")
    except (KeyError, IndexError) as e:
        raise VLMAPIError(f"Failed to parse API response: {str(e)}")


def detect_top_layer(image: Image.Image, model_name: str, use_layer_graph_reasoning: bool, prompt_override: Optional[str] = None, existing_layer_graph_json: Optional[str] = None, max_retries: int = 3, retry_delay: float = 1.0, logger: logging.Logger = None) -> Dict[str, Any]:
    image_base64 = image_to_base64_from_pil(image)
    
    if prompt_override:
        current_prompt = prompt_override
    elif use_layer_graph_reasoning:
        current_prompt = LAYER_GRAPH_PROMPT
    else:
        current_prompt = DEFAULT_PROMPT
        
    payload = create_payload(image_base64, model_name, current_prompt, existing_layer_graph_json=existing_layer_graph_json)
    
    for attempt in range(max_retries):
        try:
            return get_response(payload)
        except VLMAPIError as e:
            if attempt == max_retries - 1:
                raise VLMAPIError(f"Failed after {max_retries} attempts. Last error: {str(e)}")
            if logger:
                logger.warning(f"Attempt {attempt + 1} failed: {str(e)}. Retrying in {retry_delay} seconds...")
            else:
                print(f"Attempt {attempt + 1} failed: {str(e)}. Retrying in {retry_delay} seconds...")
            time.sleep(retry_delay)


def detect_mask(image: Image.Image, model_name: str, prompt: str, temperature: float = 0.5, max_retries: int = 3, retry_delay: float = 1.0, logger: logging.Logger = None) -> Dict[str, Any]:
    image_base64 = image_to_base64_from_pil(image)
    # lower temperature for mask & bbox detection
    payload = create_payload(image_base64, model_name, prompt, temperature)

    for attempt in range(max_retries):
        try:
            return get_response(payload)
        except VLMAPIError as e:
            if attempt == max_retries - 1:
                raise VLMAPIError(f"Failed after {max_retries} attempts. Last error: {str(e)}")
            if logger:
                logger.warning(f"Attempt {attempt + 1} failed: {str(e)}. Retrying in {retry_delay} seconds...")
            else:
                print(f"Attempt {attempt + 1} failed: {str(e)}. Retrying in {retry_delay} seconds...")
            time.sleep(retry_delay)


def extract_tag_content(tag: str, text: str) -> Optional[str]:
    match = re.search(rf'<{tag}>(.*?)</{tag}>', text, re.DOTALL | re.IGNORECASE)
    return match.group(1).strip() if match else None


def parse_json(json_output: str):
    # Parsing out the markdown fencing
    lines = json_output.splitlines()
    for i, line in enumerate(lines):
        if line == "```json":
            json_output = "\n".join(lines[i+1:])  # Remove everything before "```json"
            json_output = json_output.split("```")[0]  # Remove everything after the closing "```"
            break  # Exit the loop once "```json" is found
    return json_output
