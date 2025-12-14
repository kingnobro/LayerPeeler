import os
import logging
import json
import graphviz
from PIL import Image
from typing import Any, Callable
from collections import defaultdict


def setup_logging(output_folder):
    logger = logging.getLogger('LayerSVG')
    logger.setLevel(logging.INFO)
    
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    
    log_file = os.path.join(output_folder, 'inference.log')
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(formatter)
    file_handler.setFormatter(formatter)
    
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    
    return logger


def load_or_generate(
    file_path: str,
    generate_func: Callable[[], Any],
    save_func: Callable[[Any, str], None],
    load_func: Callable[[str], Any],
    logger: logging.Logger,
    description: str = "data"
) -> Any:
    """
    Loads data from file_path if it exists, otherwise generates it using generate_func,
    saves it using save_func, and returns the data.
    """
    if os.path.exists(file_path):
        logger.info(f"Loading existing {description} from {file_path}")
        try:
            return load_func(file_path)
        except Exception as e:
            logger.warning(f"Error loading {description} from {file_path}: {e}. Regenerating...")

    logger.info(f"Generating {description}")
    data = generate_func()
    logger.info(f"Saving {description} to {file_path}")
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    save_func(data, file_path)
    return data


def save_json(data: Any, file_path: str):
    with open(file_path, 'w') as f:
        json.dump(data, f, indent=4)


def load_json(file_path: str) -> Any:
    with open(file_path, 'r') as f:
        return json.load(f)


def save_image(image: Image.Image, file_path: str):
    image.save(file_path)


def load_image(file_path: str) -> Image.Image:
    with Image.open(file_path) as img:
        img.load()
        return img


NODE_UNIVERSAL_FILL_COLOR = "#E6F2FF"  # A very light, neutral blue for all nodes
CLUSTER_UNIVERSAL_FILL_COLOR = "#F5F5F5" # A very light grey for all cluster backgrounds
CLUSTER_UNIVERSAL_BORDER_COLOR = "#CCCCCC" # A slightly darker grey for cluster borders

# (Helper functions hex_to_rgb and get_contrasting_font_color remain the same as your last version)
def hex_to_rgb(hex_color_str):
    """Converts a hex color string (e.g., #RRGGBB or #RGB) to an (R, G, B) tuple."""
    hex_color = hex_color_str.lstrip('#')
    if len(hex_color) == 3:  # Expand short form e.g. #F0F -> #FF00FF
        hex_color = "".join([c*2 for c in hex_color])
    if len(hex_color) != 6:
        return None # Invalid hex length
    try:
        return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
    except ValueError:
        return None # Invalid hex character


def get_contrasting_font_color(hex_background_color_str):
    """
    Determines if black or white font has better contrast against a HEX background color.
    Args:
        hex_background_color_str (str): HEX color string (e.g., '#RRGGBB').
    Returns:
        str: 'black' or 'white'.
    """
    rgb = hex_to_rgb(hex_background_color_str)
    if rgb is None:
        return 'black' # Default if hex is invalid

    r, g, b = rgb
    luminance = (0.299 * r + 0.587 * g + 0.114 * b) / 255.0
    return 'black' if luminance > 0.55 else 'white'


def visualize_layer_graph(graph_data,
                          logger: logging.Logger,
                          node_fill_color=NODE_UNIVERSAL_FILL_COLOR, # Allow overriding default
                          cluster_fill_color=CLUSTER_UNIVERSAL_FILL_COLOR, # Allow overriding
                          cluster_border_color=CLUSTER_UNIVERSAL_BORDER_COLOR, # Allow overriding
                          output_filename='layer_graph',
                          output_format='svg',
                          engine='dot'):
    """
    Visualizes a layer graph using Graphviz with a consistent single color for all nodes
    and a single color for all cluster boxes, plus larger fonts.
    """
    if not isinstance(graph_data, dict) or 'nodes' not in graph_data or 'edges' not in graph_data:
        logger.error("Error: Invalid graph_data format. Must be a dict with 'nodes' and 'edges' keys.")
        return None

    full_output_path_base = output_filename

    # Main graph styling
    dot = graphviz.Digraph(comment='Layer Graph', engine=engine)
    dot.attr(rankdir='TB', splines='polyline', overlap='compress', compound='true')
    dot.attr(bgcolor="#FFFFFF", Damping="0.8") # Clean white background
    dot.attr(fontname="Helvetica", fontsize="13", fontcolor="#333333") # Increased base font size
    dot.attr(nodesep='0.7', ranksep='1.2')

    nodes_data = graph_data.get('nodes', [])
    semantic_groups = defaultdict(list)
    standalone_nodes_info = []
    node_ids_defined = set()

    for node_info in nodes_data:
        node_id = node_info.get('id')
        if not node_id:
            logger.warning(f"Warning: Node with missing ID found: {node_info}")
            continue
        node_ids_defined.add(node_id)
        
        if 'part_of_object' in node_info and node_info['part_of_object']:
            semantic_groups[node_info['part_of_object']].append(node_info)
        else:
            standalone_nodes_info.append(node_info)

    # Determine font color for all nodes based on the universal node fill color
    universal_node_font_color = get_contrasting_font_color(node_fill_color)

    # Cluster styling
    cluster_idx = 0
    universal_cluster_label_font_color = get_contrasting_font_color(cluster_fill_color)

    for group_name, group_nodes_list in semantic_groups.items():
        sane_group_name = "".join(c if c.isalnum() else "_" for c in group_name)
        with dot.subgraph(name=f'cluster_{cluster_idx}_{sane_group_name}') as c:
            c.attr(label=f'Object: {group_name}', style='filled,rounded',
                   fillcolor=cluster_fill_color, color=cluster_border_color, penwidth="1.5")
            c.attr(fontname="Helvetica-Bold", fontsize='15', fontcolor=universal_cluster_label_font_color) # Increased
            for node_info in group_nodes_list:
                node_id = node_info.get('id')
                description = node_info.get('description', '')
                label = f"{node_id}\n{description}"
                # ALL nodes get the SAME fill color and derived font color
                c.node(node_id, label=label, style='filled,rounded', shape='box',
                       fillcolor=node_fill_color, fontcolor=universal_node_font_color,
                       fontname="Helvetica", fontsize='12', color="#777777", penwidth="1") # Increased node font
        cluster_idx += 1

    # Standalone node styling
    for node_info in standalone_nodes_info:
        node_id = node_info.get('id')
        description = node_info.get('description', '')
        label = f"{node_id}\n{description}"
        # ALL nodes get the SAME fill color and derived font color
        dot.node(node_id, label=label, style='filled,rounded', shape='box',
                 fillcolor=node_fill_color, fontcolor=universal_node_font_color,
                 fontname="Helvetica", fontsize='12', color="#777777", penwidth="1") # Increased node font

    # Edge styling (maintains distinct colors for clarity of relationships)
    edges_data = graph_data.get('edges', [])
    for edge_info in edges_data:
        source = edge_info.get('source')
        target = edge_info.get('target')
        relationship = edge_info.get('relationship', '')

        if not source or not target:
            continue
        
        edge_color = "#555555" # Default edge color
        style = "solid"
        penwidth = "1.5"
        arrowhead = "normal"
        arrowsize = "0.9"
        fontcolor = "#444444"
        fontsize = "10" # Increased edge font size

        if relationship == 'occludes':
            edge_color = "#e74c3c"  # A clear red for occlusion
            penwidth = "2.2"
            fontcolor = "#c0392b"
        elif relationship == 'interrupted_shape':
            edge_color = "#9b59b6"  # A clear purple for interrupted shape
            style = "dashed"
            fontcolor = "#8e44ad"
            continue
            
        dot.edge(source, target, label=relationship, color=edge_color, style=style,
                 penwidth=penwidth, arrowhead=arrowhead, arrowsize=arrowsize,
                 fontcolor=fontcolor, fontsize=fontsize)

    # Rendering and error handling
    try:
        rendered_file_path = dot.render(filename=full_output_path_base,
                                        format=output_format,
                                        cleanup=True,
                                        quiet=False)
        logger.info(f"Layer graph visualized and saved to '{os.path.abspath(rendered_file_path)}'")
        return os.path.abspath(rendered_file_path)
    except graphviz.backend.execute.ExecutableNotFound:
        logger.error("\n--- GRAPHVIZ RENDERING FAILED ---")
        logger.error("Error: Graphviz executable not found. Please ensure it's installed and in PATH.")
        return None
    except Exception as e:
        logger.error(f"\n--- GRAPHVIZ RENDERING FAILED ---")
        logger.error(f"An error occurred during rendering: {e}")
        return None


def get_all_processor_keys(model, parent_name=''):
    all_processor_keys = []
    
    for name, module in model.named_children():
        full_name = f'{parent_name}.{name}' if parent_name else name
        
        # Check if the module has 'processor' attribute
        if hasattr(module, 'processor'):
            all_processor_keys.append(f'{full_name}.processor')
            # print(type(module.processor))
        
        # Recursively check submodules
        all_processor_keys.extend(get_all_processor_keys(module, full_name))
    
    return all_processor_keys
