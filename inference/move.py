import os
import shutil

# 路径设置
png_dir = "data/testset"
svg_src_dir = "/home/wuronghuan/LayerSVG/Generation/data/svg"
svg_dst_dir = "data/testset_svg"

# 创建目标文件夹（如果不存在）
os.makedirs(svg_dst_dir, exist_ok=True)

# 遍历 png 文件
for filename in os.listdir(png_dir):
    if not filename.lower().endswith(".png"):
        continue

    file_id = os.path.splitext(filename)[0]
    svg_name = f"{file_id}.svg"

    src_svg_path = os.path.join(svg_src_dir, svg_name)
    dst_svg_path = os.path.join(svg_dst_dir, svg_name)

    if os.path.exists(src_svg_path):
        shutil.copy(src_svg_path, dst_svg_path)
    else:
        print(f"[WARNING] SVG not found: {src_svg_path}")

print("Done.")
