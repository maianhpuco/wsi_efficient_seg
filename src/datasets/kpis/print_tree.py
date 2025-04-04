import os

def generate_tree_with_file_count(root, prefix=''):
    lines = []
    try:
        entries = sorted(os.listdir(root))
    except PermissionError:
        return lines

    dirs = [e for e in entries if os.path.isdir(os.path.join(root, e))]
    files = [e for e in entries if os.path.isfile(os.path.join(root, e))]

    # Filter for specific image extensions only
    image_extensions = {'.jpg', '.png', '.tiff', '.jpeg'}  # Add more extensions if needed
    image_files = [f for f in files if os.path.splitext(f)[1].lower() in image_extensions]

    total_items = len(dirs)

    for i, d in enumerate(dirs):
        is_last = (i == total_items - 1)
        connector = "└── " if is_last else "├── "
        lines.append(f"{prefix}{connector}{d}")
        extension = "    " if is_last else "│   "
        subdir = os.path.join(root, d)
        lines.extend(generate_tree_with_file_count(subdir, prefix + extension))

    # Only count image files
    if image_files:
        lines.append(f"{prefix}└── [Image Files: {len(image_files)}]")

    return lines

# Paths
dataset_path = "/project/hnguyen2/mvu9/datasets/kidney_pathology_image/"
readme_path = "README.md"

# Generate tree with file counts
tree_lines = generate_tree_with_file_count(dataset_path)

# Write to README.md
with open(readme_path, 'a') as f:
    f.write("\n## Dataset Directory Structure (Image file counts per folder)\n")
    f.write("```\n")
    f.write(f"{os.path.basename(dataset_path.strip('/'))}\n")
    for line in tree_lines:
        f.write(f"{line}\n")
    f.write("```\n")

print("✅ Folder tree + image file counts saved to README.md")