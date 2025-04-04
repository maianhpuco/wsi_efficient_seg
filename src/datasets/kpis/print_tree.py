import os

def generate_tree_with_one_file(root, prefix=''):
    lines = []
    try:
        entries = sorted(os.listdir(root))
    except PermissionError:
        return lines  # Skip folders we can't access

    dirs = [e for e in entries if os.path.isdir(os.path.join(root, e))]
    files = [e for e in entries if os.path.isfile(os.path.join(root, e))]

    total_items = len(dirs) + (1 if files else 0)

    for i, d in enumerate(dirs):
        is_last = (i == total_items - 1)
        connector = "└── " if is_last else "├── "
        lines.append(f"{prefix}{connector}{d}")
        extension = "    " if is_last else "│   "
        lines.extend(generate_tree_with_one_file(os.path.join(root, d), prefix + extension))

    # ✅ Only print one file per folder
    if files:
        connector = "└── "  # Always the last thing printed in a folder
        lines.append(f"{prefix}{connector}{files[0]}  [example]")

    return lines

# Paths
dataset_path = "/project/hnguyen2/mvu9/datasets/kidney_pathology_image/"
readme_path = "README.md"

# Run
tree_lines = generate_tree_with_one_file(dataset_path)

# Write to README.md
with open(readme_path, 'a') as f:
    f.write("\n## Dataset Directory Structure (1 sample file per folder)\n")
    f.write("```\n")
    f.write(f"{os.path.basename(dataset_path.strip('/'))}\n")
    for line in tree_lines:
        f.write(f"{line}\n")
    f.write("```\n")

print("✅ Only one file per folder printed to README.md")
