import os

def generate_tree_with_one_file(root, prefix=''):
    lines = []
    try:
        entries = sorted(os.listdir(root))
    except PermissionError:
        return lines  # skip folders we can't access

    dirs = [e for e in entries if os.path.isdir(os.path.join(root, e))]
    files = [e for e in entries if os.path.isfile(os.path.join(root, e))]

    # Print subdirectories
    for i, d in enumerate(dirs):
        path = os.path.join(root, d)
        is_last = (i == len(dirs) - 1 and not files)
        connector = "└── " if is_last else "├── "
        lines.append(f"{prefix}{connector}{d}")
        extension = "    " if is_last else "│   "
        lines.extend(generate_tree_with_one_file(path, prefix + extension))

    # Only print one file (if any)
    if files:
        connector = "└── " if not dirs else "├── "
        lines.append(f"{prefix}{connector}{files[0]}  [example]")

    return lines

# ✅ Your specified dataset path
dataset_path = "/project/hnguyen2/mvu9/datasets/kidney_pathology_image/"
readme_path = 'README.md'

# Generate the tree
tree_lines = generate_tree_with_one_file(dataset_path)

# Write to README.md
with open(readme_path, 'a') as f:
    f.write("\n## Dataset Directory Structure (1 sample file per folder)\n")
    f.write("```\n")
    f.write(f"{os.path.basename(dataset_path.strip('/'))}\n")
    for line in tree_lines:
        f.write(f"{line}\n")
    f.write("```\n")

print(f"✅ Wrote tree with one file per folder to {readme_path}")
