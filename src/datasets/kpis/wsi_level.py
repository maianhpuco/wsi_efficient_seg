import os

def generate_tree_with_one_file(root, prefix=''):
    lines = []
    entries = sorted(os.listdir(root))
    dirs = [e for e in entries if os.path.isdir(os.path.join(root, e))]
    files = [e for e in entries if os.path.isfile(os.path.join(root, e))]

    # Print directories first
    for i, d in enumerate(dirs):
        path = os.path.join(root, d)
        connector = "└── " if i == len(dirs) - 1 and not files else "├── "
        lines.append(f"{prefix}{connector}{d}")
        extension = "    " if i == len(dirs) - 1 and not files else "│   "
        lines.extend(generate_tree_with_one_file(path, prefix + extension))

    # Print just one file (if any)
    if files:
        connector = "└── " if not dirs else "├── "
        lines.append(f"{prefix}{connector}{files[0]}")

    return lines

 
if __name__ == "__main__":
    # Example usage
    print("Directory structure of the kidney pathology image dataset:")
    print("----------------------------------------------------------")
    # Define path to your dataset and README
    dataset_path = "/project/hnguyen2/mvu9/datasets/kidney_pathology_image/"
    readme_path = 'README.md'

    # Generate tree
    tree_lines = generate_tree_with_one_file(dataset_path)

    # Write to README.md
    with open(readme_path, 'a') as f:
        f.write("\n## Dataset Directory Structure (Sampled .jpg files)\n")
        f.write("```\n")
        f.write(f"{os.path.basename(dataset_path)}\n")
        for line in tree_lines:
            f.write(f"{line}\n")
        f.write("```\n")

    print(f"Done! Tree structure with sampled .jpg files appended to {readme_path}") 