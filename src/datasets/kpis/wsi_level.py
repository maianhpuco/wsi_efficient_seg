import os

def generate_tree_with_one_jpg(root, prefix=''):
    lines = []
    entries = sorted(os.listdir(root))
    dirs = [e for e in entries if os.path.isdir(os.path.join(root, e))]
    jpgs = [e for e in entries if e.lower().endswith('.jpg')]

    # print subdirectories first
    for i, d in enumerate(dirs):
        path = os.path.join(root, d)
        connector = "└── " if i == len(dirs) - 1 and not jpgs else "├── "
        lines.append(f"{prefix}{connector}{d}")
        extension = "    " if i == len(dirs) - 1 and not jpgs else "│   "
        lines.extend(generate_tree_with_one_jpg(path, prefix + extension))

    # only print one jpg file if any
    if jpgs:
        connector = "└── " if not dirs else "├── "
        lines.append(f"{prefix}{connector}{jpgs[0]}")

    return lines
 
if __name__ == "__main__":
    # Example usage
    print("Directory structure of the kidney pathology image dataset:")
    print("----------------------------------------------------------")
    # Replace with the actual path to your dataset
    # For example: "datasets/kidney_pathology_image" 
    dataset_path = "/project/hnguyen2/mvu9/datasets/kidney_pathology_image/"
    # print_tree(data_path)
        
    # Define the path to your dataset
    # dataset_path = 'datasets/kidney_pathology_image'

    # Generate the tree structure
    tree_output = generate_tree_with_one_jpg(dataset_path)

    # Define the path to your README.md file
    readme_path = './README.md'

    # Append the tree structure to the README.md file
    with open(readme_path, 'a') as readme_file:
        readme_file.write('\n## Dataset Directory Structure\n')
        readme_file.write('```\n')  # Start of code block
        readme_file.write(tree_output)
        readme_file.write('```\n')  # End of code block

    print(f"Directory structure has been appended to {readme_path}")
    
 