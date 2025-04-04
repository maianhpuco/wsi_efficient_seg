import os

def print_tree(root, prefix=""):
    files = sorted(os.listdir(root))
    dirs = [f for f in files if os.path.isdir(os.path.join(root, f))]
    files = [f for f in files if os.path.isfile(os.path.join(root, f))]

    for i, dirname in enumerate(dirs):
        connector = "└── " if i == len(dirs) + (1 if files else 0) - 1 else "├── "
        print(prefix + connector + dirname)
        extension = "    " if i == len(dirs) + (1 if files else 0) - 1 else "│   "
        print_tree(os.path.join(root, dirname), prefix + extension)

    if files:
        # Only show one representative file
        connector = "└── " if not dirs else "├── "
        print(prefix + connector + files[0] + "  [example]")
        
 
if __name__ == "__main__":
    # Example usage
    print("Directory structure of the kidney pathology image dataset:")
    print("----------------------------------------------------------")
    # Replace with the actual path to your dataset
    # For example: "datasets/kidney_pathology_image" 
    data_path = "/project/hnguyen2/mvu9/datasets/kidney_pathology_image/"
    print_tree(data_path)
 