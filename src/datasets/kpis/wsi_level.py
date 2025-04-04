import os

def print_tree(root, prefix=""):
    files = sorted(os.listdir(root))
    for i, filename in enumerate(files):
        path = os.path.join(root, filename)
        connector = "└── " if i == len(files) - 1 else "├── "
        print(prefix + connector + filename)
        if os.path.isdir(path):
            extension = "    " if i == len(files) - 1 else "│   "
            print_tree(path, prefix + extension)

if __name__ == "__main__":
    # Example usage
    print("Directory structure of the kidney pathology image dataset:")
    print("----------------------------------------------------------")
    # Replace with the actual path to your dataset
    # For example: "datasets/kidney_pathology_image" 
    data_path = "/project/hnguyen2/mvu9/datasets/kidney_pathology_image/"
    print_tree(data_path)
 