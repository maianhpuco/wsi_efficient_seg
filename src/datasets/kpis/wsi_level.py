import os

def generate_tree(directory, prefix=''):
    """Recursively generates a tree structure of the given directory."""
    entries = sorted(os.listdir(directory))
    tree_structure = ''
    for index, entry in enumerate(entries):
        path = os.path.join(directory, entry)
        connector = '└── ' if index == len(entries) - 1 else '├── '
        tree_structure += f"{prefix}{connector}{entry}\n"
        if os.path.isdir(path):
            extension = '    ' if index == len(entries) - 1 else '│   '
            tree_structure += generate_tree(path, prefix=prefix + extension)
    return tree_structure

 
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
    tree_output = generate_tree(dataset_path)

    # Define the path to your README.md file
    readme_path = './README.md'

    # Append the tree structure to the README.md file
    with open(readme_path, 'a') as readme_file:
        readme_file.write('\n## Dataset Directory Structure\n')
        readme_file.write('```\n')  # Start of code block
        readme_file.write(tree_output)
        readme_file.write('```\n')  # End of code block

    print(f"Directory structure has been appended to {readme_path}")
    
 