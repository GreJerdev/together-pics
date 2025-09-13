import kagglehub

# Download latest version
path = kagglehub.dataset_download("snmahsa/human-images-dataset-men-and-women")

print("Path to dataset files:", path)