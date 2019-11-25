"""
Configuration module for deep fashion dataset
"""
root_dir = "./deepfashion_data"


class DeepFashionConfig:
    root_dir = root_dir
    mean = [0.7464, 0.7155, 0.7043]
    std = [0.2606, 0.2716, 0.2744]

    # Image net mean and std
    # mean = [0.485, 0.456, 0.406]
    # std = [0.229, 0.224, 0.225]
    sizes = (224, 224)
