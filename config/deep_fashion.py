"""
Configuration module for deep fashion dataset
"""
root_dir = "./deepfashion_data"

class DeepFashionConfig:
    root_dir = root_dir
    mean = [0.7464, 0.7155, 0.7043]
    std = [0.2606, 0.2716, 0.2744]
    # sizes = (224, 224)
    sizes = (28, 28)
