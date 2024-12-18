import sys
import os

sys.path.append('/home/tumwfml-ubunt6/SAM_models/segment-anything-2')

try:
    import torch
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor
    print("Imports successful!")
except ImportError as e:
    print(f"Import error: {e}")
except Exception as e:
    print(f"An error occurred: {e}")
