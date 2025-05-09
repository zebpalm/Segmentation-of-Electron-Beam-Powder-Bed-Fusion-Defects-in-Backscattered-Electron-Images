import cv2
import numpy as np
from pathlib import Path
import os

# Define sample areas (y1, y2, x1, x2)
sample_areas = {
    'top_left': (130, 273, 128, 271),      # Sample 1
    'top_right': (130, 273, 329, 472),     # Sample 2
    'bottom_left': (329, 472, 128, 271),   # Sample 3
    'bottom_right': (329, 472, 329, 472)   # Sample 4
}

def load_image(image_path):
    """Load the image."""
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not load image: {image_path}")
    return img

def extract_sample(image, sample_name):
    """Extract a specific sample area from the image."""
    y1, y2, x1, x2 = sample_areas[sample_name]
    return image[y1:y2, x1:x2]

def process_images(input_dir, output_base_dir, max_images=10):
    """Process images and save samples in separate folders."""
    # Create output directories for each sample
    for sample_name in sample_areas.keys():
        sample_dir = Path(output_base_dir) / sample_name
        sample_dir.mkdir(parents=True, exist_ok=True)
    
    # Get list of image files and sort them
    image_files = sorted(list(Path(input_dir).glob('*.png')))
    
    # Process only the first max_images
    for image_file in image_files[:max_images]:
        try:
            print(f"Processing {image_file.name}...")
            
            # Load image
            img = load_image(str(image_file))
            
            # Process each sample area
            for sample_name in sample_areas.keys():
                print(f"  Extracting {sample_name}...")
                
                # Extract sample area
                sample_img = extract_sample(img, sample_name)
                
                # Save the sample
                output_file = Path(output_base_dir) / sample_name / image_file.name
                cv2.imwrite(str(output_file), sample_img)
            
        except Exception as e:
            print(f"Error processing {image_file.name}: {str(e)}")

if __name__ == "__main__":
    # Define input and output directories
    input_dir = "/Users/zebpalm/Exjobb 2025/BSE images/abctoanalyze/All"
    output_base_dir = "divided_samples"
    
    # Process the images
    process_images(input_dir, output_base_dir, max_images=180) 