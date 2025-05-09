import os
import shutil
import cv2
import numpy as np

# Create evaluation_images directory if it doesn't exist
eval_dir = "evaluation_images"
if not os.path.exists(eval_dir):
    os.makedirs(eval_dir)

# Source directories
source_dirs = [
    "/Users/zebpalm/Exjobb 2025/BSE images/abctoanalyze/Top 180",
    "/Users/zebpalm/Exjobb 2025/BSE images/topo1/Top 180",
    "/Users/zebpalm/Exjobb 2025/BSE images/SUM/Top 180"
]

# Sample areas for division
sample_areas = {
    'top_left': (130, 273, 128, 271),      # Sample 1
    'top_right': (130, 273, 329, 472),     # Sample 2
    'bottom_left': (329, 472, 128, 271),   # Sample 3
    'bottom_right': (329, 472, 329, 472)   # Sample 4
}

# Images to copy (45th, 90th, 135th)
image_indices = [44, 89, 134]  # 0-based indexing

# Process each source directory
for source_dir in source_dirs:
    # Get the folder name (abctoanalyze, topo1, or SUM)
    folder_name = os.path.basename(os.path.dirname(source_dir))
    
    # Create subdirectory for this source
    source_eval_dir = os.path.join(eval_dir, folder_name)
    if not os.path.exists(source_eval_dir):
        os.makedirs(source_eval_dir)
    
    # Get all image files in the source directory
    image_files = sorted([f for f in os.listdir(source_dir) if f.endswith(('.png', '.jpg', '.jpeg'))])
    
    # Copy and process specified images
    for idx in image_indices:
        if idx < len(image_files):
            # Read the image
            img_path = os.path.join(source_dir, image_files[idx])
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            
            if img is not None:
                # Divide image into samples
                for sample_name, (y_start, y_end, x_start, x_end) in sample_areas.items():
                    # Extract sample
                    sample = img[y_start:y_end, x_start:x_end]
                    
                    # Create filename for the sample
                    sample_filename = f"{folder_name}_image{idx+1}_{sample_name}.png"
                    sample_path = os.path.join(source_eval_dir, sample_filename)
                    
                    # Save the sample
                    cv2.imwrite(sample_path, sample)
                    print(f"Saved {sample_filename}")

print("Processing complete!") 