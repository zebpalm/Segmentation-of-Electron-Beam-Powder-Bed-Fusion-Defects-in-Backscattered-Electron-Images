import json
import cv2
import numpy as np
from pathlib import Path
import os

def create_binary_mask(image_path, polygons, output_path):
    """
    Create a binary mask from polygon annotations.
    
    Args:
        image_path: Path to the original image
        polygons: List of polygon points
        output_path: Path to save the binary mask
    """
    # Read the original image to get dimensions
    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f"Could not read image: {image_path}")
    
    height, width = image.shape[:2]
    
    # Create empty mask
    mask = np.zeros((height, width), dtype=np.uint8)
    
    # Draw polygons on mask
    for polygon in polygons:
        # Convert points to numpy array
        points = np.array(polygon, dtype=np.int32)
        # Reshape for cv2.fillPoly
        points = points.reshape((-1, 1, 2))
        # Fill polygon with white (255)
        cv2.fillPoly(mask, [points], 255)
    
    # Save mask
    cv2.imwrite(str(output_path), mask)
    print(f"Saved mask to: {output_path}")

def process_vgg_json(json_path, image_dir, output_dir):
    """
    Process VGG JSON file and create binary masks for all images.
    
    Args:
        json_path: Path to VGG JSON file
        image_dir: Directory containing the original images
        output_dir: Directory to save binary masks
    """
    # Create base output directory if it doesn't exist
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Read JSON file
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    # Process each image
    for filename, image_data in data.items():
        # Get image path
        image_path = Path(image_dir) / filename
        
        # Dictionary to store polygons by label
        polygons_by_label = {}
        
        # Get polygons for this image, grouped by label
        for region_id, region in image_data['regions'].items():
            if region['shape_attributes']['name'] == 'polygon':
                # Get label
                label = region['region_attributes']['label']
                
                # Get x and y coordinates
                x_points = region['shape_attributes']['all_points_x']
                y_points = region['shape_attributes']['all_points_y']
                # Combine into polygon points
                polygon = list(zip(x_points, y_points))
                
                # Add to dictionary
                if label not in polygons_by_label:
                    polygons_by_label[label] = []
                polygons_by_label[label].append(polygon)
        
        # Create masks for each label
        for label, polygons in polygons_by_label.items():
            # Create label directory
            label_dir = output_dir / label
            label_dir.mkdir(parents=True, exist_ok=True)
            
            # Create output path for mask
            mask_filename = f"{Path(filename).stem}_mask.png"
            mask_path = label_dir / mask_filename
            
            # Create and save mask
            create_binary_mask(image_path, polygons, mask_path)

def main():
    # Paths
    json_path = "makesense_annotation_test_VGG_export.json"
    image_dir = "evaluation_images/topo1"  # Directory containing the original images
    output_dir = "binary_masks"  # Directory to save the masks
    
    # Process JSON file
    process_vgg_json(json_path, image_dir, output_dir)

if __name__ == "__main__":
    main() 