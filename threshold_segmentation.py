import cv2
import numpy as np
from pathlib import Path
from skimage import img_as_float, img_as_ubyte
from skimage.filters import threshold_local, gaussian
from skimage.morphology import remove_small_objects
import matplotlib.pyplot as plt

# Define sample areas (y1, y2, x1, x2)
sample_areas = {
    'top_left': (130, 273, 128, 271),
    'top_right': (130, 273, 329, 472),
    'bottom_left': (329, 472, 128, 271),
    'bottom_right': (329, 472, 329, 472)
}

def load_image(image_path):
    """Load the image."""
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Could not load image: {image_path}")
    return img

def extract_sample(image, sample_name):
    """Extract a specific sample area from the image."""
    y1, y2, x1, x2 = sample_areas[sample_name]
    return image[y1:y2, x1:x2]

def segment_pores(image):
    """Segment pores from BSE image using threshold-based approach."""
    # Convert to float and normalize
    img = img_as_float(image)
    
    # Apply Gaussian blur to reduce noise
    smoothed = gaussian(img, sigma=1)
    
    # Use local thresholding since BSE images might have intensity variations
    threshold = threshold_local(smoothed, block_size=35, offset=0.02)
    binary = smoothed < threshold
    
    # Remove small objects (noise)
    cleaned = remove_small_objects(binary, min_size=5)
    
    return cleaned

def visualize_results(original, segmented, output_path):
    """Visualize and save original and segmented images side by side."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    
    # Plot original image
    ax1.imshow(original, cmap='gray')
    ax1.set_title('Original Image')
    ax1.axis('off')
    
    # Plot segmented image
    ax2.imshow(segmented, cmap='gray')
    ax2.set_title('Segmented Pores')
    ax2.axis('off')
    
    # Save the figure
    plt.savefig(output_path, bbox_inches='tight', dpi=300)
    plt.close()

def process_images(input_dir, output_base_dir, max_images=10):
    """Process images and save segmentation results."""
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
                print(f"  Processing {sample_name}...")
                
                # Extract sample area
                sample_img = extract_sample(img, sample_name)
                
                # Segment pores
                segmented = segment_pores(sample_img)
                
                # Save visualization
                output_file = Path(output_base_dir) / sample_name / f"segmented_{image_file.name}"
                visualize_results(sample_img, segmented, str(output_file))
                
                # Save binary mask
                mask_file = Path(output_base_dir) / sample_name / f"mask_{image_file.name}"
                cv2.imwrite(str(mask_file), img_as_ubyte(segmented))
            
        except Exception as e:
            print(f"Error processing {image_file.name}: {str(e)}")

if __name__ == "__main__":
    # Define input and output directories
    input_dir = "/Users/zebpalm/Exjobb 2025/BSE images/abctoanalyze/All"
    output_base_dir = "threshold_segmented"
    
    # Process the images
    process_images(input_dir, output_base_dir, max_images=10) 