import cv2
import numpy as np
from pathlib import Path
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

def apply_clahe(image, clip_limit=2.0, tile_size=(8, 8)):
    """Apply CLAHE to enhance image contrast."""
    # Create CLAHE object
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_size)
    
    # Apply CLAHE
    enhanced = clahe.apply(image)
    
    return enhanced

def visualize_results(original, enhanced, output_path):
    """Visualize and save original and enhanced images side by side."""
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    
    # Plot original image
    ax1.imshow(original, cmap='gray')
    ax1.set_title('Original Image')
    ax1.axis('off')
    
    # Plot enhanced image
    ax2.imshow(enhanced, cmap='gray')
    ax2.set_title('CLAHE Enhanced')
    ax2.axis('off')
    
    # Add histograms as subplots below the images
    hist_height = 2
    fig.set_figheight(fig.get_figheight() + hist_height)
    
    ax3 = plt.subplot(223)
    ax3.hist(original.ravel(), 256, [0, 256], color='gray', alpha=0.7)
    ax3.set_title('Original Histogram')
    ax3.set_xlim([0, 256])
    
    ax4 = plt.subplot(224)
    ax4.hist(enhanced.ravel(), 256, [0, 256], color='gray', alpha=0.7)
    ax4.set_title('Enhanced Histogram')
    ax4.set_xlim([0, 256])
    
    # Adjust layout and save
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight', dpi=300)
    plt.close()

def process_images(input_dir, output_base_dir, max_images=10, clip_limit=2.0, tile_size=(8, 8)):
    """Process images and save CLAHE enhanced results."""
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
                
                # Apply CLAHE
                enhanced = apply_clahe(sample_img, clip_limit, tile_size)
                
                # Save visualization with histograms
                output_file = Path(output_base_dir) / sample_name / f"enhanced_{image_file.name}"
                visualize_results(sample_img, enhanced, str(output_file))
                
                # Save enhanced image
                enhanced_file = Path(output_base_dir) / sample_name / f"clahe_{image_file.name}"
                cv2.imwrite(str(enhanced_file), enhanced)
            
        except Exception as e:
            print(f"Error processing {image_file.name}: {str(e)}")

if __name__ == "__main__":
    # Define input and output directories
    input_dir = "/Users/zebpalm/Exjobb 2025/BSE images/abctoanalyze/All"
    output_base_dir = "clahe_enhanced"
    
    # CLAHE parameters
    clip_limit = 2.0  # Contrast limit for histogram equalization
    tile_size = (8, 8)  # Size of grid for histogram equalization
    
    # Process the images
    process_images(input_dir, output_base_dir, max_images=10, 
                  clip_limit=clip_limit, tile_size=tile_size) 