import cv2
import numpy as np
from pathlib import Path
from skimage import img_as_float
from skimage.feature import blob_log
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

def detect_pores(image):
    """Detect pores using LoG blob detection."""
    # Convert to float and normalize
    img = img_as_float(image)
    
    # Invert image since we're looking for dark blobs
    img = 1 - img
    
    # Apply LoG filter at multiple scales
    blobs = blob_log(
        img,
        max_sigma=10,
        min_sigma=1,
        num_sigma=10,
        threshold=.2
    )
    
    return blobs

def visualize_results(original, blobs, output_path):
    """Visualize and save original image with detected pores."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    
    # Plot original image
    ax1.imshow(original, cmap='gray')
    ax1.set_title('Original Image')
    ax1.axis('off')
    
    # Plot original image with detected pores
    ax2.imshow(original, cmap='gray')
    ax2.set_title('Detected Pores')
    
    # Plot circles for each detected pore
    for blob in blobs:
        y, x, r = blob
        c = plt.Circle((x, y), r * np.sqrt(2), color='red', linewidth=1, fill=False)
        ax2.add_patch(c)
    
    ax2.axis('off')
    
    # Save the figure
    plt.savefig(output_path, bbox_inches='tight', dpi=300)
    plt.close()

def create_binary_mask(image_shape, blobs):
    """Create a binary mask from detected blobs."""
    mask = np.zeros(image_shape, dtype=np.uint8)
    
    for blob in blobs:
        y, x, r = blob
        rr, cc = np.ogrid[
            max(0, int(y-r)):min(image_shape[0], int(y+r+1)),
            max(0, int(x-r)):min(image_shape[1], int(x+r+1))
        ]
        dist = np.sqrt((rr - y)**2 + (cc - x)**2)
        circle_mask = dist <= r
        mask[
            max(0, int(y-r)):min(image_shape[0], int(y+r+1)),
            max(0, int(x-r)):min(image_shape[1], int(x+r+1))
        ][circle_mask] = 255
    
    return mask

def process_images(input_dir, output_base_dir, max_images=10):
    """Process images and save detection results."""
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
                
                # Detect pores
                blobs = detect_pores(sample_img)
                
                # Save visualization
                output_file = Path(output_base_dir) / sample_name / f"detected_{image_file.name}"
                visualize_results(sample_img, blobs, str(output_file))
                
                # Create and save binary mask
                mask = create_binary_mask(sample_img.shape, blobs)
                mask_file = Path(output_base_dir) / sample_name / f"mask_{image_file.name}"
                cv2.imwrite(str(mask_file), mask)
            
        except Exception as e:
            print(f"Error processing {image_file.name}: {str(e)}")

if __name__ == "__main__":
    # Define input and output directories
    input_dir = "/Users/zebpalm/Exjobb 2025/BSE images/abctoanalyze/All"
    output_base_dir = "log_detected"
    
    # Process the images
    process_images(input_dir, output_base_dir, max_images=10) 