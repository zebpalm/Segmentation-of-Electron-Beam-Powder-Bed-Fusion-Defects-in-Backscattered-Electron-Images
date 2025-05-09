import cv2
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from skimage import filters, segmentation, color
from scipy import ndimage as ndi

# Define sample areas (y1, y2, x1, x2)
sample_areas = {
    'Sample 1': (130, 273, 128, 271),
    'Sample 2': (130, 273, 329, 472),
    'Sample 3': (329, 472, 128, 271),
    'Sample 4': (329, 472, 329, 472)
}

def load_image(image_path):
    """Load and preprocess the image."""
    print(f"Attempting to load image: {image_path}")
    img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Could not load image: {image_path}")
    print(f"Successfully loaded image with shape: {img.shape}")
    return img

def extract_sample(image, sample_name):
    """Extract and upscale a specific sample area from the image."""
    y1, y2, x1, x2 = sample_areas[sample_name]
    sample = image[y1:y2, x1:x2]
    
    # Upscale the sample by a factor of 2
    upscaled_sample = cv2.resize(sample, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_LINEAR)
    
    return upscaled_sample

def apply_watershed_segmentation(img):
    """Apply watershed-based segmentation to the image."""
    print("Applying watershed segmentation...")
    # Invert the image to detect dark spots
    inverted_img = 255 - img
    
    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(inverted_img, (5, 5), 0)
    
    # Calculate elevation map using Sobel gradient
    elevation_map = filters.sobel(blurred)
    
    # Find markers based on extreme parts of the histogram
    markers = np.zeros_like(blurred)
    markers[blurred < 20] = 1  # Background markers (very bright in inverted)
    markers[blurred > 200] = 2  # Foreground markers (very dark in inverted)
    
    # Apply watershed transform
    segmentation_result = segmentation.watershed(elevation_map, markers)
    
    # Clean up the segmentation
    segmentation_result = ndi.binary_fill_holes(segmentation_result - 1)
    labeled_result, _ = ndi.label(segmentation_result)
    
    return labeled_result, segmentation_result, elevation_map, markers, blurred

def plot_segmentation_results(img, labeled_result, segmentation_result, elevation_map, markers, blurred_img, output_path):
    """Plot and save the segmentation results."""
    print(f"Saving results to: {output_path}")
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Original image with contour
    axes[0, 0].imshow(img, cmap='gray')
    axes[0, 0].contour(segmentation_result, [0.5], linewidths=1.2, colors='y')
    axes[0, 0].set_title('Original with Contour')
    axes[0, 0].axis('off')
    
    # Blurred inverted image
    axes[0, 1].imshow(blurred_img, cmap='gray')
    axes[0, 1].set_title('Blurred Inverted')
    axes[0, 1].axis('off')
    
    # Sobel gradient
    axes[1, 0].imshow(elevation_map, cmap='gray')
    axes[1, 0].set_title('Sobel Gradient')
    axes[1, 0].axis('off')
    
    # Label overlay
    image_label_overlay = color.label2rgb(labeled_result, image=img, bg_label=0)
    axes[1, 1].imshow(image_label_overlay)
    axes[1, 1].set_title('Segmentation Result')
    axes[1, 1].axis('off')
    
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print("Results saved successfully")

def process_images():
    """Process images using watershed segmentation."""
    input_dir = Path("/Users/zebpalm/Exjobb 2025/BSE images/abctoanalyze/All")
    output_dir = Path("watershed_results")
    
    print(f"Input directory: {input_dir}")
    print(f"Output directory: {output_dir}")
    
    # Check if input directory exists
    if not input_dir.exists():
        print(f"Error: Input directory does not exist: {input_dir}")
        return
    
    # Create output directory
    output_dir.mkdir(exist_ok=True)
    
    # Get list of image files
    image_files = sorted(input_dir.glob("*.png"))
    print(f"Found {len(image_files)} .png files")
    
    # Process first 10 images
    image_files = image_files[:10]
    print(f"Processing first {len(image_files)} images")
    
    for img_path in image_files:
        try:
            print(f"\nProcessing image: {img_path.name}")
            # Load image
            img = load_image(img_path)
            
            # Process each sample area
            for sample_name in sample_areas:
                print(f"Processing {sample_name}")
                # Extract sample area
                sample_img = extract_sample(img, sample_name)
                
                # Apply watershed segmentation
                labeled_result, segmentation_result, elevation_map, markers, blurred_img = apply_watershed_segmentation(sample_img)
                
                # Save results
                sample_output_dir = output_dir / sample_name
                sample_output_dir.mkdir(exist_ok=True)
                output_path = sample_output_dir / f"{img_path.stem}_watershed.png"
                plot_segmentation_results(sample_img, labeled_result, segmentation_result, elevation_map, markers, blurred_img, output_path)
                
                print(f"Completed {sample_name} from {img_path.name}")
            
        except Exception as e:
            print(f"Error processing {img_path.name}: {str(e)}")

if __name__ == "__main__":
    print("Starting watershed segmentation script...")
    process_images()
    print("Script completed")