import cv2
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from skimage import filters

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
    """Extract a specific sample area from the image."""
    y1, y2, x1, x2 = sample_areas[sample_name]
    return image[y1:y2, x1:x2]

def apply_gradient_thresholding(image, kernel_size=5, sigma=1.0, threshold=0.1):
    """Apply gradient thresholding with and without Gaussian smoothing."""
    # Calculate gradient without smoothing
    gradient_raw = filters.sobel(image)
    
    # Apply Gaussian smoothing
    smoothed = cv2.GaussianBlur(image, (kernel_size, kernel_size), sigma)
    
    # Calculate gradient after smoothing
    gradient_smoothed = filters.sobel(smoothed)
    
    # Apply thresholding
    thresholded_raw = (gradient_raw > threshold).astype(np.uint8) * 255
    thresholded_smoothed = (gradient_smoothed > threshold).astype(np.uint8) * 255
    
    return gradient_raw, gradient_smoothed, thresholded_raw, thresholded_smoothed

def plot_results(original, gradient_raw, gradient_smoothed, 
                thresholded_raw, thresholded_smoothed, output_path):
    """Plot and save all results."""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Original image
    axes[0, 0].imshow(original, cmap='gray')
    axes[0, 0].set_title('Original Sample')
    axes[0, 0].axis('off')
    
    # Raw gradient
    axes[0, 1].imshow(gradient_raw, cmap='gray')
    axes[0, 1].set_title('Raw Gradient')
    axes[0, 1].axis('off')
    
    # Smoothed gradient
    axes[0, 2].imshow(gradient_smoothed, cmap='gray')
    axes[0, 2].set_title('Smoothed Gradient')
    axes[0, 2].axis('off')
    
    # Thresholded raw gradient
    axes[1, 0].imshow(thresholded_raw, cmap='gray')
    axes[1, 0].set_title('Thresholded Raw')
    axes[1, 0].axis('off')
    
    # Thresholded smoothed gradient
    axes[1, 1].imshow(thresholded_smoothed, cmap='gray')
    axes[1, 1].set_title('Thresholded Smoothed')
    axes[1, 1].axis('off')
    
    # Leave last subplot empty for better layout
    axes[1, 2].axis('off')
    
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

def process_images(kernel_size=5, sigma=1.0, threshold=0.1):
    """Process images to apply gradient thresholding."""
    input_dir = Path("/Users/zebpalm/Exjobb 2025/BSE images/abctoanalyze/All")
    output_dir = Path("gradient_thresholding_results")
    
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
                # Extract sample
                sample = extract_sample(img, sample_name)
                
                # Apply gradient thresholding
                gradient_raw, gradient_smoothed, thresholded_raw, thresholded_smoothed = \
                    apply_gradient_thresholding(sample, kernel_size, sigma, threshold)
                
                # Save results
                sample_output_dir = output_dir / sample_name
                sample_output_dir.mkdir(exist_ok=True)
                
                # Save visualization
                output_path = sample_output_dir / f"{img_path.stem}_gradient_thresholding.png"
                plot_results(sample, gradient_raw, gradient_smoothed, 
                           thresholded_raw, thresholded_smoothed, output_path)
                
                # Save the thresholded results as separate images
                cv2.imwrite(str(sample_output_dir / f"{img_path.stem}_thresholded_raw.png"), 
                          thresholded_raw)
                cv2.imwrite(str(sample_output_dir / f"{img_path.stem}_thresholded_smoothed.png"), 
                          thresholded_smoothed)
                
                print(f"Completed {sample_name} from {img_path.name}")
            
        except Exception as e:
            print(f"Error processing {img_path.name}: {str(e)}")

if __name__ == "__main__":
    print("Starting gradient thresholding script...")
    # You can adjust these parameters:
    # kernel_size: size of the Gaussian kernel (must be odd)
    # sigma: standard deviation of the Gaussian kernel
    # threshold: gradient threshold value
    process_images(kernel_size=3, sigma=0.5, threshold=0.2)
    print("Script completed") 