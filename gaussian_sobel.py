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

def apply_gaussian_sobel(image, kernel_size=5, sigma=1.0):
    """Apply Gaussian smoothing and Sobel operator to the image."""
    # Apply Gaussian smoothing
    smoothed = cv2.GaussianBlur(image, (kernel_size, kernel_size), sigma)
    
    # Apply Sobel operator
    sobel = filters.sobel(smoothed)
    
    return smoothed, sobel

def plot_results(original, smoothed, sobel, output_path):
    """Plot and save the original, smoothed, and Sobel results."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Original image
    axes[0].imshow(original, cmap='gray')
    axes[0].set_title('Original Sample')
    axes[0].axis('off')
    
    # Smoothed image
    axes[1].imshow(smoothed, cmap='gray')
    axes[1].set_title('Gaussian Smoothed')
    axes[1].axis('off')
    
    # Sobel result
    axes[2].imshow(sobel, cmap='gray')
    axes[2].set_title('Sobel Operator')
    axes[2].axis('off')
    
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

def process_images(kernel_size=5, sigma=1.0):
    """Process images to apply Gaussian smoothing and Sobel operator."""
    input_dir = Path("/Users/zebpalm/Exjobb 2025/BSE images/abctoanalyze/All")
    output_dir = Path("gaussian_sobel_results")
    
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
                
                # Apply Gaussian smoothing and Sobel operator
                smoothed, sobel = apply_gaussian_sobel(sample, kernel_size, sigma)
                
                # Save results
                sample_output_dir = output_dir / sample_name
                sample_output_dir.mkdir(exist_ok=True)
                
                # Save visualization
                output_path = sample_output_dir / f"{img_path.stem}_gaussian_sobel.png"
                plot_results(sample, smoothed, sobel, output_path)
                
                # Save the Sobel result as a separate image
                sobel_path = sample_output_dir / f"{img_path.stem}_sobel.png"
                cv2.imwrite(str(sobel_path), (sobel * 255).astype(np.uint8))
                
                print(f"Completed {sample_name} from {img_path.name}")
            
        except Exception as e:
            print(f"Error processing {img_path.name}: {str(e)}")

if __name__ == "__main__":
    print("Starting Gaussian smoothing and Sobel operator script...")
    # You can adjust these parameters:
    # kernel_size: size of the Gaussian kernel (must be odd)
    # sigma: standard deviation of the Gaussian kernel
    process_images(kernel_size=5, sigma=1.0)
    print("Script completed") 