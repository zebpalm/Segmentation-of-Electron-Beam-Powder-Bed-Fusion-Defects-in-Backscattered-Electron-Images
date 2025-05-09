import cv2
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from skimage import exposure

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

def apply_histogram_enhancement(image):
    """Apply three histogram enhancement methods."""
    # Contrast stretching
    p2, p98 = np.percentile(image, (2, 98))
    contrast_stretched = exposure.rescale_intensity(image, in_range=(p2, p98))
    
    # Histogram equalization
    equalized = exposure.equalize_hist(image)
    
    # Adaptive histogram equalization (CLAHE)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    clahe_enhanced = clahe.apply(image)
    
    return contrast_stretched, equalized, clahe_enhanced

def apply_thresholding(image, equalized):
    """Apply Otsu's thresholding to both original and equalized images."""
    # Convert equalized image to uint8 for OpenCV thresholding
    equalized_uint8 = (equalized * 255).astype(np.uint8)
    
    # Apply Otsu's thresholding to original image
    _, thresh_original = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Apply Otsu's thresholding to equalized image
    _, thresh_equalized = cv2.threshold(equalized_uint8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    return thresh_original, thresh_equalized

def plot_histogram(image, ax, title):
    """Plot histogram of an image."""
    hist, bins = np.histogram(image.flatten(), 256, [0,256])
    ax.plot(hist, color='black')
    ax.set_title(title)
    ax.set_xlim([0, 256])
    ax.set_ylim([0, hist.max() * 1.1])

def plot_results(original, contrast_stretched, equalized, clahe_enhanced, 
                thresh_original, thresh_equalized, output_path):
    """Plot and save all results with histograms and thresholding results."""
    fig, axes = plt.subplots(5, 2, figsize=(12, 20))
    
    # Original image and histogram
    axes[0, 0].imshow(original, cmap='gray')
    axes[0, 0].set_title('Original Image')
    axes[0, 0].axis('off')
    plot_histogram(original, axes[0, 1], 'Original Histogram')
    
    # Contrast stretched image and histogram
    axes[1, 0].imshow(contrast_stretched, cmap='gray')
    axes[1, 0].set_title('Contrast Stretched')
    axes[1, 0].axis('off')
    plot_histogram(contrast_stretched, axes[1, 1], 'Contrast Stretched Histogram')
    
    # Equalized image and histogram
    axes[2, 0].imshow(equalized, cmap='gray')
    axes[2, 0].set_title('Histogram Equalized')
    axes[2, 0].axis('off')
    plot_histogram(equalized, axes[2, 1], 'Equalized Histogram')
    
    # CLAHE enhanced image and histogram
    axes[3, 0].imshow(clahe_enhanced, cmap='gray')
    axes[3, 0].set_title('CLAHE Enhanced')
    axes[3, 0].axis('off')
    plot_histogram(clahe_enhanced, axes[3, 1], 'CLAHE Histogram')
    
    # Thresholding results
    axes[4, 0].imshow(thresh_original, cmap='gray')
    axes[4, 0].set_title('Otsu Thresholding (Original)')
    axes[4, 0].axis('off')
    axes[4, 1].imshow(thresh_equalized, cmap='gray')
    axes[4, 1].set_title('Otsu Thresholding (Equalized)')
    axes[4, 1].axis('off')
    
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

def process_images():
    """Process images to apply histogram enhancement methods and thresholding."""
    input_dir = Path("/Users/zebpalm/Exjobb 2025/BSE images/abctoanalyze/All")
    output_dir = Path("histogram_enhancement_results")
    
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
                
                # Apply histogram enhancement methods
                contrast_stretched, equalized, clahe_enhanced = apply_histogram_enhancement(sample)
                
                # Apply thresholding
                thresh_original, thresh_equalized = apply_thresholding(sample, equalized)
                
                # Save results
                sample_output_dir = output_dir / sample_name
                sample_output_dir.mkdir(exist_ok=True)
                
                # Save visualization
                output_path = sample_output_dir / f"{img_path.stem}_histogram_enhancement.png"
                plot_results(sample, contrast_stretched, equalized, clahe_enhanced,
                           thresh_original, thresh_equalized, output_path)
                
                # Save the enhanced images as separate files
                cv2.imwrite(str(sample_output_dir / f"{img_path.stem}_contrast_stretched.png"), 
                          (contrast_stretched * 255).astype(np.uint8))
                cv2.imwrite(str(sample_output_dir / f"{img_path.stem}_equalized.png"), 
                          (equalized * 255).astype(np.uint8))
                cv2.imwrite(str(sample_output_dir / f"{img_path.stem}_clahe.png"), 
                          clahe_enhanced)
                cv2.imwrite(str(sample_output_dir / f"{img_path.stem}_thresh_original.png"), 
                          thresh_original)
                cv2.imwrite(str(sample_output_dir / f"{img_path.stem}_thresh_equalized.png"), 
                          thresh_equalized)
                
                print(f"Completed {sample_name} from {img_path.name}")
            
        except Exception as e:
            print(f"Error processing {img_path.name}: {str(e)}")

if __name__ == "__main__":
    print("Starting histogram enhancement script...")
    process_images()
    print("Script completed") 