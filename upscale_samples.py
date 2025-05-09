import cv2
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

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

def extract_and_upscale_sample(image, sample_name, scale_factor=2.0):
    """Extract and upscale a specific sample area from the image."""
    y1, y2, x1, x2 = sample_areas[sample_name]
    sample = image[y1:y2, x1:x2]
    
    # Upscale the sample
    upscaled_sample = cv2.resize(sample, None, fx=scale_factor, fy=scale_factor, 
                                interpolation=cv2.INTER_LINEAR)
    
    return sample, upscaled_sample

def plot_samples(original, upscaled, output_path):
    """Plot and save the original and upscaled samples."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    
    # Original sample
    axes[0].imshow(original, cmap='gray')
    axes[0].set_title('Original Sample')
    axes[0].axis('off')
    
    # Upscaled sample
    axes[1].imshow(upscaled, cmap='gray')
    axes[1].set_title(f'Upscaled Sample ({upscaled.shape[0]}x{upscaled.shape[1]})')
    axes[1].axis('off')
    
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

def process_images(scale_factor=2.0):
    """Process images to extract and upscale samples."""
    input_dir = Path("/Users/zebpalm/Exjobb 2025/BSE images/abctoanalyze/All")
    output_dir = Path("upscaled_samples")
    
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
                # Extract and upscale sample
                original_sample, upscaled_sample = extract_and_upscale_sample(img, sample_name, scale_factor)
                
                # Save results
                sample_output_dir = output_dir / sample_name
                sample_output_dir.mkdir(exist_ok=True)
                
                # Save visualization
                output_path = sample_output_dir / f"{img_path.stem}_upscaled.png"
                plot_samples(original_sample, upscaled_sample, output_path)
                
                # Save the upscaled sample as a separate image
                upscaled_path = sample_output_dir / f"{img_path.stem}_upscaled_raw.png"
                cv2.imwrite(str(upscaled_path), upscaled_sample)
                
                print(f"Completed {sample_name} from {img_path.name}")
            
        except Exception as e:
            print(f"Error processing {img_path.name}: {str(e)}")

if __name__ == "__main__":
    print("Starting sample upscaling script...")
    process_images(scale_factor=4.0)  # You can change the scale factor here
    print("Script completed") 