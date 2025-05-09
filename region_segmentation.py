import cv2
import numpy as np
from skimage.segmentation import felzenszwalb, slic, quickshift
from skimage.util import img_as_float
import os
from pathlib import Path
import matplotlib.pyplot as plt
from skimage import filters, segmentation, color, morphology
from scipy import ndimage as ndi

# Define sample areas (y1, y2, x1, x2)
sample_areas = {
    'Sample 1': (130, 273, 128, 271),
    'Sample 2': (130, 273, 329, 472),
    'Sample 3': (329, 472, 128, 271),
    'Sample 4': (329, 472, 329, 472)
}

def load_and_preprocess_image(image_path):
    """Load and preprocess the image for segmentation."""
    # Read the image
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not load image: {image_path}")
    
    # Convert to grayscale if it's not already
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Normalize the image
    img = img.astype(float) / 255.0
    return img

def extract_sample(image, sample_name):
    """Extract a specific sample area from the image."""
    y1, y2, x1, x2 = sample_areas[sample_name]
    return image[y1:y2, x1:x2]

def segment_image(image, method='felzenszwalb', **kwargs):
    """Apply region-based segmentation to the image."""
    if method == 'felzenszwalb':
        segments = felzenszwalb(image, scale=100, sigma=0.5, min_size=50)
    elif method == 'slic':
        # Convert grayscale to RGB by repeating the channel
        rgb_image = np.stack((image,) * 3, axis=-1)
        segments = slic(rgb_image, n_segments=100, compactness=10, convert2lab=False)
    elif method == 'quickshift':
        # Convert grayscale to RGB by repeating the channel
        rgb_image = np.stack((image,) * 3, axis=-1)
        segments = quickshift(rgb_image, kernel_size=3, max_dist=6, ratio=0.5, convert2lab=False)
    else:
        raise ValueError(f"Unknown segmentation method: {method}")
    
    return segments

def visualize_segments(image, segments, output_path):
    """Visualize and save the segmentation results."""
    # Create a figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    
    # Plot original image
    ax1.imshow(image, cmap='gray')
    ax1.set_title('Original Image')
    ax1.axis('off')
    
    # Plot segmented image
    ax2.imshow(segments, cmap='nipy_spectral')
    ax2.set_title('Segmented Image')
    ax2.axis('off')
    
    # Save the figure
    plt.savefig(output_path, bbox_inches='tight', dpi=300)
    plt.close()

def process_dataset(input_dir, output_dir, method='felzenszwalb', max_images=10):
    """Process images in the input directory."""
    # Create output directory if it doesn't exist
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Get list of image files and sort them
    image_files = sorted(list(Path(input_dir).glob('*.png')))
    
    # Process only the first max_images
    for image_file in image_files[:max_images]:
        try:
            print(f"Processing {image_file.name}...")
            
            # Load and preprocess image
            img = load_and_preprocess_image(str(image_file))
            
            # Process each sample area
            for sample_name, area in sample_areas.items():
                print(f"  Processing {sample_name}...")
                
                # Extract sample area
                sample_img = extract_sample(img, sample_name)
                
                # Apply segmentation
                segments = segment_image(sample_img, method=method)
                
                # Visualize and save results
                output_file = output_path / f"{sample_name}_{image_file.name}"
                visualize_segments(sample_img, segments, str(output_file))
            
        except Exception as e:
            print(f"Error processing {image_file.name}: {str(e)}")

def load_image(image_path):
    """Load and preprocess the image."""
    img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Could not load image: {image_path}")
    return img

def apply_watershed_segmentation(img):
    """Apply watershed-based segmentation to the image."""
    # Calculate elevation map using Sobel gradient
    elevation_map = filters.sobel(img)
    
    # Find markers based on extreme parts of the histogram
    markers = np.zeros_like(img)
    markers[img < np.percentile(img, 30)] = 1  # Background markers
    markers[img > np.percentile(img, 70)] = 2  # Foreground markers
    
    # Apply watershed transform
    segmentation_result = segmentation.watershed(elevation_map, markers)
    
    # Clean up the segmentation
    segmentation_result = ndi.binary_fill_holes(segmentation_result - 1)
    labeled_result, _ = ndi.label(segmentation_result)
    
    return labeled_result, segmentation_result, elevation_map, markers

def plot_watershed_results(img, labeled_result, segmentation_result, elevation_map, markers, output_path):
    """Plot and save the watershed segmentation results."""
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Original image
    axes[0, 0].imshow(img, cmap='gray')
    axes[0, 0].set_title('Original Image')
    axes[0, 0].axis('off')
    
    # Elevation map
    axes[0, 1].imshow(elevation_map, cmap='gray')
    axes[0, 1].set_title('Elevation Map')
    axes[0, 1].axis('off')
    
    # Markers
    axes[1, 0].imshow(markers, cmap='nipy_spectral')
    axes[1, 0].set_title('Markers')
    axes[1, 0].axis('off')
    
    # Final segmentation
    image_label_overlay = color.label2rgb(labeled_result, image=img, bg_label=0)
    axes[1, 1].imshow(image_label_overlay)
    axes[1, 1].set_title('Segmentation Result')
    axes[1, 1].axis('off')
    
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

def process_images():
    """Process images using watershed segmentation."""
    input_dir = Path("/Users/zebpalm/Exjobb 2025/BSE images/abctoanalyze/All")
    output_dir = Path("watershed_segmentation")
    output_dir.mkdir(exist_ok=True)
    
    # Process first 20 images
    image_files = sorted(input_dir.glob("*.tif"))[:20]
    
    for img_path in image_files:
        try:
            # Load image
            img = load_image(img_path)
            
            # Apply watershed segmentation
            labeled_result, segmentation_result, elevation_map, markers = apply_watershed_segmentation(img)
            
            # Save results
            output_path = output_dir / f"{img_path.stem}_watershed.png"
            plot_watershed_results(img, labeled_result, segmentation_result, elevation_map, markers, output_path)
            
            print(f"Processed {img_path.name}")
            
        except Exception as e:
            print(f"Error processing {img_path.name}: {str(e)}")

if __name__ == "__main__":
    # Define input and output directories
    input_dir = "/Users/zebpalm/Exjobb 2025/BSE images/abctoanalyze/All"
    output_dir = "segmented_results"
    
    # Process the dataset using different segmentation methods
    methods = ['felzenszwalb', 'slic', 'quickshift']
    
    for method in methods:
        print(f"\nProcessing with {method} segmentation...")
        method_output_dir = os.path.join(output_dir, method)
        process_dataset(input_dir, method_output_dir, method=method, max_images=10)

    process_images()