import cv2
import numpy as np
import os
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

def preprocess_image(image, kernel_size=(9,9), sigma=2.0, block_size=3, constant=2):
    """
    Preprocess grayscale image with Gaussian blur, histogram equalization, and adaptive thresholding.
    
    Args:
        image: Input grayscale image
        kernel_size: Size of Gaussian kernel
        sigma: Standard deviation for Gaussian blur
        block_size: Size of a pixel neighborhood for adaptive thresholding (must be odd)
        constant: Constant subtracted from the mean or weighted mean
    
    Returns:
        Tuple of (preprocessed image, thresholded image, filtered segmented image)
    """
    # Apply Gaussian blur
    blurred = cv2.GaussianBlur(image, kernel_size, sigma)
    
    # Apply histogram equalization first
    equalized = cv2.equalizeHist(blurred)
    
    # Apply adaptive thresholding
    # Ensure block_size is odd
    if block_size % 2 == 0:
        block_size += 1
    
    # Apply adaptive thresholding using Gaussian method
    adaptive_thresh = cv2.adaptiveThreshold(
        equalized,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        block_size,
        constant
    )
    
    # Find connected components in the thresholded image
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(adaptive_thresh, connectivity=8)
    
    # Create a mask for regions that meet our criteria
    mask = np.zeros_like(adaptive_thresh)
    
    # Filter regions based on size and shape
    min_area = 500  # Minimum area in pixels
    max_area = 100000  # Maximum area in pixels
    min_solidity = 0.7  # Minimum solidity (area / convex hull area)
    
    for i in range(1, num_labels):  # Skip background (label 0)
        area = stats[i, cv2.CC_STAT_AREA]
        
        # Get the contour of this region
        region_mask = (labels == i).astype(np.uint8) * 255
        contours, _ = cv2.findContours(region_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if len(contours) > 0:
            contour = contours[0]
            # Calculate solidity
            hull = cv2.convexHull(contour)
            hull_area = cv2.contourArea(hull)
            solidity = float(area) / hull_area if hull_area > 0 else 0
            
            # Keep region if it meets our criteria
            if (min_area <= area <= max_area) and (solidity >= min_solidity):
                mask[labels == i] = 255
    
    # Apply morphological operations to clean up the segmentation
    kernel = np.ones((5,5), np.uint8)
    segmented = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    segmented = cv2.morphologyEx(segmented, cv2.MORPH_OPEN, kernel)
    
    return equalized, adaptive_thresh, segmented

def create_interactive_comparison(original_image, output_dir, img_name):
    """Create an interactive comparison with adjustable adaptive thresholding parameters."""
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    plt.subplots_adjust(bottom=0.3)  # Make room for the sliders
    
    # Display original image
    axes[0, 0].imshow(original_image, cmap='gray')
    axes[0, 0].set_title('Original Image')
    axes[0, 0].axis('off')
    
    # Display preprocessed image (fixed)
    preprocessed, _, _ = preprocess_image(original_image, block_size=11, constant=2)
    axes[0, 1].imshow(preprocessed, cmap='gray')
    axes[0, 1].set_title('Preprocessed (Gaussian + Equalization)')
    axes[0, 1].axis('off')
    
    # Display thresholded image (will be updated)
    thresholded_img = axes[1, 0].imshow(original_image, cmap='gray')
    axes[1, 0].set_title('Adaptive Thresholded Image')
    axes[1, 0].axis('off')
    
    # Display filtered image (will be updated)
    filtered_img = axes[1, 1].imshow(original_image, cmap='gray')
    axes[1, 1].set_title('Filtered Segmented Image')
    axes[1, 1].axis('off')
    
    # Add sliders
    ax_block_size = plt.axes([0.25, 0.15, 0.65, 0.03])
    ax_constant = plt.axes([0.25, 0.1, 0.65, 0.03])
    
    block_size_slider = Slider(
        ax=ax_block_size,
        label='Block Size',
        valmin=3,
        valmax=99,
        valinit=11,
        valstep=2  # Ensure odd numbers
    )
    
    constant_slider = Slider(
        ax=ax_constant,
        label='Constant',
        valmin=-20,
        valmax=20,
        valinit=2,
        valstep=1
    )
    
    def update(val):
        block_size = int(block_size_slider.val)
        constant = int(constant_slider.val)
        
        _, thresholded, filtered = preprocess_image(
            original_image,
            block_size=block_size,
            constant=constant
        )
        
        thresholded_img.set_data(thresholded)
        filtered_img.set_data(filtered)
        axes[1, 0].set_title(f'Adaptive Thresholded (Block Size: {block_size}, Constant: {constant})')
        fig.canvas.draw_idle()
        
        # Save the current state
        comparison = create_comparison_image(original_image, preprocessed, thresholded, filtered)
        output_path = output_dir / f"comparison_{img_name}_block{block_size}_const{constant}.png"
        cv2.imwrite(str(output_path), comparison)
    
    block_size_slider.on_changed(update)
    constant_slider.on_changed(update)
    
    # Save the initial state
    plt.savefig(output_dir / f"interactive_{img_name}.png")
    plt.show()

def create_comparison_image(original, preprocessed, thresholded, filtered_segmented):
    """
    Create a side-by-side comparison of original, preprocessed, and both segmented images with labels.
    
    Args:
        original: Original image sample
        preprocessed: Preprocessed image sample (after Gaussian blur and histogram equalization)
        thresholded: Image after adaptive thresholding
        filtered_segmented: Image segmented and filtered by size/shape
    
    Returns:
        Combined image with all four versions side by side
    """
    # Ensure all images are the same height
    h1, w1 = original.shape
    h2, w2 = preprocessed.shape
    h3, w3 = thresholded.shape
    h4, w4 = filtered_segmented.shape
    height = max(h1, h2, h3, h4)
    
    # Create a white background image with extra space for labels
    comparison = np.ones((height + 30, w1 + w2 + w3 + w4 + 30), dtype=np.uint8) * 255
    
    # Place images side by side
    comparison[30:30+h1, :w1] = original
    comparison[30:30+h2, w1+10:w1+10+w2] = preprocessed
    comparison[30:30+h3, w1+w2+20:w1+w2+20+w3] = thresholded
    comparison[30:30+h4, w1+w2+w3+30:] = filtered_segmented
    
    # Add vertical lines between images
    comparison[30:, w1:w1+10] = 0
    comparison[30:, w1+w2+10:w1+w2+20] = 0
    comparison[30:, w1+w2+w3+20:w1+w2+w3+30] = 0
    
    # Add labels with smaller font
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5  # Smaller font size
    thickness = 1    # Thinner text
    
    # Calculate text positions
    text_y = 20
    cv2.putText(comparison, "Original", (w1//2 - 30, text_y), font, font_scale, 0, thickness)
    cv2.putText(comparison, "Preprocessed", (w1 + w2//2 - 40, text_y), font, font_scale, 0, thickness)
    cv2.putText(comparison, "Adaptive Thresholded", (w1 + w2 + w3//2 - 50, text_y), font, font_scale, 0, thickness)
    cv2.putText(comparison, "Filtered", (w1 + w2 + w3 + w4//2 - 20, text_y), font, font_scale, 0, thickness)
    
    return comparison

def main():
    # Input and output directories
    input_dir = Path("/Users/zebpalm/Exjobb 2025/Coding/evaluation_images/topo1")
    output_dir = Path("/Users/zebpalm/Exjobb 2025/Coding/adaptive_threshold_filtered")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Processing images from: {input_dir}")
    print(f"Saving processed images to: {output_dir}")
    
    # Process each image in the input directory
    for img_path in input_dir.glob("*.png"):
        print(f"Processing: {img_path}")
        # Read image
        image = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        if image is None:
            print(f"Could not read image: {img_path}")
            continue
        
        # Create interactive comparison
        create_interactive_comparison(image, output_dir, img_path.stem)

if __name__ == "__main__":
    main()