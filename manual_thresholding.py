import cv2
import numpy as np
import os
from pathlib import Path

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
        Tuple of (preprocessed image, thresholded image, eroded_dilated image, filtered segmented image)
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
    
    # First filtering step: Erosion and Dilation
    kernel = np.ones((7,7), np.uint8)
    eroded = cv2.erode(adaptive_thresh, kernel, iterations=1)
    eroded_dilated = cv2.dilate(eroded, kernel, iterations=1)
    
    # Find connected components in the filtered image
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(eroded_dilated, connectivity=8)
    
    # Create a mask for regions that meet our criteria
    mask = np.zeros_like(eroded_dilated)
    
    # Filter regions based on size and shape
    min_area = 1000  # Minimum area in pixels
    max_area = 100000  # Maximum area in pixels
    min_solidity = 0.67  # Minimum solidity (area / convex hull area)
    
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
    
    return equalized, adaptive_thresh, eroded_dilated, segmented

def create_comparison_image(original, preprocessed, thresholded, eroded_dilated, filtered_segmented):
    """
    Create a side-by-side comparison of original, preprocessed, and both segmented images with labels.
    
    Args:
        original: Original image sample
        preprocessed: Preprocessed image sample (after Gaussian blur and histogram equalization)
        thresholded: Image after adaptive thresholding
        eroded_dilated: Image after erosion and dilation
        filtered_segmented: Image segmented and filtered by size/shape
    
    Returns:
        Combined image with all five versions side by side
    """
    # Ensure all images are the same height
    h1, w1 = original.shape
    h2, w2 = preprocessed.shape
    h3, w3 = thresholded.shape
    h4, w4 = eroded_dilated.shape
    h5, w5 = filtered_segmented.shape
    height = max(h1, h2, h3, h4, h5)
    
    # Create a white background image with extra space for labels
    comparison = np.ones((height + 30, w1 + w2 + w3 + w4 + w5 + 40), dtype=np.uint8) * 255
    
    # Place images side by side
    comparison[30:30+h1, :w1] = original
    comparison[30:30+h2, w1+10:w1+10+w2] = preprocessed
    comparison[30:30+h3, w1+w2+20:w1+w2+20+w3] = thresholded
    comparison[30:30+h4, w1+w2+w3+30:w1+w2+w3+30+w4] = eroded_dilated
    comparison[30:30+h5, w1+w2+w3+w4+40:] = filtered_segmented
    
    # Add vertical lines between images
    comparison[30:, w1:w1+10] = 0
    comparison[30:, w1+w2+10:w1+w2+20] = 0
    comparison[30:, w1+w2+w3+20:w1+w2+w3+30] = 0
    comparison[30:, w1+w2+w3+w4+30:w1+w2+w3+w4+40] = 0
    
    # Add labels with smaller font
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5  # Smaller font size
    thickness = 1    # Thinner text
    
    # Calculate text positions
    text_y = 20
    cv2.putText(comparison, "Original", (w1//2 - 30, text_y), font, font_scale, 0, thickness)
    cv2.putText(comparison, "Preprocessed", (w1 + w2//2 - 40, text_y), font, font_scale, 0, thickness)
    cv2.putText(comparison, "Adaptive Thresholded", (w1 + w2 + w3//2 - 50, text_y), font, font_scale, 0, thickness)
    cv2.putText(comparison, "Eroded & Dilated", (w1 + w2 + w3 + w4//2 - 40, text_y), font, font_scale, 0, thickness)
    cv2.putText(comparison, "Filtered", (w1 + w2 + w3 + w4 + w5//2 - 20, text_y), font, font_scale, 0, thickness)
    
    return comparison

def main():
    # Input and output directories
    input_dir = Path("/Users/zebpalm/Exjobb 2025/BSE images/topo1/Top_180_divided_samples")
    output_dir = Path("/Users/zebpalm/Exjobb 2025/Coding/adaptive_threshold_filtered/top_180_topo1")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Processing images from: {input_dir}")
    print(f"Saving processed images to: {output_dir}")
    
    # Process each image in the input directory
    for img_path in input_dir.glob("*.png"):
        #print(f"Processing: {img_path}")
        # Read image
        image = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        if image is None:
            print(f"Could not read image: {img_path}")
            continue
        
        # Preprocess image with default parameters
        preprocessed, thresholded, eroded_dilated, filtered = preprocess_image(image)
        
        # Create comparison image
        comparison = create_comparison_image(
            image,
            preprocessed,
            thresholded,
            eroded_dilated,
            filtered
        )
        
        # Save comparison image
        comparison_path = output_dir / f"comparison_{img_path.stem}.png"
        cv2.imwrite(str(comparison_path), comparison)
        print(f"Saved comparison: {comparison_path}")

if __name__ == "__main__":
    main()