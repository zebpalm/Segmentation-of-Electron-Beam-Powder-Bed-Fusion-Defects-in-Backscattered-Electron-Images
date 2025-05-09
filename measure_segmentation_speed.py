import cv2
import numpy as np
import os
from pathlib import Path
import time
from preprocess_evaluation import divide_image, sample_areas

def measure_segmentation_speed(image_path):
    """
    Measure the processing speed of each step in the segmentation pipeline.
    
    Args:
        image_path: Path to the input image
    
    Returns:
        Dictionary containing timing information for each step
    """
    # Initialize timing dictionary
    timings = {
        'total_time': 0,
        'read_image': 0,
        'divide_samples': 0,
        'gaussian_blur': 0,
        'histogram_equalization': 0,
        'otsu_thresholding': 0,
        'connected_components': 0,
        'region_filtering': 0,
        'morphological_operations': 0,
        'create_comparison': 0,
        'save_output': 0
    }
    
    start_total = time.time()
    
    # Read image
    start = time.time()
    image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"Could not read image: {image_path}")
    timings['read_image'] = time.time() - start
    
    # Divide into samples
    start = time.time()
    original_samples = divide_image(image)
    timings['divide_samples'] = time.time() - start
    
    # Process each sample
    for sample_name in sample_areas.keys():
        sample = original_samples[sample_name]
        
        # Gaussian blur
        start = time.time()
        blurred = cv2.GaussianBlur(sample, (9,9), 2.0)
        timings['gaussian_blur'] += time.time() - start
        
        # Histogram equalization
        start = time.time()
        equalized = cv2.equalizeHist(blurred)
        timings['histogram_equalization'] += time.time() - start
        
        # Otsu thresholding
        start = time.time()
        _, otsu_thresh = cv2.threshold(equalized, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        timings['otsu_thresholding'] += time.time() - start
        
        # Connected components
        start = time.time()
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(otsu_thresh, connectivity=8)
        timings['connected_components'] += time.time() - start
        
        # Region filtering
        start = time.time()
        mask = np.zeros_like(otsu_thresh)
        min_area = 300
        max_area = 10000
        min_solidity = 0.6
        
        for i in range(1, num_labels):
            area = stats[i, cv2.CC_STAT_AREA]
            region_mask = (labels == i).astype(np.uint8) * 255
            contours, _ = cv2.findContours(region_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if len(contours) > 0:
                contour = contours[0]
                hull = cv2.convexHull(contour)
                hull_area = cv2.contourArea(hull)
                solidity = float(area) / hull_area if hull_area > 0 else 0
                
                if (min_area <= area <= max_area) and (solidity >= min_solidity):
                    mask[labels == i] = 255
        timings['region_filtering'] += time.time() - start
        
        # Morphological operations
        start = time.time()
        kernel = np.ones((5,5), np.uint8)
        segmented = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        segmented = cv2.morphologyEx(segmented, cv2.MORPH_OPEN, kernel)
        timings['morphological_operations'] += time.time() - start
    
    # Calculate total time
    timings['total_time'] = time.time() - start_total
    
    return timings

def main():
    # Input directory
    input_dir = Path("/Users/zebpalm/Exjobb 2025/BSE images/topo1/Top 180")
    
    # Process each image and collect timing information
    all_timings = []
    for img_path in input_dir.rglob("*"):
        if img_path.suffix.lower() in ['.png', '.jpg', '.jpeg']:
            print(f"\nProcessing: {img_path}")
            try:
                timings = measure_segmentation_speed(img_path)
                all_timings.append(timings)
                
                # Print timing information for this image
                print("\nTiming information (seconds):")
                print(f"Total processing time: {timings['total_time']:.3f}")
                print(f"Read image: {timings['read_image']:.3f}")
                print(f"Divide samples: {timings['divide_samples']:.3f}")
                print(f"Gaussian blur: {timings['gaussian_blur']:.3f}")
                print(f"Histogram equalization: {timings['histogram_equalization']:.3f}")
                print(f"Otsu thresholding: {timings['otsu_thresholding']:.3f}")
                print(f"Connected components: {timings['connected_components']:.3f}")
                print(f"Region filtering: {timings['region_filtering']:.3f}")
                print(f"Morphological operations: {timings['morphological_operations']:.3f}")
                
            except Exception as e:
                print(f"Error processing {img_path}: {str(e)}")
    
    # Calculate and print average timings
    if all_timings:
        print("\nAverage timing information (seconds):")
        avg_timings = {key: sum(t[key] for t in all_timings) / len(all_timings) 
                      for key in all_timings[0].keys()}
        
        print(f"Total processing time: {avg_timings['total_time']:.3f}")
        print(f"Read image: {avg_timings['read_image']:.3f}")
        print(f"Divide samples: {avg_timings['divide_samples']:.3f}")
        print(f"Gaussian blur: {avg_timings['gaussian_blur']:.3f}")
        print(f"Histogram equalization: {avg_timings['histogram_equalization']:.3f}")
        print(f"Otsu thresholding: {avg_timings['otsu_thresholding']:.3f}")
        print(f"Connected components: {avg_timings['connected_components']:.3f}")
        print(f"Region filtering: {avg_timings['region_filtering']:.3f}")
        print(f"Morphological operations: {avg_timings['morphological_operations']:.3f}")

if __name__ == "__main__":
    main() 