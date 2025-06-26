import cv2
import numpy as np
import os
from pathlib import Path
import time
from EqualizedHist_adaptiveThres import preprocess_image

def measure_segmentation_speed(image_path):
    """
    Measure the processing speed of each step in the adaptive thresholding pipeline.
    Returns timing for each internal step.
    
    Args:
        image_path: Path to the input image
    
    Returns:
        Dictionary containing timing information for each step
    """
    timings = {}
    start_total = time.time()
    # Read image
    start = time.time()
    image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"Could not read image: {image_path}")
    timings['read_image'] = time.time() - start
    
    # Preprocess image with timing
    equalized, adaptive_thresh, eroded_dilated, segmented, step_timings = preprocess_image(image, return_timings=True)
    timings.update(step_timings)
    
    # Save binary mask
    start = time.time()
    output_dir = Path("/Users/zebpalm/Exjobb 2025/Coding/adaptive_threshold_filtered/evaluation_images_topo1/topo1_evaluation/binary_masks")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"filtered_{image_path.stem}.png"
    cv2.imwrite(str(output_path), segmented)
    timings['save_mask'] = time.time() - start
    
    timings['total_time'] = time.time() - start_total
    return timings

def main():
    # Input directory
    input_dir = Path("/Users/zebpalm/Exjobb 2025/BSE images/evaluation_images/topo1_evaluation")
    
    # Process each image and collect timing information
    all_timings = []
    for img_path in input_dir.glob("*.png"):
        print(f"\nProcessing: {img_path}")
        try:
            timings = measure_segmentation_speed(img_path)
            all_timings.append(timings)
            
            # Print timing information for this image
            print("\nTiming information (milliseconds):")
            for key, value in timings.items():
                print(f"{key.replace('_', ' ').capitalize()}: {value*1000:.1f}")
        except Exception as e:
            print(f"Error processing {img_path}: {str(e)}")
    
    # Calculate and print average timings
    if all_timings:
        print("\nAverage timing information (milliseconds):")
        avg_timings = {key: sum(t.get(key,0) for t in all_timings) / len(all_timings) for key in all_timings[0].keys()}
        for key, value in avg_timings.items():
            print(f"{key.replace('_', ' ').capitalize()}: {value*1000:.1f}")

if __name__ == "__main__":
    main() 