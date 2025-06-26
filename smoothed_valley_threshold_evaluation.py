import cv2
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import precision_score, recall_score, f1_score
from smoothed_valley_threshold import load_image, preprocess_image, find_valleys, plot_histogram_with_valleys

def evaluate_segmentation(pred_mask, gt_mask):
    """Calculate all evaluation metrics for a pair of masks."""
    # Ensure masks are binary
    pred_mask = pred_mask > 0
    gt_mask = gt_mask > 0
    
    # Calculate metrics
    intersection = np.logical_and(pred_mask, gt_mask).sum()
    dice = (2. * intersection) / (pred_mask.sum() + gt_mask.sum())
    dice_dissim = 1 - dice
    
    union = np.logical_or(pred_mask, gt_mask).sum()
    jaccard = intersection / union if union > 0 else 0
    jaccard_dissim = 1 - jaccard
    
    precision = precision_score(gt_mask.flatten(), pred_mask.flatten())
    recall = recall_score(gt_mask.flatten(), pred_mask.flatten())
    f1 = f1_score(gt_mask.flatten(), pred_mask.flatten())
    
    return {
        'dice_dissimilarity': dice_dissim,
        'jaccard_dissimilarity': jaccard_dissim,
        'precision': precision,
        'recall': recall,
        'f1_score': f1
    }

def create_overlay_image(pred_mask, gt_mask, output_path):
    """Create an overlay image showing both predicted and ground truth masks."""
    # Create RGB image
    overlay = np.zeros((pred_mask.shape[0], pred_mask.shape[1], 3), dtype=np.uint8)
    
    # Convert masks to binary
    pred_mask = pred_mask > 0
    gt_mask = gt_mask > 0
    
    # Set colors:
    # - Red (255,0,0) for false positives (predicted but not in ground truth)
    # - Green (0,255,0) for true positives (predicted and in ground truth)
    # - Blue (0,0,255) for false negatives (in ground truth but not predicted)
    
    # True positives (green)
    true_positives = np.logical_and(pred_mask, gt_mask)
    overlay[true_positives] = [0, 255, 0]
    
    # False positives (red)
    false_positives = np.logical_and(pred_mask, ~gt_mask)
    overlay[false_positives] = [255, 0, 0]
    
    # False negatives (blue)
    false_negatives = np.logical_and(~pred_mask, gt_mask)
    overlay[false_negatives] = [0, 0, 255]
    
    # Save overlay image
    cv2.imwrite(str(output_path), overlay)

def process_image(img_path, gt_mask_path, output_dir):
    """Process a single image and compare with ground truth."""
    # Load and preprocess image
    img_original = load_image(img_path)
    img_processed = preprocess_image(img_original)
    
    # Compute histogram
    hist = cv2.calcHist([img_processed], [0], None, [256], [0, 256])
    hist = hist.flatten()
    
    # Find valleys
    smoothed_hist, valleys = find_valleys(hist)
    
    if not valleys:
        return None
    
    # Use the deepest valley as threshold
    threshold, _ = valleys[0]
    
    # Create binary mask
    _, pred_mask = cv2.threshold(img_processed, threshold, 255, cv2.THRESH_BINARY_INV)
    
    # Load ground truth mask
    gt_mask = cv2.imread(str(gt_mask_path), cv2.IMREAD_GRAYSCALE)
    if gt_mask is None:
        return None
    
    # Ensure masks have the same size
    if pred_mask.shape != gt_mask.shape:
        return None
    
    # Create overlay image
    overlay_path = output_dir / "overlays" / f"overlay_{img_path.stem}.png"
    create_overlay_image(pred_mask, gt_mask, overlay_path)
    
    # Save histogram visualization
    hist_path = output_dir / "histograms" / f"hist_{img_path.stem}.png"
    plot_histogram_with_valleys(smoothed_hist, valleys, img_processed, img_original, hist_path)
    
    # Calculate metrics
    metrics = evaluate_segmentation(pred_mask, gt_mask)
    metrics['image_name'] = img_path.stem
    metrics['threshold'] = threshold
    
    return metrics

def main():
    # Define paths
    pred_masks_dir = Path("/Users/zebpalm/Exjobb 2025/Coding/valley_detection/binary_masks")
    gt_masks_dir = Path("/Users/zebpalm/Exjobb 2025/Coding/binary_masks/overlapped_truth_masks/pore")
    output_dir = Path("valley_threshold_evaluation")
    output_dir.mkdir(exist_ok=True)
    (output_dir / "overlays").mkdir(exist_ok=True)
    
    # Initialize results list
    results = []
    
    # Get list of ground truth masks
    gt_masks = list(gt_masks_dir.glob("*.png"))
    
    # Process each ground truth mask
    for gt_mask_path in gt_masks:
        # Extract the base name (e.g., "topo1_image45_bottom_right")
        base_name = gt_mask_path.stem.replace("_mask", "")
        
        # Find corresponding predicted mask
        pred_mask_path = pred_masks_dir / f"filtered_{base_name}.png"
        if not pred_mask_path.exists():
            continue
        
        # Load masks
        pred_mask = cv2.imread(str(pred_mask_path), cv2.IMREAD_GRAYSCALE)
        gt_mask = cv2.imread(str(gt_mask_path), cv2.IMREAD_GRAYSCALE)
        
        if pred_mask is None or gt_mask is None:
            continue
        
        # Ensure masks have the same size
        if pred_mask.shape != gt_mask.shape:
            continue
        
        # Create overlay image
        overlay_path = output_dir / "overlays" / f"overlay_{base_name}.png"
        create_overlay_image(pred_mask, gt_mask, overlay_path)
        
        # Calculate metrics
        metrics = evaluate_segmentation(pred_mask, gt_mask)
        metrics['image_name'] = base_name
        
        results.append(metrics)
    
    if not results:
        return
    
    # Convert results to DataFrame
    df = pd.DataFrame(results)
    
    # Calculate average metrics
    avg_metrics = df.mean(numeric_only=True)
    
    # Save results
    df.to_csv(output_dir / "valley_threshold_metrics.csv", index=False)
    
    # Print average metrics
    print("\nAverage Metrics:")
    print(f"Dice Dissimilarity: {avg_metrics['dice_dissimilarity']:.4f}")
    print(f"Jaccard Dissimilarity: {avg_metrics['jaccard_dissimilarity']:.4f}")
    print(f"Precision: {avg_metrics['precision']:.4f}")
    print(f"Recall: {avg_metrics['recall']:.4f}")
    print(f"F1 Score: {avg_metrics['f1_score']:.4f}")

if __name__ == "__main__":
    main() 