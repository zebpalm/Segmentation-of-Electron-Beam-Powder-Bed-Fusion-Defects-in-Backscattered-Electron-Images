import numpy as np
import matplotlib.pyplot as plt
import skimage as ski
from skimage import filters
import cv2
from matplotlib.widgets import Slider
from scipy import ndimage

# Read the image
img = cv2.imread("/Users/zebpalm/Exjobb 2025/Coding/gaussian_filtering_all_results/Sample 4/ABC-capture-20241209-194134_topo02_13_gaussian_k7_s1.5.png", cv2.IMREAD_GRAYSCALE)

# Apply Sobel filter
sobel_horizontal = filters.sobel_h(img)  # Horizontal edges
sobel_vertical = filters.sobel_v(img)    # Vertical edges
sobel_combined = filters.sobel(img)      # Combined edges

# Normalize Sobel output to [0,1] for easier threshold selection
sobel_normalized = (sobel_combined - sobel_combined.min()) / (sobel_combined.max() - sobel_combined.min())

# Calculate different thresholds
otsu_threshold = filters.threshold_otsu(sobel_normalized)
percentile_threshold = np.percentile(sobel_normalized, 80)  # Top 20% strongest edges
mean_threshold = np.mean(sobel_normalized)

# Size threshold - change this value to adjust the maximum particle size
SIZE_THRESHOLD = 10  # pixels

# Create figure with subplots
fig = plt.figure(figsize=(15, 10))
gs = fig.add_gridspec(3, 3)

# Original image
ax1 = fig.add_subplot(gs[0, 0])
ax1.imshow(img, cmap='gray')
ax1.set_title('Original Image')
ax1.set_axis_off()

# Combined Sobel edges
ax2 = fig.add_subplot(gs[0, 1])
ax2.imshow(sobel_normalized, cmap='gray')
ax2.set_title('Sobel Edges (Normalized)')
ax2.set_axis_off()

# Otsu threshold
ax3 = fig.add_subplot(gs[0, 2])
ax3.imshow(~(sobel_normalized > otsu_threshold), cmap='gray')
ax3.set_title(f'Otsu Threshold ({otsu_threshold:.3f})')
ax3.set_axis_off()

# Percentile threshold
ax4 = fig.add_subplot(gs[1, 0])
ax4.imshow(~(sobel_normalized > percentile_threshold), cmap='gray')
ax4.set_title(f'80th Percentile ({percentile_threshold:.3f})')
ax4.set_axis_off()

# Mean threshold
ax5 = fig.add_subplot(gs[1, 1])
ax5.imshow(~(sobel_normalized > mean_threshold), cmap='gray')
ax5.set_title(f'Mean Value ({mean_threshold:.3f})')
ax5.set_axis_off()

# Size filtered Mean threshold
binary = ~(sobel_normalized > mean_threshold)  # Flip the binary - edges are black (0), background is white (1)
labeled_array, num_features = ndimage.label(binary)
sizes = np.bincount(labeled_array.ravel())
sizes[0] = 0  # Ignore background

# Create a mask where we keep only small black regions (particles) and make them white
size_mask = np.zeros_like(binary, dtype=bool)
for label in range(1, num_features + 1):
    if sizes[label] <= SIZE_THRESHOLD:  # If the region is small enough
        size_mask[labeled_array == label] = True  # Keep it and make it white

# Create red overlay
original_rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
overlay = original_rgb.copy()
overlay[size_mask] = [255, 0, 0]  # Set red channel to 255 where particles are

# Show the filtered binary image
ax6 = fig.add_subplot(gs[1, 2])
ax6.imshow(size_mask, cmap='gray')
ax6.set_title(f'Small Particles Only (Mean, ≤{SIZE_THRESHOLD}px)')
ax6.set_axis_off()

# Show the red overlay
ax7 = fig.add_subplot(gs[2, 0])
ax7.imshow(overlay)
ax7.set_title(f'Red Particles Overlay (Mean, ≤{SIZE_THRESHOLD}px)')
ax7.set_axis_off()

# Histogram
ax_hist = fig.add_subplot(gs[2, 1:])
hist, bins = np.histogram(sobel_normalized.flatten(), bins=100)
ax_hist.plot(bins[:-1], hist)
ax_hist.axvline(x=otsu_threshold, color='r', linestyle='--', label='Otsu')
ax_hist.axvline(x=percentile_threshold, color='g', linestyle='--', label='80th Percentile')
ax_hist.axvline(x=mean_threshold, color='b', linestyle='--', label='Mean')
ax_hist.set_title('Edge Magnitude Histogram')
ax_hist.set_xlabel('Edge Magnitude')
ax_hist.set_ylabel('Frequency')
ax_hist.legend()
ax_hist.grid(True)

plt.tight_layout()
plt.show()

# Print statistics
print("\nThreshold values:")
print(f"Otsu threshold: {otsu_threshold:.3f}")
print(f"80th percentile threshold: {percentile_threshold:.3f}")
print(f"Mean threshold: {mean_threshold:.3f}")
print(f"\nEdge magnitude statistics:")
print(f"Min: {sobel_normalized.min():.3f}")
print(f"Max: {sobel_normalized.max():.3f}")
print(f"Mean: {sobel_normalized.mean():.3f}")
print(f"Median: {np.median(sobel_normalized):.3f}")
print(f"Std dev: {sobel_normalized.std():.3f}")
print(f"\nSize filtering statistics:")
print(f"Total particles before filtering: {num_features}")
print(f"Particles after size filtering: {np.sum(sizes <= SIZE_THRESHOLD)}") 