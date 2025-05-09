import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage import filters

# Specify the image path
IMAGE_PATH = "/Users/zebpalm/Exjobb 2025/Coding/gaussian_filtering_all_results/Sample 4/ABC-capture-20241209-194134_topo02_13_gaussian_k7_s1.5.png"

# Read the image
img = cv2.imread(IMAGE_PATH, cv2.IMREAD_GRAYSCALE)

# Sobel operator
sobel = filters.sobel(img)

# Process Sobel output for better visualization
sobel_normalized = (sobel - sobel.min()) / (sobel.max() - sobel.min())  # Normalize to [0,1]
sobel_uint8 = (sobel_normalized * 255).astype(np.uint8)  # Convert to uint8

# Create figure with subplots for images and histograms
fig, axes = plt.subplots(2, 2, figsize=(12, 8))
fig.suptitle('Original Image vs Sobel Edge Detection', fontsize=14)

# Show original image
axes[0, 0].imshow(img, cmap='gray')
axes[0, 0].set_title('Original Image')
axes[0, 0].set_axis_off()

# Original image histogram
hist_orig, bins_orig = np.histogram(img.flatten(), bins=256, range=[0, 256])
axes[0, 1].plot(bins_orig[:-1], hist_orig, color='black')
axes[0, 1].set_title('Original Image Histogram')
axes[0, 1].set_xlabel('Pixel Value')
axes[0, 1].set_ylabel('Frequency')
axes[0, 1].grid(True)

# Show Sobel image
axes[1, 0].imshow(sobel_normalized, cmap='gray')
axes[1, 0].set_title('Sobel Edge Detection')
axes[1, 0].set_axis_off()

# Sobel histogram
hist_sobel, bins_sobel = np.histogram(sobel.flatten(), bins=100)
axes[1, 1].plot(bins_sobel[:-1], hist_sobel, color='black')
axes[1, 1].set_title('Sobel Edge Magnitude Histogram')
axes[1, 1].set_xlabel('Edge Magnitude')
axes[1, 1].set_ylabel('Frequency')
axes[1, 1].grid(True)

# Add statistics
orig_stats = f'Min: {img.min()}\nMax: {img.max()}\nMean: {img.mean():.2f}\nStd: {img.std():.2f}'
sobel_stats = f'Min: {sobel.min():.3f}\nMax: {sobel.max():.3f}\nMean: {sobel.mean():.3f}\nStd: {sobel.std():.3f}'

axes[0, 1].text(0.95, 0.95, orig_stats,
                transform=axes[0, 1].transAxes,
                verticalalignment='top',
                horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

axes[1, 1].text(0.95, 0.95, sobel_stats,
                transform=axes[1, 1].transAxes,
                verticalalignment='top',
                horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

plt.tight_layout()
plt.show()

# Print detailed statistics
print("\nOriginal Image Statistics:")
print(f"Min value: {img.min()}")
print(f"Max value: {img.max()}")
print(f"Mean value: {img.mean():.2f}")
print(f"Median value: {np.median(img):.2f}")
print(f"Standard deviation: {img.std():.2f}")

print("\nSobel Edge Detection Statistics:")
print(f"Min value: {sobel.min():.3f}")
print(f"Max value: {sobel.max():.3f}")
print(f"Mean value: {sobel.mean():.3f}")
print(f"Median value: {np.median(sobel):.3f}")
print(f"Standard deviation: {sobel.std():.3f}")
print("\nNote: Sobel values represent edge magnitudes (0 = no edge, higher values = stronger edges)") 