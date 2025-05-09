import skimage
import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.segmentation import watershed
from skimage import filters, feature

# Read image
img = cv2.imread("/Users/zebpalm/Exjobb 2025/Coding/gaussian_filtering_all_results/Sample 4/ABC-capture-20241209-194134_topo02_13_gaussian_k7_s1.5.png", cv2.IMREAD_GRAYSCALE)

# Apply edge detection
sobel = filters.sobel(img)
canny = feature.canny(img)

# Convert to uint8
canny_uint8 = (canny * 255).astype(np.uint8)
sobel_uint8 = (sobel * 255).astype(np.uint8)

# Create elevation map for watershed
elevation_map = sobel

# Create markers for watershed
markers = np.zeros_like(img)
markers[img < 30] = 1    # Mark dark regions
markers[img > 150] = 2   # Mark bright regions

# Apply watershed segmentation
segmentation = watershed(elevation_map, markers)

# Create figure with subplots
plt.figure(figsize=(12, 8))

# Plot original image
plt.subplot(231)
plt.imshow(img, cmap='gray')
plt.title('Original Image')
plt.axis('off')

# Plot Sobel edges
plt.subplot(232)
plt.imshow(sobel, cmap='gray')
plt.title('Elevation Map (Sobel)')
plt.axis('off')

# Plot markers
plt.subplot(233)
plt.imshow(markers, cmap='nipy_spectral')
plt.title('Markers')
plt.axis('off')

# Plot segmentation result
plt.subplot(234)
plt.imshow(segmentation, cmap='nipy_spectral')
plt.title('Watershed Segmentation')
plt.axis('off')

# Plot Canny edges
plt.subplot(235)
plt.imshow(canny, cmap='gray')
plt.title('Canny Edges')
plt.axis('off')

plt.tight_layout()
plt.show()
