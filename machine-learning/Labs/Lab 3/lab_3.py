# Step 1: Import libraries
import cv2
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA

# Step 2: Load the given image using OpenCV
image = cv2.imread('sat_image_plaksha1.jpg')

# Step 3: Convert the image to grayscale
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Step 4: Convert the image to double for mathematical operations
gray_image = gray_image.astype(np.float64)

# Step 5: Compute the mean of each column (pixels) and subtract it from the image
mean_column = np.mean(gray_image, axis=0)
image_mean_subtracted = gray_image - mean_column

# Step 6: Compute the covariance matrix
covariance_matrix = np.cov(image_mean_subtracted.T)

# Step 7: Get eigenvalues and eigenvectors
eigenvalues, eigenvectors = np.linalg.eig(covariance_matrix)

# Step 8: Sort eigenvectors by eigenvalues
sorted_indices = np.argsort(eigenvalues)[::-1]
eigenvalues = eigenvalues[sorted_indices]
eigenvectors = eigenvectors[:, sorted_indices]

# Step 9: Define the number of principal components to keep
num_components = [10, 20, 30, 40, 50, 60, 91]  # Adjust as needed

# Step 10: For each number of components, reconstruct the image
output_images = []
for n in num_components:
    selected_components = eigenvectors[:, :n]
    projected_data = np.dot(selected_components.T, image_mean_subtracted.T).T
    reconstructed_image = np.dot(selected_components, projected_data.T).T + mean_column
    output_images.append(reconstructed_image)

# Step 11: Display the results
plt.figure(figsize=(15, 10))
plt.suptitle("Dimensionality Reduction using PCA")

for i, n in enumerate(num_components):
    plt.subplot(2, 4, i + 1)
    plt.imshow(output_images[i], cmap='gray')
    plt.title(f"Components: {n}")
    plt.axis('off')

plt.show()

# Step 12: Calculate explained variance using sklearn's PCA
explained_variances = []
for n in num_components:
    pca = PCA(n_components=n)
    pca.fit(image_mean_subtracted)
    explained_variances.append(np.sum(pca.explained_variance_ratio_))

print("Explained Variances:")
for n, variance in zip(num_components, explained_variances):
    print(f"Components: {n}, Explained Variance: {variance:.4f}")
