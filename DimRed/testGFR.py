'''
Citation: 
B. Yaghooti, N. Raviv, and B. Sinopoli, 
"Gram-Schmidt Methods for Unsupervised Feature Extraction and Selection," 
arXiv preprint arXiv:2311.09386, Aug. 2024. [Online]. Available: https://arxiv.org/abs/2311.09386

Some code incorporated in this reimplementation came from the author's github repo:
Byaghooti, "Gram Schmidt Feature Extraction," GitHub repository, 2022. [Online]. 
Available: https://github.com/byaghooti/Gram_schmidt_feature_extraction. [Accessed: May 1, 2025].

I reimplemented the GFR process proposed in the paper. While the original work uses a series of interaction 
terms—such as {f, f₁f₂, f₁f₂f₃, ...}—as the function family, I used a simplified example in which 
the function family F consists of {f, f²}.
'''

import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

def norm(x):
    return np.sqrt(np.mean(x**2))

# Step 1: Create synthetic data (x2 = x1^2)
'''
Original Data Matrix
'''
x1 = np.array([1, 2, 3, 4, 5], dtype=float)
x2 = x1 ** 2
X_original = np.array([x1, x2])  # Shape: (2, 5)

# Step 2: Center the data
'''
StandardScalar method (mean = 0; sd = 1)
'''
# StanScal = StandardScaler()
# X = StanScal.fit_transform(X)

'''
Centering method (mean = 0)
'''
X_mean = np.mean(X_original, axis=1, keepdims=True)
X = X_original - X_mean
print(f"Mean of X_centered(data matrix X): {np.mean(X)}")

# Step 3: Compute covariance and do PCA (only first component)
'''
pca build in methods
'''
# pca = PCA(n_components=2)
# pca.fit(X.T)
# eigenvalues = pca.explained_variance_
# eigenvectors = pca.components_
# v = np.array([eigenvectors[0], eigenvectors[1]])
# z = np.array([np.matmul(v[0], X), np.matmul(v[1], X)])

'''
write from scratch
'''
Sigma = np.cov(X)
eigvals, eigenvectors = np.linalg.eigh(Sigma)  # ascending order
v = np.array([eigenvectors[1], eigenvectors[0]]) # manually adjust the order to make evectors listing descending


# Step 4: Construct function basis: z1 and z1_sqr
z = np.array([np.matmul(v[0], X), np.matmul(v[1], X)])  # only use the first one
'''
First function
'''
z1 = z[0]
z1_hat = z1/norm(z1) # f_1 = Z1_hat which is zero value in mean
F_hat = np.array([z1_hat])

'''
Second function
'''
z1_sqr = z1 ** 2 - np.mean(z1 ** 2) #key of GS! Make sure the new term is built on zero average value
z_sqr_tilde = z1_sqr
z_sqr_tilde = z_sqr_tilde - np.mean(z1_sqr*F_hat[0])*F_hat[0]
z_sqr_hat = z_sqr_tilde/norm(z_sqr_tilde) # f_2 = orthognalized z1^2
F_hat = np.append(F_hat, np.array([z_sqr_hat]), axis=0)

# Step 5: Project X_centered onto the function space and compute residual
d_j = X.copy()
for i in range(len(F_hat)):
    d_j = d_j - np.matmul(np.array([np.mean(X*F_hat[i],axis=1)]).T, np.array([F_hat[i]]))
    print(i, ':', d_j)
    print(f"norm: {norm(d_j)}")


# Step 6: Plot everything
fig, axs = plt.subplots(1, 3, figsize=(15, 4))

# Original data
axs[0].scatter(X_original[0], X_original[1], color='blue')
axs[0].set_title("Original Data (x2 = x1^2)")
axs[0].set_xlabel("x1")
axs[0].set_ylabel("x2")

# PCA direction
origin = np.mean(X_original, axis=1)
v_scaled = v[0] * 5 # note this is the direction of the first principle
axs[0].quiver(origin[0], origin[1], v_scaled[0], v_scaled[1], 
              color='red', scale=1, scale_units='xy', angles='xy', width=0.01)

# Projection z1 vs z1^2
axs[1].scatter(z1, z1_sqr, color='green')
axs[1].set_title("Function Space: z1 vs z1^2")
axs[1].set_xlabel("z1")
axs[1].set_ylabel("z1^2")

# Residual data
axs[2].scatter(d_j[0], d_j[1], color='purple')
axs[2].set_title("Residual after GFR")
axs[2].set_xlabel("Residual x1")
axs[2].set_ylabel("Residual x2")
axs[2].autoscale()

plt.tight_layout()
plt.show()
