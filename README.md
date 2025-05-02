# 📘 Project Purpose
This repository documents my efforts to independently reimplement and learn from published works in machine learning, signal processing, and data analysis.
All reimplementations are conducted strictly for educational purposes, and I take care to cite and credit original authors to ensure compliance with academic integrity standards.

# 📌 Included Reimplementations

## 1. Gram-Schmidt Feature Relearning (GFR)
This subproject reimplements the Gram-Schmidt-based unsupervised feature extraction process described in the following paper:

Citation:
B. Yaghooti, N. Raviv, and B. Sinopoli,
"Gram-Schmidt Methods for Unsupervised Feature Extraction and Selection,"
arXiv preprint arXiv:2311.09386, Aug. 2024.
Available online

Some reference code was also consulted from the author’s original repository:

Byaghooti, "Gram Schmidt Feature Extraction," GitHub repository, 2022.
GitHub Link
[Accessed: May 1, 2025]

🔍 Description of My Implementation
In this reimplementation:

I recreate the GFR process as described in the paper using synthetic data.

Unlike the original work, which uses complex higher-order interaction terms (e.g., {f, f₁f₂, f₁f₂f₃, ...}), I simplify the function family to {f, f²} for clarity and learning purposes.

The goal is to understand how orthogonalized nonlinear basis functions contribute to residual reduction in feature spaces.

All derivations, code structures, and simplifications were created from scratch unless otherwise stated and are thoroughly annotated for learning.

