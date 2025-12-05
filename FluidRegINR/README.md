# Fluid Registration INR (FRINR)

INR-based solution to register longitudinal OCT images with evolving (dissolving, newly appearing, growing etc.) 
pathologies.

For each image pair, two MLPs are adapted: One to predict the deformation field between moving and fixed image, and
one to generate a so-called residual image that contains structural differences between moving and fixed image. For 
newly developing pathologies, the residual contains a small artificial fluid region (corresponding to the appearance
offset of FluidRegNet) which is extended to the amount of fluid observed in the fixed image.

Compared to FluidRegNet, the FRINR manages to map extremely distorted retinae, but places the fluid seeds at less 
plausible locations. 

The code is based on ImpRegDec (https://conferences.miccai.org/2023/papers/331-Paper1136.html) with modifications for
3D and grayscale images.

Please cite 

FRINR: Pathology-Aware Implicit Neural Registration for Change Analysis in Retinal OCT Data
by J. Andresen, B. Kahrs, H. Handels and T. Kepp (BVM 2026)

when using this repository.
