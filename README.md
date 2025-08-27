# FluidRegNet
Code implementing the pathology-aware registration of retinal OCT images described in "FluidRegNet: Longitudinal registration of retinal OCT images with new pathological fluids" (https://proceedings.mlr.press/v250/andresen24a.html)

Basic idea of FluidRegNet: Compensate non-correspondences between time points (new or dissolving pathologies) by inserting small pathology "seeds", i.e. inpaint small dark areas, and extending these seeds to the amount of fluid observed in the subsequent time point. Using a masked regularizer, this allows the generation of more realistic, pathology-aware deformation fields. The network learns to predict deformation fields and to place the fluid seeds in an unsupervised manner, based on image similarity, masked deformation regularization and sparsity of the appearance offset maps. The masking of the regularizer is performed based on automatically generated rough fluid segmentations (simple thresholding and morphological operations).

For the results in the paper, the OCT images were flattened at the Bruch's membrane.

To run the code, you need to enter the path to your data (in loaders.py) and the path to the location where the results will be stored (in training.py). 


## Anomaly Detection with FluidRegNet
1. Re-training for registration of healthy to pathological images (trainingHealthyToPatho.py)
2. Use trained network to register several healthy images to one pathological image and use deformation fields and appearance offsets to generate (binary) segmentations of pathologies (generateAnomalyMaps.py)
3. Combine segmentations to get anomaly score (generateAnomalyMaps.py)


## FluidRegINR
INR-based implementation of fluid-aware OCT registration using two INRs per image pair that generate deformation fields
and pathology seeds, respectively. Compared to FluidRegNet, large deformations can be mapped better, the
inserted pathologies, however, are less accurate.
