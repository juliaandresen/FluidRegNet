# FluidRegNet
Code implementing the pathology-aware registration of retinal OCT images described in "FluidRegNet: Longitudinal registration of retinal OCT images with new pathological fluids" (https://proceedings.mlr.press/v250/andresen24a.html)

Basic idea of FluidRegNet: Compensate non-correspondences between time points (new or dissolving pathologies) by inserting small pathology "seeds", i.e. inpaint small dark areas, and extending these seeds to the amount of fluid observed in the subsequent time point. Using a masked regularizer, this allows the generation of more realistic, pathology-aware deformation fields. The network learns to predict deformation fields and to place the fluid seeds in an unsupervised manner, based on image similarity, masked deformation regularization and sparsity of the appearance offset maps. The masking of the regularizer is performed based on automatically generated rough fluid segmentations (simple thresholding and morphological operations).

For the results in the paper, the OCT images were flattened at the Bruch's membrane.

To run the code, you need to enter the path to your data (in loaders.py) and the path to the location where the results will be stored (in training.py). 

A demo script showing how to use the trained FluidRegNet will be added soon.
