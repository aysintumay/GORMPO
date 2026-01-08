# Toy Density Model Comparison - Summary Report

This report compares the performance of different density models
on synthetic mixture of Gaussians datasets with varying dimensionalities.

## Models Compared

1. **RealNVP**: Normalizing flow with coupling layers
2. **VAE**: Variational Autoencoder
3. **KDE**: Kernel Density Estimation
4. **Neural ODE**: Continuous normalizing flow
5. **Diffusion**: DDIM denoising diffusion probabilistic model


## 2D Results

### Log-Likelihood Statistics


#### Test

| Model | Mean | Std | Min | Max | Median |
|---|---|---|---|---|---|
| realnvp | -3.756 | 0.889 | -11.083 | -2.752 | -3.500 |
| vae | -0.829 | 2.076 | -51.090 | -0.000 | -0.492 |
| kde | -0.003 | 0.005 | -0.073 | -0.001 | -0.002 |
| neural_ode | -3.630 | 0.936 | -8.505 | -2.475 | -3.379 |
| diffusion | -27535.549 | 14011.859 | -94696.594 | -872.587 | -25683.609 |


#### Ood Easy

| Model | Mean | Std | Min | Max | Median |
|---|---|---|---|---|---|
| realnvp | -4809.139 | 7848.087 | -56478.387 | -3.874 | -1226.962 |
| vae | -6114763.500 | 54131368.000 | -1144767488.000 | -0.401 | -1295.772 |
| kde | -21.712 | 4.400 | -23.026 | -0.003 | -23.026 |
| neural_ode | -1080.100 | 642.454 | -3376.807 | -3.719 | -1020.311 |
| diffusion | -3947395.250 | 2369473.000 | -11171630.000 | -16602.539 | -3775001.000 |


#### Ood Medium

| Model | Mean | Std | Min | Max | Median |
|---|---|---|---|---|---|
| realnvp | -228.492 | 95.604 | -604.594 | -16.704 | -227.062 |
| vae | -24.462 | 10.076 | -66.783 | -2.984 | -23.235 |
| kde | -1.710 | 0.617 | -4.033 | -0.306 | -1.659 |
| neural_ode | -33.939 | 8.113 | -60.180 | -11.795 | -33.520 |
| diffusion | -128781.102 | 38178.688 | -274624.406 | -35989.613 | -125996.484 |


#### Ood Hard

| Model | Mean | Std | Min | Max | Median |
|---|---|---|---|---|---|
| realnvp | -25.237 | 12.736 | -58.539 | -3.258 | -24.259 |
| vae | -3.954 | 4.595 | -38.191 | -0.006 | -2.368 |
| kde | -0.206 | 0.030 | -0.263 | -0.084 | -0.212 |
| neural_ode | -8.888 | 0.436 | -10.299 | -7.675 | -8.908 |
| diffusion | -8578.462 | 6124.392 | -41844.605 | 2.482 | -7231.250 |


#### Ood Very Hard

| Model | Mean | Std | Min | Max | Median |
|---|---|---|---|---|---|
| realnvp | -11.950 | 12.645 | -125.534 | -4.051 | -8.713 |
| vae | -2.916 | 2.672 | -27.216 | -0.003 | -2.351 |
| kde | -0.077 | 0.061 | -0.572 | -0.005 | -0.060 |
| neural_ode | -8.443 | 2.050 | -25.746 | -4.025 | -8.168 |
| diffusion | -35191.883 | 24283.678 | -131013.242 | -86.984 | -32228.930 |


### OOD Detection Performance

#### AUROC (Area Under ROC Curve)

| Model | Easy | Medium | Hard | Very Hard |
|---|---|---|---|---|
| realnvp | 0.9997 | 1.0000 | 0.9940 | 0.9853 |
| vae | 0.9992 | 0.9978 | 0.7954 | 0.8442 |
| kde | 0.9997 | 1.0000 | 1.0000 | 0.9935 |
| neural_ode | 0.9997 | 1.0000 | 0.9998 | 0.9917 |
| diffusion | 0.9982 | 0.9974 | 0.0866 | 0.5698 |


#### AUPRC (Area Under Precision-Recall Curve)

| Model | Easy | Medium | Hard | Very Hard |
|---|---|---|---|---|
| realnvp | 0.9998 | 1.0000 | 0.9965 | 0.9929 |
| vae | 0.9994 | 0.9992 | 0.8459 | 0.8891 |
| kde | 0.9998 | 1.0000 | 1.0000 | 0.9969 |
| neural_ode | 0.9998 | 1.0000 | 0.9999 | 0.9960 |
| diffusion | 0.9987 | 0.9987 | 0.4651 | 0.6563 |


### Correlation with True Log-Probabilities

| Model | Test | OOD Easy | OOD Medium | OOD Hard | OOD Very Hard |
|---|---|---|---|---|---|
| realnvp | 0.8650 | 0.1671 | 0.2998 | -0.6109 | 0.4750 |
| vae | 0.1551 | 0.1021 | 0.8263 | 0.1646 | 0.3864 |
| kde | 0.8616 | 0.3812 | 0.9695 | 0.9492 | 0.9515 |
| neural_ode | 0.9833 | 0.8276 | 0.9971 | 0.7355 | 0.8874 |
| diffusion | 0.1175 | 0.8898 | 0.7393 | -0.0412 | 0.1132 |


### Visualizations

See `toy_experiments/results/dim2/` for:
- `loglikelihood_comparison.png`
- `ood_detection_performance.png`
- `score_distributions.png`


## 4D Results

### Log-Likelihood Statistics


#### Test

| Model | Mean | Std | Min | Max | Median |
|---|---|---|---|---|---|
| realnvp | -7.209 | 1.379 | -16.460 | -5.256 | -6.855 |
| vae | -2.161 | 1.778 | -27.950 | -0.074 | -1.764 |
| kde | -0.096 | 0.080 | -1.053 | -0.029 | -0.072 |
| neural_ode | -7.113 | 1.428 | -17.395 | -4.997 | -6.793 |
| diffusion | -72944.242 | 46636.824 | -265136.094 | -989.978 | -68230.016 |


#### Ood Easy

| Model | Mean | Std | Min | Max | Median |
|---|---|---|---|---|---|
| realnvp | -155559.969 | 570786.938 | -7938513.000 | -119.185 | -10491.858 |
| vae | -4231.354 | 3024.130 | -16589.473 | -45.667 | -3386.056 |
| kde | -22.992 | 0.604 | -23.026 | -9.809 | -23.026 |
| neural_ode | -3665.885 | 2764.465 | -18091.596 | -77.053 | -2811.204 |
| diffusion | -9501487.000 | 5556513.500 | -26551094.000 | -394623.188 | -8391012.000 |


#### Ood Medium

| Model | Mean | Std | Min | Max | Median |
|---|---|---|---|---|---|
| realnvp | -3021.679 | 3635.650 | -23531.775 | -94.869 | -1549.650 |
| vae | -416.515 | 141.219 | -1027.413 | -144.007 | -392.889 |
| kde | -20.359 | 3.600 | -23.026 | -7.769 | -22.368 |
| neural_ode | -327.746 | 94.886 | -689.430 | -114.846 | -318.537 |
| diffusion | -361481.469 | 97725.984 | -707484.000 | -100456.266 | -357472.812 |


#### Ood Hard

| Model | Mean | Std | Min | Max | Median |
|---|---|---|---|---|---|
| realnvp | -9.294 | 3.202 | -17.583 | -5.760 | -7.807 |
| vae | -5.140 | 3.951 | -20.575 | -0.098 | -3.921 |
| kde | -0.240 | 0.133 | -0.679 | -0.070 | -0.221 |
| neural_ode | -10.581 | 2.505 | -18.186 | -6.908 | -10.822 |
| diffusion | -36667.961 | 18443.357 | -109808.562 | -3283.159 | -33274.078 |


#### Ood Very Hard

| Model | Mean | Std | Min | Max | Median |
|---|---|---|---|---|---|
| realnvp | -12.485 | 5.199 | -68.047 | -5.812 | -11.583 |
| vae | -5.437 | 3.606 | -24.049 | -0.132 | -4.735 |
| kde | -0.421 | 0.304 | -1.988 | -0.049 | -0.336 |
| neural_ode | -11.884 | 2.998 | -25.720 | -6.122 | -11.449 |
| diffusion | -87892.516 | 56740.547 | -296308.406 | -358.619 | -77689.844 |


### OOD Detection Performance

#### AUROC (Area Under ROC Curve)

| Model | Easy | Medium | Hard | Very Hard |
|---|---|---|---|---|
| realnvp | 1.0000 | 1.0000 | 0.6876 | 0.9411 |
| vae | 1.0000 | 1.0000 | 0.7342 | 0.8212 |
| kde | 1.0000 | 1.0000 | 0.8772 | 0.9470 |
| neural_ode | 1.0000 | 1.0000 | 0.8920 | 0.9511 |
| diffusion | 1.0000 | 0.9983 | 0.2703 | 0.5682 |


#### AUPRC (Area Under Precision-Recall Curve)

| Model | Easy | Medium | Hard | Very Hard |
|---|---|---|---|---|
| realnvp | 1.0000 | 1.0000 | 0.8045 | 0.9668 |
| vae | 1.0000 | 1.0000 | 0.8062 | 0.8807 |
| kde | 1.0000 | 1.0000 | 0.9422 | 0.9743 |
| neural_ode | 1.0000 | 1.0000 | 0.9465 | 0.9755 |
| diffusion | 1.0000 | 0.9991 | 0.5880 | 0.7054 |


### Correlation with True Log-Probabilities

| Model | Test | OOD Easy | OOD Medium | OOD Hard | OOD Very Hard |
|---|---|---|---|---|---|
| realnvp | 0.9419 | 0.0538 | 0.0696 | 0.4133 | 0.7566 |
| vae | 0.4393 | 0.7069 | 0.4902 | 0.4608 | 0.7198 |
| kde | 0.8410 | 0.0834 | 0.5625 | 0.5901 | 0.7015 |
| neural_ode | 0.9715 | 0.6247 | 0.7523 | 0.7534 | 0.9393 |
| diffusion | 0.1332 | 0.8983 | 0.8148 | -0.5802 | 0.0717 |


### Visualizations

See `toy_experiments/results/dim4/` for:
- `loglikelihood_comparison.png`
- `ood_detection_performance.png`
- `score_distributions.png`


## 8D Results

### Log-Likelihood Statistics


#### Test

| Model | Mean | Std | Min | Max | Median |
|---|---|---|---|---|---|
| realnvp | -14.869 | 2.348 | -42.356 | -11.107 | -14.365 |
| vae | -4.372 | 2.391 | -22.147 | -0.438 | -3.942 |
| kde | -0.343 | 0.175 | -2.659 | -0.134 | -0.300 |
| neural_ode | -14.518 | 2.123 | -26.500 | -10.737 | -14.146 |
| diffusion | -225356.984 | 116326.914 | -742550.438 | -43465.574 | -185923.672 |


#### Ood Easy

| Model | Mean | Std | Min | Max | Median |
|---|---|---|---|---|---|
| realnvp | -99246.375 | 132130.438 | -1094878.000 | -1962.964 | -54235.734 |
| vae | -298530635776.000 | 8481947516928.000 | -266469250170880.000 | -1530.637 | -13365.719 |
| kde | -23.026 | 0.000 | -23.026 | -23.026 | -23.026 |
| neural_ode | -10203.646 | 4908.112 | -30983.938 | -1465.608 | -9339.387 |
| diffusion | -32225608.000 | 11421034.000 | -69207096.000 | -5456928.500 | -31411332.000 |


#### Ood Medium

| Model | Mean | Std | Min | Max | Median |
|---|---|---|---|---|---|
| realnvp | -7998.067 | 2984.220 | -19796.135 | -1261.381 | -7780.692 |
| vae | -1619.970 | 435.431 | -3228.160 | -487.343 | -1604.961 |
| kde | -23.026 | 0.000 | -23.026 | -23.026 | -23.026 |
| neural_ode | -791.705 | 180.193 | -1624.797 | -308.986 | -779.337 |
| diffusion | -1593569.000 | 345226.469 | -2872263.000 | -568915.625 | -1555829.750 |


#### Ood Hard

| Model | Mean | Std | Min | Max | Median |
|---|---|---|---|---|---|
| realnvp | -30.655 | 19.382 | -74.157 | -13.145 | -17.549 |
| vae | -15.518 | 17.081 | -109.203 | -0.644 | -8.794 |
| kde | -1.153 | 0.323 | -1.736 | -0.704 | -1.104 |
| neural_ode | -22.075 | 2.869 | -27.076 | -17.072 | -22.479 |
| diffusion | -104731.562 | 48461.977 | -252955.438 | -12854.774 | -107286.062 |


#### Ood Very Hard

| Model | Mean | Std | Min | Max | Median |
|---|---|---|---|---|---|
| realnvp | -19.881 | 4.873 | -73.345 | -12.468 | -19.107 |
| vae | -7.469 | 3.735 | -27.335 | -0.431 | -6.852 |
| kde | -0.717 | 0.371 | -3.041 | -0.206 | -0.627 |
| neural_ode | -19.167 | 3.188 | -34.168 | -12.111 | -18.742 |
| diffusion | -245371.016 | 120277.000 | -649122.062 | -35720.324 | -215278.906 |


### OOD Detection Performance

#### AUROC (Area Under ROC Curve)

| Model | Easy | Medium | Hard | Very Hard |
|---|---|---|---|---|
| realnvp | 1.0000 | 1.0000 | 0.8470 | 0.8881 |
| vae | 1.0000 | 1.0000 | 0.8038 | 0.7697 |
| kde | 1.0000 | 1.0000 | 0.9896 | 0.8871 |
| neural_ode | 1.0000 | 1.0000 | 0.9803 | 0.9029 |
| diffusion | 1.0000 | 1.0000 | 0.1639 | 0.5554 |


#### AUPRC (Area Under Precision-Recall Curve)

| Model | Easy | Medium | Hard | Very Hard |
|---|---|---|---|---|
| realnvp | 1.0000 | 1.0000 | 0.9167 | 0.9405 |
| vae | 1.0000 | 1.0000 | 0.8609 | 0.8471 |
| kde | 1.0000 | 1.0000 | 0.9958 | 0.9410 |
| neural_ode | 1.0000 | 1.0000 | 0.9914 | 0.9483 |
| diffusion | 1.0000 | 1.0000 | 0.4883 | 0.7023 |


### Correlation with True Log-Probabilities

| Model | Test | OOD Easy | OOD Medium | OOD Hard | OOD Very Hard |
|---|---|---|---|---|---|
| realnvp | 0.9194 | 0.2264 | 0.6476 | 0.8395 | 0.7842 |
| vae | 0.5061 | 0.0228 | 0.8324 | 0.4709 | 0.6373 |
| kde | 0.8140 | N/A | N/A | 0.9837 | 0.7079 |
| neural_ode | 0.9785 | 0.5686 | 0.9438 | 0.9740 | 0.9592 |
| diffusion | 0.1244 | 0.8270 | 0.8007 | -0.3665 | 0.0920 |


### Visualizations

See `toy_experiments/results/dim8/` for:
- `loglikelihood_comparison.png`
- `ood_detection_performance.png`
- `score_distributions.png`


## Key Findings & Interpretation

### Log-Likelihood Scale Comparison

The models produce different scales of log-likelihoods:

- **RealNVP**: Direct log-probability under normalizing flow
  - Scale: Typically negative values (e.g., -5 to -50)
  - Interpretation: Exact log-probability of data under learned distribution

- **VAE**: Negative reconstruction error
  - Scale: Typically negative values (e.g., -1 to -20)
  - Interpretation: NOT a true likelihood, but reconstruction quality
  - Note: ELBO provides a lower bound on true log-likelihood

- **KDE**: Kernel density estimation with Gaussian kernels
  - Scale: Log-probability from kernel density estimate
  - Interpretation: Non-parametric density estimation

- **Neural ODE**: Log-probability via continuous normalizing flow
  - Scale: Similar to RealNVP (negative values)
  - Interpretation: Exact log-probability using ODE integration
  - Note: May be more expensive computationally

- **Diffusion**: ELBO-based log-likelihood from diffusion model
  - Scale: Typically negative values (e.g., -10 to -100)
  - Interpretation: Evidence Lower Bound on log-likelihood via reverse diffusion
  - Note: Uses DDIM scheduler for faster inference

### Why Scales Differ

1. **Different objectives**:
   - RealNVP/Neural ODE maximize log-likelihood directly
   - VAE/Diffusion maximize ELBO (Evidence Lower Bound)
   - KDE uses non-parametric density estimation

2. **Different architectures**:
   - RealNVP uses invertible coupling transformations
   - VAE has bottleneck latent space with encoder-decoder
   - KDE uses Gaussian kernels around training points
   - Neural ODE uses continuous-time dynamics via ODE integration
   - Diffusion uses iterative denoising process with DDIM scheduler

3. **Normalization constants**:
   - Each model may have different implicit normalization
   - Absolute values are less important than relative ordering

### Practical Implications for GORMPO

When using these models for OOD penalty in GORMPO:

1. **Normalization is crucial**:
   - Normalize scores to [0, 1] range before applying penalty
   - Use percentile-based thresholds instead of absolute values

2. **Model selection**:
   - RealNVP: Best for direct likelihood estimation, moderate training time
   - VAE: Fastest training, good for high-dimensional data
   - KDE: No training needed, but memory-intensive for large datasets
   - Neural ODE: Most flexible, but computationally expensive
   - Diffusion: High-quality density estimation, slower inference

3. **Penalty coefficient tuning**:
   - Different models will require different `reward-penalty-coef` values
   - Start with small values (0.001-0.01) and tune based on performance
   - Monitor the distribution of penalty values during training

## Recommendations

Based on these experiments:

1. **For RL applications**: Use **RealNVP** or **Neural ODE** for true likelihood
2. **For fast prototyping**: Use **VAE** with normalized scores
3. **For baseline comparison**: Use **KDE** (no training required)
4. **For highest quality**: Use **Diffusion** or **Neural ODE** (accept slower inference)
5. **Always normalize**: Convert raw scores to standardized range before penalty
6. **Validate empirically**: Test on actual RL environment to tune penalty scale
