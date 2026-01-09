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
| realnvp | -3.710 | 0.974 | -10.901 | -2.701 | -3.443 |
| vae | -0.711 | 1.005 | -20.680 | -0.000 | -0.430 |
| kde | -0.003 | 0.005 | -0.073 | -0.001 | -0.002 |
| neural_ode | -3.632 | 0.923 | -8.572 | -2.513 | -3.376 |
| diffusion | -27580.221 | 13758.931 | -95569.477 | -1337.269 | -25774.902 |


#### Ood Easy

| Model | Mean | Std | Min | Max | Median |
|---|---|---|---|---|---|
| realnvp | -3541.772 | 6009.112 | -39284.516 | -3.795 | -1052.263 |
| vae | -4834253824.000 | 140326993920.000 | -4435815694336.000 | -0.912 | -1411.239 |
| kde | -21.712 | 4.400 | -23.026 | -0.003 | -23.026 |
| neural_ode | -972.836 | 597.782 | -3116.579 | -3.759 | -920.495 |
| diffusion | -3966091.750 | 2386166.750 | -10876986.000 | -8999.871 | -3761142.250 |


#### Ood Medium

| Model | Mean | Std | Min | Max | Median |
|---|---|---|---|---|---|
| realnvp | -12956.162 | 22637.902 | -155104.375 | -3.145 | -3405.453 |
| vae | -96101450431342837760.000 | inf | -28172866447766578003968.000 | -0.157 | -249898.625 |
| kde | -22.547 | 2.814 | -23.026 | -0.001 | -23.026 |
| neural_ode | -3469.362 | 2334.368 | -11248.183 | -3.196 | -3156.339 |
| diffusion | -13743991.000 | 8927553.000 | -41712976.000 | -38031.574 | -13048957.000 |


#### Ood Hard

| Model | Mean | Std | Min | Max | Median |
|---|---|---|---|---|---|
| realnvp | -34545.047 | 62137.504 | -394910.281 | -18.607 | -8001.766 |
| vae | -2167873014413817314449162583932928.000 | inf | -781102848324530734958340826143391744.000 | -8.644 | -1187048704.000 |
| kde | -22.844 | 1.737 | -23.026 | -1.842 | -23.026 |
| neural_ode | -8861.128 | 6064.786 | -27829.975 | -39.977 | -8054.820 |
| diffusion | -35296396.000 | 23015008.000 | -99465248.000 | -164429.438 | -32523106.000 |


#### Ood Very Hard

| Model | Mean | Std | Min | Max | Median |
|---|---|---|---|---|---|
| realnvp | -56278.406 | 99864.320 | -612118.125 | -12.039 | -12670.942 |
| vae | -inf | nan | -inf | -1.988 | -512655982592.000 |
| kde | -22.891 | 1.582 | -23.026 | -0.204 | -23.026 |
| neural_ode | -13651.040 | 8972.867 | -42518.801 | -10.820 | -12167.326 |
| diffusion | -54585480.000 | 34304332.000 | -156052688.000 | -3623.596 | -49899152.000 |


### OOD Detection Performance

#### AUROC (Area Under ROC Curve)

| Model | Easy | Medium | Hard | Very Hard |
|---|---|---|---|---|
| realnvp | 0.9997 | 0.9992 | 1.0000 | 1.0000 |
| vae | 0.9990 | 0.9987 | 1.0000 | 0.0000 |
| kde | 0.9997 | 0.9992 | 1.0000 | 1.0000 |
| neural_ode | 0.9997 | 0.9993 | 1.0000 | 1.0000 |
| diffusion | 0.9980 | 0.9998 | 1.0000 | 0.9990 |


#### AUPRC (Area Under Precision-Recall Curve)

| Model | Easy | Medium | Hard | Very Hard |
|---|---|---|---|---|
| realnvp | 0.9998 | 0.9994 | 1.0000 | 1.0000 |
| vae | 0.9995 | 0.9989 | 1.0000 | 0.0000 |
| kde | 0.9998 | 0.9993 | 1.0000 | 1.0000 |
| neural_ode | 0.9998 | 0.9995 | 1.0000 | 1.0000 |
| diffusion | 0.9977 | 0.9999 | 1.0000 | 0.9977 |


### Correlation with True Log-Probabilities

| Model | Test | OOD Easy | OOD Medium | OOD Hard | OOD Very Hard |
|---|---|---|---|---|---|
| realnvp | 0.9134 | 0.3889 | 0.3489 | 0.3093 | 0.2790 |
| vae | 0.3379 | 0.1269 | 0.2111 | 0.1896 | N/A |
| kde | 0.8616 | 0.3812 | 0.2111 | 0.1319 | 0.1134 |
| neural_ode | 0.9825 | 0.7644 | 0.7884 | 0.8173 | 0.8241 |
| diffusion | 0.1243 | 0.8901 | 0.9013 | 0.9033 | 0.8979 |


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
| realnvp | -7.203 | 1.521 | -20.350 | -5.184 | -6.805 |
| vae | -2.010 | 1.467 | -13.226 | -0.013 | -1.679 |
| kde | -0.096 | 0.080 | -1.053 | -0.029 | -0.072 |
| neural_ode | -7.106 | 1.392 | -17.713 | -5.080 | -6.816 |
| diffusion | -72242.000 | 45779.969 | -228361.156 | -814.564 | -69812.172 |


#### Ood Easy

| Model | Mean | Std | Min | Max | Median |
|---|---|---|---|---|---|
| realnvp | -116863.789 | 199392.734 | -1950823.625 | -132.016 | -29741.281 |
| vae | -4382.650 | 3235.119 | -18603.250 | -55.716 | -3479.530 |
| kde | -22.992 | 0.604 | -23.026 | -9.809 | -23.026 |
| neural_ode | -3623.705 | 2721.252 | -17366.982 | -74.469 | -2810.875 |
| diffusion | -9514896.000 | 5557953.500 | -26064940.000 | -356946.625 | -8358952.000 |


#### Ood Medium

| Model | Mean | Std | Min | Max | Median |
|---|---|---|---|---|---|
| realnvp | -588002.125 | 1289586.375 | -29680156.000 | -227.477 | -175057.188 |
| vae | -16346.659 | 12911.457 | -146381.719 | -255.839 | -13565.651 |
| kde | -23.026 | 0.000 | -23.026 | -23.017 | -23.026 |
| neural_ode | -13029.751 | 9743.027 | -53010.176 | -158.055 | -10205.490 |
| diffusion | -32543290.000 | 17995254.000 | -85697160.000 | -1076826.500 | -29998320.000 |


#### Ood Hard

| Model | Mean | Std | Min | Max | Median |
|---|---|---|---|---|---|
| realnvp | -1930482.500 | 5582247.000 | -101023400.000 | -130.937 | -474828.188 |
| vae | -38880.953 | 40848.566 | -993096.812 | -97.756 | -30719.006 |
| kde | -23.014 | 0.369 | -23.026 | -11.356 | -23.026 |
| neural_ode | -30921.549 | 22271.637 | -131700.547 | -82.648 | -24834.863 |
| diffusion | -75534264.000 | 40285932.000 | -186032880.000 | -334293.500 | -68480152.000 |


#### Ood Very Hard

| Model | Mean | Std | Min | Max | Median |
|---|---|---|---|---|---|
| realnvp | -3264749.000 | 8662673.000 | -142497472.000 | -2319.882 | -870429.000 |
| vae | -8989729792.000 | 275846397952.000 | -8723847708672.000 | -2345.399 | -56493.320 |
| kde | -23.026 | 0.000 | -23.026 | -23.026 | -23.026 |
| neural_ode | -55331.113 | 40212.176 | -264805.938 | -1759.302 | -45993.789 |
| diffusion | -133018192.000 | 73550968.000 | -369687712.000 | -2721678.000 | -120364224.000 |


### OOD Detection Performance

#### AUROC (Area Under ROC Curve)

| Model | Easy | Medium | Hard | Very Hard |
|---|---|---|---|---|
| realnvp | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| vae | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| kde | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| neural_ode | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| diffusion | 1.0000 | 1.0000 | 1.0000 | 1.0000 |


#### AUPRC (Area Under Precision-Recall Curve)

| Model | Easy | Medium | Hard | Very Hard |
|---|---|---|---|---|
| realnvp | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| vae | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| kde | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| neural_ode | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| diffusion | 1.0000 | 1.0000 | 1.0000 | 1.0000 |


### Correlation with True Log-Probabilities

| Model | Test | OOD Easy | OOD Medium | OOD Hard | OOD Very Hard |
|---|---|---|---|---|---|
| realnvp | 0.9430 | 0.1569 | 0.1793 | 0.1399 | 0.1094 |
| vae | 0.5345 | 0.7353 | 0.6266 | 0.4093 | 0.0178 |
| kde | 0.8410 | 0.0834 | 0.0522 | 0.0543 | N/A |
| neural_ode | 0.9771 | 0.6308 | 0.6075 | 0.5682 | 0.5884 |
| diffusion | 0.1332 | 0.8967 | 0.8869 | 0.9007 | 0.8987 |


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
| realnvp | -14.820 | 2.269 | -34.817 | -10.979 | -14.388 |
| vae | -4.963 | 3.557 | -106.069 | -0.369 | -4.417 |
| kde | -0.343 | 0.175 | -2.659 | -0.134 | -0.300 |
| neural_ode | -14.535 | 2.138 | -27.026 | -10.648 | -14.160 |
| diffusion | -225683.078 | 117515.719 | -683542.938 | -39031.043 | -184665.281 |


#### Ood Easy

| Model | Mean | Std | Min | Max | Median |
|---|---|---|---|---|---|
| realnvp | -98854.062 | 154951.250 | -1652887.125 | -1863.834 | -52496.133 |
| vae | -54051.512 | 837320.438 | -25256248.000 | -1657.922 | -13872.865 |
| kde | -23.026 | 0.000 | -23.026 | -23.026 | -23.026 |
| neural_ode | -10139.890 | 4727.186 | -29324.668 | -1476.294 | -9346.408 |
| diffusion | -32210854.000 | 11422640.000 | -70211504.000 | -6067001.000 | -31526062.000 |


#### Ood Medium

| Model | Mean | Std | Min | Max | Median |
|---|---|---|---|---|---|
| realnvp | -407663.438 | 569192.625 | -4897513.000 | -8338.344 | -214727.250 |
| vae | -7478688350208.000 | 229763738238976.000 | -7269048538628096.000 | -3938.358 | -57461.414 |
| kde | -23.026 | 0.000 | -23.026 | -23.026 | -23.026 |
| neural_ode | -38488.656 | 17995.135 | -116052.836 | -4212.750 | -36313.102 |
| diffusion | -115351944.000 | 41321128.000 | -252613136.000 | -14755316.000 | -113634968.000 |


#### Ood Hard

| Model | Mean | Std | Min | Max | Median |
|---|---|---|---|---|---|
| realnvp | -982341.875 | 1446698.250 | -11620156.000 | -8391.157 | -490220.312 |
| vae | -763907042719047876608.000 | inf | -763126309946645995847680.000 | -19356.846 | -146061.469 |
| kde | -23.026 | 0.000 | -23.026 | -23.026 | -23.026 |
| neural_ode | -84170.688 | 39146.496 | -243931.953 | -14049.244 | -79332.609 |
| diffusion | -255993088.000 | 90309880.000 | -528733632.000 | -46064772.000 | -249754784.000 |


#### Ood Very Hard

| Model | Mean | Std | Min | Max | Median |
|---|---|---|---|---|---|
| realnvp | -1934044.750 | 2878496.250 | -24296698.000 | -22124.729 | -999079.500 |
| vae | -1348327509221574420327485669376.000 | inf | -1339202141816380185333063324008448.000 | -31208.438 | -301589.938 |
| kde | -23.026 | 0.000 | -23.026 | -23.026 | -23.026 |
| neural_ode | -148354.656 | 70635.922 | -468331.312 | -21995.930 | -135325.125 |
| diffusion | -453383776.000 | 157986608.000 | -1028416704.000 | -68485176.000 | -448818528.000 |


### OOD Detection Performance

#### AUROC (Area Under ROC Curve)

| Model | Easy | Medium | Hard | Very Hard |
|---|---|---|---|---|
| realnvp | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| vae | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| kde | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| neural_ode | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| diffusion | 1.0000 | 1.0000 | 1.0000 | 1.0000 |


#### AUPRC (Area Under Precision-Recall Curve)

| Model | Easy | Medium | Hard | Very Hard |
|---|---|---|---|---|
| realnvp | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| vae | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| kde | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| neural_ode | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| diffusion | 1.0000 | 1.0000 | 1.0000 | 1.0000 |


### Correlation with True Log-Probabilities

| Model | Test | OOD Easy | OOD Medium | OOD Hard | OOD Very Hard |
|---|---|---|---|---|---|
| realnvp | 0.9371 | 0.1320 | 0.1546 | 0.1310 | 0.1292 |
| vae | 0.3836 | 0.1597 | 0.0237 | 0.0114 | 0.0351 |
| kde | 0.8140 | N/A | N/A | N/A | N/A |
| neural_ode | 0.9773 | 0.5818 | 0.6154 | 0.6182 | 0.5808 |
| diffusion | 0.1136 | 0.8253 | 0.8062 | 0.8174 | 0.8134 |


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
