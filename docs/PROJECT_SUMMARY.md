# Personalized HRF in TheVirtualBrain - Implementation Overview

## Project Scope

The project aims to implement a personalized hemodynamic response function (HRF) for computational models of fMRI activity in TheVirtualBrain (TVB) platform. The goal is to replace the standard Balloon-Windkessel model in TVB with a region-specific, personalized HRF that better captures the inter-subject and inter-regional variability observed in real fMRI data.

## Main Components

1. **Balloon-Windkessel Model Implementation**: A flexible, parameterized implementation of the standard Balloon-Windkessel model that converts neural activity to BOLD signals.

2. **Personalized HRF Extension**: An extension of the Balloon-Windkessel model that allows for region-specific parameter settings.

3. **HRF Parameter Estimation**: Tools for estimating personalized HRF parameters from empirical fMRI data using deconvolution and model fitting techniques.

4. **TVB Integration**: Classes for integrating the personalized HRF model with TheVirtualBrain simulations.

5. **Visualization Utilities**: Tools for visualizing HRF parameters, simulation results, and model comparisons.

## Technical Details

### Balloon-Windkessel Model

The Balloon-Windkessel model is a biophysical model that describes the hemodynamic response to neural activity through a system of differential equations. The model has several key parameters:

- **epsilon (ε)**: Neuronal efficacy (how neural activity influences blood flow)
- **tau_s (τ_s)**: Signal decay time constant
- **tau_f (τ_f)**: Autoregulatory feedback time constant
- **tau_0 (τ_0)**: Transit time
- **alpha (α)**: Grubb's vessel stiffness exponent
- **E_0**: Resting oxygen extraction fraction
- **V_0**: Resting blood volume fraction

The model consists of four state variables:
- **s**: Vasodilatory signal
- **f**: Blood flow
- **v**: Blood volume
- **q**: Deoxyhemoglobin content

These state variables evolve according to coupled differential equations, with the BOLD signal being a function of blood volume and deoxyhemoglobin content.

### Personalized HRF

Our approach to personalization involves:

1. **Parameter Estimation**: Estimating region-specific HRF parameters from empirical fMRI data.
2. **Model Extension**: Extending the Balloon-Windkessel model to support region-specific parameter settings.
3. **TVB Integration**: Creating a custom BOLD monitor for TVB that uses the personalized HRF model.

### HRF Estimation Pipeline

1. Extract BOLD time series data from fMRI experiments.
2. Identify stimulus or task timing information.
3. Use deconvolution techniques to estimate HRF shapes for each brain region.
4. Fit the Balloon-Windkessel model parameters to match the estimated HRFs.
5. Generate a dictionary of region-specific parameters for TVB simulations.

## Implementation Status

### Completed

- Complete implementation of the Balloon-Windkessel model with ODE solver
- PersonalizedHRF class that extends the model with region-specific parameters
- HRF parameter estimation tools
- Integration with TVB via custom monitor
- Visualization utilities
- Demo notebook showcasing the implementation

### Future Work

- Testing with real fMRI datasets
- Optimization for large-scale brain simulations
- Integration with the official TVB codebase
- Comparative evaluation against standard models
- Development of a user-friendly interface for parameter selection

## Validation Approach

1. **Synthetic Data Testing**: Generate synthetic BOLD data with known parameters, then use our estimation tools to recover these parameters.
2. **Real Data Testing**: Apply to real fMRI datasets with different stimuli and tasks to validate model flexibility.
3. **Comparison with Standard Models**: Compare simulation results using personalized HRFs versus standard canonical HRFs.
4. **Whole-Brain Simulations**: Test integration with TVB through whole-brain network simulations.

## Dependencies

- NumPy, SciPy for numerical computations
- Matplotlib for visualization
- TVB (The Virtual Brain) platform for brain simulations

## Next Steps

1. Continue refining the implementation based on feedback
2. Test with real fMRI datasets from EBRAINS
3. Document the codebase comprehensively
4. Prepare tutorials and examples for users

## References

1. Friston, K.J., et al. (2000). Nonlinear responses in fMRI: the Balloon model, Volterra kernels, and other hemodynamics. Neuroimage 12, 466-477.
2. Stephan, K.E., et al. (2007). Comparing hemodynamic models with DCM. Neuroimage 38, 387-401.
3. Wu, G-R., et al. (2021). rsHRF: A Toolbox for Resting-State HRF Estimation and Deconvolution. Neuroimage.
4. Sanz-Leon, P., et al. (2015). The Virtual Brain: a simulator of primate brain network dynamics. Frontiers in Neuroinformatics 9, 79. 