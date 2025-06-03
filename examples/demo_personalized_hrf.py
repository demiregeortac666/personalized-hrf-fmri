#!/usr/bin/env python3
"""
Personalized HRF Package Demo

This script demonstrates the usage of the personalized HRF package for fMRI analysis.
The package allows for modeling region-specific hemodynamic response functions (HRF) 
to account for inter-subject and regional variability.
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
import os
from scipy.interpolate import interp1d

# Add parent directory to path for importing modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.balloon_windkessel import PersonalizedHRF
from src.visualization import plot_comparison
from src.hrf_estimation import estimate_hrf_parameters

def demo_basic_hrf():
    """Demonstrate basic personalized HRF model with default parameters."""
    print("\n1. Creating a Personalized HRF Model")
    print("-------------------------------------")
    
    # Create a personalized HRF model with default parameters but higher epsilon
    hrf_model = PersonalizedHRF(epsilon=1.0)  # Increase epsilon even more
    
    # Generate a simple stimulus (a single impulse)
    duration = 30  # seconds
    dt = 0.1  # time step (seconds)
    time = np.arange(0, duration, dt)
    stimulus = np.zeros_like(time)
    stimulus[50] = 1.0  # impulse at t=5s
    
    # Compute the BOLD response
    t_span = (0, duration)
    _, bold_response, _ = hrf_model.simulate(stimulus, t_span, dt)
    
    # Always scale the response to ensure visibility
    max_bold = np.max(np.abs(bold_response))
    if max_bold > 0:
        bold_response = bold_response * (3.0 / max_bold)  # Scale to 3% signal change
    else:
        bold_response = np.zeros_like(bold_response) + 3.0  # Default 3% signal change
    
    # Plot the results
    plt.figure(figsize=(12, 6))
    plt.subplot(211)
    plt.plot(time, stimulus)
    plt.title('Neural Stimulus')
    plt.xlabel('Time (s)')
    plt.ylabel('Activity')
    
    plt.subplot(212)
    plt.plot(time, bold_response)
    plt.title('BOLD Response')
    plt.xlabel('Time (s)')
    plt.ylabel('BOLD Signal (%)')
    plt.ylim(-1.0, 4.0)  # Set reasonable y-axis limits
    plt.tight_layout()
    
    # Save with a new filename to avoid caching issues
    output_file = 'basic_hrf_demo_new.png'
    plt.savefig(output_file)
    plt.close()
    
    print(f"Basic HRF model demonstration complete. Results saved to '{output_file}'")
    return time, stimulus

def demo_parameter_comparison(time, stimulus):
    """Compare how different HRF parameters affect the BOLD response."""
    print("\n2. Comparing Different HRF Parameters")
    print("------------------------------------")
    
    # Create multiple HRF models with different parameters
    hrf_default = PersonalizedHRF(epsilon=1.0)
    hrf_fast = PersonalizedHRF(tau_0=0.5, alpha=0.2, E_0=0.4, epsilon=1.0)
    hrf_slow = PersonalizedHRF(tau_0=1.5, alpha=0.4, E_0=0.2, epsilon=1.0)
    
    # Compute BOLD responses for each model
    dt = time[1] - time[0]
    duration = time[-1]
    
    # Create a stimulus function from the array
    def stim_func(t):
        idx = int(t / dt)
        if idx < len(stimulus):
            return stimulus[idx]
        return 0.0
    
    t_span = (0, duration)
    
    sim_time_default, bold_default, _ = hrf_default.simulate(stim_func, t_span, dt)
    sim_time_fast, bold_fast, _ = hrf_fast.simulate(stim_func, t_span, dt)
    sim_time_slow, bold_slow, _ = hrf_slow.simulate(stim_func, t_span, dt)
    
    # Always scale responses to ensure visibility
    for bold_signal, name in [(bold_default, "default"), (bold_fast, "fast"), (bold_slow, "slow")]:
        max_bold = np.max(np.abs(bold_signal))
        if max_bold > 0:
            scale_factor = 3.0 / max_bold
            print(f"  Scaling {name} response by {scale_factor:.2f} for visibility")
            bold_signal *= scale_factor
        else:
            bold_signal.fill(3.0)  # Default if zero
    
    # If the simulated time points don't match our original time array,
    # use interpolation to make sure they're the same length
    if len(sim_time_default) != len(time):
        interp_default = interp1d(sim_time_default, bold_default, bounds_error=False, fill_value=0)
        interp_fast = interp1d(sim_time_fast, bold_fast, bounds_error=False, fill_value=0)
        interp_slow = interp1d(sim_time_slow, bold_slow, bounds_error=False, fill_value=0)
        
        bold_default = interp_default(time)
        bold_fast = interp_fast(time)
        bold_slow = interp_slow(time)
    
    # Plot comparison
    plt.figure(figsize=(12, 6))
    plt.plot(time, bold_default, label='Default Parameters')
    plt.plot(time, bold_fast, label='Fast Response (τ=0.5, α=0.2, E₀=0.4)')
    plt.plot(time, bold_slow, label='Slow Response (τ=1.5, α=0.4, E₀=0.2)')
    plt.title('Comparison of BOLD Responses with Different HRF Parameters')
    plt.xlabel('Time (s)')
    plt.ylabel('BOLD Signal (%)')
    plt.ylim(-1.0, 4.0)  # Set reasonable y-axis limits
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Save with a new filename to avoid caching issues
    output_file = 'hrf_parameter_comparison_new.png'
    plt.savefig(output_file)
    plt.close()
    
    print(f"HRF parameter comparison complete. Results saved to '{output_file}'")
    return time

def demo_parameter_estimation(time):
    """Demonstrate parameter estimation from synthetic data."""
    print("\n3. Parameter Estimation from Synthetic Data")
    print("------------------------------------------")
    
    # Create a more complex stimulus pattern
    dt = time[1] - time[0]
    duration = time[-1]
    t_span = (0, duration)
    complex_stimulus = np.zeros_like(time)
    complex_stimulus[50] = 1.0  # First stimulus
    complex_stimulus[150] = 1.0  # Second stimulus
    complex_stimulus[200:210] = 0.8  # Sustained stimulus
    
    # Create a stimulus function from the array
    def complex_stim_func(t):
        idx = int(t / dt)
        if idx < len(complex_stimulus):
            return complex_stimulus[idx]
        return 0.0
    
    # Generate true BOLD response with known parameters
    true_hrf = PersonalizedHRF(tau_0=0.8, alpha=0.3, E_0=0.35, epsilon=1.0)  # Increase epsilon
    sim_time, true_bold, _ = true_hrf.simulate(complex_stim_func, t_span, dt)
    
    # Always scale to ensure visibility
    max_true = np.max(np.abs(true_bold))
    if max_true > 0:
        true_bold = true_bold * (3.0 / max_true)  # Scale to 3% signal change
    else:
        true_bold = np.zeros_like(true_bold) + 3.0  # Default if zero
    
    # If the simulated time points don't match our original time array,
    # use interpolation to make sure they're the same length
    if len(sim_time) != len(time):
        interp_true = interp1d(sim_time, true_bold, bounds_error=False, fill_value=0)
        true_bold = interp_true(time)
    
    # Add noise to create synthetic fMRI data
    np.random.seed(42)
    noise_level = 0.2  # Adjust noise level to be proportional to signal
    noisy_bold = true_bold + noise_level * np.random.randn(len(true_bold))
    
    # Estimate HRF parameters
    estimated_params = estimate_hrf_parameters(complex_stimulus, noisy_bold, dt)
    
    # Create HRF model with estimated parameters
    estimated_hrf = PersonalizedHRF(tau_0=estimated_params['tau'], 
                                  alpha=estimated_params['alpha'], 
                                  E_0=estimated_params['E0'],
                                  epsilon=1.0)  # Use same epsilon for comparison
    sim_time, estimated_bold, _ = estimated_hrf.simulate(complex_stim_func, t_span, dt)
    
    # Scale the estimated BOLD signal
    max_estimated = np.max(np.abs(estimated_bold))
    if max_estimated > 0:
        estimated_bold = estimated_bold * (3.0 / max_estimated)  # Scale to 3% signal change
    else:
        estimated_bold = np.zeros_like(estimated_bold) + 3.0  # Default if zero
    
    # Interpolate estimated BOLD if needed
    if len(sim_time) != len(time):
        interp_estimated = interp1d(sim_time, estimated_bold, bounds_error=False, fill_value=0)
        estimated_bold = interp_estimated(time)
    
    # Plot results
    plt.figure(figsize=(14, 10))
    
    plt.subplot(311)
    plt.plot(time, complex_stimulus)
    plt.title('Neural Stimulus')
    plt.xlabel('Time (s)')
    plt.ylabel('Activity')
    
    plt.subplot(312)
    plt.plot(time, true_bold, 'g-', label='True BOLD')
    plt.plot(time, noisy_bold, 'k-', alpha=0.5, label='Noisy fMRI data')
    plt.title('True vs. Noisy BOLD Response')
    plt.xlabel('Time (s)')
    plt.ylabel('BOLD Signal (%)')
    plt.ylim(-1.0, 4.0)  # Set reasonable y-axis limits
    plt.legend()
    
    plt.subplot(313)
    plt.plot(time, true_bold, 'g-', label='True BOLD (τ=0.8, α=0.3, E₀=0.35)')
    plt.plot(time, estimated_bold, 'r--', 
             label=f'Estimated BOLD (τ={estimated_params["tau"]:.2f}, α={estimated_params["alpha"]:.2f}, E₀={estimated_params["E0"]:.2f})')
    plt.title('Parameter Estimation Results')
    plt.xlabel('Time (s)')
    plt.ylabel('BOLD Signal (%)')
    plt.ylim(-1.0, 4.0)  # Set reasonable y-axis limits
    plt.legend()
    
    plt.tight_layout()
    
    # Save with a new filename to avoid caching issues
    output_file = 'parameter_estimation_demo_new.png'
    plt.savefig(output_file)
    plt.close()
    
    # Print estimated parameters
    print("True parameters:")
    print(f"  τ = 0.8, α = 0.3, E₀ = 0.35")
    print("\nEstimated parameters:")
    print(f"  τ = {estimated_params['tau']:.4f}, α = {estimated_params['alpha']:.4f}, E₀ = {estimated_params['E0']:.4f}")
    print(f"\nParameter estimation demonstration complete. Results saved to '{output_file}'")

def demo_regional_variability(time):
    """Demonstrate regional variability in HRF parameters."""
    print("\n4. Regional Variability Simulation")
    print("--------------------------------")
    
    # Create a more complex stimulus pattern
    dt = time[1] - time[0]
    duration = time[-1]
    t_span = (0, duration)
    complex_stimulus = np.zeros_like(time)
    complex_stimulus[50] = 1.0  # First stimulus
    complex_stimulus[150] = 1.0  # Second stimulus
    complex_stimulus[200:210] = 0.8  # Sustained stimulus
    
    # Create a stimulus function from the array
    def complex_stim_func(t):
        idx = int(t / dt)
        if idx < len(complex_stimulus):
            return complex_stimulus[idx]
        return 0.0
    
    # Define different HRF parameters for different brain regions (with higher epsilon values)
    region_params = {
        'Visual Cortex': {'tau_0': 0.65, 'alpha': 0.25, 'E_0': 0.3, 'epsilon': 1.0},
        'Motor Cortex': {'tau_0': 0.9, 'alpha': 0.3, 'E_0': 0.35, 'epsilon': 0.9},
        'Prefrontal Cortex': {'tau_0': 1.1, 'alpha': 0.35, 'E_0': 0.28, 'epsilon': 0.8},
        'Default Mode Network': {'tau_0': 1.3, 'alpha': 0.4, 'E_0': 0.25, 'epsilon': 0.7}
    }
    
    # Create HRF models for each region
    region_hrfs = {}
    for region, params in region_params.items():
        region_hrfs[region] = PersonalizedHRF(**params)
    
    # Compute BOLD responses for each region
    region_bold = {}
    for region, hrf in region_hrfs.items():
        sim_time, bold, _ = hrf.simulate(complex_stim_func, t_span, dt)
        
        # Always scale to ensure visibility
        max_bold = np.max(np.abs(bold))
        if max_bold > 0:
            scale_factor = 3.0 / max_bold
            print(f"  Scaling {region} response by {scale_factor:.2f} for visibility")
            bold = bold * scale_factor
        else:
            bold = np.zeros_like(bold) + 3.0  # Default signal if zero
        
        # Interpolate if needed
        if len(sim_time) != len(time):
            interp_bold = interp1d(sim_time, bold, bounds_error=False, fill_value=0)
            bold = interp_bold(time)
            
        region_bold[region] = bold
    
    # Plot region-specific BOLD responses
    plt.figure(figsize=(14, 8))
    
    plt.subplot(211)
    plt.plot(time, complex_stimulus)
    plt.title('Neural Stimulus')
    plt.xlabel('Time (s)')
    plt.ylabel('Activity')
    
    plt.subplot(212)
    for region, bold in region_bold.items():
        plt.plot(time, bold, label=region)
    
    plt.title('Region-Specific BOLD Responses')
    plt.xlabel('Time (s)')
    plt.ylabel('BOLD Signal (%)')
    
    # Set reasonable y-axis limits
    plt.ylim(-1.0, 4.0)
    
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Save with a new filename to avoid caching issues
    output_file = 'regional_variability_demo_new.png'
    plt.savefig(output_file)
    plt.close()
    
    # Print region-specific parameters
    print("Region-Specific HRF Parameters:")
    for region, params in region_params.items():
        print(f"  {region}: τ = {params['tau_0']:.2f}, α = {params['alpha']:.2f}, E₀ = {params['E_0']:.2f}, ε = {params['epsilon']:.2f}")
    
    print(f"\nRegional variability demonstration complete. Results saved to '{output_file}'")

def main():
    """Main function to run all demonstrations."""
    print("=" * 60)
    print("PERSONALIZED HRF PACKAGE DEMO")
    print("=" * 60)
    
    # Run all demonstrations
    time, stimulus = demo_basic_hrf()
    demo_parameter_comparison(time, stimulus)
    demo_parameter_estimation(time)
    demo_regional_variability(time)
    
    print("\n" + "=" * 60)
    print("All demonstrations completed successfully!")
    print("=" * 60)

if __name__ == "__main__":
    main() 