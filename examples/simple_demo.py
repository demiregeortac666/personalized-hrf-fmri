#!/usr/bin/env python3

"""
Simplified demonstration of personalized HRF modeling using convolution approach.
This avoids the numerical issues in the balloon-windkessel model implementation.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gamma

def main():
    """Run the simplified HRF demo"""
    print("=" * 60)
    print("SIMPLIFIED PERSONALIZED HRF DEMO")
    print("=" * 60)
    
    # Time settings
    duration = 30  # seconds
    dt = 0.1  # time step
    time = np.arange(0, duration, dt)
    
    # Create stimulus
    print("\n1. Basic HRF Demonstration")
    print("-------------------------")
    
    impulse_stimulus = np.zeros_like(time)
    impulse_stimulus[50] = 1.0  # single impulse at t=5s
    
    # Generate HRF
    t_hrf = np.arange(0, 25, dt)  # 25 seconds is enough for the HRF to return to baseline
    hrf = canonical_hrf(t_hrf)
    
    # Convolve to get BOLD response
    impulse_bold = convolve_stimulus_with_hrf(impulse_stimulus, hrf)
    
    # Plot results
    plt.figure(figsize=(12, 6))
    plt.subplot(211)
    plt.plot(time, impulse_stimulus)
    plt.title('Neural Stimulus (Impulse)')
    plt.xlabel('Time (s)')
    plt.ylabel('Activity')
    
    plt.subplot(212)
    plt.plot(time, impulse_bold)
    plt.title('BOLD Response')
    plt.xlabel('Time (s)')
    plt.ylabel('BOLD Signal (%)')
    plt.ylim(-1, 4)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('basic_hrf_simple.png')
    plt.close()
    
    print("Basic HRF demonstration complete. Results saved to 'basic_hrf_simple.png'")
    
    # Parameter comparison
    print("\n2. HRF Parameter Comparison")
    print("-------------------------")
    
    # Generate HRFs with different parameters
    hrf_default = canonical_hrf(t_hrf, peak_time=6.0, undershoot_time=16.0, peak_disp=1.0, peak_amp=1.0)
    hrf_fast = canonical_hrf(t_hrf, peak_time=5.0, undershoot_time=14.0, peak_disp=0.8, peak_amp=1.2)
    hrf_slow = canonical_hrf(t_hrf, peak_time=7.0, undershoot_time=18.0, peak_disp=1.2, peak_amp=0.8)
    
    # Convolve to get BOLD responses
    bold_default = convolve_stimulus_with_hrf(impulse_stimulus, hrf_default)
    bold_fast = convolve_stimulus_with_hrf(impulse_stimulus, hrf_fast)
    bold_slow = convolve_stimulus_with_hrf(impulse_stimulus, hrf_slow)
    
    # Plot comparison
    plt.figure(figsize=(12, 6))
    plt.plot(time, bold_default, label='Default Parameters')
    plt.plot(time, bold_fast, label='Fast Response (earlier peak, shorter undershoot)')
    plt.plot(time, bold_slow, label='Slow Response (later peak, longer undershoot)')
    plt.title('Comparison of BOLD Responses with Different HRF Parameters')
    plt.xlabel('Time (s)')
    plt.ylabel('BOLD Signal (%)')
    plt.ylim(-1, 4)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('hrf_parameter_comparison_simple.png')
    plt.close()
    
    print("HRF parameter comparison complete. Results saved to 'hrf_parameter_comparison_simple.png'")
    
    # Regional variability demonstration
    print("\n3. Regional Variability Demonstration")
    print("---------------------------------")
    
    # Create a complex stimulus pattern
    complex_stimulus = np.zeros_like(time)
    complex_stimulus[50] = 1.0  # First stimulus at t=5s
    complex_stimulus[150] = 1.0  # Second stimulus at t=15s
    complex_stimulus[200:210] = 0.8  # Sustained stimulus from t=20s to t=21s
    
    # Define different HRF parameters for different brain regions
    # Generate HRF kernels with different shapes for different regions
    hrf_visual = canonical_hrf(t_hrf, peak_time=5.0, undershoot_time=15.0, peak_amp=1.0)
    hrf_motor = canonical_hrf(t_hrf, peak_time=5.5, undershoot_time=14.0, peak_amp=0.9)
    hrf_pfc = canonical_hrf(t_hrf, peak_time=6.0, undershoot_time=16.0, peak_amp=0.8)
    hrf_dmn = canonical_hrf(t_hrf, peak_time=7.0, undershoot_time=17.0, peak_amp=0.7)
    
    # Generate region-specific responses to complex stimulus
    bold_visual = convolve_stimulus_with_hrf(complex_stimulus, hrf_visual)
    bold_motor = convolve_stimulus_with_hrf(complex_stimulus, hrf_motor)
    bold_pfc = convolve_stimulus_with_hrf(complex_stimulus, hrf_pfc)
    bold_dmn = convolve_stimulus_with_hrf(complex_stimulus, hrf_dmn)
    
    # Plot regional variability
    plt.figure(figsize=(14, 8))
    plt.subplot(211)
    plt.plot(time, complex_stimulus)
    plt.title('Neural Stimulus (Complex)')
    plt.xlabel('Time (s)')
    plt.ylabel('Activity')
    
    plt.subplot(212)
    plt.plot(time, bold_visual, label='Visual Cortex')
    plt.plot(time, bold_motor, label='Motor Cortex')
    plt.plot(time, bold_pfc, label='Prefrontal Cortex')
    plt.plot(time, bold_dmn, label='Default Mode Network')
    plt.title('Region-Specific BOLD Responses')
    plt.xlabel('Time (s)')
    plt.ylabel('BOLD Signal (%)')
    plt.ylim(-1, 4)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('regional_variability_simple.png')
    plt.close()
    
    print("Regional variability demonstration complete. Results saved to 'regional_variability_simple.png'")
    
    print("\n" + "=" * 60)
    print("All demonstrations completed successfully!")
    print("=" * 60)

def canonical_hrf(t, peak_time=6.0, undershoot_time=16.0, peak_disp=1.0, undershoot_disp=1.0, 
                 peak_amp=1.0, undershoot_amp=0.15):
    """
    Generate a canonical two-gamma HRF
    
    Parameters
    ----------
    t : array-like
        Time points at which to evaluate the HRF
    peak_time : float
        Time to peak for the positive response
    undershoot_time : float
        Time to peak for the undershoot
    peak_disp : float
        Dispersion of positive response
    undershoot_disp : float
        Dispersion of undershoot
    peak_amp : float
        Amplitude of positive response
    undershoot_amp : float
        Amplitude of undershoot
        
    Returns
    -------
    hrf : array
        Hemodynamic response function values
    """
    # Parameters for the first gamma function (positive response)
    a1 = peak_time / peak_disp
    b1 = peak_disp
    
    # Parameters for the second gamma function (undershoot)
    a2 = undershoot_time / undershoot_disp
    b2 = undershoot_disp
    
    # Generate the HRF
    hrf = peak_amp * gamma.pdf(t, a1, scale=b1) - undershoot_amp * gamma.pdf(t, a2, scale=b2)
    
    # Normalize to peak of 1
    if np.max(hrf) > 0:
        hrf = hrf / np.max(hrf)
    
    return hrf

def convolve_stimulus_with_hrf(stimulus, hrf):
    """
    Convolve a stimulus with an HRF to generate a BOLD response
    
    Parameters
    ----------
    stimulus : array-like
        Neural stimulus time course
    hrf : array-like
        Hemodynamic response function
        
    Returns
    -------
    bold : array
        BOLD signal time course
    """
    bold = np.convolve(stimulus, hrf)[:len(stimulus)]
    return bold * 3.0  # Scale to ~3% signal change

if __name__ == "__main__":
    main() 