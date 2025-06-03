#!/usr/bin/env python3

"""
Generate synthetic BOLD responses directly without using the Balloon-Windkessel model.
This will help verify that we can at least display proper visualizations.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gamma

# Time settings
duration = 30  # seconds
dt = 0.1  # time step
time = np.arange(0, duration, dt)

# Create stimulus patterns
impulse_stimulus = np.zeros_like(time)
impulse_stimulus[50] = 1.0  # single impulse at t=5s

block_stimulus = np.zeros_like(time)
block_stimulus[50:150] = 1.0  # block from t=5s to t=15s

complex_stimulus = np.zeros_like(time)
complex_stimulus[50] = 1.0  # First stimulus at t=5s
complex_stimulus[150] = 1.0  # Second stimulus at t=15s
complex_stimulus[200:210] = 0.8  # Sustained stimulus from t=20s to t=21s

# Function to generate canonical HRF
def canonical_hrf(t, peak_time=6.0, undershoot_time=16.0, peak_disp=1.0, undershoot_disp=1.0, 
                 peak_amp=1.0, undershoot_amp=0.15):
    """
    Generate a canonical two-gamma HRF
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

# Generate HRF kernels with different shapes for different regions
t_hrf = np.arange(0, 25, dt)  # 25 seconds is enough for the HRF to return to baseline
hrf_visual = canonical_hrf(t_hrf, peak_time=5.0, undershoot_time=15.0, peak_amp=1.0)
hrf_motor = canonical_hrf(t_hrf, peak_time=5.5, undershoot_time=14.0, peak_amp=0.9)
hrf_pfc = canonical_hrf(t_hrf, peak_time=6.0, undershoot_time=16.0, peak_amp=0.8)
hrf_dmn = canonical_hrf(t_hrf, peak_time=7.0, undershoot_time=17.0, peak_amp=0.7)

# Function to convolve stimulus with HRF
def convolve_stimulus_with_hrf(stimulus, hrf):
    bold = np.convolve(stimulus, hrf)[:len(stimulus)]
    return bold * 3.0  # Scale to ~3% signal change

# Generate BOLD responses
impulse_bold = convolve_stimulus_with_hrf(impulse_stimulus, hrf_visual)
block_bold = convolve_stimulus_with_hrf(block_stimulus, hrf_visual)

# Generate region-specific responses to complex stimulus
bold_visual = convolve_stimulus_with_hrf(complex_stimulus, hrf_visual)
bold_motor = convolve_stimulus_with_hrf(complex_stimulus, hrf_motor)
bold_pfc = convolve_stimulus_with_hrf(complex_stimulus, hrf_pfc)
bold_dmn = convolve_stimulus_with_hrf(complex_stimulus, hrf_dmn)

# Save test images

# Test 1: Basic HRF
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
plt.savefig('synthetic_basic_hrf.png')
plt.close()

# Test 2: Block design
plt.figure(figsize=(12, 6))
plt.subplot(211)
plt.plot(time, block_stimulus)
plt.title('Neural Stimulus (Block)')
plt.xlabel('Time (s)')
plt.ylabel('Activity')

plt.subplot(212)
plt.plot(time, block_bold)
plt.title('BOLD Response')
plt.xlabel('Time (s)')
plt.ylabel('BOLD Signal (%)')
plt.ylim(-1, 4)
plt.grid(True)
plt.tight_layout()
plt.savefig('synthetic_block_hrf.png')
plt.close()

# Test 3: Different HRF shapes
plt.figure(figsize=(10, 6))
plt.plot(t_hrf, hrf_visual, label='Visual Cortex')
plt.plot(t_hrf, hrf_motor, label='Motor Cortex')
plt.plot(t_hrf, hrf_pfc, label='Prefrontal Cortex')
plt.plot(t_hrf, hrf_dmn, label='Default Mode Network')
plt.title('Region-Specific Hemodynamic Response Functions')
plt.xlabel('Time (s)')
plt.ylabel('Response')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('synthetic_hrf_shapes.png')
plt.close()

# Test 4: Regional variability
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
plt.savefig('synthetic_regional_variability.png')
plt.close()

print("Synthetic BOLD response generation complete.")
print("Results saved as 'synthetic_*.png' files.") 