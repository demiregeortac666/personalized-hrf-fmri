#!/usr/bin/env python3

"""
Verification script for the fixed Balloon-Windkessel model.
This script tests the model with different stimulus patterns and plots the results.
"""

import numpy as np
import matplotlib.pyplot as plt
from src.balloon_windkessel import BalloonWindkesselModel, PersonalizedHRF

# Time settings
duration = 30  # seconds
dt = 0.1  # time step
time = np.arange(0, duration, dt)

# Create different stimulus patterns
impulse_stimulus = np.zeros_like(time)
impulse_stimulus[50] = 1.0  # single impulse at t=5s

block_stimulus = np.zeros_like(time)
block_stimulus[50:150] = 1.0  # block from t=5s to t=15s

complex_stimulus = np.zeros_like(time)
complex_stimulus[50] = 1.0  # First stimulus at t=5s
complex_stimulus[150] = 1.0  # Second stimulus at t=15s
complex_stimulus[200:210] = 0.8  # Sustained stimulus from t=20s to t=21s

# Create model with default parameters
model = BalloonWindkesselModel(epsilon=1.0)  # High epsilon for stronger response

# Test 1: Impulse response
print("\n=== Testing impulse response ===")
t_span = (0, duration)
sim_time_impulse, bold_impulse, states_impulse = model.simulate(impulse_stimulus, t_span, dt)
print(f"Impulse BOLD peak: {np.max(bold_impulse):.4f}%")
print(f"Impulse BOLD min: {np.min(bold_impulse):.4f}%")

# Test 2: Block response
print("\n=== Testing block response ===")
sim_time_block, bold_block, states_block = model.simulate(block_stimulus, t_span, dt)
print(f"Block BOLD peak: {np.max(bold_block):.4f}%")
print(f"Block BOLD min: {np.min(bold_block):.4f}%")

# Test 3: Complex response
print("\n=== Testing complex response ===")
sim_time_complex, bold_complex, states_complex = model.simulate(complex_stimulus, t_span, dt)
print(f"Complex BOLD peak: {np.max(bold_complex):.4f}%")
print(f"Complex BOLD min: {np.min(bold_complex):.4f}%")

# Test 4: Regional variability
# Create personalized HRF model
phrf = PersonalizedHRF()

# Define different regions
regions = {
    'Visual Cortex': {'tau_0': 0.65, 'alpha': 0.25, 'E_0': 0.3, 'epsilon': 1.0},
    'Motor Cortex': {'tau_0': 0.9, 'alpha': 0.3, 'E_0': 0.35, 'epsilon': 0.9},
    'Prefrontal Cortex': {'tau_0': 1.1, 'alpha': 0.35, 'E_0': 0.28, 'epsilon': 0.8},
    'Default Mode Network': {'tau_0': 1.3, 'alpha': 0.4, 'E_0': 0.25, 'epsilon': 0.7}
}

# Add regions to model
for region, params in regions.items():
    phrf.set_region_params(region, **params)

# Simulate regional responses
region_results = {}
print("\n=== Testing regional variability ===")
for region in regions:
    sim_time, bold, _ = phrf.simulate_region(region, complex_stimulus, t_span, dt)
    region_results[region] = bold
    print(f"{region} BOLD peak: {np.max(bold):.4f}%")
    print(f"{region} BOLD min: {np.min(bold):.4f}%")

# Plot results
# 1. Impulse response
plt.figure(figsize=(12, 6))
plt.subplot(211)
plt.plot(time, impulse_stimulus)
plt.title('Neural Stimulus (Impulse)')
plt.xlabel('Time (s)')
plt.ylabel('Activity')

plt.subplot(212)
plt.plot(sim_time_impulse, bold_impulse)
plt.title('BOLD Response')
plt.xlabel('Time (s)')
plt.ylabel('BOLD Signal (%)')
plt.grid(True)
plt.tight_layout()
plt.savefig('balloon_impulse_response.png')
plt.close()

# 2. Block response
plt.figure(figsize=(12, 6))
plt.subplot(211)
plt.plot(time, block_stimulus)
plt.title('Neural Stimulus (Block)')
plt.xlabel('Time (s)')
plt.ylabel('Activity')

plt.subplot(212)
plt.plot(sim_time_block, bold_block)
plt.title('BOLD Response')
plt.xlabel('Time (s)')
plt.ylabel('BOLD Signal (%)')
plt.grid(True)
plt.tight_layout()
plt.savefig('balloon_block_response.png')
plt.close()

# 3. Complex response
plt.figure(figsize=(12, 6))
plt.subplot(211)
plt.plot(time, complex_stimulus)
plt.title('Neural Stimulus (Complex)')
plt.xlabel('Time (s)')
plt.ylabel('Activity')

plt.subplot(212)
plt.plot(sim_time_complex, bold_complex)
plt.title('BOLD Response')
plt.xlabel('Time (s)')
plt.ylabel('BOLD Signal (%)')
plt.grid(True)
plt.tight_layout()
plt.savefig('balloon_complex_response.png')
plt.close()

# 4. Regional variability
plt.figure(figsize=(12, 8))
plt.subplot(211)
plt.plot(time, complex_stimulus)
plt.title('Neural Stimulus (Complex)')
plt.xlabel('Time (s)')
plt.ylabel('Activity')

plt.subplot(212)
for region, bold in region_results.items():
    plt.plot(time, bold, label=region)
plt.title('Region-Specific BOLD Responses')
plt.xlabel('Time (s)')
plt.ylabel('BOLD Signal (%)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('balloon_regional_variability.png')
plt.close()

print("\nBalloon-Windkessel model verification complete.")
print("Results saved as 'balloon_*.png' files.") 