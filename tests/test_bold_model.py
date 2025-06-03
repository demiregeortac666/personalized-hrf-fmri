#!/usr/bin/env python3

"""
Simple test script to verify BOLD signal generation.
This script creates a simple stimulus and generates the BOLD response to verify it's visible.
"""

import numpy as np
import matplotlib.pyplot as plt
from src.balloon_windkessel import PersonalizedHRF

# Create stimulus
duration = 30  # seconds
dt = 0.1  # time step
time = np.arange(0, duration, dt)
stimulus = np.zeros_like(time)
stimulus[50] = 1.0  # single impulse at t=5s

# Create model with default parameters
model = PersonalizedHRF(epsilon=1.0)  # High epsilon for stronger response

# Simulate BOLD response
t_span = (0, duration)
sim_time, bold, states = model.simulate(stimulus, t_span, dt)

# Print stats
print(f"Stimulus max: {np.max(stimulus)}")
print(f"BOLD response max: {np.max(bold)}")
print(f"BOLD response min: {np.min(bold)}")
print(f"BOLD response mean: {np.mean(bold)}")

# Plot results
plt.figure(figsize=(10, 8))

plt.subplot(211)
plt.plot(time, stimulus)
plt.title('Neural Stimulus')
plt.xlabel('Time (s)')
plt.ylabel('Activity')

plt.subplot(212)
plt.plot(sim_time, bold)
plt.title('BOLD Response')
plt.xlabel('Time (s)')
plt.ylabel('BOLD Signal (%)')
plt.grid(True)

plt.tight_layout()
plt.savefig('test_bold_signal.png')
plt.close()

print(f"Test complete. Results saved to 'test_bold_signal.png'") 