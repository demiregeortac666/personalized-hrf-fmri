#!/usr/bin/env python
"""
Demo script for using the personalized HRF package.

This script demonstrates:
1. Creating a standard Balloon-Windkessel model
2. Creating a personalized HRF model
3. Simulating and comparing BOLD responses

To run: python demo.py
"""

import numpy as np
import matplotlib.pyplot as plt
from src.balloon_windkessel import BalloonWindkesselModel, PersonalizedHRF
from src.visualization import plot_hrf, plot_balloon_windkessel_states


def main():
    print("Personalized HRF Demonstration")
    print("------------------------------")
    
    # 1. Standard Balloon-Windkessel Model
    print("\n1. Standard Balloon-Windkessel Model")
    model = BalloonWindkesselModel()
    
    # Create a simple stimulus function - a block design
    def neural_activity(t):
        return 1.0 if 5 <= t <= 15 else 0.0
    
    # Simulate BOLD response
    t_span = (0, 30)  # 30 seconds simulation
    time, bold, states = model.simulate(neural_activity, t_span, dt=0.1)
    
    # Plot the standard BOLD response
    plt.figure(figsize=(10, 6))
    plt.plot(time, bold, 'k-', lw=2)
    plt.axvspan(5, 15, color='gray', alpha=0.2, label='Neural Activity')
    plt.title('BOLD Response to Neural Activity (Standard Model)', fontsize=14)
    plt.xlabel('Time (s)', fontsize=12)
    plt.ylabel('BOLD Signal', fontsize=12)
    plt.legend()
    plt.grid(True)
    plt.show()
    
    # Plot the state variables
    fig = plot_balloon_windkessel_states(time, states, title="Balloon-Windkessel Model State Variables")
    plt.show()
    
    # 2. Personalized HRF Model
    print("\n2. Personalized HRF Model")
    
    # Define different parameter sets for different brain regions
    region_params = {
        'Visual Cortex': {
            'epsilon': 0.5,  # Standard neuronal efficacy
            'tau_s': 0.8,    # Standard signal decay
            'tau_f': 0.4,    # Standard feedback time
            'E_0': 0.4       # Standard oxygen extraction
        },
        'Motor Cortex': {
            'epsilon': 0.7,  # Higher neuronal efficacy
            'tau_s': 0.6,    # Faster signal decay
            'tau_f': 0.3,    # Faster feedback
            'E_0': 0.35      # Lower oxygen extraction
        },
        'Prefrontal Cortex': {
            'epsilon': 0.4,  # Lower neuronal efficacy
            'tau_s': 1.0,    # Slower signal decay
            'tau_f': 0.5,    # Slower feedback
            'E_0': 0.45      # Higher oxygen extraction
        }
    }
    
    # Create a personalized HRF model
    personalized_model = PersonalizedHRF(region_specific_params=region_params)
    
    # Simulate for each region
    region_responses = {}
    for region_id in region_params.keys():
        t, bold, states = personalized_model.simulate_region(region_id, neural_activity, t_span, dt=0.1)
        region_responses[region_id] = (t, bold, states)
    
    # Plot all responses together
    plt.figure(figsize=(12, 6))
    
    colors = ['blue', 'red', 'green']
    for i, (region_id, (t, bold, _)) in enumerate(region_responses.items()):
        plt.plot(t, bold, color=colors[i], label=region_id, lw=2)
    
    plt.axvspan(5, 15, color='gray', alpha=0.2, label='Neural Activity')
    plt.title('Region-Specific BOLD Responses', fontsize=14)
    plt.xlabel('Time (s)', fontsize=12)
    plt.ylabel('BOLD Signal', fontsize=12)
    plt.legend()
    plt.grid(True)
    plt.show()
    
    # 3. Compare different responses
    print("\n3. Region-specific differences in the hemodynamic response")
    
    # Create a figure to show parameter effects on key aspects
    fig, axs = plt.subplots(3, 1, figsize=(12, 12))
    
    # 3.1 Response amplitude
    axs[0].set_title('Differences in BOLD Response Amplitude', fontsize=14)
    for i, (region_id, (t, bold, _)) in enumerate(region_responses.items()):
        axs[0].plot(t, bold, color=colors[i], label=f"{region_id} (max: {np.max(bold):.4f})", lw=2)
    
    axs[0].axvspan(5, 15, color='gray', alpha=0.2)
    axs[0].set_xlabel('Time (s)', fontsize=12)
    axs[0].set_ylabel('BOLD Signal', fontsize=12)
    axs[0].legend()
    axs[0].grid(True)
    
    # 3.2 Response timing (time to peak)
    axs[1].set_title('Differences in BOLD Response Timing', fontsize=14)
    
    for i, (region_id, (t, bold, _)) in enumerate(region_responses.items()):
        peak_time = t[np.argmax(bold)]
        axs[1].plot(t, bold, color=colors[i], label=f"{region_id} (peak at: {peak_time:.2f}s)", lw=2)
        axs[1].axvline(peak_time, color=colors[i], linestyle='--', alpha=0.7)
    
    axs[1].axvspan(5, 15, color='gray', alpha=0.2)
    axs[1].set_xlabel('Time (s)', fontsize=12)
    axs[1].set_ylabel('BOLD Signal', fontsize=12)
    axs[1].legend()
    axs[1].grid(True)
    
    # 3.3 Response shape (normalized to same amplitude)
    axs[2].set_title('Differences in BOLD Response Shape (Normalized)', fontsize=14)
    
    for i, (region_id, (t, bold, _)) in enumerate(region_responses.items()):
        # Normalize to same peak amplitude
        normalized_bold = bold / np.max(bold)
        axs[2].plot(t, normalized_bold, color=colors[i], label=region_id, lw=2)
    
    axs[2].axvspan(5, 15, color='gray', alpha=0.2)
    axs[2].set_xlabel('Time (s)', fontsize=12)
    axs[2].set_ylabel('Normalized BOLD', fontsize=12)
    axs[2].legend()
    axs[2].grid(True)
    
    plt.tight_layout()
    plt.show()
    
    print("\nDemonstration completed. This shows how personalized HRF parameters")
    print("affect the simulated BOLD response in different brain regions.")
    print("\nKey observations:")
    print("  1. Response amplitude varies by brain region (higher neuronal efficacy -> stronger response)")
    print("  2. Time-to-peak varies (faster signal decay and feedback -> earlier peak)")
    print("  3. Response shape varies even when normalized (different undershoot patterns)")
    print("\nThese region-specific differences are important for accurate brain simulation.")


if __name__ == "__main__":
    main() 