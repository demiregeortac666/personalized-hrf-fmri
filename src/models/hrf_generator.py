import numpy as np
import matplotlib.pyplot as plt
import base64
import io

# Set figure style
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({'font.size': 12})

def generate_hrf_curve():
    """Generate a canonical HRF curve image"""
    # Time axis
    t = np.linspace(0, 20, 200)
    
    # Create canonical double-gamma HRF
    a1, a2 = 6, 16        # Shape parameters
    b1, b2 = 1, 1         # Scale parameters
    c = 1/6               # Scale for the second gamma function
    
    hrf = (t**a1 * np.exp(-t/b1) / np.gamma(a1) - 
           c * t**a2 * np.exp(-t/b2) / np.gamma(a2))
    
    # Normalize
    hrf = hrf / np.max(np.abs(hrf))
    
    # Plot
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(t, hrf, 'b-', lw=3)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('BOLD Response')
    ax.set_title('Canonical Hemodynamic Response Function (HRF)')
    
    # Add annotations
    ax.annotate('Initial dip', xy=(0.5, -0.05), xytext=(1, -0.2), 
                arrowprops=dict(arrowstyle='->', color='red'))
    ax.annotate('Peak response', xy=(5, 1), xytext=(7, 0.8), 
                arrowprops=dict(arrowstyle='->', color='red'))
    ax.annotate('Post-stimulus undershoot', xy=(12, -0.2), xytext=(13, -0.4), 
                arrowprops=dict(arrowstyle='->', color='red'))
    
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 20)
    ax.set_ylim(-0.5, 1.1)
    
    # Convert to base64
    img_base64 = fig_to_base64(fig)
    plt.close(fig)
    
    return img_base64

def generate_region_comparison():
    """Generate an image comparing HRF across brain regions"""
    # Time axis
    t = np.linspace(0, 20, 200)
    
    # Base HRF function
    def hrf_model(t, a1=6, a2=16, b1=1, b2=1, c=1/6, delay=0):
        t_shifted = t - delay
        t_shifted[t_shifted < 0] = 0
        return (t_shifted**a1 * np.exp(-t_shifted/b1) / np.gamma(a1) - 
                c * t_shifted**a2 * np.exp(-t_shifted/b2) / np.gamma(a2))
    
    # Generate different HRFs for different regions
    visual_hrf = hrf_model(t, delay=0, a1=5.5, b1=0.9) 
    motor_hrf = hrf_model(t, delay=0.5, a1=6, b1=1.0)
    prefrontal_hrf = hrf_model(t, delay=1, a1=6.5, b1=1.1)
    dmn_hrf = hrf_model(t, delay=1.5, a1=7, b1=1.2)
    
    # Normalize
    visual_hrf = visual_hrf / np.max(np.abs(visual_hrf))
    motor_hrf = motor_hrf / np.max(np.abs(motor_hrf))
    prefrontal_hrf = prefrontal_hrf / np.max(np.abs(prefrontal_hrf))
    dmn_hrf = dmn_hrf / np.max(np.abs(dmn_hrf))
    
    # Plot
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.plot(t, visual_hrf, 'r-', lw=2, label='Visual Cortex (faster)')
    ax.plot(t, motor_hrf, 'g-', lw=2, label='Motor Cortex')
    ax.plot(t, prefrontal_hrf, 'b-', lw=2, label='Prefrontal Cortex')
    ax.plot(t, dmn_hrf, 'm-', lw=2, label='Default Mode Network (slower)')
    
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('BOLD Response')
    ax.set_title('Region-Specific HRF Variability')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 20)
    
    # Convert to base64
    img_base64 = fig_to_base64(fig)
    plt.close(fig)
    
    return img_base64

def generate_bw_model_diagram():
    """Generate a diagram illustrating the Balloon-Windkessel model"""
    # Create a flow chart style diagram
    fig, ax = plt.subplots(figsize=(8, 5))
    
    # Turn off axis
    ax.axis('off')
    
    # Box positions and sizes
    box_width = 0.2
    box_height = 0.15
    x_positions = [0.1, 0.3, 0.5, 0.7, 0.9]
    
    # Draw boxes and labels
    boxes = [
        {'pos': (x_positions[0], 0.5), 'label': 'Neural\nActivity (x)'},
        {'pos': (x_positions[1], 0.5), 'label': 'Signal (s)'},
        {'pos': (x_positions[2], 0.5), 'label': 'Flow (f)'},
        {'pos': (x_positions[3], 0.5), 'label': 'Volume (v)'},
        {'pos': (x_positions[4], 0.5), 'label': 'Deoxy (q)'},
        {'pos': (x_positions[4], 0.25), 'label': 'BOLD\nSignal'}
    ]
    
    # Parameters
    params = [
        {'pos': (x_positions[1]-0.05, 0.65), 'label': 'τ_s, τ_f'},
        {'pos': (x_positions[3]-0.05, 0.65), 'label': 'α, τ_0'},
        {'pos': (x_positions[4]-0.05, 0.65), 'label': 'E_0'}
    ]
    
    # Draw boxes
    for box in boxes:
        rect = plt.Rectangle(
            (box['pos'][0]-box_width/2, box['pos'][1]-box_height/2), 
            box_width, box_height, 
            facecolor='lightblue', 
            edgecolor='blue', 
            alpha=0.7
        )
        ax.add_patch(rect)
        ax.text(box['pos'][0], box['pos'][1], box['pos'][0], 
                ha='center', va='center', fontsize=12, fontweight='bold')
        
        # Add labels below boxes
        ax.text(box['pos'][0], box['pos'][1]-0.3*box_height, box['label'], 
                ha='center', va='center', fontsize=10)
    
    # Add arrows
    for i in range(len(boxes)-2):
        ax.annotate('', 
                    xy=(x_positions[i+1]-box_width/2, 0.5), 
                    xytext=(x_positions[i]+box_width/2, 0.5), 
                    arrowprops=dict(arrowstyle='->', color='black', lw=1.5))
    
    # Add final arrow to BOLD
    ax.annotate('', 
                xy=(x_positions[4], 0.25+box_height/2), 
                xytext=(x_positions[4], 0.5-box_height/2), 
                arrowprops=dict(arrowstyle='->', color='black', lw=1.5))
    
    # Add parameters
    for param in params:
        ax.text(param['pos'][0], param['pos'][1], param['label'], 
                ha='center', va='center', fontsize=10, 
                bbox=dict(facecolor='lightyellow', edgecolor='gold', alpha=0.7))
    
    # Add title
    ax.text(0.5, 0.9, 'Balloon-Windkessel Model', 
            ha='center', va='center', fontsize=14, fontweight='bold')
    
    # Convert to base64
    img_base64 = fig_to_base64(fig)
    plt.close(fig)
    
    return img_base64

def generate_parameter_map():
    """Generate a brain parameter map visualization"""
    # Create a simple brain region visualization with parameter values
    # Set up the brain regions
    regions = [
        {'name': 'Visual Cortex', 'pos': (0.3, 0.3), 'tau': 0.65, 'alpha': 0.25},
        {'name': 'Motor Cortex', 'pos': (0.5, 0.7), 'tau': 0.9, 'alpha': 0.3},
        {'name': 'Prefrontal', 'pos': (0.7, 0.5), 'tau': 1.1, 'alpha': 0.35},
        {'name': 'DMN', 'pos': (0.5, 0.4), 'tau': 1.3, 'alpha': 0.4}
    ]
    
    # Create plot
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Create a brain outline (simplified oval)
    brain = plt.Circle((0.5, 0.5), 0.4, color='lightgray', fill=True, alpha=0.2)
    ax.add_patch(brain)
    
    # Plot regions with colors based on tau values
    cmap = plt.cm.viridis
    tau_vals = [r['tau'] for r in regions]
    tau_min, tau_max = min(tau_vals), max(tau_vals)
    
    for region in regions:
        # Normalize tau to [0,1] for colormap
        normalized_tau = (region['tau'] - tau_min) / (tau_max - tau_min)
        color = cmap(normalized_tau)
        
        # Plot region
        circle = plt.Circle(region['pos'], 0.1, color=color, alpha=0.7)
        ax.add_patch(circle)
        
        # Add label
        ax.text(region['pos'][0], region['pos'][1], f"{region['name']}\nτ={region['tau']}", 
                ha='center', va='center', fontsize=8)
    
    # Add colorbar
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(tau_min, tau_max))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, label='Transit Time (τ)')
    
    # Set up axes
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title('Regional Variability in HRF Parameters', fontsize=14)
    
    # Convert to base64
    img_base64 = fig_to_base64(fig)
    plt.close(fig)
    
    return img_base64

def generate_demo_output():
    """Generate a demonstration of different parameter effects"""
    # Time axis
    t = np.linspace(0, 20, 200)
    
    # Stimulus
    stimulus = np.zeros_like(t)
    stimulus[(t > 5) & (t < 6)] = 1.0
    
    # Base HRF function
    def bold_response(t, stim, tau=1.0, alpha=0.3, delay=0):
        # Convolve stimulus with HRF
        t_hrf = np.linspace(0, 20, 200)
        t_shifted = t_hrf - delay
        t_shifted[t_shifted < 0] = 0
        hrf = (t_shifted**6 * np.exp(-t_shifted/1) - 1/6 * t_shifted**16 * np.exp(-t_shifted/1))
        hrf = hrf / np.max(hrf)
        
        # Apply parameter effects (simplified)
        hrf_modified = hrf.copy()
        
        # Earlier peak for lower tau
        peak_idx = np.argmax(hrf)
        new_peak_idx = int(peak_idx * tau)
        hrf_shifted = np.zeros_like(hrf)
        hrf_shifted[new_peak_idx:] = hrf[:len(hrf)-new_peak_idx]
        
        # Alpha affects undershoot
        undershoot_idx = np.argmin(hrf[peak_idx:]) + peak_idx
        hrf_modified[undershoot_idx:] = hrf[undershoot_idx:] * (alpha/0.3)
        
        # Final response is blend
        final_hrf = 0.7*hrf_shifted + 0.3*hrf_modified
        final_hrf = final_hrf / np.max(final_hrf)
        
        # Convolve
        bold = np.convolve(stim, final_hrf, mode='full')[:len(t)]
        return bold / np.max(bold)
    
    # Generate BOLD responses with different parameters
    default_bold = bold_response(t, stimulus, tau=1.0, alpha=0.3)
    fast_bold = bold_response(t, stimulus, tau=0.8, alpha=0.2)
    slow_bold = bold_response(t, stimulus, tau=1.2, alpha=0.4)
    
    # Plot
    fig, axs = plt.subplots(2, 1, figsize=(8, 8), gridspec_kw={'height_ratios': [1, 3]})
    
    # Plot stimulus
    axs[0].plot(t, stimulus, 'k-', lw=2)
    axs[0].set_ylabel('Stimulus')
    axs[0].set_xlabel('Time (s)')
    axs[0].set_title('Neural Stimulus')
    
    # Plot responses
    axs[1].plot(t, default_bold, 'b-', lw=2, label='Default (τ=1.0, α=0.3)')
    axs[1].plot(t, fast_bold, 'r-', lw=2, label='Fast (τ=0.8, α=0.2)')
    axs[1].plot(t, slow_bold, 'g-', lw=2, label='Slow (τ=1.2, α=0.4)')
    
    axs[1].set_xlabel('Time (s)')
    axs[1].set_ylabel('BOLD Response')
    axs[1].set_title('Effect of Different HRF Parameters')
    axs[1].legend()
    axs[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Convert to base64
    img_base64 = fig_to_base64(fig)
    plt.close(fig)
    
    return img_base64

def fig_to_base64(fig):
    """Convert a matplotlib figure to base64 string"""
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=100, bbox_inches='tight')
    buf.seek(0)
    img_str = base64.b64encode(buf.read()).decode('utf-8')
    return img_str

if __name__ == "__main__":
    # Generate all the images
    hrf_curve = generate_hrf_curve()
    region_comparison = generate_region_comparison()
    bw_model = generate_bw_model_diagram()
    parameter_map = generate_parameter_map()
    demo_output = generate_demo_output()
    
    # Print all base64 strings
    print("\nHRF CURVE IMAGE:")
    print(hrf_curve)
    
    print("\nREGION COMPARISON IMAGE:")
    print(region_comparison)
    
    print("\nBALLOON-WINDKESSEL MODEL IMAGE:")
    print(bw_model)
    
    print("\nPARAMETER MAP IMAGE:")
    print(parameter_map)
    
    print("\nDEMO OUTPUT IMAGE:")
    print(demo_output) 