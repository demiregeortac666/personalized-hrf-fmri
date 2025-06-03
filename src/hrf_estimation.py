"""
Hemodynamic Response Function (HRF) estimation from empirical fMRI data.

This module provides methods for estimating personalized HRF parameters from 
empirical fMRI data using various techniques including deconvolution and model fitting.
"""

import numpy as np
from scipy import signal
import scipy.optimize as opt
from .balloon_windkessel import BalloonWindkesselModel

# Add a standalone function for use in the demo script
def estimate_hrf_parameters(stimulus, bold_signal, dt):
    """
    Estimate HRF parameters from a stimulus and BOLD signal.
    
    Parameters
    ----------
    stimulus : array-like
        Neural stimulus time course
    bold_signal : array-like
        BOLD signal time course
    dt : float
        Time step in seconds
        
    Returns
    -------
    params : dict
        Estimated HRF parameters (tau, alpha, E0)
    """
    # Find stimulus onset time
    onset_indices = np.where(np.diff(np.append(0, stimulus)) > 0)[0]
    if len(onset_indices) == 0:
        onset_indices = [np.argmax(stimulus)]  # Fallback to max if no clear onset
    
    onset_idx = onset_indices[0]
    
    # Find the peak time and amplitude relative to stimulus onset
    peak_idx = onset_idx + np.argmax(bold_signal[onset_idx:onset_idx+100])  # Look within 10s after onset
    peak_time = (peak_idx - onset_idx) * dt
    peak_amplitude = bold_signal[peak_idx]
    
    # Clip peak_time to reasonable values for tau estimation
    peak_time = max(min(peak_time, 6.0), 2.0)  # Between 2-6 seconds
    
    # Estimate tau based on peak time (more realistic relationship)
    tau = peak_time * 0.2  # Transit time is approximately 1/5 of peak time
    
    # Constrain tau to be within physiological range
    tau = max(min(tau, 2.0), 0.5)
    
    # Find the time to half peak on the falling edge
    half_peak = (peak_amplitude + bold_signal[onset_idx]) / 2
    falling_edge = bold_signal[peak_idx:]
    if len(falling_edge) > 1:
        # Find where signal crosses half-peak value
        half_peak_idx = np.argmin(np.abs(falling_edge - half_peak))
        half_peak_time = (half_peak_idx) * dt
        
        # Estimate alpha based on response shape
        if half_peak_time > 3.0:
            alpha = 0.4  # Slower response
        elif half_peak_time < 1.5:
            alpha = 0.2  # Faster response
        else:
            alpha = 0.3  # Default
    else:
        alpha = 0.3  # Default value
    
    # Estimate E0 based on response amplitude
    # Scale to realistic values (0.2-0.6)
    E0 = 0.4
    if peak_amplitude > 0.05:
        E0 = 0.35  # Higher amplitude often indicates lower baseline extraction
    elif peak_amplitude < 0.01:
        E0 = 0.45  # Lower amplitude often indicates higher baseline extraction
    
    # Return parameters in a dictionary
    return {
        'tau': tau,
        'alpha': alpha,
        'E0': E0
    }

class HRFEstimator:
    """
    Class for estimating personalized Hemodynamic Response Function (HRF) from fMRI data.
    
    This class provides methods for extracting the shape of the hemodynamic response
    from empirical fMRI data and estimating the parameters of the Balloon-Windkessel
    model that best describe the observed response.
    """
    
    def __init__(self, tr=2.0, hrf_length=20.0):
        """
        Initialize the HRF estimator.
        
        Parameters
        ----------
        tr : float, optional
            Repetition time (TR) of the fMRI data in seconds, by default 2.0
        hrf_length : float, optional
            Length of the HRF in seconds, by default 20.0
        """
        self.tr = tr
        self.hrf_length = hrf_length
        
        # Default initial Balloon-Windkessel parameters
        self.default_params = {
            'epsilon': 0.5,
            'tau_s': 0.8,
            'tau_f': 0.4,
            'tau_0': 1.0,
            'alpha': 0.32,
            'E_0': 0.4,
            'V_0': 0.04
        }
    
    def canonical_hrf(self, t):
        """
        Generate a canonical double-gamma HRF at specified time points.
        
        Parameters
        ----------
        t : array-like
            Time points at which to evaluate the HRF
            
        Returns
        -------
        hrf : array
            Canonical HRF values at the specified time points
        """
        # Constants for double-gamma HRF
        a1, a2 = 6, 16           # Shape parameters
        b1, b2 = 1, 1            # Scale parameters
        c = 1/6                  # Scale for the second gamma function
        
        # Double-gamma function
        hrf = (t**a1 * np.exp(-t/b1) / np.gamma(a1) - 
               c * t**a2 * np.exp(-t/b2) / np.gamma(a2))
        
        # Normalize
        hrf = hrf / np.sum(hrf) * 0.1  # Adjust amplitude to ~5% signal change
        
        return hrf
    
    def estimate_hrf_deconvolution(self, bold_signal, stimulus_signal, method='fft'):
        """
        Estimate the HRF using deconvolution techniques.
        
        Parameters
        ----------
        bold_signal : array-like
            BOLD signal time course
        stimulus_signal : array-like
            Binary stimulus time course (same length as BOLD signal)
        method : str, optional
            Deconvolution method ('wiener' or 'fft'), by default 'fft'
            
        Returns
        -------
        hrf : array
            Estimated HRF time course
        """
        if len(bold_signal) != len(stimulus_signal):
            raise ValueError("BOLD and stimulus signals must have the same length")
        
        # Convert signals to numpy arrays
        bold = np.array(bold_signal)
        stim = np.array(stimulus_signal)
        
        # Detrend BOLD signal
        bold = signal.detrend(bold)
        
        if method == 'wiener':
            # Wiener deconvolution - using a more direct implementation
            # as scipy.signal.wiener is actually for image filtering
            hrf_length_pts = int(self.hrf_length / self.tr)
            
            # Calculate auto-correlation of stimulus
            autocorr_stim = np.correlate(stim, stim, mode='full')
            center = len(autocorr_stim) // 2
            autocorr_stim = autocorr_stim[center:center+hrf_length_pts]
            
            # Calculate cross-correlation of BOLD and stimulus
            crosscorr = np.correlate(bold, stim, mode='full')
            center = len(crosscorr) // 2
            crosscorr = crosscorr[center:center+hrf_length_pts]
            
            # Wiener deconvolution with a small regularization parameter
            noise_power = 0.1  # Can be adjusted based on SNR
            signal_power = np.abs(np.fft.fft(autocorr_stim))**2
            
            # Create Wiener filter in frequency domain
            H = np.fft.fft(crosscorr)
            G = np.conj(np.fft.fft(autocorr_stim)) / (signal_power + noise_power)
            hrf_fft = H * G
            
            # Convert back to time domain
            hrf = np.real(np.fft.ifft(hrf_fft))
            
            # Keep only the relevant part
            hrf = hrf[:hrf_length_pts]
            
        elif method == 'fft':
            # FFT-based deconvolution
            bold_fft = np.fft.fft(bold)
            stim_fft = np.fft.fft(stim)
            
            # Avoid division by zero
            eps = 1e-10
            hrf_fft = bold_fft / (stim_fft + eps * np.max(np.abs(stim_fft)))
            
            # Inverse FFT to get the HRF
            hrf = np.real(np.fft.ifft(hrf_fft))
            
            # Extract only the beginning part (length of expected HRF)
            hrf_length_pts = int(self.hrf_length / self.tr)
            hrf = hrf[:hrf_length_pts]
            
        else:
            raise ValueError(f"Unknown deconvolution method: {method}")
        
        # Normalize HRF
        if np.max(np.abs(hrf)) > 0:
            hrf = hrf / np.max(np.abs(hrf))
        
        return hrf
    
    def fit_balloon_model(self, hrf, t=None, initial_params=None, bounds=None):
        """
        Fit the Balloon-Windkessel model parameters to a given HRF.
        
        Parameters
        ----------
        hrf : array-like
            HRF time course to fit
        t : array-like, optional
            Time points corresponding to the HRF, by default None
            (will be created based on TR and HRF length)
        initial_params : dict, optional
            Initial parameters for the model, by default None
            (will use default parameters)
        bounds : dict, optional
            Bounds for parameter optimization as dict of (min, max) tuples,
            by default None (will use predefined bounds)
            
        Returns
        -------
        params : dict
            Fitted Balloon-Windkessel model parameters
        hrf_fitted : array
            HRF predicted by the fitted model
        """
        # Create time points if not provided
        if t is None:
            t = np.arange(0, len(hrf) * self.tr, self.tr)
        
        # Use default initial parameters if not provided
        if initial_params is None:
            initial_params = self.default_params.copy()
        
        # Define bounds if not provided
        if bounds is None:
            bounds = {
                'epsilon': (0.1, 2.0),
                'tau_s': (0.1, 4.0),
                'tau_f': (0.1, 2.0),
                'tau_0': (0.5, 3.0),
                'alpha': (0.1, 0.9),
                'E_0': (0.2, 0.8),
                'V_0': (0.01, 0.1)
            }
        
        # Extract parameter names, initial values, and bounds
        param_names = list(initial_params.keys())
        x0 = [initial_params[name] for name in param_names]
        
        # Convert bounds dictionary to list of (min, max) tuples in the same order
        bounds_list = [bounds[name] for name in param_names]
        
        # Define the impulse response function to simulate
        def impulse_input(t):
            # Delta function at t=0, zero elsewhere
            return 1.0 if np.isclose(t, 0.0, atol=1e-6) else 0.0
        
        # Objective function to minimize
        def objective(params):
            # Create parameter dictionary
            param_dict = {name: val for name, val in zip(param_names, params)}
            
            # Create model with these parameters
            model = BalloonWindkesselModel(**param_dict)
            
            # Simulate response to impulse
            t_span = (0, t[-1] + self.tr)
            t_sim, bold_sim, _ = model.simulate(impulse_input, t_span, dt=self.tr/10)
            
            # Interpolate to match the time points of the target HRF
            from scipy.interpolate import interp1d
            bold_interp = interp1d(t_sim, bold_sim, bounds_error=False, fill_value=0)(t)
            
            # Normalize both signals for comparison
            hrf_norm = hrf / np.max(np.abs(hrf))
            bold_interp_norm = bold_interp / np.max(np.abs(bold_interp))
            
            # Return mean squared error
            return np.mean((hrf_norm - bold_interp_norm)**2)
        
        # Run optimization
        result = opt.minimize(
            objective, 
            x0, 
            bounds=bounds_list,
            method='L-BFGS-B',
            options={'maxiter': 100, 'disp': False}
        )
        
        # Extract optimized parameters
        optimized_params = {name: val for name, val in zip(param_names, result.x)}
        
        # Simulate HRF with optimized parameters
        model = BalloonWindkesselModel(**optimized_params)
        t_span = (0, t[-1] + self.tr)
        t_sim, bold_sim, _ = model.simulate(impulse_input, t_span, dt=self.tr/10)
        hrf_fitted = np.interp(t, t_sim, bold_sim)
        
        # Normalize fitted HRF to match the amplitude of the target HRF
        max_hrf = np.max(np.abs(hrf))
        max_fitted = np.max(np.abs(hrf_fitted))
        if max_fitted > 0:
            hrf_fitted = hrf_fitted * (max_hrf / max_fitted)
        
        return optimized_params, hrf_fitted
    
    def estimate_region_specific_hrfs(self, bold_data, stimulus_signal, regions=None):
        """
        Estimate HRF parameters for multiple brain regions.
        
        Parameters
        ----------
        bold_data : array-like
            BOLD time series with shape (time_points, regions)
        stimulus_signal : array-like
            Stimulus time course
        regions : list, optional
            List of region identifiers, by default None
            
        Returns
        -------
        region_params : dict
            Dictionary of region-specific HRF parameters
        region_hrfs : dict
            Dictionary of estimated HRFs for each region
        """
        if bold_data.ndim != 2:
            raise ValueError("BOLD data should be 2D with shape (time_points, regions)")
        
        n_timepoints, n_regions = bold_data.shape
        
        if len(stimulus_signal) != n_timepoints:
            raise ValueError("Stimulus signal length must match BOLD time points")
        
        # Create region identifiers if not provided
        if regions is None:
            regions = list(range(n_regions))
        elif len(regions) != n_regions:
            raise ValueError("Number of region identifiers must match number of regions in BOLD data")
        
        region_params = {}
        region_hrfs = {}
        
        for i, region_id in enumerate(regions):
            # Extract BOLD signal for this region
            bold_signal = bold_data[:, i]
            
            # Estimate HRF
            hrf = self.estimate_hrf_deconvolution(bold_signal, stimulus_signal)
            
            # Create time points
            t = np.arange(0, len(hrf) * self.tr, self.tr)
            
            # Fit Balloon-Windkessel model
            params, hrf_fitted = self.fit_balloon_model(hrf, t)
            
            # Store results
            region_params[region_id] = params
            region_hrfs[region_id] = (t, hrf, hrf_fitted)
        
        return region_params, region_hrfs 