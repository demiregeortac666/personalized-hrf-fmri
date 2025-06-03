"""
Balloon-Windkessel model for fMRI BOLD signal estimation

This module implements the Balloon-Windkessel model which converts neural activity 
to BOLD signals. The model consists of a system of ODEs that describe the hemodynamic
response to neural activity, following Friston et al. (2000).

References:
- Friston, K.J., Mechelli, A., Turner, R., Price, C.J., 2000. Nonlinear responses in fMRI: 
  the Balloon model, Volterra kernels, and other hemodynamics. Neuroimage 12, 466-477.
- Stephan, K.E., Weiskopf, N., Drysdale, P.M., Robinson, P.A., Friston, K.J., 2007. 
  Comparing hemodynamic models with DCM. Neuroimage 38, 387-401.
"""

import numpy as np
from scipy.integrate import solve_ivp


class BalloonWindkesselModel:
    """
    Balloon-Windkessel model for converting neural activity to BOLD signals.
    
    This class implements the hemodynamic model that relates neuronal activity to 
    changes in blood flow, blood volume, and deoxyhemoglobin content, which together
    determine the BOLD signal measured in fMRI.
    
    Parameters
    ----------
    epsilon : float, optional
        Neuronal efficacy (signal increase per neural activity increase), by default 0.5
    tau_s : float, optional
        Signal decay time constant [s], by default 0.8
    tau_f : float, optional
        Autoregulatory feedback time constant [s], by default 0.4
    tau_0 : float, optional
        Transit time [s], by default 1.0
    alpha : float, optional
        Grubb's vessel stiffness exponent, by default 0.32
    E_0 : float, optional
        Resting oxygen extraction fraction, by default 0.4
    V_0 : float, optional
        Resting blood volume fraction, by default 0.04
    """

    def __init__(
        self, 
        epsilon=0.5, 
        tau_s=0.8, 
        tau_f=0.4, 
        tau_0=1.0, 
        alpha=0.32, 
        E_0=0.4, 
        V_0=0.04
    ):
        # Model parameters
        self.epsilon = epsilon  # Neuronal efficacy
        self.tau_s = tau_s      # Signal decay time constant [s]
        self.tau_f = tau_f      # Autoregulatory feedback time constant [s]
        self.tau_0 = tau_0      # Transit time [s]
        self.alpha = alpha      # Grubb's vessel stiffness exponent
        self.E_0 = E_0          # Resting oxygen extraction fraction
        self.V_0 = V_0          # Resting blood volume fraction
        
        # BOLD signal parameters - calibrated for standard percentage changes (~3%)
        self.k1 = 7.0 * self.E_0
        self.k2 = 2.0
        self.k3 = 2.0 * self.E_0 - 0.2

    def balloon_odes(self, t, y, neural_activity):
        """
        ODE system for the Balloon-Windkessel model.
        
        Parameters
        ----------
        t : float
            Current time point
        y : array-like
            Current state vector [s, f, v, q]
        neural_activity : float
            Value of neural activity at time t
            
        Returns
        -------
        dydt : array
            Derivatives of the state variables
        """
        s, f, v, q = y
        
        # Neural activity at current time (scale with epsilon)
        # Restrict neural activity to be positive
        x = max(0, neural_activity) * self.epsilon
        
        # ODEs - Corrected form based on Friston et al.
        ds = x - s / self.tau_s - (f - 1.0) / self.tau_f
        df = s
        
        # Ensure volumes are positive to avoid numerical issues
        v_safe = max(v, 1e-6)
        
        # Correct ODE for volume
        dv = (f - v_safe**(1/self.alpha)) / self.tau_0
        
        # Correct ODE for deoxyhemoglobin
        # Avoid division by zero and ensure numerical stability
        f_safe = max(f, 1e-6)
        
        # More stable implementation of the extraction term
        if f_safe > 1e-6:
            extraction = (1.0 - (1.0 - self.E_0)**(1.0/f_safe)) / self.E_0
        else:
            extraction = 0.0
            
        # Calculate outflow term safely
        if v_safe > 1e-6:
            outflow = q * v_safe**(1.0/self.alpha - 1.0)
        else:
            outflow = 0.0
        
        dq = (f_safe * extraction - outflow) / self.tau_0
        
        return [ds, df, dv, dq]

    def bold_signal(self, v, q):
        """
        Calculate BOLD signal from blood volume and deoxyhemoglobin content.
        
        Parameters
        ----------
        v : float or array
            Blood volume normalized to rest
        q : float or array
            Deoxyhemoglobin content normalized to rest
            
        Returns
        -------
        bold : float or array
            BOLD signal change (percentage)
        """
        # Ensure v is positive to avoid division by zero in q/v term
        v_safe = np.maximum(v, 1e-6)
        
        # Apply the standard BOLD signal equation from Buxton et al.
        # Scale by 100 to get percentage signal change
        bold_signal = self.V_0 * (self.k1 * (1.0 - q) + 
                                  self.k2 * (1.0 - q/v_safe) + 
                                  self.k3 * (1.0 - v))
        
        # Convert to percentage signal change (normal range: -1% to 5%)
        bold_signal = bold_signal * 100.0
        
        # Clip to reasonable physiological range
        return np.clip(bold_signal, -2.0, 6.0)

    def simulate(self, neural_activity, t_span, dt=0.01, initial_conditions=None):
        """
        Simulate the Balloon-Windkessel model for given neural activity.
        
        Parameters
        ----------
        neural_activity : callable or array-like
            Neural activity as a function of time or array of activity values
        t_span : tuple of float
            Time interval for integration (t_start, t_end)
        dt : float, optional
            Time step for output, by default 0.01
        initial_conditions : array-like, optional
            Initial conditions [s, f, v, q], by default [0, 1, 1, 1]
            
        Returns
        -------
        t : array
            Time points
        bold : array
            BOLD signal at each time point
        states : array
            Full state variables at each time point [s, f, v, q]
        """
        # Default initial conditions: resting state
        if initial_conditions is None:
            initial_conditions = [0.0, 1.0, 1.0, 1.0]
        
        # Create time points
        t = np.arange(t_span[0], t_span[1], dt)
        
        # Prepare neural activity function
        if callable(neural_activity):
            x_t = neural_activity
        else:
            # If neural_activity is an array, interpolate it
            if len(neural_activity) != len(t):
                raise ValueError("If neural_activity is an array, it must have the same length as time points")
            
            from scipy.interpolate import interp1d
            x_t = interp1d(t, neural_activity, kind='linear', bounds_error=False, fill_value=0)
        
        # Solve ODEs with appropriate method and tolerance
        result = solve_ivp(
            lambda t, y: self.balloon_odes(t, y, x_t(t)),
            t_span,
            initial_conditions,
            method='RK45',  # Use RK45 for standard ODE solving
            t_eval=t,
            rtol=1e-6,
            atol=1e-6,
            max_step=dt  # Restrict step size for better accuracy
        )
        
        # Extract results
        s = result.y[0]
        f = result.y[1]
        v = result.y[2]
        q = result.y[3]
        
        # Calculate BOLD signal
        bold = self.bold_signal(v, q)
        
        return result.t, bold, np.array([s, f, v, q])


class PersonalizedHRF(BalloonWindkesselModel):
    """
    Personalized Hemodynamic Response Function (HRF) model extending the Balloon-Windkessel model.
    
    This class allows for region-specific customization of HRF parameters to 
    account for inter-subject and inter-regional variability in the hemodynamic response.
    
    Parameters
    ----------
    All parameters from BalloonWindkesselModel, plus:
    
    region_specific_params : dict, optional
        Dictionary of region-specific parameter overrides for personalization
    """
    
    def __init__(self, region_specific_params=None, **kwargs):
        super().__init__(**kwargs)
        
        # Store default parameters for reference
        self._default_params = {
            'epsilon': self.epsilon,
            'tau_s': self.tau_s,
            'tau_f': self.tau_f,
            'tau_0': self.tau_0,
            'alpha': self.alpha,
            'E_0': self.E_0,
            'V_0': self.V_0
        }
        
        # Initialize region-specific parameters if provided
        self.region_specific_params = {} if region_specific_params is None else region_specific_params
        
    def _get_region_params(self, region_id):
        """
        Get parameters for a specific region, falling back to defaults if not specified.
        
        Parameters
        ----------
        region_id : str or int
            Identifier for the brain region
            
        Returns
        -------
        dict
            Parameters for the specified region
        """
        region_params = self._default_params.copy()
        
        if region_id in self.region_specific_params:
            region_params.update(self.region_specific_params[region_id])
            
        return region_params
    
    def set_region_params(self, region_id, **params):
        """
        Set parameters for a specific brain region.
        
        Parameters
        ----------
        region_id : str or int
            Identifier for the brain region
        **params : dict
            Parameter names and values to set for this region
        """
        if region_id not in self.region_specific_params:
            self.region_specific_params[region_id] = {}
            
        self.region_specific_params[region_id].update(params)
    
    def simulate_region(self, region_id, neural_activity, t_span, dt=0.01, initial_conditions=None):
        """
        Simulate the hemodynamic response for a specific brain region.
        
        Parameters
        ----------
        region_id : str or int
            Identifier for the brain region
        neural_activity : callable or array-like
            Neural activity as a function of time or array of activity values
        t_span : tuple of float
            Time interval for integration (t_start, t_end)
        dt : float, optional
            Time step for output, by default 0.01
        initial_conditions : array-like, optional
            Initial conditions [s, f, v, q], by default [0, 1, 1, 1]
            
        Returns
        -------
        t : array
            Time points
        bold : array
            BOLD signal at each time point
        states : array
            Full state variables at each time point [s, f, v, q]
        """
        # Get region-specific parameters
        params = self._get_region_params(region_id)
        
        # Temporarily set model parameters to region-specific values
        original_params = self._default_params.copy()
        
        # Update model parameters with region-specific values
        for param, value in params.items():
            setattr(self, param, value)
            
        # Recalculate BOLD signal constants that depend on model parameters
        self.k1 = 7.0 * self.E_0
        self.k2 = 2.0
        self.k3 = 2.0 * self.E_0 - 0.2
        
        # Run simulation with these parameters
        result = self.simulate(neural_activity, t_span, dt, initial_conditions)
        
        # Restore original parameters
        for param, value in original_params.items():
            setattr(self, param, value)
            
        # Recalculate BOLD signal constants
        self.k1 = 7.0 * self.E_0
        self.k2 = 2.0
        self.k3 = 2.0 * self.E_0 - 0.2
        
        return result 