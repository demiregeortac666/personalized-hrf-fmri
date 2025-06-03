"""
Integration of personalized HRF models with The Virtual Brain (TVB) platform.

This module provides the necessary functionality to integrate personalized
hemodynamic response functions into The Virtual Brain simulations.
"""

import numpy as np
import warnings

# Initialize variables to None to handle the case when TVB is not installed
time_series_data_type = None
coupling_module = None
integrators = None
monitors_module = None
simulator_module = None

try:
    # Try to import TVB modules
    from tvb.simulator.lab import *
    from tvb.datatypes.time_series import TimeSeries
    import tvb.datatypes.time_series as time_series_data_type
    import tvb.simulator.coupling as coupling_module
    import tvb.simulator.integrators as integrators
    import tvb.simulator.monitors as monitors_module
    import tvb.simulator.simulator as simulator_module
    HAS_TVB = True
except ImportError:
    warnings.warn("TVB (The Virtual Brain) package not found. TVB integration will not be available.")
    HAS_TVB = False


class TVBBalloonWindkesselMonitor:
    """
    Custom TVB monitor for Balloon-Windkessel model with personalized HRF parameters.
    
    This class implements a monitor for TVB that uses the personalized HRF model
    to convert neural activity to BOLD signals during simulation.
    
    Note: This class should only be instantiated if TVB is properly installed.
    """
    
    def __init__(self, period=1.0, region_params=None):
        """
        Initialize the monitor.
        
        Parameters
        ----------
        period : float, optional
            Sampling period in milliseconds, by default 1.0
        region_params : dict, optional
            Dictionary of region-specific HRF parameters keyed by region index,
            by default None (will use default parameters for all regions)
        """
        if not HAS_TVB:
            raise ImportError("TVB (The Virtual Brain) package is required for this functionality.")
            
        self.period = period
        self.region_params = {} if region_params is None else region_params
        
        # Initialize other attributes that will be set during configure
        self.dt = None
        self.connectivity = None
        self.n_regions = None
        self.period_in_steps = None
        self.istep = None
        
        # State variables of the Balloon-Windkessel model for each region
        # [s, f, v, q] = [signal, flow, volume, deoxy]
        self.bw_state = None
        
        # Store the default parameters
        self.default_params = {
            'epsilon': 0.5,  # Neuronal efficacy
            'tau_s': 0.8,    # Signal decay time constant [s]
            'tau_f': 0.4,    # Autoregulatory feedback time constant [s]
            'tau_0': 1.0,    # Transit time [s]
            'alpha': 0.32,   # Grubb's vessel stiffness exponent
            'E_0': 0.4,      # Resting oxygen extraction fraction
            'V_0': 0.04      # Resting blood volume fraction
        }
        
        # BOLD signal parameters
        self.k1 = 7 * self.default_params['E_0']
        self.k2 = 2
        self.k3 = 2 * self.default_params['E_0'] - 0.2
    
    def configure(self, simulator):
        """
        Configure the monitor with the simulator.
        
        Parameters
        ----------
        simulator : Simulator
            TVB simulator instance
        """
        self.dt = simulator.integrator.dt
        self.connectivity = simulator.connectivity
        
        # Get the number of regions from connectivity
        self.n_regions = simulator.connectivity.number_of_regions
        
        # Initialize Balloon-Windkessel state variables for each region
        # [s, f, v, q] = [signal, flow, volume, deoxy]
        # Initial conditions: all regions at rest
        self.bw_state = np.zeros((4, self.n_regions))
        self.bw_state[1, :] = 1.0  # f = 1.0 (normalized blood flow at rest)
        self.bw_state[2, :] = 1.0  # v = 1.0 (normalized blood volume at rest)
        self.bw_state[3, :] = 1.0  # q = 1.0 (normalized deoxyhemoglobin at rest)
        
        # Convert period from physical time (ms) to integration steps
        self.period_in_steps = max(1, int(self.period / self.dt))
        self.istep = 0
    
    def _get_region_params(self, region_idx):
        """
        Get Balloon-Windkessel parameters for a specific region.
        
        Parameters
        ----------
        region_idx : int
            Index of the brain region
            
        Returns
        -------
        dict
            Parameters for the specified region
        """
        params = self.default_params.copy()
        
        if region_idx in self.region_params:
            params.update(self.region_params[region_idx])
            
        return params
    
    def _balloon_windkessel_step(self, neural_activity, dt):
        """
        Perform one step of the Balloon-Windkessel model for all regions.
        
        Parameters
        ----------
        neural_activity : array
            Neural activity for each region
        dt : float
            Time step in seconds
            
        Returns
        -------
        bold : array
            BOLD signal for each region
        """
        # Create arrays to store the derivatives
        # [s, f, v, q] = [signal, flow, volume, deoxy]
        derivatives = np.zeros((4, self.n_regions))
        bold = np.zeros(self.n_regions)
        
        # Process each region with its specific parameters
        for i in range(self.n_regions):
            # Get region-specific parameters
            params = self._get_region_params(i)
            
            # Extract current state for this region
            s, f, v, q = self.bw_state[:, i]
            
            # Neural activity for this region
            x = neural_activity[i]
            
            # Calculate derivatives based on B-W model
            ds = x - s / params['tau_s'] - (f - 1) / params['tau_f']
            df = s
            dv = (f - v**(1/params['alpha'])) / params['tau_0']
            dq = (f * (1 - (1 - params['E_0'])**(1/f)) / params['E_0'] - 
                  q * v**(1/params['alpha']-1)) / params['tau_0']
            
            derivatives[:, i] = [ds, df, dv, dq]
            
            # Calculate BOLD signal for this region
            k1 = 7 * params['E_0']
            k2 = 2
            k3 = 2 * params['E_0'] - 0.2
            
            bold[i] = params['V_0'] * (k1 * (1 - q) + k2 * (1 - q/v) + k3 * (1 - v))
        
        # Update state variables using Euler integration
        self.bw_state += derivatives * dt
        
        return bold
    
    def record(self, step, state):
        """
        Record BOLD signal at the current simulation step.
        
        Parameters
        ----------
        step : int
            Current step of the simulation
        state : tuple
            Current state of the model (time, state variables)
            
        Returns
        -------
        time : float
            Current time in simulation
        bold : array
            BOLD signal for each region
        """
        time, state_variables = state
        
        # Record at specified period
        if step % self.period_in_steps == 0:
            # Extract neural activity (typically the mean field variable)
            # This depends on the specific model being used
            if isinstance(state_variables, list):
                # For models with multiple state variables, use the first one
                neural_activity = state_variables[0]
            else:
                neural_activity = state_variables
            
            # If using surface simulation, average over vertices in each region
            if neural_activity.ndim > 1 and neural_activity.shape[0] > self.n_regions:
                neural_activity = np.mean(neural_activity, axis=1)
            
            # Extract only the number of regions we have parameters for
            neural_activity = neural_activity[:self.n_regions]
            
            # Convert dt from ms to s for the B-W model
            dt_sec = self.dt / 1000.0
            
            # Update B-W model and get BOLD signal
            bold = self._balloon_windkessel_step(neural_activity, dt_sec)
            
            return time, bold
        
        return None
    
    def create_time_series(self, connectivity, storage_path=""):
        """
        Create a TimeSeries object from recorded data.
        
        Parameters
        ----------
        connectivity : Connectivity
            TVB connectivity object
        storage_path : str, optional
            Path for storage, by default ""
            
        Returns
        -------
        time_series : TimeSeries
            TVB TimeSeries object containing the recorded BOLD data
        """
        time_series = time_series_data_type.TimeSeriesRegion(
            data=np.array(self.recorded_data),
            connectivity=connectivity,
            sample_period=self.period,
            title="BOLD signals (Personalized HRF)"
        )
        
        return time_series


class TVBPersonalizedHRFIntegrator:
    """
    Class for integrating personalized HRF models into TVB simulations.
    
    This class provides utility functions for creating TVB configurations with
    personalized HRF parameters, running simulations, and processing results.
    """
    
    def __init__(self):
        """
        Initialize the integrator.
        """
        if not HAS_TVB:
            raise ImportError("TVB (The Virtual Brain) package is required for this functionality.")
    
    def create_tvb_simulator(self, model, connectivity, region_hrf_params=None, 
                             coupling=None, integrator=None, monitors=None, 
                             simulation_length=1000.0, simulation_dt=0.1):
        """
        Create a TVB simulator with personalized HRF monitoring.
        
        Parameters
        ----------
        model : Model
            TVB neural model
        connectivity : Connectivity
            TVB connectivity object
        region_hrf_params : dict, optional
            Dictionary of region-specific HRF parameters, by default None
        coupling : Coupling, optional
            TVB coupling type, by default None (linear coupling)
        integrator : Integrator, optional
            TVB integrator, by default None (Heun deterministic)
        monitors : list, optional
            List of TVB monitors, by default None
        simulation_length : float, optional
            Length of simulation in ms, by default 1000.0
        simulation_dt : float, optional
            Integration time step in ms, by default 0.1
            
        Returns
        -------
        simulator : Simulator
            Configured TVB simulator
        """
        # Default components if not specified
        if coupling is None:
            coupling = coupling_module.Linear(a=0.1)
            
        if integrator is None:
            integrator = integrators.HeunDeterministic(dt=simulation_dt)
            
        # Create custom BOLD monitor with personalized HRF
        bold_monitor = TVBBalloonWindkesselMonitor(
            period=2000.0,  # TR in ms
            region_params=region_hrf_params
        )
        
        # Add custom monitor to existing monitors
        if monitors is None:
            monitors = [
                monitors_module.Raw(),  # Record raw neural activity
                bold_monitor  # Custom BOLD monitor
            ]
        else:
            monitors.append(bold_monitor)
        
        # Create and configure simulator
        simulator = simulator_module.Simulator(
            model=model,
            connectivity=connectivity,
            coupling=coupling,
            integrator=integrator,
            monitors=monitors,
            simulation_length=simulation_length
        )
        
        simulator.configure()
        
        return simulator
    
    def run_simulation_with_personalized_hrf(self, simulator):
        """
        Run a TVB simulation with personalized HRF monitoring.
        
        Parameters
        ----------
        simulator : Simulator
            Configured TVB simulator
            
        Returns
        -------
        results : dict
            Dictionary of simulation results with neural and BOLD time series
        """
        # Run simulation
        raw_data = []
        bold_data = []
        
        for time, data in simulator():
            if isinstance(data, tuple):
                # Multiple monitors
                for i, monitor_data in enumerate(data):
                    if i == 0:  # Assume first monitor is Raw
                        raw_data.append(monitor_data)
                    elif i == 1:  # Assume second monitor is our custom BOLD monitor
                        bold_data.append(monitor_data)
            else:
                # Single monitor
                raw_data.append(data)
        
        # Convert to NumPy arrays
        raw_data = np.array(raw_data)
        bold_data = np.array(bold_data) if bold_data else None
        
        return {
            'raw': raw_data,
            'bold': bold_data
        } 