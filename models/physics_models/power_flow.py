"""
Power Flow Analysis Module
Newton-Raphson method for solving power flow equations
Based on File 2 - Physics-based approach
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional, List
from dataclasses import dataclass
import logging
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import spsolve
import warnings

logger = logging.getLogger(__name__)


@dataclass
class PowerFlowConfig:
    """Configuration for Power Flow solver"""
    method: str = "newton_raphson"  # newton_raphson, gauss_seidel, fast_decoupled
    max_iterations: int = 100
    tolerance: float = 1e-6
    voltage_min: float = 0.9  # per unit
    voltage_max: float = 1.1  # per unit
    flat_start: bool = True  # Initialize with flat voltage profile
    acceleration_factor: float = 1.0  # For Gauss-Seidel
    check_limits: bool = True


@dataclass
class BusData:
    """Data for a single bus in the network"""
    bus_id: int
    bus_type: str  # 'slack', 'pv', 'pq'
    voltage_magnitude: float  # p.u.
    voltage_angle: float  # radians
    active_power: float  # MW
    reactive_power: float  # MVAr
    active_generation: float = 0.0  # MW
    reactive_generation: float = 0.0  # MVAr
    active_load: float = 0.0  # MW
    reactive_load: float = 0.0  # MVAr
    shunt_conductance: float = 0.0  # p.u.
    shunt_susceptance: float = 0.0  # p.u.


@dataclass
class LineData:
    """Data for a transmission line"""
    from_bus: int
    to_bus: int
    resistance: float  # p.u.
    reactance: float  # p.u.
    susceptance: float = 0.0  # p.u. (line charging)
    tap_ratio: float = 1.0  # Transformer tap ratio
    phase_shift: float = 0.0  # Phase shift angle (radians)
    status: bool = True  # Line in service


class AdmittanceMatrix:
    """
    Builds and manages the network admittance matrix (Y-bus)
    Y = G + jB
    """
    
    def __init__(self):
        self.Y_bus = None  # Complex admittance matrix
        self.G = None  # Conductance matrix (real part)
        self.B = None  # Susceptance matrix (imaginary part)
        self.n_buses = 0
    
    def build(self, buses: List[BusData], lines: List[LineData]) -> np.ndarray:
        """
        Build Y-bus admittance matrix
        
        Args:
            buses: List of bus data
            lines: List of line data
            
        Returns:
            Complex Y-bus matrix
        """
        self.n_buses = len(buses)
        self.Y_bus = np.zeros((self.n_buses, self.n_buses), dtype=complex)
        
        # Build Y-bus from line data
        for line in lines:
            if not line.status:
                continue
            
            from_idx = line.from_bus
            to_idx = line.to_bus
            
            # Series impedance
            z = complex(line.resistance, line.reactance)
            
            if abs(z) < 1e-10:
                warnings.warn(f"Very low impedance on line {from_idx}-{to_idx}")
                z = complex(1e-10, 1e-10)
            
            y_series = 1.0 / z
            
            # Shunt admittance (line charging)
            y_shunt = complex(0, line.susceptance / 2.0)
            
            # Tap ratio
            tap = line.tap_ratio
            if tap == 0:
                tap = 1.0
            
            # Phase shift
            shift = line.phase_shift
            tap_complex = tap * np.exp(1j * shift)
            
            # Add to Y-bus
            # Off-diagonal elements
            self.Y_bus[from_idx, to_idx] -= y_series / (tap * np.conj(tap_complex))
            self.Y_bus[to_idx, from_idx] -= y_series / tap
            
            # Diagonal elements
            self.Y_bus[from_idx, from_idx] += (y_series + y_shunt) / (tap * tap)
            self.Y_bus[to_idx, to_idx] += y_series + y_shunt
        
        # Add shunt admittances at buses
        for bus in buses:
            if bus.shunt_conductance != 0 or bus.shunt_susceptance != 0:
                y_shunt = complex(bus.shunt_conductance, bus.shunt_susceptance)
                self.Y_bus[bus.bus_id, bus.bus_id] += y_shunt
        
        # Extract G and B matrices
        self.G = self.Y_bus.real
        self.B = self.Y_bus.imag
        
        logger.info(f"Built Y-bus matrix for {self.n_buses} buses")
        
        return self.Y_bus
    
    def get_admittance(self, from_bus: int, to_bus: int) -> complex:
        """Get admittance between two buses"""
        if self.Y_bus is None:
            raise ValueError("Y-bus not built yet")
        return self.Y_bus[from_bus, to_bus]


class PowerFlowSolver:
    """
    Solves power flow equations using Newton-Raphson method
    """
    
    def __init__(self, config: PowerFlowConfig):
        self.config = config
        self.buses: List[BusData] = []
        self.lines: List[LineData] = []
        self.Y_matrix = AdmittanceMatrix()
        self.converged = False
        self.n_iterations = 0
        
        # State variables
        self.voltage_magnitude = None  # |V|
        self.voltage_angle = None  # θ
        
        # Results
        self.results = {}
    
    def setup_network(self, buses: List[BusData], lines: List[LineData]):
        """
        Setup network topology
        
        Args:
            buses: List of bus data
            lines: List of line data
        """
        self.buses = buses
        self.lines = lines
        
        # Build admittance matrix
        self.Y_matrix.build(buses, lines)
        
        # Initialize voltage vectors
        self._initialize_voltages()
        
        logger.info(f"Network setup complete: {len(buses)} buses, {len(lines)} lines")
    
    def _initialize_voltages(self):
        """Initialize voltage magnitude and angle vectors"""
        n = len(self.buses)
        self.voltage_magnitude = np.zeros(n)
        self.voltage_angle = np.zeros(n)
        
        for i, bus in enumerate(self.buses):
            if self.config.flat_start:
                # Flat start: all voltages at 1.0 p.u., 0 degrees
                if bus.bus_type == 'slack':
                    self.voltage_magnitude[i] = bus.voltage_magnitude
                    self.voltage_angle[i] = bus.voltage_angle
                else:
                    self.voltage_magnitude[i] = 1.0
                    self.voltage_angle[i] = 0.0
            else:
                # Use provided initial values
                self.voltage_magnitude[i] = bus.voltage_magnitude
                self.voltage_angle[i] = bus.voltage_angle
    
    def solve(self) -> Dict:
        """
        Solve power flow equations
        
        Returns:
            Dictionary with results
        """
        if self.config.method == "newton_raphson":
            return self._solve_newton_raphson()
        elif self.config.method == "gauss_seidel":
            return self._solve_gauss_seidel()
        else:
            raise ValueError(f"Unknown method: {self.config.method}")
    
    def _solve_newton_raphson(self) -> Dict:
        """
        Solve using Newton-Raphson method
        Iteratively solves: J * Δx = -f(x)
        where x = [θ, |V|], f(x) = [ΔP, ΔQ]
        """
        logger.info("Starting Newton-Raphson power flow solver")
        
        # Identify bus types
        slack_buses = [i for i, b in enumerate(self.buses) if b.bus_type == 'slack']
        pv_buses = [i for i, b in enumerate(self.buses) if b.bus_type == 'pv']
        pq_buses = [i for i, b in enumerate(self.buses) if b.bus_type == 'pq']
        
        if len(slack_buses) != 1:
            raise ValueError("Exactly one slack bus required")
        
        slack_bus = slack_buses[0]
        
        # Unknown variables indices (exclude slack bus angle, PV bus voltages)
        theta_unknowns = [i for i in range(len(self.buses)) if i != slack_bus]
        v_unknowns = pq_buses  # Only PQ buses have unknown voltage magnitudes
        
        n_unknowns = len(theta_unknowns) + len(v_unknowns)
        
        # Newton-Raphson iterations
        for iteration in range(self.config.max_iterations):
            self.n_iterations = iteration + 1
            
            # Calculate power mismatches
            P_calc, Q_calc = self._calculate_power_injections()
            
            # Compute mismatches ΔP and ΔQ
            delta_P = np.zeros(len(theta_unknowns))
            delta_Q = np.zeros(len(v_unknowns))
            
            for idx, bus_idx in enumerate(theta_unknowns):
                P_scheduled = self.buses[bus_idx].active_generation - self.buses[bus_idx].active_load
                delta_P[idx] = P_scheduled - P_calc[bus_idx]
            
            for idx, bus_idx in enumerate(v_unknowns):
                Q_scheduled = self.buses[bus_idx].reactive_generation - self.buses[bus_idx].reactive_load
                delta_Q[idx] = Q_scheduled - Q_calc[bus_idx]
            
            # Check convergence
            max_mismatch = max(np.max(np.abs(delta_P)), np.max(np.abs(delta_Q)) if len(delta_Q) > 0 else 0)
            
            if max_mismatch < self.config.tolerance:
                self.converged = True
                logger.info(f"Converged in {iteration + 1} iterations. Max mismatch: {max_mismatch:.2e}")
                break
            
            # Build Jacobian matrix
            J = self._build_jacobian(theta_unknowns, v_unknowns)
            
            # Solve linear system: J * Δx = -[ΔP; ΔQ]
            mismatch = np.concatenate([delta_P, delta_Q])
            
            try:
                delta_x = np.linalg.solve(J, mismatch)
            except np.linalg.LinAlgError:
                logger.error("Jacobian matrix is singular")
                self.converged = False
                break
            
            # Update state variables
            # Update angles
            for idx, bus_idx in enumerate(theta_unknowns):
                self.voltage_angle[bus_idx] += delta_x[idx]
            
            # Update voltage magnitudes
            for idx, bus_idx in enumerate(v_unknowns):
                self.voltage_magnitude[bus_idx] += delta_x[len(theta_unknowns) + idx]
            
            # Check voltage limits
            if self.config.check_limits:
                self._enforce_voltage_limits()
        
        if not self.converged:
            logger.warning(f"Power flow did not converge after {self.config.max_iterations} iterations")
        
        # Calculate final power flows
        self._calculate_final_results()
        
        return self.results
    
    def _calculate_power_injections(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculate active and reactive power injections at all buses
        
        P_i = Σ |V_i||V_j||Y_ij| cos(θ_ij + δ_j - δ_i)
        Q_i = Σ |V_i||V_j||Y_ij| sin(θ_ij + δ_j - δ_i)
        """
        n = len(self.buses)
        P = np.zeros(n)
        Q = np.zeros(n)
        
        Y = self.Y_matrix.Y_bus
        V = self.voltage_magnitude
        theta = self.voltage_angle
        
        for i in range(n):
            for j in range(n):
                Y_ij = Y[i, j]
                Y_mag = abs(Y_ij)
                Y_angle = np.angle(Y_ij)
                
                angle_diff = Y_angle + theta[j] - theta[i]
                
                P[i] += V[i] * V[j] * Y_mag * np.cos(angle_diff)
                Q[i] += V[i] * V[j] * Y_mag * np.sin(angle_diff)
        
        return P, Q
    
    def _build_jacobian(self, theta_unknowns: List[int], v_unknowns: List[int]) -> np.ndarray:
        """
        Build Jacobian matrix for Newton-Raphson
        
        J = [ ∂P/∂θ   ∂P/∂|V| ]
            [ ∂Q/∂θ   ∂Q/∂|V| ]
        """
        n_theta = len(theta_unknowns)
        n_v = len(v_unknowns)
        n = n_theta + n_v
        
        J = np.zeros((n, n))
        
        Y = self.Y_matrix.Y_bus
        V = self.voltage_magnitude
        theta = self.voltage_angle
        
        # Compute J11: ∂P/∂θ
        for idx_i, i in enumerate(theta_unknowns):
            for idx_j, j in enumerate(theta_unknowns):
                if i == j:
                    # Diagonal element
                    J[idx_i, idx_j] = -Q[i] - V[i]**2 * self.Y_matrix.B[i, i]
                else:
                    # Off-diagonal element
                    Y_ij = Y[i, j]
                    Y_mag = abs(Y_ij)
                    Y_angle = np.angle(Y_ij)
                    angle_diff = Y_angle + theta[j] - theta[i]
                    
                    J[idx_i, idx_j] = V[i] * V[j] * Y_mag * np.sin(angle_diff)
        
        # Compute J12: ∂P/∂|V|
        for idx_i, i in enumerate(theta_unknowns):
            for idx_j, j in enumerate(v_unknowns):
                if i == j:
                    # Diagonal element
                    J[idx_i, n_theta + idx_j] = P[i] / V[i] + V[i] * self.Y_matrix.G[i, i]
                else:
                    # Off-diagonal element
                    Y_ij = Y[i, j]
                    Y_mag = abs(Y_ij)
                    Y_angle = np.angle(Y_ij)
                    angle_diff = Y_angle + theta[j] - theta[i]
                    
                    J[idx_i, n_theta + idx_j] = V[i] * Y_mag * np.cos(angle_diff)
        
        # Compute J21: ∂Q/∂θ
        for idx_i, i in enumerate(v_unknowns):
            for idx_j, j in enumerate(theta_unknowns):
                if i == j:
                    # Diagonal element
                    J[n_theta + idx_i, idx_j] = P[i] - V[i]**2 * self.Y_matrix.G[i, i]
                else:
                    # Off-diagonal element
                    Y_ij = Y[i, j]
                    Y_mag = abs(Y_ij)
                    Y_angle = np.angle(Y_ij)
                    angle_diff = Y_angle + theta[j] - theta[i]
                    
                    J[n_theta + idx_i, idx_j] = -V[i] * V[j] * Y_mag * np.cos(angle_diff)
        
        # Compute J22: ∂Q/∂|V|
        for idx_i, i in enumerate(v_unknowns):
            for idx_j, j in enumerate(v_unknowns):
                if i == j:
                    # Diagonal element
                    J[n_theta + idx_i, n_theta + idx_j] = Q[i] / V[i] - V[i] * self.Y_matrix.B[i, i]
                else:
                    # Off-diagonal element
                    Y_ij = Y[i, j]
                    Y_mag = abs(Y_ij)
                    Y_angle = np.angle(Y_ij)
                    angle_diff = Y_angle + theta[j] - theta[i]
                    
                    J[n_theta + idx_i, n_theta + idx_j] = V[i] * Y_mag * np.sin(angle_diff)
        
        return J
    
    def _solve_gauss_seidel(self) -> Dict:
        """
        Solve using Gauss-Seidel method
        Simpler but slower convergence than Newton-Raphson
        """
        logger.info("Starting Gauss-Seidel power flow solver")
        
        slack_buses = [i for i, b in enumerate(self.buses) if b.bus_type == 'slack']
        if len(slack_buses) != 1:
            raise ValueError("Exactly one slack bus required")
        
        slack_bus = slack_buses[0]
        
        for iteration in range(self.config.max_iterations):
            self.n_iterations = iteration + 1
            max_change = 0.0
            
            for i, bus in enumerate(self.buses):
                if i == slack_bus:
                    continue  # Skip slack bus
                
                # Calculate power injection
                P_scheduled = bus.active_generation - bus.active_load
                Q_scheduled = bus.reactive_generation - bus.reactive_load
                
                # Calculate complex power
                S_calc = complex(0, 0)
                for j in range(len(self.buses)):
                    if i != j:
                        V_j = self.voltage_magnitude[j] * np.exp(1j * self.voltage_angle[j])
                        Y_ij = self.Y_matrix.Y_bus[i, j]
                        S_calc += V_j * np.conj(Y_ij)
                
                # Update voltage
                S_scheduled = complex(P_scheduled, Q_scheduled)
                Y_ii = self.Y_matrix.Y_bus[i, i]
                V_old = self.voltage_magnitude[i] * np.exp(1j * self.voltage_angle[i])
                
                V_new = (S_scheduled / np.conj(V_old) - S_calc) / Y_ii
                
                # Acceleration factor
                V_new = V_old + self.config.acceleration_factor * (V_new - V_old)
                
                # For PV buses, maintain voltage magnitude
                if bus.bus_type == 'pv':
                    V_new = bus.voltage_magnitude * np.exp(1j * np.angle(V_new))
                
                # Update state variables
                voltage_change = abs(V_new - V_old)
                max_change = max(max_change, voltage_change)
                
                self.voltage_magnitude[i] = abs(V_new)
                self.voltage_angle[i] = np.angle(V_new)
            
            # Check convergence
            if max_change < self.config.tolerance:
                self.converged = True
                logger.info(f"Converged in {iteration + 1} iterations. Max change: {max_change:.2e}")
                break
        
        if not self.converged:
            logger.warning(f"Power flow did not converge after {self.config.max_iterations} iterations")
        
        self._calculate_final_results()
        return self.results
    
    def _enforce_voltage_limits(self):
        """Enforce voltage magnitude limits"""
        for i in range(len(self.buses)):
            if self.voltage_magnitude[i] < self.config.voltage_min:
                self.voltage_magnitude[i] = self.config.voltage_min
                logger.warning(f"Bus {i} voltage limited to {self.config.voltage_min} p.u.")
            elif self.voltage_magnitude[i] > self.config.voltage_max:
                self.voltage_magnitude[i] = self.config.voltage_max
                logger.warning(f"Bus {i} voltage limited to {self.config.voltage_max} p.u.")
    
    def _calculate_final_results(self):
        """Calculate final power flows and losses"""
        P_calc, Q_calc = self._calculate_power_injections()
        
        # Store results
        self.results = {
            'converged': self.converged,
            'n_iterations': self.n_iterations,
            'voltage_magnitude': self.voltage_magnitude.copy(),
            'voltage_angle': self.voltage_angle.copy(),
            'active_power': P_calc,
            'reactive_power': Q_calc,
            'buses': []
        }
        
        # Per-bus results
        for i, bus in enumerate(self.buses):
            bus_result = {
                'bus_id': bus.bus_id,
                'bus_type': bus.bus_type,
                'voltage_magnitude': self.voltage_magnitude[i],
                'voltage_angle': np.degrees(self.voltage_angle[i]),
                'active_power': P_calc[i],
                'reactive_power': Q_calc[i]
            }
            self.results['buses'].append(bus_result)
        
        logger.info("Final results calculated")
    
    def get_voltage_at_bus(self, bus_id: int) -> Tuple[float, float]:
        """
        Get voltage magnitude and angle at specified bus
        
        Returns:
            (magnitude in p.u., angle in radians)
        """
        if bus_id >= len(self.buses):
            raise ValueError(f"Bus {bus_id} does not exist")
        
        return self.voltage_magnitude[bus_id], self.voltage_angle[bus_id]
    
    def get_power_at_bus(self, bus_id: int) -> Tuple[float, float]:
        """
        Get active and reactive power at specified bus
        
        Returns:
            (active_power in MW, reactive_power in MVAr)
        """
        if bus_id >= len(self.buses):
            raise ValueError(f"Bus {bus_id} does not exist")
        
        P, Q = self._calculate_power_injections()
        return P[bus_id], Q[bus_id]


# Example usage
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Create a simple 3-bus test system
    buses = [
        BusData(0, 'slack', 1.0, 0.0, 0.0, 0.0, active_generation=0.0, active_load=0.0),
        BusData(1, 'pq', 1.0, 0.0, 0.0, 0.0, active_load=0.5, reactive_load=0.2),
        BusData(2, 'pq', 1.0, 0.0, 0.0, 0.0, active_load=0.3, reactive_load=0.1)
    ]
    
    lines = [
        LineData(0, 1, resistance=0.01, reactance=0.1),
        LineData(0, 2, resistance=0.015, reactance=0.12),
        LineData(1, 2, resistance=0.02, reactance=0.15)
    ]
    
    # Configure and solve
    config = PowerFlowConfig(
        method="newton_raphson",
        max_iterations=100,
        tolerance=1e-6
    )
    
    solver = PowerFlowSolver(config)
    solver.setup_network(buses, lines)
    results = solver.solve()
    
    print("\n" + "="*60)
    print("POWER FLOW RESULTS")
    print("="*60)
    print(f"Converged: {results['converged']}")
    print(f"Iterations: {results['n_iterations']}")
    print("\nBus voltages:")
    for bus_result in results['buses']:
        print(f"  Bus {bus_result['bus_id']}: "
              f"|V|={bus_result['voltage_magnitude']:.4f} p.u., "
              f"θ={bus_result['voltage_angle']:.2f}°")