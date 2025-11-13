"""
Particle Swarm Optimization for Power Estimation
Based on File 2 - Network-based approach with nested Power Flow
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional, Callable, List
from dataclasses import dataclass
import logging
from concurrent.futures import ThreadPoolExecutor
import time

logger = logging.getLogger(__name__)


@dataclass
class PSOConfig:
    """Configuration for PSO algorithm"""
    n_particles: int = 200
    n_iterations: int = 500
    w_max: float = 0.9  # Initial inertia weight
    w_min: float = 0.4  # Final inertia weight
    c1_upper: float = 2.5  # Initial cognitive coefficient
    c1_lower: float = 1.5  # Final cognitive coefficient
    c2_upper: float = 2.5  # Final social coefficient
    c2_lower: float = 1.5  # Initial social coefficient
    objective_threshold: float = 1e-17
    max_stagnation: int = 50  # Stop if no improvement for N iterations
    parallel: bool = False  # Use parallel evaluation
    n_workers: int = 4  # Number of parallel workers


@dataclass
class PSOBounds:
    """Search space bounds for PSO"""
    p_min: float = 0.0  # Minimum active power (kW)
    p_max: float = 100.0  # Maximum active power (kW)
    q_min: float = 0.0  # Minimum reactive power (kVAr)
    q_max: float = 50.0  # Maximum reactive power (kVAr)


class Particle:
    """
    Represents a single particle in the swarm
    Each particle is a candidate solution (P, Q)
    """
    
    def __init__(self, bounds: PSOBounds):
        self.bounds = bounds
        
        # Initialize position randomly within bounds
        self.position_p = np.random.uniform(bounds.p_min, bounds.p_max)
        self.position_q = np.random.uniform(bounds.q_min, bounds.q_max)
        
        # Initialize velocity
        self.velocity_p = np.random.uniform(-1, 1)
        self.velocity_q = np.random.uniform(-1, 1)
        
        # Best known position for this particle
        self.best_position_p = self.position_p
        self.best_position_q = self.position_q
        self.best_fitness = float('inf')
        
        # Current fitness
        self.fitness = float('inf')
    
    def update_velocity(
        self,
        global_best_p: float,
        global_best_q: float,
        w: float,
        c1: float,
        c2: float
    ):
        """
        Update particle velocity using PSO equations
        
        v(t+1) = w*v(t) + c1*r1*(pbest - x(t)) + c2*r2*(gbest - x(t))
        """
        r1 = np.random.random()
        r2 = np.random.random()
        
        # Update P velocity
        cognitive_p = c1 * r1 * (self.best_position_p - self.position_p)
        social_p = c2 * r2 * (global_best_p - self.position_p)
        self.velocity_p = w * self.velocity_p + cognitive_p + social_p
        
        # Update Q velocity
        cognitive_q = c1 * r1 * (self.best_position_q - self.position_q)
        social_q = c2 * r2 * (global_best_q - self.position_q)
        self.velocity_q = w * self.velocity_q + cognitive_q + social_q
        
        # Velocity clamping to prevent explosion
        max_velocity = 10.0
        self.velocity_p = np.clip(self.velocity_p, -max_velocity, max_velocity)
        self.velocity_q = np.clip(self.velocity_q, -max_velocity, max_velocity)
    
    def update_position(self):
        """
        Update particle position: x(t+1) = x(t) + v(t+1)
        Ensure position stays within bounds
        """
        self.position_p += self.velocity_p
        self.position_q += self.velocity_q
        
        # Boundary handling - reflecting
        if self.position_p < self.bounds.p_min:
            self.position_p = self.bounds.p_min
            self.velocity_p = -self.velocity_p * 0.5
        elif self.position_p > self.bounds.p_max:
            self.position_p = self.bounds.p_max
            self.velocity_p = -self.velocity_p * 0.5
        
        if self.position_q < self.bounds.q_min:
            self.position_q = self.bounds.q_min
            self.velocity_q = -self.velocity_q * 0.5
        elif self.position_q > self.bounds.q_max:
            self.position_q = self.bounds.q_max
            self.velocity_q = -self.velocity_q * 0.5
    
    def update_best(self):
        """Update personal best if current position is better"""
        if self.fitness < self.best_fitness:
            self.best_fitness = self.fitness
            self.best_position_p = self.position_p
            self.best_position_q = self.position_q
            return True
        return False


class PSOOptimizer:
    """
    Particle Swarm Optimization for power estimation
    """
    
    def __init__(self, config: PSOConfig, bounds: PSOBounds):
        self.config = config
        self.bounds = bounds
        self.swarm: List[Particle] = []
        
        # Global best
        self.global_best_p = 0.0
        self.global_best_q = 0.0
        self.global_best_fitness = float('inf')
        
        # Optimization history
        self.history = {
            'iteration': [],
            'best_fitness': [],
            'mean_fitness': [],
            'best_p': [],
            'best_q': [],
            'convergence_speed': []
        }
        
        # Performance tracking
        self.stagnation_counter = 0
        self.last_improvement_iter = 0
        
    def initialize_swarm(self):
        """Initialize particle swarm with random positions"""
        self.swarm = [Particle(self.bounds) for _ in range(self.config.n_particles)]
        logger.info(f"Initialized swarm with {self.config.n_particles} particles")
    
    def optimize(
        self,
        objective_function: Callable[[float, float], float],
        verbose: bool = True
    ) -> Tuple[float, float, Dict]:
        """
        Run PSO optimization
        
        Args:
            objective_function: Function to minimize f(P, Q) -> fitness
            verbose: Print progress
            
        Returns:
            - best_p: Optimal active power
            - best_q: Optimal reactive power
            - optimization_info: Dictionary with optimization details
        """
        start_time = time.time()
        
        # Initialize swarm if not already done
        if not self.swarm:
            self.initialize_swarm()
        
        # Evaluate initial population
        self._evaluate_swarm(objective_function)
        
        logger.info(f"Starting PSO optimization for {self.config.n_iterations} iterations")
        
        # Main optimization loop
        for iteration in range(self.config.n_iterations):
            # Update PSO parameters (linearly decreasing/increasing)
            w = self._compute_inertia_weight(iteration)
            c1 = self._compute_cognitive_coefficient(iteration)
            c2 = self._compute_social_coefficient(iteration)
            
            # Update all particles
            for particle in self.swarm:
                # Update velocity
                particle.update_velocity(
                    self.global_best_p,
                    self.global_best_q,
                    w, c1, c2
                )
                
                # Update position
                particle.update_position()
            
            # Evaluate new positions
            self._evaluate_swarm(objective_function)
            
            # Record history
            mean_fitness = np.mean([p.fitness for p in self.swarm])
            self.history['iteration'].append(iteration)
            self.history['best_fitness'].append(self.global_best_fitness)
            self.history['mean_fitness'].append(mean_fitness)
            self.history['best_p'].append(self.global_best_p)
            self.history['best_q'].append(self.global_best_q)
            
            # Convergence speed
            if iteration > 0:
                conv_speed = abs(
                    self.history['best_fitness'][-1] - 
                    self.history['best_fitness'][-2]
                )
                self.history['convergence_speed'].append(conv_speed)
            else:
                self.history['convergence_speed'].append(0)
            
            # Check for early stopping
            if self.global_best_fitness < self.config.objective_threshold:
                logger.info(f"Objective threshold reached at iteration {iteration}")
                break
            
            # Check for stagnation
            if iteration - self.last_improvement_iter > self.config.max_stagnation:
                logger.info(f"Optimization stagnated at iteration {iteration}")
                break
            
            # Verbose output
            if verbose and (iteration % 100 == 0 or iteration == self.config.n_iterations - 1):
                logger.info(
                    f"Iter {iteration}: Best Fitness={self.global_best_fitness:.6e}, "
                    f"P={self.global_best_p:.4f}, Q={self.global_best_q:.4f}, "
                    f"w={w:.3f}, c1={c1:.3f}, c2={c2:.3f}"
                )
        
        elapsed_time = time.time() - start_time
        
        # Prepare optimization info
        optimization_info = {
            'n_iterations': iteration + 1,
            'final_fitness': self.global_best_fitness,
            'converged': self.global_best_fitness < self.config.objective_threshold,
            'elapsed_time_seconds': elapsed_time,
            'mean_fitness_final': mean_fitness,
            'convergence_rate': self._compute_convergence_rate(),
            'history': self.history
        }
        
        logger.info(
            f"PSO optimization complete in {elapsed_time:.2f}s. "
            f"Best solution: P={self.global_best_p:.4f} kW, Q={self.global_best_q:.4f} kVAr"
        )
        
        return self.global_best_p, self.global_best_q, optimization_info
    
    def _evaluate_swarm(self, objective_function: Callable[[float, float], float]):
        """
        Evaluate fitness for all particles
        Update personal and global bests
        """
        if self.config.parallel:
            self._evaluate_swarm_parallel(objective_function)
        else:
            self._evaluate_swarm_sequential(objective_function)
    
    def _evaluate_swarm_sequential(self, objective_function: Callable):
        """Sequential evaluation of particles"""
        improved = False
        
        for particle in self.swarm:
            # Evaluate objective function
            particle.fitness = objective_function(particle.position_p, particle.position_q)
            
            # Update personal best
            if particle.update_best():
                # Check if this is new global best
                if particle.best_fitness < self.global_best_fitness:
                    self.global_best_fitness = particle.best_fitness
                    self.global_best_p = particle.best_position_p
                    self.global_best_q = particle.best_position_q
                    self.last_improvement_iter = len(self.history['iteration'])
                    improved = True
        
        if not improved:
            self.stagnation_counter += 1
        else:
            self.stagnation_counter = 0
    
    def _evaluate_swarm_parallel(self, objective_function: Callable):
        """Parallel evaluation of particles using ThreadPoolExecutor"""
        improved = False
        
        def evaluate_particle(particle: Particle) -> Tuple[Particle, float]:
            fitness = objective_function(particle.position_p, particle.position_q)
            return particle, fitness
        
        with ThreadPoolExecutor(max_workers=self.config.n_workers) as executor:
            results = list(executor.map(evaluate_particle, self.swarm))
        
        # Update fitnesses
        for particle, fitness in results:
            particle.fitness = fitness
            
            # Update personal best
            if particle.update_best():
                # Check if this is new global best
                if particle.best_fitness < self.global_best_fitness:
                    self.global_best_fitness = particle.best_fitness
                    self.global_best_p = particle.best_position_p
                    self.global_best_q = particle.best_position_q
                    self.last_improvement_iter = len(self.history['iteration'])
                    improved = True
        
        if not improved:
            self.stagnation_counter += 1
        else:
            self.stagnation_counter = 0
    
    def _compute_inertia_weight(self, iteration: int) -> float:
        """
        Linearly decreasing inertia weight
        w(t) = w_max - (w_max - w_min) * t / T
        """
        return self.config.w_max - (
            (self.config.w_max - self.config.w_min) * iteration / self.config.n_iterations
        )
    
    def _compute_cognitive_coefficient(self, iteration: int) -> float:
        """
        Linearly decreasing cognitive coefficient
        c1(t) = c1_upper - (c1_upper - c1_lower) * t / T
        """
        return self.config.c1_upper - (
            (self.config.c1_upper - self.config.c1_lower) * iteration / self.config.n_iterations
        )
    
    def _compute_social_coefficient(self, iteration: int) -> float:
        """
        Linearly increasing social coefficient
        c2(t) = c2_lower + (c2_upper - c2_lower) * t / T
        """
        return self.config.c2_lower + (
            (self.config.c2_upper - self.config.c2_lower) * iteration / self.config.n_iterations
        )
    
    def _compute_convergence_rate(self) -> float:
        """
        Compute average convergence rate over optimization
        """
        if len(self.history['convergence_speed']) < 2:
            return 0.0
        
        return np.mean(self.history['convergence_speed'])
    
    def reset(self):
        """Reset optimizer state for new optimization"""
        self.swarm = []
        self.global_best_p = 0.0
        self.global_best_q = 0.0
        self.global_best_fitness = float('inf')
        self.history = {
            'iteration': [],
            'best_fitness': [],
            'mean_fitness': [],
            'best_p': [],
            'best_q': [],
            'convergence_speed': []
        }
        self.stagnation_counter = 0
        self.last_improvement_iter = 0
    
    def get_diversity_metrics(self) -> Dict:
        """
        Compute swarm diversity metrics
        High diversity = good exploration
        Low diversity = converged or premature convergence
        """
        positions_p = np.array([p.position_p for p in self.swarm])
        positions_q = np.array([p.position_q for p in self.swarm])
        
        return {
            'std_p': positions_p.std(),
            'std_q': positions_q.std(),
            'range_p': positions_p.max() - positions_p.min(),
            'range_q': positions_q.max() - positions_q.min(),
            'mean_p': positions_p.mean(),
            'mean_q': positions_q.mean()
        }


# Example usage and testing
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Define a simple test objective function (Rosenbrock-like)
    def test_objective(p: float, q: float) -> float:
        """
        Test function: minimize (p - 5)^2 + (q - 3)^2
        Global minimum at (5, 3)
        """
        return (p - 5.0)**2 + (q - 3.0)**2
    
    # Configure PSO
    config = PSOConfig(
        n_particles=50,
        n_iterations=200,
        w_max=0.9,
        w_min=0.4,
        c1_upper=2.5,
        c1_lower=1.5,
        c2_upper=2.5,
        c2_lower=1.5,
        objective_threshold=1e-6,
        parallel=False
    )
    
    bounds = PSOBounds(
        p_min=0.0,
        p_max=10.0,
        q_min=0.0,
        q_max=10.0
    )
    
    # Create optimizer
    optimizer = PSOOptimizer(config, bounds)
    
    # Run optimization
    best_p, best_q, info = optimizer.optimize(test_objective, verbose=True)
    
    print("\n" + "="*60)
    print("OPTIMIZATION RESULTS")
    print("="*60)
    print(f"Best P: {best_p:.6f} (expected: 5.0)")
    print(f"Best Q: {best_q:.6f} (expected: 3.0)")
    print(f"Final fitness: {info['final_fitness']:.6e}")
    print(f"Converged: {info['converged']}")
    print(f"Iterations: {info['n_iterations']}")
    print(f"Time: {info['elapsed_time_seconds']:.2f} seconds")
    
    # Diversity metrics
    diversity = optimizer.get_diversity_metrics()
    print("\nFinal swarm diversity:")
    print(f"  Std(P): {diversity['std_p']:.4f}")
    print(f"  Std(Q): {diversity['std_q']:.4f}")