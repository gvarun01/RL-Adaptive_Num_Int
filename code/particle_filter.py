"""
This module contains classes for implementing Particle Filter Extended
Finite Element Method (PFEM) integration.

It provides the `Particle` class to represent individual particles used in
the PFEM simulation and the `PFEMIntegrator` class to manage the collection
of particles and perform the integration process. This method is particularly
useful for integrating functions that may be complex, non-smooth, or where
the function's behavior changes significantly across the domain.
"""
import numpy as np


class Particle:
    """
    Represents a single particle used in the Particle Filter Extended Finite
    Element Method (PFEM) integration.

    Each particle has a position (x, y), a weight, an associated function value,
    a list of neighboring particles, and an estimated local error. These
    attributes are used by the PFEMIntegrator to adaptively refine the
    particle distribution and estimate the integral.

    Attributes:
        x (float): The x-coordinate of the particle.
        y (float): The y-coordinate of the particle.
        weight (float): The weight associated with the particle, used in
                        weighted sums for integration. Defaults to 1.0.
        value (float, optional): The value of the function being integrated,
                                 evaluated at the particle's (x, y) position.
                                 Initialized to None.
        neighbors (list): A list of other `Particle` objects that are considered
                          neighbors to this particle, typically within a certain
                          distance. Initialized to an empty list.
        error_estimate (float): An estimate of the local error in the function
                                approximation around this particle, often derived
                                from the variance of function values among its
                                neighbors. Initialized to 0.0.
    """
    def __init__(self, x: float, y: float, weight: float = 1.0):
        self.x = x
        self.y = y
        self.weight = weight
        self.value = None  # Will be set when function is evaluated at this particle
        self.neighbors = []
        self.error_estimate = 0.0


class PFEMIntegrator:
    """
    Handles the Particle Filter Extended Finite Element Method (PFEM) based
    integration for a given 2D function.

    This integrator manages a collection of `Particle` objects, distributing
    them across the integration domain, evaluating the function at particle
    locations, and adapting the particle distribution based on local error
    estimates to improve integration accuracy.

    Attributes:
        function (Callable[[float, float], float]): The 2D function to be
                                                    integrated.
        min_particles (int): The minimum number of particles to maintain during
                             the simulation.
        max_particles (int): The maximum number of particles allowed in the
                             simulation to manage computational cost.
        particles (list[Particle]): A list of `Particle` objects currently active
                                   in the simulation.
    """
    def __init__(self, function, min_particles=20, max_particles=100):
        self.function = function
        self.min_particles = min_particles
        self.max_particles = max_particles
        self.particles = []

    def initialize_particles(self, x0: float, x1: float, y0: float, y1: float, n_initial: int = 20):
        """
        Initializes the particle distribution within the given rectangular domain.

        Particles are typically placed on a jittered grid to cover the domain
        initially.

        Args:
            x0 (float): The lower x-bound of the integration domain.
            x1 (float): The upper x-bound of the integration domain.
            y0 (float): The lower y-bound of the integration domain.
            y1 (float): The upper y-bound of the integration domain.
            n_initial (int): The approximate number of initial particles to create.
                             The actual number will be `nx * ny`.
        """
        nx = ny = int(np.sqrt(n_initial))
        self.particles = []

        for i in range(nx):
            for j in range(ny):
                # Add jitter to avoid regular grid artifacts
                jitter_x = np.random.uniform(-0.1, 0.1) * (x1 - x0) / nx
                jitter_y = np.random.uniform(-0.1, 0.1) * (y1 - y0) / ny

                x = x0 + (i + 0.5) * (x1 - x0) / nx + jitter_x
                y = y0 + (j + 0.5) * (y1 - y0) / ny + jitter_y

                self.particles.append(Particle(x, y))

    def update_particle_values(self, eval_cache: dict):
        """
        Updates the `value` attribute of each particle by evaluating the
        integrand function at its position.

        Uses a cache to avoid redundant function evaluations.

        Args:
            eval_cache (dict): A dictionary used as a cache for function
                               evaluations. Keys are (x, y) tuples, and
                               values are the function results.
        """
        for p in self.particles:
            if (p.x, p.y) in eval_cache:
                p.value = eval_cache[(p.x, p.y)]
            else:
                p.value = self.function(p.x, p.y)
                eval_cache[(p.x, p.y)] = p.value

    def find_neighbors(self, max_dist: float):
        """
        Identifies neighboring particles for each particle in the simulation.

        A particle `p2` is considered a neighbor of `p1` if the Euclidean
        distance between them is less than `max_dist`.

        Args:
            max_dist (float): The maximum distance for two particles to be
                              considered neighbors.
        """
        for p1 in self.particles:
            p1.neighbors = []
            for p2 in self.particles:
                if p1 != p2:
                    dist = np.sqrt((p1.x - p2.x)**2 + (p1.y - p2.y)**2)
                    if dist < max_dist:
                        p1.neighbors.append(p2)

    def estimate_local_error(self):
        """
        Estimates the local error for each particle.

        The error is typically estimated as the standard deviation of the
        function values of its neighboring particles. This provides a measure
        of the function's variability in the particle's vicinity.
        """
        for p in self.particles:
            if p.neighbors:
                values = [n.value for n in p.neighbors]
                p.error_estimate = np.std(values) if values else 0.0
            else:
                p.error_estimate = 0.0 # Or some other default for isolated particles


    def adapt_particles(self, x0: float, x1: float, y0: float, y1: float):
        """
        Adapts the particle distribution based on their estimated local errors.

        Particles in regions with low error may be removed (resampled), while
        new particles may be added in regions with high error to focus
        computational effort where the function is more complex. The total
        number of particles is kept within `min_particles` and `max_particles`.

        Args:
            x0 (float): The lower x-bound of the domain (for clamping new particles).
            x1 (float): The upper x-bound of the domain.
            y0 (float): The lower y-bound of the domain.
            y1 (float): The upper y-bound of the domain.
        """
        if not self.particles:
            return

        # Calculate median error only if there are particles with error estimates
        error_estimates = [p.error_estimate for p in self.particles if hasattr(p, 'error_estimate')]
        if not error_estimates: # No particles or no error estimates yet
             self.initialize_particles(x0, x1, y0, y1, self.min_particles) # Re-initialize if empty
             return


        median_error = np.median(error_estimates)
        # Remove particles with error estimates below or equal to the median, but ensure min_particles
        survivors = [p for p in self.particles if p.error_estimate > median_error]
        if len(survivors) < self.min_particles // 2 and len(self.particles) > self.min_particles: # Ensure we don't remove too many
            survivors = sorted(self.particles, key=lambda p: p.error_estimate, reverse=True)[:self.min_particles]
        self.particles = survivors


        new_particles = []
        # Add particles in high error regions (e.g., top 25th percentile of remaining errors)
        if self.particles: # Proceed only if there are particles left
            current_error_estimates = [p.error_estimate for p in self.particles]
            if not current_error_estimates: # Should not happen if self.particles is not empty
                 return
            
            high_error_threshold = np.percentile(current_error_estimates, 75)
            for p in self.particles:
                if p.error_estimate > high_error_threshold and len(self.particles) + len(new_particles) < self.max_particles:
                    # Add new particles around high error particle
                    for _ in range(2): # Add two new particles
                        radius = 0.1 * min(x1 - x0, y1 - y0) # Relative radius
                        angle = np.random.uniform(0, 2 * np.pi)
                        new_x = np.clip(p.x + radius * np.cos(angle), x0, x1)
                        new_y = np.clip(p.y + radius * np.sin(angle), y0, y1)
                        new_particles.append(Particle(new_x, new_y))
                        if len(self.particles) + len(new_particles) >= self.max_particles:
                            break
                if len(self.particles) + len(new_particles) >= self.max_particles:
                    break
        
        self.particles.extend(new_particles)

        # Ensure particle count is within min/max bounds
        if len(self.particles) > self.max_particles:
            # Sort by error and keep the ones with highest error
            self.particles = sorted(self.particles, key=lambda p: p.error_estimate, reverse=True)[:self.max_particles]
        elif len(self.particles) < self.min_particles:
            # If below min_particles, add more particles, perhaps in areas of highest error or randomly
            needed = self.min_particles - len(self.particles)
            # Simplified: add random particles for now if adaptation doesn't cover enough
            # A better strategy would be to add them near existing high-error particles or spread them out
            for _ in range(needed):
                px = np.random.uniform(x0, x1)
                py = np.random.uniform(y0, y1)
                self.particles.append(Particle(px, py))
                if len(self.particles) >= self.max_particles: # Should not exceed max_particles
                    break


    def integrate(self, x0: float, x1: float, y0: float, y1: float, eval_cache: dict) -> tuple[float, float]:
        """
        Performs the PFEM integration over the specified rectangular domain.

        This involves updating particle values, finding neighbors, estimating
        local errors, and then computing the integral as a weighted sum of
        particle values. An overall error estimate for the integral is also
        computed.

        Args:
            x0 (float): The lower x-bound of the integration domain.
            x1 (float): The upper x-bound of the integration domain.
            y0 (float): The lower y-bound of the integration domain.
            y1 (float): The upper y-bound of the integration domain.
            eval_cache (dict): A cache for function evaluations.

        Returns:
            tuple[float, float]: A tuple containing:
                - integral (float): The estimated value of the integral.
                - error (float): An estimate of the error in the integral value.
        """
        area = (x1 - x0) * (y1 - y0)
        if not self.particles: # Ensure particles are initialized if list is empty
            self.initialize_particles(x0,x1,y0,y1, self.min_particles)

        self.update_particle_values(eval_cache)
        max_dist = 0.2 * min(x1 - x0, y1 - y0) # Heuristic for neighbor distance
        self.find_neighbors(max_dist)
        self.estimate_local_error()

        if not self.particles: # Should not happen if initialized
            return 0.0, float('inf')

        # Weighted sum of particle values
        total_weight = sum(p.weight for p in self.particles)
        if total_weight == 0: # Avoid division by zero if all weights are zero
            return 0.0, float('inf')

        integral = area * sum(p.value * p.weight for p in self.particles if p.value is not None) / total_weight

        # Overall error estimate based on particle error estimates
        # Ensure error_estimate is a float for all particles.
        particle_errors = [p.error_estimate for p in self.particles if hasattr(p, 'error_estimate') and isinstance(p.error_estimate, float)]
        error = np.mean(particle_errors) if particle_errors else float('inf')

        return integral, error
