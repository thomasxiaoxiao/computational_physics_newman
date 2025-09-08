#!/usr/bin/env python3
"""
dla_simulator.py

Core Diffusion-Limited Aggregation simulation class.
This module contains the DLASimulation class without visualization dependencies,
providing the essential simulation logic for Exercise 10.13.

Based on Newman's "Computational Physics" textbook.
"""

import numpy as np
from random import choice


class DLASimulation:
    """
    Core class for Diffusion-Limited Aggregation simulation.
    
    Particles start at the center and perform random walks until they stick to either:
    1. The edge of the system, or 
    2. An already anchored particle
    """
    
    def __init__(self, lattice_size=101):
        """
        Initialize the DLA simulation.
        
        Args:
            lattice_size (int): Size of the square lattice (should be odd)
        """
        self.L = lattice_size
        self.center = lattice_size // 2
        
        # Grid to track anchored particles (0 = empty, 1 = anchored)
        self.anchored_grid = np.zeros((lattice_size, lattice_size), dtype=int)
        
        # List to store positions of anchored particles
        self.anchored_positions = []
        
        # Possible moves: up, down, left, right
        self.moves = [(0, 1), (0, -1), (-1, 0), (1, 0)]
        
        # Current particle position (for visualization)
        self.current_particle_pos = None
        
    def is_at_edge(self, x, y):
        """
        Check if position (x, y) is at the edge of the lattice.
        
        Args:
            x, y (int): Position coordinates
            
        Returns:
            bool: True if at edge, False otherwise
        """
        return x == 0 or x == self.L-1 or y == 0 or y == self.L-1
    
    def is_next_to_anchored(self, x, y):
        """
        Check if position (x, y) is adjacent to an anchored particle.
        
        Args:
            x, y (int): Position coordinates
            
        Returns:
            bool: True if adjacent to anchored particle, False otherwise
        """
        for dx, dy in self.moves:
            nx, ny = x + dx, y + dy
            if 0 <= nx < self.L and 0 <= ny < self.L:
                if self.anchored_grid[nx, ny] == 1:
                    return True
        return False
    
    def should_stick(self, x, y):
        """
        Determine if a particle at position (x, y) should stick.
        
        Args:
            x, y (int): Position coordinates
            
        Returns:
            bool: True if particle should stick, False otherwise
        """
        return self.is_at_edge(x, y) or self.is_next_to_anchored(x, y)
    
    def anchor_particle(self, x, y):
        """
        Anchor a particle at position (x, y).
        
        Args:
            x, y (int): Position coordinates
        """
        self.anchored_grid[x, y] = 1
        self.anchored_positions.append((x, y))
    
    def is_center_blocked(self):
        """
        Check if the center position is blocked by an anchored particle.
        
        Returns:
            bool: True if center is blocked, False otherwise
        """
        return self.anchored_grid[self.center, self.center] == 1
    
    def random_walk_single_particle(self, max_steps=100000):
        """
        Perform random walk for a single particle starting from center.
        
        Args:
            max_steps (int): Maximum number of steps before giving up
            
        Returns:
            tuple: (path_x, path_y, success) where success indicates if particle stuck
        """
        # Check if center is already occupied
        if self.anchored_grid[self.center, self.center] == 1:
            return [], [], False
        
        # Start at center
        x, y = self.center, self.center
        self.current_particle_pos = (x, y)
        
        # Store path
        path_x = [x]
        path_y = [y]
        
        for step in range(max_steps):
            # Check if particle should stick
            if self.should_stick(x, y):
                return path_x, path_y, True
            
            # Choose random direction and move
            attempts = 0
            while attempts < 10:  # Prevent infinite loops
                dx, dy = choice(self.moves)
                new_x, new_y = x + dx, y + dy
                
                # Check bounds and if destination is not occupied
                if (0 <= new_x < self.L and 0 <= new_y < self.L and 
                    self.anchored_grid[new_x, new_y] == 0):
                    x, y = new_x, new_y
                    self.current_particle_pos = (x, y)
                    path_x.append(x)
                    path_y.append(y)
                    break
                    
                attempts += 1
            
            if attempts >= 10:
                # Couldn't move, consider this a failure
                break
        
        # Failed to stick within max_steps
        return path_x, path_y, False
    
    def run_single_particle(self):
        """
        Run simulation for a single particle.
        
        Returns:
            tuple: (path_x, path_y, stuck_position) or (None, None, None) if failed
        """
        path_x, path_y, success = self.random_walk_single_particle()
        
        if success and len(path_x) > 0:
            stuck_pos = (path_x[-1], path_y[-1])
            self.anchor_particle(stuck_pos[0], stuck_pos[1])
            return path_x, path_y, stuck_pos
        else:
            return None, None, None
    
    def run_simulation(self, max_particles=100, verbose=True):
        """
        Run the complete DLA simulation.
        
        Args:
            max_particles (int): Maximum number of particles to simulate
            verbose (bool): Print progress information
            
        Returns:
            list: List of successful particle paths
        """
        successful_paths = []
        
        for particle_num in range(max_particles):
            if verbose and particle_num % 10 == 0:
                print(f"Particle {particle_num + 1}/{max_particles}")
            
            # Check if center is blocked
            if self.is_center_blocked():
                if verbose:
                    print(f"Center blocked after {particle_num} particles. Stopping simulation.")
                break
            
            # Run single particle
            path_x, path_y, stuck_pos = self.run_single_particle()
            
            if stuck_pos is not None:
                successful_paths.append((path_x, path_y, stuck_pos))
            elif verbose:
                print(f"Particle {particle_num + 1} failed to stick")
        
        if verbose:
            print(f"Simulation complete: {len(self.anchored_positions)} particles anchored")
            
        return successful_paths
    
    def get_statistics(self):
        """
        Get basic statistics about the current DLA structure.
        
        Returns:
            dict: Dictionary containing simulation statistics
        """
        if not self.anchored_positions:
            return {"num_particles": 0, "coverage": 0.0, "center_blocked": False}
        
        # Calculate distances from center
        distances = []
        for x, y in self.anchored_positions:
            dist = np.sqrt((x - self.center)**2 + (y - self.center)**2)
            distances.append(dist)
        
        stats = {
            "num_particles": len(self.anchored_positions),
            "lattice_size": self.L,
            "coverage": len(self.anchored_positions) / (self.L * self.L) * 100,
            "mean_distance": np.mean(distances) if distances else 0,
            "max_distance": np.max(distances) if distances else 0,
            "center_blocked": self.is_center_blocked()
        }
        
        return stats
    
    def reset(self):
        """Reset the simulation to initial state."""
        self.anchored_grid.fill(0)
        self.anchored_positions.clear()
        self.current_particle_pos = None