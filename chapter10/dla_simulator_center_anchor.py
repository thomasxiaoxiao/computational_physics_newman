#!/usr/bin/env python3
"""
dla_simulator_center_anchor.py

Exercise 10.13 Part (c): Center-Anchored Diffusion-Limited Aggregation

This implements the more challenging version of DLA where:
- A single particle starts anchored at the center
- New particles start from random points on the perimeter  
- Particles stick only to other anchored particles (not edges)
- Uses optimization: particles start at circle of radius r+1 for efficiency
- Particles are discarded if they wander too far from center

Key features:
- Speed-optimized visualization following 10-13.py pattern
- Automatic backend detection (interactive or file output)
- Color aging option (particles colored by arrival order)
- Smart radius management and stopping conditions
- Comprehensive statistics tracking

Usage:
    # Interactive mode
    python dla_simulator_center_anchor.py
    
    # Programmatic usage
    from dla_simulator_center_anchor import run_center_anchor_optimized, run_quick_demo
    sim = run_quick_demo(lattice_size=51)
    
Based on Newman's "Computational Physics" textbook, Exercise 10.13 part (c).
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.patches import Circle
import matplotlib
from dla_simulator import DLASimulation
from random import choice, uniform
import time
import math


def setup_matplotlib_backend():
    """
    Setup appropriate matplotlib backend for visualization.
    
    Returns:
        bool: True if interactive backend is available, False otherwise
    """
    backend = matplotlib.get_backend().lower()
    
    # Check if we're in an interactive environment
    if backend == 'agg':
        print("Note: Using non-interactive backend (Agg). Plots will be saved instead of displayed.")
        return False
    
    # Try to use interactive backend on macOS
    if 'macosx' in backend:
        try:
            # Test if we can actually display
            fig = plt.figure(figsize=(1, 1))
            plt.close(fig)
            return True
        except Exception:
            print("Warning: Interactive display not available. Using non-interactive mode.")
            matplotlib.use('Agg')
            return False
    
    return True


def show_or_save_plot(filename_prefix="dla_center_anchor", interactive=None):
    """
    Show plot if interactive backend available, otherwise save to file.
    
    Args:
        filename_prefix (str): Prefix for saved filename
        interactive (bool): Force interactive/non-interactive mode
    """
    if interactive is None:
        interactive = matplotlib.get_backend().lower() != 'agg'
    
    if interactive:
        plt.show()
    else:
        filename = f"{filename_prefix}_{int(time.time())}.png"
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        print(f"Plot saved as: {filename}")
        plt.close()


class DLACenterAnchorSimulation(DLASimulation):
    """
    Center-anchored DLA simulation class.
    
    Implements the challenging version where particles start from the perimeter
    and aggregate around a central anchored particle. Uses optimization tricks
    for reasonable simulation speed.
    """
    
    def __init__(self, lattice_size=101):
        """
        Initialize center-anchored DLA simulation.
        
        Args:
            lattice_size (int): Size of the square lattice (should be odd)
        """
        super().__init__(lattice_size)
        
        # Radius tracking: furthest distance of any anchored particle from center
        self.max_radius = 0
        
        # Maximum allowed radius (prevents particles from leaving grid)
        self.max_allowed_radius = lattice_size // 2
        
        # Anchor the initial particle at the center
        self.anchor_particle(self.center, self.center)
        self.max_radius = 0  # Center particle doesn't contribute to radius
        
    def get_distance_from_center(self, x, y):
        """
        Calculate distance from center of lattice.
        
        Args:
            x, y (int): Position coordinates
            
        Returns:
            float: Distance from center
        """
        return math.sqrt((x - self.center)**2 + (y - self.center)**2)
    
    def find_random_perimeter_start(self):
        """
        Find a random starting position on or near a circle of radius r+1.
        
        The optimization from the textbook: instead of starting from the grid boundary,
        start from a circle just outside the current aggregate.
        
        Returns:
            tuple: (x, y) starting position, or None if no valid position found
        """
        # Start at radius r+1 from center
        start_radius = self.max_radius + 1
        
        # If radius is too large, start from boundary
        if start_radius > self.max_allowed_radius:
            return self.find_boundary_start()
        
        # Try to find a valid starting position on the circle
        max_attempts = 50
        for _ in range(max_attempts):
            # Random angle around circle
            angle = uniform(0, 2 * math.pi)
            
            # Calculate position on circle
            x_float = self.center + start_radius * math.cos(angle)
            y_float = self.center + start_radius * math.sin(angle)
            
            # Round to nearest grid point
            x = int(round(x_float))
            y = int(round(y_float))
            
            # Check if position is valid and unoccupied
            if (0 <= x < self.L and 0 <= y < self.L and 
                self.anchored_grid[x, y] == 0):
                return x, y
        
        # Fallback: find any valid position near the circle
        return self.find_boundary_start()
    
    def find_boundary_start(self):
        """
        Fallback: find a random starting position on the grid boundary.
        
        Returns:
            tuple: (x, y) boundary position or None if no valid position found
        """
        # Try multiple times to find a valid boundary position
        max_attempts = 20
        
        for _ in range(max_attempts):
            # Choose random edge
            edge = choice(['top', 'bottom', 'left', 'right'])
            
            if edge == 'top':
                x, y = choice(range(self.L)), self.L - 1
            elif edge == 'bottom':
                x, y = choice(range(self.L)), 0
            elif edge == 'left':
                x, y = 0, choice(range(self.L))
            else:  # right
                x, y = self.L - 1, choice(range(self.L))
            
            # Ensure position is unoccupied
            if self.anchored_grid[x, y] == 0:
                return x, y
        
        # If we can't find any boundary position, return None
        return None
    
    def should_stick(self, x, y):
        """
        Determine if a particle should stick (center-anchor version).
        
        In center-anchor DLA, particles only stick to other anchored particles,
        NOT to edges.
        
        Args:
            x, y (int): Position coordinates
            
        Returns:
            bool: True if particle should stick to an anchored particle
        """
        return self.is_next_to_anchored(x, y)
    
    def should_discard_particle(self, x, y):
        """
        Check if particle should be discarded for wandering too far.
        
        Particles are discarded if they get more than 2r away from center.
        This prevents infinite wandering and speeds up simulation.
        
        Args:
            x, y (int): Position coordinates
            
        Returns:
            bool: True if particle should be discarded
        """
        distance = self.get_distance_from_center(x, y)
        # Use a more generous discard radius: 3 * max_radius + lattice_size/8
        # This accounts for the fact that particles start at radius r+1
        base_discard_radius = 3 * max(self.max_radius, 1)
        lattice_buffer = self.L / 8  # Additional buffer based on lattice size
        discard_radius = base_discard_radius + lattice_buffer
        return distance > discard_radius
    
    def update_max_radius(self, x, y):
        """
        Update the maximum radius if this particle is further from center.
        
        Args:
            x, y (int): Position of newly anchored particle
        """
        distance = self.get_distance_from_center(x, y)
        if distance > self.max_radius:
            self.max_radius = distance
    
    def is_simulation_complete(self):
        """
        Check if simulation should stop.
        
        Simulation stops when max_radius exceeds half the distance from center to boundary.
        
        Returns:
            bool: True if simulation should stop
        """
        return self.max_radius > self.max_allowed_radius
    
    def random_walk_center_anchor(self, max_steps=100000):
        """
        Perform random walk for center-anchor DLA.
        
        Args:
            max_steps (int): Maximum steps before giving up
            
        Returns:
            tuple: (path_x, path_y, success) where success indicates if particle stuck
        """
        # Find starting position on perimeter or circle
        start_pos = self.find_random_perimeter_start()
        if start_pos is None:
            return [], [], False
        
        x, y = start_pos
        self.current_particle_pos = (x, y)
        
        # Store path
        path_x = [x]
        path_y = [y]
        
        for _ in range(max_steps):
            # Check if particle should stick
            if self.should_stick(x, y):
                return path_x, path_y, True
            
            # Check if particle should be discarded (wandered too far)
            if self.should_discard_particle(x, y):
                return path_x, path_y, False
            
            # Choose random direction and move
            attempts = 0
            while attempts < 10:
                dx, dy = choice(self.moves)
                new_x, new_y = x + dx, y + dy
                
                # Check bounds and if destination is unoccupied
                if (0 <= new_x < self.L and 0 <= new_y < self.L and 
                    self.anchored_grid[new_x, new_y] == 0):
                    x, y = new_x, new_y
                    self.current_particle_pos = (x, y)
                    path_x.append(x)
                    path_y.append(y)
                    break
                
                attempts += 1
            
            if attempts >= 10:
                # Couldn't move, discard particle
                return path_x, path_y, False
        
        # Failed to stick within max_steps
        return path_x, path_y, False
    
    def run_single_particle_center_anchor(self):
        """
        Run simulation for a single particle (center-anchor version).
        
        Returns:
            tuple: (path_x, path_y, stuck_position) or (None, None, None) if failed
        """
        path_x, path_y, success = self.random_walk_center_anchor()
        
        if success and len(path_x) > 0:
            stuck_pos = (path_x[-1], path_y[-1])
            self.anchor_particle(stuck_pos[0], stuck_pos[1])
            self.update_max_radius(stuck_pos[0], stuck_pos[1])
            return path_x, path_y, stuck_pos
        else:
            return None, None, None
    
    def run_center_anchor_simulation(self, max_particles=1000, verbose=True):
        """
        Run the complete center-anchor DLA simulation.
        
        Args:
            max_particles (int): Maximum number of particles to simulate
            verbose (bool): Print progress information
            
        Returns:
            list: List of successful particle paths
        """
        successful_paths = []
        
        for particle_num in range(max_particles):
            if verbose and particle_num % 25 == 0:
                print(f"Particle {particle_num + 1}/{max_particles}, radius: {self.max_radius:.1f}")
            
            # Check stopping condition
            if self.is_simulation_complete():
                if verbose:
                    print(f"Maximum radius reached after {particle_num} particles. Stopping.")
                break
            
            # Run single particle
            path_x, path_y, stuck_pos = self.run_single_particle_center_anchor()
            
            if stuck_pos is not None:
                successful_paths.append((path_x, path_y, stuck_pos))
            
        if verbose:
            print(f"Center-anchor simulation complete: {len(self.anchored_positions)} particles anchored")
            print(f"Final radius: {self.max_radius:.2f}")
            
        return successful_paths
    
    def get_center_anchor_statistics(self):
        """
        Get statistics for center-anchor DLA.
        
        Returns:
            dict: Dictionary containing simulation statistics
        """
        base_stats = self.get_statistics()
        
        # Add center-anchor specific statistics
        center_stats = {
            "simulation_type": "center_anchor",
            "max_radius": self.max_radius,
            "max_allowed_radius": self.max_allowed_radius, 
            "radius_efficiency": self.max_radius / self.max_allowed_radius * 100,
            "is_complete": self.is_simulation_complete()
        }
        
        # Merge statistics
        return {**base_stats, **center_stats}


def run_center_anchor_optimized(lattice_size=101, max_particles=1000, use_color_aging=True):
    """
    Run center-anchor DLA with speed optimization (refactored to match 10-13.py pattern).
    
    Args:
        lattice_size (int): Size of the lattice
        max_particles (int): Maximum number of particles
        use_color_aging (bool): Color particles by arrival order
        
    Returns:
        DLACenterAnchorSimulation: The completed simulation
    """
    # Validate lattice size (ensure odd for true center)
    if lattice_size % 2 == 0:
        lattice_size += 1
        print(f"Adjusted lattice size to {lattice_size} (must be odd for true center)")
    
    # Setup backend
    interactive = setup_matplotlib_backend()
    
    print("\n" + "=" * 70)
    print("CENTER-ANCHOR DLA: SPEED OPTIMIZED (Exercise 10.13 Part c)")
    print("=" * 70)
    print("Optimizations for speed:")
    print("• Shows ONLY anchored particles (no random walk visualization)")
    print("• Screen updates ONLY when particles stick (not during walks)")
    print("• NO animation delays or rate limiting")
    print("• Stops when radius exceeds lattice_size/4")
    if use_color_aging:
        print("• Particles colored by age: Red (newest) → Blue (oldest)")
    print(f"• Lattice size: {lattice_size}×{lattice_size}")
    print(f"• Maximum particles: {max_particles}")
    print(f"• Display mode: {'Interactive' if interactive else 'Save to file'}")
    print()
    
    # Initialize center-anchor simulation
    dla = DLACenterAnchorSimulation(lattice_size)
    
    # Setup minimal, fast visualization (following 10-13.py pattern exactly)
    plt.ion()
    _, ax = plt.subplots(figsize=(12, 10))
    ax.set_xlim(-1, lattice_size)
    ax.set_ylim(-1, lattice_size)
    ax.set_xlabel('X Position')
    ax.set_ylabel('Y Position')
    ax.set_title(f'Center-Anchor DLA: Speed Optimized ({lattice_size}×{lattice_size})')
    ax.grid(True, alpha=0.2)  # Lighter grid for speed
    ax.set_aspect('equal')
    
    # Adjust layout to make room for legend
    plt.tight_layout()
    plt.subplots_adjust(right=0.85)
    
    # Mark center anchor (already placed in simulation)
    center_marker = ax.plot(dla.center, dla.center, 'go', markersize=8,
                           markeredgecolor='darkgreen', markeredgewidth=2,
                           label='Center anchor', zorder=10)
    
    # Lattice boundaries (minimal styling for speed)
    ax.axhline(y=0, color='black', linewidth=1.5, alpha=0.6)
    ax.axhline(y=lattice_size-1, color='black', linewidth=1.5, alpha=0.6)
    ax.axvline(x=0, color='black', linewidth=1.5, alpha=0.6)
    ax.axvline(x=lattice_size-1, color='black', linewidth=1.5, alpha=0.6)
    
    # Initialize particle display (will be updated with colors)
    particles_scatter = ax.scatter([], [], s=30, alpha=0.8, zorder=5)
    
    # Add radius circle for center-anchor specific visualization
    radius_circle = Circle((dla.center, dla.center), dla.max_radius,
                          fill=False, color='cyan', alpha=0.5,
                          linestyle='--', linewidth=2)
    ax.add_patch(radius_circle)
    
    if use_color_aging:
        ax.legend([center_marker[0], particles_scatter, radius_circle],
                 ['Center anchor', 'Particles (Red=newest, Blue=oldest)', 'Max radius'],
                 loc='upper left', bbox_to_anchor=(1.02, 1))
    else:
        particles_scatter.set_color('red')
        ax.legend([center_marker[0], particles_scatter, radius_circle],
                 ['Center anchor', 'Anchored particles', 'Max radius'],
                 loc='upper left', bbox_to_anchor=(1.02, 1))
    
    # Data storage (following 10-13.py pattern)
    anchored_positions = []
    particle_ages = []  # Track order of arrival for coloring
    
    # Run maximum speed simulation (following 10-13.py pattern exactly)
    particle_count = 0
    start_time = time.time()
    
    print("Starting center-anchor simulation...")
    print("(This will run as fast as possible with minimal screen updates)")
    
    while particle_count < max_particles:
        # Critical stopping condition: maximum radius reached
        if dla.is_simulation_complete():
            print(f"\n*** MAXIMUM RADIUS REACHED after {particle_count} particles ***")
            print(f"Final radius: {dla.max_radius:.2f}, max allowed: {dla.max_allowed_radius}")
            print("Simulation complete - radius limit exceeded")
            break
        
        # Run single particle (no intermediate visualization)
        _, _, stuck_pos = dla.run_single_particle_center_anchor()
        
        if stuck_pos is not None:
            particle_count += 1
            anchored_positions.append(stuck_pos)
            particle_ages.append(particle_count)  # Store arrival order
            
            # Update visualization ONLY when particle sticks (key optimization)
            positions_x, positions_y = zip(*anchored_positions)
            particles_scatter.set_offsets(list(zip(positions_x, positions_y)))
            
            # Apply color aging if enabled
            if use_color_aging and len(particle_ages) > 1:
                # Create color map: oldest particles = blue, newest = red
                colors = cm.coolwarm(np.linspace(0, 1, len(particle_ages)))
                particles_scatter.set_color(colors)
            
            # Update radius circle
            radius_circle.set_radius(dla.max_radius)
            
            # Minimal title update (no rate calculations for max speed)
            ax.set_title(f'Center-Anchor DLA: {particle_count} particles anchored, '
                        f'radius = {dla.max_radius:.1f}/{dla.max_allowed_radius} '
                        f'({lattice_size}×{lattice_size})')
            
            # Fastest possible screen update (NO pause/delay)
            plt.draw()
            
            # Minimal progress reporting (only every 25 particles for center-anchor)
            if particle_count % 25 == 0:
                elapsed = time.time() - start_time
                print(f"  {particle_count} particles, radius: {dla.max_radius:.1f}/{dla.max_allowed_radius}, "
                      f"time: {elapsed:.1f}s")
    
    elapsed_total = time.time() - start_time
    
    # Final display
    ax.set_title(f'Center-Anchor DLA Complete: {len(anchored_positions)} particles, '
                f'radius = {dla.max_radius:.1f} in {elapsed_total:.1f}s')
    
    # Highlight center if simulation complete
    if dla.is_simulation_complete():
        center_marker[0].set_color('red')
        center_marker[0].set_markersize(12)
    
    plt.draw()
    
    # Comprehensive final statistics (following 10-13.py pattern)
    stats = dla.get_center_anchor_statistics()
    print(f"\n" + "=" * 50)
    print(f"CENTER-ANCHOR DLA RESULTS:")
    print(f"=" * 50)
    print(f"Lattice size: {lattice_size}×{lattice_size}")
    print(f"Total particles anchored: {stats['num_particles']}")
    print(f"Simulation time: {elapsed_total:.2f} seconds")
    if particle_count > 0:
        print(f"Average rate: {particle_count/elapsed_total:.1f} particles/sec")
    print(f"Final radius: {stats['max_radius']:.2f}")
    print(f"Maximum allowed radius: {stats['max_allowed_radius']:.2f}")
    print(f"Radius efficiency: {stats['radius_efficiency']:.1f}%")
    print(f"Mean distance from center: {stats['mean_distance']:.2f}")
    print(f"Maximum distance from center: {stats['max_distance']:.2f}")
    print(f"Simulation complete: {stats['is_complete']}")
    if use_color_aging:
        print(f"Color aging: Enabled (Blue=oldest → Red=newest)")
    
    plt.ioff()
    show_or_save_plot("center_anchor_optimized", interactive)
    return dla


def run_quick_demo(lattice_size=51, max_particles=200):
    """
    Quick demo function for testing center-anchor DLA.
    
    Args:
        lattice_size (int): Size of the lattice (default 51)
        max_particles (int): Maximum particles (default 200)
        
    Returns:
        DLACenterAnchorSimulation: The completed simulation
    """
    print("Running quick center-anchor DLA demo...")
    return run_center_anchor_optimized(
        lattice_size=lattice_size,
        max_particles=max_particles,
        use_color_aging=True
    )



def main():
    """
    Main function for center-anchor DLA simulation.
    """
    print("=" * 70)
    print("Exercise 10.13 Part (c): Center-Anchor DLA")
    print("Based on Newman's Computational Physics textbook")
    print("=" * 70)
    print("This implements the challenging version of DLA where particles")
    print("start from the perimeter and aggregate around a central particle.")
    print("=" * 70)
    
    try:
        print("\nSimulation Parameters:")
        size = int(input("Enter lattice size (default 101): ") or "101")
        if size % 2 == 0:
            size += 1
            print(f"Adjusted to {size} (must be odd for true center)")
        
        max_p = int(input("Enter max particles (default 1000): ") or "1000")
        aging = input("Use color aging (y/n, default y)? ").strip().lower()
        use_aging = aging == '' or aging.startswith('y')
        
        print(f"\nRunning center-anchor DLA simulation:")
        print(f"  Lattice: {size}×{size}")
        print(f"  Max particles: {max_p}")
        print(f"  Color aging: {'Enabled' if use_aging else 'Disabled'}")
        print()
        
        run_center_anchor_optimized(size, max_p, use_aging)
        
    except ValueError:
        print("Error: Invalid input. Please enter valid numbers.")
    except KeyboardInterrupt:
        print("\nSimulation interrupted by user.")
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()