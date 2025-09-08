#!/usr/bin/env python3
"""
Exercise 10.13: Diffusion-Limited Aggregation (DLA)
Building upon Exercise 10.3 (Brownian motion).

Implements both part (a) and part (b) as specified in Newman's textbook.

Part (a): Basic DLA with visualization of both walking and anchored particles
Part (b): Optimized DLA showing only anchored particles for speed

Based on Newman's "Computational Physics" textbook, Exercise 10.13.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from dla_simulator import DLASimulation
import time

# Constants from textbook
LATTICE_SIZE = 101  # 101x101 lattice as specified
MAX_PARTICLES = 100  # Reasonable number for demonstration, part (a) might need a much smaller number


def validate_lattice_size(size):
    """
    Validate and adjust lattice size to ensure it's odd (for a true center).
    
    Args:
        size (int): Requested lattice size
        
    Returns:
        int: Valid odd lattice size
    """
    # Ensure minimum size
    if size < 51:
        print("Warning: Very small lattice, using minimum of 51")
        size = 51
    elif size > 501:
        print("Warning: Very large lattice may be slow, using maximum of 501")
        size = 501
    
    # Ensure odd size for true center
    if size % 2 == 0:
        size += 1
        print(f"Adjusted to {size} (must be odd for a true center)")
    
    return size


def part_a_basic_dla(max_particles=None):
    """
    Exercise 10.13 Part (a): Basic DLA simulation.
    
    Active visualization of random walks based on Brownian motion pattern.
    Shows step-by-step particle movement and anchoring process.
    
    Args:
        max_particles (int): Maximum number of particles to simulate (None uses default)
    """
    print("=" * 60)
    print("EXERCISE 10.13 PART (A): Basic DLA with Active Random Walks")
    print("=" * 60)
    # Use provided max_particles or default
    if max_particles is None:
        max_particles = MAX_PARTICLES
    
    print("Active visualization of particle random walks")
    print("Based on Brownian motion visualization pattern")
    print(f"Lattice size: {LATTICE_SIZE}×{LATTICE_SIZE}")
    print(f"Maximum particles: {max_particles}")
    print()
    
    # Initialize DLA simulation
    dla = DLASimulation(LATTICE_SIZE)
    
    # Setup interactive visualization (based on Brownian motion)
    plt.ion()  # Interactive mode for real-time updates
    fig, ax = plt.subplots(figsize=(12, 10))
    ax.set_xlim(-1, LATTICE_SIZE)
    ax.set_ylim(-1, LATTICE_SIZE)
    ax.set_xlabel('X Position')
    ax.set_ylabel('Y Position')
    ax.set_title('DLA Part (a): Active Random Walk Visualization')
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')
    
    # Draw lattice boundaries
    ax.axhline(y=0, color='black', linewidth=2, alpha=0.8, label='Lattice boundary')
    ax.axhline(y=LATTICE_SIZE-1, color='black', linewidth=2, alpha=0.8)
    ax.axvline(x=0, color='black', linewidth=2, alpha=0.8)
    ax.axvline(x=LATTICE_SIZE-1, color='black', linewidth=2, alpha=0.8)
    
    # Mark center starting position
    center_marker = ax.plot(dla.center, dla.center, 'go', markersize=10, 
                           markeredgecolor='darkgreen', markeredgewidth=2,
                           label='Center (start)', zorder=10)
    
    # Initialize visualization elements
    anchored_scatter = ax.scatter([], [], c='red', s=50, alpha=0.9, 
                                 edgecolors='darkred', linewidth=1,
                                 label='Anchored particles', zorder=6)
    
    # Current walking particle (animated like Brownian motion)
    walking_particle, = ax.plot([], [], 'bo', markersize=10, 
                              markeredgecolor='darkblue', markeredgewidth=2,
                              label='Walking particle', zorder=8)
    
    # Current particle trail (fading as in Brownian motion)
    current_trail, = ax.plot([], [], 'b-', alpha=0.8, linewidth=3, 
                           label='Current trail', zorder=4)
    
    # Recent walking trail (shorter, more visible)
    recent_trail, = ax.plot([], [], 'cyan', alpha=0.6, linewidth=2, 
                          label='Recent steps', zorder=3)
    
    # All completed paths (background)
    all_paths, = ax.plot([], [], 'lightgray', alpha=0.3, linewidth=0.8,
                        label='All paths', zorder=1)
    
    ax.legend(loc='upper right', fontsize=9)
    
    # Data storage for visualization
    anchored_positions = []
    all_completed_paths_x = []
    all_completed_paths_y = []
    
    # Main simulation loop with active visualization
    for particle_num in range(max_particles):
        print(f"\nParticle {particle_num + 1}/{max_particles}: Starting random walk...")
        
        # Check stopping condition
        if dla.is_center_blocked():
            print(f"Center blocked after {particle_num} particles. Stopping.")
            break
        
        # Perform random walk with step-by-step visualization
        path_x, path_y, success = dla.random_walk_single_particle(max_steps=50000)
        
        if success and len(path_x) > 1:
            print(f"  Walking {len(path_x)} steps...")
            
            # Animate the random walk (like Brownian motion)
            animation_speed = max(1, len(path_x) // 50)  # Show ~50 animation frames
            
            for step in range(0, len(path_x), animation_speed):
                end_step = min(step + animation_speed, len(path_x))
                
                # Update walking particle position
                current_x, current_y = path_x[end_step-1], path_y[end_step-1]
                walking_particle.set_data([current_x], [current_y])
                
                # Update current trail (full path so far)
                current_trail.set_data(path_x[:end_step], path_y[:end_step])
                
                # Update recent trail (last 20 steps for emphasis)
                recent_start = max(0, end_step - 20)
                recent_trail.set_data(path_x[recent_start:end_step], path_y[recent_start:end_step])
                
                # Update title with current status
                ax.set_title(f'DLA Part (a) - Particle {particle_num + 1} walking '
                           f'(step {end_step}/{len(path_x)}) | '
                           f'{len(anchored_positions)} anchored')
                
                plt.draw()
                plt.pause(0.02)  # Smooth animation timing
            
            # Particle sticks - anchor it
            final_pos = (path_x[-1], path_y[-1])
            dla.anchor_particle(final_pos[0], final_pos[1])
            anchored_positions.append(final_pos)
            
            print(f"  Particle anchored at ({final_pos[0]}, {final_pos[1]})")
            
            # Update anchored particles display
            if anchored_positions:
                anchored_x, anchored_y = zip(*anchored_positions)
                anchored_scatter.set_offsets(list(zip(anchored_x, anchored_y)))
            
            # Add to complete paths collection
            all_completed_paths_x.extend(path_x + [np.nan])  # NaN for path separation
            all_completed_paths_y.extend(path_y + [np.nan])
            all_paths.set_data(all_completed_paths_x, all_completed_paths_y)
            
            # Clear walking particle animation
            walking_particle.set_data([], [])
            current_trail.set_data([], [])
            recent_trail.set_data([], [])
            
            # Show anchored state briefly
            ax.set_title(f'DLA Part (a) - Particle {particle_num + 1} ANCHORED! '
                        f'({len(anchored_positions)} total)')
            plt.draw()
            plt.pause(0.3)  # Brief pause to show anchoring
            
        else:
            print(f"  Particle {particle_num + 1} failed to stick")
            # Clear any partial visualization
            walking_particle.set_data([], [])
            current_trail.set_data([], [])
            recent_trail.set_data([], [])
            plt.draw()
    
    # Final display
    ax.set_title(f'DLA Part (a) Complete - {len(anchored_positions)} particles anchored')
    plt.draw()
    
    # Print final statistics
    stats = dla.get_statistics()
    print(f"\nPart (a) Results:")
    print(f"Total anchored particles: {stats['num_particles']}")
    print(f"Lattice coverage: {stats['coverage']:.3f}%")
    print(f"Center blocked: {stats['center_blocked']}")
    
    plt.ioff()
    plt.show()
    return dla


def part_b_optimized_dla(lattice_size=None, use_color_aging=True, max_particles=None):
    """
    Exercise 10.13 Part (b): Optimized DLA simulation.
    
    Optimized for maximum speed:
    - Shows ONLY anchored particles (no walking particles)  
    - Updates screen ONLY when particles become anchored
    - NO animation rate limiting for speed
    - Stops when center is blocked
    - Optional color aging: particles colored by arrival order
    
    Args:
        lattice_size (int): Override default lattice size (None uses LATTICE_SIZE)
        use_color_aging (bool): Color particles by age (newest=red, oldest=blue)
        max_particles (int): Maximum number of particles to simulate (None uses size*2)
    """
    # Use custom lattice size if provided, otherwise use global constant
    if lattice_size is not None:
        size = validate_lattice_size(lattice_size)
    else:
        size = LATTICE_SIZE  # Already odd (101)
    
    # Use provided max_particles or default scaling
    if max_particles is None:
        max_particles = size * 2  # Scale max particles with lattice size
    
    print("\n" + "=" * 70)
    print("EXERCISE 10.13 PART (B): MAXIMUM SPEED OPTIMIZED DLA")
    print("=" * 70)
    print("Optimizations for speed:")
    print("• Shows ONLY anchored particles (no random walk visualization)")
    print("• Screen updates ONLY when particles stick (not during walks)")
    print("• NO animation delays or rate limiting")
    print("• Stops immediately when center becomes blocked")
    if use_color_aging:
        print("• Particles colored by age: Red (newest) → Blue (oldest)")
    print(f"• Lattice size: {size}×{size}")
    print(f"• Maximum particles: {max_particles}")
    print()
    
    # Initialize DLA simulation with specified size
    dla = DLASimulation(size)
    
    # Setup minimal, fast visualization with space for external legend
    plt.ion()
    fig, ax = plt.subplots(figsize=(12, 10))  # Wider to accommodate legend
    ax.set_xlim(-1, size)
    ax.set_ylim(-1, size)
    ax.set_xlabel('X Position')
    ax.set_ylabel('Y Position')
    ax.set_title(f'DLA Part (b): Speed Optimized ({size}×{size})')
    ax.grid(True, alpha=0.2)  # Lighter grid for speed
    ax.set_aspect('equal')
    
    # Adjust layout to make room for legend
    plt.tight_layout()
    plt.subplots_adjust(right=0.85)
    
    # Mark center (starting point)
    center_marker = ax.plot(dla.center, dla.center, 'go', markersize=8, 
                           markeredgecolor='darkgreen', markeredgewidth=2,
                           label='Center (start)', zorder=10)
    
    # Lattice boundaries (minimal styling for speed)
    ax.axhline(y=0, color='black', linewidth=1.5, alpha=0.6)
    ax.axhline(y=size-1, color='black', linewidth=1.5, alpha=0.6)
    ax.axvline(x=0, color='black', linewidth=1.5, alpha=0.6)
    ax.axvline(x=size-1, color='black', linewidth=1.5, alpha=0.6)
    
    # Initialize particle display (will be updated with colors)
    particles_scatter = ax.scatter([], [], s=30, alpha=0.8, zorder=5)
    
    if use_color_aging:
        ax.legend([center_marker[0], particles_scatter], 
                 ['Center (start)', 'Particles (Red=newest, Blue=oldest)'],
                 loc='upper left', bbox_to_anchor=(1.02, 1))
    else:
        particles_scatter.set_color('red')
        ax.legend([center_marker[0], particles_scatter], 
                 ['Center (start)', 'Anchored particles'],
                 loc='upper left', bbox_to_anchor=(1.02, 1))
    
    # Data storage
    anchored_positions = []
    particle_ages = []  # Track order of arrival for coloring
    
    # Run maximum speed simulation
    particle_count = 0
    start_time = time.time()
    
    print(f"Starting speed-optimized simulation...")
    print("(This will run as fast as possible with minimal screen updates)")
    
    while particle_count < max_particles:
        # Critical stopping condition: center blocked
        if dla.is_center_blocked():
            print(f"\n*** CENTER BLOCKED after {particle_count} particles ***")
            print("Simulation complete - no more particles can be added")
            break
        
        # Run single particle (no intermediate visualization)
        _, _, stuck_pos = dla.run_single_particle()
        
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
            
            # Minimal title update (no rate calculations for max speed)
            ax.set_title(f'DLA Part (b): {particle_count} particles anchored '
                        f'({size}×{size} lattice)')
            
            # Fastest possible screen update (NO pause/delay)
            plt.draw()
            
            # Minimal progress reporting (only every 50 particles)
            if particle_count % 50 == 0:
                elapsed = time.time() - start_time
                print(f"  {particle_count} particles in {elapsed:.1f}s")
    
    elapsed_total = time.time() - start_time
    
    # Final display
    ax.set_title(f'DLA Complete: {len(anchored_positions)} particles '
                f'({size}×{size}) in {elapsed_total:.1f}s')
    
    # Highlight center if blocked
    if dla.is_center_blocked():
        center_marker[0].set_color('red')
        center_marker[0].set_markersize(12)
    
    plt.draw()
    
    # Comprehensive final statistics
    stats = dla.get_statistics()
    print(f"\n" + "=" * 50)
    print(f"PART (B) SPEED-OPTIMIZED RESULTS:")
    print(f"=" * 50)
    print(f"Lattice size: {size}×{size}")
    print(f"Total particles anchored: {stats['num_particles']}")
    print(f"Simulation time: {elapsed_total:.2f} seconds")
    print(f"Average rate: {stats['num_particles']/elapsed_total:.1f} particles/sec")
    print(f"Lattice coverage: {stats['coverage']:.4f}%")
    print(f"Mean distance from center: {stats['mean_distance']:.2f}")
    print(f"Maximum distance from center: {stats['max_distance']:.2f}")
    print(f"Center blocked: {stats['center_blocked']}")
    if use_color_aging:
        print(f"Color aging: Enabled (Blue=oldest → Red=newest)")
    
    plt.ioff()
    plt.show()
    return dla


def run_part_a():
    """Convenience function to run part (a) with active visualization."""
    return part_a_basic_dla()



def run_part_b_custom(size, color_aging=True):
    """Run part (b) with custom validated lattice size and color options."""
    return part_b_optimized_dla(lattice_size=size, use_color_aging=color_aging)



def main():
    """
    Main function with enhanced user choice interface.
    """
    print("=" * 70)
    print("Exercise 10.13: Diffusion-Limited Aggregation (DLA)")
    print("Based on Newman's Computational Physics textbook")
    print("=" * 70)
    print("Choose which simulation to run:")
    print("  (a) Part A: Active random walk visualization")
    print("  (b) Part B: Speed-optimized DLA (custom size & options)")
    print("=" * 70)
    
    while True:
        choice = input("Enter your choice (a/b): ").strip().lower()
        
        if choice == 'a':
            print("\nPart A: Active random walk visualization")
            print("Shows step-by-step particle movement and anchoring")
            try:
                max_particles = input(f"Enter max particles (default {MAX_PARTICLES}): ").strip()
                if max_particles:
                    max_particles = int(max_particles)
                else:
                    max_particles = None
                print(f"\nRunning Part A with {max_particles or MAX_PARTICLES} max particles...")
                part_a_basic_dla(max_particles=max_particles)
                break
            except ValueError:
                print("Invalid number. Please enter a valid number.")
        elif choice == 'b':
            print("\nPart B: Speed-optimized DLA with custom options")
            print("Size suggestions:")
            print("  • 101 (standard, fast)")
            print("  • 201 (impressive, as recommended in textbook)")
            print("  • 301 (very large, slow but spectacular)")
            try:
                size = int(input("Enter lattice size: "))
                size = validate_lattice_size(size)  # Ensure odd size with proper center
                
                max_particles = input(f"Enter max particles (default {size * 2}): ").strip()
                if max_particles:
                    max_particles = int(max_particles)
                else:
                    max_particles = None
                
                aging = input("Use color aging (particles colored by age) (y/n)? ").strip().lower().startswith('y')
                print(f"\nRunning Part B: {size}×{size} lattice, {max_particles or size * 2} max particles, color aging: {aging}")
                part_b_optimized_dla(lattice_size=size, use_color_aging=aging, max_particles=max_particles)
                break
            except ValueError:
                print("Invalid input. Please enter valid numbers.")
        else:
            print("Invalid choice. Please enter 'a' or 'b'.")


if __name__ == "__main__":
    main()