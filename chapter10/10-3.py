#!/usr/bin/env python3
"""
Exercise 10.3: Brownian Motion
Simulation of a particle undergoing Brownian motion on a square lattice.

The particle starts at the center of a 101x101 lattice and performs a random walk
for one million steps. On each step, it chooses a random direction (up, down, left, right)
and moves one step in that direction. If it tries to move outside the lattice bounds,
a new random direction is chosen.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from random import choice

# Constants
L = 101  # Lattice size (odd number so there's exactly one center site)
STEPS = 1_000_000  # Total number of steps to simulate
CENTER = L // 2  # Center position (50 for 101x101 lattice)

def brownian_motion():
    """
    Simulate Brownian motion of a particle on a square lattice.
    
    Returns:
        tuple: Lists of x and y positions throughout the simulation
    """
    # Starting position (center of lattice)
    x, y = CENTER, CENTER
    
    # Lists to store the path
    x_positions = [x]
    y_positions = [y]
    
    # Possible moves: up, down, left, right
    moves = [(0, 1), (0, -1), (-1, 0), (1, 0)]
    
    for step in range(STEPS):
        # Choose a random direction
        while True:
            dx, dy = choice(moves)
            new_x = x + dx
            new_y = y + dy
            
            # Check if the new position is within bounds
            if 0 <= new_x < L and 0 <= new_y < L:
                x, y = new_x, new_y
                break
            # If out of bounds, choose a new random direction (continue loop)
        
        # Store position (only store every 1000 steps to save memory for visualization)
        if step % 1000 == 0:
            x_positions.append(x)
            y_positions.append(y)
    
    return x_positions, y_positions

def visualize_brownian_motion(x_positions, y_positions):
    """
    Create an animated visualization of the Brownian motion.
    
    Args:
        x_positions (list): List of x positions
        y_positions (list): List of y positions
    """
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.set_xlim(0, L-1)
    ax.set_ylim(0, L-1)
    ax.set_xlabel('X Position')
    ax.set_ylabel('Y Position')
    ax.set_title('Brownian Motion on 101×101 Lattice')
    ax.grid(True, alpha=0.3)
    
    # Plot the complete path as a light line
    ax.plot(x_positions, y_positions, 'lightgray', alpha=0.5, linewidth=0.5, label='Complete Path')
    
    # Initialize the moving particle
    particle, = ax.plot([], [], 'ro', markersize=8, label='Current Position')
    trail, = ax.plot([], [], 'b-', alpha=0.7, linewidth=2, label='Recent Trail')
    
    ax.legend(loc="upper right")
    
    def animate(frame):
        # Show the particle position
        particle.set_data([x_positions[frame]], [y_positions[frame]])
        
        # Show trailing path (last 50 points)
        start_idx = max(0, frame - 50)
        trail.set_data(x_positions[start_idx:frame+1], y_positions[start_idx:frame+1])
        
        # Update title with current step
        ax.set_title(f'Brownian Motion - Step {frame * 1000:,}/{STEPS:,}')
        
        return particle, trail
    
    # Create animation
    ani = animation.FuncAnimation(fig, animate, frames=len(x_positions), 
                                interval=50, blit=True, repeat=True)
    
    plt.tight_layout()
    
    # Display the animation
    plt.show()
    return ani

def analyze_motion(x_positions, y_positions):
    """
    Analyze the Brownian motion results.
    
    Args:
        x_positions (list): List of x positions
        y_positions (list): List of y positions
    """
    # Calculate displacement from center
    displacements = []
    for x, y in zip(x_positions, y_positions):
        displacement = np.sqrt((x - CENTER)**2 + (y - CENTER)**2)
        displacements.append(displacement)
    
    # Plot displacement vs time
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Displacement from center
    times = np.arange(0, len(displacements) * 1000, 1000)
    ax1.plot(times, displacements)
    ax1.set_xlabel('Time (steps)')
    ax1.set_ylabel('Distance from Center')
    ax1.set_title('Distance from Center vs Time')
    ax1.grid(True, alpha=0.3)
    
    # Position scatter plot
    ax2.scatter(x_positions[::10], y_positions[::10], alpha=0.5, s=1)
    ax2.plot(CENTER, CENTER, 'ro', markersize=10, label='Starting position')
    ax2.set_xlim(0, L-1)
    ax2.set_ylim(0, L-1)
    ax2.set_xlabel('X Position')
    ax2.set_ylabel('Y Position')
    ax2.set_title('Visited Positions (every 10th point)')
    ax2.legend(loc="upper right")
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    print(f"Simulation completed: {STEPS:,} steps")
    print(f"Final position: ({x_positions[-1]}, {y_positions[-1]})")
    print(f"Final distance from center: {displacements[-1]:.2f}")
    print(f"Maximum distance from center: {max(displacements):.2f}")

def main(show_animation=True):
    """
    Main function to run the Brownian motion simulation.
    
    Args:
        show_animation (bool): Whether to show the animation
    """
    print("Starting Brownian motion simulation...")
    print(f"Lattice size: {L}×{L}")
    print(f"Number of steps: {STEPS:,}")
    print(f"Starting position: ({CENTER}, {CENTER})")
    print()
    
    # Run the simulation
    x_pos, y_pos = brownian_motion()
    
    # Analyze results
    analyze_motion(x_pos, y_pos)
    
    # Create visualization
    ani = None
    if show_animation:
        print("Creating animation...")
        ani = visualize_brownian_motion(x_pos, y_pos)
    
    return x_pos, y_pos, ani


if __name__ == "__main__":
    x_positions, y_positions, animation_obj = main(show_animation=True)