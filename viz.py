#place to hold the messy plotting functions

import numpy as np
import matplotlib as plt
from matplotlib.animation import FuncAnimation
from IPython.display import HTML
from Ising_Model import *


def animate_ising(size=50, temperature=2.5, n_steps=10000, interval=50):
    """
    Create an animation of the Ising model evolution.

    Parameters:
    -----------
    size : int
    Lattice size
    temperature : float
    Temperature (critical temperature ~2.27)
    n_steps : int
        Number of Monte Carlo steps
    interval : int
    Delay between frames in milliseconds"""

    # Initialize and run simulation
    model = IsingModel(size=size)
    model.initialize_random()
    states, energies, _ = model.simulate(temperature, n_steps)

    # Set up the figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Spin configuration plot
    im = ax1.imshow(states[0], cmap='YlGnBu', vmin=-1, vmax=1, interpolation='nearest')
    ax1.set_title(f'Spin Configuration (T={temperature:.2f})', fontsize=12)
    ax1.axis('off')

    # Energy plot
    energy_line, = ax2.plot([], [], 'b-', linewidth=2)
    ax2.set_xlim(0, len(states))
    ax2.set_ylim(min(energies) * 1.1, max(energies) * 1.1)
    ax2.set_xlabel('Time Step (×100)')
    ax2.set_ylabel('Energy')
    ax2.set_title('System Energy')
    ax2.grid(True, alpha=0.3)

    # Text annotations
    mag_text = ax1.text(0.02, 0.98, '', transform=ax1.transAxes,
                        fontsize=10, verticalalignment='top',
                        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    def init():
        im.set_data(states[0])
        energy_line.set_data([], [])
        mag_text.set_text('')
        return im, energy_line, mag_text

    def update(frame):
        # Update spin configuration
        im.set_data(states[frame])

        # Update energy plot
        x_data = list(range(frame + 1))
        y_data = energies[:frame + 1]
        energy_line.set_data(x_data, y_data)

        # Update magnetization text
        mag = np.mean(states[frame])
        mag_text.set_text(f'Step: {frame * 100}\nM = {mag:.3f}')

        return im, energy_line, mag_text

    anim = FuncAnimation(fig, update, init_func=init,
                        frames=len(states), interval=interval,
                        blit=True, repeat=True)
    
    plt.tight_layout()
    return anim

# Animation for triangular lattice
def animate_triangular_ising(size=50, temperature=2.5, n_steps=10000, interval=50):
    """Create an animation of the triangular Ising model evolution."""
    
    # Initialize and run simulation
    triangular_model = TriangularIsingModel(size=size)
    triangular_model.initialize_random()
    states_tri, energies_tri, _ = triangular_model.simulate(temperature, n_steps)

    # Set up the figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Spin configuration plot
    im = ax1.imshow(states_tri[0], cmap='YlGnBu', vmin=-1, vmax=1, interpolation='nearest')
    ax1.set_title(f'Spin Configuration (T={temperature:.2f})', fontsize=12)
    ax1.axis('off')

    # Energy plot
    energy_line, = ax2.plot([], [], 'b-', linewidth=2)
    ax2.set_xlim(0, len(states_tri))
    ax2.set_ylim(min(energies_tri) * 1.1, max(energies_tri) * 1.1)
    ax2.set_xlabel('Time Step (×100)')
    ax2.set_ylabel('Energy')
    ax2.set_title('System Energy')
    ax2.grid(True, alpha=0.3)

    # Text annotations
    mag_text = ax1.text(0.02, 0.98, '', transform=ax1.transAxes,
                        fontsize=10, verticalalignment='top',
                        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    def init():
        im.set_data(states_tri[0])
        energy_line.set_data([], [])
        mag_text.set_text('')
        return im, energy_line, mag_text

    def update(frame):
        # Update spin configuration
        im.set_data(states_tri[frame])

        # Update energy plot
        x_data = list(range(frame + 1))
        y_data = energies_tri[:frame + 1]
        energy_line.set_data(x_data, y_data)

        # Update magnetization text
        mag = np.mean(states_tri[frame])
        mag_text.set_text(f'Step: {frame * 100}\nM = {mag:.3f}')

        return im, energy_line, mag_text

    anim = FuncAnimation(fig, update, init_func=init,
                        frames=len(states_tri), interval=interval,
                        blit=True, repeat=True)

    plt.tight_layout()
    return anim