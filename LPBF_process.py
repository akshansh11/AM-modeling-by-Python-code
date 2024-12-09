import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from mpl_toolkits.mplot3d import Axes3D
import imageio.v2 as imageio
from pathlib import Path

class LPBFSimulation:
    def __init__(self, nx=1000, nz=300, dx=1.0, laser_power=200, laser_radius=30):
        self.nx = nx
        self.nz = nz
        self.dx = dx
        self.laser_power = laser_power
        self.laser_radius = laser_radius
        self.T = np.ones((nz, nx)) * 300.0
        self.thermal_conductivity = 20
        self.density = 7800
        self.specific_heat = 500
        self.alpha = self.thermal_conductivity / (self.density * self.specific_heat)
        self.dt = 0.25 * dx**2 / self.alpha
        
    def gaussian_laser_source(self, x_pos, z_pos):
        x = np.arange(self.nx)
        z = np.arange(self.nz)
        X, Z = np.meshgrid(x, z)
        q = self.laser_power * np.exp(-2 * ((X - x_pos)**2 + (Z - z_pos)**2) / self.laser_radius**2)
        return q
    
    def simulate_single_track(self, start_x, speed):
        n_steps = int((self.nx - start_x) / speed)
        results = []
        x_pos = start_x
        
        for _ in range(n_steps):
            q = self.gaussian_laser_source(x_pos, 0)
            self.T += q * self.dt
            self.T = gaussian_filter(self.T, sigma=1)
            x_pos += speed
            results.append(self.T.copy())
            self.T[:, -1] = 300.0
            self.T[-1, :] = 300.0
            
        return results
    
    def create_frame(self, T, frame_path):
        plt.figure(figsize=(10, 4))
        plt.imshow(T, aspect='auto', 
                  extent=[0, self.nx, -self.nz, 150],
                  cmap='jet', 
                  vmin=300, vmax=3000)
        plt.colorbar(label='Temperature (K)')
        plt.xlabel('x (μm)')
        plt.ylabel('z (μm)')
        
        # Add powder particles
        n_particles = 200
        x_particles = np.random.uniform(0, self.nx, n_particles)
        z_particles = np.random.uniform(100, 150, n_particles)
        plt.plot(x_particles, z_particles, 'b.', markersize=1)
        
        plt.tight_layout()
        plt.savefig(frame_path)
        plt.close()

def run_simulation_and_save_gif():
    # Create output directory
    output_dir = Path('frames')
    output_dir.mkdir(exist_ok=True)
    
    # Initialize simulation
    sim = LPBFSimulation()
    
    # Simulate track
    results = sim.simulate_single_track(start_x=200, speed=5)
    
    # Create frames
    frames = []
    for i, result in enumerate(results):
        frame_path = output_dir / f'frame_{i:04d}.png'
        sim.create_frame(result, frame_path)
        frames.append(imageio.imread(frame_path))
    
    # Save as GIF
    imageio.mimsave('lpbf_simulation.gif', frames, fps=10)
    
    # Clean up frame files
    for frame_file in output_dir.glob('*.png'):
        frame_file.unlink()
    output_dir.rmdir()

if __name__ == "__main__":
    run_simulation_and_save_gif()
