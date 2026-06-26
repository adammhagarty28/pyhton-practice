"""
Order of approach:
1.Simulation environment (grid, Gaussian hot spot, diffusion, spray delay)
2.BFS hot zone identification
3.DFS boundary detection
4.Spray pattern (snake scan default, BFS-priority alternative)
5.Proportional controller with spatially varying k and BFS weighting
6.Spray budget constraint
7.Convergence criterion — max T below target AND uniformity below threshold
8.Maximum cooling rate constraint per point (thermal shock prevention)
9.Temperature noise
10.Moving nozzle with speed constraint
11.Multiple initial conditions (centered, off-center, two hot spots)
12.Comparison scenarios and plots

Where we go from here:
POC 2-Gradient based optimizaiton
Take the same sim env, replace the hand-coded proportional rule with jax optimized sptially varying k values. Use jax.grad
or jax.value_and_grad to find optimal k per zone for each IC. Compare results against POC 1-does optimizaiton find a better
policy than the hand-coded proportional rule? How much better? This directl uses JAX. 

After that-POC 3: SAC agent
Wrap sim in gymnasium env. This env needs three things; reset(),step(action),render(). Connect stable baseline3 SAC to env. 
Train for some number of episodes. Compare POC 1 and POC 2. This is where SAC gets used from spray-cooling-poc repor but
with proper env. 

After three POCs-analysis and writing
compare all three approahces on the same metrics; convergence time, peak temp, uniformity at convergence, total spray budget
used, robustness to different IC's. This compairiosn is the research conribution. A tabl showing POC vs POC2 vs POC3 on 
those metrics, with plots, is an important and reliable result. 

Longer term-physics integration
replace simplified diffusion with real-JAX based heat equation,add leidenfrost physics,add real nozzle models. Policy does 
not change- only sim env underneath it gets more physically accurate. 

3D Geometry and real meshes
Square 2D plate->  real geometry in 3d, unstructured mesh instead of regular grid. This is where graph algorithms, BFS and DFS
become essential, and where jax-forge's FEM infrastructure becomes relevant.


"""
######1:
import jax
import jax.numpy as jnp
import numpy as np 

n = 10                          # grid points per side (10x10 plate)
L = 1.0                         # plate side length (m) - 1 meter square forging
dx = L / (n - 1)                # spacing between points (m)
dt = 0.5                        # time step (s)
alpha = 1.17e-5                 # thermal diffusivity of steel (m^2/s)
T_target = 100.0                # target temperature (C)
T_ambient = 20.0                # ambient temperature (C)
delay_steps = 1                 # spray delay - 1 timestep lag between command and effect

# Gaussian Hot Spot Initial Condition

def make_initial_temp(cx, cy, peak=900, width=.05):
    x=jnp.linspace(0,L,n)
    y=jnp.linspace(0,L,n)
    X,Y=jnp.meshgrid(x,y)
    T=T_ambient+(peak-T_ambient)*jnp.exp(-((X-cx)**2+(Y-cy)**2)/width) #
    return T

#Diffusion Step (2D Heat Equation)
def diffusion_step(T):
    d2Tdx2=jnp.zeros((n,n))
    d2Tdy2=jnp.zeros((n,n))
    d2Tdx2 = d2Tdx2.at[1:-1, :].set(
        (T[2:, :] - 2*T[1:-1, :] + T[:-2, :]) / dx**2
    )
    d2Tdy2 = d2Tdy2.at[:, 1:-1].set(
        (T[:, 2:] - 2*T[:, 1:-1] + T[:, :-2]) / dx**2
    )
    return T + alpha * dt * (d2Tdx2 + d2Tdy2)

#apply spray with one timestep delay
def apply_spray(T, spray_command, prev_spray):
    # prev_spray is what was commanded last step - now takes effect
    T_new = T - prev_spray
    T_new = jnp.clip(T_new, T_ambient, None)
    return T_new, spray_command  # return updated T and store current command for next step

#test:create centered hot spot and run 3 steps manually
T = make_initial_temp(cx=0.5, cy=0.5)
prev_spray = jnp.zeros((n, n))

print(f"Initial max temp: {T.max():.1f} C")
print(f"Initial mean temp: {T.mean():.1f} C")
print(f"Grid shape: {T.shape}")
print(f"Spacing dx: {dx:.4f} m")

for step in range(3):
    T = diffusion_step(T)
    dummy_spray = jnp.ones((n, n)) * 5.0  # placeholder spray command
    T, prev_spray = apply_spray(T, dummy_spray, prev_spray)
    print(f"Step {step}: max={T.max():.1f} C | mean={T.mean():.1f} C")







"""
What resuls wll be useful:
convergence time-how many timestpes until max temp hits target. Lower is better. Compare across 3 k values

peak temp trajectory-does it overshoot (too high k) or crawl toward it (too low k)?

temperature uniformity at convergence- standard deviation of temperature across the plate when stopping criterion is met.
A god policy cools uniformly, not just the center

total spray budget used-sum of all spray intensitities across all timestpes. Efficiency metric-did we cool plate without
wasitng money?

spray map-visual showing which zones got sprayed most. Should correlate with initial hot spot location

BFS vs BFS no comparison-same k value, with and without BFS weighting. Shows whether sptailly aware spraying outperforms blind
proportional control

"""