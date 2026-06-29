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
from collections import deque
import matplotlib.pyplot as plt

n = 10                          # grid points per side (10x10 plate)
L = 1.0                         # plate side length (m) - 1 meter square forging
dx = L / (n - 1)                # spacing between points (m)
dt = 0.5                        # time step (s)
alpha = 1.17e-5                 # thermal diffusivity of steel (m^2/s)
T_target = 100.0                # target temperature (C)
T_ambient = 20.0                # ambient temperature (C)
delay_steps = 1                 # spray delay - 1 timestep lag between command and effect
thickness= .01                  # plate thickness (m)-10mm, typical forged plate
max_steps= 2000

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

def ambient_cooling(T, h_air=.01):
    """
    Natural convection heat loss to surrounding air every timestep.
    Every point loses heat proportional to how far above ambient it is.
    h_air: natural convection coefficient (simplified, unitless for this sim)
    """
    dT = h_air * (T - T_ambient) * dt
    T_new = T - dT
    return jnp.clip(T_new, T_ambient, None)

#test:create centered hot spot and run 3 steps manually
"""T = make_initial_temp(cx=0.5, cy=0.5)
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


#2: BFS Hot Zone identification
def bfs_hot_zone(T, threshold):
    # Convert JAX array to numpy for BFS (graph traversal needs Python loops)
    T_np = np.array(T)
    
    # Find the hottest point - this is our BFS starting point
    start = np.unravel_index(np.argmax(T_np), T_np.shape)
    
    # Only run BFS if hottest point is actually above threshold
    if T_np[start] < threshold:
        return {}, np.zeros((n, n))
    
    # BFS setup
    visited = set()
    queue = deque()
    distance = {}
    
    queue.append(start)
    visited.add(start)
    distance[start] = 0
    
    # Four cardinal neighbors: up, down, left, right
    directions = [(-1,0), (1,0), (0,-1), (0,1)]
    
    while queue:
        row, col = queue.popleft()
        
        for dr, dc in directions:
            neighbor = (row + dr, col + dc)
            nr, nc = neighbor
            
            # Check: within grid bounds, not visited, above threshold
            if (0 <= nr < n and 
                0 <= nc < n and 
                neighbor not in visited and 
                T_np[neighbor] >= threshold):
                
                visited.add(neighbor)
                distance[neighbor] = distance[(row, col)] + 1
                queue.append(neighbor)
    
    # Convert distance map to a weight array
    # Points closer to core (distance=0) get weight=1.0
    # Points further away get lower weight
    weight_map = np.zeros((n, n))
    max_dist = max(distance.values()) if distance else 1
    
    for (r, c), dist in distance.items():
        weight_map[r, c] = 1.0 - (dist / (max_dist + 1))
    
    return distance, jnp.array(weight_map)

#Test BFS
T_test = make_initial_temp(cx=0.5, cy=0.5)
threshold = 200.0
distance_map, weight_map = bfs_hot_zone(T_test, threshold)

print(f"\nBFS Results:")
print(f"Hot zone size: {len(distance_map)} points above {threshold}C")
print(f"Max BFS distance from core: {max(distance_map.values()) if distance_map else 0}")
print(f"Weight map max: {weight_map.max():.3f}")
print(f"Weight map min (hot zone only): {weight_map[weight_map > 0].min():.3f}")
print(f"\nWeight map (rounded):")
print("\nWeight map (0=cool, higher=spray priority):")
for row in np.array(weight_map):
    print("  " + "  ".join(f"{v:.1f}" for v in row))

#3: BFS boundary detection
def dfs_boundary(T, threshold):
    # Convert to numpy for graph traversal
    T_np = np.array(T)
    
    # Find hottest point as starting node
    start = np.unravel_index(np.argmax(T_np), T_np.shape)
    
    if T_np[start] < threshold:
        return set(), set()
    
    # DFS setup
    visited = set()
    stack = [start]
    hot_zone = set()
    boundary = set()
    directions = [(-1,0), (1,0), (0,-1), (0,1)]
    
    while stack:
        row, col = stack.pop()  # DFS uses stack - last in first out
        
        if (row, col) in visited:
            continue
            
        visited.add((row, col))
        hot_zone.add((row, col))
        
        is_boundary = False
        
        for dr, dc in directions:
            nr, nc = row + dr, col + dc
            neighbor = (nr, nc)
            
            if not (0 <= nr < n and 0 <= nc < n):
                # Edge of plate - this point touches plate boundary
                is_boundary = True
                continue
            
            if T_np[neighbor] < threshold:
                # Neighbor is cool - this point is on hot zone boundary
                is_boundary = True
            elif neighbor not in visited:
                stack.append(neighbor)
        
        if is_boundary:
            boundary.add((row, col))
    
    return hot_zone, boundary

#Test DFS
hot_zone, boundary = dfs_boundary(T_test, threshold)
interior = hot_zone - boundary

print(f"\nDFS Results:")
print(f"Total hot zone points: {len(hot_zone)}")
print(f"Boundary points: {len(boundary)}")
print(f"Interior points: {len(interior)}")

# Visualize boundary on grid
boundary_map = np.zeros((n, n))
for r, c in boundary:
    boundary_map[r, c] = 1.0
for r, c in interior:
    boundary_map[r, c] = 0.5

print(f"\nHot zone map (1.0=boundary, 0.5=interior, 0=cool):")
print("\nHot zone map (B=boundary, I=interior, .=cool):")
for i in range(n):
    row_str = ""
    for j in range(n):
        if boundary_map[i,j] == 1.0:
            row_str += "  B"
        elif boundary_map[i,j] == 0.5:
            row_str += "  I"
        else:
            row_str += "  ."
    print(row_str)


#Step 4:Spray Pattern

def snake_scan_pattern(n):
    """Returns list of (row, col) tuples in snake scan order"""
    pattern = []
    for row in range(n):
        if row % 2 == 0:
            cols = range(n)          # left to right
        else:
            cols = range(n-1, -1, -1) # right to left
        for col in cols:
            pattern.append((row, col))
    return pattern

def bfs_priority_pattern(weight_map, min_weight=0.1):
    """Returns list of (row, col) tuples ordered by BFS weight (highest first)
    Only includes points above min_weight threshold - skips cool zones"""
    weight_np = np.array(weight_map)
    points = []
    for r in range(n):
        for c in range(n):
            if weight_np[r, c] >= min_weight:
                points.append((r, c, weight_np[r, c]))
    # Sort by weight descending - highest priority first
    points.sort(key=lambda x: x[2], reverse=True)
    return [(r, c) for r, c, w in points]

#Test spray patterns
snake_pattern = snake_scan_pattern(n)
bfs_pattern = bfs_priority_pattern(weight_map)

print(f"\n=== STEP 4: Spray Pattern ===")
print(f"Snake scan: {len(snake_pattern)} points total")
print(f"First 5 positions: {snake_pattern[:5]}")
print(f"Last 5 positions:  {snake_pattern[-5:]}")
print(f"\nBFS priority: {len(bfs_pattern)} points in hot zone")
print(f"First 5 positions (hottest first): {bfs_pattern[:5]}")
print(f"Last 5 positions (coolest in zone): {bfs_pattern[-5:]}")

#5:Proportional Controller

def proportional_spray(T, weight_map, k_base, budget):
    """
    Proportional spray policy with BFS weighting.
    spray[i,j] = k_base * weight[i,j] * max(T[i,j] - T_target, 0)
    Spray intensity is proportional to temperature excess above target,
    scaled by BFS spatial weight (higher priority at hot zone core).
    Total spray is capped by budget constraint.
    """
    T_excess = jnp.maximum(T - T_target, 0.0)
    spray = k_base * weight_map * T_excess
    
    # Apply budget constraint - scale down if over budget
    total_spray = jnp.sum(spray)
    spray = jnp.where(total_spray > budget, spray * budget / total_spray, spray)
    
    return spray

#Test three k values
distance_map, weight_map = bfs_hot_zone(T_test, threshold)

k_low    = 0.1
k_optimal = 0.5
k_high   = 1
budget   = 50000.0

spray_low     = proportional_spray(T_test, weight_map, k_low, budget)
spray_optimal = proportional_spray(T_test, weight_map, k_optimal, budget)
spray_high    = proportional_spray(T_test, weight_map, k_high, budget)

print(f"\n=== STEP 5: Proportional Controller ===")
print(f"Budget: {budget:.1f} C total spray per step")
print(f"\nk_low={k_low}:")
print(f"  Total spray: {jnp.sum(spray_low):.1f} | Max spray: {spray_low.max():.2f}")
print(f"\nk_optimal={k_optimal}:")
print(f"  Total spray: {jnp.sum(spray_optimal):.1f} | Max spray: {spray_optimal.max():.2f}")
print(f"\nk_high={k_high}:")
print(f"  Total spray: {jnp.sum(spray_high):.1f} | Max spray: {spray_high.max():.2f}")

#Step 6: Spray-budget constraint
# Already implemented inside proportional_spray via jnp.where
# Budget is enforced by scaling spray array down if total exceeds limit
# Documented here explicitly as a design decision:
spray_budget_per_step = 50000.0   # max total cooling per timestep (C summed across all points)
# This represents physical limit of nozzle flow rate

#7:Convergence Criterion
uniformity_threshold = 60.0     # max allowed std dev at convergence (C)

def has_converged(T):
    """
    Plate is considered safely cooled when:
    1. Max temperature is below T_target
    2. Temperature std dev is below uniformity_threshold
    Both conditions must be met - uniform cooling, not just peak reduction
    """
    max_ok = jnp.max(T) < T_target
    uniform_ok = jnp.std(T) < uniformity_threshold
    return bool(max_ok and uniform_ok)

#Test convergence criterion
T_hot  = make_initial_temp(cx=0.5, cy=0.5)
T_cool = jnp.ones((n, n)) * 80.0
T_nonuniform = jnp.ones((n, n)) * 80.0
T_nonuniform = T_nonuniform.at[4, 4].set(300.0)

print(f"\n=== STEP 6 & 7: Budget + Convergence ===")
print(f"Spray budget per step: {spray_budget_per_step:.1f} C")
print(f"Uniformity threshold: {uniformity_threshold:.1f} C std dev")
print(f"\nHot plate converged: {has_converged(T_hot)}")
print(f"Cool uniform plate converged: {has_converged(T_cool)}")
print(f"Cool but non-uniform plate converged: {has_converged(T_nonuniform)}")

#Step 8: Thermal Shock Constraint
max_cooling_rate = 10.0    # max allowable temperature drop per point per timestep (C)

def apply_thermal_shock_limit(spray, T):
    """
    Prevents any single point from cooling faster than max_cooling_rate per step.
    Thermal shock occurs when cooling is too rapid - causes microstructural damage
    and residual stress concentrations in the forging.
    Clamps spray intensity at each point independently.
    """
    max_spray_allowed = jnp.ones((n, n)) * max_cooling_rate
    spray_limited = jnp.minimum(spray, max_spray_allowed)
    return spray_limited

#Test thermal shock limit
T_current = make_initial_temp(cx=0.5, cy=0.5)
distance_map, weight_map = bfs_hot_zone(T_current, threshold)
spray_unlimited = proportional_spray(T_current, weight_map, k_high, spray_budget_per_step)
spray_limited   = apply_thermal_shock_limit(spray_unlimited, T_current)

print(f"\n=== STEP 8: Thermal Shock Constraint ===")
print(f"Max cooling rate: {max_cooling_rate:.1f} C per point per step")
print(f"\nWithout limit:")
print(f"  Max spray at any point: {spray_unlimited.max():.2f} C")
print(f"  Total spray: {jnp.sum(spray_unlimited):.2f} C")
print(f"\nWith thermal shock limit:")
print(f"  Max spray at any point: {spray_limited.max():.2f} C")
print(f"  Total spray: {jnp.sum(spray_limited):.2f} C")
print(f"  Points where limit was active: {int(jnp.sum(spray_unlimited > max_cooling_rate))}")

#Step 9:Temperature Noise
noise_std = 2.0    # standard deviation of sensor noise (C)
# represents real thermal camera measurement uncertainty

def add_sensor_noise(T, key):
    """
    Adds Gaussian noise to observed temperatures before policy makes decisions.
    Policy sees noisy measurements, not perfect temperatures.
    Tests whether policy is robust to measurement uncertainty.
    key: JAX random key (required for JAX random number generation)
    """
    noise = jax.random.normal(key, shape=T.shape) * noise_std
    T_noisy = T + noise
    T_noisy = jnp.clip(T_noisy, T_ambient, None)
    return T_noisy

#Test noise
key = jax.random.PRNGKey(42)
T_clean = make_initial_temp(cx=0.5, cy=0.5)
T_noisy = add_sensor_noise(T_clean, key)

print(f"\n=== STEP 9: Temperature Noise ===")
print(f"Noise std: {noise_std:.1f} C")
print(f"Clean max temp: {T_clean.max():.2f} C")
print(f"Noisy max temp: {T_noisy.max():.2f} C")
print(f"Max noise at any point: {jnp.abs(T_noisy - T_clean).max():.2f} C")
print(f"Mean noise magnitude: {jnp.abs(T_noisy - T_clean).mean():.2f} C")

#Step 10: Moving Nozzle
max_nozzle_speed = 1    # max zones nozzle can move per timestep (grid steps)

class Nozzle:
    def __init__(self, start_row=0, start_col=0):
        self.row = start_row
        self.col = start_col

    def move_toward(self, target_row, target_col):
        """
        Moves nozzle one step toward target position.
        Constrained to max_nozzle_speed zones per timestep.
        """
        dr = target_row - self.row
        dc = target_col - self.col

        # Move one step in row direction if needed
        if dr != 0:
            self.row += int(np.sign(dr)) * max_nozzle_speed
        # Move one step in col direction if needed
        if dc != 0:
            self.col += int(np.sign(dc)) * max_nozzle_speed

        # Clamp to grid bounds
        self.row = int(np.clip(self.row, 0, n-1))
        self.col = int(np.clip(self.col, 0, n-1))

    def position(self):
        return (self.row, self.col)

def nozzle_spray_map(nozzle, spray_intensity, footprint=1):
    """
    Creates spray array with intensity applied around nozzle position.
    footprint controls how many neighboring zones the nozzle covers.
    footprint=1 means 3x3 zone around nozzle position.
    """
    spray = np.zeros((n, n))
    r, c = nozzle.position()
    for dr in range(-footprint, footprint+1):
        for dc in range(-footprint, footprint+1):
            nr, nc = r + dr, c + dc
            if 0 <= nr < n and 0 <= nc < n:
                spray[nr, nc] = spray_intensity
    return jnp.array(spray)

#Test moving nozzle
nozzle = Nozzle(start_row=0, start_col=0)
target_row, target_col = 4, 4    # move toward hot zone center

print(f"\n=== STEP 10: Moving Nozzle ===")
print(f"Max nozzle speed: {max_nozzle_speed} zone(s) per timestep")
print(f"Starting position: {nozzle.position()}")

for move in range(6):
    nozzle.move_toward(target_row, target_col)
    spray_map = nozzle_spray_map(nozzle, spray_intensity=5.0)
    print(f"  Move {move+1}: position={nozzle.position()} | spray sum={jnp.sum(spray_map):.1f}")

#Step 11: Multiple IC's (Centered, off-center, two hot spots)

key_ic = jax.random.PRNGKey(99)
rand_cx = float(jax.random.uniform(key_ic, minval=0.2, maxval=0.8))
rand_cy = float(jax.random.uniform(jax.random.split(key_ic)[0], minval=0.2, maxval=0.8))

initial_conditions = {
    "centered":    make_initial_temp(cx=0.5,     cy=0.5),
    "off_center":  make_initial_temp(cx=0.3,     cy=0.7),
    "two_hotspots": make_initial_temp(cx=0.3,    cy=0.3, peak=800) + 
                    make_initial_temp(cx=0.7,     cy=0.7, peak=700) - T_ambient,
    "random":      make_initial_temp(cx=rand_cx, cy=rand_cy)
}

print(f"\n=== STEP 11: Multiple Initial Conditions ===")
for name, T_ic in initial_conditions.items():
    print(f"{name:<15} | max={T_ic.max():.1f}C | mean={T_ic.mean():.1f}C | std={T_ic.std():.1f}C")
print(f"Random hot spot location: ({rand_cx:.2f}, {rand_cy:.2f})")

#Step 12: Full-Sim loop + Comparison

def run_simulation(T_init, k, use_bfs_pattern=True, 
                   max_steps=2000, seed=0):
    """
    Full spray cooling simulation with proportional policy.
    Returns history of temperatures and spray for analysis.
    """
    T = T_init.copy() if isinstance(T_init, np.ndarray) else jnp.array(T_init)
    prev_spray = jnp.zeros((n, n))
    key = jax.random.PRNGKey(seed)
    nozzle = Nozzle(start_row=0, start_col=0)
    
    # History storage
    max_history  = []
    mean_history = []
    min_history  = []
    std_history  = []
    spray_total_history = []
    
    # Initial BFS and DFS
    distance_map, weight_map = bfs_hot_zone(T, threshold)
    hot_zone, boundary = dfs_boundary(T, threshold)
    
    # Spray pattern
    if use_bfs_pattern:
        pattern = bfs_priority_pattern(weight_map)
    else:
        pattern = snake_scan_pattern(n)
    
    total_spray_used = 0.0
    converged_step = None
    
    for step in range(max_steps):
        # Add sensor noise
        key, subkey = jax.random.split(key)
        T_observed = add_sensor_noise(T, subkey)
        
        # Recompute BFS weights every 10 steps as hot zone evolves
        if step % 10 == 0:
            distance_map, weight_map = bfs_hot_zone(T_observed, threshold)
        
        # Compute spray command
        spray_command = proportional_spray(T_observed, weight_map, k, spray_budget_per_step)
        
        # Apply thermal shock limit
        spray_command = apply_thermal_shock_limit(spray_command, T_observed)
        
        # Move nozzle toward hottest point
        hottest = np.unravel_index(np.argmax(np.array(T_observed)), (n, n))
        nozzle.move_toward(hottest[0], hottest[1])
        
        # Apply diffusion
        T = diffusion_step(T)
        T = ambient_cooling(T)
        
        # Apply spray with delay
        T, prev_spray = apply_spray(T, spray_command, prev_spray)
        
        # Track history
        max_history.append(float(T.max()))
        mean_history.append(float(T.mean()))
        min_history.append(float(T.min()))
        std_history.append(float(T.std()))
        spray_total_history.append(float(jnp.sum(spray_command)))
        total_spray_used += float(jnp.sum(spray_command))
        
        # Check convergence
        if has_converged(T) and converged_step is None:
            converged_step = step
            break

    
    return {
        "max_history":   max_history,
        "mean_history":  mean_history,
        "min_history":   min_history,
        "std_history":   std_history,
        "spray_history": spray_total_history,
        "total_spray":   total_spray_used,
        "converged_step": converged_step,
        "final_T":       T
    }

#Run all scenarios
print(f"\n=== STEP 12: Running Simulations ===")

scenarios = {}
for ic_name, T_ic in initial_conditions.items():
    print(f"Running {ic_name}...")
    scenarios[ic_name] = run_simulation(T_ic, k=k_optimal, use_bfs_pattern=True)
    cs = scenarios[ic_name]["converged_step"]
    ts = scenarios[ic_name]["total_spray"]
    print(f"  Converged at step: {cs if cs else 'did not converge'}")
    print(f"  Total spray used: {ts:.1f} C")
    print(f"  Final std dev: {scenarios[ic_name]['std_history'][-1]:.2f} C")

#k comparison on centered case
print(f"\nK value comparison (centered IC, BFS pattern):")
k_results = {}
for k_name, k_val in [("k_low", k_low), ("k_optimal", k_optimal), ("k_high", k_high)]:
    result = run_simulation(initial_conditions["centered"], k=k_val, use_bfs_pattern=True)
    k_results[k_name] = result
    cs = result["converged_step"]
    print(f"  {k_name}={k_val}: converged={cs if cs else 'no'} | spray={result['total_spray']:.1f}")

#Pattern comparison on centered case
print(f"\nPattern comparison (centered IC, k_optimal):")
pattern_results = {}
for use_bfs, label in [(True, "BFS_priority"), (False, "snake_scan")]:
    result = run_simulation(initial_conditions["centered"], k=k_optimal, use_bfs_pattern=use_bfs)
    pattern_results[label] = result
    cs = result["converged_step"]
    print(f"  {label}: converged={cs if cs else 'no'} | spray={result['total_spray']:.1f}")


#Plot 1: Temperature evolution for all initial conditions
fig, ax = plt.subplots(figsize=(10, 5))
colors = {"centered": "red", "off_center": "blue", 
          "two_hotspots": "green", "random": "orange"}
for ic_name, result in scenarios.items():
    ax.plot(result["max_history"], color=colors[ic_name], 
            linestyle="-", label=f"{ic_name} (max)")
    ax.plot(result["mean_history"], color=colors[ic_name], 
            linestyle="--", alpha=0.5)
ax.axhline(y=T_target, color="black", linestyle=":", label="Target")
ax.axhline(y=uniformity_threshold + T_target, color="gray", 
           linestyle=":", alpha=0.5, label="Uniformity threshold")
ax.set_xlabel("Timestep")
ax.set_ylabel("Temperature (C)")
ax.set_title("Temperature Evolution — All Initial Conditions")
ax.legend(fontsize=8)
ax.grid(True)
plt.tight_layout()

#Plot 2: Final temperature heatmaps
fig2, axes2 = plt.subplots(2, 2, figsize=(10, 8))
axes2 = axes2.flatten()
for idx, (ic_name, result) in enumerate(scenarios.items()):
    im = axes2[idx].imshow(np.array(result["final_T"]), 
                           cmap="hot", vmin=T_ambient, vmax=400)
    axes2[idx].set_title(f"{ic_name}\nfinal max={result['max_history'][-1]:.1f}C")
    axes2[idx].set_xlabel("x")
    axes2[idx].set_ylabel("y")
    plt.colorbar(im, ax=axes2[idx], label="Temp (C)")
plt.suptitle("Final Temperature Distribution — All Initial Conditions")
plt.tight_layout()

#Plot 3: K value and pattern comparison bar chart
fig3, axes3 = plt.subplots(1, 2, figsize=(10, 4))

# K comparison
k_names = list(k_results.keys())
k_steps = [k_results[k]["converged_step"] or max_steps 
           for k in k_names]
k_spray = [k_results[k]["total_spray"] for k in k_names]

axes3[0].bar(k_names, k_steps, color=["blue", "green", "red"])
axes3[0].set_ylabel("Steps to converge")
axes3[0].set_title("Convergence Speed vs K Value")
axes3[0].grid(True, axis="y")

# Pattern comparison
p_names = list(pattern_results.keys())
p_steps = [pattern_results[p]["converged_step"] or max_steps 
           for p in p_names]

axes3[1].bar(p_names, p_steps, color=["orange", "purple"])
axes3[1].set_ylabel("Steps to converge")
axes3[1].set_title("BFS Priority vs Snake Scan")
axes3[1].grid(True, axis="y")

plt.suptitle("Policy Comparison — Centered Initial Condition")
plt.tight_layout()

plt.show()

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