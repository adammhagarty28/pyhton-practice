import jax
import jax.numpy as jnp
from dataclasses import dataclass
from functools import partial
@dataclass
class Pose:
    position: jnp.ndarray = None  # (3,)
    rotation: jnp.ndarray = None  # (4,) xyzw

#helpers
def quat_conj(q):
    return jnp.array([-q[0], -q[1], -q[2], q[3]])

def quat_rotate(q, v):
    q_xyz = q[:3]
    q_w = q[3]
    t = 2.0 * jnp.cross(q_xyz, v)
    return v + q_w * t + jnp.cross(q_xyz, t)

def _quat_slerp(q0, q1, t):
    dot = jnp.clip(jnp.dot(q0, q1), -1.0, 1.0)
    q1 = jnp.where(dot < 0, -q1, q1)
    dot = jnp.abs(dot)
    theta = jnp.arccos(dot)
    sin_theta = jnp.sin(theta)
    safe = sin_theta > 1e-6
    w0 = jnp.where(safe, jnp.sin((1 - t) * theta) / sin_theta, 1 - t)
    w1 = jnp.where(safe, jnp.sin(t * theta) / sin_theta, t)
    return w0 * q0 + w1 * q1

#paint deposition function through projective rasterization like qimg
@partial(jax.jit, static_argnames=('resolution', 'fov'))
def deposit(pose_pos, pose_rot, sigma, a, ref_dist, resolution,
            face_v0, face_v1, face_v2, face_normals, n_faces, fov=90.0,
            depth_temperature=0.1):
    # Create 2D projection of the 3D space of triangles
    face_centers = (face_v0 + face_v1 + face_v2) / 3.0
    rel_pos = face_centers - pose_pos
    local_pos = jax.vmap(lambda r: quat_rotate(quat_conj(pose_rot), r))(rel_pos)

    x, y, z = local_pos[:, 0], local_pos[:, 1], local_pos[:, 2]

    half_fov_rad = jnp.radians(fov / 2.0)
    tan_half = jnp.tan(half_fov_rad)

    proj_x = x / (z * tan_half + 1e-8)
    proj_y = y / (z * tan_half + 1e-8)

    px = jnp.int32(jnp.floor((proj_x + 1.0) * 0.5 * resolution))
    py = jnp.int32(jnp.floor((proj_y + 1.0) * 0.5 * resolution))

    is_visible = (z > 0.05) & (px >= 0) & (px < resolution) & (py >= 0) & (py < resolution)

    pixel_indices = py * resolution + px
    num_pixels = resolution * resolution
    safe_indices = jnp.where(is_visible, pixel_indices, 0)

    #weight paint on depth, so there is a lil bleedthrough for the sake of gradient
    depth_score = jnp.where(is_visible, -z / depth_temperature, -jnp.inf)

    pixel_max = jnp.full((num_pixels,), -jnp.inf)
    pixel_max = pixel_max.at[safe_indices].max(depth_score)

    shifted_score = depth_score - pixel_max[safe_indices]
    exp_score = jnp.where(is_visible, jnp.exp(shifted_score), 0.0)

    pixel_exp_sum = jnp.zeros((num_pixels,))
    pixel_exp_sum = pixel_exp_sum.at[safe_indices].add(exp_score)

    soft_depth_weight = jnp.where(
        is_visible,
        exp_score / (pixel_exp_sum[safe_indices] + 1e-8),
        0.0
    )

    #calculate spray weights
    #gaussian
    r2 = proj_x**2 + proj_y**2
    screen_gaussian = jnp.exp(-0.5 * r2 / (sigma**2 + 1e-8))
    #incidence
    nozzle_dir = quat_rotate(pose_rot, jnp.array([0.0, 0.0, -1.0]))
    to_nozzle = -rel_pos / (jnp.linalg.norm(rel_pos, axis=-1, keepdims=True) + 1e-8)
    raw_dot = jnp.sum(face_normals * to_nozzle, axis=-1)
    flipped_normals = jnp.where(raw_dot[:, None] < 0, -face_normals, face_normals)
    incidence = jnp.maximum(0.0, jnp.sum(flipped_normals * to_nozzle, axis=-1))
    #distance
    falloff = (ref_dist / (z + 1e-8)) ** 2
    falloff = jnp.where(is_visible, falloff, 0.0)
    #calc final paint weight
    weight = a * screen_gaussian * incidence * falloff * soft_depth_weight

    return weight


def deposit(pose_pos, pose_rot, sigma, a, ref_dist, resolution,
            face_v0, face_v1, face_v2, face_normals, n_faces, fov=90.0,
            depth_temperature=0.1):
    face_centers = (face_v0 + face_v1 + face_v2) / 3.0
    rel_pos = face_centers - pose_pos

    # replace vmap'd quat_rotate with batched version
    q_xyz = quat_conj(pose_rot)[:3]
    q_w = quat_conj(pose_rot)[3]
    t = 2.0 * jnp.cross(q_xyz[None], rel_pos)          # (n_faces, 3)
    local_pos = rel_pos + q_w * t + jnp.cross(q_xyz[None], t)
    x, y, z = local_pos[:, 0], local_pos[:, 1], local_pos[:, 2]

    half_fov_rad = jnp.radians(fov / 2.0)
    tan_half = jnp.tan(half_fov_rad)

    proj_x = x / (z * tan_half + 1e-8)
    proj_y = y / (z * tan_half + 1e-8)

    px = jnp.int32(jnp.floor((proj_x + 1.0) * 0.5 * resolution))
    py = jnp.int32(jnp.floor((proj_y + 1.0) * 0.5 * resolution))

    is_visible = (z > 0.05) & (px >= 0) & (px < resolution) & (py >= 0) & (py < resolution)
    # is_visible = jnp.full((n_faces,), True)

    pixel_indices = py * resolution + px
    num_pixels = resolution * resolution
    safe_indices = jnp.where(is_visible, pixel_indices, 0)

    #weight paint on depth, so there is a lil bleedthrough for the sake of gradient
    depth_score = jnp.where(is_visible, -z / depth_temperature, -jnp.inf)

    pixel_max = jnp.full((num_pixels,), -jnp.inf)
    pixel_max = pixel_max.at[safe_indices].max(depth_score)

    shifted_score = depth_score - pixel_max[safe_indices]
    exp_score = jnp.where(is_visible, jnp.exp(shifted_score), 0.0)

    pixel_exp_sum = jnp.zeros((num_pixels,))
    pixel_exp_sum = pixel_exp_sum.at[safe_indices].add(exp_score)

    soft_depth_weight = jnp.where(
        is_visible & (pixel_exp_sum[safe_indices] > 1e-8),
        exp_score / (pixel_exp_sum[safe_indices] + 1e-8),
        0.0
    )
    soft_depth_weight = 1.0

    #calculate spray weights
    #gaussian
    r2 = proj_x**2 + proj_y**2
    screen_gaussian = jnp.exp(-0.5 * r2 / (sigma**2 + 1e-8))
    #incidence
    nozzle_dir = quat_rotate(pose_rot, jnp.array([0.0, 0.0, -1.0]))
    to_nozzle = -rel_pos / (jnp.linalg.norm(rel_pos, axis=-1, keepdims=True) + 1e-8)
    raw_dot = jnp.sum(face_normals * to_nozzle, axis=-1)
    flipped_normals = jnp.where(raw_dot[:, None] < 0, -face_normals, face_normals)
    incidence = jnp.maximum(0.0, jnp.sum(flipped_normals * to_nozzle, axis=-1))
    #distance
    falloff = (ref_dist / (z + 1e-8)) ** 2
    falloff = jnp.where(is_visible, falloff, 0.0)
    #calc final paint weight
    # jax.debug.print("pose: {nan_mask}", 
    #                 nan_mask=jnp.isnan(pose_pos).any())
    # jax.debug.print("pos_on_screen nan: {nan_mask}", 
    #                 nan_mask=jnp.isnan(pos_on_screen).any())
    weight = a * screen_gaussian * incidence * falloff * soft_depth_weight
    # jax.debug.print(
    # "w: {w} | a: {a} | gauss: {g} | incid: {i} | fall: {f} | depth: {d}",
    # w=weight, a=a, g=screen_gaussian, i=incidence, f=falloff, d=soft_depth_weight
    # )
    weight = jnp.where(is_visible, weight, 0.0)

    return weight

_deposit_jit = partial(jax.jit, static_argnames=('resolution', 'fov', 'n_faces'))(deposit)

def interpolate(start, end, steps):
    #interpolate based on num of steps
    ts = jnp.arange(steps) / float(steps)
    positions = jax.vmap(lambda t: start.position * (1 - t) + end.position * t)(ts)
    rotations = jax.vmap(lambda t: _quat_slerp(start.rotation, end.rotation, t))(ts)
    return [Pose(position=positions[i], rotation=rotations[i]) for i in range(steps)]


#pulse kernels
#vmap pulse VERY FAST RUNS OUTTA MEMORY
# def pulse(poses, sigma, a, ref_dist, resolution,
#           face_v0, face_v1, face_v2, face_normals, n_faces, fov=90.0, **kwargs):
#     if len(poses) == 0:
#         return jnp.zeros((n_faces,))

#     def deposit_one(position, rotation):
#         return deposit(position, rotation, sigma, a, ref_dist, resolution,
#                     face_v0, face_v1, face_v2, face_normals, n_faces, fov)

#     positions = jnp.stack([p.position for p in poses])
#     rotations = jnp.stack([p.rotation for p in poses])
#     #stack all poses and evaluate at once
#     all_thicknesses = jax.vmap(deposit_one)(positions, rotations)
#     return jnp.sum(all_thicknesses, axis=0)


def pulse(poses, sigma, a, ref_dist, resolution,
          face_v0, face_v1, face_v2, face_normals, n_faces, fov=90.0, **kwargs):
    if len(poses) == 0:
        return jnp.zeros((n_faces,))

    positions = jnp.stack([p.position for p in poses])
    rotations = jnp.stack([p.rotation for p in poses])

    #for each pulse in path, deposit paint and add to total
    def _accumulate_step(running_thickness, carry_inputs):
        pos, rot = carry_inputs
        
        step_paint = deposit(
            pos, rot, sigma, a, ref_dist, resolution,
            face_v0, face_v1, face_v2, face_normals, n_faces, fov
        )
        
        next_thickness = running_thickness + step_paint
        
        return next_thickness, None

    init_thickness = jnp.zeros((n_faces,))

    final_thickness, _ = jax.lax.scan(
        _accumulate_step, 
        init_thickness, 
        (positions, rotations)
    )

    return final_thickness

def pulse_batch(poses, sigma_u, sigma_v, ref_dist, a, samples_per_pulse,
                face_v0, face_v1, face_v2, n_faces, rng_key, face_normals=None):
    # Simply use the vmap logic without summing the first axis
    positions = jnp.stack([p.position for p in poses])
    rotations = jnp.stack([p.rotation for p in poses])
    
    v_thickness = jax.vmap(
        jax.vmap(_calculate_face_thickness, in_axes=(None, None, 0, 0, 0, 0, None, None, None, None)),
        in_axes=(0, 0, None, None, None, None, None, None, None, None)
    )
    
    return v_thickness(
        positions, rotations, 
        face_v0, face_v1, face_v2, face_normals,
        sigma_u, sigma_v, a * samples_per_pulse, ref_dist
    )

@partial(jax.jit, static_argnames=('resolution',))
def qimg(
    face_v0: jnp.ndarray,       
    face_v1: jnp.ndarray,
    face_v2: jnp.ndarray,
    thicknesses: jnp.ndarray,   
    ee_pos: jnp.ndarray,        
    ee_xmat: jnp.ndarray,       
    fov_deg: float = 60.0,
    resolution: int = 64,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:   
    face_centers = (face_v0 + face_v1 + face_v2) / 3.0  

    R = ee_xmat.reshape(3, 3)
    delta = face_centers - ee_pos[None]     
    local = delta @ R                       

    distances = jnp.sqrt(jnp.sum(delta ** 2, axis=-1) + 1e-12)

    in_front = local[:, 2] > 0.0

    eps = 1e-6
    ax = local[:, 0] / jnp.maximum(local[:, 2], eps)
    ay = local[:, 1] / jnp.maximum(local[:, 2], eps)

    half_tan = jnp.tan(jnp.deg2rad(fov_deg / 2.0))
    in_fov = in_front & (jnp.abs(ax) <= half_tan) & (jnp.abs(ay) <= half_tan)

    px = jnp.int32(jnp.clip(
        jnp.floor((ax + half_tan) / (2.0 * half_tan) * resolution),
        0, resolution - 1,
    ))
    py = jnp.int32(jnp.clip(
        jnp.floor((ay + half_tan) / (2.0 * half_tan) * resolution),
        0, resolution - 1,
    ))
    pixel_indices = py * resolution + px
    flat = resolution * resolution

    valid_count    = jnp.where(in_fov, 1.0, 0.0)
    count          = jnp.zeros(flat).at[pixel_indices].add(valid_count)
    safe_count     = jnp.maximum(count, 1.0)

    #thickness img is downsampled into pixels, where pixels are the mean thickness error value of all faces encompassed

    valid_thickness = jnp.where(in_fov, thicknesses, 0.0)
    thickness_sum   = jnp.zeros(flat).at[pixel_indices].add(valid_thickness)
    thickness_img = jnp.where(count > 0, thickness_sum / safe_count, -1.0)

    valid_distance = jnp.where(in_fov, distances, 0.0)
    distance_sum   = jnp.zeros(flat).at[pixel_indices].add(valid_distance)

    #distance img is downsampled by mean distance of faces included in pixels, anything outside fov is -1

    distance_img  = jnp.where(count > 0, distance_sum / safe_count, -1.0)

    #get face normals using face edge cross product
    edge1 = face_v1 - face_v0
    edge2 = face_v2 - face_v0
    face_normals = jnp.cross(edge1, edge2, axis=-1)
    
    #normalize normals to unit vectors
    normal_lengths = jnp.sqrt(jnp.sum(face_normals ** 2, axis=-1, keepdims=True)+1e-12)
    unit_normals = face_normals / jnp.maximum(normal_lengths, eps)
    
    #normalize rays to unit vectors
    unit_rays = delta / jnp.maximum(distances[:, None], eps)
    
    #arccos of dot product to get incidence angle
    dot_product = jnp.abs(jnp.sum(unit_rays * unit_normals, axis=-1))
    dot_product = jnp.clip(dot_product, 0.0, 1.0 - 1e-6)
    #incidence angles are in DEGREES NOT RADIANS, old pulse uses degrees as well
    incidence_angles = jnp.rad2deg(jnp.arccos(dot_product))

    # Downsample by the mean incidence angle of faces included in the pixels
    valid_incidence = jnp.where(in_fov, incidence_angles, 0.0)
    incidence_sum   = jnp.zeros(flat).at[pixel_indices].add(valid_incidence)
    incidence_img   = jnp.where(count > 0, incidence_sum / safe_count, -1.0)

    return (
        thickness_img.astype(jnp.float32), 
        distance_img.astype(jnp.float32), 
        incidence_img.astype(jnp.float32)
    )

