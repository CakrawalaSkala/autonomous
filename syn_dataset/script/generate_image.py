# generate_dataset_drone_v4.py
# Run: blender --background --python generate_dataset_drone_v4.py
#
# TWO-STAGE GENERATION:
#
#   Stage 1 — SINGLE shape per image, 5 discrete size steps
#             simulates drone ascending/descending (size variance).
#             Shape always placed near center ± small jitter.
#
#   Stage 2 — MULTI shape per image (2–3), 3 discrete size steps.
#             Shapes may randomly collide/overlap (collision_prob).
#             More crowded, harder detection challenge.
#
# Both stages share:
#   a) tilt variance  (pitch + roll + yaw) — placement is tilt-aware
#   b) ground color/texture randomization (5 surface modes)
#   c) sky/world background color variation (6 sky modes)
#   d) sun lighting randomization
#   e) compositor domain randomization (exposure, blur, noise…)
#
# BUG FIXES vs v4 original:
#   FIX 1 — Seed management: single seed per image, set ONCE before
#            ALL randomization (tilt, placement, scene, render). The
#            old code re-seeded inside render_image, wiping state.
#   FIX 2 — Tilt-aware placement: shapes are placed accounting for
#            the camera tilt that WILL be applied, so they stay in frame.
#            Old code placed shapes, then picked a random tilt that could
#            push them outside the visible frustum.
#   FIX 3 — Collision placement bounds: colliding shapes are still
#            clamped inside the visible frustum. Old code let them land
#            anywhere including outside the frame.
#   FIX 4 — Camera reset: hard_reset_camera() explicitly zeroes
#            location + rotation + clears animation data + flushes
#            depsgraph. Old code's reset_camera() cleared anim data
#            but didn't always flush, letting jitter keyframes bleed
#            into the next image's render.

import bpy, bmesh, math, os, csv, random, json
from mathutils import Euler

# ══════════════════════════════════════════════════════════════════════
#  GLOBAL PARAMETERS
# ══════════════════════════════════════════════════════════════════════

OUTPUT_DIR   = "./output_v4"
SEED         = None       # master seed; None = fully random
IMG_W, IMG_H = 640, 640

# ── Stage control ─────────────────────────────────────────────────────
N_STAGE1 = 10   # single-shape images
N_STAGE2 = 1   # multi-shape images

# Stage 2: probability that a new shape is allowed to overlap an existing one.
COLLISION_PROB = 0.40

# ── Camera ────────────────────────────────────────────────────────────
CAMERA_HEIGHT       = 5.0
CAMERA_ORTHO_SCALE  = 2.0   # default; overridden per image

# Drone tilt: pitch (X), roll (Y), yaw (Z)
CAMERA_TILT_X_RANGE = (-3.0,  3.0)
CAMERA_TILT_Y_RANGE = (-3.0,  3.0)
CAMERA_YAW_RANGE    = (0.0, 360.0)

# ── Stage 1: ortho_scale steps — 5 altitude levels ───────────────────
# Smaller ortho_scale == drone lower == shape appears larger in frame.
STAGE1_ORTHO_STEPS = [2.8, 2.0, 1.6, 1.2, 0.8, 0.2]   # index 0 = highest

# ── Stage 2: ortho_scale steps — 3 altitude levels ───────────────────
STAGE2_ORTHO_STEPS = [2.0, 1.2, 0.6]

# ── Occluders (stage 2 only) ──────────────────────────────────────────
ADD_OCCLUDERS        = True
OCCLUDER_PROB        = 0.35
N_OCCLUDERS_MAX      = 3
OCCLUDER_SCALE_RANGE = (0.15, 2.5)

# ── Ground texture — 5 surface types ──────────────────────────────────
GROUND_MODES = {
    "dirt":     ((0.08, 0.22), (0.06, 0.16), (0.01, 0.05)),
    "grass":    ((0.02, 0.08), (0.08, 0.22), (0.01, 0.04)),
    "concrete": ((0.14, 0.30), (0.14, 0.28), (0.12, 0.26)),
    "tarmac":   ((0.04, 0.10), (0.04, 0.10), (0.04, 0.10)),
    "sand":     ((0.28, 0.42), (0.22, 0.35), (0.08, 0.18)),
}
GROUND_NOISE_SCALE_RANGE  = (6,  24)
GROUND_NOISE_DETAIL_RANGE = (4,  14)
GROUND_ROUGHNESS_RANGE    = (0.70, 0.96)

# ── World / sky — 6 lighting conditions ───────────────────────────────
SKY_MODES = [
    (0.55, 0.75, 0.95),   # blue sky
    (0.75, 0.80, 0.90),   # overcast white-blue
    (0.20, 0.20, 0.20),   # dark / stormy
    (0.80, 0.65, 0.45),   # golden hour warm
    (0.85, 0.85, 0.88),   # flat white overcast
    # (0.06, 0.06, 0.06),   # night / sensor dark
]
WORLD_BG_STRENGTH_RANGE = (0.08, 0.30)
HDRI_PATH               = ""

# ── Sun ───────────────────────────────────────────────────────────────
SUN_ENERGY_RANGE  = (2.0,  7.0)
SUN_ANGLE_RANGE   = (1,    9)
SUN_TILT_RANGE    = (5,   25)
SUN_AZIMUTH_RANGE = (0,  360)

# ── Camera motion blur ────────────────────────────────────────────────
MOTION_JITTER_TRANS  = 0.025
MOTION_JITTER_ROT    = 0.012
MOTION_SHUTTER_RANGE = (0.06, 0.20)

# ── Renderer ──────────────────────────────────────────────────────────
USE_GPU        = True
GPU_TYPE       = "CUDA"
RENDER_SAMPLES = 64
USE_DENOISER   = False

# ── Compositor domain randomization ───────────────────────────────────
EXPOSURE_RANGE         = (-0.20, 0.20)
HUE_JITTER             = (-0.05, 0.05)
SAT_RANGE              = (0.90,  1.10)
VAL_RANGE              = (0.95,  1.05)
BRIGHTNESS_RANGE       = (-0.03, 0.03)
CONTRAST_RANGE         = (-0.05, 0.12)
BLUR_PX_RANGE          = (0,     2)
NOISE_STRENGTH_RANGE   = (0.004, 0.03)
NOISE_SCALE_RANGE      = (5.0,  30.0)
NOISE_DISTORTION_RANGE = (0.0,   1.0)

# ══════════════════════════════════════════════════════════════════════
#  CLASS DEFINITIONS
# ══════════════════════════════════════════════════════════════════════
SHAPE_CLASSES = [
    dict(class_id=0, name="blue_square",   shape_type="square",
         pbr_colors=[(0.01,0.05,0.82),(0.02,0.10,0.90),(0.00,0.07,0.72)],
         scale_range=(0.08,0.16), mask_rgb=(0.0,0.0,1.0)),
    dict(class_id=1, name="red_square",    shape_type="square",
         pbr_colors=[(0.82,0.02,0.02),(0.90,0.04,0.04),(0.72,0.01,0.01)],
         scale_range=(0.06,0.13), mask_rgb=(1.0,0.0,0.0)),
    dict(class_id=2, name="blue_hexagon",  shape_type="hexagon",
         pbr_colors=[(0.01,0.05,0.82),(0.02,0.12,0.88),(0.00,0.07,0.75)],
         scale_range=(0.07,0.13), mask_rgb=(0.0,1.0,1.0)),
    dict(class_id=3, name="red_hexagon",   shape_type="hexagon",
         pbr_colors=[(0.82,0.02,0.02),(0.88,0.04,0.04),(0.75,0.01,0.01)],
         scale_range=(0.07,0.13), mask_rgb=(1.0,0.5,0.0)),
    dict(class_id=4, name="red_triangle",  shape_type="triangle",
         pbr_colors=[(0.82,0.02,0.02),(0.88,0.04,0.04),(0.75,0.01,0.01)],
         scale_range=(0.05,0.10), mask_rgb=(1.0,1.0,0.0)),
    dict(class_id=5, name="red_circle",    shape_type="circle",
         pbr_colors=[(0.82,0.02,0.02),(0.88,0.04,0.04),(0.75,0.01,0.01)],
         scale_range=(0.06,0.12), mask_rgb=(1.0,0.0,1.0)),
    dict(class_id=6, name="blue_circle",   shape_type="circle",
         pbr_colors=[(0.01,0.05,0.82),(0.02,0.10,0.90),(0.00,0.07,0.72)],
         scale_range=(0.06,0.12), mask_rgb=(0.0,1.0,0.0)),
    dict(class_id=7, name="blue_triangle", shape_type="triangle",
         pbr_colors=[(0.01,0.05,0.82),(0.02,0.10,0.90),(0.00,0.07,0.72)],
         scale_range=(0.05,0.10), mask_rgb=(1.0,1.0,1.0)),
]

INSTANCE_PALETTE = [
    (1.0,0.15,0.15),(0.15,1.0,0.15),(0.15,0.15,1.0),(1.0,1.0,0.15),
    (1.0,0.15,1.0),(0.15,1.0,1.0),(1.0,0.55,0.15),(0.55,0.15,1.0),
    (0.15,0.75,1.0),(1.0,0.35,0.65),(0.65,1.0,0.15),(1.0,0.80,0.15),
]

# ══════════════════════════════════════════════════════════════════════
#  MATERIAL HELPERS
# ══════════════════════════════════════════════════════════════════════

def emission_mat(rgb, name="EmiMat"):
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    n = mat.node_tree.nodes; l = mat.node_tree.links; n.clear()
    out = n.new("ShaderNodeOutputMaterial")
    em  = n.new("ShaderNodeEmission")
    em.inputs["Color"].default_value    = (*rgb, 1.0)
    em.inputs["Strength"].default_value = 1.0
    l.new(em.outputs["Emission"], out.inputs["Surface"])
    return mat

def principled_mat(rgb, roughness=0.5, specular=0.4, name="PBRMat"):
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    n = mat.node_tree.nodes; l = mat.node_tree.links; n.clear()
    out  = n.new("ShaderNodeOutputMaterial")
    bsdf = n.new("ShaderNodeBsdfPrincipled")
    bsdf.inputs["Base Color"].default_value = (*rgb, 1.0)
    bsdf.inputs["Roughness"].default_value  = roughness
    for key in ("Specular", "Specular IOR Level"):
        if key in bsdf.inputs:
            bsdf.inputs[key].default_value = specular; break
    l.new(bsdf.outputs["BSDF"], out.inputs["Surface"])
    return mat

# ══════════════════════════════════════════════════════════════════════
#  SCENE HELPERS
# ══════════════════════════════════════════════════════════════════════

def purge_scene():
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)
    for d in (bpy.data.meshes, bpy.data.materials,
              bpy.data.objects, bpy.data.textures):
        for b in list(d): d.remove(b)

def link(obj):
    bpy.context.collection.objects.link(obj)

def set_display_device_safe(device_name):
    try:
        bpy.context.scene.display_settings.display_device = device_name
    except TypeError:
        pass

def configure_render():
    sc = bpy.context.scene
    sc.render.resolution_x          = IMG_W
    sc.render.resolution_y          = IMG_H
    sc.render.resolution_percentage = 100
    sc.render.image_settings.file_format  = "PNG"
    sc.render.image_settings.color_mode  = "RGB"
    sc.render.image_settings.color_depth = "8"
    sc.render.engine = "CYCLES"
    sc.cycles.samples       = RENDER_SAMPLES
    sc.cycles.use_denoising = USE_DENOISER
    sc.render.use_motion_blur       = True
    sc.cycles.motion_blur_position  = "CENTER"
    bpy.context.view_layer.use_pass_object_index = True
    if USE_DENOISER:
        for choice in ("OPENIMAGEDENOISE", "OPTIX", "NLM"):
            try: sc.cycles.denoiser = choice; break
            except Exception: pass
    if USE_GPU:
        prefs = bpy.context.preferences.addons["cycles"].preferences
        prefs.compute_device_type = GPU_TYPE
        prefs.get_devices()
        for d in prefs.devices: d.use = True
        sc.cycles.device = "GPU"
    else:
        sc.cycles.device = "CPU"

def add_camera():
    bpy.ops.object.camera_add(location=(0.0, 0.0, CAMERA_HEIGHT))
    cam = bpy.context.active_object; cam.name = "MainCamera"
    cam.data.type        = "ORTHO"
    cam.data.ortho_scale = CAMERA_ORTHO_SCALE
    cam.rotation_euler   = (0, 0, 0)
    bpy.context.scene.camera = cam
    return cam

def hard_reset_camera(cam):
    """
    FIX 4: Fully reset camera between images.
    Clear animation data first, then explicitly set all transforms,
    then flush the depsgraph. This prevents motion-blur keyframes
    from bleeding into the next image.
    """
    if cam.animation_data:
        cam.animation_data_clear()
    cam.location       = (0.0, 0.0, CAMERA_HEIGHT)
    cam.rotation_euler = (0.0, 0.0, 0.0)
    cam.scale          = (1.0, 1.0, 1.0)
    bpy.context.view_layer.update()

def apply_camera_tilt(cam, tilt_x_deg, tilt_y_deg, yaw_deg):
    """Apply pre-chosen tilt to the camera (values decided by caller)."""
    cam.rotation_euler = (
        math.radians(tilt_x_deg),
        math.radians(tilt_y_deg),
        math.radians(yaw_deg),
    )

def add_ground():
    bpy.ops.mesh.primitive_plane_add(size=8.0, location=(0, 0, -0.001))
    g = bpy.context.active_object; g.name = "Ground"
    mode_name = random.choice(list(GROUND_MODES.keys()))
    rr, rg, rb = GROUND_MODES[mode_name]
    br = random.uniform(*rr)
    bg_v = random.uniform(*rg)
    bb = random.uniform(*rb)
    mat = bpy.data.materials.new("GroundMat"); mat.use_nodes = True
    n = mat.node_tree.nodes; l = mat.node_tree.links; n.clear()
    out   = n.new("ShaderNodeOutputMaterial")
    bsdf  = n.new("ShaderNodeBsdfPrincipled")
    noise = n.new("ShaderNodeTexNoise")
    ramp  = n.new("ShaderNodeValToRGB")
    coord = n.new("ShaderNodeTexCoord")
    noise.inputs["Scale"].default_value     = random.uniform(*GROUND_NOISE_SCALE_RANGE)
    noise.inputs["Detail"].default_value    = random.uniform(*GROUND_NOISE_DETAIL_RANGE)
    noise.inputs["Roughness"].default_value = random.uniform(0.5, 0.8)
    if "W" in noise.inputs:
        noise.inputs["W"].default_value = random.uniform(0, 100)
    ramp.color_ramp.elements[0].color = (br*0.4, bg_v*0.4, bb*0.4, 1.0)
    ramp.color_ramp.elements[1].color = (br*2.0, bg_v*2.0, bb*2.0, 1.0)
    bsdf.inputs["Roughness"].default_value = random.uniform(*GROUND_ROUGHNESS_RANGE)
    l.new(coord.outputs["Generated"], noise.inputs["Vector"])
    l.new(noise.outputs["Fac"],       ramp.inputs["Fac"])
    l.new(ramp.outputs["Color"],      bsdf.inputs["Base Color"])
    l.new(bsdf.outputs["BSDF"],       out.inputs["Surface"])
    g.data.materials.append(mat)
    return g, mode_name

def add_sun():
    bpy.ops.object.light_add(type="SUN", location=(0, 0, 10))
    s = bpy.context.active_object; s.name = "Sun"
    s.data.energy = 3.0; s.data.angle = math.radians(3)
    return s

def randomise_sun(sun):
    energy    = random.uniform(*SUN_ENERGY_RANGE)
    angle_deg = random.uniform(*SUN_ANGLE_RANGE)
    sun.data.energy = energy
    sun.data.angle  = math.radians(angle_deg)
    sun.rotation_euler = Euler((
        math.radians(random.uniform(*SUN_TILT_RANGE)),
        math.radians(random.uniform(-SUN_TILT_RANGE[1]*0.5, SUN_TILT_RANGE[1]*0.5)),
        math.radians(random.uniform(*SUN_AZIMUTH_RANGE)),
    ))
    return energy, angle_deg

def setup_world():
    world = bpy.context.scene.world
    if not world:
        world = bpy.data.worlds.new("World")
        bpy.context.scene.world = world
    world.use_nodes = True
    nodes = world.node_tree.nodes
    links = world.node_tree.links
    if HDRI_PATH and os.path.isfile(HDRI_PATH):
        nodes.clear()
        out = nodes.new("ShaderNodeOutputWorld")
        bg  = nodes.new("ShaderNodeBackground")
        env = nodes.new("ShaderNodeTexEnvironment")
        env.image = bpy.data.images.load(HDRI_PATH)
        bg.inputs["Strength"].default_value = random.uniform(*WORLD_BG_STRENGTH_RANGE)
        links.new(env.outputs["Color"],     bg.inputs["Color"])
        links.new(bg.outputs["Background"], out.inputs["Surface"])
        return "hdri"
    else:
        base_sky = random.choice(SKY_MODES)
        jitter   = 0.07
        sky_rgb  = tuple(max(0.0, min(1.0, c + random.uniform(-jitter, jitter)))
                         for c in base_sky)
        bg = nodes.get("Background")
        if bg is None:
            nodes.clear()
            out = nodes.new("ShaderNodeOutputWorld")
            bg  = nodes.new("ShaderNodeBackground")
            links.new(bg.outputs["Background"], out.inputs["Surface"])
        bg.inputs["Color"].default_value    = (*sky_rgb, 1.0)
        bg.inputs["Strength"].default_value = random.uniform(*WORLD_BG_STRENGTH_RANGE)
        return sky_rgb

# ══════════════════════════════════════════════════════════════════════
#  SHAPE MESH BUILDERS
# ══════════════════════════════════════════════════════════════════════

def _extrude_bm(bm, height=0.06):
    res = bmesh.ops.extrude_face_region(bm, geom=bm.faces[:])
    top = [v for v in res["geom"] if isinstance(v, bmesh.types.BMVert)]
    bmesh.ops.translate(bm, verts=top, vec=(0, 0, height))

def build_standard_shape(shape_type):
    segs_map = {"square": 4, "hexagon": 6, "triangle": 3, "circle": 48}
    mesh = bpy.data.meshes.new(f"Mesh_{shape_type}")
    bm   = bmesh.new()
    bmesh.ops.create_circle(bm, cap_ends=True, radius=1.0,
                             segments=segs_map[shape_type])
    if shape_type == "square":
        import mathutils
        bmesh.ops.rotate(bm, verts=bm.verts, cent=(0,0,0),
                         matrix=mathutils.Matrix.Rotation(math.pi/4, 3, "Z"))
    _extrude_bm(bm)
    bm.to_mesh(mesh); bm.free()
    return mesh

# ══════════════════════════════════════════════════════════════════════
#  TILT-AWARE SAFE PLACEMENT BOUNDS  (FIX 2)
# ══════════════════════════════════════════════════════════════════════

def _safe_half(ortho_scale, tilt_x_deg, tilt_y_deg):
    """
    Compute per-axis safe placement half-extent accounting for camera tilt.

    At height H and tilt angle θ, the visible frustum centre shifts by
    H * tan(θ) in world units. We subtract this shift (plus a 5% buffer)
    from the ortho half-width so any shape placed within (safe_x, safe_y)
    remains fully inside the frame after the tilt is applied.
    """
    shift_x = CAMERA_HEIGHT * math.tan(math.radians(abs(tilt_x_deg)))
    shift_y = CAMERA_HEIGHT * math.tan(math.radians(abs(tilt_y_deg)))
    half    = ortho_scale * 0.5
    buf     = half * 0.15   # 15% safety buffer
    safe_x  = max(half - shift_x - buf, 0.02)
    safe_y  = max(half - shift_y - buf, 0.02)
    return safe_x, safe_y

# ── Stage 1: single shape ─────────────────────────────────────────────

def place_single_shape(ortho_scale, cls, tilt_x_deg, tilt_y_deg):
    """
    One shape near centre with ±20% jitter of the tilt-safe half-extent.
    The jitter keeps the shape visible even at maximum tilt angles.
    """
    scale = random.uniform(*cls["scale_range"])
    rot_z = random.uniform(0, math.tau)
    safe_x, safe_y = _safe_half(ortho_scale, tilt_x_deg, tilt_y_deg)
    x = random.uniform(-safe_x * 0.20, safe_x * 0.20)
    y = random.uniform(-safe_y * 0.20, safe_y * 0.20)
    return [dict(cls=cls, x=x, y=y, scale=scale, rot_z=rot_z, colliding=False)]

# ── Stage 2: multi shape ──────────────────────────────────────────────

def place_multi_shapes(n, ortho_scale, tilt_x_deg, tilt_y_deg,
                       allow_collision=False):
    """
    2–3 shapes with optional overlap.

    FIX 2 + 3: All shapes — including colliding ones — are clamped to the
    tilt-adjusted safe bounds so they remain inside the frame. Colliding
    shapes skip the separation check but not the frustum bounds check.
    """
    placed  = []
    results = []
    GAP     = 1.05
    safe_x, safe_y = _safe_half(ortho_scale, tilt_x_deg, tilt_y_deg)

    for _ in range(n):
        cls   = random.choice(SHAPE_CLASSES)
        scale = random.uniform(*cls["scale_range"])
        rot_z = random.uniform(0, math.tau)
        this_collision = allow_collision and (random.random() < COLLISION_PROB)

        for _attempt in range(500):
            # Shrink bounds by shape radius so shape body stays inside frame
            mx = max(safe_x - scale * 0.6, scale * 0.1)
            my = max(safe_y - scale * 0.6, scale * 0.1)
            x  = random.uniform(-mx, mx)
            y  = random.uniform(-my, my)

            if this_collision:
                # Shape may overlap others — frustum check already done above
                placed.append((x, y, scale))
                results.append(dict(cls=cls, x=x, y=y, scale=scale,
                                    rot_z=rot_z, colliding=True))
                break
            else:
                if all(math.hypot(x-px, y-py) >= (scale + pr) * GAP
                       for (px, py, pr) in placed):
                    placed.append((x, y, scale))
                    results.append(dict(cls=cls, x=x, y=y, scale=scale,
                                        rot_z=rot_z, colliding=False))
                    break
        # If no valid placement found in 500 attempts, shape is silently
        # skipped (can happen at very close ortho with many non-colliding shapes).

    return results

# ══════════════════════════════════════════════════════════════════════
#  OCCLUDERS
# ══════════════════════════════════════════════════════════════════════

def add_occluders(n, ortho_scale):
    lim  = ortho_scale * 0.42
    objs = []
    for _ in range(n):
        ox = random.uniform(-lim, lim)
        oy = random.uniform(-lim, lim)
        bpy.ops.mesh.primitive_plane_add(size=1, location=(ox, oy, 0.12))
        occ = bpy.context.active_object; occ.name = "Occluder"
        occ.scale = (random.uniform(*OCCLUDER_SCALE_RANGE),
                     random.uniform(*OCCLUDER_SCALE_RANGE), 1.0)
        occ.rotation_euler = (0, 0, random.uniform(0, math.tau))
        v = random.uniform(0.05, 0.50)
        occ.data.materials.append(principled_mat(
            (v, v*random.uniform(0.7,1.0), v*random.uniform(0.4,0.8)),
            roughness=random.uniform(0.7,1.0), name="OccMat",
        ))
        objs.append(occ)
    return objs

# ══════════════════════════════════════════════════════════════════════
#  COMPOSITOR
# ══════════════════════════════════════════════════════════════════════

def setup_compositor(exposure, hue, sat, val, bright, contrast, blur_px, noise_strength):
    sc = bpy.context.scene; sc.use_nodes = True
    tree = sc.node_tree; nodes = tree.nodes; links = tree.links; nodes.clear()
    rl   = nodes.new("CompositorNodeRLayers");       rl.location   = (-900, 0)
    exp  = nodes.new("CompositorNodeExposure");      exp.location  = (-700, 0)
    hsv  = nodes.new("CompositorNodeHueSat");        hsv.location  = (-500, 0)
    bc   = nodes.new("CompositorNodeBrightContrast");bc.location   = (-300, 0)
    blur = nodes.new("CompositorNodeBlur");          blur.location = (-100, 0)
    mix  = nodes.new("CompositorNodeMixRGB");        mix.location  = ( 300, 0)
    comp = nodes.new("CompositorNodeComposite");     comp.location = ( 520, 0)
    exp.inputs["Exposure"].default_value = exposure
    for k, v in [("Hue", 0.5+hue), ("Saturation", sat), ("Value", val)]:
        if k in hsv.inputs: hsv.inputs[k].default_value = v
    for k, v in [("Bright", bright), ("Brightness", bright)]:
        if k in bc.inputs: bc.inputs[k].default_value = v; break
    if "Contrast" in bc.inputs: bc.inputs["Contrast"].default_value = contrast
    blur.filter_type = "GAUSS"
    blur.size_x = blur.size_y = max(0, int(blur_px))
    mix.blend_type = "ADD"
    mix.inputs["Fac"].default_value = noise_strength
    tex_name = "GS_NoiseTex"
    tex = bpy.data.textures.get(tex_name) or bpy.data.textures.new(tex_name, type="DISTORTED_NOISE")
    tex.noise_scale = random.uniform(*NOISE_SCALE_RANGE)
    tex.distortion  = random.uniform(*NOISE_DISTORTION_RANGE)
    tex_node        = nodes.new("CompositorNodeTexture")
    tex_node.texture= tex; tex_node.location = (-100, -240)
    cramp           = nodes.new("CompositorNodeValToRGB"); cramp.location = (100, -240)
    cramp.color_ramp.elements[0].color = (0,0,0,1)
    cramp.color_ramp.elements[1].color = (1,1,1,1)
    links.new(rl.outputs["Image"],       exp.inputs["Image"])
    links.new(exp.outputs["Image"],      hsv.inputs["Image"])
    links.new(hsv.outputs["Image"],      bc.inputs["Image"])
    links.new(bc.outputs["Image"],       blur.inputs["Image"])
    links.new(blur.outputs["Image"],     mix.inputs[1])
    links.new(tex_node.outputs["Value"], cramp.inputs["Fac"])
    links.new(cramp.outputs["Image"],    mix.inputs[2])
    links.new(mix.outputs["Image"],      comp.inputs["Image"])

def disable_compositor():
    sc = bpy.context.scene; sc.use_nodes = False
    if sc.node_tree: sc.node_tree.nodes.clear()

# ══════════════════════════════════════════════════════════════════════
#  RENDER HELPERS
# ══════════════════════════════════════════════════════════════════════

def _set_flat_render_settings():
    sc = bpy.context.scene
    sc.render.engine = "BLENDER_EEVEE_NEXT"
    sc.use_nodes = False
    sc.view_settings.view_transform = "Raw"
    sc.view_settings.look           = "None"
    set_display_device_safe("sRGB")

def _restore_rgb_settings():
    sc = bpy.context.scene
    sc.render.engine = "CYCLES"
    sc.view_settings.view_transform = "Standard"
    sc.view_settings.look           = "None"
    set_display_device_safe("sRGB")

def _set_mat(obj, mat):
    while obj.data.materials: obj.data.materials.pop()
    obj.data.materials.append(mat)

def _save_restore_materials(all_objs):
    return {o.name: [s.material for s in o.material_slots] for o in all_objs}

def _do_restore(all_objs, saved):
    for obj in all_objs:
        if obj.name not in saved: continue
        while obj.data.materials: obj.data.materials.pop()
        for m in saved[obj.name]:
            if m: obj.data.materials.append(m)

def render_rgb(path):
    sc = bpy.context.scene
    sc.render.engine = "CYCLES"; sc.render.filepath = path
    sc.view_settings.view_transform = "Standard"
    sc.view_settings.look           = "None"
    set_display_device_safe("sRGB")
    params = dict(
        exposure       = random.uniform(*EXPOSURE_RANGE),
        hue            = random.uniform(*HUE_JITTER),
        sat            = random.uniform(*SAT_RANGE),
        val            = random.uniform(*VAL_RANGE),
        bright         = random.uniform(*BRIGHTNESS_RANGE),
        contrast       = random.uniform(*CONTRAST_RANGE),
        blur_px        = random.uniform(*BLUR_PX_RANGE),
        noise_strength = random.uniform(*NOISE_STRENGTH_RANGE),
    )
    setup_compositor(**params)
    bpy.ops.render.render(write_still=True)
    disable_compositor()
    return params

def render_mask(path, shape_objs, ground, occluders):
    all_objs = [ground] + occluders + shape_objs
    saved    = _save_restore_materials(all_objs)
    black    = emission_mat((0,0,0), "MaskBlack")
    _set_mat(ground, black)
    for occ in occluders: _set_mat(occ, black)
    for obj in shape_objs:
        _set_mat(obj, emission_mat(obj["mask_rgb"], f"Mask_{obj['class_name']}"))
    _set_flat_render_settings()
    bpy.context.scene.render.filepath = path
    bpy.ops.render.render(write_still=True)
    _restore_rgb_settings()
    _do_restore(all_objs, saved)
    for m in list(bpy.data.materials):
        if m.name.startswith("Mask") or m.name == "MaskBlack":
            bpy.data.materials.remove(m)

def render_instance_mask(path, shape_objs, ground, occluders):
    all_objs = [ground] + occluders + shape_objs
    saved    = _save_restore_materials(all_objs)
    black    = emission_mat((0,0,0), "InstBlack")
    _set_mat(ground, black)
    for occ in occluders: _set_mat(occ, black)
    for obj in shape_objs:
        inst_id = int(obj.get("inst_id", 1))
        col     = INSTANCE_PALETTE[(inst_id - 1) % len(INSTANCE_PALETTE)]
        _set_mat(obj, emission_mat(col, f"Inst_{inst_id}"))
    _set_flat_render_settings()
    bpy.context.scene.render.filepath = path
    bpy.ops.render.render(write_still=True)
    _restore_rgb_settings()
    _do_restore(all_objs, saved)
    for m in list(bpy.data.materials):
        if m.name.startswith("Inst") or m.name == "InstBlack":
            bpy.data.materials.remove(m)

# ══════════════════════════════════════════════════════════════════════
#  CAMERA MOTION BLUR (drone vibration)
# ══════════════════════════════════════════════════════════════════════

def apply_camera_motion_jitter(cam):
    sc = bpy.context.scene
    sc.frame_set(1)
    cam.keyframe_insert(data_path="location",       frame=1)
    cam.keyframe_insert(data_path="rotation_euler", frame=1)
    cam.location.x       += random.uniform(-MOTION_JITTER_TRANS, MOTION_JITTER_TRANS)
    cam.location.y       += random.uniform(-MOTION_JITTER_TRANS, MOTION_JITTER_TRANS)
    cam.rotation_euler.z += random.uniform(-MOTION_JITTER_ROT,   MOTION_JITTER_ROT)
    cam.keyframe_insert(data_path="location",       frame=2)
    cam.keyframe_insert(data_path="rotation_euler", frame=2)
    sc.frame_set(1)
    sc.cycles.motion_blur_shutter = random.uniform(*MOTION_SHUTTER_RANGE)

# ══════════════════════════════════════════════════════════════════════
#  SPAWN SHAPE OBJECTS FROM PLACEMENTS
# ══════════════════════════════════════════════════════════════════════

def spawn_shapes(placements):
    shape_objs   = []
    class_counts = {c["name"]: 0 for c in SHAPE_CLASSES}
    for p in placements:
        cls   = p["cls"]
        mesh  = build_standard_shape(cls["shape_type"])
        obj   = bpy.data.objects.new(f"{cls['name']}_{len(shape_objs)}", mesh)
        link(obj)
        s   = p["scale"]
        alt = random.uniform(0.02, 0.07)
        obj.location       = (p["x"], p["y"], alt)
        obj.rotation_euler = (0, 0, p["rot_z"])
        obj.scale          = (s, s, s * random.uniform(0.3, 0.6))
        pbr_rgb = random.choice(cls["pbr_colors"])
        obj.data.materials.append(principled_mat(
            pbr_rgb,
            roughness=random.uniform(0.15, 0.65),
            specular =random.uniform(0.10, 0.85),
            name=f"PBR_{cls['name']}",
        ))
        obj["class_id"]   = cls["class_id"]
        obj["class_name"] = cls["name"]
        obj["mask_rgb"]   = cls["mask_rgb"]
        inst_id           = len(shape_objs) + 1
        obj["inst_id"]    = inst_id
        obj.pass_index    = inst_id
        shape_objs.append(obj)
        class_counts[cls["name"]] += 1
    return shape_objs, class_counts

# ══════════════════════════════════════════════════════════════════════
#  RENDER ONE IMAGE
# ══════════════════════════════════════════════════════════════════════

def render_image(idx, stage, stage_idx, ortho_scale, size_step,
                 tilt_x_deg, tilt_y_deg, yaw_deg,
                 placements, cam, out_dirs):
    """
    FIX 1: No random.seed() call here. The caller seeds once per image
    before computing tilt + placements + calling this function.
    All randomization shares a single deterministic state per image.

    FIX 4: hard_reset_camera() is called at the END so the camera is
    guaranteed clean before the next image's tilt is applied.
    """
    tag = f"s{stage}_{idx:06d}"
    print(f"  [{tag}] stage={stage} step={size_step} ortho={ortho_scale:.3f} "
          f"tilt=({tilt_x_deg:.1f},{tilt_y_deg:.1f}) yaw={yaw_deg:.0f}")

    # Clear per-image scene objects (keep camera)
    for obj in list(bpy.data.objects):
        if obj.name != "MainCamera":
            bpy.data.objects.remove(obj, do_unlink=True)
    for d in (bpy.data.meshes, bpy.data.materials):
        for b in list(d): d.remove(b)

    ground, ground_mode = add_ground()
    sun = add_sun()
    sun_energy, sun_angle_deg = randomise_sun(sun)
    sky_info = setup_world()

    cam.data.ortho_scale = ortho_scale
    # Apply the tilt values that were chosen BEFORE placement (FIX 2)
    apply_camera_tilt(cam, tilt_x_deg, tilt_y_deg, yaw_deg)

    shape_objs, class_counts = spawn_shapes(placements)
    n_colliding = sum(1 for p in placements if p.get("colliding", False))

    occluders = []
    if stage == 2 and ADD_OCCLUDERS and random.random() < OCCLUDER_PROB:
        occluders = add_occluders(random.randint(1, N_OCCLUDERS_MAX), ortho_scale)

    apply_camera_motion_jitter(cam)

    name = f"{idx:06d}.png"
    comp_params = render_rgb(os.path.join(out_dirs["images"],  name))
    render_mask         (os.path.join(out_dirs["masks"],     name), shape_objs, ground, occluders)
    render_instance_mask(os.path.join(out_dirs["instances"], name), shape_objs, ground, occluders)

    # FIX 4: reset after render, before next image
    hard_reset_camera(cam)

    sidecar = {
        "index":       idx,
        "stage":       stage,
        "size_step":   size_step,
        "ortho_scale": round(ortho_scale, 3),
        "ground_mode": ground_mode,
        "sky_info":    str(sky_info),
        "tilt_x_deg":  round(tilt_x_deg, 3),
        "tilt_y_deg":  round(tilt_y_deg, 3),
        "yaw_deg":     round(yaw_deg,    3),
        "instances":   [
            {
                "inst_id":    int(obj["inst_id"]),
                "class_id":   int(obj["class_id"]),
                "class_name": obj["class_name"],
                "x":          round(obj.location.x, 4),
                "y":          round(obj.location.y, 4),
                "rot_z_deg":  round(math.degrees(obj.rotation_euler.z), 2),
                "scale":      round(obj.scale.x, 4),
            }
            for obj in shape_objs
        ],
    }
    with open(os.path.join(out_dirs["annotations"], f"{idx:06d}.json"), "w") as jf:
        json.dump(sidecar, jf, indent=2)

    row = {
        "index":          idx,
        "stage":          stage,
        "size_step":      size_step,
        "image":          f"images/{name}",
        "mask":           f"masks/{name}",
        "instance_mask":  f"instances/{name}",
        "annotations":    f"annotations/{idx:06d}.json",
        "ortho_scale":    round(ortho_scale, 3),
        "ground_mode":    ground_mode,
        "n_shapes":       len(shape_objs),
        "n_colliding":    n_colliding,
        "has_occluders":  int(len(occluders) > 0),
        "n_occluders":    len(occluders),
        **{f"n_{c['name']}": class_counts[c["name"]] for c in SHAPE_CLASSES},
        "cam_tilt_x_deg": round(tilt_x_deg, 3),
        "cam_tilt_y_deg": round(tilt_y_deg, 3),
        "cam_yaw_deg":    round(yaw_deg,    3),
        "sun_energy":     round(sun_energy, 3),
        "sun_angle_deg":  round(sun_angle_deg, 3),
        "exposure":       round(comp_params["exposure"], 3),
        "blur_px":        round(comp_params["blur_px"],  3),
        "noise_strength": round(comp_params["noise_strength"], 4),
    }
    return row

# ══════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════

def make_out_dirs(stage_dir):
    dirs = {
        "images":      os.path.join(stage_dir, "images"),
        "masks":       os.path.join(stage_dir, "masks"),
        "instances":   os.path.join(stage_dir, "instances"),
        "annotations": os.path.join(stage_dir, "annotations"),
    }
    for d in dirs.values(): os.makedirs(d, exist_ok=True)
    return dirs

def write_manifest(rows, path):
    if not rows: return
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader(); w.writerows(rows)

def generate():
    """
    FIX 1 — seed discipline:
      • One master seed chosen here.
      • Per-image seed = (master + idx*7919) & 0xFFFFFFFF.
      • random.seed(img_seed) is called ONCE at the top of each image loop,
        before tilt, ortho jitter, placement, and rendering.
      • render_image() never calls random.seed() — it inherits the state.
    """
    global_seed = SEED if SEED is not None else random.randint(0, 2**31)
    print(f"\n  Master seed: {global_seed}")

    purge_scene()
    configure_render()
    cam = add_camera()

    all_rows   = []
    global_idx = 0

    # ── STAGE 1 ───────────────────────────────────────────────────────
    print("\n" + "="*60)
    print(f"  STAGE 1  — single shape, {len(STAGE1_ORTHO_STEPS)} size steps, {N_STAGE1} images")
    print("="*60)
    stage1_dirs = make_out_dirs(os.path.join(OUTPUT_DIR, "stage_1"))
    # cls   = random.choice(SHAPE_CLASSES)

    for cls in SHAPE_CLASSES:
        print(f"    Class {cls}")
        for img_i in range(N_STAGE1):
            # Single seed controls EVERYTHING for this image (FIX 1)
            img_seed = (global_seed + global_idx * 7919) & 0xFFFFFFFF
            random.seed(img_seed)

            step_idx    = img_i % len(STAGE1_ORTHO_STEPS)
            ortho_scale = STAGE1_ORTHO_STEPS[step_idx] * random.uniform(0.95, 1.1)

            # Choose tilt BEFORE placement so placement is tilt-aware (FIX 2)
            tilt_x = random.gauss(*CAMERA_TILT_X_RANGE)
            tilt_y = random.gauss(*CAMERA_TILT_Y_RANGE)
            yaw    = random.uniform(*CAMERA_YAW_RANGE)

            placements = place_single_shape(ortho_scale,cls, tilt_x, tilt_y)

            row = render_image(
                idx=global_idx, stage=1, stage_idx=img_i,
                ortho_scale=ortho_scale, size_step=step_idx,
                tilt_x_deg=tilt_x, tilt_y_deg=tilt_y, yaw_deg=yaw,
                placements=placements, cam=cam, out_dirs=stage1_dirs,
            )
            row["seed"] = img_seed
            all_rows.append(row)
            global_idx += 1

    write_manifest([r for r in all_rows if r["stage"] == 1],
                   os.path.join(OUTPUT_DIR, "stage_1", "manifest.csv"))

    # ── STAGE 2 ───────────────────────────────────────────────────────
    print("\n" + "="*60)
    print(f"  STAGE 2  — multi shape (2–3), {len(STAGE2_ORTHO_STEPS)} size steps, {N_STAGE2} images")
    print(f"             collision_prob = {COLLISION_PROB}")
    print("="*60)
    stage2_dirs = make_out_dirs(os.path.join(OUTPUT_DIR, "stage_2"))

    for img_i in range(N_STAGE2):
        img_seed = (global_seed + global_idx * 7919) & 0xFFFFFFFF
        random.seed(img_seed)

        step_idx    = img_i % len(STAGE2_ORTHO_STEPS)
        ortho_scale = STAGE2_ORTHO_STEPS[step_idx] * random.uniform(0.95, 1.05)

        tilt_x = random.gauss(*CAMERA_TILT_X_RANGE)
        tilt_y = random.gauss(*CAMERA_TILT_Y_RANGE)
        yaw    = random.uniform(*CAMERA_YAW_RANGE)

        n_shapes   = random.randint(2, 3)
        allow_col  = random.random() < COLLISION_PROB
        placements = place_multi_shapes(n_shapes, ortho_scale,
                                        tilt_x, tilt_y,
                                        allow_collision=allow_col)

        row = render_image(
            idx=global_idx, stage=2, stage_idx=img_i,
            ortho_scale=ortho_scale, size_step=step_idx,
            tilt_x_deg=tilt_x, tilt_y_deg=tilt_y, yaw_deg=yaw,
            placements=placements, cam=cam, out_dirs=stage2_dirs,
        )
        row["seed"] = img_seed
        all_rows.append(row)
        global_idx += 1

    write_manifest([r for r in all_rows if r["stage"] == 2],
                   os.path.join(OUTPUT_DIR, "stage_2", "manifest.csv"))

    write_manifest(all_rows, os.path.join(OUTPUT_DIR, "manifest_combined.csv"))

    print(f"\n{'='*60}")
    print(f"  Done!  {global_idx} images  →  {OUTPUT_DIR}")
    print(f"    Stage 1: {N_STAGE1} images, {len(STAGE1_ORTHO_STEPS)} size steps")
    print(f"    Stage 2: {N_STAGE2} images, {len(STAGE2_ORTHO_STEPS)} size steps, "
          f"collision_prob={COLLISION_PROB}")
    print(f"{'='*60}\n")

if __name__ == "__main__":
    generate()