from settings import Settings
from typing import List

source_path = None
target_path = None
output_path = None
target_folder_path = None
startup_args = None

cuda_device_id = 0
frame_processors: List[str] = []
keep_fps = None
keep_frames = None
autorotate_faces = None
vr_mode = None
skip_audio = None
wait_after_extraction = None
many_faces = None
use_batch = None
source_face_index = 0
target_face_index = 0
face_position = None
video_encoder = None
video_quality = None
max_memory = None
execution_providers: List[str] = []
execution_threads = None
headless = None
log_level = 'error'
selected_enhancer = None
subsample_size = 128
face_swap_mode = None
blend_ratio = 0.5
distance_threshold = 0.65
default_det_size = True

# --- Accuracy / quality toggles (RunLeashed improvements) ---
# Derive the 5 alignment keypoints from the stable 68-point landmark model
# (instead of the detector's raw kps) and fit the warp with a RANSAC affine.
# This is the main fix for "off" swaps at extreme yaw/pitch angles.
use_landmark_alignment = True
# Build the paste-back matte from the face convex hull (follows the face
# contour) instead of a plain rectangle, removing jaw/neck/background bleed
# at angled poses. Falls back to the rectangle if landmarks are unavailable.
use_face_hull_mask = True
# How far above the eyebrows to extend the hull to cover the forehead,
# as a fraction of the chin->brow distance.
face_hull_forehead = 0.6
# Dilate the hull outward by this fraction of the crop size so the swapped
# face fully covers the target (compensates the erosion in blur_area). Raise
# if the swap looks smaller than the target; lower if it bleeds onto the neck.
face_hull_dilate = 0.10
# Below this hull-area fraction of the crop, treat the hull as degenerate
# (profile / looking up) and fall back to the full rectangle matte so the face
# is never cut in half. 0 disables the guard.
face_hull_min_area = 0.22
# Optional Reinhard (LAB) color transfer of the swapped face toward the
# target region before paste-back. Off by default (inswapper already does
# reasonable color); enable if source/target lighting differs a lot.
use_color_transfer = False

# Multi-angle detection: try rotated copies of the frame so sideways / upside
# down faces are found and landmarked upright. 'off' | 'fallback' | 'always'.
# 'fallback' (default) only rotates when 0 deg finds nothing -> nearly free.
# 'always' unions all angles (max recall, ~4x detection cost).
multi_angle_detection_mode = 'fallback'
multi_angle_angles = [90, 270, 180]

# Temporal landmark smoothing for video (reduces per-frame jitter). Active only
# while processing video in-memory, or when force_landmark_smoothing is set
# (e.g. for the extract-frames video path). Never applied to unrelated image
# batches. strength in [0,1]; higher = smoother.
landmark_smoothing = True
landmark_smoothing_strength = 0.7
force_landmark_smoothing = False

# Faceset (multi-image source) identity averaging. 'robust' (default) drops
# outlier uploads by cosine distance from the group's median embedding, then
# takes a detector-confidence-weighted mean -> identity that resembles the real
# source more closely. 'median' picks the single most-central face. 'mean' is
# the old naive average. outlier_threshold: cosine distance cut for 'robust'
# (lower = stricter; raise toward 1.0 to keep very different angles).
faceset_average_mode = 'robust'
faceset_outlier_threshold = 0.6

# LivePortrait expression restorer (optional, faceswap tab). Re-injects the
# target's real expression onto the swapped face. Off by default (heavy model).
expression_restorer = False
expression_restorer_factor = 80     # 0-100 -> blend amount
expression_restore_eyes = True
expression_restore_mouth = True
expression_restore_brows = True
# 0 = no clamp (let the full target expression through). Set ~0.1 to gently
# clamp if very strong expressions ever cause artifacts.
expression_clamp = 0.0
# Amplify the driving expression delta. 1.0 = natural target amount; try 2.0-3.0
# to make subtle expressions clearly visible (may add artifacts if pushed high).
expression_power = 2.0
# This generator inverts expression if fed the intuitive way; True applies the
# corrected (verified) mapping so target expression transfers in the right
# direction. Leave True. (Toggle only if a future model export flips again.)
expression_invert_direction = True
# Border feather for the expression restorer: fraction of the crop radius where
# the LivePortrait result fades into the aligned swapped face. Keeps edges clean
# (no 'half face'). 0 disables. 0.2 default.
expression_blend_border = 0.2
# Full LivePortrait pipeline: crop the face the LivePortrait way (wider, incl.
# forehead) from the full frame + use the stitching model to lock pose. This is
# the precise path. Set False to fall back to the in-place arcface-crop method.
expression_full_pipeline = False
expression_stitching = False
lp_crop_size = 512
lp_crop_scale = 2.3
lp_crop_vy = -0.125
# Model-free pose lock for the expression restorer: re-centre (and re-scale)
# driving keypoints to the source so only the local expression transfers and the
# head doesn't shift/enlarge/drift. Works on the (default) old in-place approach.
expression_pose_lock = True
expression_pose_lock_scale = True
expression_pose_lock_rotation = True
# Serialise the LivePortrait sessions across threads (prevents cuDNN/illegal-memory
# crashes when Max Threads is high). True is safe; False = max speed, fewer threads.
expression_serialize = True

no_face_action = 0

processing = False

g_current_face_analysis = None
g_desired_face_analysis = None

FACE_ENHANCER = None

INPUT_FACESETS = []
TARGET_FACES = []


IMAGE_CHAIN_PROCESSOR = None
VIDEO_CHAIN_PROCESSOR = None
BATCH_IMAGE_CHAIN_PROCESSOR = None

CFG: Settings = None



# --- Diagnostics & onnxruntime CUDA tuning (added for slowness investigation) ---
# When True, prints per-stage timing for each processed/preview frame and logs
# every time a model session is (re)created. This is how we find out whether the
# slowness is a per-frame model reload or slow inference. Set False once diagnosed.
profile_timings = False
# onnxruntime CUDA convolution algo search. The onnxruntime default is
# 'EXHAUSTIVE', which can take a very long time on the first inference of large
# conv models (e.g. the LivePortrait generator) and can repeat when GPU memory is
# tight. 'HEURISTIC' avoids that long search at a negligible quality cost.
cudnn_conv_algo_search = 'HEURISTIC'
# False = don't reserve max cuDNN conv workspace (saves GBs of GPU RAM; prevents
# the CUDA out-of-memory when many models are loaded). True only if you have RAM.
cudnn_conv_use_max_workspace = False

# Preview delivery: the Gradio preview image is sent to the browser (often over a
# slow gradio.live share tunnel) as a full-resolution PNG, which can take a long
# time to appear even though the swap compute is fast. For the PREVIEW ONLY we
# downscale to this max height and let Gradio encode it as JPEG. Final renders are
# unaffected. Raise this if you want a sharper (but slower-to-load) preview.
preview_max_height = 720
