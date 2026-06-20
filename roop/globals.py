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
profile_timings = True
# onnxruntime CUDA convolution algo search. The onnxruntime default is
# 'EXHAUSTIVE', which can take a very long time on the first inference of large
# conv models (e.g. the LivePortrait generator) and can repeat when GPU memory is
# tight. 'HEURISTIC' avoids that long search at a negligible quality cost.
cudnn_conv_algo_search = 'HEURISTIC'
