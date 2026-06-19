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
# Optional Reinhard (LAB) color transfer of the swapped face toward the
# target region before paste-back. Off by default (inswapper already does
# reasonable color); enable if source/target lighting differs a lot.
use_color_transfer = False

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


