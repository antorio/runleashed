import threading
from typing import Any
import insightface

import roop.globals
from roop.typing import Frame, Face

import cv2
import numpy as np
from skimage import transform as trans
from roop.capturer import get_video_frame
from roop.utilities import resolve_relative_path, conditional_thread_semaphore

FACE_ANALYSER = None
#THREAD_LOCK_ANALYSER = threading.Lock()
#THREAD_LOCK_SWAPPER = threading.Lock()
FACE_SWAPPER = None


def get_face_analyser() -> Any:
    global FACE_ANALYSER

    with conditional_thread_semaphore():
        if FACE_ANALYSER is None or roop.globals.g_current_face_analysis != roop.globals.g_desired_face_analysis:
            model_path = resolve_relative_path('..')
            # removed genderage
            allowed_modules = roop.globals.g_desired_face_analysis
            roop.globals.g_current_face_analysis = roop.globals.g_desired_face_analysis
            if roop.globals.CFG.force_cpu:
                print("Forcing CPU for Face Analysis")
                FACE_ANALYSER = insightface.app.FaceAnalysis(
                    name="buffalo_l",
                    root=model_path, providers=["CPUExecutionProvider"],allowed_modules=allowed_modules
                )
            else:
                FACE_ANALYSER = insightface.app.FaceAnalysis(
                    name="buffalo_l", root=model_path, providers=roop.globals.execution_providers,allowed_modules=allowed_modules
                )
            FACE_ANALYSER.prepare(
                ctx_id=0,
                det_size=(640, 640) if roop.globals.default_det_size else (320, 320),
            )
    return FACE_ANALYSER


def get_first_face(frame: Frame) -> Any:
    try:
        faces = get_face_analyser().get(frame)
        return min(faces, key=lambda x: x.bbox[0])
    #   return sorted(faces, reverse=True, key=lambda x: (x.bbox[2] - x.bbox[0]) * (x.bbox[3] - x.bbox[1]))[0]
    except:
        return None


def get_all_faces(frame: Frame) -> Any:
    try:
        faces = get_face_analyser().get(frame)
        return sorted(faces, key=lambda x: x.bbox[0])
    except:
        return None


def extract_face_images(source_filename, video_info, extra_padding=-1.0):
    face_data = []
    source_image = None

    if video_info[0]:
        frame = get_video_frame(source_filename, video_info[1])
        if frame is not None:
            source_image = frame
        else:
            return face_data
    else:
        source_image = cv2.imdecode(np.fromfile(source_filename, dtype=np.uint8), cv2.IMREAD_COLOR)

    faces = get_all_faces(source_image)
    if faces is None:
        return face_data

    i = 0
    for face in faces:
        (startX, startY, endX, endY) = face["bbox"].astype("int")
        startX, endX, startY, endY = clamp_cut_values(startX, endX, startY, endY, source_image)
        if extra_padding > 0.0:
            if source_image.shape[:2] == (512, 512):
                i += 1
                face_data.append([face, source_image])
                continue

            found = False
            for i in range(1, 3):
                (startX, startY, endX, endY) = face["bbox"].astype("int")
                startX, endX, startY, endY = clamp_cut_values(startX, endX, startY, endY, source_image)
                cutout_padding = extra_padding
                # top needs extra room for detection
                padding = int((endY - startY) * cutout_padding)
                oldY = startY
                startY -= padding

                factor = 0.25 if i == 1 else 0.5
                cutout_padding = factor
                padding = int((endY - oldY) * cutout_padding)
                endY += padding
                padding = int((endX - startX) * cutout_padding)
                startX -= padding
                endX += padding
                startX, endX, startY, endY = clamp_cut_values(
                    startX, endX, startY, endY, source_image
                )
                face_temp = source_image[startY:endY, startX:endX]
                face_temp = resize_image_keep_content(face_temp)
                testfaces = get_all_faces(face_temp)
                if testfaces is not None and len(testfaces) > 0:
                    i += 1
                    face_data.append([testfaces[0], face_temp])
                    found = True
                    break

            if not found:
                print("No face found after resizing, this shouldn't happen!")
            continue

        face_temp = source_image[startY:endY, startX:endX]
        if face_temp.size < 1:
            continue

        i += 1
        face_data.append([face, face_temp])
    return face_data


def clamp_cut_values(startX, endX, startY, endY, image):
    if startX < 0:
        startX = 0
    if endX > image.shape[1]:
        endX = image.shape[1]
    if startY < 0:
        startY = 0
    if endY > image.shape[0]:
        endY = image.shape[0]
    return startX, endX, startY, endY



def face_offset_top(face: Face, offset):
    face["bbox"][1] += offset
    face["bbox"][3] += offset
    lm106 = face.landmark_2d_106
    add = np.full_like(lm106, [0, offset])
    face["landmark_2d_106"] = lm106 + add
    return face


def resize_image_keep_content(image, new_width=512, new_height=512):
    dim = None
    (h, w) = image.shape[:2]
    if h > w:
        r = new_height / float(h)
        dim = (int(w * r), new_height)
    else:
        # Calculate the ratio of the width and construct the dimensions
        r = new_width / float(w)
        dim = (new_width, int(h * r))
    image = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)
    (h, w) = image.shape[:2]
    if h == new_height and w == new_width:
        return image
    resize_img = np.zeros(shape=(new_height, new_width, 3), dtype=image.dtype)
    offs = (new_width - w) if h == new_height else (new_height - h)
    startoffs = int(offs // 2) if offs % 2 == 0 else int(offs // 2) + 1
    offs = int(offs // 2)

    if h == new_height:
        resize_img[0:new_height, startoffs : new_width - offs] = image
    else:
        resize_img[startoffs : new_height - offs, 0:new_width] = image
    return resize_img


def rotate_image_90(image, rotate=True):
    if rotate:
        return np.rot90(image)
    else:
        return np.rot90(image, 1, (1, 0))


def rotate_anticlockwise(frame):
    return rotate_image_90(frame)


def rotate_clockwise(frame):
    return rotate_image_90(frame, False)


def rotate_image_180(image):
    return np.flip(image, 0)


# alignment code from insightface https://github.com/deepinsight/insightface/blob/master/python-package/insightface/utils/face_align.py

arcface_dst = np.array(
    [
        [38.2946, 51.6963],
        [73.5318, 51.5014],
        [56.0252, 71.7366],
        [41.5493, 92.3655],
        [70.7299, 92.2041],
    ],
    dtype=np.float32,
)


""" def estimate_norm(lmk, image_size=112):
    assert lmk.shape == (5, 2)
    if image_size % 112 == 0:
        ratio = float(image_size) / 112.0
        diff_x = 0
    elif image_size % 128 == 0:
        ratio = float(image_size) / 128.0
        diff_x = 8.0 * ratio
    elif image_size % 512 == 0:
        ratio = float(image_size) / 512.0
        diff_x = 32.0 * ratio

    dst = arcface_dst * ratio
    dst[:, 0] += diff_x
    tform = trans.SimilarityTransform()
    tform.estimate(lmk, dst)
    M = tform.params[0:2, :]
    return M
 """

def estimate_norm(lmk, image_size=112):
    if image_size%112==0:
        ratio = float(image_size)/112.0
        diff_x = 0
    else:
        ratio = float(image_size)/128.0
        diff_x = 8.0*ratio
    dst = arcface_dst * ratio
    dst[:,0] += diff_x

    if image_size == 160:
        dst[:,0] += 0.1
        dst[:,1] += 0.1
    elif image_size == 256:
        dst[:,0] += 0.5
        dst[:,1] += 0.5
    elif image_size == 320:
        dst[:,0] += 0.75
        dst[:,1] += 0.75
    elif image_size == 512:
        dst[:,0] += 1.5
        dst[:,1] += 1.5

    tform = trans.SimilarityTransform()
    tform.estimate(lmk, dst)
    M = tform.params[0:2, :]
    return M


def _get_pitch_roll_from_kps(kps: np.ndarray):
    """
    Estimate rough pitch (head tilt up/down) and roll (head tilt sideways)
    from 5 keypoints: [left_eye, right_eye, nose, left_mouth, right_mouth].
    Returns (pitch_deg, roll_deg).
    """
    left_eye   = kps[0]
    right_eye  = kps[1]
    nose       = kps[2]
    left_mouth = kps[3]
    right_mouth= kps[4]

    eye_center    = (left_eye + right_eye) / 2.0
    mouth_center  = (left_mouth + right_mouth) / 2.0

    # Roll: angle of eye line relative to horizontal
    eye_delta = right_eye - left_eye
    roll_deg  = float(np.degrees(np.arctan2(eye_delta[1], eye_delta[0])))

    # Pitch proxy: where nose sits relative to eye-mouth midpoint
    # Positive = nose above midpoint (head tilted back / looking up)
    face_height   = np.linalg.norm(mouth_center - eye_center) + 1e-6
    midpoint_y    = (eye_center[1] + mouth_center[1]) / 2.0
    nose_offset_y = midpoint_y - nose[1]   # positive = nose above midpoint
    pitch_deg     = float(np.degrees(np.arcsin(np.clip(nose_offset_y / face_height, -1.0, 1.0))))

    return pitch_deg, roll_deg


def _rotate_kps(kps: np.ndarray, angle_deg: float, center: np.ndarray) -> np.ndarray:
    """Rotate 5 keypoints around a center by angle_deg degrees."""
    theta  = np.radians(angle_deg)
    cos_t, sin_t = np.cos(theta), np.sin(theta)
    R = np.array([[cos_t, -sin_t],
                  [sin_t,  cos_t]], dtype=np.float32)
    return (kps - center) @ R.T + center


# aligned, M = norm_crop2(f[1], face.kps, 512)
def align_crop(img, landmark, image_size=112, mode="arcface"):
    """
    Pose-aware align_crop.

    For extreme upward/downward pitch (head leaning back or forward, e.g. lying down
    looking at the ceiling) the standard SimilarityTransform from InsightFace kps
    produces a badly-rotated crop because the 5-point template assumes a roughly
    frontal face.  We detect the pitch from kps geometry and pre-rotate the
    keypoints (and the image) so that the template fit always sees a near-frontal
    arrangement, then apply an inverse rotation to the affine matrix so the final
    paste lands in the right orientation.

    Roll correction (sideways tilt) is also applied, fixing the "tilted face" case.
    """
    kps = landmark.astype(np.float32)

    pitch_deg, roll_deg = _get_pitch_roll_from_kps(kps)

    # Decide how much to correct.
    # Roll: always compensate fully — InsightFace handles mild roll but breaks >~25 deg.
    # Pitch: compensate when nose is clearly above eye-mouth midpoint (looking-up pose).
    #        Threshold ±15 deg to avoid over-correcting normal slight tilts.
    correction_angle = 0.0

    if abs(roll_deg) > 15.0:
        correction_angle += -roll_deg          # counter-rotate

    if pitch_deg > 15.0:
        # Head tilted back (looking up) — rotate frame so face appears more frontal.
        # Amount: scale pitch to a frame rotation (empirically ~0.6× pitch works well).
        correction_angle += pitch_deg * 0.6
    elif pitch_deg < -15.0:
        correction_angle += pitch_deg * 0.6

    if abs(correction_angle) > 3.0:
        # Rotate image patch around face center before alignment
        h, w = img.shape[:2]
        face_center = kps.mean(axis=0)

        # Build rotation matrix for the image
        Mrot = cv2.getRotationMatrix2D(
            (float(face_center[0]), float(face_center[1])),
            correction_angle, 1.0
        )
        img_rot = cv2.warpAffine(img, Mrot, (w, h), borderMode=cv2.BORDER_REPLICATE)

        # Rotate keypoints the same way
        R2x2 = Mrot[:, :2].T          # 2×2 rotation part (transposed = inverse for pure rotation)
        kps_rot = (kps - face_center) @ R2x2 + face_center + Mrot[:, 2]

        M = estimate_norm(kps_rot, image_size)
        warped = cv2.warpAffine(img_rot, M, (image_size, image_size), borderValue=0.0)

        # Compose the two transforms so paste_upscale can invert correctly:
        # final_M maps original img → aligned crop space
        # M_full = M_estimate ∘ M_rotate  (apply rotate first, then estimate_norm)
        M_full = np.vstack([M, [0, 0, 1]]) @ np.vstack([Mrot, [0, 0, 1]])
        return warped, M_full[:2]

    # Standard path — no significant pose issue
    M = estimate_norm(kps, image_size)
    warped = cv2.warpAffine(img, M, (image_size, image_size), borderValue=0.0)
    return warped, M


def square_crop(im, S):
    if im.shape[0] > im.shape[1]:
        height = S
        width = int(float(im.shape[1]) / im.shape[0] * S)
        scale = float(S) / im.shape[0]
    else:
        width = S
        height = int(float(im.shape[0]) / im.shape[1] * S)
        scale = float(S) / im.shape[1]
    resized_im = cv2.resize(im, (width, height))
    det_im = np.zeros((S, S, 3), dtype=np.uint8)
    det_im[: resized_im.shape[0], : resized_im.shape[1], :] = resized_im
    return det_im, scale


def transform(data, center, output_size, scale, rotation):
    scale_ratio = scale
    rot = float(rotation) * np.pi / 180.0
    # translation = (output_size/2-center[0]*scale_ratio, output_size/2-center[1]*scale_ratio)
    t1 = trans.SimilarityTransform(scale=scale_ratio)
    cx = center[0] * scale_ratio
    cy = center[1] * scale_ratio
    t2 = trans.SimilarityTransform(translation=(-1 * cx, -1 * cy))
    t3 = trans.SimilarityTransform(rotation=rot)
    t4 = trans.SimilarityTransform(translation=(output_size / 2, output_size / 2))
    t = t1 + t2 + t3 + t4
    M = t.params[0:2]
    cropped = cv2.warpAffine(data, M, (output_size, output_size), borderValue=0.0)
    return cropped, M


def trans_points2d(pts, M):
    new_pts = np.zeros(shape=pts.shape, dtype=np.float32)
    for i in range(pts.shape[0]):
        pt = pts[i]
        new_pt = np.array([pt[0], pt[1], 1.0], dtype=np.float32)
        new_pt = np.dot(M, new_pt)
        # print('new_pt', new_pt.shape, new_pt)
        new_pts[i] = new_pt[0:2]

    return new_pts


def trans_points3d(pts, M):
    scale = np.sqrt(M[0][0] * M[0][0] + M[0][1] * M[0][1])
    # print(scale)
    new_pts = np.zeros(shape=pts.shape, dtype=np.float32)
    for i in range(pts.shape[0]):
        pt = pts[i]
        new_pt = np.array([pt[0], pt[1], 1.0], dtype=np.float32)
        new_pt = np.dot(M, new_pt)
        # print('new_pt', new_pt.shape, new_pt)
        new_pts[i][0:2] = new_pt[0:2]
        new_pts[i][2] = pts[i][2] * scale

    return new_pts


def trans_points(pts, M):
    if pts.shape[1] == 2:
        return trans_points2d(pts, M)
    else:
        return trans_points3d(pts, M)
    
def create_blank_image(width, height):
    img = np.zeros((height, width, 4), dtype=np.uint8)
    img[:] = [0,0,0,0]
    return img

