import os
import numpy as np
import cv2
import onnxruntime
import roop.globals

from roop.typing import Frame
from roop.utilities import resolve_relative_path, conditional_thread_semaphore


# BiSeNet (CelebAMask-HQ, 19 classes) face-parsing mask.
# This is an OPTIONAL masking engine, wired the same way as Mask_XSeg /
# Mask_Clip2Seg: the user selects it from the "Face masking engine" menu.
# It is never enabled by default.
#
# 19-class layout (zllrunning/face-parsing.PyTorch / facefusion bisenet):
#   0 background  1 skin      2 l_brow   3 r_brow   4 l_eye    5 r_eye
#   6 eye_glasses 7 l_ear     8 r_ear    9 earring  10 nose    11 mouth
#   12 upper_lip  13 lower_lip 14 neck    15 necklace 16 cloth  17 hair  18 hat
#
# We keep the swapped face only over the actual facial region and restore the
# original pixels everywhere else (hair, ears, neck, cloth, hat, background),
# which is exactly what fixes hair/hands/jewellery drawn over the face.
FACE_CLASSES = [1, 2, 3, 4, 5, 10, 11, 12, 13]   # glasses (6) intentionally excluded

# Common filenames people ship this model under; first existing one is used.
CANDIDATE_MODELS = [
    'faceparser.onnx',
    'face_parser.onnx',
    'bisenet_resnet_34.onnx',
    'bisenet.onnx',
    '79999_iter.onnx',
]


class Mask_FaceParser():
    plugin_options: dict = None

    model_parser = None
    input_name = None
    output_name = None
    input_size = 512
    devicename = 'cpu'

    processorname = 'mask_faceparser'
    type = 'mask'

    def _find_model(self):
        for name in CANDIDATE_MODELS:
            p = resolve_relative_path('../models/' + name)
            if os.path.exists(p):
                return p
        return None

    def Initialize(self, plugin_options: dict):
        if self.plugin_options is not None:
            if self.plugin_options["devicename"] != plugin_options["devicename"]:
                self.Release()

        self.plugin_options = plugin_options
        if self.model_parser is None:
            model_path = self._find_model()
            if model_path is None:
                # keep state consistent (instance attr, not just class default)
                self.model_parser = None
                self.input_name = None
                self.output_name = None
                print("[Mask_FaceParser] No face-parsing model found in ./models "
                      "(expected one of: %s). This mask engine will be a no-op."
                      % ", ".join(CANDIDATE_MODELS))
                return
            onnxruntime.set_default_logger_severity(3)
            self.model_parser = onnxruntime.InferenceSession(
                model_path, None, providers=roop.globals.execution_providers
            )
            inp = self.model_parser.get_inputs()[0]
            self.input_name = inp.name
            self.output_name = self.model_parser.get_outputs()[0].name
            # infer square input size from the model if it is static
            try:
                shp = inp.shape
                if isinstance(shp[-1], int) and shp[-1] > 0:
                    self.input_size = int(shp[-1])
            except Exception:
                self.input_size = 512
            self.devicename = self.plugin_options["devicename"].replace('mps', 'cpu')

    def Run(self, img1, keywords: str) -> Frame:
        # If no model is loaded, return an all-zero mask -> keep the swapped
        # face everywhere (i.e. behaves like "no masking", never crashes).
        h, w = img1.shape[:2]
        if self.model_parser is None:
            return np.zeros((h, w), dtype=np.float32)

        s = self.input_size
        temp = cv2.resize(img1, (s, s), interpolation=cv2.INTER_LINEAR)
        temp = temp[:, :, ::-1].astype(np.float32) / 255.0          # BGR -> RGB, [0,1]
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        temp = (temp - mean) / std
        temp = np.transpose(temp, (2, 0, 1))[None, ...].astype(np.float32)  # NCHW

        with conditional_thread_semaphore():
            out = self.model_parser.run([self.output_name], {self.input_name: temp})[0]

        out = np.asarray(out)
        out = np.squeeze(out)                # -> (19, S, S) or (S, S)
        if out.ndim == 3:
            seg = np.argmax(out, axis=0)     # per-pixel class id
        else:
            seg = out
        face_region = np.isin(seg, FACE_CLASSES).astype(np.float32)   # 1 = facial

        # soften, resize back to crop size
        face_region = cv2.GaussianBlur(face_region, (0, 0), sigmaX=3)
        face_region = cv2.resize(face_region, (w, h), interpolation=cv2.INTER_LINEAR)
        face_region = np.clip(face_region, 0.0, 1.0)

        # pipeline convention (see ProcessMgr.process_mask / Mask_XSeg):
        # returned mask is high where the ORIGINAL should be kept.
        return 1.0 - face_region

    def Release(self):
        # Never use `del self.attr` here: when the model file was missing,
        # Initialize returns early and model_parser is only the *class* default,
        # so `del` would raise AttributeError and wedge engine-switching.
        self.model_parser = None
        self.input_name = None
        self.output_name = None
