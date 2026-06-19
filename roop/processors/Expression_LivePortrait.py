import os
import numpy as np
import cv2
import onnxruntime
import roop.globals

from roop.typing import Face, Frame
from roop.utilities import resolve_relative_path, conditional_thread_semaphore
from roop.processors import live_portrait_util as lpu


# Optional LivePortrait-powered expression restorer. Re-injects the TARGET
# face's expression (eyes/mouth/brows) onto the already-SWAPPED face so the
# swap keeps the source identity but wears the target's real expression.
# Three ONNX models (FaceFusion live_portrait set). Never enabled by default;
# if the models are missing it is a no-op (returns the swapped crop unchanged).
MODELS = {
    'feature_extractor': 'live_portrait_feature_extractor.onnx',
    'motion_extractor':  'live_portrait_motion_extractor.onnx',
    'generator':         'live_portrait_generator.onnx',
}


class Expression_LivePortrait():
    plugin_options: dict = None

    feature_extractor = None
    motion_extractor = None
    generator = None
    devicename = 'cpu'

    processorname = 'expression_restorer'
    type = 'expression'

    def _load(self, name):
        path = resolve_relative_path('../models/liveportrait/' + MODELS[name])
        if not os.path.exists(path):
            path = resolve_relative_path('../models/' + MODELS[name])
        if not os.path.exists(path):
            return None
        onnxruntime.set_default_logger_severity(3)
        return onnxruntime.InferenceSession(path, None, providers=roop.globals.execution_providers)

    def Initialize(self, plugin_options: dict):
        if self.plugin_options is not None:
            if self.plugin_options["devicename"] != plugin_options["devicename"]:
                self.Release()
        self.plugin_options = plugin_options
        if self.feature_extractor is None:
            self.feature_extractor = self._load('feature_extractor')
            self.motion_extractor = self._load('motion_extractor')
            self.generator = self._load('generator')
            if None in (self.feature_extractor, self.motion_extractor, self.generator):
                print("[Expression_LivePortrait] LivePortrait models not found in "
                      "./models/liveportrait/. Expression restore will be a no-op.")
            self.devicename = self.plugin_options["devicename"].replace('mps', 'cpu')

    def _run_session(self, session, feeds):
        with conditional_thread_semaphore():
            return session.run(None, feeds)

    def _extract_motion(self, crop_chw):
        name = self.motion_extractor.get_inputs()[0].name
        out = self._run_session(self.motion_extractor, {name: crop_chw})
        # FaceFusion motion_extractor output order:
        # [pitch, yaw, roll, scale, translation, expression, motion_points]
        pitch, yaw, roll, scale, translation, expression, motion_points = out[:7]
        return pitch, yaw, roll, scale, translation, expression, motion_points

    def Run(self, target_crop: Frame, swapped_crop: Frame,
            factor: float, use_eyes: bool, use_mouth: bool, use_brows: bool) -> Frame:
        # Safety / no-op short circuits
        if None in (self.feature_extractor, self.motion_extractor, self.generator):
            return swapped_crop
        if factor <= 0.0 or not (use_eyes or use_mouth or use_brows):
            return swapped_crop

        try:
            src = lpu.prepare_crop(swapped_crop, 256)   # appearance + source motion
            drv = lpu.prepare_crop(target_crop, 256)    # driving expression

            f_name = self.feature_extractor.get_inputs()[0].name
            feature_volume = self._run_session(self.feature_extractor, {f_name: src})[0]

            (pitch, yaw, roll, scale, translation,
             temp_exp, motion_points) = self._extract_motion(src)
            target_exp = self._extract_motion(drv)[5]

            rotation = lpu.get_rotation_matrix(
                lpu.headpose_pred_to_degree(pitch),
                lpu.headpose_pred_to_degree(yaw),
                lpu.headpose_pred_to_degree(roll),
            )
            applied = lpu.build_applied_expression(
                temp_exp, target_exp, factor, use_eyes, use_mouth, use_brows)

            kp_source = lpu.transform_motion_points(
                motion_points, rotation, temp_exp, scale, translation)
            kp_driving = lpu.transform_motion_points(
                motion_points, rotation, applied, scale, translation)

            g_in = self.generator.get_inputs()
            feeds = {
                g_in[0].name: np.ascontiguousarray(feature_volume, dtype=np.float32),
                g_in[1].name: np.ascontiguousarray(kp_source, dtype=np.float32),
                g_in[2].name: np.ascontiguousarray(kp_driving, dtype=np.float32),
            }
            gen_out = self._run_session(self.generator, feeds)[0]

            restored = lpu.parse_output(gen_out)
            if (restored.shape[1], restored.shape[0]) != (swapped_crop.shape[1], swapped_crop.shape[0]):
                restored = cv2.resize(
                    restored, (swapped_crop.shape[1], swapped_crop.shape[0]),
                    interpolation=cv2.INTER_AREA)
            return restored
        except Exception as e:
            print("[Expression_LivePortrait] restore failed, passing through:", e)
            return swapped_crop

    def Release(self):
        self.feature_extractor = None
        self.motion_extractor = None
        self.generator = None
