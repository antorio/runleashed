import os
import threading
import numpy as np
import cv2
import onnxruntime
import roop.globals

from roop.typing import Face, Frame
from roop.utilities import resolve_relative_path, conditional_thread_semaphore
from roop.processors import live_portrait_util as lpu

# Serialises the heavy LivePortrait CUDA sessions across worker threads. roop's
# conditional_thread_semaphore is a no-op on CUDA, so 8 threads otherwise hammer
# the same generator session concurrently -> cuDNN EXECUTION_FAILED / illegal
# memory access. This lock makes the expression step thread-safe at any thread count.
_LP_LOCK = threading.Lock()


# Optional LivePortrait-powered expression restorer. Re-injects the TARGET
# face's expression (eyes/mouth/brows) onto the already-SWAPPED face so the
# swap keeps the source identity but wears the target's real expression.
# Three ONNX models (FaceFusion live_portrait set). Never enabled by default;
# if the models are missing it is a no-op (returns the swapped crop unchanged).
MODELS = {
    'feature_extractor': 'live_portrait_feature_extractor.onnx',
    'motion_extractor':  'live_portrait_motion_extractor.onnx',
    'generator':         'live_portrait_generator.onnx',
    'stitcher':          'live_portrait_stitcher.onnx',
}


class Expression_LivePortrait():
    plugin_options: dict = None

    feature_extractor = None
    motion_extractor = None
    generator = None
    stitcher = None
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
        from roop.utilities import tuned_execution_providers
        provs = tuned_execution_providers()
        print(f"[load] Creating LivePortrait '{name}' session  providers={provs}")
        return onnxruntime.InferenceSession(path, None, providers=provs)

    def Initialize(self, plugin_options: dict):
        if self.plugin_options is not None:
            if self.plugin_options["devicename"] != plugin_options["devicename"]:
                self.Release()
        self.plugin_options = plugin_options
        if self.feature_extractor is None:
            self.feature_extractor = self._load('feature_extractor')
            self.motion_extractor = self._load('motion_extractor')
            self.generator = self._load('generator')
            self.stitcher = self._load('stitcher')  # optional: locks pose (no drift)
            if None in (self.feature_extractor, self.motion_extractor, self.generator):
                print("[Expression_LivePortrait] LivePortrait models not found in "
                      "./models/liveportrait/. Expression restore will be a no-op.")
            self.devicename = self.plugin_options["devicename"].replace('mps', 'cpu')

    def _run_session(self, session, feeds):
        safe = {}
        for k, v in feeds.items():
            a = np.ascontiguousarray(v, dtype=np.float32)
            if not np.isfinite(a).all():
                a = np.nan_to_num(a, nan=0.0, posinf=0.0, neginf=0.0)
            safe[k] = a
        if getattr(roop.globals, 'expression_serialize', True):
            with _LP_LOCK:
                return session.run(None, safe)
        with conditional_thread_semaphore():
            return session.run(None, safe)

    def _extract_motion(self, crop_chw):
        name = self.motion_extractor.get_inputs()[0].name
        out = self._run_session(self.motion_extractor, {name: crop_chw})
        # FaceFusion motion_extractor output order:
        # [pitch, yaw, roll, scale, translation, expression, motion_points]
        pitch, yaw, roll, scale, translation, expression, motion_points = out[:7]
        return pitch, yaw, roll, scale, translation, expression, motion_points

    def Run(self, target_crop: Frame, swapped_crop: Frame,
            factor: float, use_eyes: bool, use_mouth: bool, use_brows: bool,
            context=None) -> Frame:
        # Safety / no-op short circuits
        if None in (self.feature_extractor, self.motion_extractor, self.generator):
            return swapped_crop
        if factor <= 0.0 or not (use_eyes or use_mouth or use_brows):
            return swapped_crop

        try:
            # ---- Full LivePortrait pipeline: crop the face the way LivePortrait
            # expects (wider framing incl. forehead) from the FULL frame, instead
            # of the tight arcface crop. This is what makes the expression precise.
            full = (getattr(roop.globals, 'expression_full_pipeline', True)
                    and context is not None and context.get('frame') is not None
                    and context.get('landmarks') is not None
                    and context.get('M') is not None)
            back_M = None
            if full:
                try:
                    frame = context['frame']
                    M_arc = np.asarray(context['M'], dtype=np.float32)
                    H, W = frame.shape[:2]
                    dsize = int(getattr(roop.globals, 'lp_crop_size', 512))
                    M_lp = lpu.lp_crop_matrix(
                        context['landmarks'], dsize=dsize,
                        scale=float(getattr(roop.globals, 'lp_crop_scale', 2.3)),
                        vy_ratio=float(getattr(roop.globals, 'lp_crop_vy', -0.125)))
                    # rebuild the swapped frame (paste the arcface swapped crop back)
                    invM = cv2.invertAffineTransform(M_arc)
                    face_full = cv2.warpAffine(swapped_crop, invM, (W, H))
                    mask_full = cv2.warpAffine(
                        np.full(swapped_crop.shape[:2], 255, np.uint8), invM, (W, H))
                    swapped_frame = frame.copy()
                    swapped_frame[mask_full > 0] = face_full[mask_full > 0]
                    source_img = cv2.warpAffine(swapped_frame, M_lp, (dsize, dsize))
                    driving_img = cv2.warpAffine(frame, M_lp, (dsize, dsize))
                    back_M = lpu.compose_affine(M_arc, cv2.invertAffineTransform(M_lp))
                except Exception:
                    full = False
            if not full:
                source_img = swapped_crop
                driving_img = target_crop

            src = lpu.prepare_crop(source_img, 256)   # appearance + source motion
            drv = lpu.prepare_crop(driving_img, 256)  # driving expression

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

            # Stitching locks the driving keypoints onto the source pose/position
            # so the generator does not drift / rotate the head (LivePortrait's
            # own fix for exactly that artefact).
            if self.stitcher is not None and getattr(roop.globals, 'expression_stitching', False):
                try:
                    kp_driving = lpu.apply_stitching(
                        self.stitcher, kp_source, kp_driving,
                        log=not getattr(self, '_stitch_io_logged', False),
                        invert=getattr(roop.globals, 'expression_invert_direction', True))
                    self._stitch_io_logged = True
                except Exception:
                    pass

            # Deterministic pose lock (model-free): keep the driving keypoints'
            # global position/scale identical to the source so only the local
            # expression transfers -- removes the head shift / enlarge / drift.
            kp_driving_raw = kp_driving  # snapshot before pose lock (diagnostics)
            if getattr(roop.globals, 'expression_pose_lock', True):
                kp_driving = lpu.lock_pose(
                    kp_driving, kp_source,
                    scale_lock=getattr(roop.globals, 'expression_pose_lock_scale', True),
                    rotation_lock=getattr(roop.globals, 'expression_pose_lock_rotation', False))

            g_in = self.generator.get_inputs()
            if not getattr(self, '_gen_io_logged', False):
                try:
                    info = [(i.name, i.shape) for i in g_in]
                    print(f"[Expression_LivePortrait] generator inputs (name, shape) = {info}")
                except Exception:
                    pass
                self._gen_io_logged = True

            feeds = {}
            feature_name = None
            kp_inputs = []
            for i in g_in:
                nm = i.name.lower()
                shp = i.shape
                if ('feature' in nm) or ('3d' in nm) or (isinstance(shp, (list, tuple)) and len(shp) == 5):
                    feature_name = i.name
                else:
                    kp_inputs.append(i.name)
            if feature_name is None:
                feature_name = g_in[0].name
                kp_inputs = [g_in[1].name, g_in[2].name]

            feeds[feature_name] = np.ascontiguousarray(feature_volume, dtype=np.float32)

            # 'source' input <- driving (target expr), 'target' input <- source
            # (verified on-GPU: this generator's names are opposite to their effect).
            src_in = next((n for n in kp_inputs if 'sourc' in n.lower()), kp_inputs[0])
            tgt_in = next((n for n in kp_inputs if ('targ' in n.lower() or 'driv' in n.lower())), kp_inputs[-1])
            if src_in == tgt_in and len(kp_inputs) >= 2:
                src_in, tgt_in = kp_inputs[0], kp_inputs[1]
            if getattr(roop.globals, 'expression_invert_direction', True):
                feeds[src_in] = np.ascontiguousarray(kp_driving, dtype=np.float32)
                feeds[tgt_in] = np.ascontiguousarray(kp_source, dtype=np.float32)
            else:
                feeds[src_in] = np.ascontiguousarray(kp_source, dtype=np.float32)
                feeds[tgt_in] = np.ascontiguousarray(kp_driving, dtype=np.float32)
            gen_out = self._run_session(self.generator, feeds)[0]

            restored = lpu.parse_output(gen_out)
            if getattr(roop.globals, 'expression_debug', False) or getattr(roop.globals, 'profile_timings', False):
                try:
                    ed = float(np.max(np.abs(target_exp.reshape(-1) - temp_exp.reshape(-1))))
                    sig_before = float(np.abs(kp_driving_raw - kp_source).sum())
                    sig_after = float(np.abs(kp_driving - kp_source).sum())
                    kept = 100.0 * sig_after / (sig_before + 1e-9)
                    # Kabsch rotation angle (deg) that pose-lock would remove between
                    # raw driving and source -- large value => rotation_lock is
                    # absorbing local expression into a fake global rotation.
                    ang = 0.0
                    try:
                        kd0 = (kp_driving_raw - kp_driving_raw.mean(axis=1, keepdims=True))[0]
                        ks0 = (kp_source - kp_source.mean(axis=1, keepdims=True))[0]
                        Hm = kd0.T @ ks0
                        U, S, Vt = np.linalg.svd(Hm)
                        dsgn = np.sign(np.linalg.det(Vt.T @ U.T))
                        Rm = Vt.T @ np.diag([1.0, 1.0, dsgn]) @ U.T
                        ang = float(np.degrees(np.arccos(np.clip((np.trace(Rm) - 1.0) / 2.0, -1.0, 1.0))))
                    except Exception:
                        pass
                    ch, cw = swapped_crop.shape[:2]
                    pl = getattr(roop.globals, 'expression_pose_lock', True)
                    pls = getattr(roop.globals, 'expression_pose_lock_scale', True)
                    plr = getattr(roop.globals, 'expression_pose_lock_rotation', False)
                    pw = getattr(roop.globals, 'expression_power', 1.0)
                    print(f"[expr-delta] exp|max={ed:.4f}  sig_before={sig_before:.4f}  "
                          f"sig_after={sig_after:.4f}  kept={kept:.0f}%  kabsch_rot={ang:.1f}deg  "
                          f"crop={cw}x{ch}  lock={pl}/scale={pls}/rot={plr}  power={pw:.2f}  "
                          f"factor={factor:.2f}  full={full}  stitch={self.stitcher is not None}")
                except Exception:
                    pass

            # Paste the 512 result back into the arcface crop space (full pipeline)
            # or just resize (fallback), then feather so borders stay clean.
            if full and back_M is not None:
                restored = cv2.warpAffine(
                    restored, back_M,
                    (swapped_crop.shape[1], swapped_crop.shape[0]),
                    flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
            elif (restored.shape[1], restored.shape[0]) != (swapped_crop.shape[1], swapped_crop.shape[0]):
                restored = cv2.resize(
                    restored, (swapped_crop.shape[1], swapped_crop.shape[0]),
                    interpolation=cv2.INTER_AREA)
            border = float(getattr(roop.globals, 'expression_blend_border', 0.2))
            if border > 0:
                restored = lpu.feather_blend(restored, swapped_crop, border=border)
            return restored
        except Exception as e:
            if not getattr(self, '_err_logged', False):
                import traceback
                print("[Expression_LivePortrait] restore FAILED (showing once); "
                      "passing the swapped face through unchanged:")
                traceback.print_exc()
                self._err_logged = True
            return swapped_crop

    def Release(self):
        self.feature_extractor = None
        self.motion_extractor = None
        self.generator = None
