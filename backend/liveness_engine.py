import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
from pathlib import Path

from models import FaceMetrics

# MediaPipe landmark indices
NOSE_TIP = 1
# Landmark 33  = person's LEFT eye outer corner
# Landmark 263 = person's RIGHT eye outer corner
# In camera.js the frame is mirrored (selfie view), so:
#   - Person's LEFT eye (33)  appears on the LEFT  side of the image → lower x
#   - Person's RIGHT eye (263) appears on the RIGHT side of the image → higher x
# Therefore:
#   - Person turns THEIR LEFT  → nose moves LEFT  in image → yaw_proxy NEGATIVE
#   - Person turns THEIR RIGHT → nose moves RIGHT in image → yaw_proxy POSITIVE
LEFT_EYE_OUTER = 33
RIGHT_EYE_OUTER = 263

# Thresholds
# yaw_proxy < -0.30  → person has turned LEFT
# yaw_proxy >  0.30  → person has turned RIGHT
TURN_LEFT_THRESHOLD  = -0.30   # yaw must be BELOW this for "turn left"
TURN_RIGHT_THRESHOLD =  0.30   # yaw must be ABOVE this for "turn right"
NEUTRAL_THRESHOLD    =  0.15   # within ±0.15 counts as "facing forward"

# Smile blendshape threshold (0–1 score from MediaPipe)
SMILE_SCORE_THRESHOLD = 0.5

SPOOF_TEXTURE_MIN = 50.0
SPOOF_Z_STD_MIN   = 0.008

MODEL_PATH = Path(__file__).parent / "face_landmarker.task"


class LivenessEngine:
    def __init__(self):
        base_options = mp_python.BaseOptions(model_asset_path=str(MODEL_PATH))
        options = mp_vision.FaceLandmarkerOptions(
            base_options=base_options,
            output_face_blendshapes=True,   # needed for smile score
            num_faces=2,
            min_face_detection_confidence=0.5,
            min_face_presence_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self.detector = mp_vision.FaceLandmarker.create_from_options(options)

    def process_frame(self, jpeg_bytes: bytes) -> "FaceMetrics":
        """
        Process one JPEG frame. Returns a fully populated FaceMetrics including
        forehead_rgb and forehead_bbox_norm when a face is detected.
        """
        nparr = np.frombuffer(jpeg_bytes, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if frame is None:
            return FaceMetrics(face_detected=False)

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = self.detector.detect(mp_image)

        if not result.face_landmarks:
            return FaceMetrics(face_detected=False)

        if len(result.face_landmarks) > 1:
            return FaceMetrics(face_detected=False)

        landmarks = result.face_landmarks[0]
        h, w = frame.shape[:2]

        yaw_proxy   = self._compute_yaw_proxy(landmarks)
        smile_score = self._compute_smile_score(result)
        texture_var, z_std, is_spoof = self._check_spoof(frame, landmarks, w, h)
        forehead    = self.extract_forehead_rgb(frame, landmarks, w, h)

        metrics = FaceMetrics(
            face_detected=True,
            yaw_proxy=round(yaw_proxy, 4),
            smile_score=round(smile_score, 4),
            texture_variance=round(texture_var, 2),
            landmark_z_std=round(z_std, 6),
            is_spoof=is_spoof,
        )

        if forehead is not None:
            r, g, b, x1n, y1n, x2n, y2n = forehead
            metrics.forehead_rgb      = [round(r, 1), round(g, 1), round(b, 1)]
            metrics.forehead_bbox_norm = [round(x1n, 4), round(y1n, 4),
                                           round(x2n, 4), round(y2n, 4)]

        return metrics

    def _compute_yaw_proxy(self, landmarks) -> float:
        """
        Symmetric ratio of nose-to-eye distances.
        ~0   = facing forward
        < 0  = person turned LEFT  (nose moves toward lower x in mirrored frame)
        > 0  = person turned RIGHT (nose moves toward higher x in mirrored frame)
        """
        nose      = landmarks[NOSE_TIP]
        left_eye  = landmarks[LEFT_EYE_OUTER]
        right_eye = landmarks[RIGHT_EYE_OUTER]

        eps = 1e-6
        dist_to_left  = nose.x - left_eye.x    # positive when nose is right of left eye
        dist_to_right = right_eye.x - nose.x   # positive when nose is left of right eye

        yaw_proxy = (dist_to_left - dist_to_right) / (dist_to_left + dist_to_right + eps)
        return float(np.clip(yaw_proxy, -1.0, 1.0))

    def extract_blendshapes(self, result) -> dict[str, float]:
        """Returns all 52 blendshape scores as a dict."""
        if not result.face_blendshapes:
            return {}
        return {bs.category_name: bs.score for bs in result.face_blendshapes[0]}

    def _compute_smile_score(self, result) -> float:
        """
        Average of mouthSmileLeft and mouthSmileRight blendshape scores.
        Range 0–1; 0 = neutral, 1 = full smile.
        """
        bs = self.extract_blendshapes(result)
        left  = bs.get("mouthSmileLeft",  0.0)
        right = bs.get("mouthSmileRight", 0.0)
        return float((left + right) / 2)

    def _check_spoof(self, frame, landmarks, w: int, h: int):
        xs = [lm.x for lm in landmarks]
        ys = [lm.y for lm in landmarks]
        x1 = max(0, int(min(xs) * w) - 10)
        y1 = max(0, int(min(ys) * h) - 10)
        x2 = min(w, int(max(xs) * w) + 10)
        y2 = min(h, int(max(ys) * h) + 10)

        face_crop = frame[y1:y2, x1:x2]
        if face_crop.size == 0:
            return 0.0, 0.0, False

        gray = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY)
        texture_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())

        z_values = [lm.z for lm in landmarks]
        z_std = float(np.std(z_values))

        is_spoof = (texture_var < SPOOF_TEXTURE_MIN) or (z_std < SPOOF_Z_STD_MIN)
        return texture_var, z_std, is_spoof

    def extract_forehead_rgb(self, frame, landmarks, w: int, h: int):
        """
        Sample mean R, G, B from the forehead region — a thin strip just above
        the eyebrows, sized relative to actual face dimensions.

        Returns (r, g, b, x1_norm, y1_norm, x2_norm, y2_norm) or None.

        Geometry rationale:
          - WIDTH: use temple-to-temple distance (landmarks 127 ↔ 356), then
            take the inner ~55%. This is the actual face width, not the much
            smaller inner-brow distance we used before.
          - HEIGHT: a strip starting just above the brow line and extending
            upward by ~25% of the inter-ocular distance. This stays on skin
            even when the user has bangs/heavy hair that covers most of the
            upper forehead. Reaching all the way up to the forehead apex (10)
            sampled hair on people with bangs and produced garbage signal.
          - CENTER X: midpoint between the two pupils (landmarks 33 ↔ 263)
            so the box stays centered even when the head yaws slightly.

        Landmarks used:
          127, 356 → left/right temples (face width)
          33,  263 → left/right outer eye corners (centering + scale)
          107, 336 → top of left/right eyebrow (vertical anchor)
        """
        lm_temple_l = landmarks[127]
        lm_temple_r = landmarks[356]
        lm_eye_l    = landmarks[33]
        lm_eye_r    = landmarks[263]
        lm_brow_l   = landmarks[107]
        lm_brow_r   = landmarks[336]

        # Pixel coords
        temple_xl = lm_temple_l.x * w
        temple_xr = lm_temple_r.x * w
        eye_xl    = lm_eye_l.x * w
        eye_xr    = lm_eye_r.x * w

        face_width   = abs(temple_xr - temple_xl)
        if face_width < 30:  # Face too small / too far away
            return None

        eye_dist     = abs(eye_xr - eye_xl)
        center_x_pix = (eye_xl + eye_xr) / 2.0

        # Width: 55% of face width, centered between the eyes
        half_w = face_width * 0.275

        # Vertical strip: from just above the brows, going up by ~30% of eye dist
        brow_y_pix = ((lm_brow_l.y + lm_brow_r.y) / 2.0) * h
        strip_h    = max(8.0, eye_dist * 0.30)

        x1 = int(max(0, center_x_pix - half_w))
        x2 = int(min(w, center_x_pix + half_w))
        y2 = int(max(0, brow_y_pix - 4))            # 4px gap above the brow
        y1 = int(max(0, y2 - strip_h))

        if x2 - x1 < 8 or y2 - y1 < 4:
            return None

        roi = frame[y1:y2, x1:x2]
        if roi.size == 0:
            return None

        b_mean = float(roi[:, :, 0].mean())
        g_mean = float(roi[:, :, 1].mean())
        r_mean = float(roi[:, :, 2].mean())

        return r_mean, g_mean, b_mean, x1 / w, y1 / h, x2 / w, y2 / h

    def close(self):
        self.detector.close()
