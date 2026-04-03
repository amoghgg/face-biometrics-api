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

    def process_frame(self, jpeg_bytes: bytes) -> FaceMetrics:
        nparr = np.frombuffer(jpeg_bytes, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if frame is None:
            return FaceMetrics(face_detected=False)

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = self.detector.detect(mp_image)

        if not result.face_landmarks:
            return FaceMetrics(face_detected=False)

        # Reject multiple faces
        if len(result.face_landmarks) > 1:
            return FaceMetrics(face_detected=False)

        landmarks = result.face_landmarks[0]
        h, w = frame.shape[:2]

        yaw_proxy  = self._compute_yaw_proxy(landmarks)
        smile_score = self._compute_smile_score(result)
        texture_var, z_std, is_spoof = self._check_spoof(frame, landmarks, w, h)

        return FaceMetrics(
            face_detected=True,
            yaw_proxy=round(yaw_proxy, 4),
            smile_score=round(smile_score, 4),
            texture_variance=round(texture_var, 2),
            landmark_z_std=round(z_std, 6),
            is_spoof=is_spoof,
        )

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

    def close(self):
        self.detector.close()
