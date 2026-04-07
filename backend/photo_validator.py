"""
Photo Validation Engine
=======================
Validates a submitted photo for onboarding. Designed to be called from the
/api/validate-photo endpoint for both:
  - File uploads (static photo from gallery)
  - Camera-captured frames (real-time validation during capture)

Checks performed (in order):
  1. Decode         — is this a valid image?
  2. Face detection — is there at least one face? (InsightFace RetinaFace)
  3. Face count     — exactly one face?
  4. Face size      — face covers ≥ 15% of image area (not too far away)?
  5. Face centering — face center within middle 60% of frame?
  6. Occlusion      — glasses / sunglasses / mask detected?
  7. AI detection   — is the image likely AI-generated / synthetic?

Returns a PhotoValidationResult with:
  - valid: bool                   — all required checks passed
  - rejection_reason: str | None  — human-readable reason for first failure
  - individual check flags
  - ai_probability: float 0–1    — higher = more likely AI-generated
  - ai_signals: dict             — the individual sub-signals for debugging

AI detection uses THREE lightweight signals (no heavy ML model needed):
  A. Spectral slope  — real camera images obey 1/f pink-noise power spectrum
                       (slope ≈ -2.5). AI images are either too flat (diffusion)
                       or show periodic GAN artifacts.
  B. Noise residual  — real cameras add grain with specific statistics.
                       AI images are synthetically clean (std < 1.0).
  C. Face symmetry   — AI faces are unnaturally symmetric. We compute the
                       normalised pixel-difference between mirrored halves of
                       the face crop. Real faces score higher; AI faces score
                       near-zero.

Tuned thresholds are conservative: we prefer false-negatives (letting some
AI images through) over false-positives (rejecting real users). Set
AI_THRESHOLD to a lower value if you want stricter rejection.
"""

import cv2
import numpy as np
from dataclasses import dataclass, field
from typing import Optional
from scipy.ndimage import median_filter

import insightface
from insightface.app import FaceAnalysis
from pathlib import Path

# ── Model setup ───────────────────────────────────────────────────────────────

MODEL_DIR = Path(__file__).parent / "insightface_models"
MODEL_DIR.mkdir(exist_ok=True)

# Minimum face area as fraction of total image area
MIN_FACE_AREA_FRACTION = 0.12   # 12% — face must be reasonably close

# How centred the face must be — allowed deviation from image centre (0–0.5)
MAX_CENTRE_OFFSET = 0.25        # face centre must be within middle 50% of frame

# AI probability above this triggers a rejection
AI_THRESHOLD = 0.72

# InsightFace detection confidence threshold
MIN_DETECTION_SCORE = 0.65


# ── Result model ──────────────────────────────────────────────────────────────

@dataclass
class PhotoValidationResult:
    # ── Top-level verdict ─────────────────────────────────────────────────────
    valid: bool
    rejection_reason: Optional[str]

    # ── Individual checks ─────────────────────────────────────────────────────
    image_valid: bool        = True
    face_detected: bool      = False
    face_count: int          = 0
    single_face: bool        = False
    face_large_enough: bool  = False
    face_centered: bool      = False
    no_occlusion: bool       = False   # True = no glasses/mask detected
    not_ai_generated: bool   = True

    # ── Face geometry (for overlay drawing on mobile) ─────────────────────────
    bbox: Optional[list[int]]   = None   # [x1, y1, x2, y2] absolute pixels
    bbox_norm: Optional[list[float]] = None  # [x1, y1, x2, y2] normalised 0–1
    face_cx_norm: float         = 0.5
    face_cy_norm: float         = 0.5

    # ── AI detection detail ───────────────────────────────────────────────────
    ai_probability: float       = 0.0
    ai_signals: dict            = field(default_factory=dict)

    # ── Face attributes ───────────────────────────────────────────────────────
    age: Optional[int]          = None
    gender: Optional[str]       = None
    detection_score: float      = 0.0


# ── Engine ────────────────────────────────────────────────────────────────────

class PhotoValidator:
    """
    Singleton-friendly validator.
    Instantiate once at app startup (InsightFace model loading is slow).
    """

    def __init__(self):
        self._app = FaceAnalysis(
            name="buffalo_l",
            root=str(MODEL_DIR),
            providers=["CPUExecutionProvider"],
        )
        self._app.prepare(ctx_id=0, det_size=(640, 640))

    # ── Public API ────────────────────────────────────────────────────────────

    def validate(self, image_bytes: bytes) -> PhotoValidationResult:
        """
        Validate image_bytes (JPEG / PNG / WebP).
        Returns a PhotoValidationResult.
        """
        # 1. Decode
        frame = self._decode(image_bytes)
        if frame is None:
            return PhotoValidationResult(
                valid=False,
                rejection_reason="Could not decode image — please upload a valid JPEG or PNG.",
                image_valid=False,
            )

        h, w = frame.shape[:2]

        # 2. Detect faces
        faces = self._app.get(frame)
        face_count = len(faces)

        if face_count == 0:
            return PhotoValidationResult(
                valid=False,
                rejection_reason="No face detected. Please submit a clear photo of your face.",
                image_valid=True,
                face_detected=False,
                face_count=0,
            )

        # Pick the dominant face — largest bounding-box area with highest score.
        # Background people are small/distant and will fail the size check later.
        # We only flag "multiple faces" if 2+ faces are both large AND prominent
        # (i.e. another person is clearly in the foreground with the subject).
        def face_area(f):
            x1, y1, x2, y2 = f.bbox
            return max(0.0, (x2 - x1) * (y2 - y1))

        faces_sorted = sorted(faces, key=lambda f: face_area(f) * float(f.det_score), reverse=True)
        face = faces_sorted[0]

        # Check for a genuinely competing foreground face (area > 40% of dominant)
        if face_count > 1:
            dominant_area = face_area(face)
            competing = [
                f for f in faces_sorted[1:]
                if face_area(f) > dominant_area * 0.40 and float(f.det_score) > 0.60
            ]
            if competing:
                return PhotoValidationResult(
                    valid=False,
                    rejection_reason="Multiple people detected in the photo. Only one person should be visible.",
                    image_valid=True,
                    face_detected=True,
                    face_count=face_count,
                    single_face=False,
                )

        det_score = float(face.det_score)

        if det_score < MIN_DETECTION_SCORE:
            return PhotoValidationResult(
                valid=False,
                rejection_reason="Face detected but too blurry or obscured. Please use a clearer photo.",
                image_valid=True,
                face_detected=True,
                face_count=1,
                detection_score=det_score,
            )

        # 3. Bounding box + geometry
        x1, y1, x2, y2 = [max(0, int(v)) for v in face.bbox]
        x2, y2 = min(w, x2), min(h, y2)
        bbox = [x1, y1, x2, y2]
        bbox_norm = [x1/w, y1/h, x2/w, y2/h]

        face_w = x2 - x1
        face_h = y2 - y1
        face_area_frac = (face_w * face_h) / (w * h)
        face_cx_norm = (x1 + x2) / 2.0 / w
        face_cy_norm = (y1 + y2) / 2.0 / h

        # 4. Size check
        face_large_enough = face_area_frac >= MIN_FACE_AREA_FRACTION
        if not face_large_enough:
            return PhotoValidationResult(
                valid=False,
                rejection_reason="Face is too small or too far from the camera. Move closer.",
                image_valid=True, face_detected=True, face_count=1, single_face=True,
                face_large_enough=False,
                bbox=bbox, bbox_norm=bbox_norm,
                face_cx_norm=face_cx_norm, face_cy_norm=face_cy_norm,
                detection_score=det_score,
            )

        # 5. Centering check
        cx_offset = abs(face_cx_norm - 0.5)
        cy_offset = abs(face_cy_norm - 0.5)
        face_centered = (cx_offset <= MAX_CENTRE_OFFSET and cy_offset <= MAX_CENTRE_OFFSET)
        if not face_centered:
            direction = ""
            if cx_offset > MAX_CENTRE_OFFSET:
                direction = "left" if face_cx_norm < 0.5 else "right"
            elif cy_offset > MAX_CENTRE_OFFSET:
                direction = "up" if face_cy_norm < 0.5 else "down"
            return PhotoValidationResult(
                valid=False,
                rejection_reason=f"Face is off-centre (too far {direction}). Centre your face in the frame.",
                image_valid=True, face_detected=True, face_count=1, single_face=True,
                face_large_enough=True, face_centered=False,
                bbox=bbox, bbox_norm=bbox_norm,
                face_cx_norm=face_cx_norm, face_cy_norm=face_cy_norm,
                detection_score=det_score,
            )

        # 6. Occlusion check (glasses / mask)
        face_crop = frame[y1:y2, x1:x2]
        occlusion_type = self._detect_occlusion(frame, face, w, h)
        no_occlusion = occlusion_type is None
        if not no_occlusion:
            return PhotoValidationResult(
                valid=False,
                rejection_reason=f"{occlusion_type} detected. Please remove it and retake the photo.",
                image_valid=True, face_detected=True, face_count=1, single_face=True,
                face_large_enough=True, face_centered=True, no_occlusion=False,
                bbox=bbox, bbox_norm=bbox_norm,
                face_cx_norm=face_cx_norm, face_cy_norm=face_cy_norm,
                age=int(face.age) if face.age is not None else None,
                gender="male" if face.gender == 1 else "female",
                detection_score=det_score,
            )

        # 7. AI generation detection
        ai_prob, ai_signals = self._detect_ai(frame, face_crop)
        not_ai = ai_prob < AI_THRESHOLD

        if not not_ai:
            return PhotoValidationResult(
                valid=False,
                rejection_reason="This image appears to be AI-generated or digitally manipulated. Please submit a real photo.",
                image_valid=True, face_detected=True, face_count=1, single_face=True,
                face_large_enough=True, face_centered=True, no_occlusion=True,
                not_ai_generated=False,
                bbox=bbox, bbox_norm=bbox_norm,
                face_cx_norm=face_cx_norm, face_cy_norm=face_cy_norm,
                age=int(face.age) if face.age is not None else None,
                gender="male" if face.gender == 1 else "female",
                detection_score=det_score,
                ai_probability=round(ai_prob, 3),
                ai_signals=ai_signals,
            )

        # ✓ All checks passed
        return PhotoValidationResult(
            valid=True,
            rejection_reason=None,
            image_valid=True, face_detected=True, face_count=1, single_face=True,
            face_large_enough=True, face_centered=True, no_occlusion=True,
            not_ai_generated=True,
            bbox=bbox, bbox_norm=bbox_norm,
            face_cx_norm=face_cx_norm, face_cy_norm=face_cy_norm,
            age=int(face.age) if face.age is not None else None,
            gender="male" if face.gender == 1 else "female",
            detection_score=det_score,
            ai_probability=round(ai_prob, 3),
            ai_signals=ai_signals,
        )

    # ── Occlusion detection ───────────────────────────────────────────────────

    def _detect_occlusion(self, frame: np.ndarray, face, w: int, h: int) -> Optional[str]:
        """
        Returns "Sunglasses", "Glasses", "Mask", or None.

        Strategy:
          - Glasses/sunglasses: detect high-density straight edges in the eye
            strip (glasses frames create distinctive parallel line patterns).
            Also check average brightness in the eye region — dark lenses
            make the eye area significantly darker than surrounding skin.
          - Mask: check the lower face strip (nose-to-chin). If it lacks the
            normal skin-tone variation and has a hard edge pattern, it's masked.
        """
        kps = face.kps  # 5 keypoints: [left_eye, right_eye, nose, mouth_l, mouth_r]
        if kps is None or len(kps) < 5:
            return None

        left_eye  = kps[0]
        right_eye = kps[1]
        nose      = kps[2]
        mouth_l   = kps[3]
        mouth_r   = kps[4]

        eye_cx = int((left_eye[0] + right_eye[0]) / 2)
        eye_cy = int((left_eye[1] + right_eye[1]) / 2)
        interocular = float(np.linalg.norm(right_eye - left_eye))

        # ── Eye strip: glasses detection ──────────────────────────────────────
        eye_pad_x = int(interocular * 0.65)
        eye_pad_y = int(interocular * 0.35)
        ex1 = max(0, eye_cx - eye_pad_x)
        ex2 = min(w, eye_cx + eye_pad_x)
        ey1 = max(0, eye_cy - eye_pad_y)
        ey2 = min(h, eye_cy + eye_pad_y)

        if ex2 > ex1 and ey2 > ey1:
            eye_strip = frame[ey1:ey2, ex1:ex2]
            gray_eye  = cv2.cvtColor(eye_strip, cv2.COLOR_BGR2GRAY)

            # Edge density in eye strip: glasses frames = many straight edges
            edges = cv2.Canny(gray_eye, threshold1=40, threshold2=120)
            edge_density = float(np.mean(edges > 0))

            # Brightness in eye strip vs overall face brightness
            x1f = int(face.bbox[0]); y1f = int(face.bbox[1])
            x2f = int(face.bbox[2]); y2f = int(face.bbox[3])
            face_crop = frame[max(0,y1f):min(h,y2f), max(0,x1f):min(w,x2f)]
            face_brightness = float(np.mean(cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY))) if face_crop.size > 0 else 128.0
            eye_brightness  = float(np.mean(gray_eye))

            darkness_ratio = eye_brightness / (face_brightness + 1e-6)

            # Sunglasses: dark lenses AND high edge density
            if darkness_ratio < 0.55 and edge_density > 0.08:
                return "Sunglasses"

            # Regular glasses: high edge density but reasonable brightness
            if edge_density > 0.14:
                return "Glasses"

        # ── Lower face strip: mask detection ──────────────────────────────────
        nose_y  = int(nose[1])
        chin_y  = int(face.bbox[3])
        mouth_cx = int((mouth_l[0] + mouth_r[0]) / 2)
        mask_pad_x = int(interocular * 0.6)
        mx1 = max(0, mouth_cx - mask_pad_x)
        mx2 = min(w, mouth_cx + mask_pad_x)
        my1 = max(0, nose_y)
        my2 = min(h, chin_y)

        if mx2 > mx1 and my2 > my1:
            lower_strip = frame[my1:my2, mx1:mx2]
            if lower_strip.size > 0:
                # Convert to HSV to check skin-tone hue
                hsv = cv2.cvtColor(lower_strip, cv2.COLOR_BGR2HSV)
                hue = hsv[:, :, 0]
                # Skin hue in OpenCV HSV: roughly 0–25 (and 160–179 for very dark skin)
                skin_mask = ((hue < 25) | (hue > 160))
                skin_fraction = float(np.mean(skin_mask))

                # Strong edges at the nose line = mask boundary
                lower_edges = cv2.Canny(
                    cv2.cvtColor(lower_strip, cv2.COLOR_BGR2GRAY), 30, 90
                )
                lower_edge_density = float(np.mean(lower_edges > 0))

                # Mask: lower face is non-skin colour with a clear boundary edge
                if skin_fraction < 0.35 and lower_edge_density > 0.06:
                    return "Mask"

        return None

    # ── AI generation detection ───────────────────────────────────────────────

    def _detect_ai(self, frame: np.ndarray, face_crop: np.ndarray) -> tuple[float, dict]:
        """
        Returns (ai_probability, signals_dict).

        Three complementary signals:

        A. Spectral slope — fit power-law to radial power spectrum in log-log
           space. Natural images: slope ≈ -2 to -3. AI images are flatter
           (diffusion) or show periodic bumps (GAN grid artifacts).

        B. Noise floor — subtract median-filtered image to isolate sensor noise.
           Real cameras: noise std typically 2–8 (depending on ISO).
           AI images: often < 1.0 (too clean) or irregular.

        C. Face symmetry — AI faces are unnaturally symmetric. Mirror the face
           horizontally and measure mean absolute pixel difference between
           original and mirrored halves. Real faces score > 8; AI often < 4.
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32)

        # ── Signal A: Spectral slope ───────────────────────────────────────────
        fft       = np.fft.fft2(gray)
        fft_shift = np.fft.fftshift(fft)
        magnitude = np.abs(fft_shift)

        hh, ww = gray.shape
        cy, cx  = hh // 2, ww // 2
        Y, X    = np.ogrid[:hh, :ww]
        R       = np.sqrt((X - cx)**2 + (Y - cy)**2).astype(int)
        R_max   = min(cy, cx)

        # Radial average of power spectrum
        radial_sum = np.bincount(R.ravel(), weights=magnitude.ravel(), minlength=R_max+1)
        radial_cnt = np.bincount(R.ravel(), minlength=R_max+1).clip(min=1)
        radial_avg = (radial_sum / radial_cnt)[1:R_max]  # skip DC

        if len(radial_avg) > 10:
            freqs  = np.arange(1, len(radial_avg) + 1, dtype=np.float32)
            log_f  = np.log(freqs)
            log_p  = np.log(radial_avg + 1e-6)
            slope, _ = np.polyfit(log_f, log_p, 1)
            # Natural images: slope ~ -2.0 to -3.0
            # Penalise if slope is outside [-1.2, -4.0]
            spectral_score = float(np.clip(abs(slope + 2.5) / 1.5, 0.0, 1.0))
        else:
            slope = -2.5
            spectral_score = 0.0

        # ── Signal B: Noise floor ──────────────────────────────────────────────
        denoised    = median_filter(gray, size=3)
        noise       = gray - denoised
        noise_std   = float(np.std(noise))

        # Score: penalise if suspiciously clean (< 0.8) or no noise pattern
        if noise_std < 0.8:
            noise_score = 0.85   # AI image: almost no noise
        elif noise_std < 1.5:
            noise_score = 0.55
        elif noise_std > 20.0:
            noise_score = 0.45   # extreme noise (heavy JPEG artifact, not AI)
        else:
            noise_score = 0.05   # normal camera noise range

        # ── Signal C: Face symmetry ────────────────────────────────────────────
        if face_crop is not None and face_crop.size > 0:
            fc_gray   = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY).astype(np.float32)
            fc_resized = cv2.resize(fc_gray, (64, 64))
            left_half  = fc_resized[:, :32]
            right_half = np.fliplr(fc_resized[:, 32:])

            sym_diff   = float(np.mean(np.abs(left_half - right_half)))
            # Real faces: diff typically 8–20. AI faces: often < 5.
            if sym_diff < 4.0:
                symmetry_score = 0.80   # near-perfect symmetry → AI
            elif sym_diff < 7.0:
                symmetry_score = 0.45
            elif sym_diff < 12.0:
                symmetry_score = 0.15
            else:
                symmetry_score = 0.05   # natural asymmetry
        else:
            sym_diff = 10.0
            symmetry_score = 0.15

        # ── Combine ───────────────────────────────────────────────────────────
        # Weighted: noise is the strongest single signal; spectral is secondary;
        # symmetry helps push ambiguous cases over the threshold.
        ai_prob = (
            0.40 * noise_score
          + 0.35 * spectral_score
          + 0.25 * symmetry_score
        )

        signals = {
            "spectral_slope":   round(float(slope), 3),
            "spectral_score":   round(spectral_score, 3),
            "noise_std":        round(noise_std, 3),
            "noise_score":      round(noise_score, 3),
            "face_sym_diff":    round(sym_diff, 3),
            "symmetry_score":   round(symmetry_score, 3),
            "ai_probability":   round(ai_prob, 3),
        }

        return float(ai_prob), signals

    # ── Helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _decode(image_bytes: bytes) -> Optional[np.ndarray]:
        nparr = np.frombuffer(image_bytes, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        return frame if frame is not None and frame.size > 0 else None
