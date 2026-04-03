from models import ChallengeType, FaceMetrics
from liveness_engine import (
    TURN_LEFT_THRESHOLD, TURN_RIGHT_THRESHOLD,
    NEUTRAL_THRESHOLD, SMILE_SCORE_THRESHOLD,
)

# Frames that must pass consecutively to complete a challenge (~1.3s at 15fps)
CONSECUTIVE_FRAMES_REQUIRED = 20


def evaluate_challenge(
    challenge: ChallengeType,
    metrics: FaceMetrics,
    consecutive_count: int,
) -> tuple[bool, int]:
    """
    Returns (challenge_fully_passed, updated_consecutive_count).
    Resets consecutive count to 0 on any non-passing frame.
    """
    if not metrics.face_detected or metrics.is_spoof:
        return False, 0

    frame_passes = _frame_passes_challenge(challenge, metrics)

    new_count = consecutive_count + 1 if frame_passes else 0
    challenge_passed = new_count >= CONSECUTIVE_FRAMES_REQUIRED
    return challenge_passed, new_count


def is_neutral(metrics: FaceMetrics) -> bool:
    """True when the face is roughly forward-facing (used to gate next challenge)."""
    return metrics.face_detected and abs(metrics.yaw_proxy) < NEUTRAL_THRESHOLD


def _frame_passes_challenge(challenge: ChallengeType, metrics: FaceMetrics) -> bool:
    # Mirrored frame (selfie view) convention:
    #   Person turns THEIR LEFT  → nose moves to lower x → yaw_proxy NEGATIVE
    #   Person turns THEIR RIGHT → nose moves to higher x → yaw_proxy POSITIVE
    if challenge == ChallengeType.TURN_LEFT:
        return metrics.yaw_proxy < TURN_LEFT_THRESHOLD   # e.g. < -0.30

    if challenge == ChallengeType.TURN_RIGHT:
        return metrics.yaw_proxy > TURN_RIGHT_THRESHOLD  # e.g. > +0.30

    if challenge == ChallengeType.SMILE:
        return metrics.smile_score >= SMILE_SCORE_THRESHOLD  # 0–1 blendshape score

    return False
