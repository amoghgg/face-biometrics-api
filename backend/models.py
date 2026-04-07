from pydantic import BaseModel
from typing import Optional, Any
from enum import Enum


# ── Liveness ──────────────────────────────────────────────────────────────────

class ChallengeType(str, Enum):
    TURN_LEFT  = "TURN_LEFT"
    TURN_RIGHT = "TURN_RIGHT"
    SMILE      = "SMILE"
    COMPLETE   = "COMPLETE"
    FAILED     = "FAILED"


class SessionState(str, Enum):
    CREATED    = "CREATED"
    IN_PROGRESS = "IN_PROGRESS"
    COMPLETE   = "COMPLETE"
    FAILED     = "FAILED"
    EXPIRED    = "EXPIRED"


CHALLENGE_SEQUENCE = [ChallengeType.TURN_LEFT, ChallengeType.TURN_RIGHT, ChallengeType.SMILE]

CHALLENGE_INSTRUCTIONS = {
    ChallengeType.TURN_LEFT:  "Slowly turn your head to the LEFT",
    ChallengeType.TURN_RIGHT: "Now turn your head to the RIGHT",
    ChallengeType.SMILE:      "Great! Now give us a big SMILE",
    ChallengeType.COMPLETE:   "Verification complete!",
    ChallengeType.FAILED:     "Verification failed. Please try again.",
}


class FaceMetrics(BaseModel):
    face_detected: bool
    yaw_proxy: float = 0.0
    smile_score: float = 0.0
    texture_variance: float = 0.0
    landmark_z_std: float = 0.0
    is_spoof: bool = False
    # rPPG fields — populated once enough frames are buffered (~5 seconds)
    rppg_bpm: Optional[float] = None         # estimated heart rate
    rppg_confidence: Optional[float] = None  # 0–1, how dominant the peak is
    rppg_ready: bool = False                 # False until MIN_SAMPLES of data collected
    rppg_is_live: Optional[bool] = None      # True = heartbeat detected
    rppg_sampling: bool = False              # True = this frame was accepted into buffer
    rppg_samples: int = 0                   # how many good samples collected so far
    rppg_verdict: str = "pending"           # "pending" | "real" | "synthetic"
    # Forehead sampling region — for debug visualization
    forehead_rgb: Optional[list[float]] = None       # [r, g, b] 0-255
    forehead_bbox_norm: Optional[list[float]] = None  # [x1, y1, x2, y2] 0-1


class FrameResponse(BaseModel):
    session_id: str
    challenge: ChallengeType
    challenge_index: int
    challenge_passed: bool
    feedback: str
    metrics: FaceMetrics
    liveness_token: Optional[str] = None


class CreateSessionResponse(BaseModel):
    session_id: str
    expires_at: str
    challenges: list[str]


class SessionStatusResponse(BaseModel):
    session_id: str
    state: SessionState
    challenges_completed: int
    liveness_token: Optional[str] = None
    smile_photo_path: Optional[str] = None


class VerifyTokenRequest(BaseModel):
    liveness_token: str


class VerifyTokenResponse(BaseModel):
    valid: bool
    session_id: Optional[str] = None
    issued_at: Optional[str] = None


# ── Face Recognition ──────────────────────────────────────────────────────────

class RegisterFaceRequest(BaseModel):
    name: str
    image: str              # base64 JPEG
    metadata: dict[str, Any] = {}


class RegisterFaceResponse(BaseModel):
    face_id: str
    name: str
    age: Optional[int] = None
    gender: Optional[str] = None
    detection_score: float = 0.0


class VerifyFaceRequest(BaseModel):
    face_id: str
    image: str              # base64 JPEG


class VerifyFaceResponse(BaseModel):
    face_id: str
    verified: bool
    similarity: float
    threshold: float


class SearchFaceRequest(BaseModel):
    image: str              # base64 JPEG
    top_k: int = 5


class SearchMatch(BaseModel):
    face_id: str
    name: str
    similarity: float
    matched: bool
    metadata: dict[str, Any] = {}


class SearchFaceResponse(BaseModel):
    matches: list[SearchMatch]
    face_detected: bool


class FaceRecordResponse(BaseModel):
    face_id: str
    name: str
    metadata: dict[str, Any] = {}


# ── Face Analysis ─────────────────────────────────────────────────────────────

class AnalyzeRequest(BaseModel):
    image: str              # base64 JPEG


class EmotionScores(BaseModel):
    happy: float = 0.0
    sad: float = 0.0
    angry: float = 0.0
    surprised: float = 0.0
    disgusted: float = 0.0
    neutral: float = 1.0


class AnalyzeResponse(BaseModel):
    face_detected: bool
    age: Optional[int] = None
    gender: Optional[str] = None
    gender_confidence: Optional[float] = None
    emotion: Optional[EmotionScores] = None
    dominant_emotion: Optional[str] = None
    smile_score: Optional[float] = None
    detection_score: Optional[float] = None
    bbox: Optional[list[int]] = None          # [x1, y1, x2, y2]
