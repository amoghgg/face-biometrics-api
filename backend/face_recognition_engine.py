"""
Face recognition engine using InsightFace (ArcFace model).
Provides: face embedding, age estimation, gender estimation.
ArcFace achieves 99.83% on LFW benchmark — on par with commercial SDKs.
"""
import cv2
import numpy as np
from pathlib import Path
from dataclasses import dataclass

import insightface
from insightface.app import FaceAnalysis

SIMILARITY_THRESHOLD = 0.45   # cosine similarity above this = same person
MODEL_DIR = Path(__file__).parent / "insightface_models"
MODEL_DIR.mkdir(exist_ok=True)


@dataclass
class FaceAnalysisResult:
    face_detected: bool
    embedding: np.ndarray | None = None   # 512-dim ArcFace embedding
    age: int | None = None
    gender: str | None = None             # "male" / "female"
    bbox: list[int] | None = None         # [x1, y1, x2, y2]
    detection_score: float = 0.0


class FaceRecognitionEngine:
    def __init__(self):
        self.app = FaceAnalysis(
            name="buffalo_l",             # ArcFace + RetinaFace — most accurate bundle
            root=str(MODEL_DIR),
            providers=["CPUExecutionProvider"],
        )
        self.app.prepare(ctx_id=0, det_size=(640, 640))

    def analyze(self, jpeg_bytes: bytes) -> FaceAnalysisResult:
        nparr = np.frombuffer(jpeg_bytes, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if frame is None:
            return FaceAnalysisResult(face_detected=False)

        faces = self.app.get(frame)
        if not faces:
            return FaceAnalysisResult(face_detected=False)

        # Use the largest / most confident face
        face = max(faces, key=lambda f: f.det_score)

        bbox = face.bbox.astype(int).tolist()
        gender = "male" if face.gender == 1 else "female"

        return FaceAnalysisResult(
            face_detected=True,
            embedding=face.normed_embedding,   # already L2-normalized
            age=int(face.age),
            gender=gender,
            bbox=bbox,
            detection_score=float(face.det_score),
        )

    @staticmethod
    def cosine_similarity(emb1: np.ndarray, emb2: np.ndarray) -> float:
        """Both embeddings are pre-normalized so dot product = cosine similarity."""
        return float(np.dot(emb1, emb2))

    @staticmethod
    def is_same_person(emb1: np.ndarray, emb2: np.ndarray) -> tuple[bool, float]:
        sim = FaceRecognitionEngine.cosine_similarity(emb1, emb2)
        return sim >= SIMILARITY_THRESHOLD, round(sim, 4)
