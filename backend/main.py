import os
import json
import base64
import time
from datetime import datetime, timezone
from pathlib import Path

CAPTURES_DIR = Path(__file__).parent / "captures"
CAPTURES_DIR.mkdir(exist_ok=True)

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from jose import jwt

from models import (
    ChallengeType, SessionState, CHALLENGE_SEQUENCE, CHALLENGE_INSTRUCTIONS,
    FrameResponse, CreateSessionResponse, SessionStatusResponse,
    VerifyTokenRequest, VerifyTokenResponse, FaceMetrics,
    RegisterFaceRequest, RegisterFaceResponse,
    VerifyFaceRequest, VerifyFaceResponse,
    SearchFaceRequest, SearchFaceResponse, SearchMatch, FaceRecordResponse,
    AnalyzeRequest, AnalyzeResponse, EmotionScores,
)
from challenge_evaluator import is_neutral, evaluate_challenge
from liveness_engine import LivenessEngine
from session_manager import session_manager
from face_recognition_engine import FaceRecognitionEngine, SIMILARITY_THRESHOLD
from face_db import face_db
from emotion_engine import blendshapes_to_emotions, dominant_emotion
from rppg_engine import RPPGEngine
from photo_validator import PhotoValidator

# ── JWT config ────────────────────────────────────────────────────────────────
JWT_SECRET = os.getenv("JWT_SECRET", "change-me-in-production")
JWT_ALGORITHM = "HS256"
JWT_EXPIRY_SECONDS = 300  # token valid for 5 minutes after issuance

# ── App setup ─────────────────────────────────────────────────────────────────
app = FastAPI(title="Face Biometrics API", version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

engine         = LivenessEngine()
rec_engine     = FaceRecognitionEngine()
photo_validator = PhotoValidator()

# Per-session rPPG engines — keyed by session_id, cleaned up on session end
rppg_engines: dict[str, RPPGEngine] = {}

# Per-session last ROI center position (normalized 0-1) — used to detect
# inter-frame head movement and discard motion-corrupted samples.
rppg_last_pos: dict[str, tuple[float, float]] = {}


# ── REST endpoints ────────────────────────────────────────────────────────────

# ── Photo validation endpoint ─────────────────────────────────────────────────
# Accepts two modes:
#   1. multipart/form-data with field "file"  — for file uploads from gallery
#   2. JSON body { "image": "<base64>" }      — for camera frame captures
#
# Returns a full PhotoValidationResult as JSON.  The mobile app (or web onboarding
# form) calls this before accepting a photo submission.

from fastapi import UploadFile, File, Form
from typing import Optional as Opt

@app.post("/api/validate-photo")
async def validate_photo(
    file: Opt[UploadFile] = File(default=None),
    image: Opt[str]       = Form(default=None),
):
    """
    Validate a profile photo for onboarding.

    Accepts either:
      - multipart upload:  field name = "file"
      - JSON/form field:   "image" = base64-encoded image (with or without
                            the data:image/jpeg;base64, prefix)

    Returns JSON with:
      valid, rejection_reason, face_detected, face_count, single_face,
      face_large_enough, face_centered, no_occlusion, not_ai_generated,
      bbox, bbox_norm, age, gender, detection_score,
      ai_probability, ai_signals
    """
    image_bytes: bytes | None = None

    if file is not None:
        image_bytes = await file.read()
    elif image is not None:
        # Strip data-URI prefix if present
        b64 = image.split(",", 1)[-1] if "," in image else image
        try:
            image_bytes = base64.b64decode(b64)
        except Exception:
            raise HTTPException(status_code=400, detail="Invalid base64 image data.")
    else:
        raise HTTPException(
            status_code=422,
            detail="Provide either a 'file' upload or a base64 'image' field.",
        )

    if not image_bytes:
        raise HTTPException(status_code=400, detail="Empty image data.")

    result = photo_validator.validate(image_bytes)

    # Convert dataclass → dict for JSON serialisation
    import dataclasses
    return dataclasses.asdict(result)


@app.post("/session/create", response_model=CreateSessionResponse)
def create_session():
    session = session_manager.create_session()
    return CreateSessionResponse(
        session_id=session.session_id,
        expires_at=session.expires_at_iso,
        challenges=[c.value for c in CHALLENGE_SEQUENCE],
    )


@app.get("/session/{session_id}/status", response_model=SessionStatusResponse)
def get_session_status(session_id: str):
    session = session_manager.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    return SessionStatusResponse(
        session_id=session.session_id,
        state=session.state,
        challenges_completed=session.challenges_completed,
        liveness_token=session.liveness_token,
        smile_photo_path=session.smile_photo_path,
    )


@app.post("/session/{session_id}/verify", response_model=VerifyTokenResponse)
def verify_token(session_id: str, body: VerifyTokenRequest):
    try:
        payload = jwt.decode(body.liveness_token, JWT_SECRET, algorithms=[JWT_ALGORITHM])
        return VerifyTokenResponse(
            valid=True,
            session_id=payload.get("sub"),
            issued_at=payload.get("iat"),
        )
    except Exception:
        return VerifyTokenResponse(valid=False)


@app.delete("/session/{session_id}")
def delete_session(session_id: str):
    deleted = session_manager.delete_session(session_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Session not found")
    return {"deleted": True}


# ── WebSocket endpoint ─────────────────────────────────────────────────────────

@app.websocket("/ws/liveness/{session_id}")
async def liveness_websocket(websocket: WebSocket, session_id: str):
    await websocket.accept()

    session = session_manager.get_session(session_id)
    if not session:
        await websocket.send_text(json.dumps({"error": "Session not found"}))
        await websocket.close(code=4004)
        return

    try:
        while True:
            # Receive frame (binary JPEG or base64-encoded text)
            message = await websocket.receive()

            if message["type"] == "websocket.disconnect":
                break

            if message.get("bytes"):
                frame_bytes = message["bytes"]
            elif message.get("text"):
                frame_bytes = base64.b64decode(message["text"])
            else:
                continue

            # Re-fetch session each iteration to catch expiry
            session = session_manager.get_session(session_id)
            if not session:
                break

            if session.state == SessionState.EXPIRED:
                await _send_response(websocket, session_id, ChallengeType.FAILED,
                                     0, False, "Session expired. Please start over.",
                                     FaceMetrics(face_detected=False))
                break

            if session.state == SessionState.COMPLETE:
                await _send_response(websocket, session_id, ChallengeType.COMPLETE,
                                     len(CHALLENGE_SEQUENCE), True,
                                     CHALLENGE_INSTRUCTIONS[ChallengeType.COMPLETE],
                                     FaceMetrics(face_detected=True),
                                     liveness_token=session.liveness_token)
                continue

            if session.state == SessionState.FAILED:
                await _send_response(websocket, session_id, ChallengeType.FAILED,
                                     session.challenges_completed, False,
                                     CHALLENGE_INSTRUCTIONS[ChallengeType.FAILED],
                                     FaceMetrics(face_detected=False))
                break

            # Replay attack check
            if session.is_replay_frame(frame_bytes):
                session.state = SessionState.FAILED
                await _send_response(websocket, session_id, ChallengeType.FAILED,
                                     session.challenges_completed, False,
                                     "Replay attack detected. Session terminated.",
                                     FaceMetrics(face_detected=False))
                break

            # Process frame with MediaPipe
            metrics = engine.process_frame(frame_bytes)

            # Feed rPPG engine.
            # We only accept a sample when the face is:
            #   1. Detected and not a spoof
            #   2. Roughly forward-facing (|yaw| < 0.15)
            #      Head turns cause motion blur and ROI drift → huge artifacts.
            #      This is the #1 source of garbage BPM readings.
            rppg = rppg_engines.setdefault(session_id, RPPGEngine())
            face_stable = (
                metrics.face_detected
                and not metrics.is_spoof
                and abs(metrics.yaw_proxy) < 0.15
                and metrics.forehead_rgb is not None
                and metrics.forehead_bbox_norm is not None
            )

            # Position drift gate: if the forehead ROI center moved more than
            # 2% of frame dimensions between consecutive frames, it's motion —
            # reject the sample. This eliminates nodding / vertical movement
            # artifacts that the yaw filter doesn't catch.
            if face_stable:
                bbox = metrics.forehead_bbox_norm
                cx = (bbox[0] + bbox[2]) / 2.0
                cy = (bbox[1] + bbox[3]) / 2.0
                last_pos = rppg_last_pos.get(session_id)
                if last_pos is not None:
                    drift = ((cx - last_pos[0]) ** 2 + (cy - last_pos[1]) ** 2) ** 0.5
                    if drift > 0.02:
                        face_stable = False
                if face_stable:
                    rppg_last_pos[session_id] = (cx, cy)

            if face_stable:
                rppg.add_sample(*metrics.forehead_rgb)

            metrics.rppg_sampling = face_stable
            metrics.rppg_samples  = rppg.n_samples

            # Attach rPPG result once buffer is ready
            if rppg.ready:
                bpm, conf = rppg.compute_bpm()
                is_live_result = rppg.is_live()
                metrics.rppg_bpm        = bpm
                metrics.rppg_confidence = conf
                metrics.rppg_ready      = True
                metrics.rppg_is_live    = is_live_result
                metrics.rppg_verdict    = (
                    "real"      if is_live_result is True  else
                    "synthetic" if is_live_result is False else
                    "pending"
                )

            current_challenge = session.current_challenge

            # Debug: log metrics every 15 frames
            if session.consecutive_count % 15 == 0:
                print(f"[DEBUG] challenge={current_challenge.value} yaw={metrics.yaw_proxy:.3f} smile={metrics.smile_score:.3f} consec={session.consecutive_count}")

            # No face — don't terminate, just guide the user back into frame
            if not metrics.face_detected:
                await _send_response(websocket, session_id, current_challenge,
                                     session.challenge_index, False,
                                     "No face detected — move into the frame",
                                     metrics)
                continue

            if metrics.is_spoof:
                session.state = SessionState.FAILED
                await _send_response(websocket, session_id, ChallengeType.FAILED,
                                     session.challenges_completed, False,
                                     "Spoof attempt detected. Session terminated.",
                                     metrics)
                break

            # After a head-turn, wait for the face to return to neutral first
            if session.waiting_for_neutral:
                if is_neutral(metrics):
                    session.waiting_for_neutral = False
                else:
                    await _send_response(websocket, session_id, current_challenge,
                                         session.challenge_index, False,
                                         "Good! Now face forward again...", metrics)
                    continue

            # Evaluate current challenge
            challenge_passed, new_count = evaluate_challenge(
                current_challenge, metrics, session.consecutive_count
            )
            session.consecutive_count = new_count

            if challenge_passed:
                # Save smile photo before advancing (smile is the last challenge, index 2)
                if current_challenge == ChallengeType.SMILE:
                    photo_path = CAPTURES_DIR / f"{session_id}.jpg"
                    photo_path.write_bytes(frame_bytes)
                    session.smile_photo_path = str(photo_path)

                session.advance_challenge()

                if session.challenge_index >= len(CHALLENGE_SEQUENCE):
                    # All challenges complete — issue liveness token
                    session.state = SessionState.COMPLETE
                    token = _issue_liveness_token(session_id)
                    session.liveness_token = token

                    await _send_response(
                        websocket, session_id, ChallengeType.COMPLETE,
                        session.challenges_completed, True,
                        CHALLENGE_INSTRUCTIONS[ChallengeType.COMPLETE],
                        metrics, liveness_token=token,
                    )
                else:
                    next_challenge = session.current_challenge
                    await _send_response(
                        websocket, session_id, next_challenge,
                        session.challenge_index, False,
                        CHALLENGE_INSTRUCTIONS[next_challenge],
                        metrics,
                    )
            else:
                feedback = _progress_feedback(current_challenge, metrics, new_count)
                await _send_response(
                    websocket, session_id, current_challenge,
                    session.challenge_index, False, feedback, metrics,
                )

    except WebSocketDisconnect:
        pass
    except Exception as e:
        import traceback
        print(f"[WS ERROR] {e}")
        traceback.print_exc()
        try:
            await websocket.send_text(json.dumps({"error": str(e)}))
        except Exception:
            pass
    finally:
        rppg_engines.pop(session_id, None)
        rppg_last_pos.pop(session_id, None)


# ── Helpers ───────────────────────────────────────────────────────────────────

async def _send_response(
    websocket: WebSocket,
    session_id: str,
    challenge: ChallengeType,
    challenge_index: int,
    challenge_passed: bool,
    feedback: str,
    metrics: FaceMetrics,
    liveness_token: str | None = None,
):
    response = FrameResponse(
        session_id=session_id,
        challenge=challenge,
        challenge_index=challenge_index,
        challenge_passed=challenge_passed,
        feedback=feedback,
        metrics=metrics,
        liveness_token=liveness_token,
    )
    await websocket.send_text(response.model_dump_json())


def _issue_liveness_token(session_id: str) -> str:
    now = int(time.time())
    payload = {
        "sub": session_id,
        "iat": now,
        "exp": now + JWT_EXPIRY_SECONDS,
        "type": "liveness",
    }
    return jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALGORITHM)


def _progress_feedback(challenge: ChallengeType, metrics: FaceMetrics, consecutive: int) -> str:
    base = CHALLENGE_INSTRUCTIONS[challenge]
    if not metrics.face_detected:
        return "No face detected. Please look at the camera."
    if consecutive > 0:
        return f"{base} (hold it...)"
    return base


# ── Face Recognition routes ───────────────────────────────────────────────────

@app.post("/face/register", response_model=RegisterFaceResponse)
def register_face(body: RegisterFaceRequest):
    """Register a face. Returns a face_id for future 1:1 verification."""
    try:
        img_bytes = base64.b64decode(body.image)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid base64 image")

    result = rec_engine.analyze(img_bytes)
    if not result.face_detected:
        raise HTTPException(status_code=422, detail="No face detected in the image")

    record = face_db.register(
        name=body.name,
        embedding=result.embedding,
        metadata=body.metadata,
    )
    return RegisterFaceResponse(
        face_id=record.face_id,
        name=record.name,
        age=result.age,
        gender=result.gender,
        detection_score=result.detection_score,
    )


@app.post("/face/verify", response_model=VerifyFaceResponse)
def verify_face(body: VerifyFaceRequest):
    """1:1 — verify a live face against a registered face_id."""
    record = face_db.get(body.face_id)
    if not record:
        raise HTTPException(status_code=404, detail="face_id not found")

    try:
        img_bytes = base64.b64decode(body.image)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid base64 image")

    result = rec_engine.analyze(img_bytes)
    if not result.face_detected:
        raise HTTPException(status_code=422, detail="No face detected in the image")

    verified, similarity = face_db.verify(body.face_id, result.embedding)
    return VerifyFaceResponse(
        face_id=body.face_id,
        verified=verified,
        similarity=similarity,
        threshold=SIMILARITY_THRESHOLD,
    )


@app.post("/face/search", response_model=SearchFaceResponse)
def search_face(body: SearchFaceRequest):
    """1:N — find the closest matching faces in the database."""
    try:
        img_bytes = base64.b64decode(body.image)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid base64 image")

    result = rec_engine.analyze(img_bytes)
    if not result.face_detected:
        return SearchFaceResponse(matches=[], face_detected=False)

    matches = face_db.search(result.embedding, top_k=body.top_k)
    return SearchFaceResponse(
        matches=[SearchMatch(**m) for m in matches],
        face_detected=True,
    )


@app.get("/face/list", response_model=list[FaceRecordResponse])
def list_faces():
    """List all registered faces."""
    return [
        FaceRecordResponse(face_id=r.face_id, name=r.name, metadata=r.metadata)
        for r in face_db.list_all()
    ]


@app.delete("/face/{face_id}")
def delete_face(face_id: str):
    if not face_db.delete(face_id):
        raise HTTPException(status_code=404, detail="face_id not found")
    return {"deleted": True}


# ── Face Analysis route ───────────────────────────────────────────────────────

@app.post("/face/analyze", response_model=AnalyzeResponse)
def analyze_face(body: AnalyzeRequest):
    """
    Single-shot face analysis: age, gender, emotion, smile score.
    Combines InsightFace (age/gender) + MediaPipe blendshapes (emotion/smile).
    """
    try:
        img_bytes = base64.b64decode(body.image)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid base64 image")

    # InsightFace for age + gender
    rec_result = rec_engine.analyze(img_bytes)
    if not rec_result.face_detected:
        return AnalyzeResponse(face_detected=False)

    # MediaPipe for emotion + smile
    liveness_metrics = engine.process_frame(img_bytes)

    emotions: dict | None = None
    dom_emotion: str | None = None
    smile: float | None = None

    if liveness_metrics.face_detected:
        import mediapipe as mp
        import cv2
        import numpy as np
        nparr = np.frombuffer(img_bytes, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        mp_result = engine.detector.detect(mp_image)
        bs = engine.extract_blendshapes(mp_result)

        if bs:
            emotions = blendshapes_to_emotions(bs)
            dom_emotion = dominant_emotion(emotions)
            smile = liveness_metrics.smile_score

    return AnalyzeResponse(
        face_detected=True,
        age=rec_result.age,
        gender=rec_result.gender,
        gender_confidence=rec_result.detection_score,
        emotion=EmotionScores(**emotions) if emotions else None,
        dominant_emotion=dom_emotion,
        smile_score=smile,
        detection_score=rec_result.detection_score,
        bbox=rec_result.bbox,
    )
