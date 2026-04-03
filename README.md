# Face Biometrics API

A production-ready face biometrics API built with FastAPI and MediaPipe. Covers active liveness detection, face recognition (1:1 and 1:N), and single-shot face analysis — age, gender, and emotion — in one unified REST + WebSocket interface.

## What it does

**Liveness Detection** — challenges the user in real-time via webcam: turn head left, turn head right, smile. Uses MediaPipe Face Landmarker blendshapes for accurate pose and expression tracking. Issues a signed JWT on completion.

**Face Recognition** — register a face, verify it against a stored identity (1:1), or search across the entire database (1:N). Powered by InsightFace's ArcFace ResNet-50, the same model family used in production biometric systems. FAISS handles the vector search.

**Face Analysis** — single image in, structured data out: estimated age, gender, smile score, and emotion breakdown (happy, sad, angry, surprised, disgusted, neutral).

## Stack

| Layer | Technology |
|---|---|
| API | FastAPI + Uvicorn |
| Liveness | MediaPipe Face Landmarker (blendshapes) |
| Face Recognition | InsightFace buffalo_l (ArcFace + RetinaFace) |
| Vector Search | FAISS flat inner-product index |
| Emotion | MediaPipe blendshapes → 6-class mapping |
| Auth | JWT (python-jose) |

## API Reference

### Liveness

| Method | Endpoint | Description |
|---|---|---|
| POST | `/session/create` | Start a liveness session |
| WS | `/ws/liveness/{session_id}` | Stream webcam frames, receive real-time challenge feedback |
| GET | `/session/{session_id}/status` | Poll session state and retrieve liveness token |
| POST | `/session/{session_id}/verify` | Validate a liveness token |

### Face Recognition

| Method | Endpoint | Description |
|---|---|---|
| POST | `/face/register` | Register a face, get back a `face_id` |
| POST | `/face/verify` | 1:1 — does this face match a stored `face_id`? |
| POST | `/face/search` | 1:N — who is this face? Returns top-k matches with similarity scores |
| GET | `/face/list` | List all registered faces |
| DELETE | `/face/{face_id}` | Remove a face from the database |

### Analysis

| Method | Endpoint | Description |
|---|---|---|
| POST | `/face/analyze` | Age, gender, emotion, smile score from a single image |

Full interactive docs at `/docs` once the server is running.

## Getting Started

```bash
git clone https://github.com/amoghgg/face-biometrics-api
cd face-biometrics-api/backend

pip install -r requirements.txt

# Download MediaPipe face landmarker model
curl -L "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task" \
  -o face_landmarker.task

uvicorn main:app --port 8000
```

InsightFace models (`buffalo_l`) download automatically on first run (~500MB).

**Frontend demo**

```bash
cd frontend
python3 -m http.server 5500
# open http://localhost:5500
```

## How liveness works

Each challenge requires 20 consecutive passing frames (~1.3s at 15fps) before it counts. Head turn direction is computed from the nose-to-eye-corner ratio on the mirrored webcam frame. Smile is detected from the `mouthSmileLeft` and `mouthSmileRight` blendshape scores — not a heuristic ratio, actual facial muscle activation coefficients from MediaPipe's model.

Spoof detection runs on every frame: Laplacian texture variance flags printed photos (flat surfaces score below threshold), and Z-coordinate standard deviation across all 468 landmarks catches flat 2D attacks.

Between each challenge, the user must return to a neutral forward-facing position before the next one begins — prevents a single head sweep from satisfying multiple challenges.

## Security notes

- Replay attack detection: rolling SHA-256 hash window across recent frames
- Multi-face rejection: session fails if more than one face is detected
- JWT liveness tokens expire after 5 minutes
- Session TTL: 2 minutes to complete all challenges

## Environment

```bash
JWT_SECRET=your-secret-here  # default is insecure, always set in production
```
