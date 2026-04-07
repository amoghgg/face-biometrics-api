# Face Biometrics API

A self-hosted, open-source face biometrics backend with **liveness detection**, **photo validation**, **face recognition**, and a **React Native camera component** — built without any paid SDKs.

Built with [InsightFace](https://github.com/deepinsight/insightface) (buffalo_l), [MediaPipe](https://developers.google.com/mediapipe), and FastAPI.

---

## What it does

### 1. Photo Validation (`POST /api/validate-photo`)
Validates a submitted photo before accepting it for onboarding. Designed to reject:

- Photos with no face detected
- Photos with multiple people in the foreground
- Faces that are too small / too far from the camera
- Off-centre faces
- Glasses or masks (occlusion)
- AI-generated / synthetic images

The AI detection uses three lightweight mathematical signals — no heavy ML model required:
- **Spectral slope** — real camera images follow a 1/f pink-noise power spectrum; AI images deviate
- **Noise floor** — real sensors add grain; AI images are suspiciously clean
- **Face symmetry** — AI faces are unnaturally symmetric; real faces are not

### 2. rPPG Liveness Detection (WebSocket `/ws/liveness/{session_id}`)
Detects whether a face is real or synthetic by measuring a heartbeat signal from subtle green-channel oscillations in the forehead skin (remote photoplethysmography using the CHROM algorithm). Verdict: `real` | `synthetic` | `pending`.

- Motion-gated: frames are rejected if the forehead ROI drifts > 2% between frames (eliminates head-movement artifacts)
- Flat-signal detector: instantly flags still images / printed photos

### 3. Challenge-based Liveness (WebSocket)
Sequential gesture challenges — look left, look right, smile — to confirm the person is present and responsive.

Each challenge requires 20 consecutive passing frames (~1.3s at 15fps). Between challenges the user must return to a neutral forward-facing position before the next begins. Spoof detection runs on every frame via Laplacian texture variance and landmark depth variance.

### 4. Face Recognition
- `POST /face/register` — enrol a face, get a `face_id`
- `POST /face/verify` — 1:1 verification against a registered face
- `POST /face/search` — 1:N search across the database (FAISS)
- `POST /face/analyze` — age, gender, emotion, smile score

### 5. React Native Camera Component (`frontend/FaceCapture.jsx`)
Drop-in Expo component that enforces face quality in real-time before enabling the capture button. Replaces paid SDK dependencies for basic capture gating. Shows live indicators for:
- Face detected
- Single person
- Face centred
- No occlusion

---

## Stack

| Layer | Technology |
|---|---|
| API | FastAPI + Uvicorn |
| Face detection | InsightFace buffalo_l (RetinaFace + ArcFace) |
| Landmarks / liveness | MediaPipe FaceLandmarker (478 landmarks + blendshapes) |
| rPPG | CHROM algorithm (custom implementation) |
| Face search | FAISS (CPU) |
| Frontend debug UI | Vanilla JS + HTML |
| Mobile component | React Native (Expo) |

---

## Project structure

```
face-biometrics-api/
├── backend/
│   ├── main.py                     # FastAPI app, all routes + WebSocket
│   ├── models.py                   # Pydantic request/response models
│   ├── photo_validator.py          # Photo validation engine (7 checks + AI detection)
│   ├── liveness_engine.py          # MediaPipe frame processing, forehead ROI extraction
│   ├── rppg_engine.py              # CHROM rPPG algorithm, heartbeat / liveness verdict
│   ├── challenge_evaluator.py      # Gesture challenge logic (look left/right, smile)
│   ├── face_recognition_engine.py  # InsightFace wrapper
│   ├── face_db.py                  # FAISS-backed face embedding store
│   ├── emotion_engine.py           # Blendshape → emotion scores
│   ├── session_manager.py          # Session lifecycle + expiry
│   └── requirements.txt
├── frontend/
│   ├── FaceCapture.jsx             # React Native (Expo) live camera component
│   ├── debug.html                  # Browser debug UI for the WebSocket liveness stream
│   ├── index.html                  # Simple demo page
│   ├── camera.js                   # Browser camera utilities
│   └── liveness_client.js          # WebSocket client helper
├── LICENSE
└── README.md
```

---

## Quick start

### Prerequisites

- Python 3.10–3.13
- `pip` or `conda`

### 1. Clone and install

```bash
git clone https://github.com/amoghgg/face-biometrics-api
cd face-biometrics-api/backend
pip install -r requirements.txt
```

> On some systems: `pip install --break-system-packages -r requirements.txt`

### 2. Download models

**MediaPipe FaceLandmarker** (~30 MB):
```bash
curl -L -o backend/face_landmarker.task \
  https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task
```

**InsightFace buffalo_l** — downloads automatically on first run (~500 MB).

### 3. Run

```bash
cd backend
uvicorn main:app --port 8000 --reload
```

Interactive API docs: `http://localhost:8000/docs`

**Frontend debug UI:**
```bash
cd frontend
python3 -m http.server 5500
# open http://localhost:5500
```

---

## API reference

### `POST /api/validate-photo`

Accepts **multipart file upload** or **base64 form field**.

```bash
curl -X POST http://localhost:8000/api/validate-photo \
  -F "file=@/path/to/photo.jpg"
```

Response:
```json
{
  "valid": true,
  "rejection_reason": null,
  "face_detected": true,
  "face_count": 1,
  "single_face": true,
  "face_large_enough": true,
  "face_centered": true,
  "no_occlusion": true,
  "not_ai_generated": true,
  "bbox": [120, 80, 400, 420],
  "bbox_norm": [0.19, 0.10, 0.63, 0.53],
  "age": 28,
  "gender": "male",
  "detection_score": 0.97,
  "ai_probability": 0.12,
  "ai_signals": {
    "spectral_slope": -2.41,
    "spectral_score": 0.06,
    "noise_std": 4.82,
    "noise_score": 0.05,
    "face_sym_diff": 11.3,
    "symmetry_score": 0.15,
    "ai_probability": 0.09
  }
}
```

### Liveness (WebSocket)

| Method | Endpoint | Description |
|---|---|---|
| POST | `/session/create` | Start a liveness session |
| WS | `/ws/liveness/{session_id}` | Stream webcam frames, receive challenge feedback + rPPG verdict |
| GET | `/session/{session_id}/status` | Poll session state, retrieve liveness token |
| POST | `/session/{session_id}/verify` | Validate a liveness JWT |
| DELETE | `/session/{session_id}` | Clean up a session |

### Face Recognition

| Method | Endpoint | Description |
|---|---|---|
| POST | `/face/register` | Enrol a face, returns `face_id` |
| POST | `/face/verify` | 1:1 — match face against a `face_id` |
| POST | `/face/search` | 1:N — find closest matches in the database |
| GET | `/face/list` | List all enrolled faces |
| DELETE | `/face/{face_id}` | Remove a face |
| POST | `/face/analyze` | Age, gender, emotion, smile score |

---

## React Native component

`frontend/FaceCapture.jsx` is a self-contained Expo component:

```jsx
import FaceCapture from './FaceCapture';

<FaceCapture
  apiBaseUrl="http://your-server:8000"
  onCapture={(base64, validationResult) => {
    console.log(validationResult.age, validationResult.gender);
  }}
/>
```

Polls `/api/validate-photo` every 600 ms. Capture button stays disabled until all four checks are green.

---

## Tuning

| Parameter | File | Default | Effect |
|---|---|---|---|
| `MIN_FACE_AREA_FRACTION` | `photo_validator.py` | `0.12` | Minimum face size (12% of frame) |
| `MAX_CENTRE_OFFSET` | `photo_validator.py` | `0.25` | How off-centre the face can be |
| `AI_THRESHOLD` | `photo_validator.py` | `0.72` | AI rejection threshold (lower = stricter) |
| `MIN_DETECTION_SCORE` | `photo_validator.py` | `0.65` | InsightFace confidence floor |
| Drift gate | `main.py` | `0.02` | Max ROI movement per frame (normalised) |

---

## Security notes

- Replay attack detection: rolling SHA-256 hash window across recent frames
- Background face tolerance: dominant-face selection — only rejects if a second face has area > 40% of the dominant AND detection score > 0.60
- JWT liveness tokens expire after 5 minutes
- Session TTL: 2 minutes to complete all challenges

## Environment

```bash
JWT_SECRET=your-secret-here  # default is insecure — always override in production
```

---

## Why not Luxand / AWS Rekognition / Azure Face?

- **No per-call cost** — runs entirely on your own infra
- **AI-image detection** — not offered by any of those services out of the box
- **rPPG liveness** — harder to spoof than passive texture/blink methods
- **Full data ownership** — biometric embeddings never leave your server
- **Customisable thresholds** — tune for your user population, not a vendor's defaults

---

## Contributing

PRs welcome. Most useful contributions:

- Improved AI-image detection (better thresholds, more signals)
- GPU support (`CUDAExecutionProvider` for InsightFace)
- Flutter version of `FaceCapture`
- Docker / docker-compose setup
- Test coverage

---

## License

MIT — see [LICENSE](LICENSE).
