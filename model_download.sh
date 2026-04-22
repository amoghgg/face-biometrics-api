#!/bin/bash
set -e

echo "=== Face Biometrics API — Model Download ==="
echo ""

cd "$(dirname "$0")/backend"

# MediaPipe FaceLandmarker
if [ -f "face_landmarker.task" ]; then
    echo "✓ face_landmarker.task already exists, skipping"
else
    echo "Downloading MediaPipe FaceLandmarker (~30 MB)..."
    curl -fL --progress-bar -o face_landmarker.task \
        https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task
    echo "✓ face_landmarker.task downloaded"
fi

echo ""
echo "Pre-downloading InsightFace buffalo_l (~500 MB)..."
echo "This runs once and caches to ~/.insightface/"
python3 -c "
from insightface.app import FaceAnalysis
print('Initialising InsightFace...')
app = FaceAnalysis(name='buffalo_l', providers=['CPUExecutionProvider'])
app.prepare(ctx_id=-1)
print('✓ InsightFace buffalo_l ready')
"

echo ""
echo "All models ready. Run the API with:"
echo "  cd backend && uvicorn main:app --port 8000 --reload"
echo ""
echo "Or with Docker:"
echo "  docker compose up"
