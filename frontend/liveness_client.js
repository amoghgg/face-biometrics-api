// liveness_client.js — WebSocket client and UI state machine

const API_BASE = "http://localhost:8000";
const WS_BASE  = "ws://localhost:8000";

const startBtn     = document.getElementById("start-btn");
const feedbackText = document.getElementById("feedback-text");
const faceOval     = document.getElementById("face-oval");
const progressBar  = document.getElementById("progress-bar");
const tokenBox     = document.getElementById("token-box");

const STEPS = [0, 1, 2];
const TOTAL_CHALLENGES = 3;

let ws = null;

startBtn.addEventListener("click", async () => {
  startBtn.disabled = true;
  tokenBox.style.display = "none";
  tokenBox.textContent = "";
  resetStepUI();

  try {
    await startCamera();
  } catch (err) {
    setFeedback("Camera access denied. Please allow camera permissions.", "error");
    startBtn.disabled = false;
    return;
  }

  // Create session
  let sessionId;
  try {
    const res = await fetch(`${API_BASE}/session/create`, { method: "POST" });
    const data = await res.json();
    sessionId = data.session_id;
  } catch {
    setFeedback("Could not reach the server. Is the API running?", "error");
    stopCamera();
    startBtn.disabled = false;
    return;
  }

  setFeedback("Position your face in the oval...");
  setOvalState("active");

  // Open WebSocket
  ws = new WebSocket(`${WS_BASE}/ws/liveness/${sessionId}`);
  ws.binaryType = "arraybuffer";

  ws.onopen = () => {
    startCapture((frameBuffer) => {
      if (ws && ws.readyState === WebSocket.OPEN) {
        ws.send(frameBuffer);
      }
    }, 15);
  };

  ws.onmessage = (event) => {
    let data;
    try { data = JSON.parse(event.data); } catch { return; }

    handleServerMessage(data);
  };

  ws.onclose = () => {
    stopCapture();
  };

  ws.onerror = () => {
    setFeedback("Connection error.", "error");
    setOvalState("error");
    cleanup();
  };
});

function handleServerMessage(data) {
  const { challenge, challenge_index, challenge_passed, feedback, liveness_token } = data;

  setFeedback(feedback);

  // Update step indicators
  updateSteps(challenge_index, challenge);

  // Update progress bar (per-challenge progress)
  updateProgressBar(challenge_index, challenge);

  if (challenge === "COMPLETE") {
    setOvalState("success");
    setFeedback("✓ Verification complete!", "success");
    updateSteps(TOTAL_CHALLENGES, "COMPLETE");
    progressBar.style.width = "100%";
    progressBar.style.background = "#34d399";

    if (liveness_token) {
      tokenBox.style.display = "block";
      tokenBox.textContent = `Liveness Token: ${liveness_token}`;
    }

    cleanup();
    startBtn.textContent = "Verify Again";
    startBtn.disabled = false;
    return;
  }

  if (challenge === "FAILED") {
    setOvalState("error");
    cleanup();
    startBtn.textContent = "Try Again";
    startBtn.disabled = false;
    return;
  }

  // Highlight oval when face is detected
  const faceDetected = data.metrics?.face_detected ?? false;
  if (!faceDetected) {
    setOvalState("active");  // dim — looking for face
  }
}

function updateSteps(challengeIndex, currentChallenge) {
  STEPS.forEach((i) => {
    const el = document.getElementById(`step-${i}`);
    el.classList.remove("active", "done");
    if (i < challengeIndex) el.classList.add("done");
    else if (i === challengeIndex && currentChallenge !== "COMPLETE" && currentChallenge !== "FAILED") {
      el.classList.add("active");
    }
  });
}

function updateProgressBar(challengeIndex, challenge) {
  if (challenge === "COMPLETE") {
    progressBar.style.width = "100%";
    return;
  }
  const pct = (challengeIndex / TOTAL_CHALLENGES) * 100;
  progressBar.style.width = `${pct}%`;
  progressBar.style.background = "#4f8ef7";
}

function setFeedback(text, type = "normal") {
  feedbackText.textContent = text;
  feedbackText.style.color =
    type === "success" ? "#34d399" :
    type === "error"   ? "#f87171" :
    "#e8eaed";
}

function setOvalState(state) {
  faceOval.classList.remove("active", "success", "error");
  if (state) faceOval.classList.add(state);
}

function resetStepUI() {
  STEPS.forEach((i) => {
    const el = document.getElementById(`step-${i}`);
    el.classList.remove("active", "done");
  });
  progressBar.style.width = "0%";
  progressBar.style.background = "#4f8ef7";
  setOvalState("");
}

function cleanup() {
  stopCapture();
  stopCamera();
  if (ws) {
    ws.close();
    ws = null;
  }
}
