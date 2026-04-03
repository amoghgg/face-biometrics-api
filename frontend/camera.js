// camera.js — webcam access and frame capture

const video = document.getElementById("video");
const canvas = document.getElementById("canvas");
const ctx = canvas.getContext("2d");

let captureInterval = null;

async function startCamera() {
  const stream = await navigator.mediaDevices.getUserMedia({
    video: { facingMode: "user", width: { ideal: 640 }, height: { ideal: 480 } },
    audio: false,
  });
  video.srcObject = stream;
  await new Promise((resolve) => (video.onloadedmetadata = resolve));
  canvas.width = video.videoWidth;
  canvas.height = video.videoHeight;
}

function stopCamera() {
  if (video.srcObject) {
    video.srcObject.getTracks().forEach((t) => t.stop());
    video.srcObject = null;
  }
  stopCapture();
}

function startCapture(onFrame, fps = 15) {
  stopCapture();
  const intervalMs = Math.round(1000 / fps);
  captureInterval = setInterval(() => {
    if (!video.srcObject || video.readyState < 2) return;
    // Mirror the canvas to match the mirrored video display
    ctx.save();
    ctx.translate(canvas.width, 0);
    ctx.scale(-1, 1);
    ctx.drawImage(video, 0, 0);
    ctx.restore();

    canvas.toBlob(
      (blob) => {
        if (blob) blob.arrayBuffer().then((buf) => onFrame(buf));
      },
      "image/jpeg",
      0.8
    );
  }, intervalMs);
}

function stopCapture() {
  if (captureInterval !== null) {
    clearInterval(captureInterval);
    captureInterval = null;
  }
}
