/**
 * FaceCapture.jsx — React Native (Expo) face capture component
 * ─────────────────────────────────────────────────────────────
 * Drop this anywhere in your onboarding flow.
 *
 * What it does:
 *   • Shows a camera feed with a centred oval face guide
 *   • Every 600ms, sends the frame to POST /api/validate-photo
 *   • Shows a real-time status row for each requirement:
 *       ✓ / ✗  Face detected
 *       ✓ / ✗  One person only
 *       ✓ / ✗  Face centred
 *       ✓ / ✗  No glasses / mask
 *   • Capture button only activates when all 4 pass
 *   • On capture: sends the still to the API one final time, then calls
 *     onCapture(base64Jpeg, validationResult) so your parent component
 *     can handle upload / navigation
 *
 * Props:
 *   apiBase      string  — e.g. "https://your-api.com"
 *   onCapture    fn      — (base64: string, result: object) => void
 *   onCancel     fn      — () => void  (optional)
 *
 * Dependencies (install once per project):
 *   expo install expo-camera
 *
 * Usage:
 *   import FaceCapture from './FaceCapture';
 *   <FaceCapture
 *     apiBase="http://localhost:8000"
 *     onCapture={(b64, result) => console.log(result)}
 *     onCancel={() => navigation.goBack()}
 *   />
 */

import React, { useRef, useState, useEffect, useCallback } from "react";
import {
  View,
  Text,
  TouchableOpacity,
  StyleSheet,
  ActivityIndicator,
  Dimensions,
} from "react-native";
import { CameraView, useCameraPermissions } from "expo-camera";

// ── Constants ─────────────────────────────────────────────────────────────────

const VALIDATE_INTERVAL_MS = 600;   // how often to send frames to the API
const JPEG_QUALITY         = 0.75;  // lower = faster round-trip; 0.75 is fine

const CHECKS = [
  { key: "face_detected",    label: "Face detected"    },
  { key: "single_face",      label: "One person only"  },
  { key: "face_centered",    label: "Face centred"     },
  { key: "no_occlusion",     label: "No glasses / mask"},
];

// ── Component ─────────────────────────────────────────────────────────────────

export default function FaceCapture({ apiBase, onCapture, onCancel }) {
  const cameraRef                   = useRef(null);
  const validationTimer             = useRef(null);
  const isValidating                = useRef(false);

  const [permission, requestPermission] = useCameraPermissions();
  const [checks,    setChecks]       = useState({});
  const [message,   setMessage]      = useState("Position your face in the oval");
  const [capturing, setCapturing]    = useState(false);
  const [allPassed, setAllPassed]    = useState(false);

  // ── Permissions ─────────────────────────────────────────────────────────────

  useEffect(() => {
    if (permission && !permission.granted) requestPermission();
  }, [permission]);

  // ── Validation loop ──────────────────────────────────────────────────────────

  const runValidation = useCallback(async () => {
    if (isValidating.current || !cameraRef.current) return;
    isValidating.current = true;

    try {
      // Take a low-quality snapshot for validation (NOT the final capture)
      const snap = await cameraRef.current.takePictureAsync({
        quality: JPEG_QUALITY,
        base64: true,
        skipProcessing: true,
        exif: false,
      });

      const res = await fetch(`${apiBase}/api/validate-photo`, {
        method: "POST",
        headers: { "Content-Type": "application/x-www-form-urlencoded" },
        body:    `image=${encodeURIComponent(snap.base64)}`,
      });

      if (!res.ok) throw new Error(`API ${res.status}`);
      const result = await res.json();

      // Update check states
      const newChecks = {};
      CHECKS.forEach(({ key }) => { newChecks[key] = !!result[key]; });
      setChecks(newChecks);

      const passed = CHECKS.every(({ key }) => !!result[key]);
      setAllPassed(passed);

      if (passed) {
        setMessage("Hold still…");
      } else {
        setMessage(result.rejection_reason || "Adjust your position");
      }
    } catch {
      // Network error — don't surface to user, just keep trying
    } finally {
      isValidating.current = false;
    }
  }, [apiBase]);

  useEffect(() => {
    if (!permission?.granted) return;
    validationTimer.current = setInterval(runValidation, VALIDATE_INTERVAL_MS);
    return () => clearInterval(validationTimer.current);
  }, [permission?.granted, runValidation]);

  // ── Capture ──────────────────────────────────────────────────────────────────

  const handleCapture = useCallback(async () => {
    if (!allPassed || capturing || !cameraRef.current) return;
    setCapturing(true);
    clearInterval(validationTimer.current);

    try {
      // Take full-quality final photo
      const photo = await cameraRef.current.takePictureAsync({
        quality: 1.0,
        base64: true,
        skipProcessing: false,
        exif: false,
      });

      // One final server-side validation on the high-quality image
      const res = await fetch(`${apiBase}/api/validate-photo`, {
        method: "POST",
        headers: { "Content-Type": "application/x-www-form-urlencoded" },
        body:    `image=${encodeURIComponent(photo.base64)}`,
      });

      const finalResult = res.ok ? await res.json() : { valid: false };

      if (finalResult.valid) {
        onCapture?.(photo.base64, finalResult);
      } else {
        // Something changed in the last frame — restart validation
        setMessage(finalResult.rejection_reason || "Please try again");
        setAllPassed(false);
        validationTimer.current = setInterval(runValidation, VALIDATE_INTERVAL_MS);
        setCapturing(false);
      }
    } catch {
      setMessage("Connection error — please try again");
      setCapturing(false);
      validationTimer.current = setInterval(runValidation, VALIDATE_INTERVAL_MS);
    }
  }, [allPassed, capturing, apiBase, onCapture, runValidation]);

  // ── Render ───────────────────────────────────────────────────────────────────

  if (!permission) return <View style={styles.center}><ActivityIndicator color="#fff" /></View>;

  if (!permission.granted) {
    return (
      <View style={styles.center}>
        <Text style={styles.permissionText}>Camera access is required.</Text>
        <TouchableOpacity style={styles.retryBtn} onPress={requestPermission}>
          <Text style={styles.retryBtnText}>Grant Permission</Text>
        </TouchableOpacity>
      </View>
    );
  }

  return (
    <View style={styles.container}>
      {/* Camera feed */}
      <CameraView
        ref={cameraRef}
        style={StyleSheet.absoluteFill}
        facing="front"
        mirror
      />

      {/* Oval face guide */}
      <OvalGuide allPassed={allPassed} />

      {/* Status bar */}
      <View style={styles.statusBar}>
        {CHECKS.map(({ key, label }) => (
          <CheckRow
            key={key}
            label={label}
            passed={!!checks[key]}
            pending={Object.keys(checks).length === 0}
          />
        ))}
      </View>

      {/* Feedback message */}
      <View style={styles.messageWrap}>
        <Text style={[styles.message, allPassed && styles.messageGood]}>
          {message}
        </Text>
      </View>

      {/* Capture button */}
      <View style={styles.footer}>
        {onCancel && (
          <TouchableOpacity style={styles.cancelBtn} onPress={onCancel}>
            <Text style={styles.cancelText}>Cancel</Text>
          </TouchableOpacity>
        )}

        <TouchableOpacity
          style={[styles.captureBtn, !allPassed && styles.captureBtnDisabled]}
          onPress={handleCapture}
          disabled={!allPassed || capturing}
          activeOpacity={0.8}
        >
          {capturing
            ? <ActivityIndicator color="#fff" />
            : <View style={styles.captureInner} />
          }
        </TouchableOpacity>

        {/* Spacer to balance the cancel button */}
        {onCancel && <View style={{ width: 72 }} />}
      </View>
    </View>
  );
}

// ── Sub-components ────────────────────────────────────────────────────────────

function OvalGuide({ allPassed }) {
  const { width, height } = Dimensions.get("window");
  const ovalW = width  * 0.68;
  const ovalH = height * 0.52;
  const borderColor = allPassed ? "#22c55e" : "rgba(255,255,255,0.6)";
  const shadowColor = allPassed ? "#22c55e" : "transparent";

  return (
    <View
      style={[
        styles.oval,
        {
          width:         ovalW,
          height:        ovalH,
          borderRadius:  ovalW / 2,
          borderColor,
          shadowColor,
        },
      ]}
    />
  );
}

function CheckRow({ label, passed, pending }) {
  const icon  = pending ? "○" : passed ? "✓" : "✗";
  const color = pending ? "#94a3b8" : passed ? "#22c55e" : "#ef4444";
  return (
    <View style={styles.checkRow}>
      <Text style={[styles.checkIcon, { color }]}>{icon}</Text>
      <Text style={[styles.checkLabel, { color }]}>{label}</Text>
    </View>
  );
}

// ── Styles ────────────────────────────────────────────────────────────────────

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: "#000",
    alignItems: "center",
    justifyContent: "center",
  },
  center: {
    flex: 1,
    backgroundColor: "#07090e",
    alignItems: "center",
    justifyContent: "center",
    padding: 24,
  },
  permissionText: {
    color: "#e2e8f0",
    fontSize: 16,
    textAlign: "center",
    marginBottom: 20,
  },
  retryBtn: {
    backgroundColor: "#3b82f6",
    paddingHorizontal: 24,
    paddingVertical: 12,
    borderRadius: 8,
  },
  retryBtnText: {
    color: "#fff",
    fontWeight: "700",
    fontSize: 15,
  },

  // Oval guide
  oval: {
    position: "absolute",
    borderWidth: 2.5,
    top: "14%",
    shadowOpacity: 0.8,
    shadowOffset: { width: 0, height: 0 },
    shadowRadius: 12,
    elevation: 6,
  },

  // Status bar (top)
  statusBar: {
    position: "absolute",
    top: 16,
    left: 16,
    backgroundColor: "rgba(7,9,14,0.75)",
    borderRadius: 10,
    padding: 10,
    gap: 4,
  },
  checkRow: {
    flexDirection: "row",
    alignItems: "center",
    gap: 6,
  },
  checkIcon: {
    fontSize: 13,
    fontWeight: "800",
    width: 16,
    textAlign: "center",
  },
  checkLabel: {
    fontSize: 12,
    fontWeight: "600",
  },

  // Message
  messageWrap: {
    position: "absolute",
    bottom: "18%",
    left: 24,
    right: 24,
    alignItems: "center",
  },
  message: {
    color: "rgba(255,255,255,0.8)",
    fontSize: 14,
    fontWeight: "600",
    textAlign: "center",
    backgroundColor: "rgba(7,9,14,0.6)",
    paddingHorizontal: 16,
    paddingVertical: 6,
    borderRadius: 20,
  },
  messageGood: {
    color: "#22c55e",
  },

  // Footer + capture button
  footer: {
    position: "absolute",
    bottom: 48,
    left: 0,
    right: 0,
    flexDirection: "row",
    alignItems: "center",
    justifyContent: "center",
    gap: 32,
  },
  cancelBtn: {
    width: 72,
    alignItems: "center",
  },
  cancelText: {
    color: "rgba(255,255,255,0.75)",
    fontSize: 15,
    fontWeight: "600",
  },
  captureBtn: {
    width: 72,
    height: 72,
    borderRadius: 36,
    borderWidth: 3,
    borderColor: "#fff",
    alignItems: "center",
    justifyContent: "center",
    backgroundColor: "transparent",
  },
  captureBtnDisabled: {
    borderColor: "rgba(255,255,255,0.3)",
    opacity: 0.5,
  },
  captureInner: {
    width: 56,
    height: 56,
    borderRadius: 28,
    backgroundColor: "#fff",
  },
});
