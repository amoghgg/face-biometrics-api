"""
Emotion detection from MediaPipe face blendshapes.
Maps the 52 blendshape coefficients to 6 emotion classes.
No extra model needed — runs on the blendshapes already computed by liveness_engine.
"""


EMOTIONS = ["happy", "sad", "angry", "surprised", "disgusted", "neutral"]


def blendshapes_to_emotions(blendshapes: dict[str, float]) -> dict[str, float]:
    """
    Maps MediaPipe blendshape scores to emotion probabilities.
    Returns a dict with emotion → score (0–1), all summing to ~1.
    """
    happy = _clamp(
        blendshapes.get("mouthSmileLeft", 0) * 0.5 +
        blendshapes.get("mouthSmileRight", 0) * 0.5
    )

    sad = _clamp(
        blendshapes.get("mouthFrownLeft", 0) * 0.3 +
        blendshapes.get("mouthFrownRight", 0) * 0.3 +
        blendshapes.get("browInnerUp", 0) * 0.25 +
        blendshapes.get("eyeSquintLeft", 0) * 0.075 +
        blendshapes.get("eyeSquintRight", 0) * 0.075
    )

    angry = _clamp(
        blendshapes.get("browDownLeft", 0) * 0.35 +
        blendshapes.get("browDownRight", 0) * 0.35 +
        blendshapes.get("noseSneerLeft", 0) * 0.15 +
        blendshapes.get("noseSneerRight", 0) * 0.15
    )

    surprised = _clamp(
        blendshapes.get("eyeWideLeft", 0) * 0.3 +
        blendshapes.get("eyeWideRight", 0) * 0.3 +
        blendshapes.get("jawOpen", 0) * 0.25 +
        blendshapes.get("browInnerUp", 0) * 0.15
    )

    disgusted = _clamp(
        blendshapes.get("noseSneerLeft", 0) * 0.4 +
        blendshapes.get("noseSneerRight", 0) * 0.4 +
        blendshapes.get("mouthUpperUpLeft", 0) * 0.1 +
        blendshapes.get("mouthUpperUpRight", 0) * 0.1
    )

    raw = {
        "happy": happy,
        "sad": sad,
        "angry": angry,
        "surprised": surprised,
        "disgusted": disgusted,
    }

    total = sum(raw.values())
    if total < 0.15:
        # Face is mostly neutral
        neutral = 1.0 - total
    else:
        neutral = max(0.0, 1.0 - total)

    raw["neutral"] = neutral
    total_with_neutral = sum(raw.values())

    # Normalize so scores sum to 1
    if total_with_neutral > 0:
        return {k: round(v / total_with_neutral, 4) for k, v in raw.items()}
    return {e: 0.0 for e in EMOTIONS}


def dominant_emotion(emotions: dict[str, float]) -> str:
    return max(emotions, key=emotions.get)


def _clamp(v: float) -> float:
    return max(0.0, min(1.0, v))
