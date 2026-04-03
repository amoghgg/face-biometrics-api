import uuid
import hashlib
import time
from datetime import datetime, timezone, timedelta
from models import ChallengeType, SessionState, CHALLENGE_SEQUENCE

SESSION_TTL_SECONDS = 120  # 2 minutes to complete all challenges
MAX_FRAME_HASHES = 20       # Rolling window for replay detection


class Session:
    def __init__(self, session_id: str):
        self.session_id = session_id
        self.state = SessionState.IN_PROGRESS
        self.challenge_index = 0
        self.consecutive_count = 0
        self.challenges_completed = 0
        self.liveness_token: str | None = None
        self.smile_photo_path: str | None = None
        self.waiting_for_neutral: bool = False  # must return to forward-facing before next challenge
        self.created_at = time.monotonic()
        self.frame_hashes: list[str] = []

    @property
    def current_challenge(self) -> ChallengeType:
        if self.challenge_index < len(CHALLENGE_SEQUENCE):
            return CHALLENGE_SEQUENCE[self.challenge_index]
        return ChallengeType.COMPLETE

    @property
    def is_expired(self) -> bool:
        return (time.monotonic() - self.created_at) > SESSION_TTL_SECONDS

    @property
    def expires_at_iso(self) -> str:
        expiry = datetime.now(timezone.utc) + timedelta(seconds=SESSION_TTL_SECONDS)
        return expiry.isoformat()

    def advance_challenge(self):
        self.challenge_index += 1
        self.challenges_completed += 1
        self.consecutive_count = 0
        # For head-turn challenges, require returning to neutral before next one
        from models import CHALLENGE_SEQUENCE, ChallengeType
        next_is_head_turn = (
            self.challenge_index < len(CHALLENGE_SEQUENCE) and
            CHALLENGE_SEQUENCE[self.challenge_index] in (ChallengeType.TURN_LEFT, ChallengeType.TURN_RIGHT)
        )
        self.waiting_for_neutral = next_is_head_turn

    def is_replay_frame(self, frame_bytes: bytes) -> bool:
        frame_hash = hashlib.sha256(frame_bytes).hexdigest()
        if frame_hash in self.frame_hashes:
            return True
        self.frame_hashes.append(frame_hash)
        if len(self.frame_hashes) > MAX_FRAME_HASHES:
            self.frame_hashes.pop(0)
        return False


class SessionManager:
    def __init__(self):
        self._sessions: dict[str, Session] = {}

    def create_session(self) -> Session:
        session_id = str(uuid.uuid4())
        session = Session(session_id)
        self._sessions[session_id] = session
        return session

    def get_session(self, session_id: str) -> Session | None:
        session = self._sessions.get(session_id)
        if session and session.is_expired and session.state == SessionState.IN_PROGRESS:
            session.state = SessionState.EXPIRED
        return session

    def delete_session(self, session_id: str) -> bool:
        if session_id in self._sessions:
            del self._sessions[session_id]
            return True
        return False

    def cleanup_expired(self):
        expired = [
            sid for sid, s in self._sessions.items()
            if s.is_expired and s.state not in (SessionState.COMPLETE,)
        ]
        for sid in expired:
            del self._sessions[sid]


# Singleton
session_manager = SessionManager()
