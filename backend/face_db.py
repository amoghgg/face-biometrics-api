"""
Face database: stores embeddings + metadata.
Supports 1:1 verify and 1:N search via FAISS.
Persists to disk so it survives server restarts.
"""
import uuid
import json
import numpy as np
import faiss
from pathlib import Path
from dataclasses import dataclass, asdict

DB_DIR = Path(__file__).parent / "face_db"
DB_DIR.mkdir(exist_ok=True)

META_FILE   = DB_DIR / "metadata.json"
EMBED_FILE  = DB_DIR / "embeddings.npy"

EMBEDDING_DIM = 512


@dataclass
class FaceRecord:
    face_id: str
    name: str
    metadata: dict   # any extra fields the caller wants to store (user_id, exam_id, etc.)


class FaceDatabase:
    def __init__(self):
        # FAISS flat inner-product index (cosine similarity on normalized vectors)
        self.index = faiss.IndexFlatIP(EMBEDDING_DIM)
        self.records: list[FaceRecord] = []
        self._load()

    # ── Public API ────────────────────────────────────────────────────────────

    def register(self, name: str, embedding: np.ndarray, metadata: dict = {}) -> FaceRecord:
        face_id = str(uuid.uuid4())
        record = FaceRecord(face_id=face_id, name=name, metadata=metadata)
        self.records.append(record)
        self.index.add(embedding.reshape(1, -1).astype("float32"))
        self._save()
        return record

    def verify(self, face_id: str, embedding: np.ndarray) -> tuple[bool, float]:
        """1:1 — does this embedding match the registered face?"""
        idx = self._index_of(face_id)
        if idx is None:
            return False, 0.0

        stored = self._get_embedding(idx)
        sim = float(np.dot(stored, embedding))
        from face_recognition_engine import SIMILARITY_THRESHOLD
        return sim >= SIMILARITY_THRESHOLD, round(sim, 4)

    def search(self, embedding: np.ndarray, top_k: int = 5) -> list[dict]:
        """1:N — find the top-k most similar registered faces."""
        if self.index.ntotal == 0:
            return []

        k = min(top_k, self.index.ntotal)
        D, I = self.index.search(embedding.reshape(1, -1).astype("float32"), k)

        results = []
        for sim, idx in zip(D[0], I[0]):
            if idx < 0:
                continue
            rec = self.records[idx]
            from face_recognition_engine import SIMILARITY_THRESHOLD
            results.append({
                "face_id": rec.face_id,
                "name": rec.name,
                "similarity": round(float(sim), 4),
                "matched": float(sim) >= SIMILARITY_THRESHOLD,
                "metadata": rec.metadata,
            })
        return results

    def delete(self, face_id: str) -> bool:
        idx = self._index_of(face_id)
        if idx is None:
            return False

        # Rebuild index without the deleted entry
        self.records.pop(idx)
        all_embeddings = self._all_embeddings_except(idx)

        self.index = faiss.IndexFlatIP(EMBEDDING_DIM)
        if all_embeddings.shape[0] > 0:
            self.index.add(all_embeddings)
        self._save()
        return True

    def get(self, face_id: str) -> FaceRecord | None:
        idx = self._index_of(face_id)
        return self.records[idx] if idx is not None else None

    def list_all(self) -> list[FaceRecord]:
        return list(self.records)

    # ── Persistence ───────────────────────────────────────────────────────────

    def _save(self):
        meta = [asdict(r) for r in self.records]
        META_FILE.write_text(json.dumps(meta, indent=2))

        if self.index.ntotal > 0:
            vecs = faiss.rev_swig_ptr(self.index.get_xb(), self.index.ntotal * EMBEDDING_DIM)
            arr = np.array(vecs).reshape(self.index.ntotal, EMBEDDING_DIM)
            np.save(str(EMBED_FILE), arr)

    def _load(self):
        if not META_FILE.exists():
            return
        meta = json.loads(META_FILE.read_text())
        self.records = [FaceRecord(**m) for m in meta]

        if EMBED_FILE.exists() and self.records:
            arr = np.load(str(EMBED_FILE)).astype("float32")
            self.index.add(arr)

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _index_of(self, face_id: str) -> int | None:
        for i, r in enumerate(self.records):
            if r.face_id == face_id:
                return i
        return None

    def _get_embedding(self, idx: int) -> np.ndarray:
        vecs = faiss.rev_swig_ptr(self.index.get_xb(), self.index.ntotal * EMBEDDING_DIM)
        arr = np.array(vecs).reshape(self.index.ntotal, EMBEDDING_DIM)
        return arr[idx]

    def _all_embeddings_except(self, skip_idx: int) -> np.ndarray:
        if self.index.ntotal == 0:
            return np.zeros((0, EMBEDDING_DIM), dtype="float32")
        vecs = faiss.rev_swig_ptr(self.index.get_xb(), self.index.ntotal * EMBEDDING_DIM)
        arr = np.array(vecs).reshape(self.index.ntotal, EMBEDDING_DIM).copy()
        return np.delete(arr, skip_idx, axis=0).astype("float32")


# Singleton
face_db = FaceDatabase()
