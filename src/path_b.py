"""
工程 7: Path B — Embedding 逐次更新（性格履歴ループ）

ウェルフォード法による重心・分散の逐次更新、
クラスタ親和性計算（ソフトアサインメント）、
章スナップショット保存。
"""

import logging
import numpy as np

from src.models import PathBState
from src.full_text_scan import get_embedding_model
from src.config import EMBEDDING_BATCH_SIZE

logger = logging.getLogger(__name__)


# ──────────────────────────────────────
# ウェルフォード法
# ──────────────────────────────────────

def welford_update(state: PathBState,
                   new_vector: np.ndarray) -> PathBState:
    """
    ウェルフォードのオンラインアルゴリズムで
    重心と分散をO(1)更新する。
    """
    state.n += 1
    delta = new_vector - state.centroid
    state.centroid = state.centroid + delta / state.n
    delta2 = new_vector - state.centroid
    state.M2 = state.M2 + delta * delta2
    return state


def get_current_variance(state: PathBState) -> float:
    """現時点での分散スカラー値を返す"""
    if state.n < 2:
        return 0.0
    return float(np.mean(state.M2 / state.n))


# ──────────────────────────────────────
# ソフトマックス
# ──────────────────────────────────────

def softmax(x: np.ndarray, temperature: float = 0.5) -> np.ndarray:
    """温度付きソフトマックス"""
    x = x / temperature
    x = x - np.max(x)
    exp_x = np.exp(x)
    return exp_x / exp_x.sum()


# ──────────────────────────────────────
# クラスタ親和性の計算
# ──────────────────────────────────────

def compute_chapter_cluster_affinity(vectors: np.ndarray,
                                     cluster_centroids: list) -> dict:
    """今章のセリフ群が各クラスタにどれだけ近いかを計算する"""
    if cluster_centroids is None or len(vectors) == 0:
        return {}

    centroids = np.array(cluster_centroids)
    affinities: dict[int, float] = {i: 0.0 for i in range(len(centroids))}

    for vec in vectors:
        similarities = np.dot(centroids, vec)
        weights = softmax(similarities)
        for i, w in enumerate(weights):
            affinities[i] += float(w)

    total = sum(affinities.values())
    if total > 0:
        return {i: v / total for i, v in affinities.items()}
    return affinities


# ──────────────────────────────────────
# スナップショット保存
# ──────────────────────────────────────

def save_chapter_snapshot(state: PathBState,
                          chapter_id: str,
                          cluster_affinity: dict = None,
                          chapter_vectors: np.ndarray = None):
    """章末時点のPath B状態を保存する"""
    chapter_centroid = None
    if chapter_vectors is not None and len(chapter_vectors) > 0:
        chapter_centroid = np.mean(chapter_vectors, axis=0).tolist()

    snapshot = {
        "chapter_id": chapter_id,
        "n_utterances_so_far": state.n,
        "centroid": state.centroid.copy(),
        "variance": get_current_variance(state),
        "cluster_affinity": cluster_affinity or {},
        "chapter_centroid": chapter_centroid,
    }
    state.chapter_snapshots[chapter_id] = snapshot


# ──────────────────────────────────────
# 章単位の更新処理
# ──────────────────────────────────────

def update_path_b_for_chapter(char: str,
                              chapter_id: str,
                              segments: list[dict],
                              path_b_states: dict) -> PathBState:
    """1章分のセリフをまとめて受け取り、Path B状態を更新する"""
    state = path_b_states[char]
    model = get_embedding_model()

    # そのキャラクターのセリフのみ抽出
    char_utterances = [
        seg["text"] for seg in segments
        if seg["type"] == "dialogue" and seg.get("speaker") == char
    ]

    if not char_utterances:
        save_chapter_snapshot(state, chapter_id)
        return state

    # バッチEmbedding
    vectors = model.encode(
        char_utterances,
        batch_size=EMBEDDING_BATCH_SIZE,
        normalize_embeddings=True,
    )

    # セリフごとにウェルフォード更新
    for vec in vectors:
        state = welford_update(state, vec)

    # クラスタ親和性を計算
    chapter_cluster_distribution = compute_chapter_cluster_affinity(
        vectors, state.cluster_centroids
    )

    # スナップショット保存
    save_chapter_snapshot(
        state, chapter_id, chapter_cluster_distribution, vectors
    )

    return state


# ──────────────────────────────────────
# 統合エントリポイント
# ──────────────────────────────────────

def run_path_b_chapter(char: str,
                       chapter_id: str,
                       segments: list[dict],
                       path_b_states: dict,
                       cluster_profiles: dict) -> dict:
    """章ごとのPath B処理の統合エントリポイント"""
    # クラスタ情報を状態に持たせる（全文スキャンで確定済み）
    if char in cluster_profiles:
        if path_b_states[char].cluster_centroids is None:
            path_b_states[char].cluster_centroids = \
                cluster_profiles[char]["centroids"]
            path_b_states[char].cluster_labels_text = \
                cluster_profiles[char]["cluster_labels"]

    # 章の逐次更新
    path_b_states[char] = update_path_b_for_chapter(
        char, chapter_id, segments, path_b_states
    )

    return path_b_states[char].chapter_snapshots.get(chapter_id, {})


def initialize_path_b_states(known_characters: list[str],
                             embedding_dim: int = 1024) -> dict:
    """全キャラクターのPath B状態を初期化する"""
    return {
        char: PathBState(
            centroid=np.zeros(embedding_dim),
            M2=np.zeros(embedding_dim),
        )
        for char in known_characters
    }
