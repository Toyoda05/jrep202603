"""
工程 2: 全文スキャン（一度だけ実行）

全セリフの一括Embedding、グローバルプロファイル算出、
クラスタ初期化（KMeans＋シルエットスコア）、LLMによるクラスタラベル付与。
"""

import logging
import numpy as np

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    SentenceTransformer = None

try:
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score
except ImportError:
    KMeans = None
    silhouette_score = None

from src.config import (
    EMBEDDING_MODEL, EMBEDDING_BATCH_SIZE,
    MIN_UTTERANCES_FOR_CLUSTERING, MAX_CLUSTER_K,
)
from src.llm_client import LLMClient

logger = logging.getLogger(__name__)

# グローバルモデル（遅延初期化）
_embedding_model = None


def get_embedding_model():
    """Embeddingモデルを遅延初期化して返す"""
    global _embedding_model
    if _embedding_model is None:
        if SentenceTransformer is None:
            raise ImportError(
                "sentence-transformers が必要です: "
                "pip install sentence-transformers"
            )
        logger.info("Embeddingモデルをロード中: %s", EMBEDDING_MODEL)
        _embedding_model = SentenceTransformer(EMBEDDING_MODEL)
    return _embedding_model


# ──────────────────────────────────────
# Step 1: 全セリフの一括Embedding
# ──────────────────────────────────────

def embed_all_utterances(all_segments: list[dict]) -> dict:
    """
    全章・全セリフをキャラクターごとにEmbeddingする。

    Returns:
        {character: [(vector, segment_meta), ...], ...}
    """
    model = get_embedding_model()

    utterances_by_char: dict[str, list[dict]] = {}
    for seg in all_segments:
        if seg["type"] != "dialogue" or not seg.get("speaker"):
            continue
        char = seg["speaker"]
        utterances_by_char.setdefault(char, [])
        utterances_by_char[char].append(seg)

    embeddings_by_char: dict = {}
    for char, segs in utterances_by_char.items():
        texts = [s["text"] for s in segs]
        if not texts:
            continue

        logger.info("Embedding: %s (%d セリフ)", char, len(texts))
        vectors = model.encode(
            texts,
            batch_size=EMBEDDING_BATCH_SIZE,
            show_progress_bar=False,
            normalize_embeddings=True,
        )
        embeddings_by_char[char] = list(zip(vectors, segs))

    return embeddings_by_char


# ──────────────────────────────────────
# Step 2: グローバルプロファイルの算出
# ──────────────────────────────────────

def compute_chapter_centroids(vec_seg_pairs: list) -> dict:
    """章ごとのセリフベクトル平均を算出する"""
    by_chapter: dict[str, list] = {}
    for vec, seg in vec_seg_pairs:
        ch_id = seg["scene_id"].split("_scene_")[0]
        by_chapter.setdefault(ch_id, [])
        by_chapter[ch_id].append(vec)

    return {
        ch_id: np.mean(vecs, axis=0)
        for ch_id, vecs in by_chapter.items()
    }


def compute_global_profile(embeddings_by_char: dict) -> dict:
    """キャラクターごとのグローバルプロファイルを算出する"""
    global_profiles = {}

    for char, vec_seg_pairs in embeddings_by_char.items():
        vectors = np.array([v for v, _ in vec_seg_pairs])

        global_profiles[char] = {
            "centroid": np.mean(vectors, axis=0),
            "variance": float(np.var(vectors, axis=0).mean()),
            "total_utterances": len(vectors),
            "chapter_centroids": compute_chapter_centroids(vec_seg_pairs),
        }

    return global_profiles


# ──────────────────────────────────────
# Step 3: クラスタ初期化
# ──────────────────────────────────────

def compute_cluster_distribution(labels, segs) -> dict:
    """クラスタごとの章別出現頻度を集計する"""
    distribution: dict = {}
    for label, seg in zip(labels, segs):
        ch_id = seg["scene_id"].split("_scene_")[0]
        distribution.setdefault(int(label), {})
        distribution[int(label)][ch_id] = (
            distribution[int(label)].get(ch_id, 0) + 1
        )
    return distribution


def initialize_clusters(embeddings_by_char: dict,
                        global_profiles: dict,
                        max_k: int = None) -> dict:
    """
    キャラクターごとに最適なクラスタ数kを決定し、
    クラスタ構造を初期化する。
    """
    if KMeans is None:
        raise ImportError(
            "scikit-learn が必要です: pip install scikit-learn"
        )

    max_k = max_k or MAX_CLUSTER_K
    cluster_profiles: dict = {}

    for char, vec_seg_pairs in embeddings_by_char.items():
        vectors = np.array([v for v, _ in vec_seg_pairs])

        if len(vectors) < MIN_UTTERANCES_FOR_CLUSTERING:
            cluster_profiles[char] = {
                "n_clusters": 1,
                "centroids": [global_profiles[char]["centroid"].tolist()],
                "labels": [0] * len(vectors),
                "cluster_chapter_distribution": {},
                "cluster_labels": ["通常モード"],
            }
            continue

        # シルエットスコアで最適kを決定
        best_k, best_score = 1, -1
        for k in range(2, min(max_k + 1, len(vectors) // 5 + 1)):
            if k >= len(vectors):
                break
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            labels = kmeans.fit_predict(vectors)
            score = silhouette_score(vectors, labels)
            if score > best_score:
                best_k, best_score = k, score

        # 最適kでクラスタリング確定
        if best_k == 1:
            cluster_profiles[char] = {
                "n_clusters": 1,
                "centroids": [global_profiles[char]["centroid"].tolist()],
                "labels": [0] * len(vectors),
                "cluster_chapter_distribution": {},
                "cluster_labels": ["通常モード"],
            }
        else:
            kmeans = KMeans(n_clusters=best_k, random_state=42, n_init=10)
            labels = kmeans.fit_predict(vectors)

            cluster_profiles[char] = {
                "n_clusters": best_k,
                "centroids": kmeans.cluster_centers_.tolist(),
                "labels": labels.tolist(),
                "cluster_chapter_distribution": compute_cluster_distribution(
                    labels, [seg for _, seg in vec_seg_pairs]
                ),
                "cluster_labels": [None] * best_k,
            }

        logger.info("クラスタ: %s → k=%d", char, cluster_profiles[char]["n_clusters"])

    return cluster_profiles


# ──────────────────────────────────────
# Step 4: LLMによるクラスタラベル付与
# ──────────────────────────────────────

def label_clusters_with_llm(cluster_profiles: dict,
                            embeddings_by_char: dict,
                            llm: LLMClient) -> dict:
    """各クラスタの代表セリフをLLMに見せて意味ラベルを付与する"""
    for char, profile in cluster_profiles.items():
        if profile["n_clusters"] == 1:
            profile["cluster_labels"] = ["通常モード"]
            continue

        vec_seg_pairs = embeddings_by_char[char]
        labels_list = profile["labels"]

        labeled = []
        for cluster_id in range(profile["n_clusters"]):
            cluster_segs = [
                seg for (_, seg), label
                in zip(vec_seg_pairs, labels_list)
                if label == cluster_id
            ]
            representative = cluster_segs[:5]
            rep_texts = [s["text"] for s in representative]

            ch_dist = profile["cluster_chapter_distribution"].get(cluster_id, {})
            main_chapters = sorted(ch_dist, key=ch_dist.get, reverse=True)[:3]

            prompt = (
                f"キャラクター「{char}」のセリフ群を"
                f"クラスタリングした結果、このクラスタに分類されました。\n\n"
                f"代表セリフ:\n"
                + "\n".join(f'- 「{t}」' for t in rep_texts) + "\n\n"
                f"主に出現する章: {main_chapters}\n\n"
                f"このクラスタはキャラクターのどのような"
                f"「状態・顔・モード」を表していますか？\n"
                f"10文字以内のラベルを付けてください。\n"
                f"例: 「攻撃モード」「内省モード」「信頼モード」"
            )
            label_text = llm.call(prompt).strip().strip("「」")
            labeled.append(label_text)

        profile["cluster_labels"] = labeled
        logger.info("クラスタラベル: %s → %s", char, labeled)

    return cluster_profiles


# ──────────────────────────────────────
# 全文スキャン統合エントリポイント
# ──────────────────────────────────────

def run_full_text_scan(all_segments: list[dict],
                       llm: LLMClient) -> dict:
    """
    全文スキャンのメイン処理。

    Returns:
        {
            "global_profiles": {...},
            "cluster_profiles": {...},
            "embeddings_by_char": {...},
        }
    """
    logger.info("全文スキャンを開始します")

    # Step 1: 一括Embedding
    embeddings_by_char = embed_all_utterances(all_segments)

    # Step 2: グローバルプロファイル算出
    global_profiles = compute_global_profile(embeddings_by_char)

    # Step 3: クラスタ初期化
    cluster_profiles = initialize_clusters(embeddings_by_char, global_profiles)

    # Step 4: クラスタラベル付与
    cluster_profiles = label_clusters_with_llm(
        cluster_profiles, embeddings_by_char, llm
    )

    logger.info("全文スキャン完了")
    return {
        "global_profiles": global_profiles,
        "cluster_profiles": cluster_profiles,
        "embeddings_by_char": embeddings_by_char,
    }
