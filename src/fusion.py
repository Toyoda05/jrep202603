"""
工程 8: 矛盾検出・統合

Path A（LLM解釈）とPath B（Embedding統計）の結果を比較し、
食い違いの深刻度に応じて統合方式を決定する。
"""

import logging
import numpy as np

from src.models import PersonalityState, FusionResult
from src.config import (
    CONTRADICTION_NONE_THRESHOLD,
    CONTRADICTION_SOFT_THRESHOLD,
    MAX_REGENERATION_ATTEMPTS,
    CLUSTER_SCORE_MAP,
)
from src.llm_client import LLMClient
from src.full_text_scan import get_embedding_model

logger = logging.getLogger(__name__)


# ──────────────────────────────────────
# 矛盾スコアの算出
# ──────────────────────────────────────

def compute_path_ab_contradiction(path_a_state: PersonalityState,
                                  path_b_snapshot: dict,
                                  cluster_labels: list = None) -> dict:
    """Path AとPath Bの矛盾スコアを計算する"""
    model = get_embedding_model()

    # ① サマリーEmbeddingと重心のコサイン類似度
    if path_a_state.summary:
        summary_vec = model.encode(
            [path_a_state.summary],
            normalize_embeddings=True,
        )[0]
    else:
        summary_vec = np.zeros_like(path_b_snapshot.get(
            "centroid", np.zeros(1024)
        ))

    centroid = np.array(path_b_snapshot.get("centroid", np.zeros(1024)))
    norm = np.linalg.norm(centroid)
    if norm > 1e-8:
        centroid = centroid / norm
    cosine_sim = float(np.dot(summary_vec, centroid))

    # ② クラスタ親和性とPath Aスコアの整合性チェック
    affinity_ok = check_affinity_score_consistency(
        path_b_snapshot.get("cluster_affinity", {}),
        path_a_state,
        cluster_labels or [],
    )

    contradiction_score = max(0.0, 1.0 - cosine_sim)

    return {
        "cosine_similarity": cosine_sim,
        "contradiction_score": contradiction_score,
        "affinity_consistency": affinity_ok,
        "needs_regeneration": (
            contradiction_score > CONTRADICTION_SOFT_THRESHOLD
            or not affinity_ok
        ),
    }


def check_affinity_score_consistency(cluster_affinity: dict,
                                     path_a_state: PersonalityState,
                                     cluster_labels: list) -> bool:
    """クラスタ親和性とPath Aスコアの整合チェック"""
    if not cluster_affinity or not cluster_labels:
        return True

    dominant_cluster_id = max(cluster_affinity, key=cluster_affinity.get)
    idx = int(dominant_cluster_id)
    if idx >= len(cluster_labels):
        return True
    dominant_label = cluster_labels[idx]

    rules = {
        "激情モード": lambda s: s.aggression > 0.6,
        "内省モード": lambda s: s.anxiety > 0.5 or s.openness < 0.4,
        "信頼モード": lambda s: s.loyalty > 0.6 and s.anxiety < 0.4,
        "攻撃モード": lambda s: s.aggression > 0.7,
    }
    rule = rules.get(dominant_label)
    if rule is None:
        return True
    return rule(path_a_state)


# ──────────────────────────────────────
# 分類
# ──────────────────────────────────────

def classify_contradiction(contradiction_result: dict) -> str:
    """矛盾の深刻度を分類する"""
    score = contradiction_result["contradiction_score"]
    if score <= CONTRADICTION_NONE_THRESHOLD:
        return "none"
    elif score <= CONTRADICTION_SOFT_THRESHOLD:
        return "soft"
    else:
        return "hard"


# ──────────────────────────────────────
# Path BからPath A相当のスコア導出
# ──────────────────────────────────────

def derive_scores_from_path_b(path_b_snapshot: dict) -> PersonalityState:
    """クラスタ親和性と分散から性格スコアを近似する"""
    affinity = path_b_snapshot.get("cluster_affinity", {})
    variance = path_b_snapshot.get("variance", 0.05)
    cluster_labels_map = path_b_snapshot.get("cluster_labels", {})

    scores = {
        k: 0.0 for k in
        ["aggression", "loyalty", "anxiety", "openness", "confidence"]
    }
    total_weight = 0.0

    for cluster_id, weight in affinity.items():
        label = ""
        if isinstance(cluster_labels_map, dict):
            label = cluster_labels_map.get(int(cluster_id), "")
        elif isinstance(cluster_labels_map, list):
            idx = int(cluster_id)
            if idx < len(cluster_labels_map):
                label = cluster_labels_map[idx]

        if label in CLUSTER_SCORE_MAP:
            for key, val in CLUSTER_SCORE_MAP[label].items():
                scores[key] += val * weight
            total_weight += weight

    if total_weight > 0:
        scores = {k: v / total_weight for k, v in scores.items()}
    else:
        scores = {k: 0.5 for k in scores}

    scores["anxiety"] = min(1.0, scores["anxiety"] + variance * 3.0)

    return PersonalityState(**scores)


# ──────────────────────────────────────
# 矛盾なし / 軽微な矛盾の統合
# ──────────────────────────────────────

def fuse_no_contradiction(path_a_state: PersonalityState,
                          path_b_snapshot: dict) -> PersonalityState:
    """矛盾なし: Path A 60% + Path B 40%"""
    path_b_scores = derive_scores_from_path_b(path_b_snapshot)

    return PersonalityState(
        aggression=path_a_state.aggression * 0.6
                   + path_b_scores.aggression * 0.4,
        loyalty=path_a_state.loyalty * 0.6
                + path_b_scores.loyalty * 0.4,
        anxiety=path_a_state.anxiety * 0.6
                + path_b_scores.anxiety * 0.4,
        openness=path_a_state.openness * 0.6
                 + path_b_scores.openness * 0.4,
        confidence=path_a_state.confidence * 0.6
                   + path_b_scores.confidence * 0.4,
        summary=path_a_state.summary,
        update_reason=path_a_state.update_reason,
        chapter_id=path_a_state.chapter_id,
    )


def fuse_soft_contradiction(path_a_state: PersonalityState,
                            path_b_snapshot: dict,
                            contradiction: dict) -> PersonalityState:
    """軽微な矛盾: Path Bの重みを動的に増やす"""
    score = contradiction["contradiction_score"]
    path_b_weight = 0.4 + (score - 0.2) * 1.5
    path_a_weight = 1.0 - path_b_weight
    path_b_scores = derive_scores_from_path_b(path_b_snapshot)

    return PersonalityState(
        aggression=path_a_state.aggression * path_a_weight
                   + path_b_scores.aggression * path_b_weight,
        loyalty=path_a_state.loyalty * path_a_weight
                + path_b_scores.loyalty * path_b_weight,
        anxiety=path_a_state.anxiety * path_a_weight
                + path_b_scores.anxiety * path_b_weight,
        openness=path_a_state.openness * path_a_weight
                 + path_b_scores.openness * path_b_weight,
        confidence=path_a_state.confidence * path_a_weight
                   + path_b_scores.confidence * path_b_weight,
        summary=path_a_state.summary,
        update_reason=(
            f"[軽微矛盾・Path B重み{path_b_weight:.2f}]"
            + path_a_state.update_reason
        ),
        chapter_id=path_a_state.chapter_id,
    )


# ──────────────────────────────────────
# 深刻な矛盾の対処（再生成）
# ──────────────────────────────────────

def diagnose_contradiction(path_a_state: PersonalityState,
                           path_b_snapshot: dict,
                           cluster_labels: list,
                           llm: LLMClient) -> str:
    """矛盾の原因をLLMに診断させる"""
    affinity = path_b_snapshot.get("cluster_affinity", {})
    if affinity:
        dominant_id = max(affinity, key=affinity.get)
        idx = int(dominant_id)
        dominant_label = (
            cluster_labels[idx] if idx < len(cluster_labels) else "不明"
        )
        dominant_aff = affinity[dominant_id]
    else:
        dominant_label = "不明"
        dominant_aff = 0.0

    variance = path_b_snapshot.get("variance", 0.0)

    prompt = (
        "Path A（LLM）とPath B（Embedding）の結果が大きく食い違っています。\n\n"
        f"Path A のサマリー:\n{path_a_state.summary}\n\n"
        f"Path A のスコア:\n"
        f"攻撃性={path_a_state.aggression:.2f}, "
        f"忠誠心={path_a_state.loyalty:.2f}, "
        f"不安={path_a_state.anxiety:.2f}, "
        f"開放性={path_a_state.openness:.2f}, "
        f"自信={path_a_state.confidence:.2f}\n\n"
        f"Path B の統計:\n"
        f"支配的クラスタ: {dominant_label}（親和度{dominant_aff:.2f}）\n"
        f"分散: {variance:.3f}\n\n"
        "食い違いの最も可能性が高い原因を1つ選んでください:\n"
        "A. Path Aが特定の場面に引きずられ全体を見誤っている\n"
        "B. Path Aが知識・情報に基づく解釈を混入させている\n"
        "C. Path Bのクラスタラベルがこの章の文脈に合っていない\n"
        "D. キャラクターがこの章で意図的に普段と異なる振る舞いをしている\n\n"
        "選択肢の記号と、その根拠を1文で返してください。\n"
    )
    return llm.call(prompt)


def regenerate_path_a(path_a_state: PersonalityState,
                      path_b_snapshot: dict,
                      diagnosis: str,
                      prev_state: PersonalityState | None,
                      llm: LLMClient) -> PersonalityState:
    """診断結果を踏まえてPath Aを再生成する"""
    prompt = (
        f"前回の性格分析に以下の問題がありました:\n{diagnosis}\n\n"
        f"前回のサマリー（問題あり）:\n{path_a_state.summary}\n\n"
        f"Embeddingの統計が示す客観的な傾向:\n"
        f"発言の分散: {path_b_snapshot.get('variance', 0):.3f}\n\n"
        f"前章までの確定済みサマリー:\n"
        f"{prev_state.summary if prev_state else '（初章）'}\n\n"
        "上記の問題点を修正し、観察された行動のみに基づいて\n"
        "性格スコアとサマリーを再生成してください。\n\n"
        "特に注意:\n"
        "- キャラクターの知識・情報状態への言及を含めないこと\n"
        "- 今章で観察できない推測を含めないこと\n\n"
        "JSONで返してください。\n"
    )
    result = llm.call_json(prompt)

    return PersonalityState(
        aggression=float(result.get("aggression", path_a_state.aggression)),
        loyalty=float(result.get("loyalty", path_a_state.loyalty)),
        anxiety=float(result.get("anxiety", path_a_state.anxiety)),
        openness=float(result.get("openness", path_a_state.openness)),
        confidence=float(result.get("confidence", path_a_state.confidence)),
        summary=result.get("summary", path_a_state.summary),
        update_reason=result.get("update_reason", "[再生成]"),
        chapter_id=path_a_state.chapter_id,
    )


def fuse_hard_contradiction(path_a_state: PersonalityState,
                            path_b_snapshot: dict,
                            prev_state: PersonalityState | None,
                            contradiction: dict,
                            cluster_labels: list,
                            llm: LLMClient) -> FusionResult:
    """深刻な矛盾: 原因特定 → Path A 再生成 → 再チェック"""
    regeneration_count = 0
    current_path_a = path_a_state

    while regeneration_count < MAX_REGENERATION_ATTEMPTS:
        diagnosis = diagnose_contradiction(
            current_path_a, path_b_snapshot, cluster_labels, llm
        )
        current_path_a = regenerate_path_a(
            current_path_a, path_b_snapshot, diagnosis, prev_state, llm
        )
        regeneration_count += 1

        new_contradiction = compute_path_ab_contradiction(
            current_path_a, path_b_snapshot, cluster_labels
        )
        new_level = classify_contradiction(new_contradiction)

        if new_level != "hard":
            if new_level == "none":
                confirmed = fuse_no_contradiction(
                    current_path_a, path_b_snapshot
                )
            else:
                confirmed = fuse_soft_contradiction(
                    current_path_a, path_b_snapshot, new_contradiction
                )
            return FusionResult(
                confirmed_state=confirmed,
                contradiction_score=new_contradiction["contradiction_score"],
                cosine_similarity=new_contradiction["cosine_similarity"],
                needed_regeneration=True,
                regeneration_count=regeneration_count,
                fusion_method=f"regenerated_{new_level}",
            )

    # 解消不可: Path B 100% 信頼
    path_b_scores = derive_scores_from_path_b(path_b_snapshot)
    path_b_scores.summary = current_path_a.summary
    path_b_scores.update_reason = "[矛盾未解消・Path B優先確定]"
    path_b_scores.chapter_id = current_path_a.chapter_id

    return FusionResult(
        confirmed_state=path_b_scores,
        contradiction_score=contradiction["contradiction_score"],
        cosine_similarity=contradiction["cosine_similarity"],
        needed_regeneration=True,
        regeneration_count=regeneration_count,
        fusion_method="path_b_fallback",
    )


# ──────────────────────────────────────
# 統合エントリポイント
# ──────────────────────────────────────

def detect_and_fuse(path_a_state: PersonalityState,
                    path_b_snapshot: dict,
                    prev_state: PersonalityState | None,
                    cluster_labels: list = None,
                    llm: LLMClient = None) -> FusionResult:
    """矛盾検出・統合のメイン処理"""
    cluster_labels = cluster_labels or []

    contradiction = compute_path_ab_contradiction(
        path_a_state, path_b_snapshot, cluster_labels
    )
    level = classify_contradiction(contradiction)

    if level == "none":
        confirmed = fuse_no_contradiction(path_a_state, path_b_snapshot)
        return FusionResult(
            confirmed_state=confirmed,
            contradiction_score=contradiction["contradiction_score"],
            cosine_similarity=contradiction["cosine_similarity"],
            needed_regeneration=False,
            fusion_method="weighted_average",
        )

    elif level == "soft":
        confirmed = fuse_soft_contradiction(
            path_a_state, path_b_snapshot, contradiction
        )
        return FusionResult(
            confirmed_state=confirmed,
            contradiction_score=contradiction["contradiction_score"],
            cosine_similarity=contradiction["cosine_similarity"],
            needed_regeneration=False,
            fusion_method="path_b_weighted",
        )

    else:
        return fuse_hard_contradiction(
            path_a_state, path_b_snapshot,
            prev_state, contradiction,
            cluster_labels, llm,
        )
