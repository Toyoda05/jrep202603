"""
パイプライン全体の設定値
"""

import os
from pathlib import Path

# ──────────────────────────────────────
# パス設定
# ──────────────────────────────────────

PROJECT_ROOT = Path(__file__).resolve().parent.parent
NOVEL_PATH = PROJECT_ROOT / "novel" / "novel.txt"
OUTPUT_DIR = PROJECT_ROOT / "output"

# ──────────────────────────────────────
# LLM設定
# ──────────────────────────────────────

LLM_MODEL = os.environ.get("LLM_MODEL", "gpt-4o-mini")
LLM_API_KEY = os.environ.get("OPENAI_API_KEY", "")
LLM_BASE_URL = os.environ.get("LLM_BASE_URL", "https://api.openai.com/v1")
LLM_TEMPERATURE = 0.3
LLM_MAX_TOKENS = 4096

# ──────────────────────────────────────
# Embedding設定
# ──────────────────────────────────────

EMBEDDING_MODEL = "intfloat/multilingual-e5-large"
EMBEDDING_DIM = 1024
EMBEDDING_BATCH_SIZE = 64

# ──────────────────────────────────────
# 前処理パラメータ
# ──────────────────────────────────────

MIN_SCENE_LENGTH = 200          # 場面の最小文字数
SPEAKER_CONTEXT_WINDOW = 50     # 話者検索の前後文字数
LLM_CONTEXT_SEGMENTS = 3        # LLM話者同定の前後セグメント数

# ──────────────────────────────────────
# クラスタリング
# ──────────────────────────────────────

MIN_UTTERANCES_FOR_CLUSTERING = 10
MAX_CLUSTER_K = 5

# ──────────────────────────────────────
# 知識DB
# ──────────────────────────────────────

CONFIDENCE_THRESHOLD = 0.7      # 遡及修正の信頼度閾値

# ──────────────────────────────────────
# Path A（性格履歴）
# ──────────────────────────────────────

MAX_PERSONALITY_DRIFT = 0.15    # 章ごとの性格スコア変化上限

# ──────────────────────────────────────
# 矛盾検出・統合
# ──────────────────────────────────────

CONTRADICTION_NONE_THRESHOLD = 0.2
CONTRADICTION_SOFT_THRESHOLD = 0.4
MAX_REGENERATION_ATTEMPTS = 2

# ──────────────────────────────────────
# Path Bスコアマッピング
# ──────────────────────────────────────

CLUSTER_SCORE_MAP = {
    "激情モード": {"aggression": 0.8, "anxiety": 0.6,
                    "confidence": 0.7, "openness": 0.4, "loyalty": 0.5},
    "内省モード": {"aggression": 0.2, "anxiety": 0.7,
                    "confidence": 0.3, "openness": 0.7, "loyalty": 0.6},
    "信頼モード": {"aggression": 0.2, "anxiety": 0.2,
                    "confidence": 0.7, "openness": 0.7, "loyalty": 0.9},
    "攻撃モード": {"aggression": 0.9, "anxiety": 0.4,
                    "confidence": 0.8, "openness": 0.2, "loyalty": 0.3},
}

# ──────────────────────────────────────
# 抑制重み（検出パターン別）
# ──────────────────────────────────────

SUPPRESSION_WEIGHTS = {
    "A": 0.7,   # 明示的告白 → 強く抑制
    "B": 0.4,   # 行動矛盾   → 中程度抑制
    "C": 0.3,   # 第三者証言 → 弱めに抑制
}

# ──────────────────────────────────────
# 感情→音声マッピング
# ──────────────────────────────────────

EMOTION_VOICE_MAP = {
    "anger":       {"rate_delta": +0.15, "pitch_delta": +0.05,
                    "variation_delta": -0.10, "energy_delta": +0.25},
    "sadness":     {"rate_delta": -0.15, "pitch_delta": -0.05,
                    "variation_delta": -0.15, "energy_delta": -0.20},
    "fear":        {"rate_delta": +0.10, "pitch_delta": +0.08,
                    "variation_delta": +0.05, "energy_delta": -0.10},
    "joy":         {"rate_delta": +0.05, "pitch_delta": +0.08,
                    "variation_delta": +0.20, "energy_delta": +0.15},
    "tension":     {"rate_delta": +0.08, "pitch_delta": +0.03,
                    "variation_delta": -0.20, "energy_delta": +0.05},
    "resignation": {"rate_delta": -0.10, "pitch_delta": -0.08,
                    "variation_delta": -0.20, "energy_delta": -0.25},
    "disgust":     {"rate_delta": -0.05, "pitch_delta": -0.03,
                    "variation_delta": -0.05, "energy_delta": +0.05},
    "surprise":    {"rate_delta": +0.00, "pitch_delta": +0.10,
                    "variation_delta": +0.25, "energy_delta": +0.10},
}

# ──────────────────────────────────────
# Style-BERT-VITS2 スタイルマッピング
# ──────────────────────────────────────

EMOTION_STYLE_MAP = {
    "anger":       "Angry",
    "sadness":     "Sad",
    "fear":        "Fearful",
    "joy":         "Happy",
    "tension":     "Nervous",
    "resignation": "Sad",
    "disgust":     "Disgusted",
    "surprise":    "Surprised",
    "neutral":     "Neutral",
}
