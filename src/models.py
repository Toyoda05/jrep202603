"""
共通データクラス定義

パイプライン全体で使用される構造化データ型を一元管理する。
"""

from dataclasses import dataclass, field
from typing import Optional
import numpy as np


# ──────────────────────────────────────
# 工程 3: 事象マスタ
# ──────────────────────────────────────

@dataclass
class Event:
    """物語内で明らかになった重要な事実"""
    event_id: str
    description: str          # 何が明らかになったか
    scene_id: str             # どの場面で発生したか
    chapter_id: str


@dataclass
class KnowledgeEntry:
    """キャラクター × 事象の知識状態"""
    known: bool
    learned_at: Optional[str] = None          # chapter_id
    evidence: Optional[str] = None            # 根拠となったセリフ・描写
    retroactive: bool = False                 # 遡及修正されたか
    originally_detected_at: Optional[str] = None  # 遡及前の検出章


# ──────────────────────────────────────
# 工程 6: Path A 性格状態
# ──────────────────────────────────────

@dataclass
class PersonalityState:
    """キャラクターの性格パラメタ（0.0〜1.0）"""
    aggression: float = 0.5
    loyalty: float = 0.5
    anxiety: float = 0.5
    openness: float = 0.5
    confidence: float = 0.5

    # LLMサマリー
    summary: str = ""

    # 更新の根拠（評価用）
    update_reason: str = ""
    chapter_id: str = ""


# ──────────────────────────────────────
# 工程 7: Path B 状態
# ──────────────────────────────────────

@dataclass
class PathBState:
    """Embedding逐次更新用の累積状態"""
    n: int = 0                                # セリフ総数
    centroid: np.ndarray = None               # 現在の重心ベクトル
    M2: np.ndarray = None                     # 分散計算用累積値

    # 章ごとのスナップショット
    chapter_snapshots: dict = field(default_factory=dict)

    # 全文スキャンで確定したクラスタ構造（変更不可）
    cluster_centroids: list = None
    cluster_labels_text: list = None

    def __post_init__(self):
        if self.centroid is None:
            self.centroid = np.zeros(1024)
        if self.M2 is None:
            self.M2 = np.zeros(1024)


# ──────────────────────────────────────
# 工程 8: 矛盾検出・統合結果
# ──────────────────────────────────────

@dataclass
class FusionResult:
    """Path A/B 統合結果"""
    confirmed_state: PersonalityState

    # 矛盾検出の記録（論文評価用）
    contradiction_score: float = 0.0
    cosine_similarity: float = 1.0
    needed_regeneration: bool = False
    regeneration_count: int = 0
    fusion_method: str = ""


# ──────────────────────────────────────
# 工程 9: 心情層
# ──────────────────────────────────────

@dataclass
class EmotionalState:
    """場面・セリフ単位の感情状態"""
    # 基本感情（0.0〜1.0）
    anger: float = 0.0
    sadness: float = 0.0
    fear: float = 0.0
    joy: float = 0.0
    disgust: float = 0.0
    surprise: float = 0.0

    # 複合状態
    tension: float = 0.0
    resignation: float = 0.0

    # 感情変化のトリガー
    trigger: str = ""
    scene_id: str = ""

    # 抑制フラグ（知識制御との接続点）
    suppressed: bool = False
    suppression_weight: float = 0.0


# ──────────────────────────────────────
# 工程 10: TTSパラメタ
# ──────────────────────────────────────

@dataclass
class TTSParams:
    """セリフごとの最終音声パラメタ"""
    # 感情制御
    emotion: str = "neutral"
    emotion_intensity: float = 0.0
    secondary_emotion: str = "neutral"
    secondary_intensity: float = 0.0

    # 音声パラメタ
    speech_rate: float = 1.0
    pitch_scale: float = 1.0
    pitch_variation: float = 0.3
    energy: float = 0.5
    pause_before_ms: int = 0
    pause_after_ms: int = 0

    # 語調制御
    speaking_style: str = "標準"
    breathiness: float = 0.0

    # 知識制御
    hiding_knowledge: bool = False
    suppression_weight: float = 0.0

    # メタデータ
    dominant_layer: str = "emotion"
    chapter_id: str = ""
    scene_id: str = ""
