"""
工程 9: 心情層の更新（場面単位・第2パス）

性格層が「ありうる感情の幅」を決め、
心情層が「今この瞬間の実値」を決める。
"""

import logging
import numpy as np

from src.models import PersonalityState, EmotionalState
from src.llm_client import LLMClient

logger = logging.getLogger(__name__)


# ──────────────────────────────────────
# Step 1: 性格層による感情の許容範囲
# ──────────────────────────────────────

def compute_emotion_bounds(personality: PersonalityState) -> dict:
    """性格層から各感情の「ありうる範囲」を算出する"""
    amplitude = 0.4 + personality.openness * 0.6

    bounds = {
        "anger": (
            personality.aggression * 0.1,
            min(1.0, personality.aggression * amplitude * 1.3),
        ),
        "sadness": (
            0.0,
            min(1.0, (1.0 - personality.confidence) * amplitude * 1.2),
        ),
        "fear": (
            0.0,
            min(1.0, personality.anxiety * amplitude * 1.1),
        ),
        "joy": (
            0.0,
            min(1.0, (1.0 - personality.anxiety * 0.5) * amplitude),
        ),
        "disgust": (
            0.0,
            min(1.0, personality.aggression * amplitude * 0.9),
        ),
        "surprise": (
            0.0,
            amplitude,
        ),
        "tension": (
            personality.anxiety * 0.1,
            min(1.0, personality.anxiety * amplitude * 1.2),
        ),
        "resignation": (
            0.0,
            min(1.0, (1.0 - personality.confidence)
                * (1.0 - personality.loyalty * 0.3) * amplitude),
        ),
    }
    return bounds


# ──────────────────────────────────────
# Step 2: 場面テキストからの感情推定
# ──────────────────────────────────────

def emotion_to_text(emotion: EmotionalState) -> str:
    """感情状態を簡潔なテキストに変換する"""
    items = []
    for name, val in [
        ("怒り", emotion.anger), ("悲しみ", emotion.sadness),
        ("恐怖", emotion.fear), ("喜び", emotion.joy),
        ("嫌悪", emotion.disgust), ("驚き", emotion.surprise),
        ("緊張", emotion.tension), ("諦め", emotion.resignation),
    ]:
        if val > 0.1:
            items.append(f"{name}={val:.2f}")
    return ", ".join(items) if items else "中性"


def estimate_raw_emotion_from_scene(scene_segments: list[dict],
                                    char: str,
                                    prev_emotion: EmotionalState,
                                    llm: LLMClient) -> EmotionalState:
    """場面のセリフ・描写から感情の生の値を推定する"""
    char_segs = [
        seg for seg in scene_segments
        if seg.get("speaker") == char
        or (seg["type"] == "narration" and char in seg["text"])
    ]
    if not char_segs:
        return decay_emotion(prev_emotion, decay_rate=0.3)

    seg_text = "\n".join(
        f"[{'セリフ' if s['type'] == 'dialogue' else '描写'}] {s['text']}"
        for s in char_segs
    )
    prev_summary = emotion_to_text(prev_emotion)

    prompt = (
        f"キャラクター「{char}」の直前の感情状態:\n{prev_summary}\n\n"
        f"今場面のセリフ・描写:\n{seg_text}\n\n"
        f"この場面での「{char}」の感情状態を推定してください。\n\n"
        "ルール:\n"
        "- セリフ・描写から直接読み取れる感情のみを評価する\n"
        "- 「なぜそう感じるか」の背景知識には言及しない\n"
        "- 感情を表に出しているか、抑えているかも判定する\n\n"
        "JSON形式で返してください:\n"
        "{\n"
        '  "anger": 0.0〜1.0,\n'
        '  "sadness": 0.0〜1.0,\n'
        '  "fear": 0.0〜1.0,\n'
        '  "joy": 0.0〜1.0,\n'
        '  "disgust": 0.0〜1.0,\n'
        '  "surprise": 0.0〜1.0,\n'
        '  "tension": 0.0〜1.0,\n'
        '  "resignation": 0.0〜1.0,\n'
        '  "suppressed": true/false,\n'
        '  "trigger": "感情変化の原因となった出来事を1文で"\n'
        "}\n"
    )
    result = llm.call_json(prompt)

    return EmotionalState(
        anger=float(result.get("anger", 0.0)),
        sadness=float(result.get("sadness", 0.0)),
        fear=float(result.get("fear", 0.0)),
        joy=float(result.get("joy", 0.0)),
        disgust=float(result.get("disgust", 0.0)),
        surprise=float(result.get("surprise", 0.0)),
        tension=float(result.get("tension", 0.0)),
        resignation=float(result.get("resignation", 0.0)),
        suppressed=bool(result.get("suppressed", False)),
        trigger=result.get("trigger", ""),
        scene_id=scene_segments[0]["scene_id"] if scene_segments else "",
    )


# ──────────────────────────────────────
# Step 3: 性格層によるクリッピング
# ──────────────────────────────────────

def apply_personality_constraint(raw_emotion: EmotionalState,
                                 personality: PersonalityState) -> EmotionalState:
    """性格層の許容範囲に感情値をクリッピングする"""
    bounds = compute_emotion_bounds(personality)

    def clip_val(value, field_name):
        lo, hi = bounds.get(field_name, (0.0, 1.0))
        return float(np.clip(value, lo, hi))

    return EmotionalState(
        anger=clip_val(raw_emotion.anger, "anger"),
        sadness=clip_val(raw_emotion.sadness, "sadness"),
        fear=clip_val(raw_emotion.fear, "fear"),
        joy=clip_val(raw_emotion.joy, "joy"),
        disgust=clip_val(raw_emotion.disgust, "disgust"),
        surprise=clip_val(raw_emotion.surprise, "surprise"),
        tension=clip_val(raw_emotion.tension, "tension"),
        resignation=clip_val(raw_emotion.resignation, "resignation"),
        suppressed=raw_emotion.suppressed,
        suppression_weight=raw_emotion.suppression_weight,
        trigger=raw_emotion.trigger,
        scene_id=raw_emotion.scene_id,
    )


# ──────────────────────────────────────
# Step 4: 感情の減衰処理
# ──────────────────────────────────────

def decay_emotion(emotion: EmotionalState,
                  decay_rate: float = 0.3,
                  personality: PersonalityState = None) -> EmotionalState:
    """感情を中性値に向けて減衰させる"""
    if personality:
        anger_decay = decay_rate * (1.0 - personality.aggression * 0.5)
        default_decay = decay_rate * (1.0 - personality.openness * 0.3)
    else:
        anger_decay = default_decay = decay_rate

    def decay(val, rate, neutral=0.0):
        return val + (neutral - val) * rate

    return EmotionalState(
        anger=decay(emotion.anger, anger_decay),
        sadness=decay(emotion.sadness, default_decay),
        fear=decay(emotion.fear, default_decay),
        joy=decay(emotion.joy, default_decay),
        disgust=decay(emotion.disgust, default_decay),
        surprise=decay(emotion.surprise, default_decay * 1.5),
        tension=decay(emotion.tension, anger_decay),
        resignation=decay(emotion.resignation, default_decay * 0.7),
        suppressed=emotion.suppressed,
        suppression_weight=emotion.suppression_weight * (1.0 - default_decay),
        trigger="",
        scene_id=emotion.scene_id,
    )


# ──────────────────────────────────────
# Step 5: 知識制御との接合
# ──────────────────────────────────────

def apply_knowledge_suppression_to_emotion(
        emotion: EmotionalState,
        knowledge_control: dict) -> EmotionalState:
    """知識隠蔽フラグを心情層の抑制重みに反映する"""
    if not knowledge_control:
        return emotion

    hiding_weights = [
        c["suppression_weight"]
        for c in knowledge_control.values()
        if c.get("hiding_knowledge")
    ]
    if not hiding_weights:
        return emotion

    max_suppression = max(hiding_weights)
    if max_suppression > 0:
        emotion.suppressed = True
        emotion.suppression_weight = max(
            emotion.suppression_weight, max_suppression
        )

    return emotion


# ──────────────────────────────────────
# 統合エントリポイント
# ──────────────────────────────────────

def update_emotional_state(char: str,
                           scene_segments: list[dict],
                           prev_emotion: EmotionalState,
                           personality: PersonalityState,
                           knowledge_control: dict,
                           llm: LLMClient) -> EmotionalState:
    """心情層更新のメイン処理"""
    # Step 2: 生の感情を推定
    raw_emotion = estimate_raw_emotion_from_scene(
        scene_segments, char, prev_emotion, llm
    )

    # Step 3: 性格層による制約を適用
    constrained = apply_personality_constraint(raw_emotion, personality)

    # Step 5: 知識制御による抑制を適用
    final_emotion = apply_knowledge_suppression_to_emotion(
        constrained, knowledge_control
    )

    return final_emotion
