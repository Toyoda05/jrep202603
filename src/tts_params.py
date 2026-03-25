"""
工程 10: セリフごとのTTSパラメタ推定（3層統合）

性格層 × 心情層 × 知識制御 → TTSパラメタ
"""

import logging
import numpy as np

from src.models import PersonalityState, EmotionalState, TTSParams
from src.config import EMOTION_VOICE_MAP
from src.llm_client import LLMClient

logger = logging.getLogger(__name__)


# ──────────────────────────────────────
# Step 1: セリフの文脈分析
# ──────────────────────────────────────

def analyze_utterance_context(utterance: dict,
                              surrounding_segs: list[dict],
                              llm: LLMClient) -> dict:
    """セリフ単体の文脈特徴を分析する"""
    context_text = "\n".join(
        f"[{s['type']}] {s.get('speaker', '')}: {s['text']}"
        for s in surrounding_segs
    )

    prompt = (
        "以下の文脈の中のセリフを分析してください。\n\n"
        f"文脈:\n{context_text}\n\n"
        f"対象セリフ: 「{utterance['text']}」\n\n"
        "以下をJSONで返してください:\n"
        "{\n"
        '  "utterance_type": "断言/疑問/命令/懇願/独白/皮肉",\n'
        '  "addressee": "誰に向けて言っているか（独白の場合はnull）",\n'
        '  "is_interrupted": false,\n'
        '  "has_subtext": false,\n'
        '  "emphasis_words": ["強調すべき単語のリスト"],\n'
        '  "natural_pause_positions": []\n'
        "}\n"
    )
    result = llm.call_json(prompt)
    if not result:
        result = {
            "utterance_type": "断言",
            "addressee": None,
            "is_interrupted": False,
            "has_subtext": False,
            "emphasis_words": [],
            "natural_pause_positions": [],
        }
    return result


# ──────────────────────────────────────
# Step 2: 3層の重み計算
# ──────────────────────────────────────

def compute_layer_weights(utterance_context: dict,
                          emotion: EmotionalState,
                          knowledge_control: dict) -> dict:
    """セリフの性質に応じて3層の影響度を動的に決定する"""
    weights = {
        "personality": 0.3,
        "emotion": 0.5,
        "knowledge": 0.2,
    }

    max_suppression = max(
        (c.get("suppression_weight", 0.0) for c in knowledge_control.values()
         if c.get("hiding_knowledge")),
        default=0.0,
    ) if knowledge_control else 0.0

    if max_suppression > 0.5:
        weights["knowledge"] += 0.15
        weights["emotion"] -= 0.10
        weights["personality"] -= 0.05

    max_emotion = max(
        emotion.anger, emotion.sadness, emotion.fear,
        emotion.joy, emotion.tension,
    )
    if max_emotion > 0.8 and not emotion.suppressed:
        weights["emotion"] += 0.15
        weights["personality"] -= 0.10
        weights["knowledge"] -= 0.05

    if utterance_context.get("addressee") is None:
        weights["personality"] += 0.10
        weights["emotion"] -= 0.10

    total = sum(weights.values())
    return {k: v / total for k, v in weights.items()}


# ──────────────────────────────────────
# Step 3: 性格層からの音声パラメタ
# ──────────────────────────────────────

def derive_params_from_personality(personality: PersonalityState,
                                   utterance_context: dict) -> dict:
    """性格層 → 話し方の癖・ベースライン"""
    if personality.aggression > 0.7:
        speaking_style = "強め・短文傾向"
    elif personality.openness > 0.7:
        speaking_style = "表現豊か・抑揚あり"
    elif personality.anxiety > 0.7:
        speaking_style = "不安定・語尾が弱め"
    else:
        speaking_style = "標準"

    rate_adj = (
        1.05 if utterance_context.get("utterance_type") == "命令" else 1.0
    )

    return {
        "base_speech_rate": (
            0.9 + personality.aggression * 0.15
            + personality.confidence * 0.10
        ) * rate_adj,
        "base_pitch_scale": (
            1.0 - personality.anxiety * 0.08
            + personality.confidence * 0.05
        ),
        "base_pitch_variation": 0.3 + personality.openness * 0.4,
        "base_breathiness": personality.anxiety * 0.3,
        "speaking_style": speaking_style,
    }


# ──────────────────────────────────────
# Step 4: 心情層からの音声パラメタ
# ──────────────────────────────────────

def derive_params_from_emotion(emotion: EmotionalState,
                               utterance_context: dict) -> dict:
    """心情層 → 今この瞬間の感情の出方"""
    emotion_vals = {
        "anger": emotion.anger,
        "sadness": emotion.sadness,
        "fear": emotion.fear,
        "joy": emotion.joy,
        "tension": emotion.tension,
        "resignation": emotion.resignation,
        "disgust": emotion.disgust,
        "surprise": emotion.surprise,
    }
    sorted_emotions = sorted(
        emotion_vals.items(), key=lambda x: x[1], reverse=True
    )
    dominant = sorted_emotions[0]
    secondary = (
        sorted_emotions[1]
        if sorted_emotions[1][1] > 0.2
        else ("neutral", 0.0)
    )

    zero_map = {"rate_delta": 0.0, "pitch_delta": 0.0,
                "variation_delta": 0.0, "energy_delta": 0.0}
    dom_map = EMOTION_VOICE_MAP.get(dominant[0], zero_map)
    sec_map = EMOTION_VOICE_MAP.get(secondary[0], zero_map)

    blended = {
        k: dom_map.get(k, 0.0) * dominant[1] * 0.7
        + sec_map.get(k, 0.0) * secondary[1] * 0.3
        for k in dom_map
    }

    pause_before = 0
    if emotion.sadness > 0.6 or emotion.resignation > 0.5:
        pause_before += 300
    if utterance_context.get("has_subtext"):
        pause_before += 200

    return {
        "dominant_emotion": dominant[0],
        "dominant_intensity": dominant[1],
        "secondary_emotion": secondary[0],
        "secondary_intensity": secondary[1],
        "rate_delta": blended["rate_delta"],
        "pitch_delta": blended["pitch_delta"],
        "variation_delta": blended["variation_delta"],
        "energy_delta": blended["energy_delta"],
        "pause_before_ms": pause_before,
    }


# ──────────────────────────────────────
# Step 5: 知識制御レイヤーの適用
# ──────────────────────────────────────

def apply_knowledge_control_layer(params: dict,
                                  knowledge_control: dict,
                                  emotion: EmotionalState) -> dict:
    """知識制御は表に出る量のみを制限する"""
    if not knowledge_control:
        return params

    hiding = any(
        c.get("hiding_knowledge") for c in knowledge_control.values()
    )
    if not hiding:
        return params

    sw = emotion.suppression_weight

    params["emotion_intensity"] *= (1.0 - sw * 0.8)
    params["pitch_variation"] *= (1.0 - sw * 0.6)
    params["speech_rate"] *= (1.0 + sw * 0.08)
    params["breathiness"] = min(
        1.0, params.get("breathiness", 0.0) + sw * 0.4
    )
    params["pause_before_ms"] += int(sw * 400)
    params["hiding_knowledge"] = True
    params["suppression_weight"] = sw

    return params


# ──────────────────────────────────────
# Step 6: 3層の統合
# ──────────────────────────────────────

def integrate_three_layers(utterance: dict,
                           personality: PersonalityState,
                           emotion: EmotionalState,
                           knowledge_control: dict,
                           surrounding_segs: list[dict],
                           llm: LLMClient) -> TTSParams:
    """3層統合のメインエントリポイント"""
    context = analyze_utterance_context(utterance, surrounding_segs, llm)

    p_params = derive_params_from_personality(personality, context)
    e_params = derive_params_from_emotion(emotion, context)
    weights = compute_layer_weights(context, emotion, knowledge_control)

    raw_params = {
        "emotion": e_params["dominant_emotion"],
        "emotion_intensity": e_params["dominant_intensity"],
        "secondary_emotion": e_params["secondary_emotion"],
        "secondary_intensity": e_params["secondary_intensity"],

        "speech_rate": (
            p_params["base_speech_rate"]
            + e_params["rate_delta"] * weights["emotion"]
        ),
        "pitch_scale": (
            p_params["base_pitch_scale"]
            + e_params["pitch_delta"] * weights["emotion"]
        ),
        "pitch_variation": (
            p_params["base_pitch_variation"]
            + e_params["variation_delta"] * weights["emotion"]
        ),
        "energy": 0.5 + e_params["energy_delta"] * weights["emotion"],

        "breathiness": p_params["base_breathiness"],
        "speaking_style": p_params["speaking_style"],
        "pause_before_ms": e_params["pause_before_ms"],
        "pause_after_ms": 0,
        "hiding_knowledge": False,
        "suppression_weight": 0.0,
        "dominant_layer": max(weights, key=weights.get),
    }

    # 知識制御を最後に適用
    final_params = apply_knowledge_control_layer(
        raw_params, knowledge_control, emotion
    )

    # 値域クランプ
    final_params["speech_rate"] = float(
        np.clip(final_params["speech_rate"], 0.6, 1.6)
    )
    final_params["pitch_scale"] = float(
        np.clip(final_params["pitch_scale"], 0.7, 1.4)
    )
    final_params["pitch_variation"] = float(
        np.clip(final_params["pitch_variation"], 0.0, 1.0)
    )
    final_params["energy"] = float(
        np.clip(final_params["energy"], 0.1, 1.0)
    )

    return TTSParams(
        **final_params,
        chapter_id=personality.chapter_id,
        scene_id=utterance.get("scene_id", ""),
    )
