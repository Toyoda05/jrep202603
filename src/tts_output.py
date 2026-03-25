"""
工程 11: TTSパラメタ出力（JSON）

構造化JSON出力、Style-BERT-VITS2変換ロジック、バッチ出力。
"""

import json
import logging
from dataclasses import asdict
from pathlib import Path

from src.models import PersonalityState, EmotionalState, TTSParams
from src.config import EMOTION_STYLE_MAP, OUTPUT_DIR

logger = logging.getLogger(__name__)


# ──────────────────────────────────────
# 1セリフ分の完全JSON構築
# ──────────────────────────────────────

def build_utterance_json(utterance: dict,
                         personality: PersonalityState,
                         emotion: EmotionalState,
                         tts: TTSParams,
                         fusion_meta: dict = None,
                         utterance_id: str = "") -> dict:
    """1セリフ分の完全なJSONオブジェクトを構築する"""
    return {
        "$schema": "audiobook_tts_params_v1",
        "meta": {
            "character": utterance.get("speaker", ""),
            "chapter_id": tts.chapter_id,
            "scene_id": tts.scene_id,
            "utterance_id": utterance_id,
            "text": utterance.get("text", ""),
        },
        "personality_layer": {
            "aggression": round(personality.aggression, 3),
            "loyalty": round(personality.loyalty, 3),
            "anxiety": round(personality.anxiety, 3),
            "openness": round(personality.openness, 3),
            "confidence": round(personality.confidence, 3),
            "summary": personality.summary,
            "chapter_confirmed_at": personality.chapter_id,
        },
        "emotional_layer": {
            "dominant_emotion": tts.emotion,
            "dominant_intensity": round(tts.emotion_intensity, 3),
            "secondary_emotion": tts.secondary_emotion,
            "secondary_intensity": round(tts.secondary_intensity, 3),
            "suppressed": emotion.suppressed,
            "trigger": emotion.trigger,
            "scene_confirmed_at": tts.scene_id,
        },
        "knowledge_control": utterance.get("knowledge_control", {}),
        "tts_params": {
            "emotion": {
                "label": tts.emotion,
                "intensity": round(tts.emotion_intensity, 3),
                "secondary_label": tts.secondary_emotion,
                "secondary_intensity": round(tts.secondary_intensity, 3),
            },
            "prosody": {
                "speech_rate": round(tts.speech_rate, 3),
                "pitch_scale": round(tts.pitch_scale, 3),
                "pitch_variation": round(tts.pitch_variation, 3),
                "energy": round(tts.energy, 3),
            },
            "voice_quality": {
                "breathiness": round(tts.breathiness, 3),
                "speaking_style": tts.speaking_style,
            },
            "timing": {
                "pause_before_ms": tts.pause_before_ms,
                "pause_after_ms": tts.pause_after_ms,
            },
            "knowledge_suppression": {
                "hiding_knowledge": tts.hiding_knowledge,
                "suppression_weight": round(tts.suppression_weight, 3),
            },
        },
        "fusion_meta": fusion_meta or {
            "dominant_layer": tts.dominant_layer,
        },
    }


# ──────────────────────────────────────
# 場面バッチ出力
# ──────────────────────────────────────

def build_scene_batch(scene_id: str,
                      character: str,
                      utterance_jsons: list[dict]) -> dict:
    """場面ごとのバッチ出力を構築する"""
    return {
        "scene_id": scene_id,
        "character": character,
        "utterances": utterance_jsons,
    }


# ──────────────────────────────────────
# Style-BERT-VITS2 変換
# ──────────────────────────────────────

def to_style_bert_vits2(tts_json: dict, character: str) -> dict:
    """TTSパラメタをStyle-BERT-VITS2の入力形式に変換する"""
    p = tts_json.get("tts_params", {})

    emotion_block = p.get("emotion", {})
    dominant = emotion_block.get("label", "neutral")
    secondary = emotion_block.get("secondary_label", "neutral")
    d_intensity = emotion_block.get("intensity", 0.0)
    s_intensity = emotion_block.get("secondary_intensity", 0.0)

    style_weight = {
        EMOTION_STYLE_MAP.get(dominant, "Neutral"): d_intensity * 0.7,
        EMOTION_STYLE_MAP.get(secondary, "Neutral"): s_intensity * 0.3,
    }

    ks = p.get("knowledge_suppression", {})
    if ks.get("hiding_knowledge"):
        sw = ks.get("suppression_weight", 0.0)
        for key in list(style_weight.keys()):
            style_weight[key] *= (1.0 - sw * 0.6)
        style_weight["Neutral"] = style_weight.get("Neutral", 0.0) + sw * 0.4

    prosody = p.get("prosody", {})
    voice_q = p.get("voice_quality", {})
    timing = p.get("timing", {})

    return {
        "text": tts_json.get("meta", {}).get("text", ""),
        "speaker": character,
        "style_weight": style_weight,
        "speed": prosody.get("speech_rate", 1.0),
        "pitch": prosody.get("pitch_scale", 1.0),
        "intonation_scale": prosody.get("pitch_variation", 0.3),
        "volume": prosody.get("energy", 0.5),
        "noise_scale": voice_q.get("breathiness", 0.0),
        "pre_phoneme_length": timing.get("pause_before_ms", 0) / 1000.0,
        "post_phoneme_length": timing.get("pause_after_ms", 0) / 1000.0,
    }


# ──────────────────────────────────────
# ファイル出力
# ──────────────────────────────────────

def save_output(data: dict | list, filename: str,
                output_dir: Path = None) -> Path:
    """JSONデータをファイルに保存する"""
    out_dir = output_dir or OUTPUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    filepath = out_dir / filename
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2, default=str)
    logger.info("出力保存: %s", filepath)
    return filepath


def save_full_output(all_utterance_jsons: list[dict],
                     personality_history: dict,
                     knowledge_db: dict,
                     output_dir: Path = None) -> dict:
    """全結果を出力ファイルに保存する"""
    out_dir = output_dir or OUTPUT_DIR

    # メイン出力: 全セリフのTTSパラメタ
    tts_path = save_output(all_utterance_jsons, "tts_params.json", out_dir)

    # Style-BERT-VITS2 変換版
    sbv2_outputs = []
    for utt_json in all_utterance_jsons:
        char = utt_json.get("meta", {}).get("character", "")
        sbv2_outputs.append(to_style_bert_vits2(utt_json, char))
    sbv2_path = save_output(sbv2_outputs, "style_bert_vits2.json", out_dir)

    # 性格履歴の出力
    personality_out = {}
    for char, chapters in personality_history.items():
        personality_out[char] = {}
        for ch_id, entry in chapters.items():
            if isinstance(entry, dict):
                state = entry.get("state")
                meta = entry.get("meta", {})
            else:
                state = entry
                meta = {}
            if state:
                personality_out[char][ch_id] = {
                    "state": asdict(state) if hasattr(state, '__dataclass_fields__') else str(state),
                    "meta": meta,
                }
    pers_path = save_output(personality_out, "personality_history.json", out_dir)

    # 知識DBの出力
    knowledge_out = {}
    for char, events in knowledge_db.items():
        knowledge_out[char] = {}
        for eid, entry in events.items():
            if hasattr(entry, '__dataclass_fields__'):
                knowledge_out[char][eid] = asdict(entry)
            else:
                knowledge_out[char][eid] = str(entry)
    kb_path = save_output(knowledge_out, "knowledge_db.json", out_dir)

    return {
        "tts_params": str(tts_path),
        "style_bert_vits2": str(sbv2_path),
        "personality_history": str(pers_path),
        "knowledge_db": str(kb_path),
    }
