"""
工程 6: Path A — LLM要約（性格履歴ループ）

各キャラクターの性格を章ごとに逐次更新する。
3段階プロンプト: 観察事実抽出 → 変化差分抽出 → スコア更新
"""

import logging
import numpy as np

from src.models import PersonalityState
from src.config import MAX_PERSONALITY_DRIFT
from src.llm_client import LLMClient

logger = logging.getLogger(__name__)


# ──────────────────────────────────────
# 第1段階: 観察可能な行動・発言の抽出
# ──────────────────────────────────────

def extract_observable_behaviors(char: str,
                                segments: list[dict],
                                llm: LLMClient) -> str:
    """テキストから直接読み取れる行動・発言の事実のみを抽出する"""
    char_segments = [
        seg for seg in segments
        if seg.get("speaker") == char
        or (seg["type"] == "narration" and char in seg["text"])
    ]
    if not char_segments:
        return ""

    seg_text = "\n".join(
        f"[{'セリフ' if s['type'] == 'dialogue' else '描写'}] {s['text']}"
        for s in char_segments
    )

    prompt = (
        f"以下はキャラクター「{char}」に関するこの章のセリフと描写です。\n\n"
        f"{seg_text}\n\n"
        "以下のルールに従って、観察可能な事実のみを箇条書きで抽出してください。\n\n"
        "ルール:\n"
        "- テキストに明示されている行動・発言のみを書く\n"
        "- 「なぜそうしたか」の推測は書かない\n"
        "- 知識・情報の有無に関する記述は含めない\n"
        "- 他キャラクターへの態度・反応を具体的に記述する\n\n"
        "出力形式:\n"
        "- （行動・発言の事実を1つずつ記述）\n"
    )
    return llm.call(prompt)


# ──────────────────────────────────────
# 第2段階: 性格変化の差分抽出
# ──────────────────────────────────────

def extract_personality_delta(char: str,
                              behaviors: str,
                              prev_state: PersonalityState | None,
                              llm: LLMClient) -> str:
    """観察された行動から前章との性格変化の差分を抽出する"""
    prev_summary = prev_state.summary if prev_state else "（初登場・情報なし）"

    prompt = (
        f"キャラクター「{char}」について分析します。\n\n"
        f"【前章までの性格サマリー】\n{prev_summary}\n\n"
        f"【今章での観察された行動・発言】\n{behaviors}\n\n"
        "前章と比較して、以下の観点で「変化があった点のみ」を\n"
        "記述してください。変化がない項目は記述不要です。\n\n"
        "分析観点:\n"
        "1. 他者への態度・接し方の変化\n"
        "2. 感情表現の変化\n"
        "3. 意思決定のパターンの変化\n"
        "4. 対人スタンスの変化\n\n"
        "重要: 変化の根拠として「どの行動・発言から読み取れるか」を\n"
        "必ず添えてください。\n"
    )
    return llm.call(prompt)


# ──────────────────────────────────────
# 第3段階: パラメタ数値の更新
# ──────────────────────────────────────

def update_personality_scores(char: str,
                              delta_text: str,
                              prev_state: PersonalityState | None,
                              chapter_id: str,
                              llm: LLMClient) -> PersonalityState:
    """差分テキストを受けて数値パラメタを更新する"""
    prev_scores = {
        "aggression":  prev_state.aggression if prev_state else 0.5,
        "loyalty":     prev_state.loyalty    if prev_state else 0.5,
        "anxiety":     prev_state.anxiety    if prev_state else 0.5,
        "openness":    prev_state.openness   if prev_state else 0.5,
        "confidence":  prev_state.confidence if prev_state else 0.5,
    }

    prompt = (
        f"キャラクター「{char}」の今章での性格変化:\n{delta_text}\n\n"
        f"前章末時点の性格スコア（0.0〜1.0）:\n{prev_scores}\n\n"
        "今章末時点の新しいスコアをJSONで返してください。\n\n"
        "制約:\n"
        f"- 各スコアは前章の値から±{MAX_PERSONALITY_DRIFT}以内の変化に留めること\n"
        "- 変化の根拠が明確な項目のみ変動させること\n"
        "- 根拠がない項目は前章の値をそのまま使うこと\n\n"
        "出力形式:\n"
        "{\n"
        '  "aggression":  数値,\n'
        '  "loyalty":     数値,\n'
        '  "anxiety":     数値,\n'
        '  "openness":    数値,\n'
        '  "confidence":  数値,\n'
        '  "update_reason": "変化した項目とその根拠を1〜2文で"\n'
        "}\n"
    )
    result = llm.call_json(prompt)

    # ±MAX_DRIFT 制限をコードでも保証（二重保証）
    clamped = {}
    for key in prev_scores:
        raw = result.get(key, prev_scores[key])
        try:
            raw = float(raw)
        except (ValueError, TypeError):
            raw = prev_scores[key]
        clamped[key] = float(np.clip(
            raw,
            prev_scores[key] - MAX_PERSONALITY_DRIFT,
            prev_scores[key] + MAX_PERSONALITY_DRIFT,
        ))
        clamped[key] = float(np.clip(clamped[key], 0.0, 1.0))

    return PersonalityState(
        **clamped,
        update_reason=result.get("update_reason", ""),
        chapter_id=chapter_id,
    )


# ──────────────────────────────────────
# サマリー生成
# ──────────────────────────────────────

def build_summary(char: str,
                  new_state: PersonalityState,
                  prev_state: PersonalityState | None,
                  llm: LLMClient) -> str:
    """次章のPath Aに渡す前章サマリーを生成する"""
    prompt = (
        f"キャラクター「{char}」の現時点での性格を\n"
        f"声優への演技指示書として3〜4文でまとめてください。\n\n"
        f"性格スコア:\n"
        f"- 攻撃性: {new_state.aggression:.2f}\n"
        f"- 忠誠心: {new_state.loyalty:.2f}\n"
        f"- 不安傾向: {new_state.anxiety:.2f}\n"
        f"- 開放性: {new_state.openness:.2f}\n"
        f"- 自信: {new_state.confidence:.2f}\n\n"
        f"今章での変化: {new_state.update_reason}\n\n"
        f"出力は「{char}は〜」で始めてください。\n"
        f"知識・情報の有無には言及しないでください。\n"
    )
    return llm.call(prompt)


# ──────────────────────────────────────
# Path A 統合エントリポイント
# ──────────────────────────────────────

def update_path_a(char: str,
                  chapter: dict,
                  segments: list[dict],
                  prev_state: PersonalityState | None,
                  llm: LLMClient) -> PersonalityState:
    """Path Aのメイン処理。3段階プロンプトで逐次更新する。"""
    # 第1段階: 観察事実の抽出
    behaviors = extract_observable_behaviors(char, segments, llm)
    if not behaviors:
        return prev_state or PersonalityState(chapter_id=chapter["id"])

    # 第2段階: 変化差分の抽出
    delta = extract_personality_delta(char, behaviors, prev_state, llm)

    # 第3段階: スコア更新
    new_state = update_personality_scores(
        char, delta, prev_state, chapter["id"], llm
    )

    # サマリー生成
    new_state.summary = build_summary(char, new_state, prev_state, llm)

    return new_state


def character_appears_in_chapter(segments: list[dict], char: str) -> bool:
    """キャラクターがこの章に登場するか確認する"""
    for seg in segments:
        if seg.get("speaker") == char:
            return True
        if seg["type"] == "narration" and char in seg["text"]:
            return True
    return False


def get_previous_state(char: str,
                       chapter_id: str,
                       personality_history: dict) -> PersonalityState | None:
    """前章の性格状態を取得する"""
    history = personality_history.get(char, {})
    if not history:
        return None

    # 直前の章を探す
    try:
        ch_num = int(chapter_id.split("_")[-1])
    except (ValueError, IndexError):
        return None

    for prev_num in range(ch_num - 1, 0, -1):
        prev_id = f"chapter_{prev_num}"
        if prev_id in history:
            entry = history[prev_id]
            return entry["state"] if isinstance(entry, dict) else entry

    return None
