"""
工程 1: 前処理・セリフ分離

生の小説テキストを構造化データに変換する。
章分割 → 場面分割 → セリフ/地の文分離 → 話者タグ付け
"""

import re
import logging
from src.config import (
    MIN_SCENE_LENGTH, SPEAKER_CONTEXT_WINDOW, LLM_CONTEXT_SEGMENTS,
)
from src.llm_client import LLMClient

logger = logging.getLogger(__name__)


# ──────────────────────────────────────
# 章見出しパターン
# ──────────────────────────────────────

CHAPTER_PATTERN = re.compile(
    r'(?m)^(?:'
    r'第[一二三四五六七八九十百\d]+章'
    r'|Chapter\s*\d+'
    r'|第[一二三四五六七八九十\d]+話'
    r'|プロローグ|エピローグ'
    r')\s*[^\n]*$'
)

# ──────────────────────────────────────
# 話者動詞句パターン
# ──────────────────────────────────────

SPEAKER_PATTERN = re.compile(
    r'([^\s、。「」]{1,10})'
    r'(?:は|が|も)?'
    r'(?:言った|言う|答えた|答える|叫んだ|叫ぶ|'
    r'囁いた|囁く|呟いた|呟く|続けた|続ける|'
    r'尋ねた|尋ねる|笑った|笑う|怒鳴った|怒鳴る|'
    r'云った|云う|罵った|罵る|念を押した|押した|'
    r'つぶやいた|つぶやく|'
    r'こう云った|そう云った)'
)


# ──────────────────────────────────────
# Step 1: 章分割
# ──────────────────────────────────────

def split_chapters(text: str) -> list[dict]:
    """正規表現で章見出しを検出し分割する"""
    boundaries = [m.start() for m in CHAPTER_PATTERN.finditer(text)]
    if not boundaries:
        return [{"id": "chapter_1", "heading": "", "text": text}]

    chapters = []
    for i, start in enumerate(boundaries):
        end = boundaries[i + 1] if i + 1 < len(boundaries) else len(text)
        chapter_text = text[start:end].strip()
        chapters.append({
            "id": f"chapter_{i + 1}",
            "heading": chapter_text.split('\n')[0],
            "text": chapter_text,
        })
    return chapters


# ──────────────────────────────────────
# Step 2: 場面分割
# ──────────────────────────────────────

def split_scenes(chapter: dict) -> list[dict]:
    """空行2行以上で場面を区切り、短いブロックは前に結合"""
    text = chapter["text"]
    raw_scenes = re.split(r'\n{2,}', text)

    scenes = []
    buffer = ""
    for block in raw_scenes:
        buffer += "\n\n" + block if buffer else block
        if len(buffer) > MIN_SCENE_LENGTH:
            scenes.append(buffer.strip())
            buffer = ""
    if buffer:
        if scenes:
            scenes[-1] += "\n\n" + buffer.strip()
        else:
            scenes.append(buffer.strip())

    return [
        {"id": f"{chapter['id']}_scene_{i + 1}", "text": s}
        for i, s in enumerate(scenes)
    ]


# ──────────────────────────────────────
# Step 3: セリフ・地の文の分離
# ──────────────────────────────────────

DIALOGUE_PATTERN = re.compile(r'「(.*?)」', re.DOTALL)


def split_segments(scene: dict) -> list[dict]:
    """「」括弧でセリフと地の文を分離する"""
    text = scene["text"]
    segments = []
    pos = 0

    for m in DIALOGUE_PATTERN.finditer(text):
        # セリフの前の地の文
        if m.start() > pos:
            narration = text[pos:m.start()].strip()
            if narration:
                segments.append({
                    "type": "narration",
                    "text": narration,
                    "speaker": None,
                    "scene_id": scene["id"],
                })

        # セリフ本体
        segments.append({
            "type": "dialogue",
            "text": m.group(1),
            "speaker": None,
            "scene_id": scene["id"],
        })
        pos = m.end()

    # 末尾の地の文
    if pos < len(text):
        tail = text[pos:].strip()
        if tail:
            segments.append({
                "type": "narration",
                "text": tail,
                "speaker": None,
                "scene_id": scene["id"],
            })

    return segments


# ──────────────────────────────────────
# Step 4: 話者タグ付け（ルールベース）
# ──────────────────────────────────────

def assign_speakers(segments: list[dict],
                    known_characters: list[str]) -> list[dict]:
    """セリフの前後の地の文から話者を推定する"""
    window = SPEAKER_CONTEXT_WINDOW

    for i, seg in enumerate(segments):
        if seg["type"] != "dialogue":
            continue

        context = ""
        if i > 0 and segments[i - 1]["type"] == "narration":
            context += segments[i - 1]["text"][-window:]
        if i + 1 < len(segments) and segments[i + 1]["type"] == "narration":
            context += segments[i + 1]["text"][:window]

        m = SPEAKER_PATTERN.search(context)
        if m:
            candidate = m.group(1)
            for char in known_characters:
                if candidate in char or char in candidate:
                    seg["speaker"] = char
                    break
            else:
                seg["speaker"] = candidate

    return segments


# ──────────────────────────────────────
# Step 5: LLMによる補完
# ──────────────────────────────────────

def resolve_unknown_speakers(segments: list[dict],
                             llm: LLMClient,
                             known_characters: list[str]) -> list[dict]:
    """ルールで解決できなかったセリフをLLMに投げる"""
    unresolved = [
        s for s in segments
        if s["type"] == "dialogue" and s["speaker"] is None
    ]
    if not unresolved:
        return segments

    ctx_window = LLM_CONTEXT_SEGMENTS

    for seg in unresolved:
        idx = segments.index(seg)
        context = segments[max(0, idx - ctx_window):idx + ctx_window + 1]
        context_text = "\n".join(
            f"[{s['type']}] {s['speaker'] or '?'}: {s['text']}"
            for s in context
        )
        prompt = (
            f"以下の会話文脈で「?」の話者は誰ですか？\n"
            f"登場人物候補: {known_characters}\n\n"
            f"{context_text}\n\n"
            f"話者名だけを返してください。\n"
            f"特定不能なら「不明」と返してください。"
        )
        result = llm.call(prompt).strip()
        seg["speaker"] = result if result != "不明" else None

    return segments


# ──────────────────────────────────────
# キャラクター抽出（補助）
# ──────────────────────────────────────

def extract_character_names(segments: list[dict],
                            llm: LLMClient) -> list[str]:
    """セグメントからキャラクター名を抽出する"""
    # まずルールベースで既に特定されたspeakerを集める
    speakers = {
        seg["speaker"] for seg in segments
        if seg["speaker"] is not None
    }

    if speakers:
        return sorted(speakers)

    # ルールベースで見つからない場合はLLMに聞く
    all_text = "\n".join(seg["text"][:100] for seg in segments[:30])
    prompt = (
        "以下の小説テキストに登場するキャラクター名を列挙してください。\n"
        "名前だけをカンマ区切りで返してください。\n\n"
        f"{all_text}"
    )
    result = llm.call(prompt)
    return [name.strip() for name in result.split(",") if name.strip()]


# ──────────────────────────────────────
# 前処理統合エントリポイント
# ──────────────────────────────────────

def preprocess(text: str, llm: LLMClient) -> dict:
    """
    前処理のメイン関数。

    Returns:
        {
            "chapters": [chapter_dict, ...],
            "segments_by_chapter": {chapter_id: [segment, ...]},
            "all_segments": [segment, ...],
            "known_characters": [name, ...]
        }
    """
    logger.info("前処理を開始します")

    # Step 1: 章分割
    chapters = split_chapters(text)
    logger.info("章分割完了: %d章", len(chapters))

    all_segments: list[dict] = []
    segments_by_chapter: dict[str, list[dict]] = {}

    for chapter in chapters:
        # Step 2: 場面分割
        scenes = split_scenes(chapter)

        chapter_segs: list[dict] = []
        for scene in scenes:
            # Step 3: セリフ/地の文分離
            segs = split_segments(scene)
            chapter_segs.extend(segs)

        segments_by_chapter[chapter["id"]] = chapter_segs
        all_segments.extend(chapter_segs)

    # Step 4: キャラクター名抽出 + 話者タグ付け
    known_characters = extract_character_names(all_segments, llm)
    logger.info("登場人物: %s", known_characters)

    for chapter_id, segs in segments_by_chapter.items():
        assign_speakers(segs, known_characters)

    # Step 5: LLMによる話者補完
    for chapter_id, segs in segments_by_chapter.items():
        resolve_unknown_speakers(segs, llm, known_characters)

    # 統計
    dialogues = [s for s in all_segments if s["type"] == "dialogue"]
    identified = [s for s in dialogues if s["speaker"] is not None]
    logger.info(
        "話者同定: %d/%d セリフ (%.1f%%)",
        len(identified), len(dialogues),
        len(identified) / max(1, len(dialogues)) * 100,
    )

    return {
        "chapters": chapters,
        "segments_by_chapter": segments_by_chapter,
        "all_segments": all_segments,
        "known_characters": known_characters,
    }
