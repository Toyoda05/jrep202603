"""
工程 4: 知識DB更新（全章ループ）

事象マスタを参照しながら、各キャラクターが
どの章・場面で何を知ったかを確定する。
"""

import logging
from src.models import Event, KnowledgeEntry
from src.llm_client import LLMClient

logger = logging.getLogger(__name__)


# ──────────────────────────────────────
# 初期化
# ──────────────────────────────────────

def initialize_knowledge_db(known_characters: list[str],
                            event_master: dict[str, Event]) -> dict:
    """キャラクター × 事象 の知識状態マトリクスを初期化"""
    knowledge_db: dict[str, dict[str, KnowledgeEntry]] = {}
    for char in known_characters:
        knowledge_db[char] = {}
        for eid in event_master:
            knowledge_db[char][eid] = KnowledgeEntry(known=False)
    return knowledge_db


# ──────────────────────────────────────
# ① 登場人物の特定
# ──────────────────────────────────────

def detect_present_characters(scene_segments: list[dict],
                              known_characters: list[str]) -> list[str]:
    """場面テキストに登場するキャラクターを特定する"""
    present = set()
    for seg in scene_segments:
        if seg["type"] == "dialogue" and seg.get("speaker"):
            present.add(seg["speaker"])
        for char in known_characters:
            if char in seg["text"]:
                present.add(char)
    return list(present)


# ──────────────────────────────────────
# ② 未解決フラグの絞り込み
# ──────────────────────────────────────

def get_unresolved_events(present_characters: list[str],
                          knowledge_db: dict,
                          event_master: dict[str, Event]) -> dict:
    """登場人物ごとに、まだ知得していない事象のみを返す"""
    unresolved: dict[str, list[str]] = {}
    for character in present_characters:
        if character not in knowledge_db:
            continue
        unresolved[character] = [
            event_id
            for event_id, entry in knowledge_db[character].items()
            if not entry.known
        ]
    return unresolved


# ──────────────────────────────────────
# ③ 知得判定（LLM）
# ──────────────────────────────────────

def judge_knowledge_acquisition(scene_segments: list[dict],
                                character: str,
                                event: Event,
                                llm: LLMClient) -> dict:
    """キャラクターが場面で事象を知り得たかを判定する"""
    scene_text = "\n".join(seg["text"] for seg in scene_segments)

    prompt = (
        f"場面テキスト:\n{scene_text}\n\n"
        f"キャラクター「{character}」はこの場面で\n"
        f"以下の事実を知り得ましたか？\n\n"
        f"事実: {event.description}\n\n"
        "判定基準:\n"
        "- 直接目撃・聴取した場合 → True\n"
        "- その場にいたが知覚できなかった場合（気絶・別室など） → False\n"
        "- 伝聞・推測のみの場合 → False（推測は知識として扱わない）\n\n"
        "以下のJSON形式で返してください:\n"
        '{\n'
        '  "learned": true/false,\n'
        '  "evidence": "根拠となった文章を原文から抜粋"\n'
        '}\n'
    )
    result = llm.call_json(prompt)
    return {
        "learned": result.get("learned", False),
        "evidence": result.get("evidence", ""),
    }


# ──────────────────────────────────────
# メインループ
# ──────────────────────────────────────

def run_knowledge_db_loop(chapters: list[dict],
                          segments_by_chapter: dict,
                          event_master: dict[str, Event],
                          known_characters: list[str],
                          llm: LLMClient) -> dict:
    """
    知識DB更新のメインループ。

    Returns:
        knowledge_db: {character: {event_id: KnowledgeEntry, ...}, ...}
    """
    logger.info("知識DBループを開始します")

    knowledge_db = initialize_knowledge_db(known_characters, event_master)

    if not event_master:
        logger.info("事象マスタが空のため知識DBループをスキップ")
        return knowledge_db

    for chapter in chapters:
        chapter_id = chapter["id"]
        segs = segments_by_chapter.get(chapter_id, [])

        # 場面ごとにグループ化
        scenes: dict[str, list[dict]] = {}
        for seg in segs:
            sid = seg["scene_id"]
            scenes.setdefault(sid, [])
            scenes[sid].append(seg)

        for scene_id, scene_segs in scenes.items():
            # ① 登場人物を特定
            present = detect_present_characters(scene_segs, known_characters)

            # ② 未解決フラグのみ判定対象
            unresolved = get_unresolved_events(
                present, knowledge_db, event_master
            )

            # ③ 知得判定
            for character in present:
                for event_id in unresolved.get(character, []):
                    result = judge_knowledge_acquisition(
                        scene_segs, character,
                        event_master[event_id], llm,
                    )
                    if result["learned"]:
                        knowledge_db[character][event_id] = KnowledgeEntry(
                            known=True,
                            learned_at=chapter_id,
                            evidence=result["evidence"],
                        )

        logger.info("知識DB更新: %s 完了", chapter_id)

    # 統計
    total = sum(
        1 for char_events in knowledge_db.values()
        for entry in char_events.values()
        if entry.known
    )
    logger.info("知識DB確定: %d件の知得エントリ", total)

    return knowledge_db
