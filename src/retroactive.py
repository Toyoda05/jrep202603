"""
工程 5: 遡及検出・過去エントリ修正

「実は前から知っていた」を検出し、知識DBを遡及修正し、
影響セリフに抑制フラグを付与する。
"""

import logging
from src.models import Event, KnowledgeEntry
from src.config import CONFIDENCE_THRESHOLD, SUPPRESSION_WEIGHTS
from src.llm_client import LLMClient
from src.event_master import match_event_to_master, register_new_event

logger = logging.getLogger(__name__)


# ──────────────────────────────────────
# Stage 1: 遡及的知識の検出
# ──────────────────────────────────────

def detect_explicit_retroactive(chapter_text: str, llm: LLMClient) -> list:
    """パターンA: 明示的な告白・回想"""
    prompt = (
        "以下の章テキストに、キャラクターが\n"
        "「実は以前から知っていた・気づいていた」ことを\n"
        "明示的に示す表現はありますか？\n\n"
        "検出対象の表現例:\n"
        "- 「ずっと知っていた」「前から気づいていた」\n"
        "- 回想シーン\n"
        "- 告白\n\n"
        f"章テキスト:\n{chapter_text}\n\n"
        "JSON形式で返してください（なければ空リスト）:\n"
        "[\n"
        '  {\n'
        '    "character": "キャラクター名",\n'
        '    "fact": "知っていた事実",\n'
        '    "pattern": "A",\n'
        '    "estimated_known_from": "推定される最初の知得章（不明ならnull）",\n'
        '    "evidence_text": "根拠となった原文抜粋",\n'
        '    "confidence": 0.0\n'
        '  }\n'
        "]\n"
    )
    result = llm.call_json(prompt)
    return result if isinstance(result, list) else []


def detect_behavioral_contradiction(chapter_text: str,
                                    knowledge_db: dict,
                                    event_master: dict[str, Event],
                                    llm: LLMClient) -> list:
    """パターンB: 行動の矛盾"""
    unknown_facts = [
        (char, event_id, event_master[event_id].description)
        for char, events in knowledge_db.items()
        for event_id, entry in events.items()
        if not entry.known and event_id in event_master
    ]
    if not unknown_facts:
        return []

    unknown_summary = "\n".join(
        f"- {char}は「{desc}」をまだ知らないはず"
        for char, _, desc in unknown_facts[:20]   # コスト制限
    )

    prompt = (
        "以下のキャラクターは現時点でこれらの事実を「知らないはず」です:\n"
        f"{unknown_summary}\n\n"
        "しかし以下の章テキストで、これらの事実を知っているかのような\n"
        "行動・発言をしているキャラクターはいますか？\n\n"
        f"章テキスト:\n{chapter_text}\n\n"
        "JSON形式で返してください（なければ空リスト）:\n"
        "[\n"
        '  {\n'
        '    "character": "キャラクター名",\n'
        '    "fact": "矛盾している事実",\n'
        '    "pattern": "B",\n'
        '    "estimated_known_from": null,\n'
        '    "evidence_text": "矛盾を示す原文抜粋",\n'
        '    "confidence": 0.0\n'
        '  }\n'
        "]\n"
    )
    result = llm.call_json(prompt)
    return result if isinstance(result, list) else []


def detect_third_party_testimony(chapter_text: str,
                                 llm: LLMClient) -> list:
    """パターンC: 第三者証言"""
    prompt = (
        "以下の章テキストに、第三者が\n"
        "「あのキャラクターは以前からこの事実を知っていたはずだ」\n"
        "という趣旨の発言・描写はありますか？\n\n"
        f"章テキスト:\n{chapter_text}\n\n"
        "JSON形式で返してください（なければ空リスト）:\n"
        "[\n"
        '  {\n'
        '    "character": "知っていたとされるキャラクター名",\n'
        '    "fact": "知っていたとされる事実",\n'
        '    "pattern": "C",\n'
        '    "estimated_known_from": null,\n'
        '    "evidence_text": "根拠となった原文抜粋",\n'
        '    "confidence": 0.0\n'
        '  }\n'
        "]\n"
    )
    result = llm.call_json(prompt)
    return result if isinstance(result, list) else []


def deduplicate_candidates(candidates: list) -> list:
    """同一キャラ×同一事実の重複を除去"""
    seen = set()
    deduped = []
    for c in candidates:
        key = (c.get("character", ""), c.get("fact", ""))
        if key not in seen:
            seen.add(key)
            deduped.append(c)
    return deduped


def detect_retroactive_knowledge(chapter: dict,
                                 knowledge_db: dict,
                                 event_master: dict[str, Event],
                                 llm: LLMClient) -> list:
    """遡及的知識を3パターンで検出する"""
    chapter_text = chapter["text"]

    pattern_a = detect_explicit_retroactive(chapter_text, llm)
    pattern_b = detect_behavioral_contradiction(
        chapter_text, knowledge_db, event_master, llm
    )
    pattern_c = detect_third_party_testimony(chapter_text, llm)

    candidates = pattern_a + pattern_b + pattern_c
    return deduplicate_candidates(candidates)


# ──────────────────────────────────────
# Stage 2: 知識DBの遡及修正
# ──────────────────────────────────────

def should_apply_retroactive(old_learned_at: str | None,
                             new_learned_at: str | None) -> bool:
    """new_learned_atがold_learned_atより前の章かを判定"""
    if new_learned_at is None:
        return False
    if old_learned_at is None:
        return True
    try:
        old_num = int(old_learned_at.split("_")[-1])
        new_num = int(new_learned_at.split("_")[-1])
        return new_num < old_num
    except (ValueError, IndexError):
        return False


def apply_retroactive_updates(candidates: list,
                              knowledge_db: dict,
                              event_master: dict[str, Event],
                              current_chapter_id: str,
                              llm: LLMClient) -> list:
    """
    検出候補を知識DBに反映する。

    Returns:
        affected_ranges: 影響範囲のリスト
    """
    affected_ranges = []

    for candidate in candidates:
        conf = candidate.get("confidence", 0.0)
        if isinstance(conf, str):
            try:
                conf = float(conf)
            except ValueError:
                conf = 0.0

        if conf < CONFIDENCE_THRESHOLD:
            continue

        character = candidate.get("character", "")
        if character not in knowledge_db:
            continue

        matched_event_id = match_event_to_master(
            candidate.get("fact", ""), event_master, llm
        )
        if not matched_event_id:
            matched_event_id = register_new_event(
                candidate.get("fact", ""),
                current_chapter_id,
                f"{current_chapter_id}_scene_1",
                event_master,
            )
            # 新規イベント用の知識エントリを初期化
            for char in knowledge_db:
                knowledge_db[char].setdefault(
                    matched_event_id, KnowledgeEntry(known=False)
                )

        current_entry = knowledge_db[character].get(
            matched_event_id, KnowledgeEntry(known=False)
        )
        old_learned_at = current_entry.learned_at
        new_learned_at = candidate.get("estimated_known_from")

        if should_apply_retroactive(old_learned_at, new_learned_at):
            knowledge_db[character][matched_event_id] = KnowledgeEntry(
                known=True,
                learned_at=new_learned_at,
                evidence=candidate.get("evidence_text", ""),
                retroactive=True,
                originally_detected_at=old_learned_at,
            )
            affected_ranges.append({
                "character": character,
                "event_id": matched_event_id,
                "from_chapter": new_learned_at,
                "to_chapter": old_learned_at,
                "pattern": candidate.get("pattern", "B"),
            })

    return affected_ranges


# ──────────────────────────────────────
# Stage 3: 影響セリフへのフラグ付与
# ──────────────────────────────────────

def get_chapter_range(from_ch: str | None, to_ch: str | None) -> list[str]:
    """章IDの範囲をリストで返す"""
    if not from_ch or not to_ch:
        return []
    try:
        from_n = int(from_ch.split("_")[-1])
        to_n = int(to_ch.split("_")[-1])
        return [f"chapter_{i}" for i in range(from_n, to_n + 1)]
    except (ValueError, IndexError):
        return []


def calc_suppression_weight(pattern: str) -> float:
    """検出パターンに応じた抑制重みを返す"""
    return SUPPRESSION_WEIGHTS.get(pattern, 0.4)


def flag_affected_utterances(affected_ranges: list,
                             segments_by_chapter: dict) -> None:
    """遡及修正の影響セリフに抑制フラグを付与する（in-place）"""
    for affected in affected_ranges:
        character = affected["character"]
        event_id = affected["event_id"]
        from_ch = affected["from_chapter"]
        to_ch = affected["to_chapter"]

        affected_chapters = get_chapter_range(from_ch, to_ch)
        for chapter_id in affected_chapters:
            if chapter_id not in segments_by_chapter:
                continue
            for seg in segments_by_chapter[chapter_id]:
                if seg.get("speaker") != character:
                    continue
                if seg["type"] != "dialogue":
                    continue
                seg.setdefault("knowledge_control", {})
                seg["knowledge_control"][event_id] = {
                    "hiding_knowledge": True,
                    "known_since": from_ch,
                    "suppression_weight": calc_suppression_weight(
                        affected["pattern"]
                    ),
                }

    flagged = sum(
        1 for segs in segments_by_chapter.values()
        for seg in segs
        if seg.get("knowledge_control")
    )
    if flagged:
        logger.info("知識制御フラグ付与: %dセリフ", flagged)


# ──────────────────────────────────────
# 統合エントリポイント
# ──────────────────────────────────────

def run_retroactive_detection(chapters: list[dict],
                              knowledge_db: dict,
                              event_master: dict[str, Event],
                              segments_by_chapter: dict,
                              llm: LLMClient) -> list:
    """
    遡及検出のメイン処理。

    Returns:
        all_affected_ranges: 全章で検出された影響範囲
    """
    logger.info("遡及検出を開始します")
    all_affected = []

    for chapter in chapters:
        candidates = detect_retroactive_knowledge(
            chapter, knowledge_db, event_master, llm
        )
        if candidates:
            affected = apply_retroactive_updates(
                candidates, knowledge_db, event_master,
                chapter["id"], llm,
            )
            flag_affected_utterances(affected, segments_by_chapter)
            all_affected.extend(affected)

    logger.info("遡及検出完了: %d件の遡及修正", len(all_affected))
    return all_affected
