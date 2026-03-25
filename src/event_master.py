"""
工程 3: 事象マスタ定義

物語内で明らかになった重要な事実を台帳として管理する。
知識DBの参照基盤になる。
"""

import logging
from src.models import Event
from src.llm_client import LLMClient

logger = logging.getLogger(__name__)


def build_event_master(chapters: list[dict],
                       segments_by_chapter: dict,
                       llm: LLMClient) -> dict[str, Event]:
    """
    各章・場面を走査し、物語的に重要な事象を登録する。

    Returns:
        {event_id: Event, ...}
    """
    logger.info("事象マスタの構築を開始します")
    event_master: dict[str, Event] = {}
    event_counter = 0

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
            scene_text = "\n".join(seg["text"] for seg in scene_segs)

            prompt = (
                "以下の場面テキストから、物語の展開に影響を与える\n"
                "重要な事実・情報が明らかになった箇所を抽出してください。\n\n"
                "抽出基準:\n"
                "- 登場人物の秘密・正体が明かされた\n"
                "- 重要なアイテム・場所が示された\n"
                "- 人間関係に影響する事実が判明した\n"
                "- 物語の転換点となる出来事が起きた\n\n"
                "些細な出来事（右を向いた等）は含めないでください。\n\n"
                f"場面テキスト:\n{scene_text}\n\n"
                "JSON形式で返してください（なければ空リスト）:\n"
                "[\n"
                '  {"description": "何が明らかになったか（1文で）"}\n'
                "]\n"
            )
            results = llm.call_json(prompt)
            if isinstance(results, dict):
                results = [results] if results else []

            for item in results:
                desc = item.get("description", "")
                if not desc:
                    continue
                event_counter += 1
                eid = f"event_{event_counter:03d}"
                event_master[eid] = Event(
                    event_id=eid,
                    description=desc,
                    scene_id=scene_id,
                    chapter_id=chapter_id,
                )

    logger.info("事象マスタ構築完了: %d件", len(event_master))
    return event_master


def match_event_to_master(fact_description: str,
                          event_master: dict[str, Event],
                          llm: LLMClient = None) -> str | None:
    """事実の記述を事象マスタ内の既存イベントと照合する"""
    # 単純な部分一致検索
    for eid, event in event_master.items():
        if (fact_description in event.description
                or event.description in fact_description):
            return eid

    # LLMによる意味照合
    if llm and event_master:
        event_list = "\n".join(
            f"- {eid}: {ev.description}"
            for eid, ev in event_master.items()
        )
        prompt = (
            "以下の事象マスタの中から、次の事実に最も近いものを選んでください。\n"
            "該当するものがなければ「なし」と返してください。\n\n"
            f"事実: {fact_description}\n\n"
            f"事象マスタ:\n{event_list}\n\n"
            "event_idだけを返してください。"
        )
        result = llm.call(prompt).strip()
        if result in event_master:
            return result

    return None


def register_new_event(description: str,
                       chapter_id: str,
                       scene_id: str,
                       event_master: dict[str, Event]) -> str:
    """新しい事象を登録する"""
    event_counter = len(event_master) + 1
    eid = f"event_{event_counter:03d}"
    event_master[eid] = Event(
        event_id=eid,
        description=description,
        scene_id=scene_id,
        chapter_id=chapter_id,
    )
    logger.info("新規事象登録: %s = %s", eid, description)
    return eid
