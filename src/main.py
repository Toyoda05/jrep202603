"""
オーディオブック生成パイプライン — メインエントリポイント

2パス構成:
  第1パス: 知識DBループ（全文）＋ 性格履歴ループ（章ごと逐次）
  第2パス: TTSパラメタ生成ループ（性格履歴・知識DB参照）
"""

import logging
import sys
from pathlib import Path

from src.config import NOVEL_PATH, OUTPUT_DIR
from src.llm_client import LLMClient
from src.models import PersonalityState, EmotionalState

# 各工程モジュール
from src.preprocessing import preprocess
from src.full_text_scan import run_full_text_scan
from src.event_master import build_event_master
from src.knowledge_db import run_knowledge_db_loop
from src.retroactive import run_retroactive_detection
from src.path_a import (
    update_path_a, character_appears_in_chapter, get_previous_state,
)
from src.path_b import run_path_b_chapter, initialize_path_b_states
from src.fusion import detect_and_fuse
from src.emotion import update_emotional_state
from src.tts_params import integrate_three_layers
from src.tts_output import build_utterance_json, save_full_output


# ──────────────────────────────────────
# ログ設定
# ──────────────────────────────────────

def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )


# ──────────────────────────────────────
# 第1パス: 性格履歴ループ
# ──────────────────────────────────────

def run_personality_history_loop(chapters, segments_by_chapter,
                                 known_characters, cluster_profiles,
                                 llm):
    """
    性格履歴ループ（章ごと逐次）。
    Path A (LLM要約) + Path B (Embedding) + 矛盾検出・統合。
    """
    logger = logging.getLogger(__name__)
    logger.info("性格履歴ループを開始します")

    personality_history = {char: {} for char in known_characters}
    path_b_states = initialize_path_b_states(known_characters)

    for chapter in chapters:
        chapter_id = chapter["id"]
        segs = segments_by_chapter.get(chapter_id, [])

        for char in known_characters:
            if not character_appears_in_chapter(segs, char):
                prev = get_previous_state(char, chapter_id, personality_history)
                if prev:
                    personality_history[char][chapter_id] = {
                        "state": prev,
                        "meta": {"fusion_method": "inherited"},
                    }
                continue

            prev_state = get_previous_state(
                char, chapter_id, personality_history
            )

            # Path A: LLMによる観察行動ベース更新
            path_a_result = update_path_a(
                char, chapter, segs, prev_state, llm
            )

            # Path B: Embedding逐次更新
            path_b_snapshot = run_path_b_chapter(
                char, chapter_id, segs,
                path_b_states, cluster_profiles,
            )

            # 矛盾検出・統合
            cluster_labels = cluster_profiles.get(
                char, {}
            ).get("cluster_labels", [])

            fusion_result = detect_and_fuse(
                path_a_result, path_b_snapshot,
                prev_state, cluster_labels, llm,
            )

            personality_history[char][chapter_id] = {
                "state": fusion_result.confirmed_state,
                "meta": {
                    "contradiction_score": fusion_result.contradiction_score,
                    "cosine_similarity": fusion_result.cosine_similarity,
                    "needed_regeneration": fusion_result.needed_regeneration,
                    "regeneration_count": fusion_result.regeneration_count,
                    "fusion_method": fusion_result.fusion_method,
                },
            }

        logger.info("性格履歴: %s 完了", chapter_id)

    return personality_history


# ──────────────────────────────────────
# 第2パス: TTSパラメタ生成ループ
# ──────────────────────────────────────

def run_tts_generation_loop(chapters, segments_by_chapter,
                            personality_history, knowledge_db,
                            known_characters, llm):
    """
    第2パス: 確定済みの性格履歴・知識DBを参照しながら
    セリフごとのTTSパラメタを生成する。
    """
    logger = logging.getLogger(__name__)
    logger.info("TTSパラメタ生成ループを開始します")

    all_utterance_jsons = []
    utt_counter = 0

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
            # キャラクターごとの感情状態を場面内で管理
            char_emotions: dict[str, EmotionalState] = {}

            for char in known_characters:
                # 性格層を取得（章ごとに確定済み）
                ch_entry = personality_history.get(char, {}).get(chapter_id)
                if ch_entry:
                    personality = (
                        ch_entry["state"]
                        if isinstance(ch_entry, dict) else ch_entry
                    )
                else:
                    personality = PersonalityState()

                # 心情層を場面単位で推定
                knowledge_control = {}
                for seg in scene_segs:
                    if seg.get("speaker") == char:
                        knowledge_control = seg.get("knowledge_control", {})
                        break

                prev_emotion = char_emotions.get(
                    char, EmotionalState()
                )
                emotion = update_emotional_state(
                    char, scene_segs, prev_emotion,
                    personality, knowledge_control, llm,
                )
                char_emotions[char] = emotion

            # セリフごとにTTSパラメタを生成
            for i, seg in enumerate(scene_segs):
                if seg["type"] != "dialogue" or not seg.get("speaker"):
                    continue

                char = seg["speaker"]
                utt_counter += 1

                # 性格層
                ch_entry = personality_history.get(char, {}).get(chapter_id)
                if ch_entry:
                    personality = (
                        ch_entry["state"]
                        if isinstance(ch_entry, dict) else ch_entry
                    )
                else:
                    personality = PersonalityState()

                # 心情層
                emotion = char_emotions.get(char, EmotionalState())

                # 知識制御
                kc = seg.get("knowledge_control", {})

                # 前後セグメント
                start = max(0, i - 2)
                end = min(len(scene_segs), i + 3)
                surrounding = scene_segs[start:end]

                # 3層統合
                tts = integrate_three_layers(
                    seg, personality, emotion, kc, surrounding, llm,
                )

                # 融合メタデータ
                fusion_meta_data = None
                if ch_entry and isinstance(ch_entry, dict):
                    fusion_meta_data = ch_entry.get("meta")

                # JSON構築
                utt_json = build_utterance_json(
                    utterance=seg,
                    personality=personality,
                    emotion=emotion,
                    tts=tts,
                    fusion_meta=fusion_meta_data,
                    utterance_id=f"utt_{utt_counter:04d}",
                )
                all_utterance_jsons.append(utt_json)

        logger.info("TTS生成: %s 完了", chapter_id)

    return all_utterance_jsons


# ──────────────────────────────────────
# メイン関数
# ──────────────────────────────────────

def main(novel_path: str = None, output_dir: str = None):
    """パイプラインのメインエントリポイント"""
    setup_logging()
    logger = logging.getLogger(__name__)

    novel_file = Path(novel_path) if novel_path else NOVEL_PATH
    out_dir = Path(output_dir) if output_dir else OUTPUT_DIR

    # ─── 入力読み込み ───
    if not novel_file.exists():
        logger.error("小説ファイルが見つかりません: %s", novel_file)
        sys.exit(1)

    text = novel_file.read_text(encoding="utf-8")
    logger.info("入力: %s (%d文字)", novel_file, len(text))

    # ─── LLMクライアント初期化 ───
    llm = LLMClient()

    # ═══════════════════════════════════
    # 工程 1: 前処理・セリフ分離
    # ═══════════════════════════════════
    logger.info("=" * 50)
    logger.info("工程 1: 前処理・セリフ分離")
    logger.info("=" * 50)
    prep = preprocess(text, llm)
    chapters = prep["chapters"]
    segments_by_chapter = prep["segments_by_chapter"]
    all_segments = prep["all_segments"]
    known_characters = prep["known_characters"]

    # ═══════════════════════════════════
    # 工程 2: 全文スキャン
    # ═══════════════════════════════════
    logger.info("=" * 50)
    logger.info("工程 2: 全文スキャン")
    logger.info("=" * 50)
    scan_result = run_full_text_scan(all_segments, llm)
    cluster_profiles = scan_result["cluster_profiles"]

    # ═══════════════════════════════════
    # 工程 3: 事象マスタ定義
    # ═══════════════════════════════════
    logger.info("=" * 50)
    logger.info("工程 3: 事象マスタ定義")
    logger.info("=" * 50)
    event_master = build_event_master(chapters, segments_by_chapter, llm)

    # ═══════════════════════════════════
    # 工程 4: 知識DB更新
    # ═══════════════════════════════════
    logger.info("=" * 50)
    logger.info("工程 4: 知識DB更新")
    logger.info("=" * 50)
    knowledge_db = run_knowledge_db_loop(
        chapters, segments_by_chapter,
        event_master, known_characters, llm,
    )

    # ═══════════════════════════════════
    # 工程 5: 遡及検出
    # ═══════════════════════════════════
    logger.info("=" * 50)
    logger.info("工程 5: 遡及検出")
    logger.info("=" * 50)
    run_retroactive_detection(
        chapters, knowledge_db, event_master,
        segments_by_chapter, llm,
    )

    # ═══════════════════════════════════
    # 工程 6-8: 性格履歴ループ（Path A + B + 統合）
    # ═══════════════════════════════════
    logger.info("=" * 50)
    logger.info("工程 6-8: 性格履歴ループ")
    logger.info("=" * 50)
    personality_history = run_personality_history_loop(
        chapters, segments_by_chapter,
        known_characters, cluster_profiles, llm,
    )

    # ═══════════════════════════════════
    # 工程 9-10: TTSパラメタ生成（第2パス）
    # ═══════════════════════════════════
    logger.info("=" * 50)
    logger.info("工程 9-10: TTSパラメタ生成")
    logger.info("=" * 50)
    all_utterance_jsons = run_tts_generation_loop(
        chapters, segments_by_chapter,
        personality_history, knowledge_db,
        known_characters, llm,
    )

    # ═══════════════════════════════════
    # 工程 11: 出力
    # ═══════════════════════════════════
    logger.info("=" * 50)
    logger.info("工程 11: TTSパラメタ出力")
    logger.info("=" * 50)
    output_paths = save_full_output(
        all_utterance_jsons, personality_history,
        knowledge_db, out_dir,
    )

    logger.info("パイプライン完了")
    for name, path in output_paths.items():
        logger.info("  %s: %s", name, path)

    return output_paths


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="オーディオブック生成パイプライン"
    )
    parser.add_argument(
        "--novel", type=str, default=None,
        help="入力小説ファイルのパス (default: novel/novel.txt)",
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="出力ディレクトリのパス (default: output/)",
    )
    args = parser.parse_args()

    main(novel_path=args.novel, output_dir=args.output)
