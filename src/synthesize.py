"""
Style-BERT-VITS2 音声合成スクリプト

出力JSONを読み込み、Style-BERT-VITS2 APIサーバーに送信して
WAVファイルを生成する。

前提条件:
  1. Style-BERT-VITS2 をインストール済み
     https://github.com/litagin02/Style-BERT-VITS2
  2. APIサーバーが起動済み
     python server_fastapi.py
     → デフォルトで http://localhost:5000 で起動
  3. 音声モデルが配置済み

使い方:
  python src/synthesize.py
  python src/synthesize.py --input output/style_bert_vits2.json --output output/audio
  python src/synthesize.py --server http://localhost:5000
"""

import json
import logging
import sys
import time
import wave
import struct
from pathlib import Path
from urllib.parse import urlencode

try:
    import requests
except ImportError:
    requests = None

from src.config import OUTPUT_DIR

logger = logging.getLogger(__name__)

# ──────────────────────────────────────
# デフォルト設定
# ──────────────────────────────────────

DEFAULT_SERVER = "http://127.0.0.1:5000"
DEFAULT_MODEL_ID = 0
DEFAULT_SPEAKER_ID = 0
DEFAULT_LANGUAGE = "JP"


# ──────────────────────────────────────
# APIクライアント
# ──────────────────────────────────────

class StyleBertVits2Client:
    """Style-BERT-VITS2 APIサーバーとの通信"""

    def __init__(self, server_url: str = None,
                 model_id: int = None,
                 speaker_id: int = None):
        if requests is None:
            raise ImportError(
                "requests パッケージが必要です: pip install requests"
            )
        self.server_url = (server_url or DEFAULT_SERVER).rstrip("/")
        self.model_id = model_id if model_id is not None else DEFAULT_MODEL_ID
        self.speaker_id = (
            speaker_id if speaker_id is not None else DEFAULT_SPEAKER_ID
        )

    def check_server(self) -> bool:
        """サーバーの接続確認"""
        try:
            resp = requests.get(f"{self.server_url}/models/info", timeout=5)
            if resp.status_code == 200:
                models = resp.json()
                logger.info("利用可能モデル:")
                for mid, info in models.items():
                    logger.info(
                        "  [%s] %s (スタイル: %s)",
                        mid,
                        info.get("config_path", ""),
                        list(info.get("style2id", {}).keys()),
                    )
                return True
        except requests.ConnectionError:
            logger.error(
                "Style-BERT-VITS2 サーバーに接続できません: %s",
                self.server_url,
            )
            logger.error(
                "サーバーを起動してください: python server_fastapi.py"
            )
        except Exception as e:
            logger.error("サーバー接続エラー: %s", e)
        return False

    def synthesize(self, params: dict) -> bytes | None:
        """1セリフの音声合成を実行し、WAVバイナリを返す"""
        text = params.get("text", "")
        if not text or text.strip() == "":
            return None

        # Style-BERT-VITS2 APIパラメータ構築
        style_weight = params.get("style_weight", {})
        if style_weight:
            dominant_style = max(style_weight, key=style_weight.get)
            dominant_weight = style_weight[dominant_style]
        else:
            dominant_style = "Neutral"
            dominant_weight = 0.0

        # style_weightが0の場合はNeutralにフォールバック
        if dominant_weight < 0.01:
            dominant_style = "Neutral"

        query_params = {
            "text": text,
            "model_id": self.model_id,
            "speaker_id": self.speaker_id,
            "language": DEFAULT_LANGUAGE,
            "style": dominant_style,
            "style_weight": min(dominant_weight * 5.0, 2.0),
            "length": 1.0 / max(params.get("speed", 1.0), 0.5),
            "sdp_ratio": 0.2,
            "noise": 0.6,
            "noisew": 0.8,
            "auto_split": "true",
            "split_interval": 0.5,
        }

        # pitch, intonation の追加（対応版の場合）
        pitch = params.get("pitch", 1.0)
        if pitch != 1.0:
            query_params["pitch_scale"] = pitch

        intonation = params.get("intonation_scale", 0.5)
        if intonation != 0.3:
            query_params["intonation_scale"] = intonation

        # 前後の無音
        pre_silence = params.get("pre_phoneme_length", 0.0)
        post_silence = params.get("post_phoneme_length", 0.0)
        if pre_silence > 0:
            query_params["pre_phoneme_length"] = pre_silence
        if post_silence > 0:
            query_params["post_phoneme_length"] = post_silence

        try:
            url = f"{self.server_url}/voice?{urlencode(query_params)}"
            resp = requests.get(url, timeout=60)

            if resp.status_code == 200:
                return resp.content
            else:
                logger.warning(
                    "合成失敗 [%d]: %s… → %s",
                    resp.status_code,
                    text[:30],
                    resp.text[:100],
                )
                return None

        except requests.Timeout:
            logger.warning("タイムアウト: %s…", text[:30])
            return None
        except Exception as e:
            logger.error("合成エラー: %s → %s", text[:30], e)
            return None


# ──────────────────────────────────────
# 無音WAV生成
# ──────────────────────────────────────

def generate_silence_wav(duration_sec: float,
                         sample_rate: int = 44100) -> bytes:
    """指定秒数の無音WAVデータを生成する"""
    n_samples = int(sample_rate * duration_sec)
    data = struct.pack(f"<{n_samples}h", *([0] * n_samples))

    import io
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(data)
    return buf.getvalue()


# ──────────────────────────────────────
# WAV結合
# ──────────────────────────────────────

def concatenate_wavs(wav_data_list: list[bytes],
                     output_path: Path) -> Path:
    """複数のWAVバイナリを1つのファイルに結合する"""
    import io

    all_frames = b""
    params_set = False
    n_channels, samp_width, frame_rate = 1, 2, 44100

    for wav_data in wav_data_list:
        if not wav_data:
            continue
        buf = io.BytesIO(wav_data)
        try:
            with wave.open(buf, "rb") as wf:
                if not params_set:
                    n_channels = wf.getnchannels()
                    samp_width = wf.getsampwidth()
                    frame_rate = wf.getframerate()
                    params_set = True
                all_frames += wf.readframes(wf.getnframes())
        except wave.Error as e:
            logger.warning("WAV読み取りスキップ: %s", e)
            continue

    if not all_frames:
        logger.warning("結合するWAVデータがありません")
        return None

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(output_path), "wb") as wf:
        wf.setnchannels(n_channels)
        wf.setsampwidth(samp_width)
        wf.setframerate(frame_rate)
        wf.writeframes(all_frames)

    logger.info("音声ファイル生成: %s", output_path)
    return output_path


# ──────────────────────────────────────
# メイン処理
# ──────────────────────────────────────

def synthesize_audiobook(input_json: Path,
                         output_dir: Path,
                         server_url: str = None,
                         model_id: int = None,
                         speaker_id: int = None) -> Path | None:
    """
    JSONを読み込み、Style-BERT-VITS2で音声化し、
    結合してオーディオブックWAVを生成する。
    """
    logger.info("音声合成を開始します")

    # クライアント初期化 & サーバー確認
    client = StyleBertVits2Client(server_url, model_id, speaker_id)
    if not client.check_server():
        return None

    # JSON読み込み
    with open(input_json, "r", encoding="utf-8") as f:
        utterances = json.load(f)
    logger.info("入力: %d セリフ", len(utterances))

    # 個別WAVの生成
    wav_parts: list[bytes] = []
    output_dir.mkdir(parents=True, exist_ok=True)

    for i, utt in enumerate(utterances):
        text = utt.get("text", "")
        logger.info("[%d/%d] %s…", i + 1, len(utterances), text[:30])

        # 前の無音
        pre_silence = utt.get("pre_phoneme_length", 0.0)
        if pre_silence > 0.05:
            wav_parts.append(generate_silence_wav(pre_silence))

        # 音声合成
        wav_data = client.synthesize(utt)
        if wav_data:
            wav_parts.append(wav_data)

            # 個別ファイルも保存
            individual_path = output_dir / f"utt_{i + 1:04d}.wav"
            individual_path.write_bytes(wav_data)

        # 後の無音
        post_silence = utt.get("post_phoneme_length", 0.0)
        if post_silence > 0.05:
            wav_parts.append(generate_silence_wav(post_silence))

        # セリフ間の基本間隔（0.3秒）
        wav_parts.append(generate_silence_wav(0.3))

        # APIレート制限対策
        time.sleep(0.1)

    # 全体を結合
    final_path = output_dir / "audiobook.wav"
    result = concatenate_wavs(wav_parts, final_path)

    if result:
        logger.info("オーディオブック生成完了: %s", result)
    return result


# ──────────────────────────────────────
# CLI
# ──────────────────────────────────────

def main():
    import argparse

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    parser = argparse.ArgumentParser(
        description="Style-BERT-VITS2 音声合成"
    )
    parser.add_argument(
        "--input", type=str,
        default=str(OUTPUT_DIR / "style_bert_vits2.json"),
        help="入力JSONファイルのパス",
    )
    parser.add_argument(
        "--output", type=str,
        default=str(OUTPUT_DIR / "audio"),
        help="音声出力ディレクトリ",
    )
    parser.add_argument(
        "--server", type=str,
        default=DEFAULT_SERVER,
        help="Style-BERT-VITS2 APIサーバーのURL",
    )
    parser.add_argument(
        "--model-id", type=int, default=DEFAULT_MODEL_ID,
        help="使用するモデルID",
    )
    parser.add_argument(
        "--speaker-id", type=int, default=DEFAULT_SPEAKER_ID,
        help="使用するスピーカーID",
    )
    args = parser.parse_args()

    result = synthesize_audiobook(
        input_json=Path(args.input),
        output_dir=Path(args.output),
        server_url=args.server,
        model_id=args.model_id,
        speaker_id=args.speaker_id,
    )
    if result:
        print(f"\n完了: {result}")
    else:
        print("\n音声合成に失敗しました。")
        sys.exit(1)


if __name__ == "__main__":
    main()
