"""
LLMクライアント抽象レイヤー

OpenAI API 互換のバックエンドと通信する。
`call()` でテキスト応答、`call_json()` でJSON応答を得る。
"""

import json
import re
import logging

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None

from src.config import (
    LLM_MODEL, LLM_API_KEY, LLM_BASE_URL,
    LLM_TEMPERATURE, LLM_MAX_TOKENS,
)

logger = logging.getLogger(__name__)


class LLMClient:
    """OpenAI API 互換 LLM クライアント"""

    def __init__(self, model: str = None, api_key: str = None,
                 base_url: str = None):
        self.model = model or LLM_MODEL
        self.api_key = api_key or LLM_API_KEY
        self.base_url = base_url or LLM_BASE_URL
        self._client = None

    def _get_client(self):
        if self._client is None:
            if OpenAI is None:
                raise ImportError(
                    "openai パッケージが必要です: pip install openai"
                )
            self._client = OpenAI(
                api_key=self.api_key,
                base_url=self.base_url,
            )
        return self._client

    def call(self, prompt: str) -> str:
        """テキスト応答を返す"""
        client = self._get_client()
        try:
            response = client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=LLM_TEMPERATURE,
                max_tokens=LLM_MAX_TOKENS,
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            logger.error("LLM呼び出しエラー: %s", e)
            return ""

    def call_json(self, prompt: str) -> dict | list:
        """JSON応答を返す。パース失敗時は空dict/listを返す"""
        raw = self.call(prompt)
        if not raw:
            return {}
        return self._parse_json(raw)

    @staticmethod
    def _parse_json(text: str) -> dict | list:
        """LLM応答からJSONを抽出・パースする"""
        # まず直接パースを試みる
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass

        # マークダウンコードブロックの中身を抽出
        m = re.search(r'```(?:json)?\s*\n?(.*?)\n?```', text, re.DOTALL)
        if m:
            try:
                return json.loads(m.group(1))
            except json.JSONDecodeError:
                pass

        # 最初の { ... } または [ ... ] を抽出
        for start_char, end_char in [('{', '}'), ('[', ']')]:
            start = text.find(start_char)
            end = text.rfind(end_char)
            if start != -1 and end != -1 and end > start:
                try:
                    return json.loads(text[start:end + 1])
                except json.JSONDecodeError:
                    pass

        logger.warning("JSONパースに失敗: %s…", text[:100])
        return {}
