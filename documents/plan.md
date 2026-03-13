# 前処理・セリフ分離
日本語小説を前提に、実装を順番に説明します。

---

## 全体の流れ

```python
def preprocess(text: str) -> dict:
    chapters = split_chapters(text)
    result = []
    for chapter in chapters:
        scenes = split_scenes(chapter)
        for scene in scenes:
            segments = split_segments(scene)
            result.append(segments)
    return result
```

---

## Step 1：章分割

日本語小説の章見出しパターンを正規表現で検出します。

```python
import re

def split_chapters(text: str) -> list[dict]:
    # 代表的な章見出しパターン
    pattern = re.compile(
        r'(?m)^(?:'
        r'第[一二三四五六七八九十百\d]+章'   # 第一章・第1章
        r'|Chapter\s*\d+'                    # Chapter 1
        r'|第[一二三四五六七八九十\d]+話'    # 第一話
        r'|プロローグ|エピローグ'
        r')\s*[^\n]*$'
    )

    boundaries = [m.start() for m in pattern.finditer(text)]
    if not boundaries:
        # 見出しが見つからない場合は全体を1章として扱う
        return [{"id": "chapter_1", "text": text}]

    chapters = []
    for i, start in enumerate(boundaries):
        end = boundaries[i + 1] if i + 1 < len(boundaries) else len(text)
        chapter_text = text[start:end].strip()
        chapters.append({
            "id": f"chapter_{i + 1}",
            "heading": chapter_text.split('\n')[0],
            "text": chapter_text
        })
    return chapters
```

---

## Step 2：場面分割

空行2行以上・時間経過表現・場所転換などを場面の境界として検出します。

```python
def split_scenes(chapter: dict) -> list[dict]:
    text = chapter["text"]

    # 空行2行以上で分割
    raw_scenes = re.split(r'\n{2,}', text)

    # 短すぎるブロックは前の場面に結合
    scenes = []
    buffer = ""
    for block in raw_scenes:
        buffer += "\n\n" + block if buffer else block
        if len(buffer) > 200:  # 200文字以上で1場面として確定
            scenes.append(buffer.strip())
            buffer = ""
    if buffer:
        scenes.append(buffer.strip())

    return [
        {"id": f"{chapter['id']}_scene_{i+1}", "text": s}
        for i, s in enumerate(scenes)
    ]
```

---

## Step 3：セリフ・地の文の分離

日本語のセリフは「」で囲まれている前提で分離します。

```python
def split_segments(scene: dict) -> list[dict]:
    text = scene["text"]
    segments = []
    pos = 0

    # 「」で囲まれたセリフを検出
    dialogue_pattern = re.compile(r'「(.*?)」', re.DOTALL)

    for m in dialogue_pattern.finditer(text):
        # セリフの前の地の文
        if m.start() > pos:
            narration = text[pos:m.start()].strip()
            if narration:
                segments.append({
                    "type": "narration",
                    "text": narration,
                    "speaker": None,
                    "scene_id": scene["id"]
                })

        # セリフ本体
        segments.append({
            "type": "dialogue",
            "text": m.group(1),
            "speaker": None,          # 次のStep 4で付与
            "scene_id": scene["id"]
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
                "scene_id": scene["id"]
            })

    return segments
```

---

## Step 4：話者タグ付け

地の文のパターンからセリフの直前・直後に話者を示す動詞句を検出します。

```python
SPEAKER_PATTERN = re.compile(
    r'([^\s、。「」]{1,10})'     # 人名（最大10文字）
    r'(?:は|が|も)?'
    r'(?:言った|言う|答えた|答える|叫んだ|叫ぶ|'
    r'囁いた|囁く|呟いた|呟く|続けた|続ける|'
    r'尋ねた|尋ねる|笑った|笑う|怒鳴った|怒鳴る)'
)

def assign_speakers(segments: list[dict], known_characters: list[str]) -> list[dict]:
    for i, seg in enumerate(segments):
        if seg["type"] != "dialogue":
            continue

        # セリフの直前・直後の地の文から話者を探す
        context = ""
        if i > 0 and segments[i-1]["type"] == "narration":
            context += segments[i-1]["text"][-50:]   # 直前50文字
        if i + 1 < len(segments) and segments[i+1]["type"] == "narration":
            context += segments[i+1]["text"][:50]    # 直後50文字

        m = SPEAKER_PATTERN.search(context)
        if m:
            candidate = m.group(1)
            # 既知キャラクター名と照合（部分一致も許容）
            for char in known_characters:
                if candidate in char or char in candidate:
                    seg["speaker"] = char
                    break
            else:
                seg["speaker"] = candidate  # 未知キャラとして記録

    return segments
```

---

## Step 5：LLMによる補完（ルールで解決できなかった場合）

ルールベースで話者が特定できなかったセリフをLLMに投げます。

```python
def resolve_unknown_speakers(segments, llm, known_characters):
    unresolved = [s for s in segments
                  if s["type"] == "dialogue" and s["speaker"] is None]
    if not unresolved:
        return segments

    # 前後の文脈をまとめてLLMに一括投げ
    context_window = 3  # 前後3セグメントを文脈として渡す

    for seg in unresolved:
        idx = segments.index(seg)
        context = segments[max(0, idx-context_window):idx+context_window+1]
        context_text = "\n".join(
            f"[{s['type']}] {s['speaker'] or '?'}: {s['text']}"
            for s in context
        )
        prompt = f"""
        以下の会話文脈で「?」の話者は誰ですか？
        登場人物候補: {known_characters}

        {context_text}

        話者名だけを返してください。
        特定不能なら「不明」と返してください。
        """
        result = llm.call(prompt).strip()
        seg["speaker"] = result if result != "不明" else None

    return segments
```

---

## 出力データ構造

最終的に以下のような構造になります。

```json
[
  {
    "type": "narration",
    "text": "朝の光が差し込む中、エレンは立ち上がった。",
    "speaker": null,
    "scene_id": "chapter_1_scene_1"
  },
  {
    "type": "dialogue",
    "text": "行かなければならない。",
    "speaker": "エレン",
    "scene_id": "chapter_1_scene_1"
  }
]
```

---

## 実用上の注意点

**ルールベースだけでは限界があります。** 体感で話者が特定できるのは全セリフの70〜80%程度で、残りはLLMに頼ることになります。論文の評価では「話者同定精度」を人手アノテーションと比較する実験を組むと、前処理の品質を定量的に示せます。

---

# 全文スキャン
順番に説明します。

---

## 全文スキャンの目的

全文スキャンはPath Bが章ごとの逐次更新を行う際の「ベースライン」を確立するための一度だけ実行される処理です。具体的には以下の3つを算出します。

```
① グローバル重心ベクトル   → 各キャラクターの「平均的な発言傾向」
② グローバル分散          → 発言スタイルの幅広さ・感情の揮発性
③ クラスタ構造            → キャラクターが持つ「顔の数」
                            （逐次更新では再計算しないため全文で確定）
```

---

## Step 1：全セリフの一括Embedding

```python
from sentence_transformers import SentenceTransformer
import numpy as np

model = SentenceTransformer("intfloat/multilingual-e5-large")
# 日本語対応・性能バランスが良い。
# 軽量版なら "paraphrase-multilingual-MiniLM-L12-v2"

def embed_all_utterances(segments_all_chapters: list[dict]) -> dict:
    """
    全章・全セリフをキャラクターごとにEmbeddingして返す。
    返り値: { character: [(vector, segment_meta), ...] }
    """
    # キャラクターごとにセリフを集約
    utterances_by_char = {}
    for seg in segments_all_chapters:
        if seg["type"] != "dialogue" or not seg["speaker"]:
            continue
        char = seg["speaker"]
        utterances_by_char.setdefault(char, [])
        utterances_by_char[char].append(seg)

    # キャラクターごとに一括Embedding
    embeddings_by_char = {}
    for char, segs in utterances_by_char.items():
        texts = [s["text"] for s in segs]

        # バッチ処理（メモリ効率のため一度に処理しすぎない）
        vectors = model.encode(
            texts,
            batch_size=64,
            show_progress_bar=False,
            normalize_embeddings=True  # コサイン類似度計算のためL2正規化
        )
        embeddings_by_char[char] = list(zip(vectors, segs))

    return embeddings_by_char
```

---

## Step 2：グローバルプロファイルの算出

```python
def compute_global_profile(embeddings_by_char: dict) -> dict:
    """
    キャラクターごとのグローバルプロファイルを算出する。
    """
    global_profiles = {}

    for char, vec_seg_pairs in embeddings_by_char.items():
        vectors = np.array([v for v, _ in vec_seg_pairs])

        global_profiles[char] = {
            # ① 全体重心：そのキャラクターの「平均的な発言の向き」
            "centroid": np.mean(vectors, axis=0),

            # ② 全体分散：発言の多様性（高い=感情が揺れやすい）
            "variance": float(np.var(vectors, axis=0).mean()),

            # ③ 発言数：後の逐次更新で章ごとの比率計算に使う
            "total_utterances": len(vectors),

            # ④ 章ごとの部分平均（逐次更新の初期値として使う）
            "chapter_centroids": compute_chapter_centroids(vec_seg_pairs),
        }

    return global_profiles


def compute_chapter_centroids(vec_seg_pairs: list) -> dict:
    """
    章ごとのセリフベクトル平均を算出する。
    逐次更新時の「前章までの状態」の初期化に使う。
    """
    by_chapter = {}
    for vec, seg in vec_seg_pairs:
        ch_id = seg["scene_id"].split("_scene_")[0]  # "chapter_3_scene_2" → "chapter_3"
        by_chapter.setdefault(ch_id, [])
        by_chapter[ch_id].append(vec)

    return {
        ch_id: np.mean(vecs, axis=0)
        for ch_id, vecs in by_chapter.items()
    }
```

---

## Step 3：クラスタ初期化

クラスタ構造は**全データが揃っている全文スキャン時にのみ計算**します。逐次更新では再計算しません（前回の議論での「案1」）。

```python
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

def initialize_clusters(embeddings_by_char: dict,
                         global_profiles: dict,
                         max_k: int = 5) -> dict:
    """
    キャラクターごとに最適なクラスタ数kを決定し、
    クラスタ構造を初期化する。

    クラスタ数 = キャラクターの「顔の数」
      k=1 → 一貫したキャラクター
      k=2 → 2つの顔を持つ（例：表と裏）
      k=3以上 → 場面ごとに大きく変わるキャラクター
    """
    cluster_profiles = {}

    for char, vec_seg_pairs in embeddings_by_char.items():
        vectors = np.array([v for v, _ in vec_seg_pairs])

        if len(vectors) < 10:
            # セリフが少なすぎる場合はクラスタリング不要
            cluster_profiles[char] = {
                "n_clusters": 1,
                "centroids": [global_profiles[char]["centroid"]],
                "labels": [0] * len(vectors)
            }
            continue

        # シルエットスコアで最適kを決定
        best_k, best_score = 1, -1
        for k in range(2, min(max_k + 1, len(vectors) // 5)):
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            labels = kmeans.fit_predict(vectors)
            score = silhouette_score(vectors, labels)
            if score > best_score:
                best_k, best_score = k, score

        # 最適kでクラスタリング確定
        kmeans = KMeans(n_clusters=best_k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(vectors)

        cluster_profiles[char] = {
            "n_clusters": best_k,
            "centroids": kmeans.cluster_centers_.tolist(),
            "labels": labels.tolist(),

            # どのクラスタがどの章に多く現れるかのマップ
            "cluster_chapter_distribution": compute_cluster_distribution(
                labels, [seg for _, seg in vec_seg_pairs]
            ),

            # クラスタの解釈ラベル（後でPath Aと統合して意味付け）
            "cluster_labels": [None] * best_k  # Path Aが後から埋める
        }

    return cluster_profiles


def compute_cluster_distribution(labels, segs) -> dict:
    """
    クラスタごとに、どの章に多く出現するかを集計する。
    「クラスタ0は主に前半、クラスタ1は後半に出現する」
    という傾向をPath Aが解釈する際に使う。
    """
    distribution = {}
    for label, seg in zip(labels, segs):
        ch_id = seg["scene_id"].split("_scene_")[0]
        distribution.setdefault(label, {})
        distribution[label][ch_id] = distribution[label].get(ch_id, 0) + 1
    return distribution
```

---

## Step 4：クラスタにPath Aで意味ラベルを付与

クラスタは数値ベクトルなので、それだけでは「このクラスタが何を意味するか」が分かりません。Path Aのサマリーと組み合わせて解釈します。

```python
def label_clusters_with_llm(cluster_profiles, embeddings_by_char,
                              global_profiles, llm):
    """
    各クラスタの代表セリフをLLMに見せて意味ラベルを付与する。
    これがPath A（LLM要約）とPath B（Embedding）の
    最初の統合ポイントになる。
    """
    for char, profile in cluster_profiles.items():
        if profile["n_clusters"] == 1:
            profile["cluster_labels"] = ["通常モード"]
            continue

        vec_seg_pairs = embeddings_by_char[char]
        labels_list = profile["labels"]

        labeled = []
        for cluster_id in range(profile["n_clusters"]):
            # そのクラスタに属するセリフを取得
            cluster_segs = [
                seg for (_, seg), label
                in zip(vec_seg_pairs, labels_list)
                if label == cluster_id
            ]
            # 代表セリフを最大5件取得
            representative = cluster_segs[:5]
            rep_texts = [s["text"] for s in representative]

            # 主に出現する章の情報
            ch_dist = profile["cluster_chapter_distribution"][cluster_id]
            main_chapters = sorted(ch_dist, key=ch_dist.get, reverse=True)[:3]

            prompt = f"""
            キャラクター「{char}」のセリフ群を
            クラスタリングした結果、このクラスタに分類されました。

            代表セリフ:
            {chr(10).join(f'- 「{t}」' for t in rep_texts)}

            主に出現する章: {main_chapters}

            このクラスタはキャラクターのどのような
            「状態・顔・モード」を表していますか？
            10文字以内のラベルを付けてください。
            例: 「攻撃モード」「内省モード」「信頼モード」
            """
            label_text = llm.call(prompt).strip()
            labeled.append(label_text)

        profile["cluster_labels"] = labeled

    return cluster_profiles
```

---

## 全文スキャンの最終出力

```python
def run_full_text_scan(segments_all_chapters, llm):
    # Step 1: 一括Embedding
    embeddings_by_char = embed_all_utterances(segments_all_chapters)

    # Step 2: グローバルプロファイル算出
    global_profiles = compute_global_profile(embeddings_by_char)

    # Step 3: クラスタ初期化
    cluster_profiles = initialize_clusters(embeddings_by_char, global_profiles)

    # Step 4: クラスタへの意味ラベル付与
    cluster_profiles = label_clusters_with_llm(
        cluster_profiles, embeddings_by_char, global_profiles, llm
    )

    return {
        "global_profiles": global_profiles,
        "cluster_profiles": cluster_profiles,
        "embeddings_by_char": embeddings_by_char  # 逐次更新でも使う
    }
```

出力例はこうなります。

```json
{
  "エレン": {
    "global_profiles": {
      "centroid": [0.12, -0.34, ...],
      "variance": 0.047,
      "total_utterances": 312
    },
    "cluster_profiles": {
      "n_clusters": 2,
      "cluster_labels": ["激情モード", "内省モード"],
      "cluster_chapter_distribution": {
        "0": {"chapter_1": 18, "chapter_2": 22, "chapter_8": 31},
        "1": {"chapter_4": 14, "chapter_6": 19, "chapter_9": 8}
      }
    }
  }
}
```

---

## Path Bの逐次更新との接続

全文スキャンの結果は章ごとのループで以下のように参照されます。

```python
def update_path_b_incremental(chapter, embeddings_by_char,
                               global_profiles, cluster_profiles):
    """
    章ごとのPath B逐次更新。
    クラスタ構造は変えず、重心・分散のみ更新する。
    """
    for char in global_profiles:
        chapter_vecs = get_chapter_vectors(chapter, char, embeddings_by_char)
        if not chapter_vecs:
            continue

        # 現在章までの累積重心を更新（逐次式）
        n_prev = global_profiles[char]["utterances_so_far"]
        n_new  = len(chapter_vecs)
        prev_centroid = global_profiles[char]["running_centroid"]

        # ウェルフォードの逐次平均更新
        new_centroid = (
            prev_centroid * n_prev + np.sum(chapter_vecs, axis=0)
        ) / (n_prev + n_new)

        global_profiles[char]["running_centroid"]   = new_centroid
        global_profiles[char]["utterances_so_far"] += n_new

        # 現在章のベクトルが全文クラスタのどれに近いかを判定
        # （クラスタ構造自体は変えない）
        for vec in chapter_vecs:
            distances = [
                np.linalg.norm(vec - np.array(c))
                for c in cluster_profiles[char]["centroids"]
            ]
            nearest_cluster = np.argmin(distances)
            # → TTSパラメタ生成時に「このセリフは激情モードに近い」
            #   という情報として使う
```

全文スキャンは「地図を作る作業」、章ごとの逐次更新は「地図の上を歩きながら現在地を更新する作業」というイメージです。地図（クラスタ構造）は最初に一度だけ作り、現在地（重心・分散）だけが随時更新されます。

---
# 知識DB更新
全文ループの設計と2軸構造を順番に説明します。

---

## 2軸構造の全体像

知識DBは「**事象マスタ**」と「**キャラクター知識状態**」の2軸で構成されます。

```
【軸1】事象マスタ（何が明らかになったか）
  event_001: "Chapter_3_scene_2で宝の場所が地図に示された"
  event_002: "Chapter_7_scene_1で犯人がエレンに正体を告白した"
  ...

【軸2】キャラクター知識状態（誰が何を知っているか）
  エレン  × event_001 → known: True,  learned_at: chapter_3
  エレン  × event_002 → known: True,  learned_at: chapter_7
  アルミン × event_001 → known: False, learned_at: None
  アルミン × event_002 → known: False, learned_at: None
  ...
```

この2軸が交差することで「誰がいつ何を知ったか」が一意に追跡できます。

---

## データ構造の定義

```python
from dataclasses import dataclass, field
from typing import Optional

@dataclass
class Event:
    event_id: str
    description: str          # 何が明らかになったか
    scene_id: str             # どの場面で発生したか
    chapter_id: str

@dataclass
class KnowledgeEntry:
    known: bool
    learned_at: Optional[str]         # chapter_id
    evidence: Optional[str]           # 根拠となったセリフ・描写
    retroactive: bool = False         # 遡及修正されたか
    originally_detected_at: Optional[str] = None  # 遡及前の検出章

# 事象マスタ
event_master: dict[str, Event] = {}

# キャラクター知識状態
# knowledge_db[character][event_id] = KnowledgeEntry
knowledge_db: dict[str, dict[str, KnowledgeEntry]] = {}
```

---

## 全文ループの処理フロー

```python
def run_knowledge_db_loop(chapters, segments_by_chapter,
                           event_master, known_characters, llm):
    # キャラクターごとに全事象をFalseで初期化
    initialize_knowledge_db(known_characters, event_master)

    for chapter in chapters:
        scenes = segments_by_chapter[chapter.id]

        for scene in scenes:
            # ① 登場人物を特定
            present = detect_present_characters(scene, known_characters)

            # ② 未解決フラグのみ判定対象に絞る
            unresolved = get_unresolved_events(present, event_master)
            if not unresolved:
                continue

            # ③ 登場→知得の2段階判定
            for character in present:
                for event_id in unresolved[character]:
                    result = judge_knowledge_acquisition(
                        scene, character, event_master[event_id], llm
                    )
                    if result["learned"]:
                        knowledge_db[character][event_id] = KnowledgeEntry(
                            known=True,
                            learned_at=chapter.id,
                            evidence=result["evidence"]
                        )

        # ④ 章末に遡及検出を実行
        retroactive_updates = detect_retroactive_knowledge(
            chapter, knowledge_db, event_master, llm
        )
        apply_retroactive_updates(retroactive_updates)
```

---

## ① 登場人物の特定

```python
def detect_present_characters(scene, known_characters):
    """
    場面テキストに登場する人物を特定する。
    セリフの話者タグと地の文の人名言及の両方を確認する。
    """
    present = set()

    for seg in scene:
        # 話者タグから直接取得
        if seg["type"] == "dialogue" and seg["speaker"]:
            present.add(seg["speaker"])

        # 地の文に人名が含まれるか
        for char in known_characters:
            if char in seg["text"]:
                present.add(char)

    return list(present)
```

---

## ② 未解決フラグの絞り込み

```python
def get_unresolved_events(present_characters, event_master):
    """
    登場人物ごとに、まだ知得していない事象のみを返す。
    既知フラグは再判定しない（コスト削減）。
    """
    unresolved = {}
    for character in present_characters:
        unresolved[character] = [
            event_id
            for event_id, entry in knowledge_db[character].items()
            if not entry.known
        ]
    return unresolved
```

---

## ③ 登場→知得の2段階判定

「その場にいた」と「知り得た」を**あえて別のプロンプトで判定**します。1つにまとめると精度が落ちます。

```python
def judge_knowledge_acquisition(scene, character, event, llm):
    scene_text = "\n".join(seg["text"] for seg in scene)

    # 第1段階：登場確認（軽量・ルールベース優先）
    # detect_present_characters で既に確認済みなのでここでは省略

    # 第2段階：知得判定（LLMが必要）
    prompt = f"""
    場面テキスト:
    {scene_text}

    キャラクター「{character}」はこの場面で
    以下の事実を知り得ましたか？

    事実: {event.description}

    判定基準:
    - 直接目撃・聴取した場合 → True
    - その場にいたが知覚できなかった場合（気絶・別室など） → False
    - 伝聞・推測のみの場合 → False（推測は知識として扱わない）

    以下のJSON形式で返してください:
    {{
      "learned": true/false,
      "evidence": "根拠となった文章を原文から抜粋"
    }}
    """
    return llm.call_json(prompt)
```

---

## ④ 遡及検出と修正

```python
def detect_retroactive_knowledge(chapter, knowledge_db, event_master, llm):
    """
    「実は前から知っていた」という記述を検出し、
    過去エントリを遡及修正する。
    """
    chapter_text = chapter.text

    prompt = f"""
    以下の章テキストに、あるキャラクターが
    「実は以前の場面から知っていた」ことを示す記述はありますか？

    章テキスト:
    {chapter_text}

    あれば以下のJSON形式で返してください（なければ空リスト）:
    [
      {{
        "character": "キャラクター名",
        "fact": "知っていた事実の説明",
        "estimated_from_chapter": "最も早い章のID（不明なら null）",
        "evidence": "根拠となった文章"
      }}
    ]
    """
    candidates = llm.call_json(prompt)
    return candidates


def apply_retroactive_updates(candidates, event_master, knowledge_db):
    for c in candidates:
        # 事象マスタ内の対応するevent_idを特定
        matched_event_id = match_event(c["fact"], event_master)
        if not matched_event_id:
            continue

        character = c["character"]
        entry = knowledge_db[character][matched_event_id]

        if not entry.known or c["estimated_from_chapter"]:
            old_chapter = entry.learned_at
            new_chapter = c["estimated_from_chapter"]

            knowledge_db[character][matched_event_id] = KnowledgeEntry(
                known=True,
                learned_at=new_chapter,
                evidence=c["evidence"],
                retroactive=True,
                originally_detected_at=old_chapter
            )
```

---

## 最終的なknowledge_dbの出力例

```json
{
  "エレン": {
    "event_001": {
      "known": true,
      "learned_at": "chapter_3",
      "evidence": "エレンは地図を受け取り、場所を確認した。",
      "retroactive": false
    },
    "event_002": {
      "known": true,
      "learned_at": "chapter_3",
      "evidence": "「ずっと知っていた」とエレンは静かに言った。",
      "retroactive": true,
      "originally_detected_at": "chapter_7"
    }
  },
  "アルミン": {
    "event_001": {
      "known": false,
      "learned_at": null,
      "evidence": null,
      "retroactive": false
    }
  }
}
```

---

## 実用上の注意点

**事象マスタの粒度設計が品質の鍵です。** 粒度が大きすぎると（例：「物語の謎が解けた」）判定が曖昧になり、小さすぎると（例：「エレンが3ページ目で右を向いた」）DBが膨大になります。論文の検証実験では**物語の展開に影響する情報のみ**を対象にする、という基準で統一するのが現実的です。

---
# 遡及検出・過去エントリの修正
前回の実装を踏まえて、3段階を順番に説明します。

---

## 3段階の概要

```
Stage 1: 遡及的知識の検出
         ↓ （遡及事例が見つかった場合のみ）
Stage 2: 知識DBの遡及修正
         ↓
Stage 3: 影響範囲の特定と再処理フラグの付与
```

---

## Stage 1：遡及的知識の検出

**何をするか**
章テキストを読んで「このキャラクターは実は以前から知っていた」という記述を検出します。検出パターンは3種類あります。

```python
def detect_retroactive_knowledge_stage1(chapter, knowledge_db,
                                        event_master, llm):
    """
    遡及的知識の検出。3つのパターンに分けて検出する。
    """
    chapter_text = chapter.text
    candidates = []

    # パターンA: 明示的な告白・回想
    # 「ずっと知っていた」「あの時から気づいていた」など
    pattern_a = detect_explicit_retroactive(chapter_text, llm)

    # パターンB: 行動の矛盾
    # 知らないはずのキャラクターが知っているかのように行動している
    pattern_b = detect_behavioral_contradiction(
        chapter_text, knowledge_db, event_master, llm
    )

    # パターンC: 他キャラクターによる証言
    # 「あいつはあの時から知っていたはずだ」という第三者の発言
    pattern_c = detect_third_party_testimony(chapter_text, llm)

    candidates = pattern_a + pattern_b + pattern_c

    # 重複除去（同一キャラ×同一事実の重複を統合）
    return deduplicate_candidates(candidates)


def detect_explicit_retroactive(chapter_text, llm):
    prompt = f"""
    以下の章テキストに、キャラクターが
    「実は以前から知っていた・気づいていた」ことを
    明示的に示す表現はありますか？

    検出対象の表現例:
    - 「ずっと知っていた」「前から気づいていた」
    - 回想シーン（「あの時、私は〜を見ていた」）
    - 告白（「実は〜のことは知っていたんだ」）

    章テキスト:
    {chapter_text}

    JSON形式で返してください（なければ空リスト）:
    [
      {{
        "character": "キャラクター名",
        "fact": "知っていた事実",
        "pattern": "A",
        "estimated_known_from": "推定される最初の知得章（不明ならnull）",
        "evidence_text": "根拠となった原文抜粋",
        "confidence": 0.0〜1.0
      }}
    ]
    """
    return llm.call_json(prompt)


def detect_behavioral_contradiction(chapter_text, knowledge_db,
                                    event_master, llm):
    """
    「知らないはず」なのに知っているかのように振る舞っている
    矛盾した行動を検出する。
    """
    # 現時点でknown=Falseのキャラクター×事象の組み合わせを取得
    unknown_facts = [
        (char, event_id, event_master[event_id].description)
        for char, events in knowledge_db.items()
        for event_id, entry in events.items()
        if not entry.known
    ]

    if not unknown_facts:
        return []

    unknown_summary = "\n".join(
        f"- {char}は「{desc}」をまだ知らないはず"
        for char, _, desc in unknown_facts
    )

    prompt = f"""
    以下のキャラクターは現時点でこれらの事実を「知らないはず」です:
    {unknown_summary}

    しかし以下の章テキストで、
    これらの事実を知っているかのような行動・発言をしている
    キャラクターはいますか？

    章テキスト:
    {chapter_text}

    JSON形式で返してください（なければ空リスト）:
    [
      {{
        "character": "キャラクター名",
        "fact": "矛盾している事実",
        "pattern": "B",
        "estimated_known_from": null,
        "evidence_text": "矛盾を示す原文抜粋",
        "confidence": 0.0〜1.0
      }}
    ]
    """
    return llm.call_json(prompt)


def detect_third_party_testimony(chapter_text, llm):
    prompt = f"""
    以下の章テキストに、第三者が
    「あのキャラクターは以前からこの事実を知っていたはずだ」
    という趣旨の発言・描写はありますか？

    章テキスト:
    {chapter_text}

    JSON形式で返してください（なければ空リスト）:
    [
      {{
        "character": "知っていたとされるキャラクター名",
        "fact": "知っていたとされる事実",
        "pattern": "C",
        "estimated_known_from": null,
        "evidence_text": "根拠となった原文抜粋",
        "confidence": 0.0〜1.0
      }}
    ]
    """
    return llm.call_json(prompt)
```

---

## Stage 2：知識DBの遡及修正

**何をするか**
Stage 1で検出した候補を実際にDBに書き込みます。信頼度スコアによるフィルタリングと、修正前後の記録を残すことが重要です。

```python
CONFIDENCE_THRESHOLD = 0.7  # これ未満は修正しない

def apply_retroactive_updates_stage2(candidates, knowledge_db,
                                     event_master, current_chapter_id):
    """
    知識DBを遡及修正する。
    修正前の状態を必ず保存し、追跡可能にする。
    """
    affected_ranges = []  # Stage 3で使う影響範囲

    for candidate in candidates:
        # 信頼度が低い場合はスキップ
        if candidate["confidence"] < CONFIDENCE_THRESHOLD:
            continue

        character = candidate["character"]
        matched_event_id = match_event_to_master(
            candidate["fact"], event_master
        )
        if not matched_event_id:
            # 事象マスタに対応する事象がない場合は新規登録
            matched_event_id = register_new_event(
                candidate["fact"], current_chapter_id, event_master
            )

        current_entry = knowledge_db[character][matched_event_id]
        old_learned_at = current_entry.learned_at
        new_learned_at = candidate["estimated_known_from"]

        # 遡及修正が実際に「より早い章」を指している場合のみ適用
        if should_apply_retroactive(old_learned_at, new_learned_at):
            knowledge_db[character][matched_event_id] = KnowledgeEntry(
                known=True,
                learned_at=new_learned_at,
                evidence=candidate["evidence_text"],
                retroactive=True,
                originally_detected_at=old_learned_at
            )

            affected_ranges.append({
                "character": character,
                "event_id": matched_event_id,
                "from_chapter": new_learned_at,
                "to_chapter": old_learned_at,  # 修正前の検出章まで
                "pattern": candidate["pattern"]
            })

    return affected_ranges


def should_apply_retroactive(old_learned_at, new_learned_at):
    """
    new_learned_at が old_learned_at より前の章かを判定する。
    """
    if new_learned_at is None:
        return False  # 推定章が不明な場合は適用しない
    if old_learned_at is None:
        return True   # 元々未検出だった場合は適用する

    # chapter_id が "chapter_3" 形式の場合
    old_num = int(old_learned_at.split("_")[-1])
    new_num = int(new_learned_at.split("_")[-1])
    return new_num < old_num
```

---

## Stage 3：影響範囲の特定と再処理フラグの付与

**何をするか**
遡及修正によって「知識状態が変わった章の範囲」を特定し、第2パス（TTSパラメタ生成）で該当セリフを正しく処理するためのフラグを付与します。

```python
def flag_affected_utterances_stage3(affected_ranges, segments_by_chapter):
    """
    遡及修正の影響を受けるセリフに
    「知識を隠している」フラグを付与する。
    """
    for affected in affected_ranges:
        character   = affected["character"]
        event_id    = affected["event_id"]
        from_ch     = affected["from_chapter"]
        to_ch       = affected["to_chapter"]

        # 影響章の範囲を算出
        affected_chapters = get_chapter_range(from_ch, to_ch)

        for chapter_id in affected_chapters:
            if chapter_id not in segments_by_chapter:
                continue

            for seg in segments_by_chapter[chapter_id]:
                if seg["speaker"] != character:
                    continue
                if seg["type"] != "dialogue":
                    continue

                # 知識制御フラグを付与
                seg.setdefault("knowledge_control", {})
                seg["knowledge_control"][event_id] = {
                    "hiding_knowledge": True,
                    "known_since": from_ch,
                    "suppression_weight": calc_suppression_weight(
                        affected["pattern"]
                    )
                }


def calc_suppression_weight(pattern: str) -> float:
    """
    検出パターンによって感情抑制の強さを変える。

    パターンAは「ずっと知っていた」と明示されているので
    強く抑制する。
    パターンBは行動の矛盾から推測なので弱めに抑制する。
    """
    return {
        "A": 0.7,  # 明示的告白 → 強く抑制
        "B": 0.4,  # 行動矛盾   → 中程度抑制
        "C": 0.3,  # 第三者証言 → 弱めに抑制
    }.get(pattern, 0.4)
```

---

## 3段階の出力と第2パスへの接続

Stage 3まで完了すると、各セリフは以下のような状態になります。

```json
{
  "type": "dialogue",
  "speaker": "エレン",
  "text": "何も知らないよ、そんなこと。",
  "scene_id": "chapter_5_scene_2",
  "knowledge_control": {
    "event_002": {
      "hiding_knowledge": true,
      "known_since": "chapter_3",
      "suppression_weight": 0.7
    }
  }
}
```

第2パスのTTSパラメタ生成はこのフラグを参照し、`suppression_weight`に応じて感情値を抑制します。

```python
def apply_knowledge_suppression(tts_params, seg):
    controls = seg.get("knowledge_control", {})
    if not controls:
        return tts_params

    # 最大の抑制重みを適用
    max_weight = max(c["suppression_weight"] for c in controls.values())

    tts_params["emotion_intensity"] *= (1.0 - max_weight)
    tts_params["speech_rate"]       *= (1.0 + max_weight * 0.1)  # 少し早口に
    tts_params["pitch_variation"]   *= (1.0 - max_weight * 0.3)  # 抑揚を抑える

    return tts_params
```

---

## 精度の現実的な評価

| 検出パターン | 自動検出精度（目安） | 理由 |
|---|---|---|
| A：明示的告白 | 高（80〜90%） | 表現パターンが明確 |
| B：行動矛盾 | 中（50〜65%） | 文脈依存が強くLLMが迷いやすい |
| C：第三者証言 | 中（60〜75%） | 発話の対象が曖昧なことがある |

パターンBの精度が最も低くなります。論文では「パターン別の再現率・適合率」を評価指標に加えると、Stage 1の設計の丁寧さが査読に伝わります。

---
# pathA:LLM要約
順番に説明します。

---

## 「観察行動ベース」の意味

前回の議論で確認した通り、Path Aのプロンプトは**知識状態に依存せず、観察可能な行動・発言のみを入力**として設計します。

```
❌ 知識依存の設計（避けるべき）
「エレンは犯人の正体を知っている。それを踏まえてこの章での
 性格変化を分析せよ」
→ 知識DBの遡及修正が性格履歴に波及する

✅ 観察行動ベースの設計（正しい）
「この章でエレンが実際に取った行動・発したセリフのみから
 性格変化を分析せよ」
→ 知識DBと完全に独立
```

---

## データ構造

```python
from dataclasses import dataclass, field
from typing import Optional

@dataclass
class PersonalityState:
    # 性格パラメタ（0.0〜1.0）
    aggression:   float = 0.5
    loyalty:      float = 0.5
    anxiety:      float = 0.5
    openness:     float = 0.5
    confidence:   float = 0.5

    # サマリー（Path AのLLM出力）
    summary: str = ""

    # 更新の根拠（査読対策・評価用）
    update_reason: str = ""
    chapter_id: str = ""

# キャラクターごとの性格履歴
# personality_history[character][chapter_id] = PersonalityState
personality_history: dict[str, dict[str, PersonalityState]] = {}
```

---

## メインループ：章ごとの逐次更新

```python
def run_personality_loop(chapters, segments_by_chapter,
                          known_characters, llm):
    # 全キャラクターを中間値で初期化
    for char in known_characters:
        personality_history[char] = {}

    for chapter in chapters:
        segments = segments_by_chapter[chapter.id]

        for char in known_characters:
            # そのキャラクターがこの章に登場するか確認
            if not character_appears_in_chapter(segments, char):
                # 登場しない章では前章の状態を引き継ぐ
                prev = get_previous_state(char, chapter.id)
                if prev:
                    personality_history[char][chapter.id] = prev
                continue

            # 前章までの性格サマリーを取得
            prev_state = get_previous_state(char, chapter.id)

            # Path A：LLMによる観察行動ベース更新
            path_a_result = update_path_a(
                char, chapter, segments, prev_state, llm
            )

            # Path B：Embedding逐次更新（別モジュール）
            path_b_result = update_path_b_incremental(
                char, chapter, segments
            )

            # 矛盾検出・統合（別モジュール）
            confirmed_state = detect_and_fuse(
                path_a_result, path_b_result, prev_state
            )

            personality_history[char][chapter.id] = confirmed_state

    return personality_history
```

---

## Path A のコア：3段階プロンプト設計

1回のLLM呼び出しで全部やろうとすると精度が落ちます。**観察→変化抽出→スコア更新**の3段階に分けます。

### 第1段階：観察可能な行動・発言の抽出

```python
def extract_observable_behaviors(char, segments, llm) -> str:
    """
    その章でキャラクターが実際に示した行動・発言を抽出する。
    解釈や推測を含まず、テキストから直接読み取れる事実のみ。
    """
    # キャラクターのセリフと、そのキャラクターが登場する地の文を抽出
    char_segments = [
        seg for seg in segments
        if seg["speaker"] == char
        or (seg["type"] == "narration" and char in seg["text"])
    ]

    if not char_segments:
        return ""

    seg_text = "\n".join(
        f"[{'セリフ' if s['type'] == 'dialogue' else '描写'}] {s['text']}"
        for s in char_segments
    )

    prompt = f"""
    以下はキャラクター「{char}」に関するこの章のセリフと描写です。

    {seg_text}

    以下のルールに従って、観察可能な事実のみを箇条書きで抽出してください。

    ルール:
    - テキストに明示されている行動・発言のみを書く
    - 「なぜそうしたか」の推測は書かない
    - 知識・情報の有無に関する記述は含めない
    - 他キャラクターへの態度・反応を具体的に記述する

    出力形式:
    - （行動・発言の事実を1つずつ記述）
    """
    return llm.call(prompt)
```

### 第2段階：性格変化の差分抽出

```python
def extract_personality_delta(char, behaviors, prev_state, llm) -> dict:
    """
    観察された行動から、前章との性格変化の差分を抽出する。
    前章サマリーとの比較で「何が変わったか」だけを言語化する。
    """
    prev_summary = prev_state.summary if prev_state else "（初登場・情報なし）"

    prompt = f"""
    キャラクター「{char}」について分析します。

    【前章までの性格サマリー】
    {prev_summary}

    【今章での観察された行動・発言】
    {behaviors}

    前章と比較して、以下の観点で「変化があった点のみ」を
    記述してください。変化がない項目は記述不要です。

    分析観点:
    1. 他者への態度・接し方の変化
    2. 感情表現の変化（表に出やすくなった・抑えるようになったなど）
    3. 意思決定のパターンの変化（慎重になった・大胆になったなど）
    4. 対人スタンスの変化（信頼・疑念・距離感など）

    重要: 変化の根拠として「どの行動・発言から読み取れるか」を
    必ず添えてください。
    """
    return llm.call(prompt)
```

### 第3段階：パラメタ数値の更新

```python
def update_personality_scores(char, delta_text,
                               prev_state, chapter_id, llm) -> PersonalityState:
    """
    差分テキストを受けて数値パラメタを更新する。
    前章の値からの増減幅を制限することで急変を防ぐ。
    """
    prev_scores = {
        "aggression":  prev_state.aggression  if prev_state else 0.5,
        "loyalty":     prev_state.loyalty     if prev_state else 0.5,
        "anxiety":     prev_state.anxiety     if prev_state else 0.5,
        "openness":    prev_state.openness    if prev_state else 0.5,
        "confidence":  prev_state.confidence  if prev_state else 0.5,
    }

    prompt = f"""
    キャラクター「{char}」の今章での性格変化:
    {delta_text}

    前章末時点の性格スコア（0.0〜1.0）:
    {prev_scores}

    今章末時点の新しいスコアをJSONで返してください。

    制約:
    - 各スコアは前章の値から±0.15以内の変化に留めること
      （急激な人格変化を防ぐため）
    - 変化の根拠が明確な項目のみ変動させること
    - 根拠がない項目は前章の値をそのまま使うこと

    出力形式:
    {{
      "aggression":  数値,
      "loyalty":     数値,
      "anxiety":     数値,
      "openness":    数値,
      "confidence":  数値,
      "update_reason": "変化した項目とその根拠を1〜2文で"
    }}
    """
    result = llm.call_json(prompt)

    # ±0.15制限をコードでも保証（LLMが守らない場合に備えて）
    MAX_DRIFT = 0.15
    clamped = {}
    for key in prev_scores:
        raw = result.get(key, prev_scores[key])
        clamped[key] = float(np.clip(
            raw,
            prev_scores[key] - MAX_DRIFT,
            prev_scores[key] + MAX_DRIFT
        ))

    return PersonalityState(
        **clamped,
        update_reason=result.get("update_reason", ""),
        chapter_id=chapter_id
    )
```

---

## 3段階をまとめたPath A更新関数

```python
def update_path_a(char, chapter, segments, prev_state, llm) -> PersonalityState:
    """
    Path A のメイン処理。3段階プロンプトで逐次更新する。
    """
    # 第1段階：観察事実の抽出
    behaviors = extract_observable_behaviors(char, segments, llm)
    if not behaviors:
        return prev_state  # 観察事実がなければ前章を引き継ぐ

    # 第2段階：変化差分の抽出
    delta = extract_personality_delta(char, behaviors, prev_state, llm)

    # 第3段階：スコア更新
    new_state = update_personality_scores(
        char, delta, prev_state, chapter.id, llm
    )

    # 更新サマリーを生成（次章のPath Aに渡すため）
    new_state.summary = build_summary(char, new_state, prev_state, llm)

    return new_state


def build_summary(char, new_state, prev_state, llm) -> str:
    """
    次章のPath Aに渡す「前章サマリー」を生成する。
    これが逐次的に書き換わっていくLLMサマリーの実体。
    """
    prompt = f"""
    キャラクター「{char}」の現時点での性格を
    声優への演技指示書として3〜4文でまとめてください。

    性格スコア:
    - 攻撃性: {new_state.aggression:.2f}
    - 忠誠心: {new_state.loyalty:.2f}
    - 不安傾向: {new_state.anxiety:.2f}
    - 開放性: {new_state.openness:.2f}
    - 自信: {new_state.confidence:.2f}

    今章での変化: {new_state.update_reason}

    出力は「{char}は〜」で始めてください。
    知識・情報の有無には言及しないでください。
    """
    return llm.call(prompt)
```

---

## サマリーの逐次変化イメージ

```
Chapter 1:
「エレンは感情が表に出やすく、仲間への強い思いを
 直接的な言葉で表現する。やや衝動的だが迷いは少ない。」

Chapter 5（裏切りを目撃した後）:
「エレンは怒りを内側に溜め込むようになり、
 発言が短く鋭くなっている。仲間への信頼は揺らいでいるが、
 表面上は普段通りを装おうとしている。」

Chapter 12（覚悟を固めた後）:
「エレンは感情を完全に制御下に置いており、
 発言は簡潔で目的指向的になっている。
 かつての衝動性は失われ、冷静さと諦念が共存している。」
```

このサマリーが次章のPath Aへの入力として渡され、「前章との差分のみを書き換える」という逐次処理の連鎖が成立します。

---

## 実装上の注意点

**LLMの呼び出し回数について**
3段階設計では1キャラクター×1章あたり最大4回（抽出・差分・スコア・サマリー）のLLM呼び出しが発生します。20章・5キャラクターなら最大400回です。コストを抑えるには第1・2段階を1回のプロンプトに統合することも可能ですが、精度とのトレードオフになります。論文の検証実験では「3段階版」と「統合版」をアブレーション比較することで、段階分割の有効性を示せます。

**ハルシネーション対策**
`update_reason`に根拠文を必ず含めさせることで、Path Bとの矛盾検出時の追跡可能性が高まります。また第3段階の±0.15制限はコードとプロンプトの両方で二重に保証しています。LLMは制約を守らないことがあるため、コード側での強制が必須です。

---
# pathB:embedding
順番に説明します。

---

## 逐次更新の基本原理

全データを毎回再計算せず、**前章までの集計値に今章分を加算する**だけで更新できます。これをウェルフォードのオンラインアルゴリズムと呼びます。

```
【通常の平均計算】全セリフを毎章読み直す → O(n) × 章数
【逐次更新】      前章の平均に今章分を加算する → O(1)
```

重心・分散・共分散の3つがすべてこの方式で更新できます。

---

## データ構造

```python
from dataclasses import dataclass, field
import numpy as np

@dataclass
class PathBState:
    # 逐次更新に必要な累積値
    n: int = 0                        # これまでのセリフ総数
    centroid: np.ndarray = None       # 現在の重心ベクトル
    M2: np.ndarray = None             # 分散計算用累積値（ウェルフォード法）

    # 章ごとのスナップショット（TTSパラメタ生成時に参照）
    chapter_snapshots: dict = field(default_factory=dict)

    # 全文スキャンで確定したクラスタ構造（変更しない）
    cluster_centroids: list = None    # shape: (k, embedding_dim)
    cluster_labels_text: list = None  # ["激情モード", "内省モード", ...]

    def __post_init__(self):
        if self.centroid is None:
            self.centroid = np.zeros(1024)  # multilingual-e5-largeの次元数
        if self.M2 is None:
            self.M2 = np.zeros(1024)

# キャラクターごとのPath B状態
path_b_states: dict[str, PathBState] = {}
```

---

## Step 1：ウェルフォード法による重心・分散の逐次更新

```python
def welford_update(state: PathBState, new_vector: np.ndarray) -> PathBState:
    """
    新しいセリフベクトルを1つ受け取り、
    重心と分散をO(1)で更新する。

    ウェルフォードのオンラインアルゴリズム:
    n += 1
    delta  = x - mean
    mean  += delta / n
    delta2 = x - mean  （更新後のmeanを使う）
    M2    += delta * delta2
    variance = M2 / n
    """
    state.n += 1
    delta  = new_vector - state.centroid
    state.centroid += delta / state.n
    delta2 = new_vector - state.centroid  # 更新後の重心で再計算
    state.M2 += delta * delta2

    return state


def get_current_variance(state: PathBState) -> float:
    """
    現時点での分散スカラー値を返す。
    （ベクトルの各次元の分散の平均）
    """
    if state.n < 2:
        return 0.0
    return float(np.mean(state.M2 / state.n))
```

---

## Step 2：章単位の更新処理

```python
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("intfloat/multilingual-e5-large")

def update_path_b_for_chapter(char: str,
                                chapter_id: str,
                                segments: list[dict],
                                path_b_states: dict) -> PathBState:
    """
    1章分のセリフをまとめて受け取り、Path B状態を更新する。
    """
    state = path_b_states[char]

    # そのキャラクターのセリフのみ抽出
    char_utterances = [
        seg["text"] for seg in segments
        if seg["type"] == "dialogue" and seg["speaker"] == char
    ]

    if not char_utterances:
        # 登場しない章でもスナップショットは保存する
        save_chapter_snapshot(state, chapter_id)
        return state

    # バッチEmbedding
    vectors = model.encode(
        char_utterances,
        batch_size=64,
        normalize_embeddings=True
    )

    # セリフごとにウェルフォード更新
    for vec in vectors:
        state = welford_update(state, vec)

    # 今章でどのクラスタに近いセリフが多かったかを記録
    chapter_cluster_distribution = compute_chapter_cluster_affinity(
        vectors, state.cluster_centroids
    )

    # 章末のスナップショットを保存
    save_chapter_snapshot(
        state, chapter_id, chapter_cluster_distribution, vectors
    )

    return state
```

---

## Step 3：クラスタ親和性の計算

章ごとに「どのクラスタ（顔）が多く出現したか」を記録します。これがTTSパラメタ生成時の「このセリフは激情モードに近い」という情報の根拠になります。

```python
def compute_chapter_cluster_affinity(vectors: np.ndarray,
                                      cluster_centroids: list) -> dict:
    """
    今章のセリフ群が各クラスタにどれだけ近いかを計算する。
    クラスタ構造自体は変えず、「親和度」のみを計算する。
    """
    if cluster_centroids is None or len(vectors) == 0:
        return {}

    centroids = np.array(cluster_centroids)
    affinities = {i: 0.0 for i in range(len(centroids))}

    for vec in vectors:
        # 各クラスタ重心とのコサイン類似度を計算
        similarities = np.dot(centroids, vec)  # L2正規化済みなのでそのまま内積

        # ソフトアサインメント（最近傍のみでなく全クラスタへの重みを計算）
        weights = softmax(similarities)
        for i, w in enumerate(weights):
            affinities[i] += float(w)

    # 正規化
    total = sum(affinities.values())
    return {i: v / total for i, v in affinities.items()}


def softmax(x: np.ndarray, temperature: float = 0.5) -> np.ndarray:
    """
    temperatureが低いほど最近傍クラスタへの集中度が高まる。
    0.5はソフトとハードの中間。
    """
    x = x / temperature
    x = x - np.max(x)  # 数値安定性のためのシフト
    exp_x = np.exp(x)
    return exp_x / exp_x.sum()
```

---

## Step 4：章末スナップショットの保存

```python
def save_chapter_snapshot(state: PathBState,
                           chapter_id: str,
                           cluster_affinity: dict = None,
                           chapter_vectors: np.ndarray = None):
    """
    章末時点のPath B状態を保存する。
    第2パス（TTSパラメタ生成）でこのスナップショットを参照する。
    """
    snapshot = {
        "chapter_id":          chapter_id,
        "n_utterances_so_far": state.n,
        "centroid":            state.centroid.copy(),
        "variance":            get_current_variance(state),

        # そのクラスタとの距離（TTSの音声モード選択に使う）
        "cluster_affinity":    cluster_affinity or {},

        # 今章内での重心（章内の一貫性チェックに使う）
        "chapter_centroid":    np.mean(chapter_vectors, axis=0).tolist()
                               if chapter_vectors is not None
                               and len(chapter_vectors) > 0
                               else None,
    }
    state.chapter_snapshots[chapter_id] = snapshot
```

---

## Step 5：Path Aとの矛盾検出への接続

Path Bのスナップショットと、Path Aのスコア・サマリーを比較して矛盾を検出します。

```python
def compute_path_ab_contradiction(path_a_state,
                                   path_b_snapshot,
                                   char: str) -> dict:
    """
    Path AとPath Bの矛盾スコアを計算する。

    比較ポイント:
    1. Path AサマリーのEmbeddingとPath B重心のコサイン類似度
    2. クラスタ親和性とPath Aスコアの整合性
    """
    # ① サマリーEmbeddingと重心の比較
    summary_vec = model.encode(
        [path_a_state.summary],
        normalize_embeddings=True
    )[0]

    centroid = np.array(path_b_snapshot["centroid"])
    centroid = centroid / (np.linalg.norm(centroid) + 1e-8)

    cosine_sim = float(np.dot(summary_vec, centroid))
    # 1.0に近い = 矛盾なし
    # 0.5以下   = 矛盾あり → LLMに再生成を要求

    # ② クラスタ親和性とPath Aスコアの整合性チェック
    affinity_consistency = check_affinity_score_consistency(
        path_b_snapshot["cluster_affinity"],
        path_a_state
    )

    contradiction_score = 1.0 - cosine_sim  # 高いほど矛盾

    return {
        "cosine_similarity":      cosine_sim,
        "contradiction_score":    contradiction_score,
        "affinity_consistency":   affinity_consistency,
        "needs_regeneration":     contradiction_score > 0.4
                                  or not affinity_consistency
    }


def check_affinity_score_consistency(cluster_affinity: dict,
                                      path_a_state,
                                      cluster_labels: list) -> bool:
    """
    最も親和度が高いクラスタと、Path Aのスコアが
    概念的に整合しているかを簡易チェックする。

    例: 最も親和度が高いクラスタ = "激情モード"
        Path A の aggression = 0.2（低い）
        → 矛盾あり
    """
    if not cluster_affinity:
        return True  # クラスタなしなら整合性チェック不要

    dominant_cluster_id = max(cluster_affinity, key=cluster_affinity.get)
    dominant_label = cluster_labels[dominant_cluster_id]

    # 簡易ルール（論文では機械学習で精緻化可能）
    rules = {
        "激情モード":  lambda s: s.aggression > 0.6,
        "内省モード":  lambda s: s.anxiety > 0.5 or s.openness < 0.4,
        "信頼モード":  lambda s: s.loyalty > 0.6 and s.anxiety < 0.4,
        "攻撃モード":  lambda s: s.aggression > 0.7,
    }

    rule = rules.get(dominant_label)
    if rule is None:
        return True  # 未定義ルールは整合性ありとみなす

    return rule(path_a_state)
```

---

## 全体の呼び出しフロー

```python
def run_path_b_chapter(char, chapter_id, segments,
                        path_b_states, cluster_profiles):
    """
    章ごとのPath B処理の統合エントリポイント。
    """
    # クラスタ情報を状態に持たせる（全文スキャンで確定済み）
    if path_b_states[char].cluster_centroids is None:
        path_b_states[char].cluster_centroids = \
            cluster_profiles[char]["centroids"]
        path_b_states[char].cluster_labels_text = \
            cluster_profiles[char]["cluster_labels"]

    # 章の逐次更新
    path_b_states[char] = update_path_b_for_chapter(
        char, chapter_id, segments, path_b_states
    )

    return path_b_states[char].chapter_snapshots[chapter_id]
```

---

## スナップショットの出力例

```json
{
  "chapter_id": "chapter_5",
  "n_utterances_so_far": 87,
  "centroid": [0.14, -0.31, 0.08, ...],
  "variance": 0.061,
  "cluster_affinity": {
    "0": 0.71,
    "1": 0.29
  },
  "chapter_centroid": [0.19, -0.38, 0.11, ...]
}
```

`variance: 0.061`は第1章の`0.031`から上昇しており、「エレンの発言スタイルが第5章にかけて不安定になってきた」ことを統計的に示しています。`cluster_affinity`の`"0": 0.71`は「激情モード」クラスタへの親和度が高く、Path Aの`aggression: 0.78`と整合しているため矛盾なしと判定されます。この数値の積み重ねが論文の定量評価の根拠になります。

---
# 矛盾検出・統合
順番に説明します。

---

## 矛盾検出・統合の役割

Path AとPath Bは独立して性格を推定するため、結果が食い違うことがあります。矛盾検出・統合レイヤーの仕事は以下の3つです。

```
① 矛盾の検出    → どの程度食い違っているかをスコア化
② 矛盾の解消    → 食い違いの原因を特定し再生成を要求
③ 確定値の出力  → 次章のPath Aに渡す「確定済み性格状態」を生成
```

---

## データ構造

```python
from dataclasses import dataclass
from typing import Optional

@dataclass
class FusionResult:
    # 確定した性格状態（Path AとBを統合した最終値）
    confirmed_state: PersonalityState

    # 矛盾検出の記録（論文の評価指標として使う）
    contradiction_score: float        # 0.0=完全一致, 1.0=完全矛盾
    cosine_similarity: float          # Path AサマリーとPath B重心の類似度
    needed_regeneration: bool         # 再生成が必要だったか
    regeneration_count: int = 0       # 再生成した回数
    fusion_method: str = ""           # どの方法で統合したか（査読対策）
```

---

## Step 1：矛盾スコアの算出

前章で実装した`compute_path_ab_contradiction`を受け取り、矛盾の種類と深刻度を判定します。

```python
THRESHOLDS = {
    "no_contradiction":   0.2,   # 0.0〜0.2: 矛盾なし → そのまま統合
    "soft_contradiction": 0.4,   # 0.2〜0.4: 軽微な矛盾 → 重み付き統合
    "hard_contradiction": 1.0,   # 0.4以上:  深刻な矛盾 → 再生成
}

def classify_contradiction(contradiction_result: dict) -> str:
    score = contradiction_result["contradiction_score"]
    if score <= THRESHOLDS["no_contradiction"]:
        return "none"
    elif score <= THRESHOLDS["soft_contradiction"]:
        return "soft"
    else:
        return "hard"
```

---

## Step 2：矛盾の種類別の対処

```python
def detect_and_fuse(path_a_state: PersonalityState,
                     path_b_snapshot: dict,
                     prev_state: Optional[PersonalityState],
                     cluster_labels: list,
                     llm,
                     max_regeneration: int = 2) -> FusionResult:
    """
    矛盾検出・統合のメイン処理。
    矛盾の深刻度に応じて3つの処理に分岐する。
    """
    contradiction = compute_path_ab_contradiction(
        path_a_state, path_b_snapshot, cluster_labels
    )
    level = classify_contradiction(contradiction)

    if level == "none":
        # ケース1: 矛盾なし → 重み付き平均で統合
        confirmed = fuse_no_contradiction(
            path_a_state, path_b_snapshot
        )
        return FusionResult(
            confirmed_state=confirmed,
            contradiction_score=contradiction["contradiction_score"],
            cosine_similarity=contradiction["cosine_similarity"],
            needed_regeneration=False,
            fusion_method="weighted_average"
        )

    elif level == "soft":
        # ケース2: 軽微な矛盾 → Path Bを優先した重み付き統合
        confirmed = fuse_soft_contradiction(
            path_a_state, path_b_snapshot, contradiction
        )
        return FusionResult(
            confirmed_state=confirmed,
            contradiction_score=contradiction["contradiction_score"],
            cosine_similarity=contradiction["cosine_similarity"],
            needed_regeneration=False,
            fusion_method="path_b_weighted"
        )

    else:
        # ケース3: 深刻な矛盾 → 原因特定 → Path A再生成
        return fuse_hard_contradiction(
            path_a_state, path_b_snapshot,
            prev_state, contradiction,
            cluster_labels, llm, max_regeneration
        )
```

---

## Step 3：矛盾なし / 軽微な矛盾の統合

```python
def fuse_no_contradiction(path_a_state: PersonalityState,
                           path_b_snapshot: dict) -> PersonalityState:
    """
    矛盾なしのケース。
    Path Aスコア 60% + Path B親和性由来のスコア 40% で統合する。

    Path Aは解釈可能性が高い反面ハルシネーションリスクがあるため、
    Path Bで客観的に裏付けられた値をやや優先する。
    """
    path_b_scores = derive_scores_from_path_b(path_b_snapshot)

    fused = PersonalityState(
        aggression  = path_a_state.aggression  * 0.6
                    + path_b_scores.aggression * 0.4,
        loyalty     = path_a_state.loyalty     * 0.6
                    + path_b_scores.loyalty    * 0.4,
        anxiety     = path_a_state.anxiety     * 0.6
                    + path_b_scores.anxiety    * 0.4,
        openness    = path_a_state.openness    * 0.6
                    + path_b_scores.openness   * 0.4,
        confidence  = path_a_state.confidence  * 0.6
                    + path_b_scores.confidence * 0.4,
        summary       = path_a_state.summary,
        update_reason = path_a_state.update_reason,
        chapter_id    = path_a_state.chapter_id
    )
    return fused


def fuse_soft_contradiction(path_a_state: PersonalityState,
                              path_b_snapshot: dict,
                              contradiction: dict) -> PersonalityState:
    """
    軽微な矛盾のケース。
    矛盾スコアに応じてPath Bの重みを増やす。
    Path Bは幻覚がないため、矛盾が大きいほどPath Bを信頼する。
    """
    score = contradiction["contradiction_score"]

    # 矛盾スコアが0.4に近いほどPath Bの重みが増す
    path_b_weight = 0.4 + (score - 0.2) * 1.5  # 0.4〜0.7の範囲
    path_a_weight = 1.0 - path_b_weight

    path_b_scores = derive_scores_from_path_b(path_b_snapshot)

    fused = PersonalityState(
        aggression  = path_a_state.aggression  * path_a_weight
                    + path_b_scores.aggression * path_b_weight,
        loyalty     = path_a_state.loyalty     * path_a_weight
                    + path_b_scores.loyalty    * path_b_weight,
        anxiety     = path_a_state.anxiety     * path_a_weight
                    + path_b_scores.anxiety    * path_b_weight,
        openness    = path_a_state.openness    * path_a_weight
                    + path_b_scores.openness   * path_b_weight,
        confidence  = path_a_state.confidence  * path_a_weight
                    + path_b_scores.confidence * path_b_weight,
        summary       = path_a_state.summary,
        update_reason = f"[軽微矛盾・Path B重み{path_b_weight:.2f}]"
                      + path_a_state.update_reason,
        chapter_id    = path_a_state.chapter_id
    )
    return fused
```

---

## Step 4：Path BスナップショットからPath A相当のスコアを導出

Path Bは数値ベクトルなので、クラスタ親和性から性格スコアを近似します。

```python
def derive_scores_from_path_b(path_b_snapshot: dict) -> PersonalityState:
    """
    クラスタ親和性と分散から性格スコアを近似する。
    クラスタラベルとスコアのマッピングは全文スキャン時に確定した
    cluster_labelsを使う。
    """
    affinity = path_b_snapshot.get("cluster_affinity", {})
    variance = path_b_snapshot.get("variance", 0.05)

    # クラスタラベルとスコアのマッピング（全文スキャン時に確定）
    # ここでは簡易マッピングの例
    CLUSTER_SCORE_MAP = {
        "激情モード":  {"aggression": 0.8, "anxiety": 0.6,
                        "confidence": 0.7, "openness": 0.4, "loyalty": 0.5},
        "内省モード":  {"aggression": 0.2, "anxiety": 0.7,
                        "confidence": 0.3, "openness": 0.7, "loyalty": 0.6},
        "信頼モード":  {"aggression": 0.2, "anxiety": 0.2,
                        "confidence": 0.7, "openness": 0.7, "loyalty": 0.9},
        "攻撃モード":  {"aggression": 0.9, "anxiety": 0.4,
                        "confidence": 0.8, "openness": 0.2, "loyalty": 0.3},
    }

    # 親和度による加重平均
    scores = {k: 0.0 for k in ["aggression","loyalty",
                                 "anxiety","openness","confidence"]}
    total_weight = 0.0

    for cluster_id, weight in affinity.items():
        label = path_b_snapshot.get(
            "cluster_labels", {}
        ).get(int(cluster_id), "")
        if label in CLUSTER_SCORE_MAP:
            for key, val in CLUSTER_SCORE_MAP[label].items():
                scores[key] += val * weight
            total_weight += weight

    if total_weight > 0:
        scores = {k: v / total_weight for k, v in scores.items()}

    # 分散が高い = 感情が不安定 → anxiety補正
    scores["anxiety"] = min(1.0, scores["anxiety"] + variance * 3.0)

    return PersonalityState(**scores)
```

---

## Step 5：深刻な矛盾の対処（再生成）

```python
def fuse_hard_contradiction(path_a_state, path_b_snapshot,
                              prev_state, contradiction,
                              cluster_labels, llm,
                              max_regeneration) -> FusionResult:
    """
    深刻な矛盾のケース。
    原因をLLMに特定させ、Path Aに再生成を要求する。
    max_regeneration回試みて解消しない場合はPath B優先で確定する。
    """
    regeneration_count = 0
    current_path_a = path_a_state

    while regeneration_count < max_regeneration:
        # 矛盾の原因をLLMに診断させる
        diagnosis = diagnose_contradiction(
            current_path_a, path_b_snapshot, cluster_labels, llm
        )

        # 診断を踏まえてPath Aを再生成
        current_path_a = regenerate_path_a(
            current_path_a, path_b_snapshot,
            diagnosis, prev_state, llm
        )
        regeneration_count += 1

        # 再生成後の矛盾スコアを再チェック
        new_contradiction = compute_path_ab_contradiction(
            current_path_a, path_b_snapshot, cluster_labels
        )
        new_level = classify_contradiction(new_contradiction)

        if new_level != "hard":
            # 矛盾が解消された → 統合して確定
            confirmed = fuse_no_contradiction(
                current_path_a, path_b_snapshot
            ) if new_level == "none" else fuse_soft_contradiction(
                current_path_a, path_b_snapshot, new_contradiction
            )
            return FusionResult(
                confirmed_state=confirmed,
                contradiction_score=new_contradiction["contradiction_score"],
                cosine_similarity=new_contradiction["cosine_similarity"],
                needed_regeneration=True,
                regeneration_count=regeneration_count,
                fusion_method=f"regenerated_{new_level}"
            )

    # max_regeneration回試みても解消しない場合
    # → Path Bを100%信頼して確定（LLMより統計の方が信頼できる）
    path_b_scores = derive_scores_from_path_b(path_b_snapshot)
    path_b_scores.summary       = current_path_a.summary  # サマリーはそのまま
    path_b_scores.update_reason = "[矛盾未解消・Path B優先確定]"
    path_b_scores.chapter_id    = current_path_a.chapter_id

    return FusionResult(
        confirmed_state=path_b_scores,
        contradiction_score=contradiction["contradiction_score"],
        cosine_similarity=contradiction["cosine_similarity"],
        needed_regeneration=True,
        regeneration_count=regeneration_count,
        fusion_method="path_b_fallback"
    )


def diagnose_contradiction(path_a_state, path_b_snapshot,
                            cluster_labels, llm) -> str:
    dominant_id = max(
        path_b_snapshot["cluster_affinity"],
        key=path_b_snapshot["cluster_affinity"].get
    )
    dominant_label = cluster_labels[int(dominant_id)]
    variance = path_b_snapshot["variance"]

    prompt = f"""
    Path A（LLM）とPath B（Embedding）の結果が大きく食い違っています。

    Path A のサマリー:
    {path_a_state.summary}

    Path A のスコア:
    攻撃性={path_a_state.aggression:.2f}, 忠誠心={path_a_state.loyalty:.2f},
    不安={path_a_state.anxiety:.2f}, 開放性={path_a_state.openness:.2f},
    自信={path_a_state.confidence:.2f}

    Path B の統計:
    支配的クラスタ: {dominant_label}（親和度{path_b_snapshot['cluster_affinity'][dominant_id]:.2f}）
    分散: {variance:.3f}（高いほど発言が不安定）

    食い違いの最も可能性が高い原因を1つ選んでください:
    A. Path Aが特定の場面に引きずられ全体を見誤っている
    B. Path Aが知識・情報に基づく解釈を混入させている
    C. Path Bのクラスタラベルがこの章の文脈に合っていない
    D. キャラクターがこの章で意図的に普段と異なる振る舞いをしている

    選択肢の記号と、その根拠を1文で返してください。
    """
    return llm.call(prompt)


def regenerate_path_a(path_a_state, path_b_snapshot,
                       diagnosis, prev_state, llm) -> PersonalityState:
    """
    診断結果を踏まえてPath Aのサマリー・スコアを再生成する。
    """
    prompt = f"""
    前回の性格分析に以下の問題がありました:
    {diagnosis}

    前回のサマリー（問題あり）:
    {path_a_state.summary}

    Embeddingの統計が示す客観的な傾向:
    支配的クラスタ: {path_b_snapshot.get('dominant_label', '不明')}
    発言の分散（不安定さ）: {path_b_snapshot['variance']:.3f}

    前章までの確定済みサマリー:
    {prev_state.summary if prev_state else '（初章）'}

    上記の問題点を修正し、観察された行動のみに基づいて
    性格スコアとサマリーを再生成してください。

    特に注意:
    - キャラクターの知識・情報状態への言及を含めないこと
    - 今章で観察できない推測を含めないこと

    JSONで返してください（前回と同じスキーマ）。
    """
    result = llm.call_json(prompt)

    return PersonalityState(
        aggression    = float(result.get("aggression",   path_a_state.aggression)),
        loyalty       = float(result.get("loyalty",      path_a_state.loyalty)),
        anxiety       = float(result.get("anxiety",      path_a_state.anxiety)),
        openness      = float(result.get("openness",     path_a_state.openness)),
        confidence    = float(result.get("confidence",   path_a_state.confidence)),
        summary       = result.get("summary",       path_a_state.summary),
        update_reason = result.get("update_reason", "[再生成]"),
        chapter_id    = path_a_state.chapter_id
    )
```

---

## 統合後の出力と性格履歴への保存

```python
def confirm_and_store(char: str,
                       chapter_id: str,
                       fusion_result: FusionResult,
                       personality_history: dict):
    """
    確定した性格状態を履歴に保存する。
    FusionResultのメタデータも保存し、論文の評価で使えるようにする。
    """
    personality_history[char][chapter_id] = {
        "state":   fusion_result.confirmed_state,
        "meta": {
            "contradiction_score":  fusion_result.contradiction_score,
            "cosine_similarity":    fusion_result.cosine_similarity,
            "needed_regeneration":  fusion_result.needed_regeneration,
            "regeneration_count":   fusion_result.regeneration_count,
            "fusion_method":        fusion_result.fusion_method,
        }
    }
```

---

## 出力例

```json
{
  "character": "エレン",
  "chapter_id": "chapter_5",
  "confirmed_state": {
    "aggression":  0.76,
    "loyalty":     0.81,
    "anxiety":     0.58,
    "openness":    0.34,
    "confidence":  0.62,
    "summary": "エレンは怒りを内側に溜め込むようになり発言が短く鋭くなっている。仲間への態度は表面上変わらないが随所に緊張感が滲み出ている。"
  },
  "meta": {
    "contradiction_score":  0.31,
    "cosine_similarity":    0.69,
    "needed_regeneration":  false,
    "fusion_method":        "path_b_weighted"
  }
}
```

---

## 論文の評価指標としての活用

`meta`フィールドに蓄積されたデータは以下の評価に直接使えます。

| メタデータ | 評価での使い方 |
|---|---|
| `contradiction_score`の章推移 | キャラクターの性格変化が激しい章を自動検出できているか |
| `needed_regeneration`の発生率 | Path Aのハルシネーション頻度の定量的な証拠 |
| `fusion_method`の分布 | 3種類の統合方式がどの割合で使われたかを示す |
| `cosine_similarity`の平均 | Path AとBの全体的な整合性の指標 |

これらをベースライン（単純なPath Aのみ）と比較することで、Dual-Path設計の有効性を定量的に主張できます。

---
# 心情層の更新
順番に説明します。

---

## 性格層と心情層の関係の整理

まず両者の役割を明確にします。

```
【性格層】Personality Layer
  更新単位：章
  表すもの：そのキャラクターの「傾向・素質」
  例：エレンは怒りやすく、仲間思いである
  → Path A/B + 矛盾検出で章ごとに確定済み

【心情層】Emotional State Layer
  更新単位：場面・セリフ
  表すもの：その瞬間の「状態・気分」
  例：今この場面のエレンは悲しみを堪えている
  → 場面の出来事から逐次推定
```

性格層と心情層の関係は**「許容範囲」と「実値」**です。

```python
# 性格層が「ありうる感情の幅」を決める
# 心情層が「今この瞬間の実際の値」を決める

# 例：攻撃性が高いキャラクター（性格層）でも
#     穏やかな場面では怒りは低い（心情層）
#     ただし穏やかさには上限がある（性格層による制約）

emotion_actual = clip(
    emotion_raw,                          # 場面から推定した生の値
    min = personality_floor(personality), # 性格層による下限
    max = personality_ceiling(personality)# 性格層による上限
)
```

---

## データ構造

```python
from dataclasses import dataclass, field
import numpy as np

@dataclass
class EmotionalState:
    # 基本6感情（0.0〜1.0）
    anger:      float = 0.0
    sadness:    float = 0.0
    fear:       float = 0.0
    joy:        float = 0.0
    disgust:    float = 0.0
    surprise:   float = 0.0

    # 複合状態
    tension:    float = 0.0   # 緊張（恐怖×自制心）
    resignation:float = 0.0   # 諦め（悲しみ×低自信）

    # 感情の変化トリガー（TTSの「なぜこのトーンか」の根拠）
    trigger:    str = ""
    scene_id:   str = ""

    # 抑制フラグ（知識制御との接続点）
    suppressed: bool = False      # 感情を意図的に隠しているか
    suppression_weight: float = 0.0
```

---

## Step 1：性格層による感情の許容範囲算出

```python
def compute_emotion_bounds(personality: PersonalityState) -> dict:
    """
    性格層から各感情の「ありうる範囲」を算出する。

    設計思想:
    - 攻撃性が高い → 怒りの上限が高く、下限も低くはない
    - 不安傾向が高い → 恐怖・緊張の上限が高い
    - 自信が高い → 恐怖の上限が低め
    - 開放性が高い → 感情全体の振れ幅が大きい
    """
    # 振れ幅係数（開放性が高いほど感情が表に出やすい）
    amplitude = 0.4 + personality.openness * 0.6

    bounds = {
        "anger": (
            personality.aggression * 0.1,           # 下限（最低限この程度は出やすい）
            min(1.0, personality.aggression * amplitude * 1.3)  # 上限
        ),
        "sadness": (
            0.0,
            min(1.0, (1.0 - personality.confidence) * amplitude * 1.2)
        ),
        "fear": (
            0.0,
            min(1.0, personality.anxiety * amplitude * 1.1)
        ),
        "joy": (
            0.0,
            min(1.0, (1.0 - personality.anxiety * 0.5) * amplitude)
        ),
        "disgust": (
            0.0,
            min(1.0, personality.aggression * amplitude * 0.9)
        ),
        "surprise": (
            0.0,
            amplitude  # 開放性が直接上限に影響
        ),
        "tension": (
            personality.anxiety * 0.1,
            min(1.0, personality.anxiety * amplitude * 1.2)
        ),
        "resignation": (
            0.0,
            min(1.0, (1.0 - personality.confidence)
                     * (1.0 - personality.loyalty * 0.3) * amplitude)
        ),
    }
    return bounds
```

---

## Step 2：場面テキストからの感情推定

```python
def estimate_raw_emotion_from_scene(scene_segments: list[dict],
                                     char: str,
                                     prev_emotion: EmotionalState,
                                     llm) -> EmotionalState:
    """
    場面のセリフ・描写から感情の「生の値」を推定する。
    性格層の制約はまだ適用しない。
    """
    # キャラクター関連セグメントを抽出
    char_segs = [
        seg for seg in scene_segments
        if seg["speaker"] == char
        or (seg["type"] == "narration" and char in seg["text"])
    ]
    if not char_segs:
        # 登場しない場面は前の状態を減衰させて引き継ぐ
        return decay_emotion(prev_emotion, decay_rate=0.3)

    seg_text = "\n".join(
        f"[{'セリフ' if s['type'] == 'dialogue' else '描写'}] {s['text']}"
        for s in char_segs
    )

    prev_summary = emotion_to_text(prev_emotion)

    prompt = f"""
    キャラクター「{char}」の直前の感情状態:
    {prev_summary}

    今場面のセリフ・描写:
    {seg_text}

    この場面での「{char}」の感情状態を推定してください。

    ルール:
    - セリフ・描写から直接読み取れる感情のみを評価する
    - 「なぜそう感じるか」の背景知識には言及しない
    - 感情を表に出しているか、抑えているかも判定する

    JSON形式で返してください:
    {{
      "anger":       0.0〜1.0,
      "sadness":     0.0〜1.0,
      "fear":        0.0〜1.0,
      "joy":         0.0〜1.0,
      "disgust":     0.0〜1.0,
      "surprise":    0.0〜1.0,
      "tension":     0.0〜1.0,
      "resignation": 0.0〜1.0,
      "suppressed":  true/false,
      "trigger":     "感情変化の原因となった出来事を1文で",
    }}
    """
    result = llm.call_json(prompt)

    return EmotionalState(
        anger       = float(result.get("anger",       0.0)),
        sadness     = float(result.get("sadness",     0.0)),
        fear        = float(result.get("fear",        0.0)),
        joy         = float(result.get("joy",         0.0)),
        disgust     = float(result.get("disgust",     0.0)),
        surprise    = float(result.get("surprise",    0.0)),
        tension     = float(result.get("tension",     0.0)),
        resignation = float(result.get("resignation", 0.0)),
        suppressed  = bool(result.get("suppressed",   False)),
        trigger     = result.get("trigger", ""),
        scene_id    = scene_segments[0]["scene_id"] if scene_segments else ""
    )
```

---

## Step 3：性格層による感情のクリッピング

```python
def apply_personality_constraint(raw_emotion: EmotionalState,
                                   personality: PersonalityState) -> EmotionalState:
    """
    性格層の許容範囲に収まるよう感情値をクリッピングする。
    これが性格層と心情層の「接合点」。
    """
    bounds = compute_emotion_bounds(personality)

    def clip_emotion(value, field_name):
        lo, hi = bounds[field_name]
        return float(np.clip(value, lo, hi))

    return EmotionalState(
        anger       = clip_emotion(raw_emotion.anger,       "anger"),
        sadness     = clip_emotion(raw_emotion.sadness,     "sadness"),
        fear        = clip_emotion(raw_emotion.fear,        "fear"),
        joy         = clip_emotion(raw_emotion.joy,         "joy"),
        disgust     = clip_emotion(raw_emotion.disgust,     "disgust"),
        surprise    = clip_emotion(raw_emotion.surprise,    "surprise"),
        tension     = clip_emotion(raw_emotion.tension,     "tension"),
        resignation = clip_emotion(raw_emotion.resignation, "resignation"),
        suppressed          = raw_emotion.suppressed,
        suppression_weight  = raw_emotion.suppression_weight,
        trigger             = raw_emotion.trigger,
        scene_id            = raw_emotion.scene_id
    )
```

---

## Step 4：感情の減衰処理

登場しない場面や次場面への引き継ぎで、感情が自然に収束するよう減衰させます。

```python
def decay_emotion(emotion: EmotionalState,
                   decay_rate: float = 0.3,
                   personality: PersonalityState = None) -> EmotionalState:
    """
    感情を中性値に向けて減衰させる。
    性格層の特性により減衰速度が異なる。

    - 攻撃性が高い → 怒りが冷めにくい（減衰が遅い）
    - 開放性が低い → 感情全体が中性に戻りにくい
    """
    # 性格層による減衰速度の補正
    if personality:
        anger_decay   = decay_rate * (1.0 - personality.aggression * 0.5)
        default_decay = decay_rate * (1.0 - personality.openness * 0.3)
    else:
        anger_decay = default_decay = decay_rate

    def decay(val, rate, neutral=0.0):
        return val + (neutral - val) * rate

    return EmotionalState(
        anger       = decay(emotion.anger,       anger_decay),
        sadness     = decay(emotion.sadness,     default_decay),
        fear        = decay(emotion.fear,        default_decay),
        joy         = decay(emotion.joy,         default_decay),
        disgust     = decay(emotion.disgust,     default_decay),
        surprise    = decay(emotion.surprise,    default_decay * 1.5),  # 驚きは特に早く収束
        tension     = decay(emotion.tension,     anger_decay),
        resignation = decay(emotion.resignation, default_decay * 0.7),  # 諦めは収束が遅い
        suppressed  = emotion.suppressed,
        suppression_weight = emotion.suppression_weight * (1.0 - default_decay),
        trigger     = "",
        scene_id    = emotion.scene_id
    )
```

---

## Step 5：知識制御との接合

知識DBの`hiding_knowledge`フラグを心情層に反映します。

```python
def apply_knowledge_suppression_to_emotion(
        emotion: EmotionalState,
        knowledge_control: dict) -> EmotionalState:
    """
    「知っているのに隠している」フラグを
    心情層の抑制重みに反映する。

    知識制御は感情の「実値」を変えるのではなく
    「表に出る量」を制限する。
    → TTSパラメタ生成時に intensity を絞るための情報として使う。
    """
    if not knowledge_control:
        return emotion

    max_suppression = max(
        c["suppression_weight"]
        for c in knowledge_control.values()
        if c.get("hiding_knowledge")
    ) if any(
        c.get("hiding_knowledge")
        for c in knowledge_control.values()
    ) else 0.0

    if max_suppression > 0:
        emotion.suppressed = True
        emotion.suppression_weight = max(
            emotion.suppression_weight,
            max_suppression
        )

    return emotion
```

---

## 統合エントリポイント

```python
def update_emotional_state(char: str,
                            scene_segments: list[dict],
                            prev_emotion: EmotionalState,
                            personality: PersonalityState,
                            knowledge_control: dict,
                            llm) -> EmotionalState:
    """
    心情層更新のメイン処理。
    上記5ステップを統合して呼び出す。
    """
    # Step 2: 生の感情を推定
    raw_emotion = estimate_raw_emotion_from_scene(
        scene_segments, char, prev_emotion, llm
    )

    # Step 3: 性格層による制約を適用
    constrained = apply_personality_constraint(raw_emotion, personality)

    # Step 5: 知識制御による抑制を適用
    final_emotion = apply_knowledge_suppression_to_emotion(
        constrained, knowledge_control
    )

    return final_emotion
```

---

## TTSパラメタへの接続

```python
def emotion_to_tts_params(emotion: EmotionalState,
                            personality: PersonalityState) -> dict:
    """
    性格層（傾向）× 心情層（実値）→ TTS音声パラメタ
    """
    # 支配的感情を特定
    emotion_dict = {
        "anger":    emotion.anger,
        "sadness":  emotion.sadness,
        "fear":     emotion.fear,
        "joy":      emotion.joy,
        "tension":  emotion.tension,
    }
    dominant = max(emotion_dict, key=emotion_dict.get)
    intensity = emotion_dict[dominant]

    # 抑制重みを適用（知識隠蔽がある場合）
    effective_intensity = intensity * (1.0 - emotion.suppression_weight)

    return {
        "emotion":          dominant,
        "emotion_intensity": effective_intensity,

        # 性格層が「話し方の癖」として常に影響する
        "speech_rate":      0.9 + personality.aggression * 0.2
                              + emotion.tension * 0.15,
        "pitch_scale":      1.0 - personality.anxiety * 0.1
                              + emotion.anger * 0.05,
        "pitch_variation":  (0.3 + personality.openness * 0.4)
                              * (1.0 - emotion.suppression_weight * 0.5),
        "energy":           0.5 + effective_intensity * 0.4,

        # 知識隠蔽がある場合の追加制御
        "pause_before_ms":  200 + int(emotion.suppression_weight * 300),
        "hiding_knowledge": emotion.suppressed,
    }
```

---

## 出力例

```json
{
  "character": "エレン",
  "scene_id": "chapter_5_scene_3",
  "emotional_state": {
    "anger":       0.71,
    "sadness":     0.38,
    "tension":     0.65,
    "suppressed":  true,
    "suppression_weight": 0.7,
    "trigger": "仲間が傷つくのを目撃したが、感情を抑えて平静を装っている"
  },
  "tts_params": {
    "emotion":           "anger",
    "emotion_intensity":  0.21,
    "speech_rate":        1.05,
    "pitch_variation":    0.18,
    "pause_before_ms":    410,
    "hiding_knowledge":   true
  }
}
```

`emotion_intensity`が`anger: 0.71`に対して`0.21`まで下がっているのは`suppression_weight: 0.7`が効いているためです。内面では強い怒りを抱えているが、それを意図的に隠しているという状態がTTSパラメタに反映されています。

---
# セリフごとのTTSパラメタ設定
順番に説明します。

---

## 3層統合の全体像

第2パスでは確定済みの3つのレイヤーを1つのセリフに対して統合します。

```
【性格層】章ごとに確定済み（personality_history から参照）
  → 話し方の「癖・傾向」を決める。セリフが変わっても変化しない

【心情層】場面ごとに推定済み（emotional_state から参照）
  → 今この瞬間の「感情の実値」を決める

【知識制御】セリフごとのフラグ（knowledge_control から参照）
  → 感情を「表に出す量」を制限する

                    ↓ 統合
         TTSパラメタ（セリフごとに確定）
```

---

## データ構造

```python
from dataclasses import dataclass

@dataclass
class TTSParams:
    # 感情制御
    emotion: str              # 支配的感情ラベル
    emotion_intensity: float  # 実効強度（抑制後）
    secondary_emotion: str    # 副次感情（複雑な感情表現に使う）
    secondary_intensity: float

    # 音声パラメタ
    speech_rate: float        # 発話速度（1.0が標準）
    pitch_scale: float        # 音程（1.0が標準）
    pitch_variation: float    # 抑揚の大きさ
    energy: float             # 声の張り・音量
    pause_before_ms: int      # 発話前の間（ミリ秒）
    pause_after_ms: int       # 発話後の間（ミリ秒）

    # 語調制御
    speaking_style: str       # 丁寧語・タメ口・命令口調など
    breathiness: float        # 息漏れ（抑えた発話に使う）

    # 知識制御フラグ（TTSへの最終指示）
    hiding_knowledge: bool
    suppression_weight: float

    # デバッグ・評価用メタデータ
    dominant_layer: str       # どの層が最も影響したか
    chapter_id: str
    scene_id: str
```

---

## Step 1：セリフの文脈分析

統合の前に、セリフ単体の特徴を分析します。性格・心情・知識とは独立した「このセリフ自体が持つ性質」を先に抽出します。

```python
def analyze_utterance_context(utterance: dict,
                               surrounding_segs: list[dict],
                               llm) -> dict:
    """
    セリフ単体の文脈特徴を分析する。
    3層とは独立して実行する。
    """
    # 前後2セグメントを文脈として取得
    context_text = "\n".join(
        f"[{s['type']}] {s.get('speaker','')}: {s['text']}"
        for s in surrounding_segs
    )

    prompt = f"""
    以下の文脈の中のセリフを分析してください。

    文脈:
    {context_text}

    対象セリフ: 「{utterance['text']}」

    以下をJSONで返してください:
    {{
      "utterance_type": "断言/疑問/命令/懇願/独白/皮肉",
      "addressee": "誰に向けて言っているか（独白の場合はnull）",
      "is_interrupted": false,      // 発話が途中で遮られているか
      "has_subtext":    false,       // 言葉と裏腹な意図があるか
      "emphasis_words": ["強調すべき単語のリスト"],
      "natural_pause_positions": [3, 7]  // 自然な間を置く位置（文字インデックス）
    }}
    """
    return llm.call_json(prompt)
```

---

## Step 2：3層の重み計算

3層は常に均等に影響するわけではありません。セリフの種類・文脈によって各層の影響度を動的に変えます。

```python
def compute_layer_weights(utterance_context: dict,
                           emotion: EmotionalState,
                           knowledge_control: dict) -> dict:
    """
    セリフの性質に応じて3層の影響度を動的に決定する。

    基本重み:
    - 性格層: 0.3  （常に背景として効く）
    - 心情層: 0.5  （最も直接的に声に出る）
    - 知識制御: 0.2 （抑制として効く）

    ただし以下の条件で調整する。
    """
    weights = {
        "personality": 0.3,
        "emotion":     0.5,
        "knowledge":   0.2,
    }

    # 知識隠蔽が強い場合 → 知識制御の影響を増大
    max_suppression = max(
        (c["suppression_weight"] for c in knowledge_control.values()
         if c.get("hiding_knowledge")),
        default=0.0
    )
    if max_suppression > 0.5:
        weights["knowledge"]   += 0.15
        weights["emotion"]     -= 0.10
        weights["personality"] -= 0.05

    # 感情が非常に強い場合 → 感情層が支配的になる
    max_emotion = max(
        emotion.anger, emotion.sadness, emotion.fear,
        emotion.joy, emotion.tension
    )
    if max_emotion > 0.8 and not emotion.suppressed:
        weights["emotion"]     += 0.15
        weights["personality"] -= 0.10
        weights["knowledge"]   -= 0.05

    # 独白・内省的なセリフ → 性格層が相対的に強く出る
    if utterance_context.get("addressee") is None:
        weights["personality"] += 0.10
        weights["emotion"]     -= 0.10

    # 正規化（合計が1.0になるように）
    total = sum(weights.values())
    return {k: v / total for k, v in weights.items()}
```

---

## Step 3：性格層からの音声パラメタ算出

```python
def derive_params_from_personality(personality: PersonalityState,
                                    utterance_context: dict) -> dict:
    """
    性格層 → 話し方の「癖・ベースライン」
    セリフが変わっても、このキャラクターが変わらない限り一定。
    """
    # 語調スタイルの決定
    if personality.aggression > 0.7:
        speaking_style = "強め・短文傾向"
    elif personality.openness > 0.7:
        speaking_style = "表現豊か・抑揚あり"
    elif personality.anxiety > 0.7:
        speaking_style = "不安定・語尾が弱め"
    else:
        speaking_style = "標準"

    # 命令口調かどうかで速度補正
    rate_adjustment = 1.05 if utterance_context.get(
        "utterance_type") == "命令" else 1.0

    return {
        "base_speech_rate":  (0.9 + personality.aggression * 0.15
                               + personality.confidence * 0.10) * rate_adjustment,
        "base_pitch_scale":   1.0 - personality.anxiety * 0.08
                                  + personality.confidence * 0.05,
        "base_pitch_variation": 0.3 + personality.openness * 0.4,
        "base_breathiness":   personality.anxiety * 0.3,
        "speaking_style":     speaking_style,
    }
```

---

## Step 4：心情層からの音声パラメタ算出

```python
def derive_params_from_emotion(emotion: EmotionalState,
                                utterance_context: dict) -> dict:
    """
    心情層 → 今この瞬間の感情の出方
    性格層のベースラインに対するデルタとして計算する。
    """
    # 支配的感情と副次感情を特定
    emotion_vals = {
        "anger":       emotion.anger,
        "sadness":     emotion.sadness,
        "fear":        emotion.fear,
        "joy":         emotion.joy,
        "tension":     emotion.tension,
        "resignation": emotion.resignation,
        "disgust":     emotion.disgust,
        "surprise":    emotion.surprise,
    }
    sorted_emotions = sorted(
        emotion_vals.items(), key=lambda x: x[1], reverse=True
    )
    dominant   = sorted_emotions[0]
    secondary  = sorted_emotions[1] if sorted_emotions[1][1] > 0.2 else ("neutral", 0.0)

    # 感情ごとの音声パラメタへの影響
    EMOTION_VOICE_MAP = {
        "anger":       {"rate_delta": +0.15, "pitch_delta": +0.05,
                        "variation_delta": -0.10, "energy_delta": +0.25},
        "sadness":     {"rate_delta": -0.15, "pitch_delta": -0.05,
                        "variation_delta": -0.15, "energy_delta": -0.20},
        "fear":        {"rate_delta": +0.10, "pitch_delta": +0.08,
                        "variation_delta": +0.05, "energy_delta": -0.10},
        "joy":         {"rate_delta": +0.05, "pitch_delta": +0.08,
                        "variation_delta": +0.20, "energy_delta": +0.15},
        "tension":     {"rate_delta": +0.08, "pitch_delta": +0.03,
                        "variation_delta": -0.20, "energy_delta": +0.05},
        "resignation": {"rate_delta": -0.10, "pitch_delta": -0.08,
                        "variation_delta": -0.20, "energy_delta": -0.25},
        "disgust":     {"rate_delta": -0.05, "pitch_delta": -0.03,
                        "variation_delta": -0.05, "energy_delta": +0.05},
        "surprise":    {"rate_delta": +0.00, "pitch_delta": +0.10,
                        "variation_delta": +0.25, "energy_delta": +0.10},
    }

    dom_map = EMOTION_VOICE_MAP.get(dominant[0], {k: 0.0 for k in EMOTION_VOICE_MAP["anger"]})
    sec_map = EMOTION_VOICE_MAP.get(secondary[0], {k: 0.0 for k in EMOTION_VOICE_MAP["anger"]})

    # 支配的感情70%・副次感情30%でブレンド
    blended = {
        k: dom_map[k] * dominant[1] * 0.7 + sec_map[k] * secondary[1] * 0.3
        for k in dom_map
    }

    # 間（ポーズ）の計算
    pause_before = 0
    if emotion.sadness > 0.6 or emotion.resignation > 0.5:
        pause_before += 300
    if utterance_context.get("has_subtext"):
        pause_before += 200

    return {
        "dominant_emotion":      dominant[0],
        "dominant_intensity":    dominant[1],
        "secondary_emotion":     secondary[0],
        "secondary_intensity":   secondary[1],
        "rate_delta":            blended["rate_delta"],
        "pitch_delta":           blended["pitch_delta"],
        "variation_delta":       blended["variation_delta"],
        "energy_delta":          blended["energy_delta"],
        "pause_before_ms":       pause_before,
    }
```

---

## Step 5：知識制御レイヤーの適用

```python
def apply_knowledge_control_layer(params_so_far: dict,
                                   knowledge_control: dict,
                                   emotion: EmotionalState) -> dict:
    """
    知識制御は感情の「実値」を変えるのではなく
    「表に出る量」のみを制限する。
    内面の状態は変えず、表出だけを抑える。
    """
    if not knowledge_control:
        return params_so_far

    hiding = any(
        c.get("hiding_knowledge")
        for c in knowledge_control.values()
    )
    if not hiding:
        return params_so_far

    sw = emotion.suppression_weight  # 0.0〜1.0

    # 感情強度を抑制（内面は強くても外に出さない）
    params_so_far["emotion_intensity"] *= (1.0 - sw * 0.8)

    # 抑揚を平坦に（感情を隠すため）
    params_so_far["pitch_variation"] *= (1.0 - sw * 0.6)

    # 少し速度を上げる（緊張・不自然さの表現）
    params_so_far["speech_rate"] *= (1.0 + sw * 0.08)

    # 息漏れを増やす（抑制した発話の物理的な特徴）
    params_so_far["breathiness"] = min(
        1.0,
        params_so_far.get("breathiness", 0.0) + sw * 0.4
    )

    # 発話前の間を長くする（隠すための一拍）
    params_so_far["pause_before_ms"] += int(sw * 400)

    params_so_far["hiding_knowledge"] = True
    params_so_far["suppression_weight"] = sw

    return params_so_far
```

---

## Step 6：3層の統合

```python
def integrate_three_layers(utterance: dict,
                             personality: PersonalityState,
                             emotion: EmotionalState,
                             knowledge_control: dict,
                             surrounding_segs: list[dict],
                             llm) -> TTSParams:
    """
    3層統合のメインエントリポイント。
    """
    # セリフ文脈分析
    context = analyze_utterance_context(utterance, surrounding_segs, llm)

    # 各層のパラメタを計算
    p_params = derive_params_from_personality(personality, context)
    e_params = derive_params_from_emotion(emotion, context)
    weights  = compute_layer_weights(context, emotion, knowledge_control)

    # 性格層ベースライン + 心情層デルタ（加算）
    # → 知識制御による抑制（乗算）の順で適用
    raw_params = {
        "emotion":           e_params["dominant_emotion"],
        "emotion_intensity": e_params["dominant_intensity"],
        "secondary_emotion": e_params["secondary_emotion"],
        "secondary_intensity": e_params["secondary_intensity"],

        # 性格層ベース + 心情層デルタ
        "speech_rate":    p_params["base_speech_rate"]
                        + e_params["rate_delta"]      * weights["emotion"],
        "pitch_scale":    p_params["base_pitch_scale"]
                        + e_params["pitch_delta"]     * weights["emotion"],
        "pitch_variation": p_params["base_pitch_variation"]
                        + e_params["variation_delta"] * weights["emotion"],
        "energy":          0.5
                        + e_params["energy_delta"]    * weights["emotion"],

        "breathiness":     p_params["base_breathiness"],
        "speaking_style":  p_params["speaking_style"],
        "pause_before_ms": e_params["pause_before_ms"],
        "pause_after_ms":  0,
        "hiding_knowledge": False,
        "suppression_weight": 0.0,

        # どの層が最も効いたかを記録
        "dominant_layer": max(weights, key=weights.get),
    }

    # 知識制御レイヤーを最後に適用（乗算で上書き）
    final_params = apply_knowledge_control_layer(
        raw_params, knowledge_control, emotion
    )

    # 値域を安全にクランプ
    final_params["speech_rate"]    = float(np.clip(final_params["speech_rate"],    0.6, 1.6))
    final_params["pitch_scale"]    = float(np.clip(final_params["pitch_scale"],    0.7, 1.4))
    final_params["pitch_variation"]= float(np.clip(final_params["pitch_variation"],0.0, 1.0))
    final_params["energy"]         = float(np.clip(final_params["energy"],         0.1, 1.0))

    return TTSParams(
        **final_params,
        chapter_id = personality.chapter_id,
        scene_id   = utterance["scene_id"]
    )
```

---

## 出力例と3層の寄与の可視化

```json
{
  "utterance": "何も知らないよ、そんなこと。",
  "character": "エレン",
  "scene_id":  "chapter_5_scene_3",

  "tts_params": {
    "emotion":             "tension",
    "emotion_intensity":    0.19,
    "secondary_emotion":   "anger",
    "secondary_intensity":  0.14,

    "speech_rate":          1.06,
    "pitch_scale":          0.97,
    "pitch_variation":      0.11,
    "energy":               0.54,
    "breathiness":          0.42,
    "pause_before_ms":      510,
    "speaking_style":       "強め・短文傾向",
    "hiding_knowledge":     true,
    "suppression_weight":   0.70,
    "dominant_layer":       "knowledge"
  }
}
```

各層の寄与はこうなっています。

```
性格層（aggression=0.78）
  → speech_rateのベースを1.02に設定
  → speaking_styleを「強め・短文傾向」に設定

心情層（tension=0.65, anger=0.71, suppressed=true）
  → tensionのrate_delta(+0.08)を加算 → 1.10
  → pitch_variationをベース0.32から-0.20 → 0.12

知識制御（suppression_weight=0.70）
  → emotion_intensityを0.65×(1-0.56) → 0.19 に圧縮
  → pitch_variationをさらに×0.40 → 0.11 に圧縮
  → breathinessを0.42に引き上げ
  → pause_before_msに+280ms追加
```

内面では強い怒りと緊張を抱えているにもかかわらず、TTSへの指示は「抑揚がほぼない・息漏れがある・長い間の後に発話する」という出力になっており、感情を隠しているという状態が音声レベルで表現されます。

---
# TTSパラメタ出力
これまでの実装を統合した完全なJSONスキーマを示します。

---

## 完全なJSONスキーマ（1セリフ分）

```json
{
  "$schema": "audiobook_tts_params_v1",

  "meta": {
    "character":   "エレン",
    "chapter_id":  "chapter_5",
    "scene_id":    "chapter_5_scene_3",
    "utterance_id": "ch5_sc3_utt_007",
    "text":        "何も知らないよ、そんなこと。",
    "utterance_type": "断言",
    "addressee":   "アルミン"
  },

  "personality_layer": {
    "aggression":   0.78,
    "loyalty":      0.82,
    "anxiety":      0.55,
    "openness":     0.31,
    "confidence":   0.64,
    "summary":      "エレンは怒りを内側に溜め込むようになり発言が短く鋭くなっている。",
    "chapter_confirmed_at": "chapter_5"
  },

  "emotional_layer": {
    "dominant_emotion":      "tension",
    "dominant_intensity":     0.65,
    "secondary_emotion":     "anger",
    "secondary_intensity":    0.71,
    "suppressed":             true,
    "trigger":  "仲間が傷つくのを目撃したが感情を抑えて平静を装っている",
    "scene_confirmed_at":    "chapter_5_scene_3"
  },

  "knowledge_control": {
    "event_002": {
      "hiding_knowledge":    true,
      "known_since":         "chapter_3",
      "suppression_weight":  0.70,
      "detection_pattern":   "A"
    }
  },

  "tts_params": {

    "emotion": {
      "label":               "tension",
      "intensity":            0.19,
      "secondary_label":     "anger",
      "secondary_intensity":  0.14
    },

    "prosody": {
      "speech_rate":          1.06,
      "pitch_scale":          0.97,
      "pitch_variation":      0.11,
      "energy":               0.54
    },

    "voice_quality": {
      "breathiness":          0.42,
      "speaking_style":       "強め・短文傾向",
      "emphasis_words":       ["知らない"]
    },

    "timing": {
      "pause_before_ms":      510,
      "pause_after_ms":        80,
      "natural_pause_positions": [5]
    },

    "knowledge_suppression": {
      "hiding_knowledge":     true,
      "suppression_weight":   0.70
    }
  },

  "fusion_meta": {
    "dominant_layer":         "knowledge",
    "layer_weights": {
      "personality":           0.27,
      "emotion":               0.41,
      "knowledge":             0.32
    },
    "contradiction_score":     0.31,
    "cosine_similarity":       0.69,
    "fusion_method":           "path_b_weighted",
    "needed_regeneration":     false
  }
}
```

---

## 各ブロックの役割

| ブロック | 内容 | 生成タイミング |
|---|---|---|
| `meta` | セリフの識別情報・文脈 | 前処理 |
| `personality_layer` | 章ごとに確定した性格状態 | 性格履歴ループ |
| `emotional_layer` | 場面ごとの心情状態 | 第2パス場面サブループ |
| `knowledge_control` | 知識隠蔽フラグ | 第1パス知識DBループ |
| `tts_params` | TTSへの最終指示 | 3層統合後に生成 |
| `fusion_meta` | デバッグ・評価用記録 | 矛盾検出・統合時 |

---

## Style-BERT-VITS2へのマッピング例

`tts_params`を実際のTTSツールの入力形式に変換します。

```python
def to_style_bert_vits2(tts_params: dict,
                         character: str) -> dict:
    """
    TTSパラメタをStyle-BERT-VITS2の入力形式に変換する。
    """
    p = tts_params["tts_params"]

    # 感情ラベル → Style-BERT-VITS2のスタイルIDへのマッピング
    EMOTION_STYLE_MAP = {
        "anger":       "Angry",
        "sadness":     "Sad",
        "fear":        "Fearful",
        "joy":         "Happy",
        "tension":     "Nervous",
        "resignation": "Sad",
        "disgust":     "Disgusted",
        "surprise":    "Surprised",
        "neutral":     "Neutral",
    }

    dominant  = p["emotion"]["label"]
    secondary = p["emotion"]["secondary_label"]
    d_intensity = p["emotion"]["intensity"]
    s_intensity = p["emotion"]["secondary_intensity"]

    # 感情をブレンド（支配的70%・副次30%）
    style_weight = {
        EMOTION_STYLE_MAP[dominant]:  d_intensity * 0.7,
        EMOTION_STYLE_MAP[secondary]: s_intensity * 0.3,
    }

    # 知識隠蔽がある場合はNeutral寄りにシフト
    if p["knowledge_suppression"]["hiding_knowledge"]:
        sw = p["knowledge_suppression"]["suppression_weight"]
        for key in style_weight:
            style_weight[key] *= (1.0 - sw * 0.6)
        style_weight["Neutral"] = style_weight.get("Neutral", 0.0) + sw * 0.4

    return {
        "text":            tts_params["meta"]["text"],
        "speaker":         character,
        "style_weight":    style_weight,

        # prosody
        "speed":           p["prosody"]["speech_rate"],
        "pitch":           p["prosody"]["pitch_scale"],
        "intonation_scale": p["prosody"]["pitch_variation"],
        "volume":          p["prosody"]["energy"],

        # voice quality
        "noise_scale":     p["voice_quality"]["breathiness"],

        # timing
        "pre_phoneme_length":  p["timing"]["pause_before_ms"] / 1000.0,
        "post_phoneme_length": p["timing"]["pause_after_ms"]  / 1000.0,

        # emphasis（対応している場合）
        "accent_phrases":  build_accent_phrases(
            tts_params["meta"]["text"],
            p["voice_quality"]["emphasis_words"],
            p["timing"]["natural_pause_positions"]
        )
    }


def build_accent_phrases(text: str,
                          emphasis_words: list[str],
                          pause_positions: list[int]) -> list[dict]:
    """
    強調語と間のポジションをアクセント句として構造化する。
    """
    phrases = []
    current = ""
    for i, char in enumerate(text):
        current += char
        if i in pause_positions or i == len(text) - 1:
            has_emphasis = any(w in current for w in emphasis_words)
            phrases.append({
                "text":         current,
                "is_emphasized": has_emphasis,
                "pause_after_ms": 150 if i in pause_positions else 0
            })
            current = ""
    return phrases
```

---

## 複数セリフのバッチ出力例

実際の処理では1場面分をまとめて出力します。

```json
{
  "scene_id": "chapter_5_scene_3",
  "character": "エレン",
  "utterances": [
    {
      "utterance_id": "ch5_sc3_utt_005",
      "text": "別に。",
      "tts_params": {
        "emotion":      {"label": "tension",  "intensity": 0.22},
        "prosody":      {"speech_rate": 1.02, "pitch_variation": 0.08},
        "timing":       {"pause_before_ms": 320, "pause_after_ms": 200},
        "knowledge_suppression": {"hiding_knowledge": true, "suppression_weight": 0.70}
      }
    },
    {
      "utterance_id": "ch5_sc3_utt_007",
      "text": "何も知らないよ、そんなこと。",
      "tts_params": {
        "emotion":      {"label": "tension",  "intensity": 0.19,
                         "secondary_label": "anger", "secondary_intensity": 0.14},
        "prosody":      {"speech_rate": 1.06, "pitch_scale": 0.97,
                         "pitch_variation": 0.11, "energy": 0.54},
        "voice_quality": {"breathiness": 0.42, "emphasis_words": ["知らない"]},
        "timing":       {"pause_before_ms": 510, "pause_after_ms": 80,
                         "natural_pause_positions": [5]},
        "knowledge_suppression": {"hiding_knowledge": true, "suppression_weight": 0.70}
      }
    },
    {
      "utterance_id": "ch5_sc3_utt_012",
      "text": "行こう。",
      "tts_params": {
        "emotion":      {"label": "anger", "intensity": 0.61},
        "prosody":      {"speech_rate": 1.14, "pitch_variation": 0.28},
        "timing":       {"pause_before_ms": 180, "pause_after_ms": 0},
        "knowledge_suppression": {"hiding_knowledge": false, "suppression_weight": 0.0}
      }
    }
  ]
}
```

3つのセリフを比較すると、`utt_012`の「行こう。」では`hiding_knowledge: false`かつ`emotion_intensity: 0.61`と高く、前の2つのセリフで感情を抑制していたエレンが、その場面の終盤で感情を露わにする瞬間の変化がパラメタに反映されています。