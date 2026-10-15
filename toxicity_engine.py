#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
modules/toxicity_engine.py — локальный ИИ-детектор токсичности для Inferno Bot.
modules/toxicity_engine.py — local AI toxicity detector for Inferno Bot.

Модель / Model: cointegrated/rubert-tiny-toxicity (BERT-tiny, 3 слоя, hidden=312).
Загружается лениво ТОЛЬКО в load(); torch импортируется внутри.
Loaded lazily inside load(); torch is imported inside only.
Если веса отсутствуют / torch недоступен — работает детерминированный
rule-engine (backend == "rules").
If weights are missing / torch is unavailable — a deterministic rule engine
runs instead (backend == "rules").

Контракт API (жёсткий, используется toxicity_dashboard.py):
Public API contract (stable, used by toxicity_dashboard.py):
    MODEL_DIR
    class ToxicityEngine:
        __init__, load() -> bool, available (property),
        classify(text) -> dict, classify_many(texts) -> list[dict]
    get_engine() -> ToxicityEngine

Гарантии / Guarantees:
  * import модуля мгновенный, без torch/numpy в топ-уровне.
    module import is instant, no torch/numpy at top level.
  * load/classify/classify_many НИКОГДА не бросают исключений.
    load/classify/classify_many NEVER raise.
  * тяжёлый CPU — только внутри методов (не в import-time).
    heavy CPU work happens only inside methods (never at import time).
  * синхронные методы (никаких asyncio), вызываются из run_in_executor.
    synchronous methods (no asyncio), meant to run via run_in_executor.
"""

from __future__ import annotations

import json
import logging
import math
import os
import re
import string
import threading
from dataclasses import dataclass
from typing import Iterable, Optional, Sequence

__all__ = ["MODEL_DIR", "ToxicityEngine", "get_engine"]

logger = logging.getLogger("toxicity_engine")


# ─────────────────────────────────────────────────────────────────────────────
# Конфигурация / Configuration
# ─────────────────────────────────────────────────────────────────────────────
def _env_int(name: str, default: int) -> int:
    """Безопасное чтение int из окружения / Safely read an int from env."""
    try:
        raw = os.environ.get(name)
        return int(raw) if raw not in (None, "") else default
    except (TypeError, ValueError):
        return default


def _env_float(name: str, default: float) -> float:
    """Безопасное чтение float из окружения / Safely read a float from env."""
    try:
        raw = os.environ.get(name)
        return float(raw) if raw not in (None, "") else default
    except (TypeError, ValueError):
        return default


def _env_flag(name: str, default: bool = False) -> bool:
    """Флаг из окружения: 1/true/yes/on / Env boolean flag."""
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on", "y"}


@dataclass(frozen=True)
class _Config:
    """Неизменяемая конфигурация движка / Immutable engine configuration."""

    model_dir: str = os.environ.get(
        "TOXICITY_MODEL_DIR",
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", "toxicity"),
    )
    model_bin: str = "pytorch_model.bin"
    vocab: str = "vocab.txt"
    config_json: str = "config.json"
    special_tokens_map: str = "special_tokens_map.json"

    max_seq_len: int = _env_int("TOXICITY_MAX_SEQ_LEN", 64)
    batch_size: int = _env_int("TOXICITY_BATCH_SIZE", 32)
    threshold: float = _env_float("TOXICITY_THRESHOLD", 0.5)
    num_threads: int = _env_int("TOXICITY_NUM_THREADS", 0)  # 0 => auto

    # Ожидаемый размер pytorch_model.bin; строгая проверка — по флагу.
    # Expected pytorch_model.bin size; strict check is opt-in.
    model_bin_size: int = _env_int("TOXICITY_MODEL_BIN_SIZE", 47168503)
    strict_size: bool = _env_flag("TOXICITY_STRICT_SIZE", False)

    # Дефолты архитектуры (fallback, если config.json недоступен).
    # Architecture defaults (fallback when config.json is unavailable).
    vocab_size: int = _env_int("TOXICITY_VOCAB_SIZE", 29564)
    hidden: int = _env_int("TOXICITY_HIDDEN", 312)
    intermediate: int = _env_int("TOXICITY_INTERMEDIATE", 600)
    num_layers: int = _env_int("TOXICITY_NUM_LAYERS", 3)
    num_heads: int = _env_int("TOXICITY_NUM_HEADS", 12)
    pos_max: int = _env_int("TOXICITY_POS_MAX", 512)
    type_vocab: int = _env_int("TOXICITY_TYPE_VOCAB", 2)


CFG = _Config()
MODEL_DIR = CFG.model_dir

# Классы токсичности (id 1..4 в чекпоинте) / Toxicity classes (ids 1..4).
CLASSES: tuple[str, ...] = ("insult", "obscenity", "threat", "dangerous")

# Приоритет для детерминированного разрешения ничьих при argmax.
# Priority used to deterministically break argmax ties.
_CLASS_PRIORITY: dict[str, int] = {
    "dangerous": 3, "threat": 2, "obscenity": 1, "insult": 0,
}

_EPS = 1e-12
_SPECIAL = {"[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"}


# ─────────────────────────────────────────────────────────────────────────────
# Ручной WordPiece-токенизатор / Hand-rolled WordPiece tokenizer
# ─────────────────────────────────────────────────────────────────────────────
class _WordPieceTokenizer:
    """WordPiece: do_lower_case=False, basic-tokenize = пробелы + пунктуация.
    WordPiece tokenizer: do_lower_case=False, basic tokenize = whitespace + punct.
    """

    def __init__(self, vocab_path: str, max_len: int = CFG.max_seq_len):
        self.vocab: dict[str, int] = {}
        self.tokens: list[str] = []          # id -> token (для отладки / for debug)
        self._saw_unk = False
        with open(vocab_path, "r", encoding="utf-8", errors="replace") as fh:
            for idx, line in enumerate(fh):
                token = line.rstrip("\n")
                self.vocab[token] = idx
                self.tokens.append(token)

        # id спецтокенов из vocab.txt (config: pad_token_id=0).
        # Special-token ids from vocab.txt (config: pad_token_id=0).
        self.pad_id = self.vocab.get("[PAD]", 0)
        self.unk_id = self.vocab.get("[UNK]", 1)
        self.cls_id = self.vocab.get("[CLS]", 2)
        self.sep_id = self.vocab.get("[SEP]", 3)
        self.mask_id = self.vocab.get("[MASK]", 4)
        self.max_len = max(2, int(max_len))

        # Пунктуация, отделяемая от слов (RU + ASCII).
        # Punctuation split off words (Russian + ASCII).
        self._punct: frozenset[str] = frozenset(string.punctuation) | frozenset(
            "«»—–…„“”‘’·№"
        )

    # -- basic tokenize ------------------------------------------------------
    def _basic_tokenize(self, text: str) -> list[str]:
        out: list[str] = []
        for raw in text.split():
            if raw in _SPECIAL:
                out.append(raw)
                continue
            out.extend(self._split_punct(raw))
        return out

    def _split_punct(self, token: str) -> list[str]:
        out: list[str] = []
        start: Optional[int] = None
        for i, ch in enumerate(token):
            if ch in self._punct:
                if start is not None:
                    out.append(token[start:i])
                    start = None
                out.append(ch)
            elif start is None:
                start = i
        if start is not None:
            out.append(token[start:])
        return out

    # -- wordpiece -----------------------------------------------------------
    def _wordpiece(self, word: str) -> list[int]:
        if not word:
            return []
        direct = self.vocab.get(word)
        if direct is not None:
            return [direct]

        ids: list[int] = []
        start = 0
        n = len(word)
        while start < n:
            end = n
            found: Optional[str] = None
            while start < end:
                sub = word[start:end]
                cand = sub if start == 0 else "##" + sub
                if cand in self.vocab:
                    found = cand
                    break
                end -= 1
            if found is None:
                self._saw_unk = True
                return [self.unk_id]
            ids.append(self.vocab[found])
            start = end
        return ids

    # -- публичные / public --------------------------------------------------
    def tokenize(self, text: str) -> list[int]:
        """WordPiece-id фразы без [CLS]/[SEP].
        WordPiece ids of the phrase without [CLS]/[SEP].
        """
        ids: list[int] = []
        for word in self._basic_tokenize(text or ""):
            ids.extend(self._wordpiece(word))
        return ids

    def encode(self, text: str) -> list[int]:
        """[CLS] + wordpiece + [SEP], обрезка до max_len.
        [CLS] + wordpiece + [SEP], truncated to max_len.
        """
        ids: list[int] = [self.cls_id]
        for word in self._basic_tokenize(text or ""):
            ids.extend(self._wordpiece(word))
            if len(ids) >= self.max_len - 1:
                break
        ids.append(self.sep_id)
        return ids[: self.max_len]


# ─────────────────────────────────────────────────────────────────────────────
# Fallback rule-engine / Fallback rule engine
# ─────────────────────────────────────────────────────────────────────────────
_NEG = frozenset({"не", "ни"})
_NEG_MOD = frozenset({
    "очень", "оч", "такой", "такая", "такие", "совсем", "вовсе",
    "особо", "настолько", "столь", "прям", "прямо", "ничуть",
    "нисколько", "вообще", "слишком",
})

# Непересекающиеся лексиконы (морфоформы RU).
# Disjoint lexicons (Russian word forms).
_LEX_INSULT = frozenset({
    "идиот", "идиотка", "идиоты", "идиотский", "идиотская",
    "дурак", "дура", "дурачок", "дурочка", "дурацкий", "дурацкая",
    "тупой", "тупая", "тупое", "тупые", "тупица", "тупень", "тупарь",
    "дебил", "дебилка", "дебилы", "дебильный",
    "кретин", "кретинка", "кретины",
    "урод", "уродка", "уроды", "уродливый", "уродина",
    "придурок", "придурки", "придурочный",
    "мудак", "мудаки", "мудацкий",
    "чмо", "чмошник", "чмошница",
    "сволочь", "сволочи",
    "тварь", "твари",
    "мразь", "мрази", "мразота",
    "гад", "гадина", "гадёныш", "гаденыш",
    "мерзавец", "мерзавка",
    "подонок", "подонки",
    "отморозок", "отморозки",
    "ублюдок", "ублюдки",
    "негодяй", "негодяи",
    "козёл", "козел", "козлы",
    "бестолочь", "бездарь", "недоумок", "глупец",
    "засранец", "говнюк", "говнючка",
    "скотина", "быдло",
})

_LEX_OBSCENITY = frozenset({
    "сука", "суки", "сучара", "сучка", "сучонок", "сучёнок", "сучий",
    "блядь", "бля", "блять", "блядина", "бляди", "блядский", "блядская",
    "хуй", "хуи", "хуйня", "хуёвый", "хуевый", "хуёво", "хуево",
    "нахуй", "нахер", "нахуя", "похуй", "похер", "дохуя", "нихуя",
    "охуел", "охуеть", "ахуел", "ахуеть", "хуесос", "хуйло",
    "заебал", "заебали", "заёб", "заеб", "ебал", "ебать", "ебаный", "ёбаный",
    "ебанутый", "еблан", "ебланы", "долбоёб", "долбоеб", "выебок", "мудень",
    "пизда", "пиздец", "пиздатый", "пиздишь", "спиздил",
    "пидор", "пидоры", "пидорас", "пидорасы", "петух", "петушара",
    "говно", "говённый", "говенный", "говнище",
    "жопа", "жопы", "срака", "сраный", "дерьмо", "дерьмовый", "мудила",
})

_LEX_THREAT = frozenset({
    "убью", "убьёшь", "убьешь", "убьёт", "убьет", "убьём", "убьем",
    "убьёте", "убьете", "убьют", "убей", "убейся", "убейтесь",
    "зарежу", "зарежешь", "зарежет", "прирежу", "прирежет",
    "вырежу", "вырежет", "перережу",
    "сдохни", "сдохнешь", "умри", "умрите", "умирай",
    "задушу", "задушишь", "задушит", "придушу", "удушу",
    "разорву", "разорвёшь", "изуродую", "покалечу", "изобью", "избил",
    "похороню", "закопаю", "прикончу", "грохну", "уничтожу", "уничтожишь",
    "застрелю", "пристрелю", "расстреляю", "взорву", "подорву",
    "повешу", "казню", "казнишь",
})

_LEX_DANGEROUS = frozenset({
    "повесься", "повесьтесь", "повешусь", "повесился",
    "застрелись", "застрелитесь", "застрелюсь",
    "спрыгни", "спрыгну", "спрыгнул", "прыгни", "выпрыгни",
    "вскройся", "вскрой", "порежься", "порежусь", "перережь",
    "суицид", "суицидальный", "самоубийство", "самоубийца", "суициднуться",
    "отбеливатель", "отравись", "отравлюсь", "вены",
})

_LABEL_TO_LEX: dict[str, frozenset[str]] = {
    "insult": _LEX_INSULT,
    "obscenity": _LEX_OBSCENITY,
    "threat": _LEX_THREAT,
    "dangerous": _LEX_DANGEROUS,
}

_ALL_LEXICON: frozenset[str] = frozenset().union(*_LABEL_TO_LEX.values())

_WORD_RE = re.compile(r"[а-яёa-z0-9]+", re.IGNORECASE)


def _make_result(probs: dict[str, float], backend: str) -> dict:
    """Единый сборщик результата: детерминированный argmax.
    Single result builder: deterministic argmax with priority tie-break.
    """
    top = max(CLASSES, key=lambda c: (probs.get(c, 0.0), _CLASS_PRIORITY[c]))
    if probs.get(top, 0.0) <= CFG.threshold:
        top = None
    return {
        "toxic": top is not None,
        "probs": {c: float(round(float(probs.get(c, 0.0)), 6)) for c in CLASSES},
        "top_class": top,
        "severity": top or "none",
        "backend": backend,
    }


def _empty_result() -> dict:
    """Безопасный результат «нет токсичности» / Safe non-toxic result."""
    return {
        "toxic": False,
        "probs": {c: 0.0 for c in CLASSES},
        "top_class": None,
        "severity": "none",
        "backend": "rules",
    }


class _RuleEngine:
    """Детерминированный rule-engine: те же поля, backend='rules'.
    Deterministic rule engine: same result shape, backend='rules'.
    """

    @staticmethod
    def _is_negated(tokens: Sequence[str], i: int) -> bool:
        w = tokens[i]
        # Приклеенное отрицание: "нетупой" -> база "тупой" токсична.
        # Glued negation: "нетупой" -> base "тупой" is toxic.
        if w.startswith("не") and w[2:] in _ALL_LEXICON:
            return True
        if i >= 1 and tokens[i - 1] in _NEG:
            return True
        if i >= 2 and tokens[i - 1] in _NEG_MOD and tokens[i - 2] in _NEG:
            return True
        return False

    def _scan(self, text: str) -> dict[str, list[str]]:
        found: dict[str, list[str]] = {c: [] for c in CLASSES}
        tokens = [t.lower() for t in _WORD_RE.findall(text or "")]
        for i, w in enumerate(tokens):
            if self._is_negated(tokens, i):
                continue
            for cls in CLASSES:
                if w in _LABEL_TO_LEX[cls]:
                    found[cls].append(w)
                    break
        return found

    def classify(self, text: str) -> dict:
        found = self._scan(text)
        probs: dict[str, float] = {}
        for cls in CLASSES:
            n = len(found[cls])
            probs[cls] = round(min(0.7 + 0.15 * n, 0.97), 6) if n > 0 else 0.05
        return _make_result(probs, "rules")


# ─────────────────────────────────────────────────────────────────────────────
# Архитектура BERT-tiny (torch) — определяется ТОЛЬКО внутри load().
# BERT-tiny architecture (torch) — defined ONLY inside load().
# ─────────────────────────────────────────────────────────────────────────────
def _load_arch_config(path: str) -> dict:
    """Читает config.json (если есть) и возвращает параметры архитектуры.
    Reads config.json (if present) and returns architecture parameters.
    """
    defaults = {
        "vocab_size": CFG.vocab_size,
        "hidden": CFG.hidden,
        "intermediate": CFG.intermediate,
        "num_layers": CFG.num_layers,
        "num_heads": CFG.num_heads,
        "pos_max": CFG.pos_max,
        "type_vocab": CFG.type_vocab,
    }
    try:
        if not os.path.exists(path):
            return defaults
        with open(path, "r", encoding="utf-8", errors="replace") as fh:
            raw = json.load(fh)
    except Exception as exc:
        logger.warning("toxicity: config.json не прочитан (%s) -> дефолты", exc)
        return defaults

    def pick(*keys, default):
        for k in keys:
            v = raw.get(k)
            if isinstance(v, int) and v > 0:
                return v
        return default

    return {
        "vocab_size": pick("vocab_size", default=defaults["vocab_size"]),
        "hidden": pick("hidden_size", default=defaults["hidden"]),
        "intermediate": pick("intermediate_size", default=defaults["intermediate"]),
        "num_layers": pick("num_hidden_layers", default=defaults["num_layers"]),
        "num_heads": pick("num_attention_heads", default=defaults["num_heads"]),
        "pos_max": pick("max_position_embeddings", default=defaults["pos_max"]),
        "type_vocab": pick("type_vocab_size", default=defaults["type_vocab"]),
    }


def _build_torch_model(torch, nn, F, bin_path: str, arch: dict):
    """Собирает rubert-tiny-toxicity и грузит веса.
    Builds rubert-tiny-toxicity and loads weights.

    Ключи state_dict совпадают с оригинальным чекпоинтом:
        bert.embeddings.* / bert.encoder.layer.{i}.* / bert.pooler.dense.* /
        classifier.*
    Возвращает модель в eval-режиме или None при ошибке.
    Returns the model in eval mode, or None on failure.
    """
    emb: int = arch["hidden"]
    n_heads: int = arch["num_heads"]
    head_size: int = max(1, emb // n_heads)
    inter: int = arch["intermediate"]
    n_layers: int = arch["num_layers"]
    vocab: int = arch["vocab_size"]
    pos_max: int = arch["pos_max"]
    type_vocab: int = arch["type_vocab"]

    class BertSelfAttention(nn.Module):
        def __init__(self):
            super().__init__()
            self.query = nn.Linear(emb, emb)
            self.key = nn.Linear(emb, emb)
            self.value = nn.Linear(emb, emb)
            self.n_head = n_heads
            self.head_size = head_size

        def _split(self, x):
            b, l, _ = x.shape
            return x.view(b, l, self.n_head, self.head_size).permute(0, 2, 1, 3)

        def forward(self, hidden, ext_mask):
            q = self._split(self.query(hidden))
            k = self._split(self.key(hidden))
            v = self._split(self.value(hidden))
            scores = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(self.head_size)
            if ext_mask is not None:
                scores = scores + ext_mask
            attn = F.softmax(scores, dim=-1)
            out = torch.matmul(attn, v)
            b, n, l, hd = out.shape
            return out.permute(0, 2, 1, 3).contiguous().view(b, l, emb)

    class BertSelfOutput(nn.Module):
        def __init__(self):
            super().__init__()
            self.dense = nn.Linear(emb, emb)
            self.LayerNorm = nn.LayerNorm(emb, eps=_EPS)

        def forward(self, hidden, residual):
            return self.LayerNorm(self.dense(hidden) + residual)

    class BertAttention(nn.Module):
        def __init__(self):
            super().__init__()
            self.self = BertSelfAttention()
            self.output = BertSelfOutput()

        def forward(self, hidden, ext_mask):
            return self.output(self.self(hidden, ext_mask), hidden)

    class BertIntermediate(nn.Module):
        def __init__(self):
            super().__init__()
            self.dense = nn.Linear(emb, inter)

        def forward(self, hidden):
            return F.gelu(self.dense(hidden))

    class BertOutput(nn.Module):
        def __init__(self):
            super().__init__()
            self.dense = nn.Linear(inter, emb)
            self.LayerNorm = nn.LayerNorm(emb, eps=_EPS)

        def forward(self, hidden, residual):
            return self.LayerNorm(self.dense(hidden) + residual)

    class BertLayer(nn.Module):
        def __init__(self):
            super().__init__()
            self.attention = BertAttention()
            self.intermediate = BertIntermediate()
            self.output = BertOutput()

        def forward(self, hidden, ext_mask):
            hidden = self.attention(hidden, ext_mask)
            return self.output(self.intermediate(hidden), hidden)

    class BertEncoder(nn.Module):
        def __init__(self):
            super().__init__()
            self.layer = nn.ModuleList([BertLayer() for _ in range(n_layers)])

        def forward(self, hidden, ext_mask):
            for layer in self.layer:
                hidden = layer(hidden, ext_mask)
            return hidden

    class BertEmbeddings(nn.Module):
        def __init__(self):
            super().__init__()
            self.word_embeddings = nn.Embedding(vocab, emb, padding_idx=0)
            self.position_embeddings = nn.Embedding(pos_max, emb)
            self.token_type_embeddings = nn.Embedding(type_vocab, emb)
            self.LayerNorm = nn.LayerNorm(emb, eps=_EPS)

        def forward(self, input_ids):
            b, l = input_ids.shape
            words = self.word_embeddings(input_ids)
            pos_ids = torch.arange(l, device=input_ids.device).unsqueeze(0).expand(b, l)
            positions = self.position_embeddings(pos_ids)
            token_types = self.token_type_embeddings(torch.zeros_like(input_ids))
            return self.LayerNorm(words + positions + token_types)

    class BertPooler(nn.Module):
        def __init__(self):
            super().__init__()
            self.dense = nn.Linear(emb, emb)

        def forward(self, hidden):
            return torch.tanh(self.dense(hidden[:, 0]))

    class BertModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.embeddings = BertEmbeddings()
            self.encoder = BertEncoder()
            self.pooler = BertPooler()

        def forward(self, input_ids, attention_mask=None):
            ext_mask = None
            if attention_mask is not None:
                m = attention_mask.unsqueeze(1).unsqueeze(2).to(torch.float32)
                ext_mask = (1.0 - m) * -10000.0
            hidden = self.embeddings(input_ids)
            hidden = self.encoder(hidden, ext_mask)
            return self.pooler(hidden)

    class RuBertTinyToxicity(nn.Module):
        def __init__(self):
            super().__init__()
            self.bert = BertModel()
            self.classifier = nn.Linear(emb, 5)

        def forward(self, input_ids, attention_mask=None):
            pooled = self.bert(input_ids, attention_mask)
            return self.classifier(pooled)

    # ── Загрузка state_dict / Load state_dict ────────────────────────────────
    try:
        try:
            state_dict = torch.load(bin_path, map_location="cpu", weights_only=True)
        except TypeError:
            # Старые версии torch без weights_only / Older torch without weights_only.
            state_dict = torch.load(bin_path, map_location="cpu")
    except Exception as exc:
        logger.warning("toxicity: не удалось прочитать state_dict: %s", exc)
        return None

    # ── Проверка ключей и форм / Validate keys and shapes ────────────────────
    try:
        we = state_dict.get("bert.embeddings.word_embeddings.weight")
        cw = state_dict.get("classifier.weight")
        if we is None or cw is None:
            logger.warning("toxicity: state_dict без word_embeddings/classifier")
            return None
        if tuple(we.shape) != (vocab, emb):
            logger.warning("toxicity: word_embeddings.shape=%s != %s", tuple(we.shape), (vocab, emb))
            return None
        if tuple(cw.shape) != (5, emb):
            logger.warning("toxicity: classifier.weight.shape=%s != %s", tuple(cw.shape), (5, emb))
            return None
    except Exception as exc:
        logger.warning("toxicity: ошибка проверки размеров весов: %s", exc)
        return None

    model = RuBertTinyToxicity()
    try:
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        if missing:
            logger.warning("toxicity: missing keys (%d): %s", len(missing), list(missing)[:4])
        if unexpected:
            # Допустимы только служебные буферы вроде position_ids.
            # Only service buffers such as position_ids are acceptable.
            real_unexpected = [k for k in unexpected if not k.endswith("position_ids")]
            if real_unexpected:
                logger.warning("toxicity: unexpected keys: %s", real_unexpected[:4])
    except Exception as exc:
        logger.warning("toxicity: load_state_dict не удался: %s", exc)
        return None

    model.eval()
    return model


# ─────────────────────────────────────────────────────────────────────────────
# Движок / Engine
# ─────────────────────────────────────────────────────────────────────────────
class ToxicityEngine:
    """Лениво загружаемый детектор токсичности.
    Lazily-loaded toxicity detector.

    Ничего тяжёлого в __init__; все публичные методы безопасны и не бросают.
    Nothing heavy in __init__; all public methods are safe and never raise.
    """

    def __init__(self) -> None:
        self._model = None            # torch nn.Module или None
        self._torch = None            # модуль torch (если загружен)
        self._tokenizer: Optional[_WordPieceTokenizer] = None
        self._rule_engine: Optional[_RuleEngine] = None
        self._attempted = False       # была ли попытка загрузки
        self._lock = threading.Lock()

    # ── Внутреннее / Internals ───────────────────────────────────────────────
    def _rule(self) -> _RuleEngine:
        if self._rule_engine is None:
            self._rule_engine = _RuleEngine()
        return self._rule_engine

    def _load_tokenizer(self) -> None:
        """Лёгкая загрузка vocab.txt (stdlib).
        Lightweight vocab.txt load (stdlib only).
        """
        vocab_path = os.path.join(CFG.model_dir, CFG.vocab)
        if not os.path.exists(vocab_path):
            logger.warning("toxicity: vocab.txt не найден -> токенизатор недоступен")
            self._tokenizer = None
            return
        try:
            self._tokenizer = _WordPieceTokenizer(vocab_path, max_len=CFG.max_seq_len)
        except Exception as exc:
            logger.warning("toxicity: ошибка загрузки токенизатора: %s", exc)
            self._tokenizer = None

    def _check_weights(self) -> Optional[str]:
        """Проверяет наличие и размер весов. Возвращает путь или None.
        Checks weights presence/size. Returns path or None.
        """
        bin_path = os.path.join(CFG.model_dir, CFG.model_bin)
        if not os.path.exists(bin_path):
            logger.warning("toxicity: %s отсутствует -> fallback на rules", CFG.model_bin)
            return None
        try:
            size = os.path.getsize(bin_path)
        except Exception as exc:
            logger.warning("toxicity: не удалось определить размер весов: %s", exc)
            return None
        if size != CFG.model_bin_size:
            msg = "toxicity: %s size=%s != %s" % (CFG.model_bin, size, CFG.model_bin_size)
            if CFG.strict_size:
                logger.warning("%s -> fallback на rules (strict)", msg)
                return None
            logger.warning("%s -> продолжаем (non-strict)", msg)
        return bin_path

    def _load_impl(self) -> bool:
        self._load_tokenizer()
        bin_path = self._check_weights()
        if bin_path is None:
            return False

        # ── Тяжёлый импорт torch только здесь / heavy torch import only here ──
        try:
            import torch  # noqa: F401
            import torch.nn as nn  # noqa: F401
            import torch.nn.functional as F  # noqa: F401

            try:
                threads = CFG.num_threads or max(1, min(4, os.cpu_count() or 1))
                torch.set_num_threads(threads)
            except Exception:
                pass

            arch = _load_arch_config(os.path.join(CFG.model_dir, CFG.config_json))
            model = _build_torch_model(torch, nn, F, bin_path, arch)
            if model is None:
                return False

            if self._tokenizer is None:
                logger.warning("toxicity: модель есть, но токенизатор отсутствует")
                return False

            self._torch = torch
            self._model = model
            logger.info("toxicity: модель rubert-tiny-toxicity загружена (bert)")
            return True
        except Exception as exc:
            logger.warning("toxicity: загрузка модели не удалась: %s", exc)
            self._model = None
            return False

    # ── Публичное / Public ───────────────────────────────────────────────────
    def load(self) -> bool:
        """Лениво и идемпотентно грузит токенизатор и веса.
        Lazily and idempotently loads tokenizer and weights.

        Returns:
            True, если torch-модель доступна; иначе False (работает rules).
            True if the torch model is available; otherwise False (rules mode).
        """
        with self._lock:
            if self._model is not None:
                return True
            if self._attempted:
                return False
            self._attempted = True
            try:
                return self._load_impl()
            except Exception as exc:  # pragma: no cover — последняя линия
                logger.warning("toxicity: неожиданная ошибка load(): %s", exc)
                self._model = None
                return False

    @property
    def available(self) -> bool:
        """True, если torch-модель загружена / True if the torch model is loaded."""
        return self._model is not None

    def _ensure_loaded(self) -> None:
        if self._model is None:
            try:
                self.load()
            except Exception:
                pass

    # ── Инференс / Inference ─────────────────────────────────────────────────
    def _encode_batch(self, texts: Sequence[str]):
        tok = self._tokenizer
        if tok is None:
            raise RuntimeError("tokenizer is not loaded")
        ids_batch: list[list[int]] = []
        max_len = 1
        for t in texts:
            ids = tok.encode(t)
            ids_batch.append(ids)
            if len(ids) > max_len:
                max_len = len(ids)
        pad_id = tok.pad_id
        padded: list[list[int]] = []
        mask: list[list[int]] = []
        for ids in ids_batch:
            n = len(ids)
            padded.append(ids + [pad_id] * (max_len - n))
            mask.append([1] * n + [0] * (max_len - n))
        return padded, mask

    def _predict(self, texts: Sequence[str]) -> list[list[float]]:
        """Прогон батча -> list[list[float]] из 5 sigmoid-вероятностей.
        Batch forward -> list[list[float]] of 5 sigmoid probabilities.
        """
        ids_batch, mask_batch = self._encode_batch(texts)
        torch = self._torch
        x = torch.as_tensor(ids_batch, dtype=torch.long)
        m = torch.as_tensor(mask_batch, dtype=torch.long)
        with torch.inference_mode():
            logits = self._model(x, m)
            probs = torch.sigmoid(logits)
        return probs.cpu().tolist()

    # ── Утилиты / Helpers ────────────────────────────────────────────────────
    @staticmethod
    def _probs4(five: Sequence[float]) -> dict[str, float]:
        """5-классовые вероятности -> 4 токсичных класса.
        Five-class probabilities -> four toxic classes.
        """
        return {
            "insult": five[1],
            "obscenity": five[2],
            "threat": five[3],
            "dangerous": five[4],
        }

    @staticmethod
    def _coerce(text) -> str:
        """Мягко приводит вход к строке / Softly coerce input to str."""
        if text is None:
            return ""
        if isinstance(text, str):
            return text
        try:
            return str(text)
        except Exception:
            return ""

    # ── Публичное API / Public API ───────────────────────────────────────────
    def classify(self, text: str) -> dict:
        """Классифицирует один текст. Никогда не бросает.
        Classifies a single text. Never raises.
        """
        try:
            txt = self._coerce(text)
            self._ensure_loaded()
            if self._model is not None and self._tokenizer is not None:
                try:
                    five = self._predict([txt])[0]
                    return _make_result(self._probs4(five), "bert")
                except Exception as exc:
                    logger.warning("toxicity: инференс упал, fallback rules: %s", exc)
            return self._rule().classify(txt)
        except Exception:
            return _empty_result()

    def classify_many(self, texts: Iterable[str]) -> list[dict]:
        """Классифицирует батч текстов. Никогда не бросает.
        Classifies a batch of texts. Never raises.
        """
        # ── Нормализация входа / Input normalization ────────────────────────
        try:
            if texts is None:
                return []
            if isinstance(texts, (str, bytes)):
                seq: list[str] = [self._coerce(texts)]
            else:
                try:
                    seq = [self._coerce(t) for t in texts]
                except TypeError:
                    seq = [self._coerce(texts)]
        except Exception:
            return []

        if not seq:
            return []

        try:
            self._ensure_loaded()
            if self._model is not None and self._tokenizer is not None:
                # Сортируем по длине в токенах, чтобы батчи были однородными:
                # меньше паддинга => быстрее на длинных сообщениях.
                # Sort by token length for homogeneous batches: less padding.
                try:
                    lengths = [len(self._tokenizer.encode(t)) for t in seq]
                except Exception:
                    lengths = [0] * len(seq)
                order = sorted(range(len(seq)), key=lambda i: lengths[i])
                results: list[Optional[dict]] = [None] * len(seq)
                for start in range(0, len(seq), CFG.batch_size):
                    idxs = order[start:start + CFG.batch_size]
                    chunk = [seq[j] for j in idxs]
                    for j, five in zip(idxs, self._predict(chunk)):
                        results[j] = _make_result(self._probs4(five), "bert")
                return [r if r is not None else _empty_result() for r in results]
        except Exception as exc:
            logger.warning("toxicity: batched инференс упал, fallback rules: %s", exc)

        # Fallback: rules для каждого (без смешивания backend'ов).
        # Fallback: rules for every item (no backend mixing).
        try:
            rule = self._rule()
            return [rule.classify(t) for t in seq]
        except Exception:
            return [_empty_result() for _ in seq]


# ─────────────────────────────────────────────────────────────────────────────
# Синглтон / Singleton
# ─────────────────────────────────────────────────────────────────────────────
_engine: Optional[ToxicityEngine] = None
_engine_lock = threading.Lock()


def get_engine() -> ToxicityEngine:
    """Возвращает общий экземпляр движка (ленивый синглтон).
    Returns the shared engine instance (lazy singleton).
    """
    global _engine
    if _engine is None:
        with _engine_lock:
            if _engine is None:
                _engine = ToxicityEngine()
    return _engine