# Copyright (c) Meta Platforms, Inc. and affiliates.
from __future__ import annotations
from typing import List, Optional, Tuple
from dataclasses import dataclass

from lst.tokenizers.abstract_tokenizer import Tokenizer
# from .sentence_piece_tokenizer import SentencePieceTokenizer
from sentencepiece import SentencePieceProcessor
import re
import random


random.seed(23)

from lst.tokenizers.constants import (
    BOE_ID,
    BOS_ID,
    BPE_ID,
    BYTE_UNITS,
    EOS_ID,
    OFFSET,
    PAD_ID,
    TEXT_MODE_ID,
    SPEECH_MODE_ID,
    HUBERT_UNITS,
)
import logging

logger = logging.getLogger()


def split_head_payload(segment: str) -> Tuple[str, str]:
    seg = segment.strip()
    if not seg:
        return "", ""
    sp = seg.split(" ", 1)
    head = sp[0]
    payload = sp[1] if len(sp) > 1 else ""
    return head, payload


def parse_interleaved_pairs(s: str) -> List[Tuple[List[str], List[str]]]:
    """
    Parse an interleaved string into [(text_tokens, hubert_tokens), ...].
    Only handles TEXT/HUBERT pairs; if consecutive TEXTs or HUBERTs appear,
    pair them with the nearest segment of the other type when possible.
    """
    pairs: List[Tuple[List[str], List[str]]] = []
    if not s.startswith("<META MODE SEP>"):
        s = f"<META MODE SEP>TEXT {s}"

    last_text: Optional[List[str]] = None

    for segment in s.split("<META MODE SEP>"):
        if not segment.strip():
            continue
        head, payload = split_head_payload(segment)
        if head == "TEXT":
            toks = payload.strip().split() if payload.strip() else []
            if last_text is not None:
                pairs.append((last_text, []))
            last_text = toks
        elif head == "HUBERT":
            hu = payload.strip().split() if payload.strip() else []
            if last_text is None:
                pairs.append(([], hu))
            else:
                pairs.append((last_text, hu))
                last_text = None
        else:
            raise ValueError(f"Unknown segment head: {head}")

    if last_text is not None:
        pairs.append((last_text, []))
    return pairs


def _take_span(tokens: List[str], start_idx: int, word_len: int, specials: set[str] = {"[BPE]", "[SIL]"}):
    """
    From start_idx, take exactly word_len non-special tokens ([BPE]/[SIL] don't count but are kept in the span).
    Returns (span_tokens, new_start_idx, next_start_idx).
    """
    i = start_idx
    words = 0
    n = len(tokens)
    while i < n and words < word_len:
        if tokens[i] not in specials:
            words += 1
        i += 1
    while i < n and tokens[i] in specials:
        i += 1
    if start_idx > 0 and tokens[start_idx - 1] == '[SIL]':
        start_idx -= 1
    return tokens[start_idx:i], start_idx, i


def _get_word_type_length(
    n_core: int,
    *,
    ratio_speech_text: float = 0.5,
    lower_bound_text: int = 5,
    upper_bound_text: int = 30,
    min_seq: int = 2,
    constant_text_speech_ratio: bool = True,
    upper_bound_by_seq_len: float = 0.6,
    mode: str = "interleaved",
):
    if upper_bound_by_seq_len > 0:
        upper_bound_text = max(min_seq, min(upper_bound_text, int(n_core * upper_bound_by_seq_len)))
        lower_bound_text = max(min_seq, min(lower_bound_text, upper_bound_text))

    prev_is_text = False
    prev_text_len = -1
    word_length_seq: List[int] = []
    word_type_seq: List[str] = []

    curr_type = random.choice(["speech", "text"])
    acc = 0
    while acc < n_core:
        if curr_type == "speech":
            if prev_is_text and constant_text_speech_ratio:
                curr_len = max(min_seq, int(ratio_speech_text * prev_text_len))
            else:
                lo = max(min_seq, int(ratio_speech_text * lower_bound_text))
                hi = max(min_seq, int(ratio_speech_text * upper_bound_text))
                curr_len = random.randint(lo, hi)
            prev_is_text = False
        else:
            curr_len = random.randint(lower_bound_text, upper_bound_text)
            prev_is_text = True
            prev_text_len = curr_len

        word_type_seq.append(curr_type)
        word_length_seq.append(curr_len)
        acc += curr_len
        curr_type = "text" if curr_type == "speech" else "speech"

    overflow = acc - n_core
    if overflow > 0:
        word_length_seq[-1] -= overflow

    if word_length_seq[-1] < 3 and len(word_length_seq) > 1:
        word_length_seq[-2] += word_length_seq[-1]
        word_length_seq.pop()
        word_type_seq.pop()

    return word_type_seq, word_length_seq


def _flatten_pairs_with_frame_counts(
    pairs: List[Tuple[List[str], List[str]]]
) -> Tuple[List[str], List[str], List[int]]:
    """
    Flatten (text_tokens, hubert_tokens) pairs into:
      - flat_text: tokens per TEXT (keep [SIL]/[BPE])
      - flat_frames: concatenated HUBERT frame ids
      - frame_counts_per_text: same length as flat_text; number of HUBERT frames per text token
    If a pair has multiple TEXT tokens, split hubert_tokens proportionally across them (remainder to the last).
    If a pair has HUBERT but no TEXT, insert a virtual [SIL] token (not counted as a core word) to hold the frames.

    """
    flat_text: List[str] = []
    flat_frames: List[str] = []
    frame_counts_per_text: List[int] = []

    for text_toks, hub_toks in pairs:
        # No TEXT but has HUBERT: insert a [SIL] to attach frames (does not increase core count)
        if (not text_toks) and hub_toks:
            flat_text.append("[SIL]")
            flat_frames.extend(hub_toks)
            frame_counts_per_text.append(len(hub_toks))
            continue

        # Has TEXT (possibly 1 or many)
        n_tok = len(text_toks)
        if n_tok == 0:
            continue

        tok = text_toks[0]
        flat_text.append(tok)
        flat_frames.extend(hub_toks)
        frame_counts_per_text.append(len(hub_toks))

    return flat_text, flat_frames, frame_counts_per_text


def _slice_frames_half_sil(span_text, start, end, cum, frame_counts, flat_text, flat_frames):
    """
    Returns:
      - span_frames: sliced frames list
      - frame_lens: lengths per audio token (units aligned to text tokens)
    If the boundary token is [SIL] and it's not the very beginning/end, cut that [SIL] by half.
    """
    f_start = cum[start]
    f_end = cum[end]

    head_is_sil = (len(span_text) > 0 and span_text[0] == '[SIL]' and start > 0)
    tail_is_sil = (len(span_text) > 0 and span_text[-1] == '[SIL]' and end < len(flat_text))
    single_sil = (len(span_text) == 1 and span_text[0] == '[SIL]' and (start > 0 or end < len(flat_text)))

    adj_f_start, adj_f_end = f_start, f_end

    frame_lens = frame_counts[start:end]

    if head_is_sil and not single_sil:
        sil_len = frame_counts[start]
        adj_f_start = min(adj_f_start + sil_len // 2, adj_f_end)
        frame_lens[0] -= sil_len // 2

    if tail_is_sil:
        sil_len_tail = frame_counts[end - 1]
        adj_f_end = max(adj_f_start, adj_f_end - sil_len_tail // 2)
        frame_lens[-1] -= sil_len_tail // 2

    span_frames = flat_frames[adj_f_start:adj_f_end]

    frame_lens = [l for l in frame_lens if l > 0]

    if sum(frame_lens) != len(span_frames):
        import pdb; pdb.set_trace()
    assert sum(frame_lens) == len(span_frames), (
        f"Frame length mismatch: sum(frame_lens)={sum(frame_lens)}, "
        f"len(span_frames)={len(span_frames)}"
    )

    return span_frames, frame_lens


def _random_interleaved_string_and_hu(self, s: str, *, lower_bound_text: int = 5, upper_bound_text: int = 30):
    """
    segment by spans of text tokens, but slice audio using prefix sums over flat_frames for the corresponding lengths
    """
    pairs = parse_interleaved_pairs(s)

    # Flatten and build per-text-token frame counts
    flat_text, flat_frames, frame_counts = _flatten_pairs_with_frame_counts(pairs)

    # Count of non-special tokens (to decide the total length of random blocks)
    text_core_count = sum(1 for t in flat_text if t not in {"[SIL]", "[BPE]"})
    if text_core_count == 0:
        # No core text: output a single HUBERT segment (fallback)
        inter_str = ""
        if flat_frames:
            inter_str = "<META MODE SEP>HUBERT " + " ".join(flat_frames)
        hu_length = [len(flat_frames)] if flat_frames else []
        inter_str = inter_str[15:] if inter_str.startswith("<META MODE SEP>") else inter_str
        return inter_str, hu_length

    # Generate random block layout ("text"/"speech" with lengths in core-text tokens)
    word_type_seq, word_length_seq = _get_word_type_length(
        text_core_count,
        lower_bound_text=lower_bound_text,
        upper_bound_text=upper_bound_text,
        mode="interleaved",
    )

    # Prepare prefix sums over frames so [start:end) token range maps to frame indices
    # cum[i] = total frames before the i-th text token
    cum = [0]
    for c in frame_counts:
        cum.append(cum[-1] + c)

    pieces: List[str] = []
    hu_length_total: List[int] = []
    end = 0

    for block_type, block_len in zip(word_type_seq, word_length_seq):
        start = end
        # Take block_len non-special TEXT tokens; include trailing [SIL]/[BPE]
        span_text, start, end = _take_span(flat_text, start, block_len)

        if block_type == "text":
            if span_text:
                pieces.append("<META MODE SEP>TEXT " + " ".join(span_text))
        else:  # speech
            span_frames, audio_lens = _slice_frames_half_sil(span_text, start, end, cum, frame_counts, flat_text, flat_frames)
            pieces.append("<META MODE SEP>HUBERT " + " ".join(span_frames))

            if isinstance(audio_lens, list):
                hu_length_total += [audio_lens]

    inter_str = " ".join(pieces)
    inter_str = inter_str[15:] if inter_str.startswith("<META MODE SEP>") else inter_str
    return inter_str.replace(" [SIL]", ""), ([hu_length_total] if hu_length_total else [])


class LstTokenizer(Tokenizer):
    """
    LstTokenizer
    - TEXT: SentencePieceTokenizer
    - SPEECH: built-in "sep" / "hubert_bpe" (mirrors the full version)
    - interleaved encode/decode supported
    - Optional hu_length: when encode(..., return_hu_length=True),
      only meaningful for mode="interleaved" (None otherwise)

    Special offset rules (compatible with the full version):
      base_offset = 4
      if add_mode_token=True then +2
      speech token id = text_vocab_size + base_offset (+ optional mode token) + raw id
    """

    def __init__(
        self,
        *,
        # === text ===
        bpe_tokenizer_path: str,
        tokenization_mode: str = "lst",
        add_bos: bool = True,
        add_eos: bool = True,

        # === speech ===
        speech_mode: str = "sep",
        hubert_bpe_model_path: Optional[str] = None,

        # === offsets & options ===
        add_mode_token: bool = False,
        speech_vocab_size_unit: int = HUBERT_UNITS,

    ) -> None:
        assert tokenization_mode == "lst"
        assert speech_mode in ("sep", "hubert_bpe")

        # text side
        self.add_bos = add_bos
        self.add_eos = add_eos

        self.text_tok = SentencePieceProcessor(model_file=bpe_tokenizer_path)
        self.n_words = self.text_tok.vocab_size()
        self.boe_id = BOE_ID
        self.bos_id = self.text_tok.bos_id()
        self.eos_id = self.text_tok.eos_id()
        self.pad_id = self.text_tok.pad_id()

        # self.bpe_id = BPE_ID
        logger.info(
            f"#words: {self.n_words} - BOS ID: {self.bos_id} - EOS ID: {self.eos_id} - PAD ID {self.pad_id} - BOE ID {self.boe_id}"
        )

        assert self.text_tok.vocab_size() == self.text_tok.get_piece_size()

        self.text_vocab_size = self.text_tok.vocab_size()
        self.speech_vocab_size_unit = speech_vocab_size_unit


        # --- BLT offset rule (add_mode_token always True) ---
        self.add_mode_token = True
        self.offsetting_special_char = OFFSET + 2  # 4 + 2 = 6
        self.speech_offset_start = self.text_vocab_size + self.offsetting_special_char
        # Mode token ids (same offset base as the full version)
        self.text_mode_id = TEXT_MODE_ID + self.text_vocab_size
        self.speech_mode_id = SPEECH_MODE_ID + self.text_vocab_size

        # speech side (internal)
        self.speech_mode = speech_mode

        # offsets
        base_offset = OFFSET
        if self.add_mode_token:
            base_offset += 2
        self.offsetting_special_char: int = base_offset
        self.speech_offset_start: int = self.text_vocab_size + self.offsetting_special_char

        self.n_words = (
            self.text_vocab_size
            + self.offsetting_special_char
            + self.speech_vocab_size_unit
        )

        self.modes = ["text", self.speech_mode]

    def get_vocab_size(self) -> int:
        return self.n_words

    # ---------- public API ----------
    def encode(
        self,
        s: str,
        mode: Optional[str] = None,
        *,
        bos: Optional[bool] = None,
        eos: Optional[bool] = None,
        return_hu_length: bool = True,
    ):
        """
        Supports:
        - None/"text"/"jsonl" -> pure text via SentencePiece
        - "sep"/"hubert_bpe" -> pure speech (non-recursive)
        - "interleaved" -> segment-wise (TEXT via text, HUBERT via self.speech_mode)
        - "on_the_fly" -> randomly interleave to a string, then run interleaved; can return hu_length
        - "on_the_fly_eval" -> deterministic interleaving from existing pairs; hu_length can be omitted
        """

        # Default: treat as text
        if mode is None or mode in ("text", "jsonl"):
            t = self._encode_text(s, bos=bos, eos=eos)
            t = self._wrap_bos_eos(t, bos=bos, eos=eos)
            return (t, None) if return_hu_length else t

        # Pure speech
        if mode in ("sep", "hubert_bpe"):
            t = self._encode_speech(s, mode=mode, bos=bos, eos=eos)
            t = self._wrap_bos_eos(t, bos=bos, eos=eos)
            return (t, None) if return_hu_length else t

        # on-the-fly: random interleave → encode → optionally return hu_length
        if mode == "on_the_fly":
            interleaved_string, hu_length = _random_interleaved_string_and_hu(self, s)
            tokens = self._encode_interleaved(
                interleaved_string,
                bos=bos,
                eos=eos,
                compute_hu=True
            )
            tokens = self._wrap_bos_eos(tokens, bos=bos, eos=eos)

            return (tokens, hu_length) if return_hu_length else tokens

        # on-the-fly eval: no randomness; interleave pairs in order
        if mode == "on_the_fly_eval":
            tokens, _ = self._encode_interleaved(s, bos=bos, eos=eos, compute_hu=True)
            return (tokens, None) if return_hu_length else tokens

        raise ValueError(f"{mode} not supported")

    def decode(self, ids: List[int], *, mode: Optional[str] = None, cut_at_eos: bool = True) -> str:
        if mode is None or mode == "text":
            return self._decode_text(ids, cut_at_eos=cut_at_eos)

        if mode in ("sep", "hubert_bpe"):
            return self._decode_speech(ids, mode=mode, cut_at_eos=cut_at_eos)

        if mode == "interleaved":
            return self._decode_interleaved_runs(ids, cut_at_eos=cut_at_eos)

        raise ValueError(f"{mode} not supported")

    def encode_str(self, s: str) -> List[str]:
        return self.text_tok.encode_str(s)

    def get_mode(self, tokens: List[int]) -> str:
        boundary = self.speech_offset_start
        return self.modes[1] if any(t >= boundary for t in tokens) else self.modes[0]

    # ---------- text ----------
    def _encode_text(self, s: str, *, bos: Optional[bool], eos: Optional[bool]) -> List[int]:
        """
        Encode TEXT with the SentencePiece model.
        - Never call encode() recursively.
        - Optionally prepend a mode token when self.add_mode_token is True.
        - Only wrap BOS/EOS according to the given flags.
        """
        ids = self.text_tok.encode(s) if s else []
        if self.add_mode_token and self.text_mode_id is not None:
            ids = [self.text_mode_id] + ids

        if bos:
            ids = [self.bos_id] + ids
        if eos:
            ids = ids + [self.eos_id]
        return ids

    def _decode_text(self, ids: List[int], *, cut_at_eos: bool) -> str:
        return self.text_tok.decode(ids, cut_at_eos=cut_at_eos)


    def _encode_speech(self, s: str, *, mode: str, bos: Optional[bool], eos: Optional[bool]) -> List[int]:
        """
        Encode HUBERT frames in 'sep' mode:
        - Input s is a whitespace-separated list of frame ids (e.g., "117 313 6 ...").
        - Each id is shifted by self.speech_offset_start.
        - Optionally add a speech mode token if self.add_mode_token is True.
        - Only wrap BOS/EOS according to the given flags.
        """
        if mode != "sep":
            raise ValueError(f"Unsupported speech mode {mode}; expected 'sep'.")

        toks = s.split() if s.strip() else []
        # Offset HUBERT ids into the shared vocab space
        ids = [int(u) + self.speech_offset_start for u in toks]

        if self.add_mode_token and self.speech_mode_id is not None:
            ids = [self.speech_mode_id] + ids

        if bos:
            ids = [self.bos_id] + ids
        if eos:
            ids = ids + [self.eos_id]
        return ids

    def _decode_speech(self, ids: List[int], *, mode: str, cut_at_eos: bool) -> str:
        toks = ids
        if cut_at_eos and self.eos_id in toks:
            toks = toks[: toks.index(self.eos_id) + 1]

        body = []
        for t in toks:
            if t == self.bos_id or t == self.eos_id:
                continue
            body.append(t)

        if mode == "sep":
            raw = [str(t - self.speech_offset_start) for t in body]
            return " ".join(raw)
        elif mode == "hubert_bpe":
            raw = [str(t - self.speech_offset_start) for t in body]
            return " ".join(raw)
        else:
            raise ValueError(f"Unsupported speech mode {mode}")

    # ---------- interleaved ----------
    def _encode_interleaved(
        self,
        s: str,
        *,
        bos: Optional[bool],
        eos: Optional[bool],
        compute_hu: bool
    ) -> Tuple[List[int], Optional[List[int]]]:
        """
        Non-recursive interleaved encoder:
        - Parse into (TEXT, HUBERT) pairs
        - Encode each sub-segment WITHOUT BOS/EOS
        - Concatenate and wrap BOS/EOS once at the outer level
        - Optionally compute hu_length per pair
        """

        tokens: List[int] = []
        hu_length_total: List[int] = []

        for segment in s.split("<META MODE SEP>"):
            # TEXT segment
            if segment[0] == "T":
                txt_ids = self._encode_text(segment[5:], bos=False, eos=False)
                tokens.extend(txt_ids)

            # HUBERT segment
            if segment[0] == "H":
                sp_ids = self._encode_speech(segment[7:], mode=self.speech_mode, bos=False, eos=False)
                tokens.extend(sp_ids)

        return tokens

    def _decode_interleaved_runs(self, ids: List[int], *, cut_at_eos: bool) -> str:
        if not ids:
            return ""
        toks = ids
        if cut_at_eos and self.eos_id in toks:
            toks = toks[: toks.index(self.eos_id) + 1]

        pieces: List[str] = []
        buf: List[int] = []
        kind: Optional[str] = None
        boundary = self.speech_offset_start

        def flush(k: Optional[str], b: List[int]) -> None:
            if k is None or not b:
                return
            if k == "text":
                txt = self._decode_text(b, cut_at_eos=False)
                if txt:
                    pieces.append(f"<META MODE SEP>TEXT {txt}")
            else:
                sp = self._decode_speech(b, mode=self.speech_mode, cut_at_eos=False)
                if sp:
                    pieces.append(f"<META MODE SEP>HUBERT {sp}")
            b.clear()

        for t in toks:
            if t == self.bos_id or t == self.eos_id:
                continue
            curr = "speech" if t >= boundary else "text"
            if kind is None:
                kind = curr
            if curr != kind:
                flush(kind, buf)
                kind = curr
            buf.append(t)
        flush(kind, buf)

        out = " ".join(pieces)
        return out[15:] if out.startswith("<META MODE SEP>") else out

    # ---------- helpers ----------
    @staticmethod
    def _pick(x: Optional[bool], default: bool) -> bool:
        return default if x is None else x

    def _wrap_bos_eos(self, ids: List[int], *, bos: Optional[bool], eos: Optional[bool]) -> List[int]:
        _bos = self._pick(bos, self.bos_id)
        _eos = self._pick(eos, self.eos_id)
        if _bos:
            ids = [self.bos_id] + ids
        if _eos:
            ids = ids + [self.eos_id]
        return ids

    def get_token_offsets(self, text: str, tokens: list[int] | None = None):
        # TODO: Figure out what this does
        raise NotImplementedError()
