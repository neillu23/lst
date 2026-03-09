# Copyright (c) Meta Platforms, Inc. and affiliates.
import math
import os
import time
from collections import defaultdict
from contextlib import nullcontext
from enum import Enum
import random

import torch
from pydantic import BaseModel
from torch.nn import functional as F

from lst.distributed import get_local_rank
from lst.entropy_model import load_entropy_model

# from src.slurm import get_local_rank
from lst.tokenizers.blt_tokenizer import BPE_ID, OFFSET
from lst.tokenizers.lst_tokenizer import LstTokenizer
from lst.tokenizers.constants import BPE_ID, OFFSET


class PatchingModeEnum(str, Enum):
    entropy = "entropy"
    bpe = "bpe"
    bpe_patcher = "bpe_patcher"
    space = "space"
    static = "static"
    byte = "byte"
    lst_static = "lst_static"
    lst_align = "lst_align"
    lst_curr = "lst_curriculum"


class PatcherArgs(BaseModel):
    patching_mode: PatchingModeEnum = PatchingModeEnum.entropy
    patching_device: str = "cuda"
    entropy_model_checkpoint_dir: str | None = None
    realtime_patching: bool = False
    threshold: float = 1.335442066192627
    threshold_add: float | None = None
    max_patch_length: int | None = None
    patch_size: float = 4.5
    patching_batch_size: int = 1
    patcher_curr_align_steps: int | None = None
    patcher_curr_mixed_steps: int | None = None
    device: str = "cuda"
    monotonicity: bool = False
    log_time: bool = False

    def build(self) -> "Patcher":
        return Patcher(self)


def patching_availability(tokens, tokenizer, patch_speech_mode=True):
    """
    Set speech tokens to 1 for patching available
    Set text tokens and mode tokens to 0 for separate patching
    """
    # Return 1 for padding available,
    patching_avail = torch.full_like(tokens, 1, dtype=torch.int)
    if patch_speech_mode:
        # not patching 4 special tokens, 1 text mode token
        # aligned with the speech only static
        mask = tokens < tokenizer.text_vocab_size + tokenizer.offsetting_special_char - 1
    else:
        mask = tokens < tokenizer.text_vocab_size + tokenizer.offsetting_special_char
    patching_avail[mask] = 0
    return patching_avail


def tokens_to_hubert_tokens(tokens, tokenizer):
    """
    Calculate hubert only tokens for bpe model/mode checking
    """
    return tokens - tokenizer.offsetting_special_char - tokenizer.text_vocab_size


def static_patch_lengths(patching_avail, patch_size):
    """
    Static Patching for interleaved mode

    Args:
        patching_avail (torch.Tensor): 2D Tensor (batch_size, sequence_length)
        patch_size (int)

    Returns:
        torch.Tensor: patch_lengths (batch_size, num_patches)
    """
    batch_size, sequence_length = patching_avail.shape
    patch_lengths = []
    for i in range(batch_size):
        run = 0
        patch_lengths_i = []
        for v in patching_avail[i].tolist():
            if v == 1:
                run += 1
                if run == patch_size:
                    patch_lengths_i.append(patch_size)
                    run = 0
            else:
                if run > 0:
                    patch_lengths_i.append(run)
                    run = 0
                patch_lengths_i.append(1)
        if run > 0:
            patch_lengths_i.append(run)
        patch_lengths.append(torch.tensor(patch_lengths_i, dtype=torch.long))
    return torch.stack(patch_lengths, dim=0)



def aligned_patch_lengths(hubert_tokens, hu_length, sph_txt_boundary = 0):
    """
    Aligned Patching to align with word tokens

    Args:
        hubert_tokens (torch.Tensor): 2D Tensor (batch_size, sequence_length)
        hu_length (list of list of int)

    Returns:
        torch.Tensor: patch_lengths (batch_size, num_patches)
    """
    batch_size, sequence_length = hubert_tokens.shape
    patch_lengths = []
    for i in range(batch_size):
        hu_length_iter = iter(hu_length[i])
        hu_tokens = hubert_tokens[i].tolist()

        index = 0
        patch_lengths_i = []
        while index < len(hu_tokens):
            if hu_tokens[index] < sph_txt_boundary:
                patch_lengths_i.append(1)
                index += 1
            else:
                start = index
                while index < len(hu_tokens) and hu_tokens[index] >= sph_txt_boundary:
                    index += 1
                run_length = index - start
                segment = next(hu_length_iter)
                if sum(segment) != run_length:
                    raise ValueError(f"Align segment {segment} does not match run length {run_length}")
                patch_lengths_i.extend(segment)
        patch_lengths.append(torch.tensor(patch_lengths_i, dtype=torch.long))
    return torch.stack(patch_lengths, dim=0)


def entropy(scores):
    """
    scores: [bs, seq_len, vocab]
    returns [bs, seq_len]

    Computes the entropy for each token in the batch.
    Note: uses natural log.
    """
    log_probs = F.log_softmax(scores, dim=-1)
    probs = torch.exp(log_probs)
    p_log_p = log_probs * probs
    entropy = -p_log_p.sum(dim=-1)
    return entropy


def calculate_entropies(
    tokens: torch.tensor,
    entropy_model,
    patching_batch_size,
    device: str | None = None,
    enable_grad: bool = False,
):
    """
    tokens: 2D tensor of shape [batch_size, seq_len]
    Return 2D tensor of shape [batch_size, seq_len] with entropies for each token.

    Splits the tokens into chunks of size max_length and calculates entropies for each chunk.
    Entropy model can be executed on cpu or gpu, specify either 'cuda' or 'cpu' in the device argument.
    """

    grad_context = nullcontext() if enable_grad else torch.no_grad()

    with grad_context:
        entropies = []
        preds = []
        max_length = getattr(entropy_model, "max_length", 8192)
        batch_numel = max_length * patching_batch_size
        splits = torch.split(tokens.flatten(), batch_numel)
        for split in splits:
            pad_size = (max_length - (split.numel() % max_length)) % max_length
            pad = torch.zeros(
                pad_size, dtype=split.dtype, device=split.device, requires_grad=False
            )
            split = torch.cat((split, pad), dim=0)
            split = split.reshape(-1, max_length)
            if device is not None:
                split = split.to(device)
            # assert torch.all(split >= 0) and torch.all(split < 260)
            pred = entropy_model(split)
            pred = pred.reshape(-1, pred.shape[-1])[
                : split.numel() - pad_size, :
            ]  # [batch_size * seq_len, vocab]
            preds.append(pred)
            pred_entropies = entropy(pred)
            entropies.append(pred_entropies)

        concat_entropies = torch.cat(entropies, dim=0)
        concat_entropies = concat_entropies.reshape(tokens.shape)
        concat_preds = torch.cat(preds, dim=0)
        concat_preds = concat_preds.reshape(tokens.shape[0], -1)
    return concat_entropies, concat_preds


def patch_start_mask_from_entropy_with_monotonicity(entropies, t):
    """
    entropies: [bs, seq_len] torch tensor of entropies
    t: threshold
    returns [bs, seq_len] mask where True indicates the start of a patch
    """
    bs, seq_len = entropies.shape

    if seq_len == 0:
        return entropies > t

    mask = torch.zeros_like(entropies, dtype=torch.bool)
    mask[:, 0] = True

    # Calculate differences between consecutive elements along the sequence length
    differences = entropies[:, 1:] - entropies[:, :-1]

    # Calculate conditions for all elements except the first one in each sequence
    condition = differences > t

    # Update the mask based on the condition
    mask[:, 1:] = condition

    return mask


def patch_start_mask_global_and_monotonicity(entropies, t, t_add=0):
    """
    entropies: [bs, seq_len] torch tensor of entropies
    t: threshold
    returns [bs, seq_len] mask where True indicates the start of a patch
    """
    bs, seq_len = entropies.shape

    if seq_len == 0:
        return entropies > t

    mask = torch.zeros_like(entropies, dtype=torch.bool)
    mask[:, 0] = True

    # Calculate differences between consecutive elements along the sequence length
    differences = entropies[:, 1:] - entropies[:, :-1]

    # Calculate conditions for all elements except the first one in each sequence
    condition = (differences > t_add) & (entropies[:, 1:] > t) & (~mask[:, :-1])

    # Update the mask based on the condition
    mask[:, 1:] = condition

    return mask


def patch_start_ids_from_patch_start_mask(patch_start_mask):
    bs, trunc_seq_len = patch_start_mask.shape
    max_patches = patch_start_mask.sum(dim=1).max()
    if max_patches == 0:
        patch_start_ids = torch.full(
            (bs, trunc_seq_len),
            trunc_seq_len,
            dtype=torch.long,
            device=patch_start_mask.device,
        )
    else:
        patch_ids = (
            torch.arange(trunc_seq_len, device=patch_start_mask.device)
            .unsqueeze(0)
            .repeat(bs, 1)
        )
        extra_patch_ids = torch.full(
            (bs, trunc_seq_len),
            trunc_seq_len,
            dtype=torch.long,
            device=patch_start_mask.device,
        )
        all_patch_ids = torch.cat((patch_ids, extra_patch_ids), dim=1)
        patch_start_mask_padded = torch.cat(
            (patch_start_mask, ~patch_start_mask), dim=1
        )
        patch_start_ids = all_patch_ids[patch_start_mask_padded].reshape(
            bs, trunc_seq_len
        )[:, :max_patches]
    return patch_start_ids


def check_non_zero_after_zero(tensor):
    zero_mask = tensor == 0
    shifted_mask = torch.cat(
        [
            torch.zeros(tensor.shape[0], 1, dtype=torch.bool, device=tensor.device),
            zero_mask[:, :-1],
        ],
        dim=1,
    )
    non_zero_after_zero = (tensor != 0) & shifted_mask
    return non_zero_after_zero.any()


def patch_lengths_from_start_ids(patch_start_ids, seq_len):
    """
    Calculate patch lengths from start ids.
    start ids: ex: [0, 1, 7, 7, 7, 7, 7], it has the start ids of the patches (here 0, 1), and then
        the rest are filled to the seq len.
    seq_len: ex: 7 length of the sequence

    returns the patch lengths:
    [1, 6] for the above example.
    """
    last_ids = torch.full_like(patch_start_ids[:, :1], seq_len - 1)
    patch_end_ids = torch.cat((patch_start_ids[:, 1:] - 1, last_ids), dim=1)
    patch_lengths = patch_end_ids - patch_start_ids + 1
    assert torch.all(patch_lengths >= 0), f"{patch_lengths}"
    assert not check_non_zero_after_zero(patch_lengths), f"{patch_lengths}"
    return patch_lengths


def find_space_patch_start_ids(tokens):
    bs, seq_len = tokens.shape
    tokens_no_offset = tokens - OFFSET
    patch_end_mask = (
        (tokens_no_offset < ord("0"))
        | ((ord("9") < tokens_no_offset) & (tokens_no_offset < ord("A")))
        | ((ord("Z") < tokens_no_offset) & (tokens_no_offset < ord("a")))
        | ((ord("z") < tokens_no_offset) & (tokens_no_offset < 0b1000_0000))
        | (0b1100_0000 <= tokens_no_offset)
    )
    patch_end_mask[:, 1:] &= patch_end_mask[:, :-1].bitwise_not()
    patch_end_mask |= tokens < OFFSET

    patch_start_mask = torch.cat(
        [
            torch.tensor([1, 1], device=tokens.device, dtype=torch.bool)
            .unsqueeze(0)
            .repeat(bs, 1),
            patch_end_mask[:, 1:],
        ],
        dim=1,
    )
    max_patches = patch_start_mask.sum(dim=1).max()

    patch_ids = (
        torch.arange(seq_len + 1, device=tokens.device).unsqueeze(0).repeat(bs, 1)
    )
    extra_patch_ids = torch.full(
        (bs, seq_len + 1), seq_len + 1, dtype=torch.long, device=tokens.device
    )
    all_patch_ids = torch.cat((patch_ids, extra_patch_ids), dim=1)
    patch_start_mask_padded = torch.cat((patch_start_mask, ~patch_start_mask), dim=1)

    patch_start_ids = all_patch_ids[patch_start_mask_padded].reshape(bs, -1)[
        :, :max_patches
    ]
    return patch_start_ids


def to_device(entropy_model, device=None):
    if device == "cuda":
        rank = get_local_rank()
        device = f"cuda:{rank}"
    entropy_model = entropy_model.to(device)
    return entropy_model, device


def model_pred_to_bpe_patching_pred(pred):
    _, indices = torch.max(pred, dim=1)
    return indices == BPE_ID


def apply_bpe_patcher(tokens, bpe_patcher, patching_batch_size, device=None):
    assert tokens.device == torch.device(
        "cpu"
    ), f"{tokens.device} != cpu expects tokens to be on cpu"
    with torch.no_grad():
        bpe_patcher_device, device = to_device(
            bpe_patcher, device
        )  # Get entropy model to right rank device.
        bpe_patching_mask = []
        max_length = getattr(bpe_patcher, "max_length", 8192)
        batch_numel = max_length * patching_batch_size
        splits = torch.split(tokens.flatten(), batch_numel)
        for split in splits:
            pad_size = (max_length - (split.numel() % max_length)) % max_length
            pad = torch.zeros(
                pad_size, dtype=split.dtype, device=split.device, requires_grad=False
            )
            split = torch.cat((split, pad), dim=0)
            split = split.reshape(-1, max_length).to(device)
            assert torch.all(split >= 0) and torch.all(split < 260)
            pred = bpe_patcher_device(split)
            pred_cpu = pred[0].cpu()
            pred_cpu = pred_cpu.reshape(-1, pred_cpu.shape[-1])[
                : split.numel() - pad_size, :
            ]  # [batch_size * seq_len, vocab]
            bpe_patching_pred = model_pred_to_bpe_patching_pred(pred_cpu)
            bpe_patching_mask.append(bpe_patching_pred)
        bpe_patching_mask = torch.cat(bpe_patching_mask, dim=0)
        bpe_patching_mask = bpe_patching_mask.reshape(tokens.shape)
    return bpe_patching_mask


def find_bpe_patcher_patch_start_ids(
    tokens, bpe_patcher, patching_batch_size, device=None, include_next_token=True
):
    bs, seq_len = tokens.shape

    first_ids = (
        torch.tensor([0, 1], dtype=torch.long, device=tokens.device)
        .unsqueeze(0)
        .repeat(bs, 1)
    )
    preds_truncation_len = first_ids.shape[1]
    token_input = tokens[:, 1:] if include_next_token else tokens[:, 1:-1]
    if token_input.shape[1] >= 1:
        patch_start_mask = apply_bpe_patcher(
            token_input, bpe_patcher, patching_batch_size, device
        )
        assert (
            patch_start_mask.shape[1]
            == tokens.shape[1] + include_next_token - preds_truncation_len
        ), f"{patch_start_mask.shape[1]} != {tokens.shape[1] + include_next_token - preds_truncation_len}"
        patch_start_ids = patch_start_ids_from_patch_start_mask(patch_start_mask)
        patch_start_ids = torch.cat(
            (first_ids, patch_start_ids + preds_truncation_len), dim=1
        )
    else:
        patch_start_ids = first_ids
    return patch_start_ids


def find_entropy_patch_start_ids(
    entropies,
    patch_size=None,
    threshold=None,
    threshold_add=None,
    monotonicity=False,
    include_next_token=True,
):
    """
    Use entropies to find the start ids of each patch.
    Use patch_size or threshold to figure out the total number of patches to allocate.

    When threshold is not None the number of patches is not constant between
    different sequences, but patches can be identified incrementally rather than
    decided globally using the entire sequence.
    """
    bs, seq_len = entropies.shape[:2]

    first_ids = (
        torch.tensor([0, 1], dtype=torch.long, device=entropies.device)
        .unsqueeze(0)
        .repeat(bs, 1)
    )
    preds_truncation_len = first_ids.shape[
        1
    ]  # remove the first preds because they will be start of patches.
    entropies = entropies[:, 1:]
    if threshold is None:
        num_patches = seq_len // patch_size
        patch_start_ids = entropies.topk(num_patches - 2, dim=1).indices
        patch_start_ids = patch_start_ids.sort(dim=1).values
    else:
        # Assumes that there is at least one token going over the threshold
        if monotonicity:
            patch_start_mask = patch_start_mask_from_entropy_with_monotonicity(
                entropies, threshold
            )
        elif threshold_add is not None and threshold is not None:
            patch_start_mask = patch_start_mask_global_and_monotonicity(
                entropies, threshold, threshold_add
            )
        else:
            patch_start_mask = entropies > threshold
        if not include_next_token:
            patch_start_mask = patch_start_mask[:, :-1]
        # patch_start_mask[1:] |= tokens[:-1] < OFFSET
        patch_start_ids = patch_start_ids_from_patch_start_mask(patch_start_mask)

    patch_start_ids = torch.cat(
        (first_ids, patch_start_ids + preds_truncation_len), dim=1
    )
    return patch_start_ids


def rightpad(seq, pad_id, max_len):
    return seq + [pad_id] * (max_len - len(seq))


def find_bpe_delim_patch_start_ids(tokens, delim):
    ids = (tokens[:, :-1] == delim).nonzero(as_tuple=False)
    out = [[0, 1] for _ in range(tokens.shape[0])]
    for x, y in ids:
        # start is at delim + 1, delim should be the last element in the patch.
        out[x.item()].append(y.item() + 1)
    max_len = max([len(elt) for elt in out])
    out = [rightpad(elt, tokens.shape[1], max_len) for elt in out]
    patch_start_ids = torch.tensor(out, dtype=tokens.dtype, device=tokens.device)
    return patch_start_ids


def find_lookup_table_start_mask(
    tokens: torch.Tensor, lookup_table: torch.Tensor, include_next_token=True
):
    window_size = lookup_table.ndim
    # Unfold the tensor to get sliding windows
    unfolded = tokens.unfold(1, window_size, 1)
    # Gather indices for each dimension
    indices = [unfolded[..., i] for i in range(window_size)]
    # Access the lookup table using the gathered indices
    result = lookup_table[indices]
    return result


def find_lookup_table_patch_start_ids(
    tokens: torch.Tensor, lookup_table: torch.Tensor, include_next_token=True
):
    bs, seq_len = tokens.shape

    first_ids = (
        torch.tensor([0, 1], dtype=torch.long, device=tokens.device)
        .unsqueeze(0)
        .repeat(bs, 1)
    )
    preds_truncation_len = first_ids.shape[1]
    window_size = lookup_table.ndim
    assert window_size == 2, f"{window_size} != 2"
    # output dimensions: token_input shape - window_size + 1   --> we want first ids + this = tokens shape + 1 if next token otherwise just token shape
    token_input = (
        tokens if include_next_token else tokens[:, : -preds_truncation_len + 1]
    )
    if token_input.shape[1] >= window_size:
        patch_start_mask = find_lookup_table_start_mask(
            token_input, lookup_table, include_next_token
        )
        assert (
            patch_start_mask.shape[1]
            == tokens.shape[1] + include_next_token - preds_truncation_len
        ), f"{patch_start_mask.shape[1]} != {tokens.shape[1] + include_next_token - preds_truncation_len}"
        patch_start_ids = patch_start_ids_from_patch_start_mask(patch_start_mask)
        patch_start_ids = torch.cat(
            (first_ids, patch_start_ids + preds_truncation_len), dim=1
        )
    else:
        patch_start_ids = first_ids
    return patch_start_ids


def split_large_numbers(lst, m):
    new_lst = []
    for i in lst:
        if i > m:
            while i > m:
                new_lst.append(m)
                i -= m
            new_lst.append(i)
        else:
            new_lst.append(i)
    assert sum(new_lst) == sum(lst), f"{sum(new_lst)} != {sum(lst)}"
    return new_lst


class Patcher:
    def __init__(self, patcher_args: PatcherArgs):
        self.patcher_args = patcher_args
        self.patching_mode = patcher_args.patching_mode
        self.realtime_patching = patcher_args.realtime_patching
        if self.realtime_patching:
            assert (
                patcher_args.entropy_model_checkpoint_dir is not None
            ), "Cannot require realtime patching without an entropy model checkpoint"
            entropy_model = load_entropy_model(
                patcher_args.entropy_model_checkpoint_dir,
                os.path.join(
                    patcher_args.entropy_model_checkpoint_dir,
                    "consolidated/consolidated.pth",
                ),
            )
            entropy_model, _ = to_device(entropy_model, patcher_args.patching_device)
            self.entropy_model = entropy_model
        else:
            self.entropy_model = None
        self.threshold = patcher_args.threshold
        self.threshold_add = patcher_args.threshold_add
        self.max_patch_length = patcher_args.max_patch_length
        self.patch_size = patcher_args.patch_size
        self.patching_batch_size = patcher_args.patching_batch_size
        self.patcher_curr_align_steps = patcher_args.patcher_curr_align_steps
        self.patcher_curr_mixed_steps = patcher_args.patcher_curr_mixed_steps
        self.device = patcher_args.device
        self.monotonicity = patcher_args.monotonicity
        self.log_time = patcher_args.log_time
        if self.log_time:
            self.log = defaultdict(float)

    def get_curr_type(self, curr_step):
        if curr_step is None or  curr_step > self.patcher_curr_mixed_steps:
            curr_type = 'static'
        elif curr_step > self.patcher_curr_align_steps and curr_step <= self.patcher_curr_mixed_steps:
            prob_align = 1.0 - (curr_step - self.patcher_curr_align_steps) / (self.patcher_curr_mixed_steps - self.patcher_curr_align_steps)
            if random.random() < prob_align:
                curr_type = 'align'
            else:
                curr_type = 'static'
        elif curr_step <= self.patcher_curr_align_steps:
            curr_type = 'align'
        return curr_type

    def patch(
        self,
        tokens: torch.Tensor,
        include_next_token: bool = False,
        preds: torch.Tensor | None = None,
        entropies: torch.Tensor | None = None,
        threshold: float = None,
        hu_length: list[int] = None,
        curr_step: int = None,
        tokenizer: LstTokenizer | None = None,
    ) -> torch.Tensor:
        """
        tokens: 2D tensor of shape [batch_size, seq_len] that needs to be patched
        Returns patch lengths and optionally scores associated with the tokens (i.e. entropies, logprobs etc.)
        -> output tensor: [batch_size, max_num_patches]
            each tensor is processed independently and gets right padded with zeros.

        Patching with the following modes:
        1. patching_mode = None: static patch size
        2. patching_mode = "entropy":
            calculate entropy of each token, allocate patches so that the total
            number of patches is the same as static patching but choose to begin
            patches on tokens where the model is most uncertain (highest entropy).

            When threshold is provided, it uses the threshold to decide when to
            start a new patch.
        3. patching_mode = "space":
            use space like tokens to define the patches.
        4. patching_mode = "bpe":
            use bpe delim tokens to define the patches.

        To correctly patch the last token, it may be necessary to include the next token in the patch
        lengths calculations. This is controlled by the include_next_token argument.
        """
        bs, seq_len = tokens.shape
        seq_len_next_tok = seq_len + 1 if include_next_token else seq_len
        scores = None
        # STATIC
        if self.patching_mode == PatchingModeEnum.static:
            patch_lengths = torch.zeros(
                (bs, math.ceil(seq_len_next_tok / self.patch_size)),
                dtype=tokens.dtype,
                device=tokens.device,
            ).fill_(self.patch_size)
            if seq_len_next_tok % self.patch_size != 0:
                patch_lengths[:, -1] = seq_len_next_tok % self.patch_size
        elif self.patching_mode == PatchingModeEnum.byte:
            patch_lengths = torch.ones(
                (bs, seq_len_next_tok), dtype=tokens.dtype, device=tokens.device
            )
        # ENTROPY
        elif self.patching_mode == PatchingModeEnum.entropy:
            if self.log_time:
                s = time.time()
            if entropies is not None:
                scores = entropies.to(dtype=torch.float32)
            elif preds is not None:
                scores = entropy(preds)
            else:
                start_entropies = time.time()
                scores, _ = calculate_entropies(
                    tokens,
                    self.entropy_model,
                    self.patching_batch_size,
                    self.device,
                )
            if self.log_time:
                self.log["calculate_entropies"] += time.time() - s
                s = time.time()
            patch_start_ids = find_entropy_patch_start_ids(
                scores,
                self.patch_size,
                include_next_token=include_next_token,
                threshold=threshold if threshold is not None else self.threshold,
                threshold_add=self.threshold_add,
                monotonicity=self.monotonicity,
            )
            if self.log_time:
                self.log["find_entropy_patch_start_ids"] += time.time() - s
                s = time.time()
            patch_lengths = patch_lengths_from_start_ids(
                patch_start_ids, seq_len_next_tok
            )
            if self.log_time:
                self.log["patch_lengths_from_start_ids"] += time.time() - s
                s = time.time()
        # BPE
        elif self.patching_mode == PatchingModeEnum.bpe:
            patch_start_ids = find_bpe_delim_patch_start_ids(tokens, delim=BPE_ID)
            patch_lengths = patch_lengths_from_start_ids(
                patch_start_ids, seq_len_next_tok
            )
        elif self.patching_mode == PatchingModeEnum.bpe_patcher:
            patch_start_ids = find_bpe_patcher_patch_start_ids(
                tokens,
                self.entropy_model,
                self.patching_batch_size,
                self.device,
                include_next_token,
            )
            patch_lengths = patch_lengths_from_start_ids(
                patch_start_ids, seq_len_next_tok
            )
        # SPACE
        elif self.patching_mode == PatchingModeEnum.space:
            patch_start_ids = find_space_patch_start_ids(tokens)
            patch_lengths = patch_lengths_from_start_ids(
                patch_start_ids, seq_len_next_tok
            )

        # LST Static
        elif self.patching_mode == PatchingModeEnum.lst_static:
            assert(int(self.patch_size) == self.patch_size)
            patching_available_tokens = patching_availability(tokens, tokenizer)

            if torch.all(patching_available_tokens[:,1:-1] == 1):
                # speech only
                start_patch = []
                end_patch = []
                patching_length = seq_len
                if torch.all(patching_available_tokens[:,0] == 0):
                    start_patch = [1]
                    patching_length -= 1
                if torch.all(patching_available_tokens[:,-1] == 0):
                    end_patch = [1]
                    patching_length -= 1

                remaining = (patching_length) % self.patch_size
                patch_lengths = start_patch + [int(self.patch_size)] * int((patching_length) // self.patch_size)  + ([remaining] if remaining > 0 else []) + end_patch
                patch_lengths = torch.tensor(patch_lengths, dtype=tokens.dtype, device=tokens.device).unsqueeze(0).repeat(bs, 1)
            elif torch.sum(patching_available_tokens) == 0:
                # Text
                patch_lengths = torch.ones_like(tokens)
            else:
                # Interleaved
                patch_lengths = static_patch_lengths(
                    patching_available_tokens,
                    self.patch_size,
                )

        elif self.patching_mode == PatchingModeEnum.lst_align:
            hubert_tokens = tokens_to_hubert_tokens(tokens, tokenizer)

            if torch.max(hubert_tokens) < 0:
                # Text, all not patching
                patch_lengths = torch.ones_like(tokens)
            else:
                assert(hu_length is not None), "speech only should not be in interleaved_aligned_patcher mode"
                # Interleaved
                patch_lengths = aligned_patch_lengths(
                    hubert_tokens,
                    hu_length,
                )


        elif self.patching_mode == PatchingModeEnum.lst_curr:
            hubert_tokens = tokens_to_hubert_tokens(tokens, tokenizer)
            if torch.max(hubert_tokens) < 0:
                # Text, all not patching
                patch_lengths = torch.ones_like(tokens)
            elif hu_length is not None and hu_length[0] is not None:
                curr_type = self.get_curr_type(curr_step)
                if curr_type == 'align':
                    patch_lengths = aligned_patch_lengths(
                        hubert_tokens,
                        hu_length,
                    )
                elif curr_type == 'static':
                    patching_available_tokens = patching_availability(tokens, tokenizer, patch_speech_mode=self.patch_speech_mode)
                    patch_lengths = static_patch_lengths(
                        patching_available_tokens,
                        self.patch_size,
                    )



        else:
            raise NotImplementedError(f"self.patching_mode {self.patching_mode}")

        # Apply any processing to patch lengths
        if self.max_patch_length is not None:
            # TODO: avoid going back to a list here.
            patch_lengths = [
                split_large_numbers(pl, self.max_patch_length)
                for pl in patch_lengths.tolist()
            ]
            max_len = max([len(pl) for pl in patch_lengths])
            patch_lengths = [rightpad(pl, 0, max_len=max_len) for pl in patch_lengths]
            patch_lengths = torch.tensor(
                patch_lengths, dtype=tokens.dtype, device=tokens.device
            )
        assert not check_non_zero_after_zero(patch_lengths)
        # Find the last non-zero column index using argmax on a reversed version of the tensor
        last_non_zero_col_reversed = (
            (patch_lengths != 0).flip(dims=[1]).int().argmax(dim=1).min()
        )
        # Slice the tensor up to the last non-zero column
        patch_lengths = patch_lengths[
            :, : patch_lengths.shape[1] - last_non_zero_col_reversed
        ]
        assert (
            torch.sum(patch_lengths)
            == tokens.numel() + include_next_token * tokens.shape[0]
        ), f"{torch.sum(patch_lengths)} != {tokens.numel() + include_next_token * tokens.shape[0]}"
        if self.log_time:
            self.log["postprocessing_patch_lengths"] += time.time() - s
            self.log["tokens"] += patch_lengths.sum().item()
        return patch_lengths, scores
