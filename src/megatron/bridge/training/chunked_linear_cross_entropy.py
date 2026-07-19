# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Memory-efficient fused LM-head and vocab-parallel cross entropy."""

from __future__ import annotations

from typing import Any

import torch
from megatron.core.tensor_parallel.mappings import (
    copy_to_tensor_model_parallel_region,
    gather_from_sequence_parallel_region,
)
from megatron.core.tensor_parallel.utils import VocabUtility


def _group_size(group: torch.distributed.ProcessGroup) -> int:
    return group.size()


def _group_rank(group: torch.distributed.ProcessGroup) -> int:
    return group.rank()


def _all_reduce(
    tensor: torch.Tensor,
    *,
    op: torch.distributed.ReduceOp.RedOpType,
    group: torch.distributed.ProcessGroup,
) -> None:
    if _group_size(group) > 1:
        torch.distributed.all_reduce(tensor, op=op, group=group)


def _accumulate_weight_gradient(
    hidden_states: torch.Tensor,
    grad_logits: torch.Tensor,
    main_grad: torch.Tensor,
) -> None:
    """Accumulate a BF16 GEMM directly into an FP32 main-gradient slice."""
    import fused_weight_gradient_mlp_cuda

    if main_grad.dtype == torch.float32:
        fused_weight_gradient_mlp_cuda.wgrad_gemm_accum_fp32(hidden_states, grad_logits, main_grad)
    elif main_grad.dtype in (torch.float16, torch.bfloat16):
        fused_weight_gradient_mlp_cuda.wgrad_gemm_accum_fp16(hidden_states, grad_logits, main_grad)
    else:
        raise RuntimeError(f"Unsupported LM-head main-gradient dtype: {main_grad.dtype}")


def _matmul_with_fp32_output(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    """Run a low-precision GEMM while retaining its FP32 accumulator output."""
    if left.is_cuda and left.dtype in (torch.float16, torch.bfloat16):
        return torch.mm(left, right, out_dtype=torch.float32)
    return torch.matmul(left, right).float()


def _dummy_weight_gradient(weight: torch.Tensor, *, zero: bool) -> torch.Tensor:
    """Return the reusable dummy gradient used by Megatron DDP hooks."""
    # The output weight is roughly 1 GiB, so allocating a full dummy gradient
    # solely to wake DDP's AccumulateGrad hook would erase much of this
    # optimization. A one-element view is sufficient for the hook because
    # Megatron DDP ignores its value when grad_added_to_main_grad is set.
    dummy = weight.new_zeros(1) if zero else weight.new_empty(1)
    return dummy.expand_as(weight)


class _ChunkedLinearCrossEntropy(torch.autograd.Function):
    """Autograd implementation that never materializes a complete token-vocab tensor."""

    @staticmethod
    def forward(
        ctx: Any,
        hidden_states: torch.Tensor,
        weight: torch.Tensor,
        labels: torch.Tensor,
        vocab_chunk_size: int,
        vocab_start_index: int,
        tp_group: torch.distributed.ProcessGroup,
        logit_scale: float,
        gradient_accumulation_fusion: bool,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if hidden_states.ndim != 3:
            raise ValueError(f"hidden_states must have shape [S, B, D], got {hidden_states.shape}")
        if weight.ndim != 2 or weight.shape[1] != hidden_states.shape[-1]:
            raise ValueError(f"weight must have shape [V_local, D={hidden_states.shape[-1]}], got {weight.shape}")
        if labels.shape != (hidden_states.shape[1], hidden_states.shape[0]):
            raise ValueError(
                "labels must have shape [B, S] matching hidden_states; "
                f"got labels={labels.shape}, hidden_states={hidden_states.shape}"
            )
        if vocab_chunk_size <= 0:
            raise ValueError(f"vocab_chunk_size must be positive, got {vocab_chunk_size}")
        if not hidden_states.dtype.is_floating_point or hidden_states.dtype != weight.dtype:
            raise ValueError(
                f"hidden_states and weight must have the same floating dtype, got "
                f"{hidden_states.dtype} and {weight.dtype}"
            )

        sequence_length, micro_batch_size, hidden_size = hidden_states.shape
        hidden_2d = hidden_states.reshape(-1, hidden_size)
        targets = labels.transpose(0, 1).contiguous().reshape(-1)
        token_count = hidden_2d.shape[0]
        local_vocab_size = weight.shape[0]
        running_max = torch.full((token_count,), -torch.inf, dtype=torch.float32, device=hidden_states.device)
        running_sum = torch.zeros_like(running_max)
        target_logits = torch.zeros_like(running_max)
        target_found = torch.zeros(token_count, dtype=torch.int32, device=hidden_states.device)
        local_predictions = torch.zeros(token_count, dtype=torch.long, device=hidden_states.device)

        for chunk_start in range(0, local_vocab_size, vocab_chunk_size):
            chunk_end = min(chunk_start + vocab_chunk_size, local_vocab_size)
            logits = torch.matmul(hidden_2d, weight[chunk_start:chunk_end].t())
            if logit_scale != 1.0:
                logits = logits * logit_scale
            logits_fp32 = logits.float()
            chunk_max, chunk_predictions = logits_fp32.max(dim=-1)
            replace_prediction = chunk_max > running_max
            local_predictions = torch.where(
                replace_prediction,
                chunk_predictions + vocab_start_index + chunk_start,
                local_predictions,
            )

            global_chunk_start = vocab_start_index + chunk_start
            global_chunk_end = vocab_start_index + chunk_end
            target_in_chunk = (targets >= global_chunk_start) & (targets < global_chunk_end)
            local_targets = (targets - global_chunk_start).clamp(0, chunk_end - chunk_start - 1)
            selected_logits = logits_fp32.gather(dim=1, index=local_targets.unsqueeze(-1)).squeeze(-1)
            target_logits = torch.where(target_in_chunk, selected_logits, target_logits)
            target_found = torch.where(target_in_chunk, 1, target_found)

            merged_max = torch.maximum(running_max, chunk_max)
            running_sum = running_sum * torch.exp(running_max - merged_max)
            logits_fp32.sub_(merged_max.unsqueeze(-1)).exp_()
            running_sum.add_(logits_fp32.sum(dim=-1))
            running_max = merged_max

            del chunk_max, chunk_predictions, logits, logits_fp32, merged_max, selected_logits

        global_max = running_max.clone()
        _all_reduce(global_max, op=torch.distributed.ReduceOp.MAX, group=tp_group)
        prediction_sentinel = torch.iinfo(local_predictions.dtype).max
        predictions = torch.where(running_max == global_max, local_predictions, prediction_sentinel)
        _all_reduce(predictions, op=torch.distributed.ReduceOp.MIN, group=tp_group)
        running_sum.mul_(torch.exp(running_max - global_max))
        _all_reduce(running_sum, op=torch.distributed.ReduceOp.SUM, group=tp_group)
        _all_reduce(target_logits, op=torch.distributed.ReduceOp.SUM, group=tp_group)
        _all_reduce(target_found, op=torch.distributed.ReduceOp.MAX, group=tp_group)
        logsumexp = global_max + torch.log(running_sum)
        # Native vocab-parallel CE reports log(sum(exp(logits - max))) for an
        # out-of-range target (for example label=-100 under a zero loss mask).
        target_logits = torch.where(target_found.bool(), target_logits, global_max)
        loss = logsumexp - target_logits

        ctx.save_for_backward(hidden_states, weight, targets, logsumexp)
        ctx.vocab_chunk_size = vocab_chunk_size
        ctx.vocab_start_index = vocab_start_index
        ctx.tp_group = tp_group
        ctx.logit_scale = logit_scale
        ctx.gradient_accumulation_fusion = gradient_accumulation_fusion
        ctx.hidden_shape = hidden_states.shape
        predictions = predictions.view(sequence_length, micro_batch_size).transpose(0, 1).contiguous()
        ctx.mark_non_differentiable(predictions)
        return loss.view(sequence_length, micro_batch_size).transpose(0, 1).contiguous(), predictions

    @staticmethod
    def backward(
        ctx: Any,
        grad_output: torch.Tensor,
        grad_predictions: torch.Tensor | None,
    ) -> tuple[torch.Tensor | None, ...]:
        del grad_predictions
        hidden_states, weight, targets, logsumexp = ctx.saved_tensors
        sequence_length, micro_batch_size, hidden_size = ctx.hidden_shape
        hidden_2d = hidden_states.reshape(-1, hidden_size)
        grad_loss = grad_output.transpose(0, 1).contiguous().reshape(-1).float()
        grad_hidden_accum = torch.zeros_like(hidden_2d, dtype=torch.float32) if ctx.needs_input_grad[0] else None
        grad_weight = None
        if ctx.needs_input_grad[1] and not ctx.gradient_accumulation_fusion:
            grad_weight = torch.empty_like(weight)
        if ctx.needs_input_grad[1] and ctx.gradient_accumulation_fusion:
            if hasattr(weight, "__fsdp_param__"):
                raise RuntimeError("Chunked LM-head loss does not support Megatron FSDP")
            if not hasattr(weight, "main_grad"):
                raise RuntimeError("gradient_accumulation_fusion requires weight.main_grad")

        local_vocab_size = weight.shape[0]
        for chunk_start in range(0, local_vocab_size, ctx.vocab_chunk_size):
            chunk_end = min(chunk_start + ctx.vocab_chunk_size, local_vocab_size)
            weight_chunk = weight[chunk_start:chunk_end]
            logits = torch.matmul(hidden_2d, weight_chunk.t())
            if ctx.logit_scale != 1.0:
                logits = logits * ctx.logit_scale
            grad_logits_fp32 = logits.float()
            grad_logits_fp32.sub_(logsumexp.unsqueeze(-1)).exp_()

            global_chunk_start = ctx.vocab_start_index + chunk_start
            global_chunk_end = ctx.vocab_start_index + chunk_end
            target_in_chunk = (targets >= global_chunk_start) & (targets < global_chunk_end)
            local_targets = (targets - global_chunk_start).clamp(0, chunk_end - chunk_start - 1)
            row_indices = torch.arange(targets.numel(), device=targets.device)
            grad_logits_fp32[row_indices, local_targets] -= target_in_chunk.float()
            grad_logits_fp32.mul_(grad_loss.unsqueeze(-1))
            if ctx.logit_scale != 1.0:
                grad_logits_fp32.mul_(ctx.logit_scale)
            grad_logits = grad_logits_fp32.to(hidden_states.dtype)

            if grad_hidden_accum is not None:
                # A full LM-head backward accumulates across the whole vocab in
                # FP32 inside GEMM and casts dH once. Preserve that behavior by
                # retaining each chunk GEMM's FP32 output until all chunks merge.
                grad_hidden_accum.add_(_matmul_with_fp32_output(grad_logits, weight_chunk))
            if ctx.needs_input_grad[1]:
                if ctx.gradient_accumulation_fusion:
                    _accumulate_weight_gradient(
                        hidden_2d.contiguous(),
                        grad_logits.contiguous(),
                        weight.main_grad[chunk_start:chunk_end],
                    )
                else:
                    grad_weight[chunk_start:chunk_end] = torch.matmul(grad_logits.t(), hidden_2d)
            del grad_logits, grad_logits_fp32, logits, weight_chunk

        grad_hidden = None
        if grad_hidden_accum is not None:
            grad_hidden = grad_hidden_accum.to(hidden_states.dtype).view(
                sequence_length, micro_batch_size, hidden_size
            )

        if ctx.needs_input_grad[1] and ctx.gradient_accumulation_fusion:
            if hasattr(weight, "grad_added_to_main_grad"):
                weight.grad_added_to_main_grad = True
                grad_weight = _dummy_weight_gradient(weight, zero=bool(getattr(weight, "zero_out_wgrad", False)))
            else:
                grad_weight = None

        return grad_hidden, grad_weight, None, None, None, None, None, None


def chunked_linear_cross_entropy(
    hidden_states: torch.Tensor,
    weight: torch.Tensor,
    labels: torch.Tensor,
    *,
    vocab_chunk_size: int,
    vocab_start_index: int,
    tp_group: torch.distributed.ProcessGroup,
    logit_scale: float,
    gradient_accumulation_fusion: bool,
) -> torch.Tensor:
    """Compute per-token LM loss without materializing complete logits.

    Args:
        hidden_states: Decoder output with shape ``[S, B, D]``.
        weight: Local vocab shard of the output weight, shape ``[V_local, D]``.
        labels: Global token ids with shape ``[B, S]``.
        vocab_chunk_size: Maximum local-vocab width materialized at once.
        vocab_start_index: Global token-id offset of this TP rank's shard.
        tp_group: Tensor-parallel process group.
        logit_scale: MuP output multiplier applied before softmax.
        gradient_accumulation_fusion: Accumulate wgrad directly into ``main_grad``.

    Returns:
        Unmasked, unreduced per-token loss with shape ``[B, S]``.
    """
    loss, _ = _ChunkedLinearCrossEntropy.apply(
        hidden_states,
        weight,
        labels,
        vocab_chunk_size,
        vocab_start_index,
        tp_group,
        logit_scale,
        gradient_accumulation_fusion,
    )
    return loss


def chunked_linear_cross_entropy_with_predictions(
    hidden_states: torch.Tensor,
    weight: torch.Tensor,
    labels: torch.Tensor,
    *,
    vocab_chunk_size: int,
    vocab_start_index: int,
    tp_group: torch.distributed.ProcessGroup,
    logit_scale: float,
    gradient_accumulation_fusion: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute chunked per-token loss and global vocabulary argmax ids."""
    return _ChunkedLinearCrossEntropy.apply(
        hidden_states,
        weight,
        labels,
        vocab_chunk_size,
        vocab_start_index,
        tp_group,
        logit_scale,
        gradient_accumulation_fusion,
    )


def chunked_lm_head_output_processor(
    *,
    vocab_chunk_size: int,
    mtp_vocab_chunk_size: int | None = None,
    is_mtp: bool = False,
    **kwargs: Any,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """GPT output processor for the chunked training loss path."""
    hidden_states = kwargs["hidden_states"]
    output_layer = kwargs["output_layer"]
    output_weight = kwargs["output_weight"]
    labels = kwargs["labels"]
    runtime_gather_output = kwargs["runtime_gather_output"]
    config = kwargs["config"]

    if labels is None:
        raise ValueError("Chunked LM-head loss requires labels")
    if runtime_gather_output not in (None, False) or output_layer.gather_output:
        raise ValueError("Chunked LM-head loss requires sharded, non-gathered logits")
    if output_layer.bias is not None:
        raise ValueError("Chunked LM-head loss does not yet support output-layer bias")
    if config.defer_embedding_wgrad_compute:
        raise ValueError("Chunked LM-head loss does not support deferred embedding wgrad")
    if not config.gradient_accumulation_fusion:
        raise ValueError("The training integration requires gradient_accumulation_fusion")

    weight = output_weight if output_weight is not None else output_layer.weight
    if weight is None:
        raise ValueError("Output weight is unavailable")
    if output_layer.sequence_parallel:
        projection_input = gather_from_sequence_parallel_region(
            hidden_states,
            tensor_parallel_output_grad=True,
            group=output_layer.tp_group,
        )
    else:
        projection_input = copy_to_tensor_model_parallel_region(hidden_states, group=output_layer.tp_group)

    local_vocab_size = weight.shape[0]
    vocab_start_index, _ = VocabUtility.vocab_range_from_per_partition_vocab_size(
        local_vocab_size,
        _group_rank(output_layer.tp_group),
        _group_size(output_layer.tp_group),
    )
    logit_scale = float(config.mup_output_mult) if config.use_mup else 1.0
    selected_chunk_size = mtp_vocab_chunk_size if is_mtp and mtp_vocab_chunk_size is not None else vocab_chunk_size
    loss_kwargs = {
        "vocab_chunk_size": selected_chunk_size,
        "vocab_start_index": vocab_start_index,
        "tp_group": output_layer.tp_group,
        "logit_scale": logit_scale,
        "gradient_accumulation_fusion": config.gradient_accumulation_fusion,
    }
    if not is_mtp:
        return chunked_linear_cross_entropy(projection_input, weight, labels, **loss_kwargs)

    loss_mask = kwargs["loss_mask"]
    loss, predictions = chunked_linear_cross_entropy_with_predictions(
        projection_input,
        weight,
        labels,
        **loss_kwargs,
    )
    valid_positions = loss_mask.bool()
    correct = ((predictions == labels) & valid_positions).sum().float()
    total = valid_positions.sum().float()
    return loss, correct, total
