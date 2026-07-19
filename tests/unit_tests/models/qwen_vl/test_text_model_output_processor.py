# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.

from unittest.mock import Mock

import torch

from megatron.bridge.models.qwen_vl.modelling_qwen3_vl.text_model import Qwen3VLGPTModel


class _FakeQwenTextModel:
    def __init__(self) -> None:
        self.mtp_process = False
        self.config = Mock(sequence_parallel=False)
        self._preprocess = Mock(
            return_value=(
                torch.zeros(2, 1, 4),
                None,
                None,
                None,
                None,
                None,
            )
        )
        self.decoder = Mock(return_value=torch.ones(2, 1, 4))
        self._postprocess = Mock(return_value=torch.tensor(3.0))


def test_qwen_text_model_threads_output_processor_to_postprocess() -> None:
    model = _FakeQwenTextModel()
    output_processor = Mock()
    context = {"vocab_chunk_size": 17}

    output = Qwen3VLGPTModel.forward(
        model,
        input_ids=torch.ones(1, 2, dtype=torch.long),
        position_ids=torch.arange(2).view(1, 2),
        attention_mask=None,
        labels=torch.ones(1, 2, dtype=torch.long),
        output_processor=output_processor,
        output_processor_context=context,
    )

    assert output.item() == 3.0
    postprocess_kwargs = model._postprocess.call_args.kwargs
    assert postprocess_kwargs["output_processor"] is output_processor
    assert postprocess_kwargs["output_processor_context"] is context
