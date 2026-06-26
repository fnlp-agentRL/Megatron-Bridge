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

import torch

from megatron.bridge.data.datasets.packed_sequence import _pre_pad_data_point
from megatron.bridge.data.datasets.packing_utils import fill_packing_strategy


PAD_ID = 0


def test_pre_pad_data_point_chat_tensors_do_not_raise():
    """Chat path returns torch tensors; padding must not raise TypeError (see issue #2610)."""
    data = {
        "input_ids": torch.LongTensor([5, 6, 7]),
        "loss_mask": torch.BoolTensor([False, True, True]),
        "context_ids": torch.LongTensor([5, 6]),
        "padding_mask": torch.BoolTensor([False, False, False]),
    }
    # max_length_to_pad=8 -> input_ids padded to 8 - 3 + 1 = 6 extra -> length 9
    _pre_pad_data_point(data, max_seq_length=16, max_length_to_pad=8, pad_id=PAD_ID)

    assert isinstance(data["input_ids"], list)
    assert isinstance(data["loss_mask"], list)
    assert isinstance(data["padding_mask"], list)
    # loss_mask must end up the same length as input_ids, otherwise fill_packing_strategy's
    # np.array([...loss_mask...]) raises an inhomogeneous-shape error when samples are grouped.
    assert len(data["loss_mask"]) == len(data["input_ids"])
    # padded loss_mask positions carry False (no loss on pad tokens)
    assert data["loss_mask"][3:] == [False] * (len(data["loss_mask"]) - 3)
    assert data["padding_mask"][3:] == [True] * (len(data["padding_mask"]) - 3)
    assert data["input_ids"][3:] == [PAD_ID] * (len(data["input_ids"]) - 3)


def test_pre_pad_data_point_equalizes_loss_mask_lengths():
    """Two samples that round to the same padded input length must get equal-length loss_masks."""
    a = {"input_ids": torch.LongTensor([1, 2, 3]), "loss_mask": torch.BoolTensor([False, True, True])}
    b = {
        "input_ids": torch.LongTensor([1, 2, 3, 4, 5]),
        "loss_mask": torch.BoolTensor([False, False, True, True, True]),
    }
    # both round up to the same multiple-of-8 target
    _pre_pad_data_point(a, max_seq_length=16, max_length_to_pad=8, pad_id=PAD_ID)
    _pre_pad_data_point(b, max_seq_length=16, max_length_to_pad=8, pad_id=PAD_ID)

    assert len(a["input_ids"]) == len(b["input_ids"])
    assert len(a["loss_mask"]) == len(b["loss_mask"]) == len(a["input_ids"])


def test_pre_pad_data_point_non_chat_lists_still_work():
    """Non-chat (GPTSFTDataset) path returns plain lists without loss_mask; must still get padding_mask."""
    data = {"input_ids": [9, 9, 9], "context_ids": [9, 9]}
    _pre_pad_data_point(data, max_seq_length=16, max_length_to_pad=8, pad_id=PAD_ID)

    assert data["input_ids"] == [9, 9, 9] + [PAD_ID] * 6
    assert "loss_mask" not in data
    assert data["padding_mask"] == [False, False, False] + [True] * 6


def test_pre_pad_data_point_truncates_overlong():
    """Sequences longer than max_seq_length are truncated."""
    data = {"input_ids": list(range(20)), "loss_mask": [True] * 20}
    _pre_pad_data_point(data, max_seq_length=16, max_length_to_pad=8, pad_id=PAD_ID)

    assert len(data["input_ids"]) == 16
    assert len(data["loss_mask"]) == 16
    assert len(data["padding_mask"]) == 16


def test_fill_packing_strategy_preserves_padding_mask():
    sequences = {idx: [] for idx in range(5)}
    sequences[4] = [
        {
            "input_ids": [10, 11, PAD_ID, PAD_ID, PAD_ID],
            "loss_mask": [False, True, False, False, False],
            "padding_mask": [False, False, True, True, True],
        }
    ]

    output_data = fill_packing_strategy([[4]], sequences, pack_size=4, pad_id=PAD_ID)

    assert output_data == [
        {
            "input_ids": [10, 11, PAD_ID, PAD_ID, PAD_ID],
            "loss_mask": [True, False, False, False, False],
            "padding_mask": [False, False, True, True, True],
            "seq_start_id": [0],
        }
    ]


def test_fill_packing_strategy_defaults_missing_padding_mask_to_zeros():
    sequences = {idx: [] for idx in range(3)}
    sequences[2] = [{"input_ids": [10, 11, 12], "loss_mask": [False, True, True]}]

    output_data = fill_packing_strategy([[2]], sequences, pack_size=2, pad_id=PAD_ID)

    assert output_data[0]["padding_mask"] == [False, False, False]
