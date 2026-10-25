# Copyright 2026 Google LLC
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

"""explorer.detector_point_order: the object detector asks for, and reads, the model's native point order.

Gemini points [y, x] (the default, unchanged). Qwen-VL-family models point [x, y] and keep doing so when the
prompt asks for [y, x], so with the fixed swap their points land transposed.
"""

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from artemis.agents.object_detector import object_detector
from artemis.agents.object_detector.object_detector import _run_object_detection
from artemis.config.agent import ExplorerConfig

TEMPLATE = "Point to {labels_str}. Each object has a 'point' array [y, x] (normalized to 0-1000)."


def test_default_point_order_is_gemini_yx():
    assert ExplorerConfig().detector_point_order == "yx"
    assert ExplorerConfig(detector_point_order="xy").detector_point_order == "xy"


def test_point_order_is_read_from_agent_config():
    cfg = SimpleNamespace(explorer=SimpleNamespace(detector_point_order="xy"))
    with patch("artemis.config.load_agent_config", return_value=cfg):
        assert object_detector._detector_point_order() == "xy"
    with patch("artemis.config.load_agent_config", side_effect=RuntimeError("no config")):
        assert object_detector._detector_point_order() == "yx"


async def _detect(order: str, raw_point: list[int]):
    captured = {}

    async def fake_detect(llm, ctx, image, mime, label, templates, timeout_val=None):
        captured["templates"] = templates
        return [{"label": label, "point": list(raw_point)}]

    with (
        patch.object(object_detector, "_detector_point_order", return_value=order),
        patch.object(
            object_detector, "_detect_single_label", new=AsyncMock(side_effect=fake_detect)
        ),
        patch.object(object_detector, "get_llm"),
    ):
        out = await _run_object_detection(
            ctx=MagicMock(), image_bytes=b"img", queries=["apply button"], templates=[TEMPLATE]
        )
    return out["detected"][0]["point"], captured["templates"][0]


@pytest.mark.asyncio
async def test_yx_order_asks_for_yx_and_swaps_to_xy_as_before():
    point, prompt = await _detect("yx", [800, 200])
    assert point == [200, 800]
    assert "[y, x]" in prompt


@pytest.mark.asyncio
async def test_xy_order_asks_for_xy_and_keeps_the_point():
    point, prompt = await _detect("xy", [850, 840])
    assert point == [850, 840]
    assert "[x, y]" in prompt and "[y, x]" not in prompt
