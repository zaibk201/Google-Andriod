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

"""Gemini connection pre-warming only runs when the configured operator is a Gemini model."""

from types import SimpleNamespace
from unittest.mock import patch

from artemis.sdk import agent as sdk_agent


def _cfg(provider: str) -> SimpleNamespace:
    return SimpleNamespace(get_agent=lambda name: SimpleNamespace(provider=provider))


def test_non_google_operator_skips_prewarm():
    for provider in ("custom", "openai", "anthropic", "ollama"):
        with patch("artemis.config.llm.get_default_llm_config", return_value=_cfg(provider)):
            assert sdk_agent._operator_uses_google() is False, provider


def test_google_operator_keeps_prewarm():
    for provider in ("google", "gemini"):
        with patch("artemis.config.llm.get_default_llm_config", return_value=_cfg(provider)):
            assert sdk_agent._operator_uses_google() is True, provider


def test_unreadable_config_keeps_the_original_behavior():
    with patch("artemis.config.llm.get_default_llm_config", side_effect=RuntimeError("no config")):
        assert sdk_agent._operator_uses_google() is True
