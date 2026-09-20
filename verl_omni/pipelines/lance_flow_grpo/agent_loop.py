# Copyright 2026 Bytedance Ltd. and/or its affiliates
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

"""Lance agent loop: reference media travels beside the prompt, not inside it.

Lance's bundle ships no processor, and the default loop only turns image content
into media streams through one.  Without an override an image-conditioned request
reaches the pipeline with empty ``multi_modal_data``, so ``LancePipeline.forward``
finds no reference, falls through to its text-to-image path and the reference is
dropped without an error.

Text tokenization is deliberately left alone: Lance recipes pre-tokenize prompts
in the model's own format (``prompt_token_ids``), and the default loop honours
those ids.
"""

from typing import Any

from verl.experimental.agent_loop.agent_loop import register

from verl_omni.agent_loop.single_turn_agent_loop import DiffusionSingleTurnAgentLoop


@register("lance_diffusion_single_turn_agent")
class LanceDiffusionSingleTurnAgentLoop(DiffusionSingleTurnAgentLoop):
    """Extract reference media without a processor, as LTX-2.3's loop does."""

    async def process_multi_modal_info(self, messages: list[dict]) -> dict[str, Any]:
        """Collect reference media from the message content itself.

        Nothing else can: Lance ships no processor, so the dataset layer's
        content items are the only place the reference lives, and it has already
        turned the ``<image>`` marker into an image item by the time this runs.
        """
        if self.processor is not None:
            return await super().process_multi_modal_info(messages)
        media: dict[str, list[Any]] = {"images": [], "videos": [], "audios": []}
        for message in messages:
            content = message.get("content")
            if not isinstance(content, list):
                continue
            for item in content:
                if not isinstance(item, dict):
                    continue
                kind = item.get("type")
                if kind == "image":
                    media["images"].append(item["image"])
                elif kind == "video":
                    media["videos"].append(item["video"])
                elif kind == "audio":
                    media["audios"].append(item["audio"])
        return {key: values for key, values in media.items() if values}

    def _assert_mm_supported(self, has_multi_modal: bool) -> None:
        """Allow reference media carried beside the prompt."""
        del has_multi_modal
