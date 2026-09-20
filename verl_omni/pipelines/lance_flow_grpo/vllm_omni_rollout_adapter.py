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


"""Lance rollout-side adapter for FlowGRPO.

vllm-omni's ``LancePipeline`` inherits ``BagelPipeline`` and overrides only
model construction, so the RL wrapper composes the two: the method resolution
order puts :class:`BagelPipelineWithLogProb` first, which adds the SDE
scheduler, trajectory recording and SDE windowing, and its ``super().forward``
lands on :class:`LancePipeline`, which builds and runs the Lance model.

Only Lance's sigma shift (3.5) differs from BAGEL's 3.0, and that is a class
attribute on the BAGEL adapter.
"""

from __future__ import annotations

import logging
from types import SimpleNamespace

from vllm_omni.diffusion.data import OmniDiffusionConfig
from vllm_omni.diffusion.models.lance.pipeline_lance import LancePipeline

from verl_omni.pipelines.bagel_flow_grpo.vllm_omni_rollout_adapter import BagelPipelineWithLogProb
from verl_omni.pipelines.model_base import VllmOmniPipelineBase
from verl_omni.pipelines.rollout_media import DiffusionIOSpec, MediaSpec
from verl_omni.pipelines.schedulers import FlowMatchSDEDiscreteScheduler
from verl_omni.workers.config import DiffusionModelConfig

from .common import LANCE_TIMESTEP_SHIFT, setup_lance_sigmas

logger = logging.getLogger(__name__)


@VllmOmniPipelineBase.register("OmniLanceForConditionalGeneration", algorithm="flow_grpo")
class LancePipelineWithLogProb(BagelPipelineWithLogProb, LancePipeline):
    """Lance pipeline variant for RL rollouts with verl-omni.

    ``LancePipeline.__init__`` deliberately does not call up the MRO - it
    replaces BAGEL's construction wholesale - so ``BagelPipelineWithLogProb``
    still reaches it through ``super().__init__`` and installs the SDE
    scheduler afterwards.

    One class serves both Lance checkpoints, text-to-image and text-to-video,
    because that is how vllm-omni ships them: ``Lance_3B`` and
    ``Lance_3B_Video`` share the ``OmniLanceForConditionalGeneration``
    architecture and differ only in the checkpoint the pipeline is pointed at.
    The primary media stream is therefore resolved per run by
    :meth:`flowgrpo_io_spec` rather than fixed per architecture.
    """

    #: Text-to-image, overridden for the video checkpoint.  This is the value
    #: the shared strategy turns into the ``modalities`` entry
    #: ``LancePipeline.forward`` routes on, so it is what decides between the
    #: image path and ``_forward_t2v``.
    diffusion_io_spec = DiffusionIOSpec(primary=MediaSpec("image"))

    #: Lance's rollout default (``LANCE_DEFAULTS.timestep_shift``).
    flowgrpo_timestep_shift: float = LANCE_TIMESTEP_SHIFT

    @classmethod
    def flowgrpo_io_spec(cls, model_config: DiffusionModelConfig) -> DiffusionIOSpec:
        """Declare a video stream when the run is pointed at the video checkpoint.

        ``LancePipeline`` picks its video variant from the model path
        (``LancePipeline._select_video_variant``), and the rollout has to make
        the same choice or a video checkpoint would be asked for a text-to-image
        request.  Ask the pipeline itself for the decision instead of restating
        its rule here, so the two cannot drift apart.
        """
        model = getattr(model_config, "local_path", None) or getattr(model_config, "path", "") or ""
        od_config = SimpleNamespace(model=model, extra=getattr(model_config, "extra", None))
        if LancePipeline._select_video_variant(od_config):
            return DiffusionIOSpec(primary=MediaSpec("video"))
        return cls.diffusion_io_spec

    @classmethod
    def flowgrpo_setup_sigmas(cls, scheduler: FlowMatchSDEDiscreteScheduler, num_steps: int, shift: float) -> None:
        """Lance's schedule: one more point than BAGEL's, and a 3.5 shift."""
        setup_lance_sigmas(scheduler, num_steps, shift=shift)

    def __init__(self, *, od_config: OmniDiffusionConfig, prefix: str = ""):
        super().__init__(od_config=od_config, prefix=prefix)
        logger.info("LancePipelineWithLogProb: SDE scheduler enabled, timestep_shift=%s", LANCE_TIMESTEP_SHIFT)

    @classmethod
    def flowgrpo_media_keys(cls, model_config: DiffusionModelConfig) -> dict[str, str]:
        """Name a video run's reference frame the way the i2v node reads it.

        The rollout transport only knows media modalities, so a conditioning
        frame arrives as ``multi_modal_data["image"]``.  On a video checkpoint
        that frame is a first frame, and ``LancePipeline.forward`` routes to
        ``_forward_i2v`` only when it arrives as ``first_frame``: under the plain
        key the request falls through to ``_forward_t2v`` and the reference is
        dropped without an error.  The image checkpoint keeps the plain key,
        which is what ``_forward_image_edit`` reads.
        """
        model = getattr(model_config, "local_path", None) or getattr(model_config, "path", "") or ""
        od_config = SimpleNamespace(model=model, extra=getattr(model_config, "extra", None))
        if LancePipeline._select_video_variant(od_config):
            return {"image": "first_frame"}
        return {}
