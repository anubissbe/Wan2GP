import copy
import math
import re
from typing import List

import gradio as gr

from shared.utils.plugins import WAN2GPPlugin


SEGMENT_SECONDS_DEFAULT = 7


class AutoVideoComposerPlugin(WAN2GPPlugin):
    def __init__(self):
        super().__init__()
        self.name = "Auto Video Composer"
        self.version = "1.0.0"
        self.description = (
            "Split long prompts into coherent 7-second scenes and queue them automatically."
        )

    def setup_ui(self):
        self.request_global("get_current_model_settings")
        self.request_global("get_gen_info")
        self.request_global("get_base_model_type")
        self.request_global("get_computed_fps")
        self.request_global("get_model_min_frames_and_step")
        self.request_global("add_video_task")
        self.request_global("update_queue_data")

        self.request_component("state")
        self.request_component("queue_html_container")

        self.add_tab(
            tab_id="auto_video_composer",
            label="Auto Video Composer",
            component_constructor=self._build_ui,
        )

    def _build_ui(self):
        with gr.Column():
            gr.Markdown(
                """
                ### Auto Video Composer
                Geef één grote prompt en de gewenste totale duur in minuten. De plugin splitst de prompt
                automatisch op in coherente scènes van 7 seconden, vult de generatie-queue en houdt de
                prompts voor elke scène consistent zodat de uiteindelijke video één geheel lijkt.
                """
            )
            prompt = gr.Textbox(
                label="Volledige prompt",
                lines=8,
                placeholder="Beschrijf het volledige verhaal of de hele scène in één prompt...",
            )
            with gr.Row():
                total_minutes = gr.Number(
                    label="Totale duur (minuten)",
                    value=1.0,
                    precision=2,
                    minimum=0.1,
                )
                segment_seconds = gr.Number(
                    label="Lengte per scène (seconden)",
                    value=SEGMENT_SECONDS_DEFAULT,
                    precision=1,
                    minimum=1,
                    maximum=30,
                    info="Standaard 7 seconden per segment.",
                )

            summary = gr.Markdown("Nog geen segmenten aangemaakt.")
            update_queue = gr.Button("Genereer segmenten en vul queue", variant="primary")

        outputs = [summary]
        if hasattr(self, "queue_html_container"):
            outputs.append(self.queue_html_container)

        update_queue.click(
            fn=self._create_segments,
            inputs=[self.state, prompt, total_minutes, segment_seconds],
            outputs=outputs,
        )

        return summary

    def _create_segments(
        self,
        state: dict,
        prompt: str,
        total_minutes: float,
        segment_seconds: float,
    ):
        errors = self._validate_inputs(prompt, total_minutes, segment_seconds)
        has_queue_output = hasattr(self, "queue_html_container")
        if errors:
            return self._build_error_response(errors, has_queue_output)

        settings = copy.deepcopy(self.get_current_model_settings(state))
        if settings is None:
            return self._build_error_response("Kon de huidige modelinstellingen niet ophalen.")

        settings.pop("lset_name", None)
        settings.setdefault("mode", "")

        model_type = state.get("model_type")
        model_filename = state.get("model_filename")
        base_model_type = self.get_base_model_type(model_type)

        if base_model_type is None:
            return self._build_error_response(
                "Kon de basismodelgegevens niet bepalen. Kies een geldig model en probeer opnieuw.",
                has_queue_output,
            )

        fps = self.get_computed_fps(
            settings.get("force_fps", ""),
            base_model_type,
            settings.get("video_guide"),
            settings.get("video_source"),
        )

        if not isinstance(fps, (int, float)) or fps <= 0:
            return self._build_error_response(
                "Kon een geldige FPS-waarde niet bepalen. Controleer de modelinstellingen.",
                has_queue_output,
            )

        min_frames, frame_step, _ = self.get_model_min_frames_and_step(base_model_type)

        if min_frames <= 0 or frame_step <= 0:
            return self._build_error_response(
                "Modelconfiguratie heeft ongeldige frame-waarden. Controleer het geselecteerde model.",
                has_queue_output,
            )

        segments_plan = self._build_segments_plan(
            prompt, total_minutes, segment_seconds, fps, min_frames, frame_step
        )

        gen = self.get_gen_info(state)
        queue = gen.setdefault("queue", [])

        for segment in segments_plan:
            segment_settings = copy.deepcopy(settings)
            segment_settings.update(
                {
                    "prompt": segment["prompt"],
                    "video_length": segment["frames"],
                    "state": state,
                    "model_type": model_type,
                    "model_filename": model_filename,
                    "plugin_data": {
                        "auto_video_composer": {
                            "segment": segment["index"],
                            "total_segments": len(segments_plan),
                            "segment_seconds": segment["seconds"],
                            "fps": fps,
                        }
                    },
                }
            )
            self.add_video_task(**segment_settings)

        gen["prompts_max"] = gen.get("prompts_max", 0) + len(segments_plan)

        queue_update = (
            self.update_queue_data(queue) if hasattr(self, "update_queue_data") else None
        )

        summary_md = self._render_summary(segments_plan)

        if queue_update is None:
            return summary_md

        return summary_md, queue_update

    @staticmethod
    def _validate_inputs(prompt: str, total_minutes: float, segment_seconds: float):
        if not prompt or len(prompt.strip()) == 0:
            return "Prompt mag niet leeg zijn."
        if total_minutes <= 0:
            return "Totale duur moet groter dan nul zijn."
        if segment_seconds <= 0:
            return "Segmentlengte moet groter dan nul zijn."
        return None

    @staticmethod
    def _build_error_response(message: str, include_queue: bool = False):
        gr.Warning(message)
        if include_queue:
            return f"⚠️ {message}", gr.update()
        return f"⚠️ {message}"

    @staticmethod
    def _split_prompt(prompt: str, segments: int) -> List[str]:
        sentences = [
            part.strip()
            for part in re.split(r"[\n\r]+|(?<=[.!?])\s+", prompt)
            if part.strip()
        ]
        if not sentences:
            return [prompt.strip()] * segments

        chunk_size = max(1, math.ceil(len(sentences) / segments))
        chunks = []
        cursor = 0
        for _ in range(segments):
            chunk = sentences[cursor : cursor + chunk_size]
            if not chunk:
                chunk = [sentences[-1]]
            chunks.append(" ".join(chunk))
            cursor += chunk_size
        return chunks

    def _build_segments_plan(
        self,
        prompt: str,
        total_minutes: float,
        segment_seconds: float,
        fps: float,
        min_frames: int,
        frame_step: int,
    ) -> List[dict]:
        total_seconds = total_minutes * 60
        segments_count = max(1, math.ceil(total_seconds / segment_seconds))
        chunks = self._split_prompt(prompt, segments_count)

        plan = []
        for idx in range(segments_count):
            duration = (
                segment_seconds
                if idx < segments_count - 1
                else max(segment_seconds, total_seconds - segment_seconds * (segments_count - 1))
            )
            frames = max(
                min_frames,
                int(round(duration * fps / frame_step)) * frame_step,
            )
            segment_prompt = (
                f"{chunks[idx]} "
                f"(Segment {idx + 1}/{segments_count}. Houd stijl, personages en camera gelijkmatig)."
            )
            plan.append(
                {
                    "index": idx + 1,
                    "prompt": segment_prompt,
                    "seconds": round(duration, 2),
                    "frames": frames,
                }
            )
        return plan

    @staticmethod
    def _render_summary(plan: List[dict]) -> str:
        rows = [
            f"- Segment {item['index']}: {item['seconds']}s (~{item['frames']} frames) — {item['prompt']}"
            for item in plan
        ]
        return "\n".join(["### Segmenten", *rows])
