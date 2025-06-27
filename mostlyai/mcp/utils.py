# Copyright 2025 MOSTLY AI
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

import json
import re
import time
from collections.abc import Callable

import pandas as pd
import requests
from fastmcp import Context

from mostlyai.sdk.domain import ProgressStatus, TaskType

_DOC_CACHE = {}
DF_AS_DICT_MAX_ROWS = 100
PROGRESS_INTERVAL_SECONDS = 1


def doc_section(starts_with: str, doc_url="https://mostly-ai.github.io/mostlyai/llms-full.txt") -> str:
    if doc_url not in _DOC_CACHE:
        try:
            _DOC_CACHE[doc_url] = requests.get(doc_url).text
        except Exception as e:
            return f"Error fetching document: {e}"

    lines = _DOC_CACHE[doc_url].splitlines()
    section_lines = []
    capture = False

    for line in lines:
        if not capture and line.strip().startswith(starts_with):
            capture = True
            section_lines.append(line)
        elif capture:
            if re.match(r"^\s*##+", line):  # next section begins
                break
            section_lines.append(line)

    return "\n".join(section_lines).strip() or starts_with


def df_as_dict(obj):
    """convert a pd.DataFrame or dict[str, pd.DataFrame] to a dict recursively, limiting DataFrame rows to DF_AS_DICT_MAX_ROWS, and including row count metadata. ensures all values are JSON-serializable."""
    if isinstance(obj, pd.DataFrame):
        total_rows = len(obj)
        truncated = total_rows > DF_AS_DICT_MAX_ROWS
        # use to_json for robust serialization, then parse back to Python object
        json_str = obj.head(DF_AS_DICT_MAX_ROWS).to_json(orient="records")
        data = json.loads(json_str)
        return {"data": data, "row_count": total_rows, "truncated": truncated}
    if isinstance(obj, dict):
        return {k: df_as_dict(v) for k, v in obj.items()}
    return obj


async def job_wait(ctx: Context, get_progress_fn: Callable, progress_bar: bool = True) -> None:
    """
    Similar to mostlyai.sdk.client._utils.job_wait, but for MCP.
    """

    job_progress = get_progress_fn()
    task_type = job_progress.steps[0].task_type
    while job_progress.status not in [ProgressStatus.done, ProgressStatus.failed, ProgressStatus.canceled]:
        if progress_bar:
            job_progress_value = job_progress.progress.value
            job_progress_max = job_progress.progress.max
            job_progress_percentage = (
                float(job_progress_value) / job_progress_max * 100.0 if job_progress_max > 0 else 0.0
            )
            if job_progress.status == ProgressStatus.queued:
                await ctx.report_progress(
                    progress=job_progress_value,
                    total=job_progress_max,
                    message=f"Your {'synthetic dataset' if task_type == TaskType.generate else 'generator'} is queued. Please wait...",
                )
            else:
                for step in job_progress.steps:
                    if step.status not in [ProgressStatus.done, ProgressStatus.failed, ProgressStatus.canceled]:
                        step_progress_value = step.progress.value
                        step_progress_max = step.progress.max
                        step_progress_percentage = (
                            float(step_progress_value) / step_progress_max * 100.0 if step_progress_max > 0 else 0.0
                        )
                        await ctx.report_progress(
                            progress=job_progress_value,
                            total=job_progress_max,
                            message=f"[{job_progress_percentage:3.0f}%] {step.model_label} - {step.step_code.value}: {step_progress_percentage:3.0f}%",
                        )
                        break
        time.sleep(PROGRESS_INTERVAL_SECONDS)
        job_progress = get_progress_fn()

    if progress_bar:
        if job_progress.status == ProgressStatus.done:
            message = f"🎉 Your {'synthetic dataset' if task_type == TaskType.generate else 'generator'} is ready!"
        elif job_progress.status == ProgressStatus.failed:
            message = f"{'Generation' if task_type == TaskType.generate else 'Training'} failed."
        elif job_progress.status == ProgressStatus.canceled:
            message = f"{'Generation' if task_type == TaskType.generate else 'Training'} was canceled."
        await ctx.report_progress(
            progress=job_progress_max,
            total=job_progress_max,
            message=message,
        )
