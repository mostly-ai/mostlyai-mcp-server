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

import pandas as pd
import requests

_DOC_CACHE = {}
DF_AS_DICT_MAX_ROWS = 100


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
