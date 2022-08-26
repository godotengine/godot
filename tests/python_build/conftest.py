import os
import sys
from pathlib import Path

import pytest

CWD = Path(__file__).parent
ROOT = CWD.parent.parent
# append directory with build files to sys.path to import them
sys.path.append(str(ROOT))


@pytest.fixture
def shader_files(request):
    shader_path = request.param

    res = {
        "path_input": str(CWD / "fixtures" / f"{shader_path}.glsl"),
        "path_output": str(CWD / "fixtures" / f"{shader_path}.glsl.gen.h"),
        "path_expected_full": str(CWD / "fixtures" / f"{shader_path}_expected_full.glsl"),
        "path_expected_parts": str(CWD / "fixtures" / f"{shader_path}_expected_parts.json"),
    }
    yield res

    if not os.getenv("PYTEST_KEEP_GENERATED_FILES"):
        os.remove(res["path_output"])
