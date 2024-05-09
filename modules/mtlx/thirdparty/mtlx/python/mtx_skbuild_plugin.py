"""
This is a custom scikit-build-core plugin that will
fetch the MaterialX version from the CMake project.
"""
import os
import tempfile
import subprocess
from pathlib import Path
from typing import FrozenSet, Dict, Optional, Union, List

from scikit_build_core.file_api.query import stateless_query
from scikit_build_core.file_api.reply import load_reply_dir


def dynamic_metadata(
    fields: FrozenSet[str],
    settings: Optional[Dict[str, object]] = None,
) -> Dict[str, Union[str, Dict[str, Optional[str]]]]:
    print("mtx_skbuild_plugin: Computing MaterialX version from CMake...")

    if fields != {"version"}:
        msg = "Only the 'version' field is supported"
        raise ValueError(msg)

    if settings:
        msg = "No inline configuration is supported"
        raise ValueError(msg)

    current_dir = os.path.dirname(__file__)

    with tempfile.TemporaryDirectory() as tmpdir:
        # We will use CMake's file API to get the version
        # instead of parsing the CMakeLists files.

        # First generate the query folder so that CMake can generate replies.
        reply_dir = stateless_query(Path(tmpdir))

        # Run cmake (configure). CMake will generate a reply automatically.
        try:
            subprocess.run(
                [
                    "cmake",
                    "-S",
                    os.path.dirname(current_dir),
                    "-B",
                    tmpdir,
                    "-DMATERIALX_BUILD_SHARED_LIBS=OFF",
                    "-DMATERIALX_BUILD_PYTHON=OFF",
                    "-DMATERIALX_TEST_RENDER=OFF",
                    "-DMATERIALX_BUILD_TESTS=OFF",
                    "-DMATERIALX_INSTALL_PYTHON=OFF",
                    "-DMATERIALX_BUILD_RENDER=OFF",
                ],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                check=True,
                text=True,
            )
        except subprocess.CalledProcessError as exc:
            print(exc.stdout)
            raise RuntimeError(
                "Failed to configure project to get the version"
            ) from exc

        # Get the generated replies.
        index = load_reply_dir(reply_dir)

        # Get the version from the CMAKE_PROJECT_VERSION variable.
        entries = [
            entry
            for entry in index.reply.cache_v2.entries
            if entry.name == "CMAKE_PROJECT_VERSION"
        ]

        if not entries:
            raise ValueError("Could not find MaterialX version from CMake project")

        if len(entries) > 1:
            raise ValueError("More than one entry for CMAKE_PROJECT_VERSION found...")

    version = entries[0].value
    print("mtx_skbuild_plugin: Computed version: {0}".format(version))

    return {"version": version}


def get_requires_for_dynamic_metadata(
    _settings: Optional[Dict[str, object]] = None,
) -> List[str]:
    return ["cmake"]
