"""Helper methods for SCons interpreter"""
from contextlib import contextmanager
import os
from io import StringIO, TextIOBase
from typing import Generator, Optional

@contextmanager
def generated_wrapper(
    target,
    guard: Optional[bool] = None,
    prefix: str = "",
    suffix: str = "",
) -> Generator[TextIOBase, None, None]:
    """Context manager for generating files"""
    if isinstance(target, list):
        target = target[0]
    if isinstance(target, str):
        os.makedirs(os.path.dirname(target), exist_ok=True)
        with open(target, 'w') as f:
            f.write("/* THIS FILE IS GENERATED DO NOT EDIT */\n")
            if guard:
                guard_name = os.path.basename(target).replace(".", "_").upper()
                if prefix:
                    guard_name = prefix + "_" + guard_name
                if suffix:
                    guard_name = guard_name + "_" + suffix
                f.write(f"#ifndef {guard_name}\n")
                f.write(f"#define {guard_name}\n\n")
            yield f
            if guard:
                f.write(f"\n#endif // {guard_name}\n")
    else:
        yield target