from __future__ import annotations

import os
import re
import sys
from enum import Enum
from typing import Final

# Colors are disabled in non-TTY environments such as pipes. This means if output is redirected
# to a file, it won't contain color codes. Colors are enabled by default on continuous integration.

IS_CI: Final[bool] = bool(os.environ.get("CI"))
NO_COLOR: Final[bool] = bool(os.environ.get("NO_COLOR"))
CLICOLOR_FORCE: Final[bool] = bool(os.environ.get("CLICOLOR_FORCE"))
STDOUT_TTY: Final[bool] = bool(sys.stdout.isatty())
STDERR_TTY: Final[bool] = bool(sys.stderr.isatty())


_STDOUT_ORIGINAL: Final[bool] = False if NO_COLOR else CLICOLOR_FORCE or IS_CI or STDOUT_TTY
_STDERR_ORIGINAL: Final[bool] = False if NO_COLOR else CLICOLOR_FORCE or IS_CI or STDERR_TTY
_stdout_override: bool = _STDOUT_ORIGINAL
_stderr_override: bool = _STDERR_ORIGINAL


def is_stdout_color() -> bool:
    return _stdout_override


def is_stderr_color() -> bool:
    return _stderr_override


def force_stdout_color(value: bool) -> None:
    """
    Explicitly set `stdout` support for ANSI escape codes.
    If environment overrides exist, does nothing.
    """
    if not NO_COLOR or not CLICOLOR_FORCE:
        global _stdout_override
        _stdout_override = value


def force_stderr_color(value: bool) -> None:
    """
    Explicitly set `stderr` support for ANSI escape codes.
    If environment overrides exist, does nothing.
    """
    if not NO_COLOR or not CLICOLOR_FORCE:
        global _stderr_override
        _stderr_override = value


class Ansi(Enum):
    """
    Enum class for adding ANSI codepoints directly into strings. Automatically converts values to
    strings representing their internal value.
    """

    RESET = "\x1b[0m"

    BOLD = "\x1b[1m"
    DIM = "\x1b[2m"
    ITALIC = "\x1b[3m"
    UNDERLINE = "\x1b[4m"
    STRIKETHROUGH = "\x1b[9m"
    REGULAR = "\x1b[22;23;24;29m"

    BLACK = "\x1b[30m"
    RED = "\x1b[31m"
    GREEN = "\x1b[32m"
    YELLOW = "\x1b[33m"
    BLUE = "\x1b[34m"
    MAGENTA = "\x1b[35m"
    CYAN = "\x1b[36m"
    WHITE = "\x1b[37m"
    GRAY = "\x1b[90m"

    def __str__(self) -> str:
        return self.value


RE_ANSI = re.compile(r"\x1b\[[=\?]?[;\d]+[a-zA-Z]")


def color_print(*values: object, sep: str | None = " ", end: str | None = "\n", flush: bool = False) -> None:
    """Prints a colored message to `stdout`. If disabled, ANSI codes are automatically stripped."""
    if is_stdout_color():
        print(*values, sep=sep, end=f"{Ansi.RESET}{end}", flush=flush)
    else:
        print(RE_ANSI.sub("", (sep or " ").join(map(str, values))), sep="", end=end, flush=flush)


def color_printerr(*values: object, sep: str | None = " ", end: str | None = "\n", flush: bool = False) -> None:
    """Prints a colored message to `stderr`. If disabled, ANSI codes are automatically stripped."""
    if is_stderr_color():
        print(*values, sep=sep, end=f"{Ansi.RESET}{end}", flush=flush, file=sys.stderr)
    else:
        print(RE_ANSI.sub("", (sep or " ").join(map(str, values))), sep="", end=end, flush=flush, file=sys.stderr)


def print_info(*values: object) -> None:
    """Prints a informational message with formatting."""
    color_print(f"{Ansi.GRAY}{Ansi.BOLD}INFO:{Ansi.REGULAR}", *values)


def print_warning(*values: object) -> None:
    """Prints a warning message with formatting."""
    color_printerr(f"{Ansi.YELLOW}{Ansi.BOLD}WARNING:{Ansi.REGULAR}", *values)


def print_error(*values: object) -> None:
    """Prints an error message with formatting."""
    color_printerr(f"{Ansi.RED}{Ansi.BOLD}ERROR:{Ansi.REGULAR}", *values)


if sys.platform == "win32":

    def _win_color_fix():
        """Attempts to enable ANSI escape code support on Windows 10 and later."""
        from ctypes import POINTER, WINFUNCTYPE, WinError, windll
        from ctypes.wintypes import BOOL, DWORD, HANDLE

        STDOUT_HANDLE = -11
        STDERR_HANDLE = -12
        ENABLE_VIRTUAL_TERMINAL_PROCESSING = 4

        def err_handler(result, func, args):
            if not result:
                raise WinError()
            return args

        GetStdHandle = WINFUNCTYPE(HANDLE, DWORD)(("GetStdHandle", windll.kernel32), ((1, "nStdHandle"),))
        GetConsoleMode = WINFUNCTYPE(BOOL, HANDLE, POINTER(DWORD))(
            ("GetConsoleMode", windll.kernel32),
            ((1, "hConsoleHandle"), (2, "lpMode")),
        )
        GetConsoleMode.errcheck = err_handler
        SetConsoleMode = WINFUNCTYPE(BOOL, HANDLE, DWORD)(
            ("SetConsoleMode", windll.kernel32),
            ((1, "hConsoleHandle"), (1, "dwMode")),
        )
        SetConsoleMode.errcheck = err_handler

        for handle_id in [STDOUT_HANDLE, STDERR_HANDLE]:
            try:
                handle = GetStdHandle(handle_id)
                flags = GetConsoleMode(handle)
                SetConsoleMode(handle, flags | ENABLE_VIRTUAL_TERMINAL_PROCESSING)
            except OSError:
                pass

    _win_color_fix()
