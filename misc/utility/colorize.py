import os
import sys
from enum import Enum
from typing import Optional

IS_CI: bool = bool(os.environ.get("CI"))
IS_TTY: bool = sys.stdout.isatty()


def _color_supported() -> bool:
    """
    Enables ANSI escape code support on Windows 10 and later (for colored console output).
    See here: https://github.com/python/cpython/issues/73245
    """
    if sys.platform == "win32" and IS_TTY:
        try:
            from ctypes import WinError, byref, windll  # type: ignore
            from ctypes.wintypes import DWORD  # type: ignore

            stdout_handle = windll.kernel32.GetStdHandle(DWORD(-11))
            mode = DWORD(0)
            if not windll.kernel32.GetConsoleMode(stdout_handle, byref(mode)):
                raise WinError()
            mode = DWORD(mode.value | 4)
            if not windll.kernel32.SetConsoleMode(stdout_handle, mode):
                raise WinError()
        except (TypeError, OSError) as e:
            print(f"Failed to enable ANSI escape code support, disabling color output.\n{e}", file=sys.stderr)
            return False

    return IS_TTY or IS_CI


# Colors are disabled in non-TTY environments such as pipes. This means
# that if output is redirected to a file, it won't contain color codes.
# Colors are always enabled on continuous integration.
COLOR_SUPPORTED: bool = _color_supported()
_can_color: bool = COLOR_SUPPORTED


def toggle_color(value: Optional[bool] = None) -> None:
    """
    Explicitly toggle color codes, regardless of support.

    - `value`: An optional boolean to explicitly set the color
    state instead of toggling.
    """
    global _can_color
    _can_color = value if value is not None else not _can_color


class Ansi(Enum):
    """
    Enum class for adding ansi colorcodes directly into strings.
    Automatically converts values to strings representing their
    internal value, or an empty string in a non-colorized scope.
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

    LIGHT_BLACK = "\x1b[90m"
    LIGHT_RED = "\x1b[91m"
    LIGHT_GREEN = "\x1b[92m"
    LIGHT_YELLOW = "\x1b[93m"
    LIGHT_BLUE = "\x1b[94m"
    LIGHT_MAGENTA = "\x1b[95m"
    LIGHT_CYAN = "\x1b[96m"
    LIGHT_WHITE = "\x1b[97m"

    GRAY = LIGHT_BLACK if IS_CI else BLACK
    """
    Special case. GitHub Actions doesn't convert `BLACK` to gray as expected, but does convert `LIGHT_BLACK`.
    By implementing `GRAY`, we handle both cases dynamically, while still allowing for explicit values if desired.
    """

    def __str__(self) -> str:
        global _can_color
        return str(self.value) if _can_color else ""


def print_info(*values: object) -> None:
    """Prints a informational message with formatting."""
    print(f"{Ansi.GRAY}{Ansi.BOLD}INFO:{Ansi.REGULAR}", *values, Ansi.RESET)


def print_warning(*values: object) -> None:
    """Prints a warning message with formatting."""
    print(f"{Ansi.YELLOW}{Ansi.BOLD}WARNING:{Ansi.REGULAR}", *values, Ansi.RESET, file=sys.stderr)


def print_error(*values: object) -> None:
    """Prints an error message with formatting."""
    print(f"{Ansi.RED}{Ansi.BOLD}ERROR:{Ansi.REGULAR}", *values, Ansi.RESET, file=sys.stderr)
