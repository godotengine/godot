import sys
from godot_consts import *

verbose_enabled = True
debug_enabled = True
should_color = True

STYLES: Dict[str, str] = {}

def set_should_color(should):
    global should_color

    # Enable ANSI escape code support on Windows 10 and later (for colored console output).
    # <https://github.com/python/cpython/issues/73245>
    if should_color and sys.stdout.isatty() and sys.platform == "win32":
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
        except Exception:
            should_color = False

    STYLES["red"] = "\x1b[91m" if should_color else ""
    STYLES["green"] = "\x1b[92m" if should_color else ""
    STYLES["yellow"] = "\x1b[93m" if should_color else ""
    STYLES["bold"] = "\x1b[1m" if should_color else ""
    STYLES["regular"] = "\x1b[22m" if should_color else ""
    STYLES["reset"] = "\x1b[0m" if should_color else ""


def print_error(error: str, state=None) -> None:
    print(f'{STYLES["red"]}{STYLES["bold"]}ERROR:{STYLES["regular"]} {error}{STYLES["reset"]}')
    if(state != None):
        state.num_errors += 1


def print_warning(warning: str, state=None) -> None:
    print(f'{STYLES["yellow"]}{STYLES["bold"]}WARNING:{STYLES["regular"]} {warning}{STYLES["reset"]}')
    if(state != None):
        state.num_warnings += 1


def print_style(sname: str, text: str, plain_text="") -> None:
    print(f'{STYLES[sname]}{text}{STYLES["reset"]}{plain_text}')


def _print_arg(sname, prefix, arg):
    out = ""
    for a in arg:
        out += str(a) + " "
    print_style(sname, prefix, out)


def vprint(*arg) -> None:
    if(verbose_enabled):
        _print_arg("bold", "[Info]  ", arg)


def dbg(*arg) -> None:
    if(debug_enabled):
        _print_arg("bold", "[Debug]  ", arg)

