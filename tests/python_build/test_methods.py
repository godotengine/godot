import pytest
import sys

from unittest.mock import MagicMock

from methods import get_debug_symbols_enum


@pytest.mark.parametrize(
    "default,debug_symbols,separate_debug_symbols,expected",
    [
        (False, "no", None, "no"),
        (True, None, None, "embedded"),
        (False, "embedded", None, "embedded"),
        (False, "separate", None, "separate"),
        # backwards compatibility checks
        (False, "yes", None, "embedded"),
        (False, "yes", "no", "embedded"),
        (False, "yes", "yes", "separate"),
        (False, "true", None, "embedded"),
        (False, "false", None, "no"),
    ],
)
def test_get_debug_symbols_enum(default, debug_symbols, separate_debug_symbols, expected):
    # mock SCons via sys.modules so we can run this test even if SCons isn't installed
    scons_script_mock = MagicMock()
    scons_script_mock.ARGUMENTS = {"debug_symbols": debug_symbols, "separate_debug_symbols": separate_debug_symbols}
    sys.modules["SCons.Script"] = scons_script_mock
    scons_vars_mock = MagicMock()
    scons_vars_mock.TRUE_STRINGS = ("y", "yes", "true", "t", "1", "on", "all")
    scons_vars_mock._text2bool = lambda x: x in scons_vars_mock.TRUE_STRINGS
    sys.modules["SCons.Variables.BoolVariable"] = scons_vars_mock
    result = get_debug_symbols_enum(default)

    assert expected == result
