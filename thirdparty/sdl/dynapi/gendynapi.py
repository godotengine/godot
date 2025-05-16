#!/usr/bin/env python3

#  Simple DirectMedia Layer
#  Copyright (C) 1997-2025 Sam Lantinga <slouken@libsdl.org>
#
#  This software is provided 'as-is', without any express or implied
#  warranty.  In no event will the authors be held liable for any damages
#  arising from the use of this software.
#
#  Permission is granted to anyone to use this software for any purpose,
#  including commercial applications, and to alter it and redistribute it
#  freely, subject to the following restrictions:
#
#  1. The origin of this software must not be misrepresented; you must not
#     claim that you wrote the original software. If you use this software
#     in a product, an acknowledgment in the product documentation would be
#     appreciated but is not required.
#  2. Altered source versions must be plainly marked as such, and must not be
#     misrepresented as being the original software.
#  3. This notice may not be removed or altered from any source distribution.

# WHAT IS THIS?
#  When you add a public API to SDL, please run this script, make sure the
#  output looks sane (git diff, it adds to existing files), and commit it.
#  It keeps the dynamic API jump table operating correctly.
#
#  Platform-specific API:
#   After running the script, you have to manually add #ifdef SDL_PLATFORM_WIN32
#   or similar around the function in 'SDL_dynapi_procs.h'.
#

import argparse
import dataclasses
import json
import logging
import os
from pathlib import Path
import pprint
import re


SDL_ROOT = Path(__file__).resolve().parents[2]

SDL_INCLUDE_DIR = SDL_ROOT / "include/SDL3"
SDL_DYNAPI_PROCS_H = SDL_ROOT / "src/dynapi/SDL_dynapi_procs.h"
SDL_DYNAPI_OVERRIDES_H = SDL_ROOT / "src/dynapi/SDL_dynapi_overrides.h"
SDL_DYNAPI_SYM = SDL_ROOT / "src/dynapi/SDL_dynapi.sym"

RE_EXTERN_C = re.compile(r'.*extern[ "]*C[ "].*')
RE_COMMENT_REMOVE_CONTENT = re.compile(r'\/\*.*\*/')
RE_PARSING_FUNCTION = re.compile(r'(.*SDLCALL[^\(\)]*) ([a-zA-Z0-9_]+) *\((.*)\) *;.*')

#eg:
# void (SDLCALL *callback)(void*, int)
# \1(\2)\3
RE_PARSING_CALLBACK = re.compile(r'([^\(\)]*)\(([^\(\)]+)\)(.*)')


logger = logging.getLogger(__name__)


@dataclasses.dataclass(frozen=True)
class SdlProcedure:
    retval: str
    name: str
    parameter: list[str]
    parameter_name: list[str]
    header: str
    comment: str

    @property
    def variadic(self) -> bool:
        return "..." in self.parameter


def parse_header(header_path: Path) -> list[SdlProcedure]:
    logger.debug("Parse header: %s", header_path)

    header_procedures = []

    parsing_function = False
    current_func = ""
    parsing_comment = False
    current_comment = ""
    ignore_wiki_documentation = False

    with header_path.open() as f:
        for line in f:

            # Skip lines if we're in a wiki documentation block.
            if ignore_wiki_documentation:
                if line.startswith("#endif"):
                    ignore_wiki_documentation = False
                continue

            # Discard wiki documentations blocks.
            if line.startswith("#ifdef SDL_WIKI_DOCUMENTATION_SECTION"):
                ignore_wiki_documentation = True
                continue

            # Discard pre-processor directives ^#.*
            if line.startswith("#"):
                continue

            # Discard "extern C" line
            match = RE_EXTERN_C.match(line)
            if match:
                continue

            # Remove one line comment // ...
            # eg: extern SDL_DECLSPEC SDL_hid_device * SDLCALL SDL_hid_open_path(const char *path, int bExclusive /* = false */)
            line = RE_COMMENT_REMOVE_CONTENT.sub('', line)

            # Get the comment block /* ... */ across several lines
            match_start = "/*" in line
            match_end = "*/" in line
            if match_start and match_end:
                continue
            if match_start:
                parsing_comment = True
                current_comment = line
                continue
            if match_end:
                parsing_comment = False
                current_comment += line
                continue
            if parsing_comment:
                current_comment += line
                continue

            # Get the function prototype across several lines
            if parsing_function:
                # Append to the current function
                current_func += " "
                current_func += line.strip()
            else:
                # if is contains "extern", start grabbing
                if "extern" not in line:
                    continue
                # Start grabbing the new function
                current_func = line.strip()
                parsing_function = True

            # If it contains ';', then the function is complete
            if ";" not in current_func:
                continue

            # Got function/comment, reset vars
            parsing_function = False
            func = current_func
            comment = current_comment
            current_func = ""
            current_comment = ""

            # Discard if it doesn't contain 'SDLCALL'
            if "SDLCALL" not in func:
                logger.debug("  Discard, doesn't have SDLCALL: %r", func)
                continue

            # Discard if it contains 'SDLMAIN_DECLSPEC' (these are not SDL symbols).
            if "SDLMAIN_DECLSPEC" in func:
                logger.debug("  Discard, has SDLMAIN_DECLSPEC: %r", func)
                continue

            logger.debug("Raw data: %r", func)

            # Replace unusual stuff...
            func = func.replace(" SDL_PRINTF_VARARG_FUNC(1)", "")
            func = func.replace(" SDL_PRINTF_VARARG_FUNC(2)", "")
            func = func.replace(" SDL_PRINTF_VARARG_FUNC(3)", "")
            func = func.replace(" SDL_PRINTF_VARARG_FUNC(4)", "")
            func = func.replace(" SDL_PRINTF_VARARG_FUNCV(1)", "")
            func = func.replace(" SDL_PRINTF_VARARG_FUNCV(2)", "")
            func = func.replace(" SDL_PRINTF_VARARG_FUNCV(3)", "")
            func = func.replace(" SDL_PRINTF_VARARG_FUNCV(4)", "")
            func = func.replace(" SDL_WPRINTF_VARARG_FUNC(3)", "")
            func = func.replace(" SDL_WPRINTF_VARARG_FUNCV(3)", "")
            func = func.replace(" SDL_SCANF_VARARG_FUNC(2)", "")
            func = func.replace(" SDL_SCANF_VARARG_FUNCV(2)", "")
            func = func.replace(" SDL_ANALYZER_NORETURN", "")
            func = func.replace(" SDL_MALLOC", "")
            func = func.replace(" SDL_ALLOC_SIZE2(1, 2)", "")
            func = func.replace(" SDL_ALLOC_SIZE(2)", "")
            func = re.sub(r" SDL_ACQUIRE\(.*\)", "", func)
            func = re.sub(r" SDL_ACQUIRE_SHARED\(.*\)", "", func)
            func = re.sub(r" SDL_TRY_ACQUIRE\(.*\)", "", func)
            func = re.sub(r" SDL_TRY_ACQUIRE_SHARED\(.*\)", "", func)
            func = re.sub(r" SDL_RELEASE\(.*\)", "", func)
            func = re.sub(r" SDL_RELEASE_SHARED\(.*\)", "", func)
            func = re.sub(r" SDL_RELEASE_GENERIC\(.*\)", "", func)
            func = re.sub(r"([ (),])(SDL_IN_BYTECAP\([^)]*\))", r"\1", func)
            func = re.sub(r"([ (),])(SDL_OUT_BYTECAP\([^)]*\))", r"\1", func)
            func = re.sub(r"([ (),])(SDL_INOUT_Z_CAP\([^)]*\))", r"\1", func)
            func = re.sub(r"([ (),])(SDL_OUT_Z_CAP\([^)]*\))", r"\1", func)

            # Should be a valid function here
            match = RE_PARSING_FUNCTION.match(func)
            if not match:
                logger.error("Cannot parse: %s", func)
                raise ValueError(func)

            func_ret = match.group(1)
            func_name = match.group(2)
            func_params = match.group(3)

            #
            # Parse return value
            #
            func_ret = func_ret.replace('extern', ' ')
            func_ret = func_ret.replace('SDLCALL', ' ')
            func_ret = func_ret.replace('SDL_DECLSPEC', ' ')
            func_ret, _ = re.subn('([ ]{2,})', ' ', func_ret)
            # Remove trailing spaces in front of '*'
            func_ret = func_ret.replace(' *', '*')
            func_ret = func_ret.strip()

            #
            # Parse parameters
            #
            func_params = func_params.strip()
            if func_params == "":
                func_params = "void"

            # Identify each function parameters with type and name
            # (eventually there are callbacks of several parameters)
            tmp = func_params.split(',')
            tmp2 = []
            param = ""
            for t in tmp:
                if param == "":
                    param = t
                else:
                    param = param + "," + t
                # Identify a callback or parameter when there is same count of '(' and ')'
                if param.count('(') == param.count(')'):
                    tmp2.append(param.strip())
                    param = ""

            # Process each parameters, separation name and type
            func_param_type = []
            func_param_name = []
            for t in tmp2:
                if t == "void":
                    func_param_type.append(t)
                    func_param_name.append("")
                    continue

                if t == "...":
                    func_param_type.append(t)
                    func_param_name.append("")
                    continue

                param_name = ""

                # parameter is a callback
                if '(' in t:
                    match = RE_PARSING_CALLBACK.match(t)
                    if not match:
                        logger.error("cannot parse callback: %s", t)
                        raise ValueError(t)
                    a = match.group(1).strip()
                    b = match.group(2).strip()
                    c = match.group(3).strip()

                    try:
                        (param_type, param_name) = b.rsplit('*', 1)
                    except:
                        param_type = t
                        param_name = "param_name_not_specified"

                    # bug rsplit ??
                    if param_name == "":
                        param_name = "param_name_not_specified"

                    # reconstruct a callback name for future parsing
                    func_param_type.append(a + " (" + param_type.strip() + " *REWRITE_NAME)" + c)
                    func_param_name.append(param_name.strip())

                    continue

                # array like "char *buf[]"
                has_array = False
                if t.endswith("[]"):
                    t = t.replace("[]", "")
                    has_array = True

                # pointer
                if '*' in t:
                    try:
                        (param_type, param_name) = t.rsplit('*', 1)
                    except:
                        param_type = t
                        param_name = "param_name_not_specified"

                    # bug rsplit ??
                    if param_name == "":
                        param_name = "param_name_not_specified"

                    val = param_type.strip() + "*REWRITE_NAME"

                    # Remove trailing spaces in front of '*'
                    tmp = ""
                    while val != tmp:
                        tmp = val
                        val = val.replace('  ', ' ')
                    val = val.replace(' *', '*')
                    # first occurrence
                    val = val.replace('*', ' *', 1)
                    val = val.strip()

                else: # non pointer
                    # cut-off last word on
                    try:
                        (param_type, param_name) = t.rsplit(' ', 1)
                    except:
                        param_type = t
                        param_name = "param_name_not_specified"

                    val = param_type.strip() + " REWRITE_NAME"

                # set back array
                if has_array:
                    val += "[]"

                func_param_type.append(val)
                func_param_name.append(param_name.strip())

            new_proc = SdlProcedure(
                retval=func_ret,                # Return value type
                name=func_name,                 # Function name
                comment=comment,                # Function comment
                header=header_path.name,        # Header file
                parameter=func_param_type,      # List of parameters (type + anonymized param name 'REWRITE_NAME')
                parameter_name=func_param_name, # Real parameter name, or 'param_name_not_specified'
            )

            header_procedures.append(new_proc)

            if logger.getEffectiveLevel() <= logging.DEBUG:
                logger.debug("%s", pprint.pformat(new_proc))

    return header_procedures


# Dump API into a json file
def full_API_json(path: Path, procedures: list[SdlProcedure]):
    with path.open('w', newline='') as f:
        json.dump([dataclasses.asdict(proc) for proc in procedures], f, indent=4, sort_keys=True)
        logger.info("dump API to '%s'", path)


class CallOnce:
    def __init__(self, cb):
        self._cb = cb
        self._called = False
    def __call__(self, *args, **kwargs):
        if self._called:
            return
        self._called = True
        self._cb(*args, **kwargs)


# Check public function comments are correct
def print_check_comment_header():
    logger.warning("")
    logger.warning("Please fix following warning(s):")
    logger.warning("--------------------------------")


def check_documentations(procedures: list[SdlProcedure]) -> None:

    check_comment_header = CallOnce(print_check_comment_header)

    warning_header_printed = False

    # Check \param
    for proc in procedures:
        expected = len(proc.parameter)
        if expected == 1:
            if proc.parameter[0] == 'void':
                expected = 0
        count = proc.comment.count("\\param")
        if count != expected:
            # skip SDL_stdinc.h
            if proc.header != 'SDL_stdinc.h':
                # Warning mismatch \param and function prototype
                check_comment_header()
                logger.warning("  In file %s: function %s() has %d '\\param' but expected %d", proc.header, proc.name, count, expected)

        # Warning check \param uses the correct parameter name
        # skip SDL_stdinc.h
        if proc.header != 'SDL_stdinc.h':
            for n in proc.parameter_name:
                if n != "" and "\\param " + n not in proc.comment and "\\param[out] " + n not in proc.comment:
                    check_comment_header()
                    logger.warning("  In file %s: function %s() missing '\\param %s'", proc.header, proc.name, n)

    # Check \returns
    for proc in procedures:
        expected = 1
        if proc.retval == 'void':
            expected = 0

        count = proc.comment.count("\\returns")
        if count != expected:
            # skip SDL_stdinc.h
            if proc.header != 'SDL_stdinc.h':
                # Warning mismatch \param and function prototype
                check_comment_header()
                logger.warning("  In file %s: function %s() has %d '\\returns' but expected %d" % (proc.header, proc.name, count, expected))

    # Check \since
    for proc in procedures:
        expected = 1
        count = proc.comment.count("\\since")
        if count != expected:
            # skip SDL_stdinc.h
            if proc.header != 'SDL_stdinc.h':
                # Warning mismatch \param and function prototype
                check_comment_header()
                logger.warning("  In file %s: function %s() has %d '\\since' but expected %d" % (proc.header, proc.name, count, expected))


# Parse 'sdl_dynapi_procs_h' file to find existing functions
def find_existing_proc_names() -> list[str]:
    reg = re.compile(r'SDL_DYNAPI_PROC\([^,]*,([^,]*),.*\)')
    ret = []

    with SDL_DYNAPI_PROCS_H.open() as f:
        for line in f:
            match = reg.match(line)
            if not match:
                continue
            existing_func = match.group(1)
            ret.append(existing_func)
    return ret

# Get list of SDL headers
def get_header_list() -> list[Path]:
    ret = []

    for f in SDL_INCLUDE_DIR.iterdir():
        # Only *.h files
        if f.is_file() and f.suffix == ".h":
            ret.append(f)
        else:
            logger.debug("Skip %s", f)

    # Order headers for reproducible behavior
    ret.sort()

    return ret

# Write the new API in files: _procs.h _overrivides.h and .sym
def add_dyn_api(proc: SdlProcedure) -> None:
    decl_args: list[str] = []
    call_args = []
    for i, argtype in enumerate(proc.parameter):
        # Special case, void has no parameter name
        if argtype == "void":
            assert len(decl_args) == 0
            assert len(proc.parameter) == 1
            decl_args.append("void")
            continue

        # Var name: a, b, c, ...
        varname = chr(ord('a') + i)

        decl_args.append(argtype.replace("REWRITE_NAME", varname))
        if argtype != "...":
            call_args.append(varname)

    macro_args = (
        proc.retval,
        proc.name,
        "({})".format(",".join(decl_args)),
        "({})".format(",".join(call_args)),
        "" if proc.retval == "void" else "return",
    )

    # File: SDL_dynapi_procs.h
    #
    # Add at last
    # SDL_DYNAPI_PROC(SDL_EGLConfig,SDL_EGL_GetCurrentConfig,(void),(),return)
    with SDL_DYNAPI_PROCS_H.open("a", newline="") as f:
        if proc.variadic:
            f.write("#ifndef SDL_DYNAPI_PROC_NO_VARARGS\n")
        f.write(f"SDL_DYNAPI_PROC({','.join(macro_args)})\n")
        if proc.variadic:
            f.write("#endif\n")

    # File: SDL_dynapi_overrides.h
    #
    # Add at last
    # "#define SDL_DelayNS SDL_DelayNS_REAL
    f = open(SDL_DYNAPI_OVERRIDES_H, "a", newline="")
    f.write(f"#define {proc.name} {proc.name}_REAL\n")
    f.close()

    # File: SDL_dynapi.sym
    #
    # Add before "extra symbols go here" line
    with SDL_DYNAPI_SYM.open() as f:
        new_input = []
        for line in f:
            if "extra symbols go here" in line:
                new_input.append(f"    {proc.name};\n")
            new_input.append(line)

    with SDL_DYNAPI_SYM.open('w', newline='') as f:
        for line in new_input:
            f.write(line)


def main():
    parser = argparse.ArgumentParser()
    parser.set_defaults(loglevel=logging.INFO)
    parser.add_argument('--dump', nargs='?', default=None, const="sdl.json", metavar="JSON", help='output all SDL API into a .json file')
    parser.add_argument('--debug', action='store_const', const=logging.DEBUG, dest="loglevel", help='add debug traces')
    args = parser.parse_args()

    logging.basicConfig(level=args.loglevel, format='[%(levelname)s] %(message)s')

    # Get list of SDL headers
    sdl_list_includes = get_header_list()
    procedures = []
    for filename in sdl_list_includes:
        header_procedures = parse_header(filename)
        procedures.extend(header_procedures)

    # Parse 'sdl_dynapi_procs_h' file to find existing functions
    existing_proc_names = find_existing_proc_names()
    for procedure in procedures:
        if procedure.name not in existing_proc_names:
            logger.info("NEW %s", procedure.name)
            add_dyn_api(procedure)

    if args.dump:
        # Dump API into a json file
        full_API_json(path=Path(args.dump), procedures=procedures)

    # Check comment formatting
    check_documentations(procedures)


if __name__ == '__main__':
    raise SystemExit(main())
