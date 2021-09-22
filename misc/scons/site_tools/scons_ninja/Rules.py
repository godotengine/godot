# MIT License
#
# Copyright The SCons Foundation
#
# Permission is hereby granted, free of charge, to any person obtaining
# a copy of this software and associated documentation files (the
# "Software"), to deal in the Software without restriction, including
# without limitation the rights to use, copy, modify, merge, publish,
# distribute, sublicense, and/or sell copies of the Software, and to
# permit persons to whom the Software is furnished to do so, subject to
# the following conditions:
#
# The above copyright notice and this permission notice shall be included
# in all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY
# KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
# WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
# NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE
# LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
# OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
# WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

from Utils import get_outputs, get_rule, get_inputs, get_dependencies


def _install_action_function(_env, node):
    """Install files using the install or copy commands"""
    return {
        "outputs": get_outputs(node),
        "rule": get_rule(node, "INSTALL"),
        "inputs": get_inputs(node),
        "implicit": get_dependencies(node),
    }


def _mkdir_action_function(env, node):
    return {
        "outputs": get_outputs(node),
        "rule": get_rule(node, "GENERATED_CMD"),
        # implicit explicitly omitted, we translate these so they can be
        # used by anything that depends on these but commonly this is
        # hit with a node that will depend on all of the fake
        # srcnode's that SCons will never give us a rule for leading
        # to an invalid ninja file.
        "variables": {
            # On Windows mkdir "-p" is always on
            "cmd": "mkdir {args}".format(
                args=" ".join(get_outputs(node)) + " & exit /b 0"
                if env["PLATFORM"] == "win32"
                else "-p " + " ".join(get_outputs(node)),
            ),
        },
    }


def _copy_action_function(env, node):
    return {
        "outputs": get_outputs(node),
        "inputs": get_inputs(node),
        "rule": get_rule(node, "CMD"),
        "variables": {
            "cmd": "$COPY",
        },
    }


def _lib_symlink_action_function(_env, node):
    """Create shared object symlinks if any need to be created"""
    symlinks = node.check_attributes("shliblinks")

    if not symlinks or symlinks is None:
        return None

    outputs = [link.get_dir().rel_path(linktgt) for link, linktgt in symlinks]
    inputs = [link.get_path() for link, _ in symlinks]

    return {
        "outputs": outputs,
        "inputs": inputs,
        "rule": get_rule(node, "SYMLINK"),
        "implicit": get_dependencies(node),
    }
