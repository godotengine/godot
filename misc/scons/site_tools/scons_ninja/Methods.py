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

import os
import shlex
import textwrap

import SCons

from scons_ninja import NINJA_CUSTOM_HANDLERS, NINJA_RULES, NINJA_POOLS
import Globals
from Globals import __NINJA_RULE_MAPPING
from Utils import (
    get_targets_sources,
    get_dependencies,
    get_order_only,
    get_outputs,
    get_inputs,
    get_rule,
    get_path,
    generate_command,
    get_command_env,
    get_comstr,
)


def register_custom_handler(env, name, handler):
    """Register a custom handler for SCons function actions."""
    env[NINJA_CUSTOM_HANDLERS][name] = handler


def register_custom_rule_mapping(env, pre_subst_string, rule):
    """Register a function to call for a given rule."""
    Globals.__NINJA_RULE_MAPPING[pre_subst_string] = rule


def register_custom_rule(
    env,
    rule,
    command,
    description="",
    deps=None,
    pool=None,
    use_depfile=False,
    use_response_file=False,
    response_file_content="$rspc",
):
    """Allows specification of Ninja rules from inside SCons files."""
    rule_obj = {
        "command": command,
        "description": description if description else "{} $out".format(rule),
    }

    if use_depfile:
        rule_obj["depfile"] = os.path.join(get_path(env["NINJA_DIR"]), "$out.depfile")

    if deps is not None:
        rule_obj["deps"] = deps

    if pool is not None:
        rule_obj["pool"] = pool

    if use_response_file:
        rule_obj["rspfile"] = "$out.rsp"
        rule_obj["rspfile_content"] = response_file_content

    env[NINJA_RULES][rule] = rule_obj


def register_custom_pool(env, pool, size):
    """Allows the creation of custom Ninja pools"""
    env[NINJA_POOLS][pool] = size


def set_build_node_callback(env, node, callback):
    if not node.is_conftest():
        node.attributes.ninja_build_callback = callback


def get_generic_shell_command(env, node, action, targets, sources, executor=None):
    return (
        "GENERATED_CMD",
        {
            "cmd": generate_command(env, node, action, targets, sources, executor=executor),
            "env": get_command_env(env),
        },
        # Since this function is a rule mapping provider, it must return a list of dependencies,
        # and usually this would be the path to a tool, such as a compiler, used for this rule.
        # However this function is to generic to be able to reliably extract such deps
        # from the command, so we return a placeholder empty list. It should be noted that
        # generally this function will not be used solely and is more like a template to generate
        # the basics for a custom provider which may have more specific options for a provider
        # function for a custom NinjaRuleMapping.
        [],
    )


def CheckNinjaCompdbExpand(env, context):
    """Configure check testing if ninja's compdb can expand response files"""

    # TODO: When would this be false?
    context.Message("Checking if ninja compdb can expand response files... ")
    ret, output = context.TryAction(
        action="ninja -f $SOURCE -t compdb -x CMD_RSP > $TARGET",
        extension=".ninja",
        text=textwrap.dedent(
            """
            rule CMD_RSP
              command = $cmd @$out.rsp > fake_output.txt
              description = Building $out
              rspfile = $out.rsp
              rspfile_content = $rspc
            build fake_output.txt: CMD_RSP fake_input.txt
              cmd = echo
              pool = console
              rspc = "test"
            """
        ),
    )
    result = "@fake_output.txt.rsp" not in output
    context.Result(result)
    return result


def get_command(env, node, action):  # pylint: disable=too-many-branches
    """Get the command to execute for node."""
    if node.env:
        sub_env = node.env
    else:
        sub_env = env
    executor = node.get_executor()
    tlist, slist = get_targets_sources(node)

    # Generate a real CommandAction
    if isinstance(action, SCons.Action.CommandGeneratorAction):
        # pylint: disable=protected-access
        action = action._generate(tlist, slist, sub_env, 0, executor=executor)

    variables = {}

    comstr = get_comstr(sub_env, action, tlist, slist)
    if not comstr:
        return None

    provider = __NINJA_RULE_MAPPING.get(comstr, get_generic_shell_command)
    rule, variables, provider_deps = provider(sub_env, node, action, tlist, slist, executor=executor)
    if node.get_env().get("NINJA_FORCE_SCONS_BUILD"):
        rule = "TEMPLATE"

    # Get the dependencies for all targets
    implicit = list({dep for tgt in tlist for dep in get_dependencies(tgt)})

    # Now add in the other dependencies related to the command,
    # e.g. the compiler binary. The ninja rule can be user provided so
    # we must do some validation to resolve the dependency path for ninja.
    for provider_dep in provider_deps:

        provider_dep = sub_env.subst(provider_dep)
        if not provider_dep:
            continue

        # If the tool is a node, then SCons will resolve the path later, if its not
        # a node then we assume it generated from build and make sure it is existing.
        if isinstance(provider_dep, SCons.Node.Node) or os.path.exists(provider_dep):
            implicit.append(provider_dep)
            continue

        # in some case the tool could be in the local directory and be supplied without the ext
        # such as in windows, so append the executable suffix and check.
        prog_suffix = sub_env.get("PROGSUFFIX", "")
        provider_dep_ext = provider_dep if provider_dep.endswith(prog_suffix) else provider_dep + prog_suffix
        if os.path.exists(provider_dep_ext):
            implicit.append(provider_dep_ext)
            continue

        # Many commands will assume the binary is in the path, so
        # we accept this as a possible input from a given command.

        provider_dep_abspath = sub_env.WhereIs(provider_dep) or sub_env.WhereIs(provider_dep, path=os.environ["PATH"])
        if provider_dep_abspath:
            implicit.append(provider_dep_abspath)
            continue

        # Possibly these could be ignore and the build would still work, however it may not always
        # rebuild correctly, so we hard stop, and force the user to fix the issue with the provided
        # ninja rule.
        raise Exception("Could not resolve path for %s dependency on node '%s'" % (provider_dep, node))

    ninja_build = {
        "order_only": get_order_only(node),
        "outputs": get_outputs(node),
        "inputs": get_inputs(node),
        "implicit": implicit,
        "rule": get_rule(node, rule),
        "variables": variables,
    }

    # Don't use sub_env here because we require that NINJA_POOL be set
    # on a per-builder call basis to prevent accidental strange
    # behavior like env['NINJA_POOL'] = 'console' and sub_env can be
    # the global Environment object if node.env is None.
    # Example:
    #
    # Allowed:
    #
    #     env.Command("ls", NINJA_POOL="ls_pool")
    #
    # Not allowed and ignored:
    #
    #     env["NINJA_POOL"] = "ls_pool"
    #     env.Command("ls")
    #
    # TODO: Why not alloe env['NINJA_POOL'] ? (bdbaddog)
    if node.env and node.env.get("NINJA_POOL", None) is not None:
        ninja_build["pool"] = node.env["NINJA_POOL"]

    return ninja_build


def gen_get_response_file_command(env, rule, tool, tool_is_dynamic=False, custom_env={}):
    """Generate a response file command provider for rule name."""

    # If win32 using the environment with a response file command will cause
    # ninja to fail to create the response file. Additionally since these rules
    # generally are not piping through cmd.exe /c any environment variables will
    # make CreateProcess fail to start.
    #
    # On POSIX we can still set environment variables even for compile
    # commands so we do so.
    use_command_env = not env["PLATFORM"] == "win32"
    if "$" in tool:
        tool_is_dynamic = True

    def get_response_file_command(env, node, action, targets, sources, executor=None):
        if hasattr(action, "process"):
            cmd_list, _, _ = action.process(targets, sources, env, executor=executor)
            cmd_list = [str(c).replace("$", "$$") for c in cmd_list[0]]
        else:
            command = generate_command(env, node, action, targets, sources, executor=executor)
            cmd_list = shlex.split(command)

        if tool_is_dynamic:
            tool_command = env.subst(tool, target=targets, source=sources, executor=executor)
        else:
            tool_command = tool

        try:
            # Add 1 so we always keep the actual tool inside of cmd
            tool_idx = cmd_list.index(tool_command) + 1
        except ValueError:
            raise Exception(
                "Could not find tool {} in {} generated from {}".format(
                    tool, cmd_list, get_comstr(env, action, targets, sources)
                )
            )

        cmd, rsp_content = cmd_list[:tool_idx], cmd_list[tool_idx:]
        rsp_content = ['"' + rsp_content_item + '"' for rsp_content_item in rsp_content]
        rsp_content = " ".join(rsp_content)

        variables = {"rspc": rsp_content, rule: cmd}
        if use_command_env:
            variables["env"] = get_command_env(env)

            for key, value in custom_env.items():
                variables["env"] += (
                    env.subst("export %s=%s;" % (key, value), target=targets, source=sources, executor=executor) + " "
                )

        if node.get_env().get("NINJA_FORCE_SCONS_BUILD"):
            ret_rule = "TEMPLATE"
        else:
            ret_rule = rule

        return ret_rule, variables, [tool_command]

    return get_response_file_command
