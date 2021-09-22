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
import shutil
from os.path import join as joinpath

import SCons
from SCons.Action import get_default_ENV, _string_from_cmd_list
from SCons.Script import AddOption
from SCons.Util import is_List, flatten_sequence

import Globals


def ninja_add_command_line_options():
    """
    Add additional command line arguments to SCons specific to the ninja tool
    """
    AddOption(
        "--disable-execute-ninja",
        dest="disable_execute_ninja",
        metavar="BOOL",
        action="store_true",
        default=False,
        help="Disable automatically running ninja after scons",
    )

    AddOption(
        "--disable-ninja",
        dest="disable_ninja",
        metavar="BOOL",
        action="store_true",
        default=False,
        help="Disable ninja generation and build with scons even if tool is loaded. "
        + "Also used by ninja to build targets which only scons can build.",
    )


def is_valid_dependent_node(node):
    """
    Return True if node is not an alias or is an alias that has children

    This prevents us from making phony targets that depend on other
    phony targets that will never have an associated ninja build
    target.

    We also have to specify that it's an alias when doing the builder
    check because some nodes (like src files) won't have builders but
    are valid implicit dependencies.
    """
    if isinstance(node, SCons.Node.Alias.Alias):
        return node.children()

    return not node.get_env().get("NINJA_SKIP")


def alias_to_ninja_build(node):
    """Convert an Alias node into a Ninja phony target"""
    return {
        "outputs": get_outputs(node),
        "rule": "phony",
        "implicit": [get_path(src_file(n)) for n in node.children() if is_valid_dependent_node(n)],
    }


def check_invalid_ninja_node(node):
    return not isinstance(node, (SCons.Node.FS.Base, SCons.Node.Alias.Alias))


def filter_ninja_nodes(node_list):
    ninja_nodes = []
    for node in node_list:
        if isinstance(node, (SCons.Node.FS.Base, SCons.Node.Alias.Alias)) and not node.get_env().get("NINJA_SKIP"):
            ninja_nodes.append(node)
        else:
            continue
    return ninja_nodes


def get_input_nodes(node):
    if node.get_executor() is not None:
        inputs = node.get_executor().get_all_sources()
    else:
        inputs = node.sources
    return inputs


def invalid_ninja_nodes(node, targets):
    result = False
    for node_list in [node.prerequisites, get_input_nodes(node), node.children(), targets]:
        if node_list:
            result = result or any([check_invalid_ninja_node(node) for node in node_list])
    return result


def get_order_only(node):
    """Return a list of order only dependencies for node."""
    if node.prerequisites is None:
        return []
    return [get_path(src_file(prereq)) for prereq in filter_ninja_nodes(node.prerequisites)]


def get_dependencies(node, skip_sources=False):
    """Return a list of dependencies for node."""
    if skip_sources:
        return [get_path(src_file(child)) for child in filter_ninja_nodes(node.children()) if child not in node.sources]
    return [get_path(src_file(child)) for child in filter_ninja_nodes(node.children())]


def get_inputs(node):
    """Collect the Ninja inputs for node."""
    return [get_path(src_file(o)) for o in filter_ninja_nodes(get_input_nodes(node))]


def get_outputs(node):
    """Collect the Ninja outputs for node."""
    executor = node.get_executor()
    if executor is not None:
        outputs = executor.get_all_targets()
    else:
        if hasattr(node, "target_peers"):
            outputs = node.target_peers
        else:
            outputs = [node]

    outputs = [get_path(o) for o in filter_ninja_nodes(outputs)]

    return outputs


def get_targets_sources(node):
    executor = node.get_executor()
    if executor is not None:
        tlist = executor.get_all_targets()
        slist = executor.get_all_sources()
    else:
        if hasattr(node, "target_peers"):
            tlist = node.target_peers
        else:
            tlist = [node]
        slist = node.sources

    # Retrieve the repository file for all sources
    slist = [rfile(s) for s in slist]
    return tlist, slist


def get_path(node):
    """
    Return a fake path if necessary.

    As an example Aliases use this as their target name in Ninja.
    """
    if hasattr(node, "get_path"):
        return node.get_path()
    return str(node)


def rfile(node):
    """
    Return the repository file for node if it has one. Otherwise return node
    """
    if hasattr(node, "rfile"):
        return node.rfile()
    return node


def src_file(node):
    """Returns the src code file if it exists."""
    if hasattr(node, "srcnode"):
        src = node.srcnode()
        if src.stat() is not None:
            return src
    return get_path(node)


def get_rule(node, rule):
    tlist, slist = get_targets_sources(node)
    if invalid_ninja_nodes(node, tlist):
        return "TEMPLATE"
    else:
        return rule


def generate_depfile(env, node, dependencies):
    """
    Ninja tool function for writing a depfile. The depfile should include
    the node path followed by all the dependent files in a makefile format.

    dependencies arg can be a list or a subst generator which returns a list.
    """

    depfile = os.path.join(get_path(env["NINJA_DIR"]), str(node) + ".depfile")

    # subst_list will take in either a raw list or a subst callable which generates
    # a list, and return a list of CmdStringHolders which can be converted into raw strings.
    # If a raw list was passed in, then scons_list will make a list of lists from the original
    # values and even subst items in the list if they are substitutable. Flatten will flatten
    # the list in that case, to ensure for either input we have a list of CmdStringHolders.
    deps_list = env.Flatten(env.subst_list(dependencies))

    # Now that we have the deps in a list as CmdStringHolders, we can convert them into raw strings
    # and make sure to escape the strings to handle spaces in paths. We also will sort the result
    # keep the order of the list consistent.
    escaped_depends = sorted([dep.escape(env.get("ESCAPE", lambda x: x)) for dep in deps_list])
    depfile_contents = str(node) + ": " + " ".join(escaped_depends)

    need_rewrite = False
    try:
        with open(depfile, "r") as f:
            need_rewrite = f.read() != depfile_contents
    except FileNotFoundError:
        need_rewrite = True

    if need_rewrite:
        os.makedirs(os.path.dirname(depfile) or ".", exist_ok=True)
        with open(depfile, "w") as f:
            f.write(depfile_contents)


def ninja_noop(*_args, **_kwargs):
    """
    A general purpose no-op function.

    There are many things that happen in SCons that we don't need and
    also don't return anything. We use this to disable those functions
    instead of creating multiple definitions of the same thing.
    """
    return None


def get_command_env(env):
    """
    Return a string that sets the environment for any environment variables that
    differ between the OS environment and the SCons command ENV.

    It will be compatible with the default shell of the operating system.
    """
    try:
        return env["NINJA_ENV_VAR_CACHE"]
    except KeyError:
        pass

    # Scan the ENV looking for any keys which do not exist in
    # os.environ or differ from it. We assume if it's a new or
    # differing key from the process environment then it's
    # important to pass down to commands in the Ninja file.
    ENV = get_default_ENV(env)
    scons_specified_env = {
        key: value
        for key, value in ENV.items()
        # TODO: Remove this filter, unless there's a good reason to keep. SCons's behavior shouldn't depend on shell's.
        if key not in os.environ or os.environ.get(key, None) != value
    }

    windows = env["PLATFORM"] == "win32"
    command_env = ""
    for key, value in scons_specified_env.items():
        # Ensure that the ENV values are all strings:
        if is_List(value):
            # If the value is a list, then we assume it is a
            # path list, because that's a pretty common list-like
            # value to stick in an environment variable:
            value = flatten_sequence(value)
            value = joinpath(map(str, value))
        else:
            # If it isn't a string or a list, then we just coerce
            # it to a string, which is the proper way to handle
            # Dir and File instances and will produce something
            # reasonable for just about everything else:
            value = str(value)

        if windows:
            command_env += "set '{}={}' && ".format(key, value)
        else:
            # We address here *only* the specific case that a user might have
            # an environment variable which somehow gets included and has
            # spaces in the value. These are escapes that Ninja handles. This
            # doesn't make builds on paths with spaces (Ninja and SCons issues)
            # nor expanding response file paths with spaces (Ninja issue) work.
            value = value.replace(r" ", r"$ ")
            command_env += "export {}='{}';".format(key, value)

    env["NINJA_ENV_VAR_CACHE"] = command_env
    return command_env


def get_comstr(env, action, targets, sources):
    """Get the un-substituted string for action."""
    # Despite being having "list" in it's name this member is not
    # actually a list. It's the pre-subst'd string of the command. We
    # use it to determine if the command we're about to generate needs
    # to use a custom Ninja rule. By default this redirects CC, CXX,
    # AR, SHLINK, and LINK commands to their respective rules but the
    # user can inject custom Ninja rules and tie them to commands by
    # using their pre-subst'd string.
    if hasattr(action, "process"):
        return action.cmd_list

    return action.genstring(targets, sources, env)


def generate_command(env, node, action, targets, sources, executor=None):
    # Actions like CommandAction have a method called process that is
    # used by SCons to generate the cmd_line they need to run. So
    # check if it's a thing like CommandAction and call it if we can.
    if hasattr(action, "process"):
        cmd_list, _, _ = action.process(targets, sources, env, executor=executor)
        cmd = _string_from_cmd_list(cmd_list[0])
    else:
        # Anything else works with genstring, this is most commonly hit by
        # ListActions which essentially call process on all of their
        # commands and concatenate it for us.
        genstring = action.genstring(targets, sources, env)
        if executor is not None:
            cmd = env.subst(genstring, executor=executor)
        else:
            cmd = env.subst(genstring, targets, sources)

        cmd = cmd.replace("\n", " && ").strip()
        if cmd.endswith("&&"):
            cmd = cmd[0:-2].strip()

    # Escape dollars as necessary
    return cmd.replace("$", "$$")


def ninja_csig(original):
    """Return a dummy csig"""

    def wrapper(self):
        if isinstance(self, SCons.Node.Node) and self.is_sconscript():
            return original(self)
        return "dummy_ninja_csig"

    return wrapper


def ninja_contents(original):
    """Return a dummy content without doing IO"""

    def wrapper(self):
        if isinstance(self, SCons.Node.Node) and (self.is_sconscript() or self.is_conftest()):
            return original(self)
        return bytes("dummy_ninja_contents", encoding="utf-8")

    return wrapper


def ninja_stat(_self, path):
    """
    Eternally memoized stat call.

    SCons is very aggressive about clearing out cached values. For our
    purposes everything should only ever call stat once since we're
    running in a no_exec build the file system state should not
    change. For these reasons we patch SCons.Node.FS.LocalFS.stat to
    use our eternal memoized dictionary.
    """

    try:
        return Globals.NINJA_STAT_MEMO[path]
    except KeyError:
        try:
            result = os.stat(path)
        except os.error:
            result = None

        Globals.NINJA_STAT_MEMO[path] = result
        return result


def ninja_whereis(thing, *_args, **_kwargs):
    """Replace env.WhereIs with a much faster version"""

    # Optimize for success, this gets called significantly more often
    # when the value is already memoized than when it's not.
    try:
        return Globals.NINJA_WHEREIS_MEMO[thing]
    except KeyError:
        # TODO: Fix this to respect env['ENV']['PATH']... WPD
        # We do not honor any env['ENV'] or env[*] variables in the
        # generated ninja file. Ninja passes your raw shell environment
        # down to it's subprocess so the only sane option is to do the
        # same during generation. At some point, if and when we try to
        # upstream this, I'm sure a sticking point will be respecting
        # env['ENV'] variables and such but it's actually quite
        # complicated. I have a naive version but making it always work
        # with shell quoting is nigh impossible. So I've decided to
        # cross that bridge when it's absolutely required.
        path = shutil.which(thing)
        Globals.NINJA_WHEREIS_MEMO[thing] = path
        return path


def ninja_print_conf_log(s, target, source, env):
    """Command line print only for conftest to generate a correct conf log."""
    if target and target[0].is_conftest():
        action = SCons.Action._ActionAction()
        action.print_cmd_line(s, target, source, env)
