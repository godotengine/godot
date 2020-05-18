# Copyright 2015 MongoDB Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
import SCons
import itertools

# Implements the ability for SCons to emit a compilation database for the MongoDB project. See
# http://clang.llvm.org/docs/JSONCompilationDatabase.html for details on what a compilation
# database is, and why you might want one. The only user visible entry point here is
# 'env.CompilationDatabase'. This method takes an optional 'target' to name the file that
# should hold the compilation database, otherwise, the file defaults to compile_commands.json,
# which is the name that most clang tools search for by default.

# TODO: Is there a better way to do this than this global? Right now this exists so that the
# emitter we add can record all of the things it emits, so that the scanner for the top level
# compilation database can access the complete list, and also so that the writer has easy
# access to write all of the files. But it seems clunky. How can the emitter and the scanner
# communicate more gracefully?
__COMPILATION_DB_ENTRIES = []

# We make no effort to avoid rebuilding the entries. Someday, perhaps we could and even
# integrate with the cache, but there doesn't seem to be much call for it.
class __CompilationDbNode(SCons.Node.Python.Value):
    def __init__(self, value):
        SCons.Node.Python.Value.__init__(self, value)
        self.Decider(changed_since_last_build_node)


def changed_since_last_build_node(child, target, prev_ni, node):
    """ Dummy decider to force always building"""
    return True


def makeEmitCompilationDbEntry(comstr):
    """
    Effectively this creates a lambda function to capture:
    * command line
    * source
    * target
    :param comstr: unevaluated command line
    :return: an emitter which has captured the above
    """
    user_action = SCons.Action.Action(comstr)

    def EmitCompilationDbEntry(target, source, env):
        """
        This emitter will be added to each c/c++ object build to capture the info needed
        for clang tools
        :param target: target node(s)
        :param source: source node(s)
        :param env: Environment for use building this node
        :return: target(s), source(s)
        """

        dbtarget = __CompilationDbNode(source)

        entry = env.__COMPILATIONDB_Entry(
            target=dbtarget,
            source=[],
            __COMPILATIONDB_UTARGET=target,
            __COMPILATIONDB_USOURCE=source,
            __COMPILATIONDB_UACTION=user_action,
            __COMPILATIONDB_ENV=env,
        )

        # TODO: Technically, these next two lines should not be required: it should be fine to
        # cache the entries. However, they don't seem to update properly. Since they are quick
        # to re-generate disable caching and sidestep this problem.
        env.AlwaysBuild(entry)
        env.NoCache(entry)

        __COMPILATION_DB_ENTRIES.append(dbtarget)

        return target, source

    return EmitCompilationDbEntry


def CompilationDbEntryAction(target, source, env, **kw):
    """
    Create a dictionary with evaluated command line, target, source
    and store that info as an attribute on the target
    (Which has been stored in __COMPILATION_DB_ENTRIES array
    :param target: target node(s)
    :param source: source node(s)
    :param env: Environment for use building this node
    :param kw:
    :return: None
    """

    command = env["__COMPILATIONDB_UACTION"].strfunction(
        target=env["__COMPILATIONDB_UTARGET"], source=env["__COMPILATIONDB_USOURCE"], env=env["__COMPILATIONDB_ENV"],
    )

    entry = {
        "directory": env.Dir("#").abspath,
        "command": command,
        "file": str(env["__COMPILATIONDB_USOURCE"][0]),
    }

    target[0].write(entry)


def WriteCompilationDb(target, source, env):
    entries = []

    for s in __COMPILATION_DB_ENTRIES:
        entries.append(s.read())

    with open(str(target[0]), "w") as target_file:
        json.dump(entries, target_file, sort_keys=True, indent=4, separators=(",", ": "))


def ScanCompilationDb(node, env, path):
    return __COMPILATION_DB_ENTRIES


def generate(env, **kwargs):

    static_obj, shared_obj = SCons.Tool.createObjBuilders(env)

    env["COMPILATIONDB_COMSTR"] = kwargs.get("COMPILATIONDB_COMSTR", "Building compilation database $TARGET")

    components_by_suffix = itertools.chain(
        itertools.product(
            env["CPPSUFFIXES"],
            [
                (static_obj, SCons.Defaults.StaticObjectEmitter, "$CXXCOM"),
                (shared_obj, SCons.Defaults.SharedObjectEmitter, "$SHCXXCOM"),
            ],
        ),
    )

    for entry in components_by_suffix:
        suffix = entry[0]
        builder, base_emitter, command = entry[1]

        # Ensure we have a valid entry
        # used to auto ignore header files
        if suffix in builder.emitter:
            emitter = builder.emitter[suffix]
            builder.emitter[suffix] = SCons.Builder.ListEmitter([emitter, makeEmitCompilationDbEntry(command),])

    env["BUILDERS"]["__COMPILATIONDB_Entry"] = SCons.Builder.Builder(
        action=SCons.Action.Action(CompilationDbEntryAction, None),
    )

    env["BUILDERS"]["__COMPILATIONDB_Database"] = SCons.Builder.Builder(
        action=SCons.Action.Action(WriteCompilationDb, "$COMPILATIONDB_COMSTR"),
        target_scanner=SCons.Scanner.Scanner(function=ScanCompilationDb, node_class=None),
    )

    def CompilationDatabase(env, target):
        result = env.__COMPILATIONDB_Database(target=target, source=[])

        env.AlwaysBuild(result)
        env.NoCache(result)

        return result

    env.AddMethod(CompilationDatabase, "CompilationDatabase")


def exists(env):
    return True
