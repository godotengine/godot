import os
import subprocess
from git import Repo
import SCons
from SCons.Script import *

black_bin = None


def run_changeset_black(env, target, source):
    subprocess.getoutput("black -l 120 " + " ".join([s.path for s in source]) + " > " + target[0].path)
    # because we have modified the files during a build, we should clear the nodes.
    # this prevent unnecessary rebuids.
    for s in source:
        s.del_binfo()
        s.clear_memoized_values()
        s.ninfo = s.new_ninfo()
        s.executor_cleanup()


def exists(env):
    global black_bin

    black_bin = env.WhereIs("black")
    if not black_bin:
        black_bin = env.WhereIs("black", os.environ["PATH"])

    return black_bin


def generate(env):

    if not exists(env):
        print("black not found, not running black formatter.")
        return

    env["BLACK_OUTPUT"] = env.get("BLACK_OUTPUT", ".black.out")
    repo = Repo(env.Dir("#").abspath)

    files = [
        file
        for file in repo.git.diff("HEAD~1..HEAD", name_only=True).split("\n") + repo.untracked_files
        if (
            not file.startswith("thirdparty")
            and (file.endswith("SConstruct") or file.endswith("SCsub") or file.endswith(".py"))
        )
    ]

    if files:
        changeset_black = env.Command(
            target="$BLACK_OUTPUT",
            source=files,
            action=SCons.Action.Action(run_changeset_black, cmdstr="Formatting changeset with black..."),
        )

        # these next few lines for scons to always run the black formatter task
        Default(changeset_black)
        for target in COMMAND_LINE_TARGETS:
            if target == "run-black":
                continue
            env.Depends(target, changeset_black)

        env.Alias("run-black", changeset_black)
