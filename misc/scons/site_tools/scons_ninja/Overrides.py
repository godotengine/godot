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
"""
This module is to hold logic which overrides default SCons behaviors to enable
ninja file generation
"""
import SCons


def ninja_hack_linkcom(env):
    # TODO: change LINKCOM and SHLINKCOM to handle embedding manifest exe checks
    # without relying on the SCons hacks that SCons uses by default.
    if env["PLATFORM"] == "win32":
        from SCons.Tool.mslink import compositeLinkAction

        if env.get("LINKCOM", None) == compositeLinkAction:
            env[
                "LINKCOM"
            ] = '${TEMPFILE("$LINK $LINKFLAGS /OUT:$TARGET.windows $_LIBDIRFLAGS $_LIBFLAGS $_PDB $SOURCES.windows", "$LINKCOMSTR")}'
            env[
                "SHLINKCOM"
            ] = '${TEMPFILE("$SHLINK $SHLINKFLAGS $_SHLINK_TARGETS $_LIBDIRFLAGS $_LIBFLAGS $_PDB $_SHLINK_SOURCES", "$SHLINKCOMSTR")}'


def ninja_hack_arcom(env):
    """
    Force ARCOM so use 's' flag on ar instead of separately running ranlib
    """
    if env["PLATFORM"] != "win32" and env.get("RANLIBCOM"):
        # There is no way to translate the ranlib list action into
        # Ninja so add the s flag and disable ranlib.
        #
        # This is equivalent to Meson.
        # https://github.com/mesonbuild/meson/blob/master/mesonbuild/linkers.py#L143
        old_arflags = str(env["ARFLAGS"])
        if "s" not in old_arflags:
            old_arflags += "s"

        env["ARFLAGS"] = SCons.Util.CLVar([old_arflags])

        # Disable running ranlib, since we added 's' above
        env["RANLIBCOM"] = ""


class NinjaNoResponseFiles(SCons.Platform.TempFileMunge):
    """Overwrite the __call__ method of SCons' TempFileMunge to not delete."""

    def __call__(self, target, source, env, for_signature):
        return self.cmd

    def _print_cmd_str(*_args, **_kwargs):
        """Disable this method"""
        pass


def ninja_always_serial(self, num, taskmaster):
    """Replacement for SCons.Job.Jobs constructor which always uses the Serial Job class."""
    # We still set self.num_jobs to num even though it's a lie. The
    # only consumer of this attribute is the Parallel Job class AND
    # the Main.py function which instantiates a Jobs class. It checks
    # if Jobs.num_jobs is equal to options.num_jobs, so if the user
    # provides -j12 but we set self.num_jobs = 1 they get an incorrect
    # warning about this version of Python not supporting parallel
    # builds. So here we lie so the Main.py will not give a false
    # warning to users.
    self.num_jobs = num
    self.job = SCons.Job.Serial(taskmaster)


# pylint: disable=too-few-public-methods
class AlwaysExecAction(SCons.Action.FunctionAction):
    """Override FunctionAction.__call__ to always execute."""

    def __call__(self, *args, **kwargs):
        kwargs["execute"] = 1
        return super().__call__(*args, **kwargs)
