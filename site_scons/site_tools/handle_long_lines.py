"""
This is a modified copy of SCons's TempFileMunge.
It will accept TEMPFILE_ARG_COUNT which is the number of command line arguments which
are excepted from being included in the generated response file if the comand line is too
long for the OS's shell
"""

import atexit
import contextlib
import os
import sys
import tempfile

import SCons.Action

MY_TEMPFILE_DEFAULT_ENCODING = "utf-8"


class MyTempFileMunge:
    """Convert long command lines to use a temporary file.

    You can set an Environment variable (usually ``TEMPFILE``) to this,
    then call it with a string argument, and it will perform temporary
    file substitution on it.  This is used to circumvent limitations on
    the length of command lines. Example::

        env["TEMPFILE"] = TempFileMunge
        env["LINKCOM"] = "${TEMPFILE('$LINK $TARGET $SOURCES', '$LINKCOMSTR')}"

    By default, the name of the temporary file used begins with a
    prefix of '@'.  This may be configured for other tool chains by
    setting the ``TEMPFILEPREFIX`` variable. Example::

        env["TEMPFILEPREFIX"] = '-@'        # diab compiler
        env["TEMPFILEPREFIX"] = '-via'      # arm tool chain
        env["TEMPFILEPREFIX"] = ''          # (the empty string) PC Lint

    You can configure the extension of the temporary file through the
    ``TEMPFILESUFFIX`` variable, which defaults to '.lnk' (see comments
    in the code below). Example::

        env["TEMPFILESUFFIX"] = '.lnt'   # PC Lint

    Entries in the temporary file are separated by the value of the
    ``TEMPFILEARGJOIN`` variable, which defaults to an OS-appropriate value.

    A default argument escape function is ``SCons.Subst.quote_spaces``.
    If you need to apply extra operations on a command argument before
    writing to a temporary file(fix Windows slashes, normalize paths, etc.),
    please set `TEMPFILEARGESCFUNC` variable to a custom function. Example::

        import sys
        import re
        from SCons.Subst import quote_spaces

        WINPATHSEP_RE = re.compile(r"\\([^\"'\\]|$)")


        def tempfile_arg_esc_func(arg):
            arg = quote_spaces(arg)
            if sys.platform != "win32":
                return arg
            # GCC requires double Windows slashes, let's use UNIX separator
            return WINPATHSEP_RE.sub(r"/\1", arg)


        env["TEMPFILEARGESCFUNC"] = tempfile_arg_esc_func

    """

    def __init__(self, cmd, cmdstr=None) -> None:
        self.cmd = cmd
        self.cmdstr = cmdstr

    def __call__(self, target, source, env, for_signature):
        if for_signature:
            # If we're being called for signature calculation, it's
            # because we're being called by the string expansion in
            # Subst.py, which has the logic to strip any $( $) that
            # may be in the command line we squirreled away.  So we
            # just return the raw command line and let the upper
            # string substitution layers do their thing.
            return self.cmd

        # Now we're actually being called because someone is actually
        # going to try to execute the command, so we have to do our
        # own expansion.
        cmd = env.subst_list(self.cmd, SCons.Subst.SUBST_CMD, target, source)[0]
        try:
            maxline = int(env.subst("$MAXLINELENGTH"))
        except ValueError:
            maxline = 2048

        length = 0
        for c in cmd:
            length += len(c)
        length += len(cmd) - 1
        if length <= maxline:
            return self.cmd

        # Check if we already created the temporary file for this target
        # It should have been previously done by Action.strfunction() call
        if SCons.Util.is_List(target):
            node = target[0]
        else:
            node = target

        cmdlist = None

        if SCons.Util.is_List(self.cmd):
            cmdlist_key = tuple(self.cmd)
        else:
            cmdlist_key = self.cmd

        if node and hasattr(node.attributes, "tempfile_cmdlist"):
            cmdlist = node.attributes.tempfile_cmdlist.get(cmdlist_key, None)
        if cmdlist is not None:
            return cmdlist

        # try encoding the tempfile data before creating the file -
        # avoid orphaned files
        tempfile_esc_func = env.get("TEMPFILEARGESCFUNC", SCons.Subst.quote_spaces)
        tempfile_arg_count = env.get("TEMPFILE_ARG_COUNT", 1)
        args = [tempfile_esc_func(arg) for arg in cmd[tempfile_arg_count:]]
        join_char = env.get("TEMPFILEARGJOIN", " ")
        contents = join_char.join(args) + "\n"
        encoding = env.get("TEMPFILEENCODING", MY_TEMPFILE_DEFAULT_ENCODING)
        try:
            tempfile_contents = bytes(contents, encoding=encoding)
        except (UnicodeError, LookupError, TypeError):
            exc_type, exc_value, _ = sys.exc_info()
            if "TEMPFILEENCODING" in env:
                encoding_msg = "env['TEMPFILEENCODING']"
            else:
                encoding_msg = "default"
            err_msg = f"tempfile encoding error: [{exc_type.__name__}] {exc_value!s}"
            err_msg += f"\n  {type(self).__name__} encoding: {encoding_msg} = {encoding!r}"
            raise SCons.Errors.UserError(err_msg)

        # Default to the .lnk suffix for the benefit of the Phar Lap
        # linkloc linker, which likes to append an .lnk suffix if
        # none is given.
        if "TEMPFILESUFFIX" in env:
            suffix = env.subst("$TEMPFILESUFFIX")
        else:
            suffix = ".lnk"

        if "TEMPFILEDIR" in env:
            tempfile_dir = env.subst("$TEMPFILEDIR")
            os.makedirs(tempfile_dir, exist_ok=True)
        else:
            tempfile_dir = None

        fd, tmp = tempfile.mkstemp(suffix, dir=tempfile_dir)
        try:
            os.write(fd, tempfile_contents)
        finally:
            os.close(fd)
        native_tmp = SCons.Util.get_native_path(tmp)

        # arrange for cleanup on exit:

        def tmpfile_cleanup(file) -> None:
            with contextlib.suppress(FileNotFoundError):
                os.remove(file)

        atexit.register(tmpfile_cleanup, tmp)

        # XXX Using the SCons.Action.print_actions value directly
        # like this is bogus, but expedient.  This class should
        # really be rewritten as an Action that defines the
        # __call__() and strfunction() methods and lets the
        # normal action-execution logic handle whether or not to
        # print/execute the action.  The problem, though, is all
        # of that is decided before we execute this method as
        # part of expanding the $TEMPFILE construction variable.
        # Consequently, refactoring this will have to wait until
        # we get more flexible with allowing Actions to exist
        # independently and get strung together arbitrarily like
        # Ant tasks.  In the meantime, it's going to be more
        # user-friendly to not let obsession with architectural
        # purity get in the way of just being helpful, so we'll
        # reach into SCons.Action directly.
        if SCons.Action.print_actions:
            cmdstr = env.subst(self.cmdstr, SCons.Subst.SUBST_RAW, target, source) if self.cmdstr is not None else ""
            # Print our message only if XXXCOMSTR returns an empty string
            if not cmdstr:
                cmdstr = f"Using tempfile {native_tmp} for command line:\n{cmd[tempfile_arg_count]} {' '.join(args)}"
                self._print_cmd_str(target, source, env, cmdstr)

        if env.get("SHELL", None) == "sh":
            # The sh shell will try to escape the backslashes in the
            # path, so unescape them.
            native_tmp = native_tmp.replace("\\", r"\\\\")
        if "TEMPFILEPREFIX" in env:
            prefix = env.subst("$TEMPFILEPREFIX")
        else:
            prefix = "@"
        cmdlist = cmd[:tempfile_arg_count] + [prefix + native_tmp]

        # Store the temporary file command list into the target Node.attributes
        # to avoid creating separate temporary files for print and execute.
        if node is not None:
            try:
                # Storing in tempfile_cmdlist by self.cmd provided when intializing
                # $TEMPFILE{} fixes issue raised in PR #3140 and #3553
                node.attributes.tempfile_cmdlist[cmdlist_key] = cmdlist
            except AttributeError:
                node.attributes.tempfile_cmdlist = {cmdlist_key: cmdlist}

        return cmdlist

    def _print_cmd_str(self, target, source, env, cmdstr) -> None:
        # check if the user has specified a cmd line print function
        print_func = None
        try:
            get = env.get
        except AttributeError:
            pass
        else:
            print_func = get("PRINT_CMD_LINE_FUNC")

        # use the default action cmd line print if user did not supply one
        if not print_func:
            action = SCons.Action._ActionAction()
            action.print_cmd_line(cmdstr, target, source, env)
        else:
            print_func(cmdstr, target, source, env)


def generate(env):
    env["MYTEMPFILE"] = MyTempFileMunge
    env["TEMPFILEPREFIX"] = "@"
    env["MAXLINELENGTH"] = 2048
    env["TEMPFILE_ARG_COUNT"] = 1


def exists(env):
    return True
