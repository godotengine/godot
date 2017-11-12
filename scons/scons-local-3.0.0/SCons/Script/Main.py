"""SCons.Script

This file implements the main() function used by the scons script.

Architecturally, this *is* the scons script, and will likely only be
called from the external "scons" wrapper.  Consequently, anything here
should not be, or be considered, part of the build engine.  If it's
something that we expect other software to want to use, it should go in
some other module.  If it's specific to the "scons" script invocation,
it goes here.
"""

from __future__ import print_function


unsupported_python_version = (2, 6, 0)
deprecated_python_version = (2, 7, 0)


# Copyright (c) 2001 - 2017 The SCons Foundation
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

__revision__ = "src/engine/SCons/Script/Main.py rel_3.0.0:4395:8972f6a2f699 2017/09/18 12:59:24 bdbaddog"


import SCons.compat

import os
import sys
import time
import traceback
import sysconfig

import SCons.CacheDir
import SCons.Debug
import SCons.Defaults
import SCons.Environment
import SCons.Errors
import SCons.Job
import SCons.Node
import SCons.Node.FS
import SCons.Platform
import SCons.SConf
import SCons.Script
import SCons.Taskmaster
import SCons.Util
import SCons.Warnings

import SCons.Script.Interactive


def fetch_win32_parallel_msg():
    # A subsidiary function that exists solely to isolate this import
    # so we don't have to pull it in on all platforms, and so that an
    # in-line "import" statement in the _main() function below doesn't
    # cause warnings about local names shadowing use of the 'SCons'
    # global in nest scopes and UnboundLocalErrors and the like in some
    # versions (2.1) of Python.
    import SCons.Platform.win32
    return SCons.Platform.win32.parallel_msg


def revert_io():
    # This call is added to revert stderr and stdout to the original
    # ones just in case some build rule or something else in the system
    # has redirected them elsewhere.
    sys.stderr = sys.__stderr__
    sys.stdout = sys.__stdout__

class SConsPrintHelpException(Exception):
    pass

display = SCons.Util.display
progress_display = SCons.Util.DisplayEngine()

first_command_start = None
last_command_end = None


class Progressor(object):
    prev = ''
    count = 0
    target_string = '$TARGET'

    def __init__(self, obj, interval=1, file=None, overwrite=False):
        if file is None:
            file = sys.stdout

        self.obj = obj
        self.file = file
        self.interval = interval
        self.overwrite = overwrite

        if callable(obj):
            self.func = obj
        elif SCons.Util.is_List(obj):
            self.func = self.spinner
        elif obj.find(self.target_string) != -1:
            self.func = self.replace_string
        else:
            self.func = self.string

    def write(self, s):
        self.file.write(s)
        self.file.flush()
        self.prev = s

    def erase_previous(self):
        if self.prev:
            length = len(self.prev)
            if self.prev[-1] in ('\n', '\r'):
                length = length - 1
            self.write(' ' * length + '\r')
            self.prev = ''

    def spinner(self, node):
        self.write(self.obj[self.count % len(self.obj)])

    def string(self, node):
        self.write(self.obj)

    def replace_string(self, node):
        self.write(self.obj.replace(self.target_string, str(node)))

    def __call__(self, node):
        self.count = self.count + 1
        if (self.count % self.interval) == 0:
            if self.overwrite:
                self.erase_previous()
            self.func(node)

ProgressObject = SCons.Util.Null()

def Progress(*args, **kw):
    global ProgressObject
    ProgressObject = Progressor(*args, **kw)

# Task control.
#

_BuildFailures = []


def GetBuildFailures():
    return _BuildFailures


class BuildTask(SCons.Taskmaster.OutOfDateTask):
    """An SCons build task."""
    progress = ProgressObject

    def display(self, message):
        display('scons: ' + message)

    def prepare(self):
        self.progress(self.targets[0])
        return SCons.Taskmaster.OutOfDateTask.prepare(self)

    def needs_execute(self):
        if SCons.Taskmaster.OutOfDateTask.needs_execute(self):
            return True
        if self.top and self.targets[0].has_builder():
            display("scons: `%s' is up to date." % str(self.node))
        return False

    def execute(self):
        if print_time:
            start_time = time.time()
            global first_command_start
            if first_command_start is None:
                first_command_start = start_time
        SCons.Taskmaster.OutOfDateTask.execute(self)
        if print_time:
            global cumulative_command_time
            global last_command_end
            finish_time = time.time()
            last_command_end = finish_time
            cumulative_command_time = cumulative_command_time+finish_time-start_time
            sys.stdout.write("Command execution time: %s: %f seconds\n"%(str(self.node), finish_time-start_time))

    def do_failed(self, status=2):
        _BuildFailures.append(self.exception[1])
        global exit_status
        global this_build_status
        if self.options.ignore_errors:
            SCons.Taskmaster.OutOfDateTask.executed(self)
        elif self.options.keep_going:
            SCons.Taskmaster.OutOfDateTask.fail_continue(self)
            exit_status = status
            this_build_status = status
        else:
            SCons.Taskmaster.OutOfDateTask.fail_stop(self)
            exit_status = status
            this_build_status = status

    def executed(self):
        t = self.targets[0]
        if self.top and not t.has_builder() and not t.side_effect:
            if not t.exists():
                if t.__class__.__name__ in ('File', 'Dir', 'Entry'):
                    errstr="Do not know how to make %s target `%s' (%s)." % (t.__class__.__name__, t, t.get_abspath())
                else: # Alias or Python or ...
                    errstr="Do not know how to make %s target `%s'." % (t.__class__.__name__, t)
                sys.stderr.write("scons: *** " + errstr)
                if not self.options.keep_going:
                    sys.stderr.write("  Stop.")
                sys.stderr.write("\n")
                try:
                    raise SCons.Errors.BuildError(t, errstr)
                except KeyboardInterrupt:
                    raise
                except:
                    self.exception_set()
                self.do_failed()
            else:
                print("scons: Nothing to be done for `%s'." % t)
                SCons.Taskmaster.OutOfDateTask.executed(self)
        else:
            SCons.Taskmaster.OutOfDateTask.executed(self)

    def failed(self):
        # Handle the failure of a build task.  The primary purpose here
        # is to display the various types of Errors and Exceptions
        # appropriately.
        exc_info = self.exc_info()
        try:
            t, e, tb = exc_info
        except ValueError:
            t, e = exc_info
            tb = None

        if t is None:
            # The Taskmaster didn't record an exception for this Task;
            # see if the sys module has one.
            try:
                t, e, tb = sys.exc_info()[:]
            except ValueError:
                t, e = exc_info
                tb = None

        # Deprecated string exceptions will have their string stored
        # in the first entry of the tuple.
        if e is None:
            e = t

        buildError = SCons.Errors.convert_to_BuildError(e)
        if not buildError.node:
            buildError.node = self.node

        node = buildError.node
        if not SCons.Util.is_List(node):
                node = [ node ]
        nodename = ', '.join(map(str, node))

        errfmt = "scons: *** [%s] %s\n"
        sys.stderr.write(errfmt % (nodename, buildError))

        if (buildError.exc_info[2] and buildError.exc_info[1] and
           not isinstance(
               buildError.exc_info[1],
               (EnvironmentError, SCons.Errors.StopError,
                            SCons.Errors.UserError))):
            type, value, trace = buildError.exc_info
            if tb and print_stacktrace:
                sys.stderr.write("scons: internal stack trace:\n")
                traceback.print_tb(tb, file=sys.stderr)
            traceback.print_exception(type, value, trace)
        elif tb and print_stacktrace:
            sys.stderr.write("scons: internal stack trace:\n")
            traceback.print_tb(tb, file=sys.stderr)

        self.exception = (e, buildError, tb) # type, value, traceback
        self.do_failed(buildError.exitstatus)

        self.exc_clear()

    def postprocess(self):
        if self.top:
            t = self.targets[0]
            for tp in self.options.tree_printers:
                tp.display(t)
            if self.options.debug_includes:
                tree = t.render_include_tree()
                if tree:
                    print()
                    print(tree)
        SCons.Taskmaster.OutOfDateTask.postprocess(self)

    def make_ready(self):
        """Make a task ready for execution"""
        SCons.Taskmaster.OutOfDateTask.make_ready(self)
        if self.out_of_date and self.options.debug_explain:
            explanation = self.out_of_date[0].explain()
            if explanation:
                sys.stdout.write("scons: " + explanation)


class CleanTask(SCons.Taskmaster.AlwaysTask):
    """An SCons clean task."""
    def fs_delete(self, path, pathstr, remove=True):
        try:
            if os.path.lexists(path):
                if os.path.isfile(path) or os.path.islink(path):
                    if remove: os.unlink(path)
                    display("Removed " + pathstr)
                elif os.path.isdir(path) and not os.path.islink(path):
                    # delete everything in the dir
                    for e in sorted(os.listdir(path)):
                        p = os.path.join(path, e)
                        s = os.path.join(pathstr, e)
                        if os.path.isfile(p):
                            if remove: os.unlink(p)
                            display("Removed " + s)
                        else:
                            self.fs_delete(p, s, remove)
                    # then delete dir itself
                    if remove: os.rmdir(path)
                    display("Removed directory " + pathstr)
                else:
                    errstr = "Path '%s' exists but isn't a file or directory."
                    raise SCons.Errors.UserError(errstr % (pathstr))
        except SCons.Errors.UserError as e:
            print(e)
        except (IOError, OSError) as e:
            print("scons: Could not remove '%s':" % pathstr, e.strerror)

    def _get_files_to_clean(self):
        result = []
        target = self.targets[0]
        if target.has_builder() or target.side_effect:
            result = [t for t in self.targets if not t.noclean]
        return result

    def _clean_targets(self, remove=True):
        target = self.targets[0]
        if target in SCons.Environment.CleanTargets:
            files = SCons.Environment.CleanTargets[target]
            for f in files:
                self.fs_delete(f.get_abspath(), str(f), remove)

    def show(self):
        for t in self._get_files_to_clean():
            if not t.isdir():
                display("Removed " + str(t))
        self._clean_targets(remove=False)

    def remove(self):
        for t in self._get_files_to_clean():
            try:
                removed = t.remove()
            except OSError as e:
                # An OSError may indicate something like a permissions
                # issue, an IOError would indicate something like
                # the file not existing.  In either case, print a
                # message and keep going to try to remove as many
                # targets as possible.
                print("scons: Could not remove '{0}'".format(str(t)), e.strerror)
            else:
                if removed:
                    display("Removed " + str(t))
        self._clean_targets(remove=True)

    execute = remove

    # We want the Taskmaster to update the Node states (and therefore
    # handle reference counts, etc.), but we don't want to call
    # back to the Node's post-build methods, which would do things
    # we don't want, like store .sconsign information.
    executed = SCons.Taskmaster.Task.executed_without_callbacks

    # Have the Taskmaster arrange to "execute" all of the targets, because
    # we'll figure out ourselves (in remove() or show() above) whether
    # anything really needs to be done.
    make_ready = SCons.Taskmaster.Task.make_ready_all

    def prepare(self):
        pass

class QuestionTask(SCons.Taskmaster.AlwaysTask):
    """An SCons task for the -q (question) option."""
    def prepare(self):
        pass

    def execute(self):
        if self.targets[0].get_state() != SCons.Node.up_to_date or \
           (self.top and not self.targets[0].exists()):
            global exit_status
            global this_build_status
            exit_status = 1
            this_build_status = 1
            self.tm.stop()

    def executed(self):
        pass


class TreePrinter(object):
    def __init__(self, derived=False, prune=False, status=False):
        self.derived = derived
        self.prune = prune
        self.status = status
    def get_all_children(self, node):
        return node.all_children()
    def get_derived_children(self, node):
        children = node.all_children(None)
        return [x for x in children if x.has_builder()]
    def display(self, t):
        if self.derived:
            func = self.get_derived_children
        else:
            func = self.get_all_children
        s = self.status and 2 or 0
        SCons.Util.print_tree(t, func, prune=self.prune, showtags=s)


def python_version_string():
    return sys.version.split()[0]

def python_version_unsupported(version=sys.version_info):
    return version < unsupported_python_version

def python_version_deprecated(version=sys.version_info):
    return version < deprecated_python_version


# Global variables

print_objects = 0
print_memoizer = 0
print_stacktrace = 0
print_time = 0
sconscript_time = 0
cumulative_command_time = 0
exit_status = 0 # final exit status, assume success by default
this_build_status = 0 # "exit status" of an individual build
num_jobs = None
delayed_warnings = []

class FakeOptionParser(object):
    """
    A do-nothing option parser, used for the initial OptionsParser variable.

    During normal SCons operation, the OptionsParser is created right
    away by the main() function.  Certain tests scripts however, can
    introspect on different Tool modules, the initialization of which
    can try to add a new, local option to an otherwise uninitialized
    OptionsParser object.  This allows that introspection to happen
    without blowing up.

    """
    class FakeOptionValues(object):
        def __getattr__(self, attr):
            return None
    values = FakeOptionValues()
    def add_local_option(self, *args, **kw):
        pass

OptionsParser = FakeOptionParser()

def AddOption(*args, **kw):
    if 'default' not in kw:
        kw['default'] = None
    result = OptionsParser.add_local_option(*args, **kw)
    return result

def GetOption(name):
    return getattr(OptionsParser.values, name)

def SetOption(name, value):
    return OptionsParser.values.set_option(name, value)

def PrintHelp(file=None):
    OptionsParser.print_help(file=file)

class Stats(object):
    def __init__(self):
        self.stats = []
        self.labels = []
        self.append = self.do_nothing
        self.print_stats = self.do_nothing
    def enable(self, outfp):
        self.outfp = outfp
        self.append = self.do_append
        self.print_stats = self.do_print
    def do_nothing(self, *args, **kw):
        pass

class CountStats(Stats):
    def do_append(self, label):
        self.labels.append(label)
        self.stats.append(SCons.Debug.fetchLoggedInstances())
    def do_print(self):
        stats_table = {}
        for s in self.stats:
            for n in [t[0] for t in s]:
                stats_table[n] = [0, 0, 0, 0]
        i = 0
        for s in self.stats:
            for n, c in s:
                stats_table[n][i] = c
            i = i + 1
        self.outfp.write("Object counts:\n")
        pre = ["   "]
        post = ["   %s\n"]
        l = len(self.stats)
        fmt1 = ''.join(pre + [' %7s']*l + post)
        fmt2 = ''.join(pre + [' %7d']*l + post)
        labels = self.labels[:l]
        labels.append(("", "Class"))
        self.outfp.write(fmt1 % tuple([x[0] for x in labels]))
        self.outfp.write(fmt1 % tuple([x[1] for x in labels]))
        for k in sorted(stats_table.keys()):
            r = stats_table[k][:l] + [k]
            self.outfp.write(fmt2 % tuple(r))

count_stats = CountStats()

class MemStats(Stats):
    def do_append(self, label):
        self.labels.append(label)
        self.stats.append(SCons.Debug.memory())
    def do_print(self):
        fmt = 'Memory %-32s %12d\n'
        for label, stats in zip(self.labels, self.stats):
            self.outfp.write(fmt % (label, stats))

memory_stats = MemStats()

# utility functions

def _scons_syntax_error(e):
    """Handle syntax errors. Print out a message and show where the error
    occurred.
    """
    etype, value, tb = sys.exc_info()
    lines = traceback.format_exception_only(etype, value)
    for line in lines:
        sys.stderr.write(line+'\n')
    sys.exit(2)

def find_deepest_user_frame(tb):
    """
    Find the deepest stack frame that is not part of SCons.

    Input is a "pre-processed" stack trace in the form
    returned by traceback.extract_tb() or traceback.extract_stack()
    """

    tb.reverse()

    # find the deepest traceback frame that is not part
    # of SCons:
    for frame in tb:
        filename = frame[0]
        if filename.find(os.sep+'SCons'+os.sep) == -1:
            return frame
    return tb[0]

def _scons_user_error(e):
    """Handle user errors. Print out a message and a description of the
    error, along with the line number and routine where it occured.
    The file and line number will be the deepest stack frame that is
    not part of SCons itself.
    """
    global print_stacktrace
    etype, value, tb = sys.exc_info()
    if print_stacktrace:
        traceback.print_exception(etype, value, tb)
    filename, lineno, routine, dummy = find_deepest_user_frame(traceback.extract_tb(tb))
    sys.stderr.write("\nscons: *** %s\n" % value)
    sys.stderr.write('File "%s", line %d, in %s\n' % (filename, lineno, routine))
    sys.exit(2)

def _scons_user_warning(e):
    """Handle user warnings. Print out a message and a description of
    the warning, along with the line number and routine where it occured.
    The file and line number will be the deepest stack frame that is
    not part of SCons itself.
    """
    etype, value, tb = sys.exc_info()
    filename, lineno, routine, dummy = find_deepest_user_frame(traceback.extract_tb(tb))
    sys.stderr.write("\nscons: warning: %s\n" % e)
    sys.stderr.write('File "%s", line %d, in %s\n' % (filename, lineno, routine))

def _scons_internal_warning(e):
    """Slightly different from _scons_user_warning in that we use the
    *current call stack* rather than sys.exc_info() to get our stack trace.
    This is used by the warnings framework to print warnings."""
    filename, lineno, routine, dummy = find_deepest_user_frame(traceback.extract_stack())
    sys.stderr.write("\nscons: warning: %s\n" % e.args[0])
    sys.stderr.write('File "%s", line %d, in %s\n' % (filename, lineno, routine))

def _scons_internal_error():
    """Handle all errors but user errors. Print out a message telling
    the user what to do in this case and print a normal trace.
    """
    print('internal error')
    traceback.print_exc()
    sys.exit(2)

def _SConstruct_exists(dirname='', repositories=[], filelist=None):
    """This function checks that an SConstruct file exists in a directory.
    If so, it returns the path of the file. By default, it checks the
    current directory.
    """
    if not filelist:
        filelist = ['SConstruct', 'Sconstruct', 'sconstruct']
    for file in filelist:
        sfile = os.path.join(dirname, file)
        if os.path.isfile(sfile):
            return sfile
        if not os.path.isabs(sfile):
            for rep in repositories:
                if os.path.isfile(os.path.join(rep, sfile)):
                    return sfile
    return None

def _set_debug_values(options):
    global print_memoizer, print_objects, print_stacktrace, print_time

    debug_values = options.debug

    if "count" in debug_values:
        # All of the object counts are within "if track_instances:" blocks,
        # which get stripped when running optimized (with python -O or
        # from compiled *.pyo files).  Provide a warning if __debug__ is
        # stripped, so it doesn't just look like --debug=count is broken.
        enable_count = False
        if __debug__: enable_count = True
        if enable_count:
            count_stats.enable(sys.stdout)
            SCons.Debug.track_instances = True
        else:
            msg = "--debug=count is not supported when running SCons\n" + \
                  "\twith the python -O option or optimized (.pyo) modules."
            SCons.Warnings.warn(SCons.Warnings.NoObjectCountWarning, msg)
    if "dtree" in debug_values:
        options.tree_printers.append(TreePrinter(derived=True))
    options.debug_explain = ("explain" in debug_values)
    if "findlibs" in debug_values:
        SCons.Scanner.Prog.print_find_libs = "findlibs"
    options.debug_includes = ("includes" in debug_values)
    print_memoizer = ("memoizer" in debug_values)
    if "memory" in debug_values:
        memory_stats.enable(sys.stdout)
    print_objects = ("objects" in debug_values)
    if print_objects:
        SCons.Debug.track_instances = True
    if "presub" in debug_values:
        SCons.Action.print_actions_presub = 1
    if "stacktrace" in debug_values:
        print_stacktrace = 1
    if "stree" in debug_values:
        options.tree_printers.append(TreePrinter(status=True))
    if "time" in debug_values:
        print_time = 1
    if "tree" in debug_values:
        options.tree_printers.append(TreePrinter())
    if "prepare" in debug_values:
        SCons.Taskmaster.print_prepare = 1
    if "duplicate" in debug_values:
        SCons.Node.print_duplicate = 1

def _create_path(plist):
    path = '.'
    for d in plist:
        if os.path.isabs(d):
            path = d
        else:
            path = path + '/' + d
    return path

def _load_site_scons_dir(topdir, site_dir_name=None):
    """Load the site_scons dir under topdir.
    Prepends site_scons to sys.path, imports site_scons/site_init.py,
    and prepends site_scons/site_tools to default toolpath."""
    if site_dir_name:
        err_if_not_found = True       # user specified: err if missing
    else:
        site_dir_name = "site_scons"
        err_if_not_found = False

    site_dir = os.path.join(topdir, site_dir_name)
    if not os.path.exists(site_dir):
        if err_if_not_found:
            raise SCons.Errors.UserError("site dir %s not found."%site_dir)
        return

    site_init_filename = "site_init.py"
    site_init_modname = "site_init"
    site_tools_dirname = "site_tools"
    # prepend to sys.path
    sys.path = [os.path.abspath(site_dir)] + sys.path
    site_init_file = os.path.join(site_dir, site_init_filename)
    site_tools_dir = os.path.join(site_dir, site_tools_dirname)
    if os.path.exists(site_init_file):
        import imp, re
        try:
            try:
                fp, pathname, description = imp.find_module(site_init_modname,
                                                            [site_dir])
                # Load the file into SCons.Script namespace.  This is
                # opaque and clever; m is the module object for the
                # SCons.Script module, and the exec ... in call executes a
                # file (or string containing code) in the context of the
                # module's dictionary, so anything that code defines ends
                # up adding to that module.  This is really short, but all
                # the error checking makes it longer.
                try:
                    m = sys.modules['SCons.Script']
                except Exception as e:
                    fmt = 'cannot import site_init.py: missing SCons.Script module %s'
                    raise SCons.Errors.InternalError(fmt % repr(e))
                try:
                    sfx = description[0]
                    modname = os.path.basename(pathname)[:-len(sfx)]
                    site_m = {"__file__": pathname, "__name__": modname, "__doc__": None}
                    re_special = re.compile("__[^_]+__")
                    for k in list(m.__dict__.keys()):
                        if not re_special.match(k):
                            site_m[k] = m.__dict__[k]

                    # This is the magic.
                    exec(compile(fp.read(), fp.name, 'exec'), site_m)
                except KeyboardInterrupt:
                    raise
                except Exception as e:
                    fmt = '*** Error loading site_init file %s:\n'
                    sys.stderr.write(fmt % repr(site_init_file))
                    raise
                else:
                    for k in site_m:
                        if not re_special.match(k):
                            m.__dict__[k] = site_m[k]
            except KeyboardInterrupt:
                raise
            except ImportError as e:
                fmt = '*** cannot import site init file %s:\n'
                sys.stderr.write(fmt % repr(site_init_file))
                raise
        finally:
            if fp:
                fp.close()
    if os.path.exists(site_tools_dir):
        # prepend to DefaultToolpath
        SCons.Tool.DefaultToolpath.insert(0, os.path.abspath(site_tools_dir))

def _load_all_site_scons_dirs(topdir, verbose=None):
    """Load all of the predefined site_scons dir.
    Order is significant; we load them in order from most generic
    (machine-wide) to most specific (topdir).
    The verbose argument is only for testing.
    """
    platform = SCons.Platform.platform_default()

    def homedir(d):
        return os.path.expanduser('~/'+d)

    if platform == 'win32' or platform == 'cygwin':
        # Note we use $ here instead of %...% because older
        # pythons (prior to 2.6?) didn't expand %...% on Windows.
        # This set of dirs should work on XP, Vista, 7 and later.
        sysdirs=[
            os.path.expandvars('$ALLUSERSPROFILE\\Application Data\\scons'),
            os.path.expandvars('$USERPROFILE\\Local Settings\\Application Data\\scons')]
        appdatadir = os.path.expandvars('$APPDATA\\scons')
        if appdatadir not in sysdirs:
            sysdirs.append(appdatadir)
        sysdirs.append(homedir('.scons'))

    elif platform == 'darwin':  # MacOS X
        sysdirs=['/Library/Application Support/SCons',
                 '/opt/local/share/scons', # (for MacPorts)
                 '/sw/share/scons', # (for Fink)
                  homedir('Library/Application Support/SCons'),
                  homedir('.scons')]
    elif platform == 'sunos':   # Solaris
        sysdirs=['/opt/sfw/scons',
                 '/usr/share/scons',
                 homedir('.scons')]
    else:                       # Linux, HPUX, etc.
        # assume posix-like, i.e. platform == 'posix'
        sysdirs=['/usr/share/scons',
                 homedir('.scons')]

    dirs=sysdirs + [topdir]
    for d in dirs:
        if verbose:    # this is used by unit tests.
            print("Loading site dir ", d)
        _load_site_scons_dir(d)

def test_load_all_site_scons_dirs(d):
    _load_all_site_scons_dirs(d, True)

def version_string(label, module):
    version = module.__version__
    build = module.__build__
    if build:
        if build[0] != '.':
            build = '.' + build
        version = version + build
    fmt = "\t%s: v%s, %s, by %s on %s\n"
    return fmt % (label,
                  version,
                  module.__date__,
                  module.__developer__,
                  module.__buildsys__)

def path_string(label, module):
    path = module.__path__
    return "\t%s path: %s\n"%(label,path)

def _main(parser):
    global exit_status
    global this_build_status

    options = parser.values

    # Here's where everything really happens.

    # First order of business:  set up default warnings and then
    # handle the user's warning options, so that we can issue (or
    # suppress) appropriate warnings about anything that might happen,
    # as configured by the user.

    default_warnings = [ SCons.Warnings.WarningOnByDefault,
                         SCons.Warnings.DeprecatedWarning,
                       ]

    for warning in default_warnings:
        SCons.Warnings.enableWarningClass(warning)
    SCons.Warnings._warningOut = _scons_internal_warning
    SCons.Warnings.process_warn_strings(options.warn)

    # Now that we have the warnings configuration set up, we can actually
    # issue (or suppress) any warnings about warning-worthy things that
    # occurred while the command-line options were getting parsed.
    try:
        dw = options.delayed_warnings
    except AttributeError:
        pass
    else:
        delayed_warnings.extend(dw)
    for warning_type, message in delayed_warnings:
        SCons.Warnings.warn(warning_type, message)

    if options.diskcheck:
        SCons.Node.FS.set_diskcheck(options.diskcheck)

    # Next, we want to create the FS object that represents the outside
    # world's file system, as that's central to a lot of initialization.
    # To do this, however, we need to be in the directory from which we
    # want to start everything, which means first handling any relevant
    # options that might cause us to chdir somewhere (-C, -D, -U, -u).
    if options.directory:
        script_dir = os.path.abspath(_create_path(options.directory))
    else:
        script_dir = os.getcwd()

    target_top = None
    if options.climb_up:
        target_top = '.'  # directory to prepend to targets
        while script_dir and not _SConstruct_exists(script_dir,
                                                    options.repository,
                                                    options.file):
            script_dir, last_part = os.path.split(script_dir)
            if last_part:
                target_top = os.path.join(last_part, target_top)
            else:
                script_dir = ''

    if script_dir and script_dir != os.getcwd():
        if not options.silent:
            display("scons: Entering directory `%s'" % script_dir)
        try:
            os.chdir(script_dir)
        except OSError:
            sys.stderr.write("Could not change directory to %s\n" % script_dir)

    # Now that we're in the top-level SConstruct directory, go ahead
    # and initialize the FS object that represents the file system,
    # and make it the build engine default.
    fs = SCons.Node.FS.get_default_fs()

    for rep in options.repository:
        fs.Repository(rep)

    # Now that we have the FS object, the next order of business is to
    # check for an SConstruct file (or other specified config file).
    # If there isn't one, we can bail before doing any more work.
    scripts = []
    if options.file:
        scripts.extend(options.file)
    if not scripts:
        sfile = _SConstruct_exists(repositories=options.repository,
                                   filelist=options.file)
        if sfile:
            scripts.append(sfile)

    if not scripts:
        if options.help:
            # There's no SConstruct, but they specified -h.
            # Give them the options usage now, before we fail
            # trying to read a non-existent SConstruct file.
            raise SConsPrintHelpException
        raise SCons.Errors.UserError("No SConstruct file found.")

    if scripts[0] == "-":
        d = fs.getcwd()
    else:
        d = fs.File(scripts[0]).dir
    fs.set_SConstruct_dir(d)

    _set_debug_values(options)
    SCons.Node.implicit_cache = options.implicit_cache
    SCons.Node.implicit_deps_changed = options.implicit_deps_changed
    SCons.Node.implicit_deps_unchanged = options.implicit_deps_unchanged

    if options.no_exec:
        SCons.SConf.dryrun = 1
        SCons.Action.execute_actions = None
    if options.question:
        SCons.SConf.dryrun = 1
    if options.clean:
        SCons.SConf.SetBuildType('clean')
    if options.help:
        SCons.SConf.SetBuildType('help')
    SCons.SConf.SetCacheMode(options.config)
    SCons.SConf.SetProgressDisplay(progress_display)

    if options.no_progress or options.silent:
        progress_display.set_mode(0)

    if options.site_dir:
        _load_site_scons_dir(d.get_internal_path(), options.site_dir)
    elif not options.no_site_dir:
        _load_all_site_scons_dirs(d.get_internal_path())

    if options.include_dir:
        sys.path = options.include_dir + sys.path

    # If we're about to start SCons in the interactive mode,
    # inform the FS about this right here. Else, the release_target_info
    # method could get called on some nodes, like the used "gcc" compiler,
    # when using the Configure methods within the SConscripts.
    # This would then cause subtle bugs, as already happened in #2971.
    if options.interactive:
        SCons.Node.interactive = True

    # That should cover (most of) the options.  Next, set up the variables
    # that hold command-line arguments, so the SConscript files that we
    # read and execute have access to them.
    targets = []
    xmit_args = []
    for a in parser.largs:
        if a[:1] == '-':
            continue
        if '=' in a:
            xmit_args.append(a)
        else:
            targets.append(a)
    SCons.Script._Add_Targets(targets + parser.rargs)
    SCons.Script._Add_Arguments(xmit_args)

    # If stdout is not a tty, replace it with a wrapper object to call flush
    # after every write.
    #
    # Tty devices automatically flush after every newline, so the replacement
    # isn't necessary.  Furthermore, if we replace sys.stdout, the readline
    # module will no longer work.  This affects the behavior during
    # --interactive mode.  --interactive should only be used when stdin and
    # stdout refer to a tty.
    if not hasattr(sys.stdout, 'isatty') or not sys.stdout.isatty():
        sys.stdout = SCons.Util.Unbuffered(sys.stdout)
    if not hasattr(sys.stderr, 'isatty') or not sys.stderr.isatty():
        sys.stderr = SCons.Util.Unbuffered(sys.stderr)

    memory_stats.append('before reading SConscript files:')
    count_stats.append(('pre-', 'read'))

    # And here's where we (finally) read the SConscript files.

    progress_display("scons: Reading SConscript files ...")

    start_time = time.time()
    try:
        for script in scripts:
            SCons.Script._SConscript._SConscript(fs, script)
    except SCons.Errors.StopError as e:
        # We had problems reading an SConscript file, such as it
        # couldn't be copied in to the VariantDir.  Since we're just
        # reading SConscript files and haven't started building
        # things yet, stop regardless of whether they used -i or -k
        # or anything else.
        revert_io()
        sys.stderr.write("scons: *** %s  Stop.\n" % e)
        sys.exit(2)
    global sconscript_time
    sconscript_time = time.time() - start_time

    progress_display("scons: done reading SConscript files.")

    memory_stats.append('after reading SConscript files:')
    count_stats.append(('post-', 'read'))

    # Re-{enable,disable} warnings in case they disabled some in
    # the SConscript file.
    #
    # We delay enabling the PythonVersionWarning class until here so that,
    # if they explicitly disabled it in either in the command line or in
    # $SCONSFLAGS, or in the SConscript file, then the search through
    # the list of deprecated warning classes will find that disabling
    # first and not issue the warning.
    #SCons.Warnings.enableWarningClass(SCons.Warnings.PythonVersionWarning)
    SCons.Warnings.process_warn_strings(options.warn)

    # Now that we've read the SConscript files, we can check for the
    # warning about deprecated Python versions--delayed until here
    # in case they disabled the warning in the SConscript files.
    if python_version_deprecated():
        msg = "Support for pre-%s Python version (%s) is deprecated.\n" + \
              "    If this will cause hardship, contact scons-dev@scons.org"
        deprecated_version_string = ".".join(map(str, deprecated_python_version))
        SCons.Warnings.warn(SCons.Warnings.PythonVersionWarning,
                            msg % (deprecated_version_string, python_version_string()))

    if not options.help:
        # [ ] Clarify why we need to create Builder here at all, and
        #     why it is created in DefaultEnvironment
        # https://bitbucket.org/scons/scons/commits/d27a548aeee8ad5e67ea75c2d19a7d305f784e30
        if SCons.SConf.NeedConfigHBuilder():
            SCons.SConf.CreateConfigHBuilder(SCons.Defaults.DefaultEnvironment())

    # Now re-parse the command-line options (any to the left of a '--'
    # argument, that is) with any user-defined command-line options that
    # the SConscript files may have added to the parser object.  This will
    # emit the appropriate error message and exit if any unknown option
    # was specified on the command line.

    parser.preserve_unknown_options = False
    parser.parse_args(parser.largs, options)

    if options.help:
        help_text = SCons.Script.help_text
        if help_text is None:
            # They specified -h, but there was no Help() inside the
            # SConscript files.  Give them the options usage.
            raise SConsPrintHelpException
        else:
            print(help_text)
            print("Use scons -H for help about command-line options.")
        exit_status = 0
        return

    # Change directory to the top-level SConstruct directory, then tell
    # the Node.FS subsystem that we're all done reading the SConscript
    # files and calling Repository() and VariantDir() and changing
    # directories and the like, so it can go ahead and start memoizing
    # the string values of file system nodes.

    fs.chdir(fs.Top)

    SCons.Node.FS.save_strings(1)

    # Now that we've read the SConscripts we can set the options
    # that are SConscript settable:
    SCons.Node.implicit_cache = options.implicit_cache
    SCons.Node.FS.set_duplicate(options.duplicate)
    fs.set_max_drift(options.max_drift)

    SCons.Job.explicit_stack_size = options.stack_size

    if options.md5_chunksize:
        SCons.Node.FS.File.md5_chunksize = options.md5_chunksize

    platform = SCons.Platform.platform_module()

    if options.interactive:
        SCons.Script.Interactive.interact(fs, OptionsParser, options,
                                          targets, target_top)

    else:

        # Build the targets
        nodes = _build_targets(fs, options, targets, target_top)
        if not nodes:
            revert_io()
            print('Found nothing to build')
            exit_status = 2

def _build_targets(fs, options, targets, target_top):

    global this_build_status
    this_build_status = 0

    progress_display.set_mode(not (options.no_progress or options.silent))
    display.set_mode(not options.silent)
    SCons.Action.print_actions          = not options.silent
    SCons.Action.execute_actions        = not options.no_exec
    SCons.Node.do_store_info            = not options.no_exec
    SCons.SConf.dryrun                  = options.no_exec

    if options.diskcheck:
        SCons.Node.FS.set_diskcheck(options.diskcheck)

    SCons.CacheDir.cache_enabled = not options.cache_disable
    SCons.CacheDir.cache_readonly = options.cache_readonly
    SCons.CacheDir.cache_debug = options.cache_debug
    SCons.CacheDir.cache_force = options.cache_force
    SCons.CacheDir.cache_show = options.cache_show

    if options.no_exec:
        CleanTask.execute = CleanTask.show
    else:
        CleanTask.execute = CleanTask.remove

    lookup_top = None
    if targets or SCons.Script.BUILD_TARGETS != SCons.Script._build_plus_default:
        # They specified targets on the command line or modified
        # BUILD_TARGETS in the SConscript file(s), so if they used -u,
        # -U or -D, we have to look up targets relative to the top,
        # but we build whatever they specified.
        if target_top:
            lookup_top = fs.Dir(target_top)
            target_top = None

        targets = SCons.Script.BUILD_TARGETS
    else:
        # There are no targets specified on the command line,
        # so if they used -u, -U or -D, we may have to restrict
        # what actually gets built.
        d = None
        if target_top:
            if options.climb_up == 1:
                # -u, local directory and below
                target_top = fs.Dir(target_top)
                lookup_top = target_top
            elif options.climb_up == 2:
                # -D, all Default() targets
                target_top = None
                lookup_top = None
            elif options.climb_up == 3:
                # -U, local SConscript Default() targets
                target_top = fs.Dir(target_top)
                def check_dir(x, target_top=target_top):
                    if hasattr(x, 'cwd') and not x.cwd is None:
                        cwd = x.cwd.srcnode()
                        return cwd == target_top
                    else:
                        # x doesn't have a cwd, so it's either not a target,
                        # or not a file, so go ahead and keep it as a default
                        # target and let the engine sort it out:
                        return 1
                d = [tgt for tgt in SCons.Script.DEFAULT_TARGETS if check_dir(tgt)]
                SCons.Script.DEFAULT_TARGETS[:] = d
                target_top = None
                lookup_top = None

        targets = SCons.Script._Get_Default_Targets(d, fs)

    if not targets:
        sys.stderr.write("scons: *** No targets specified and no Default() targets found.  Stop.\n")
        return None

    def Entry(x, ltop=lookup_top, ttop=target_top, fs=fs):
        if isinstance(x, SCons.Node.Node):
            node = x
        else:
            node = None
            # Why would ltop be None? Unfortunately this happens.
            if ltop is None: ltop = ''
            # Curdir becomes important when SCons is called with -u, -C,
            # or similar option that changes directory, and so the paths
            # of targets given on the command line need to be adjusted.
            curdir = os.path.join(os.getcwd(), str(ltop))
            for lookup in SCons.Node.arg2nodes_lookups:
                node = lookup(x, curdir=curdir)
                if node is not None:
                    break
            if node is None:
                node = fs.Entry(x, directory=ltop, create=1)
        if ttop and not node.is_under(ttop):
            if isinstance(node, SCons.Node.FS.Dir) and ttop.is_under(node):
                node = ttop
            else:
                node = None
        return node

    nodes = [_f for _f in map(Entry, targets) if _f]

    task_class = BuildTask      # default action is to build targets
    opening_message = "Building targets ..."
    closing_message = "done building targets."
    if options.keep_going:
        failure_message = "done building targets (errors occurred during build)."
    else:
        failure_message = "building terminated because of errors."
    if options.question:
        task_class = QuestionTask
    try:
        if options.clean:
            task_class = CleanTask
            opening_message = "Cleaning targets ..."
            closing_message = "done cleaning targets."
            if options.keep_going:
                failure_message = "done cleaning targets (errors occurred during clean)."
            else:
                failure_message = "cleaning terminated because of errors."
    except AttributeError:
        pass

    task_class.progress = ProgressObject

    if options.random:
        def order(dependencies):
            """Randomize the dependencies."""
            import random
            random.shuffle(dependencies)
            return dependencies
    else:
        def order(dependencies):
            """Leave the order of dependencies alone."""
            return dependencies

    if options.taskmastertrace_file == '-':
        tmtrace = sys.stdout
    elif options.taskmastertrace_file:
        tmtrace = open(options.taskmastertrace_file, 'w')
    else:
        tmtrace = None
    taskmaster = SCons.Taskmaster.Taskmaster(nodes, task_class, order, tmtrace)

    # Let the BuildTask objects get at the options to respond to the
    # various print_* settings, tree_printer list, etc.
    BuildTask.options = options


    python_has_threads = sysconfig.get_config_var('WITH_THREAD')
    # to check if python configured with threads.
    global num_jobs
    num_jobs = options.num_jobs
    jobs = SCons.Job.Jobs(num_jobs, taskmaster)
    if num_jobs > 1:
        msg = None
        if sys.platform == 'win32':
            msg = fetch_win32_parallel_msg()
        elif jobs.num_jobs == 1 or not python_has_threads:
            msg = "parallel builds are unsupported by this version of Python;\n" + \
                  "\tignoring -j or num_jobs option.\n"
        if msg:
            SCons.Warnings.warn(SCons.Warnings.NoParallelSupportWarning, msg)

    memory_stats.append('before building targets:')
    count_stats.append(('pre-', 'build'))

    def jobs_postfunc(
        jobs=jobs,
        options=options,
        closing_message=closing_message,
        failure_message=failure_message
        ):
        if jobs.were_interrupted():
            if not options.no_progress and not options.silent:
                sys.stderr.write("scons: Build interrupted.\n")
            global exit_status
            global this_build_status
            exit_status = 2
            this_build_status = 2

        if this_build_status:
            progress_display("scons: " + failure_message)
        else:
            progress_display("scons: " + closing_message)
        if not options.no_exec:
            if jobs.were_interrupted():
                progress_display("scons: writing .sconsign file.")
            SCons.SConsign.write()

    progress_display("scons: " + opening_message)
    jobs.run(postfunc = jobs_postfunc)

    memory_stats.append('after building targets:')
    count_stats.append(('post-', 'build'))

    return nodes

def _exec_main(parser, values):
    sconsflags = os.environ.get('SCONSFLAGS', '')
    all_args = sconsflags.split() + sys.argv[1:]

    options, args = parser.parse_args(all_args, values)

    if isinstance(options.debug, list) and "pdb" in options.debug:
        import pdb
        pdb.Pdb().runcall(_main, parser)
    elif options.profile_file:
        # compat layer imports "cProfile" for us if it's available.
        from profile import Profile

        prof = Profile()
        try:
            prof.runcall(_main, parser)
        finally:
            prof.dump_stats(options.profile_file)
    else:
        _main(parser)

def main():
    global OptionsParser
    global exit_status
    global first_command_start

    # Check up front for a Python version we do not support.  We
    # delay the check for deprecated Python versions until later,
    # after the SConscript files have been read, in case they
    # disable that warning.
    if python_version_unsupported():
        msg = "scons: *** SCons version %s does not run under Python version %s.\n"
        sys.stderr.write(msg % (SCons.__version__, python_version_string()))
        sys.exit(1)

    parts = ["SCons by Steven Knight et al.:\n"]
    try:
        import __main__
        parts.append(version_string("script", __main__))
    except (ImportError, AttributeError):
        # On Windows there is no scons.py, so there is no
        # __main__.__version__, hence there is no script version.
        pass
    parts.append(version_string("engine", SCons))
    parts.append(path_string("engine", SCons))
    parts.append("Copyright (c) 2001 - 2017 The SCons Foundation")
    version = ''.join(parts)

    from . import SConsOptions
    parser = SConsOptions.Parser(version)
    values = SConsOptions.SConsValues(parser.get_default_values())

    OptionsParser = parser

    try:
        try:
            _exec_main(parser, values)
        finally:
            revert_io()
    except SystemExit as s:
        if s:
            exit_status = s
    except KeyboardInterrupt:
        print("scons: Build interrupted.")
        sys.exit(2)
    except SyntaxError as e:
        _scons_syntax_error(e)
    except SCons.Errors.InternalError:
        _scons_internal_error()
    except SCons.Errors.UserError as e:
        _scons_user_error(e)
    except SConsPrintHelpException:
        parser.print_help()
        exit_status = 0
    except SCons.Errors.BuildError as e:
        print(e)
        exit_status = e.exitstatus
    except:
        # An exception here is likely a builtin Python exception Python
        # code in an SConscript file.  Show them precisely what the
        # problem was and where it happened.
        SCons.Script._SConscript.SConscript_exception()
        sys.exit(2)

    memory_stats.print_stats()
    count_stats.print_stats()

    if print_objects:
        SCons.Debug.listLoggedInstances('*')
        #SCons.Debug.dumpLoggedInstances('*')

    if print_memoizer:
        SCons.Memoize.Dump("Memoizer (memory cache) hits and misses:")

    # Dump any development debug info that may have been enabled.
    # These are purely for internal debugging during development, so
    # there's no need to control them with --debug= options; they're
    # controlled by changing the source code.
    SCons.Debug.dump_caller_counts()
    SCons.Taskmaster.dump_stats()

    if print_time:
        total_time = time.time() - SCons.Script.start_time
        if num_jobs == 1:
            ct = cumulative_command_time
        else:
            if last_command_end is None or first_command_start is None:
                ct = 0.0
            else:
                ct = last_command_end - first_command_start
        scons_time = total_time - sconscript_time - ct
        print("Total build time: %f seconds"%total_time)
        print("Total SConscript file execution time: %f seconds"%sconscript_time)
        print("Total SCons execution time: %f seconds"%scons_time)
        print("Total command execution time: %f seconds"%ct)

    sys.exit(exit_status)

# Local Variables:
# tab-width:4
# indent-tabs-mode:nil
# End:
# vim: set expandtab tabstop=4 shiftwidth=4:
