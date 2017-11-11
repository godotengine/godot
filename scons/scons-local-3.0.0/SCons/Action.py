"""SCons.Action

This encapsulates information about executing any sort of action that
can build one or more target Nodes (typically files) from one or more
source Nodes (also typically files) given a specific Environment.

The base class here is ActionBase.  The base class supplies just a few
OO utility methods and some generic methods for displaying information
about an Action in response to the various commands that control printing.

A second-level base class is _ActionAction.  This extends ActionBase
by providing the methods that can be used to show and perform an
action.  True Action objects will subclass _ActionAction; Action
factory class objects will subclass ActionBase.

The heavy lifting is handled by subclasses for the different types of
actions we might execute:

    CommandAction
    CommandGeneratorAction
    FunctionAction
    ListAction

The subclasses supply the following public interface methods used by
other modules:

    __call__()
        THE public interface, "calling" an Action object executes the
        command or Python function.  This also takes care of printing
        a pre-substitution command for debugging purposes.

    get_contents()
        Fetches the "contents" of an Action for signature calculation
        plus the varlist.  This is what gets MD5 checksummed to decide
        if a target needs to be rebuilt because its action changed.

    genstring()
        Returns a string representation of the Action *without*
        command substitution, but allows a CommandGeneratorAction to
        generate the right action based on the specified target,
        source and env.  This is used by the Signature subsystem
        (through the Executor) to obtain an (imprecise) representation
        of the Action operation for informative purposes.


Subclasses also supply the following methods for internal use within
this module:

    __str__()
        Returns a string approximation of the Action; no variable
        substitution is performed.

    execute()
        The internal method that really, truly, actually handles the
        execution of a command or Python function.  This is used so
        that the __call__() methods can take care of displaying any
        pre-substitution representations, and *then* execute an action
        without worrying about the specific Actions involved.

    get_presig()
        Fetches the "contents" of a subclass for signature calculation.
        The varlist is added to this to produce the Action's contents.
        TODO(?): Change this to always return ascii/bytes and not unicode (or py3 strings)

    strfunction()
        Returns a substituted string representation of the Action.
        This is used by the _ActionAction.show() command to display the
        command/function that will be executed to generate the target(s).

There is a related independent ActionCaller class that looks like a
regular Action, and which serves as a wrapper for arbitrary functions
that we want to let the user specify the arguments to now, but actually
execute later (when an out-of-date check determines that it's needed to
be executed, for example).  Objects of this class are returned by an
ActionFactory class that provides a __call__() method as a convenient
way for wrapping up the functions.

"""

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

__revision__ = "src/engine/SCons/Action.py rel_3.0.0:4395:8972f6a2f699 2017/09/18 12:59:24 bdbaddog"

import os
import pickle
import re
import sys
import subprocess
import itertools
import inspect

import SCons.Debug
from SCons.Debug import logInstanceCreation
import SCons.Errors
import SCons.Util
import SCons.Subst

# we use these a lot, so try to optimize them
is_String = SCons.Util.is_String
is_List = SCons.Util.is_List

class _null(object):
    pass

print_actions = 1
execute_actions = 1
print_actions_presub = 0

# Use pickle protocol 1 when pickling functions for signature
# otherwise python3 and python2 will yield different pickles
# for the same object.
# This is due to default being 1 for python 2.7, and 3 for 3.x
# TODO: We can roll this forward to 2 (if it has value), but not
# before a deprecation cycle as the sconsigns will change
ACTION_SIGNATURE_PICKLE_PROTOCOL = 1


def rfile(n):
    try:
        return n.rfile()
    except AttributeError:
        return n


def default_exitstatfunc(s):
    return s

strip_quotes = re.compile('^[\'"](.*)[\'"]$')


def _callable_contents(obj):
    """Return the signature contents of a callable Python object.
    """
    try:
        # Test if obj is a method.
        return _function_contents(obj.__func__)

    except AttributeError:
        try:
            # Test if obj is a callable object.
            return _function_contents(obj.__call__.__func__)

        except AttributeError:
            try:
                # Test if obj is a code object.
                return _code_contents(obj)

            except AttributeError:
                # Test if obj is a function object.
                return _function_contents(obj)


def _object_contents(obj):
    """Return the signature contents of any Python object.

    We have to handle the case where object contains a code object
    since it can be pickled directly.
    """
    try:
        # Test if obj is a method.
        return _function_contents(obj.__func__)

    except AttributeError:
        try:
            # Test if obj is a callable object.
            return _function_contents(obj.__call__.__func__)

        except AttributeError:
            try:
                # Test if obj is a code object.
                return _code_contents(obj)

            except AttributeError:
                try:
                    # Test if obj is a function object.
                    return _function_contents(obj)

                except AttributeError as ae:
                    # Should be a pickle-able Python object.
                    try:
                        return _object_instance_content(obj)
                        # pickling an Action instance or object doesn't yield a stable
                        # content as instance property may be dumped in different orders
                        # return pickle.dumps(obj, ACTION_SIGNATURE_PICKLE_PROTOCOL)
                    except (pickle.PicklingError, TypeError, AttributeError) as ex:
                        # This is weird, but it seems that nested classes
                        # are unpickable. The Python docs say it should
                        # always be a PicklingError, but some Python
                        # versions seem to return TypeError.  Just do
                        # the best we can.
                        return bytearray(repr(obj), 'utf-8')


def _code_contents(code, docstring=None):
    """Return the signature contents of a code object.

    By providing direct access to the code object of the
    function, Python makes this extremely easy.  Hooray!

    Unfortunately, older versions of Python include line
    number indications in the compiled byte code.  Boo!
    So we remove the line number byte codes to prevent
    recompilations from moving a Python function.

    See:
      - https://docs.python.org/2/library/inspect.html
      - http://python-reference.readthedocs.io/en/latest/docs/code/index.html

    For info on what each co\_ variable provides

    The signature is as follows (should be byte/chars):
    co_argcount, len(co_varnames), len(co_cellvars), len(co_freevars),
    ( comma separated signature for each object in co_consts ),
    ( comma separated signature for each object in co_names ),
    ( The bytecode with line number bytecodes removed from  co_code )

    co_argcount - Returns the number of positional arguments (including arguments with default values).
    co_varnames - Returns a tuple containing the names of the local variables (starting with the argument names).
    co_cellvars - Returns a tuple containing the names of local variables that are referenced by nested functions.
    co_freevars - Returns a tuple containing the names of free variables. (?)
    co_consts   - Returns a tuple containing the literals used by the bytecode.
    co_names    - Returns a tuple containing the names used by the bytecode.
    co_code     - Returns a string representing the sequence of bytecode instructions.

    """

    # contents = []

    # The code contents depends on the number of local variables
    # but not their actual names.
    contents = bytearray("{}, {}".format(code.co_argcount, len(code.co_varnames)), 'utf-8')

    contents.extend(b", ")
    contents.extend(bytearray(str(len(code.co_cellvars)), 'utf-8'))
    contents.extend(b", ")
    contents.extend(bytearray(str(len(code.co_freevars)), 'utf-8'))

    # The code contents depends on any constants accessed by the
    # function. Note that we have to call _object_contents on each
    # constants because the code object of nested functions can
    # show-up among the constants.

    z = [_object_contents(cc) for cc in code.co_consts[1:]]
    contents.extend(b',(')
    contents.extend(bytearray(',', 'utf-8').join(z))
    contents.extend(b')')

    # The code contents depends on the variable names used to
    # accessed global variable, as changing the variable name changes
    # the variable actually accessed and therefore changes the
    # function result.
    z= [bytearray(_object_contents(cc)) for cc in code.co_names]
    contents.extend(b',(')
    contents.extend(bytearray(',','utf-8').join(z))
    contents.extend(b')')

    # The code contents depends on its actual code!!!
    contents.extend(b',(')
    contents.extend(code.co_code)
    contents.extend(b')')

    return contents


def _function_contents(func):
    """
    The signature is as follows (should be byte/chars):
    < _code_contents (see above) from func.__code__ >
    ,( comma separated _object_contents for function argument defaults)
    ,( comma separated _object_contents for any closure contents )


    See also: https://docs.python.org/3/reference/datamodel.html
      - func.__code__     - The code object representing the compiled function body.
      - func.__defaults__ - A tuple containing default argument values for those arguments that have defaults, or None if no arguments have a default value
      - func.__closure__  - None or a tuple of cells that contain bindings for the function's free variables.

    :Returns:
      Signature contents of a function. (in bytes)
    """

    contents = [_code_contents(func.__code__, func.__doc__)]

    # The function contents depends on the value of defaults arguments
    if func.__defaults__:

        function_defaults_contents = [_object_contents(cc) for cc in func.__defaults__]

        defaults = bytearray(b',(')
        defaults.extend(bytearray(b',').join(function_defaults_contents))
        defaults.extend(b')')

        contents.append(defaults)
    else:
        contents.append(b',()')

    # The function contents depends on the closure captured cell values.
    closure = func.__closure__ or []

    try:
        closure_contents = [_object_contents(x.cell_contents) for x in closure]
    except AttributeError:
        closure_contents = []

    contents.append(b',(')
    contents.append(bytearray(b',').join(closure_contents))
    contents.append(b')')

    retval = bytearray(b'').join(contents)
    return retval


def _object_instance_content(obj):
    """
    Returns consistant content for a action class or an instance thereof

    :Parameters:
      - `obj` Should be either and action class or an instance thereof

    :Returns:
      bytearray or bytes representing the obj suitable for generating a signature from.
    """
    retval = bytearray()

    if obj is None:
        return b'N.'

    if isinstance(obj, SCons.Util.BaseStringTypes):
        return SCons.Util.to_bytes(obj)

    inst_class = obj.__class__
    inst_class_name = bytearray(obj.__class__.__name__,'utf-8')
    inst_class_module = bytearray(obj.__class__.__module__,'utf-8')
    inst_class_hierarchy = bytearray(repr(inspect.getclasstree([obj.__class__,])),'utf-8')
    # print("ICH:%s : %s"%(inst_class_hierarchy, repr(obj)))

    properties = [(p, getattr(obj, p, "None")) for p in dir(obj) if not (p[:2] == '__' or inspect.ismethod(getattr(obj, p)) or inspect.isbuiltin(getattr(obj,p))) ]
    properties.sort()
    properties_str = ','.join(["%s=%s"%(p[0],p[1]) for p in properties])
    properties_bytes = bytearray(properties_str,'utf-8')

    methods = [p for p in dir(obj) if inspect.ismethod(getattr(obj, p))]
    methods.sort()

    method_contents = []
    for m in methods:
        # print("Method:%s"%m)
        v = _function_contents(getattr(obj, m))
        # print("[%s->]V:%s [%s]"%(m,v,type(v)))
        method_contents.append(v)

    retval = bytearray(b'{')
    retval.extend(inst_class_name)
    retval.extend(b":")
    retval.extend(inst_class_module)
    retval.extend(b'}[[')
    retval.extend(inst_class_hierarchy)
    retval.extend(b']]{{')
    retval.extend(bytearray(b",").join(method_contents))
    retval.extend(b"}}{{{")
    retval.extend(properties_bytes)
    retval.extend(b'}}}')
    return retval

    # print("class          :%s"%inst_class)
    # print("class_name     :%s"%inst_class_name)
    # print("class_module   :%s"%inst_class_module)
    # print("Class hier     :\n%s"%pp.pformat(inst_class_hierarchy))
    # print("Inst Properties:\n%s"%pp.pformat(properties))
    # print("Inst Methods   :\n%s"%pp.pformat(methods))

def _actionAppend(act1, act2):
    # This function knows how to slap two actions together.
    # Mainly, it handles ListActions by concatenating into
    # a single ListAction.
    a1 = Action(act1)
    a2 = Action(act2)
    if a1 is None:
        return a2
    if a2 is None:
        return a1
    if isinstance(a1, ListAction):
        if isinstance(a2, ListAction):
            return ListAction(a1.list + a2.list)
        else:
            return ListAction(a1.list + [ a2 ])
    else:
        if isinstance(a2, ListAction):
            return ListAction([ a1 ] + a2.list)
        else:
            return ListAction([ a1, a2 ])


def _do_create_keywords(args, kw):
    """This converts any arguments after the action argument into
    their equivalent keywords and adds them to the kw argument.
    """
    v = kw.get('varlist', ())
    # prevent varlist="FOO" from being interpreted as ['F', 'O', 'O']
    if is_String(v): v = (v,)
    kw['varlist'] = tuple(v)
    if args:
        # turn positional args into equivalent keywords
        cmdstrfunc = args[0]
        if cmdstrfunc is None or is_String(cmdstrfunc):
            kw['cmdstr'] = cmdstrfunc
        elif callable(cmdstrfunc):
            kw['strfunction'] = cmdstrfunc
        else:
            raise SCons.Errors.UserError(
                'Invalid command display variable type. '
                'You must either pass a string or a callback which '
                'accepts (target, source, env) as parameters.')
        if len(args) > 1:
            kw['varlist'] = tuple(SCons.Util.flatten(args[1:])) + kw['varlist']
    if kw.get('strfunction', _null) is not _null \
                      and kw.get('cmdstr', _null) is not _null:
        raise SCons.Errors.UserError(
            'Cannot have both strfunction and cmdstr args to Action()')


def _do_create_action(act, kw):
    """This is the actual "implementation" for the
    Action factory method, below.  This handles the
    fact that passing lists to Action() itself has
    different semantics than passing lists as elements
    of lists.

    The former will create a ListAction, the latter
    will create a CommandAction by converting the inner
    list elements to strings."""

    if isinstance(act, ActionBase):
        return act

    if is_String(act):
        var=SCons.Util.get_environment_var(act)
        if var:
            # This looks like a string that is purely an Environment
            # variable reference, like "$FOO" or "${FOO}".  We do
            # something special here...we lazily evaluate the contents
            # of that Environment variable, so a user could put something
            # like a function or a CommandGenerator in that variable
            # instead of a string.
            return LazyAction(var, kw)
        commands = str(act).split('\n')
        if len(commands) == 1:
            return CommandAction(commands[0], **kw)
        # The list of string commands may include a LazyAction, so we
        # reprocess them via _do_create_list_action.
        return _do_create_list_action(commands, kw)

    if is_List(act):
        return CommandAction(act, **kw)

    if callable(act):
        try:
            gen = kw['generator']
            del kw['generator']
        except KeyError:
            gen = 0
        if gen:
            action_type = CommandGeneratorAction
        else:
            action_type = FunctionAction
        return action_type(act, kw)

    # Catch a common error case with a nice message:
    if isinstance(act, int) or isinstance(act, float):
        raise TypeError("Don't know how to create an Action from a number (%s)"%act)
    # Else fail silently (???)
    return None


def _do_create_list_action(act, kw):
    """A factory for list actions.  Convert the input list into Actions
    and then wrap them in a ListAction."""
    acts = []
    for a in act:
        aa = _do_create_action(a, kw)
        if aa is not None: acts.append(aa)
    if not acts:
        return ListAction([])
    elif len(acts) == 1:
        return acts[0]
    else:
        return ListAction(acts)


def Action(act, *args, **kw):
    """A factory for action objects."""
    # Really simple: the _do_create_* routines do the heavy lifting.
    _do_create_keywords(args, kw)
    if is_List(act):
        return _do_create_list_action(act, kw)
    return _do_create_action(act, kw)


class ActionBase(object):
    """Base class for all types of action objects that can be held by
    other objects (Builders, Executors, etc.)  This provides the
    common methods for manipulating and combining those actions."""

    def __eq__(self, other):
        return self.__dict__ == other

    def no_batch_key(self, env, target, source):
        return None

    batch_key = no_batch_key

    def genstring(self, target, source, env):
        return str(self)

    def get_contents(self, target, source, env):
        result = self.get_presig(target, source, env)

        if not isinstance(result,(bytes, bytearray)):
            result = bytearray("",'utf-8').join([ SCons.Util.to_bytes(r) for r in result ])
        else:
            # Make a copy and put in bytearray, without this the contents returned by get_presig
            # can be changed by the logic below, appending with each call and causing very
            # hard to track down issues...
            result = bytearray(result)

        # At this point everything should be a bytearray

        # This should never happen, as the Action() factory should wrap
        # the varlist, but just in case an action is created directly,
        # we duplicate this check here.
        vl = self.get_varlist(target, source, env)
        if is_String(vl): vl = (vl,)
        for v in vl:
            # do the subst this way to ignore $(...$) parts:
            if isinstance(result, bytearray):
                result.extend(SCons.Util.to_bytes(env.subst_target_source('${'+v+'}', SCons.Subst.SUBST_SIG, target, source)))
            else:
                raise Exception("WE SHOULD NEVER GET HERE result should be bytearray not:%s"%type(result))
                # result.append(SCons.Util.to_bytes(env.subst_target_source('${'+v+'}', SCons.Subst.SUBST_SIG, target, source)))


        if isinstance(result, (bytes,bytearray)):
            return result
        else:
            raise Exception("WE SHOULD NEVER GET HERE - #2 result should be bytearray not:%s" % type(result))
            # return b''.join(result)

    def __add__(self, other):
        return _actionAppend(self, other)

    def __radd__(self, other):
        return _actionAppend(other, self)

    def presub_lines(self, env):
        # CommandGeneratorAction needs a real environment
        # in order to return the proper string here, since
        # it may call LazyAction, which looks up a key
        # in that env.  So we temporarily remember the env here,
        # and CommandGeneratorAction will use this env
        # when it calls its _generate method.
        self.presub_env = env
        lines = str(self).split('\n')
        self.presub_env = None      # don't need this any more
        return lines

    def get_varlist(self, target, source, env, executor=None):
        return self.varlist

    def get_targets(self, env, executor):
        """
        Returns the type of targets ($TARGETS, $CHANGED_TARGETS) used
        by this action.
        """
        return self.targets


class _ActionAction(ActionBase):
    """Base class for actions that create output objects."""
    def __init__(self, cmdstr=_null, strfunction=_null, varlist=(),
                       presub=_null, chdir=None, exitstatfunc=None,
                       batch_key=None, targets='$TARGETS',
                 **kw):
        self.cmdstr = cmdstr
        if strfunction is not _null:
            if strfunction is None:
                self.cmdstr = None
            else:
                self.strfunction = strfunction
        self.varlist = varlist
        self.presub = presub
        self.chdir = chdir
        if not exitstatfunc:
            exitstatfunc = default_exitstatfunc
        self.exitstatfunc = exitstatfunc

        self.targets = targets

        if batch_key:
            if not callable(batch_key):
                # They have set batch_key, but not to their own
                # callable.  The default behavior here will batch
                # *all* targets+sources using this action, separated
                # for each construction environment.
                def default_batch_key(self, env, target, source):
                    return (id(self), id(env))
                batch_key = default_batch_key
            SCons.Util.AddMethod(self, batch_key, 'batch_key')

    def print_cmd_line(self, s, target, source, env):
        """
        In python 3, and in some of our tests, sys.stdout is
        a String io object, and it takes unicode strings only
        In other cases it's a regular Python 2.x file object
        which takes strings (bytes), and if you pass those a
        unicode object they try to decode with 'ascii' codec
        which fails if the cmd line has any hi-bit-set chars.
        This code assumes s is a regular string, but should
        work if it's unicode too.
        """
        try:
            sys.stdout.write(s + u"\n")
        except UnicodeDecodeError:
            sys.stdout.write(s + "\n")

    def __call__(self, target, source, env,
                               exitstatfunc=_null,
                               presub=_null,
                               show=_null,
                               execute=_null,
                               chdir=_null,
                               executor=None):
        if not is_List(target):
            target = [target]
        if not is_List(source):
            source = [source]

        if presub is _null:
            presub = self.presub
            if presub is _null:
                presub = print_actions_presub
        if exitstatfunc is _null: exitstatfunc = self.exitstatfunc
        if show is _null:  show = print_actions
        if execute is _null:  execute = execute_actions
        if chdir is _null: chdir = self.chdir
        save_cwd = None
        if chdir:
            save_cwd = os.getcwd()
            try:
                chdir = str(chdir.get_abspath())
            except AttributeError:
                if not is_String(chdir):
                    if executor:
                        chdir = str(executor.batches[0].targets[0].dir)
                    else:
                        chdir = str(target[0].dir)
        if presub:
            if executor:
                target = executor.get_all_targets()
                source = executor.get_all_sources()
            t = ' and '.join(map(str, target))
            l = '\n  '.join(self.presub_lines(env))
            out = u"Building %s with action:\n  %s\n" % (t, l)
            sys.stdout.write(out)
        cmd = None
        if show and self.strfunction:
            if executor:
                target = executor.get_all_targets()
                source = executor.get_all_sources()
            try:
                cmd = self.strfunction(target, source, env, executor)
            except TypeError:
                cmd = self.strfunction(target, source, env)
            if cmd:
                if chdir:
                    cmd = ('os.chdir(%s)\n' % repr(chdir)) + cmd
                try:
                    get = env.get
                except AttributeError:
                    print_func = self.print_cmd_line
                else:
                    print_func = get('PRINT_CMD_LINE_FUNC')
                    if not print_func:
                        print_func = self.print_cmd_line
                print_func(cmd, target, source, env)
        stat = 0
        if execute:
            if chdir:
                os.chdir(chdir)
            try:
                stat = self.execute(target, source, env, executor=executor)
                if isinstance(stat, SCons.Errors.BuildError):
                    s = exitstatfunc(stat.status)
                    if s:
                        stat.status = s
                    else:
                        stat = s
                else:
                    stat = exitstatfunc(stat)
            finally:
                if save_cwd:
                    os.chdir(save_cwd)
        if cmd and save_cwd:
            print_func('os.chdir(%s)' % repr(save_cwd), target, source, env)

        return stat


def _string_from_cmd_list(cmd_list):
    """Takes a list of command line arguments and returns a pretty
    representation for printing."""
    cl = []
    for arg in map(str, cmd_list):
        if ' ' in arg or '\t' in arg:
            arg = '"' + arg + '"'
        cl.append(arg)
    return ' '.join(cl)

default_ENV = None


def get_default_ENV(env):
    """
    A fiddlin' little function that has an 'import SCons.Environment' which
    can't be moved to the top level without creating an import loop.  Since
    this import creates a local variable named 'SCons', it blocks access to
    the global variable, so we move it here to prevent complaints about local
    variables being used uninitialized.
    """
    global default_ENV
    try:
        return env['ENV']
    except KeyError:
        if not default_ENV:
            import SCons.Environment
            # This is a hideously expensive way to get a default shell
            # environment.  What it really should do is run the platform
            # setup to get the default ENV.  Fortunately, it's incredibly
            # rare for an Environment not to have a shell environment, so
            # we're not going to worry about it overmuch.
            default_ENV = SCons.Environment.Environment()['ENV']
        return default_ENV


def _subproc(scons_env, cmd, error = 'ignore', **kw):
    """Do common setup for a subprocess.Popen() call

    This function is still in draft mode.  We're going to need something like
    it in the long run as more and more places use subprocess, but I'm sure
    it'll have to be tweaked to get the full desired functionality.
    one special arg (so far?), 'error', to tell what to do with exceptions.
    """
    # allow std{in,out,err} to be "'devnull'"
    io = kw.get('stdin')
    if is_String(io) and io == 'devnull':
        kw['stdin'] = open(os.devnull)
    io = kw.get('stdout')
    if is_String(io) and io == 'devnull':
        kw['stdout'] = open(os.devnull, 'w')
    io = kw.get('stderr')
    if is_String(io) and io == 'devnull':
        kw['stderr'] = open(os.devnull, 'w')

    # Figure out what shell environment to use
    ENV = kw.get('env', None)
    if ENV is None: ENV = get_default_ENV(scons_env)

    # Ensure that the ENV values are all strings:
    new_env = {}
    for key, value in ENV.items():
        if is_List(value):
            # If the value is a list, then we assume it is a path list,
            # because that's a pretty common list-like value to stick
            # in an environment variable:
            value = SCons.Util.flatten_sequence(value)
            new_env[key] = os.pathsep.join(map(str, value))
        else:
            # It's either a string or something else.  If it's a string,
            # we still want to call str() because it might be a *Unicode*
            # string, which makes subprocess.Popen() gag.  If it isn't a
            # string or a list, then we just coerce it to a string, which
            # is the proper way to handle Dir and File instances and will
            # produce something reasonable for just about everything else:
            new_env[key] = str(value)
    kw['env'] = new_env

    try:
        return subprocess.Popen(cmd, **kw)
    except EnvironmentError as e:
        if error == 'raise': raise
        # return a dummy Popen instance that only returns error
        class dummyPopen(object):
            def __init__(self, e): self.exception = e
            def communicate(self, input=None): return ('', '')
            def wait(self): return -self.exception.errno
            stdin = None
            class f(object):
                def read(self): return ''
                def readline(self): return ''
                def __iter__(self): return iter(())
            stdout = stderr = f()
        return dummyPopen(e)


class CommandAction(_ActionAction):
    """Class for command-execution actions."""
    def __init__(self, cmd, **kw):
        # Cmd can actually be a list or a single item; if it's a
        # single item it should be the command string to execute; if a
        # list then it should be the words of the command string to
        # execute.  Only a single command should be executed by this
        # object; lists of commands should be handled by embedding
        # these objects in a ListAction object (which the Action()
        # factory above does).  cmd will be passed to
        # Environment.subst_list() for substituting environment
        # variables.
        if SCons.Debug.track_instances: logInstanceCreation(self, 'Action.CommandAction')

        _ActionAction.__init__(self, **kw)
        if is_List(cmd):
            if [c for c in cmd if is_List(c)]:
                raise TypeError("CommandAction should be given only " \
                      "a single command")
        self.cmd_list = cmd

    def __str__(self):
        if is_List(self.cmd_list):
            return ' '.join(map(str, self.cmd_list))
        return str(self.cmd_list)

    def process(self, target, source, env, executor=None):
        if executor:
            result = env.subst_list(self.cmd_list, 0, executor=executor)
        else:
            result = env.subst_list(self.cmd_list, 0, target, source)
        silent = None
        ignore = None
        while True:
            try: c = result[0][0][0]
            except IndexError: c = None
            if c == '@': silent = 1
            elif c == '-': ignore = 1
            else: break
            result[0][0] = result[0][0][1:]
        try:
            if not result[0][0]:
                result[0] = result[0][1:]
        except IndexError:
            pass
        return result, ignore, silent

    def strfunction(self, target, source, env, executor=None):
        if self.cmdstr is None:
            return None
        if self.cmdstr is not _null:
            from SCons.Subst import SUBST_RAW
            if executor:
                c = env.subst(self.cmdstr, SUBST_RAW, executor=executor)
            else:
                c = env.subst(self.cmdstr, SUBST_RAW, target, source)
            if c:
                return c
        cmd_list, ignore, silent = self.process(target, source, env, executor)
        if silent:
            return ''
        return _string_from_cmd_list(cmd_list[0])

    def execute(self, target, source, env, executor=None):
        """Execute a command action.

        This will handle lists of commands as well as individual commands,
        because construction variable substitution may turn a single
        "command" into a list.  This means that this class can actually
        handle lists of commands, even though that's not how we use it
        externally.
        """
        escape_list = SCons.Subst.escape_list
        flatten_sequence = SCons.Util.flatten_sequence

        try:
            shell = env['SHELL']
        except KeyError:
            raise SCons.Errors.UserError('Missing SHELL construction variable.')

        try:
            spawn = env['SPAWN']
        except KeyError:
            raise SCons.Errors.UserError('Missing SPAWN construction variable.')
        else:
            if is_String(spawn):
                spawn = env.subst(spawn, raw=1, conv=lambda x: x)

        escape = env.get('ESCAPE', lambda x: x)

        ENV = get_default_ENV(env)

        # Ensure that the ENV values are all strings:
        for key, value in ENV.items():
            if not is_String(value):
                if is_List(value):
                    # If the value is a list, then we assume it is a
                    # path list, because that's a pretty common list-like
                    # value to stick in an environment variable:
                    value = flatten_sequence(value)
                    ENV[key] = os.pathsep.join(map(str, value))
                else:
                    # If it isn't a string or a list, then we just coerce
                    # it to a string, which is the proper way to handle
                    # Dir and File instances and will produce something
                    # reasonable for just about everything else:
                    ENV[key] = str(value)

        if executor:
            target = executor.get_all_targets()
            source = executor.get_all_sources()
        cmd_list, ignore, silent = self.process(target, list(map(rfile, source)), env, executor)

        # Use len() to filter out any "command" that's zero-length.
        for cmd_line in filter(len, cmd_list):
            # Escape the command line for the interpreter we are using.
            cmd_line = escape_list(cmd_line, escape)
            result = spawn(shell, escape, cmd_line[0], cmd_line, ENV)
            if not ignore and result:
                msg = "Error %s" % result
                return SCons.Errors.BuildError(errstr=msg,
                                               status=result,
                                               action=self,
                                               command=cmd_line)
        return 0

    def get_presig(self, target, source, env, executor=None):
        """Return the signature contents of this action's command line.

        This strips $(-$) and everything in between the string,
        since those parts don't affect signatures.
        """
        from SCons.Subst import SUBST_SIG
        cmd = self.cmd_list
        if is_List(cmd):
            cmd = ' '.join(map(str, cmd))
        else:
            cmd = str(cmd)
        if executor:
            return env.subst_target_source(cmd, SUBST_SIG, executor=executor)
        else:
            return env.subst_target_source(cmd, SUBST_SIG, target, source)

    def get_implicit_deps(self, target, source, env, executor=None):
        icd = env.get('IMPLICIT_COMMAND_DEPENDENCIES', True)
        if is_String(icd) and icd[:1] == '$':
            icd = env.subst(icd)
        if not icd or icd in ('0', 'None'):
            return []
        from SCons.Subst import SUBST_SIG
        if executor:
            cmd_list = env.subst_list(self.cmd_list, SUBST_SIG, executor=executor)
        else:
            cmd_list = env.subst_list(self.cmd_list, SUBST_SIG, target, source)
        res = []
        for cmd_line in cmd_list:
            if cmd_line:
                d = str(cmd_line[0])
                m = strip_quotes.match(d)
                if m:
                    d = m.group(1)
                d = env.WhereIs(d)
                if d:
                    res.append(env.fs.File(d))
        return res


class CommandGeneratorAction(ActionBase):
    """Class for command-generator actions."""
    def __init__(self, generator, kw):
        if SCons.Debug.track_instances: logInstanceCreation(self, 'Action.CommandGeneratorAction')
        self.generator = generator
        self.gen_kw = kw
        self.varlist = kw.get('varlist', ())
        self.targets = kw.get('targets', '$TARGETS')

    def _generate(self, target, source, env, for_signature, executor=None):
        # ensure that target is a list, to make it easier to write
        # generator functions:
        if not is_List(target):
            target = [target]

        if executor:
            target = executor.get_all_targets()
            source = executor.get_all_sources()
        ret = self.generator(target=target,
                             source=source,
                             env=env,
                             for_signature=for_signature)
        gen_cmd = Action(ret, **self.gen_kw)
        if not gen_cmd:
            raise SCons.Errors.UserError("Object returned from command generator: %s cannot be used to create an Action." % repr(ret))
        return gen_cmd

    def __str__(self):
        try:
            env = self.presub_env
        except AttributeError:
            env = None
        if env is None:
            env = SCons.Defaults.DefaultEnvironment()
        act = self._generate([], [], env, 1)
        return str(act)

    def batch_key(self, env, target, source):
        return self._generate(target, source, env, 1).batch_key(env, target, source)

    def genstring(self, target, source, env, executor=None):
        return self._generate(target, source, env, 1, executor).genstring(target, source, env)

    def __call__(self, target, source, env, exitstatfunc=_null, presub=_null,
                 show=_null, execute=_null, chdir=_null, executor=None):
        act = self._generate(target, source, env, 0, executor)
        if act is None:
            raise SCons.Errors.UserError("While building `%s': "
                            "Cannot deduce file extension from source files: %s"
                % (repr(list(map(str, target))), repr(list(map(str, source)))))
        return act(target, source, env, exitstatfunc, presub,
                   show, execute, chdir, executor)

    def get_presig(self, target, source, env, executor=None):
        """Return the signature contents of this action's command line.

        This strips $(-$) and everything in between the string,
        since those parts don't affect signatures.
        """
        return self._generate(target, source, env, 1, executor).get_presig(target, source, env)

    def get_implicit_deps(self, target, source, env, executor=None):
        return self._generate(target, source, env, 1, executor).get_implicit_deps(target, source, env)

    def get_varlist(self, target, source, env, executor=None):
        return self._generate(target, source, env, 1, executor).get_varlist(target, source, env, executor)

    def get_targets(self, env, executor):
        return self._generate(None, None, env, 1, executor).get_targets(env, executor)


class LazyAction(CommandGeneratorAction, CommandAction):
    """
    A LazyAction is a kind of hybrid generator and command action for
    strings of the form "$VAR".  These strings normally expand to other
    strings (think "$CCCOM" to "$CC -c -o $TARGET $SOURCE"), but we also
    want to be able to replace them with functions in the construction
    environment.  Consequently, we want lazy evaluation and creation of
    an Action in the case of the function, but that's overkill in the more
    normal case of expansion to other strings.

    So we do this with a subclass that's both a generator *and*
    a command action.  The overridden methods all do a quick check
    of the construction variable, and if it's a string we just call
    the corresponding CommandAction method to do the heavy lifting.
    If not, then we call the same-named CommandGeneratorAction method.
    The CommandGeneratorAction methods work by using the overridden
    _generate() method, that is, our own way of handling "generation" of
    an action based on what's in the construction variable.
    """

    def __init__(self, var, kw):
        if SCons.Debug.track_instances: logInstanceCreation(self, 'Action.LazyAction')
        CommandAction.__init__(self, '${'+var+'}', **kw)
        self.var = SCons.Util.to_String(var)
        self.gen_kw = kw

    def get_parent_class(self, env):
        c = env.get(self.var)
        if is_String(c) and not '\n' in c:
            return CommandAction
        return CommandGeneratorAction

    def _generate_cache(self, env):
        if env:
            c = env.get(self.var, '')
        else:
            c = ''
        gen_cmd = Action(c, **self.gen_kw)
        if not gen_cmd:
            raise SCons.Errors.UserError("$%s value %s cannot be used to create an Action." % (self.var, repr(c)))
        return gen_cmd

    def _generate(self, target, source, env, for_signature, executor=None):
        return self._generate_cache(env)

    def __call__(self, target, source, env, *args, **kw):
        c = self.get_parent_class(env)
        return c.__call__(self, target, source, env, *args, **kw)

    def get_presig(self, target, source, env):
        c = self.get_parent_class(env)
        return c.get_presig(self, target, source, env)

    def get_varlist(self, target, source, env, executor=None):
        c = self.get_parent_class(env)
        return c.get_varlist(self, target, source, env, executor)


class FunctionAction(_ActionAction):
    """Class for Python function actions."""

    def __init__(self, execfunction, kw):
        if SCons.Debug.track_instances: logInstanceCreation(self, 'Action.FunctionAction')

        self.execfunction = execfunction
        try:
            self.funccontents = _callable_contents(execfunction)
        except AttributeError:
            try:
                # See if execfunction will do the heavy lifting for us.
                self.gc = execfunction.get_contents
            except AttributeError:
                # This is weird, just do the best we can.
                self.funccontents = _object_contents(execfunction)

        _ActionAction.__init__(self, **kw)

    def function_name(self):
        try:
            return self.execfunction.__name__
        except AttributeError:
            try:
                return self.execfunction.__class__.__name__
            except AttributeError:
                return "unknown_python_function"

    def strfunction(self, target, source, env, executor=None):
        if self.cmdstr is None:
            return None
        if self.cmdstr is not _null:
            from SCons.Subst import SUBST_RAW
            if executor:
                c = env.subst(self.cmdstr, SUBST_RAW, executor=executor)
            else:
                c = env.subst(self.cmdstr, SUBST_RAW, target, source)
            if c:
                return c

        def array(a):
            def quote(s):
                try:
                    str_for_display = s.str_for_display
                except AttributeError:
                    s = repr(s)
                else:
                    s = str_for_display()
                return s
            return '[' + ", ".join(map(quote, a)) + ']'
        try:
            strfunc = self.execfunction.strfunction
        except AttributeError:
            pass
        else:
            if strfunc is None:
                return None
            if callable(strfunc):
                return strfunc(target, source, env)
        name = self.function_name()
        tstr = array(target)
        sstr = array(source)
        return "%s(%s, %s)" % (name, tstr, sstr)

    def __str__(self):
        name = self.function_name()
        if name == 'ActionCaller':
            return str(self.execfunction)
        return "%s(target, source, env)" % name

    def execute(self, target, source, env, executor=None):
        exc_info = (None,None,None)
        try:
            if executor:
                target = executor.get_all_targets()
                source = executor.get_all_sources()
            rsources = list(map(rfile, source))
            try:
                result = self.execfunction(target=target, source=rsources, env=env)
            except KeyboardInterrupt as e:
                raise
            except SystemExit as e:
                raise
            except Exception as e:
                result = e
                exc_info = sys.exc_info()

            if result:
                result = SCons.Errors.convert_to_BuildError(result, exc_info)
                result.node=target
                result.action=self
                try:
                    result.command=self.strfunction(target, source, env, executor)
                except TypeError:
                    result.command=self.strfunction(target, source, env)

                # FIXME: This maintains backward compatibility with respect to
                # which type of exceptions were returned by raising an
                # exception and which ones were returned by value. It would
                # probably be best to always return them by value here, but
                # some codes do not check the return value of Actions and I do
                # not have the time to modify them at this point.
                if (exc_info[1] and
                    not isinstance(exc_info[1],EnvironmentError)):
                    raise result

            return result
        finally:
            # Break the cycle between the traceback object and this
            # function stack frame. See the sys.exc_info() doc info for
            # more information about this issue.
            del exc_info

    def get_presig(self, target, source, env):
        """Return the signature contents of this callable action."""
        try:
            return self.gc(target, source, env)
        except AttributeError:
            return self.funccontents

    def get_implicit_deps(self, target, source, env):
        return []

class ListAction(ActionBase):
    """Class for lists of other actions."""
    def __init__(self, actionlist):
        if SCons.Debug.track_instances: logInstanceCreation(self, 'Action.ListAction')
        def list_of_actions(x):
            if isinstance(x, ActionBase):
                return x
            return Action(x)
        self.list = list(map(list_of_actions, actionlist))
        # our children will have had any varlist
        # applied; we don't need to do it again
        self.varlist = ()
        self.targets = '$TARGETS'

    def genstring(self, target, source, env):
        return '\n'.join([a.genstring(target, source, env) for a in self.list])

    def __str__(self):
        return '\n'.join(map(str, self.list))

    def presub_lines(self, env):
        return SCons.Util.flatten_sequence(
            [a.presub_lines(env) for a in self.list])

    def get_presig(self, target, source, env):
        """Return the signature contents of this action list.

        Simple concatenation of the signatures of the elements.
        """
        return b"".join([bytes(x.get_contents(target, source, env)) for x in self.list])

    def __call__(self, target, source, env, exitstatfunc=_null, presub=_null,
                 show=_null, execute=_null, chdir=_null, executor=None):
        if executor:
            target = executor.get_all_targets()
            source = executor.get_all_sources()
        for act in self.list:
            stat = act(target, source, env, exitstatfunc, presub,
                       show, execute, chdir, executor)
            if stat:
                return stat
        return 0

    def get_implicit_deps(self, target, source, env):
        result = []
        for act in self.list:
            result.extend(act.get_implicit_deps(target, source, env))
        return result

    def get_varlist(self, target, source, env, executor=None):
        result = SCons.Util.OrderedDict()
        for act in self.list:
            for var in act.get_varlist(target, source, env, executor):
                result[var] = True
        return list(result.keys())


class ActionCaller(object):
    """A class for delaying calling an Action function with specific
    (positional and keyword) arguments until the Action is actually
    executed.

    This class looks to the rest of the world like a normal Action object,
    but what it's really doing is hanging on to the arguments until we
    have a target, source and env to use for the expansion.
    """
    def __init__(self, parent, args, kw):
        self.parent = parent
        self.args = args
        self.kw = kw

    def get_contents(self, target, source, env):
        actfunc = self.parent.actfunc
        try:
            # "self.actfunc" is a function.
            contents = actfunc.__code__.co_code
        except AttributeError:
            # "self.actfunc" is a callable object.
            try:
                contents = actfunc.__call__.__func__.__code__.co_code
            except AttributeError:
                # No __call__() method, so it might be a builtin
                # or something like that.  Do the best we can.
                contents = repr(actfunc)

        return contents

    def subst(self, s, target, source, env):
        # If s is a list, recursively apply subst()
        # to every element in the list
        if is_List(s):
            result = []
            for elem in s:
                result.append(self.subst(elem, target, source, env))
            return self.parent.convert(result)

        # Special-case hack:  Let a custom function wrapped in an
        # ActionCaller get at the environment through which the action
        # was called by using this hard-coded value as a special return.
        if s == '$__env__':
            return env
        elif is_String(s):
            return env.subst(s, 1, target, source)
        return self.parent.convert(s)

    def subst_args(self, target, source, env):
        return [self.subst(x, target, source, env) for x in self.args]

    def subst_kw(self, target, source, env):
        kw = {}
        for key in list(self.kw.keys()):
            kw[key] = self.subst(self.kw[key], target, source, env)
        return kw

    def __call__(self, target, source, env, executor=None):
        args = self.subst_args(target, source, env)
        kw = self.subst_kw(target, source, env)
        return self.parent.actfunc(*args, **kw)

    def strfunction(self, target, source, env):
        args = self.subst_args(target, source, env)
        kw = self.subst_kw(target, source, env)
        return self.parent.strfunc(*args, **kw)

    def __str__(self):
        return self.parent.strfunc(*self.args, **self.kw)


class ActionFactory(object):
    """A factory class that will wrap up an arbitrary function
    as an SCons-executable Action object.

    The real heavy lifting here is done by the ActionCaller class.
    We just collect the (positional and keyword) arguments that we're
    called with and give them to the ActionCaller object we create,
    so it can hang onto them until it needs them.
    """
    def __init__(self, actfunc, strfunc, convert=lambda x: x):
        self.actfunc = actfunc
        self.strfunc = strfunc
        self.convert = convert

    def __call__(self, *args, **kw):
        ac = ActionCaller(self, args, kw)
        action = Action(ac, strfunction=ac.strfunction)
        return action

# Local Variables:
# tab-width:4
# indent-tabs-mode:nil
# End:
# vim: set expandtab tabstop=4 shiftwidth=4:
