"""SCons.Node

The Node package for the SCons software construction utility.

This is, in many ways, the heart of SCons.

A Node is where we encapsulate all of the dependency information about
any thing that SCons can build, or about any thing which SCons can use
to build some other thing.  The canonical "thing," of course, is a file,
but a Node can also represent something remote (like a web page) or
something completely abstract (like an Alias).

Each specific type of "thing" is specifically represented by a subclass
of the Node base class:  Node.FS.File for files, Node.Alias for aliases,
etc.  Dependency information is kept here in the base class, and
information specific to files/aliases/etc. is in the subclass.  The
goal, if we've done this correctly, is that any type of "thing" should
be able to depend on any other type of "thing."

"""

from __future__ import print_function

#
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

__revision__ = "src/engine/SCons/Node/__init__.py rel_3.0.0:4395:8972f6a2f699 2017/09/18 12:59:24 bdbaddog"

import collections
import copy
from itertools import chain

import SCons.Debug
from SCons.Debug import logInstanceCreation
import SCons.Executor
import SCons.Memoize
import SCons.Util

from SCons.Debug import Trace

from SCons.compat import with_metaclass, NoSlotsPyPy

print_duplicate = 0

def classname(obj):
    return str(obj.__class__).split('.')[-1]

# Set to false if we're doing a dry run. There's more than one of these
# little treats
do_store_info = True

# Node states
#
# These are in "priority" order, so that the maximum value for any
# child/dependency of a node represents the state of that node if
# it has no builder of its own.  The canonical example is a file
# system directory, which is only up to date if all of its children
# were up to date.
no_state = 0
pending = 1
executing = 2
up_to_date = 3
executed = 4
failed = 5

StateString = {
    0 : "no_state",
    1 : "pending",
    2 : "executing",
    3 : "up_to_date",
    4 : "executed",
    5 : "failed",
}

# controls whether implicit dependencies are cached:
implicit_cache = 0

# controls whether implicit dep changes are ignored:
implicit_deps_unchanged = 0

# controls whether the cached implicit deps are ignored:
implicit_deps_changed = 0

# A variable that can be set to an interface-specific function be called
# to annotate a Node with information about its creation.
def do_nothing(node): pass

Annotate = do_nothing

# Gets set to 'True' if we're running in interactive mode. Is
# currently used to release parts of a target's info during
# clean builds and update runs (see release_target_info).
interactive = False

def is_derived_none(node):
    raise NotImplementedError

def is_derived_node(node):
    """
        Returns true if this node is derived (i.e. built).
    """
    return node.has_builder() or node.side_effect

_is_derived_map = {0 : is_derived_none,
                   1 : is_derived_node}

def exists_none(node):
    raise NotImplementedError

def exists_always(node):
    return 1

def exists_base(node):
    return node.stat() is not None

def exists_entry(node):
    """Return if the Entry exists.  Check the file system to see
    what we should turn into first.  Assume a file if there's no
    directory."""
    node.disambiguate()
    return _exists_map[node._func_exists](node)

def exists_file(node):
    # Duplicate from source path if we are set up to do this.
    if node.duplicate and not node.is_derived() and not node.linked:
        src = node.srcnode()
        if src is not node:
            # At this point, src is meant to be copied in a variant directory.
            src = src.rfile()
            if src.get_abspath() != node.get_abspath():
                if src.exists():
                    node.do_duplicate(src)
                    # Can't return 1 here because the duplication might
                    # not actually occur if the -n option is being used.
                else:
                    # The source file does not exist.  Make sure no old
                    # copy remains in the variant directory.
                    if print_duplicate:
                        print("dup: no src for %s, unlinking old variant copy"%self)
                    if exists_base(node) or node.islink():
                        node.fs.unlink(node.get_internal_path())
                    # Return None explicitly because the Base.exists() call
                    # above will have cached its value if the file existed.
                    return None
    return exists_base(node)

_exists_map = {0 : exists_none,
               1 : exists_always,
               2 : exists_base,
               3 : exists_entry,
               4 : exists_file}


def rexists_none(node):
    raise NotImplementedError

def rexists_node(node):
    return node.exists()

def rexists_base(node):
    return node.rfile().exists()

_rexists_map = {0 : rexists_none,
                1 : rexists_node,
                2 : rexists_base}

def get_contents_none(node):
    raise NotImplementedError

def get_contents_entry(node):
    """Fetch the contents of the entry.  Returns the exact binary
    contents of the file."""
    try:
        node = node.disambiguate(must_exist=1)
    except SCons.Errors.UserError:
        # There was nothing on disk with which to disambiguate
        # this entry.  Leave it as an Entry, but return a null
        # string so calls to get_contents() in emitters and the
        # like (e.g. in qt.py) don't have to disambiguate by hand
        # or catch the exception.
        return ''
    else:
        return _get_contents_map[node._func_get_contents](node)

def get_contents_dir(node):
    """Return content signatures and names of all our children
    separated by new-lines. Ensure that the nodes are sorted."""
    contents = []
    for n in sorted(node.children(), key=lambda t: t.name):
        contents.append('%s %s\n' % (n.get_csig(), n.name))
    return ''.join(contents)

def get_contents_file(node):
    if not node.rexists():
        return b''
    fname = node.rfile().get_abspath()
    try:
        with open(fname, "rb") as fp:
            contents = fp.read()
    except EnvironmentError as e:
        if not e.filename:
            e.filename = fname
        raise
    return contents

_get_contents_map = {0 : get_contents_none,
                     1 : get_contents_entry,
                     2 : get_contents_dir,
                     3 : get_contents_file}

def target_from_source_none(node, prefix, suffix, splitext):
    raise NotImplementedError

def target_from_source_base(node, prefix, suffix, splitext):
    return node.dir.Entry(prefix + splitext(node.name)[0] + suffix)

_target_from_source_map = {0 : target_from_source_none,
                           1 : target_from_source_base}

#
# The new decider subsystem for Nodes
#
# We would set and overwrite the changed_since_last_build function
# before, but for being able to use slots (less memory!) we now have
# a dictionary of the different decider functions. Then in the Node
# subclasses we simply store the index to the decider that should be
# used by it.
#

#
# First, the single decider functions
#
def changed_since_last_build_node(node, target, prev_ni):
    """

    Must be overridden in a specific subclass to return True if this
    Node (a dependency) has changed since the last time it was used
    to build the specified target.  prev_ni is this Node's state (for
    example, its file timestamp, length, maybe content signature)
    as of the last time the target was built.

    Note that this method is called through the dependency, not the
    target, because a dependency Node must be able to use its own
    logic to decide if it changed.  For example, File Nodes need to
    obey if we're configured to use timestamps, but Python Value Nodes
    never use timestamps and always use the content.  If this method
    were called through the target, then each Node's implementation
    of this method would have to have more complicated logic to
    handle all the different Node types on which it might depend.
    """
    raise NotImplementedError

def changed_since_last_build_alias(node, target, prev_ni):
    cur_csig = node.get_csig()
    try:
        return cur_csig != prev_ni.csig
    except AttributeError:
        return 1

def changed_since_last_build_entry(node, target, prev_ni):
    node.disambiguate()
    return _decider_map[node.changed_since_last_build](node, target, prev_ni)

def changed_since_last_build_state_changed(node, target, prev_ni):
    return (node.state != SCons.Node.up_to_date)

def decide_source(node, target, prev_ni):
    return target.get_build_env().decide_source(node, target, prev_ni)

def decide_target(node, target, prev_ni):
    return target.get_build_env().decide_target(node, target, prev_ni)

def changed_since_last_build_python(node, target, prev_ni):
    cur_csig = node.get_csig()
    try:
        return cur_csig != prev_ni.csig
    except AttributeError:
        return 1


#
# Now, the mapping from indices to decider functions
#
_decider_map = {0 : changed_since_last_build_node,
               1 : changed_since_last_build_alias,
               2 : changed_since_last_build_entry,
               3 : changed_since_last_build_state_changed,
               4 : decide_source,
               5 : decide_target,
               6 : changed_since_last_build_python}

do_store_info = True

#
# The new store_info subsystem for Nodes
#
# We would set and overwrite the store_info function
# before, but for being able to use slots (less memory!) we now have
# a dictionary of the different functions. Then in the Node
# subclasses we simply store the index to the info method that should be
# used by it.
#

#
# First, the single info functions
#

def store_info_pass(node):
    pass

def store_info_file(node):
    # Merge our build information into the already-stored entry.
    # This accommodates "chained builds" where a file that's a target
    # in one build (SConstruct file) is a source in a different build.
    # See test/chained-build.py for the use case.
    if do_store_info:
        node.dir.sconsign().store_info(node.name, node)


store_info_map = {0 : store_info_pass,
                  1 : store_info_file}

# Classes for signature info for Nodes.

class NodeInfoBase(object):
    """
    The generic base class for signature information for a Node.

    Node subclasses should subclass NodeInfoBase to provide their own
    logic for dealing with their own Node-specific signature information.
    """
    __slots__ = ('__weakref__',)
    current_version_id = 2

    def update(self, node):
        try:
            field_list = self.field_list
        except AttributeError:
            return
        for f in field_list:
            try:
                delattr(self, f)
            except AttributeError:
                pass
            try:
                func = getattr(node, 'get_' + f)
            except AttributeError:
                pass
            else:
                setattr(self, f, func())

    def convert(self, node, val):
        pass

    def merge(self, other):
        """
        Merge the fields of another object into this object. Already existing
        information is overwritten by the other instance's data.
        WARNING: If a '__dict__' slot is added, it should be updated instead of
        replaced.
        """
        state = other.__getstate__()
        self.__setstate__(state)
    def format(self, field_list=None, names=0):
        if field_list is None:
            try:
                field_list = self.field_list
            except AttributeError:
                field_list = list(getattr(self, '__dict__', {}).keys())
                for obj in type(self).mro():
                    for slot in getattr(obj, '__slots__', ()):
                        if slot not in ('__weakref__', '__dict__'):
                            field_list.append(slot)
                field_list.sort()
        fields = []
        for field in field_list:
            try:
                f = getattr(self, field)
            except AttributeError:
                f = None
            f = str(f)
            if names:
                f = field + ': ' + f
            fields.append(f)
        return fields

    def __getstate__(self):
        """
        Return all fields that shall be pickled. Walk the slots in the class
        hierarchy and add those to the state dictionary. If a '__dict__' slot is
        available, copy all entries to the dictionary. Also include the version
        id, which is fixed for all instances of a class.
        """
        state = getattr(self, '__dict__', {}).copy()
        for obj in type(self).mro():
            for name in getattr(obj,'__slots__',()):
                if hasattr(self, name):
                    state[name] = getattr(self, name)

        state['_version_id'] = self.current_version_id
        try:
            del state['__weakref__']
        except KeyError:
            pass
        return state

    def __setstate__(self, state):
        """
        Restore the attributes from a pickled state. The version is discarded.
        """
        # TODO check or discard version
        del state['_version_id']

        for key, value in state.items():
            if key not in ('__weakref__',):
                setattr(self, key, value)


class BuildInfoBase(object):
    """
    The generic base class for build information for a Node.

    This is what gets stored in a .sconsign file for each target file.
    It contains a NodeInfo instance for this node (signature information
    that's specific to the type of Node) and direct attributes for the
    generic build stuff we have to track:  sources, explicit dependencies,
    implicit dependencies, and action information.
    """
    __slots__ = ("bsourcesigs", "bdependsigs", "bimplicitsigs", "bactsig",
                 "bsources", "bdepends", "bact", "bimplicit", "__weakref__")
    current_version_id = 2

    def __init__(self):
        # Create an object attribute from the class attribute so it ends up
        # in the pickled data in the .sconsign file.
        self.bsourcesigs = []
        self.bdependsigs = []
        self.bimplicitsigs = []
        self.bactsig = None

    def merge(self, other):
        """
        Merge the fields of another object into this object. Already existing
        information is overwritten by the other instance's data.
        WARNING: If a '__dict__' slot is added, it should be updated instead of
        replaced.
        """
        state = other.__getstate__()
        self.__setstate__(state)

    def __getstate__(self):
        """
        Return all fields that shall be pickled. Walk the slots in the class
        hierarchy and add those to the state dictionary. If a '__dict__' slot is
        available, copy all entries to the dictionary. Also include the version
        id, which is fixed for all instances of a class.
        """
        state = getattr(self, '__dict__', {}).copy()
        for obj in type(self).mro():
            for name in getattr(obj,'__slots__',()):
                if hasattr(self, name):
                    state[name] = getattr(self, name)

        state['_version_id'] = self.current_version_id
        try:
            del state['__weakref__']
        except KeyError:
            pass
        return state

    def __setstate__(self, state):
        """
        Restore the attributes from a pickled state.
        """
        # TODO check or discard version
        del state['_version_id']
        for key, value in state.items():
            if key not in ('__weakref__',):
                setattr(self, key, value)


class Node(object, with_metaclass(NoSlotsPyPy)):
    """The base Node class, for entities that we know how to
    build, or use to build other Nodes.
    """

    __slots__ = ['sources',
                 'sources_set',
                 '_specific_sources',
                 'depends',
                 'depends_set',
                 'ignore',
                 'ignore_set',
                 'prerequisites',
                 'implicit',
                 'waiting_parents',
                 'waiting_s_e',
                 'ref_count',
                 'wkids',
                 'env',
                 'state',
                 'precious',
                 'noclean',
                 'nocache',
                 'cached',
                 'always_build',
                 'includes',
                 'attributes',
                 'side_effect',
                 'side_effects',
                 'linked',
                 '_memo',
                 'executor',
                 'binfo',
                 'ninfo',
                 'builder',
                 'is_explicit',
                 'implicit_set',
                 'changed_since_last_build',
                 'store_info',
                 'pseudo',
                 '_tags',
                 '_func_is_derived',
                 '_func_exists',
                 '_func_rexists',
                 '_func_get_contents',
                 '_func_target_from_source']

    class Attrs(object):
        __slots__ = ('shared', '__dict__')


    def __init__(self):
        if SCons.Debug.track_instances: logInstanceCreation(self, 'Node.Node')
        # Note that we no longer explicitly initialize a self.builder
        # attribute to None here.  That's because the self.builder
        # attribute may be created on-the-fly later by a subclass (the
        # canonical example being a builder to fetch a file from a
        # source code system like CVS or Subversion).

        # Each list of children that we maintain is accompanied by a
        # dictionary used to look up quickly whether a node is already
        # present in the list.  Empirical tests showed that it was
        # fastest to maintain them as side-by-side Node attributes in
        # this way, instead of wrapping up each list+dictionary pair in
        # a class.  (Of course, we could always still do that in the
        # future if we had a good reason to...).
        self.sources = []       # source files used to build node
        self.sources_set = set()
        self._specific_sources = False
        self.depends = []       # explicit dependencies (from Depends)
        self.depends_set = set()
        self.ignore = []        # dependencies to ignore
        self.ignore_set = set()
        self.prerequisites = None
        self.implicit = None    # implicit (scanned) dependencies (None means not scanned yet)
        self.waiting_parents = set()
        self.waiting_s_e = set()
        self.ref_count = 0
        self.wkids = None       # Kids yet to walk, when it's an array

        self.env = None
        self.state = no_state
        self.precious = None
        self.pseudo = False
        self.noclean = 0
        self.nocache = 0
        self.cached = 0 # is this node pulled from cache?
        self.always_build = None
        self.includes = None
        self.attributes = self.Attrs() # Generic place to stick information about the Node.
        self.side_effect = 0 # true iff this node is a side effect
        self.side_effects = [] # the side effects of building this target
        self.linked = 0 # is this node linked to the variant directory?
        self.changed_since_last_build = 0
        self.store_info = 0
        self._tags = None
        self._func_is_derived = 1
        self._func_exists = 1
        self._func_rexists = 1
        self._func_get_contents = 0
        self._func_target_from_source = 0

        self.clear_memoized_values()

        # Let the interface in which the build engine is embedded
        # annotate this Node with its own info (like a description of
        # what line in what file created the node, for example).
        Annotate(self)

    def disambiguate(self, must_exist=None):
        return self

    def get_suffix(self):
        return ''

    @SCons.Memoize.CountMethodCall
    def get_build_env(self):
        """Fetch the appropriate Environment to build this node.
        """
        try:
            return self._memo['get_build_env']
        except KeyError:
            pass
        result = self.get_executor().get_build_env()
        self._memo['get_build_env'] = result
        return result

    def get_build_scanner_path(self, scanner):
        """Fetch the appropriate scanner path for this node."""
        return self.get_executor().get_build_scanner_path(scanner)

    def set_executor(self, executor):
        """Set the action executor for this node."""
        self.executor = executor

    def get_executor(self, create=1):
        """Fetch the action executor for this node.  Create one if
        there isn't already one, and requested to do so."""
        try:
            executor = self.executor
        except AttributeError:
            if not create:
                raise
            try:
                act = self.builder.action
            except AttributeError:
                executor = SCons.Executor.Null(targets=[self])
            else:
                executor = SCons.Executor.Executor(act,
                                                   self.env or self.builder.env,
                                                   [self.builder.overrides],
                                                   [self],
                                                   self.sources)
            self.executor = executor
        return executor

    def executor_cleanup(self):
        """Let the executor clean up any cached information."""
        try:
            executor = self.get_executor(create=None)
        except AttributeError:
            pass
        else:
            if executor is not None:
                executor.cleanup()

    def reset_executor(self):
        "Remove cached executor; forces recompute when needed."
        try:
            delattr(self, 'executor')
        except AttributeError:
            pass

    def push_to_cache(self):
        """Try to push a node into a cache
        """
        pass

    def retrieve_from_cache(self):
        """Try to retrieve the node's content from a cache

        This method is called from multiple threads in a parallel build,
        so only do thread safe stuff here. Do thread unsafe stuff in
        built().

        Returns true if the node was successfully retrieved.
        """
        return 0

    #
    # Taskmaster interface subsystem
    #

    def make_ready(self):
        """Get a Node ready for evaluation.

        This is called before the Taskmaster decides if the Node is
        up-to-date or not.  Overriding this method allows for a Node
        subclass to be disambiguated if necessary, or for an implicit
        source builder to be attached.
        """
        pass

    def prepare(self):
        """Prepare for this Node to be built.

        This is called after the Taskmaster has decided that the Node
        is out-of-date and must be rebuilt, but before actually calling
        the method to build the Node.

        This default implementation checks that explicit or implicit
        dependencies either exist or are derived, and initializes the
        BuildInfo structure that will hold the information about how
        this node is, uh, built.

        (The existence of source files is checked separately by the
        Executor, which aggregates checks for all of the targets built
        by a specific action.)

        Overriding this method allows for for a Node subclass to remove
        the underlying file from the file system.  Note that subclass
        methods should call this base class method to get the child
        check and the BuildInfo structure.
        """
        if self.depends is not None:
            for d in self.depends:
                if d.missing():
                    msg = "Explicit dependency `%s' not found, needed by target `%s'."
                    raise SCons.Errors.StopError(msg % (d, self))
        if self.implicit is not None:
            for i in self.implicit:
                if i.missing():
                    msg = "Implicit dependency `%s' not found, needed by target `%s'."
                    raise SCons.Errors.StopError(msg % (i, self))
        self.binfo = self.get_binfo()

    def build(self, **kw):
        """Actually build the node.

        This is called by the Taskmaster after it's decided that the
        Node is out-of-date and must be rebuilt, and after the prepare()
        method has gotten everything, uh, prepared.

        This method is called from multiple threads in a parallel build,
        so only do thread safe stuff here. Do thread unsafe stuff
        in built().

        """
        try:
            self.get_executor()(self, **kw)
        except SCons.Errors.BuildError as e:
            e.node = self
            raise

    def built(self):
        """Called just after this node is successfully built."""

        # Clear the implicit dependency caches of any Nodes
        # waiting for this Node to be built.
        for parent in self.waiting_parents:
            parent.implicit = None

        self.clear()

        if self.pseudo:
            if self.exists():
                raise SCons.Errors.UserError("Pseudo target " + str(self) + " must not exist")
        else:
            if not self.exists() and do_store_info:
                SCons.Warnings.warn(SCons.Warnings.TargetNotBuiltWarning,
                                    "Cannot find target " + str(self) + " after building")
        self.ninfo.update(self)

    def visited(self):
        """Called just after this node has been visited (with or
        without a build)."""
        try:
            binfo = self.binfo
        except AttributeError:
            # Apparently this node doesn't need build info, so
            # don't bother calculating or storing it.
            pass
        else:
            self.ninfo.update(self)
            SCons.Node.store_info_map[self.store_info](self)

    def release_target_info(self):
        """Called just after this node has been marked
         up-to-date or was built completely.

         This is where we try to release as many target node infos
         as possible for clean builds and update runs, in order
         to minimize the overall memory consumption.

         By purging attributes that aren't needed any longer after
         a Node (=File) got built, we don't have to care that much how
         many KBytes a Node actually requires...as long as we free
         the memory shortly afterwards.

         @see: built() and File.release_target_info()
         """
        pass

    #
    #
    #

    def add_to_waiting_s_e(self, node):
        self.waiting_s_e.add(node)

    def add_to_waiting_parents(self, node):
        """
        Returns the number of nodes added to our waiting parents list:
        1 if we add a unique waiting parent, 0 if not.  (Note that the
        returned values are intended to be used to increment a reference
        count, so don't think you can "clean up" this function by using
        True and False instead...)
        """
        wp = self.waiting_parents
        if node in wp:
            return 0
        wp.add(node)
        return 1

    def postprocess(self):
        """Clean up anything we don't need to hang onto after we've
        been built."""
        self.executor_cleanup()
        self.waiting_parents = set()

    def clear(self):
        """Completely clear a Node of all its cached state (so that it
        can be re-evaluated by interfaces that do continuous integration
        builds).
        """
        # The del_binfo() call here isn't necessary for normal execution,
        # but is for interactive mode, where we might rebuild the same
        # target and need to start from scratch.
        self.del_binfo()
        self.clear_memoized_values()
        self.ninfo = self.new_ninfo()
        self.executor_cleanup()
        try:
            delattr(self, '_calculated_sig')
        except AttributeError:
            pass
        self.includes = None

    def clear_memoized_values(self):
        self._memo = {}

    def builder_set(self, builder):
        self.builder = builder
        try:
            del self.executor
        except AttributeError:
            pass

    def has_builder(self):
        """Return whether this Node has a builder or not.

        In Boolean tests, this turns out to be a *lot* more efficient
        than simply examining the builder attribute directly ("if
        node.builder: ..."). When the builder attribute is examined
        directly, it ends up calling __getattr__ for both the __len__
        and __nonzero__ attributes on instances of our Builder Proxy
        class(es), generating a bazillion extra calls and slowing
        things down immensely.
        """
        try:
            b = self.builder
        except AttributeError:
            # There was no explicit builder for this Node, so initialize
            # the self.builder attribute to None now.
            b = self.builder = None
        return b is not None

    def set_explicit(self, is_explicit):
        self.is_explicit = is_explicit

    def has_explicit_builder(self):
        """Return whether this Node has an explicit builder

        This allows an internal Builder created by SCons to be marked
        non-explicit, so that it can be overridden by an explicit
        builder that the user supplies (the canonical example being
        directories)."""
        try:
            return self.is_explicit
        except AttributeError:
            self.is_explicit = None
            return self.is_explicit

    def get_builder(self, default_builder=None):
        """Return the set builder, or a specified default value"""
        try:
            return self.builder
        except AttributeError:
            return default_builder

    multiple_side_effect_has_builder = has_builder

    def is_derived(self):
        """
        Returns true if this node is derived (i.e. built).

        This should return true only for nodes whose path should be in
        the variant directory when duplicate=0 and should contribute their build
        signatures when they are used as source files to other derived files. For
        example: source with source builders are not derived in this sense,
        and hence should not return true.
        """
        return _is_derived_map[self._func_is_derived](self)

    def alter_targets(self):
        """Return a list of alternate targets for this Node.
        """
        return [], None

    def get_found_includes(self, env, scanner, path):
        """Return the scanned include lines (implicit dependencies)
        found in this node.

        The default is no implicit dependencies.  We expect this method
        to be overridden by any subclass that can be scanned for
        implicit dependencies.
        """
        return []

    def get_implicit_deps(self, env, initial_scanner, path_func, kw = {}):
        """Return a list of implicit dependencies for this node.

        This method exists to handle recursive invocation of the scanner
        on the implicit dependencies returned by the scanner, if the
        scanner's recursive flag says that we should.
        """
        nodes = [self]
        seen = set(nodes)
        dependencies = []
        path_memo = {}

        root_node_scanner = self._get_scanner(env, initial_scanner, None, kw)

        while nodes:
            node = nodes.pop(0)

            scanner = node._get_scanner(env, initial_scanner, root_node_scanner, kw)
            if not scanner:
                continue

            try:
                path = path_memo[scanner]
            except KeyError:
                path = path_func(scanner)
                path_memo[scanner] = path

            included_deps = [x for x in node.get_found_includes(env, scanner, path) if x not in seen]
            if included_deps:
                dependencies.extend(included_deps)
                seen.update(included_deps)
                nodes.extend(scanner.recurse_nodes(included_deps))

        return dependencies

    def _get_scanner(self, env, initial_scanner, root_node_scanner, kw):
        if initial_scanner:
            # handle explicit scanner case
            scanner = initial_scanner.select(self)
        else:
            # handle implicit scanner case
            scanner = self.get_env_scanner(env, kw)
            if scanner:
                scanner = scanner.select(self)

        if not scanner:
            # no scanner could be found for the given node's scanner key;
            # thus, make an attempt at using a default.
            scanner = root_node_scanner

        return scanner

    def get_env_scanner(self, env, kw={}):
        return env.get_scanner(self.scanner_key())

    def get_target_scanner(self):
        return self.builder.target_scanner

    def get_source_scanner(self, node):
        """Fetch the source scanner for the specified node

        NOTE:  "self" is the target being built, "node" is
        the source file for which we want to fetch the scanner.

        Implies self.has_builder() is true; again, expect to only be
        called from locations where this is already verified.

        This function may be called very often; it attempts to cache
        the scanner found to improve performance.
        """
        scanner = None
        try:
            scanner = self.builder.source_scanner
        except AttributeError:
            pass
        if not scanner:
            # The builder didn't have an explicit scanner, so go look up
            # a scanner from env['SCANNERS'] based on the node's scanner
            # key (usually the file extension).
            scanner = self.get_env_scanner(self.get_build_env())
        if scanner:
            scanner = scanner.select(node)
        return scanner

    def add_to_implicit(self, deps):
        if not hasattr(self, 'implicit') or self.implicit is None:
            self.implicit = []
            self.implicit_set = set()
            self._children_reset()
        self._add_child(self.implicit, self.implicit_set, deps)

    def scan(self):
        """Scan this node's dependents for implicit dependencies."""
        # Don't bother scanning non-derived files, because we don't
        # care what their dependencies are.
        # Don't scan again, if we already have scanned.
        if self.implicit is not None:
            return
        self.implicit = []
        self.implicit_set = set()
        self._children_reset()
        if not self.has_builder():
            return

        build_env = self.get_build_env()
        executor = self.get_executor()

        # Here's where we implement --implicit-cache.
        if implicit_cache and not implicit_deps_changed:
            implicit = self.get_stored_implicit()
            if implicit is not None:
                # We now add the implicit dependencies returned from the
                # stored .sconsign entry to have already been converted
                # to Nodes for us.  (We used to run them through a
                # source_factory function here.)

                # Update all of the targets with them.  This
                # essentially short-circuits an N*M scan of the
                # sources for each individual target, which is a hell
                # of a lot more efficient.
                for tgt in executor.get_all_targets():
                    tgt.add_to_implicit(implicit)

                if implicit_deps_unchanged or self.is_up_to_date():
                    return
                # one of this node's sources has changed,
                # so we must recalculate the implicit deps for all targets
                for tgt in executor.get_all_targets():
                    tgt.implicit = []
                    tgt.implicit_set = set()

        # Have the executor scan the sources.
        executor.scan_sources(self.builder.source_scanner)

        # If there's a target scanner, have the executor scan the target
        # node itself and associated targets that might be built.
        scanner = self.get_target_scanner()
        if scanner:
            executor.scan_targets(scanner)

    def scanner_key(self):
        return None

    def select_scanner(self, scanner):
        """Selects a scanner for this Node.

        This is a separate method so it can be overridden by Node
        subclasses (specifically, Node.FS.Dir) that *must* use their
        own Scanner and don't select one the Scanner.Selector that's
        configured for the target.
        """
        return scanner.select(self)

    def env_set(self, env, safe=0):
        if safe and self.env:
            return
        self.env = env

    #
    # SIGNATURE SUBSYSTEM
    #

    NodeInfo = NodeInfoBase
    BuildInfo = BuildInfoBase

    def new_ninfo(self):
        ninfo = self.NodeInfo()
        return ninfo

    def get_ninfo(self):
        try:
            return self.ninfo
        except AttributeError:
            self.ninfo = self.new_ninfo()
            return self.ninfo

    def new_binfo(self):
        binfo = self.BuildInfo()
        return binfo

    def get_binfo(self):
        """
        Fetch a node's build information.

        node - the node whose sources will be collected
        cache - alternate node to use for the signature cache
        returns - the build signature

        This no longer handles the recursive descent of the
        node's children's signatures.  We expect that they're
        already built and updated by someone else, if that's
        what's wanted.
        """
        try:
            return self.binfo
        except AttributeError:
            pass

        binfo = self.new_binfo()
        self.binfo = binfo

        executor = self.get_executor()
        ignore_set = self.ignore_set

        if self.has_builder():
            binfo.bact = str(executor)
            binfo.bactsig = SCons.Util.MD5signature(executor.get_contents())

        if self._specific_sources:
            sources = [ s for s in self.sources if not s in ignore_set]

        else:
            sources = executor.get_unignored_sources(self, self.ignore)

        seen = set()
        binfo.bsources = [s for s in sources if s not in seen and not seen.add(s)]
        binfo.bsourcesigs = [s.get_ninfo() for s in binfo.bsources]


        binfo.bdepends = self.depends
        binfo.bdependsigs = [d.get_ninfo() for d in self.depends if d not in ignore_set]

        binfo.bimplicit = self.implicit or []
        binfo.bimplicitsigs = [i.get_ninfo() for i in binfo.bimplicit if i not in ignore_set]


        return binfo

    def del_binfo(self):
        """Delete the build info from this node."""
        try:
            delattr(self, 'binfo')
        except AttributeError:
            pass

    def get_csig(self):
        try:
            return self.ninfo.csig
        except AttributeError:
            ninfo = self.get_ninfo()
            ninfo.csig = SCons.Util.MD5signature(self.get_contents())
            return self.ninfo.csig

    def get_cachedir_csig(self):
        return self.get_csig()

    def get_stored_info(self):
        return None

    def get_stored_implicit(self):
        """Fetch the stored implicit dependencies"""
        return None

    #
    #
    #

    def set_precious(self, precious = 1):
        """Set the Node's precious value."""
        self.precious = precious

    def set_pseudo(self, pseudo = True):
        """Set the Node's precious value."""
        self.pseudo = pseudo

    def set_noclean(self, noclean = 1):
        """Set the Node's noclean value."""
        # Make sure noclean is an integer so the --debug=stree
        # output in Util.py can use it as an index.
        self.noclean = noclean and 1 or 0

    def set_nocache(self, nocache = 1):
        """Set the Node's nocache value."""
        # Make sure nocache is an integer so the --debug=stree
        # output in Util.py can use it as an index.
        self.nocache = nocache and 1 or 0

    def set_always_build(self, always_build = 1):
        """Set the Node's always_build value."""
        self.always_build = always_build

    def exists(self):
        """Does this node exists?"""
        return _exists_map[self._func_exists](self)

    def rexists(self):
        """Does this node exist locally or in a repositiory?"""
        # There are no repositories by default:
        return _rexists_map[self._func_rexists](self)

    def get_contents(self):
        """Fetch the contents of the entry."""
        return _get_contents_map[self._func_get_contents](self)

    def missing(self):
        return not self.is_derived() and \
               not self.linked and \
               not self.rexists()

    def remove(self):
        """Remove this Node:  no-op by default."""
        return None

    def add_dependency(self, depend):
        """Adds dependencies."""
        try:
            self._add_child(self.depends, self.depends_set, depend)
        except TypeError as e:
            e = e.args[0]
            if SCons.Util.is_List(e):
                s = list(map(str, e))
            else:
                s = str(e)
            raise SCons.Errors.UserError("attempted to add a non-Node dependency to %s:\n\t%s is a %s, not a Node" % (str(self), s, type(e)))

    def add_prerequisite(self, prerequisite):
        """Adds prerequisites"""
        if self.prerequisites is None:
            self.prerequisites = SCons.Util.UniqueList()
        self.prerequisites.extend(prerequisite)
        self._children_reset()

    def add_ignore(self, depend):
        """Adds dependencies to ignore."""
        try:
            self._add_child(self.ignore, self.ignore_set, depend)
        except TypeError as e:
            e = e.args[0]
            if SCons.Util.is_List(e):
                s = list(map(str, e))
            else:
                s = str(e)
            raise SCons.Errors.UserError("attempted to ignore a non-Node dependency of %s:\n\t%s is a %s, not a Node" % (str(self), s, type(e)))

    def add_source(self, source):
        """Adds sources."""
        if self._specific_sources:
            return
        try:
            self._add_child(self.sources, self.sources_set, source)
        except TypeError as e:
            e = e.args[0]
            if SCons.Util.is_List(e):
                s = list(map(str, e))
            else:
                s = str(e)
            raise SCons.Errors.UserError("attempted to add a non-Node as source of %s:\n\t%s is a %s, not a Node" % (str(self), s, type(e)))

    def _add_child(self, collection, set, child):
        """Adds 'child' to 'collection', first checking 'set' to see if it's
        already present."""
        added = None
        for c in child:
            if c not in set:
                set.add(c)
                collection.append(c)
                added = 1
        if added:
            self._children_reset()

    def set_specific_source(self, source):
        self.add_source(source)
        self._specific_sources = True

    def add_wkid(self, wkid):
        """Add a node to the list of kids waiting to be evaluated"""
        if self.wkids is not None:
            self.wkids.append(wkid)

    def _children_reset(self):
        self.clear_memoized_values()
        # We need to let the Executor clear out any calculated
        # build info that it's cached so we can re-calculate it.
        self.executor_cleanup()

    @SCons.Memoize.CountMethodCall
    def _children_get(self):
        try:
            return self._memo['_children_get']
        except KeyError:
            pass

        # The return list may contain duplicate Nodes, especially in
        # source trees where there are a lot of repeated #includes
        # of a tangle of .h files.  Profiling shows, however, that
        # eliminating the duplicates with a brute-force approach that
        # preserves the order (that is, something like:
        #
        #       u = []
        #       for n in list:
        #           if n not in u:
        #               u.append(n)"
        #
        # takes more cycles than just letting the underlying methods
        # hand back cached values if a Node's information is requested
        # multiple times.  (Other methods of removing duplicates, like
        # using dictionary keys, lose the order, and the only ordered
        # dictionary patterns I found all ended up using "not in"
        # internally anyway...)
        if self.ignore_set:
            iter = chain.from_iterable([_f for _f in [self.sources, self.depends, self.implicit] if _f])

            children = []
            for i in iter:
                if i not in self.ignore_set:
                    children.append(i)
        else:
            children = self.all_children(scan=0)

        self._memo['_children_get'] = children
        return children

    def all_children(self, scan=1):
        """Return a list of all the node's direct children."""
        if scan:
            self.scan()

        # The return list may contain duplicate Nodes, especially in
        # source trees where there are a lot of repeated #includes
        # of a tangle of .h files.  Profiling shows, however, that
        # eliminating the duplicates with a brute-force approach that
        # preserves the order (that is, something like:
        #
        #       u = []
        #       for n in list:
        #           if n not in u:
        #               u.append(n)"
        #
        # takes more cycles than just letting the underlying methods
        # hand back cached values if a Node's information is requested
        # multiple times.  (Other methods of removing duplicates, like
        # using dictionary keys, lose the order, and the only ordered
        # dictionary patterns I found all ended up using "not in"
        # internally anyway...)
        return list(chain.from_iterable([_f for _f in [self.sources, self.depends, self.implicit] if _f]))

    def children(self, scan=1):
        """Return a list of the node's direct children, minus those
        that are ignored by this node."""
        if scan:
            self.scan()
        return self._children_get()

    def set_state(self, state):
        self.state = state

    def get_state(self):
        return self.state

    def get_env(self):
        env = self.env
        if not env:
            import SCons.Defaults
            env = SCons.Defaults.DefaultEnvironment()
        return env

    def Decider(self, function):
        foundkey = None
        for k, v in _decider_map.items():
            if v == function:
                foundkey = k
                break
        if not foundkey:
            foundkey = len(_decider_map)
            _decider_map[foundkey] = function
        self.changed_since_last_build = foundkey

    def Tag(self, key, value):
        """ Add a user-defined tag. """
        if not self._tags:
            self._tags = {}
        self._tags[key] = value

    def GetTag(self, key):
        """ Return a user-defined tag. """
        if not self._tags:
            return None
        return self._tags.get(key, None)

    def changed(self, node=None, allowcache=False):
        """
        Returns if the node is up-to-date with respect to the BuildInfo
        stored last time it was built.  The default behavior is to compare
        it against our own previously stored BuildInfo, but the stored
        BuildInfo from another Node (typically one in a Repository)
        can be used instead.

        Note that we now *always* check every dependency.  We used to
        short-circuit the check by returning as soon as we detected
        any difference, but we now rely on checking every dependency
        to make sure that any necessary Node information (for example,
        the content signature of an #included .h file) is updated.

        The allowcache option was added for supporting the early
        release of the executor/builder structures, right after
        a File target was built. When set to true, the return
        value of this changed method gets cached for File nodes.
        Like this, the executor isn't needed any longer for subsequent
        calls to changed().

        @see: FS.File.changed(), FS.File.release_target_info()
        """
        t = 0
        if t: Trace('changed(%s [%s], %s)' % (self, classname(self), node))
        if node is None:
            node = self

        result = False

        bi = node.get_stored_info().binfo
        then = bi.bsourcesigs + bi.bdependsigs + bi.bimplicitsigs
        children = self.children()

        diff = len(children) - len(then)
        if diff:
            # The old and new dependency lists are different lengths.
            # This always indicates that the Node must be rebuilt.
            # We also extend the old dependency list with enough None
            # entries to equal the new dependency list, for the benefit
            # of the loop below that updates node information.
            then.extend([None] * diff)
            if t: Trace(': old %s new %s' % (len(then), len(children)))
            result = True

        for child, prev_ni in zip(children, then):
            if _decider_map[child.changed_since_last_build](child, self, prev_ni):
                if t: Trace(': %s changed' % child)
                result = True

        contents = self.get_executor().get_contents()
        if self.has_builder():
            import SCons.Util
            newsig = SCons.Util.MD5signature(contents)
            if bi.bactsig != newsig:
                if t: Trace(': bactsig %s != newsig %s' % (bi.bactsig, newsig))
                result = True

        if not result:
            if t: Trace(': up to date')

        if t: Trace('\n')

        return result

    def is_up_to_date(self):
        """Default check for whether the Node is current: unknown Node
        subtypes are always out of date, so they will always get built."""
        return None

    def children_are_up_to_date(self):
        """Alternate check for whether the Node is current:  If all of
        our children were up-to-date, then this Node was up-to-date, too.

        The SCons.Node.Alias and SCons.Node.Python.Value subclasses
        rebind their current() method to this method."""
        # Allow the children to calculate their signatures.
        self.binfo = self.get_binfo()
        if self.always_build:
            return None
        state = 0
        for kid in self.children(None):
            s = kid.get_state()
            if s and (not state or s > state):
                state = s
        return (state == 0 or state == SCons.Node.up_to_date)

    def is_literal(self):
        """Always pass the string representation of a Node to
        the command interpreter literally."""
        return 1

    def render_include_tree(self):
        """
        Return a text representation, suitable for displaying to the
        user, of the include tree for the sources of this node.
        """
        if self.is_derived():
            env = self.get_build_env()
            if env:
                for s in self.sources:
                    scanner = self.get_source_scanner(s)
                    if scanner:
                        path = self.get_build_scanner_path(scanner)
                    else:
                        path = None
                    def f(node, env=env, scanner=scanner, path=path):
                        return node.get_found_includes(env, scanner, path)
                    return SCons.Util.render_tree(s, f, 1)
        else:
            return None

    def get_abspath(self):
        """
        Return an absolute path to the Node.  This will return simply
        str(Node) by default, but for Node types that have a concept of
        relative path, this might return something different.
        """
        return str(self)

    def for_signature(self):
        """
        Return a string representation of the Node that will always
        be the same for this particular Node, no matter what.  This
        is by contrast to the __str__() method, which might, for
        instance, return a relative path for a file Node.  The purpose
        of this method is to generate a value to be used in signature
        calculation for the command line used to build a target, and
        we use this method instead of str() to avoid unnecessary
        rebuilds.  This method does not need to return something that
        would actually work in a command line; it can return any kind of
        nonsense, so long as it does not change.
        """
        return str(self)

    def get_string(self, for_signature):
        """This is a convenience function designed primarily to be
        used in command generators (i.e., CommandGeneratorActions or
        Environment variables that are callable), which are called
        with a for_signature argument that is nonzero if the command
        generator is being called to generate a signature for the
        command line, which determines if we should rebuild or not.

        Such command generators should use this method in preference
        to str(Node) when converting a Node to a string, passing
        in the for_signature parameter, such that we will call
        Node.for_signature() or str(Node) properly, depending on whether
        we are calculating a signature or actually constructing a
        command line."""
        if for_signature:
            return self.for_signature()
        return str(self)

    def get_subst_proxy(self):
        """
        This method is expected to return an object that will function
        exactly like this Node, except that it implements any additional
        special features that we would like to be in effect for
        Environment variable substitution.  The principle use is that
        some Nodes would like to implement a __getattr__() method,
        but putting that in the Node type itself has a tendency to kill
        performance.  We instead put it in a proxy and return it from
        this method.  It is legal for this method to return self
        if no new functionality is needed for Environment substitution.
        """
        return self

    def explain(self):
        if not self.exists():
            return "building `%s' because it doesn't exist\n" % self

        if self.always_build:
            return "rebuilding `%s' because AlwaysBuild() is specified\n" % self

        old = self.get_stored_info()
        if old is None:
            return None

        old = old.binfo
        old.prepare_dependencies()

        try:
            old_bkids    = old.bsources    + old.bdepends    + old.bimplicit
            old_bkidsigs = old.bsourcesigs + old.bdependsigs + old.bimplicitsigs
        except AttributeError:
            return "Cannot explain why `%s' is being rebuilt: No previous build information found\n" % self

        new = self.get_binfo()

        new_bkids    = new.bsources    + new.bdepends    + new.bimplicit
        new_bkidsigs = new.bsourcesigs + new.bdependsigs + new.bimplicitsigs

        osig = dict(list(zip(old_bkids, old_bkidsigs)))
        nsig = dict(list(zip(new_bkids, new_bkidsigs)))

        # The sources and dependencies we'll want to report are all stored
        # as relative paths to this target's directory, but we want to
        # report them relative to the top-level SConstruct directory,
        # so we only print them after running them through this lambda
        # to turn them into the right relative Node and then return
        # its string.
        def stringify( s, E=self.dir.Entry ) :
            if hasattr( s, 'dir' ) :
                return str(E(s))
            return str(s)

        lines = []

        removed = [x for x in old_bkids if not x in new_bkids]
        if removed:
            removed = list(map(stringify, removed))
            fmt = "`%s' is no longer a dependency\n"
            lines.extend([fmt % s for s in removed])

        for k in new_bkids:
            if not k in old_bkids:
                lines.append("`%s' is a new dependency\n" % stringify(k))
            elif _decider_map[k.changed_since_last_build](k, self, osig[k]):
                lines.append("`%s' changed\n" % stringify(k))

        if len(lines) == 0 and old_bkids != new_bkids:
            lines.append("the dependency order changed:\n" +
                         "%sold: %s\n" % (' '*15, list(map(stringify, old_bkids))) +
                         "%snew: %s\n" % (' '*15, list(map(stringify, new_bkids))))

        if len(lines) == 0:
            def fmt_with_title(title, strlines):
                lines = strlines.split('\n')
                sep = '\n' + ' '*(15 + len(title))
                return ' '*15 + title + sep.join(lines) + '\n'
            if old.bactsig != new.bactsig:
                if old.bact == new.bact:
                    lines.append("the contents of the build action changed\n" +
                                 fmt_with_title('action: ', new.bact))

                    # lines.append("the contents of the build action changed [%s] [%s]\n"%(old.bactsig,new.bactsig) +
                    #              fmt_with_title('action: ', new.bact))
                else:
                    lines.append("the build action changed:\n" +
                                 fmt_with_title('old: ', old.bact) +
                                 fmt_with_title('new: ', new.bact))

        if len(lines) == 0:
            return "rebuilding `%s' for unknown reasons\n" % self

        preamble = "rebuilding `%s' because" % self
        if len(lines) == 1:
            return "%s %s"  % (preamble, lines[0])
        else:
            lines = ["%s:\n" % preamble] + lines
            return ( ' '*11).join(lines)

class NodeList(collections.UserList):
    def __str__(self):
        return str(list(map(str, self.data)))

def get_children(node, parent): return node.children()
def ignore_cycle(node, stack): pass
def do_nothing(node, parent): pass

class Walker(object):
    """An iterator for walking a Node tree.

    This is depth-first, children are visited before the parent.
    The Walker object can be initialized with any node, and
    returns the next node on the descent with each get_next() call.
    'kids_func' is an optional function that will be called to
    get the children of a node instead of calling 'children'.
    'cycle_func' is an optional function that will be called
    when a cycle is detected.

    This class does not get caught in node cycles caused, for example,
    by C header file include loops.
    """
    def __init__(self, node, kids_func=get_children,
                             cycle_func=ignore_cycle,
                             eval_func=do_nothing):
        self.kids_func = kids_func
        self.cycle_func = cycle_func
        self.eval_func = eval_func
        node.wkids = copy.copy(kids_func(node, None))
        self.stack = [node]
        self.history = {} # used to efficiently detect and avoid cycles
        self.history[node] = None

    def get_next(self):
        """Return the next node for this walk of the tree.

        This function is intentionally iterative, not recursive,
        to sidestep any issues of stack size limitations.
        """

        while self.stack:
            if self.stack[-1].wkids:
                node = self.stack[-1].wkids.pop(0)
                if not self.stack[-1].wkids:
                    self.stack[-1].wkids = None
                if node in self.history:
                    self.cycle_func(node, self.stack)
                else:
                    node.wkids = copy.copy(self.kids_func(node, self.stack[-1]))
                    self.stack.append(node)
                    self.history[node] = None
            else:
                node = self.stack.pop()
                del self.history[node]
                if node:
                    if self.stack:
                        parent = self.stack[-1]
                    else:
                        parent = None
                    self.eval_func(node, parent)
                return node
        return None

    def is_done(self):
        return not self.stack


arg2nodes_lookups = []

# Local Variables:
# tab-width:4
# indent-tabs-mode:nil
# End:
# vim: set expandtab tabstop=4 shiftwidth=4:
