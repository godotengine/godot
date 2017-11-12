"""SCons.SConsign

Writing and reading information to the .sconsign file or files.

"""

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
#

from __future__ import print_function

__revision__ = "src/engine/SCons/SConsign.py rel_3.0.0:4395:8972f6a2f699 2017/09/18 12:59:24 bdbaddog"

import SCons.compat

import os
import pickle

import SCons.dblite
import SCons.Warnings

from SCons.compat import PICKLE_PROTOCOL


def corrupt_dblite_warning(filename):
    SCons.Warnings.warn(SCons.Warnings.CorruptSConsignWarning,
                        "Ignoring corrupt .sconsign file: %s"%filename)

SCons.dblite.ignore_corrupt_dbfiles = 1
SCons.dblite.corruption_warning = corrupt_dblite_warning

# XXX Get rid of the global array so this becomes re-entrant.
sig_files = []

# Info for the database SConsign implementation (now the default):
# "DataBase" is a dictionary that maps top-level SConstruct directories
# to open database handles.
# "DB_Module" is the Python database module to create the handles.
# "DB_Name" is the base name of the database file (minus any
# extension the underlying DB module will add).
DataBase = {}
DB_Module = SCons.dblite
DB_Name = ".sconsign"
DB_sync_list = []


def Get_DataBase(dir):
    global DataBase, DB_Module, DB_Name
    top = dir.fs.Top
    if not os.path.isabs(DB_Name) and top.repositories:
        mode = "c"
        for d in [top] + top.repositories:
            if dir.is_under(d):
                try:
                    return DataBase[d], mode
                except KeyError:
                    path = d.entry_abspath(DB_Name)
                    try: db = DataBase[d] = DB_Module.open(path, mode)
                    except (IOError, OSError): pass
                    else:
                        if mode != "r":
                            DB_sync_list.append(db)
                        return db, mode
            mode = "r"
    try:
        return DataBase[top], "c"
    except KeyError:
        db = DataBase[top] = DB_Module.open(DB_Name, "c")
        DB_sync_list.append(db)
        return db, "c"
    except TypeError:
        print("DataBase =", DataBase)
        raise


def Reset():
    """Reset global state.  Used by unit tests that end up using
    SConsign multiple times to get a clean slate for each test."""
    global sig_files, DB_sync_list
    sig_files = []
    DB_sync_list = []

normcase = os.path.normcase


def write():
    global sig_files
    for sig_file in sig_files:
        sig_file.write(sync=0)
    for db in DB_sync_list:
        try:
            syncmethod = db.sync
        except AttributeError:
            pass # Not all dbm modules have sync() methods.
        else:
            syncmethod()
        try:
            closemethod = db.close
        except AttributeError:
            pass # Not all dbm modules have close() methods.
        else:
            closemethod()


class SConsignEntry(object):
    """
    Wrapper class for the generic entry in a .sconsign file.
    The Node subclass populates it with attributes as it pleases.

    XXX As coded below, we do expect a '.binfo' attribute to be added,
    but we'll probably generalize this in the next refactorings.
    """
    __slots__ = ("binfo", "ninfo", "__weakref__")
    current_version_id = 2

    def __init__(self):
        # Create an object attribute from the class attribute so it ends up
        # in the pickled data in the .sconsign file.
        #_version_id = self.current_version_id
        pass

    def convert_to_sconsign(self):
        self.binfo.convert_to_sconsign()

    def convert_from_sconsign(self, dir, name):
        self.binfo.convert_from_sconsign(dir, name)

    def __getstate__(self):
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
        for key, value in state.items():
            if key not in ('_version_id','__weakref__'):
                setattr(self, key, value)


class Base(object):
    """
    This is the controlling class for the signatures for the collection of
    entries associated with a specific directory.  The actual directory
    association will be maintained by a subclass that is specific to
    the underlying storage method.  This class provides a common set of
    methods for fetching and storing the individual bits of information
    that make up signature entry.
    """
    def __init__(self):
        self.entries = {}
        self.dirty = False
        self.to_be_merged = {}

    def get_entry(self, filename):
        """
        Fetch the specified entry attribute.
        """
        return self.entries[filename]

    def set_entry(self, filename, obj):
        """
        Set the entry.
        """
        self.entries[filename] = obj
        self.dirty = True

    def do_not_set_entry(self, filename, obj):
        pass

    def store_info(self, filename, node):
        entry = node.get_stored_info()
        entry.binfo.merge(node.get_binfo())
        self.to_be_merged[filename] = node
        self.dirty = True

    def do_not_store_info(self, filename, node):
        pass

    def merge(self):
        for key, node in self.to_be_merged.items():
            entry = node.get_stored_info()
            try:
                ninfo = entry.ninfo
            except AttributeError:
                # This happens with SConf Nodes, because the configuration
                # subsystem takes direct control over how the build decision
                # is made and its information stored.
                pass
            else:
                ninfo.merge(node.get_ninfo())
            self.entries[key] = entry
        self.to_be_merged = {}


class DB(Base):
    """
    A Base subclass that reads and writes signature information
    from a global .sconsign.db* file--the actual file suffix is
    determined by the database module.
    """
    def __init__(self, dir):
        Base.__init__(self)

        self.dir = dir

        db, mode = Get_DataBase(dir)

        # Read using the path relative to the top of the Repository
        # (self.dir.tpath) from which we're fetching the signature
        # information.
        path = normcase(dir.get_tpath())
        try:
            rawentries = db[path]
        except KeyError:
            pass
        else:
            try:
                self.entries = pickle.loads(rawentries)
                if not isinstance(self.entries, dict):
                    self.entries = {}
                    raise TypeError
            except KeyboardInterrupt:
                raise
            except Exception as e:
                SCons.Warnings.warn(SCons.Warnings.CorruptSConsignWarning,
                                    "Ignoring corrupt sconsign entry : %s (%s)\n"%(self.dir.get_tpath(), e))
            for key, entry in self.entries.items():
                entry.convert_from_sconsign(dir, key)

        if mode == "r":
            # This directory is actually under a repository, which means
            # likely they're reaching in directly for a dependency on
            # a file there.  Don't actually set any entry info, so we
            # won't try to write to that .sconsign.dblite file.
            self.set_entry = self.do_not_set_entry
            self.store_info = self.do_not_store_info

        global sig_files
        sig_files.append(self)

    def write(self, sync=1):
        if not self.dirty:
            return

        self.merge()

        db, mode = Get_DataBase(self.dir)

        # Write using the path relative to the top of the SConstruct
        # directory (self.dir.path), not relative to the top of
        # the Repository; we only write to our own .sconsign file,
        # not to .sconsign files in Repositories.
        path = normcase(self.dir.get_internal_path())
        for key, entry in self.entries.items():
            entry.convert_to_sconsign()
        db[path] = pickle.dumps(self.entries, PICKLE_PROTOCOL)

        if sync:
            try:
                syncmethod = db.sync
            except AttributeError:
                # Not all anydbm modules have sync() methods.
                pass
            else:
                syncmethod()


class Dir(Base):
    def __init__(self, fp=None, dir=None):
        """
        fp - file pointer to read entries from
        """
        Base.__init__(self)

        if not fp:
            return

        self.entries = pickle.load(fp)
        if not isinstance(self.entries, dict):
            self.entries = {}
            raise TypeError

        if dir:
            for key, entry in self.entries.items():
                entry.convert_from_sconsign(dir, key)


class DirFile(Dir):
    """
    Encapsulates reading and writing a per-directory .sconsign file.
    """
    def __init__(self, dir):
        """
        dir - the directory for the file
        """

        self.dir = dir
        self.sconsign = os.path.join(dir.get_internal_path(), '.sconsign')

        try:
            fp = open(self.sconsign, 'rb')
        except IOError:
            fp = None

        try:
            Dir.__init__(self, fp, dir)
        except KeyboardInterrupt:
            raise
        except:
            SCons.Warnings.warn(SCons.Warnings.CorruptSConsignWarning,
                                "Ignoring corrupt .sconsign file: %s"%self.sconsign)

        global sig_files
        sig_files.append(self)

    def write(self, sync=1):
        """
        Write the .sconsign file to disk.

        Try to write to a temporary file first, and rename it if we
        succeed.  If we can't write to the temporary file, it's
        probably because the directory isn't writable (and if so,
        how did we build anything in this directory, anyway?), so
        try to write directly to the .sconsign file as a backup.
        If we can't rename, try to copy the temporary contents back
        to the .sconsign file.  Either way, always try to remove
        the temporary file at the end.
        """
        if not self.dirty:
            return

        self.merge()

        temp = os.path.join(self.dir.get_internal_path(), '.scons%d' % os.getpid())
        try:
            file = open(temp, 'wb')
            fname = temp
        except IOError:
            try:
                file = open(self.sconsign, 'wb')
                fname = self.sconsign
            except IOError:
                return
        for key, entry in self.entries.items():
            entry.convert_to_sconsign()
        pickle.dump(self.entries, file, PICKLE_PROTOCOL)
        file.close()
        if fname != self.sconsign:
            try:
                mode = os.stat(self.sconsign)[0]
                os.chmod(self.sconsign, 0o666)
                os.unlink(self.sconsign)
            except (IOError, OSError):
                # Try to carry on in the face of either OSError
                # (things like permission issues) or IOError (disk
                # or network issues).  If there's a really dangerous
                # issue, it should get re-raised by the calls below.
                pass
            try:
                os.rename(fname, self.sconsign)
            except OSError:
                # An OSError failure to rename may indicate something
                # like the directory has no write permission, but
                # the .sconsign file itself might still be writable,
                # so try writing on top of it directly.  An IOError
                # here, or in any of the following calls, would get
                # raised, indicating something like a potentially
                # serious disk or network issue.
                open(self.sconsign, 'wb').write(open(fname, 'rb').read())
                os.chmod(self.sconsign, mode)
        try:
            os.unlink(temp)
        except (IOError, OSError):
            pass

ForDirectory = DB


def File(name, dbm_module=None):
    """
    Arrange for all signatures to be stored in a global .sconsign.db*
    file.
    """
    global ForDirectory, DB_Name, DB_Module
    if name is None:
        ForDirectory = DirFile
        DB_Module = None
    else:
        ForDirectory = DB
        DB_Name = name
        if not dbm_module is None:
            DB_Module = dbm_module

# Local Variables:
# tab-width:4
# indent-tabs-mode:nil
# End:
# vim: set expandtab tabstop=4 shiftwidth=4:
