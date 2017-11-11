"""SCons.Tool.javac

Tool-specific initialization for javac.

There normally shouldn't be any need to import this module directly.
It will usually be imported through the generic SCons.Tool.Tool()
selection method.

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

__revision__ = "src/engine/SCons/Tool/javac.py rel_3.0.0:4395:8972f6a2f699 2017/09/18 12:59:24 bdbaddog"

import os
import os.path

import SCons.Action
import SCons.Builder
from SCons.Node.FS import _my_normcase
from SCons.Tool.JavaCommon import parse_java_file
import SCons.Util

def classname(path):
    """Turn a string (path name) into a Java class name."""
    return os.path.normpath(path).replace(os.sep, '.')

def emit_java_classes(target, source, env):
    """Create and return lists of source java files
    and their corresponding target class files.
    """
    java_suffix = env.get('JAVASUFFIX', '.java')
    class_suffix = env.get('JAVACLASSSUFFIX', '.class')

    target[0].must_be_same(SCons.Node.FS.Dir)
    classdir = target[0]

    s = source[0].rentry().disambiguate()
    if isinstance(s, SCons.Node.FS.File):
        sourcedir = s.dir.rdir()
    elif isinstance(s, SCons.Node.FS.Dir):
        sourcedir = s.rdir()
    else:
        raise SCons.Errors.UserError("Java source must be File or Dir, not '%s'" % s.__class__)

    slist = []
    js = _my_normcase(java_suffix)
    for entry in source:
        entry = entry.rentry().disambiguate()
        if isinstance(entry, SCons.Node.FS.File):
            slist.append(entry)
        elif isinstance(entry, SCons.Node.FS.Dir):
            result = SCons.Util.OrderedDict()
            dirnode = entry.rdir()
            def find_java_files(arg, dirpath, filenames):
                java_files = sorted([n for n in filenames
                                       if _my_normcase(n).endswith(js)])
                mydir = dirnode.Dir(dirpath)
                java_paths = [mydir.File(f) for f in java_files]
                for jp in java_paths:
                     arg[jp] = True
            for dirpath, dirnames, filenames in os.walk(dirnode.get_abspath()):
               find_java_files(result, dirpath, filenames)
            entry.walk(find_java_files, result)

            slist.extend(list(result.keys()))
        else:
            raise SCons.Errors.UserError("Java source must be File or Dir, not '%s'" % entry.__class__)

    version = env.get('JAVAVERSION', '1.4')
    full_tlist = []
    for f in slist:
        tlist = []
        source_file_based = True
        pkg_dir = None
        if not f.is_derived():
            pkg_dir, classes = parse_java_file(f.rfile().get_abspath(), version)
            if classes:
                source_file_based = False
                if pkg_dir:
                    d = target[0].Dir(pkg_dir)
                    p = pkg_dir + os.sep
                else:
                    d = target[0]
                    p = ''
                for c in classes:
                    t = d.File(c + class_suffix)
                    t.attributes.java_classdir = classdir
                    t.attributes.java_sourcedir = sourcedir
                    t.attributes.java_classname = classname(p + c)
                    tlist.append(t)

        if source_file_based:
            base = f.name[:-len(java_suffix)]
            if pkg_dir:
                t = target[0].Dir(pkg_dir).File(base + class_suffix)
            else:
                t = target[0].File(base + class_suffix)
            t.attributes.java_classdir = classdir
            t.attributes.java_sourcedir = f.dir
            t.attributes.java_classname = classname(base)
            tlist.append(t)

        for t in tlist:
            t.set_specific_source([f])

        full_tlist.extend(tlist)

    return full_tlist, slist

JavaAction = SCons.Action.Action('$JAVACCOM', '$JAVACCOMSTR')

JavaBuilder = SCons.Builder.Builder(action = JavaAction,
                    emitter = emit_java_classes,
                    target_factory = SCons.Node.FS.Entry,
                    source_factory = SCons.Node.FS.Entry)

class pathopt(object):
    """
    Callable object for generating javac-style path options from
    a construction variable (e.g. -classpath, -sourcepath).
    """
    def __init__(self, opt, var, default=None):
        self.opt = opt
        self.var = var
        self.default = default

    def __call__(self, target, source, env, for_signature):
        path = env[self.var]
        if path and not SCons.Util.is_List(path):
            path = [path]
        if self.default:
            default = env[self.default]
            if default:
                if not SCons.Util.is_List(default):
                    default = [default]
                path = path + default
        if path:
            return [self.opt, os.pathsep.join(map(str, path))]
        else:
            return []

def Java(env, target, source, *args, **kw):
    """
    A pseudo-Builder wrapper around the separate JavaClass{File,Dir}
    Builders.
    """
    if not SCons.Util.is_List(target):
        target = [target]
    if not SCons.Util.is_List(source):
        source = [source]

    # Pad the target list with repetitions of the last element in the
    # list so we have a target for every source element.
    target = target + ([target[-1]] * (len(source) - len(target)))

    java_suffix = env.subst('$JAVASUFFIX')
    result = []

    for t, s in zip(target, source):
        if isinstance(s, SCons.Node.FS.Base):
            if isinstance(s, SCons.Node.FS.File):
                b = env.JavaClassFile
            else:
                b = env.JavaClassDir
        else:
            if os.path.isfile(s):
                b = env.JavaClassFile
            elif os.path.isdir(s):
                b = env.JavaClassDir
            elif s[-len(java_suffix):] == java_suffix:
                b = env.JavaClassFile
            else:
                b = env.JavaClassDir
        result.extend(b(t, s, *args, **kw))

    return result

def generate(env):
    """Add Builders and construction variables for javac to an Environment."""
    java_file = SCons.Tool.CreateJavaFileBuilder(env)
    java_class = SCons.Tool.CreateJavaClassFileBuilder(env)
    java_class_dir = SCons.Tool.CreateJavaClassDirBuilder(env)
    java_class.add_emitter(None, emit_java_classes)
    java_class.add_emitter(env.subst('$JAVASUFFIX'), emit_java_classes)
    java_class_dir.emitter = emit_java_classes

    env.AddMethod(Java)

    env['JAVAC']                    = 'javac'
    env['JAVACFLAGS']               = SCons.Util.CLVar('')
    env['JAVABOOTCLASSPATH']        = []
    env['JAVACLASSPATH']            = []
    env['JAVASOURCEPATH']           = []
    env['_javapathopt']             = pathopt
    env['_JAVABOOTCLASSPATH']       = '${_javapathopt("-bootclasspath", "JAVABOOTCLASSPATH")} '
    env['_JAVACLASSPATH']           = '${_javapathopt("-classpath", "JAVACLASSPATH")} '
    env['_JAVASOURCEPATH']          = '${_javapathopt("-sourcepath", "JAVASOURCEPATH", "_JAVASOURCEPATHDEFAULT")} '
    env['_JAVASOURCEPATHDEFAULT']   = '${TARGET.attributes.java_sourcedir}'
    env['_JAVACCOM']                = '$JAVAC $JAVACFLAGS $_JAVABOOTCLASSPATH $_JAVACLASSPATH -d ${TARGET.attributes.java_classdir} $_JAVASOURCEPATH $SOURCES'
    env['JAVACCOM']                 = "${TEMPFILE('$_JAVACCOM','$JAVACCOMSTR')}"
    env['JAVACLASSSUFFIX']          = '.class'
    env['JAVASUFFIX']               = '.java'

def exists(env):
    return 1

# Local Variables:
# tab-width:4
# indent-tabs-mode:nil
# End:
# vim: set expandtab tabstop=4 shiftwidth=4:
