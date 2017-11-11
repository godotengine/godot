"""SCons.Tool.rmic

Tool-specific initialization for rmic.

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
#

__revision__ = "src/engine/SCons/Tool/rmic.py rel_3.0.0:4395:8972f6a2f699 2017/09/18 12:59:24 bdbaddog"

import os.path

import SCons.Action
import SCons.Builder
import SCons.Node.FS
import SCons.Util

def emit_rmic_classes(target, source, env):
    """Create and return lists of Java RMI stub and skeleton
    class files to be created from a set of class files.
    """
    class_suffix = env.get('JAVACLASSSUFFIX', '.class')
    classdir = env.get('JAVACLASSDIR')

    if not classdir:
        try:
            s = source[0]
        except IndexError:
            classdir = '.'
        else:
            try:
                classdir = s.attributes.java_classdir
            except AttributeError:
                classdir = '.'
    classdir = env.Dir(classdir).rdir()
    if str(classdir) == '.':
        c_ = None
    else:
        c_ = str(classdir) + os.sep

    slist = []
    for src in source:
        try:
            classname = src.attributes.java_classname
        except AttributeError:
            classname = str(src)
            if c_ and classname[:len(c_)] == c_:
                classname = classname[len(c_):]
            if class_suffix and classname[:-len(class_suffix)] == class_suffix:
                classname = classname[-len(class_suffix):]
        s = src.rfile()
        s.attributes.java_classdir = classdir
        s.attributes.java_classname = classname
        slist.append(s)

    stub_suffixes = ['_Stub']
    if env.get('JAVAVERSION') == '1.4':
        stub_suffixes.append('_Skel')

    tlist = []
    for s in source:
        for suff in stub_suffixes:
            fname = s.attributes.java_classname.replace('.', os.sep) + \
                    suff + class_suffix
            t = target[0].File(fname)
            t.attributes.java_lookupdir = target[0]
            tlist.append(t)

    return tlist, source

RMICAction = SCons.Action.Action('$RMICCOM', '$RMICCOMSTR')

RMICBuilder = SCons.Builder.Builder(action = RMICAction,
                     emitter = emit_rmic_classes,
                     src_suffix = '$JAVACLASSSUFFIX',
                     target_factory = SCons.Node.FS.Dir,
                     source_factory = SCons.Node.FS.File)

def generate(env):
    """Add Builders and construction variables for rmic to an Environment."""
    env['BUILDERS']['RMIC'] = RMICBuilder

    env['RMIC']            = 'rmic'
    env['RMICFLAGS']       = SCons.Util.CLVar('')
    env['RMICCOM']         = '$RMIC $RMICFLAGS -d ${TARGET.attributes.java_lookupdir} -classpath ${SOURCE.attributes.java_classdir} ${SOURCES.attributes.java_classname}'
    env['JAVACLASSSUFFIX']  = '.class'

def exists(env):
    # As reported by Jan Nijtmans in issue #2730, the simple
    #    return env.Detect('rmic')
    # doesn't always work during initialization. For now, we
    # stop trying to detect an executable (analogous to the
    # javac Builder).
    # TODO: Come up with a proper detect() routine...and enable it.
    return 1

# Local Variables:
# tab-width:4
# indent-tabs-mode:nil
# End:
# vim: set expandtab tabstop=4 shiftwidth=4:
