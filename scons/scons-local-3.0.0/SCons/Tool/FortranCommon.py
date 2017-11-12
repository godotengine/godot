"""SCons.Tool.FortranCommon

Stuff for processing Fortran, common to all fortran dialects.

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

__revision__ = "src/engine/SCons/Tool/FortranCommon.py rel_3.0.0:4395:8972f6a2f699 2017/09/18 12:59:24 bdbaddog"

import re
import os.path

import SCons.Action
import SCons.Defaults
import SCons.Scanner.Fortran
import SCons.Tool
import SCons.Util

def isfortran(env, source):
    """Return 1 if any of code in source has fortran files in it, 0
    otherwise."""
    try:
        fsuffixes = env['FORTRANSUFFIXES']
    except KeyError:
        # If no FORTRANSUFFIXES, no fortran tool, so there is no need to look
        # for fortran sources.
        return 0

    if not source:
        # Source might be None for unusual cases like SConf.
        return 0
    for s in source:
        if s.sources:
            ext = os.path.splitext(str(s.sources[0]))[1]
            if ext in fsuffixes:
                return 1
    return 0

def _fortranEmitter(target, source, env):
    node = source[0].rfile()
    if not node.exists() and not node.is_derived():
       print("Could not locate " + str(node.name))
       return ([], [])
    mod_regex = """(?i)^\s*MODULE\s+(?!PROCEDURE)(\w+)"""
    cre = re.compile(mod_regex,re.M)
    # Retrieve all USE'd module names
    modules = cre.findall(node.get_text_contents())
    # Remove unique items from the list
    modules = SCons.Util.unique(modules)
    # Convert module name to a .mod filename
    suffix = env.subst('$FORTRANMODSUFFIX', target=target, source=source)
    moddir = env.subst('$FORTRANMODDIR', target=target, source=source)
    modules = [x.lower() + suffix for x in modules]
    for m in modules:
       target.append(env.fs.File(m, moddir))
    return (target, source)

def FortranEmitter(target, source, env):
    target, source = _fortranEmitter(target, source, env)
    return SCons.Defaults.StaticObjectEmitter(target, source, env)

def ShFortranEmitter(target, source, env):
    target, source = _fortranEmitter(target, source, env)
    return SCons.Defaults.SharedObjectEmitter(target, source, env)

def ComputeFortranSuffixes(suffixes, ppsuffixes):
    """suffixes are fortran source files, and ppsuffixes the ones to be
    pre-processed. Both should be sequences, not strings."""
    assert len(suffixes) > 0
    s = suffixes[0]
    sup = s.upper()
    upper_suffixes = [_.upper() for _ in suffixes]
    if SCons.Util.case_sensitive_suffixes(s, sup):
        ppsuffixes.extend(upper_suffixes)
    else:
        suffixes.extend(upper_suffixes)

def CreateDialectActions(dialect):
    """Create dialect specific actions."""
    CompAction = SCons.Action.Action('$%sCOM ' % dialect, '$%sCOMSTR' % dialect)
    CompPPAction = SCons.Action.Action('$%sPPCOM ' % dialect, '$%sPPCOMSTR' % dialect)
    ShCompAction = SCons.Action.Action('$SH%sCOM ' % dialect, '$SH%sCOMSTR' % dialect)
    ShCompPPAction = SCons.Action.Action('$SH%sPPCOM ' % dialect, '$SH%sPPCOMSTR' % dialect)

    return CompAction, CompPPAction, ShCompAction, ShCompPPAction

def DialectAddToEnv(env, dialect, suffixes, ppsuffixes, support_module = 0):
    """Add dialect specific construction variables."""
    ComputeFortranSuffixes(suffixes, ppsuffixes)

    fscan = SCons.Scanner.Fortran.FortranScan("%sPATH" % dialect)

    for suffix in suffixes + ppsuffixes:
        SCons.Tool.SourceFileScanner.add_scanner(suffix, fscan)

    env.AppendUnique(FORTRANSUFFIXES = suffixes + ppsuffixes)

    compaction, compppaction, shcompaction, shcompppaction = \
            CreateDialectActions(dialect)

    static_obj, shared_obj = SCons.Tool.createObjBuilders(env)

    for suffix in suffixes:
        static_obj.add_action(suffix, compaction)
        shared_obj.add_action(suffix, shcompaction)
        static_obj.add_emitter(suffix, FortranEmitter)
        shared_obj.add_emitter(suffix, ShFortranEmitter)

    for suffix in ppsuffixes:
        static_obj.add_action(suffix, compppaction)
        shared_obj.add_action(suffix, shcompppaction)
        static_obj.add_emitter(suffix, FortranEmitter)
        shared_obj.add_emitter(suffix, ShFortranEmitter)

    if '%sFLAGS' % dialect not in env:
        env['%sFLAGS' % dialect] = SCons.Util.CLVar('')

    if 'SH%sFLAGS' % dialect not in env:
        env['SH%sFLAGS' % dialect] = SCons.Util.CLVar('$%sFLAGS' % dialect)

    # If a tool does not define fortran prefix/suffix for include path, use C ones
    if 'INC%sPREFIX' % dialect not in env:
        env['INC%sPREFIX' % dialect] = '$INCPREFIX'

    if 'INC%sSUFFIX' % dialect not in env:
        env['INC%sSUFFIX' % dialect] = '$INCSUFFIX'

    env['_%sINCFLAGS' % dialect] = '$( ${_concat(INC%sPREFIX, %sPATH, INC%sSUFFIX, __env__, RDirs, TARGET, SOURCE)} $)' % (dialect, dialect, dialect)

    if support_module == 1:
        env['%sCOM' % dialect]     = '$%s -o $TARGET -c $%sFLAGS $_%sINCFLAGS $_FORTRANMODFLAG $SOURCES' % (dialect, dialect, dialect)
        env['%sPPCOM' % dialect]   = '$%s -o $TARGET -c $%sFLAGS $CPPFLAGS $_CPPDEFFLAGS $_%sINCFLAGS $_FORTRANMODFLAG $SOURCES' % (dialect, dialect, dialect)
        env['SH%sCOM' % dialect]    = '$SH%s -o $TARGET -c $SH%sFLAGS $_%sINCFLAGS $_FORTRANMODFLAG $SOURCES' % (dialect, dialect, dialect)
        env['SH%sPPCOM' % dialect]  = '$SH%s -o $TARGET -c $SH%sFLAGS $CPPFLAGS $_CPPDEFFLAGS $_%sINCFLAGS $_FORTRANMODFLAG $SOURCES' % (dialect, dialect, dialect)
    else:
        env['%sCOM' % dialect]     = '$%s -o $TARGET -c $%sFLAGS $_%sINCFLAGS $SOURCES' % (dialect, dialect, dialect)
        env['%sPPCOM' % dialect]   = '$%s -o $TARGET -c $%sFLAGS $CPPFLAGS $_CPPDEFFLAGS $_%sINCFLAGS $SOURCES' % (dialect, dialect, dialect)
        env['SH%sCOM' % dialect]    = '$SH%s -o $TARGET -c $SH%sFLAGS $_%sINCFLAGS $SOURCES' % (dialect, dialect, dialect)
        env['SH%sPPCOM' % dialect]  = '$SH%s -o $TARGET -c $SH%sFLAGS $CPPFLAGS $_CPPDEFFLAGS $_%sINCFLAGS $SOURCES' % (dialect, dialect, dialect)

def add_fortran_to_env(env):
    """Add Builders and construction variables for Fortran to an Environment."""
    try:
        FortranSuffixes = env['FORTRANFILESUFFIXES']
    except KeyError:
        FortranSuffixes = ['.f', '.for', '.ftn']

    #print("Adding %s to fortran suffixes" % FortranSuffixes)
    try:
        FortranPPSuffixes = env['FORTRANPPFILESUFFIXES']
    except KeyError:
        FortranPPSuffixes = ['.fpp', '.FPP']

    DialectAddToEnv(env, "FORTRAN", FortranSuffixes,
                    FortranPPSuffixes, support_module = 1)

    env['FORTRANMODPREFIX'] = ''     # like $LIBPREFIX
    env['FORTRANMODSUFFIX'] = '.mod' # like $LIBSUFFIX

    env['FORTRANMODDIR'] = ''          # where the compiler should place .mod files
    env['FORTRANMODDIRPREFIX'] = ''    # some prefix to $FORTRANMODDIR - similar to $INCPREFIX
    env['FORTRANMODDIRSUFFIX'] = ''    # some suffix to $FORTRANMODDIR - similar to $INCSUFFIX
    env['_FORTRANMODFLAG'] = '$( ${_concat(FORTRANMODDIRPREFIX, FORTRANMODDIR, FORTRANMODDIRSUFFIX, __env__, RDirs, TARGET, SOURCE)} $)'

def add_f77_to_env(env):
    """Add Builders and construction variables for f77 to an Environment."""
    try:
        F77Suffixes = env['F77FILESUFFIXES']
    except KeyError:
        F77Suffixes = ['.f77']

    #print("Adding %s to f77 suffixes" % F77Suffixes)
    try:
        F77PPSuffixes = env['F77PPFILESUFFIXES']
    except KeyError:
        F77PPSuffixes = []

    DialectAddToEnv(env, "F77", F77Suffixes, F77PPSuffixes)

def add_f90_to_env(env):
    """Add Builders and construction variables for f90 to an Environment."""
    try:
        F90Suffixes = env['F90FILESUFFIXES']
    except KeyError:
        F90Suffixes = ['.f90']

    #print("Adding %s to f90 suffixes" % F90Suffixes)
    try:
        F90PPSuffixes = env['F90PPFILESUFFIXES']
    except KeyError:
        F90PPSuffixes = []

    DialectAddToEnv(env, "F90", F90Suffixes, F90PPSuffixes,
                    support_module = 1)

def add_f95_to_env(env):
    """Add Builders and construction variables for f95 to an Environment."""
    try:
        F95Suffixes = env['F95FILESUFFIXES']
    except KeyError:
        F95Suffixes = ['.f95']

    #print("Adding %s to f95 suffixes" % F95Suffixes)
    try:
        F95PPSuffixes = env['F95PPFILESUFFIXES']
    except KeyError:
        F95PPSuffixes = []

    DialectAddToEnv(env, "F95", F95Suffixes, F95PPSuffixes,
                    support_module = 1)

def add_f03_to_env(env):
    """Add Builders and construction variables for f03 to an Environment."""
    try:
        F03Suffixes = env['F03FILESUFFIXES']
    except KeyError:
        F03Suffixes = ['.f03']

    #print("Adding %s to f95 suffixes" % F95Suffixes)
    try:
        F03PPSuffixes = env['F03PPFILESUFFIXES']
    except KeyError:
        F03PPSuffixes = []

    DialectAddToEnv(env, "F03", F03Suffixes, F03PPSuffixes,
                    support_module = 1)

def add_f08_to_env(env):
    """Add Builders and construction variables for f08 to an Environment."""
    try:
        F08Suffixes = env['F08FILESUFFIXES']
    except KeyError:
        F08Suffixes = ['.f08']

    try:
        F08PPSuffixes = env['F08PPFILESUFFIXES']
    except KeyError:
        F08PPSuffixes = []

    DialectAddToEnv(env, "F08", F08Suffixes, F08PPSuffixes,
                    support_module = 1)

def add_all_to_env(env):
    """Add builders and construction variables for all supported fortran
    dialects."""
    add_fortran_to_env(env)
    add_f77_to_env(env)
    add_f90_to_env(env)
    add_f95_to_env(env)
    add_f03_to_env(env)
    add_f08_to_env(env)

# Local Variables:
# tab-width:4
# indent-tabs-mode:nil
# End:
# vim: set expandtab tabstop=4 shiftwidth=4:
