"""SCons.Tool.rpmutils.py

RPM specific helper routines for general usage in the test framework
and SCons core modules.

Since we check for the RPM package target name in several places,
we have to know which machine/system name RPM will use for the current
hardware setup. The following dictionaries and functions try to
mimic the exact naming rules of the RPM source code.
They were directly derived from the file "rpmrc.in" of the version
rpm-4.9.1.3. For updating to a more recent version of RPM, this Python
script can be used standalone. The usage() function below shows the
exact syntax.

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
from __future__ import print_function

__revision__ = "src/engine/SCons/Tool/rpmutils.py rel_3.0.0:4395:8972f6a2f699 2017/09/18 12:59:24 bdbaddog"


import platform
import subprocess

import SCons.Util

# Start of rpmrc dictionaries (Marker, don't change or remove!)
os_canon = {
  'AIX' : ['AIX','5'],
  'AmigaOS' : ['AmigaOS','5'],
  'BSD_OS' : ['bsdi','12'],
  'CYGWIN32_95' : ['cygwin32','15'],
  'CYGWIN32_NT' : ['cygwin32','14'],
  'Darwin' : ['darwin','21'],
  'FreeBSD' : ['FreeBSD','8'],
  'HP-UX' : ['hpux10','6'],
  'IRIX' : ['Irix','2'],
  'IRIX64' : ['Irix64','10'],
  'Linux' : ['Linux','1'],
  'Linux/390' : ['OS/390','20'],
  'Linux/ESA' : ['VM/ESA','20'],
  'MacOSX' : ['macosx','21'],
  'MiNT' : ['FreeMiNT','17'],
  'NEXTSTEP' : ['NextStep','11'],
  'OS/390' : ['OS/390','18'],
  'OSF1' : ['osf1','7'],
  'SCO_SV' : ['SCO_SV3.2v5.0.2','9'],
  'SunOS4' : ['SunOS','4'],
  'SunOS5' : ['solaris','3'],
  'UNIX_SV' : ['MP_RAS','16'],
  'VM/ESA' : ['VM/ESA','19'],
  'machten' : ['machten','13'],
  'osf3.2' : ['osf1','7'],
  'osf4.0' : ['osf1','7'],
}

buildarch_compat = {
  'alpha' : ['noarch'],
  'alphaev5' : ['alpha'],
  'alphaev56' : ['alphaev5'],
  'alphaev6' : ['alphapca56'],
  'alphaev67' : ['alphaev6'],
  'alphapca56' : ['alphaev56'],
  'amd64' : ['x86_64'],
  'armv3l' : ['noarch'],
  'armv4b' : ['noarch'],
  'armv4l' : ['armv3l'],
  'armv4tl' : ['armv4l'],
  'armv5tejl' : ['armv5tel'],
  'armv5tel' : ['armv4tl'],
  'armv6l' : ['armv5tejl'],
  'armv7l' : ['armv6l'],
  'atariclone' : ['m68kmint','noarch'],
  'atarist' : ['m68kmint','noarch'],
  'atariste' : ['m68kmint','noarch'],
  'ataritt' : ['m68kmint','noarch'],
  'athlon' : ['i686'],
  'falcon' : ['m68kmint','noarch'],
  'geode' : ['i586'],
  'hades' : ['m68kmint','noarch'],
  'hppa1.0' : ['parisc'],
  'hppa1.1' : ['hppa1.0'],
  'hppa1.2' : ['hppa1.1'],
  'hppa2.0' : ['hppa1.2'],
  'i386' : ['noarch','fat'],
  'i486' : ['i386'],
  'i586' : ['i486'],
  'i686' : ['i586'],
  'ia32e' : ['x86_64'],
  'ia64' : ['noarch'],
  'm68k' : ['noarch'],
  'milan' : ['m68kmint','noarch'],
  'mips' : ['noarch'],
  'mipsel' : ['noarch'],
  'parisc' : ['noarch'],
  'pentium3' : ['i686'],
  'pentium4' : ['pentium3'],
  'ppc' : ['noarch','fat'],
  'ppc32dy4' : ['noarch'],
  'ppc64' : ['noarch','fat'],
  'ppc64iseries' : ['ppc64'],
  'ppc64pseries' : ['ppc64'],
  'ppc8260' : ['noarch'],
  'ppc8560' : ['noarch'],
  'ppciseries' : ['noarch'],
  'ppcpseries' : ['noarch'],
  's390' : ['noarch'],
  's390x' : ['noarch'],
  'sh3' : ['noarch'],
  'sh4' : ['noarch'],
  'sh4a' : ['sh4'],
  'sparc' : ['noarch'],
  'sparc64' : ['sparcv9v'],
  'sparc64v' : ['sparc64'],
  'sparcv8' : ['sparc'],
  'sparcv9' : ['sparcv8'],
  'sparcv9v' : ['sparcv9'],
  'sun4c' : ['noarch'],
  'sun4d' : ['noarch'],
  'sun4m' : ['noarch'],
  'sun4u' : ['noarch'],
  'x86_64' : ['noarch'],
}

os_compat = {
  'BSD_OS' : ['bsdi'],
  'Darwin' : ['MacOSX'],
  'FreeMiNT' : ['mint','MiNT','TOS'],
  'IRIX64' : ['IRIX'],
  'MiNT' : ['FreeMiNT','mint','TOS'],
  'TOS' : ['FreeMiNT','MiNT','mint'],
  'bsdi4.0' : ['bsdi'],
  'hpux10.00' : ['hpux9.07'],
  'hpux10.01' : ['hpux10.00'],
  'hpux10.10' : ['hpux10.01'],
  'hpux10.20' : ['hpux10.10'],
  'hpux10.30' : ['hpux10.20'],
  'hpux11.00' : ['hpux10.30'],
  'hpux9.05' : ['hpux9.04'],
  'hpux9.07' : ['hpux9.05'],
  'mint' : ['FreeMiNT','MiNT','TOS'],
  'ncr-sysv4.3' : ['ncr-sysv4.2'],
  'osf4.0' : ['osf3.2','osf1'],
  'solaris2.4' : ['solaris2.3'],
  'solaris2.5' : ['solaris2.3','solaris2.4'],
  'solaris2.6' : ['solaris2.3','solaris2.4','solaris2.5'],
  'solaris2.7' : ['solaris2.3','solaris2.4','solaris2.5','solaris2.6'],
}

arch_compat = {
  'alpha' : ['axp','noarch'],
  'alphaev5' : ['alpha'],
  'alphaev56' : ['alphaev5'],
  'alphaev6' : ['alphapca56'],
  'alphaev67' : ['alphaev6'],
  'alphapca56' : ['alphaev56'],
  'amd64' : ['x86_64','athlon','noarch'],
  'armv3l' : ['noarch'],
  'armv4b' : ['noarch'],
  'armv4l' : ['armv3l'],
  'armv4tl' : ['armv4l'],
  'armv5tejl' : ['armv5tel'],
  'armv5tel' : ['armv4tl'],
  'armv6l' : ['armv5tejl'],
  'armv7l' : ['armv6l'],
  'atariclone' : ['m68kmint','noarch'],
  'atarist' : ['m68kmint','noarch'],
  'atariste' : ['m68kmint','noarch'],
  'ataritt' : ['m68kmint','noarch'],
  'athlon' : ['i686'],
  'falcon' : ['m68kmint','noarch'],
  'geode' : ['i586'],
  'hades' : ['m68kmint','noarch'],
  'hppa1.0' : ['parisc'],
  'hppa1.1' : ['hppa1.0'],
  'hppa1.2' : ['hppa1.1'],
  'hppa2.0' : ['hppa1.2'],
  'i370' : ['noarch'],
  'i386' : ['noarch','fat'],
  'i486' : ['i386'],
  'i586' : ['i486'],
  'i686' : ['i586'],
  'ia32e' : ['x86_64','athlon','noarch'],
  'ia64' : ['noarch'],
  'milan' : ['m68kmint','noarch'],
  'mips' : ['noarch'],
  'mipsel' : ['noarch'],
  'osfmach3_i386' : ['i486'],
  'osfmach3_i486' : ['i486','osfmach3_i386'],
  'osfmach3_i586' : ['i586','osfmach3_i486'],
  'osfmach3_i686' : ['i686','osfmach3_i586'],
  'osfmach3_ppc' : ['ppc'],
  'parisc' : ['noarch'],
  'pentium3' : ['i686'],
  'pentium4' : ['pentium3'],
  'powerpc' : ['ppc'],
  'powerppc' : ['ppc'],
  'ppc' : ['rs6000'],
  'ppc32dy4' : ['ppc'],
  'ppc64' : ['ppc'],
  'ppc64iseries' : ['ppc64'],
  'ppc64pseries' : ['ppc64'],
  'ppc8260' : ['ppc'],
  'ppc8560' : ['ppc'],
  'ppciseries' : ['ppc'],
  'ppcpseries' : ['ppc'],
  'rs6000' : ['noarch','fat'],
  's390' : ['noarch'],
  's390x' : ['s390','noarch'],
  'sh3' : ['noarch'],
  'sh4' : ['noarch'],
  'sh4a' : ['sh4'],
  'sparc' : ['noarch'],
  'sparc64' : ['sparcv9'],
  'sparc64v' : ['sparc64'],
  'sparcv8' : ['sparc'],
  'sparcv9' : ['sparcv8'],
  'sparcv9v' : ['sparcv9'],
  'sun4c' : ['sparc'],
  'sun4d' : ['sparc'],
  'sun4m' : ['sparc'],
  'sun4u' : ['sparc64'],
  'x86_64' : ['amd64','athlon','noarch'],
}

buildarchtranslate = {
  'alphaev5' : ['alpha'],
  'alphaev56' : ['alpha'],
  'alphaev6' : ['alpha'],
  'alphaev67' : ['alpha'],
  'alphapca56' : ['alpha'],
  'amd64' : ['x86_64'],
  'armv3l' : ['armv3l'],
  'armv4b' : ['armv4b'],
  'armv4l' : ['armv4l'],
  'armv4tl' : ['armv4tl'],
  'armv5tejl' : ['armv5tejl'],
  'armv5tel' : ['armv5tel'],
  'armv6l' : ['armv6l'],
  'armv7l' : ['armv7l'],
  'atariclone' : ['m68kmint'],
  'atarist' : ['m68kmint'],
  'atariste' : ['m68kmint'],
  'ataritt' : ['m68kmint'],
  'athlon' : ['i386'],
  'falcon' : ['m68kmint'],
  'geode' : ['i386'],
  'hades' : ['m68kmint'],
  'i386' : ['i386'],
  'i486' : ['i386'],
  'i586' : ['i386'],
  'i686' : ['i386'],
  'ia32e' : ['x86_64'],
  'ia64' : ['ia64'],
  'milan' : ['m68kmint'],
  'osfmach3_i386' : ['i386'],
  'osfmach3_i486' : ['i386'],
  'osfmach3_i586' : ['i386'],
  'osfmach3_i686' : ['i386'],
  'osfmach3_ppc' : ['ppc'],
  'pentium3' : ['i386'],
  'pentium4' : ['i386'],
  'powerpc' : ['ppc'],
  'powerppc' : ['ppc'],
  'ppc32dy4' : ['ppc'],
  'ppc64iseries' : ['ppc64'],
  'ppc64pseries' : ['ppc64'],
  'ppc8260' : ['ppc'],
  'ppc8560' : ['ppc'],
  'ppciseries' : ['ppc'],
  'ppcpseries' : ['ppc'],
  's390' : ['s390'],
  's390x' : ['s390x'],
  'sh3' : ['sh3'],
  'sh4' : ['sh4'],
  'sh4a' : ['sh4'],
  'sparc64v' : ['sparc64'],
  'sparcv8' : ['sparc'],
  'sparcv9' : ['sparc'],
  'sparcv9v' : ['sparc'],
  'sun4c' : ['sparc'],
  'sun4d' : ['sparc'],
  'sun4m' : ['sparc'],
  'sun4u' : ['sparc64'],
  'x86_64' : ['x86_64'],
}

optflags = {
  'alpha' : ['-O2','-g','-mieee'],
  'alphaev5' : ['-O2','-g','-mieee','-mtune=ev5'],
  'alphaev56' : ['-O2','-g','-mieee','-mtune=ev56'],
  'alphaev6' : ['-O2','-g','-mieee','-mtune=ev6'],
  'alphaev67' : ['-O2','-g','-mieee','-mtune=ev67'],
  'alphapca56' : ['-O2','-g','-mieee','-mtune=pca56'],
  'amd64' : ['-O2','-g'],
  'armv3l' : ['-O2','-g','-march=armv3'],
  'armv4b' : ['-O2','-g','-march=armv4'],
  'armv4l' : ['-O2','-g','-march=armv4'],
  'armv4tl' : ['-O2','-g','-march=armv4t'],
  'armv5tejl' : ['-O2','-g','-march=armv5te'],
  'armv5tel' : ['-O2','-g','-march=armv5te'],
  'armv6l' : ['-O2','-g','-march=armv6'],
  'armv7l' : ['-O2','-g','-march=armv7'],
  'atariclone' : ['-O2','-g','-fomit-frame-pointer'],
  'atarist' : ['-O2','-g','-fomit-frame-pointer'],
  'atariste' : ['-O2','-g','-fomit-frame-pointer'],
  'ataritt' : ['-O2','-g','-fomit-frame-pointer'],
  'athlon' : ['-O2','-g','-march=athlon'],
  'falcon' : ['-O2','-g','-fomit-frame-pointer'],
  'fat' : ['-O2','-g','-arch','i386','-arch','ppc'],
  'geode' : ['-Os','-g','-m32','-march=geode'],
  'hades' : ['-O2','-g','-fomit-frame-pointer'],
  'hppa1.0' : ['-O2','-g','-mpa-risc-1-0'],
  'hppa1.1' : ['-O2','-g','-mpa-risc-1-0'],
  'hppa1.2' : ['-O2','-g','-mpa-risc-1-0'],
  'hppa2.0' : ['-O2','-g','-mpa-risc-1-0'],
  'i386' : ['-O2','-g','-march=i386','-mtune=i686'],
  'i486' : ['-O2','-g','-march=i486'],
  'i586' : ['-O2','-g','-march=i586'],
  'i686' : ['-O2','-g','-march=i686'],
  'ia32e' : ['-O2','-g'],
  'ia64' : ['-O2','-g'],
  'm68k' : ['-O2','-g','-fomit-frame-pointer'],
  'milan' : ['-O2','-g','-fomit-frame-pointer'],
  'mips' : ['-O2','-g'],
  'mipsel' : ['-O2','-g'],
  'parisc' : ['-O2','-g','-mpa-risc-1-0'],
  'pentium3' : ['-O2','-g','-march=pentium3'],
  'pentium4' : ['-O2','-g','-march=pentium4'],
  'ppc' : ['-O2','-g','-fsigned-char'],
  'ppc32dy4' : ['-O2','-g','-fsigned-char'],
  'ppc64' : ['-O2','-g','-fsigned-char'],
  'ppc8260' : ['-O2','-g','-fsigned-char'],
  'ppc8560' : ['-O2','-g','-fsigned-char'],
  'ppciseries' : ['-O2','-g','-fsigned-char'],
  'ppcpseries' : ['-O2','-g','-fsigned-char'],
  's390' : ['-O2','-g'],
  's390x' : ['-O2','-g'],
  'sh3' : ['-O2','-g'],
  'sh4' : ['-O2','-g','-mieee'],
  'sh4a' : ['-O2','-g','-mieee'],
  'sparc' : ['-O2','-g','-m32','-mtune=ultrasparc'],
  'sparc64' : ['-O2','-g','-m64','-mtune=ultrasparc'],
  'sparc64v' : ['-O2','-g','-m64','-mtune=niagara'],
  'sparcv8' : ['-O2','-g','-m32','-mtune=ultrasparc','-mv8'],
  'sparcv9' : ['-O2','-g','-m32','-mtune=ultrasparc'],
  'sparcv9v' : ['-O2','-g','-m32','-mtune=niagara'],
  'x86_64' : ['-O2','-g'],
}

arch_canon = {
  'IP' : ['sgi','7'],
  'alpha' : ['alpha','2'],
  'alphaev5' : ['alphaev5','2'],
  'alphaev56' : ['alphaev56','2'],
  'alphaev6' : ['alphaev6','2'],
  'alphaev67' : ['alphaev67','2'],
  'alphapca56' : ['alphapca56','2'],
  'amd64' : ['amd64','1'],
  'armv3l' : ['armv3l','12'],
  'armv4b' : ['armv4b','12'],
  'armv4l' : ['armv4l','12'],
  'armv5tejl' : ['armv5tejl','12'],
  'armv5tel' : ['armv5tel','12'],
  'armv6l' : ['armv6l','12'],
  'armv7l' : ['armv7l','12'],
  'atariclone' : ['m68kmint','13'],
  'atarist' : ['m68kmint','13'],
  'atariste' : ['m68kmint','13'],
  'ataritt' : ['m68kmint','13'],
  'athlon' : ['athlon','1'],
  'falcon' : ['m68kmint','13'],
  'geode' : ['geode','1'],
  'hades' : ['m68kmint','13'],
  'i370' : ['i370','14'],
  'i386' : ['i386','1'],
  'i486' : ['i486','1'],
  'i586' : ['i586','1'],
  'i686' : ['i686','1'],
  'ia32e' : ['ia32e','1'],
  'ia64' : ['ia64','9'],
  'm68k' : ['m68k','6'],
  'm68kmint' : ['m68kmint','13'],
  'milan' : ['m68kmint','13'],
  'mips' : ['mips','4'],
  'mipsel' : ['mipsel','11'],
  'pentium3' : ['pentium3','1'],
  'pentium4' : ['pentium4','1'],
  'ppc' : ['ppc','5'],
  'ppc32dy4' : ['ppc32dy4','5'],
  'ppc64' : ['ppc64','16'],
  'ppc64iseries' : ['ppc64iseries','16'],
  'ppc64pseries' : ['ppc64pseries','16'],
  'ppc8260' : ['ppc8260','5'],
  'ppc8560' : ['ppc8560','5'],
  'ppciseries' : ['ppciseries','5'],
  'ppcpseries' : ['ppcpseries','5'],
  'rs6000' : ['rs6000','8'],
  's390' : ['s390','14'],
  's390x' : ['s390x','15'],
  'sh' : ['sh','17'],
  'sh3' : ['sh3','17'],
  'sh4' : ['sh4','17'],
  'sh4a' : ['sh4a','17'],
  'sparc' : ['sparc','3'],
  'sparc64' : ['sparc64','2'],
  'sparc64v' : ['sparc64v','2'],
  'sparcv8' : ['sparcv8','3'],
  'sparcv9' : ['sparcv9','3'],
  'sparcv9v' : ['sparcv9v','3'],
  'sun4' : ['sparc','3'],
  'sun4c' : ['sparc','3'],
  'sun4d' : ['sparc','3'],
  'sun4m' : ['sparc','3'],
  'sun4u' : ['sparc64','2'],
  'x86_64' : ['x86_64','1'],
  'xtensa' : ['xtensa','18'],
}

# End of rpmrc dictionaries (Marker, don't change or remove!)

def defaultMachine(use_rpm_default=True):
    """ Return the canonicalized machine name. """

    if use_rpm_default:
        try:
            # This should be the most reliable way to get the default arch
            rmachine = subprocess.check_output(['rpm', '--eval=%_target_cpu'], shell=False).rstrip()
            rmachine = SCons.Util.to_str(rmachine)
        except Exception as e:
            # Something went wrong, try again by looking up platform.machine()
            return defaultMachine(False)
    else:
        rmachine = platform.machine()

        # Try to lookup the string in the canon table
        if rmachine in arch_canon:
            rmachine = arch_canon[rmachine][0]

    return rmachine

def defaultSystem():
    """ Return the canonicalized system name. """
    rsystem = platform.system()

    # Try to lookup the string in the canon tables
    if rsystem in os_canon:
        rsystem = os_canon[rsystem][0]

    return rsystem

def defaultNames():
    """ Return the canonicalized machine and system name. """
    return defaultMachine(), defaultSystem()

def updateRpmDicts(rpmrc, pyfile):
    """ Read the given rpmrc file with RPM definitions and update the
        info dictionaries in the file pyfile with it.
        The arguments will usually be 'rpmrc.in' from a recent RPM source
        tree, and 'rpmutils.py' referring to this script itself.
        See also usage() below.
    """
    try:
        # Read old rpmutils.py file
        oldpy = open(pyfile,"r").readlines()
        # Read current rpmrc.in file
        rpm = open(rpmrc,"r").readlines()
        # Parse for data
        data = {}
        # Allowed section names that get parsed
        sections = ['optflags',
                    'arch_canon',
                    'os_canon',
                    'buildarchtranslate',
                    'arch_compat',
                    'os_compat',
                    'buildarch_compat']
        for l in rpm:
            l = l.rstrip('\n').replace(':',' ')
            # Skip comments
            if l.lstrip().startswith('#'):
                continue
            tokens = l.strip().split()
            if len(tokens):
                key = tokens[0]
                if key in sections:
                    # Have we met this section before?
                    if tokens[0] not in data:
                        # No, so insert it
                        data[key] = {}
                    # Insert data
                    data[key][tokens[1]] = tokens[2:]
        # Write new rpmutils.py file
        out = open(pyfile,"w")
        pm = 0
        for l in oldpy:
            if pm:
                if l.startswith('# End of rpmrc dictionaries'):
                    pm = 0
                    out.write(l)
            else:
                out.write(l)
                if l.startswith('# Start of rpmrc dictionaries'):
                    pm = 1
                    # Write data sections to single dictionaries
                    for key, entries in data.items():
                        out.write("%s = {\n" % key)
                        for arch in sorted(entries.keys()):
                            out.write("  '%s' : ['%s'],\n" % (arch, "','".join(entries[arch])))
                        out.write("}\n\n")
        out.close()
    except:
        pass

def usage():
    print("rpmutils.py rpmrc.in rpmutils.py")

def main():
    import sys

    if len(sys.argv) < 3:
        usage()
        sys.exit(0)
    updateRpmDicts(sys.argv[1], sys.argv[2])

if __name__ == "__main__":
    main()
