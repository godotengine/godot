#!/usr/bin/env python

# Copyright 2017 The Chromium Authors. All rights reserved.
# Use of this source code is governed by a BSD-style license that can be
# found in the LICENSE file.

import _winreg
import os
import re
import subprocess
import sys


def _ExtractImportantEnvironment(output_of_set):
  """Extracts environment variables required for the toolchain to run from
  a textual dump output by the cmd.exe 'set' command."""
  envvars_to_save = (
      'include',
      'lib',
      'libpath',
      'path',
      'pathext',
      'systemroot',
      'temp',
      'tmp',
      )
  env = {}
  for line in output_of_set.splitlines():
    for envvar in envvars_to_save:
      if re.match(envvar + '=', line.lower()):
        var, setting = line.split('=', 1)
        env[var.upper()] = setting
        break
  for required in ('SYSTEMROOT', 'TEMP', 'TMP'):
    if required not in env:
      raise Exception('Environment variable "%s" '
                      'required to be set to valid path' % required)
  return env


def _FormatAsEnvironmentBlock(envvar_dict):
  """Format as an 'environment block' directly suitable for CreateProcess.
  Briefly this is a list of key=value\0, terminated by an additional \0. See
  CreateProcess() documentation for more details."""
  block = ''
  nul = '\0'
  for key, value in envvar_dict.iteritems():
    block += key + '=' + value + nul
  block += nul
  return block


def _GenerateEnvironmentFiles(install_dir, out_dir, script_path):
  """It's not sufficient to have the absolute path to the compiler, linker, etc.
  on Windows, as those tools rely on .dlls being in the PATH. We also need to
  support both x86 and x64 compilers. Different architectures require a
  different compiler binary, and different supporting environment variables
  (INCLUDE, LIB, LIBPATH). So, we extract the environment here, wrap all
  invocations of compiler tools (cl, link, lib, rc, midl, etc.) to set up the
  environment, and then do not prefix the compiler with an absolute path,
  instead preferring something like "cl.exe" in the rule which will then run
  whichever the environment setup has put in the path."""
  archs = ('x86', 'amd64')
  result = []
  for arch in archs:
    # Extract environment variables for subprocesses.
    args = [os.path.join(install_dir, script_path)]
    script_arch_name = arch
    if script_path.endswith('SetEnv.cmd') and arch == 'amd64':
      script_arch_name = '/x64'
    args.extend((script_arch_name, '&&', 'set'))
    popen = subprocess.Popen(
        args, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    variables, _ = popen.communicate()
    if popen.returncode != 0:
      raise Exception('"%s" failed with error %d' % (args, popen.returncode))
    env = _ExtractImportantEnvironment(variables)

    env_block = _FormatAsEnvironmentBlock(env)
    basename = 'environment.' + arch
    with open(os.path.join(out_dir, basename), 'wb') as f:
      f.write(env_block)
    result.append(basename)
  return result


def _GetEnvAsDict(arch):
  """Gets the saved environment from a file for a given architecture."""
  # The environment is saved as an "environment block" (see CreateProcess()
  # for details, which is the format required for ninja). We convert to a dict
  # here. Drop last 2 NULs, one for list terminator, one for trailing vs.
  # separator.
  pairs = open(arch).read()[:-2].split('\0')
  kvs = [item.split('=', 1) for item in pairs]
  return dict(kvs)


class WinTool(object):
  def Dispatch(self, args):
    """Dispatches a string command to a method."""
    if len(args) < 1:
      raise Exception("Not enough arguments")

    method = "Exec%s" % self._CommandifyName(args[0])
    return getattr(self, method)(*args[1:])

  def _CommandifyName(self, name_string):
    """Transforms a tool name like recursive-mirror to RecursiveMirror."""
    return name_string.title().replace('-', '')

  def ExecLinkWrapper(self, arch, *args):
    """Filter diagnostic output from link that looks like:
    '   Creating library ui.dll.lib and object ui.dll.exp'
    This happens when there are exports from the dll or exe.
    """
    env = _GetEnvAsDict(arch)
    args = list(args)  # *args is a tuple by default, which is read-only.
    args[0] = args[0].replace('/', '\\')
    link = subprocess.Popen(args, env=env, shell=True, stdout=subprocess.PIPE)
    out, _ = link.communicate()
    for line in out.splitlines():
      if (not line.startswith('   Creating library ') and
          not line.startswith('Generating code') and
          not line.startswith('Finished generating code')):
        print line
    return link.returncode

  def ExecAsmWrapper(self, arch, *args):
    """Filter logo banner from invocations of asm.exe."""
    env = _GetEnvAsDict(arch)
    popen = subprocess.Popen(args, env=env, shell=True,
                             stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    out, _ = popen.communicate()
    for line in out.splitlines():
      if (not line.startswith('Copyright (C) Microsoft Corporation') and
          not line.startswith('Microsoft (R) Macro Assembler') and
          not line.startswith(' Assembling: ') and
          line):
        print line
    return popen.returncode

  def _GetVisualStudioInstallDirOrDie(self):
    # Try vswhere, which will find VS2017.2+. Note that earlier VS2017s will not
    # be found.
    vswhere_path = os.path.join(os.environ.get('ProgramFiles(x86)'),
        'Microsoft Visual Studio', 'Installer', 'vswhere.exe')
    if os.path.exists(vswhere_path):
      installation_path = subprocess.check_output(
          [vswhere_path, '-latest', '-property', 'installationPath']).strip()
      if installation_path:
        return (installation_path,
                os.path.join('VC', 'Auxiliary', 'Build', 'vcvarsall.bat'))

    raise Exception('Visual Studio installation dir not found')

  def ExecGetVisualStudioData(self, outdir, toolchain_path):
    # Use an explicitly specified toolchain path, if provided and found.
    setenv_path = os.path.join('win_sdk', 'bin', 'SetEnv.cmd')
    if os.path.exists(os.path.join(toolchain_path, setenv_path)):
      install_dir, script_path = toolchain_path, setenv_path
    else:
      # Otherwise, try to autodetect.
      install_dir, script_path = self._GetVisualStudioInstallDirOrDie()

    x86_file, x64_file = _GenerateEnvironmentFiles(
        install_dir, outdir, script_path)
    result = '''install_dir = "%s"
x86_environment_file = "%s"
x64_environment_file = "%s"''' % (install_dir, x86_file, x64_file)
    print result
    return 0

  def ExecStamp(self, path):
    """Simple stamp command."""
    open(path, 'w').close()
    return 0


if __name__ == '__main__':
  sys.exit(WinTool().Dispatch(sys.argv[1:]))
