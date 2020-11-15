#!/usr/bin/env python
# coding: utf-8

# Copyright 2012 The Chromium Authors. All rights reserved.
# Use of this source code is governed by a BSD-style license that can be
# found in the LICENSE file.

from __future__ import print_function

import argparse
import distutils.version
import os
import re
import subprocess
import sys
import textwrap


def _AsVersion(string):
  return distutils.version.StrictVersion(string)


def _RunXCRun(args, sdk=None):
  xcrun_args = ['xcrun']
  if sdk is not None:
    xcrun_args.extend(['--sdk', sdk])
  xcrun_args.extend(args)
  return subprocess.check_output(xcrun_args).decode('utf-8').rstrip()


def _SDKPath(sdk=None):
  return _RunXCRun(['--show-sdk-path'], sdk)


def _SDKVersion(sdk=None):
  return _AsVersion(_RunXCRun(['--show-sdk-version'], sdk))


class DidNotMeetCriteria(Exception):
  pass


def _FindPlatformSDKWithMinimumVersion(platform, minimum_sdk_version_str):
  minimum_sdk_version = _AsVersion(minimum_sdk_version_str)

  # Try the SDKs that Xcode knows about.
  xcodebuild_showsdks_subprocess = subprocess.Popen(
      ['xcodebuild', '-showsdks'],
      stdout=subprocess.PIPE,
      stderr=open(os.devnull, 'w'))
  xcodebuild_showsdks_output = (
      xcodebuild_showsdks_subprocess.communicate()[0].decode('utf-8'))
  if xcodebuild_showsdks_subprocess.returncode == 0:
    # Collect strings instead of version objects to preserve the precise
    # format used to identify each SDK.
    sdk_version_strs = []
    for line in xcodebuild_showsdks_output.splitlines():
      match = re.match('[ \t].+[ \t]-sdk ' + re.escape(platform) + '(.+)$',
                       line)
      if match:
        sdk_version_str = match.group(1)
        if _AsVersion(sdk_version_str) >= minimum_sdk_version:
          sdk_version_strs.append(sdk_version_str)

    if len(sdk_version_strs) == 0:
      raise DidNotMeetCriteria({'minimum': minimum_sdk_version_str,
                                'platform': platform})
    sdk_version_str = sorted(sdk_version_strs, key=_AsVersion)[0]
    sdk_path = _SDKPath(platform + sdk_version_str)
    sdk_version = _AsVersion(sdk_version_str)
  else:
    # Xcode may not be installed. If the command-line tools are installed, use
    # the system’s default SDK if it meets the requirements.
    sdk_path = _SDKPath()
    sdk_version = _SDKVersion()
    if sdk_version < minimum_sdk_version:
      raise DidNotMeetCriteria({'minimum': minimum_sdk_version_str,
                                'platform': platform,
                                'sdk_path': sdk_path,
                                'sdk_version': str(sdk_version)})

  return (sdk_version, sdk_path)


def main(args):
  parser = argparse.ArgumentParser(
      description='Find an appropriate platform SDK',
      epilog='Two lines will be written to standard output: the version of the '
             'selected SDK, and its path.')
  parser.add_argument('--developer-dir',
                      help='path to Xcode or Command Line Tools')
  parser.add_argument('--exact', help='an exact SDK version to find')
  parser.add_argument('--minimum', help='the minimum SDK version to find')
  parser.add_argument('--path', help='a known SDK path to validate')
  parser.add_argument('--platform',
                      default='macosx',
                      help='the platform to target')
  parsed = parser.parse_args(args)

  if parsed.developer_dir is not None:
    os.environ['DEVELOPER_DIR'] = parsed.developer_dir

  if (os.environ.get('DEVELOPER_DIR') is None and
      subprocess.call(['xcode-select', '--print-path'],
                      stdout=open(os.devnull, 'w'),
                      stderr=open(os.devnull, 'w')) != 0):
    # This is friendlier than letting the first invocation of xcrun or
    # xcodebuild show the UI prompting to install developer tools at an
    # inopportune time.
    hint = 'Install Xcode and run "sudo xcodebuild -license"'
    if parsed.platform == 'macosx':
      hint += ', or install Command Line Tools with "xcode-select --install"'
    hint += ('. If necessary, run "sudo xcode-select --switch" to select an '
             'active developer tools installation.')
    hint = '\n'.join(textwrap.wrap(hint, 80))
    print(os.path.basename(sys.argv[0]) +
              ': No developer tools found.\n' +
              hint,
          file=sys.stderr)
    return 1

  if parsed.path is not None:
    # _SDKVersion() doesn’t work with a relative pathname argument or one that’s
    # a symbolic link. Such paths are suitable for other purposes, like “clang
    # -isysroot”, so use an absolute non-symbolic link path for _SDKVersion(),
    # but preserve the user’s path in sdk_path.
    sdk_version = _SDKVersion(os.path.realpath(parsed.path))
    sdk_path = parsed.path
  elif parsed.exact is None and parsed.minimum is None:
    # Use the platform’s default SDK.
    sdk_version = _SDKVersion(parsed.platform)
    sdk_path = _SDKPath(parsed.platform)
  elif parsed.exact is not None:
    sdk_version = _SDKVersion(parsed.platform + parsed.exact)
    sdk_path = _SDKPath(parsed.platform + parsed.exact)
  else:
    (sdk_version,
     sdk_path) = _FindPlatformSDKWithMinimumVersion(parsed.platform,
                                                    parsed.minimum)

  # These checks may be redundant depending on how the SDK was chosen.
  if ((parsed.exact is not None and sdk_version != _AsVersion(parsed.exact)) or
      (parsed.minimum is not None and
       sdk_version < _AsVersion(parsed.minimum))):
    raise DidNotMeetCriteria({'developer_dir': parsed.developer_dir,
                              'exact': parsed.exact,
                              'minimum': parsed.minimum,
                              'path': parsed.path,
                              'platform': parsed.platform,
                              'sdk_path': sdk_path,
                              'sdk_version': str(sdk_version)})

  # Nobody wants trailing slashes. This is true even if “/” is the SDK: it’s
  # better to return an empty string, which will be interpreted as “no sysroot.”
  sdk_path = sdk_path.rstrip(os.path.sep)

  print(sdk_version)
  print(sdk_path)

  return 0


if __name__ == '__main__':
  sys.exit(main(sys.argv[1:]))
