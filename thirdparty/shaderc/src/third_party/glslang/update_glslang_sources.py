#!/usr/bin/env python

# Copyright 2017 The Glslang Authors. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Get source files for Glslang and its dependencies from public repositories.
"""

from __future__ import print_function

import argparse
import json
import distutils.dir_util
import os.path
import subprocess
import sys

KNOWN_GOOD_FILE = 'known_good.json'

SITE_TO_KNOWN_GOOD_FILE = { 'github' : 'known_good.json',
                            'gitlab' : 'known_good_khr.json' }

# Maps a site name to its hostname.
SITE_TO_HOST = { 'github' : 'https://github.com/',
                 'gitlab' : 'git@gitlab.khronos.org:' }

VERBOSE = True


def command_output(cmd, directory, fail_ok=False):
    """Runs a command in a directory and returns its standard output stream.

    Captures the standard error stream.

    Raises a RuntimeError if the command fails to launch or otherwise fails.
    """
    if VERBOSE:
        print('In {d}: {cmd}'.format(d=directory, cmd=cmd))
    p = subprocess.Popen(cmd,
                         cwd=directory,
                         stdout=subprocess.PIPE)
    (stdout, _) = p.communicate()
    if p.returncode != 0 and not fail_ok:
        raise RuntimeError('Failed to run {} in {}'.format(cmd, directory))
    if VERBOSE:
        print(stdout)
    return stdout


def command_retval(cmd, directory):
    """Runs a command in a directory and returns its return value.

    Captures the standard error stream.
    """
    p = subprocess.Popen(cmd,
                         cwd=directory,
                         stdout=subprocess.PIPE)
    p.communicate()
    return p.returncode


class GoodCommit(object):
    """Represents a good commit for a repository."""

    def __init__(self, json):
        """Initializes this good commit object.

        Args:
        'json':  A fully populated JSON object describing the commit.
        """
        self._json = json
        self.name = json['name']
        self.site = json['site']
        self.subrepo = json['subrepo']
        self.subdir = json['subdir'] if ('subdir' in json) else '.'
        self.commit = json['commit']

    def GetUrl(self):
        """Returns the URL for the repository."""
        host = SITE_TO_HOST[self.site]
        return '{host}{subrepo}'.format(
                    host=host,
                    subrepo=self.subrepo)

    def AddRemote(self):
        """Add the remote 'known-good' if it does not exist."""
        remotes = command_output(['git', 'remote'], self.subdir).splitlines()
        if b'known-good' not in remotes:
            command_output(['git', 'remote', 'add', 'known-good', self.GetUrl()], self.subdir)

    def HasCommit(self):
        """Check if the repository contains the known-good commit."""
        return 0 == subprocess.call(['git', 'rev-parse', '--verify', '--quiet',
                                     self.commit + "^{commit}"],
                                    cwd=self.subdir)

    def Clone(self):
        distutils.dir_util.mkpath(self.subdir)
        command_output(['git', 'clone', self.GetUrl(), '.'], self.subdir)

    def Fetch(self):
        command_output(['git', 'fetch', 'known-good'], self.subdir)

    def Checkout(self):
        if not os.path.exists(os.path.join(self.subdir,'.git')):
            self.Clone()
        self.AddRemote()
        if not self.HasCommit():
            self.Fetch()
        command_output(['git', 'checkout', self.commit], self.subdir)


def GetGoodCommits(site):
    """Returns the latest list of GoodCommit objects."""
    known_good_file = SITE_TO_KNOWN_GOOD_FILE[site]
    with open(known_good_file) as known_good:
        return [GoodCommit(c) for c in json.loads(known_good.read())['commits']]


def main():
    parser = argparse.ArgumentParser(description='Get Glslang source dependencies at a known-good commit')
    parser.add_argument('--dir', dest='dir', default='.',
                        help="Set target directory for Glslang source root. Default is \'.\'.")
    parser.add_argument('--site', dest='site', default='github',
                        help="Set git server site. Default is github.")

    args = parser.parse_args()

    commits = GetGoodCommits(args.site)

    distutils.dir_util.mkpath(args.dir)
    print('Change directory to {d}'.format(d=args.dir))
    os.chdir(args.dir)

    # Create the subdirectories in sorted order so that parent git repositories
    # are created first.
    for c in sorted(commits, key=lambda x: x.subdir):
        print('Get {n}\n'.format(n=c.name))
        c.Checkout()
    sys.exit(0)


if __name__ == '__main__':
    main()
