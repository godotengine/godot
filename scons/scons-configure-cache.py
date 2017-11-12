#! /usr/bin/env python
#
# SCons - a Software Constructor
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

from __future__ import print_function

__revision__ = "src/script/scons-configure-cache.py rel_3.0.0:4395:8972f6a2f699 2017/09/18 12:59:24 bdbaddog"

__version__ = "3.0.0"

__build__ = "rel_3.0.0:4395:8972f6a2f699"

__buildsys__ = "ubuntu-16"

__date__ = "2017/09/18 12:59:24"

__developer__ = "bdbaddog"

import argparse
import glob
import json
import os

def rearrange_cache_entries(current_prefix_len, new_prefix_len):
    print('Changing prefix length from', current_prefix_len, 'to', new_prefix_len)
    dirs = set()
    old_dirs = set()
    for file in glob.iglob(os.path.join('*', '*')):
        name = os.path.basename(file)
        dir = name[:current_prefix_len].upper()
        if dir not in old_dirs:
            print('Migrating', dir)
            old_dirs.add(dir)
        dir = name[:new_prefix_len].upper()
        if dir not in dirs:
            os.mkdir(dir)
            dirs.add(dir)
        os.rename(file, os.path.join(dir, name))

    # Now delete the original directories
    for dir in old_dirs:
        os.rmdir(dir)

# This dictionary should have one entry per entry in the cache config
# Each entry should have the following:
#   implicit - (optional) This is to allow adding a new config entry and also
#              changing the behaviour of the system at the same time. This
#              indicates the value the config entry would have had if it had been
#              specified.
#   default - The value the config entry should have if it wasn't previously
#             specified
#   command-line - parameters to pass to ArgumentParser.add_argument
#   converter - (optional) Function to call if it's necessary to do some work
#               if this configuration entry changes
config_entries = {
    'prefix_len' : { 
        'implicit' : 1, 
        'default' : 2 ,
        'command-line' : {
            'help' : 'Length of cache file name used as subdirectory prefix',
            'metavar' : '<number>',
            'type' : int
            },
        'converter' : rearrange_cache_entries
    }
}
parser = argparse.ArgumentParser(
    description = 'Modify the configuration of an scons cache directory',
    epilog = '''
             Unless you specify an option, it will not be changed (if it is
             already set in the cache config), or changed to an appropriate
             default (it it is not set).
             '''
    )

parser.add_argument('cache-dir', help='Path to scons cache directory')
for param in config_entries:
    parser.add_argument('--' + param.replace('_', '-'), 
                        **config_entries[param]['command-line'])
parser.add_argument('--version', action='version', version='%(prog)s 1.0')

# Get the command line as a dict without any of the unspecified entries.
args = dict([x for x in vars(parser.parse_args()).items() if x[1]])

# It seems somewhat strange to me, but positional arguments don't get the -
# in the name changed to _, whereas optional arguments do...
os.chdir(args['cache-dir'])
del args['cache-dir']

if not os.path.exists('config'):
    # Validate the only files in the directory are directories 0-9, a-f
    expected = [ '{:X}'.format(x) for x in range(0, 16) ]
    if not set(os.listdir('.')).issubset(expected):
        raise RuntimeError("This doesn't look like a version 1 cache directory")
    config = dict()
else:
    with open('config') as conf:
        config = json.load(conf)

# Find any keys that aren't currently set but should be
for key in config_entries:
    if key not in config:
        if 'implicit' in config_entries[key]:
            config[key] = config_entries[key]['implicit']
        else:
            config[key] = config_entries[key]['default']
        if key not in args:
            args[key] = config_entries[key]['default']

#Now we go through each entry in args to see if it changes an existing config
#setting.
for key in args:
    if args[key] != config[key]:        
        if 'converter' in config_entries[key]:
            config_entries[key]['converter'](config[key], args[key])
        config[key] = args[key]

# and write the updated config file
with open('config', 'w') as conf:
    json.dump(config, conf)
