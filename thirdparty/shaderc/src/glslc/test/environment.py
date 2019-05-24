# Copyright 2015 The Shaderc Authors. All rights reserved.
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

"""Classes for conveniently specifying a test environment.

These classes have write() methods that create objects in a test's environment.
For instance, File creates a file, and Directory creates a directory with some
files or subdirectories in it.

Example:
  test.environment = Directory('.', [
    File('a.vert', 'void main(){}'),
    Directory('subdir', [
      File('b', 'b content'),
      File('c', 'c content')
    ])
  ])

In general, these classes don't clean up the disk content they create.  They
were written in a test framework that handles clean-up at a higher level.

"""

import os

class Directory:
    """Specifies a directory or a subdirectory."""
    def __init__(self, name, content):
        """content is a list of File or Directory objects."""
        self.name = name
        self.content = content

    @staticmethod
    def create(path):
        """Creates a directory path if it doesn't already exist."""
        try:
            os.makedirs(path)
        except OSError: # Handles both pre-existence and a racing creation.
            if not os.path.isdir(path):
                raise

    def write(self, parent):
        """Creates a self.name directory within parent (which is a string path) and
        recursively writes the content in it.

        """
        full_path = os.path.join(parent, self.name)
        Directory.create(full_path)
        for c in self.content:
            c.write(full_path)

class File:
    """Specifies a file by name and content."""
    def __init__(self, name, content):
        self.name = name
        self.content = content

    def write(self, directory):
        """Writes content to directory/name."""
        with open(os.path.join(directory, self.name), 'w') as f:
            f.write(self.content)
