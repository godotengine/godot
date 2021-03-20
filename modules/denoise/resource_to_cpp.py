#!/usr/bin/env python

## ======================================================================== ##
## Copyright 2009-2019 Intel Corporation                                    ##
##                                                                          ##
## Licensed under the Apache License, Version 2.0 (the "License");          ##
## you may not use this file except in compliance with the License.         ##
## You may obtain a copy of the License at                                  ##
##                                                                          ##
##     http://www.apache.org/licenses/LICENSE-2.0                           ##
##                                                                          ##
## Unless required by applicable law or agreed to in writing, software      ##
## distributed under the License is distributed on an "AS IS" BASIS,        ##
## WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. ##
## See the License for the specific language governing permissions and      ##
## limitations under the License.                                           ##
## ======================================================================== ##

import os
from array import array

# Generates a C++ file from the specified binary resource file
def generate(in_path, out_path):

    namespace = "oidn::weights"
    scopes = namespace.split("::")

    file_name = os.path.basename(in_path)
    var_name = os.path.splitext(file_name)[0]

    with open(in_path, "rb") as in_file, open(out_path, "w") as out_file:
        # Header
        out_file.write("// Generated from: %s\n" % file_name)
        out_file.write("#include <cstddef>\n\n")

        # Open the namespaces
        for s in scopes:
            out_file.write("namespace %s {\n" % s)
        if scopes:
            out_file.write("\n")

        # Read the file
        in_data = array("B", in_file.read())

        # Write the size
        out_file.write("//const size_t %s_size = %d;\n\n" % (var_name, len(in_data)))

        # Write the data
        out_file.write("unsigned char %s[] = {" % var_name)
        for i in range(len(in_data)):
            c = in_data[i]
            if i > 0:
                out_file.write(",")
            if (i + 1) % 20 == 1:
                out_file.write("\n")
            out_file.write("%d" % c)
        out_file.write("\n};\n")

        # Close the namespaces
        if scopes:
            out_file.write("\n")
        for scope in reversed(scopes):
            out_file.write("} // namespace %s\n" % scope)


def tza_to_cpp(target, source, env):
    for x in zip(source, target):
        generate(str(x[0]), str(x[1]))
