#!/usr/bin/env python3

"""
Checks that all of the "catch_foo_all.hpp" headers include all subheaders.

The logic is simple: given a folder, e.g. `catch2/matchers`, then the
ccorresponding header is called `catch_matchers_all.hpp` and contains
* all headers in `catch2/matchers`,
* all headers in `catch2/matchers/{internal, detail}`,
* all convenience catch_matchers_*_all.hpp headers from any non-internal subfolders

The top level header is called `catch_all.hpp`.
"""

internal_dirs = ['detail', 'internal']

from scriptCommon import catchPath
from glob import glob
from pprint import pprint
import os
import re

def normalized_path(path):
    r"""Replaces \ in paths on Windows with /"""
    return path.replace('\\', '/')

def normalized_paths(paths):
    r"""Replaces \ with / in every path"""
    return [normalized_path(path) for path in paths]

source_path = catchPath + '/src/catch2'
source_path = normalized_path(source_path)
include_parser = re.compile(r'#include <(catch2/.+\.hpp)>')

errors_found = False

def headers_in_folder(folder):
    return glob(folder + '/*.hpp')

def folders_in_folder(folder):
    return [x for x in os.scandir(folder) if x.is_dir()]

def collated_includes(folder):
    base = headers_in_folder(folder)
    for subfolder in folders_in_folder(folder):
        if subfolder.name in internal_dirs:
            base.extend(headers_in_folder(subfolder.path))
        else:
            base.append(subfolder.path + '/catch_{}_all.hpp'.format(subfolder.name))
    return normalized_paths(sorted(base))

def includes_from_file(header):
    includes = []
    with open(header, 'r', encoding = 'utf-8') as file:
        for line in file:
            if not line.startswith('#include'):
                continue
            match = include_parser.match(line)
            if match:
                includes.append(match.group(1))
    return normalized_paths(includes)

def normalize_includes(includes):
    """Returns """
    return [include[len(catchPath)+5:] for include in includes]

def get_duplicates(xs):
    seen = set()
    duplicated = []
    for x in xs:
        if x in seen:
            duplicated.append(x)
        seen.add(x)
    return duplicated

def verify_convenience_header(folder):
    """
    Performs the actual checking of convenience header for specific folder.
    Checks that
    1) The header even exists
    2) That all includes in the header are sorted
    3) That there are no duplicated includes
    4) That all includes that should be in the header are actually present in the header
    5) That there are no superfluous includes that should not be in the header
    """
    global errors_found

    path = normalized_path(folder.path)

    assert path.startswith(source_path), '{} does not start with {}'.format(path, source_path)
    stripped_path = path[len(source_path) + 1:]
    path_pieces = stripped_path.split('/')

    if path == source_path:
        header_name = 'catch_all.hpp'
    else:
        header_name = 'catch_{}_all.hpp'.format('_'.join(path_pieces))

    # 1) Does it exist?
    full_path = path + '/' + header_name
    if not os.path.isfile(full_path):
        errors_found = True
        print('Missing convenience header: {}'.format(full_path))
        return
    file_incs = includes_from_file(path + '/' + header_name)
    # 2) Are the includes are sorted?
    if sorted(file_incs) != file_incs:
        errors_found = True
        print("'{}': Includes are not in sorted order!".format(header_name))

    # 3) Are there no duplicates?
    duplicated = get_duplicates(file_incs)
    for duplicate in duplicated:
        errors_found = True
        print("'{}': Duplicated include: '{}'".format(header_name, duplicate))

    target_includes = normalize_includes(collated_includes(path))
    # Avoid requiring the convenience header to include itself
    target_includes = [x for x in target_includes if header_name not in x]
    # 4) Are all required headers present?
    file_incs_set = set(file_incs)
    for include in target_includes:
        if (include not in file_incs_set and
            include != 'catch2/internal/catch_windows_h_proxy.hpp'):
            errors_found = True
            print("'{}': missing include '{}'".format(header_name, include))

    # 5) Are there any superfluous headers?
    desired_set = set(target_includes)
    for include in file_incs:
        if include not in desired_set:
            errors_found = True
            print("'{}': superfluous include '{}'".format(header_name, include))



def walk_source_folders(current):
    verify_convenience_header(current)
    for folder in folders_in_folder(current.path):
        fname = folder.name
        if fname not in internal_dirs:
            walk_source_folders(folder)

# This is an ugly hack because we cannot instantiate DirEntry manually
base_dir = [x for x in os.scandir(catchPath + '/src') if x.name == 'catch2']
walk_source_folders(base_dir[0])

# Propagate error "code" upwards
if not errors_found:
    print('Everything ok')
exit(errors_found)
