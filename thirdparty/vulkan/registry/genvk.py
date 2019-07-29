#!/usr/bin/python3
#
# Copyright (c) 2013-2019 The Khronos Group Inc.
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

import argparse
import pdb
import re
import sys
import time
import xml.etree.ElementTree as etree

from cgenerator import CGeneratorOptions, COutputGenerator
from docgenerator import DocGeneratorOptions, DocOutputGenerator
from extensionmetadocgenerator import (ExtensionMetaDocGeneratorOptions,
                                       ExtensionMetaDocOutputGenerator)
from generator import write
from hostsyncgenerator import HostSynchronizationOutputGenerator
from pygenerator import PyOutputGenerator
from reg import Registry
from validitygenerator import ValidityOutputGenerator
from vkconventions import VulkanConventions

# Simple timer functions
startTime = None


def startTimer(timeit):
    global startTime
    if timeit:
        startTime = time.process_time()

def endTimer(timeit, msg):
    global startTime
    if timeit:
        endTime = time.process_time()
        write(msg, endTime - startTime, file=sys.stderr)
        startTime = None

# Turn a list of strings into a regexp string matching exactly those strings
def makeREstring(list, default = None):
    if len(list) > 0 or default is None:
        return '^(' + '|'.join(list) + ')$'
    else:
        return default

# Returns a directory of [ generator function, generator options ] indexed
# by specified short names. The generator options incorporate the following
# parameters:
#
# args is an parsed argument object; see below for the fields that are used.
def makeGenOpts(args):
    global genOpts
    genOpts = {}

    # Default class of extensions to include, or None
    defaultExtensions = args.defaultExtensions

    # Additional extensions to include (list of extensions)
    extensions = args.extension

    # Extensions to remove (list of extensions)
    removeExtensions = args.removeExtensions

    # Extensions to emit (list of extensions)
    emitExtensions = args.emitExtensions

    # Features to include (list of features)
    features = args.feature

    # Whether to disable inclusion protect in headers
    protect = args.protect

    # Output target directory
    directory = args.directory

    # Descriptive names for various regexp patterns used to select
    # versions and extensions
    allFeatures = allExtensions = r'.*'

    # Turn lists of names/patterns into matching regular expressions
    addExtensionsPat     = makeREstring(extensions, None)
    removeExtensionsPat  = makeREstring(removeExtensions, None)
    emitExtensionsPat    = makeREstring(emitExtensions, allExtensions)
    featuresPat          = makeREstring(features, allFeatures)

    # Copyright text prefixing all headers (list of strings).
    prefixStrings = [
        '/*',
        '** Copyright (c) 2015-2019 The Khronos Group Inc.',
        '**',
        '** Licensed under the Apache License, Version 2.0 (the "License");',
        '** you may not use this file except in compliance with the License.',
        '** You may obtain a copy of the License at',
        '**',
        '**     http://www.apache.org/licenses/LICENSE-2.0',
        '**',
        '** Unless required by applicable law or agreed to in writing, software',
        '** distributed under the License is distributed on an "AS IS" BASIS,',
        '** WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.',
        '** See the License for the specific language governing permissions and',
        '** limitations under the License.',
        '*/',
        ''
    ]

    # Text specific to Vulkan headers
    vkPrefixStrings = [
        '/*',
        '** This header is generated from the Khronos Vulkan XML API Registry.',
        '**',
        '*/',
        ''
    ]

    # Defaults for generating re-inclusion protection wrappers (or not)
    protectFile = protect

    # An API style conventions object
    conventions = VulkanConventions()

    # API include files for spec and ref pages
    # Overwrites include subdirectories in spec source tree
    # The generated include files do not include the calling convention
    # macros (apientry etc.), unlike the header files.
    # Because the 1.0 core branch includes ref pages for extensions,
    # all the extension interfaces need to be generated, even though
    # none are used by the core spec itself.
    genOpts['apiinc'] = [
          DocOutputGenerator,
          DocGeneratorOptions(
            conventions       = conventions,
            filename          = 'timeMarker',
            directory         = directory,
            apiname           = 'vulkan',
            profile           = None,
            versions          = featuresPat,
            emitversions      = featuresPat,
            defaultExtensions = None,
            addExtensions     = addExtensionsPat,
            removeExtensions  = removeExtensionsPat,
            emitExtensions    = emitExtensionsPat,
            prefixText        = prefixStrings + vkPrefixStrings,
            apicall           = '',
            apientry          = '',
            apientryp         = '*',
            alignFuncParam    = 48,
            expandEnumerants  = False)
        ]

    # API names to validate man/api spec includes & links
    genOpts['vkapi.py'] = [
          PyOutputGenerator,
          DocGeneratorOptions(
            conventions       = conventions,
            filename          = 'vkapi.py',
            directory         = directory,
            apiname           = 'vulkan',
            profile           = None,
            versions          = featuresPat,
            emitversions      = featuresPat,
            defaultExtensions = None,
            addExtensions     = addExtensionsPat,
            removeExtensions  = removeExtensionsPat,
            emitExtensions    = emitExtensionsPat)
        ]

    # API validity files for spec
    genOpts['validinc'] = [
          ValidityOutputGenerator,
          DocGeneratorOptions(
            conventions       = conventions,
            filename          = 'timeMarker',
            directory         = directory,
            apiname           = 'vulkan',
            profile           = None,
            versions          = featuresPat,
            emitversions      = featuresPat,
            defaultExtensions = None,
            addExtensions     = addExtensionsPat,
            removeExtensions  = removeExtensionsPat,
            emitExtensions    = emitExtensionsPat)
        ]

    # API host sync table files for spec
    genOpts['hostsyncinc'] = [
          HostSynchronizationOutputGenerator,
          DocGeneratorOptions(
            conventions       = conventions,
            filename          = 'timeMarker',
            directory         = directory,
            apiname           = 'vulkan',
            profile           = None,
            versions          = featuresPat,
            emitversions      = featuresPat,
            defaultExtensions = None,
            addExtensions     = addExtensionsPat,
            removeExtensions  = removeExtensionsPat,
            emitExtensions    = emitExtensionsPat)
        ]

    # Extension metainformation for spec extension appendices
    genOpts['extinc'] = [
          ExtensionMetaDocOutputGenerator,
          ExtensionMetaDocGeneratorOptions(
            conventions       = conventions,
            filename          = 'timeMarker',
            directory         = directory,
            apiname           = 'vulkan',
            profile           = None,
            versions          = featuresPat,
            emitversions      = None,
            defaultExtensions = defaultExtensions,
            addExtensions     = None,
            removeExtensions  = None,
            emitExtensions    = emitExtensionsPat)
        ]

    # Platform extensions, in their own header files
    # Each element of the platforms[] array defines information for
    # generating a single platform:
    #   [0] is the generated header file name
    #   [1] is the set of platform extensions to generate
    #   [2] is additional extensions whose interfaces should be considered,
    #   but suppressed in the output, to avoid duplicate definitions of
    #   dependent types like VkDisplayKHR and VkSurfaceKHR which come from
    #   non-platform extensions.

    # Track all platform extensions, for exclusion from vulkan_core.h
    allPlatformExtensions = []

    # Extensions suppressed for all platforms.
    # Covers common WSI extension types.
    commonSuppressExtensions = [ 'VK_KHR_display', 'VK_KHR_swapchain' ]

    platforms = [
        [ 'vulkan_android.h',     [ 'VK_KHR_android_surface',
                                    'VK_ANDROID_external_memory_android_hardware_buffer'
                                                                  ], commonSuppressExtensions ],
        [ 'vulkan_fuchsia.h',     [ 'VK_FUCHSIA_imagepipe_surface'], commonSuppressExtensions ],
        [ 'vulkan_ggp.h',         [ 'VK_GGP_stream_descriptor_surface',
                                    'VK_GGP_frame_token'          ], commonSuppressExtensions ],
        [ 'vulkan_ios.h',         [ 'VK_MVK_ios_surface'          ], commonSuppressExtensions ],
        [ 'vulkan_macos.h',       [ 'VK_MVK_macos_surface'        ], commonSuppressExtensions ],
        [ 'vulkan_vi.h',          [ 'VK_NN_vi_surface'            ], commonSuppressExtensions ],
        [ 'vulkan_wayland.h',     [ 'VK_KHR_wayland_surface'      ], commonSuppressExtensions ],
        [ 'vulkan_win32.h',       [ 'VK_.*_win32(|_.*)', 'VK_EXT_full_screen_exclusive' ],
                                                                     commonSuppressExtensions +
                                                                     [ 'VK_KHR_external_semaphore',
                                                                       'VK_KHR_external_memory_capabilities',
                                                                       'VK_KHR_external_fence',
                                                                       'VK_KHR_external_fence_capabilities',
                                                                       'VK_KHR_get_surface_capabilities2',
                                                                       'VK_NV_external_memory_capabilities',
                                                                     ] ],
        [ 'vulkan_xcb.h',         [ 'VK_KHR_xcb_surface'          ], commonSuppressExtensions ],
        [ 'vulkan_xlib.h',        [ 'VK_KHR_xlib_surface'         ], commonSuppressExtensions ],
        [ 'vulkan_xlib_xrandr.h', [ 'VK_EXT_acquire_xlib_display' ], commonSuppressExtensions ],
        [ 'vulkan_metal.h',       [ 'VK_EXT_metal_surface'        ], commonSuppressExtensions ],
    ]

    for platform in platforms:
        headername = platform[0]

        allPlatformExtensions += platform[1]

        addPlatformExtensionsRE = makeREstring(platform[1] + platform[2])
        emitPlatformExtensionsRE = makeREstring(platform[1])

        opts = CGeneratorOptions(
            conventions       = conventions,
            filename          = headername,
            directory         = directory,
            apiname           = 'vulkan',
            profile           = None,
            versions          = featuresPat,
            emitversions      = None,
            defaultExtensions = None,
            addExtensions     = addPlatformExtensionsRE,
            removeExtensions  = None,
            emitExtensions    = emitPlatformExtensionsRE,
            prefixText        = prefixStrings + vkPrefixStrings,
            genFuncPointers   = True,
            protectFile       = protectFile,
            protectFeature    = False,
            protectProto      = '#ifndef',
            protectProtoStr   = 'VK_NO_PROTOTYPES',
            apicall           = 'VKAPI_ATTR ',
            apientry          = 'VKAPI_CALL ',
            apientryp         = 'VKAPI_PTR *',
            alignFuncParam    = 48,
            genEnumBeginEndRange = True)

        genOpts[headername] = [ COutputGenerator, opts ]

    # Header for core API + extensions.
    # To generate just the core API,
    # change to 'defaultExtensions = None' below.
    #
    # By default this adds all enabled, non-platform extensions.
    # It removes all platform extensions (from the platform headers options
    # constructed above) as well as any explicitly specified removals.

    removeExtensionsPat = makeREstring(allPlatformExtensions + removeExtensions, None)

    genOpts['vulkan_core.h'] = [
          COutputGenerator,
          CGeneratorOptions(
            conventions       = conventions,
            filename          = 'vulkan_core.h',
            directory         = directory,
            apiname           = 'vulkan',
            profile           = None,
            versions          = featuresPat,
            emitversions      = featuresPat,
            defaultExtensions = defaultExtensions,
            addExtensions     = None,
            removeExtensions  = removeExtensionsPat,
            emitExtensions    = emitExtensionsPat,
            prefixText        = prefixStrings + vkPrefixStrings,
            genFuncPointers   = True,
            protectFile       = protectFile,
            protectFeature    = False,
            protectProto      = '#ifndef',
            protectProtoStr   = 'VK_NO_PROTOTYPES',
            apicall           = 'VKAPI_ATTR ',
            apientry          = 'VKAPI_CALL ',
            apientryp         = 'VKAPI_PTR *',
            alignFuncParam    = 48,
            genEnumBeginEndRange = True)
        ]

    # Unused - vulkan10.h target.
    # It is possible to generate a header with just the Vulkan 1.0 +
    # extension interfaces defined, but since the promoted KHR extensions
    # are now defined in terms of the 1.1 interfaces, such a header is very
    # similar to vulkan_core.h.
    genOpts['vulkan10.h'] = [
          COutputGenerator,
          CGeneratorOptions(
            conventions       = conventions,
            filename          = 'vulkan10.h',
            directory         = directory,
            apiname           = 'vulkan',
            profile           = None,
            versions          = 'VK_VERSION_1_0',
            emitversions      = 'VK_VERSION_1_0',
            defaultExtensions = defaultExtensions,
            addExtensions     = None,
            removeExtensions  = removeExtensionsPat,
            emitExtensions    = emitExtensionsPat,
            prefixText        = prefixStrings + vkPrefixStrings,
            genFuncPointers   = True,
            protectFile       = protectFile,
            protectFeature    = False,
            protectProto      = '#ifndef',
            protectProtoStr   = 'VK_NO_PROTOTYPES',
            apicall           = 'VKAPI_ATTR ',
            apientry          = 'VKAPI_CALL ',
            apientryp         = 'VKAPI_PTR *',
            alignFuncParam    = 48)
        ]

    genOpts['alias.h'] = [
          COutputGenerator,
          CGeneratorOptions(
            conventions       = conventions,
            filename          = 'alias.h',
            directory         = directory,
            apiname           = 'vulkan',
            profile           = None,
            versions          = featuresPat,
            emitversions      = featuresPat,
            defaultExtensions = defaultExtensions,
            addExtensions     = None,
            removeExtensions  = removeExtensionsPat,
            emitExtensions    = emitExtensionsPat,
            prefixText        = None,
            genFuncPointers   = False,
            protectFile       = False,
            protectFeature    = False,
            protectProto      = '',
            protectProtoStr   = '',
            apicall           = '',
            apientry          = '',
            apientryp         = '',
            alignFuncParam    = 36)
        ]

# Generate a target based on the options in the matching genOpts{} object.
# This is encapsulated in a function so it can be profiled and/or timed.
# The args parameter is an parsed argument object containing the following
# fields that are used:
#   target - target to generate
#   directory - directory to generate it in
#   protect - True if re-inclusion wrappers should be created
#   extensions - list of additional extensions to include in generated
#   interfaces
def genTarget(args):
    # Create generator options with specified parameters
    makeGenOpts(args)

    if args.target in genOpts:
        createGenerator = genOpts[args.target][0]
        options = genOpts[args.target][1]

        if not args.quiet:
            write('* Building', options.filename, file=sys.stderr)
            write('* options.versions          =', options.versions, file=sys.stderr)
            write('* options.emitversions      =', options.emitversions, file=sys.stderr)
            write('* options.defaultExtensions =', options.defaultExtensions, file=sys.stderr)
            write('* options.addExtensions     =', options.addExtensions, file=sys.stderr)
            write('* options.removeExtensions  =', options.removeExtensions, file=sys.stderr)
            write('* options.emitExtensions    =', options.emitExtensions, file=sys.stderr)

        startTimer(args.time)
        gen = createGenerator(errFile=errWarn,
                              warnFile=errWarn,
                              diagFile=diag)
        reg.setGenerator(gen)
        reg.apiGen(options)

        if not args.quiet:
            write('* Generated', options.filename, file=sys.stderr)
        endTimer(args.time, '* Time to generate ' + options.filename + ' =')
    else:
        write('No generator options for unknown target:',
              args.target, file=sys.stderr)


# -feature name
# -extension name
# For both, "name" may be a single name, or a space-separated list
# of names, or a regular expression.
if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('-defaultExtensions', action='store',
                        default='vulkan',
                        help='Specify a single class of extensions to add to targets')
    parser.add_argument('-extension', action='append',
                        default=[],
                        help='Specify an extension or extensions to add to targets')
    parser.add_argument('-removeExtensions', action='append',
                        default=[],
                        help='Specify an extension or extensions to remove from targets')
    parser.add_argument('-emitExtensions', action='append',
                        default=[],
                        help='Specify an extension or extensions to emit in targets')
    parser.add_argument('-feature', action='append',
                        default=[],
                        help='Specify a core API feature name or names to add to targets')
    parser.add_argument('-debug', action='store_true',
                        help='Enable debugging')
    parser.add_argument('-dump', action='store_true',
                        help='Enable dump to stderr')
    parser.add_argument('-diagfile', action='store',
                        default=None,
                        help='Write diagnostics to specified file')
    parser.add_argument('-errfile', action='store',
                        default=None,
                        help='Write errors and warnings to specified file instead of stderr')
    parser.add_argument('-noprotect', dest='protect', action='store_false',
                        help='Disable inclusion protection in output headers')
    parser.add_argument('-profile', action='store_true',
                        help='Enable profiling')
    parser.add_argument('-registry', action='store',
                        default='vk.xml',
                        help='Use specified registry file instead of vk.xml')
    parser.add_argument('-time', action='store_true',
                        help='Enable timing')
    parser.add_argument('-validate', action='store_true',
                        help='Enable group validation')
    parser.add_argument('-o', action='store', dest='directory',
                        default='.',
                        help='Create target and related files in specified directory')
    parser.add_argument('target', metavar='target', nargs='?',
                        help='Specify target')
    parser.add_argument('-quiet', action='store_true', default=True,
                        help='Suppress script output during normal execution.')
    parser.add_argument('-verbose', action='store_false', dest='quiet', default=True,
                        help='Enable script output during normal execution.')

    args = parser.parse_args()

    # This splits arguments which are space-separated lists
    args.feature = [name for arg in args.feature for name in arg.split()]
    args.extension = [name for arg in args.extension for name in arg.split()]

    # Load & parse registry
    reg = Registry()

    startTimer(args.time)
    tree = etree.parse(args.registry)
    endTimer(args.time, '* Time to make ElementTree =')

    if args.debug:
        pdb.run('reg.loadElementTree(tree)')
    else:
        startTimer(args.time)
        reg.loadElementTree(tree)
        endTimer(args.time, '* Time to parse ElementTree =')

    if args.validate:
        reg.validateGroups()

    if args.dump:
        write('* Dumping registry to regdump.txt', file=sys.stderr)
        reg.dumpReg(filehandle = open('regdump.txt', 'w', encoding='utf-8'))

    # create error/warning & diagnostic files
    if args.errfile:
        errWarn = open(args.errfile, 'w', encoding='utf-8')
    else:
        errWarn = sys.stderr

    if args.diagfile:
        diag = open(args.diagfile, 'w', encoding='utf-8')
    else:
        diag = None

    if args.debug:
        pdb.run('genTarget(args)')
    elif args.profile:
        import cProfile, pstats
        cProfile.run('genTarget(args)', 'profile.txt')
        p = pstats.Stats('profile.txt')
        p.strip_dirs().sort_stats('time').print_stats(50)
    else:
        genTarget(args)
