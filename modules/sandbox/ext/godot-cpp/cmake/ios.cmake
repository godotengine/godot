#[=======================================================================[.rst:
iOS
---

This file contains functions for options and configuration for targeting the
iOS platform

]=======================================================================]

#[==============================[ iOS Options ]==============================]
function(ios_options)
    #[[ Options from SCons

    TODO ios_simulator: Target iOS Simulator
        Default: False

    TODO ios_min_version: Target minimum iphoneos/iphonesimulator version
        Default: 12.0

    TODO IOS_TOOLCHAIN_PATH: Path to iOS toolchain
        Default: "/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain",

    TODO IOS_SDK_PATH: Path to the iOS SDK
        Default: ''

    TODO ios_triple: Triple for ios toolchain
        Default: if has_ios_osxcross(): 'ios_triple' else ''
    ]]
endfunction()

#[===========================[ Target Generation ]===========================]
function(ios_generate)
    target_compile_definitions(godot-cpp PUBLIC IOS_ENABLED UNIX_ENABLED)

    common_compiler_flags()
endfunction()
