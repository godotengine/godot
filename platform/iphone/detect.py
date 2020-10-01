import os
import sys
from methods import detect_darwin_sdk_path


def is_active():
    return True


def get_name():
    return "iOS"


def can_build():

    if sys.platform == "darwin" or ("OSXCROSS_IOS" in os.environ):
        return True

    return False


def get_opts():
    from SCons.Variables import BoolVariable

    return [
        (
            "IPHONEPATH",
            "Path to iPhone toolchain",
            "/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain",
        ),
        ("IPHONESDK", "Path to the iPhone SDK", ""),
        BoolVariable(
            "use_static_mvk",
            "Link MoltenVK statically as Level-0 driver (better portability) or use Vulkan ICD loader (enables"
            " validation layers)",
            False,
        ),
        BoolVariable("game_center", "Support for game center", True),
        BoolVariable("store_kit", "Support for in-app store", True),
        BoolVariable("icloud", "Support for iCloud", True),
        BoolVariable("ios_exceptions", "Enable exceptions", False),
        ("ios_triple", "Triple for ios toolchain", ""),
    ]


def get_flags():

    return [
        ("tools", False),
    ]


def configure(env):

    ## Build type

    if env["target"].startswith("release"):
        env.Append(CPPDEFINES=["NDEBUG", ("NS_BLOCK_ASSERTIONS", 1)])
        if env["optimize"] == "speed":  # optimize for speed (default)
            env.Append(CCFLAGS=["-O2", "-ftree-vectorize", "-fomit-frame-pointer"])
            env.Append(LINKFLAGS=["-O2"])
        else:  # optimize for size
            env.Append(CCFLAGS=["-Os", "-ftree-vectorize"])
            env.Append(LINKFLAGS=["-Os"])

        if env["target"] == "release_debug":
            env.Append(CPPDEFINES=["DEBUG_ENABLED"])

    elif env["target"] == "debug":
        env.Append(CCFLAGS=["-gdwarf-2", "-O0"])
        env.Append(CPPDEFINES=["_DEBUG", ("DEBUG", 1), "DEBUG_ENABLED"])

    if env["use_lto"]:
        env.Append(CCFLAGS=["-flto"])
        env.Append(LINKFLAGS=["-flto"])

    ## Architecture
    if env["arch"] == "x86":  # i386
        env["bits"] = "32"
    elif env["arch"] == "x86_64":
        env["bits"] = "64"
    elif env["arch"] == "arm" or env["arch"] == "arm32" or env["arch"] == "armv7" or env["bits"] == "32":  # arm
        env["arch"] = "arm"
        env["bits"] = "32"
    else:  # armv64
        env["arch"] = "arm64"
        env["bits"] = "64"

    ## Compiler configuration

    # Save this in environment for use by other modules
    if "OSXCROSS_IOS" in os.environ:
        env["osxcross"] = True

    env["ENV"]["PATH"] = env["IPHONEPATH"] + "/Developer/usr/bin/:" + env["ENV"]["PATH"]

    compiler_path = "$IPHONEPATH/usr/bin/${ios_triple}"
    s_compiler_path = "$IPHONEPATH/Developer/usr/bin/"

    ccache_path = os.environ.get("CCACHE")
    if ccache_path is None:
        env["CC"] = compiler_path + "clang"
        env["CXX"] = compiler_path + "clang++"
        env["S_compiler"] = s_compiler_path + "gcc"
    else:
        # there aren't any ccache wrappers available for iOS,
        # to enable caching we need to prepend the path to the ccache binary
        env["CC"] = ccache_path + " " + compiler_path + "clang"
        env["CXX"] = ccache_path + " " + compiler_path + "clang++"
        env["S_compiler"] = ccache_path + " " + s_compiler_path + "gcc"
    env["AR"] = compiler_path + "ar"
    env["RANLIB"] = compiler_path + "ranlib"

    ## Compile flags

    if env["arch"] == "x86" or env["arch"] == "x86_64":
        detect_darwin_sdk_path("iphonesimulator", env)
        env["ENV"]["MACOSX_DEPLOYMENT_TARGET"] = "10.9"
        arch_flag = "i386" if env["arch"] == "x86" else env["arch"]
        env.Append(
            CCFLAGS=(
                "-arch "
                + arch_flag
                + " -fobjc-abi-version=2 -fobjc-legacy-dispatch -fmessage-length=0 -fpascal-strings -fblocks"
                " -fasm-blocks -isysroot $IPHONESDK -mios-simulator-version-min=13.0"
            ).split()
        )
    elif env["arch"] == "arm":
        detect_darwin_sdk_path("iphone", env)
        env.Append(
            CCFLAGS=(
                "-fobjc-arc -arch armv7 -fmessage-length=0 -fno-strict-aliasing"
                " -fdiagnostics-print-source-range-info -fdiagnostics-show-category=id -fdiagnostics-parseable-fixits"
                " -fpascal-strings -fblocks -isysroot $IPHONESDK -fvisibility=hidden -mthumb"
                ' "-DIBOutlet=__attribute__((iboutlet))"'
                ' "-DIBOutletCollection(ClassName)=__attribute__((iboutletcollection(ClassName)))"'
                ' "-DIBAction=void)__attribute__((ibaction)" -miphoneos-version-min=11.0 -MMD -MT dependencies'.split()
            )
        )
    elif env["arch"] == "arm64":
        detect_darwin_sdk_path("iphone", env)
        env.Append(
            CCFLAGS=(
                "-fobjc-arc -arch arm64 -fmessage-length=0 -fno-strict-aliasing"
                " -fdiagnostics-print-source-range-info -fdiagnostics-show-category=id -fdiagnostics-parseable-fixits"
                " -fpascal-strings -fblocks -fvisibility=hidden -MMD -MT dependencies -miphoneos-version-min=11.0"
                " -isysroot $IPHONESDK".split()
            )
        )
        env.Append(CPPDEFINES=["NEED_LONG_INT"])
        env.Append(CPPDEFINES=["LIBYUV_DISABLE_NEON"])

    # Disable exceptions on non-tools (template) builds
    if not env["tools"]:
        if env["ios_exceptions"]:
            env.Append(CCFLAGS=["-fexceptions"])
        else:
            env.Append(CCFLAGS=["-fno-exceptions"])

    # Temp fix for ABS/MAX/MIN macros in iPhone SDK blocking compilation
    env.Append(CCFLAGS=["-Wno-ambiguous-macro"])

    ## Link flags

    if env["arch"] == "x86" or env["arch"] == "x86_64":
        arch_flag = "i386" if env["arch"] == "x86" else env["arch"]
        env.Append(
            LINKFLAGS=[
                "-arch",
                arch_flag,
                "-mios-simulator-version-min=13.0",
                "-isysroot",
                "$IPHONESDK",
                "-Xlinker",
                "-objc_abi_version",
                "-Xlinker",
                "2",
                "-F$IPHONESDK",
            ]
        )
    elif env["arch"] == "arm":
        env.Append(LINKFLAGS=["-arch", "armv7", "-Wl,-dead_strip", "-miphoneos-version-min=11.0"])
    if env["arch"] == "arm64":
        env.Append(LINKFLAGS=["-arch", "arm64", "-Wl,-dead_strip", "-miphoneos-version-min=11.0"])

    env.Append(
        LINKFLAGS=[
            "-isysroot",
            "$IPHONESDK",
            "-framework",
            "AudioToolbox",
            "-framework",
            "AVFoundation",
            "-framework",
            "CoreAudio",
            "-framework",
            "CoreGraphics",
            "-framework",
            "CoreMedia",
            "-framework",
            "CoreVideo",
            "-framework",
            "CoreMotion",
            "-framework",
            "Foundation",
            "-framework",
            "GameController",
            "-framework",
            "MediaPlayer",
            "-framework",
            "Metal",
            "-framework",
            "QuartzCore",
            "-framework",
            "Security",
            "-framework",
            "SystemConfiguration",
            "-framework",
            "UIKit",
            "-framework",
            "ARKit",
        ]
    )

    # Feature options
    if env["game_center"]:
        env.Append(CPPDEFINES=["GAME_CENTER_ENABLED"])
        env.Append(LINKFLAGS=["-framework", "GameKit"])

    if env["store_kit"]:
        env.Append(CPPDEFINES=["STOREKIT_ENABLED"])
        env.Append(LINKFLAGS=["-framework", "StoreKit"])

    if env["icloud"]:
        env.Append(CPPDEFINES=["ICLOUD_ENABLED"])

    env.Prepend(
        CPPPATH=[
            "$IPHONESDK/usr/include",
            "$IPHONESDK/System/Library/Frameworks/AudioUnit.framework/Headers",
        ]
    )

    env["ENV"]["CODESIGN_ALLOCATE"] = "/Developer/Platforms/iPhoneOS.platform/Developer/usr/bin/codesign_allocate"

    env.Prepend(CPPPATH=["#platform/iphone"])
    env.Append(CPPDEFINES=["IPHONE_ENABLED", "UNIX_ENABLED", "COREAUDIO_ENABLED"])

    env.Append(CPPDEFINES=["VULKAN_ENABLED"])
    env.Append(LINKFLAGS=["-framework", "IOSurface"])

    # Use Static Vulkan for iOS. Dynamic Framework works fine too.
    env.Append(LINKFLAGS=["-framework", "MoltenVK"])
    env["builtin_vulkan"] = False
