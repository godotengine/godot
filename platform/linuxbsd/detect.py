import os
import platform
import sys
from methods import get_compiler_version, using_gcc
from platform_methods import detect_arch

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from SCons import Environment


def get_name():
    return "LinuxBSD"


def can_build():
    if os.name != "posix" or sys.platform == "darwin":
        return False

    pkgconf_error = os.system("pkg-config --version > /dev/null")
    if pkgconf_error:
        print("Error: pkg-config not found. Aborting.")
        return False

    return True


def get_opts():
    from SCons.Variables import BoolVariable, EnumVariable

    return [
        EnumVariable("linker", "Linker program", "default", ("default", "bfd", "gold", "lld", "mold")),
        BoolVariable("use_llvm", "Use the LLVM compiler", False),
        BoolVariable("use_static_cpp", "Link libgcc and libstdc++ statically for better portability", True),
        BoolVariable("use_coverage", "Test Godot coverage", False),
        BoolVariable("use_ubsan", "Use LLVM/GCC compiler undefined behavior sanitizer (UBSAN)", False),
        BoolVariable("use_asan", "Use LLVM/GCC compiler address sanitizer (ASAN)", False),
        BoolVariable("use_lsan", "Use LLVM/GCC compiler leak sanitizer (LSAN)", False),
        BoolVariable("use_tsan", "Use LLVM/GCC compiler thread sanitizer (TSAN)", False),
        BoolVariable("use_msan", "Use LLVM compiler memory sanitizer (MSAN)", False),
        BoolVariable("use_sowrap", "Dynamically load system libraries", True),
        BoolVariable("alsa", "Use ALSA", True),
        BoolVariable("pulseaudio", "Use PulseAudio", True),
        BoolVariable("dbus", "Use D-Bus to handle screensaver and portal desktop settings", True),
        BoolVariable("speechd", "Use Speech Dispatcher for Text-to-Speech support", True),
        BoolVariable("fontconfig", "Use fontconfig for system fonts support", True),
        BoolVariable("udev", "Use udev for gamepad connection callbacks", True),
        BoolVariable("x11", "Enable X11 display", True),
        BoolVariable("touch", "Enable touch events", True),
        BoolVariable("execinfo", "Use libexecinfo on systems where glibc is not available", False),
    ]


def get_doc_classes():
    return [
        "EditorExportPlatformLinuxBSD",
    ]


def get_doc_path():
    return "doc_classes"


def get_flags():
    return [
        ("arch", detect_arch()),
    ]


def configure(env: "Environment"):
    # Validate arch.
    supported_arches = ["x86_32", "x86_64", "arm32", "arm64", "rv64", "ppc32", "ppc64"]
    if env["arch"] not in supported_arches:
        print(
            'Unsupported CPU architecture "%s" for Linux / *BSD. Supported architectures are: %s.'
            % (env["arch"], ", ".join(supported_arches))
        )
        sys.exit(255)

    ## Build type

    if env.dev_build:
        # This is needed for our crash handler to work properly.
        # gdb works fine without it though, so maybe our crash handler could too.
        env.Append(LINKFLAGS=["-rdynamic"])

    # CPU architecture flags.
    if env["arch"] == "rv64":
        # G = General-purpose extensions, C = Compression extension (very common).
        env.Append(CCFLAGS=["-march=rv64gc"])

    ## Compiler configuration

    if "CXX" in env and "clang" in os.path.basename(env["CXX"]):
        # Convenience check to enforce the use_llvm overrides when CXX is clang(++)
        env["use_llvm"] = True

    if env["use_llvm"]:
        if "clang++" not in os.path.basename(env["CXX"]):
            env["CC"] = "clang"
            env["CXX"] = "clang++"
        env.extra_suffix = ".llvm" + env.extra_suffix

    if env["linker"] != "default":
        print("Using linker program: " + env["linker"])
        if env["linker"] == "mold" and using_gcc(env):  # GCC < 12.1 doesn't support -fuse-ld=mold.
            cc_version = get_compiler_version(env)
            cc_semver = (cc_version["major"], cc_version["minor"])
            if cc_semver < (12, 1):
                found_wrapper = False
                for path in ["/usr/libexec", "/usr/local/libexec", "/usr/lib", "/usr/local/lib"]:
                    if os.path.isfile(path + "/mold/ld"):
                        env.Append(LINKFLAGS=["-B" + path + "/mold"])
                        found_wrapper = True
                        break
                if not found_wrapper:
                    print("Couldn't locate mold installation path. Make sure it's installed in /usr or /usr/local.")
                    sys.exit(255)
            else:
                env.Append(LINKFLAGS=["-fuse-ld=mold"])
        else:
            env.Append(LINKFLAGS=["-fuse-ld=%s" % env["linker"]])

    if env["use_coverage"]:
        env.Append(CCFLAGS=["-ftest-coverage", "-fprofile-arcs"])
        env.Append(LINKFLAGS=["-ftest-coverage", "-fprofile-arcs"])

    if env["use_ubsan"] or env["use_asan"] or env["use_lsan"] or env["use_tsan"] or env["use_msan"]:
        env.extra_suffix += ".san"
        env.Append(CCFLAGS=["-DSANITIZERS_ENABLED"])

        if env["use_ubsan"]:
            env.Append(
                CCFLAGS=[
                    "-fsanitize=undefined,shift,shift-exponent,integer-divide-by-zero,unreachable,vla-bound,null,return,signed-integer-overflow,bounds,float-divide-by-zero,float-cast-overflow,nonnull-attribute,returns-nonnull-attribute,bool,enum,vptr,pointer-overflow,builtin"
                ]
            )
            env.Append(LINKFLAGS=["-fsanitize=undefined"])
            if env["use_llvm"]:
                env.Append(
                    CCFLAGS=[
                        "-fsanitize=nullability-return,nullability-arg,function,nullability-assign,implicit-integer-sign-change"
                    ]
                )
            else:
                env.Append(CCFLAGS=["-fsanitize=bounds-strict"])

        if env["use_asan"]:
            env.Append(CCFLAGS=["-fsanitize=address,pointer-subtract,pointer-compare"])
            env.Append(LINKFLAGS=["-fsanitize=address"])

        if env["use_lsan"]:
            env.Append(CCFLAGS=["-fsanitize=leak"])
            env.Append(LINKFLAGS=["-fsanitize=leak"])

        if env["use_tsan"]:
            env.Append(CCFLAGS=["-fsanitize=thread"])
            env.Append(LINKFLAGS=["-fsanitize=thread"])

        if env["use_msan"] and env["use_llvm"]:
            env.Append(CCFLAGS=["-fsanitize=memory"])
            env.Append(CCFLAGS=["-fsanitize-memory-track-origins"])
            env.Append(CCFLAGS=["-fsanitize-recover=memory"])
            env.Append(LINKFLAGS=["-fsanitize=memory"])

    # LTO

    if env["lto"] == "auto":  # Full LTO for production.
        env["lto"] = "full"

    if env["lto"] != "none":
        if env["lto"] == "thin":
            if not env["use_llvm"]:
                print("ThinLTO is only compatible with LLVM, use `use_llvm=yes` or `lto=full`.")
                sys.exit(255)
            env.Append(CCFLAGS=["-flto=thin"])
            env.Append(LINKFLAGS=["-flto=thin"])
        elif not env["use_llvm"] and env.GetOption("num_jobs") > 1:
            env.Append(CCFLAGS=["-flto"])
            env.Append(LINKFLAGS=["-flto=" + str(env.GetOption("num_jobs"))])
        else:
            env.Append(CCFLAGS=["-flto"])
            env.Append(LINKFLAGS=["-flto"])

        if not env["use_llvm"]:
            env["RANLIB"] = "gcc-ranlib"
            env["AR"] = "gcc-ar"

    env.Append(CCFLAGS=["-pipe"])

    ## Dependencies

    if env["use_sowrap"]:
        env.Append(CPPDEFINES=["SOWRAP_ENABLED"])

    if env["touch"]:
        env.Append(CPPDEFINES=["TOUCH_ENABLED"])

    # Check for existence of the libs, and if so,
    # parse their flags with pkg-config
    def add_external_libs(list):
        # Keep track of the number of libraries in the list
        # that couldn't be found
        not_found = 0
        for lib in list:
            if os.system("pkg-config --exists " + lib) == 0:  # 0 means found
                env.ParseConfig("pkg-config --cflags --libs " + lib)
            else:
                not_found += 1
        # A return value of 0 means all libs were found
        return not_found

    # freetype depends on libpng and zlib, so bundling one of them
    # while keeping others as shared libraries leads to weird issues.
    # And graphite and harfbuzz need freetype.
    ft_linked_deps = [
        env["builtin_freetype"],
        env["builtin_libpng"],
        env["builtin_zlib"],
        env["builtin_graphite"],
        env["builtin_harfbuzz"],
    ]
    if (not all(ft_linked_deps)) and any(ft_linked_deps):  # All or nothing.
        print(
            "These libraries should be either all builtin, or all system provided:\n"
            "freetype, libpng, zlib, graphite, harfbuzz.\n"
            "Please specify `builtin_<name>=no` for all of them, or none."
        )
        sys.exit(255)

    if not env["builtin_freetype"]:
        add_external_libs(["freetype2"])

    if not env["builtin_msdfgen"]:
        add_external_libs(["msdfgen"])

    if not env["builtin_graphite"]:
        add_external_libs(["graphite2"])

    if not env["builtin_icu4c"]:
        add_external_libs(["icu-i18n", "icu-uc"])

    if not env["builtin_harfbuzz"]:
        add_external_libs(["harfbuzz", "harfbuzz-icu"])

    if not env["builtin_libpng"]:
        add_external_libs(["libpng16"])

    if not env["builtin_enet"]:
        add_external_libs(["libenet"])

    if not env["builtin_squish"]:
        if add_external_libs(["libsquish"]) != 0:
            # libsquish doesn't reliably install its .pc file,
            # so some distros lack it.
            env.Append(LIBS=["libsquish"])

    if not env["builtin_zstd"]:
        add_external_libs(["libzstd"])

    if env["brotli"] and not env["builtin_brotli"]:
        add_external_libs(["libbrotlicommon", "libbrotlidec"])

    # Sound and video libraries
    # Keep the order as it triggers chained dependencies
    # (ogg needed by others, etc.)

    if not env["builtin_libtheora"]:
        env["builtin_libogg"] = False  # Needed to link against system libtheora
        env["builtin_libvorbis"] = False  # Needed to link against system libtheora
        add_external_libs(["theora", "theoradec"])
    else:
        if env["arch"] in ["x86_64", "x86_32"]:
            env["x86_libtheora_opt_gcc"] = True

    if not env["builtin_libvorbis"]:
        env["builtin_libogg"] = False  # Needed to link against system libvorbis
        add_external_libs(["vorbis", "vorbisfile"])

    if not env["builtin_libogg"]:
        add_external_libs(["ogg"])

    if not env["builtin_libwebp"]:
        add_external_libs(["libwebp"])

    if not env["builtin_mbedtls"]:
        # mbedTLS does not provide a pkgconfig config yet.
        # See https://github.com/ARMmbed/mbedtls/issues/228
        env.Append(LIBS=["mbedtls", "mbedcrypto", "mbedx509"])

    if not env["builtin_wslay"]:
        add_external_libs(["libwslay"])

    if not env["builtin_miniupnpc"]:
        # No pkgconfig file so far, hardcode default paths.
        env.Prepend(CPPPATH=["/usr/include/miniupnpc"])
        env.Append(LIBS=["miniupnpc"])

    # On Linux wchar_t should be 32-bits
    # 16-bit library shouldn't be required due to compiler optimizations
    if not env["builtin_pcre2"]:
        add_external_libs(["libpcre2-32"])

    if not env["builtin_recastnavigation"]:
        # No pkgconfig file so far, hardcode default paths.
        env.Prepend(CPPPATH=["/usr/include/recastnavigation"])
        env.Append(LIBS=["Recast"])

    if not env["builtin_embree"] and env["arch"] in ["x86_64", "arm64"]:
        # No pkgconfig file so far, hardcode expected lib name.
        env.Append(LIBS=["embree3"])

    if not env["builtin_openxr"]:
        add_external_libs(["openxr"])

    if env["fontconfig"]:
        if not env["use_sowrap"]:
            if add_external_libs(["fontconfig"]) == 0:  # 0 means found
                env.Append(CPPDEFINES=["FONTCONFIG_ENABLED"])
            else:
                print("Warning: fontconfig development libraries not found. Disabling the system fonts support.")
                env["fontconfig"] = False
        else:
            env.Append(CPPDEFINES=["FONTCONFIG_ENABLED"])

    if env["alsa"]:
        if not env["use_sowrap"]:
            if add_external_libs(["alsa"]) == 0:  # 0 means found
                env.Append(CPPDEFINES=["ALSA_ENABLED", "ALSAMIDI_ENABLED"])
            else:
                print("Warning: ALSA development libraries not found. Disabling the ALSA audio driver.")
                env["alsa"] = False
        else:
            env.Append(CPPDEFINES=["ALSA_ENABLED", "ALSAMIDI_ENABLED"])

    if env["pulseaudio"]:
        if not env["use_sowrap"]:
            if add_external_libs(["libpulse"]) == 0:  # 0 means found
                env.Append(CPPDEFINES=["PULSEAUDIO_ENABLED"])
            else:
                print("Warning: PulseAudio development libraries not found. Disabling the PulseAudio audio driver.")
                env["pulseaudio"] = False
        else:
            env.Append(CPPDEFINES=["PULSEAUDIO_ENABLED", "_REENTRANT"])

    if env["dbus"]:
        if not env["use_sowrap"]:
            if add_external_libs(["dbus-1"]) == 0:  # 0 means found
                env.Append(CPPDEFINES=["DBUS_ENABLED"])
            else:
                print("Warning: D-Bus development libraries not found. Disabling screensaver prevention.")
                env["dbus"] = False
        else:
            env.Append(CPPDEFINES=["DBUS_ENABLED"])

    if env["speechd"]:
        if not env["use_sowrap"]:
            if add_external_libs(["speech-dispatcher"]) == 0:  # 0 means found
                env.Append(CPPDEFINES=["SPEECHD_ENABLED"])
            else:
                print("Warning: speech-dispatcher development libraries not found. Disabling text to speech support.")
                env["speechd"] = False
        else:
            env.Append(CPPDEFINES=["SPEECHD_ENABLED"])

    if not env["use_sowrap"]:
        if add_external_libs(["xkbcommon"]) == 0:  # 0 means found
            env.Append(CPPDEFINES=["XKB_ENABLED"])
        else:
            print(
                "Warning: libxkbcommon development libraries not found.",
                "Disabling dead key composition and key label support.",
            )
    else:
        env.Append(CPPDEFINES=["XKB_ENABLED"])

    if platform.system() == "Linux":
        env.Append(CPPDEFINES=["JOYDEV_ENABLED"])
        if env["udev"]:
            if not env["use_sowrap"]:
                if add_external_libs(["libudev"]) == 0:  # 0 means found
                    env.Append(CPPDEFINES=["UDEV_ENABLED"])
                else:
                    print("Warning: libudev development libraries not found. Disabling controller hotplugging support.")
                    env["udev"] = False
            else:
                env.Append(CPPDEFINES=["UDEV_ENABLED"])
    else:
        env["udev"] = False  # Linux specific

    # Linkflags below this line should typically stay the last ones
    if not env["builtin_zlib"]:
        add_external_libs(["zlib"])

    env.Prepend(CPPPATH=["#platform/linuxbsd"])
    if env["use_sowrap"]:
        env.Prepend(CPPPATH=["#thirdparty/linuxbsd_headers"])

    env.Append(
        CPPDEFINES=[
            "LINUXBSD_ENABLED",
            "UNIX_ENABLED",
            ("_FILE_OFFSET_BITS", 64),
        ]
    )

    if env["x11"]:
        if not env["use_sowrap"]:
            if add_external_libs(["x11"]) != 0:
                print("Error: X11 libraries not found. Aborting.")
                sys.exit(255)
            if add_external_libs(["xcursor"]) != 0:
                print("Error: Xcursor library not found. Aborting.")
                sys.exit(255)
            if add_external_libs(["xinerama"]) != 0:
                print("Error: Xinerama library not found. Aborting.")
                sys.exit(255)
            if add_external_libs(["xext"]) != 0:
                print("Error: Xext library not found. Aborting.")
                sys.exit(255)
            if add_external_libs(["xrandr"]) != 0:
                print("Error: XrandR library not found. Aborting.")
                sys.exit(255)
            if add_external_libs(["xrender"]) != 0:
                print("Error: XRender library not found. Aborting.")
                sys.exit(255)
            if add_external_libs(["xi"]) != 0:
                print("Error: Xi library not found. Aborting.")
                sys.exit(255)
        env.Append(CPPDEFINES=["X11_ENABLED"])

    if env["vulkan"]:
        env.Append(CPPDEFINES=["VULKAN_ENABLED"])
        if not env["use_volk"]:
            add_external_libs(["vulkan"])
        if not env["builtin_glslang"]:
            # No pkgconfig file so far, hardcode expected lib name.
            # A patch to add a pkg-config file has been submitted
            # upstream: https://github.com/KhronosGroup/glslang/pull/3371
            # Once it gets added to glslang and we revbump to match
            # that version, we can start using add_external_libs()
            env.Append(LIBS=["glslang", "SPIRV"])

    if env["opengl3"]:
        env.Append(CPPDEFINES=["GLES3_ENABLED"])

    env.Append(LIBS=["pthread"])

    if platform.system() == "Linux":
        env.Append(LIBS=["dl"])

    if not env["execinfo"] and platform.libc_ver()[0] != "glibc":
        # The default crash handler depends on glibc, so if the host uses
        # a different libc (BSD libc, musl), fall back to libexecinfo.
        print("Note: Using `execinfo=yes` for the crash handler as required on platforms where glibc is missing.")
        env["execinfo"] = True

    if env["execinfo"]:
        env.Append(LIBS=["execinfo"])

    if not env.editor_build:
        import subprocess
        import re

        linker_version_str = subprocess.check_output(
            [env.subst(env["LINK"]), "-Wl,--version"] + env.subst(env["LINKFLAGS"])
        ).decode("utf-8")
        gnu_ld_version = re.search(r"^GNU ld [^$]*(\d+\.\d+)$", linker_version_str, re.MULTILINE)
        if not gnu_ld_version:
            print(
                "Warning: Creating export template binaries enabled for PCK embedding is currently only supported with GNU ld, not gold, LLD or mold."
            )
        else:
            if float(gnu_ld_version.group(1)) >= 2.30:
                env.Append(LINKFLAGS=["-T", "platform/linuxbsd/pck_embed.ld"])
            else:
                env.Append(LINKFLAGS=["-T", "platform/linuxbsd/pck_embed.legacy.ld"])

    if platform.system() == "FreeBSD":
        env.Append(LINKFLAGS=["-lkvm"])

    ## Cross-compilation
    # TODO: Support cross-compilation on architectures other than x86.
    host_is_64_bit = sys.maxsize > 2**32
    if host_is_64_bit and env["arch"] == "x86_32":
        env.Append(CCFLAGS=["-m32"])
        env.Append(LINKFLAGS=["-m32", "-L/usr/lib/i386-linux-gnu"])
    elif not host_is_64_bit and env["arch"] == "x86_64":
        env.Append(CCFLAGS=["-m64"])
        env.Append(LINKFLAGS=["-m64", "-L/usr/lib/i686-linux-gnu"])

    # Link those statically for portability
    if env["use_static_cpp"]:
        env.Append(LINKFLAGS=["-static-libgcc", "-static-libstdc++"])
        if env["use_llvm"] and platform.system() != "FreeBSD":
            env["LINKCOM"] = env["LINKCOM"] + " -l:libatomic.a"

    else:
        if env["use_llvm"] and platform.system() != "FreeBSD":
            env.Append(LIBS=["atomic"])
