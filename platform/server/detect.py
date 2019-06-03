
import os
import sys


def is_active():
    return True


def get_name():
    return "Server"


def can_build():

    if (os.name != "posix"):
        return False

    return True  # enabled


def get_opts():

    return [
        ('use_llvm', 'Use llvm compiler', 'no'),
        ('force_32_bits', 'Force 32 bits binary', 'no')
    ]


def get_flags():

    return [
    ]


def configure(env):

    env.Append(CPPPATH=['#platform/server'])
    if (env["use_llvm"] == "yes"):
        env["CC"] = "clang"
        env["CXX"] = "clang++"
        env["LD"] = "clang++"
    elif (os.system("gcc --version > /dev/null 2>&1") == 0): # GCC
        # Hack to prevent building this branch with GCC 6+, which trigger segfaults due to UB when dereferencing pointers in Object::cast_to
        # This is fixed in the master branch, for 2.1 we just prevent using too recent GCC versions.
        import subprocess
        gcc_major = subprocess.check_output(['gcc', '-dumpversion']).decode('ascii').split('.')[0]
        if (int(gcc_major) > 5):
            print("Your configured compiler appears to be GCC %s, which triggers issues in release builds for this version of Godot (fixed in Godot 3.0+)." % gcc_major)
            print("You can use the Clang compiler instead with the `use_llvm=yes` option, or configure another compiler such as GCC 5 using the CC, CXX and LD flags.")
            print("Aborting..")
            sys.exit(255)

    is64 = sys.maxsize > 2**32

    if (env["bits"] == "default"):
        if (is64):
            env["bits"] = "64"
        else:
            env["bits"] = "32"

    # if (env["tools"]=="no"):
    #	#no tools suffix
    #	env['OBJSUFFIX'] = ".nt"+env['OBJSUFFIX']
    #	env['LIBSUFFIX'] = ".nt"+env['LIBSUFFIX']

    if (env["target"] == "release"):

        env.Append(CCFLAGS=['-O2', '-ffast-math', '-fomit-frame-pointer'])

    elif (env["target"] == "release_debug"):

        env.Append(CCFLAGS=['-O2', '-ffast-math', '-DDEBUG_ENABLED'])

    elif (env["target"] == "debug"):

        env.Append(CCFLAGS=['-g2', '-DDEBUG_ENABLED', '-DDEBUG_MEMORY_ENABLED'])


    # Shared libraries, when requested

    if (env['builtin_openssl'] == 'no'):
        env.ParseConfig('pkg-config openssl --cflags --libs')

    if (env['builtin_libwebp'] == 'no'):
        env.ParseConfig('pkg-config libwebp --cflags --libs')

    if (env['builtin_freetype'] == 'no'):
        env.ParseConfig('pkg-config freetype2 --cflags --libs')

    if (env['builtin_libpng'] == 'no'):
        env.ParseConfig('pkg-config libpng --cflags --libs')

    # Sound and video libraries
    # Keep the order as it triggers chained dependencies (ogg needed by others, etc.)

    if (env['builtin_libtheora'] == 'no'):
        env['builtin_libogg'] = 'no'  # Needed to link against system libtheora
        env['builtin_libvorbis'] = 'no'  # Needed to link against system libtheora
        env.ParseConfig('pkg-config theora theoradec --cflags --libs')

    if (env['builtin_libvorbis'] == 'no'):
        env['builtin_libogg'] = 'no'  # Needed to link against system libvorbis
        env.ParseConfig('pkg-config vorbis vorbisfile --cflags --libs')

    if (env['builtin_opus'] == 'no'):
        env['builtin_libogg'] = 'no'  # Needed to link against system opus
        env.ParseConfig('pkg-config opus opusfile --cflags --libs')

    if (env['builtin_libogg'] == 'no'):
        env.ParseConfig('pkg-config ogg --cflags --libs')


    env.Append(CPPFLAGS=['-DSERVER_ENABLED', '-DUNIX_ENABLED'])
    env.Append(LIBS=['pthread', 'z'])  # TODO detect linux/BSD!

    if (env["CXX"] == "clang++"):
        env.Append(CPPFLAGS=['-DTYPED_METHOD_BIND'])
        env["CC"] = "clang"
        env["LD"] = "clang++"
