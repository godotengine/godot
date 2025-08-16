CMake
=====

.. warning::

    The CMake scripts do not have feature parity with the SCons ones at this
    stage and are still a work in progress. There are a number of people who
    have been working on alternative CMake solutions that are frequently
    referenced in the discord chats: Ivan's cmake-rewrite_ branch and
    Vorlac's godot-roguelite_ Project

.. _cmake-rewrite: https://github.com/IvanInventor/godot-cpp/tree/cmake-rewrite
.. _godot-roguelite: https://github.com/vorlac/godot-roguelite

Introduction
------------

Compiling godot-cpp independently of an extension project is mainly for
godot-cpp developers, package maintainers, and CI/CD. Look to the
godot-cpp-template_ for a practical example on how to consume the godot-cpp
library as part of a Godot extension.

Configuration examples are listed at the bottom of the page.

.. _godot-cpp-template: https://github.com/godotengine/godot-cpp-template

Debug vs template_debug
-----------------------

Something I've seen come up many times is the conflation of a compilation of c++
source code with debug symbols enabled, and compiling a Godot extension with
debug features enabled. The two concepts are not mutually inclusive.

- debug_features
    Enables a pre-processor definition to selectively compile code to help
    users of a Godot extension with their own project.

    debug features are enabled in editor and template_debug builds, which can be specified during the configure phase like so

	``cmake -S . -B cmake-build -DGODOTCPP_TARGET=<target choice>``

- Debug
    Sets compiler flags so that debug symbols are generated to help godot
    extension developers debug their extension.

    ``Debug`` is the default build type for CMake projects, to select another it depends on the generator used

    For single configuration generators, add to the configure command:

	``-DCMAKE_BUILD_TYPE=<type>``

    For multi-config generators add to the build command:

	``--config <type>``

    where ``<type>`` is one of ``Debug``, ``Release``, ``RelWithDebInfo``, ``MinSizeRel``


SCons Deviations
----------------

Not everything from SCons can be perfectly representable in CMake, here are
the notable differences.

- debug_symbols
    No longer has an explicit option, and is enabled via Debug-like CMake
    build configurations; ``Debug``, ``RelWithDebInfo``.

- dev_build
    Does not define ``NDEBUG`` when disabled, ``NDEBUG`` is set via Release-like
    CMake build configurations; ``Release``, ``MinSizeRel``.

- arch
    CMake sets the architecture via the toolchain files, macos universal is controlled vua the ``CMAKE_OSX_ARCHITECTURES``
    property which is copied to targets when they are defined.

- debug_crt
    CMake controls linking to windows runtime libraries by copying the value of ``CMAKE_MSVC_RUNTIME_LIBRARIES`` to targets as they are defined.
    godot-cpp will set this variable if it isn't already set. so include it before other dependencies to have the value propagate across the projects.

Testing Integration
-------------------
The testing target ``godot-cpp-test`` is guarded by ``GODOTCPP_ENABLE_TESTING`` which is off by default.

To configure and build the godot-cpp project to enable the integration
testing targets the command will look something like:

.. code-block::

    # Assuming our current directory is the godot-cpp source root
    cmake -S . -B cmake-build -DGODOTCPP_ENABLE_TESTING=YES
    cmake --build cmake-build --target godot-cpp-test

Basic walkthrough
-----------------

.. topic:: Clone the git repository

    .. code-block::

        git clone https://github.com/godotengine/godot-cpp.git
        Cloning into 'godot-cpp'...
        ...
        cd godot-cpp

.. topic:: Options

    To list the available options CMake use the ``-L[AH]`` option. ``A`` is for
    advanced, and ``H`` is for help strings.

    .. code-block::

        cmake .. -LH

    Options are specified on the command line when configuring eg.

    .. code-block::

        cmake .. -DGODOTCPP_USE_HOT_RELOAD:BOOL=ON \
            -DGODOTCPP_PRECISION:STRING=double \
            -DCMAKE_BUILD_TYPE:STRING=Debug

    Review setting-build-variables_ and build-configurations_ for more information.

    .. _setting-build-variables: https://cmake.org/cmake/help/latest/guide/user-interaction/index.html#setting-build-variables
    .. _build-configurations: https://cmake.org/cmake/help/latest/manual/cmake-buildsystem.7.html#build-configurations

    A non-exhaustive list of options:

    .. code-block::

        // Path to a custom GDExtension API JSON file (takes precedence over `GODOTCPP_GDEXTENSION_DIR`) ( /path/to/custom_api_file )
        `GODOTCPP_CUSTOM_API_FILE:FILEPATH=`

        // Force disabling exception handling code (ON|OFF)
        GODOTCPP_DISABLE_EXCEPTIONS:BOOL=ON

        // Path to a custom directory containing GDExtension interface header and API JSON file ( /path/to/gdextension_dir )
        GODOTCPP_GDEXTENSION_DIR:PATH=gdextension

        // Set the floating-point precision level (single|double)
        GODOTCPP_PRECISION:STRING=single

        // Enable the extra accounting required to support hot reload. (ON|OFF)
        GODOTCPP_USE_HOT_RELOAD:BOOL=

.. topic:: Configure the build

    .. code-block::

        cmake -S . -B cmake-build -G Ninja

    ``-S .`` Specifies the source directory

    ``-B cmake-build`` Specifies the build directory

    ``-G Ninja`` Specifies the Generator

    The source directory in this example is the source code for godot-cpp.
    The build directory is so that generated files do not clutter up the source tree.
    CMake doesn't build the code, it generates the files that another tool uses
    to build the code, in this case Ninja.
    To see the list of generators run ``cmake --help``.

.. topic:: Compiling

    Tell cmake to invoke the build system it generated in the specified directory.
    The default target is template_debug and the default build configuration is Debug.

    .. code-block::

        cmake --build cmake-build

Examples
--------

Windows and MSVC - Release
~~~~~~~~~~~~~~~~~~~~~~~~~~
So long as CMake is installed from the `CMake Downloads`_ page and in the PATH,
and Microsoft Visual Studio is installed with c++ support, CMake will detect
the MSVC compiler.

Note that Visual Studio is a Multi-Config Generator so the build configuration
needs to be specified at build time ie ``--config Release``

.. _CMake downloads: https://cmake.org/download/

.. code-block::

    # Assuming our current directory is the godot-cpp source root
    cmake -S . -B cmake-build -DGODOTCPP_ENABLE_TESTING=YES
    cmake --build cmake-build -t godot-cpp-test --config Release


MSys2/clang64, "Ninja" - Debug
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Assumes the ming-w64-clang-x86_64-toolchain is installed

Note that Ninja is a Single-Config Generator so the build type
needs to be specified at Configure time.

Using the msys2/clang64 shell

.. code-block::

    # Assuming our current directory is the godot-cpp source root
    cmake -S . -B cmake-build -G"Ninja" -DGODOTCPP_ENABLE_TESTING=YES -DCMAKE_BUILD_TYPE=Release
    cmake --build cmake-build -t godot-cpp-test

MSys2/clang64, "Ninja Multi-Config" - dev_build, Debug Symbols
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Assumes the ming-w64-clang-x86_64-toolchain is installed

This time we are choosing the 'Ninja Multi-Config' generator, so the build
type is specified at build time.

Using the msys2/clang64 shell

.. code-block::

    # Assuming our current directory is the godot-cpp source root
    cmake -S . -B cmake-build -G"Ninja Multi-Config" -DGODOTCPP_ENABLE_TESTING=YES -DGODOTCPP_DEV_BUILD:BOOL=ON
    cmake --build cmake-build -t godot-cpp-test --config Debug

Emscripten for web platform
~~~~~~~~~~~~~~~~~~~~~~~~~~~
I've only tested this on windows so far.

I cloned and installed the latest Emscripten tools to ``c:\emsdk``
At the time of writing that was v3.1.69

I've been using ``C:\emsdk\emsdk.ps1 activate latest`` to enable the
environment from powershell in the current shell.

The ``emcmake.bat`` utility adds the emscripten toolchain to the CMake command
It can also be added manually, the location is listed inside the emcmake.bat file

.. code-block::

    # Assuming our current directory is the godot-cpp source root
    C:\emsdk\emsdk.ps1 activate latest
    emcmake.bat cmake -S . -B cmake-build-web -DCMAKE_BUILD_TYPE=Release
    cmake --build cmake-build-web

Android Cross Compile from Windows
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
There are two separate paths you can choose when configuring for android.

Use the ``CMAKE_ANDROID_*`` variables specified on the commandline or in your
own toolchain file as listed in the cmake-toolchains_ documentation

.. _cmake-toolchains: https://cmake.org/cmake/help/latest/manual/cmake-toolchains.7.html#cross-compiling-for-android-with-the-ndk

Or use the toolchain and scripts provided by the Android SDK and make changes
using the ``ANDROID_*`` variables listed there. Where ``<version>`` is whatever
ndk version you have installed (tested with `28.1.13356709`) and ``<platform>``
is for android sdk platform, (tested with ``android-29``)

.. warning::

    The Android SDK website explicitly states that they do not support using
    the CMake built-in method, and recommends you stick with their toolchain
    files.

.. topic:: Using your own toolchain file as described in the CMake documentation

    .. code-block::

        # Assuming our current directory is the godot-cpp source root
        cmake -S . -B cmake-build --toolchain my_toolchain.cmake
        cmake --build cmake-build -t template_release

    Doing the equivalent on just using the command line

    .. code-block::

        # Assuming our current directory is the godot-cpp source root
        cmake -S . -B cmake-build \
            -DCMAKE_SYSTEM_NAME=Android \
            -DCMAKE_SYSTEM_VERSION=<platform> \
            -DCMAKE_ANDROID_ARCH_ABI=<arch> \
            -DCMAKE_ANDROID_NDK=/path/to/android-ndk
        cmake --build cmake-build

.. topic:: Using the toolchain file from the Android SDK

    Defaults to minimum supported version( android-16 in my case) and armv7-a.

    .. code-block::

        # Assuming our current directory is the godot-cpp source root
        cmake -S . -B cmake-build --toolchain $ANDROID_HOME/ndk/<version>/build/cmake/android.toolchain.cmake
        cmake --build cmake-build

    Specify Android platform and ABI

    .. code-block::

        # Assuming our current directory is the godot-cpp source root
        cmake -S . -B cmake-build --toolchain $ANDROID_HOME/ndk/<version>/build/cmake/android.toolchain.cmake \
            -DANDROID_PLATFORM:STRING=android-29 \
            -DANDROID_ABI:STRING=armeabi-v7a
        cmake --build cmake-build


Toolchains
----------
This section attempts to list the host and target combinations that have been
at tested.

Linux Host
~~~~~~~~~~

Macos Host
~~~~~~~~~~

:System: Mac Mini
:OS Name: Sequoia 15.0.1
:Processor: Apple M2

* AppleClang

Windows Host
~~~~~~~~~~~~

:OS Name: Windows 11
:Processor: AMD Ryzen 7 6800HS Creator Edition


* `Microsoft Visual Studio 17 2022 <https://visualstudio.microsoft.com/vs/>`_
* `LLVM <https://llvm.org/>`_
* `LLVM-MinGW <https://github.com/mstorsjo/llvm-mingw/releases>`_

    * aarch64-w64-mingw32
    * armv7-w64-mingw32
    * i686-w64-mingw32
    * x86_64-w64-mingw32

* `AndroidSDK <https://developer.android.com/studio/#command-tools>`_
* `Emscripten <https://emscripten.org/>`_
* `MinGW-W64-builds <https://github.com/niXman/mingw-builds-binaries/releases>`_
* `Jetbrains-CLion <https://www.jetbrains.com/clion/>`_

    Jetbrains builtin compiler is just the MingW64 above.

* `MSYS2 <https://www.msys2.org/>`_
    Necessary reading about MSYS2 `environments <https://www.msys2.org/docs/environments/>`_

    * ucrt64
    * clang64
    * mingw32
    * mingw64
    * clangarm64
