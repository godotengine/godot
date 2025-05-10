# Building HIDAPI from Source

## Table of content

* [Intro](#intro)
* [Prerequisites](#prerequisites)
    * [Linux](#linux)
    * [FreeBSD](#freebsd)
    * [Mac](#mac)
    * [Windows](#windows)
* [Embedding HIDAPI directly into your source tree](#embedding-hidapi-directly-into-your-source-tree)
* [Building the manual way on Unix platforms](#building-the-manual-way-on-unix-platforms)
* [Building on Windows](#building-on-windows)

## Intro

For various reasons, you may need to build HIDAPI on your own.

It can be done in several different ways:
- using [CMake](BUILD.cmake.md);
- using [Autotools](BUILD.autotools.md) (deprecated);
- using [manual makefiles](#building-the-manual-way-on-unix-platforms);
- using `Meson` (requires CMake);

**Autotools** build system is historically the first mature build system for
HIDAPI. The most common usage of it is in its separate README: [BUILD.autotools.md](BUILD.autotools.md).<br/>
NOTE: for all intentions and purposes the Autotools build scripts for HIDAPI are _deprecated_ and going to be obsolete in the future.
HIDAPI Team recommends using CMake build for HIDAPI.

**CMake** build system is de facto an industry standard for many open-source and proprietary projects and solutions.
HIDAPI is one of the projects which use the power of CMake to its advantage.
More documentation is available in its separate README: [BUILD.cmake.md](BUILD.cmake.md).

**Meson** build system for HIDAPI is designed as a [wrapper](https://mesonbuild.com/CMake-module.html) over CMake build script.
It is present for the convenience of Meson users who need to use HIDAPI and need to be sure HIDAPI is built in accordance with officially supported build scripts.<br>
In the Meson script of your project you need a `hidapi = subproject('hidapi')` subproject, and `hidapi.get_variable('hidapi_dep')` as your dependency.
There are also backend/platform-specific dependencies available: `hidapi_winapi`, `hidapi_darwin`, `hidapi_hidraw`, `hidapi_libusb`.

If you don't know where to start to build HIDAPI, we recommend starting with [CMake](BUILD.cmake.md) build.

## Prerequisites:

Regardless of what build system you choose to use, there are specific dependencies for each platform/backend.

### Linux:

Depending on which backend you're going to build, you'll need to install
additional development packages. For `linux/hidraw` backend, you need a
development package for `libudev`. For `libusb` backend, naturally, you need
`libusb` development package.

On Debian/Ubuntu systems these can be installed by running:
```sh
# required only by hidraw backend
sudo apt install libudev-dev
# required only by libusb backend
sudo apt install libusb-1.0-0-dev
```

### FreeBSD:

On FreeBSD, you will need to install libiconv. This is done by running
the following:
```sh
pkg_add -r libiconv
```

### Mac:

Make sure you have XCode installed and its Command Line Tools.

### Windows:

You just need a compiler. You may use Visual Studio or Cygwin/MinGW,
depending on which environment is best for your needs.

## Embedding HIDAPI directly into your source tree

Instead of using one of the provided standalone build systems,
you may want to integrate HIDAPI directly into your source tree.

---
If your project uses CMake as a build system, it is safe to add HIDAPI as a [subdirectory](BUILD.cmake.md#hidapi-as-a-subdirectory).

---
If _the only option_ that works for you is adding HIDAPI sources directly
to your project's build system, then you need:
- include a _single source file_ into your project's build system,
depending on your platform and the backend you want to use:
    - [`windows\hid.c`](windows/hid.c);
    - [`linux/hid.c`](linux/hid.c);
    - [`libusb/hid.c`](libusb/hid.c);
    - [`mac/hid.c`](mac/hid.c);
- add a [`hidapi`](hidapi) folder to the include path when building `hid.c`;
- make the platform/backend specific [dependencies](#prerequisites) available during the compilation/linking, when building `hid.c`;

NOTE: the above doesn't guarantee that having a copy of `<backend>/hid.c` and `hidapi/hidapi.h` is enough to build HIDAPI.
The only guarantee that `<backend>/hid.c` includes all necessary sources to compile it as a single file.

Check the manual makefiles for a simple example/reference of what are the dependencies of each specific backend.

## Building the manual way on Unix platforms

Manual Makefiles are provided mostly to give the user an idea what it takes
to build a program which embeds HIDAPI directly inside of it. These should
really be used as examples only. If you want to build a system-wide shared
library, use one of the build systems mentioned above.

To build HIDAPI using the manual Makefiles, change the directory
of your platform and run make. For example, on Linux run:
```sh
cd linux/
make -f Makefile-manual
```

## Building on Windows

To build the HIDAPI DLL on Windows using Visual Studio, build the `.sln` file
in the `windows/` directory.

To build HIDAPI using MinGW or Cygwin using Autotools, use general Autotools
 [instruction](BUILD.autotools.md).

Any windows builds (MSVC or MinGW/Cygwin) are also supported by [CMake](BUILD.cmake.md).

If you are looking for information regarding DDK build of HIDAPI:
- the build has been broken for a while and now the support files are obsolete.
