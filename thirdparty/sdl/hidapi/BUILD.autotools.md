# Building HIDAPI using Autotools (deprecated)

---
**NOTE**: for all intentions and purposes the Autotools build scripts for HIDAPI are _deprecated_ and going to be obsolete in the future.
HIDAPI Team recommends using CMake build for HIDAPI.
If you are already using Autotools build scripts provided by HIDAPI,
consider switching to CMake build scripts as soon as possible.

---

To be able to use Autotools to build HIDAPI, it has to be [installed](#installing-autotools)/available in the system.

Make sure you've checked [prerequisites](BUILD.md#prerequisites) and installed all required dependencies.

## Installing Autotools

HIDAPI uses few specific tools/packages from Autotools: `autoconf`, `automake`, `libtool`.

On different platforms or package managers, those could be named a bit differently or packaged together.
You'll have to check the documentation/package list for your specific package manager.

### Linux

On Ubuntu the tools are available via APT:

```sh
sudo apt install autoconf automake libtool
```

### FreeBSD

FreeBSD Autotools can be installed as:

```sh
pkg_add -r autotools
```

Additionally, on FreeBSD you will need to install GNU make:
```sh
pkg_add -r gmake
```

## Building HIDAPI with Autotools

A simple command list, to build HIDAPI with Autotools as a _shared library_ and install in into your system:

```sh
./bootstrap # this prepares the configure script
./configure
make # build the library
make install # as root, or using sudo, this will install hidapi into your system
```

`./configure` can take several arguments which control the build. A few commonly used options:
```sh
	--enable-testgui
		# Enable the build of Foxit-based Test GUI. This requires Fox toolkit to
		# be installed/available. See README.md#test-gui for remarks.

	--prefix=/usr
		# Specify where you want the output headers and libraries to
		# be installed. The example above will put the headers in
		# /usr/include and the binaries in /usr/lib. The default is to
		# install into /usr/local which is fine on most systems.

	--disable-shared
		# By default, both shared and static libraries are going to be built/installed.
		# This option disables shared library build, if only static library is required.
```


## Cross Compiling

This section talks about cross compiling HIDAPI for Linux using Autotools.
This is useful for using HIDAPI on embedded Linux targets. These
instructions assume the most raw kind of embedded Linux build, where all
prerequisites will need to be built first. This process will of course vary
based on your embedded Linux build system if you are using one, such as
OpenEmbedded or Buildroot.

For the purpose of this section, it will be assumed that the following
environment variables are exported.
```sh
$ export STAGING=$HOME/out
$ export HOST=arm-linux
```

`STAGING` and `HOST` can be modified to suit your setup.

### Prerequisites

Depending on what backend you want to cross-compile, you also need to prepare the dependencies:
`libusb` for libusb HIDAPI backend, or `libudev` for hidraw HIDAPI backend.

An example of cross-compiling `libusb`. From `libusb` source directory, run:
```sh
./configure --host=$HOST --prefix=$STAGING
make
make install
```

An example of cross-comping `libudev` is not covered by this section.
Check `libudev`'s documentation for details.

### Building HIDAPI

Build HIDAPI:
```sh
PKG_CONFIG_DIR= \
PKG_CONFIG_LIBDIR=$STAGING/lib/pkgconfig:$STAGING/share/pkgconfig \
PKG_CONFIG_SYSROOT_DIR=$STAGING \
./configure --host=$HOST --prefix=$STAGING
# make / make install - same as for a regular build
```
