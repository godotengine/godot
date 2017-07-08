         HIDAPI library for Windows, Linux, FreeBSD and Mac OS X
        =========================================================

About
======

HIDAPI is a multi-platform library which allows an application to interface
with USB and Bluetooth HID-Class devices on Windows, Linux, FreeBSD, and Mac
OS X.  HIDAPI can be either built as a shared library (.so or .dll) or
can be embedded directly into a target application by adding a single source
file (per platform) and a single header.

HIDAPI has four back-ends:
	* Windows (using hid.dll)
	* Linux/hidraw (using the Kernel's hidraw driver)
	* Linux/libusb (using libusb-1.0)
	* FreeBSD (using libusb-1.0)
	* Mac (using IOHidManager)

On Linux, either the hidraw or the libusb back-end can be used. There are
tradeoffs, and the functionality supported is slightly different.

Linux/hidraw (linux/hid.c):
This back-end uses the hidraw interface in the Linux kernel.  While this
back-end will support both USB and Bluetooth, it has some limitations on
kernels prior to 2.6.39, including the inability to send or receive feature
reports.  In addition, it will only communicate with devices which have
hidraw nodes associated with them.  Keyboards, mice, and some other devices
which are blacklisted from having hidraw nodes will not work. Fortunately,
for nearly all the uses of hidraw, this is not a problem.

Linux/FreeBSD/libusb (libusb/hid.c):
This back-end uses libusb-1.0 to communicate directly to a USB device. This
back-end will of course not work with Bluetooth devices.

HIDAPI also comes with a Test GUI. The Test GUI is cross-platform and uses
Fox Toolkit (http://www.fox-toolkit.org).  It will build on every platform
which HIDAPI supports.  Since it relies on a 3rd party library, building it
is optional but recommended because it is so useful when debugging hardware.

What Does the API Look Like?
=============================
The API provides the the most commonly used HID functions including sending
and receiving of input, output, and feature reports.  The sample program,
which communicates with a heavily hacked up version of the Microchip USB
Generic HID sample looks like this (with error checking removed for
simplicity):

#ifdef WIN32
#include <windows.h>
#endif
#include <stdio.h>
#include <stdlib.h>
#include "hidapi.h"

#define MAX_STR 255

int main(int argc, char* argv[])
{
	int res;
	unsigned char buf[65];
	wchar_t wstr[MAX_STR];
	hid_device *handle;
	int i;

	// Initialize the hidapi library
	res = hid_init();

	// Open the device using the VID, PID,
	// and optionally the Serial number.
	handle = hid_open(0x4d8, 0x3f, NULL);

	// Read the Manufacturer String
	res = hid_get_manufacturer_string(handle, wstr, MAX_STR);
	wprintf(L"Manufacturer String: %s\n", wstr);

	// Read the Product String
	res = hid_get_product_string(handle, wstr, MAX_STR);
	wprintf(L"Product String: %s\n", wstr);

	// Read the Serial Number String
	res = hid_get_serial_number_string(handle, wstr, MAX_STR);
	wprintf(L"Serial Number String: (%d) %s\n", wstr[0], wstr);

	// Read Indexed String 1
	res = hid_get_indexed_string(handle, 1, wstr, MAX_STR);
	wprintf(L"Indexed String 1: %s\n", wstr);

	// Toggle LED (cmd 0x80). The first byte is the report number (0x0).
	buf[0] = 0x0;
	buf[1] = 0x80;
	res = hid_write(handle, buf, 65);

	// Request state (cmd 0x81). The first byte is the report number (0x0).
	buf[0] = 0x0;
	buf[1] = 0x81;
	res = hid_write(handle, buf, 65);

	// Read requested state
	res = hid_read(handle, buf, 65);

	// Print out the returned buffer.
	for (i = 0; i < 4; i++)
		printf("buf[%d]: %d\n", i, buf[i]);

	// Finalize the hidapi library
	res = hid_exit();

	return 0;
}

If you have your own simple test programs which communicate with standard
hardware development boards (such as those from Microchip, TI, Atmel,
FreeScale and others), please consider sending me something like the above
for inclusion into the HIDAPI source.  This will help others who have the
same hardware as you do.

License
========
HIDAPI may be used by one of three licenses as outlined in LICENSE.txt.

Download
=========
HIDAPI can be downloaded from github
	git clone git://github.com/signal11/hidapi.git

Build Instructions
===================

This section is long. Don't be put off by this. It's not long because it's
complicated to build HIDAPI; it's quite the opposite.  This section is long
because of the flexibility of HIDAPI and the large number of ways in which
it can be built and used.  You will likely pick a single build method.

HIDAPI can be built in several different ways. If you elect to build a
shared library, you will need to build it from the HIDAPI source
distribution.  If you choose instead to embed HIDAPI directly into your
application, you can skip the building and look at the provided platform
Makefiles for guidance.  These platform Makefiles are located in linux/
libusb/ mac/ and windows/ and are called Makefile-manual.  In addition,
Visual Studio projects are provided.  Even if you're going to embed HIDAPI
into your project, it is still beneficial to build the example programs.


Prerequisites:
---------------

	Linux:
	-------
	On Linux, you will need to install development packages for libudev,
	libusb and optionally Fox-toolkit (for the test GUI). On
	Debian/Ubuntu systems these can be installed by running:
	    sudo apt-get install libudev-dev libusb-1.0-0-dev libfox-1.6-dev

	If you downloaded the source directly from the git repository (using
	git clone), you'll need Autotools:
	    sudo apt-get install autotools-dev autoconf automake libtool

	FreeBSD:
	---------
	On FreeBSD you will need to install GNU make, libiconv, and
	optionally Fox-Toolkit (for the test GUI). This is done by running
	the following:
	    pkg_add -r gmake libiconv fox16

	If you downloaded the source directly from the git repository (using
	git clone), you'll need Autotools:
	    pkg_add -r autotools

	Mac:
	-----
	On Mac, you will need to install Fox-Toolkit if you wish to build
	the Test GUI. There are two ways to do this, and each has a slight
	complication. Which method you use depends on your use case.

	If you wish to build the Test GUI just for your own testing on your
	own computer, then the easiest method is to install Fox-Toolkit
	using ports:
		sudo port install fox

	If you wish to build the TestGUI app bundle to redistribute to
	others, you will need to install Fox-toolkit from source.  This is
	because the version of fox that gets installed using ports uses the
	ports X11 libraries which are not compatible with the Apple X11
	libraries.  If you install Fox with ports and then try to distribute
	your built app bundle, it will simply fail to run on other systems.
	To install Fox-Toolkit manually, download the source package from
	http://www.fox-toolkit.org, extract it, and run the following from
	within the extracted source:
		./configure && make && make install

	Windows:
	---------
	On Windows, if you want to build the test GUI, you will need to get
	the hidapi-externals.zip package from the download site.  This
	contains pre-built binaries for Fox-toolkit.  Extract
	hidapi-externals.zip just outside of hidapi, so that
	hidapi-externals and hidapi are on the same level, as shown:

	     Parent_Folder
	       |
	       +hidapi
	       +hidapi-externals

	Again, this step is not required if you do not wish to build the
	test GUI.


Building HIDAPI into a shared library on Unix Platforms:
---------------------------------------------------------

On Unix-like systems such as Linux, FreeBSD, Mac, and even Windows, using
Mingw or Cygwin, the easiest way to build a standard system-installed shared
library is to use the GNU Autotools build system.  If you checked out the
source from the git repository, run the following:

	./bootstrap
	./configure
	make
	make install     <----- as root, or using sudo

If you downloaded a source package (ie: if you did not run git clone), you
can skip the ./bootstrap step.

./configure can take several arguments which control the build. The two most
likely to be used are:
	--enable-testgui
		Enable build of the Test GUI. This requires Fox toolkit to
		be installed.  Instructions for installing Fox-Toolkit on
		each platform are in the Prerequisites section above.

	--prefix=/usr
		Specify where you want the output headers and libraries to
		be installed. The example above will put the headers in
		/usr/include and the binaries in /usr/lib. The default is to
		install into /usr/local which is fine on most systems.

Building the manual way on Unix platforms:
-------------------------------------------

Manual Makefiles are provided mostly to give the user and idea what it takes
to build a program which embeds HIDAPI directly inside of it. These should
really be used as examples only. If you want to build a system-wide shared
library, use the Autotools method described above.

	To build HIDAPI using the manual makefiles, change to the directory
	of your platform and run make. For example, on Linux run:
		cd linux/
		make -f Makefile-manual

	To build the Test GUI using the manual makefiles:
		cd testgui/
		make -f Makefile-manual

Building on Windows:
---------------------

To build the HIDAPI DLL on Windows using Visual Studio, build the .sln file
in the windows/ directory.

To build the Test GUI on windows using Visual Studio, build the .sln file in
the testgui/ directory.

To build HIDAPI using MinGW or Cygwin using Autotools, use the instructions
in the section titled "Building HIDAPI into a shared library on Unix
Platforms" above.  Note that building the Test GUI with MinGW or Cygwin will
require the Windows procedure in the Prerequisites section above (ie:
hidapi-externals.zip).

To build HIDAPI using MinGW using the Manual Makefiles, see the section
"Building the manual way on Unix platforms" above.

HIDAPI can also be built using the Windows DDK (now also called the Windows
Driver Kit or WDK). This method was originally required for the HIDAPI build
but not anymore. However, some users still prefer this method. It is not as
well supported anymore but should still work. Patches are welcome if it does
not. To build using the DDK:

   1. Install the Windows Driver Kit (WDK) from Microsoft.
   2. From the Start menu, in the Windows Driver Kits folder, select Build
      Environments, then your operating system, then the x86 Free Build
      Environment (or one that is appropriate for your system).
   3. From the console, change directory to the windows/ddk_build/ directory,
      which is part of the HIDAPI distribution.
   4. Type build.
   5. You can find the output files (DLL and LIB) in a subdirectory created
      by the build system which is appropriate for your environment. On
      Windows XP, this directory is objfre_wxp_x86/i386.

Cross Compiling
================

This section talks about cross compiling HIDAPI for Linux using autotools.
This is useful for using HIDAPI on embedded Linux targets.  These
instructions assume the most raw kind of embedded Linux build, where all
prerequisites will need to be built first.  This process will of course vary
based on your embedded Linux build system if you are using one, such as
OpenEmbedded or Buildroot.

For the purpose of this section, it will be assumed that the following
environment variables are exported.

	$ export STAGING=$HOME/out
	$ export HOST=arm-linux

STAGING and HOST can be modified to suit your setup.

Prerequisites
--------------

Note that the build of libudev is the very basic configuration.

Build Libusb. From the libusb source directory, run:
	./configure --host=$HOST --prefix=$STAGING
	make
	make install

Build libudev. From the libudev source directory, run:
	./configure --disable-gudev --disable-introspection --disable-hwdb \
		 --host=$HOST --prefix=$STAGING
	make
	make install

Building HIDAPI
----------------

Build HIDAPI:

	PKG_CONFIG_DIR= \
	PKG_CONFIG_LIBDIR=$STAGING/lib/pkgconfig:$STAGING/share/pkgconfig \
	PKG_CONFIG_SYSROOT_DIR=$STAGING \
	./configure --host=$HOST --prefix=$STAGING


Signal 11 Software - 2010-04-11
                     2010-07-28
                     2011-09-10
                     2012-05-01
                     2012-07-03
