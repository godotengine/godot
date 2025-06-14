## HIDAPI library for Windows, Linux, FreeBSD and macOS

| CI instance          | Status |
|----------------------|--------|
| `Linux/macOS/Windows (master)` | [![GitHub Builds](https://github.com/libusb/hidapi/workflows/GitHub%20Builds/badge.svg?branch=master)](https://github.com/libusb/hidapi/actions/workflows/builds.yml?query=branch%3Amaster) |
| `Windows (master)` | [![Build status](https://ci.appveyor.com/api/projects/status/xfmr5fo8w0re8ded/branch/master?svg=true)](https://ci.appveyor.com/project/libusb/hidapi/branch/master) |
| `BSD, last build (branch/PR)` | [![builds.sr.ht status](https://builds.sr.ht/~z3ntu/hidapi.svg)](https://builds.sr.ht/~z3ntu/hidapi) |
| `Coverity Scan (last)` | ![Coverity Scan](https://scan.coverity.com/projects/583/badge.svg) |

HIDAPI is a multi-platform library which allows an application to interface
with USB and Bluetooth HID-Class devices on Windows, Linux, FreeBSD, and macOS.
HIDAPI can be either built as a shared library (`.so`, `.dll` or `.dylib`) or
can be embedded directly into a target application by adding a _single source_
file (per platform) and a single header.<br>
See [remarks](BUILD.md#embedding-hidapi-directly-into-your-source-tree) on embedding _directly_ into your build system.

HIDAPI library was originally developed by Alan Ott ([signal11](https://github.com/signal11)).

It was moved to [libusb/hidapi](https://github.com/libusb/hidapi) on June 4th, 2019, in order to merge important bugfixes and continue development of the library.

## Table of Contents

* [About](#about)
    * [Test GUI](#test-gui)
    * [Console Test App](#console-test-app)
* [What Does the API Look Like?](#what-does-the-api-look-like)
* [License](#license)
* [Installing HIDAPI](#installing-hidapi)
* [Build from Source](#build-from-source)


## About

### HIDAPI has four back-ends:
* Windows (using `hid.dll`)
* Linux/hidraw (using the Kernel's hidraw driver)
* libusb (using libusb-1.0 - Linux/BSD/other UNIX-like systems)
* macOS (using IOHidManager)

On Linux, either the hidraw or the libusb back-end can be used. There are
tradeoffs, and the functionality supported is slightly different. Both are
built by default. It is up to the application linking to hidapi to choose
the backend at link time by linking to either `libhidapi-libusb` or
`libhidapi-hidraw`.

Note that you will need to install an udev rule file with your application
for unprivileged users to be able to access HID devices with hidapi. Refer
to the [69-hid.rules](udev/69-hid.rules) file in the `udev` directory
for an example.

#### __Linux/hidraw__ (`linux/hid.c`):

This back-end uses the hidraw interface in the Linux kernel, and supports
both USB and Bluetooth HID devices. It requires kernel version at least 2.6.39
to build. In addition, it will only communicate with devices which have hidraw
nodes associated with them.
Keyboards, mice, and some other devices which are blacklisted from having
hidraw nodes will not work. Fortunately, for nearly all the uses of hidraw,
this is not a problem.

#### __Linux/FreeBSD/libusb__ (`libusb/hid.c`):

This back-end uses libusb-1.0 to communicate directly to a USB device. This
back-end will of course not work with Bluetooth devices.

### Test GUI

HIDAPI also comes with a Test GUI. The Test GUI is cross-platform and uses
Fox Toolkit <http://www.fox-toolkit.org>.  It will build on every platform
which HIDAPI supports.  Since it relies on a 3rd party library, building it
is optional but it is useful when debugging hardware.

NOTE: Test GUI based on Fox Toolkit is not actively developed nor supported
by HIDAPI team. It is kept as a historical artifact. It may even work sometime
or on some platforms, but it is not going to get any new features or bugfixes.

Instructions for installing Fox-Toolkit on each platform is not provided.
Make sure to use Fox-Toolkit v1.6 if you choose to use it.

### Console Test App

If you want to play around with your HID device before starting
any development with HIDAPI and using a GUI app is not an option for you, you may try [`hidapitester`](https://github.com/todbot/hidapitester).

This app has a console interface for most of the features supported
by HIDAPI library.

## What Does the API Look Like?

The API provides the most commonly used HID functions including sending
and receiving of input, output, and feature reports. The sample program,
which communicates with a heavily hacked up version of the Microchip USB
Generic HID sample looks like this (with error checking removed for
simplicity):

**Warning: Only run the code you understand, and only when it conforms to the
device spec. Writing data (`hid_write`) at random to your HID devices can break them.**

```c
#include <stdio.h> // printf
#include <wchar.h> // wchar_t

#include <hidapi.h>

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
	if (!handle) {
		printf("Unable to open device\n");
		hid_exit();
 		return 1;
	}

	// Read the Manufacturer String
	res = hid_get_manufacturer_string(handle, wstr, MAX_STR);
	printf("Manufacturer String: %ls\n", wstr);

	// Read the Product String
	res = hid_get_product_string(handle, wstr, MAX_STR);
	printf("Product String: %ls\n", wstr);

	// Read the Serial Number String
	res = hid_get_serial_number_string(handle, wstr, MAX_STR);
	printf("Serial Number String: (%d) %ls\n", wstr[0], wstr);

	// Read Indexed String 1
	res = hid_get_indexed_string(handle, 1, wstr, MAX_STR);
	printf("Indexed String 1: %ls\n", wstr);

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

	// Close the device
	hid_close(handle);

	// Finalize the hidapi library
	res = hid_exit();

	return 0;
}
```

You can also use [hidtest/test.c](hidtest/test.c)
as a starting point for your applications.


## License

HIDAPI may be used by one of three licenses as outlined in [LICENSE.txt](LICENSE.txt).

## Installing HIDAPI

If you want to build your own application that uses HID devices with HIDAPI,
you need to get HIDAPI development package.

Depending on what your development environment is, HIDAPI likely to be provided
by your package manager.

For instance on Ubuntu, HIDAPI is available via APT:
```sh
sudo apt install libhidapi-dev
```

HIDAPI package name for other systems/package managers may differ.
Check the documentation/package list of your package manager.

## Build from Source

Check [BUILD.md](BUILD.md) for details.
