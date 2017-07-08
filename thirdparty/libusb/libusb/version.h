/* This file is parsed by m4 and windres and RC.EXE so please keep it simple. */
#include "version_nano.h"
#ifndef LIBUSB_MAJOR
#define LIBUSB_MAJOR 1
#endif
#ifndef LIBUSB_MINOR
#define LIBUSB_MINOR 0
#endif
#ifndef LIBUSB_MICRO
#define LIBUSB_MICRO 21
#endif
#ifndef LIBUSB_NANO
#define LIBUSB_NANO 0
#endif
/* LIBUSB_RC is the release candidate suffix. Should normally be empty. */
#ifndef LIBUSB_RC
#define LIBUSB_RC ""
#endif
