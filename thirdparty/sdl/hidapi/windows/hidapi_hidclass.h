/*******************************************************
 HIDAPI - Multi-Platform library for
 communication with HID devices.

 libusb/hidapi Team

 Copyright 2022, All Rights Reserved.

 At the discretion of the user of this library,
 this software may be licensed under the terms of the
 GNU General Public License v3, a BSD-Style license, or the
 original HIDAPI license as outlined in the LICENSE.txt,
 LICENSE-gpl3.txt, LICENSE-bsd.txt, and LICENSE-orig.txt
 files located at the root of the source distribution.
 These files may also be found in the public source
 code repository located at:
        https://github.com/libusb/hidapi .
********************************************************/

#ifndef HIDAPI_HIDCLASS_H
#define HIDAPI_HIDCLASS_H

#ifdef HIDAPI_USE_DDK

#include <hidclass.h>

#else

#include <winioctl.h>

/* This part of the header mimics hidclass.h,
    but only what is used by HIDAPI */

#define HID_OUT_CTL_CODE(id) CTL_CODE(FILE_DEVICE_KEYBOARD, (id), METHOD_OUT_DIRECT, FILE_ANY_ACCESS)
#define IOCTL_HID_GET_FEATURE HID_OUT_CTL_CODE(100)
#define IOCTL_HID_GET_INPUT_REPORT HID_OUT_CTL_CODE(104)

#endif

#endif /* HIDAPI_HIDCLASS_H */
