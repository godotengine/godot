/*
 * darwin backend for libusb 1.0
 * Copyright © 2008-2015 Nathan Hjelm <hjelmn@users.sourceforge.net>
 *
 * This library is free software; you can redistribute it and/or
 * modify it under the terms of the GNU Lesser General Public
 * License as published by the Free Software Foundation; either
 * version 2.1 of the License, or (at your option) any later version.
 *
 * This library is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public
 * License along with this library; if not, write to the Free Software
 * Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA
 */

#if !defined(LIBUSB_DARWIN_H)
#define LIBUSB_DARWIN_H

#include "libusbi.h"

#include <IOKit/IOTypes.h>
#include <IOKit/IOCFBundle.h>
#include <IOKit/usb/IOUSBLib.h>
#include <IOKit/IOCFPlugIn.h>

/* IOUSBInterfaceInferface */
#if defined (kIOUSBInterfaceInterfaceID700) && MAC_OS_X_VERSION_MIN_REQUIRED >= MAC_OS_X_VERSION_10_9

#define usb_interface_t IOUSBInterfaceInterface700
#define InterfaceInterfaceID kIOUSBInterfaceInterfaceID700
#define InterfaceVersion 700

#elif defined (kIOUSBInterfaceInterfaceID550) && MAC_OS_X_VERSION_MIN_REQUIRED >= MAC_OS_X_VERSION_10_9

#define usb_interface_t IOUSBInterfaceInterface550
#define InterfaceInterfaceID kIOUSBInterfaceInterfaceID550
#define InterfaceVersion 550

#elif defined (kIOUSBInterfaceInterfaceID500)

#define usb_interface_t IOUSBInterfaceInterface500
#define InterfaceInterfaceID kIOUSBInterfaceInterfaceID500
#define InterfaceVersion 500

#elif defined (kIOUSBInterfaceInterfaceID300)

#define usb_interface_t IOUSBInterfaceInterface300
#define InterfaceInterfaceID kIOUSBInterfaceInterfaceID300
#define InterfaceVersion 300

#elif defined (kIOUSBInterfaceInterfaceID245)

#define usb_interface_t IOUSBInterfaceInterface245
#define InterfaceInterfaceID kIOUSBInterfaceInterfaceID245
#define InterfaceVersion 245

#elif defined (kIOUSBInterfaceInterfaceID220)

#define usb_interface_t IOUSBInterfaceInterface220
#define InterfaceInterfaceID kIOUSBInterfaceInterfaceID220
#define InterfaceVersion 220

#else

#error "IOUSBFamily is too old. Please upgrade your OS"

#endif

/* IOUSBDeviceInterface */
#if defined (kIOUSBDeviceInterfaceID500) && MAC_OS_X_VERSION_MIN_REQUIRED >= MAC_OS_X_VERSION_10_9

#define usb_device_t    IOUSBDeviceInterface500
#define DeviceInterfaceID kIOUSBDeviceInterfaceID500
#define DeviceVersion 500

#elif defined (kIOUSBDeviceInterfaceID320)

#define usb_device_t    IOUSBDeviceInterface320
#define DeviceInterfaceID kIOUSBDeviceInterfaceID320
#define DeviceVersion 320

#elif defined (kIOUSBDeviceInterfaceID300)

#define usb_device_t    IOUSBDeviceInterface300
#define DeviceInterfaceID kIOUSBDeviceInterfaceID300
#define DeviceVersion 300

#elif defined (kIOUSBDeviceInterfaceID245)

#define usb_device_t    IOUSBDeviceInterface245
#define DeviceInterfaceID kIOUSBDeviceInterfaceID245
#define DeviceVersion 245

#elif defined (kIOUSBDeviceInterfaceID220)
#define usb_device_t    IOUSBDeviceInterface197
#define DeviceInterfaceID kIOUSBDeviceInterfaceID197
#define DeviceVersion 197

#else

#error "IOUSBFamily is too old. Please upgrade your OS"

#endif

#if !defined(IO_OBJECT_NULL)
#define IO_OBJECT_NULL ((io_object_t) 0)
#endif

typedef IOCFPlugInInterface *io_cf_plugin_ref_t;
typedef IONotificationPortRef io_notification_port_t;

/* private structures */
struct darwin_cached_device {
  struct list_head      list;
  IOUSBDeviceDescriptor dev_descriptor;
  UInt32                location;
  UInt64                parent_session;
  UInt64                session;
  UInt16                address;
  char                  sys_path[21];
  usb_device_t        **device;
  int                   open_count;
  UInt8                 first_config, active_config, port;  
  int                   can_enumerate;
  int                   refcount;
};

struct darwin_device_priv {
  struct darwin_cached_device *dev;
};

struct darwin_device_handle_priv {
  int                  is_open;
  CFRunLoopSourceRef   cfSource;

  struct darwin_interface {
    usb_interface_t    **interface;
    uint8_t              num_endpoints;
    CFRunLoopSourceRef   cfSource;
    uint64_t             frames[256];
    uint8_t              endpoint_addrs[USB_MAXENDPOINTS];
  } interfaces[USB_MAXINTERFACES];
};

struct darwin_transfer_priv {
  /* Isoc */
  IOUSBIsocFrame *isoc_framelist;
  int num_iso_packets;

  /* Control */
  IOUSBDevRequestTO req;

  /* Bulk */

  /* Completion status */
  IOReturn result;
  UInt32 size;
};

#endif
