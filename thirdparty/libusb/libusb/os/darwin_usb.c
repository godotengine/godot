/* -*- Mode: C; indent-tabs-mode:nil -*- */
/*
 * darwin backend for libusb 1.0
 * Copyright © 2008-2017 Nathan Hjelm <hjelmn@users.sourceforge.net>
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

#include "config.h"
#include <time.h>
#include <ctype.h>
#include <errno.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/types.h>
#include <unistd.h>
#include <fcntl.h>
#include <sys/sysctl.h>

#include <mach/clock.h>
#include <mach/clock_types.h>
#include <mach/mach_host.h>
#include <mach/mach_port.h>

#include <AvailabilityMacros.h>
#if MAC_OS_X_VERSION_MIN_REQUIRED >= 1060 && MAC_OS_X_VERSION_MIN_REQUIRED < 101200
  #include <objc/objc-auto.h>
#endif

#if MAC_OS_X_VERSION_MIN_REQUIRED >= 101200
/* Apple deprecated the darwin atomics in 10.12 in favor of C11 atomics */
#include <stdatomic.h>
#define libusb_darwin_atomic_fetch_add(x, y) atomic_fetch_add(x, y)
#define OSX_USE_CLOCK_GETTIME 1

_Atomic int32_t initCount = ATOMIC_VAR_INIT(0);
#else
/* use darwin atomics if the target is older than 10.12 */
#include <libkern/OSAtomic.h>

/* OSAtomicAdd32Barrier returns the new value */
#define libusb_darwin_atomic_fetch_add(x, y) (OSAtomicAdd32Barrier(y, x) - y)

static volatile int32_t initCount = 0;

#define OSX_USE_CLOCK_GETTIME 0
#endif

#include "darwin_usb.h"

/* async event thread */
static pthread_mutex_t libusb_darwin_at_mutex = PTHREAD_MUTEX_INITIALIZER;
static pthread_cond_t  libusb_darwin_at_cond = PTHREAD_COND_INITIALIZER;

static pthread_once_t darwin_init_once = PTHREAD_ONCE_INIT;

#if !OSX_USE_CLOCK_GETTIME
static clock_serv_t clock_realtime;
static clock_serv_t clock_monotonic;
#endif

static CFRunLoopRef libusb_darwin_acfl = NULL; /* event cf loop */
static CFRunLoopSourceRef libusb_darwin_acfls = NULL; /* shutdown signal for event cf loop */

static usbi_mutex_t darwin_cached_devices_lock = PTHREAD_MUTEX_INITIALIZER;
static struct list_head darwin_cached_devices = {&darwin_cached_devices, &darwin_cached_devices};
static char *darwin_device_class = kIOUSBDeviceClassName;

#define DARWIN_CACHED_DEVICE(a) ((struct darwin_cached_device *) (((struct darwin_device_priv *)((a)->os_priv))->dev))

/* async event thread */
static pthread_t libusb_darwin_at;

static int darwin_get_config_descriptor(struct libusb_device *dev, uint8_t config_index, unsigned char *buffer, size_t len, int *host_endian);
static int darwin_claim_interface(struct libusb_device_handle *dev_handle, int iface);
static int darwin_release_interface(struct libusb_device_handle *dev_handle, int iface);
static int darwin_reset_device(struct libusb_device_handle *dev_handle);
static void darwin_async_io_callback (void *refcon, IOReturn result, void *arg0);

static int darwin_scan_devices(struct libusb_context *ctx);
static int process_new_device (struct libusb_context *ctx, io_service_t service);

#if defined(ENABLE_LOGGING)
static const char *darwin_error_str (int result) {
  static char string_buffer[50];
  switch (result) {
  case kIOReturnSuccess:
    return "no error";
  case kIOReturnNotOpen:
    return "device not opened for exclusive access";
  case kIOReturnNoDevice:
    return "no connection to an IOService";
  case kIOUSBNoAsyncPortErr:
    return "no async port has been opened for interface";
  case kIOReturnExclusiveAccess:
    return "another process has device opened for exclusive access";
  case kIOUSBPipeStalled:
    return "pipe is stalled";
  case kIOReturnError:
    return "could not establish a connection to the Darwin kernel";
  case kIOUSBTransactionTimeout:
    return "transaction timed out";
  case kIOReturnBadArgument:
    return "invalid argument";
  case kIOReturnAborted:
    return "transaction aborted";
  case kIOReturnNotResponding:
    return "device not responding";
  case kIOReturnOverrun:
    return "data overrun";
  case kIOReturnCannotWire:
    return "physical memory can not be wired down";
  case kIOReturnNoResources:
    return "out of resources";
  case kIOUSBHighSpeedSplitError:
    return "high speed split error";
  default:
    snprintf(string_buffer, sizeof(string_buffer), "unknown error (0x%x)", result);
    return string_buffer;
  }
}
#endif

static int darwin_to_libusb (int result) {
  switch (result) {
  case kIOReturnUnderrun:
  case kIOReturnSuccess:
    return LIBUSB_SUCCESS;
  case kIOReturnNotOpen:
  case kIOReturnNoDevice:
    return LIBUSB_ERROR_NO_DEVICE;
  case kIOReturnExclusiveAccess:
    return LIBUSB_ERROR_ACCESS;
  case kIOUSBPipeStalled:
    return LIBUSB_ERROR_PIPE;
  case kIOReturnBadArgument:
    return LIBUSB_ERROR_INVALID_PARAM;
  case kIOUSBTransactionTimeout:
    return LIBUSB_ERROR_TIMEOUT;
  case kIOReturnNotResponding:
  case kIOReturnAborted:
  case kIOReturnError:
  case kIOUSBNoAsyncPortErr:
  default:
    return LIBUSB_ERROR_OTHER;
  }
}

/* this function must be called with the darwin_cached_devices_lock held */
static void darwin_deref_cached_device(struct darwin_cached_device *cached_dev) {
  cached_dev->refcount--;
  /* free the device and remove it from the cache */
  if (0 == cached_dev->refcount) {
    list_del(&cached_dev->list);

    (*(cached_dev->device))->Release(cached_dev->device);
    free (cached_dev);
  }
}

static void darwin_ref_cached_device(struct darwin_cached_device *cached_dev) {
  cached_dev->refcount++;
}

static int ep_to_pipeRef(struct libusb_device_handle *dev_handle, uint8_t ep, uint8_t *pipep, uint8_t *ifcp, struct darwin_interface **interface_out) {
  struct darwin_device_handle_priv *priv = (struct darwin_device_handle_priv *)dev_handle->os_priv;

  /* current interface */
  struct darwin_interface *cInterface;

  int8_t i, iface;

  usbi_dbg ("converting ep address 0x%02x to pipeRef and interface", ep);

  for (iface = 0 ; iface < USB_MAXINTERFACES ; iface++) {
    cInterface = &priv->interfaces[iface];

    if (dev_handle->claimed_interfaces & (1 << iface)) {
      for (i = 0 ; i < cInterface->num_endpoints ; i++) {
        if (cInterface->endpoint_addrs[i] == ep) {
          *pipep = i + 1;

          if (ifcp)
            *ifcp = iface;

          if (interface_out)
            *interface_out = cInterface;

          usbi_dbg ("pipe %d on interface %d matches", *pipep, iface);
          return 0;
        }
      }
    }
  }

  /* No pipe found with the correct endpoint address */
  usbi_warn (HANDLE_CTX(dev_handle), "no pipeRef found with endpoint address 0x%02x.", ep);

  return LIBUSB_ERROR_NOT_FOUND;
}

static int usb_setup_device_iterator (io_iterator_t *deviceIterator, UInt32 location) {
  CFMutableDictionaryRef matchingDict = IOServiceMatching(darwin_device_class);

  if (!matchingDict)
    return kIOReturnError;

  if (location) {
    CFMutableDictionaryRef propertyMatchDict = CFDictionaryCreateMutable(kCFAllocatorDefault, 0,
                                                                         &kCFTypeDictionaryKeyCallBacks,
                                                                         &kCFTypeDictionaryValueCallBacks);

    if (propertyMatchDict) {
      /* there are no unsigned CFNumber types so treat the value as signed. the os seems to do this
         internally (CFNumberType of locationID is 3) */
      CFTypeRef locationCF = CFNumberCreate (NULL, kCFNumberSInt32Type, &location);

      CFDictionarySetValue (propertyMatchDict, CFSTR(kUSBDevicePropertyLocationID), locationCF);
      /* release our reference to the CFNumber (CFDictionarySetValue retains it) */
      CFRelease (locationCF);

      CFDictionarySetValue (matchingDict, CFSTR(kIOPropertyMatchKey), propertyMatchDict);
      /* release out reference to the CFMutableDictionaryRef (CFDictionarySetValue retains it) */
      CFRelease (propertyMatchDict);
    }
    /* else we can still proceed as long as the caller accounts for the possibility of other devices in the iterator */
  }

  return IOServiceGetMatchingServices(kIOMasterPortDefault, matchingDict, deviceIterator);
}

/* Returns 1 on success, 0 on failure. */
static int get_ioregistry_value_number (io_service_t service, CFStringRef property, CFNumberType type, void *p) {
  CFTypeRef cfNumber = IORegistryEntryCreateCFProperty (service, property, kCFAllocatorDefault, 0);
  int ret = 0;

  if (cfNumber) {
    if (CFGetTypeID(cfNumber) == CFNumberGetTypeID()) {
      ret = CFNumberGetValue(cfNumber, type, p);
    }

    CFRelease (cfNumber);
  }

  return ret;
}

static int get_ioregistry_value_data (io_service_t service, CFStringRef property, ssize_t size, void *p) {
  CFTypeRef cfData = IORegistryEntryCreateCFProperty (service, property, kCFAllocatorDefault, 0);
  int ret = 0;

  if (cfData) {
    if (CFGetTypeID (cfData) == CFDataGetTypeID ()) {
      CFIndex length = CFDataGetLength (cfData);
      if (length < size) {
        size = length;
      }

      CFDataGetBytes (cfData, CFRangeMake(0, size), p);
      ret = 1;
    }

    CFRelease (cfData);
  }

  return ret;
}

static usb_device_t **darwin_device_from_service (io_service_t service)
{
  io_cf_plugin_ref_t *plugInInterface = NULL;
  usb_device_t **device;
  kern_return_t result;
  SInt32 score;

  result = IOCreatePlugInInterfaceForService(service, kIOUSBDeviceUserClientTypeID,
                                             kIOCFPlugInInterfaceID, &plugInInterface,
                                             &score);

  if (kIOReturnSuccess != result || !plugInInterface) {
    usbi_dbg ("could not set up plugin for service: %s", darwin_error_str (result));
    return NULL;
  }

  (void)(*plugInInterface)->QueryInterface(plugInInterface, CFUUIDGetUUIDBytes(DeviceInterfaceID),
                                           (LPVOID)&device);
  /* Use release instead of IODestroyPlugInInterface to avoid stopping IOServices associated with this device */
  (*plugInInterface)->Release (plugInInterface);

  return device;
}

static void darwin_devices_attached (void *ptr, io_iterator_t add_devices) {
  struct libusb_context *ctx;
  io_service_t service;

  usbi_mutex_lock(&active_contexts_lock);

  while ((service = IOIteratorNext(add_devices))) {
    /* add this device to each active context's device list */
    list_for_each_entry(ctx, &active_contexts_list, list, struct libusb_context) {
      process_new_device (ctx, service);;
    }

    IOObjectRelease(service);
  }

  usbi_mutex_unlock(&active_contexts_lock);
}

static void darwin_devices_detached (void *ptr, io_iterator_t rem_devices) {
  struct libusb_device *dev = NULL;
  struct libusb_context *ctx;
  struct darwin_cached_device *old_device;

  io_service_t device;
  UInt64 session;
  int ret;

  usbi_mutex_lock(&active_contexts_lock);

  while ((device = IOIteratorNext (rem_devices)) != 0) {
    /* get the location from the i/o registry */
    ret = get_ioregistry_value_number (device, CFSTR("sessionID"), kCFNumberSInt64Type, &session);
    IOObjectRelease (device);
    if (!ret)
      continue;

    /* we need to match darwin_ref_cached_device call made in darwin_get_cached_device function
       otherwise no cached device will ever get freed */
    usbi_mutex_lock(&darwin_cached_devices_lock);
    list_for_each_entry(old_device, &darwin_cached_devices, list, struct darwin_cached_device) {
      if (old_device->session == session) {
        darwin_deref_cached_device (old_device);
        break;
      }
    }
    usbi_mutex_unlock(&darwin_cached_devices_lock);

    list_for_each_entry(ctx, &active_contexts_list, list, struct libusb_context) {
      usbi_dbg ("notifying context %p of device disconnect", ctx);

      dev = usbi_get_device_by_session_id(ctx, (unsigned long) session);
      if (dev) {
        /* signal the core that this device has been disconnected. the core will tear down this device
           when the reference count reaches 0 */
        usbi_disconnect_device(dev);
        libusb_unref_device(dev);
      }
    }
  }

  usbi_mutex_unlock(&active_contexts_lock);
}

static void darwin_hotplug_poll (void)
{
  /* not sure if 5 seconds will be too long/short but it should work ok */
  mach_timespec_t timeout = {.tv_sec = 5, .tv_nsec = 0};

  /* since a kernel thread may nodify the IOInterators used for
   * hotplug notidication we can't just clear the iterators.
   * instead just wait until all IOService providers are quiet */
  (void) IOKitWaitQuiet (kIOMasterPortDefault, &timeout);
}

static void darwin_clear_iterator (io_iterator_t iter) {
  io_service_t device;

  while ((device = IOIteratorNext (iter)) != 0)
    IOObjectRelease (device);
}

static void *darwin_event_thread_main (void *arg0) {
  IOReturn kresult;
  struct libusb_context *ctx = (struct libusb_context *)arg0;
  CFRunLoopRef runloop;

#if MAC_OS_X_VERSION_MIN_REQUIRED >= 1060
  /* Set this thread's name, so it can be seen in the debugger
     and crash reports. */
  pthread_setname_np ("org.libusb.device-hotplug");
#endif

#if MAC_OS_X_VERSION_MIN_REQUIRED >= 1060 && MAC_OS_X_VERSION_MIN_REQUIRED < 101200
  /* Tell the Objective-C garbage collector about this thread.
     This is required because, unlike NSThreads, pthreads are
     not automatically registered. Although we don't use
     Objective-C, we use CoreFoundation, which does.
     Garbage collection support was entirely removed in 10.12,
     so don't bother there. */
  objc_registerThreadWithCollector();
#endif

  /* hotplug (device arrival/removal) sources */
  CFRunLoopSourceContext libusb_shutdown_cfsourcectx;
  CFRunLoopSourceRef     libusb_notification_cfsource;
  io_notification_port_t libusb_notification_port;
  io_iterator_t          libusb_rem_device_iterator;
  io_iterator_t          libusb_add_device_iterator;

  usbi_dbg ("creating hotplug event source");

  runloop = CFRunLoopGetCurrent ();
  CFRetain (runloop);

  /* add the shutdown cfsource to the run loop */
  memset(&libusb_shutdown_cfsourcectx, 0, sizeof(libusb_shutdown_cfsourcectx));
  libusb_shutdown_cfsourcectx.info = runloop;
  libusb_shutdown_cfsourcectx.perform = (void (*)(void *))CFRunLoopStop;
  libusb_darwin_acfls = CFRunLoopSourceCreate(NULL, 0, &libusb_shutdown_cfsourcectx);
  CFRunLoopAddSource(runloop, libusb_darwin_acfls, kCFRunLoopDefaultMode);

  /* add the notification port to the run loop */
  libusb_notification_port     = IONotificationPortCreate (kIOMasterPortDefault);
  libusb_notification_cfsource = IONotificationPortGetRunLoopSource (libusb_notification_port);
  CFRunLoopAddSource(runloop, libusb_notification_cfsource, kCFRunLoopDefaultMode);

  /* create notifications for removed devices */
  kresult = IOServiceAddMatchingNotification (libusb_notification_port, kIOTerminatedNotification,
                                              IOServiceMatching(darwin_device_class),
                                              darwin_devices_detached,
                                              ctx, &libusb_rem_device_iterator);

  if (kresult != kIOReturnSuccess) {
    usbi_err (ctx, "could not add hotplug event source: %s", darwin_error_str (kresult));

    pthread_exit (NULL);
  }

  /* create notifications for attached devices */
  kresult = IOServiceAddMatchingNotification(libusb_notification_port, kIOFirstMatchNotification,
                                              IOServiceMatching(darwin_device_class),
                                              darwin_devices_attached,
                                              ctx, &libusb_add_device_iterator);

  if (kresult != kIOReturnSuccess) {
    usbi_err (ctx, "could not add hotplug event source: %s", darwin_error_str (kresult));

    pthread_exit (NULL);
  }

  /* arm notifiers */
  darwin_clear_iterator (libusb_rem_device_iterator);
  darwin_clear_iterator (libusb_add_device_iterator);

  usbi_dbg ("darwin event thread ready to receive events");

  /* signal the main thread that the hotplug runloop has been created. */
  pthread_mutex_lock (&libusb_darwin_at_mutex);
  libusb_darwin_acfl = runloop;
  pthread_cond_signal (&libusb_darwin_at_cond);
  pthread_mutex_unlock (&libusb_darwin_at_mutex);

  /* run the runloop */
  CFRunLoopRun();

  usbi_dbg ("darwin event thread exiting");

  /* remove the notification cfsource */
  CFRunLoopRemoveSource(runloop, libusb_notification_cfsource, kCFRunLoopDefaultMode);

  /* remove the shutdown cfsource */
  CFRunLoopRemoveSource(runloop, libusb_darwin_acfls, kCFRunLoopDefaultMode);

  /* delete notification port */
  IONotificationPortDestroy (libusb_notification_port);

  /* delete iterators */
  IOObjectRelease (libusb_rem_device_iterator);
  IOObjectRelease (libusb_add_device_iterator);

  CFRelease (libusb_darwin_acfls);
  CFRelease (runloop);

  libusb_darwin_acfls = NULL;
  libusb_darwin_acfl = NULL;

  pthread_exit (NULL);
}

/* cleanup function to destroy cached devices */
static void __attribute__((destructor)) _darwin_finalize(void) {
  struct darwin_cached_device *dev, *next;

  usbi_mutex_lock(&darwin_cached_devices_lock);
  list_for_each_entry_safe(dev, next, &darwin_cached_devices, list, struct darwin_cached_device) {
    darwin_deref_cached_device(dev);
  }
  usbi_mutex_unlock(&darwin_cached_devices_lock);
}

static void darwin_check_version (void) {
  /* adjust for changes in the USB stack in xnu 15 */
  int sysctl_args[] = {CTL_KERN, KERN_OSRELEASE};
  long version;
  char version_string[256] = {'\0',};
  size_t length = 256;

  sysctl(sysctl_args, 2, version_string, &length, NULL, 0);

  errno = 0;
  version = strtol (version_string, NULL, 10);
  if (0 == errno && version >= 15) {
    darwin_device_class = "IOUSBHostDevice";
  }
}

static int darwin_init(struct libusb_context *ctx) {
  int rc;

  rc = pthread_once (&darwin_init_once, darwin_check_version);
  if (rc) {
    return LIBUSB_ERROR_OTHER;
  }

  rc = darwin_scan_devices (ctx);
  if (LIBUSB_SUCCESS != rc) {
    return rc;
  }

  if (libusb_darwin_atomic_fetch_add (&initCount, 1) == 0) {
#if !OSX_USE_CLOCK_GETTIME
    /* create the clocks that will be used if clock_gettime() is not available */
    host_name_port_t host_self;

    host_self = mach_host_self();
    host_get_clock_service(host_self, CALENDAR_CLOCK, &clock_realtime);
    host_get_clock_service(host_self, SYSTEM_CLOCK, &clock_monotonic);
    mach_port_deallocate(mach_task_self(), host_self);
#endif

    pthread_create (&libusb_darwin_at, NULL, darwin_event_thread_main, ctx);

    pthread_mutex_lock (&libusb_darwin_at_mutex);
    while (!libusb_darwin_acfl)
      pthread_cond_wait (&libusb_darwin_at_cond, &libusb_darwin_at_mutex);
    pthread_mutex_unlock (&libusb_darwin_at_mutex);
  }

  return rc;
}

static void darwin_exit (void) {
  if (libusb_darwin_atomic_fetch_add (&initCount, -1) == 1) {
#if !OSX_USE_CLOCK_GETTIME
    mach_port_deallocate(mach_task_self(), clock_realtime);
    mach_port_deallocate(mach_task_self(), clock_monotonic);
#endif

    /* stop the event runloop and wait for the thread to terminate. */
    CFRunLoopSourceSignal(libusb_darwin_acfls);
    CFRunLoopWakeUp (libusb_darwin_acfl);
    pthread_join (libusb_darwin_at, NULL);
  }
}

static int darwin_get_device_descriptor(struct libusb_device *dev, unsigned char *buffer, int *host_endian) {
  struct darwin_cached_device *priv = DARWIN_CACHED_DEVICE(dev);

  /* return cached copy */
  memmove (buffer, &(priv->dev_descriptor), DEVICE_DESC_LENGTH);

  *host_endian = 0;

  return 0;
}

static int get_configuration_index (struct libusb_device *dev, int config_value) {
  struct darwin_cached_device *priv = DARWIN_CACHED_DEVICE(dev);
  UInt8 i, numConfig;
  IOUSBConfigurationDescriptorPtr desc;
  IOReturn kresult;

  /* is there a simpler way to determine the index? */
  kresult = (*(priv->device))->GetNumberOfConfigurations (priv->device, &numConfig);
  if (kresult != kIOReturnSuccess)
    return darwin_to_libusb (kresult);

  for (i = 0 ; i < numConfig ; i++) {
    (*(priv->device))->GetConfigurationDescriptorPtr (priv->device, i, &desc);

    if (desc->bConfigurationValue == config_value)
      return i;
  }

  /* configuration not found */
  return LIBUSB_ERROR_NOT_FOUND;
}

static int darwin_get_active_config_descriptor(struct libusb_device *dev, unsigned char *buffer, size_t len, int *host_endian) {
  struct darwin_cached_device *priv = DARWIN_CACHED_DEVICE(dev);
  int config_index;

  if (0 == priv->active_config)
    return LIBUSB_ERROR_NOT_FOUND;

  config_index = get_configuration_index (dev, priv->active_config);
  if (config_index < 0)
    return config_index;

  return darwin_get_config_descriptor (dev, config_index, buffer, len, host_endian);
}

static int darwin_get_config_descriptor(struct libusb_device *dev, uint8_t config_index, unsigned char *buffer, size_t len, int *host_endian) {
  struct darwin_cached_device *priv = DARWIN_CACHED_DEVICE(dev);
  IOUSBConfigurationDescriptorPtr desc;
  IOReturn kresult;
  int ret;

  if (!priv || !priv->device)
    return LIBUSB_ERROR_OTHER;

  kresult = (*priv->device)->GetConfigurationDescriptorPtr (priv->device, config_index, &desc);
  if (kresult == kIOReturnSuccess) {
    /* copy descriptor */
    if (libusb_le16_to_cpu(desc->wTotalLength) < len)
      len = libusb_le16_to_cpu(desc->wTotalLength);

    memmove (buffer, desc, len);

    /* GetConfigurationDescriptorPtr returns the descriptor in USB bus order */
    *host_endian = 0;
  }

  ret = darwin_to_libusb (kresult);
  if (ret != LIBUSB_SUCCESS)
    return ret;

  return (int) len;
}

/* check whether the os has configured the device */
static int darwin_check_configuration (struct libusb_context *ctx, struct darwin_cached_device *dev) {
  usb_device_t **darwin_device = dev->device;

  IOUSBConfigurationDescriptorPtr configDesc;
  IOUSBFindInterfaceRequest request;
  kern_return_t             kresult;
  io_iterator_t             interface_iterator;
  io_service_t              firstInterface;

  if (dev->dev_descriptor.bNumConfigurations < 1) {
    usbi_err (ctx, "device has no configurations");
    return LIBUSB_ERROR_OTHER; /* no configurations at this speed so we can't use it */
  }

  /* checking the configuration of a root hub simulation takes ~1 s in 10.11. the device is
     not usable anyway */
  if (0x05ac == dev->dev_descriptor.idVendor && 0x8005 == dev->dev_descriptor.idProduct) {
    usbi_dbg ("ignoring configuration on root hub simulation");
    dev->active_config = 0;
    return 0;
  }

  /* find the first configuration */
  kresult = (*darwin_device)->GetConfigurationDescriptorPtr (darwin_device, 0, &configDesc);
  dev->first_config = (kIOReturnSuccess == kresult) ? configDesc->bConfigurationValue : 1;

  /* check if the device is already configured. there is probably a better way than iterating over the
     to accomplish this (the trick is we need to avoid a call to GetConfigurations since buggy devices
     might lock up on the device request) */

  /* Setup the Interface Request */
  request.bInterfaceClass    = kIOUSBFindInterfaceDontCare;
  request.bInterfaceSubClass = kIOUSBFindInterfaceDontCare;
  request.bInterfaceProtocol = kIOUSBFindInterfaceDontCare;
  request.bAlternateSetting  = kIOUSBFindInterfaceDontCare;

  kresult = (*(darwin_device))->CreateInterfaceIterator(darwin_device, &request, &interface_iterator);
  if (kresult)
    return darwin_to_libusb (kresult);

  /* iterate once */
  firstInterface = IOIteratorNext(interface_iterator);

  /* done with the interface iterator */
  IOObjectRelease(interface_iterator);

  if (firstInterface) {
    IOObjectRelease (firstInterface);

    /* device is configured */
    if (dev->dev_descriptor.bNumConfigurations == 1)
      /* to avoid problems with some devices get the configurations value from the configuration descriptor */
      dev->active_config = dev->first_config;
    else
      /* devices with more than one configuration should work with GetConfiguration */
      (*darwin_device)->GetConfiguration (darwin_device, &dev->active_config);
  } else
    /* not configured */
    dev->active_config = 0;
  
  usbi_dbg ("active config: %u, first config: %u", dev->active_config, dev->first_config);

  return 0;
}

static int darwin_request_descriptor (usb_device_t **device, UInt8 desc, UInt8 desc_index, void *buffer, size_t buffer_size) {
  IOUSBDevRequestTO req;

  memset (buffer, 0, buffer_size);

  /* Set up request for descriptor/ */
  req.bmRequestType = USBmakebmRequestType(kUSBIn, kUSBStandard, kUSBDevice);
  req.bRequest      = kUSBRqGetDescriptor;
  req.wValue        = desc << 8;
  req.wIndex        = desc_index;
  req.wLength       = buffer_size;
  req.pData         = buffer;
  req.noDataTimeout = 20;
  req.completionTimeout = 100;

  return (*device)->DeviceRequestTO (device, &req);
}

static int darwin_cache_device_descriptor (struct libusb_context *ctx, struct darwin_cached_device *dev) {
  usb_device_t **device = dev->device;
  int retries = 1, delay = 30000;
  int unsuspended = 0, try_unsuspend = 1, try_reconfigure = 1;
  int is_open = 0;
  int ret = 0, ret2;
  UInt8 bDeviceClass;
  UInt16 idProduct, idVendor;

  dev->can_enumerate = 0;

  (*device)->GetDeviceClass (device, &bDeviceClass);
  (*device)->GetDeviceProduct (device, &idProduct);
  (*device)->GetDeviceVendor (device, &idVendor);

  /* According to Apple's documentation the device must be open for DeviceRequest but we may not be able to open some
   * devices and Apple's USB Prober doesn't bother to open the device before issuing a descriptor request.  Still,
   * to follow the spec as closely as possible, try opening the device */
  is_open = ((*device)->USBDeviceOpenSeize(device) == kIOReturnSuccess);

  do {
    /**** retrieve device descriptor ****/
    ret = darwin_request_descriptor (device, kUSBDeviceDesc, 0, &dev->dev_descriptor, sizeof(dev->dev_descriptor));

    if (kIOReturnOverrun == ret && kUSBDeviceDesc == dev->dev_descriptor.bDescriptorType)
      /* received an overrun error but we still received a device descriptor */
      ret = kIOReturnSuccess;

    if (kIOUSBVendorIDAppleComputer == idVendor) {
      /* NTH: don't bother retrying or unsuspending Apple devices */
      break;
    }

    if (kIOReturnSuccess == ret && (0 == dev->dev_descriptor.bNumConfigurations ||
                                    0 == dev->dev_descriptor.bcdUSB)) {
      /* work around for incorrectly configured devices */
      if (try_reconfigure && is_open) {
        usbi_dbg("descriptor appears to be invalid. resetting configuration before trying again...");

        /* set the first configuration */
        (*device)->SetConfiguration(device, 1);

        /* don't try to reconfigure again */
        try_reconfigure = 0;
      }

      ret = kIOUSBPipeStalled;
    }

    if (kIOReturnSuccess != ret && is_open && try_unsuspend) {
      /* device may be suspended. unsuspend it and try again */
#if DeviceVersion >= 320
      UInt32 info = 0;

      /* IOUSBFamily 320+ provides a way to detect device suspension but earlier versions do not */
      (void)(*device)->GetUSBDeviceInformation (device, &info);

      /* note that the device was suspended */
      if (info & (1 << kUSBInformationDeviceIsSuspendedBit) || 0 == info)
        try_unsuspend = 1;
#endif

      if (try_unsuspend) {
        /* try to unsuspend the device */
        ret2 = (*device)->USBDeviceSuspend (device, 0);
        if (kIOReturnSuccess != ret2) {
          /* prevent log spew from poorly behaving devices.  this indicates the
             os actually had trouble communicating with the device */
          usbi_dbg("could not retrieve device descriptor. failed to unsuspend: %s",darwin_error_str(ret2));
        } else
          unsuspended = 1;

        try_unsuspend = 0;
      }
    }

    if (kIOReturnSuccess != ret) {
      usbi_dbg("kernel responded with code: 0x%08x. sleeping for %d ms before trying again", ret, delay/1000);
      /* sleep for a little while before trying again */
      nanosleep(&(struct timespec){delay / 1000000, (delay * 1000) % 1000000000UL}, NULL);
    }
  } while (kIOReturnSuccess != ret && retries--);

  if (unsuspended)
    /* resuspend the device */
    (void)(*device)->USBDeviceSuspend (device, 1);

  if (is_open)
    (void) (*device)->USBDeviceClose (device);

  if (ret != kIOReturnSuccess) {
    /* a debug message was already printed out for this error */
    if (LIBUSB_CLASS_HUB == bDeviceClass)
      usbi_dbg ("could not retrieve device descriptor %.4x:%.4x: %s (%x). skipping device",
                idVendor, idProduct, darwin_error_str (ret), ret);
    else
      usbi_warn (ctx, "could not retrieve device descriptor %.4x:%.4x: %s (%x). skipping device",
                 idVendor, idProduct, darwin_error_str (ret), ret);
    return darwin_to_libusb (ret);
  }

  /* catch buggy hubs (which appear to be virtual). Apple's own USB prober has problems with these devices. */
  if (libusb_le16_to_cpu (dev->dev_descriptor.idProduct) != idProduct) {
    /* not a valid device */
    usbi_warn (ctx, "idProduct from iokit (%04x) does not match idProduct in descriptor (%04x). skipping device",
               idProduct, libusb_le16_to_cpu (dev->dev_descriptor.idProduct));
    return LIBUSB_ERROR_NO_DEVICE;
  }

  usbi_dbg ("cached device descriptor:");
  usbi_dbg ("  bDescriptorType:    0x%02x", dev->dev_descriptor.bDescriptorType);
  usbi_dbg ("  bcdUSB:             0x%04x", dev->dev_descriptor.bcdUSB);
  usbi_dbg ("  bDeviceClass:       0x%02x", dev->dev_descriptor.bDeviceClass);
  usbi_dbg ("  bDeviceSubClass:    0x%02x", dev->dev_descriptor.bDeviceSubClass);
  usbi_dbg ("  bDeviceProtocol:    0x%02x", dev->dev_descriptor.bDeviceProtocol);
  usbi_dbg ("  bMaxPacketSize0:    0x%02x", dev->dev_descriptor.bMaxPacketSize0);
  usbi_dbg ("  idVendor:           0x%04x", dev->dev_descriptor.idVendor);
  usbi_dbg ("  idProduct:          0x%04x", dev->dev_descriptor.idProduct);
  usbi_dbg ("  bcdDevice:          0x%04x", dev->dev_descriptor.bcdDevice);
  usbi_dbg ("  iManufacturer:      0x%02x", dev->dev_descriptor.iManufacturer);
  usbi_dbg ("  iProduct:           0x%02x", dev->dev_descriptor.iProduct);
  usbi_dbg ("  iSerialNumber:      0x%02x", dev->dev_descriptor.iSerialNumber);
  usbi_dbg ("  bNumConfigurations: 0x%02x", dev->dev_descriptor.bNumConfigurations);

  dev->can_enumerate = 1;

  return LIBUSB_SUCCESS;
}

static int get_device_port (io_service_t service, UInt8 *port) {
  kern_return_t result;
  io_service_t parent;
  int ret = 0;

  if (get_ioregistry_value_number (service, CFSTR("PortNum"), kCFNumberSInt8Type, port)) {
    return 1;
  }

  result = IORegistryEntryGetParentEntry (service, kIOServicePlane, &parent);
  if (kIOReturnSuccess == result) {
    ret = get_ioregistry_value_data (parent, CFSTR("port"), 1, port);
    IOObjectRelease (parent);
  }

  return ret;
}

static int get_device_parent_sessionID(io_service_t service, UInt64 *parent_sessionID) {
  kern_return_t result;
  io_service_t parent;

  /* Walk up the tree in the IOService plane until we find a parent that has a sessionID */
  parent = service;
  while((result = IORegistryEntryGetParentEntry (parent, kIOServicePlane, &parent)) == kIOReturnSuccess) {
    if (get_ioregistry_value_number (parent, CFSTR("sessionID"), kCFNumberSInt64Type, parent_sessionID)) {
        /* Success */
        return 1;
    }
  }

  /* We ran out of parents */
  return 0;
}

static int darwin_get_cached_device(struct libusb_context *ctx, io_service_t service,
                                    struct darwin_cached_device **cached_out) {
  struct darwin_cached_device *new_device;
  UInt64 sessionID = 0, parent_sessionID = 0;
  int ret = LIBUSB_SUCCESS;
  usb_device_t **device;
  UInt8 port = 0;

  /* get some info from the io registry */
  (void) get_ioregistry_value_number (service, CFSTR("sessionID"), kCFNumberSInt64Type, &sessionID);
  if (!get_device_port (service, &port)) {
    usbi_dbg("could not get connected port number");
  }

  usbi_dbg("finding cached device for sessionID 0x%" PRIx64, sessionID);

  if (get_device_parent_sessionID(service, &parent_sessionID)) {
    usbi_dbg("parent sessionID: 0x%" PRIx64, parent_sessionID);
  }

  usbi_mutex_lock(&darwin_cached_devices_lock);
  do {
    *cached_out = NULL;

    list_for_each_entry(new_device, &darwin_cached_devices, list, struct darwin_cached_device) {
      usbi_dbg("matching sessionID 0x%" PRIx64 " against cached device with sessionID 0x%" PRIx64, sessionID, new_device->session);
      if (new_device->session == sessionID) {
        usbi_dbg("using cached device for device");
        *cached_out = new_device;
        break;
      }
    }

    if (*cached_out)
      break;

    usbi_dbg("caching new device with sessionID 0x%" PRIx64, sessionID);

    device = darwin_device_from_service (service);
    if (!device) {
      ret = LIBUSB_ERROR_NO_DEVICE;
      break;
    }

    new_device = calloc (1, sizeof (*new_device));
    if (!new_device) {
      ret = LIBUSB_ERROR_NO_MEM;
      break;
    }

    /* add this device to the cached device list */
    list_add(&new_device->list, &darwin_cached_devices);

    (*device)->GetDeviceAddress (device, (USBDeviceAddress *)&new_device->address);

    /* keep a reference to this device */
    darwin_ref_cached_device(new_device);

    new_device->device = device;
    new_device->session = sessionID;
    (*device)->GetLocationID (device, &new_device->location);
    new_device->port = port;
    new_device->parent_session = parent_sessionID;

    /* cache the device descriptor */
    ret = darwin_cache_device_descriptor(ctx, new_device);
    if (ret)
      break;

    if (new_device->can_enumerate) {
      snprintf(new_device->sys_path, 20, "%03i-%04x-%04x-%02x-%02x", new_device->address,
               new_device->dev_descriptor.idVendor, new_device->dev_descriptor.idProduct,
               new_device->dev_descriptor.bDeviceClass, new_device->dev_descriptor.bDeviceSubClass);
    }
  } while (0);

  usbi_mutex_unlock(&darwin_cached_devices_lock);

  /* keep track of devices regardless of if we successfully enumerate them to
     prevent them from being enumerated multiple times */

  *cached_out = new_device;

  return ret;
}

static int process_new_device (struct libusb_context *ctx, io_service_t service) {
  struct darwin_device_priv *priv;
  struct libusb_device *dev = NULL;
  struct darwin_cached_device *cached_device;
  UInt8 devSpeed;
  int ret = 0;

  do {
    ret = darwin_get_cached_device (ctx, service, &cached_device);

    if (ret < 0 || !cached_device->can_enumerate) {
      return ret;
    }

    /* check current active configuration (and cache the first configuration value--
       which may be used by claim_interface) */
    ret = darwin_check_configuration (ctx, cached_device);
    if (ret)
      break;

    usbi_dbg ("allocating new device in context %p for with session 0x%" PRIx64,
              ctx, cached_device->session);

    dev = usbi_alloc_device(ctx, (unsigned long) cached_device->session);
    if (!dev) {
      return LIBUSB_ERROR_NO_MEM;
    }

    priv = (struct darwin_device_priv *)dev->os_priv;

    priv->dev = cached_device;
    darwin_ref_cached_device (priv->dev);

    if (cached_device->parent_session > 0) {
      dev->parent_dev = usbi_get_device_by_session_id (ctx, (unsigned long) cached_device->parent_session);
    } else {
      dev->parent_dev = NULL;
    }
    dev->port_number    = cached_device->port;
    dev->bus_number     = cached_device->location >> 24;
    dev->device_address = cached_device->address;

    (*(priv->dev->device))->GetDeviceSpeed (priv->dev->device, &devSpeed);

    switch (devSpeed) {
    case kUSBDeviceSpeedLow: dev->speed = LIBUSB_SPEED_LOW; break;
    case kUSBDeviceSpeedFull: dev->speed = LIBUSB_SPEED_FULL; break;
    case kUSBDeviceSpeedHigh: dev->speed = LIBUSB_SPEED_HIGH; break;
#if DeviceVersion >= 500
    case kUSBDeviceSpeedSuper: dev->speed = LIBUSB_SPEED_SUPER; break;
#endif
    default:
      usbi_warn (ctx, "Got unknown device speed %d", devSpeed);
    }

    ret = usbi_sanitize_device (dev);
    if (ret < 0)
      break;

    usbi_dbg ("found device with address %d port = %d parent = %p at %p", dev->device_address,
              dev->port_number, (void *) dev->parent_dev, priv->dev->sys_path);
  } while (0);

  if (0 == ret) {
    usbi_connect_device (dev);
  } else {
    libusb_unref_device (dev);
  }

  return ret;
}

static int darwin_scan_devices(struct libusb_context *ctx) {
  io_iterator_t deviceIterator;
  io_service_t service;
  kern_return_t kresult;

  kresult = usb_setup_device_iterator (&deviceIterator, 0);
  if (kresult != kIOReturnSuccess)
    return darwin_to_libusb (kresult);

  while ((service = IOIteratorNext (deviceIterator))) {
    (void) process_new_device (ctx, service);

    IOObjectRelease(service);
  }

  IOObjectRelease(deviceIterator);

  return 0;
}

static int darwin_open (struct libusb_device_handle *dev_handle) {
  struct darwin_device_handle_priv *priv = (struct darwin_device_handle_priv *)dev_handle->os_priv;
  struct darwin_cached_device *dpriv = DARWIN_CACHED_DEVICE(dev_handle->dev);
  IOReturn kresult;

  if (0 == dpriv->open_count) {
    /* try to open the device */
    kresult = (*(dpriv->device))->USBDeviceOpenSeize (dpriv->device);
    if (kresult != kIOReturnSuccess) {
      usbi_warn (HANDLE_CTX (dev_handle), "USBDeviceOpen: %s", darwin_error_str(kresult));

      if (kIOReturnExclusiveAccess != kresult) {
        return darwin_to_libusb (kresult);
      }

      /* it is possible to perform some actions on a device that is not open so do not return an error */
      priv->is_open = 0;
    } else {
      priv->is_open = 1;
    }

    /* create async event source */
    kresult = (*(dpriv->device))->CreateDeviceAsyncEventSource (dpriv->device, &priv->cfSource);
    if (kresult != kIOReturnSuccess) {
      usbi_err (HANDLE_CTX (dev_handle), "CreateDeviceAsyncEventSource: %s", darwin_error_str(kresult));

      if (priv->is_open) {
        (*(dpriv->device))->USBDeviceClose (dpriv->device);
      }

      priv->is_open = 0;

      return darwin_to_libusb (kresult);
    }

    CFRetain (libusb_darwin_acfl);

    /* add the cfSource to the aync run loop */
    CFRunLoopAddSource(libusb_darwin_acfl, priv->cfSource, kCFRunLoopCommonModes);
  }

  /* device opened successfully */
  dpriv->open_count++;

  usbi_dbg ("device open for access");

  return 0;
}

static void darwin_close (struct libusb_device_handle *dev_handle) {
  struct darwin_device_handle_priv *priv = (struct darwin_device_handle_priv *)dev_handle->os_priv;
  struct darwin_cached_device *dpriv = DARWIN_CACHED_DEVICE(dev_handle->dev);
  IOReturn kresult;
  int i;

  if (dpriv->open_count == 0) {
    /* something is probably very wrong if this is the case */
    usbi_err (HANDLE_CTX (dev_handle), "Close called on a device that was not open!");
    return;
  }

  dpriv->open_count--;

  /* make sure all interfaces are released */
  for (i = 0 ; i < USB_MAXINTERFACES ; i++)
    if (dev_handle->claimed_interfaces & (1 << i))
      libusb_release_interface (dev_handle, i);

  if (0 == dpriv->open_count) {
    /* delete the device's async event source */
    if (priv->cfSource) {
      CFRunLoopRemoveSource (libusb_darwin_acfl, priv->cfSource, kCFRunLoopDefaultMode);
      CFRelease (priv->cfSource);
      priv->cfSource = NULL;
      CFRelease (libusb_darwin_acfl);
    }

    if (priv->is_open) {
      /* close the device */
      kresult = (*(dpriv->device))->USBDeviceClose(dpriv->device);
      if (kresult) {
        /* Log the fact that we had a problem closing the file, however failing a
         * close isn't really an error, so return success anyway */
        usbi_warn (HANDLE_CTX (dev_handle), "USBDeviceClose: %s", darwin_error_str(kresult));
      }
    }
  }
}

static int darwin_get_configuration(struct libusb_device_handle *dev_handle, int *config) {
  struct darwin_cached_device *dpriv = DARWIN_CACHED_DEVICE(dev_handle->dev);

  *config = (int) dpriv->active_config;

  return 0;
}

static int darwin_set_configuration(struct libusb_device_handle *dev_handle, int config) {
  struct darwin_cached_device *dpriv = DARWIN_CACHED_DEVICE(dev_handle->dev);
  IOReturn kresult;
  int i;

  /* Setting configuration will invalidate the interface, so we need
     to reclaim it. First, dispose of existing interfaces, if any. */
  for (i = 0 ; i < USB_MAXINTERFACES ; i++)
    if (dev_handle->claimed_interfaces & (1 << i))
      darwin_release_interface (dev_handle, i);

  kresult = (*(dpriv->device))->SetConfiguration (dpriv->device, config);
  if (kresult != kIOReturnSuccess)
    return darwin_to_libusb (kresult);

  /* Reclaim any interfaces. */
  for (i = 0 ; i < USB_MAXINTERFACES ; i++)
    if (dev_handle->claimed_interfaces & (1 << i))
      darwin_claim_interface (dev_handle, i);

  dpriv->active_config = config;

  return 0;
}

static int darwin_get_interface (usb_device_t **darwin_device, uint8_t ifc, io_service_t *usbInterfacep) {
  IOUSBFindInterfaceRequest request;
  kern_return_t             kresult;
  io_iterator_t             interface_iterator;
  UInt8                     bInterfaceNumber;
  int                       ret;

  *usbInterfacep = IO_OBJECT_NULL;

  /* Setup the Interface Request */
  request.bInterfaceClass    = kIOUSBFindInterfaceDontCare;
  request.bInterfaceSubClass = kIOUSBFindInterfaceDontCare;
  request.bInterfaceProtocol = kIOUSBFindInterfaceDontCare;
  request.bAlternateSetting  = kIOUSBFindInterfaceDontCare;

  kresult = (*(darwin_device))->CreateInterfaceIterator(darwin_device, &request, &interface_iterator);
  if (kresult)
    return kresult;

  while ((*usbInterfacep = IOIteratorNext(interface_iterator))) {
    /* find the interface number */
    ret = get_ioregistry_value_number (*usbInterfacep, CFSTR("bInterfaceNumber"), kCFNumberSInt8Type,
                                       &bInterfaceNumber);

    if (ret && bInterfaceNumber == ifc) {
      break;
    }

    (void) IOObjectRelease (*usbInterfacep);
  }

  /* done with the interface iterator */
  IOObjectRelease(interface_iterator);

  return 0;
}

static int get_endpoints (struct libusb_device_handle *dev_handle, int iface) {
  struct darwin_device_handle_priv *priv = (struct darwin_device_handle_priv *)dev_handle->os_priv;

  /* current interface */
  struct darwin_interface *cInterface = &priv->interfaces[iface];

  kern_return_t kresult;

  u_int8_t numep, direction, number;
  u_int8_t dont_care1, dont_care3;
  u_int16_t dont_care2;
  int rc;

  usbi_dbg ("building table of endpoints.");

  /* retrieve the total number of endpoints on this interface */
  kresult = (*(cInterface->interface))->GetNumEndpoints(cInterface->interface, &numep);
  if (kresult) {
    usbi_err (HANDLE_CTX (dev_handle), "can't get number of endpoints for interface: %s", darwin_error_str(kresult));
    return darwin_to_libusb (kresult);
  }

  /* iterate through pipe references */
  for (int i = 1 ; i <= numep ; i++) {
    kresult = (*(cInterface->interface))->GetPipeProperties(cInterface->interface, i, &direction, &number, &dont_care1,
                                                            &dont_care2, &dont_care3);

    if (kresult != kIOReturnSuccess) {
      /* probably a buggy device. try to get the endpoint address from the descriptors */
      struct libusb_config_descriptor *config;
      const struct libusb_endpoint_descriptor *endpoint_desc;
      UInt8 alt_setting;

      kresult = (*(cInterface->interface))->GetAlternateSetting (cInterface->interface, &alt_setting);
      if (kresult) {
        usbi_err (HANDLE_CTX (dev_handle), "can't get alternate setting for interface");
        return darwin_to_libusb (kresult);
      }

      rc = libusb_get_active_config_descriptor (dev_handle->dev, &config);
      if (LIBUSB_SUCCESS != rc) {
        return rc;
      }

      endpoint_desc = config->interface[iface].altsetting[alt_setting].endpoint + i - 1;

      cInterface->endpoint_addrs[i - 1] = endpoint_desc->bEndpointAddress;
    } else {
      cInterface->endpoint_addrs[i - 1] = (((kUSBIn == direction) << kUSBRqDirnShift) | (number & LIBUSB_ENDPOINT_ADDRESS_MASK));
    }

    usbi_dbg ("interface: %i pipe %i: dir: %i number: %i", iface, i, cInterface->endpoint_addrs[i - 1] >> kUSBRqDirnShift,
              cInterface->endpoint_addrs[i - 1] & LIBUSB_ENDPOINT_ADDRESS_MASK);
  }

  cInterface->num_endpoints = numep;

  return 0;
}

static int darwin_claim_interface(struct libusb_device_handle *dev_handle, int iface) {
  struct darwin_cached_device *dpriv = DARWIN_CACHED_DEVICE(dev_handle->dev);
  struct darwin_device_handle_priv *priv = (struct darwin_device_handle_priv *)dev_handle->os_priv;
  io_service_t          usbInterface = IO_OBJECT_NULL;
  IOReturn kresult;
  IOCFPlugInInterface **plugInInterface = NULL;
  SInt32                score;

  /* current interface */
  struct darwin_interface *cInterface = &priv->interfaces[iface];

  kresult = darwin_get_interface (dpriv->device, iface, &usbInterface);
  if (kresult != kIOReturnSuccess)
    return darwin_to_libusb (kresult);

  /* make sure we have an interface */
  if (!usbInterface && dpriv->first_config != 0) {
    usbi_info (HANDLE_CTX (dev_handle), "no interface found; setting configuration: %d", dpriv->first_config);

    /* set the configuration */
    kresult = darwin_set_configuration (dev_handle, dpriv->first_config);
    if (kresult != LIBUSB_SUCCESS) {
      usbi_err (HANDLE_CTX (dev_handle), "could not set configuration");
      return kresult;
    }

    kresult = darwin_get_interface (dpriv->device, iface, &usbInterface);
    if (kresult) {
      usbi_err (HANDLE_CTX (dev_handle), "darwin_get_interface: %s", darwin_error_str(kresult));
      return darwin_to_libusb (kresult);
    }
  }

  if (!usbInterface) {
    usbi_err (HANDLE_CTX (dev_handle), "interface not found");
    return LIBUSB_ERROR_NOT_FOUND;
  }

  /* get an interface to the device's interface */
  kresult = IOCreatePlugInInterfaceForService (usbInterface, kIOUSBInterfaceUserClientTypeID,
                                               kIOCFPlugInInterfaceID, &plugInInterface, &score);

  /* ignore release error */
  (void)IOObjectRelease (usbInterface);

  if (kresult) {
    usbi_err (HANDLE_CTX (dev_handle), "IOCreatePlugInInterfaceForService: %s", darwin_error_str(kresult));
    return darwin_to_libusb (kresult);
  }

  if (!plugInInterface) {
    usbi_err (HANDLE_CTX (dev_handle), "plugin interface not found");
    return LIBUSB_ERROR_NOT_FOUND;
  }

  /* Do the actual claim */
  kresult = (*plugInInterface)->QueryInterface(plugInInterface,
                                               CFUUIDGetUUIDBytes(kIOUSBInterfaceInterfaceID),
                                               (LPVOID)&cInterface->interface);
  /* We no longer need the intermediate plug-in */
  /* Use release instead of IODestroyPlugInInterface to avoid stopping IOServices associated with this device */
  (*plugInInterface)->Release (plugInInterface);
  if (kresult || !cInterface->interface) {
    usbi_err (HANDLE_CTX (dev_handle), "QueryInterface: %s", darwin_error_str(kresult));
    return darwin_to_libusb (kresult);
  }

  /* claim the interface */
  kresult = (*(cInterface->interface))->USBInterfaceOpen(cInterface->interface);
  if (kresult) {
    usbi_err (HANDLE_CTX (dev_handle), "USBInterfaceOpen: %s", darwin_error_str(kresult));
    return darwin_to_libusb (kresult);
  }

  /* update list of endpoints */
  kresult = get_endpoints (dev_handle, iface);
  if (kresult) {
    /* this should not happen */
    darwin_release_interface (dev_handle, iface);
    usbi_err (HANDLE_CTX (dev_handle), "could not build endpoint table");
    return kresult;
  }

  cInterface->cfSource = NULL;

  /* create async event source */
  kresult = (*(cInterface->interface))->CreateInterfaceAsyncEventSource (cInterface->interface, &cInterface->cfSource);
  if (kresult != kIOReturnSuccess) {
    usbi_err (HANDLE_CTX (dev_handle), "could not create async event source");

    /* can't continue without an async event source */
    (void)darwin_release_interface (dev_handle, iface);

    return darwin_to_libusb (kresult);
  }

  /* add the cfSource to the async thread's run loop */
  CFRunLoopAddSource(libusb_darwin_acfl, cInterface->cfSource, kCFRunLoopDefaultMode);

  usbi_dbg ("interface opened");

  return 0;
}

static int darwin_release_interface(struct libusb_device_handle *dev_handle, int iface) {
  struct darwin_device_handle_priv *priv = (struct darwin_device_handle_priv *)dev_handle->os_priv;
  IOReturn kresult;

  /* current interface */
  struct darwin_interface *cInterface = &priv->interfaces[iface];

  /* Check to see if an interface is open */
  if (!cInterface->interface)
    return LIBUSB_SUCCESS;

  /* clean up endpoint data */
  cInterface->num_endpoints = 0;

  /* delete the interface's async event source */
  if (cInterface->cfSource) {
    CFRunLoopRemoveSource (libusb_darwin_acfl, cInterface->cfSource, kCFRunLoopDefaultMode);
    CFRelease (cInterface->cfSource);
  }

  kresult = (*(cInterface->interface))->USBInterfaceClose(cInterface->interface);
  if (kresult)
    usbi_warn (HANDLE_CTX (dev_handle), "USBInterfaceClose: %s", darwin_error_str(kresult));

  kresult = (*(cInterface->interface))->Release(cInterface->interface);
  if (kresult != kIOReturnSuccess)
    usbi_warn (HANDLE_CTX (dev_handle), "Release: %s", darwin_error_str(kresult));

  cInterface->interface = (usb_interface_t **) IO_OBJECT_NULL;

  return darwin_to_libusb (kresult);
}

static int darwin_set_interface_altsetting(struct libusb_device_handle *dev_handle, int iface, int altsetting) {
  struct darwin_device_handle_priv *priv = (struct darwin_device_handle_priv *)dev_handle->os_priv;
  IOReturn kresult;

  /* current interface */
  struct darwin_interface *cInterface = &priv->interfaces[iface];

  if (!cInterface->interface)
    return LIBUSB_ERROR_NO_DEVICE;

  kresult = (*(cInterface->interface))->SetAlternateInterface (cInterface->interface, altsetting);
  if (kresult != kIOReturnSuccess)
    darwin_reset_device (dev_handle);

  /* update list of endpoints */
  kresult = get_endpoints (dev_handle, iface);
  if (kresult) {
    /* this should not happen */
    darwin_release_interface (dev_handle, iface);
    usbi_err (HANDLE_CTX (dev_handle), "could not build endpoint table");
    return kresult;
  }

  return darwin_to_libusb (kresult);
}

static int darwin_clear_halt(struct libusb_device_handle *dev_handle, unsigned char endpoint) {
  /* current interface */
  struct darwin_interface *cInterface;
  IOReturn kresult;
  uint8_t pipeRef;

  /* determine the interface/endpoint to use */
  if (ep_to_pipeRef (dev_handle, endpoint, &pipeRef, NULL, &cInterface) != 0) {
    usbi_err (HANDLE_CTX (dev_handle), "endpoint not found on any open interface");

    return LIBUSB_ERROR_NOT_FOUND;
  }

  /* newer versions of darwin support clearing additional bits on the device's endpoint */
  kresult = (*(cInterface->interface))->ClearPipeStallBothEnds(cInterface->interface, pipeRef);
  if (kresult)
    usbi_warn (HANDLE_CTX (dev_handle), "ClearPipeStall: %s", darwin_error_str (kresult));

  return darwin_to_libusb (kresult);
}

static int darwin_reset_device(struct libusb_device_handle *dev_handle) {
  struct darwin_cached_device *dpriv = DARWIN_CACHED_DEVICE(dev_handle->dev);
  IOUSBDeviceDescriptor descriptor;
  IOUSBConfigurationDescriptorPtr cached_configuration;
  IOUSBConfigurationDescriptor configuration;
  bool reenumerate = false;
  IOReturn kresult;
  int i;

  kresult = (*(dpriv->device))->ResetDevice (dpriv->device);
  if (kresult) {
    usbi_err (HANDLE_CTX (dev_handle), "ResetDevice: %s", darwin_error_str (kresult));
    return darwin_to_libusb (kresult);
  }

  do {
    usbi_dbg ("darwin/reset_device: checking if device descriptor changed");

    /* ignore return code. if we can't get a descriptor it might be worthwhile re-enumerating anway */
    (void) darwin_request_descriptor (dpriv->device, kUSBDeviceDesc, 0, &descriptor, sizeof (descriptor));

    /* check if the device descriptor has changed */
    if (0 != memcmp (&dpriv->dev_descriptor, &descriptor, sizeof (descriptor))) {
      reenumerate = true;
      break;
    }

    /* check if any configuration descriptor has changed */
    for (i = 0 ; i < descriptor.bNumConfigurations ; ++i) {
      usbi_dbg ("darwin/reset_device: checking if configuration descriptor %d changed", i);

      (void) darwin_request_descriptor (dpriv->device, kUSBConfDesc, i, &configuration, sizeof (configuration));
      (*(dpriv->device))->GetConfigurationDescriptorPtr (dpriv->device, i, &cached_configuration);

      if (!cached_configuration || 0 != memcmp (cached_configuration, &configuration, sizeof (configuration))) {
        reenumerate = true;
        break;
      }
    }
  } while (0);

  if (reenumerate) {
    usbi_dbg ("darwin/reset_device: device requires reenumeration");
    (void) (*(dpriv->device))->USBDeviceReEnumerate (dpriv->device, 0);
    return LIBUSB_ERROR_NOT_FOUND;
  }

  usbi_dbg ("darwin/reset_device: device reset complete");

  return LIBUSB_SUCCESS;
}

static int darwin_kernel_driver_active(struct libusb_device_handle *dev_handle, int interface) {
  struct darwin_cached_device *dpriv = DARWIN_CACHED_DEVICE(dev_handle->dev);
  io_service_t usbInterface;
  CFTypeRef driver;
  IOReturn kresult;

  kresult = darwin_get_interface (dpriv->device, interface, &usbInterface);
  if (kresult) {
    usbi_err (HANDLE_CTX (dev_handle), "darwin_get_interface: %s", darwin_error_str(kresult));

    return darwin_to_libusb (kresult);
  }

  driver = IORegistryEntryCreateCFProperty (usbInterface, kIOBundleIdentifierKey, kCFAllocatorDefault, 0);
  IOObjectRelease (usbInterface);

  if (driver) {
    CFRelease (driver);

    return 1;
  }

  /* no driver */
  return 0;
}

/* attaching/detaching kernel drivers is not currently supported (maybe in the future?) */
static int darwin_attach_kernel_driver (struct libusb_device_handle *dev_handle, int interface) {
  UNUSED(dev_handle);
  UNUSED(interface);
  return LIBUSB_ERROR_NOT_SUPPORTED;
}

static int darwin_detach_kernel_driver (struct libusb_device_handle *dev_handle, int interface) {
  UNUSED(dev_handle);
  UNUSED(interface);
  return LIBUSB_ERROR_NOT_SUPPORTED;
}

static void darwin_destroy_device(struct libusb_device *dev) {
  struct darwin_device_priv *dpriv = (struct darwin_device_priv *) dev->os_priv;

  if (dpriv->dev) {
    /* need to hold the lock in case this is the last reference to the device */
    usbi_mutex_lock(&darwin_cached_devices_lock);
    darwin_deref_cached_device (dpriv->dev);
    dpriv->dev = NULL;
    usbi_mutex_unlock(&darwin_cached_devices_lock);
  }
}

static int submit_bulk_transfer(struct usbi_transfer *itransfer) {
  struct libusb_transfer *transfer = USBI_TRANSFER_TO_LIBUSB_TRANSFER(itransfer);

  IOReturn               ret;
  uint8_t                transferType;
  /* None of the values below are used in libusbx for bulk transfers */
  uint8_t                direction, number, interval, pipeRef;
  uint16_t               maxPacketSize;

  struct darwin_interface *cInterface;

  if (ep_to_pipeRef (transfer->dev_handle, transfer->endpoint, &pipeRef, NULL, &cInterface) != 0) {
    usbi_err (TRANSFER_CTX (transfer), "endpoint not found on any open interface");

    return LIBUSB_ERROR_NOT_FOUND;
  }

  ret = (*(cInterface->interface))->GetPipeProperties (cInterface->interface, pipeRef, &direction, &number,
                                                       &transferType, &maxPacketSize, &interval);

  if (ret) {
    usbi_err (TRANSFER_CTX (transfer), "bulk transfer failed (dir = %s): %s (code = 0x%08x)", IS_XFERIN(transfer) ? "In" : "Out",
              darwin_error_str(ret), ret);
    return darwin_to_libusb (ret);
  }

  if (0 != (transfer->length % maxPacketSize)) {
    /* do not need a zero packet */
    transfer->flags &= ~LIBUSB_TRANSFER_ADD_ZERO_PACKET;
  }

  /* submit the request */
  /* timeouts are unavailable on interrupt endpoints */
  if (transferType == kUSBInterrupt) {
    if (IS_XFERIN(transfer))
      ret = (*(cInterface->interface))->ReadPipeAsync(cInterface->interface, pipeRef, transfer->buffer,
                                                      transfer->length, darwin_async_io_callback, itransfer);
    else
      ret = (*(cInterface->interface))->WritePipeAsync(cInterface->interface, pipeRef, transfer->buffer,
                                                       transfer->length, darwin_async_io_callback, itransfer);
  } else {
    itransfer->timeout_flags |= USBI_TRANSFER_OS_HANDLES_TIMEOUT;

    if (IS_XFERIN(transfer))
      ret = (*(cInterface->interface))->ReadPipeAsyncTO(cInterface->interface, pipeRef, transfer->buffer,
                                                        transfer->length, transfer->timeout, transfer->timeout,
                                                        darwin_async_io_callback, (void *)itransfer);
    else
      ret = (*(cInterface->interface))->WritePipeAsyncTO(cInterface->interface, pipeRef, transfer->buffer,
                                                         transfer->length, transfer->timeout, transfer->timeout,
                                                         darwin_async_io_callback, (void *)itransfer);
  }

  if (ret)
    usbi_err (TRANSFER_CTX (transfer), "bulk transfer failed (dir = %s): %s (code = 0x%08x)", IS_XFERIN(transfer) ? "In" : "Out",
               darwin_error_str(ret), ret);

  return darwin_to_libusb (ret);
}

#if InterfaceVersion >= 550
static int submit_stream_transfer(struct usbi_transfer *itransfer) {
  struct libusb_transfer *transfer = USBI_TRANSFER_TO_LIBUSB_TRANSFER(itransfer);
  struct darwin_interface *cInterface;
  uint8_t pipeRef;
  IOReturn ret;

  if (ep_to_pipeRef (transfer->dev_handle, transfer->endpoint, &pipeRef, NULL, &cInterface) != 0) {
    usbi_err (TRANSFER_CTX (transfer), "endpoint not found on any open interface");

    return LIBUSB_ERROR_NOT_FOUND;
  }

  itransfer->timeout_flags |= USBI_TRANSFER_OS_HANDLES_TIMEOUT;

  if (IS_XFERIN(transfer))
    ret = (*(cInterface->interface))->ReadStreamsPipeAsyncTO(cInterface->interface, pipeRef, itransfer->stream_id,
                                                             transfer->buffer, transfer->length, transfer->timeout,
                                                             transfer->timeout, darwin_async_io_callback, (void *)itransfer);
  else
    ret = (*(cInterface->interface))->WriteStreamsPipeAsyncTO(cInterface->interface, pipeRef, itransfer->stream_id,
                                                              transfer->buffer, transfer->length, transfer->timeout,
                                                              transfer->timeout, darwin_async_io_callback, (void *)itransfer);

  if (ret)
    usbi_err (TRANSFER_CTX (transfer), "bulk stream transfer failed (dir = %s): %s (code = 0x%08x)", IS_XFERIN(transfer) ? "In" : "Out",
               darwin_error_str(ret), ret);

  return darwin_to_libusb (ret);
}
#endif

static int submit_iso_transfer(struct usbi_transfer *itransfer) {
  struct libusb_transfer *transfer = USBI_TRANSFER_TO_LIBUSB_TRANSFER(itransfer);
  struct darwin_transfer_priv *tpriv = usbi_transfer_get_os_priv(itransfer);

  IOReturn kresult;
  uint8_t direction, number, interval, pipeRef, transferType;
  uint16_t maxPacketSize;
  UInt64 frame;
  AbsoluteTime atTime;
  int i;

  struct darwin_interface *cInterface;

  /* construct an array of IOUSBIsocFrames, reuse the old one if possible */
  if (tpriv->isoc_framelist && tpriv->num_iso_packets != transfer->num_iso_packets) {
    free(tpriv->isoc_framelist);
    tpriv->isoc_framelist = NULL;
  }

  if (!tpriv->isoc_framelist) {
    tpriv->num_iso_packets = transfer->num_iso_packets;
    tpriv->isoc_framelist = (IOUSBIsocFrame*) calloc (transfer->num_iso_packets, sizeof(IOUSBIsocFrame));
    if (!tpriv->isoc_framelist)
      return LIBUSB_ERROR_NO_MEM;
  }

  /* copy the frame list from the libusb descriptor (the structures differ only is member order) */
  for (i = 0 ; i < transfer->num_iso_packets ; i++)
    tpriv->isoc_framelist[i].frReqCount = transfer->iso_packet_desc[i].length;

  /* determine the interface/endpoint to use */
  if (ep_to_pipeRef (transfer->dev_handle, transfer->endpoint, &pipeRef, NULL, &cInterface) != 0) {
    usbi_err (TRANSFER_CTX (transfer), "endpoint not found on any open interface");

    return LIBUSB_ERROR_NOT_FOUND;
  }

  /* determine the properties of this endpoint and the speed of the device */
  (*(cInterface->interface))->GetPipeProperties (cInterface->interface, pipeRef, &direction, &number,
                                                 &transferType, &maxPacketSize, &interval);

  /* Last but not least we need the bus frame number */
  kresult = (*(cInterface->interface))->GetBusFrameNumber(cInterface->interface, &frame, &atTime);
  if (kresult) {
    usbi_err (TRANSFER_CTX (transfer), "failed to get bus frame number: %d", kresult);
    free(tpriv->isoc_framelist);
    tpriv->isoc_framelist = NULL;

    return darwin_to_libusb (kresult);
  }

  (*(cInterface->interface))->GetPipeProperties (cInterface->interface, pipeRef, &direction, &number,
                                                 &transferType, &maxPacketSize, &interval);

  /* schedule for a frame a little in the future */
  frame += 4;

  if (cInterface->frames[transfer->endpoint] && frame < cInterface->frames[transfer->endpoint])
    frame = cInterface->frames[transfer->endpoint];

  /* submit the request */
  if (IS_XFERIN(transfer))
    kresult = (*(cInterface->interface))->ReadIsochPipeAsync(cInterface->interface, pipeRef, transfer->buffer, frame,
                                                             transfer->num_iso_packets, tpriv->isoc_framelist, darwin_async_io_callback,
                                                             itransfer);
  else
    kresult = (*(cInterface->interface))->WriteIsochPipeAsync(cInterface->interface, pipeRef, transfer->buffer, frame,
                                                              transfer->num_iso_packets, tpriv->isoc_framelist, darwin_async_io_callback,
                                                              itransfer);

  if (LIBUSB_SPEED_FULL == transfer->dev_handle->dev->speed)
    /* Full speed */
    cInterface->frames[transfer->endpoint] = frame + transfer->num_iso_packets * (1 << (interval - 1));
  else
    /* High/super speed */
    cInterface->frames[transfer->endpoint] = frame + transfer->num_iso_packets * (1 << (interval - 1)) / 8;

  if (kresult != kIOReturnSuccess) {
    usbi_err (TRANSFER_CTX (transfer), "isochronous transfer failed (dir: %s): %s", IS_XFERIN(transfer) ? "In" : "Out",
               darwin_error_str(kresult));
    free (tpriv->isoc_framelist);
    tpriv->isoc_framelist = NULL;
  }

  return darwin_to_libusb (kresult);
}

static int submit_control_transfer(struct usbi_transfer *itransfer) {
  struct libusb_transfer *transfer = USBI_TRANSFER_TO_LIBUSB_TRANSFER(itransfer);
  struct libusb_control_setup *setup = (struct libusb_control_setup *) transfer->buffer;
  struct darwin_cached_device *dpriv = DARWIN_CACHED_DEVICE(transfer->dev_handle->dev);
  struct darwin_transfer_priv *tpriv = usbi_transfer_get_os_priv(itransfer);

  IOReturn               kresult;

  memset(&tpriv->req, 0, sizeof(tpriv->req));

  /* IOUSBDeviceInterface expects the request in cpu endianness */
  tpriv->req.bmRequestType     = setup->bmRequestType;
  tpriv->req.bRequest          = setup->bRequest;
  /* these values should be in bus order from libusb_fill_control_setup */
  tpriv->req.wValue            = OSSwapLittleToHostInt16 (setup->wValue);
  tpriv->req.wIndex            = OSSwapLittleToHostInt16 (setup->wIndex);
  tpriv->req.wLength           = OSSwapLittleToHostInt16 (setup->wLength);
  /* data is stored after the libusb control block */
  tpriv->req.pData             = transfer->buffer + LIBUSB_CONTROL_SETUP_SIZE;
  tpriv->req.completionTimeout = transfer->timeout;
  tpriv->req.noDataTimeout     = transfer->timeout;

  itransfer->timeout_flags |= USBI_TRANSFER_OS_HANDLES_TIMEOUT;

  /* all transfers in libusb-1.0 are async */

  if (transfer->endpoint) {
    struct darwin_interface *cInterface;
    uint8_t                 pipeRef;

    if (ep_to_pipeRef (transfer->dev_handle, transfer->endpoint, &pipeRef, NULL, &cInterface) != 0) {
      usbi_err (TRANSFER_CTX (transfer), "endpoint not found on any open interface");

      return LIBUSB_ERROR_NOT_FOUND;
    }

    kresult = (*(cInterface->interface))->ControlRequestAsyncTO (cInterface->interface, pipeRef, &(tpriv->req), darwin_async_io_callback, itransfer);
  } else
    /* control request on endpoint 0 */
    kresult = (*(dpriv->device))->DeviceRequestAsyncTO(dpriv->device, &(tpriv->req), darwin_async_io_callback, itransfer);

  if (kresult != kIOReturnSuccess)
    usbi_err (TRANSFER_CTX (transfer), "control request failed: %s", darwin_error_str(kresult));

  return darwin_to_libusb (kresult);
}

static int darwin_submit_transfer(struct usbi_transfer *itransfer) {
  struct libusb_transfer *transfer = USBI_TRANSFER_TO_LIBUSB_TRANSFER(itransfer);

  switch (transfer->type) {
  case LIBUSB_TRANSFER_TYPE_CONTROL:
    return submit_control_transfer(itransfer);
  case LIBUSB_TRANSFER_TYPE_BULK:
  case LIBUSB_TRANSFER_TYPE_INTERRUPT:
    return submit_bulk_transfer(itransfer);
  case LIBUSB_TRANSFER_TYPE_ISOCHRONOUS:
    return submit_iso_transfer(itransfer);
  case LIBUSB_TRANSFER_TYPE_BULK_STREAM:
#if InterfaceVersion >= 550
    return submit_stream_transfer(itransfer);
#else
    usbi_err (TRANSFER_CTX(transfer), "IOUSBFamily version does not support bulk stream transfers");
    return LIBUSB_ERROR_NOT_SUPPORTED;
#endif
  default:
    usbi_err (TRANSFER_CTX(transfer), "unknown endpoint type %d", transfer->type);
    return LIBUSB_ERROR_INVALID_PARAM;
  }
}

static int cancel_control_transfer(struct usbi_transfer *itransfer) {
  struct libusb_transfer *transfer = USBI_TRANSFER_TO_LIBUSB_TRANSFER(itransfer);
  struct darwin_cached_device *dpriv = DARWIN_CACHED_DEVICE(transfer->dev_handle->dev);
  IOReturn kresult;

  usbi_warn (ITRANSFER_CTX (itransfer), "aborting all transactions control pipe");

  if (!dpriv->device)
    return LIBUSB_ERROR_NO_DEVICE;

  kresult = (*(dpriv->device))->USBDeviceAbortPipeZero (dpriv->device);

  return darwin_to_libusb (kresult);
}

static int darwin_abort_transfers (struct usbi_transfer *itransfer) {
  struct libusb_transfer *transfer = USBI_TRANSFER_TO_LIBUSB_TRANSFER(itransfer);
  struct darwin_cached_device *dpriv = DARWIN_CACHED_DEVICE(transfer->dev_handle->dev);
  struct darwin_interface *cInterface;
  uint8_t pipeRef, iface;
  IOReturn kresult;

  if (ep_to_pipeRef (transfer->dev_handle, transfer->endpoint, &pipeRef, &iface, &cInterface) != 0) {
    usbi_err (TRANSFER_CTX (transfer), "endpoint not found on any open interface");

    return LIBUSB_ERROR_NOT_FOUND;
  }

  if (!dpriv->device)
    return LIBUSB_ERROR_NO_DEVICE;

  usbi_warn (ITRANSFER_CTX (itransfer), "aborting all transactions on interface %d pipe %d", iface, pipeRef);

  /* abort transactions */
#if InterfaceVersion >= 550
  if (LIBUSB_TRANSFER_TYPE_BULK_STREAM == transfer->type)
    (*(cInterface->interface))->AbortStreamsPipe (cInterface->interface, pipeRef, itransfer->stream_id);
  else
#endif
    (*(cInterface->interface))->AbortPipe (cInterface->interface, pipeRef);

  usbi_dbg ("calling clear pipe stall to clear the data toggle bit");

  /* newer versions of darwin support clearing additional bits on the device's endpoint */
  kresult = (*(cInterface->interface))->ClearPipeStallBothEnds(cInterface->interface, pipeRef);

  return darwin_to_libusb (kresult);
}

static int darwin_cancel_transfer(struct usbi_transfer *itransfer) {
  struct libusb_transfer *transfer = USBI_TRANSFER_TO_LIBUSB_TRANSFER(itransfer);

  switch (transfer->type) {
  case LIBUSB_TRANSFER_TYPE_CONTROL:
    return cancel_control_transfer(itransfer);
  case LIBUSB_TRANSFER_TYPE_BULK:
  case LIBUSB_TRANSFER_TYPE_INTERRUPT:
  case LIBUSB_TRANSFER_TYPE_ISOCHRONOUS:
    return darwin_abort_transfers (itransfer);
  default:
    usbi_err (TRANSFER_CTX(transfer), "unknown endpoint type %d", transfer->type);
    return LIBUSB_ERROR_INVALID_PARAM;
  }
}

static void darwin_clear_transfer_priv (struct usbi_transfer *itransfer) {
  struct libusb_transfer *transfer = USBI_TRANSFER_TO_LIBUSB_TRANSFER(itransfer);
  struct darwin_transfer_priv *tpriv = usbi_transfer_get_os_priv(itransfer);

  if (transfer->type == LIBUSB_TRANSFER_TYPE_ISOCHRONOUS && tpriv->isoc_framelist) {
    free (tpriv->isoc_framelist);
    tpriv->isoc_framelist = NULL;
  }
}

static void darwin_async_io_callback (void *refcon, IOReturn result, void *arg0) {
  struct usbi_transfer *itransfer = (struct usbi_transfer *)refcon;
  struct libusb_transfer *transfer = USBI_TRANSFER_TO_LIBUSB_TRANSFER(itransfer);
  struct darwin_transfer_priv *tpriv = usbi_transfer_get_os_priv(itransfer);

  usbi_dbg ("an async io operation has completed");

  /* if requested write a zero packet */
  if (kIOReturnSuccess == result && IS_XFEROUT(transfer) && transfer->flags & LIBUSB_TRANSFER_ADD_ZERO_PACKET) {
    struct darwin_interface *cInterface;
    uint8_t pipeRef;

    (void) ep_to_pipeRef (transfer->dev_handle, transfer->endpoint, &pipeRef, NULL, &cInterface);

    (*(cInterface->interface))->WritePipe (cInterface->interface, pipeRef, transfer->buffer, 0);
  }

  tpriv->result = result;
  tpriv->size = (UInt32) (uintptr_t) arg0;

  /* signal the core that this transfer is complete */
  usbi_signal_transfer_completion(itransfer);
}

static int darwin_transfer_status (struct usbi_transfer *itransfer, kern_return_t result) {
  if (itransfer->timeout_flags & USBI_TRANSFER_TIMED_OUT)
    result = kIOUSBTransactionTimeout;

  switch (result) {
  case kIOReturnUnderrun:
  case kIOReturnSuccess:
    return LIBUSB_TRANSFER_COMPLETED;
  case kIOReturnAborted:
    return LIBUSB_TRANSFER_CANCELLED;
  case kIOUSBPipeStalled:
    usbi_dbg ("transfer error: pipe is stalled");
    return LIBUSB_TRANSFER_STALL;
  case kIOReturnOverrun:
    usbi_warn (ITRANSFER_CTX (itransfer), "transfer error: data overrun");
    return LIBUSB_TRANSFER_OVERFLOW;
  case kIOUSBTransactionTimeout:
    usbi_warn (ITRANSFER_CTX (itransfer), "transfer error: timed out");
    itransfer->timeout_flags |= USBI_TRANSFER_TIMED_OUT;
    return LIBUSB_TRANSFER_TIMED_OUT;
  default:
    usbi_warn (ITRANSFER_CTX (itransfer), "transfer error: %s (value = 0x%08x)", darwin_error_str (result), result);
    return LIBUSB_TRANSFER_ERROR;
  }
}

static int darwin_handle_transfer_completion (struct usbi_transfer *itransfer) {
  struct libusb_transfer *transfer = USBI_TRANSFER_TO_LIBUSB_TRANSFER(itransfer);
  struct darwin_transfer_priv *tpriv = usbi_transfer_get_os_priv(itransfer);
  int isIsoc      = LIBUSB_TRANSFER_TYPE_ISOCHRONOUS == transfer->type;
  int isBulk      = LIBUSB_TRANSFER_TYPE_BULK == transfer->type;
  int isControl   = LIBUSB_TRANSFER_TYPE_CONTROL == transfer->type;
  int isInterrupt = LIBUSB_TRANSFER_TYPE_INTERRUPT == transfer->type;
  int i;

  if (!isIsoc && !isBulk && !isControl && !isInterrupt) {
    usbi_err (TRANSFER_CTX(transfer), "unknown endpoint type %d", transfer->type);
    return LIBUSB_ERROR_INVALID_PARAM;
  }

  usbi_dbg ("handling %s completion with kernel status %d",
             isControl ? "control" : isBulk ? "bulk" : isIsoc ? "isoc" : "interrupt", tpriv->result);

  if (kIOReturnSuccess == tpriv->result || kIOReturnUnderrun == tpriv->result) {
    if (isIsoc && tpriv->isoc_framelist) {
      /* copy isochronous results back */

      for (i = 0; i < transfer->num_iso_packets ; i++) {
        struct libusb_iso_packet_descriptor *lib_desc = &transfer->iso_packet_desc[i];
        lib_desc->status = darwin_to_libusb (tpriv->isoc_framelist[i].frStatus);
        lib_desc->actual_length = tpriv->isoc_framelist[i].frActCount;
      }
    } else if (!isIsoc)
      itransfer->transferred += tpriv->size;
  }

  /* it is ok to handle cancelled transfers without calling usbi_handle_transfer_cancellation (we catch timeout transfers) */
  return usbi_handle_transfer_completion (itransfer, darwin_transfer_status (itransfer, tpriv->result));
}

static int darwin_clock_gettime(int clk_id, struct timespec *tp) {
#if !OSX_USE_CLOCK_GETTIME
  mach_timespec_t sys_time;
  clock_serv_t clock_ref;

  switch (clk_id) {
  case USBI_CLOCK_REALTIME:
    /* CLOCK_REALTIME represents time since the epoch */
    clock_ref = clock_realtime;
    break;
  case USBI_CLOCK_MONOTONIC:
    /* use system boot time as reference for the monotonic clock */
    clock_ref = clock_monotonic;
    break;
  default:
    return LIBUSB_ERROR_INVALID_PARAM;
  }

  clock_get_time (clock_ref, &sys_time);

  tp->tv_sec  = sys_time.tv_sec;
  tp->tv_nsec = sys_time.tv_nsec;

  return 0;
#else
  switch (clk_id) {
  case USBI_CLOCK_MONOTONIC:
    return clock_gettime(CLOCK_MONOTONIC, tp);
  case USBI_CLOCK_REALTIME:
    return clock_gettime(CLOCK_REALTIME, tp);
  default:
    return LIBUSB_ERROR_INVALID_PARAM;
  }
#endif
}

#if InterfaceVersion >= 550
static int darwin_alloc_streams (struct libusb_device_handle *dev_handle, uint32_t num_streams, unsigned char *endpoints,
                                 int num_endpoints) {
  struct darwin_interface *cInterface;
  UInt32 supportsStreams;
  uint8_t pipeRef;
  int rc, i;

  /* find the mimimum number of supported streams on the endpoint list */
  for (i = 0 ; i < num_endpoints ; ++i) {
    if (0 != (rc = ep_to_pipeRef (dev_handle, endpoints[i], &pipeRef, NULL, &cInterface))) {
      return rc;
    }

    (*(cInterface->interface))->SupportsStreams (cInterface->interface, pipeRef, &supportsStreams);
    if (num_streams > supportsStreams)
      num_streams = supportsStreams;
  }

  /* it is an error if any endpoint in endpoints does not support streams */
  if (0 == num_streams)
    return LIBUSB_ERROR_INVALID_PARAM;

  /* create the streams */
  for (i = 0 ; i < num_endpoints ; ++i) {
    (void) ep_to_pipeRef (dev_handle, endpoints[i], &pipeRef, NULL, &cInterface);

    rc = (*(cInterface->interface))->CreateStreams (cInterface->interface, pipeRef, num_streams);
    if (kIOReturnSuccess != rc)
      return darwin_to_libusb(rc);
  }

  return num_streams;
}

static int darwin_free_streams (struct libusb_device_handle *dev_handle, unsigned char *endpoints, int num_endpoints) {
  struct darwin_interface *cInterface;
  UInt32 supportsStreams;
  uint8_t pipeRef;
  int rc;

  for (int i = 0 ; i < num_endpoints ; ++i) {
    if (0 != (rc = ep_to_pipeRef (dev_handle, endpoints[i], &pipeRef, NULL, &cInterface)))
      return rc;

    (*(cInterface->interface))->SupportsStreams (cInterface->interface, pipeRef, &supportsStreams);
    if (0 == supportsStreams)
      return LIBUSB_ERROR_INVALID_PARAM;

    rc = (*(cInterface->interface))->CreateStreams (cInterface->interface, pipeRef, 0);
    if (kIOReturnSuccess != rc)
      return darwin_to_libusb(rc);
  }

  return LIBUSB_SUCCESS;
}
#endif

const struct usbi_os_backend usbi_backend = {
        .name = "Darwin",
        .caps = 0,
        .init = darwin_init,
        .exit = darwin_exit,
        .get_device_list = NULL, /* not needed */
        .get_device_descriptor = darwin_get_device_descriptor,
        .get_active_config_descriptor = darwin_get_active_config_descriptor,
        .get_config_descriptor = darwin_get_config_descriptor,
        .hotplug_poll = darwin_hotplug_poll,

        .open = darwin_open,
        .close = darwin_close,
        .get_configuration = darwin_get_configuration,
        .set_configuration = darwin_set_configuration,
        .claim_interface = darwin_claim_interface,
        .release_interface = darwin_release_interface,

        .set_interface_altsetting = darwin_set_interface_altsetting,
        .clear_halt = darwin_clear_halt,
        .reset_device = darwin_reset_device,

#if InterfaceVersion >= 550
        .alloc_streams = darwin_alloc_streams,
        .free_streams = darwin_free_streams,
#endif

        .kernel_driver_active = darwin_kernel_driver_active,
        .detach_kernel_driver = darwin_detach_kernel_driver,
        .attach_kernel_driver = darwin_attach_kernel_driver,

        .destroy_device = darwin_destroy_device,

        .submit_transfer = darwin_submit_transfer,
        .cancel_transfer = darwin_cancel_transfer,
        .clear_transfer_priv = darwin_clear_transfer_priv,

        .handle_transfer_completion = darwin_handle_transfer_completion,

        .clock_gettime = darwin_clock_gettime,

        .device_priv_size = sizeof(struct darwin_device_priv),
        .device_handle_priv_size = sizeof(struct darwin_device_handle_priv),
        .transfer_priv_size = sizeof(struct darwin_transfer_priv),
};
