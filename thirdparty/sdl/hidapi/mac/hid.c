/*******************************************************
 HIDAPI - Multi-Platform library for
 communication with HID devices.

 Alan Ott
 Signal 11 Software

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

/* See Apple Technical Note TN2187 for details on IOHidManager. */

#include <IOKit/hid/IOHIDManager.h>
#include <IOKit/hid/IOHIDKeys.h>
#include <IOKit/IOKitLib.h>
#include <IOKit/usb/USBSpec.h>
#include <CoreFoundation/CoreFoundation.h>
#include <mach/mach_error.h>
#include <stdbool.h>
#include <wchar.h>
#include <locale.h>
#include <pthread.h>
#include <sys/time.h>
#include <unistd.h>
#include <dlfcn.h>

#include "hidapi_darwin.h"

/* Barrier implementation because Mac OSX doesn't have pthread_barrier.
   It also doesn't have clock_gettime(). So much for POSIX and SUSv2.
   This implementation came from Brent Priddy and was posted on
   StackOverflow. It is used with his permission. */
typedef int pthread_barrierattr_t;
typedef struct pthread_barrier {
    pthread_mutex_t mutex;
    pthread_cond_t cond;
    int count;
    int trip_count;
} pthread_barrier_t;

static int pthread_barrier_init(pthread_barrier_t *barrier, const pthread_barrierattr_t *attr, unsigned int count)
{
	(void) attr;

	if (count == 0) {
		errno = EINVAL;
		return -1;
	}

	if (pthread_mutex_init(&barrier->mutex, 0) < 0) {
		return -1;
	}
	if (pthread_cond_init(&barrier->cond, 0) < 0) {
		pthread_mutex_destroy(&barrier->mutex);
		return -1;
	}
	barrier->trip_count = count;
	barrier->count = 0;

	return 0;
}

static int pthread_barrier_destroy(pthread_barrier_t *barrier)
{
	pthread_cond_destroy(&barrier->cond);
	pthread_mutex_destroy(&barrier->mutex);
	return 0;
}

static int pthread_barrier_wait(pthread_barrier_t *barrier)
{
	pthread_mutex_lock(&barrier->mutex);
	++(barrier->count);
	if (barrier->count >= barrier->trip_count) {
		barrier->count = 0;
		pthread_mutex_unlock(&barrier->mutex);
		pthread_cond_broadcast(&barrier->cond);
		return 1;
	}
	else {
		do {
			pthread_cond_wait(&barrier->cond, &(barrier->mutex));
		}
		while (barrier->count != 0);

		pthread_mutex_unlock(&barrier->mutex);
		return 0;
	}
}

static int return_data(hid_device *dev, unsigned char *data, size_t length);

/* Linked List of input reports received from the device. */
struct input_report {
	uint8_t *data;
	size_t len;
	struct input_report *next;
};

static struct hid_api_version api_version = {
	.major = HID_API_VERSION_MAJOR,
	.minor = HID_API_VERSION_MINOR,
	.patch = HID_API_VERSION_PATCH
};

/* - Run context - */
static	IOHIDManagerRef hid_mgr = 0x0;
static	int is_macos_10_10_or_greater = 0;
static	IOOptionBits device_open_options = 0;
static	wchar_t *last_global_error_str = NULL;
/* --- */

struct hid_device_ {
	IOHIDDeviceRef device_handle;
	IOOptionBits open_options;
	int blocking;
	int disconnected;
	CFStringRef run_loop_mode;
	CFRunLoopRef run_loop;
	CFRunLoopSourceRef source;
	uint8_t *input_report_buf;
	CFIndex max_input_report_len;
	struct input_report *input_reports;
	struct hid_device_info* device_info;

	pthread_t thread;
	pthread_mutex_t mutex; /* Protects input_reports */
	pthread_cond_t condition;
	pthread_barrier_t barrier; /* Ensures correct startup sequence */
	pthread_barrier_t shutdown_barrier; /* Ensures correct shutdown sequence */
	int shutdown_thread;
	wchar_t *last_error_str;
};

static hid_device *new_hid_device(void)
{
	hid_device *dev = (hid_device*) calloc(1, sizeof(hid_device));
	if (dev == NULL) {
		return NULL;
	}

	dev->device_handle = NULL;
	dev->open_options = device_open_options;
	dev->blocking = 1;
	dev->disconnected = 0;
	dev->run_loop_mode = NULL;
	dev->run_loop = NULL;
	dev->source = NULL;
	dev->input_report_buf = NULL;
	dev->input_reports = NULL;
	dev->device_info = NULL;
	dev->shutdown_thread = 0;
	dev->last_error_str = NULL;

	/* Thread objects */
	pthread_mutex_init(&dev->mutex, NULL);
	pthread_cond_init(&dev->condition, NULL);
	pthread_barrier_init(&dev->barrier, NULL, 2);
	pthread_barrier_init(&dev->shutdown_barrier, NULL, 2);

	return dev;
}

static void free_hid_device(hid_device *dev)
{
	if (!dev)
		return;

	/* Delete any input reports still left over. */
	struct input_report *rpt = dev->input_reports;
	while (rpt) {
		struct input_report *next = rpt->next;
		free(rpt->data);
		free(rpt);
		rpt = next;
	}

	/* Free the string and the report buffer. The check for NULL
	   is necessary here as CFRelease() doesn't handle NULL like
	   free() and others do. */
	if (dev->run_loop_mode)
		CFRelease(dev->run_loop_mode);
	if (dev->source)
		CFRelease(dev->source);
	free(dev->input_report_buf);
	hid_free_enumeration(dev->device_info);

	/* Clean up the thread objects */
	pthread_barrier_destroy(&dev->shutdown_barrier);
	pthread_barrier_destroy(&dev->barrier);
	pthread_cond_destroy(&dev->condition);
	pthread_mutex_destroy(&dev->mutex);

	/* Free the structure itself. */
	free(dev);
}


#ifndef HIDAPI_USING_SDL_RUNTIME
/* The caller must free the returned string with free(). */
static wchar_t *utf8_to_wchar_t(const char *utf8)
{
	wchar_t *ret = NULL;

	if (utf8) {
		size_t wlen = mbstowcs(NULL, utf8, 0);
		if ((size_t) -1 == wlen) {
			return wcsdup(L"");
		}
		ret = (wchar_t*) calloc(wlen+1, sizeof(wchar_t));
		if (ret == NULL) {
			/* as much as we can do at this point */
			return NULL;
		}
		mbstowcs(ret, utf8, wlen+1);
		ret[wlen] = 0x0000;
	}

	return ret;
}
#endif


/* Makes a copy of the given error message (and decoded according to the
 * currently locale) into the wide string pointer pointed by error_str.
 * The last stored error string is freed.
 * Use register_error_str(NULL) to free the error message completely. */
static void register_error_str(wchar_t **error_str, const char *msg)
{
	free(*error_str);
#ifdef HIDAPI_USING_SDL_RUNTIME
	/* Thread-safe error handling */
	if (msg) {
		SDL_SetError("%s", msg);
	} else {
		SDL_ClearError();
	}
#else
	*error_str = utf8_to_wchar_t(msg);
#endif
}

/* Similar to register_error_str, but allows passing a format string with va_list args into this function. */
static void register_error_str_vformat(wchar_t **error_str, const char *format, va_list args)
{
	char msg[1024];
	vsnprintf(msg, sizeof(msg), format, args);

	register_error_str(error_str, msg);
}

/* Set the last global error to be reported by hid_error(NULL).
 * The given error message will be copied (and decoded according to the
 * currently locale, so do not pass in string constants).
 * The last stored global error message is freed.
 * Use register_global_error(NULL) to indicate "no error". */
static void register_global_error(const char *msg)
{
	register_error_str(&last_global_error_str, msg);
}

/* Similar to register_global_error, but allows passing a format string into this function. */
static void register_global_error_format(const char *format, ...)
{
	va_list args;
	va_start(args, format);
	register_error_str_vformat(&last_global_error_str, format, args);
	va_end(args);
}

/* Set the last error for a device to be reported by hid_error(dev).
 * The given error message will be copied (and decoded according to the
 * currently locale, so do not pass in string constants).
 * The last stored device error message is freed.
 * Use register_device_error(dev, NULL) to indicate "no error". */
static void register_device_error(hid_device *dev, const char *msg)
{
	register_error_str(&dev->last_error_str, msg);
}

/* Similar to register_device_error, but you can pass a format string into this function. */
static void register_device_error_format(hid_device *dev, const char *format, ...)
{
	va_list args;
	va_start(args, format);
	register_error_str_vformat(&dev->last_error_str, format, args);
	va_end(args);
}


static CFArrayRef get_array_property(IOHIDDeviceRef device, CFStringRef key)
{
	CFTypeRef ref = IOHIDDeviceGetProperty(device, key);
	if (ref != NULL && CFGetTypeID(ref) == CFArrayGetTypeID()) {
		return (CFArrayRef)ref;
	} else {
		return NULL;
	}
}

static int32_t get_int_property(IOHIDDeviceRef device, CFStringRef key)
{
	CFTypeRef ref;
	int32_t value = 0;

	ref = IOHIDDeviceGetProperty(device, key);
	if (ref) {
		if (CFGetTypeID(ref) == CFNumberGetTypeID()) {
			CFNumberGetValue((CFNumberRef) ref, kCFNumberSInt32Type, &value);
			return value;
		}
	}
	return 0;
}

static bool try_get_int_property(IOHIDDeviceRef device, CFStringRef key, int32_t *out_val)
{
	bool result = false;
	CFTypeRef ref;

	ref = IOHIDDeviceGetProperty(device, key);
	if (ref) {
		if (CFGetTypeID(ref) == CFNumberGetTypeID()) {
			result = CFNumberGetValue((CFNumberRef) ref, kCFNumberSInt32Type, out_val);
		}
	}
	return result;
}

static bool try_get_ioregistry_int_property(io_service_t service, CFStringRef property, int32_t *out_val)
{
	bool result = false;
	CFTypeRef ref = IORegistryEntryCreateCFProperty(service, property, kCFAllocatorDefault, 0);

	if (ref) {
		if (CFGetTypeID(ref) == CFNumberGetTypeID()) {
			result = CFNumberGetValue(ref, kCFNumberSInt32Type, out_val);
		}

		CFRelease(ref);
	}

	return result;
}

static CFArrayRef get_usage_pairs(IOHIDDeviceRef device)
{
	return get_array_property(device, CFSTR(kIOHIDDeviceUsagePairsKey));
}

static unsigned short get_vendor_id(IOHIDDeviceRef device)
{
	return get_int_property(device, CFSTR(kIOHIDVendorIDKey));
}

static unsigned short get_product_id(IOHIDDeviceRef device)
{
	return get_int_property(device, CFSTR(kIOHIDProductIDKey));
}

static int32_t get_max_report_length(IOHIDDeviceRef device)
{
	return get_int_property(device, CFSTR(kIOHIDMaxInputReportSizeKey));
}

static int get_string_property(IOHIDDeviceRef device, CFStringRef prop, wchar_t *buf, size_t len)
{
	CFStringRef str;

	if (!len)
		return 0;

	str = (CFStringRef) IOHIDDeviceGetProperty(device, prop);

	buf[0] = 0;

	if (str && CFGetTypeID(str) == CFStringGetTypeID()) {
		CFIndex str_len = CFStringGetLength(str);
		CFRange range;
		CFIndex used_buf_len;
		CFIndex chars_copied;

		len --;

		range.location = 0;
		range.length = ((size_t) str_len > len)? len: (size_t) str_len;
		chars_copied = CFStringGetBytes(str,
			range,
			kCFStringEncodingUTF32LE,
			(char) '?',
			FALSE,
			(UInt8*)buf,
			len * sizeof(wchar_t),
			&used_buf_len);

		if (chars_copied <= 0)
			buf[0] = 0;
		else
			buf[chars_copied] = 0;

		return 0;
	}
	else
		return -1;

}

static int get_serial_number(IOHIDDeviceRef device, wchar_t *buf, size_t len)
{
	return get_string_property(device, CFSTR(kIOHIDSerialNumberKey), buf, len);
}

static int get_manufacturer_string(IOHIDDeviceRef device, wchar_t *buf, size_t len)
{
	return get_string_property(device, CFSTR(kIOHIDManufacturerKey), buf, len);
}

static int get_product_string(IOHIDDeviceRef device, wchar_t *buf, size_t len)
{
	return get_string_property(device, CFSTR(kIOHIDProductKey), buf, len);
}


/* Implementation of wcsdup() for Mac. */
static wchar_t *dup_wcs(const wchar_t *s)
{
	size_t len = wcslen(s);
	wchar_t *ret = (wchar_t*) malloc((len+1)*sizeof(wchar_t));
	wcscpy(ret, s);

	return ret;
}

/* Initialize the IOHIDManager. Return 0 for success and -1 for failure. */
static int init_hid_manager(void)
{
	/* Initialize all the HID Manager Objects */
	hid_mgr = IOHIDManagerCreate(kCFAllocatorDefault, kIOHIDOptionsTypeNone);
	if (hid_mgr) {
		IOHIDManagerSetDeviceMatching(hid_mgr, NULL);
		IOHIDManagerScheduleWithRunLoop(hid_mgr, CFRunLoopGetCurrent(), kCFRunLoopDefaultMode);
		return 0;
	}

	register_global_error("Failed to create IOHIDManager");
	return -1;
}

HID_API_EXPORT const struct hid_api_version* HID_API_CALL hid_version(void)
{
	return &api_version;
}

HID_API_EXPORT const char* HID_API_CALL hid_version_str(void)
{
	return HID_API_VERSION_STR;
}

/* Initialize the IOHIDManager if necessary. This is the public function, and
   it is safe to call this function repeatedly. Return 0 for success and -1
   for failure. */
int HID_API_EXPORT hid_init(void)
{
	register_global_error(NULL);

	if (!hid_mgr) {
		is_macos_10_10_or_greater = (kCFCoreFoundationVersionNumber >= 1151.16); /* kCFCoreFoundationVersionNumber10_10 */
		hid_darwin_set_open_exclusive(1); /* Backward compatibility */
		return init_hid_manager();
	}

	/* Already initialized. */
	return 0;
}

int HID_API_EXPORT hid_exit(void)
{
	if (hid_mgr) {
		/* Close the HID manager. */
		IOHIDManagerClose(hid_mgr, kIOHIDOptionsTypeNone);
		CFRelease(hid_mgr);
		hid_mgr = NULL;
	}

	/* Free global error message */
	register_global_error(NULL);

	return 0;
}

static void process_pending_events(void) {
	SInt32 res;
	do {
		res = CFRunLoopRunInMode(kCFRunLoopDefaultMode, 0.001, FALSE);
	} while(res != kCFRunLoopRunFinished && res != kCFRunLoopRunTimedOut);
}

static int read_usb_interface_from_hid_service_parent(io_service_t hid_service)
{
	int32_t result = -1;
	bool success = false;
	io_registry_entry_t current = IO_OBJECT_NULL;
	kern_return_t res;
	int parent_number = 0;

	res = IORegistryEntryGetParentEntry(hid_service, kIOServicePlane, &current);
	while (KERN_SUCCESS == res
			/* Only search up to 3 parent entries.
			 * With the default driver - the parent-of-interest supposed to be the first one,
			 * but lets assume some custom drivers or so, with deeper tree. */
			&& parent_number < 3) {
		io_registry_entry_t parent = IO_OBJECT_NULL;
		int32_t interface_number = -1;
		parent_number++;

		success = try_get_ioregistry_int_property(current, CFSTR(kUSBInterfaceNumber), &interface_number);
		if (success) {
			result = interface_number;
			break;
		}

		res = IORegistryEntryGetParentEntry(current, kIOServicePlane, &parent);
		if (parent) {
			IOObjectRelease(current);
			current = parent;
		}

	}

	if (current) {
		IOObjectRelease(current);
		current = IO_OBJECT_NULL;
	}

	return result;
}

#ifdef HIDAPI_IGNORE_DEVICE
static hid_bus_type get_bus_type(IOHIDDeviceRef dev)
{
	hid_bus_type bus_type = HID_API_BUS_UNKNOWN;

	CFTypeRef transport_prop = IOHIDDeviceGetProperty(dev, CFSTR(kIOHIDTransportKey));

	if (transport_prop != NULL && CFGetTypeID(transport_prop) == CFStringGetTypeID()) {
		if (CFStringCompare((CFStringRef)transport_prop, CFSTR(kIOHIDTransportUSBValue), 0) == kCFCompareEqualTo) {
			bus_type = HID_API_BUS_USB;
		} else if (CFStringHasPrefix((CFStringRef)transport_prop, CFSTR(kIOHIDTransportBluetoothValue))) {
			bus_type = HID_API_BUS_BLUETOOTH;
		} else if (CFStringCompare((CFStringRef)transport_prop, CFSTR(kIOHIDTransportI2CValue), 0) == kCFCompareEqualTo) {
			bus_type = HID_API_BUS_I2C;
		} else  if (CFStringCompare((CFStringRef)transport_prop, CFSTR(kIOHIDTransportSPIValue), 0) == kCFCompareEqualTo) {
			bus_type = HID_API_BUS_SPI;
		}
	}
	return bus_type;
}
#endif /* HIDAPI_IGNORE_DEVICE */

static struct hid_device_info *create_device_info_with_usage(IOHIDDeviceRef dev, int32_t usage_page, int32_t usage)
{
	unsigned short dev_vid;
	unsigned short dev_pid;
	int BUF_LEN = 256;
	wchar_t buf[BUF_LEN];
	CFTypeRef transport_prop;

	struct hid_device_info *cur_dev;
	io_service_t hid_service;
	kern_return_t res;
	uint64_t entry_id = 0;

	if (dev == NULL) {
		return NULL;
	}

	cur_dev = (struct hid_device_info *)calloc(1, sizeof(struct hid_device_info));
	if (cur_dev == NULL) {
		return NULL;
	}

	dev_vid = get_vendor_id(dev);
	dev_pid = get_product_id(dev);

#ifdef HIDAPI_IGNORE_DEVICE
	/* See if there are any devices we should skip in enumeration */
	if (HIDAPI_IGNORE_DEVICE(get_bus_type(dev), dev_vid, dev_pid, usage_page, usage)) {
		free(cur_dev);
		return NULL;
	}
#endif

	cur_dev->usage_page = usage_page;
	cur_dev->usage = usage;

	/* Fill out the record */
	cur_dev->next = NULL;

	/* Fill in the path (as a unique ID of the service entry) */
	cur_dev->path = NULL;
	hid_service = IOHIDDeviceGetService(dev);
	if (hid_service != MACH_PORT_NULL) {
		res = IORegistryEntryGetRegistryEntryID(hid_service, &entry_id);
	}
	else {
		res = KERN_INVALID_ARGUMENT;
	}

	if (res == KERN_SUCCESS) {
		/* max value of entry_id(uint64_t) is 18446744073709551615 which is 20 characters long,
		   so for (max) "path" string 'DevSrvsID:18446744073709551615' we would need
		   9+1+20+1=31 bytes buffer, but allocate 32 for simple alignment */
		const size_t path_len = 32;
		cur_dev->path = calloc(1, path_len);
		if (cur_dev->path != NULL) {
			snprintf(cur_dev->path, path_len, "DevSrvsID:%llu", entry_id);
		}
	}

	if (cur_dev->path == NULL) {
		/* for whatever reason, trying to keep it a non-NULL string */
		cur_dev->path = strdup("");
	}

	/* Serial Number */
	get_serial_number(dev, buf, BUF_LEN);
	cur_dev->serial_number = dup_wcs(buf);

	/* Manufacturer and Product strings */
	get_manufacturer_string(dev, buf, BUF_LEN);
	cur_dev->manufacturer_string = dup_wcs(buf);
	get_product_string(dev, buf, BUF_LEN);
	cur_dev->product_string = dup_wcs(buf);

	/* VID/PID */
	cur_dev->vendor_id = dev_vid;
	cur_dev->product_id = dev_pid;

	/* Release Number */
	cur_dev->release_number = get_int_property(dev, CFSTR(kIOHIDVersionNumberKey));

	/* Interface Number.
	 * We can only retrieve the interface number for USB HID devices.
	 * See below */
	cur_dev->interface_number = -1;

	/* Bus Type */
	transport_prop = IOHIDDeviceGetProperty(dev, CFSTR(kIOHIDTransportKey));

	if (transport_prop != NULL && CFGetTypeID(transport_prop) == CFStringGetTypeID()) {
		if (CFStringCompare((CFStringRef)transport_prop, CFSTR(kIOHIDTransportUSBValue), 0) == kCFCompareEqualTo) {
			int32_t interface_number = -1;
			cur_dev->bus_type = HID_API_BUS_USB;

			/* A IOHIDDeviceRef used to have this simple property,
			 * until macOS 13.3 - we will try to use it. */
			if (try_get_int_property(dev, CFSTR(kUSBInterfaceNumber), &interface_number)) {
				cur_dev->interface_number = interface_number;
			} else {
				/* Otherwise fallback to io_service_t property.
				 * (of one of the parent services). */
				cur_dev->interface_number = read_usb_interface_from_hid_service_parent(hid_service);

				/* If the above doesn't work -
				 * no (known) fallback exists at this point. */
			}

		/* Match "Bluetooth", "BluetoothLowEnergy" and "Bluetooth Low Energy" strings */
		} else if (CFStringHasPrefix((CFStringRef)transport_prop, CFSTR(kIOHIDTransportBluetoothValue))) {
			cur_dev->bus_type = HID_API_BUS_BLUETOOTH;
		} else if (CFStringCompare((CFStringRef)transport_prop, CFSTR(kIOHIDTransportI2CValue), 0) == kCFCompareEqualTo) {
			cur_dev->bus_type = HID_API_BUS_I2C;
		} else  if (CFStringCompare((CFStringRef)transport_prop, CFSTR(kIOHIDTransportSPIValue), 0) == kCFCompareEqualTo) {
			cur_dev->bus_type = HID_API_BUS_SPI;
		}
	}

	return cur_dev;
}

static struct hid_device_info *create_device_info(IOHIDDeviceRef device)
{
	const int32_t primary_usage_page = get_int_property(device, CFSTR(kIOHIDPrimaryUsagePageKey));
	const int32_t primary_usage = get_int_property(device, CFSTR(kIOHIDPrimaryUsageKey));

	/* Primary should always be first, to match previous behavior. */
	struct hid_device_info *root = create_device_info_with_usage(device, primary_usage_page, primary_usage);
	struct hid_device_info *cur = root;

	CFArrayRef usage_pairs = get_usage_pairs(device);

	if (usage_pairs != NULL) {
		struct hid_device_info *next = NULL;
		for (CFIndex i = 0; i < CFArrayGetCount(usage_pairs); i++) {
			CFTypeRef dict = CFArrayGetValueAtIndex(usage_pairs, i);
			if (CFGetTypeID(dict) != CFDictionaryGetTypeID()) {
				continue;
			}

			CFTypeRef usage_page_ref, usage_ref;
			int32_t usage_page, usage;

			if (!CFDictionaryGetValueIfPresent((CFDictionaryRef)dict, CFSTR(kIOHIDDeviceUsagePageKey), &usage_page_ref) ||
			    !CFDictionaryGetValueIfPresent((CFDictionaryRef)dict, CFSTR(kIOHIDDeviceUsageKey), &usage_ref) ||
					CFGetTypeID(usage_page_ref) != CFNumberGetTypeID() ||
					CFGetTypeID(usage_ref) != CFNumberGetTypeID() ||
					!CFNumberGetValue((CFNumberRef)usage_page_ref, kCFNumberSInt32Type, &usage_page) ||
					!CFNumberGetValue((CFNumberRef)usage_ref, kCFNumberSInt32Type, &usage)) {
					continue;
			}
			if (usage_page == primary_usage_page && usage == primary_usage)
				continue; /* Already added. */

			next = create_device_info_with_usage(device, usage_page, usage);
			if (cur) {
				if (next != NULL) {
					cur->next = next;
					cur = next;
				}
			} else {
				root = cur = next;
			}
		}
	}

	return root;
}

struct hid_device_info  HID_API_EXPORT *hid_enumerate(unsigned short vendor_id, unsigned short product_id)
{
	struct hid_device_info *root = NULL; /* return object */
	struct hid_device_info *cur_dev = NULL;
	CFIndex num_devices;
	int i;

	/* Set up the HID Manager if it hasn't been done */
	if (hid_init() < 0) {
		return NULL;
	}
	/* register_global_error: global error is set/reset by hid_init */

	/* give the IOHIDManager a chance to update itself */
	process_pending_events();

	/* Get a list of the Devices */
	CFMutableDictionaryRef matching = NULL;
	if (vendor_id != 0 || product_id != 0) {
		matching = CFDictionaryCreateMutable(kCFAllocatorDefault, kIOHIDOptionsTypeNone, &kCFTypeDictionaryKeyCallBacks, &kCFTypeDictionaryValueCallBacks);

		if (matching && vendor_id != 0) {
			CFNumberRef v = CFNumberCreate(kCFAllocatorDefault, kCFNumberShortType, &vendor_id);
			CFDictionarySetValue(matching, CFSTR(kIOHIDVendorIDKey), v);
			CFRelease(v);
		}

		if (matching && product_id != 0) {
			CFNumberRef p = CFNumberCreate(kCFAllocatorDefault, kCFNumberShortType, &product_id);
			CFDictionarySetValue(matching, CFSTR(kIOHIDProductIDKey), p);
			CFRelease(p);
		}
	}
	IOHIDManagerSetDeviceMatching(hid_mgr, matching);
	if (matching != NULL) {
		CFRelease(matching);
	}

	CFSetRef device_set = IOHIDManagerCopyDevices(hid_mgr);

	IOHIDDeviceRef *device_array = NULL;

	if (device_set != NULL) {
		/* Convert the list into a C array so we can iterate easily. */
		num_devices = CFSetGetCount(device_set);
		device_array = (IOHIDDeviceRef*) calloc(num_devices, sizeof(IOHIDDeviceRef));
		CFSetGetValues(device_set, (const void **) device_array);
	} else {
		num_devices = 0;
	}

	/* Iterate over each device, making an entry for it. */
	for (i = 0; i < num_devices; i++) {

		IOHIDDeviceRef dev = device_array[i];
		if (!dev) {
			continue;
		}

		struct hid_device_info *tmp = create_device_info(dev);
		if (tmp == NULL) {
			continue;
		}

		if (cur_dev) {
			cur_dev->next = tmp;
		}
		else {
			root = tmp;
		}
		cur_dev = tmp;

		/* move the pointer to the tail of returned list */
		while (cur_dev->next != NULL) {
			cur_dev = cur_dev->next;
		}
	}

	free(device_array);
	if (device_set != NULL)
		CFRelease(device_set);

	if (root == NULL) {
		if (vendor_id == 0 && product_id == 0) {
			register_global_error("No HID devices found in the system.");
		} else {
			register_global_error("No HID devices with requested VID/PID found in the system.");
		}
	}

	return root;
}

void  HID_API_EXPORT hid_free_enumeration(struct hid_device_info *devs)
{
	/* This function is identical to the Linux version. Platform independent. */
	struct hid_device_info *d = devs;
	while (d) {
		struct hid_device_info *next = d->next;
		free(d->path);
		free(d->serial_number);
		free(d->manufacturer_string);
		free(d->product_string);
		free(d);
		d = next;
	}
}

hid_device * HID_API_EXPORT hid_open(unsigned short vendor_id, unsigned short product_id, const wchar_t *serial_number)
{
	/* This function is identical to the Linux version. Platform independent. */

	struct hid_device_info *devs, *cur_dev;
	const char *path_to_open = NULL;
	hid_device * handle = NULL;

	/* register_global_error: global error is reset by hid_enumerate/hid_init */
	devs = hid_enumerate(vendor_id, product_id);
	if (devs == NULL) {
		/* register_global_error: global error is already set by hid_enumerate */
		return NULL;
	}

	cur_dev = devs;
	while (cur_dev) {
		if (cur_dev->vendor_id == vendor_id &&
		    cur_dev->product_id == product_id) {
			if (serial_number) {
				if (wcscmp(serial_number, cur_dev->serial_number) == 0) {
					path_to_open = cur_dev->path;
					break;
				}
			}
			else {
				path_to_open = cur_dev->path;
				break;
			}
		}
		cur_dev = cur_dev->next;
	}

	if (path_to_open) {
		handle = hid_open_path(path_to_open);
	} else {
		register_global_error("Device with requested VID/PID/(SerialNumber) not found");
	}

	hid_free_enumeration(devs);

	return handle;
}

static void hid_device_removal_callback(void *context, IOReturn result,
                                        void *sender)
{
	(void) result;
	(void) sender;

	/* Stop the Run Loop for this device. */
	hid_device *d = (hid_device*) context;

	d->disconnected = 1;
	CFRunLoopStop(d->run_loop);
}

/* The Run Loop calls this function for each input report received.
   This function puts the data into a linked list to be picked up by
   hid_read(). */
static void hid_report_callback(void *context, IOReturn result, void *sender,
                         IOHIDReportType report_type, uint32_t report_id,
                         uint8_t *report, CFIndex report_length)
{
	(void) result;
	(void) sender;
	(void) report_type;
	(void) report_id;

	struct input_report *rpt;
	hid_device *dev = (hid_device*) context;

	/* Make a new Input Report object */
	rpt = (struct input_report*) calloc(1, sizeof(struct input_report));
	rpt->data = (uint8_t*) calloc(1, report_length);
	memcpy(rpt->data, report, report_length);
	rpt->len = report_length;
	rpt->next = NULL;

	/* Lock this section */
	pthread_mutex_lock(&dev->mutex);

	/* Attach the new report object to the end of the list. */
	if (dev->input_reports == NULL) {
		/* The list is empty. Put it at the root. */
		dev->input_reports = rpt;
	}
	else {
		/* Find the end of the list and attach. */
		struct input_report *cur = dev->input_reports;
		int num_queued = 0;
		while (cur->next != NULL) {
			cur = cur->next;
			num_queued++;
		}
		cur->next = rpt;

		/* Pop one off if we've reached 30 in the queue. This
		   way we don't grow forever if the user never reads
		   anything from the device. */
		if (num_queued > 30) {
			return_data(dev, NULL, 0);
		}
	}

	/* Signal a waiting thread that there is data. */
	pthread_cond_signal(&dev->condition);

	/* Unlock */
	pthread_mutex_unlock(&dev->mutex);

}

/* This gets called when the read_thread's run loop gets signaled by
   hid_close(), and serves to stop the read_thread's run loop. */
static void perform_signal_callback(void *context)
{
	hid_device *dev = (hid_device*) context;
	CFRunLoopStop(dev->run_loop); /*TODO: CFRunLoopGetCurrent()*/
}

static void *read_thread(void *param)
{
	hid_device *dev = (hid_device*) param;
	SInt32 code;

	/* Move the device's run loop to this thread. */
	IOHIDDeviceScheduleWithRunLoop(dev->device_handle, CFRunLoopGetCurrent(), dev->run_loop_mode);

	/* Create the RunLoopSource which is used to signal the
	   event loop to stop when hid_close() is called. */
	CFRunLoopSourceContext ctx;
	memset(&ctx, 0, sizeof(ctx));
	ctx.version = 0;
	ctx.info = dev;
	ctx.perform = &perform_signal_callback;
	dev->source = CFRunLoopSourceCreate(kCFAllocatorDefault, 0/*order*/, &ctx);
	CFRunLoopAddSource(CFRunLoopGetCurrent(), dev->source, dev->run_loop_mode);

	/* Store off the Run Loop so it can be stopped from hid_close()
	   and on device disconnection. */
	dev->run_loop = CFRunLoopGetCurrent();

	/* Notify the main thread that the read thread is up and running. */
	pthread_barrier_wait(&dev->barrier);

	/* Run the Event Loop. CFRunLoopRunInMode() will dispatch HID input
	   reports into the hid_report_callback(). */
	while (!dev->shutdown_thread && !dev->disconnected) {
		code = CFRunLoopRunInMode(dev->run_loop_mode, 1000/*sec*/, FALSE);
		/* Return if the device has been disconnected */
		if (code == kCFRunLoopRunFinished || code == kCFRunLoopRunStopped) {
			dev->disconnected = 1;
			break;
		}


		/* Break if The Run Loop returns Finished or Stopped. */
		if (code != kCFRunLoopRunTimedOut &&
		    code != kCFRunLoopRunHandledSource) {
			/* There was some kind of error. Setting
			   shutdown seems to make sense, but
			   there may be something else more appropriate */
			dev->shutdown_thread = 1;
			break;
		}
	}

	/* Now that the read thread is stopping, Wake any threads which are
	   waiting on data (in hid_read_timeout()). Do this under a mutex to
	   make sure that a thread which is about to go to sleep waiting on
	   the condition actually will go to sleep before the condition is
	   signaled. */
	pthread_mutex_lock(&dev->mutex);
	pthread_cond_broadcast(&dev->condition);
	pthread_mutex_unlock(&dev->mutex);

	/* Wait here until hid_close() is called and makes it past
	   the call to CFRunLoopWakeUp(). This thread still needs to
	   be valid when that function is called on the other thread. */
	pthread_barrier_wait(&dev->shutdown_barrier);

	return NULL;
}

/* \p path must be one of:
     - in format 'DevSrvsID:<RegistryEntryID>' (as returned by hid_enumerate);
     - a valid path to an IOHIDDevice in the IOService plane (as returned by IORegistryEntryGetPath,
       e.g.: "IOService:/AppleACPIPlatformExpert/PCI0@0/AppleACPIPCI/EHC1@1D,7/AppleUSBEHCI/PLAYSTATION(R)3 Controller@fd120000/IOUSBInterface@0/IOUSBHIDDriver");
   Second format is for compatibility with paths accepted by older versions of HIDAPI.
*/
static io_registry_entry_t hid_open_service_registry_from_path(const char *path)
{
	if (path == NULL)
		return MACH_PORT_NULL;

	/* Get the IORegistry entry for the given path */
	if (strncmp("DevSrvsID:", path, 10) == 0) {
		char *endptr;
		uint64_t entry_id = strtoull(path + 10, &endptr, 10);
		if (*endptr == '\0') {
			return IOServiceGetMatchingService((mach_port_t) 0, IORegistryEntryIDMatching(entry_id));
		}
	}
	else {
		/* Fallback to older format of the path */
		return IORegistryEntryFromPath((mach_port_t) 0, path);
	}

	return MACH_PORT_NULL;
}

hid_device * HID_API_EXPORT hid_open_path(const char *path)
{
	hid_device *dev = NULL;
	io_registry_entry_t entry = MACH_PORT_NULL;
	IOReturn ret = kIOReturnInvalid;
	char str[32];

	/* Set up the HID Manager if it hasn't been done */
	if (hid_init() < 0) {
		return NULL;
	}
	/* register_global_error: global error is set/reset by hid_init */

	dev = new_hid_device();
	if (!dev) {
		register_global_error("Couldn't allocate memory");
		return NULL;
	}

	/* Get the IORegistry entry for the given path */
	entry = hid_open_service_registry_from_path(path);
	if (entry == MACH_PORT_NULL) {
		/* Path wasn't valid (maybe device was removed?) */
		register_global_error("hid_open_path: device mach entry not found with the given path");
		goto return_error;
	}

	/* Create an IOHIDDevice for the entry */
	dev->device_handle = IOHIDDeviceCreate(kCFAllocatorDefault, entry);
	if (dev->device_handle == NULL) {
		/* Error creating the HID device */
		register_global_error("hid_open_path: failed to create IOHIDDevice from the mach entry");
		goto return_error;
	}

	/* Open the IOHIDDevice */
	ret = IOHIDDeviceOpen(dev->device_handle, dev->open_options);
	if (ret != kIOReturnSuccess) {
		register_global_error_format("hid_open_path: failed to open IOHIDDevice from mach entry: (0x%08X) %s", ret, mach_error_string(ret));
		goto return_error;
	}

	/* Create the buffers for receiving data */
	dev->max_input_report_len = (CFIndex) get_max_report_length(dev->device_handle);
	dev->input_report_buf = (uint8_t*) calloc(dev->max_input_report_len, sizeof(uint8_t));

	/* Create the Run Loop Mode for this device.
	   printing the reference seems to work. */
	snprintf(str, sizeof(str), "HIDAPI_%p", (void*) dev->device_handle);
	dev->run_loop_mode =
		CFStringCreateWithCString(NULL, str, kCFStringEncodingASCII);

	/* Attach the device to a Run Loop */
	IOHIDDeviceRegisterInputReportCallback(
		dev->device_handle, dev->input_report_buf, dev->max_input_report_len,
		&hid_report_callback, dev);
	IOHIDDeviceRegisterRemovalCallback(dev->device_handle, hid_device_removal_callback, dev);

	/* Start the read thread */
	pthread_create(&dev->thread, NULL, read_thread, dev);

	/* Wait here for the read thread to be initialized. */
	pthread_barrier_wait(&dev->barrier);

	IOObjectRelease(entry);
	return dev;

return_error:
	if (dev->device_handle != NULL)
		CFRelease(dev->device_handle);

	if (entry != MACH_PORT_NULL)
		IOObjectRelease(entry);

	free_hid_device(dev);
	return NULL;
}

static int set_report(hid_device *dev, IOHIDReportType type, const unsigned char *data, size_t length)
{
	const unsigned char *data_to_send = data;
	CFIndex length_to_send = length;
	IOReturn res;
	unsigned char report_id;

	register_device_error(dev, NULL);

	if (!data || (length == 0)) {
		register_device_error(dev, strerror(EINVAL));
		return -1;
	}

	report_id = data[0];

	if (report_id == 0x0) {
		/* Not using numbered Reports.
		   Don't send the report number. */
		data_to_send = data+1;
		length_to_send = length-1;
	}

	/* Avoid crash if the device has been unplugged. */
	if (dev->disconnected) {
		register_device_error(dev, "Device is disconnected");
		return -1;
	}

	res = IOHIDDeviceSetReport(dev->device_handle,
	                           type,
	                           report_id,
	                           data_to_send, length_to_send);

	if (res != kIOReturnSuccess) {
		register_device_error_format(dev, "IOHIDDeviceSetReport failed: (0x%08X) %s", res, mach_error_string(res));
		return -1;
	}

	return (int) length;
}

static int get_report(hid_device *dev, IOHIDReportType type, unsigned char *data, size_t length)
{
	unsigned char *report = data;
	CFIndex report_length = length;
	IOReturn res = kIOReturnSuccess;
	const unsigned char report_id = data[0];

	register_device_error(dev, NULL);

	if (report_id == 0x0) {
		/* Not using numbered Reports.
		   Don't send the report number. */
		report = data+1;
		report_length = length-1;
	}

	/* Avoid crash if the device has been unplugged. */
	if (dev->disconnected) {
		register_device_error(dev, "Device is disconnected");
		return -1;
	}

	res = IOHIDDeviceGetReport(dev->device_handle,
	                           type,
	                           report_id,
	                           report, &report_length);

	if (res != kIOReturnSuccess) {
		register_device_error_format(dev, "IOHIDDeviceGetReport failed: (0x%08X) %s", res, mach_error_string(res));
		return -1;
	}

	if (report_id == 0x0) { /* 0 report number still present at the beginning */
		report_length++;
	}

	return (int) report_length;
}

int HID_API_EXPORT hid_write(hid_device *dev, const unsigned char *data, size_t length)
{
	return set_report(dev, kIOHIDReportTypeOutput, data, length);
}

/* Helper function, so that this isn't duplicated in hid_read(). */
static int return_data(hid_device *dev, unsigned char *data, size_t length)
{
	/* Copy the data out of the linked list item (rpt) into the
	   return buffer (data), and delete the liked list item. */
	struct input_report *rpt = dev->input_reports;
	size_t len = (length < rpt->len)? length: rpt->len;
	if (data != NULL) {
		memcpy(data, rpt->data, len);
	}
	dev->input_reports = rpt->next;
	free(rpt->data);
	free(rpt);
	return (int) len;
}

static int cond_wait(hid_device *dev, pthread_cond_t *cond, pthread_mutex_t *mutex)
{
	while (!dev->input_reports) {
		int res = pthread_cond_wait(cond, mutex);
		if (res != 0)
			return res;

		/* A res of 0 means we may have been signaled or it may
		   be a spurious wakeup. Check to see that there's actually
		   data in the queue before returning, and if not, go back
		   to sleep. See the pthread_cond_timedwait() man page for
		   details. */

		if (dev->shutdown_thread || dev->disconnected) {
			return -1;
		}
	}

	return 0;
}

static int cond_timedwait(hid_device *dev, pthread_cond_t *cond, pthread_mutex_t *mutex, const struct timespec *abstime)
{
	while (!dev->input_reports) {
		int res = pthread_cond_timedwait(cond, mutex, abstime);
		if (res != 0)
			return res;

		/* A res of 0 means we may have been signaled or it may
		   be a spurious wakeup. Check to see that there's actually
		   data in the queue before returning, and if not, go back
		   to sleep. See the pthread_cond_timedwait() man page for
		   details. */

		if (dev->shutdown_thread || dev->disconnected) {
			return -1;
		}
	}

	return 0;

}

int HID_API_EXPORT hid_read_timeout(hid_device *dev, unsigned char *data, size_t length, int milliseconds)
{
	int bytes_read = -1;

	/* Lock the access to the report list. */
	pthread_mutex_lock(&dev->mutex);

	/* There's an input report queued up. Return it. */
	if (dev->input_reports) {
		/* Return the first one */
		bytes_read = return_data(dev, data, length);
		goto ret;
	}

	/* Return if the device has been disconnected. */
	if (dev->disconnected) {
		bytes_read = -1;
		register_device_error(dev, "hid_read_timeout: device disconnected");
		goto ret;
	}

	if (dev->shutdown_thread) {
		/* This means the device has been closed (or there
		   has been an error. An error code of -1 should
		   be returned. */
		bytes_read = -1;
		register_device_error(dev, "hid_read_timeout: thread shutdown");
		goto ret;
	}

	/* There is no data. Go to sleep and wait for data. */

	if (milliseconds == -1) {
		/* Blocking */
		int res;
		res = cond_wait(dev, &dev->condition, &dev->mutex);
		if (res == 0)
			bytes_read = return_data(dev, data, length);
		else {
			/* There was an error, or a device disconnection. */
			register_device_error(dev, "hid_read_timeout: error waiting for more data");
			bytes_read = -1;
		}
	}
	else if (milliseconds > 0) {
		/* Non-blocking, but called with timeout. */
		int res;
		struct timespec ts;
		struct timeval tv;
		gettimeofday(&tv, NULL);
		TIMEVAL_TO_TIMESPEC(&tv, &ts);
		ts.tv_sec += milliseconds / 1000;
		ts.tv_nsec += (milliseconds % 1000) * 1000000;
		if (ts.tv_nsec >= 1000000000L) {
			ts.tv_sec++;
			ts.tv_nsec -= 1000000000L;
		}

		res = cond_timedwait(dev, &dev->condition, &dev->mutex, &ts);
		if (res == 0) {
			bytes_read = return_data(dev, data, length);
		} else if (res == ETIMEDOUT) {
			bytes_read = 0;
		} else {
			register_device_error(dev, "hid_read_timeout:  error waiting for more data");
			bytes_read = -1;
		}
	}
	else {
		/* Purely non-blocking */
		bytes_read = 0;
	}

ret:
	/* Unlock */
	pthread_mutex_unlock(&dev->mutex);
	return bytes_read;
}

int HID_API_EXPORT hid_read(hid_device *dev, unsigned char *data, size_t length)
{
	return hid_read_timeout(dev, data, length, (dev->blocking)? -1: 0);
}

int HID_API_EXPORT hid_set_nonblocking(hid_device *dev, int nonblock)
{
	/* All Nonblocking operation is handled by the library. */
	dev->blocking = !nonblock;

	return 0;
}

int HID_API_EXPORT hid_send_feature_report(hid_device *dev, const unsigned char *data, size_t length)
{
	return set_report(dev, kIOHIDReportTypeFeature, data, length);
}

int HID_API_EXPORT hid_get_feature_report(hid_device *dev, unsigned char *data, size_t length)
{
	return get_report(dev, kIOHIDReportTypeFeature, data, length);
}

int HID_API_EXPORT HID_API_CALL hid_get_input_report(hid_device *dev, unsigned char *data, size_t length)
{
	return get_report(dev, kIOHIDReportTypeInput, data, length);
}

void HID_API_EXPORT hid_close(hid_device *dev)
{
	if (!dev)
		return;

	/* Disconnect the report callback before close.
	   See comment below.
	*/
	if (is_macos_10_10_or_greater || !dev->disconnected) {
		IOHIDDeviceRegisterInputReportCallback(
			dev->device_handle, dev->input_report_buf, dev->max_input_report_len,
			NULL, dev);
		IOHIDDeviceRegisterRemovalCallback(dev->device_handle, NULL, dev);
		IOHIDDeviceUnscheduleFromRunLoop(dev->device_handle, dev->run_loop, dev->run_loop_mode);
		IOHIDDeviceScheduleWithRunLoop(dev->device_handle, CFRunLoopGetMain(), kCFRunLoopDefaultMode);
	}

	/* Cause read_thread() to stop. */
	dev->shutdown_thread = 1;

	/* Wake up the run thread's event loop so that the thread can exit. */
	CFRunLoopSourceSignal(dev->source);
	CFRunLoopWakeUp(dev->run_loop);

	/* Notify the read thread that it can shut down now. */
	pthread_barrier_wait(&dev->shutdown_barrier);

	/* Wait for read_thread() to end. */
	pthread_join(dev->thread, NULL);

	/* Close the OS handle to the device, but only if it's not
	   been unplugged. If it's been unplugged, then calling
	   IOHIDDeviceClose() will crash.

	   UPD: The crash part was true in/until some version of macOS.
	   Starting with macOS 10.15, there is an opposite effect in some environments:
	   crash happenes if IOHIDDeviceClose() is not called.
	   Not leaking a resource in all tested environments.
	*/
	if (is_macos_10_10_or_greater || !dev->disconnected) {
		IOHIDDeviceClose(dev->device_handle, dev->open_options);
	}

	/* Clear out the queue of received reports. */
	pthread_mutex_lock(&dev->mutex);
	while (dev->input_reports) {
		return_data(dev, NULL, 0);
	}
	pthread_mutex_unlock(&dev->mutex);
	CFRelease(dev->device_handle);

	free_hid_device(dev);
}

int HID_API_EXPORT_CALL hid_get_manufacturer_string(hid_device *dev, wchar_t *string, size_t maxlen)
{
	if (!string || !maxlen)
	{
		register_device_error(dev, "Zero buffer/length");
		return -1;
	}

	struct hid_device_info *info = hid_get_device_info(dev);
	if (!info)
	{
		// hid_get_device_info will have set an error already
		return -1;
	}

	wcsncpy(string, info->manufacturer_string, maxlen);
	string[maxlen - 1] = L'\0';

	return 0;
}

int HID_API_EXPORT_CALL hid_get_product_string(hid_device *dev, wchar_t *string, size_t maxlen)
{
	if (!string || !maxlen) {
		register_device_error(dev, "Zero buffer/length");
		return -1;
	}

	struct hid_device_info *info = hid_get_device_info(dev);
	if (!info) {
		// hid_get_device_info will have set an error already
		return -1;
	}

	wcsncpy(string, info->product_string, maxlen);
	string[maxlen - 1] = L'\0';

	return 0;
}

int HID_API_EXPORT_CALL hid_get_serial_number_string(hid_device *dev, wchar_t *string, size_t maxlen)
{
	if (!string || !maxlen) {
		register_device_error(dev, "Zero buffer/length");
		return -1;
	}

	struct hid_device_info *info = hid_get_device_info(dev);
	if (!info) {
		// hid_get_device_info will have set an error already
		return -1;
	}

	wcsncpy(string, info->serial_number, maxlen);
	string[maxlen - 1] = L'\0';

	return 0;
}

HID_API_EXPORT struct hid_device_info *HID_API_CALL hid_get_device_info(hid_device *dev) {
	if (!dev->device_info) {
		dev->device_info = create_device_info(dev->device_handle);
		if (!dev->device_info) {
			register_device_error(dev, "Failed to create hid_device_info");
		}
	}

	return dev->device_info;
}

int HID_API_EXPORT_CALL hid_get_indexed_string(hid_device *dev, int string_index, wchar_t *string, size_t maxlen)
{
	(void) dev;
	(void) string_index;
	(void) string;
	(void) maxlen;

	register_device_error(dev, "hid_get_indexed_string: not available on this platform");
	return -1;
}

int HID_API_EXPORT_CALL hid_darwin_get_location_id(hid_device *dev, uint32_t *location_id)
{
	int res = get_int_property(dev->device_handle, CFSTR(kIOHIDLocationIDKey));
	if (res != 0) {
		*location_id = (uint32_t) res;
		return 0;
	} else {
		register_device_error(dev, "Failed to get IOHIDLocationID property");
		return -1;
	}
}

void HID_API_EXPORT_CALL hid_darwin_set_open_exclusive(int open_exclusive)
{
	device_open_options = (open_exclusive == 0) ? kIOHIDOptionsTypeNone : kIOHIDOptionsTypeSeizeDevice;
}

int HID_API_EXPORT_CALL hid_darwin_get_open_exclusive(void)
{
	return (device_open_options == kIOHIDOptionsTypeSeizeDevice) ? 1 : 0;
}

int HID_API_EXPORT_CALL hid_darwin_is_device_open_exclusive(hid_device *dev)
{
	if (!dev)
		return -1;

	return (dev->open_options == kIOHIDOptionsTypeSeizeDevice) ? 1 : 0;
}

int HID_API_EXPORT_CALL hid_get_report_descriptor(hid_device *dev, unsigned char *buf, size_t buf_size)
{
	CFTypeRef ref = IOHIDDeviceGetProperty(dev->device_handle, CFSTR(kIOHIDReportDescriptorKey));
	if (ref != NULL && CFGetTypeID(ref) == CFDataGetTypeID()) {
		CFDataRef report_descriptor = (CFDataRef) ref;
		const UInt8 *descriptor_buf = CFDataGetBytePtr(report_descriptor);
		CFIndex descriptor_buf_len = CFDataGetLength(report_descriptor);
		size_t copy_len = (size_t) descriptor_buf_len;

		if (descriptor_buf == NULL || descriptor_buf_len < 0) {
			register_device_error(dev, "Zero buffer/length");
			return -1;
		}

		if (buf_size < copy_len) {
			copy_len = buf_size;
		}

		memcpy(buf, descriptor_buf, copy_len);
		return (int)copy_len;
	}
	else {
		register_device_error(dev, "Failed to get kIOHIDReportDescriptorKey property");
		return -1;
	}
}

HID_API_EXPORT const wchar_t * HID_API_CALL  hid_error(hid_device *dev)
{
	if (dev) {
		if (dev->last_error_str == NULL)
			return L"Success";
		return dev->last_error_str;
	}

	if (last_global_error_str == NULL)
		return L"Success";
	return last_global_error_str;
}
