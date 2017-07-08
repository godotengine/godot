/*
 * Internal header for libusb
 * Copyright © 2007-2009 Daniel Drake <dsd@gentoo.org>
 * Copyright © 2001 Johannes Erdfelt <johannes@erdfelt.com>
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

#ifndef LIBUSBI_H
#define LIBUSBI_H

#include <config.h>

#include <stdlib.h>

#include <stddef.h>
#include <stdint.h>
#include <time.h>
#include <stdarg.h>
#ifdef HAVE_POLL_H
#include <poll.h>
#endif
#ifdef HAVE_MISSING_H
#include <missing.h>
#endif

#include "libusb.h"
#include "version.h"

/* Attribute to ensure that a structure member is aligned to a natural
 * pointer alignment. Used for os_priv member. */
#if defined(_MSC_VER)
#if defined(_WIN64)
#define PTR_ALIGNED __declspec(align(8))
#else
#define PTR_ALIGNED __declspec(align(4))
#endif
#elif defined(__GNUC__)
#define PTR_ALIGNED __attribute__((aligned(sizeof(void *))))
#else
#define PTR_ALIGNED
#endif

/* Inside the libusb code, mark all public functions as follows:
 *   return_type API_EXPORTED function_name(params) { ... }
 * But if the function returns a pointer, mark it as follows:
 *   DEFAULT_VISIBILITY return_type * LIBUSB_CALL function_name(params) { ... }
 * In the libusb public header, mark all declarations as:
 *   return_type LIBUSB_CALL function_name(params);
 */
#define API_EXPORTED LIBUSB_CALL DEFAULT_VISIBILITY

#ifdef __cplusplus
extern "C" {
#endif

#define DEVICE_DESC_LENGTH	18

#define USB_MAXENDPOINTS	32
#define USB_MAXINTERFACES	32
#define USB_MAXCONFIG		8

/* Backend specific capabilities */
#define USBI_CAP_HAS_HID_ACCESS			0x00010000
#define USBI_CAP_SUPPORTS_DETACH_KERNEL_DRIVER	0x00020000

/* Maximum number of bytes in a log line */
#define USBI_MAX_LOG_LEN	1024
/* Terminator for log lines */
#define USBI_LOG_LINE_END	"\n"

/* The following is used to silence warnings for unused variables */
#define UNUSED(var)		do { (void)(var); } while(0)

#if !defined(ARRAYSIZE)
#define ARRAYSIZE(array) (sizeof(array) / sizeof(array[0]))
#endif

struct list_head {
	struct list_head *prev, *next;
};

/* Get an entry from the list
 *  ptr - the address of this list_head element in "type"
 *  type - the data type that contains "member"
 *  member - the list_head element in "type"
 */
#define list_entry(ptr, type, member) \
	((type *)((uintptr_t)(ptr) - (uintptr_t)offsetof(type, member)))

#define list_first_entry(ptr, type, member) \
	list_entry((ptr)->next, type, member)

/* Get each entry from a list
 *  pos - A structure pointer has a "member" element
 *  head - list head
 *  member - the list_head element in "pos"
 *  type - the type of the first parameter
 */
#define list_for_each_entry(pos, head, member, type)			\
	for (pos = list_entry((head)->next, type, member);		\
		 &pos->member != (head);				\
		 pos = list_entry(pos->member.next, type, member))

#define list_for_each_entry_safe(pos, n, head, member, type)		\
	for (pos = list_entry((head)->next, type, member),		\
		 n = list_entry(pos->member.next, type, member);	\
		 &pos->member != (head);				\
		 pos = n, n = list_entry(n->member.next, type, member))

#define list_empty(entry) ((entry)->next == (entry))

static inline void list_init(struct list_head *entry)
{
	entry->prev = entry->next = entry;
}

static inline void list_add(struct list_head *entry, struct list_head *head)
{
	entry->next = head->next;
	entry->prev = head;

	head->next->prev = entry;
	head->next = entry;
}

static inline void list_add_tail(struct list_head *entry,
	struct list_head *head)
{
	entry->next = head;
	entry->prev = head->prev;

	head->prev->next = entry;
	head->prev = entry;
}

static inline void list_del(struct list_head *entry)
{
	entry->next->prev = entry->prev;
	entry->prev->next = entry->next;
	entry->next = entry->prev = NULL;
}

static inline void list_cut(struct list_head *list, struct list_head *head)
{
	if (list_empty(head))
		return;

	list->next = head->next;
	list->next->prev = list;
	list->prev = head->prev;
	list->prev->next = list;

	list_init(head);
}

static inline void *usbi_reallocf(void *ptr, size_t size)
{
	void *ret = realloc(ptr, size);
	if (!ret)
		free(ptr);
	return ret;
}

#define container_of(ptr, type, member) ({			\
	const typeof( ((type *)0)->member ) *mptr = (ptr);	\
	(type *)( (char *)mptr - offsetof(type,member) );})

#ifndef MIN
#define MIN(a, b)	((a) < (b) ? (a) : (b))
#endif
#ifndef MAX
#define MAX(a, b)	((a) > (b) ? (a) : (b))
#endif

#define TIMESPEC_IS_SET(ts) ((ts)->tv_sec != 0 || (ts)->tv_nsec != 0)

#if defined(_WIN32) || defined(__CYGWIN__) || defined(_WIN32_WCE)
#define TIMEVAL_TV_SEC_TYPE	long
#else
#define TIMEVAL_TV_SEC_TYPE	time_t
#endif

/* Some platforms don't have this define */
#ifndef TIMESPEC_TO_TIMEVAL
#define TIMESPEC_TO_TIMEVAL(tv, ts)					\
	do {								\
		(tv)->tv_sec = (TIMEVAL_TV_SEC_TYPE) (ts)->tv_sec;	\
		(tv)->tv_usec = (ts)->tv_nsec / 1000;			\
	} while (0)
#endif

void usbi_log(struct libusb_context *ctx, enum libusb_log_level level,
	const char *function, const char *format, ...);

void usbi_log_v(struct libusb_context *ctx, enum libusb_log_level level,
	const char *function, const char *format, va_list args);

#if !defined(_MSC_VER) || _MSC_VER >= 1400

#ifdef ENABLE_LOGGING
#define _usbi_log(ctx, level, ...) usbi_log(ctx, level, __FUNCTION__, __VA_ARGS__)
#define usbi_dbg(...) _usbi_log(NULL, LIBUSB_LOG_LEVEL_DEBUG, __VA_ARGS__)
#else
#define _usbi_log(ctx, level, ...) do { (void)(ctx); } while(0)
#define usbi_dbg(...) do {} while(0)
#endif

#define usbi_info(ctx, ...) _usbi_log(ctx, LIBUSB_LOG_LEVEL_INFO, __VA_ARGS__)
#define usbi_warn(ctx, ...) _usbi_log(ctx, LIBUSB_LOG_LEVEL_WARNING, __VA_ARGS__)
#define usbi_err(ctx, ...) _usbi_log(ctx, LIBUSB_LOG_LEVEL_ERROR, __VA_ARGS__)

#else /* !defined(_MSC_VER) || _MSC_VER >= 1400 */

#ifdef ENABLE_LOGGING
#define LOG_BODY(ctxt, level)				\
{							\
	va_list args;					\
	va_start(args, format);				\
	usbi_log_v(ctxt, level, "", format, args);	\
	va_end(args);					\
}
#else
#define LOG_BODY(ctxt, level)				\
{							\
	(void)(ctxt);					\
}
#endif

static inline void usbi_info(struct libusb_context *ctx, const char *format, ...)
	LOG_BODY(ctx, LIBUSB_LOG_LEVEL_INFO)
static inline void usbi_warn(struct libusb_context *ctx, const char *format, ...)
	LOG_BODY(ctx, LIBUSB_LOG_LEVEL_WARNING)
static inline void usbi_err(struct libusb_context *ctx, const char *format, ...)
	LOG_BODY(ctx, LIBUSB_LOG_LEVEL_ERROR)

static inline void usbi_dbg(const char *format, ...)
	LOG_BODY(NULL, LIBUSB_LOG_LEVEL_DEBUG)

#endif /* !defined(_MSC_VER) || _MSC_VER >= 1400 */

#define USBI_GET_CONTEXT(ctx)				\
	do {						\
		if (!(ctx))				\
			(ctx) = usbi_default_context;	\
	} while(0)

#define DEVICE_CTX(dev)		((dev)->ctx)
#define HANDLE_CTX(handle)	(DEVICE_CTX((handle)->dev))
#define TRANSFER_CTX(transfer)	(HANDLE_CTX((transfer)->dev_handle))
#define ITRANSFER_CTX(transfer) \
	(TRANSFER_CTX(USBI_TRANSFER_TO_LIBUSB_TRANSFER(transfer)))

#define IS_EPIN(ep)		(0 != ((ep) & LIBUSB_ENDPOINT_IN))
#define IS_EPOUT(ep)		(!IS_EPIN(ep))
#define IS_XFERIN(xfer)		(0 != ((xfer)->endpoint & LIBUSB_ENDPOINT_IN))
#define IS_XFEROUT(xfer)	(!IS_XFERIN(xfer))

/* Internal abstraction for thread synchronization */
#if defined(THREADS_POSIX)
#include "os/threads_posix.h"
#elif defined(OS_WINDOWS) || defined(OS_WINCE)
#include "os/threads_windows.h"
#endif

extern struct libusb_context *usbi_default_context;

/* Forward declaration for use in context (fully defined inside poll abstraction) */
struct pollfd;

struct libusb_context {
	int debug;
	int debug_fixed;

	/* internal event pipe, used for signalling occurrence of an internal event. */
	int event_pipe[2];

	struct list_head usb_devs;
	usbi_mutex_t usb_devs_lock;

	/* A list of open handles. Backends are free to traverse this if required.
	 */
	struct list_head open_devs;
	usbi_mutex_t open_devs_lock;

	/* A list of registered hotplug callbacks */
	struct list_head hotplug_cbs;
	usbi_mutex_t hotplug_cbs_lock;

	/* this is a list of in-flight transfer handles, sorted by timeout
	 * expiration. URBs to timeout the soonest are placed at the beginning of
	 * the list, URBs that will time out later are placed after, and urbs with
	 * infinite timeout are always placed at the very end. */
	struct list_head flying_transfers;
	/* Note paths taking both this and usbi_transfer->lock must always
	 * take this lock first */
	usbi_mutex_t flying_transfers_lock;

	/* user callbacks for pollfd changes */
	libusb_pollfd_added_cb fd_added_cb;
	libusb_pollfd_removed_cb fd_removed_cb;
	void *fd_cb_user_data;

	/* ensures that only one thread is handling events at any one time */
	usbi_mutex_t events_lock;

	/* used to see if there is an active thread doing event handling */
	int event_handler_active;

	/* A thread-local storage key to track which thread is performing event
	 * handling */
	usbi_tls_key_t event_handling_key;

	/* used to wait for event completion in threads other than the one that is
	 * event handling */
	usbi_mutex_t event_waiters_lock;
	usbi_cond_t event_waiters_cond;

	/* A lock to protect internal context event data. */
	usbi_mutex_t event_data_lock;

	/* A bitmask of flags that are set to indicate specific events that need to
	 * be handled. Protected by event_data_lock. */
	unsigned int event_flags;

	/* A counter that is set when we want to interrupt and prevent event handling,
	 * in order to safely close a device. Protected by event_data_lock. */
	unsigned int device_close;

	/* list and count of poll fds and an array of poll fd structures that is
	 * (re)allocated as necessary prior to polling. Protected by event_data_lock. */
	struct list_head ipollfds;
	struct pollfd *pollfds;
	POLL_NFDS_TYPE pollfds_cnt;

	/* A list of pending hotplug messages. Protected by event_data_lock. */
	struct list_head hotplug_msgs;

	/* A list of pending completed transfers. Protected by event_data_lock. */
	struct list_head completed_transfers;

#ifdef USBI_TIMERFD_AVAILABLE
	/* used for timeout handling, if supported by OS.
	 * this timerfd is maintained to trigger on the next pending timeout */
	int timerfd;
#endif

	struct list_head list;

	PTR_ALIGNED unsigned char os_priv[ZERO_SIZED_ARRAY];
};

enum usbi_event_flags {
	/* The list of pollfds has been modified */
	USBI_EVENT_POLLFDS_MODIFIED = 1 << 0,

	/* The user has interrupted the event handler */
	USBI_EVENT_USER_INTERRUPT = 1 << 1,
};

/* Macros for managing event handling state */
#define usbi_handling_events(ctx) \
	(usbi_tls_key_get((ctx)->event_handling_key) != NULL)

#define usbi_start_event_handling(ctx) \
	usbi_tls_key_set((ctx)->event_handling_key, ctx)

#define usbi_end_event_handling(ctx) \
	usbi_tls_key_set((ctx)->event_handling_key, NULL)

/* Update the following macro if new event sources are added */
#define usbi_pending_events(ctx) \
	((ctx)->event_flags || (ctx)->device_close \
	 || !list_empty(&(ctx)->hotplug_msgs) || !list_empty(&(ctx)->completed_transfers))

#ifdef USBI_TIMERFD_AVAILABLE
#define usbi_using_timerfd(ctx) ((ctx)->timerfd >= 0)
#else
#define usbi_using_timerfd(ctx) (0)
#endif

struct libusb_device {
	/* lock protects refcnt, everything else is finalized at initialization
	 * time */
	usbi_mutex_t lock;
	int refcnt;

	struct libusb_context *ctx;

	uint8_t bus_number;
	uint8_t port_number;
	struct libusb_device* parent_dev;
	uint8_t device_address;
	uint8_t num_configurations;
	enum libusb_speed speed;

	struct list_head list;
	unsigned long session_data;

	struct libusb_device_descriptor device_descriptor;
	int attached;

	PTR_ALIGNED unsigned char os_priv[ZERO_SIZED_ARRAY];
};

struct libusb_device_handle {
	/* lock protects claimed_interfaces */
	usbi_mutex_t lock;
	unsigned long claimed_interfaces;

	struct list_head list;
	struct libusb_device *dev;
	int auto_detach_kernel_driver;

	PTR_ALIGNED unsigned char os_priv[ZERO_SIZED_ARRAY];
};

enum {
	USBI_CLOCK_MONOTONIC,
	USBI_CLOCK_REALTIME
};

/* in-memory transfer layout:
 *
 * 1. struct usbi_transfer
 * 2. struct libusb_transfer (which includes iso packets) [variable size]
 * 3. os private data [variable size]
 *
 * from a libusb_transfer, you can get the usbi_transfer by rewinding the
 * appropriate number of bytes.
 * the usbi_transfer includes the number of allocated packets, so you can
 * determine the size of the transfer and hence the start and length of the
 * OS-private data.
 */

struct usbi_transfer {
	int num_iso_packets;
	struct list_head list;
	struct list_head completed_list;
	struct timeval timeout;
	int transferred;
	uint32_t stream_id;
	uint8_t state_flags;   /* Protected by usbi_transfer->lock */
	uint8_t timeout_flags; /* Protected by the flying_stransfers_lock */

	/* this lock is held during libusb_submit_transfer() and
	 * libusb_cancel_transfer() (allowing the OS backend to prevent duplicate
	 * cancellation, submission-during-cancellation, etc). the OS backend
	 * should also take this lock in the handle_events path, to prevent the user
	 * cancelling the transfer from another thread while you are processing
	 * its completion (presumably there would be races within your OS backend
	 * if this were possible).
	 * Note paths taking both this and the flying_transfers_lock must
	 * always take the flying_transfers_lock first */
	usbi_mutex_t lock;
};

enum usbi_transfer_state_flags {
	/* Transfer successfully submitted by backend */
	USBI_TRANSFER_IN_FLIGHT = 1 << 0,

	/* Cancellation was requested via libusb_cancel_transfer() */
	USBI_TRANSFER_CANCELLING = 1 << 1,

	/* Operation on the transfer failed because the device disappeared */
	USBI_TRANSFER_DEVICE_DISAPPEARED = 1 << 2,
};

enum usbi_transfer_timeout_flags {
	/* Set by backend submit_transfer() if the OS handles timeout */
	USBI_TRANSFER_OS_HANDLES_TIMEOUT = 1 << 0,

	/* The transfer timeout has been handled */
	USBI_TRANSFER_TIMEOUT_HANDLED = 1 << 1,

	/* The transfer timeout was successfully processed */
	USBI_TRANSFER_TIMED_OUT = 1 << 2,
};

#define USBI_TRANSFER_TO_LIBUSB_TRANSFER(transfer)			\
	((struct libusb_transfer *)(((unsigned char *)(transfer))	\
		+ sizeof(struct usbi_transfer)))
#define LIBUSB_TRANSFER_TO_USBI_TRANSFER(transfer)			\
	((struct usbi_transfer *)(((unsigned char *)(transfer))		\
		- sizeof(struct usbi_transfer)))

static inline void *usbi_transfer_get_os_priv(struct usbi_transfer *transfer)
{
	return ((unsigned char *)transfer) + sizeof(struct usbi_transfer)
		+ sizeof(struct libusb_transfer)
		+ (transfer->num_iso_packets
			* sizeof(struct libusb_iso_packet_descriptor));
}

/* bus structures */

/* All standard descriptors have these 2 fields in common */
struct usb_descriptor_header {
	uint8_t bLength;
	uint8_t bDescriptorType;
};

/* shared data and functions */

int usbi_io_init(struct libusb_context *ctx);
void usbi_io_exit(struct libusb_context *ctx);

struct libusb_device *usbi_alloc_device(struct libusb_context *ctx,
	unsigned long session_id);
struct libusb_device *usbi_get_device_by_session_id(struct libusb_context *ctx,
	unsigned long session_id);
int usbi_sanitize_device(struct libusb_device *dev);
void usbi_handle_disconnect(struct libusb_device_handle *dev_handle);

int usbi_handle_transfer_completion(struct usbi_transfer *itransfer,
	enum libusb_transfer_status status);
int usbi_handle_transfer_cancellation(struct usbi_transfer *transfer);
void usbi_signal_transfer_completion(struct usbi_transfer *transfer);

int usbi_parse_descriptor(const unsigned char *source, const char *descriptor,
	void *dest, int host_endian);
int usbi_device_cache_descriptor(libusb_device *dev);
int usbi_get_config_index_by_value(struct libusb_device *dev,
	uint8_t bConfigurationValue, int *idx);

void usbi_connect_device (struct libusb_device *dev);
void usbi_disconnect_device (struct libusb_device *dev);

int usbi_signal_event(struct libusb_context *ctx);
int usbi_clear_event(struct libusb_context *ctx);

/* Internal abstraction for poll (needs struct usbi_transfer on Windows) */
#if defined(OS_LINUX) || defined(OS_DARWIN) || defined(OS_OPENBSD) || defined(OS_NETBSD) ||\
	defined(OS_HAIKU) || defined(OS_SUNOS)
#include <unistd.h>
#include "os/poll_posix.h"
#elif defined(OS_WINDOWS) || defined(OS_WINCE)
#include "os/poll_windows.h"
#endif

#if defined(_MSC_VER) && (_MSC_VER < 1900)
#define snprintf usbi_snprintf
#define vsnprintf usbi_vsnprintf
int usbi_snprintf(char *dst, size_t size, const char *format, ...);
int usbi_vsnprintf(char *dst, size_t size, const char *format, va_list ap);
#define LIBUSB_PRINTF_WIN32
#endif

struct usbi_pollfd {
	/* must come first */
	struct libusb_pollfd pollfd;

	struct list_head list;
};

int usbi_add_pollfd(struct libusb_context *ctx, int fd, short events);
void usbi_remove_pollfd(struct libusb_context *ctx, int fd);

/* device discovery */

/* we traverse usbfs without knowing how many devices we are going to find.
 * so we create this discovered_devs model which is similar to a linked-list
 * which grows when required. it can be freed once discovery has completed,
 * eliminating the need for a list node in the libusb_device structure
 * itself. */
struct discovered_devs {
	size_t len;
	size_t capacity;
	struct libusb_device *devices[ZERO_SIZED_ARRAY];
};

struct discovered_devs *discovered_devs_append(
	struct discovered_devs *discdevs, struct libusb_device *dev);

/* OS abstraction */

/* This is the interface that OS backends need to implement.
 * All fields are mandatory, except ones explicitly noted as optional. */
struct usbi_os_backend {
	/* A human-readable name for your backend, e.g. "Linux usbfs" */
	const char *name;

	/* Binary mask for backend specific capabilities */
	uint32_t caps;

	/* Perform initialization of your backend. You might use this function
	 * to determine specific capabilities of the system, allocate required
	 * data structures for later, etc.
	 *
	 * This function is called when a libusb user initializes the library
	 * prior to use.
	 *
	 * Return 0 on success, or a LIBUSB_ERROR code on failure.
	 */
	int (*init)(struct libusb_context *ctx);

	/* Deinitialization. Optional. This function should destroy anything
	 * that was set up by init.
	 *
	 * This function is called when the user deinitializes the library.
	 */
	void (*exit)(void);

	/* Enumerate all the USB devices on the system, returning them in a list
	 * of discovered devices.
	 *
	 * Your implementation should enumerate all devices on the system,
	 * regardless of whether they have been seen before or not.
	 *
	 * When you have found a device, compute a session ID for it. The session
	 * ID should uniquely represent that particular device for that particular
	 * connection session since boot (i.e. if you disconnect and reconnect a
	 * device immediately after, it should be assigned a different session ID).
	 * If your OS cannot provide a unique session ID as described above,
	 * presenting a session ID of (bus_number << 8 | device_address) should
	 * be sufficient. Bus numbers and device addresses wrap and get reused,
	 * but that is an unlikely case.
	 *
	 * After computing a session ID for a device, call
	 * usbi_get_device_by_session_id(). This function checks if libusb already
	 * knows about the device, and if so, it provides you with a reference
	 * to a libusb_device structure for it.
	 *
	 * If usbi_get_device_by_session_id() returns NULL, it is time to allocate
	 * a new device structure for the device. Call usbi_alloc_device() to
	 * obtain a new libusb_device structure with reference count 1. Populate
	 * the bus_number and device_address attributes of the new device, and
	 * perform any other internal backend initialization you need to do. At
	 * this point, you should be ready to provide device descriptors and so
	 * on through the get_*_descriptor functions. Finally, call
	 * usbi_sanitize_device() to perform some final sanity checks on the
	 * device. Assuming all of the above succeeded, we can now continue.
	 * If any of the above failed, remember to unreference the device that
	 * was returned by usbi_alloc_device().
	 *
	 * At this stage we have a populated libusb_device structure (either one
	 * that was found earlier, or one that we have just allocated and
	 * populated). This can now be added to the discovered devices list
	 * using discovered_devs_append(). Note that discovered_devs_append()
	 * may reallocate the list, returning a new location for it, and also
	 * note that reallocation can fail. Your backend should handle these
	 * error conditions appropriately.
	 *
	 * This function should not generate any bus I/O and should not block.
	 * If I/O is required (e.g. reading the active configuration value), it is
	 * OK to ignore these suggestions :)
	 *
	 * This function is executed when the user wishes to retrieve a list
	 * of USB devices connected to the system.
	 *
	 * If the backend has hotplug support, this function is not used!
	 *
	 * Return 0 on success, or a LIBUSB_ERROR code on failure.
	 */
	int (*get_device_list)(struct libusb_context *ctx,
		struct discovered_devs **discdevs);

	/* Apps which were written before hotplug support, may listen for
	 * hotplug events on their own and call libusb_get_device_list on
	 * device addition. In this case libusb_get_device_list will likely
	 * return a list without the new device in there, as the hotplug
	 * event thread will still be busy enumerating the device, which may
	 * take a while, or may not even have seen the event yet.
	 *
	 * To avoid this libusb_get_device_list will call this optional
	 * function for backends with hotplug support before copying
	 * ctx->usb_devs to the user. In this function the backend should
	 * ensure any pending hotplug events are fully processed before
	 * returning.
	 *
	 * Optional, should be implemented by backends with hotplug support.
	 */
	void (*hotplug_poll)(void);

	/* Open a device for I/O and other USB operations. The device handle
	 * is preallocated for you, you can retrieve the device in question
	 * through handle->dev.
	 *
	 * Your backend should allocate any internal resources required for I/O
	 * and other operations so that those operations can happen (hopefully)
	 * without hiccup. This is also a good place to inform libusb that it
	 * should monitor certain file descriptors related to this device -
	 * see the usbi_add_pollfd() function.
	 *
	 * This function should not generate any bus I/O and should not block.
	 *
	 * This function is called when the user attempts to obtain a device
	 * handle for a device.
	 *
	 * Return:
	 * - 0 on success
	 * - LIBUSB_ERROR_ACCESS if the user has insufficient permissions
	 * - LIBUSB_ERROR_NO_DEVICE if the device has been disconnected since
	 *   discovery
	 * - another LIBUSB_ERROR code on other failure
	 *
	 * Do not worry about freeing the handle on failed open, the upper layers
	 * do this for you.
	 */
	int (*open)(struct libusb_device_handle *dev_handle);

	/* Close a device such that the handle cannot be used again. Your backend
	 * should destroy any resources that were allocated in the open path.
	 * This may also be a good place to call usbi_remove_pollfd() to inform
	 * libusb of any file descriptors associated with this device that should
	 * no longer be monitored.
	 *
	 * This function is called when the user closes a device handle.
	 */
	void (*close)(struct libusb_device_handle *dev_handle);

	/* Retrieve the device descriptor from a device.
	 *
	 * The descriptor should be retrieved from memory, NOT via bus I/O to the
	 * device. This means that you may have to cache it in a private structure
	 * during get_device_list enumeration. Alternatively, you may be able
	 * to retrieve it from a kernel interface (some Linux setups can do this)
	 * still without generating bus I/O.
	 *
	 * This function is expected to write DEVICE_DESC_LENGTH (18) bytes into
	 * buffer, which is guaranteed to be big enough.
	 *
	 * This function is called when sanity-checking a device before adding
	 * it to the list of discovered devices, and also when the user requests
	 * to read the device descriptor.
	 *
	 * This function is expected to return the descriptor in bus-endian format
	 * (LE). If it returns the multi-byte values in host-endian format,
	 * set the host_endian output parameter to "1".
	 *
	 * Return 0 on success or a LIBUSB_ERROR code on failure.
	 */
	int (*get_device_descriptor)(struct libusb_device *device,
		unsigned char *buffer, int *host_endian);

	/* Get the ACTIVE configuration descriptor for a device.
	 *
	 * The descriptor should be retrieved from memory, NOT via bus I/O to the
	 * device. This means that you may have to cache it in a private structure
	 * during get_device_list enumeration. You may also have to keep track
	 * of which configuration is active when the user changes it.
	 *
	 * This function is expected to write len bytes of data into buffer, which
	 * is guaranteed to be big enough. If you can only do a partial write,
	 * return an error code.
	 *
	 * This function is expected to return the descriptor in bus-endian format
	 * (LE). If it returns the multi-byte values in host-endian format,
	 * set the host_endian output parameter to "1".
	 *
	 * Return:
	 * - 0 on success
	 * - LIBUSB_ERROR_NOT_FOUND if the device is in unconfigured state
	 * - another LIBUSB_ERROR code on other failure
	 */
	int (*get_active_config_descriptor)(struct libusb_device *device,
		unsigned char *buffer, size_t len, int *host_endian);

	/* Get a specific configuration descriptor for a device.
	 *
	 * The descriptor should be retrieved from memory, NOT via bus I/O to the
	 * device. This means that you may have to cache it in a private structure
	 * during get_device_list enumeration.
	 *
	 * The requested descriptor is expressed as a zero-based index (i.e. 0
	 * indicates that we are requesting the first descriptor). The index does
	 * not (necessarily) equal the bConfigurationValue of the configuration
	 * being requested.
	 *
	 * This function is expected to write len bytes of data into buffer, which
	 * is guaranteed to be big enough. If you can only do a partial write,
	 * return an error code.
	 *
	 * This function is expected to return the descriptor in bus-endian format
	 * (LE). If it returns the multi-byte values in host-endian format,
	 * set the host_endian output parameter to "1".
	 *
	 * Return the length read on success or a LIBUSB_ERROR code on failure.
	 */
	int (*get_config_descriptor)(struct libusb_device *device,
		uint8_t config_index, unsigned char *buffer, size_t len,
		int *host_endian);

	/* Like get_config_descriptor but then by bConfigurationValue instead
	 * of by index.
	 *
	 * Optional, if not present the core will call get_config_descriptor
	 * for all configs until it finds the desired bConfigurationValue.
	 *
	 * Returns a pointer to the raw-descriptor in *buffer, this memory
	 * is valid as long as device is valid.
	 *
	 * Returns the length of the returned raw-descriptor on success,
	 * or a LIBUSB_ERROR code on failure.
	 */
	int (*get_config_descriptor_by_value)(struct libusb_device *device,
		uint8_t bConfigurationValue, unsigned char **buffer,
		int *host_endian);

	/* Get the bConfigurationValue for the active configuration for a device.
	 * Optional. This should only be implemented if you can retrieve it from
	 * cache (don't generate I/O).
	 *
	 * If you cannot retrieve this from cache, either do not implement this
	 * function, or return LIBUSB_ERROR_NOT_SUPPORTED. This will cause
	 * libusb to retrieve the information through a standard control transfer.
	 *
	 * This function must be non-blocking.
	 * Return:
	 * - 0 on success
	 * - LIBUSB_ERROR_NO_DEVICE if the device has been disconnected since it
	 *   was opened
	 * - LIBUSB_ERROR_NOT_SUPPORTED if the value cannot be retrieved without
	 *   blocking
	 * - another LIBUSB_ERROR code on other failure.
	 */
	int (*get_configuration)(struct libusb_device_handle *dev_handle, int *config);

	/* Set the active configuration for a device.
	 *
	 * A configuration value of -1 should put the device in unconfigured state.
	 *
	 * This function can block.
	 *
	 * Return:
	 * - 0 on success
	 * - LIBUSB_ERROR_NOT_FOUND if the configuration does not exist
	 * - LIBUSB_ERROR_BUSY if interfaces are currently claimed (and hence
	 *   configuration cannot be changed)
	 * - LIBUSB_ERROR_NO_DEVICE if the device has been disconnected since it
	 *   was opened
	 * - another LIBUSB_ERROR code on other failure.
	 */
	int (*set_configuration)(struct libusb_device_handle *dev_handle, int config);

	/* Claim an interface. When claimed, the application can then perform
	 * I/O to an interface's endpoints.
	 *
	 * This function should not generate any bus I/O and should not block.
	 * Interface claiming is a logical operation that simply ensures that
	 * no other drivers/applications are using the interface, and after
	 * claiming, no other drivers/applications can use the interface because
	 * we now "own" it.
	 *
	 * Return:
	 * - 0 on success
	 * - LIBUSB_ERROR_NOT_FOUND if the interface does not exist
	 * - LIBUSB_ERROR_BUSY if the interface is in use by another driver/app
	 * - LIBUSB_ERROR_NO_DEVICE if the device has been disconnected since it
	 *   was opened
	 * - another LIBUSB_ERROR code on other failure
	 */
	int (*claim_interface)(struct libusb_device_handle *dev_handle, int interface_number);

	/* Release a previously claimed interface.
	 *
	 * This function should also generate a SET_INTERFACE control request,
	 * resetting the alternate setting of that interface to 0. It's OK for
	 * this function to block as a result.
	 *
	 * You will only ever be asked to release an interface which was
	 * successfully claimed earlier.
	 *
	 * Return:
	 * - 0 on success
	 * - LIBUSB_ERROR_NO_DEVICE if the device has been disconnected since it
	 *   was opened
	 * - another LIBUSB_ERROR code on other failure
	 */
	int (*release_interface)(struct libusb_device_handle *dev_handle, int interface_number);

	/* Set the alternate setting for an interface.
	 *
	 * You will only ever be asked to set the alternate setting for an
	 * interface which was successfully claimed earlier.
	 *
	 * It's OK for this function to block.
	 *
	 * Return:
	 * - 0 on success
	 * - LIBUSB_ERROR_NOT_FOUND if the alternate setting does not exist
	 * - LIBUSB_ERROR_NO_DEVICE if the device has been disconnected since it
	 *   was opened
	 * - another LIBUSB_ERROR code on other failure
	 */
	int (*set_interface_altsetting)(struct libusb_device_handle *dev_handle,
		int interface_number, int altsetting);

	/* Clear a halt/stall condition on an endpoint.
	 *
	 * It's OK for this function to block.
	 *
	 * Return:
	 * - 0 on success
	 * - LIBUSB_ERROR_NOT_FOUND if the endpoint does not exist
	 * - LIBUSB_ERROR_NO_DEVICE if the device has been disconnected since it
	 *   was opened
	 * - another LIBUSB_ERROR code on other failure
	 */
	int (*clear_halt)(struct libusb_device_handle *dev_handle,
		unsigned char endpoint);

	/* Perform a USB port reset to reinitialize a device.
	 *
	 * If possible, the device handle should still be usable after the reset
	 * completes, assuming that the device descriptors did not change during
	 * reset and all previous interface state can be restored.
	 *
	 * If something changes, or you cannot easily locate/verify the resetted
	 * device, return LIBUSB_ERROR_NOT_FOUND. This prompts the application
	 * to close the old handle and re-enumerate the device.
	 *
	 * Return:
	 * - 0 on success
	 * - LIBUSB_ERROR_NOT_FOUND if re-enumeration is required, or if the device
	 *   has been disconnected since it was opened
	 * - another LIBUSB_ERROR code on other failure
	 */
	int (*reset_device)(struct libusb_device_handle *dev_handle);

	/* Alloc num_streams usb3 bulk streams on the passed in endpoints */
	int (*alloc_streams)(struct libusb_device_handle *dev_handle,
		uint32_t num_streams, unsigned char *endpoints, int num_endpoints);

	/* Free usb3 bulk streams allocated with alloc_streams */
	int (*free_streams)(struct libusb_device_handle *dev_handle,
		unsigned char *endpoints, int num_endpoints);

	/* Allocate persistent DMA memory for the given device, suitable for
	 * zerocopy. May return NULL on failure. Optional to implement.
	 */
	unsigned char *(*dev_mem_alloc)(struct libusb_device_handle *handle,
		size_t len);

	/* Free memory allocated by dev_mem_alloc. */
	int (*dev_mem_free)(struct libusb_device_handle *handle,
		unsigned char *buffer, size_t len);

	/* Determine if a kernel driver is active on an interface. Optional.
	 *
	 * The presence of a kernel driver on an interface indicates that any
	 * calls to claim_interface would fail with the LIBUSB_ERROR_BUSY code.
	 *
	 * Return:
	 * - 0 if no driver is active
	 * - 1 if a driver is active
	 * - LIBUSB_ERROR_NO_DEVICE if the device has been disconnected since it
	 *   was opened
	 * - another LIBUSB_ERROR code on other failure
	 */
	int (*kernel_driver_active)(struct libusb_device_handle *dev_handle,
		int interface_number);

	/* Detach a kernel driver from an interface. Optional.
	 *
	 * After detaching a kernel driver, the interface should be available
	 * for claim.
	 *
	 * Return:
	 * - 0 on success
	 * - LIBUSB_ERROR_NOT_FOUND if no kernel driver was active
	 * - LIBUSB_ERROR_INVALID_PARAM if the interface does not exist
	 * - LIBUSB_ERROR_NO_DEVICE if the device has been disconnected since it
	 *   was opened
	 * - another LIBUSB_ERROR code on other failure
	 */
	int (*detach_kernel_driver)(struct libusb_device_handle *dev_handle,
		int interface_number);

	/* Attach a kernel driver to an interface. Optional.
	 *
	 * Reattach a kernel driver to the device.
	 *
	 * Return:
	 * - 0 on success
	 * - LIBUSB_ERROR_NOT_FOUND if no kernel driver was active
	 * - LIBUSB_ERROR_INVALID_PARAM if the interface does not exist
	 * - LIBUSB_ERROR_NO_DEVICE if the device has been disconnected since it
	 *   was opened
	 * - LIBUSB_ERROR_BUSY if a program or driver has claimed the interface,
	 *   preventing reattachment
	 * - another LIBUSB_ERROR code on other failure
	 */
	int (*attach_kernel_driver)(struct libusb_device_handle *dev_handle,
		int interface_number);

	/* Destroy a device. Optional.
	 *
	 * This function is called when the last reference to a device is
	 * destroyed. It should free any resources allocated in the get_device_list
	 * path.
	 */
	void (*destroy_device)(struct libusb_device *dev);

	/* Submit a transfer. Your implementation should take the transfer,
	 * morph it into whatever form your platform requires, and submit it
	 * asynchronously.
	 *
	 * This function must not block.
	 *
	 * This function gets called with the flying_transfers_lock locked!
	 *
	 * Return:
	 * - 0 on success
	 * - LIBUSB_ERROR_NO_DEVICE if the device has been disconnected
	 * - another LIBUSB_ERROR code on other failure
	 */
	int (*submit_transfer)(struct usbi_transfer *itransfer);

	/* Cancel a previously submitted transfer.
	 *
	 * This function must not block. The transfer cancellation must complete
	 * later, resulting in a call to usbi_handle_transfer_cancellation()
	 * from the context of handle_events.
	 */
	int (*cancel_transfer)(struct usbi_transfer *itransfer);

	/* Clear a transfer as if it has completed or cancelled, but do not
	 * report any completion/cancellation to the library. You should free
	 * all private data from the transfer as if you were just about to report
	 * completion or cancellation.
	 *
	 * This function might seem a bit out of place. It is used when libusb
	 * detects a disconnected device - it calls this function for all pending
	 * transfers before reporting completion (with the disconnect code) to
	 * the user. Maybe we can improve upon this internal interface in future.
	 */
	void (*clear_transfer_priv)(struct usbi_transfer *itransfer);

	/* Handle any pending events on file descriptors. Optional.
	 *
	 * Provide this function when file descriptors directly indicate device
	 * or transfer activity. If your backend does not have such file descriptors,
	 * implement the handle_transfer_completion function below.
	 *
	 * This involves monitoring any active transfers and processing their
	 * completion or cancellation.
	 *
	 * The function is passed an array of pollfd structures (size nfds)
	 * as a result of the poll() system call. The num_ready parameter
	 * indicates the number of file descriptors that have reported events
	 * (i.e. the poll() return value). This should be enough information
	 * for you to determine which actions need to be taken on the currently
	 * active transfers.
	 *
	 * For any cancelled transfers, call usbi_handle_transfer_cancellation().
	 * For completed transfers, call usbi_handle_transfer_completion().
	 * For control/bulk/interrupt transfers, populate the "transferred"
	 * element of the appropriate usbi_transfer structure before calling the
	 * above functions. For isochronous transfers, populate the status and
	 * transferred fields of the iso packet descriptors of the transfer.
	 *
	 * This function should also be able to detect disconnection of the
	 * device, reporting that situation with usbi_handle_disconnect().
	 *
	 * When processing an event related to a transfer, you probably want to
	 * take usbi_transfer.lock to prevent races. See the documentation for
	 * the usbi_transfer structure.
	 *
	 * Return 0 on success, or a LIBUSB_ERROR code on failure.
	 */
	int (*handle_events)(struct libusb_context *ctx,
		struct pollfd *fds, POLL_NFDS_TYPE nfds, int num_ready);

	/* Handle transfer completion. Optional.
	 *
	 * Provide this function when there are no file descriptors available
	 * that directly indicate device or transfer activity. If your backend does
	 * have such file descriptors, implement the handle_events function above.
	 *
	 * Your backend must tell the library when a transfer has completed by
	 * calling usbi_signal_transfer_completion(). You should store any private
	 * information about the transfer and its completion status in the transfer's
	 * private backend data.
	 *
	 * During event handling, this function will be called on each transfer for
	 * which usbi_signal_transfer_completion() was called.
	 *
	 * For any cancelled transfers, call usbi_handle_transfer_cancellation().
	 * For completed transfers, call usbi_handle_transfer_completion().
	 * For control/bulk/interrupt transfers, populate the "transferred"
	 * element of the appropriate usbi_transfer structure before calling the
	 * above functions. For isochronous transfers, populate the status and
	 * transferred fields of the iso packet descriptors of the transfer.
	 *
	 * Return 0 on success, or a LIBUSB_ERROR code on failure.
	 */
	int (*handle_transfer_completion)(struct usbi_transfer *itransfer);

	/* Get time from specified clock. At least two clocks must be implemented
	   by the backend: USBI_CLOCK_REALTIME, and USBI_CLOCK_MONOTONIC.

	   Description of clocks:
	     USBI_CLOCK_REALTIME : clock returns time since system epoch.
	     USBI_CLOCK_MONOTONIC: clock returns time since unspecified start
	                             time (usually boot).
	 */
	int (*clock_gettime)(int clkid, struct timespec *tp);

#ifdef USBI_TIMERFD_AVAILABLE
	/* clock ID of the clock that should be used for timerfd */
	clockid_t (*get_timerfd_clockid)(void);
#endif

	/* Number of bytes to reserve for per-context private backend data.
	 * This private data area is accessible through the "os_priv" field of
	 * struct libusb_context. */
	size_t context_priv_size;

	/* Number of bytes to reserve for per-device private backend data.
	 * This private data area is accessible through the "os_priv" field of
	 * struct libusb_device. */
	size_t device_priv_size;

	/* Number of bytes to reserve for per-handle private backend data.
	 * This private data area is accessible through the "os_priv" field of
	 * struct libusb_device. */
	size_t device_handle_priv_size;

	/* Number of bytes to reserve for per-transfer private backend data.
	 * This private data area is accessible by calling
	 * usbi_transfer_get_os_priv() on the appropriate usbi_transfer instance.
	 */
	size_t transfer_priv_size;
};

extern const struct usbi_os_backend usbi_backend;

extern struct list_head active_contexts_list;
extern usbi_mutex_static_t active_contexts_lock;

#ifdef __cplusplus
}
#endif

#endif
