/* -*- Mode: C; indent-tabs-mode:t ; c-basic-offset:8 -*- */
/*
 * Hotplug functions for libusb
 * Copyright © 2012-2013 Nathan Hjelm <hjelmn@mac.com>
 * Copyright © 2012-2013 Peter Stuge <peter@stuge.se>
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

#include <config.h>

#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#ifdef HAVE_SYS_TYPES_H
#include <sys/types.h>
#endif
#include <assert.h>

#include "libusbi.h"
#include "hotplug.h"

/**
 * @defgroup libusb_hotplug Device hotplug event notification
 * This page details how to use the libusb hotplug interface, where available.
 *
 * Be mindful that not all platforms currently implement hotplug notification and
 * that you should first call on \ref libusb_has_capability() with parameter
 * \ref LIBUSB_CAP_HAS_HOTPLUG to confirm that hotplug support is available.
 *
 * \page libusb_hotplug Device hotplug event notification
 *
 * \section hotplug_intro Introduction
 *
 * Version 1.0.16, \ref LIBUSB_API_VERSION >= 0x01000102, has added support
 * for hotplug events on <b>some</b> platforms (you should test if your platform
 * supports hotplug notification by calling \ref libusb_has_capability() with
 * parameter \ref LIBUSB_CAP_HAS_HOTPLUG). 
 *
 * This interface allows you to request notification for the arrival and departure
 * of matching USB devices.
 *
 * To receive hotplug notification you register a callback by calling
 * \ref libusb_hotplug_register_callback(). This function will optionally return
 * a callback handle that can be passed to \ref libusb_hotplug_deregister_callback().
 *
 * A callback function must return an int (0 or 1) indicating whether the callback is
 * expecting additional events. Returning 0 will rearm the callback and 1 will cause
 * the callback to be deregistered. Note that when callbacks are called from
 * libusb_hotplug_register_callback() because of the \ref LIBUSB_HOTPLUG_ENUMERATE
 * flag, the callback return value is ignored, iow you cannot cause a callback
 * to be deregistered by returning 1 when it is called from
 * libusb_hotplug_register_callback().
 *
 * Callbacks for a particular context are automatically deregistered by libusb_exit().
 *
 * As of 1.0.16 there are two supported hotplug events:
 *  - LIBUSB_HOTPLUG_EVENT_DEVICE_ARRIVED: A device has arrived and is ready to use
 *  - LIBUSB_HOTPLUG_EVENT_DEVICE_LEFT: A device has left and is no longer available
 *
 * A hotplug event can listen for either or both of these events.
 *
 * Note: If you receive notification that a device has left and you have any
 * a libusb_device_handles for the device it is up to you to call libusb_close()
 * on each device handle to free up any remaining resources associated with the device.
 * Once a device has left any libusb_device_handle associated with the device
 * are invalid and will remain so even if the device comes back.
 *
 * When handling a LIBUSB_HOTPLUG_EVENT_DEVICE_ARRIVED event it is considered
 * safe to call any libusb function that takes a libusb_device. It also safe to
 * open a device and submit asynchronous transfers. However, most other functions
 * that take a libusb_device_handle are <b>not</b> safe to call. Examples of such
 * functions are any of the \ref libusb_syncio "synchronous API" functions or the blocking
 * functions that retrieve various \ref libusb_desc "USB descriptors". These functions must
 * be used outside of the context of the hotplug callback.
 *
 * When handling a LIBUSB_HOTPLUG_EVENT_DEVICE_LEFT event the only safe function
 * is libusb_get_device_descriptor().
 *
 * The following code provides an example of the usage of the hotplug interface:
\code
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <libusb.h>

static int count = 0;

int hotplug_callback(struct libusb_context *ctx, struct libusb_device *dev,
                     libusb_hotplug_event event, void *user_data) {
  static libusb_device_handle *dev_handle = NULL;
  struct libusb_device_descriptor desc;
  int rc;

  (void)libusb_get_device_descriptor(dev, &desc);

  if (LIBUSB_HOTPLUG_EVENT_DEVICE_ARRIVED == event) {
    rc = libusb_open(dev, &dev_handle);
    if (LIBUSB_SUCCESS != rc) {
      printf("Could not open USB device\n");
    }
  } else if (LIBUSB_HOTPLUG_EVENT_DEVICE_LEFT == event) {
    if (dev_handle) {
      libusb_close(dev_handle);
      dev_handle = NULL;
    }
  } else {
    printf("Unhandled event %d\n", event);
  }
  count++;

  return 0;
}

int main (void) {
  libusb_hotplug_callback_handle callback_handle;
  int rc;

  libusb_init(NULL);

  rc = libusb_hotplug_register_callback(NULL, LIBUSB_HOTPLUG_EVENT_DEVICE_ARRIVED |
                                        LIBUSB_HOTPLUG_EVENT_DEVICE_LEFT, 0, 0x045a, 0x5005,
                                        LIBUSB_HOTPLUG_MATCH_ANY, hotplug_callback, NULL,
                                        &callback_handle);
  if (LIBUSB_SUCCESS != rc) {
    printf("Error creating a hotplug callback\n");
    libusb_exit(NULL);
    return EXIT_FAILURE;
  }

  while (count < 2) {
    libusb_handle_events_completed(NULL, NULL);
    nanosleep(&(struct timespec){0, 10000000UL}, NULL);
  }

  libusb_hotplug_deregister_callback(NULL, callback_handle);
  libusb_exit(NULL);

  return 0;
}
\endcode
 */

static int usbi_hotplug_match_cb (struct libusb_context *ctx,
	struct libusb_device *dev, libusb_hotplug_event event,
	struct libusb_hotplug_callback *hotplug_cb)
{
	/* Handle lazy deregistration of callback */
	if (hotplug_cb->needs_free) {
		/* Free callback */
		return 1;
	}

	if (!(hotplug_cb->events & event)) {
		return 0;
	}

	if (LIBUSB_HOTPLUG_MATCH_ANY != hotplug_cb->vendor_id &&
	    hotplug_cb->vendor_id != dev->device_descriptor.idVendor) {
		return 0;
	}

	if (LIBUSB_HOTPLUG_MATCH_ANY != hotplug_cb->product_id &&
	    hotplug_cb->product_id != dev->device_descriptor.idProduct) {
		return 0;
	}

	if (LIBUSB_HOTPLUG_MATCH_ANY != hotplug_cb->dev_class &&
	    hotplug_cb->dev_class != dev->device_descriptor.bDeviceClass) {
		return 0;
	}

	return hotplug_cb->cb (ctx, dev, event, hotplug_cb->user_data);
}

void usbi_hotplug_match(struct libusb_context *ctx, struct libusb_device *dev,
	libusb_hotplug_event event)
{
	struct libusb_hotplug_callback *hotplug_cb, *next;
	int ret;

	usbi_mutex_lock(&ctx->hotplug_cbs_lock);

	list_for_each_entry_safe(hotplug_cb, next, &ctx->hotplug_cbs, list, struct libusb_hotplug_callback) {
		usbi_mutex_unlock(&ctx->hotplug_cbs_lock);
		ret = usbi_hotplug_match_cb (ctx, dev, event, hotplug_cb);
		usbi_mutex_lock(&ctx->hotplug_cbs_lock);

		if (ret) {
			list_del(&hotplug_cb->list);
			free(hotplug_cb);
		}
	}

	usbi_mutex_unlock(&ctx->hotplug_cbs_lock);

	/* the backend is expected to call the callback for each active transfer */
}

void usbi_hotplug_notification(struct libusb_context *ctx, struct libusb_device *dev,
	libusb_hotplug_event event)
{
	int pending_events;
	libusb_hotplug_message *message = calloc(1, sizeof(*message));

	if (!message) {
		usbi_err(ctx, "error allocating hotplug message");
		return;
	}

	message->event = event;
	message->device = dev;

	/* Take the event data lock and add this message to the list.
	 * Only signal an event if there are no prior pending events. */
	usbi_mutex_lock(&ctx->event_data_lock);
	pending_events = usbi_pending_events(ctx);
	list_add_tail(&message->list, &ctx->hotplug_msgs);
	if (!pending_events)
		usbi_signal_event(ctx);
	usbi_mutex_unlock(&ctx->event_data_lock);
}

int API_EXPORTED libusb_hotplug_register_callback(libusb_context *ctx,
	libusb_hotplug_event events, libusb_hotplug_flag flags,
	int vendor_id, int product_id, int dev_class,
	libusb_hotplug_callback_fn cb_fn, void *user_data,
	libusb_hotplug_callback_handle *callback_handle)
{
	libusb_hotplug_callback *new_callback;
	static int handle_id = 1;

	/* check for hotplug support */
	if (!libusb_has_capability(LIBUSB_CAP_HAS_HOTPLUG)) {
		return LIBUSB_ERROR_NOT_SUPPORTED;
	}

	/* check for sane values */
	if ((LIBUSB_HOTPLUG_MATCH_ANY != vendor_id && (~0xffff & vendor_id)) ||
	    (LIBUSB_HOTPLUG_MATCH_ANY != product_id && (~0xffff & product_id)) ||
	    (LIBUSB_HOTPLUG_MATCH_ANY != dev_class && (~0xff & dev_class)) ||
	    !cb_fn) {
		return LIBUSB_ERROR_INVALID_PARAM;
	}

	USBI_GET_CONTEXT(ctx);

	new_callback = (libusb_hotplug_callback *)calloc(1, sizeof (*new_callback));
	if (!new_callback) {
		return LIBUSB_ERROR_NO_MEM;
	}

	new_callback->ctx = ctx;
	new_callback->vendor_id = vendor_id;
	new_callback->product_id = product_id;
	new_callback->dev_class = dev_class;
	new_callback->flags = flags;
	new_callback->events = events;
	new_callback->cb = cb_fn;
	new_callback->user_data = user_data;
	new_callback->needs_free = 0;

	usbi_mutex_lock(&ctx->hotplug_cbs_lock);

	/* protect the handle by the context hotplug lock. it doesn't matter if the same handle
	 * is used for different contexts only that the handle is unique for this context */
	new_callback->handle = handle_id++;

	list_add(&new_callback->list, &ctx->hotplug_cbs);

	usbi_mutex_unlock(&ctx->hotplug_cbs_lock);


	if (flags & LIBUSB_HOTPLUG_ENUMERATE) {
		int i, len;
		struct libusb_device **devs;

		len = (int) libusb_get_device_list(ctx, &devs);
		if (len < 0) {
			libusb_hotplug_deregister_callback(ctx,
							new_callback->handle);
			return len;
		}

		for (i = 0; i < len; i++) {
			usbi_hotplug_match_cb(ctx, devs[i],
					LIBUSB_HOTPLUG_EVENT_DEVICE_ARRIVED,
					new_callback);
		}

		libusb_free_device_list(devs, 1);
	}


	if (callback_handle)
		*callback_handle = new_callback->handle;

	return LIBUSB_SUCCESS;
}

void API_EXPORTED libusb_hotplug_deregister_callback (struct libusb_context *ctx,
	libusb_hotplug_callback_handle callback_handle)
{
	struct libusb_hotplug_callback *hotplug_cb;

	/* check for hotplug support */
	if (!libusb_has_capability(LIBUSB_CAP_HAS_HOTPLUG)) {
		return;
	}

	USBI_GET_CONTEXT(ctx);

	usbi_mutex_lock(&ctx->hotplug_cbs_lock);
	list_for_each_entry(hotplug_cb, &ctx->hotplug_cbs, list,
			    struct libusb_hotplug_callback) {
		if (callback_handle == hotplug_cb->handle) {
			/* Mark this callback for deregistration */
			hotplug_cb->needs_free = 1;
		}
	}
	usbi_mutex_unlock(&ctx->hotplug_cbs_lock);

	usbi_hotplug_notification(ctx, NULL, 0);
}

void usbi_hotplug_deregister_all(struct libusb_context *ctx) {
	struct libusb_hotplug_callback *hotplug_cb, *next;

	usbi_mutex_lock(&ctx->hotplug_cbs_lock);
	list_for_each_entry_safe(hotplug_cb, next, &ctx->hotplug_cbs, list,
				 struct libusb_hotplug_callback) {
		list_del(&hotplug_cb->list);
		free(hotplug_cb);
	}

	usbi_mutex_unlock(&ctx->hotplug_cbs_lock);
}
