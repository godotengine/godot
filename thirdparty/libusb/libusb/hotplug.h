/* -*- Mode: C; indent-tabs-mode:t ; c-basic-offset:8 -*- */
/*
 * Hotplug support for libusb
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

#if !defined(USBI_HOTPLUG_H)
#define USBI_HOTPLUG_H

#ifndef LIBUSBI_H
#include "libusbi.h"
#endif

/** \ingroup hotplug
 * The hotplug callback structure. The user populates this structure with
 * libusb_hotplug_prepare_callback() and then calls libusb_hotplug_register_callback()
 * to receive notification of hotplug events.
 */
struct libusb_hotplug_callback {
	/** Context this callback is associated with */
	struct libusb_context *ctx;

	/** Vendor ID to match or LIBUSB_HOTPLUG_MATCH_ANY */
	int vendor_id;

	/** Product ID to match or LIBUSB_HOTPLUG_MATCH_ANY */
	int product_id;

	/** Device class to match or LIBUSB_HOTPLUG_MATCH_ANY */
	int dev_class;

	/** Hotplug callback flags */
	libusb_hotplug_flag flags;

	/** Event(s) that will trigger this callback */
	libusb_hotplug_event events;

	/** Callback function to invoke for matching event/device */
	libusb_hotplug_callback_fn cb;

	/** Handle for this callback (used to match on deregister) */
	libusb_hotplug_callback_handle handle;

	/** User data that will be passed to the callback function */
	void *user_data;

	/** Callback is marked for deletion */
	int needs_free;

	/** List this callback is registered in (ctx->hotplug_cbs) */
	struct list_head list;
};

typedef struct libusb_hotplug_callback libusb_hotplug_callback;

struct libusb_hotplug_message {
	/** The hotplug event that occurred */
	libusb_hotplug_event event;

	/** The device for which this hotplug event occurred */
	struct libusb_device *device;

	/** List this message is contained in (ctx->hotplug_msgs) */
	struct list_head list;
};

typedef struct libusb_hotplug_message libusb_hotplug_message;

void usbi_hotplug_deregister_all(struct libusb_context *ctx);
void usbi_hotplug_match(struct libusb_context *ctx, struct libusb_device *dev,
			libusb_hotplug_event event);
void usbi_hotplug_notification(struct libusb_context *ctx, struct libusb_device *dev,
			libusb_hotplug_event event);

#endif
