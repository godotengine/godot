/*
 *
 * Copyright (c) 2016, Oracle and/or its affiliates.
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

#ifndef	LIBUSB_SUNOS_H
#define	LIBUSB_SUNOS_H

#include <libdevinfo.h>
#include <pthread.h>
#include "libusbi.h"

#define	READ	0
#define	WRITE	1

typedef struct sunos_device_priv {
	uint8_t	cfgvalue;		/* active config value */
	uint8_t	*raw_cfgdescr;		/* active config descriptor */
	struct libusb_device_descriptor	dev_descr;	/* usb device descriptor */
	char	*ugenpath;		/* name of the ugen(4) node */
	char	*phypath;		/* physical path */
} sunos_dev_priv_t;

typedef	struct endpoint {
	int datafd;	/* data file */
	int statfd;	/* state file */
} sunos_ep_priv_t;

typedef struct sunos_device_handle_priv {
	uint8_t			altsetting[USB_MAXINTERFACES];	/* a interface's alt */
	uint8_t			config_index;
	sunos_ep_priv_t		eps[USB_MAXENDPOINTS];
	sunos_dev_priv_t	*dpriv; /* device private */
} sunos_dev_handle_priv_t;

typedef	struct sunos_transfer_priv {
	struct aiocb		aiocb;
	struct libusb_transfer	*transfer;
} sunos_xfer_priv_t;

struct node_args {
	struct libusb_context	*ctx;
	struct discovered_devs	**discdevs;
	const char		*last_ugenpath;
	di_devlink_handle_t	dlink_hdl;
};

struct devlink_cbarg {
	struct node_args	*nargs;	/* di node walk arguments */
	di_node_t		myself;	/* the di node */
	di_minor_t		minor;
};

/* AIO callback args */
struct aio_callback_args{
	struct libusb_transfer *transfer;
	struct aiocb aiocb;
};

#endif /* LIBUSB_SUNOS_H */
