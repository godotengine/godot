/* -*- Mode: C; indent-tabs-mode:t ; c-basic-offset:8 -*- */
/*
 * USB descriptor handling functions for libusb
 * Copyright © 2007 Daniel Drake <dsd@gentoo.org>
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

#include <config.h>

#include <errno.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include "libusbi.h"

#define DESC_HEADER_LENGTH		2
#define DEVICE_DESC_LENGTH		18
#define CONFIG_DESC_LENGTH		9
#define INTERFACE_DESC_LENGTH		9
#define ENDPOINT_DESC_LENGTH		7
#define ENDPOINT_AUDIO_DESC_LENGTH	9

/** @defgroup libusb_desc USB descriptors
 * This page details how to examine the various standard USB descriptors
 * for detected devices
 */

/* set host_endian if the w values are already in host endian format,
 * as opposed to bus endian. */
int usbi_parse_descriptor(const unsigned char *source, const char *descriptor,
	void *dest, int host_endian)
{
	const unsigned char *sp = source;
	unsigned char *dp = dest;
	uint16_t w;
	const char *cp;
	uint32_t d;

	for (cp = descriptor; *cp; cp++) {
		switch (*cp) {
			case 'b':	/* 8-bit byte */
				*dp++ = *sp++;
				break;
			case 'w':	/* 16-bit word, convert from little endian to CPU */
				dp += ((uintptr_t)dp & 1);	/* Align to word boundary */

				if (host_endian) {
					memcpy(dp, sp, 2);
				} else {
					w = (sp[1] << 8) | sp[0];
					*((uint16_t *)dp) = w;
				}
				sp += 2;
				dp += 2;
				break;
			case 'd':	/* 32-bit word, convert from little endian to CPU */
				dp += ((uintptr_t)dp & 1);	/* Align to word boundary */

				if (host_endian) {
					memcpy(dp, sp, 4);
				} else {
					d = (sp[3] << 24) | (sp[2] << 16) |
						(sp[1] << 8) | sp[0];
					*((uint32_t *)dp) = d;
				}
				sp += 4;
				dp += 4;
				break;
			case 'u':	/* 16 byte UUID */
				memcpy(dp, sp, 16);
				sp += 16;
				dp += 16;
				break;
		}
	}

	return (int) (sp - source);
}

static void clear_endpoint(struct libusb_endpoint_descriptor *endpoint)
{
	free((void *) endpoint->extra);
}

static int parse_endpoint(struct libusb_context *ctx,
	struct libusb_endpoint_descriptor *endpoint, unsigned char *buffer,
	int size, int host_endian)
{
	struct usb_descriptor_header header;
	unsigned char *extra;
	unsigned char *begin;
	int parsed = 0;
	int len;

	if (size < DESC_HEADER_LENGTH) {
		usbi_err(ctx, "short endpoint descriptor read %d/%d",
			 size, DESC_HEADER_LENGTH);
		return LIBUSB_ERROR_IO;
	}

	usbi_parse_descriptor(buffer, "bb", &header, 0);
	if (header.bDescriptorType != LIBUSB_DT_ENDPOINT) {
		usbi_err(ctx, "unexpected descriptor %x (expected %x)",
			header.bDescriptorType, LIBUSB_DT_ENDPOINT);
		return parsed;
	}
	if (header.bLength > size) {
		usbi_warn(ctx, "short endpoint descriptor read %d/%d",
			  size, header.bLength);
		return parsed;
	}
	if (header.bLength >= ENDPOINT_AUDIO_DESC_LENGTH)
		usbi_parse_descriptor(buffer, "bbbbwbbb", endpoint, host_endian);
	else if (header.bLength >= ENDPOINT_DESC_LENGTH)
		usbi_parse_descriptor(buffer, "bbbbwb", endpoint, host_endian);
	else {
		usbi_err(ctx, "invalid endpoint bLength (%d)", header.bLength);
		return LIBUSB_ERROR_IO;
	}

	buffer += header.bLength;
	size -= header.bLength;
	parsed += header.bLength;

	/* Skip over the rest of the Class Specific or Vendor Specific */
	/*  descriptors */
	begin = buffer;
	while (size >= DESC_HEADER_LENGTH) {
		usbi_parse_descriptor(buffer, "bb", &header, 0);
		if (header.bLength < DESC_HEADER_LENGTH) {
			usbi_err(ctx, "invalid extra ep desc len (%d)",
				 header.bLength);
			return LIBUSB_ERROR_IO;
		} else if (header.bLength > size) {
			usbi_warn(ctx, "short extra ep desc read %d/%d",
				  size, header.bLength);
			return parsed;
		}

		/* If we find another "proper" descriptor then we're done  */
		if ((header.bDescriptorType == LIBUSB_DT_ENDPOINT) ||
				(header.bDescriptorType == LIBUSB_DT_INTERFACE) ||
				(header.bDescriptorType == LIBUSB_DT_CONFIG) ||
				(header.bDescriptorType == LIBUSB_DT_DEVICE))
			break;

		usbi_dbg("skipping descriptor %x", header.bDescriptorType);
		buffer += header.bLength;
		size -= header.bLength;
		parsed += header.bLength;
	}

	/* Copy any unknown descriptors into a storage area for drivers */
	/*  to later parse */
	len = (int)(buffer - begin);
	if (!len) {
		endpoint->extra = NULL;
		endpoint->extra_length = 0;
		return parsed;
	}

	extra = malloc(len);
	endpoint->extra = extra;
	if (!extra) {
		endpoint->extra_length = 0;
		return LIBUSB_ERROR_NO_MEM;
	}

	memcpy(extra, begin, len);
	endpoint->extra_length = len;

	return parsed;
}

static void clear_interface(struct libusb_interface *usb_interface)
{
	int i;
	int j;

	if (usb_interface->altsetting) {
		for (i = 0; i < usb_interface->num_altsetting; i++) {
			struct libusb_interface_descriptor *ifp =
				(struct libusb_interface_descriptor *)
				usb_interface->altsetting + i;
			free((void *) ifp->extra);
			if (ifp->endpoint) {
				for (j = 0; j < ifp->bNumEndpoints; j++)
					clear_endpoint((struct libusb_endpoint_descriptor *)
						       ifp->endpoint + j);
			}
			free((void *) ifp->endpoint);
		}
	}
	free((void *) usb_interface->altsetting);
	usb_interface->altsetting = NULL;
}

static int parse_interface(libusb_context *ctx,
	struct libusb_interface *usb_interface, unsigned char *buffer, int size,
	int host_endian)
{
	int i;
	int len;
	int r;
	int parsed = 0;
	int interface_number = -1;
	struct usb_descriptor_header header;
	struct libusb_interface_descriptor *ifp;
	unsigned char *begin;

	usb_interface->num_altsetting = 0;

	while (size >= INTERFACE_DESC_LENGTH) {
		struct libusb_interface_descriptor *altsetting =
			(struct libusb_interface_descriptor *) usb_interface->altsetting;
		altsetting = usbi_reallocf(altsetting,
			sizeof(struct libusb_interface_descriptor) *
			(usb_interface->num_altsetting + 1));
		if (!altsetting) {
			r = LIBUSB_ERROR_NO_MEM;
			goto err;
		}
		usb_interface->altsetting = altsetting;

		ifp = altsetting + usb_interface->num_altsetting;
		usbi_parse_descriptor(buffer, "bbbbbbbbb", ifp, 0);
		if (ifp->bDescriptorType != LIBUSB_DT_INTERFACE) {
			usbi_err(ctx, "unexpected descriptor %x (expected %x)",
				 ifp->bDescriptorType, LIBUSB_DT_INTERFACE);
			return parsed;
		}
		if (ifp->bLength < INTERFACE_DESC_LENGTH) {
			usbi_err(ctx, "invalid interface bLength (%d)",
				 ifp->bLength);
			r = LIBUSB_ERROR_IO;
			goto err;
		}
		if (ifp->bLength > size) {
			usbi_warn(ctx, "short intf descriptor read %d/%d",
				 size, ifp->bLength);
			return parsed;
		}
		if (ifp->bNumEndpoints > USB_MAXENDPOINTS) {
			usbi_err(ctx, "too many endpoints (%d)", ifp->bNumEndpoints);
			r = LIBUSB_ERROR_IO;
			goto err;
		}

		usb_interface->num_altsetting++;
		ifp->extra = NULL;
		ifp->extra_length = 0;
		ifp->endpoint = NULL;

		if (interface_number == -1)
			interface_number = ifp->bInterfaceNumber;

		/* Skip over the interface */
		buffer += ifp->bLength;
		parsed += ifp->bLength;
		size -= ifp->bLength;

		begin = buffer;

		/* Skip over any interface, class or vendor descriptors */
		while (size >= DESC_HEADER_LENGTH) {
			usbi_parse_descriptor(buffer, "bb", &header, 0);
			if (header.bLength < DESC_HEADER_LENGTH) {
				usbi_err(ctx,
					 "invalid extra intf desc len (%d)",
					 header.bLength);
				r = LIBUSB_ERROR_IO;
				goto err;
			} else if (header.bLength > size) {
				usbi_warn(ctx,
					  "short extra intf desc read %d/%d",
					  size, header.bLength);
				return parsed;
			}

			/* If we find another "proper" descriptor then we're done */
			if ((header.bDescriptorType == LIBUSB_DT_INTERFACE) ||
					(header.bDescriptorType == LIBUSB_DT_ENDPOINT) ||
					(header.bDescriptorType == LIBUSB_DT_CONFIG) ||
					(header.bDescriptorType == LIBUSB_DT_DEVICE))
				break;

			buffer += header.bLength;
			parsed += header.bLength;
			size -= header.bLength;
		}

		/* Copy any unknown descriptors into a storage area for */
		/*  drivers to later parse */
		len = (int)(buffer - begin);
		if (len) {
			ifp->extra = malloc(len);
			if (!ifp->extra) {
				r = LIBUSB_ERROR_NO_MEM;
				goto err;
			}
			memcpy((unsigned char *) ifp->extra, begin, len);
			ifp->extra_length = len;
		}

		if (ifp->bNumEndpoints > 0) {
			struct libusb_endpoint_descriptor *endpoint;
			endpoint = calloc(ifp->bNumEndpoints, sizeof(struct libusb_endpoint_descriptor));
			ifp->endpoint = endpoint;
			if (!endpoint) {
				r = LIBUSB_ERROR_NO_MEM;
				goto err;
			}

			for (i = 0; i < ifp->bNumEndpoints; i++) {
				r = parse_endpoint(ctx, endpoint + i, buffer, size,
					host_endian);
				if (r < 0)
					goto err;
				if (r == 0) {
					ifp->bNumEndpoints = (uint8_t)i;
					break;;
				}

				buffer += r;
				parsed += r;
				size -= r;
			}
		}

		/* We check to see if it's an alternate to this one */
		ifp = (struct libusb_interface_descriptor *) buffer;
		if (size < LIBUSB_DT_INTERFACE_SIZE ||
				ifp->bDescriptorType != LIBUSB_DT_INTERFACE ||
				ifp->bInterfaceNumber != interface_number)
			return parsed;
	}

	return parsed;
err:
	clear_interface(usb_interface);
	return r;
}

static void clear_configuration(struct libusb_config_descriptor *config)
{
	int i;
	if (config->interface) {
		for (i = 0; i < config->bNumInterfaces; i++)
			clear_interface((struct libusb_interface *)
					config->interface + i);
	}
	free((void *) config->interface);
	free((void *) config->extra);
}

static int parse_configuration(struct libusb_context *ctx,
	struct libusb_config_descriptor *config, unsigned char *buffer,
	int size, int host_endian)
{
	int i;
	int r;
	struct usb_descriptor_header header;
	struct libusb_interface *usb_interface;

	if (size < LIBUSB_DT_CONFIG_SIZE) {
		usbi_err(ctx, "short config descriptor read %d/%d",
			 size, LIBUSB_DT_CONFIG_SIZE);
		return LIBUSB_ERROR_IO;
	}

	usbi_parse_descriptor(buffer, "bbwbbbbb", config, host_endian);
	if (config->bDescriptorType != LIBUSB_DT_CONFIG) {
		usbi_err(ctx, "unexpected descriptor %x (expected %x)",
			 config->bDescriptorType, LIBUSB_DT_CONFIG);
		return LIBUSB_ERROR_IO;
	}
	if (config->bLength < LIBUSB_DT_CONFIG_SIZE) {
		usbi_err(ctx, "invalid config bLength (%d)", config->bLength);
		return LIBUSB_ERROR_IO;
	}
	if (config->bLength > size) {
		usbi_err(ctx, "short config descriptor read %d/%d",
			 size, config->bLength);
		return LIBUSB_ERROR_IO;
	}
	if (config->bNumInterfaces > USB_MAXINTERFACES) {
		usbi_err(ctx, "too many interfaces (%d)", config->bNumInterfaces);
		return LIBUSB_ERROR_IO;
	}

	usb_interface = calloc(config->bNumInterfaces, sizeof(struct libusb_interface));
	config->interface = usb_interface;
	if (!usb_interface)
		return LIBUSB_ERROR_NO_MEM;

	buffer += config->bLength;
	size -= config->bLength;

	config->extra = NULL;
	config->extra_length = 0;

	for (i = 0; i < config->bNumInterfaces; i++) {
		int len;
		unsigned char *begin;

		/* Skip over the rest of the Class Specific or Vendor */
		/*  Specific descriptors */
		begin = buffer;
		while (size >= DESC_HEADER_LENGTH) {
			usbi_parse_descriptor(buffer, "bb", &header, 0);

			if (header.bLength < DESC_HEADER_LENGTH) {
				usbi_err(ctx,
					 "invalid extra config desc len (%d)",
					 header.bLength);
				r = LIBUSB_ERROR_IO;
				goto err;
			} else if (header.bLength > size) {
				usbi_warn(ctx,
					  "short extra config desc read %d/%d",
					  size, header.bLength);
				config->bNumInterfaces = (uint8_t)i;
				return size;
			}

			/* If we find another "proper" descriptor then we're done */
			if ((header.bDescriptorType == LIBUSB_DT_ENDPOINT) ||
					(header.bDescriptorType == LIBUSB_DT_INTERFACE) ||
					(header.bDescriptorType == LIBUSB_DT_CONFIG) ||
					(header.bDescriptorType == LIBUSB_DT_DEVICE))
				break;

			usbi_dbg("skipping descriptor 0x%x", header.bDescriptorType);
			buffer += header.bLength;
			size -= header.bLength;
		}

		/* Copy any unknown descriptors into a storage area for */
		/*  drivers to later parse */
		len = (int)(buffer - begin);
		if (len) {
			/* FIXME: We should realloc and append here */
			if (!config->extra_length) {
				config->extra = malloc(len);
				if (!config->extra) {
					r = LIBUSB_ERROR_NO_MEM;
					goto err;
				}

				memcpy((unsigned char *) config->extra, begin, len);
				config->extra_length = len;
			}
		}

		r = parse_interface(ctx, usb_interface + i, buffer, size, host_endian);
		if (r < 0)
			goto err;
		if (r == 0) {
			config->bNumInterfaces = (uint8_t)i;
			break;
		}

		buffer += r;
		size -= r;
	}

	return size;

err:
	clear_configuration(config);
	return r;
}

static int raw_desc_to_config(struct libusb_context *ctx,
	unsigned char *buf, int size, int host_endian,
	struct libusb_config_descriptor **config)
{
	struct libusb_config_descriptor *_config = malloc(sizeof(*_config));
	int r;
	
	if (!_config)
		return LIBUSB_ERROR_NO_MEM;

	r = parse_configuration(ctx, _config, buf, size, host_endian);
	if (r < 0) {
		usbi_err(ctx, "parse_configuration failed with error %d", r);
		free(_config);
		return r;
	} else if (r > 0) {
		usbi_warn(ctx, "still %d bytes of descriptor data left", r);
	}
	
	*config = _config;
	return LIBUSB_SUCCESS;
}

int usbi_device_cache_descriptor(libusb_device *dev)
{
	int r, host_endian = 0;

	r = usbi_backend.get_device_descriptor(dev, (unsigned char *) &dev->device_descriptor,
						&host_endian);
	if (r < 0)
		return r;

	if (!host_endian) {
		dev->device_descriptor.bcdUSB = libusb_le16_to_cpu(dev->device_descriptor.bcdUSB);
		dev->device_descriptor.idVendor = libusb_le16_to_cpu(dev->device_descriptor.idVendor);
		dev->device_descriptor.idProduct = libusb_le16_to_cpu(dev->device_descriptor.idProduct);
		dev->device_descriptor.bcdDevice = libusb_le16_to_cpu(dev->device_descriptor.bcdDevice);
	}

	return LIBUSB_SUCCESS;
}

/** \ingroup libusb_desc
 * Get the USB device descriptor for a given device.
 *
 * This is a non-blocking function; the device descriptor is cached in memory.
 *
 * Note since libusb-1.0.16, \ref LIBUSB_API_VERSION >= 0x01000102, this
 * function always succeeds.
 *
 * \param dev the device
 * \param desc output location for the descriptor data
 * \returns 0 on success or a LIBUSB_ERROR code on failure
 */
int API_EXPORTED libusb_get_device_descriptor(libusb_device *dev,
	struct libusb_device_descriptor *desc)
{
	usbi_dbg("");
	memcpy((unsigned char *) desc, (unsigned char *) &dev->device_descriptor,
	       sizeof (dev->device_descriptor));
	return 0;
}

/** \ingroup libusb_desc
 * Get the USB configuration descriptor for the currently active configuration.
 * This is a non-blocking function which does not involve any requests being
 * sent to the device.
 *
 * \param dev a device
 * \param config output location for the USB configuration descriptor. Only
 * valid if 0 was returned. Must be freed with libusb_free_config_descriptor()
 * after use.
 * \returns 0 on success
 * \returns LIBUSB_ERROR_NOT_FOUND if the device is in unconfigured state
 * \returns another LIBUSB_ERROR code on error
 * \see libusb_get_config_descriptor
 */
int API_EXPORTED libusb_get_active_config_descriptor(libusb_device *dev,
	struct libusb_config_descriptor **config)
{
	struct libusb_config_descriptor _config;
	unsigned char tmp[LIBUSB_DT_CONFIG_SIZE];
	unsigned char *buf = NULL;
	int host_endian = 0;
	int r;

	r = usbi_backend.get_active_config_descriptor(dev, tmp,
		LIBUSB_DT_CONFIG_SIZE, &host_endian);
	if (r < 0)
		return r;
	if (r < LIBUSB_DT_CONFIG_SIZE) {
		usbi_err(dev->ctx, "short config descriptor read %d/%d",
			 r, LIBUSB_DT_CONFIG_SIZE);
		return LIBUSB_ERROR_IO;
	}

	usbi_parse_descriptor(tmp, "bbw", &_config, host_endian);
	buf = malloc(_config.wTotalLength);
	if (!buf)
		return LIBUSB_ERROR_NO_MEM;

	r = usbi_backend.get_active_config_descriptor(dev, buf,
		_config.wTotalLength, &host_endian);
	if (r >= 0)
		r = raw_desc_to_config(dev->ctx, buf, r, host_endian, config);

	free(buf);
	return r;
}

/** \ingroup libusb_desc
 * Get a USB configuration descriptor based on its index.
 * This is a non-blocking function which does not involve any requests being
 * sent to the device.
 *
 * \param dev a device
 * \param config_index the index of the configuration you wish to retrieve
 * \param config output location for the USB configuration descriptor. Only
 * valid if 0 was returned. Must be freed with libusb_free_config_descriptor()
 * after use.
 * \returns 0 on success
 * \returns LIBUSB_ERROR_NOT_FOUND if the configuration does not exist
 * \returns another LIBUSB_ERROR code on error
 * \see libusb_get_active_config_descriptor()
 * \see libusb_get_config_descriptor_by_value()
 */
int API_EXPORTED libusb_get_config_descriptor(libusb_device *dev,
	uint8_t config_index, struct libusb_config_descriptor **config)
{
	struct libusb_config_descriptor _config;
	unsigned char tmp[LIBUSB_DT_CONFIG_SIZE];
	unsigned char *buf = NULL;
	int host_endian = 0;
	int r;

	usbi_dbg("index %d", config_index);
	if (config_index >= dev->num_configurations)
		return LIBUSB_ERROR_NOT_FOUND;

	r = usbi_backend.get_config_descriptor(dev, config_index, tmp,
		LIBUSB_DT_CONFIG_SIZE, &host_endian);
	if (r < 0)
		return r;
	if (r < LIBUSB_DT_CONFIG_SIZE) {
		usbi_err(dev->ctx, "short config descriptor read %d/%d",
			 r, LIBUSB_DT_CONFIG_SIZE);
		return LIBUSB_ERROR_IO;
	}

	usbi_parse_descriptor(tmp, "bbw", &_config, host_endian);
	buf = malloc(_config.wTotalLength);
	if (!buf)
		return LIBUSB_ERROR_NO_MEM;

	r = usbi_backend.get_config_descriptor(dev, config_index, buf,
		_config.wTotalLength, &host_endian);
	if (r >= 0)
		r = raw_desc_to_config(dev->ctx, buf, r, host_endian, config);

	free(buf);
	return r;
}

/* iterate through all configurations, returning the index of the configuration
 * matching a specific bConfigurationValue in the idx output parameter, or -1
 * if the config was not found.
 * returns 0 on success or a LIBUSB_ERROR code
 */
int usbi_get_config_index_by_value(struct libusb_device *dev,
	uint8_t bConfigurationValue, int *idx)
{
	uint8_t i;

	usbi_dbg("value %d", bConfigurationValue);
	for (i = 0; i < dev->num_configurations; i++) {
		unsigned char tmp[6];
		int host_endian;
		int r = usbi_backend.get_config_descriptor(dev, i, tmp, sizeof(tmp),
			&host_endian);
		if (r < 0) {
			*idx = -1;
			return r;
		}
		if (tmp[5] == bConfigurationValue) {
			*idx = i;
			return 0;
		}
	}

	*idx = -1;
	return 0;
}

/** \ingroup libusb_desc
 * Get a USB configuration descriptor with a specific bConfigurationValue.
 * This is a non-blocking function which does not involve any requests being
 * sent to the device.
 *
 * \param dev a device
 * \param bConfigurationValue the bConfigurationValue of the configuration you
 * wish to retrieve
 * \param config output location for the USB configuration descriptor. Only
 * valid if 0 was returned. Must be freed with libusb_free_config_descriptor()
 * after use.
 * \returns 0 on success
 * \returns LIBUSB_ERROR_NOT_FOUND if the configuration does not exist
 * \returns another LIBUSB_ERROR code on error
 * \see libusb_get_active_config_descriptor()
 * \see libusb_get_config_descriptor()
 */
int API_EXPORTED libusb_get_config_descriptor_by_value(libusb_device *dev,
	uint8_t bConfigurationValue, struct libusb_config_descriptor **config)
{
	int r, idx, host_endian;
	unsigned char *buf = NULL;

	if (usbi_backend.get_config_descriptor_by_value) {
		r = usbi_backend.get_config_descriptor_by_value(dev,
			bConfigurationValue, &buf, &host_endian);
		if (r < 0)
			return r;
		return raw_desc_to_config(dev->ctx, buf, r, host_endian, config);
	}

	r = usbi_get_config_index_by_value(dev, bConfigurationValue, &idx);
	if (r < 0)
		return r;
	else if (idx == -1)
		return LIBUSB_ERROR_NOT_FOUND;
	else
		return libusb_get_config_descriptor(dev, (uint8_t) idx, config);
}

/** \ingroup libusb_desc
 * Free a configuration descriptor obtained from
 * libusb_get_active_config_descriptor() or libusb_get_config_descriptor().
 * It is safe to call this function with a NULL config parameter, in which
 * case the function simply returns.
 *
 * \param config the configuration descriptor to free
 */
void API_EXPORTED libusb_free_config_descriptor(
	struct libusb_config_descriptor *config)
{
	if (!config)
		return;

	clear_configuration(config);
	free(config);
}

/** \ingroup libusb_desc
 * Get an endpoints superspeed endpoint companion descriptor (if any)
 *
 * \param ctx the context to operate on, or NULL for the default context
 * \param endpoint endpoint descriptor from which to get the superspeed
 * endpoint companion descriptor
 * \param ep_comp output location for the superspeed endpoint companion
 * descriptor. Only valid if 0 was returned. Must be freed with
 * libusb_free_ss_endpoint_companion_descriptor() after use.
 * \returns 0 on success
 * \returns LIBUSB_ERROR_NOT_FOUND if the configuration does not exist
 * \returns another LIBUSB_ERROR code on error
 */
int API_EXPORTED libusb_get_ss_endpoint_companion_descriptor(
	struct libusb_context *ctx,
	const struct libusb_endpoint_descriptor *endpoint,
	struct libusb_ss_endpoint_companion_descriptor **ep_comp)
{
	struct usb_descriptor_header header;
	int size = endpoint->extra_length;
	const unsigned char *buffer = endpoint->extra;

	*ep_comp = NULL;

	while (size >= DESC_HEADER_LENGTH) {
		usbi_parse_descriptor(buffer, "bb", &header, 0);
		if (header.bLength < 2 || header.bLength > size) {
			usbi_err(ctx, "invalid descriptor length %d",
				 header.bLength);
			return LIBUSB_ERROR_IO;
		}
		if (header.bDescriptorType != LIBUSB_DT_SS_ENDPOINT_COMPANION) {
			buffer += header.bLength;
			size -= header.bLength;
			continue;
		}
		if (header.bLength < LIBUSB_DT_SS_ENDPOINT_COMPANION_SIZE) {
			usbi_err(ctx, "invalid ss-ep-comp-desc length %d",
				 header.bLength);
			return LIBUSB_ERROR_IO;
		}
		*ep_comp = malloc(sizeof(**ep_comp));
		if (*ep_comp == NULL)
			return LIBUSB_ERROR_NO_MEM;
		usbi_parse_descriptor(buffer, "bbbbw", *ep_comp, 0);
		return LIBUSB_SUCCESS;
	}
	return LIBUSB_ERROR_NOT_FOUND;
}

/** \ingroup libusb_desc
 * Free a superspeed endpoint companion descriptor obtained from
 * libusb_get_ss_endpoint_companion_descriptor().
 * It is safe to call this function with a NULL ep_comp parameter, in which
 * case the function simply returns.
 *
 * \param ep_comp the superspeed endpoint companion descriptor to free
 */
void API_EXPORTED libusb_free_ss_endpoint_companion_descriptor(
	struct libusb_ss_endpoint_companion_descriptor *ep_comp)
{
	free(ep_comp);
}

static int parse_bos(struct libusb_context *ctx,
	struct libusb_bos_descriptor **bos,
	unsigned char *buffer, int size, int host_endian)
{
	struct libusb_bos_descriptor bos_header, *_bos;
	struct libusb_bos_dev_capability_descriptor dev_cap;
	int i;

	if (size < LIBUSB_DT_BOS_SIZE) {
		usbi_err(ctx, "short bos descriptor read %d/%d",
			 size, LIBUSB_DT_BOS_SIZE);
		return LIBUSB_ERROR_IO;
	}

	usbi_parse_descriptor(buffer, "bbwb", &bos_header, host_endian);
	if (bos_header.bDescriptorType != LIBUSB_DT_BOS) {
		usbi_err(ctx, "unexpected descriptor %x (expected %x)",
			 bos_header.bDescriptorType, LIBUSB_DT_BOS);
		return LIBUSB_ERROR_IO;
	}
	if (bos_header.bLength < LIBUSB_DT_BOS_SIZE) {
		usbi_err(ctx, "invalid bos bLength (%d)", bos_header.bLength);
		return LIBUSB_ERROR_IO;
	}
	if (bos_header.bLength > size) {
		usbi_err(ctx, "short bos descriptor read %d/%d",
			 size, bos_header.bLength);
		return LIBUSB_ERROR_IO;
	}

	_bos = calloc (1,
		sizeof(*_bos) + bos_header.bNumDeviceCaps * sizeof(void *));
	if (!_bos)
		return LIBUSB_ERROR_NO_MEM;

	usbi_parse_descriptor(buffer, "bbwb", _bos, host_endian);
	buffer += bos_header.bLength;
	size -= bos_header.bLength;

	/* Get the device capability descriptors */
	for (i = 0; i < bos_header.bNumDeviceCaps; i++) {
		if (size < LIBUSB_DT_DEVICE_CAPABILITY_SIZE) {
			usbi_warn(ctx, "short dev-cap descriptor read %d/%d",
				  size, LIBUSB_DT_DEVICE_CAPABILITY_SIZE);
			break;
		}
		usbi_parse_descriptor(buffer, "bbb", &dev_cap, host_endian);
		if (dev_cap.bDescriptorType != LIBUSB_DT_DEVICE_CAPABILITY) {
			usbi_warn(ctx, "unexpected descriptor %x (expected %x)",
				  dev_cap.bDescriptorType, LIBUSB_DT_DEVICE_CAPABILITY);
			break;
		}
		if (dev_cap.bLength < LIBUSB_DT_DEVICE_CAPABILITY_SIZE) {
			usbi_err(ctx, "invalid dev-cap bLength (%d)",
				 dev_cap.bLength);
			libusb_free_bos_descriptor(_bos);
			return LIBUSB_ERROR_IO;
		}
		if (dev_cap.bLength > size) {
			usbi_warn(ctx, "short dev-cap descriptor read %d/%d",
				  size, dev_cap.bLength);
			break;
		}

		_bos->dev_capability[i] = malloc(dev_cap.bLength);
		if (!_bos->dev_capability[i]) {
			libusb_free_bos_descriptor(_bos);
			return LIBUSB_ERROR_NO_MEM;
		}
		memcpy(_bos->dev_capability[i], buffer, dev_cap.bLength);
		buffer += dev_cap.bLength;
		size -= dev_cap.bLength;
	}
	_bos->bNumDeviceCaps = (uint8_t)i;
	*bos = _bos;

	return LIBUSB_SUCCESS;
}

/** \ingroup libusb_desc
 * Get a Binary Object Store (BOS) descriptor
 * This is a BLOCKING function, which will send requests to the device.
 *
 * \param dev_handle the handle of an open libusb device
 * \param bos output location for the BOS descriptor. Only valid if 0 was returned.
 * Must be freed with \ref libusb_free_bos_descriptor() after use.
 * \returns 0 on success
 * \returns LIBUSB_ERROR_NOT_FOUND if the device doesn't have a BOS descriptor
 * \returns another LIBUSB_ERROR code on error
 */
int API_EXPORTED libusb_get_bos_descriptor(libusb_device_handle *dev_handle,
	struct libusb_bos_descriptor **bos)
{
	struct libusb_bos_descriptor _bos;
	uint8_t bos_header[LIBUSB_DT_BOS_SIZE] = {0};
	unsigned char *bos_data = NULL;
	const int host_endian = 0;
	int r;

	/* Read the BOS. This generates 2 requests on the bus,
	 * one for the header, and one for the full BOS */
	r = libusb_get_descriptor(dev_handle, LIBUSB_DT_BOS, 0, bos_header,
				  LIBUSB_DT_BOS_SIZE);
	if (r < 0) {
		if (r != LIBUSB_ERROR_PIPE)
			usbi_err(HANDLE_CTX(dev_handle), "failed to read BOS (%d)", r);
		return r;
	}
	if (r < LIBUSB_DT_BOS_SIZE) {
		usbi_err(HANDLE_CTX(dev_handle), "short BOS read %d/%d",
			 r, LIBUSB_DT_BOS_SIZE);
		return LIBUSB_ERROR_IO;
	}

	usbi_parse_descriptor(bos_header, "bbwb", &_bos, host_endian);
	usbi_dbg("found BOS descriptor: size %d bytes, %d capabilities",
		 _bos.wTotalLength, _bos.bNumDeviceCaps);
	bos_data = calloc(_bos.wTotalLength, 1);
	if (bos_data == NULL)
		return LIBUSB_ERROR_NO_MEM;

	r = libusb_get_descriptor(dev_handle, LIBUSB_DT_BOS, 0, bos_data,
				  _bos.wTotalLength);
	if (r >= 0)
		r = parse_bos(HANDLE_CTX(dev_handle), bos, bos_data, r, host_endian);
	else
		usbi_err(HANDLE_CTX(dev_handle), "failed to read BOS (%d)", r);

	free(bos_data);
	return r;
}

/** \ingroup libusb_desc
 * Free a BOS descriptor obtained from libusb_get_bos_descriptor().
 * It is safe to call this function with a NULL bos parameter, in which
 * case the function simply returns.
 *
 * \param bos the BOS descriptor to free
 */
void API_EXPORTED libusb_free_bos_descriptor(struct libusb_bos_descriptor *bos)
{
	int i;

	if (!bos)
		return;

	for (i = 0; i < bos->bNumDeviceCaps; i++)
		free(bos->dev_capability[i]);
	free(bos);
}

/** \ingroup libusb_desc
 * Get an USB 2.0 Extension descriptor
 *
 * \param ctx the context to operate on, or NULL for the default context
 * \param dev_cap Device Capability descriptor with a bDevCapabilityType of
 * \ref libusb_capability_type::LIBUSB_BT_USB_2_0_EXTENSION
 * LIBUSB_BT_USB_2_0_EXTENSION
 * \param usb_2_0_extension output location for the USB 2.0 Extension
 * descriptor. Only valid if 0 was returned. Must be freed with
 * libusb_free_usb_2_0_extension_descriptor() after use.
 * \returns 0 on success
 * \returns a LIBUSB_ERROR code on error
 */
int API_EXPORTED libusb_get_usb_2_0_extension_descriptor(
	struct libusb_context *ctx,
	struct libusb_bos_dev_capability_descriptor *dev_cap,
	struct libusb_usb_2_0_extension_descriptor **usb_2_0_extension)
{
	struct libusb_usb_2_0_extension_descriptor *_usb_2_0_extension;
	const int host_endian = 0;

	if (dev_cap->bDevCapabilityType != LIBUSB_BT_USB_2_0_EXTENSION) {
		usbi_err(ctx, "unexpected bDevCapabilityType %x (expected %x)",
			 dev_cap->bDevCapabilityType,
			 LIBUSB_BT_USB_2_0_EXTENSION);
		return LIBUSB_ERROR_INVALID_PARAM;
	}
	if (dev_cap->bLength < LIBUSB_BT_USB_2_0_EXTENSION_SIZE) {
		usbi_err(ctx, "short dev-cap descriptor read %d/%d",
			 dev_cap->bLength, LIBUSB_BT_USB_2_0_EXTENSION_SIZE);
		return LIBUSB_ERROR_IO;
	}

	_usb_2_0_extension = malloc(sizeof(*_usb_2_0_extension));
	if (!_usb_2_0_extension)
		return LIBUSB_ERROR_NO_MEM;

	usbi_parse_descriptor((unsigned char *)dev_cap, "bbbd",
			      _usb_2_0_extension, host_endian);

	*usb_2_0_extension = _usb_2_0_extension;
	return LIBUSB_SUCCESS;
}

/** \ingroup libusb_desc
 * Free a USB 2.0 Extension descriptor obtained from
 * libusb_get_usb_2_0_extension_descriptor().
 * It is safe to call this function with a NULL usb_2_0_extension parameter,
 * in which case the function simply returns.
 *
 * \param usb_2_0_extension the USB 2.0 Extension descriptor to free
 */
void API_EXPORTED libusb_free_usb_2_0_extension_descriptor(
	struct libusb_usb_2_0_extension_descriptor *usb_2_0_extension)
{
	free(usb_2_0_extension);
}

/** \ingroup libusb_desc
 * Get a SuperSpeed USB Device Capability descriptor
 *
 * \param ctx the context to operate on, or NULL for the default context
 * \param dev_cap Device Capability descriptor with a bDevCapabilityType of
 * \ref libusb_capability_type::LIBUSB_BT_SS_USB_DEVICE_CAPABILITY
 * LIBUSB_BT_SS_USB_DEVICE_CAPABILITY
 * \param ss_usb_device_cap output location for the SuperSpeed USB Device
 * Capability descriptor. Only valid if 0 was returned. Must be freed with
 * libusb_free_ss_usb_device_capability_descriptor() after use.
 * \returns 0 on success
 * \returns a LIBUSB_ERROR code on error
 */
int API_EXPORTED libusb_get_ss_usb_device_capability_descriptor(
	struct libusb_context *ctx,
	struct libusb_bos_dev_capability_descriptor *dev_cap,
	struct libusb_ss_usb_device_capability_descriptor **ss_usb_device_cap)
{
	struct libusb_ss_usb_device_capability_descriptor *_ss_usb_device_cap;
	const int host_endian = 0;

	if (dev_cap->bDevCapabilityType != LIBUSB_BT_SS_USB_DEVICE_CAPABILITY) {
		usbi_err(ctx, "unexpected bDevCapabilityType %x (expected %x)",
			 dev_cap->bDevCapabilityType,
			 LIBUSB_BT_SS_USB_DEVICE_CAPABILITY);
		return LIBUSB_ERROR_INVALID_PARAM;
	}
	if (dev_cap->bLength < LIBUSB_BT_SS_USB_DEVICE_CAPABILITY_SIZE) {
		usbi_err(ctx, "short dev-cap descriptor read %d/%d",
			 dev_cap->bLength, LIBUSB_BT_SS_USB_DEVICE_CAPABILITY_SIZE);
		return LIBUSB_ERROR_IO;
	}

	_ss_usb_device_cap = malloc(sizeof(*_ss_usb_device_cap));
	if (!_ss_usb_device_cap)
		return LIBUSB_ERROR_NO_MEM;

	usbi_parse_descriptor((unsigned char *)dev_cap, "bbbbwbbw",
			      _ss_usb_device_cap, host_endian);

	*ss_usb_device_cap = _ss_usb_device_cap;
	return LIBUSB_SUCCESS;
}

/** \ingroup libusb_desc
 * Free a SuperSpeed USB Device Capability descriptor obtained from
 * libusb_get_ss_usb_device_capability_descriptor().
 * It is safe to call this function with a NULL ss_usb_device_cap
 * parameter, in which case the function simply returns.
 *
 * \param ss_usb_device_cap the USB 2.0 Extension descriptor to free
 */
void API_EXPORTED libusb_free_ss_usb_device_capability_descriptor(
	struct libusb_ss_usb_device_capability_descriptor *ss_usb_device_cap)
{
	free(ss_usb_device_cap);
}

/** \ingroup libusb_desc
 * Get a Container ID descriptor
 *
 * \param ctx the context to operate on, or NULL for the default context
 * \param dev_cap Device Capability descriptor with a bDevCapabilityType of
 * \ref libusb_capability_type::LIBUSB_BT_CONTAINER_ID
 * LIBUSB_BT_CONTAINER_ID
 * \param container_id output location for the Container ID descriptor.
 * Only valid if 0 was returned. Must be freed with
 * libusb_free_container_id_descriptor() after use.
 * \returns 0 on success
 * \returns a LIBUSB_ERROR code on error
 */
int API_EXPORTED libusb_get_container_id_descriptor(struct libusb_context *ctx,
	struct libusb_bos_dev_capability_descriptor *dev_cap,
	struct libusb_container_id_descriptor **container_id)
{
	struct libusb_container_id_descriptor *_container_id;
	const int host_endian = 0;

	if (dev_cap->bDevCapabilityType != LIBUSB_BT_CONTAINER_ID) {
		usbi_err(ctx, "unexpected bDevCapabilityType %x (expected %x)",
			 dev_cap->bDevCapabilityType,
			 LIBUSB_BT_CONTAINER_ID);
		return LIBUSB_ERROR_INVALID_PARAM;
	}
	if (dev_cap->bLength < LIBUSB_BT_CONTAINER_ID_SIZE) {
		usbi_err(ctx, "short dev-cap descriptor read %d/%d",
			 dev_cap->bLength, LIBUSB_BT_CONTAINER_ID_SIZE);
		return LIBUSB_ERROR_IO;
	}

	_container_id = malloc(sizeof(*_container_id));
	if (!_container_id)
		return LIBUSB_ERROR_NO_MEM;

	usbi_parse_descriptor((unsigned char *)dev_cap, "bbbbu",
			      _container_id, host_endian);

	*container_id = _container_id;
	return LIBUSB_SUCCESS;
}

/** \ingroup libusb_desc
 * Free a Container ID descriptor obtained from
 * libusb_get_container_id_descriptor().
 * It is safe to call this function with a NULL container_id parameter,
 * in which case the function simply returns.
 *
 * \param container_id the USB 2.0 Extension descriptor to free
 */
void API_EXPORTED libusb_free_container_id_descriptor(
	struct libusb_container_id_descriptor *container_id)
{
	free(container_id);
}

/** \ingroup libusb_desc
 * Retrieve a string descriptor in C style ASCII.
 *
 * Wrapper around libusb_get_string_descriptor(). Uses the first language
 * supported by the device.
 *
 * \param dev_handle a device handle
 * \param desc_index the index of the descriptor to retrieve
 * \param data output buffer for ASCII string descriptor
 * \param length size of data buffer
 * \returns number of bytes returned in data, or LIBUSB_ERROR code on failure
 */
int API_EXPORTED libusb_get_string_descriptor_ascii(libusb_device_handle *dev_handle,
	uint8_t desc_index, unsigned char *data, int length)
{
	unsigned char tbuf[255]; /* Some devices choke on size > 255 */
	int r, si, di;
	uint16_t langid;

	/* Asking for the zero'th index is special - it returns a string
	 * descriptor that contains all the language IDs supported by the
	 * device. Typically there aren't many - often only one. Language
	 * IDs are 16 bit numbers, and they start at the third byte in the
	 * descriptor. There's also no point in trying to read descriptor 0
	 * with this function. See USB 2.0 specification section 9.6.7 for
	 * more information.
	 */

	if (desc_index == 0)
		return LIBUSB_ERROR_INVALID_PARAM;

	r = libusb_get_string_descriptor(dev_handle, 0, 0, tbuf, sizeof(tbuf));
	if (r < 0)
		return r;

	if (r < 4)
		return LIBUSB_ERROR_IO;

	langid = tbuf[2] | (tbuf[3] << 8);

	r = libusb_get_string_descriptor(dev_handle, desc_index, langid, tbuf,
		sizeof(tbuf));
	if (r < 0)
		return r;

	if (tbuf[1] != LIBUSB_DT_STRING)
		return LIBUSB_ERROR_IO;

	if (tbuf[0] > r)
		return LIBUSB_ERROR_IO;

	for (di = 0, si = 2; si < tbuf[0]; si += 2) {
		if (di >= (length - 1))
			break;

		if ((tbuf[si] & 0x80) || (tbuf[si + 1])) /* non-ASCII */
			data[di++] = '?';
		else
			data[di++] = tbuf[si];
	}

	data[di] = 0;
	return di;
}
