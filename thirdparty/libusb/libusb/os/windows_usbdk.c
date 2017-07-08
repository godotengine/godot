/*
 * windows UsbDk backend for libusb 1.0
 * Copyright © 2014 Red Hat, Inc.

 * Authors:
 * Dmitry Fleytman <dmitry@daynix.com>
 * Pavel Gurvich <pavel@daynix.com>
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

#if defined(USE_USBDK)

#include <windows.h>
#include <cfgmgr32.h>
#include <stdio.h>

#include "libusbi.h"
#include "windows_common.h"
#include "windows_nt_common.h"

#define ULONG64 uint64_t
#define PVOID64 uint64_t

typedef CONST WCHAR *PCWCHAR;
#define wcsncpy_s wcsncpy

#include "windows_usbdk.h"

#if !defined(STATUS_SUCCESS)
typedef LONG NTSTATUS;
#define STATUS_SUCCESS			((NTSTATUS)0x00000000L)
#endif

#if !defined(STATUS_CANCELLED)
#define STATUS_CANCELLED		((NTSTATUS)0xC0000120L)
#endif

#if !defined(STATUS_REQUEST_CANCELED)
#define STATUS_REQUEST_CANCELED		((NTSTATUS)0xC0000703L)
#endif

#if !defined(USBD_SUCCESS)
typedef int32_t USBD_STATUS;
#define USBD_SUCCESS(Status)		((USBD_STATUS) (Status) >= 0)
#define USBD_PENDING(Status)		((ULONG) (Status) >> 30 == 1)
#define USBD_ERROR(Status)		((USBD_STATUS) (Status) < 0)
#define USBD_STATUS_STALL_PID		((USBD_STATUS) 0xc0000004)
#define USBD_STATUS_ENDPOINT_HALTED	((USBD_STATUS) 0xc0000030)
#define USBD_STATUS_BAD_START_FRAME	((USBD_STATUS) 0xc0000a00)
#define USBD_STATUS_TIMEOUT		((USBD_STATUS) 0xc0006000)
#define USBD_STATUS_CANCELED		((USBD_STATUS) 0xc0010000)
#endif

static int concurrent_usage = -1;

struct usbdk_device_priv {
	USB_DK_DEVICE_INFO info;
	PUSB_CONFIGURATION_DESCRIPTOR *config_descriptors;
	HANDLE redirector_handle;
	uint8_t active_configuration;
};

struct usbdk_transfer_priv {
	USB_DK_TRANSFER_REQUEST request;
	struct winfd pollable_fd;
	PULONG64 IsochronousPacketsArray;
	PUSB_DK_ISO_TRANSFER_RESULT IsochronousResultsArray;
};

static inline struct usbdk_device_priv *_usbdk_device_priv(struct libusb_device *dev)
{
	return (struct usbdk_device_priv *)dev->os_priv;
}

static inline struct usbdk_transfer_priv *_usbdk_transfer_priv(struct usbi_transfer *itransfer)
{
	return (struct usbdk_transfer_priv *)usbi_transfer_get_os_priv(itransfer);
}

static struct {
	HMODULE module;

	USBDK_GET_DEVICES_LIST			GetDevicesList;
	USBDK_RELEASE_DEVICES_LIST		ReleaseDevicesList;
	USBDK_START_REDIRECT			StartRedirect;
	USBDK_STOP_REDIRECT			StopRedirect;
	USBDK_GET_CONFIGURATION_DESCRIPTOR	GetConfigurationDescriptor;
	USBDK_RELEASE_CONFIGURATION_DESCRIPTOR	ReleaseConfigurationDescriptor;
	USBDK_READ_PIPE				ReadPipe;
	USBDK_WRITE_PIPE			WritePipe;
	USBDK_ABORT_PIPE			AbortPipe;
	USBDK_RESET_PIPE			ResetPipe;
	USBDK_SET_ALTSETTING			SetAltsetting;
	USBDK_RESET_DEVICE			ResetDevice;
	USBDK_GET_REDIRECTOR_SYSTEM_HANDLE	GetRedirectorSystemHandle;
} usbdk_helper;

static FARPROC get_usbdk_proc_addr(struct libusb_context *ctx, LPCSTR api_name)
{
	FARPROC api_ptr = GetProcAddress(usbdk_helper.module, api_name);

	if (api_ptr == NULL)
		usbi_err(ctx, "UsbDkHelper API %s not found, error %d", api_name, GetLastError());

	return api_ptr;
}

static void unload_usbdk_helper_dll(void)
{
	if (usbdk_helper.module != NULL) {
		FreeLibrary(usbdk_helper.module);
		usbdk_helper.module = NULL;
	}
}

static int load_usbdk_helper_dll(struct libusb_context *ctx)
{
	usbdk_helper.module = LoadLibraryA("UsbDkHelper");
	if (usbdk_helper.module == NULL) {
		usbi_err(ctx, "Failed to load UsbDkHelper.dll, error %d", GetLastError());
		return LIBUSB_ERROR_NOT_FOUND;
	}

	usbdk_helper.GetDevicesList = (USBDK_GET_DEVICES_LIST)get_usbdk_proc_addr(ctx, "UsbDk_GetDevicesList");
	if (usbdk_helper.GetDevicesList == NULL)
		goto error_unload;

	usbdk_helper.ReleaseDevicesList = (USBDK_RELEASE_DEVICES_LIST)get_usbdk_proc_addr(ctx, "UsbDk_ReleaseDevicesList");
	if (usbdk_helper.ReleaseDevicesList == NULL)
		goto error_unload;

	usbdk_helper.StartRedirect = (USBDK_START_REDIRECT)get_usbdk_proc_addr(ctx, "UsbDk_StartRedirect");
	if (usbdk_helper.StartRedirect == NULL)
		goto error_unload;

	usbdk_helper.StopRedirect = (USBDK_STOP_REDIRECT)get_usbdk_proc_addr(ctx, "UsbDk_StopRedirect");
	if (usbdk_helper.StopRedirect == NULL)
		goto error_unload;

	usbdk_helper.GetConfigurationDescriptor = (USBDK_GET_CONFIGURATION_DESCRIPTOR)get_usbdk_proc_addr(ctx, "UsbDk_GetConfigurationDescriptor");
	if (usbdk_helper.GetConfigurationDescriptor == NULL)
		goto error_unload;

	usbdk_helper.ReleaseConfigurationDescriptor = (USBDK_RELEASE_CONFIGURATION_DESCRIPTOR)get_usbdk_proc_addr(ctx, "UsbDk_ReleaseConfigurationDescriptor");
	if (usbdk_helper.ReleaseConfigurationDescriptor == NULL)
		goto error_unload;

	usbdk_helper.ReadPipe = (USBDK_READ_PIPE)get_usbdk_proc_addr(ctx, "UsbDk_ReadPipe");
	if (usbdk_helper.ReadPipe == NULL)
		goto error_unload;

	usbdk_helper.WritePipe = (USBDK_WRITE_PIPE)get_usbdk_proc_addr(ctx, "UsbDk_WritePipe");
	if (usbdk_helper.WritePipe == NULL)
		goto error_unload;

	usbdk_helper.AbortPipe = (USBDK_ABORT_PIPE)get_usbdk_proc_addr(ctx, "UsbDk_AbortPipe");
	if (usbdk_helper.AbortPipe == NULL)
		goto error_unload;

	usbdk_helper.ResetPipe = (USBDK_RESET_PIPE)get_usbdk_proc_addr(ctx, "UsbDk_ResetPipe");
	if (usbdk_helper.ResetPipe == NULL)
		goto error_unload;

	usbdk_helper.SetAltsetting = (USBDK_SET_ALTSETTING)get_usbdk_proc_addr(ctx, "UsbDk_SetAltsetting");
	if (usbdk_helper.SetAltsetting == NULL)
		goto error_unload;

	usbdk_helper.ResetDevice = (USBDK_RESET_DEVICE)get_usbdk_proc_addr(ctx, "UsbDk_ResetDevice");
	if (usbdk_helper.ResetDevice == NULL)
		goto error_unload;

	usbdk_helper.GetRedirectorSystemHandle = (USBDK_GET_REDIRECTOR_SYSTEM_HANDLE)get_usbdk_proc_addr(ctx, "UsbDk_GetRedirectorSystemHandle");
	if (usbdk_helper.GetRedirectorSystemHandle == NULL)
		goto error_unload;

	return LIBUSB_SUCCESS;

error_unload:
	FreeLibrary(usbdk_helper.module);
	usbdk_helper.module = NULL;
	return LIBUSB_ERROR_NOT_FOUND;
}

static int usbdk_init(struct libusb_context *ctx)
{
	int r;

	if (++concurrent_usage == 0) { // First init?
		r = load_usbdk_helper_dll(ctx);
		if (r)
			goto init_exit;

		init_polling();

		r = windows_common_init(ctx);
		if (r)
			goto init_exit;
	}
	// At this stage, either we went through full init successfully, or didn't need to
	r = LIBUSB_SUCCESS;

init_exit:
	if (!concurrent_usage && r != LIBUSB_SUCCESS) { // First init failed?
		exit_polling();
		windows_common_exit();
		unload_usbdk_helper_dll();
	}

	if (r != LIBUSB_SUCCESS)
		--concurrent_usage; // Not expected to call libusb_exit if we failed.

	return r;
}

static int usbdk_get_session_id_for_device(struct libusb_context *ctx,
	PUSB_DK_DEVICE_ID id, unsigned long *session_id)
{
	char dev_identity[ARRAYSIZE(id->DeviceID) + ARRAYSIZE(id->InstanceID)];

	if (sprintf(dev_identity, "%S%S", id->DeviceID, id->InstanceID) == -1) {
		usbi_warn(ctx, "cannot form device identity", id->DeviceID);
		return LIBUSB_ERROR_NOT_SUPPORTED;
	}

	*session_id = htab_hash(dev_identity);

	return LIBUSB_SUCCESS;
}

static void usbdk_release_config_descriptors(struct usbdk_device_priv *p, uint8_t count)
{
	uint8_t i;

	for (i = 0; i < count; i++)
		usbdk_helper.ReleaseConfigurationDescriptor(p->config_descriptors[i]);

	free(p->config_descriptors);
	p->config_descriptors = NULL;
}

static int usbdk_cache_config_descriptors(struct libusb_context *ctx,
	struct usbdk_device_priv *p, PUSB_DK_DEVICE_INFO info)
{
	uint8_t i;
	USB_DK_CONFIG_DESCRIPTOR_REQUEST Request;
	Request.ID = info->ID;

	p->config_descriptors = calloc(info->DeviceDescriptor.bNumConfigurations, sizeof(PUSB_CONFIGURATION_DESCRIPTOR));
	if (p->config_descriptors == NULL) {
		usbi_err(ctx, "failed to allocate configuration descriptors holder");
		return LIBUSB_ERROR_NO_MEM;
	}

	for (i = 0; i < info->DeviceDescriptor.bNumConfigurations; i++) {
		ULONG Length;

		Request.Index = i;
		if (!usbdk_helper.GetConfigurationDescriptor(&Request, &p->config_descriptors[i], &Length)) {
			usbi_err(ctx, "failed to retrieve configuration descriptors");
			usbdk_release_config_descriptors(p, i);
			return LIBUSB_ERROR_OTHER;
		}
	}

	return LIBUSB_SUCCESS;
}

static inline int usbdk_device_priv_init(struct libusb_context *ctx, struct libusb_device *dev, PUSB_DK_DEVICE_INFO info)
{
	struct usbdk_device_priv *p = _usbdk_device_priv(dev);

	p->info = *info;
	p->active_configuration = 0;

	return usbdk_cache_config_descriptors(ctx, p, info);
}

static void usbdk_device_init(libusb_device *dev, PUSB_DK_DEVICE_INFO info)
{
	dev->bus_number = (uint8_t)info->FilterID;
	dev->port_number = (uint8_t)info->Port;
	dev->parent_dev = NULL;

	//Addresses in libusb are 1-based
	dev->device_address = (uint8_t)(info->Port + 1);

	dev->num_configurations = info->DeviceDescriptor.bNumConfigurations;
	dev->device_descriptor = info->DeviceDescriptor;

	switch (info->Speed) {
	case LowSpeed:
		dev->speed = LIBUSB_SPEED_LOW;
		break;
	case FullSpeed:
		dev->speed = LIBUSB_SPEED_FULL;
		break;
	case HighSpeed:
		dev->speed = LIBUSB_SPEED_HIGH;
		break;
	case SuperSpeed:
		dev->speed = LIBUSB_SPEED_SUPER;
		break;
	case NoSpeed:
	default:
		dev->speed = LIBUSB_SPEED_UNKNOWN;
		break;
	}
}

static int usbdk_get_device_list(struct libusb_context *ctx, struct discovered_devs **_discdevs)
{
	int r = LIBUSB_SUCCESS;
	ULONG i;
	struct discovered_devs *discdevs = NULL;
	ULONG dev_number;
	PUSB_DK_DEVICE_INFO devices;

	if(!usbdk_helper.GetDevicesList(&devices, &dev_number))
		return LIBUSB_ERROR_OTHER;

	for (i = 0; i < dev_number; i++) {
		unsigned long session_id;
		struct libusb_device *dev = NULL;

		if (usbdk_get_session_id_for_device(ctx, &devices[i].ID, &session_id))
			continue;

		dev = usbi_get_device_by_session_id(ctx, session_id);
		if (dev == NULL) {
			dev = usbi_alloc_device(ctx, session_id);
			if (dev == NULL) {
				usbi_err(ctx, "failed to allocate a new device structure");
				continue;
			}

			usbdk_device_init(dev, &devices[i]);
			if (usbdk_device_priv_init(ctx, dev, &devices[i]) != LIBUSB_SUCCESS) {
				libusb_unref_device(dev);
				continue;
			}
		}

		discdevs = discovered_devs_append(*_discdevs, dev);
		libusb_unref_device(dev);
		if (!discdevs) {
			usbi_err(ctx, "cannot append new device to list");
			r = LIBUSB_ERROR_NO_MEM;
			goto func_exit;
		}

		*_discdevs = discdevs;
	}

func_exit:
	usbdk_helper.ReleaseDevicesList(devices);
	return r;
}

static void usbdk_exit(void)
{
	if (--concurrent_usage < 0) {
		windows_common_exit();
		exit_polling();
		unload_usbdk_helper_dll();
	}
}

static int usbdk_get_device_descriptor(struct libusb_device *dev, unsigned char *buffer, int *host_endian)
{
	struct usbdk_device_priv *priv = _usbdk_device_priv(dev);

	memcpy(buffer, &priv->info.DeviceDescriptor, DEVICE_DESC_LENGTH);
	*host_endian = 0;

	return LIBUSB_SUCCESS;
}

static int usbdk_get_config_descriptor(struct libusb_device *dev, uint8_t config_index, unsigned char *buffer, size_t len, int *host_endian)
{
	struct usbdk_device_priv *priv = _usbdk_device_priv(dev);
	PUSB_CONFIGURATION_DESCRIPTOR config_header;
	size_t size;

	if (config_index >= dev->num_configurations)
		return LIBUSB_ERROR_INVALID_PARAM;

	config_header = (PUSB_CONFIGURATION_DESCRIPTOR)priv->config_descriptors[config_index];

	size = min(config_header->wTotalLength, len);
	memcpy(buffer, config_header, size);
	*host_endian = 0;

	return (int)size;
}

static inline int usbdk_get_active_config_descriptor(struct libusb_device *dev, unsigned char *buffer, size_t len, int *host_endian)
{
	return usbdk_get_config_descriptor(dev, _usbdk_device_priv(dev)->active_configuration,
			buffer, len, host_endian);
}

static int usbdk_open(struct libusb_device_handle *dev_handle)
{
	struct usbdk_device_priv *priv = _usbdk_device_priv(dev_handle->dev);

	priv->redirector_handle = usbdk_helper.StartRedirect(&priv->info.ID);
	if (priv->redirector_handle == INVALID_HANDLE_VALUE) {
		usbi_err(DEVICE_CTX(dev_handle->dev), "Redirector startup failed");
		return LIBUSB_ERROR_OTHER;
	}

	return LIBUSB_SUCCESS;
}

static void usbdk_close(struct libusb_device_handle *dev_handle)
{
	struct usbdk_device_priv *priv = _usbdk_device_priv(dev_handle->dev);

	if (!usbdk_helper.StopRedirect(priv->redirector_handle)) {
		struct libusb_context *ctx = DEVICE_CTX(dev_handle->dev);
		usbi_err(ctx, "Redirector shutdown failed");
	}
}

static int usbdk_get_configuration(struct libusb_device_handle *dev_handle, int *config)
{
	*config = _usbdk_device_priv(dev_handle->dev)->active_configuration;

	return LIBUSB_SUCCESS;
}

static int usbdk_set_configuration(struct libusb_device_handle *dev_handle, int config)
{
	UNUSED(dev_handle);
	UNUSED(config);
	return LIBUSB_SUCCESS;
}

static int usbdk_claim_interface(struct libusb_device_handle *dev_handle, int iface)
{
	UNUSED(dev_handle);
	UNUSED(iface);
	return LIBUSB_SUCCESS;
}

static int usbdk_set_interface_altsetting(struct libusb_device_handle *dev_handle, int iface, int altsetting)
{
	struct libusb_context *ctx = DEVICE_CTX(dev_handle->dev);
	struct usbdk_device_priv *priv = _usbdk_device_priv(dev_handle->dev);

	if (!usbdk_helper.SetAltsetting(priv->redirector_handle, iface, altsetting)) {
		usbi_err(ctx, "SetAltsetting failed: %s", windows_error_str(0));
		return LIBUSB_ERROR_NO_DEVICE;
	}

	return LIBUSB_SUCCESS;
}

static int usbdk_release_interface(struct libusb_device_handle *dev_handle, int iface)
{
	UNUSED(dev_handle);
	UNUSED(iface);
	return LIBUSB_SUCCESS;
}

static int usbdk_clear_halt(struct libusb_device_handle *dev_handle, unsigned char endpoint)
{
	struct libusb_context *ctx = DEVICE_CTX(dev_handle->dev);
	struct usbdk_device_priv *priv = _usbdk_device_priv(dev_handle->dev);

	if (!usbdk_helper.ResetPipe(priv->redirector_handle, endpoint)) {
		usbi_err(ctx, "ResetPipe failed: %s", windows_error_str(0));
		return LIBUSB_ERROR_NO_DEVICE;
	}

	return LIBUSB_SUCCESS;
}

static int usbdk_reset_device(struct libusb_device_handle *dev_handle)
{
	struct libusb_context *ctx = DEVICE_CTX(dev_handle->dev);
	struct usbdk_device_priv *priv = _usbdk_device_priv(dev_handle->dev);

	if (!usbdk_helper.ResetDevice(priv->redirector_handle)) {
		usbi_err(ctx, "ResetDevice failed: %s", windows_error_str(0));
		return LIBUSB_ERROR_NO_DEVICE;
	}

	return LIBUSB_SUCCESS;
}

static int usbdk_kernel_driver_active(struct libusb_device_handle *dev_handle, int iface)
{
	UNUSED(dev_handle);
	UNUSED(iface);
	return LIBUSB_ERROR_NOT_SUPPORTED;
}

static int usbdk_attach_kernel_driver(struct libusb_device_handle *dev_handle, int iface)
{
	UNUSED(dev_handle);
	UNUSED(iface);
	return LIBUSB_ERROR_NOT_SUPPORTED;
}

static int usbdk_detach_kernel_driver(struct libusb_device_handle *dev_handle, int iface)
{
	UNUSED(dev_handle);
	UNUSED(iface);
	return LIBUSB_ERROR_NOT_SUPPORTED;
}

static void usbdk_destroy_device(struct libusb_device *dev)
{
	struct usbdk_device_priv* p = _usbdk_device_priv(dev);

	if (p->config_descriptors != NULL)
		usbdk_release_config_descriptors(p, p->info.DeviceDescriptor.bNumConfigurations);
}

void windows_clear_transfer_priv(struct usbi_transfer *itransfer)
{
	struct usbdk_transfer_priv *transfer_priv = _usbdk_transfer_priv(itransfer);
	struct libusb_transfer *transfer = USBI_TRANSFER_TO_LIBUSB_TRANSFER(itransfer);

	usbi_free_fd(&transfer_priv->pollable_fd);

	if (transfer->type == LIBUSB_TRANSFER_TYPE_ISOCHRONOUS) {
		safe_free(transfer_priv->IsochronousPacketsArray);
		safe_free(transfer_priv->IsochronousResultsArray);
	}
}

static int usbdk_do_control_transfer(struct usbi_transfer *itransfer)
{
	struct libusb_transfer *transfer = USBI_TRANSFER_TO_LIBUSB_TRANSFER(itransfer);
	struct usbdk_device_priv *priv = _usbdk_device_priv(transfer->dev_handle->dev);
	struct usbdk_transfer_priv *transfer_priv = _usbdk_transfer_priv(itransfer);
	struct libusb_context *ctx = DEVICE_CTX(transfer->dev_handle->dev);
	struct winfd wfd;
	ULONG Length;
	TransferResult transResult;
	HANDLE sysHandle;

	sysHandle = usbdk_helper.GetRedirectorSystemHandle(priv->redirector_handle);

	wfd = usbi_create_fd(sysHandle, RW_READ, NULL, NULL);
	// Always use the handle returned from usbi_create_fd (wfd.handle)
	if (wfd.fd < 0)
		return LIBUSB_ERROR_NO_MEM;

	transfer_priv->request.Buffer = (PVOID64)(uintptr_t)transfer->buffer;
	transfer_priv->request.BufferLength = transfer->length;
	transfer_priv->request.TransferType = ControlTransferType;
	transfer_priv->pollable_fd = INVALID_WINFD;
	Length = (ULONG)transfer->length;

	if (IS_XFERIN(transfer))
		transResult = usbdk_helper.ReadPipe(priv->redirector_handle, &transfer_priv->request, wfd.overlapped);
	else
		transResult = usbdk_helper.WritePipe(priv->redirector_handle, &transfer_priv->request, wfd.overlapped);

	switch (transResult) {
	case TransferSuccess:
		wfd.overlapped->Internal = STATUS_COMPLETED_SYNCHRONOUSLY;
		wfd.overlapped->InternalHigh = (DWORD)Length;
		break;
	case TransferSuccessAsync:
		break;
	case TransferFailure:
		usbi_err(ctx, "ControlTransfer failed: %s", windows_error_str(0));
		usbi_free_fd(&wfd);
		return LIBUSB_ERROR_IO;
	}

	// Use priv_transfer to store data needed for async polling
	transfer_priv->pollable_fd = wfd;
	usbi_add_pollfd(ctx, transfer_priv->pollable_fd.fd, POLLIN);

	return LIBUSB_SUCCESS;
}

static int usbdk_do_bulk_transfer(struct usbi_transfer *itransfer)
{
	struct libusb_transfer *transfer = USBI_TRANSFER_TO_LIBUSB_TRANSFER(itransfer);
	struct usbdk_device_priv *priv = _usbdk_device_priv(transfer->dev_handle->dev);
	struct usbdk_transfer_priv *transfer_priv = _usbdk_transfer_priv(itransfer);
	struct libusb_context *ctx = DEVICE_CTX(transfer->dev_handle->dev);
	struct winfd wfd;
	TransferResult transferRes;
	HANDLE sysHandle;

	transfer_priv->request.Buffer = (PVOID64)(uintptr_t)transfer->buffer;
	transfer_priv->request.BufferLength = transfer->length;
	transfer_priv->request.EndpointAddress = transfer->endpoint;

	switch (transfer->type) {
	case LIBUSB_TRANSFER_TYPE_BULK:
		transfer_priv->request.TransferType = BulkTransferType;
		break;
	case LIBUSB_TRANSFER_TYPE_INTERRUPT:
		transfer_priv->request.TransferType = IntertuptTransferType;
		break;
	default:
		usbi_err(ctx, "Wrong transfer type (%d) in usbdk_do_bulk_transfer. %s", transfer->type, windows_error_str(0));
		return LIBUSB_ERROR_INVALID_PARAM;
	}

	transfer_priv->pollable_fd = INVALID_WINFD;

	sysHandle = usbdk_helper.GetRedirectorSystemHandle(priv->redirector_handle);

	wfd = usbi_create_fd(sysHandle, IS_XFERIN(transfer) ? RW_READ : RW_WRITE, NULL, NULL);
	// Always use the handle returned from usbi_create_fd (wfd.handle)
	if (wfd.fd < 0)
		return LIBUSB_ERROR_NO_MEM;

	if (IS_XFERIN(transfer))
		transferRes = usbdk_helper.ReadPipe(priv->redirector_handle, &transfer_priv->request, wfd.overlapped);
	else
		transferRes = usbdk_helper.WritePipe(priv->redirector_handle, &transfer_priv->request, wfd.overlapped);

	switch (transferRes) {
	case TransferSuccess:
		wfd.overlapped->Internal = STATUS_COMPLETED_SYNCHRONOUSLY;
		break;
	case TransferSuccessAsync:
		break;
	case TransferFailure:
		usbi_err(ctx, "ReadPipe/WritePipe failed: %s", windows_error_str(0));
		usbi_free_fd(&wfd);
		return LIBUSB_ERROR_IO;
	}

	transfer_priv->pollable_fd = wfd;
	usbi_add_pollfd(ctx, transfer_priv->pollable_fd.fd, IS_XFERIN(transfer) ? POLLIN : POLLOUT);

	return LIBUSB_SUCCESS;
}

static int usbdk_do_iso_transfer(struct usbi_transfer *itransfer)
{
	struct libusb_transfer *transfer = USBI_TRANSFER_TO_LIBUSB_TRANSFER(itransfer);
	struct usbdk_device_priv *priv = _usbdk_device_priv(transfer->dev_handle->dev);
	struct usbdk_transfer_priv *transfer_priv = _usbdk_transfer_priv(itransfer);
	struct libusb_context *ctx = DEVICE_CTX(transfer->dev_handle->dev);
	struct winfd wfd;
	TransferResult transferRes;
	int i;
	HANDLE sysHandle;

	transfer_priv->request.Buffer = (PVOID64)(uintptr_t)transfer->buffer;
	transfer_priv->request.BufferLength = transfer->length;
	transfer_priv->request.EndpointAddress = transfer->endpoint;
	transfer_priv->request.TransferType = IsochronousTransferType;
	transfer_priv->request.IsochronousPacketsArraySize = transfer->num_iso_packets;
	transfer_priv->IsochronousPacketsArray = malloc(transfer->num_iso_packets * sizeof(ULONG64));
	transfer_priv->request.IsochronousPacketsArray = (PVOID64)(uintptr_t)transfer_priv->IsochronousPacketsArray;
	if (!transfer_priv->IsochronousPacketsArray) {
		usbi_err(ctx, "Allocation of IsochronousPacketsArray is failed, %s", windows_error_str(0));
		return LIBUSB_ERROR_IO;
	}

	transfer_priv->IsochronousResultsArray = malloc(transfer->num_iso_packets * sizeof(USB_DK_ISO_TRANSFER_RESULT));
	transfer_priv->request.Result.IsochronousResultsArray = (PVOID64)(uintptr_t)transfer_priv->IsochronousResultsArray;
	if (!transfer_priv->IsochronousResultsArray) {
		usbi_err(ctx, "Allocation of isochronousResultsArray is failed, %s", windows_error_str(0));
		free(transfer_priv->IsochronousPacketsArray);
		return LIBUSB_ERROR_IO;
	}

	for (i = 0; i < transfer->num_iso_packets; i++)
		transfer_priv->IsochronousPacketsArray[i] = transfer->iso_packet_desc[i].length;

	transfer_priv->pollable_fd = INVALID_WINFD;

	sysHandle = usbdk_helper.GetRedirectorSystemHandle(priv->redirector_handle);

	wfd = usbi_create_fd(sysHandle, IS_XFERIN(transfer) ? RW_READ : RW_WRITE, NULL, NULL);
	// Always use the handle returned from usbi_create_fd (wfd.handle)
	if (wfd.fd < 0) {
		free(transfer_priv->IsochronousPacketsArray);
		free(transfer_priv->IsochronousResultsArray);
		return LIBUSB_ERROR_NO_MEM;
	}

	if (IS_XFERIN(transfer))
		transferRes = usbdk_helper.ReadPipe(priv->redirector_handle, &transfer_priv->request, wfd.overlapped);
	else
		transferRes = usbdk_helper.WritePipe(priv->redirector_handle, &transfer_priv->request, wfd.overlapped);

	switch (transferRes) {
	case TransferSuccess:
		wfd.overlapped->Internal = STATUS_COMPLETED_SYNCHRONOUSLY;
		break;
	case TransferSuccessAsync:
		break;
	case TransferFailure:
		usbi_err(ctx, "ReadPipe/WritePipe failed: %s", windows_error_str(0));
		usbi_free_fd(&wfd);
		free(transfer_priv->IsochronousPacketsArray);
		free(transfer_priv->IsochronousResultsArray);
		return LIBUSB_ERROR_IO;
	}

	transfer_priv->pollable_fd = wfd;
	usbi_add_pollfd(ctx, transfer_priv->pollable_fd.fd, IS_XFERIN(transfer) ? POLLIN : POLLOUT);

	return LIBUSB_SUCCESS;
}

static int usbdk_submit_transfer(struct usbi_transfer *itransfer)
{
	struct libusb_transfer *transfer = USBI_TRANSFER_TO_LIBUSB_TRANSFER(itransfer);

	switch (transfer->type) {
	case LIBUSB_TRANSFER_TYPE_CONTROL:
		return usbdk_do_control_transfer(itransfer);
	case LIBUSB_TRANSFER_TYPE_BULK:
	case LIBUSB_TRANSFER_TYPE_INTERRUPT:
		if (IS_XFEROUT(transfer) && (transfer->flags & LIBUSB_TRANSFER_ADD_ZERO_PACKET))
			return LIBUSB_ERROR_NOT_SUPPORTED; //TODO: Check whether we can support this in UsbDk
		else
			return usbdk_do_bulk_transfer(itransfer);
	case LIBUSB_TRANSFER_TYPE_ISOCHRONOUS:
		return usbdk_do_iso_transfer(itransfer);
	default:
		usbi_err(TRANSFER_CTX(transfer), "unknown endpoint type %d", transfer->type);
		return LIBUSB_ERROR_INVALID_PARAM;
	}
}

static int usbdk_abort_transfers(struct usbi_transfer *itransfer)
{
	struct libusb_transfer *transfer = USBI_TRANSFER_TO_LIBUSB_TRANSFER(itransfer);
	struct libusb_context *ctx = DEVICE_CTX(transfer->dev_handle->dev);
	struct usbdk_device_priv *priv = _usbdk_device_priv(transfer->dev_handle->dev);

	if (!usbdk_helper.AbortPipe(priv->redirector_handle, transfer->endpoint)) {
		usbi_err(ctx, "AbortPipe failed: %s", windows_error_str(0));
		return LIBUSB_ERROR_NO_DEVICE;
	}

	return LIBUSB_SUCCESS;
}

static int usbdk_cancel_transfer(struct usbi_transfer *itransfer)
{
	struct libusb_transfer *transfer = USBI_TRANSFER_TO_LIBUSB_TRANSFER(itransfer);

	switch (transfer->type) {
	case LIBUSB_TRANSFER_TYPE_CONTROL:
		// Control transfers cancelled by IoCancelXXX() API
		// No special treatment needed
		return LIBUSB_SUCCESS;
	case LIBUSB_TRANSFER_TYPE_BULK:
	case LIBUSB_TRANSFER_TYPE_INTERRUPT:
	case LIBUSB_TRANSFER_TYPE_ISOCHRONOUS:
		return usbdk_abort_transfers(itransfer);
	default:
		usbi_err(ITRANSFER_CTX(itransfer), "unknown endpoint type %d", transfer->type);
		return LIBUSB_ERROR_INVALID_PARAM;
	}
}

int windows_copy_transfer_data(struct usbi_transfer *itransfer, uint32_t io_size)
{
	itransfer->transferred += io_size;
	return LIBUSB_TRANSFER_COMPLETED;
}

struct winfd *windows_get_fd(struct usbi_transfer *transfer)
{
	struct usbdk_transfer_priv *transfer_priv = _usbdk_transfer_priv(transfer);
	return &transfer_priv->pollable_fd;
}

static DWORD usbdk_translate_usbd_status(USBD_STATUS UsbdStatus)
{
	if (USBD_SUCCESS(UsbdStatus))
		return NO_ERROR;

	switch (UsbdStatus) {
	case USBD_STATUS_TIMEOUT:
		return ERROR_SEM_TIMEOUT;
	case USBD_STATUS_CANCELED:
		return ERROR_OPERATION_ABORTED;
	default:
		return ERROR_GEN_FAILURE;
	}
}

void windows_get_overlapped_result(struct usbi_transfer *transfer, struct winfd *pollable_fd, DWORD *io_result, DWORD *io_size)
{
	if (HasOverlappedIoCompletedSync(pollable_fd->overlapped) // Handle async requests that completed synchronously first
			|| GetOverlappedResult(pollable_fd->handle, pollable_fd->overlapped, io_size, false)) { // Regular async overlapped
		struct libusb_transfer *ltransfer = USBI_TRANSFER_TO_LIBUSB_TRANSFER(transfer);
		struct usbdk_transfer_priv *transfer_priv = _usbdk_transfer_priv(transfer);

		if (ltransfer->type == LIBUSB_TRANSFER_TYPE_ISOCHRONOUS) {
			int i;
			for (i = 0; i < transfer_priv->request.IsochronousPacketsArraySize; i++) {
				struct libusb_iso_packet_descriptor *lib_desc = &ltransfer->iso_packet_desc[i];

				switch (transfer_priv->IsochronousResultsArray[i].TransferResult) {
				case STATUS_SUCCESS:
				case STATUS_CANCELLED:
				case STATUS_REQUEST_CANCELED:
					lib_desc->status = LIBUSB_TRANSFER_COMPLETED; // == ERROR_SUCCESS
					break;
				default:
					lib_desc->status = LIBUSB_TRANSFER_ERROR; // ERROR_UNKNOWN_EXCEPTION;
					break;
				}

				lib_desc->actual_length = (unsigned int)transfer_priv->IsochronousResultsArray[i].ActualLength;
			}
		}

		*io_size = (DWORD) transfer_priv->request.Result.GenResult.BytesTransferred;
		*io_result = usbdk_translate_usbd_status((USBD_STATUS) transfer_priv->request.Result.GenResult.UsbdStatus);
	}
	else {
		*io_result = GetLastError();
	}
}

static int usbdk_clock_gettime(int clk_id, struct timespec *tp)
{
	return windows_clock_gettime(clk_id, tp);
}

const struct usbi_os_backend usbi_backend = {
	"Windows",
	USBI_CAP_HAS_HID_ACCESS,
	usbdk_init,
	usbdk_exit,

	usbdk_get_device_list,
	NULL,
	usbdk_open,
	usbdk_close,

	usbdk_get_device_descriptor,
	usbdk_get_active_config_descriptor,
	usbdk_get_config_descriptor,
	NULL,

	usbdk_get_configuration,
	usbdk_set_configuration,
	usbdk_claim_interface,
	usbdk_release_interface,

	usbdk_set_interface_altsetting,
	usbdk_clear_halt,
	usbdk_reset_device,

	NULL,
	NULL,

	NULL,	// dev_mem_alloc()
	NULL,	// dev_mem_free()

	usbdk_kernel_driver_active,
	usbdk_detach_kernel_driver,
	usbdk_attach_kernel_driver,

	usbdk_destroy_device,

	usbdk_submit_transfer,
	usbdk_cancel_transfer,
	windows_clear_transfer_priv,

	windows_handle_events,
	NULL,

	usbdk_clock_gettime,
#if defined(USBI_TIMERFD_AVAILABLE)
	NULL,
#endif
	0,
	sizeof(struct usbdk_device_priv),
	0,
	sizeof(struct usbdk_transfer_priv),
};

#endif /* USE_USBDK */
