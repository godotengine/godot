/*
 * Copyright 2006-2008, Haiku Inc. All rights reserved.
 * Distributed under the terms of the MIT License.
 */

#ifndef _USB_RAW_H_
#define _USB_RAW_H_

#include <USB3.h>

#define B_USB_RAW_PROTOCOL_VERSION	0x0015
#define B_USB_RAW_ACTIVE_ALTERNATE	0xffffffff

typedef enum {
	B_USB_RAW_COMMAND_GET_VERSION = 0x1000,

	B_USB_RAW_COMMAND_GET_DEVICE_DESCRIPTOR = 0x2000,
	B_USB_RAW_COMMAND_GET_CONFIGURATION_DESCRIPTOR,
	B_USB_RAW_COMMAND_GET_INTERFACE_DESCRIPTOR,
	B_USB_RAW_COMMAND_GET_ENDPOINT_DESCRIPTOR,
	B_USB_RAW_COMMAND_GET_STRING_DESCRIPTOR,
	B_USB_RAW_COMMAND_GET_GENERIC_DESCRIPTOR,
	B_USB_RAW_COMMAND_GET_ALT_INTERFACE_COUNT,
	B_USB_RAW_COMMAND_GET_ACTIVE_ALT_INTERFACE_INDEX,
	B_USB_RAW_COMMAND_GET_INTERFACE_DESCRIPTOR_ETC,
	B_USB_RAW_COMMAND_GET_ENDPOINT_DESCRIPTOR_ETC,
	B_USB_RAW_COMMAND_GET_GENERIC_DESCRIPTOR_ETC,

	B_USB_RAW_COMMAND_SET_CONFIGURATION = 0x3000,
	B_USB_RAW_COMMAND_SET_FEATURE,
	B_USB_RAW_COMMAND_CLEAR_FEATURE,
	B_USB_RAW_COMMAND_GET_STATUS,
	B_USB_RAW_COMMAND_GET_DESCRIPTOR,
	B_USB_RAW_COMMAND_SET_ALT_INTERFACE,

	B_USB_RAW_COMMAND_CONTROL_TRANSFER = 0x4000,
	B_USB_RAW_COMMAND_INTERRUPT_TRANSFER,
	B_USB_RAW_COMMAND_BULK_TRANSFER,
	B_USB_RAW_COMMAND_ISOCHRONOUS_TRANSFER
} usb_raw_command_id;


typedef enum {
	B_USB_RAW_STATUS_SUCCESS = 0,

	B_USB_RAW_STATUS_FAILED,
	B_USB_RAW_STATUS_ABORTED,
	B_USB_RAW_STATUS_STALLED,
	B_USB_RAW_STATUS_CRC_ERROR,
	B_USB_RAW_STATUS_TIMEOUT,

	B_USB_RAW_STATUS_INVALID_CONFIGURATION,
	B_USB_RAW_STATUS_INVALID_INTERFACE,
	B_USB_RAW_STATUS_INVALID_ENDPOINT,
	B_USB_RAW_STATUS_INVALID_STRING,

	B_USB_RAW_STATUS_NO_MEMORY
} usb_raw_command_status;


typedef union {
	struct {
		status_t status;
	} version;

	struct {
		status_t status;
		usb_device_descriptor *descriptor;
	} device;

	struct {
		status_t status;
		usb_configuration_descriptor *descriptor;
		uint32 config_index;
	} config;

	struct {
		status_t status;
		uint32 alternate_info;
		uint32 config_index;
		uint32 interface_index;
	} alternate;

	struct {
		status_t status;
		usb_interface_descriptor *descriptor;
		uint32 config_index;
		uint32 interface_index;
	} interface;

	struct {
		status_t status;
		usb_interface_descriptor *descriptor;
		uint32 config_index;
		uint32 interface_index;
		uint32 alternate_index;
	} interface_etc;

	struct {
		status_t status;
		usb_endpoint_descriptor *descriptor;
		uint32 config_index;
		uint32 interface_index;
		uint32 endpoint_index;
	} endpoint;

	struct {
		status_t status;
		usb_endpoint_descriptor *descriptor;
		uint32 config_index;
		uint32 interface_index;
		uint32 alternate_index;
		uint32 endpoint_index;
	} endpoint_etc;

	struct {
		status_t status;
		usb_descriptor *descriptor;
		uint32 config_index;
		uint32 interface_index;
		uint32 generic_index;
		size_t length;
	} generic;

	struct {
		status_t status;
		usb_descriptor *descriptor;
		uint32 config_index;
		uint32 interface_index;
		uint32 alternate_index;
		uint32 generic_index;
		size_t length;
	} generic_etc;

	struct {
		status_t status;
		usb_string_descriptor *descriptor;
		uint32 string_index;
		size_t length;
	} string;

	struct {
		status_t status;
		uint8 type;
		uint8 index;
		uint16 language_id;
		void *data;
		size_t length;
	} descriptor;

	struct {
		status_t status;
		uint8 request_type;
		uint8 request;
		uint16 value;
		uint16 index;
		uint16 length;
		void *data;
	} control;

	struct {
		status_t status;
		uint32 interface;
		uint32 endpoint;
		void *data;
		size_t length;
	} transfer;

	struct {
		status_t status;
		uint32 interface;
		uint32 endpoint;
		void *data;
		size_t length;
		usb_iso_packet_descriptor *packet_descriptors;
		uint32 packet_count;
	} isochronous;
} usb_raw_command;

#endif // _USB_RAW_H_
