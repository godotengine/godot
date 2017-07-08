/*
 * OpenHMD - Free and Open Source API and drivers for immersive technology.
 * Copyright (C) 2013 Fredrik Hultin.
 * Copyright (C) 2013 Jakob Bornecrantz.
 * Copyright (C) 2017 Joey Ferwerda.
 * Distributed under the Boost 1.0 licence, see LICENSE for full text.
 *
 * Original implementation by: Yann Vernier.
 */

/* NOLO VR- HID/USB Driver Implementation */

#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <assert.h>

#include "nolo.h"

#define NOLO_ID					0x0483 //ST microcontroller
#define NOLO_HMD				0x5750

static const int controllerLength = 3 + (3+4)*2 + 2 + 2 + 1;
static devices_t* nolo_devices;

static drv_priv* drv_priv_get(ohmd_device* device)
{
	return (drv_priv*)device;
}

static int send_feature_report(drv_priv* priv, const unsigned char *data, size_t length)
{
	return hid_send_feature_report(priv->handle, data, length);
}

static void update_device(ohmd_device* device)
{
	drv_priv* priv = drv_priv_get(device);
	unsigned char buffer[FEATURE_BUFFER_SIZE];

	// Only update when physical device
	if (priv->id != 0)
		return;

	devices_t* current = nolo_devices;
	drv_priv* controller0 = NULL;
	drv_priv* controller1 = NULL;

	//Check if controllers exist
	while (current != NULL) {
		if (current->drv->hmd_tracker == priv)
		{
			if (current->drv->controller0)
				controller0 = current->drv->controller0;
			if (current->drv->controller1)
				controller1 = current->drv->controller1;
			break;
		}
		current = current->next;
	}

	// Read all the messages from the device.
	while(true){
		int size = hid_read(priv->handle, buffer, FEATURE_BUFFER_SIZE);
		if(size < 0){
			LOGE("error reading from device");
			return;
		} else if(size == 0) {
			return; // No more messages, return.
		}

		nolo_decrypt_data(buffer);

		// currently the only message type the hardware supports
		switch (buffer[0]) {
			case 0xa5:  // Controllers packet
			{
				if (controller0)
					nolo_decode_controller(controller0, buffer+1);
				if (controller1)
					nolo_decode_controller(controller1, buffer+64-controllerLength);
			break;
			}
			case 0xa6: // HMD packet
				nolo_decode_hmd_marker(priv, buffer+0x15);
				nolo_decode_base_station(priv, buffer+0x36);
			break;
			default:
				LOGE("unknown message type: %u", buffer[0]);
		}
	}


	return;
}

static int getf(ohmd_device* device, ohmd_float_value type, float* out)
{
	drv_priv* priv = drv_priv_get(device);

	switch(type){

	case OHMD_ROTATION_QUAT: {
			*(quatf*)out = priv->base.rotation;
			break;
		}

	case OHMD_POSITION_VECTOR:
		if(priv->id == 0) {
			// HMD
			*(vec3f*)out = priv->base.position;
		}
		else if(priv->id == 1) {
			// Controller 0
			*(vec3f*)out = priv->base.position;
		}
		else if(priv->id == 2) {
			// Controller 1
			*(vec3f*)out = priv->base.position;
		}
		break;

	case OHMD_CONTROLS_STATE:
		if(priv->id > 0) {
			for (int i = 0; i < 8; i++){
				out[i] = priv->controller_values[i];
			}
		}
		break;

	default:
		ohmd_set_error(priv->base.ctx, "invalid type given to getf (%ud)", type);
		return -1;
		break;
	}

	return 0;
}

static void close_device(ohmd_device* device)
{
	LOGD("closing device");
	drv_priv* priv = drv_priv_get(device);
	hid_close(priv->handle);
	free(priv);
}

static char* _hid_to_unix_path(char* path)
{
	const int len = 16;
	char bus [16];
	char dev [16];
	char *result = malloc( sizeof(char) * ( 20 + 1 ) );

	sprintf (bus, "%.*s\n", len, path);
	sprintf (dev, "%.*s\n", len, path + 5);

	sprintf (result, "/dev/bus/usb/%03d/%03d",
		(int)strtol(bus, NULL, 16),
		(int)strtol(dev, NULL, 16));
	return result;
}


void push_device(devices_t * head, drv_nolo* val) {
	devices_t* current = head;

	if (!nolo_devices)
	{
		nolo_devices = calloc(1, sizeof(devices_t));
		nolo_devices->drv = val;
		nolo_devices->next = NULL;
		return;
	}

	while (current->next != NULL) {
		current = current->next;
	}

	/* now we can add a new variable */
	current->next = calloc(1, sizeof(devices_t));
	current->next->drv = val;
	current->next->next = NULL;
}

static ohmd_device* open_device(ohmd_driver* driver, ohmd_device_desc* desc)
{
	drv_priv* priv = ohmd_alloc(driver->ctx, sizeof(drv_priv));
	if(!priv)
		goto cleanup;

	priv->id = desc->id;
	priv->base.ctx = driver->ctx;

	// Open the HID device when physical device
	if (priv->id == 0)
	{
		priv->handle = hid_open_path(desc->path);

		if(!priv->handle) {
			char* path = _hid_to_unix_path(desc->path);
			ohmd_set_error(driver->ctx, "Could not open %s. "
										"Check your rights.", path);
			free(path);
			goto cleanup;
		}

		if(hid_set_nonblocking(priv->handle, 1) == -1){
			ohmd_set_error(driver->ctx, "failed to set non-blocking on device");
			goto cleanup;
		}

	}

	devices_t* current = nolo_devices;
	drv_nolo* mNOLO = NULL;

	//Check if the opened device is part of a group
	while (current != NULL) {
		if (strcmp(current->drv->path, desc->path)==0)
			mNOLO = current->drv;
		current = current->next;
	}

	if (!mNOLO)
	{
		//Create new group
		mNOLO = calloc(1, sizeof(drv_nolo));
		mNOLO->hmd_tracker = NULL;
		mNOLO->controller0 = NULL;
		mNOLO->controller1 = NULL;
		strcpy(mNOLO->path, desc->path);
		push_device(nolo_devices, mNOLO);
	}

	if (priv->id == 0) {
		mNOLO->hmd_tracker = priv;
	}
	else if (priv->id == 1) {
		mNOLO->controller0 = priv;
		priv->base.properties.control_count = 8;
		priv->base.properties.controls_hints[0] = OHMD_ANALOG_PRESS;
		priv->base.properties.controls_hints[1] = OHMD_TRIGGER_CLICK;
		priv->base.properties.controls_hints[2] = OHMD_MENU;
		priv->base.properties.controls_hints[3] = OHMD_HOME;
		priv->base.properties.controls_hints[4] = OHMD_SQUEEZE;
		priv->base.properties.controls_hints[5] = OHMD_GENERIC; //touching the XY pad
		priv->base.properties.controls_hints[6] = OHMD_ANALOG_X;
		priv->base.properties.controls_hints[7] = OHMD_ANALOG_Y;
		priv->base.properties.controls_types[0] = OHMD_DIGITAL;
		priv->base.properties.controls_types[1] = OHMD_DIGITAL;
		priv->base.properties.controls_types[2] = OHMD_DIGITAL;
		priv->base.properties.controls_types[3] = OHMD_DIGITAL;
		priv->base.properties.controls_types[4] = OHMD_DIGITAL;
		priv->base.properties.controls_types[5] = OHMD_DIGITAL;
		priv->base.properties.controls_types[6] = OHMD_ANALOG;
		priv->base.properties.controls_types[7] = OHMD_ANALOG;
	}
	else if (priv->id == 2) {
		mNOLO->controller1 = priv;
		priv->base.properties.control_count = 8;
		priv->base.properties.controls_hints[0] = OHMD_ANALOG_PRESS;
		priv->base.properties.controls_hints[1] = OHMD_TRIGGER_CLICK;
		priv->base.properties.controls_hints[2] = OHMD_MENU;
		priv->base.properties.controls_hints[3] = OHMD_HOME;
		priv->base.properties.controls_hints[4] = OHMD_SQUEEZE;
		priv->base.properties.controls_hints[5] = OHMD_GENERIC; //touching the XY pad
		priv->base.properties.controls_hints[6] = OHMD_ANALOG_X;
		priv->base.properties.controls_hints[7] = OHMD_ANALOG_Y;
		priv->base.properties.controls_types[0] = OHMD_DIGITAL;
		priv->base.properties.controls_types[1] = OHMD_DIGITAL;
		priv->base.properties.controls_types[2] = OHMD_DIGITAL;
		priv->base.properties.controls_types[3] = OHMD_DIGITAL;
		priv->base.properties.controls_types[4] = OHMD_DIGITAL;
		priv->base.properties.controls_types[5] = OHMD_DIGITAL;
		priv->base.properties.controls_types[6] = OHMD_ANALOG;
		priv->base.properties.controls_types[7] = OHMD_ANALOG;
	}

	// Set default device properties
	ohmd_set_default_device_properties(&priv->base.properties);

	// set up device callbacks
	priv->base.update = update_device;
	priv->base.close = close_device;
	priv->base.getf = getf;

	return &priv->base;

cleanup:
	if(priv)
		free(priv);

	return NULL;
}

static void get_device_list(ohmd_driver* driver, ohmd_device_list* list)
{
	struct hid_device_info* devs = hid_enumerate(NOLO_ID, NOLO_HMD);
	struct hid_device_info* cur_dev = devs;

	int id = 0;
	while (cur_dev) {
		if (wcscmp(cur_dev->manufacturer_string, L"LYRobotix")==0 &&
			wcscmp(cur_dev->product_string, L"NOLO")==0) {
			ohmd_device_desc* desc = &list->devices[list->num_devices++];

			strcpy(desc->driver, "OpenHMD NOLO VR CV1 driver");
			strcpy(desc->vendor, "LYRobotix");
			strcpy(desc->product, "NOLO CV1");

			desc->revision = 0;

			strcpy(desc->path, cur_dev->path);

			desc->device_flags = OHMD_DEVICE_FLAGS_POSITIONAL_TRACKING | OHMD_DEVICE_FLAGS_ROTATIONAL_TRACKING;
			desc->device_class = OHMD_DEVICE_CLASS_HMD;

			desc->driver_ptr = driver;
			desc->id = id++;

			//Controller 0
			desc = &list->devices[list->num_devices++];

			strcpy(desc->driver, "OpenHMD NOLO VR CV1 driver");
			strcpy(desc->vendor, "LYRobotix");
			strcpy(desc->product, "NOLO CV1: Controller 0");

			strcpy(desc->path, cur_dev->path);

			desc->device_flags =
				OHMD_DEVICE_FLAGS_POSITIONAL_TRACKING |
				OHMD_DEVICE_FLAGS_ROTATIONAL_TRACKING |
				OHMD_DEVICE_FLAGS_RIGHT_CONTROLLER;

			desc->driver_ptr = driver;
			desc->id = id++;

			// Controller 1
			desc = &list->devices[list->num_devices++];

			strcpy(desc->driver, "OpenHMD NOLO VR CV1 driver");
			strcpy(desc->vendor, "LYRobotix");
			strcpy(desc->product, "NOLO CV1: Controller 1");

			strcpy(desc->path, cur_dev->path);

			desc->device_flags =
				OHMD_DEVICE_FLAGS_POSITIONAL_TRACKING |
				OHMD_DEVICE_FLAGS_ROTATIONAL_TRACKING |
				OHMD_DEVICE_FLAGS_LEFT_CONTROLLER;

			desc->driver_ptr = driver;
			desc->id = id++;
		}
		cur_dev = cur_dev->next;
	}
	hid_free_enumeration(devs);
}

static void destroy_driver(ohmd_driver* drv)
{
	LOGD("shutting down NOLO CV1 driver");
	hid_exit();
	free(drv);
}

ohmd_driver* ohmd_create_nolo_drv(ohmd_context* ctx)
{
	ohmd_driver* drv = ohmd_alloc(ctx, sizeof(ohmd_driver));
	if(drv == NULL)
		return NULL;

	drv->get_device_list = get_device_list;
	drv->open_device = open_device;
	drv->destroy = destroy_driver;
	drv->ctx = ctx;

	return drv;
}
