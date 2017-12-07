/*
 * OpenHMD - Free and Open Source API and drivers for immersive technology.
 * Copyright (C) 2017 Joey Ferwerda.
 * Distributed under the Boost 1.0 licence, see LICENSE for full text.
 */

/* NOLO VR - Internal Interface */

#ifndef NOLODRIVER_H
#define NOLODRIVER_H

#include "../openhmdi.h"
#include <hidapi.h>

#define FEATURE_BUFFER_SIZE 64

typedef struct {
	ohmd_device base;

	hid_device* handle;
	int id;
	float controller_values[8];
} drv_priv;

typedef struct{
	char path[OHMD_STR_SIZE];
	drv_priv* hmd_tracker;
	drv_priv* controller0;
	drv_priv* controller1;
} drv_nolo;

typedef struct devices{
	drv_nolo* drv;
	struct devices * next;
} devices_t;

void btea_decrypt(uint32_t *v, int n, int base_rounds, uint32_t const key[4]);
void nolo_decrypt_data(unsigned char* buf);

void nolo_decode_base_station(drv_priv* priv, unsigned char* data);
void nolo_decode_hmd_marker(drv_priv* priv, unsigned char* data);
void nolo_decode_controller(drv_priv* priv, unsigned char* data);
void nolo_decode_orientation(const unsigned char* data, quatf* quat);
void nolo_decode_position(const unsigned char* data, vec3f* pos);

#endif

