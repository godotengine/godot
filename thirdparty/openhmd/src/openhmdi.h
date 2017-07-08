/*
 * OpenHMD - Free and Open Source API and drivers for immersive technology.
 * Copyright (C) 2013 Fredrik Hultin.
 * Copyright (C) 2013 Jakob Bornecrantz.
 * Distributed under the Boost 1.0 licence, see LICENSE for full text.
 */

/* Internal interface */

#ifndef OPENHMDI_H
#define OPENHMDI_H

#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#include "openhmd.h"
#include "omath.h"
#include "platform.h"

#define OHMD_MAX_DEVICES 16

#define OHMD_MAX(_a, _b) ((_a) > (_b) ? (_a) : (_b))
#define OHMD_MIN(_a, _b) ((_a) < (_b) ? (_a) : (_b))

#define OHMD_STRINGIFY(_what) #_what

typedef struct ohmd_driver ohmd_driver;

typedef struct {
	char driver[OHMD_STR_SIZE];
	char vendor[OHMD_STR_SIZE];
	char product[OHMD_STR_SIZE];
	char path[OHMD_STR_SIZE];
	int revision;
	int id;
	ohmd_device_flags device_flags;
	ohmd_device_class device_class;
	ohmd_driver* driver_ptr;
} ohmd_device_desc;

typedef struct {
	int num_devices;
	ohmd_device_desc devices[OHMD_MAX_DEVICES];
} ohmd_device_list;

struct ohmd_driver {
	void (*get_device_list)(ohmd_driver* driver, ohmd_device_list* list);
	ohmd_device* (*open_device)(ohmd_driver* driver, ohmd_device_desc* desc);
	void (*destroy)(ohmd_driver* driver);
	ohmd_context* ctx;
};

typedef struct {
		int hres;
		int vres;
		int control_count;
		int controls_hints[64];
		int controls_types[64];

		float hsize;
		float vsize;

		float lens_sep;
		float lens_vpos;

		float fov;
		float ratio;

		float ipd;
		float zfar;
		float znear;

		int accel_only; //bool-like for setting acceleration only fallback (android driver)

		mat4x4f proj_left; // adjusted projection matrix for left screen
		mat4x4f proj_right; // adjusted projection matrix for right screen
		float universal_distortion_k[4]; //PanoTools lens distiorion model [a,b,c,d]
		float universal_aberration_k[3]; //post-warp per channel scaling [r,g,b]
} ohmd_device_properties;

struct ohmd_device_settings
{
	bool automatic_update;
};

struct ohmd_device {
	ohmd_device_properties properties;

	quatf rotation_correction;
	vec3f position_correction;

	int (*getf)(ohmd_device* device, ohmd_float_value type, float* out);
	int (*setf)(ohmd_device* device, ohmd_float_value type, const float* in);
	int (*seti)(ohmd_device* device, ohmd_int_value type, const int* in);
	int (*set_data)(ohmd_device* device, ohmd_data_value type, const void* in);

	void (*update)(ohmd_device* device);
	void (*close)(ohmd_device* device);

	ohmd_context* ctx;

	ohmd_device_settings settings;

	int active_device_idx; // index into ohmd_device->active_devices[]

	quatf rotation;
	vec3f position;
};


struct ohmd_context {
	ohmd_driver* drivers[16];
	int num_drivers;

	ohmd_device_list list;

	ohmd_device* active_devices[256];
	int num_active_devices;

	ohmd_thread* update_thread;
	ohmd_mutex* update_mutex;

	bool update_request_quit;

	uint64_t monotonic_ticks_per_sec;

	char error_msg[OHMD_STR_SIZE];
};

// helper functions
void ohmd_monotonic_init(ohmd_context* ctx);
uint64_t ohmd_monotonic_get(ohmd_context* ctx);
uint64_t ohmd_monotonic_per_sec(ohmd_context* ctx);
uint64_t ohmd_monotonic_conv(uint64_t ticks, uint64_t srcTicksPerSecond, uint64_t dstTicksPerSecond);
void ohmd_set_default_device_properties(ohmd_device_properties* props);
void ohmd_calc_default_proj_matrices(ohmd_device_properties* props);
void ohmd_set_universal_distortion_k(ohmd_device_properties* props, float a, float b, float c, float d);
void ohmd_set_universal_aberration_k(ohmd_device_properties* props, float r, float g, float b);

// drivers
ohmd_driver* ohmd_create_dummy_drv(ohmd_context* ctx);
ohmd_driver* ohmd_create_oculus_rift_drv(ohmd_context* ctx);
ohmd_driver* ohmd_create_deepoon_drv(ohmd_context* ctx);
ohmd_driver* ohmd_create_htc_vive_drv(ohmd_context* ctx);
ohmd_driver* ohmd_create_psvr_drv(ohmd_context* ctx);
ohmd_driver* ohmd_create_nolo_drv(ohmd_context* ctx);
ohmd_driver* ohmd_create_external_drv(ohmd_context* ctx);
ohmd_driver* ohmd_create_android_drv(ohmd_context* ctx);

#include "log.h"
#include "omath.h"
#include "fusion.h"

#endif
