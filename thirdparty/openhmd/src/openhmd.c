/*
 * OpenHMD - Free and Open Source API and drivers for immersive technology.
 * Copyright (C) 2013 Fredrik Hultin.
 * Copyright (C) 2013 Jakob Bornecrantz.
 * Distributed under the Boost 1.0 licence, see LICENSE for full text.
 */

/* Main Lib Implementation */

#include "openhmdi.h"
#include "shaders.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

// Running automatic updates at 1000 Hz
#define AUTOMATIC_UPDATE_SLEEP (1.0 / 1000.0)

ohmd_context* OHMD_APIENTRY ohmd_ctx_create(void)
{
	ohmd_context* ctx = calloc(1, sizeof(ohmd_context));
	if(!ctx){
		LOGE("could not allocate RAM for context");
		return NULL;
	}

	ohmd_monotonic_init(ctx);

#if DRIVER_OCULUS_RIFT
	ctx->drivers[ctx->num_drivers++] = ohmd_create_oculus_rift_drv(ctx);
#endif

#if DRIVER_DEEPOON
	ctx->drivers[ctx->num_drivers++] = ohmd_create_deepoon_drv(ctx);
#endif

#if DRIVER_HTC_VIVE
	ctx->drivers[ctx->num_drivers++] = ohmd_create_htc_vive_drv(ctx);
#endif

#if DRIVER_PSVR
	ctx->drivers[ctx->num_drivers++] = ohmd_create_psvr_drv(ctx);
#endif

#if DRIVER_NOLO
	ctx->drivers[ctx->num_drivers++] = ohmd_create_nolo_drv(ctx);
#endif

#if DRIVER_EXTERNAL
	ctx->drivers[ctx->num_drivers++] = ohmd_create_external_drv(ctx);
#endif

#if DRIVER_ANDROID
	ctx->drivers[ctx->num_drivers++] = ohmd_create_android_drv(ctx);
#endif
	// add dummy driver last to make it the lowest priority
	ctx->drivers[ctx->num_drivers++] = ohmd_create_dummy_drv(ctx);

	ctx->update_request_quit = false;

	return ctx;
}

void OHMD_APIENTRY ohmd_ctx_destroy(ohmd_context* ctx)
{
	ctx->update_request_quit = true;

	for(int i = 0; i < ctx->num_active_devices; i++){
		ctx->active_devices[i]->close(ctx->active_devices[i]);
	}

	for(int i = 0; i < ctx->num_drivers; i++){
		ctx->drivers[i]->destroy(ctx->drivers[i]);
	}

	if(ctx->update_thread){
		ohmd_destroy_thread(ctx->update_thread);
		ohmd_destroy_mutex(ctx->update_mutex);
	}

	free(ctx);
}

void OHMD_APIENTRY ohmd_ctx_update(ohmd_context* ctx)
{
	for(int i = 0; i < ctx->num_active_devices; i++){
		ohmd_device* dev = ctx->active_devices[i];
		if(!dev->settings.automatic_update && dev->update)
			dev->update(dev);

		ohmd_lock_mutex(ctx->update_mutex);
		dev->getf(dev, OHMD_POSITION_VECTOR, (float*)&dev->position);
		dev->getf(dev, OHMD_ROTATION_QUAT, (float*)&dev->rotation);
		ohmd_unlock_mutex(ctx->update_mutex);
	}
}

const char* OHMD_APIENTRY ohmd_ctx_get_error(ohmd_context* ctx)
{
	return ctx->error_msg;
}

int OHMD_APIENTRY ohmd_ctx_probe(ohmd_context* ctx)
{
	memset(&ctx->list, 0, sizeof(ohmd_device_list));
	for(int i = 0; i < ctx->num_drivers; i++){
		ctx->drivers[i]->get_device_list(ctx->drivers[i], &ctx->list);
	}

	return ctx->list.num_devices;
}

int OHMD_APIENTRY ohmd_gets(ohmd_string_description type, const char ** out)
{
	switch(type){
	case OHMD_GLSL_DISTORTION_VERT_SRC:
		*out = distortion_vert;
		return OHMD_S_OK;
	case OHMD_GLSL_DISTORTION_FRAG_SRC:
		*out = distortion_frag;
		return OHMD_S_OK;
	default:
		return OHMD_S_UNSUPPORTED;
	}
}

const char* OHMD_APIENTRY ohmd_list_gets(ohmd_context* ctx, int index, ohmd_string_value type)
{
	if(index >= ctx->list.num_devices)
		return NULL;

	switch(type){
	case OHMD_VENDOR:
		return ctx->list.devices[index].vendor;
	case OHMD_PRODUCT:
		return ctx->list.devices[index].product;
	case OHMD_PATH:
		return ctx->list.devices[index].path;
	default:
		return NULL;
	}
}

int OHMD_APIENTRY ohmd_list_geti(ohmd_context* ctx, int index, ohmd_int_value type, int* out)
{
	if(index >= ctx->list.num_devices)
		return OHMD_S_INVALID_PARAMETER;

	switch(type){
	case OHMD_DEVICE_CLASS:
		*out = ctx->list.devices[index].device_class;
		return OHMD_S_OK;

	case OHMD_DEVICE_FLAGS:
		*out = ctx->list.devices[index].device_flags;
		return OHMD_S_OK;

	default:
		return OHMD_S_INVALID_PARAMETER;
	}
}

static unsigned int ohmd_update_thread(void* arg)
{
	ohmd_context* ctx = (ohmd_context*)arg;

	while(!ctx->update_request_quit)
	{
		ohmd_lock_mutex(ctx->update_mutex);

		for(int i = 0; i < ctx->num_active_devices; i++){
			if(ctx->active_devices[i]->settings.automatic_update && ctx->active_devices[i]->update)
				ctx->active_devices[i]->update(ctx->active_devices[i]);
		}

		ohmd_unlock_mutex(ctx->update_mutex);

		ohmd_sleep(AUTOMATIC_UPDATE_SLEEP);
	}

	return 0;
}

static void ohmd_set_up_update_thread(ohmd_context* ctx)
{
	if(!ctx->update_thread){
		ctx->update_mutex = ohmd_create_mutex(ctx);
		ctx->update_thread = ohmd_create_thread(ctx, ohmd_update_thread, ctx);
	}
}

ohmd_device* OHMD_APIENTRY ohmd_list_open_device_s(ohmd_context* ctx, int index, ohmd_device_settings* settings)
{
	ohmd_lock_mutex(ctx->update_mutex);

	if(index >= 0 && index < ctx->list.num_devices){

		ohmd_device_desc* desc = &ctx->list.devices[index];
		ohmd_driver* driver = (ohmd_driver*)desc->driver_ptr;
		ohmd_device* device = driver->open_device(driver, desc);

		if (device == NULL) {
			ohmd_unlock_mutex(ctx->update_mutex);
			return NULL;
		}

		device->rotation_correction.w = 1;

		device->settings = *settings;

		device->ctx = ctx;
		device->active_device_idx = ctx->num_active_devices;
		ctx->active_devices[ctx->num_active_devices++] = device;

		ohmd_unlock_mutex(ctx->update_mutex);

		if(device->settings.automatic_update)
			ohmd_set_up_update_thread(ctx);

		return device;
	}

	ohmd_unlock_mutex(ctx->update_mutex);

	ohmd_set_error(ctx, "no device with index: %d", index);
	return NULL;
}

ohmd_device* OHMD_APIENTRY ohmd_list_open_device(ohmd_context* ctx, int index)
{
	ohmd_device_settings settings;

	settings.automatic_update = true;

	return ohmd_list_open_device_s(ctx, index, &settings);
}

int OHMD_APIENTRY ohmd_close_device(ohmd_device* device)
{
	ohmd_lock_mutex(device->ctx->update_mutex);

	ohmd_context* ctx = device->ctx;
	int idx = device->active_device_idx;

	memmove(ctx->active_devices + idx, ctx->active_devices + idx + 1,
		sizeof(ohmd_device*) * (ctx->num_active_devices - idx - 1));

	device->close(device);

	ctx->num_active_devices--;

	for(int i = idx; i < ctx->num_active_devices; i++)
		ctx->active_devices[i]->active_device_idx--;

	ohmd_unlock_mutex(ctx->update_mutex);

	return OHMD_S_OK;
}

static int ohmd_device_getf_unp(ohmd_device* device, ohmd_float_value type, float* out)
{
	switch(type){
	case OHMD_LEFT_EYE_GL_MODELVIEW_MATRIX: {
			vec3f point = {{0, 0, 0}};
			quatf rot = device->rotation;
			quatf tmp = device->rotation_correction;
			oquatf_mult_me(&tmp, &rot);
			rot = tmp;
			mat4x4f orient, world_shift, result;
			omat4x4f_init_look_at(&orient, &rot, &point);
			omat4x4f_init_translate(&world_shift, -device->position.x +(device->properties.ipd / 2.0f), -device->position.y, -device->position.z);
			omat4x4f_mult(&world_shift, &orient, &result);
			omat4x4f_transpose(&result, (mat4x4f*)out);
			return OHMD_S_OK;
		}
	case OHMD_RIGHT_EYE_GL_MODELVIEW_MATRIX: {
			vec3f point = {{0, 0, 0}};
			quatf rot = device->rotation;
			oquatf_mult_me(&rot, &device->rotation_correction);
			mat4x4f orient, world_shift, result;
			omat4x4f_init_look_at(&orient, &rot, &point);
			omat4x4f_init_translate(&world_shift, -device->position.x + -(device->properties.ipd / 2.0f), -device->position.y, -device->position.z);
			omat4x4f_mult(&world_shift, &orient, &result);
			omat4x4f_transpose(&result, (mat4x4f*)out);
			return OHMD_S_OK;
		}
	case OHMD_LEFT_EYE_GL_PROJECTION_MATRIX:
		omat4x4f_transpose(&device->properties.proj_left, (mat4x4f*)out);
		return OHMD_S_OK;
	case OHMD_RIGHT_EYE_GL_PROJECTION_MATRIX:
		omat4x4f_transpose(&device->properties.proj_right, (mat4x4f*)out);
		return OHMD_S_OK;

	case OHMD_SCREEN_HORIZONTAL_SIZE:
		*out = device->properties.hsize;
		return OHMD_S_OK;
	case OHMD_SCREEN_VERTICAL_SIZE:
		*out = device->properties.vsize;
		return OHMD_S_OK;

	case OHMD_LENS_HORIZONTAL_SEPARATION:
		*out = device->properties.lens_sep;
		return OHMD_S_OK;
	case OHMD_LENS_VERTICAL_POSITION:
		*out = device->properties.lens_vpos;
		return OHMD_S_OK;

	case OHMD_RIGHT_EYE_FOV:
	case OHMD_LEFT_EYE_FOV:
		*out = device->properties.fov;
		return OHMD_S_OK;
	case OHMD_RIGHT_EYE_ASPECT_RATIO:
	case OHMD_LEFT_EYE_ASPECT_RATIO:
		*out = device->properties.ratio;
		return OHMD_S_OK;

	case OHMD_EYE_IPD:
		*out = device->properties.ipd;
		return OHMD_S_OK;

	case OHMD_PROJECTION_ZFAR:
		*out = device->properties.zfar;
		return OHMD_S_OK;
	case OHMD_PROJECTION_ZNEAR:
		*out = device->properties.znear;
		return OHMD_S_OK;

	case OHMD_ROTATION_QUAT:
	{
		*(quatf*)out = device->rotation;

		oquatf_mult_me((quatf*)out, &device->rotation_correction);
		quatf tmp = device->rotation_correction;
		oquatf_mult_me(&tmp, (quatf*)out);
		*(quatf*)out = tmp;
		return OHMD_S_OK;
	}
	case OHMD_POSITION_VECTOR:
	{
		*(vec3f*)out = device->position;
		for(int i = 0; i < 3; i++)
			out[i] += device->position_correction.arr[i];
		
		return OHMD_S_OK;
	}
	case OHMD_UNIVERSAL_DISTORTION_K: {
		for (int i = 0; i < 4; i++) {
			out[i] = device->properties.universal_distortion_k[i];
		}
		return OHMD_S_OK;
	}
	case OHMD_UNIVERSAL_ABERRATION_K: {
		for (int i = 0; i < 3; i++) {
			out[i] = device->properties.universal_aberration_k[i];
		}
		return OHMD_S_OK;
	}
	default:
		return device->getf(device, type, out);
	}
}

int OHMD_APIENTRY ohmd_device_getf(ohmd_device* device, ohmd_float_value type, float* out)
{
	ohmd_lock_mutex(device->ctx->update_mutex);
	int ret = ohmd_device_getf_unp(device, type, out);
	ohmd_unlock_mutex(device->ctx->update_mutex);

	return ret;
}

int ohmd_device_setf_unp(ohmd_device* device, ohmd_float_value type, const float* in)
{
	switch(type){
	case OHMD_EYE_IPD:
		device->properties.ipd = *in;
		return OHMD_S_OK;
	case OHMD_PROJECTION_ZFAR:
		device->properties.zfar = *in;
		return OHMD_S_OK;
	case OHMD_PROJECTION_ZNEAR:
		device->properties.znear = *in;
		return OHMD_S_OK;
	case OHMD_ROTATION_QUAT:
		{
			// adjust rotation correction
			quatf q;
			int ret = device->getf(device, OHMD_ROTATION_QUAT, (float*)&q);

			if(ret != 0){
				return ret;
			}

			oquatf_diff(&q, (quatf*)in, &device->rotation_correction);
			return OHMD_S_OK;
		}
	case OHMD_POSITION_VECTOR:
		{
			// adjust position correction
			vec3f v;
			int ret = device->getf(device, OHMD_POSITION_VECTOR, (float*)&v);

			if(ret != 0){
				return ret;
			}

			for(int i = 0; i < 3; i++)
				device->position_correction.arr[i] = in[i] - v.arr[i];

			return OHMD_S_OK;
		}
	case OHMD_EXTERNAL_SENSOR_FUSION:
		{
			if(device->setf == NULL)
				return OHMD_S_UNSUPPORTED;

			return device->setf(device, type, in);
		}
	default:
		return OHMD_S_INVALID_PARAMETER;
	}
}

int OHMD_APIENTRY ohmd_device_setf(ohmd_device* device, ohmd_float_value type, const float* in)
{
	ohmd_lock_mutex(device->ctx->update_mutex);
	int ret = ohmd_device_setf_unp(device, type, in);
	ohmd_unlock_mutex(device->ctx->update_mutex);

	return ret;
}

int OHMD_APIENTRY ohmd_device_geti(ohmd_device* device, ohmd_int_value type, int* out)
{
	switch(type){
		case OHMD_SCREEN_HORIZONTAL_RESOLUTION:
			*out = device->properties.hres;
			return OHMD_S_OK;

		case OHMD_SCREEN_VERTICAL_RESOLUTION:
			*out = device->properties.vres;
			return OHMD_S_OK;
		
		case OHMD_CONTROL_COUNT:
			*out = device->properties.control_count;
			return OHMD_S_OK;

		case OHMD_CONTROLS_TYPES:
			memcpy(out, device->properties.controls_types, device->properties.control_count * sizeof(int));
			return OHMD_S_OK;
		
		case OHMD_CONTROLS_HINTS:
			memcpy(out, device->properties.controls_hints, device->properties.control_count * sizeof(int));
			return OHMD_S_OK;

		default:
				return OHMD_S_INVALID_PARAMETER;
	}
}

int OHMD_APIENTRY ohmd_device_seti(ohmd_device* device, ohmd_int_value type, const int* in)
{
	switch(type){
	default:
		return OHMD_S_INVALID_PARAMETER;
	}
}


int ohmd_device_set_data_unp(ohmd_device* device, ohmd_data_value type, const void* in)
{
    switch(type){
    case OHMD_DRIVER_DATA:
			device->set_data(device, OHMD_DRIVER_DATA, in);
			return OHMD_S_OK;

    case OHMD_DRIVER_PROPERTIES:
			device->set_data(device, OHMD_DRIVER_PROPERTIES, in);
			return OHMD_S_OK;

    default:
      return OHMD_S_INVALID_PARAMETER;
    }
}

int OHMD_APIENTRY ohmd_device_set_data(ohmd_device* device, ohmd_data_value type, const void* in)
{
	ohmd_lock_mutex(device->ctx->update_mutex);
	int ret = ohmd_device_set_data_unp(device, type, in);
	ohmd_unlock_mutex(device->ctx->update_mutex);

	return ret;
}

ohmd_status OHMD_APIENTRY ohmd_device_settings_seti(ohmd_device_settings* settings, ohmd_int_settings key, const int* val)
{
	switch(key){
	case OHMD_IDS_AUTOMATIC_UPDATE:
		settings->automatic_update = val[0] == 0 ? false : true;
		return OHMD_S_OK;

	default:
		return OHMD_S_INVALID_PARAMETER;
	}
}

ohmd_device_settings* OHMD_APIENTRY ohmd_device_settings_create(ohmd_context* ctx)
{
	return ohmd_alloc(ctx, sizeof(ohmd_device_settings));
}

void OHMD_APIENTRY ohmd_device_settings_destroy(ohmd_device_settings* settings)
{
	free(settings);
}

void* ohmd_allocfn(ohmd_context* ctx, const char* e_msg, size_t size)
{
	void* ret = calloc(1, size);
	if(!ret)
		ohmd_set_error(ctx, "%s", e_msg);
	return ret;
}

void ohmd_set_default_device_properties(ohmd_device_properties* props)
{
	props->ipd = 0.061f;
	props->znear = 0.1f;
	props->zfar = 1000.0f;
	ohmd_set_universal_distortion_k(props, 0, 0, 0, 1);
	ohmd_set_universal_aberration_k(props, 1.0, 1.0, 1.0);
}

void ohmd_calc_default_proj_matrices(ohmd_device_properties* props)
{
	mat4x4f proj_base; // base projection matrix

	// Calculate where the lens is on each screen,
	// and with the given value offset the projection matrix.
	float screen_center = props->hsize / 4.0f;
	float lens_shift = screen_center - props->lens_sep / 2.0f;
	// XXX: on CV1, props->hsize > props->lens_sep / 2.0,
	// I am not sure about the implications, but just taking the absolute
	// value of the offset seems to work.
	float proj_offset = fabs(4.0f * lens_shift / props->hsize);

	// Setup the base projection matrix. Each eye mostly have the
	// same projection matrix with the exception of the offset.
	omat4x4f_init_perspective(&proj_base, props->fov, props->ratio, props->znear, props->zfar);

	// Setup the two adjusted projection matrices. Each is setup to deal
	// with the fact that the lens is not in the center of the screen.
	// These matrices only change of the hardware changes, so static.
	mat4x4f translate;

	omat4x4f_init_translate(&translate, proj_offset, 0, 0);
	omat4x4f_mult(&translate, &proj_base, &props->proj_left);

	omat4x4f_init_translate(&translate, -proj_offset, 0, 0);
	omat4x4f_mult(&translate, &proj_base, &props->proj_right);
}

void ohmd_set_universal_distortion_k(ohmd_device_properties* props, float a, float b, float c, float d)
{
	props->universal_distortion_k[0] = a;
	props->universal_distortion_k[1] = b;
	props->universal_distortion_k[2] = c;
	props->universal_distortion_k[3] = d;
}

void ohmd_set_universal_aberration_k(ohmd_device_properties* props, float r, float g, float b)
{
	props->universal_aberration_k[0] = r;
	props->universal_aberration_k[1] = g;
	props->universal_aberration_k[2] = b;
}

uint64_t ohmd_monotonic_per_sec(ohmd_context* ctx)
{
	return ctx->monotonic_ticks_per_sec;
}

/*
 * Grabbed from druntime, good thing it's BOOST v1.0 as well.
 */
uint64_t ohmd_monotonic_conv(uint64_t ticks, uint64_t srcTicksPerSecond, uint64_t dstTicksPerSecond)
{
	// This would be more straightforward with floating point arithmetic,
	// but we avoid it here in order to avoid the rounding errors that that
	// introduces. Also, by splitting out the units in this way, we're able
	// to deal with much larger values before running into problems with
	// integer overflow.
	return ticks / srcTicksPerSecond * dstTicksPerSecond +
		ticks % srcTicksPerSecond * dstTicksPerSecond / srcTicksPerSecond;
}
