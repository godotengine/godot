/*
 * OpenHMD - Free and Open Source API and drivers for immersive technology.
 * Copyright (C) 2013 Fredrik Hultin.
 * Copyright (C) 2013 Jakob Bornecrantz.
 * Distributed under the Boost 1.0 licence, see LICENSE for full text.
 */

/* Sony PSVR Driver */

#define FEATURE_BUFFER_SIZE 256

#define TICK_LEN (1.0f / 1000000.0f) // 1000 Hz ticks

#define SONY_ID                  0x054c
#define PSVR_HMD                 0x09af

#include <string.h>
#include <wchar.h>
#include <hidapi.h>
#include <assert.h>
#include <limits.h>
#include <stdint.h>
#include <stdbool.h>

#include "psvr.h"

typedef struct {
	ohmd_device base;

	hid_device* hmd_handle;
	hid_device* hmd_control;
	fusion sensor_fusion;
	vec3f raw_accel, raw_gyro;
	uint32_t last_ticks;
	uint8_t last_seq;
	psvr_sensor_packet sensor;

} psvr_priv;

void vec3f_from_psvr_vec(const int16_t* smp, vec3f* out_vec)
{
	out_vec->x = (float)smp[1] * 0.001f;
	out_vec->y = (float)smp[0] * 0.001f;
	out_vec->z = (float)smp[2] * 0.001f * -1.0f;
}

static void handle_tracker_sensor_msg(psvr_priv* priv, unsigned char* buffer, int size)
{
	uint32_t last_sample_tick = priv->sensor.tick;

	if(!psvr_decode_sensor_packet(&priv->sensor, buffer, size)){
		LOGE("couldn't decode tracker sensor message");
	}

	psvr_sensor_packet* s = &priv->sensor;

	uint32_t tick_delta = 1000;
	if(last_sample_tick > 0) //startup correction
		tick_delta = s->tick - last_sample_tick;

	float dt = tick_delta * TICK_LEN;
	vec3f mag = {{0.0f, 0.0f, 0.0f}};

	for(int i = 0; i < 1; i++){ //just use 1 sample since we don't have sample order for 	 frame
		vec3f_from_psvr_vec(s->samples[i].accel, &priv->raw_accel);
		vec3f_from_psvr_vec(s->samples[i].gyro, &priv->raw_gyro);

		ofusion_update(&priv->sensor_fusion, dt, &priv->raw_gyro, &priv->raw_accel, &mag);

		// reset dt to tick_len for the last samples if there were more than one sample
		dt = TICK_LEN;
	}
}

static void update_device(ohmd_device* device)
{
	psvr_priv* priv = (psvr_priv*)device;

	int size = 0;
	unsigned char buffer[FEATURE_BUFFER_SIZE];

	while(true){
		int size = hid_read(priv->hmd_handle, buffer, FEATURE_BUFFER_SIZE);
		if(size < 0){
			LOGE("error reading from device");
			return;
		} else if(size == 0) {
			return; // No more messages, return.
		}

		// currently the only message type the hardware supports (I think)
		if(buffer[0] == PSVR_IRQ_SENSORS){
			handle_tracker_sensor_msg(priv, buffer, size);
		}else if (buffer[0] == PSVR_IRQ_VOLUME_PLUS){
			//TODO implement
		}else if (buffer[0] == PSVR_IRQ_VOLUME_MINUS){
			//TODO implement
		}else if (buffer[0] == PSVR_IRQ_MIC_MUTE){
			//TODO implement
		}else{
			LOGE("unknown message type: %u", buffer[0]);
		}
	}

	if(size < 0){
		LOGE("error reading from device");
	}
}

static int getf(ohmd_device* device, ohmd_float_value type, float* out)
{
	psvr_priv* priv = (psvr_priv*)device;

	switch(type){
	case OHMD_ROTATION_QUAT:
		*(quatf*)out = priv->sensor_fusion.orient;
		break;

	case OHMD_POSITION_VECTOR:
		out[0] = out[1] = out[2] = 0;
		break;

	case OHMD_DISTORTION_K:
		// TODO this should be set to the equivalent of no distortion
		memset(out, 0, sizeof(float) * 6);
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
	psvr_priv* priv = (psvr_priv*)device;

	LOGD("closing HTC PSVR device");

	hid_close(priv->hmd_handle);
	hid_close(priv->hmd_control);

	free(device);
}

static hid_device* open_device_idx(int manufacturer, int product, int iface, int iface_tot, int device_index)
{
	struct hid_device_info* devs = hid_enumerate(manufacturer, product);
	struct hid_device_info* cur_dev = devs;

	int idx = 0;
	int iface_cur = 0;
	hid_device* ret = NULL;

	while (cur_dev) {
		printf("%04x:%04x %s\n", manufacturer, product, cur_dev->path);

		if(findEndPoint(cur_dev->path, device_index) > 0 && iface == iface_cur){
			ret = hid_open_path(cur_dev->path);
			printf("opening\n");
		}

		cur_dev = cur_dev->next;

		iface_cur++;

		if(iface_cur >= iface_tot){
			idx++;
			iface_cur = 0;
		}
	}

	hid_free_enumeration(devs);

	return ret;
}

static ohmd_device* open_device(ohmd_driver* driver, ohmd_device_desc* desc)
{
	psvr_priv* priv = ohmd_alloc(driver->ctx, sizeof(psvr_priv));

	if(!priv)
		return NULL;

	priv->base.ctx = driver->ctx;

	int idx = atoi(desc->path);

	// Open the HMD device
	priv->hmd_handle = open_device_idx(SONY_ID, PSVR_HMD, 0, 0, 4);

	if(!priv->hmd_handle)
		goto cleanup;

	if(hid_set_nonblocking(priv->hmd_handle, 1) == -1){
		ohmd_set_error(driver->ctx, "failed to set non-blocking on device");
		goto cleanup;
	}

	// Open the HMD Control device
	priv->hmd_control = open_device_idx(SONY_ID, PSVR_HMD, 0, 0, 5);

	if(!priv->hmd_control)
		goto cleanup;

	if(hid_set_nonblocking(priv->hmd_control, 1) == -1){
		ohmd_set_error(driver->ctx, "failed to set non-blocking on device");
		goto cleanup;
	}

	// turn the display on
	hid_write(priv->hmd_control, psvr_power_on, sizeof(psvr_power_on));
	
	// set VR mode for the hmd
	hid_write(priv->hmd_control, psvr_vrmode_on, sizeof(psvr_vrmode_on));

	// Set default device properties
	ohmd_set_default_device_properties(&priv->base.properties);

	// Set device properties TODO: Get from device
	priv->base.properties.hsize = 0.126; //from calculated specs
	priv->base.properties.vsize = 0.071; //from calculated specs
	priv->base.properties.hres = 1920;
	priv->base.properties.vres = 1080;
	priv->base.properties.lens_sep = 0.063500;
	priv->base.properties.lens_vpos = 0.049694;
	priv->base.properties.fov = DEG_TO_RAD(103.57f); //TODO: Confirm exact mesurements
	priv->base.properties.ratio = (1920.0f / 1080.0f) / 2.0f;

	// calculate projection eye projection matrices from the device properties
	ohmd_calc_default_proj_matrices(&priv->base.properties);

	// set up device callbacks
	priv->base.update = update_device;
	priv->base.close = close_device;
	priv->base.getf = getf;

	ofusion_init(&priv->sensor_fusion);

	return (ohmd_device*)priv;

cleanup:
	if(priv)
		free(priv);

	return NULL;
}

static void get_device_list(ohmd_driver* driver, ohmd_device_list* list)
{
	struct hid_device_info* devs = hid_enumerate(SONY_ID, PSVR_HMD);
	struct hid_device_info* cur_dev = devs;

	int idx = 0;
	while (cur_dev) {
		ohmd_device_desc* desc = &list->devices[list->num_devices++];

		strcpy(desc->driver, "OpenHMD Sony PSVR Driver");
		strcpy(desc->vendor, "Sony");
		strcpy(desc->product, "PSVR");

		desc->revision = 0;

		snprintf(desc->path, OHMD_STR_SIZE, "%d", idx);

		desc->driver_ptr = driver;
		
		desc->device_class = OHMD_DEVICE_CLASS_HMD;
		desc->device_flags = OHMD_DEVICE_FLAGS_ROTATIONAL_TRACKING;

		cur_dev = cur_dev->next;
		idx++;
	}

	hid_free_enumeration(devs);
}

static void destroy_driver(ohmd_driver* drv)
{
	LOGD("shutting down Sony PSVR driver");
	free(drv);
}

ohmd_driver* ohmd_create_psvr_drv(ohmd_context* ctx)
{
	ohmd_driver* drv = ohmd_alloc(ctx, sizeof(ohmd_driver));

	if(!drv)
		return NULL;

	drv->get_device_list = get_device_list;
	drv->open_device = open_device;
	drv->destroy = destroy_driver;
	drv->ctx = ctx;

	return drv;
}
