/*
 * OpenHMD - Free and Open Source API and drivers for immersive technology.
 * Copyright (C) 2013 Fredrik Hultin.
 * Copyright (C) 2013 Jakob Bornecrantz.
 * Distributed under the Boost 1.0 licence, see LICENSE for full text.
 */

/* Deepoon Driver - HID/USB Driver Implementation */

#include <stdlib.h>
#include <hidapi.h>
#include <string.h>
#include <stdio.h>
#include <time.h>
#include <assert.h>

#include "deepoon.h"

#define TICK_LEN (1.0f / 1000000.0f) // 1000 Hz ticks
#define KEEP_ALIVE_VALUE (10 * 1000)
#define SETFLAG(_s, _flag, _val) (_s) = ((_s) & ~(_flag)) | ((_val) ? (_flag) : 0)

#define DEEPOON_ID					0x0483
#define DEEPOON_HMD					0x5750

typedef struct {
	ohmd_device base;

	hid_device* handle;
	pkt_sensor_range sensor_range;
	pkt_sensor_display_info display_info;
	rift_coordinate_frame coordinate_frame, hw_coordinate_frame;
	pkt_sensor_config sensor_config;
	pkt_tracker_sensor sensor;
	double last_keep_alive;
	fusion sensor_fusion;
	vec3f raw_mag, raw_accel, raw_gyro;
} rift_priv;

static rift_priv* rift_priv_get(ohmd_device* device)
{
	return (rift_priv*)device;
}

static int get_feature_report(rift_priv* priv, rift_sensor_feature_cmd cmd, unsigned char* buf)
{
	memset(buf, 0, FEATURE_BUFFER_SIZE);
	buf[0] = (unsigned char)cmd;
	return hid_get_feature_report(priv->handle, buf, FEATURE_BUFFER_SIZE);
}

static int send_feature_report(rift_priv* priv, const unsigned char *data, size_t length)
{
	return hid_send_feature_report(priv->handle, data, length);
}

static void set_coordinate_frame(rift_priv* priv, rift_coordinate_frame coordframe)
{
	priv->coordinate_frame = coordframe;

	// set the RIFT_SCF_SENSOR_COORDINATES in the sensor config to match whether coordframe is hmd or sensor
	SETFLAG(priv->sensor_config.flags, RIFT_SCF_SENSOR_COORDINATES, coordframe == RIFT_CF_SENSOR);

	// encode send the new config to the Rift
	unsigned char buf[FEATURE_BUFFER_SIZE];
	int size = dp_encode_sensor_config(buf, &priv->sensor_config);
	if(send_feature_report(priv, buf, size) == -1){
		ohmd_set_error(priv->base.ctx, "send_feature_report failed in set_coordinate frame");
		return;
	}

	// read the state again, set the hw_coordinate_frame to match what
	// the hardware actually is set to just incase it doesn't stick.
	size = get_feature_report(priv, RIFT_CMD_SENSOR_CONFIG, buf);
	if(size <= 0){
		LOGW("could not set coordinate frame");
		priv->hw_coordinate_frame = RIFT_CF_HMD;
		return;
	}

	priv->hw_coordinate_frame = (priv->sensor_config.flags & RIFT_SCF_SENSOR_COORDINATES) ? RIFT_CF_SENSOR : RIFT_CF_HMD;

	if(priv->hw_coordinate_frame != coordframe) {
		LOGW("coordinate frame didn't stick");
	}
}

static void handle_tracker_sensor_msg(rift_priv* priv, unsigned char* buffer, int size)
{
	uint32_t last_sample_tick = priv->sensor.tick;

	if(!dp_decode_tracker_sensor_msg(&priv->sensor, buffer, size)){
		LOGE("couldn't decode tracker sensor message");
	}

	pkt_tracker_sensor* s = &priv->sensor;

	dp_dump_packet_tracker_sensor(s);

	uint32_t tick_delta = 1000;
	if(last_sample_tick > 0) //startup correction
		tick_delta = s->tick - last_sample_tick;

	float dt = tick_delta * TICK_LEN;
	vec3f mag = {{0.0f, 0.0f, 0.0f}};

	for(int i = 0; i < 1; i++){ //just use 1 sample since we don't have sample order for this frame
		vec3f_from_dp_vec(s->samples[i].accel, &priv->raw_accel);
		vec3f_from_dp_vec(s->samples[i].gyro, &priv->raw_gyro);

		ofusion_update(&priv->sensor_fusion, dt, &priv->raw_gyro, &priv->raw_accel, &mag);

		// reset dt to tick_len for the last samples if there were more than one sample
		dt = TICK_LEN;
	}
}

static void update_device(ohmd_device* device)
{
	rift_priv* priv = rift_priv_get(device);
	unsigned char buffer[FEATURE_BUFFER_SIZE];

	// Handle keep alive messages
	double t = ohmd_get_tick();
	if(t - priv->last_keep_alive >= (double)priv->sensor_config.keep_alive_interval / 1000.0 - .2){
		// send keep alive message
		pkt_keep_alive keep_alive = { 0, priv->sensor_config.keep_alive_interval };
		int ka_size = dp_encode_keep_alive(buffer, &keep_alive);
		send_feature_report(priv, buffer, ka_size);

		// Update the time of the last keep alive we have sent.
		priv->last_keep_alive = t;
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

		// currently the only message type the hardware supports (I think)
		if(buffer[0] == RIFT_IRQ_SENSORS || buffer[0] == 11){
			handle_tracker_sensor_msg(priv, buffer, size);
		}else{
			LOGE("unknown message type: %u", buffer[0]);
		}
	}
}

static int getf(ohmd_device* device, ohmd_float_value type, float* out)
{
	rift_priv* priv = rift_priv_get(device);

	switch(type){
	case OHMD_DISTORTION_K: {
			for (int i = 0; i < 6; i++) {
				out[i] = priv->display_info.distortion_k[i];
			}
			break;
		}

	case OHMD_ROTATION_QUAT: {
			*(quatf*)out = priv->sensor_fusion.orient;
			break;
		}

	case OHMD_POSITION_VECTOR:
		out[0] = out[1] = out[2] = 0;
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
	rift_priv* priv = rift_priv_get(device);
	hid_close(priv->handle);
	free(priv);
}

static char* _hid_to_unix_path(char* path)
{
	char bus [4];
	char dev [4];
	char *result = malloc( sizeof(char) * ( 20 + 1 ) );

	sprintf (bus, "%.*s\n", 4, path);
	sprintf (dev, "%.*s\n", 4, path + 5);

	sprintf (result, "/dev/bus/usb/%03d/%03d",
		(int)strtol(bus, NULL, 16),
		(int)strtol(dev, NULL, 16));
	return result;
}

static ohmd_device* open_device(ohmd_driver* driver, ohmd_device_desc* desc)
{
	rift_priv* priv = ohmd_alloc(driver->ctx, sizeof(rift_priv));
	if(!priv)
		goto cleanup;

	priv->base.ctx = driver->ctx;

	// Open the HID device
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

	unsigned char buf[FEATURE_BUFFER_SIZE];

	int size;

	// if the sensor has display info data, use HMD coordinate frame
	priv->coordinate_frame = priv->display_info.distortion_type != RIFT_DT_NONE ? RIFT_CF_HMD : RIFT_CF_SENSOR;

	// enable calibration
	SETFLAG(priv->sensor_config.flags, RIFT_SCF_USE_CALIBRATION, 1);
	SETFLAG(priv->sensor_config.flags, RIFT_SCF_AUTO_CALIBRATION, 1);

	// apply sensor config
	set_coordinate_frame(priv, priv->coordinate_frame);

	// set keep alive interval to n seconds
	pkt_keep_alive keep_alive = { 0, KEEP_ALIVE_VALUE };
	size = dp_encode_keep_alive(buf, &keep_alive);
	send_feature_report(priv, buf, size);

	// Update the time of the last keep alive we have sent.
	priv->last_keep_alive = ohmd_get_tick();

	// Set default device properties
	ohmd_set_default_device_properties(&priv->base.properties);

	// Set device properties
	//NOTE: These values are estimations, no one has taken one apart to check
	priv->base.properties.hsize = 0.1698f;
	priv->base.properties.vsize = 0.0936f;
	priv->base.properties.hres = 1920;
	priv->base.properties.vres = 1080;
	priv->base.properties.lens_sep = 0.0849f;
	priv->base.properties.lens_vpos = 0.0468f;
	priv->base.properties.fov = DEG_TO_RAD(110.0); // TODO calculate.
	priv->base.properties.ratio = ((float)1920 / (float)1080) / 2.0f;

	// calculate projection eye projection matrices from the device properties
	ohmd_calc_default_proj_matrices(&priv->base.properties);

	// set up device callbacks
	priv->base.update = update_device;
	priv->base.close = close_device;
	priv->base.getf = getf;

	// initialize sensor fusion
	ofusion_init(&priv->sensor_fusion);

	return &priv->base;

cleanup:
	if(priv)
		free(priv);

	return NULL;
}

static void get_device_list(ohmd_driver* driver, ohmd_device_list* list)
{
	struct hid_device_info* devs = hid_enumerate(DEEPOON_ID, DEEPOON_HMD);
	struct hid_device_info* cur_dev = devs;

	while (cur_dev) {
		// This is needed because DeePoon share USB IDs with the Nolo.
		if (wcscmp(cur_dev->manufacturer_string, L"DeePoon VR, Inc.")==0 &&
			wcscmp(cur_dev->product_string, L"DeePoon Tracker Device")==0) {

			ohmd_device_desc* desc = &list->devices[list->num_devices++];

			strcpy(desc->driver, "Deepoon Driver");
			strcpy(desc->vendor, "Deepoon");
			strcpy(desc->product, "Deepoon E2");

			desc->device_class = OHMD_DEVICE_CLASS_HMD;
			desc->device_flags = OHMD_DEVICE_FLAGS_ROTATIONAL_TRACKING;

			desc->revision = 0;
			strcpy(desc->path, cur_dev->path);
			desc->driver_ptr = driver;
		}
		cur_dev = cur_dev->next;
	}

	hid_free_enumeration(devs);
}

static void destroy_driver(ohmd_driver* drv)
{
	LOGD("shutting down driver");
	hid_exit();
	free(drv);
}

ohmd_driver* ohmd_create_deepoon_drv(ohmd_context* ctx)
{
	ohmd_driver* drv = ohmd_alloc(ctx, sizeof(ohmd_driver));
	if(drv == NULL)
		return NULL;

	drv->get_device_list = get_device_list;
	drv->open_device = open_device;
	drv->ctx = ctx;
	drv->get_device_list = get_device_list;
	drv->open_device = open_device;
	drv->destroy = destroy_driver;

	return drv;
}
