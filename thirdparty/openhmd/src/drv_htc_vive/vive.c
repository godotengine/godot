/*
 * OpenHMD - Free and Open Source API and drivers for immersive technology.
 * Copyright (C) 2013 Fredrik Hultin.
 * Copyright (C) 2013 Jakob Bornecrantz.
 * Distributed under the Boost 1.0 licence, see LICENSE for full text.
 */

/* HTC Vive Driver */

#define FEATURE_BUFFER_SIZE 256

#define HTC_ID                   0x0bb4
#define VIVE_HMD                 0x2c87

#define VALVE_ID                 0x28de
#define VIVE_WATCHMAN_DONGLE     0x2101
#define VIVE_LIGHTHOUSE_FPGA_RX  0x2000

#define VIVE_TIME_DIV 48000000.0f

#include <string.h>
#include <wchar.h>
#include <hidapi.h>
#include <assert.h>
#include <limits.h>
#include <stdint.h>
#include <stdbool.h>

#include "vive.h"

typedef struct {
	ohmd_device base;

	hid_device* hmd_handle;
	hid_device* imu_handle;
	fusion sensor_fusion;
	vec3f raw_accel, raw_gyro;
	uint32_t last_ticks;
	uint8_t last_seq;

	vive_config_packet vive_config;
	vec3f gyro_error;
	filter_queue gyro_q;
} vive_priv;

void vec3f_from_vive_vec_accel(const int16_t* smp, vec3f* out_vec)
{
	float gravity = 9.81f;
	float scaler = 4.0f * gravity / 32768.0f;

	out_vec->x = (float)smp[0] * scaler;
	out_vec->y = (float)smp[1] * scaler * -1;
	out_vec->z = (float)smp[2] * scaler * -1;
}

void vec3f_from_vive_vec_gyro(const int16_t* smp, vec3f* out_vec)
{
	float scaler = 8.7f / 32768.0f;
	out_vec->x = (float)smp[0] * scaler;
	out_vec->y = (float)smp[1] * scaler * -1;
	out_vec->z = (float)smp[2] * scaler * -1;
}

static bool process_error(vive_priv* priv)
{
	if(priv->gyro_q.at >= priv->gyro_q.size - 1)
		return true;

	ofq_add(&priv->gyro_q, &priv->raw_gyro);

	if(priv->gyro_q.at >= priv->gyro_q.size - 1){
		ofq_get_mean(&priv->gyro_q, &priv->gyro_error);
		printf("gyro error: %f, %f, %f\n", priv->gyro_error.x, priv->gyro_error.y, priv->gyro_error.z);
	}

	return false;
}

vive_sensor_sample* get_next_sample(vive_sensor_packet* pkt, int last_seq)
{
	int diff[3];

	for(int i = 0; i < 3; i++)
	{
		diff[i] = (int)pkt->samples[i].seq - last_seq;

		if(diff[i] < -128){
			diff[i] += 256;
		}
	}

	int closest_diff = INT_MAX;
	int closest_idx = -1;

	for(int i = 0; i < 3; i++)
	{
		if(diff[i] < closest_diff && diff[i] > 0 && diff[i] < 128){
			closest_diff = diff[i];
			closest_idx = i;
		}
	}

	if(closest_idx != -1)
		return pkt->samples + closest_idx;

	return NULL;
}

static void update_device(ohmd_device* device)
{
	vive_priv* priv = (vive_priv*)device;

	int size = 0;
	unsigned char buffer[FEATURE_BUFFER_SIZE];

	while((size = hid_read(priv->imu_handle, buffer, FEATURE_BUFFER_SIZE)) > 0){
		if(buffer[0] == VIVE_IRQ_SENSORS){
			vive_sensor_packet pkt;
			vive_decode_sensor_packet(&pkt, buffer, size);

			vive_sensor_sample* smp = NULL;

			while((smp = get_next_sample(&pkt, priv->last_seq)) != NULL)
			{
				if(priv->last_ticks == 0)
					priv->last_ticks = smp->time_ticks;

				uint32_t t1, t2;
				t1 = smp->time_ticks;
				t2 = priv->last_ticks;

				float dt = (t1 - t2) / VIVE_TIME_DIV;

				priv->last_ticks = smp->time_ticks;

				vec3f_from_vive_vec_accel(smp->acc, &priv->raw_accel);
				vec3f_from_vive_vec_gyro(smp->rot, &priv->raw_gyro);

				if(process_error(priv)){
					vec3f mag = {{0.0f, 0.0f, 0.0f}};
					vec3f gyro;
					ovec3f_subtract(&priv->raw_gyro, &priv->gyro_error, &gyro);

					ofusion_update(&priv->sensor_fusion, dt, &gyro, &priv->raw_accel, &mag);
				}

				priv->last_seq = smp->seq;
			}
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
	vive_priv* priv = (vive_priv*)device;

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
	int hret = 0;
	vive_priv* priv = (vive_priv*)device;

	LOGD("closing HTC Vive device");

	// turn the display off
	hret = hid_send_feature_report(priv->hmd_handle, vive_magic_power_off1, sizeof(vive_magic_power_off1));
	printf("power off magic 1: %d\n", hret);

	hret = hid_send_feature_report(priv->hmd_handle, vive_magic_power_off2, sizeof(vive_magic_power_off2));
	printf("power off magic 2: %d\n", hret);

	hid_close(priv->hmd_handle);
	hid_close(priv->imu_handle);

	free(device);
}

#if 0
static void dump_indexed_string(hid_device* device, int index)
{
	wchar_t wbuffer[512] = {0};
	char buffer[1024] = {0};

	int hret = hid_get_indexed_string(device, index, wbuffer, 511);

	if(hret == 0){
		wcstombs(buffer, wbuffer, sizeof(buffer));
		printf("indexed string 0x%02x: '%s'\n", index, buffer);
	}
}
#endif

static void dump_info_string(int (*fun)(hid_device*, wchar_t*, size_t), const char* what, hid_device* device)
{
	wchar_t wbuffer[512] = {0};
	char buffer[1024] = {0};

	int hret = fun(device, wbuffer, 511);

	if(hret == 0){
		wcstombs(buffer, wbuffer, sizeof(buffer));
		printf("%s: '%s'\n", what, buffer);
	}
}

#if 0
static void dumpbin(const char* label, const unsigned char* data, int length)
{
	printf("%s:\n", label);
	for(int i = 0; i < length; i++){
		printf("%02x ", data[i]);
		if((i % 16) == 15)
			printf("\n");
	}
	printf("\n");
}
#endif

static hid_device* open_device_idx(int manufacturer, int product, int iface, int iface_tot, int device_index)
{
	struct hid_device_info* devs = hid_enumerate(manufacturer, product);
	struct hid_device_info* cur_dev = devs;

	int idx = 0;
	int iface_cur = 0;
	hid_device* ret = NULL;

	while (cur_dev) {
		printf("%04x:%04x %s\n", manufacturer, product, cur_dev->path);

		if(idx == device_index && iface == iface_cur){
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
	vive_priv* priv = ohmd_alloc(driver->ctx, sizeof(vive_priv));

	if(!priv)
		return NULL;

	int hret = 0;

	priv->base.ctx = driver->ctx;

	int idx = atoi(desc->path);

	// Open the HMD device
	priv->hmd_handle = open_device_idx(HTC_ID, VIVE_HMD, 0, 1, idx);

	if(!priv->hmd_handle)
		goto cleanup;

	if(hid_set_nonblocking(priv->hmd_handle, 1) == -1){
		ohmd_set_error(driver->ctx, "failed to set non-blocking on device");
		goto cleanup;
	}

	// Open the lighthouse device
	priv->imu_handle = open_device_idx(VALVE_ID, VIVE_LIGHTHOUSE_FPGA_RX, 0, 2, idx);

	if(!priv->imu_handle)
		goto cleanup;

	if(hid_set_nonblocking(priv->imu_handle, 1) == -1){
		ohmd_set_error(driver->ctx, "failed to set non-blocking on device");
		goto cleanup;
	}

	dump_info_string(hid_get_manufacturer_string, "manufacturer", priv->hmd_handle);
	dump_info_string(hid_get_product_string , "product", priv->hmd_handle);
	dump_info_string(hid_get_serial_number_string, "serial number", priv->hmd_handle);

	// turn the display on
	hret = hid_send_feature_report(priv->hmd_handle, vive_magic_power_on, sizeof(vive_magic_power_on));
	printf("power on magic: %d\n", hret);

	// enable lighthouse
	//hret = hid_send_feature_report(priv->hmd_handle, vive_magic_enable_lighthouse, sizeof(vive_magic_enable_lighthouse));
	//printf("enable lighthouse magic: %d\n", hret);

	unsigned char buffer[128];
	int bytes;

	printf("Getting feature report 16 to 39\n");
	buffer[0] = 16;
	bytes = hid_get_feature_report(priv->imu_handle, buffer, sizeof(buffer));
	printf("got %i bytes\n", bytes);
	for (int i = 0; i < bytes; i++) {
		printf("%02hx ", buffer[i]);
	}
	printf("\n\n");

	unsigned char* packet_buffer = malloc(4096);

	int offset = 0;
	while (buffer[1] != 0) {
		buffer[0] = 17;
		bytes = hid_get_feature_report(priv->imu_handle, buffer, sizeof(buffer));

 		memcpy((uint8_t*)packet_buffer + offset, buffer+2, buffer[1]);
 		offset += buffer[1];
	}
	packet_buffer[offset] = '\0';
	//printf("Result: %s\n", packet_buffer);
	vive_decode_config_packet(&priv->vive_config, packet_buffer, offset);

	free(packet_buffer);

	// Set default device properties
	ohmd_set_default_device_properties(&priv->base.properties);

	// Set device properties TODO: Get from device
	priv->base.properties.hsize = 0.122822f;
	priv->base.properties.vsize = 0.068234f;
	priv->base.properties.hres = 2160;
	priv->base.properties.vres = 1200;
	priv->base.properties.lens_sep = 0.063500;
	priv->base.properties.lens_vpos = 0.049694;
	priv->base.properties.fov = DEG_TO_RAD(111.435f); //TODO: Confirm exact mesurements
	priv->base.properties.ratio = (2160.0f / 1200.0f) / 2.0f;

	// calculate projection eye projection matrices from the device properties
	ohmd_calc_default_proj_matrices(&priv->base.properties);

	// set up device callbacks
	priv->base.update = update_device;
	priv->base.close = close_device;
	priv->base.getf = getf;

	ofusion_init(&priv->sensor_fusion);

	ofq_init(&priv->gyro_q, 128);

	return (ohmd_device*)priv;

cleanup:
	if(priv)
		free(priv);

	return NULL;
}

static void get_device_list(ohmd_driver* driver, ohmd_device_list* list)
{
	struct hid_device_info* devs = hid_enumerate(HTC_ID, VIVE_HMD);
	struct hid_device_info* cur_dev = devs;

	int idx = 0;
	while (cur_dev) {
		ohmd_device_desc* desc = &list->devices[list->num_devices++];

		strcpy(desc->driver, "OpenHMD HTC Vive Driver");
		strcpy(desc->vendor, "HTC/Valve");
		strcpy(desc->product, "HTC Vive");

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
	LOGD("shutting down HTC Vive driver");
	free(drv);
}

ohmd_driver* ohmd_create_htc_vive_drv(ohmd_context* ctx)
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
