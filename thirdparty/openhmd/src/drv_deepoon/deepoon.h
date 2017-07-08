/*
 * OpenHMD - Free and Open Source API and drivers for immersive technology.
 * Copyright (C) 2013 Fredrik Hultin.
 * Copyright (C) 2013 Jakob Bornecrantz.
 * Distributed under the Boost 1.0 licence, see LICENSE for full text.
 */

/* Deepoon Driver Internal Interface */

#ifndef DEEPOON_H
#define DEEPOON_H

#include "../openhmdi.h"

#define FEATURE_BUFFER_SIZE 256

typedef enum {
	RIFT_CMD_SENSOR_CONFIG = 2,
	RIFT_CMD_RANGE = 4,
	RIFT_CMD_KEEP_ALIVE = 8,
	RIFT_CMD_DISPLAY_INFO = 9
} rift_sensor_feature_cmd;

typedef enum {
	RIFT_CF_SENSOR,
	RIFT_CF_HMD
} rift_coordinate_frame;

typedef enum {
	RIFT_IRQ_SENSORS = 1
} rift_irq_cmd;

typedef enum {
	RIFT_DT_NONE,
	RIFT_DT_SCREEN_ONLY,
	RIFT_DT_DISTORTION
} rift_distortion_type;

// Sensor config flags
#define RIFT_SCF_RAW_MODE           0x01
#define RIFT_SCF_CALIBRATION_TEST   0x02
#define RIFT_SCF_USE_CALIBRATION    0x04
#define RIFT_SCF_AUTO_CALIBRATION   0x08
#define RIFT_SCF_MOTION_KEEP_ALIVE  0x10
#define RIFT_SCF_COMMAND_KEEP_ALIVE 0x20
#define RIFT_SCF_SENSOR_COORDINATES 0x40



typedef struct {
	uint16_t command_id;
	uint16_t accel_scale;
	uint16_t gyro_scale;
	uint16_t mag_scale;
} pkt_sensor_range;

typedef struct {
	int32_t accel[3];
	int32_t gyro[3];
} pkt_tracker_sample;

typedef struct {
	uint8_t report_id;
	uint8_t sample_delta;
	uint16_t sample_number;
	uint32_t tick;
	pkt_tracker_sample samples[2];
	int16_t mag[3];
} pkt_tracker_sensor;

typedef struct {
    uint16_t command_id;
    uint8_t flags;
    uint16_t packet_interval;
    uint16_t keep_alive_interval; // in ms
} pkt_sensor_config;

typedef struct {
	uint16_t command_id;
	rift_distortion_type distortion_type;
	uint8_t distortion_type_opts;
	uint16_t h_resolution, v_resolution;
	float h_screen_size, v_screen_size;
	float v_center;
	float lens_separation;
	float eye_to_screen_distance[2];
	float distortion_k[6];
} pkt_sensor_display_info;

typedef struct {
	uint16_t command_id;
	uint16_t keep_alive_interval;
} pkt_keep_alive;


bool dp_decode_sensor_range(pkt_sensor_range* range, const unsigned char* buffer, int size);
bool dp_decode_sensor_display_info(pkt_sensor_display_info* info, const unsigned char* buffer, int size);
bool dp_decode_sensor_config(pkt_sensor_config* config, const unsigned char* buffer, int size);
bool dp_decode_tracker_sensor_msg(pkt_tracker_sensor* msg, const unsigned char* buffer, int size);

void vec3f_from_dp_vec(const int32_t* smp, vec3f* out_vec);

int dp_encode_sensor_config(unsigned char* buffer, const pkt_sensor_config* config);
int dp_encode_keep_alive(unsigned char* buffer, const pkt_keep_alive* keep_alive);

void dp_dump_packet_sensor_range(const pkt_sensor_range* range);
void dp_dump_packet_sensor_config(const pkt_sensor_config* config);
void dp_dump_packet_sensor_display_info(const pkt_sensor_display_info* info);
void dp_dump_packet_tracker_sensor(const pkt_tracker_sensor* sensor);

#endif
