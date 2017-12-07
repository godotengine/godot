/*
 * OpenHMD - Free and Open Source API and drivers for immersive technology.
 * Copyright (C) 2013 Fredrik Hultin.
 * Copyright (C) 2013 Jakob Bornecrantz.
 * Distributed under the Boost 1.0 licence, see LICENSE for full text.
 */

/* Oculus Rift Driver Internal Interface */

#ifndef RIFT_H
#define RIFT_H

#include "../openhmdi.h"

#define FEATURE_BUFFER_SIZE 256

typedef enum {
	RIFT_CMD_SENSOR_CONFIG = 2,
	RIFT_CMD_RANGE = 4,
	RIFT_CMD_KEEP_ALIVE = 8,
	RIFT_CMD_DISPLAY_INFO = 9,
	RIFT_CMD_ENABLE_COMPONENTS = 0x1d,
	RIFT_CMD_POSITION_INFO = 15,
} rift_sensor_feature_cmd;

typedef enum {
	RIFT_CF_SENSOR,
	RIFT_CF_HMD
} rift_coordinate_frame;

typedef enum {
	RIFT_IRQ_SENSORS = 1,
	RIFT_IRQ_SENSORS_DK2 = 11
} rift_irq_cmd;

typedef enum {
	RIFT_DT_NONE,
	RIFT_DT_SCREEN_ONLY,
	RIFT_DT_DISTORTION
} rift_distortion_type;

typedef enum {
	RIFT_COMPONENT_DISPLAY = 1,
	RIFT_COMPONENT_AUDIO = 2,
	RIFT_COMPONENT_LEDS = 4
} rift_component_type;

// Sensor config flags
#define RIFT_SCF_RAW_MODE           0x01
#define RIFT_SCF_CALIBRATION_TEST   0x02
#define RIFT_SCF_USE_CALIBRATION    0x04
#define RIFT_SCF_AUTO_CALIBRATION   0x08
#define RIFT_SCF_MOTION_KEEP_ALIVE  0x10
#define RIFT_SCF_COMMAND_KEEP_ALIVE 0x20
#define RIFT_SCF_SENSOR_COORDINATES 0x40

static const unsigned char rift_enable_leds_dk2[17] = {
	0x0c, 0x00, 0x00, 0x00, 0x01, 0x00, 0x5E, 0x01, 0x1A, 0x41, 0x00, 0x00, 0x7F,
};

static const unsigned char rift_enable_leds_cv1[17] = {
	0x0c, 0x00, 0x00, 0xFF, 0x05, 0x00, 0x8F, 0x01, 0x00, 0x4B, 0x00, 0x00, 0x7F,
};

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
	uint8_t num_samples;
	uint32_t timestamp;
	uint16_t last_command_id;
	int16_t temperature;
	pkt_tracker_sample samples[3];
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

typedef struct {
	uint8_t flags;
	int32_t pos_x;
	int32_t pos_y;
	int32_t pos_z;
	int16_t dir_x;
	int16_t dir_y;
	int16_t dir_z;
	uint8_t index;
	uint8_t num;
	uint8_t type;
} pkt_position_info;

typedef struct {
	// Relative position in micrometers
	vec3f pos;
	// Normal
	vec3f dir;
} rift_led;

bool decode_sensor_range(pkt_sensor_range* range, const unsigned char* buffer, int size);
bool decode_sensor_display_info(pkt_sensor_display_info* info, const unsigned char* buffer, int size);
bool decode_sensor_config(pkt_sensor_config* config, const unsigned char* buffer, int size);
bool decode_tracker_sensor_msg(pkt_tracker_sensor* msg, const unsigned char* buffer, int size);
bool decode_tracker_sensor_msg_dk2(pkt_tracker_sensor* msg, const unsigned char* buffer, int size);
bool decode_position_info(pkt_position_info* p, const unsigned char* buffer, int size);

void vec3f_from_rift_vec(const int32_t* smp, vec3f* out_vec);

int encode_sensor_config(unsigned char* buffer, const pkt_sensor_config* config);
int encode_keep_alive(unsigned char* buffer, const pkt_keep_alive* keep_alive);
int encode_enable_components(unsigned char* buffer, bool display, bool audio, bool leds);

void dump_packet_sensor_range(const pkt_sensor_range* range);
void dump_packet_sensor_config(const pkt_sensor_config* config);
void dump_packet_sensor_display_info(const pkt_sensor_display_info* info);
void dump_packet_tracker_sensor(const pkt_tracker_sensor* sensor);

#endif
