#ifndef PSVR_H
#define PSVR_H

#include <stdint.h>
#include <stdbool.h>

#include "../openhmdi.h"

typedef enum
{
	PSVR_IRQ_SENSORS = 0,
	PSVR_IRQ_VOLUME_PLUS = 2,
	PSVR_IRQ_VOLUME_MINUS = 4,
	PSVR_IRQ_MIC_MUTE = 8
} psvr_irq_cmd;

typedef struct
{
	int16_t accel[3];
	int16_t gyro[3];
	uint32_t tick;
	uint8_t seq;
	uint8_t volume;
	uint8_t proximity;
	uint8_t proximity_state;
} psvr_sensor_sample;

typedef struct
{
	uint8_t report_id;
	uint32_t tick;
	psvr_sensor_sample samples[1];
} psvr_sensor_packet;

static const unsigned char psvr_vrmode_on[8]  = {
	0x23, 0x00, 0xaa, 0x04, 0x01, 0x00, 0x00, 0x00
};

static const unsigned char psvr_tracking_on[12]  = {
	0x11, 0x00, 0xaa, 0x08, 0x00, 0xff, 0xff, 0xff, 0x00, 0x00, 0x00, 0x00
};


static const unsigned char psvr_power_on[8]  = {
	0x17, 0x76, 0xaa, 0x04, 0x01, 0x00, 0x00, 0x00
};


void vec3f_from_psvr_vec(const int16_t* smp, vec3f* out_vec);
bool psvr_decode_sensor_packet(psvr_sensor_packet* pkt, const unsigned char* buffer, int size);

#endif
