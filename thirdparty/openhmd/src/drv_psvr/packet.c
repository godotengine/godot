#include "psvr.h"

#ifdef _MSC_VER
#define inline __inline
#endif

inline static uint8_t read8(const unsigned char** buffer)
{
	uint8_t ret = **buffer;
	*buffer += 1;
	return ret;
}

inline static int16_t read16(const unsigned char** buffer)
{
	int16_t ret = **buffer | (*(*buffer + 1) << 8);
	*buffer += 2;
	return ret;
}

inline static uint32_t read32(const unsigned char** buffer)
{
	uint32_t ret = **buffer | (*(*buffer + 1) << 8) | (*(*buffer + 2) << 16) | (*(*buffer + 3) << 24);
	*buffer += 4;
	return ret;
}

bool psvr_decode_sensor_packet(psvr_sensor_packet* pkt, const unsigned char* buffer, int size)
{
	if(size != 64){
		LOGE("invalid psvr sensor packet size (expected 64 but got %d)", size);
		return false;
	}

	buffer += 2; //skip 2
	pkt->samples[0].volume = read16(&buffer); //volume
	buffer += 12; //unknown, skip 12
	pkt->samples[0].tick = read32(&buffer); //TICK
	// acceleration
	for(int i = 0; i < 3; i++){
		pkt->samples[0].gyro[i] = read16(&buffer);
	}

	// rotation
	for(int i = 0; i < 3; i++){
		pkt->samples[0].accel[i] = read16(&buffer);
	}//34
	buffer += 23; //probably other sample somewhere
	pkt->samples[0].proximity = read8(&buffer); //255 for close
	pkt->samples[0].proximity_state = read8(&buffer); // 0 (nothing) to 3 (headset is on)

	return true;
}
