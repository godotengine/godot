#include "vive.h"
#include "vive_config.h"

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

bool vive_decode_sensor_packet(vive_sensor_packet* pkt, const unsigned char* buffer, int size)
{
	if(size != 52){
		LOGE("invalid vive sensor packet size (expected 52 but got %d)", size);
		return false;
	}

	pkt->report_id = read8(&buffer);

	for(int j = 0; j < 3; j++){
		// acceleration
		for(int i = 0; i < 3; i++){
			pkt->samples[j].acc[i] = read16(&buffer);
		}

		// rotation
		for(int i = 0; i < 3; i++){
			pkt->samples[j].rot[i] = read16(&buffer);
		}

		pkt->samples[j].time_ticks = read32(&buffer);
		pkt->samples[j].seq = read8(&buffer);
	}

	return true;
}

//Trim function for removing tabs and spaces from string buffers
void trim(const char* src, char* buff, const unsigned int sizeBuff)
{
    if(sizeBuff < 1)
    return;

    const char* current = src;
    unsigned int i = 0;
    while(*current != '\0' && i < sizeBuff-1)
    {
        if(*current != ' ' && *current != '\t')
            buff[i++] = *current;
        ++current;
    }
    buff[i] = '\0';
}

bool vive_decode_config_packet(vive_config_packet* pkt, const unsigned char* buffer, uint16_t size)
{/*
	if(size != 4069){
		LOGE("invalid vive sensor packet size (expected 4069 but got %d)", size);
		return false;
	}*/

	pkt->report_id = 17;
	pkt->length = size;

	unsigned char output[32768];
	int output_size = 32768;

	//int cmp_status = uncompress(pUncomp, &uncomp_len, pCmp, cmp_len);
	int cmp_status = uncompress(output, (mz_ulong*)&output_size, buffer, (mz_ulong)pkt->length);
	if (cmp_status != Z_OK){
		LOGE("invalid vive config, could not uncompress");
		return false;
	}

	LOGE("Decompressed from %u to %u bytes\n", (mz_uint32)pkt->length, (mz_uint32)output_size);

	//printf("Debug print all the RAW JSON things!\n%s", output);
	//pUncomp should now be the uncompressed data, lets get the json from it
	/** DEBUG JSON PARSER CODE **/
	trim((char*)output,(char*)output,output_size);
	//printf("%s\n",output);
	/*
	FILE* dfp;
	dfp = fopen("jsondebug.json","w");
	json_enable_debug(3, dfp);*/
	int status = json_read_object((char*)output, sensor_offsets, NULL);
	printf("\n--- Converted Vive JSON Data ---\n\n");
	printf("acc_bias = %f %f %f\n", acc_bias[0], acc_bias[1], acc_bias[2]);
	printf("acc_scale = %f %f %f\n", acc_scale[0], acc_scale[1], acc_scale[2]);
	printf("gyro_bias = %f %f %f\n", gyro_bias[0], gyro_bias[1], gyro_bias[2]);
	printf("gyro_scale = %f %f %f\n", gyro_scale[0], gyro_scale[1], gyro_scale[2]);
	printf("\n--- End of Vive JSON Data ---\n\n");

	if (status != 0)
		puts(json_error_string(status));
	/** END OF DEBUG JSON PARSER CODE **/

//	free(pCmp);
//	free(pUncomp);

	return true;
}
