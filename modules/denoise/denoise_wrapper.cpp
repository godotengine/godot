#include "denoise_wrapper.h"
#include "thirdparty/oidn/include/OpenImageDenoise/oidn.h"
#include <stdio.h>

void *oidn_denoiser_init() {
	OIDNDeviceImpl *device = oidnNewDevice(OIDN_DEVICE_TYPE_CPU);
	oidnCommitDevice(device);
	return device;
}

bool oidn_denoise(void *deviceptr, float *p_floats, int p_width, int p_height) {
	OIDNDeviceImpl *device = (OIDNDeviceImpl *)deviceptr;
	OIDNFilter filter = oidnNewFilter(device, "RTLightmap");
	oidnSetSharedFilterImage(filter, "color", (void *)p_floats, OIDN_FORMAT_FLOAT3, p_width, p_height, 0, 0, 0);
	oidnSetSharedFilterImage(filter, "output", (void *)p_floats, OIDN_FORMAT_FLOAT3, p_width, p_height, 0, 0, 0);
	oidnSetFilter1b(filter, "hdr", true);
	//oidnSetFilter1f(filter, "hdrScale", 1.0f);
	oidnCommitFilter(filter);
	oidnExecuteFilter(filter);

	const char *msg;
	bool success = true;
	if (oidnGetDeviceError(device, &msg) != OIDN_ERROR_NONE) {
		printf("LightmapDenoiser: %s\n", msg);
		success = false;
	}

	oidnReleaseFilter(filter);
	return success;
}

void oidn_denoiser_finish(void *device) {
	oidnReleaseDevice((OIDNDeviceImpl *)device);
}
