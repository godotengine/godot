/**************************************************************************/
/*  denoise_wrapper.cpp                                                   */
/**************************************************************************/
/*                         This file is part of:                          */
/*                             GODOT ENGINE                               */
/*                        https://godotengine.org                         */
/**************************************************************************/
/* Copyright (c) 2014-present Godot Engine contributors (see AUTHORS.md). */
/* Copyright (c) 2007-2014 Juan Linietsky, Ariel Manzur.                  */
/*                                                                        */
/* Permission is hereby granted, free of charge, to any person obtaining  */
/* a copy of this software and associated documentation files (the        */
/* "Software"), to deal in the Software without restriction, including    */
/* without limitation the rights to use, copy, modify, merge, publish,    */
/* distribute, sublicense, and/or sell copies of the Software, and to     */
/* permit persons to whom the Software is furnished to do so, subject to  */
/* the following conditions:                                              */
/*                                                                        */
/* The above copyright notice and this permission notice shall be         */
/* included in all copies or substantial portions of the Software.        */
/*                                                                        */
/* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,        */
/* EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF     */
/* MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. */
/* IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY   */
/* CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,   */
/* TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE      */
/* SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.                 */
/**************************************************************************/

#include "denoise_wrapper.h"
#include "core/os/memory.h"
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
