/*************************************************************************/
/*  lightmap_denoiser.cpp                                                */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2020 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2020 Godot Engine contributors (cf. AUTHORS.md).   */
/*                                                                       */
/* Permission is hereby granted, free of charge, to any person obtaining */
/* a copy of this software and associated documentation files (the       */
/* "Software"), to deal in the Software without restriction, including   */
/* without limitation the rights to use, copy, modify, merge, publish,   */
/* distribute, sublicense, and/or sell copies of the Software, and to    */
/* permit persons to whom the Software is furnished to do so, subject to */
/* the following conditions:                                             */
/*                                                                       */
/* The above copyright notice and this permission notice shall be        */
/* included in all copies or substantial portions of the Software.       */
/*                                                                       */
/* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,       */
/* EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF    */
/* MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.*/
/* IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY  */
/* CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,  */
/* TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE     */
/* SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.                */
/*************************************************************************/

#include "lightmap_denoiser.h"
#include "include/OpenImageDenoise/oidn.h"

LightmapDenoiser *LightmapDenoiser::singleton = NULL;

LightmapDenoiser *LightmapDenoiser::get_singleton() {
	return singleton;
}

void LightmapDenoiser::init() {
	if (!device) {
		device = oidnNewDevice(OIDN_DEVICE_TYPE_CPU);
		oidnCommitDevice(device);
	}
}

bool LightmapDenoiser::denoise_lightmap(float *r_lightmap, const Vector2i &p_size) {

	OIDNFilter filter = oidnNewFilter(device, "RTLightmap");
	oidnSetSharedFilterImage(filter, "color", (void *)r_lightmap, OIDN_FORMAT_FLOAT3, p_size.x, p_size.y, 0, 0, 0);
	oidnSetSharedFilterImage(filter, "output", (void *)r_lightmap, OIDN_FORMAT_FLOAT3, p_size.x, p_size.y, 0, 0, 0);
	oidnSetFilter1b(filter, "hdr", true);
	oidnCommitFilter(filter);

	oidnExecuteFilter(filter);

	const char *msg;
	bool success = true;
	if (oidnGetDeviceError(device, &msg) != OIDN_ERROR_NONE) {
		print_error("LightmapDenoiser: " + String(msg));
		success = false;
	}

	oidnReleaseFilter(filter);
	return success;
}

void LightmapDenoiser::free() {
	oidnReleaseDevice(device);
	device = NULL;
}

LightmapDenoiser::LightmapDenoiser() {
	singleton = this;
}
