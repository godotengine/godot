/**************************************************************************/
/*  metal_device_profile.cpp                                              */
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

#include "metal_device_profile.h"

#include "metal_utils.h"

Mutex MetalDeviceProfile::profiles_lock;
HashMap<MetalDeviceProfile::ProfileKey, MetalDeviceProfile> MetalDeviceProfile::profiles;

const MetalDeviceProfile *MetalDeviceProfile::get_profile(Platform p_platform, GPU p_gpu, MinOsVersion p_min_os_version) {
	DEV_ASSERT(p_platform == Platform::macOS || p_platform == Platform::iOS || p_platform == Platform::visionOS);

	MutexLock lock(profiles_lock);

	ProfileKey key(p_min_os_version, p_platform, p_gpu);
	if (MetalDeviceProfile *profile = profiles.getptr(key)) {
		return profile;
	}

	MetalDeviceProfile res;
	res.platform = p_platform;
	res.gpu = p_gpu;
	res.min_os_version = p_min_os_version;

	switch (p_platform) {
		case Platform::macOS: {
			if (p_min_os_version >= os_version::MACOS_26_0) {
				res.features.msl_version = MSL_VERSION_40;
			} else if (p_min_os_version >= os_version::MACOS_15_0) {
				res.features.msl_version = MSL_VERSION_32;
			} else if (p_min_os_version >= os_version::MACOS_14_0) {
				res.features.msl_version = MSL_VERSION_31;
			} else if (p_min_os_version >= os_version::MACOS_13_0) {
				res.features.msl_version = MSL_VERSION_30;
			} else if (p_min_os_version >= os_version::MACOS_12_0) {
				res.features.msl_version = MSL_VERSION_24;
			} else {
				res.features.msl_version = MSL_VERSION_23;
			}
			res.features.use_argument_buffers = p_min_os_version >= os_version::MACOS_13_0;
			res.features.simdPermute = true;
		} break;

		case Platform::iOS: {
			if (p_min_os_version >= os_version::IOS_26_0) {
				res.features.msl_version = MSL_VERSION_40;
			} else if (p_min_os_version >= os_version::IOS_18_0) {
				res.features.msl_version = MSL_VERSION_32;
			} else if (p_min_os_version >= os_version::IOS_17_0) {
				res.features.msl_version = MSL_VERSION_31;
			} else if (p_min_os_version >= os_version::IOS_16_0) {
				res.features.msl_version = MSL_VERSION_30;
			} else if (p_min_os_version >= os_version::IOS_15_0) {
				res.features.msl_version = MSL_VERSION_24;
			} else {
				res.features.msl_version = MSL_VERSION_23;
			}

			switch (p_gpu) {
				case GPU::Apple1:
				case GPU::Apple2:
				case GPU::Apple3:
				case GPU::Apple4:
				case GPU::Apple5: {
					res.features.simdPermute = false;
					res.features.use_argument_buffers = false;
				} break;
				case GPU::Apple6:
				case GPU::Apple7:
				case GPU::Apple8:
				case GPU::Apple9: {
					res.features.use_argument_buffers = p_min_os_version >= os_version::IOS_16_0;
					res.features.simdPermute = true;
				} break;
			}
		} break;

		case Platform::visionOS: {
			if (p_min_os_version >= os_version::VISIONOS_26_0) {
				res.features.msl_version = MSL_VERSION_40;
			} else if (p_min_os_version >= os_version::VISIONOS_02_4) {
				res.features.msl_version = MSL_VERSION_32;
			} else {
				ERR_FAIL_V_MSG(nullptr, "visionOS 2.4 is the minimum supported version for visionOS.");
			}

			switch (p_gpu) {
				case GPU::Apple8:
				case GPU::Apple9: {
					res.features.use_argument_buffers = true;
					res.features.simdPermute = true;
				} break;
				default: {
					CRASH_NOW_MSG("visionOS hardware has a minimum Apple8 GPU.");
				}
			}
		} break;
	}

	return &profiles.insert(key, res)->value;
}
