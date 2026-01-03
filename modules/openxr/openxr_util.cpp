/**************************************************************************/
/*  openxr_util.cpp                                                       */
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

#include "openxr_util.h"

#include "core/math/math_funcs.h"

#include <openxr/openxr_reflection.h>

String OpenXRUtil::get_result_string(XrResult p_result){
	XR_ENUM_SWITCH(XrResult, p_result)
}

String OpenXRUtil::get_view_configuration_name(XrViewConfigurationType p_view_configuration){
	XR_ENUM_SWITCH(XrViewConfigurationType, p_view_configuration)
}

String OpenXRUtil::get_reference_space_name(XrReferenceSpaceType p_reference_space){
	XR_ENUM_SWITCH(XrReferenceSpaceType, p_reference_space)
}

String OpenXRUtil::get_structure_type_name(XrStructureType p_structure_type){
	XR_ENUM_SWITCH(XrStructureType, p_structure_type)
}

String OpenXRUtil::get_session_state_name(XrSessionState p_session_state){
	XR_ENUM_SWITCH(XrSessionState, p_session_state)
}

String OpenXRUtil::get_action_type_name(XrActionType p_action_type){
	XR_ENUM_SWITCH(XrActionType, p_action_type)
}

String OpenXRUtil::get_environment_blend_mode_name(XrEnvironmentBlendMode p_blend_mode) {
	XR_ENUM_SWITCH(XrEnvironmentBlendMode, p_blend_mode);
}

String OpenXRUtil::make_xr_version_string(XrVersion p_version) {
	String version;

	version += String::num_int64(XR_VERSION_MAJOR(p_version));
	version += String(".");
	version += String::num_int64(XR_VERSION_MINOR(p_version));
	version += String(".");
	version += String::num_int64(XR_VERSION_PATCH(p_version));

	return version;
}

String OpenXRUtil::get_handle_as_hex_string(void *p_handle) {
	String hex;

	if (p_handle == XR_NULL_HANDLE) {
		return "null";
	}

	uint64_t handle = (uint64_t)p_handle;

	while (handle != 0) {
		uint8_t a = handle & 0x0F;
		uint8_t b = (handle & 0xF0) >> 4;
		handle = handle >> 8;

		if (a < 10) {
			hex = (a + '0') + hex;
		} else {
			hex = (a + 'a' - 10) + hex;
		}

		if (b < 10) {
			hex = (b + '0') + hex;
		} else {
			hex = (b + 'a' - 10) + hex;
		}
	}

	return "0x" + hex;
}

String OpenXRUtil::string_from_xruuid(const XrUuid &xr_uuid) {
	String ret;
	bool non_zero = false;

	for (int i = 0; i < XR_UUID_SIZE; i++) {
		non_zero |= xr_uuid.data[i] != 0;

		char a = xr_uuid.data[i] & 0xF0 >> 4;
		char b = xr_uuid.data[i] & 0x0F;

		if (a < 10) {
			ret += '0' + a;
		} else {
			ret += 'a' + a - 10;
		}

		if (b < 10) {
			ret += '0' + b;
		} else {
			ret += 'a' + b - 10;
		}
	}

	if (non_zero) {
		return ret;
	} else {
		return "";
	}
}

XrUuid OpenXRUtil::xruuid_from_string(const String &p_uuid) {
	XrUuid new_uuid = { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 };

	int len = p_uuid.length();
	if (len == 0) {
		return new_uuid;
	} else if (len != (2 * XR_UUID_SIZE)) {
		WARN_PRINT("OpenXR: Unexpected UUID length: " + String::num_int64(len) + " != " + String::num_int64(2 * XR_UUID_SIZE));
	}

	int j = 0;
	for (int i = 0; i < XR_UUID_SIZE; i++) {
		uint8_t val = 0;

		// 2 chars per byte.
		for (int k = 0; k < 2; k++) {
			if (j < len) {
				val <<= 4;

				char32_t c = p_uuid[j++];
				if (c >= '0' && c <= '9') {
					val += uint8_t(c - '0');
				} else if (c >= 'a' && c <= 'f') {
					val += uint8_t(10 + c - 'a');
				} else if (c >= 'A' && c <= 'F') {
					val += uint8_t(10 + c - 'A');
				} else {
					WARN_PRINT("OpenXR: Unexpected character in UUID: " + String::num_int64(c));
				}
			}
		}

		new_uuid.data[i] = val;
	}

	return new_uuid;
}

// Copied from OpenXR xr_linear.h private header, so we can still link against
// system-provided packages without relying on our `thirdparty` code.

// Copyright (c) 2017 The Khronos Group Inc.
// Copyright (c) 2016 Oculus VR, LLC.
//
// SPDX-License-Identifier: Apache-2.0

// Creates a projection matrix based on the specified dimensions.
// The projection matrix transforms -Z=forward, +Y=up, +X=right to the appropriate clip space for Godot (OpenGL convention).
// The far plane is placed at infinity if farZ <= nearZ.
// An infinite projection matrix is preferred for rasterization because, except for
// things *right* up against the near plane, it always provides better precision:
//              "Tightening the Precision of Perspective Rendering"
//              Paul Upchurch, Mathieu Desbrun
//              Journal of Graphics Tools, Volume 16, Issue 1, 2012
void OpenXRUtil::XrMatrix4x4f_CreateProjection(XrMatrix4x4f *result, const float tanAngleLeft, const float tanAngleRight,
		const float tanAngleUp, float const tanAngleDown, const float nearZ, const float farZ) {
	const float tanAngleWidth = tanAngleRight - tanAngleLeft;

	// Set to tanAngleUp - tanAngleDown for a clip space with positive Y up.
	const float tanAngleHeight = (tanAngleUp - tanAngleDown);

	// Set to nearZ for a [-1,1] Z clip space.
	const float offsetZ = nearZ;

	if (farZ <= nearZ) {
		// place the far plane at infinity
		result->m[0] = 2.0f / tanAngleWidth;
		result->m[4] = 0.0f;
		result->m[8] = (tanAngleRight + tanAngleLeft) / tanAngleWidth;
		result->m[12] = 0.0f;

		result->m[1] = 0.0f;
		result->m[5] = 2.0f / tanAngleHeight;
		result->m[9] = (tanAngleUp + tanAngleDown) / tanAngleHeight;
		result->m[13] = 0.0f;

		result->m[2] = 0.0f;
		result->m[6] = 0.0f;
		result->m[10] = -1.0f;
		result->m[14] = -(nearZ + offsetZ);

		result->m[3] = 0.0f;
		result->m[7] = 0.0f;
		result->m[11] = -1.0f;
		result->m[15] = 0.0f;
	} else {
		// normal projection
		result->m[0] = 2.0f / tanAngleWidth;
		result->m[4] = 0.0f;
		result->m[8] = (tanAngleRight + tanAngleLeft) / tanAngleWidth;
		result->m[12] = 0.0f;

		result->m[1] = 0.0f;
		result->m[5] = 2.0f / tanAngleHeight;
		result->m[9] = (tanAngleUp + tanAngleDown) / tanAngleHeight;
		result->m[13] = 0.0f;

		result->m[2] = 0.0f;
		result->m[6] = 0.0f;
		result->m[10] = -(farZ + offsetZ) / (farZ - nearZ);
		result->m[14] = -(farZ * (nearZ + offsetZ)) / (farZ - nearZ);

		result->m[3] = 0.0f;
		result->m[7] = 0.0f;
		result->m[11] = -1.0f;
		result->m[15] = 0.0f;
	}
}

// Creates a projection matrix based on the specified FOV.
void OpenXRUtil::XrMatrix4x4f_CreateProjectionFov(XrMatrix4x4f *result, const XrFovf fov, const float nearZ, const float farZ) {
	const float tanLeft = std::tan(fov.angleLeft);
	const float tanRight = std::tan(fov.angleRight);

	const float tanDown = std::tan(fov.angleDown);
	const float tanUp = std::tan(fov.angleUp);

	XrMatrix4x4f_CreateProjection(result, tanLeft, tanRight, tanUp, tanDown, nearZ, farZ);
}
