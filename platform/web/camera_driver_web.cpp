/**************************************************************************/
/*  camera_driver_web.cpp                                                 */
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

#include "camera_driver_web.h"

#include "core/io/json.h"

#include <cstdlib>

CameraDriverWeb *CameraDriverWeb::singleton = nullptr;
Array CameraDriverWeb::_camera_info_key;

CameraDriverWeb *CameraDriverWeb::get_singleton() {
	if (_camera_info_key.is_empty()) {
		_camera_info_key.push_back(KEY_INDEX);
		_camera_info_key.push_back(KEY_ID);
		_camera_info_key.push_back(KEY_LABEL);
	}
	return singleton;
}

// Helper to extract 'max' from a capability dictionary or use direct value
int CameraDriverWeb::_get_max_or_direct(const Variant &p_val) {
	if (p_val.get_type() == Variant::DICTIONARY) {
		Dictionary d = p_val;
		if (d.has(KEY_MAX)) {
			return d[KEY_MAX];
		}
	} else if (p_val.get_type() == Variant::INT) {
		return p_val;
	} else if (p_val.get_type() == Variant::FLOAT) {
		return static_cast<int>(p_val.operator float());
	}
	return 0;
}

void CameraDriverWeb::_on_get_cameras_callback(void *context, void *callback, const char *json_ptr) {
	if (!json_ptr) {
		ERR_PRINT("CameraDriverWeb::_on_get_cameras_callback: json_ptr is null");
		return;
	}
	String json_string = String::utf8(json_ptr);
	Variant json_variant = JSON::parse_string(json_string);

	if (json_variant.get_type() != Variant::DICTIONARY) {
		ERR_PRINT("CameraDriverWeb::_on_get_cameras_callback: Failed to parse JSON response or response is not a Dictionary.");
		return;
	}

	Dictionary json_dict = json_variant;
	Variant v_error = json_dict[KEY_ERROR];
	if (v_error.get_type() == Variant::STRING) {
		String error_str = v_error;
		ERR_PRINT(vformat("Camera error from JS: %s", error_str));
		return;
	}

	Variant v_devices = json_dict.get(KEY_CAMERAS, Variant());
	if (v_devices.get_type() != Variant::ARRAY) {
		ERR_PRINT("Camera error: 'cameras' is not an array or missing.");
		return;
	}

	Array devices_array = v_devices;
	Vector<CameraInfo> camera_info;
	for (int i = 0; i < devices_array.size(); i++) {
		Variant device_variant = devices_array.get(i);
		if (device_variant.get_type() != Variant::DICTIONARY) {
			continue;
		}

		Dictionary device_dict = device_variant;
		if (!device_dict.has_all(_camera_info_key)) {
			WARN_PRINT("Camera info entry missing required keys (index, id, label).");
			continue;
		}

		CameraInfo info;
		info.index = device_dict[KEY_INDEX];
		info.device_id = device_dict[KEY_ID];
		info.label = device_dict[KEY_LABEL];
		// Initialize capability with safe defaults to avoid uninitialized usage downstream.
		{
			CapabilityInfo capability = {};
			capability.width = 0;
			capability.height = 0;
			info.capability = capability;
		}

		Variant v_caps_data = device_dict.get(KEY_CAPABILITIES, Variant());
		if (v_caps_data.get_type() != Variant::DICTIONARY) {
			WARN_PRINT("Camera info entry has no capabilities or capabilities are not a dictionary.");
			camera_info.push_back(info);
			continue;
		}

		Dictionary caps_dict = v_caps_data;
		if (!caps_dict.has(KEY_WIDTH) || !caps_dict.has(KEY_HEIGHT)) {
			WARN_PRINT("Capabilities object does not directly contain top-level width/height keys.");
			camera_info.push_back(info);
			continue;
		}

		Variant v_width_val = caps_dict.get(KEY_WIDTH, Variant());
		Variant v_height_val = caps_dict.get(KEY_HEIGHT, Variant());

		int width = _get_max_or_direct(v_width_val);
		int height = _get_max_or_direct(v_height_val);

		if (width <= 0 || height <= 0) {
			WARN_PRINT("Could not extract valid width/height from capabilities structure.");
			// Still include the device in the list; keep zeroed capabilities.
			camera_info.push_back(info);
		} else {
			CapabilityInfo capability;
			capability.width = width;
			capability.height = height;
			info.capability = capability;
			camera_info.push_back(info);
		}
	}

	CameraDriverWeb_OnGetCamerasCallback on_get_cameras_callback = reinterpret_cast<CameraDriverWeb_OnGetCamerasCallback>(callback);
	on_get_cameras_callback(context, camera_info);
}

void CameraDriverWeb::get_cameras(void *context, CameraDriverWeb_OnGetCamerasCallback callback) {
	godot_js_camera_get_cameras(context, (void *)callback, &_on_get_cameras_callback);
}

void CameraDriverWeb::get_pixel_data(void *context, const String &p_device_id, const int width, const int height, CameraLibrary_OnGetPixelDataCallback p_callback, CameraLibrary_OnDeniedCallback p_denied_callback) {
	godot_js_camera_get_pixel_data(context, p_device_id.utf8().get_data(), width, height, p_callback, p_denied_callback);
}

void CameraDriverWeb::stop_stream(const String &device_id) {
	godot_js_camera_stop_stream(device_id.utf8().get_data());
}

CameraDriverWeb::CameraDriverWeb() {
	ERR_FAIL_COND_MSG(singleton != nullptr, "CameraDriverWeb singleton already exists.");
	singleton = this;
}

CameraDriverWeb::~CameraDriverWeb() {
	if (singleton == this) {
		singleton = nullptr;
	}
}
