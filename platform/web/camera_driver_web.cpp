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

namespace {
const String KEY_CAMERAS("cameras");
const String KEY_ERROR("error");
const String KEY_FORMATS("formats");
const String KEY_HEIGHT("height");
const String KEY_ID("id");
const String KEY_INDEX("index");
const String KEY_LABEL("label");
const String KEY_WIDTH("width");
} //namespace

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

// Helper to extract integer value from Variant.
int CameraDriverWeb::_get_int_value(const Variant &p_val) {
	if (p_val.get_type() == Variant::INT) {
		return p_val;
	} else if (p_val.get_type() == Variant::FLOAT) {
		return static_cast<int>(p_val.operator float());
	}
	return 0;
}

void CameraDriverWeb::_on_get_cameras_callback(void *context, void *callback, const char *json_ptr) {
	if (!json_ptr) {
		ERR_PRINT("CameraDriverWeb::_on_get_cameras_callback: json_ptr is null.");
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

		// Parse formats array.
		Variant v_formats = device_dict.get(KEY_FORMATS, Variant());
		if (v_formats.get_type() == Variant::ARRAY) {
			Array formats_array = v_formats;
			for (int j = 0; j < formats_array.size(); j++) {
				Variant format_variant = formats_array.get(j);
				if (format_variant.get_type() != Variant::DICTIONARY) {
					continue;
				}

				Dictionary format_dict = format_variant;
				if (!format_dict.has(KEY_WIDTH) || !format_dict.has(KEY_HEIGHT)) {
					continue;
				}

				int width = _get_int_value(format_dict.get(KEY_WIDTH, Variant()));
				int height = _get_int_value(format_dict.get(KEY_HEIGHT, Variant()));

				if (width > 0 && height > 0) {
					FormatInfo format_info;
					format_info.width = width;
					format_info.height = height;
					info.formats.push_back(format_info);
				}
			}
		}

		if (info.formats.is_empty()) {
			WARN_PRINT("Camera info entry has no valid formats.");
		}

		camera_info.push_back(info);
	}

	CameraDriverWebGetCamerasCallback on_get_cameras_callback = reinterpret_cast<CameraDriverWebGetCamerasCallback>(callback);
	on_get_cameras_callback(context, camera_info);
}

void CameraDriverWeb::get_cameras(void *p_context, CameraDriverWebGetCamerasCallback p_callback) {
	godot_js_camera_get_cameras(p_context, (void *)p_callback, &_on_get_cameras_callback);
}

void CameraDriverWeb::get_pixel_data(void *p_context, const String &p_device_id, const int p_width, const int p_height, void (*p_callback)(void *, const uint8_t *, const int, const int, const int, const int, const int, const char *), void (*p_denied_callback)(void *)) {
	godot_js_camera_get_pixel_data(p_context, p_device_id.utf8().get_data(), p_width, p_height, p_callback, p_denied_callback);
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
