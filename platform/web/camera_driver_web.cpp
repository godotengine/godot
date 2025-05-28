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

EM_ASYNC_JS(void, godot_js_camera_get_cameras, (void *context, CameraLibrary_OnGetCamerasCallback p_callback_ptr), {
	await GodotCamera.api.getCameras(context, p_callback_ptr);
});

EM_ASYNC_JS(void, godot_js_camera_get_capabilities, (void *context, const char *p_device_id_ptr, CameraLibrary_OnGetCapabilitiesCallback p_callback_ptr), {
	await GodotCamera.api.getCameraCapabilities(p_device_id_ptr, context, p_callback_ptr);
});

CameraDriverWeb *CameraDriverWeb::singleton = nullptr;
Array CameraDriverWeb::_camera_info_key;

CameraDriverWeb *CameraDriverWeb::get_singleton() {
	_camera_info_key.clear();
	_camera_info_key.push_back(KEY_INDEX);
	_camera_info_key.push_back(KEY_ID);
	_camera_info_key.push_back(KEY_LABEL);
	return singleton;
}

void CameraDriverWeb::_on_get_cameras_callback(void *context, const char *json_ptr) {
	if (!json_ptr) {
		print_error("CameraDriverWeb::_on_get_cameras_callback: json_ptr is null");
		return;
	}
	String json_string = String::utf8(json_ptr);
	Variant json_variant = JSON::parse_string(json_string);

	if (json_variant.get_type() == Variant::DICTIONARY) {
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
		Vector<CameraInfo> *camera_info = reinterpret_cast<Vector<CameraInfo> *>(context);
		camera_info->clear();
		for (int i = 0; i < devices_array.size(); i++) {
			Variant device_variant = devices_array.get(i);
			if (device_variant.get_type() == Variant::DICTIONARY) {
				Dictionary device_dict = device_variant;
				if (device_dict.has_all(_camera_info_key)) {
					CameraInfo info;
					info.index = device_dict[KEY_INDEX];
					info.device_id = device_dict[KEY_ID];
					info.label = device_dict[KEY_LABEL];
					camera_info->push_back(info);
				} else {
					WARN_PRINT("Camera info entry missing required keys (index, id, label).");
				}
			}
		}
	} else {
		ERR_PRINT("CameraDriverWeb::_on_get_cameras_callback: Failed to parse JSON response or response is not a Dictionary.");
	}
}

void CameraDriverWeb::_on_get_capabilities_callback(void *context, const char *json_ptr) {
	if (!json_ptr) {
		ERR_PRINT("CameraDriverWeb::_on_get_capabilities_callback: json_ptr is null");
		return;
	}
	String json_string = String::utf8(json_ptr);
	Variant json_variant = JSON::parse_string(json_string);

	if (json_variant.get_type() == Variant::DICTIONARY) {
		Dictionary json_dict = json_variant;
		Variant v_error = json_dict[KEY_ERROR];
		if (v_error.get_type() == Variant::STRING) {
			String error_str = v_error;
			ERR_PRINT(vformat("Camera capabilities error from JS: %s", error_str));
			return;
		}
		Variant v_caps_data = json_dict.get(KEY_CAPABILITIES, Variant());
		if (v_caps_data.get_type() != Variant::DICTIONARY) {
			ERR_PRINT("Camera capabilities error: 'capabilities' data is not a dictionary or missing.");
			return;
		}
		Dictionary caps_dict = v_caps_data;
		Vector<CapabilityInfo> *capabilities = reinterpret_cast<Vector<CapabilityInfo> *>(context);
		capabilities->clear();

		if (caps_dict.has(KEY_WIDTH) && caps_dict.has(KEY_HEIGHT)) {
			Variant v_width_val = caps_dict.get(KEY_WIDTH, Variant());
			Variant v_height_val = caps_dict.get(KEY_HEIGHT, Variant());
			int width = 0;
			int height = 0;

			// Helper to extract 'max' from a capability dictionary or use direct value
			auto get_max_or_direct = [](const Variant &p_val) -> int {
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
			};

			width = get_max_or_direct(v_width_val);
			height = get_max_or_direct(v_height_val);

			if (width > 0 && height > 0) {
				CapabilityInfo info;
				info.width = width;
				info.height = height;
				capabilities->push_back(info);
			} else {
				WARN_PRINT("Could not extract valid width/height from capabilities structure.");
			}
		} else {
			WARN_PRINT("Capabilities object does not directly contain top-level width/height keys.");
		}
	} else {
		ERR_PRINT("CameraDriverWeb::_on_get_capabilities_callback: Failed to parse JSON response or response is not a Dictionary.");
	}
}

void CameraDriverWeb::get_cameras(Vector<CameraInfo> *r_camera_info) {
	godot_js_camera_get_cameras((void *)r_camera_info, &_on_get_cameras_callback);
}

void CameraDriverWeb::get_capabilities(Vector<CapabilityInfo> *r_capabilities, const String &p_device_id) {
	godot_js_camera_get_capabilities((void *)r_capabilities, p_device_id.utf8().get_data(), &_on_get_capabilities_callback);
}

void CameraDriverWeb::get_pixel_data(void *context, const String &p_device_id, const int width, const int height, CameraLibrary_OnGetPixelDataCallback p_callback, CameraLibrary_OnDeniedCallback p_denied_callback) {
	godot_js_camera_get_pixel_data(context, p_device_id.utf8().get_data(), width, height, p_callback, p_denied_callback);
}

void CameraDriverWeb::stop_stream(const String &device_id) {
	godot_js_camera_stop_stream(device_id.utf8().get_data());
}

CameraDriverWeb::CameraDriverWeb() {
	if (singleton == nullptr) {
		singleton = this;
	}
}

CameraDriverWeb::~CameraDriverWeb() {
	if (singleton == this) {
		singleton = nullptr;
	}
}
