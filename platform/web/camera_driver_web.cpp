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

#include "godot_camera.h"

#include "core/io/json.h"

namespace {
const String KEY_CAMERAS("cameras");
const String KEY_ERROR("error");
const String KEY_ID("id");
const String KEY_LABEL("label");
} // namespace

CameraDriverWeb *CameraDriverWeb::singleton = nullptr;

CameraDriverWeb *CameraDriverWeb::get_singleton() {
	return singleton;
}

void CameraDriverWeb::_on_get_cameras_callback(void *context, void *callback, const char *json_ptr) {
	// Always call the user callback, even on error, so callers can finish the
	// pending camera list update regardless of outcome.
	CameraDriverWebGetCamerasCallback on_get_cameras_callback =
			reinterpret_cast<CameraDriverWebGetCamerasCallback>(callback);

	if (!json_ptr) {
		ERR_PRINT("CameraDriverWeb::_on_get_cameras_callback: json_ptr is null.");
		on_get_cameras_callback(context, Vector<CameraInfo>());
		return;
	}
	String json_string = String::utf8(json_ptr);
	Variant json_variant = JSON::parse_string(json_string);

	if (json_variant.get_type() != Variant::DICTIONARY) {
		ERR_PRINT("CameraDriverWeb::_on_get_cameras_callback: Failed to parse JSON response or response is not a Dictionary.");
		on_get_cameras_callback(context, Vector<CameraInfo>());
		return;
	}

	Dictionary json_dict = json_variant;
	Variant v_error = json_dict[KEY_ERROR];
	if (v_error.get_type() == Variant::STRING) {
		String error_str = v_error;
		ERR_PRINT(vformat("Camera error from JS: %s", error_str));
		on_get_cameras_callback(context, Vector<CameraInfo>());
		return;
	}

	Variant v_devices = json_dict.get(KEY_CAMERAS, Variant());
	if (v_devices.get_type() != Variant::ARRAY) {
		ERR_PRINT("Camera error: 'cameras' is not an array or missing.");
		on_get_cameras_callback(context, Vector<CameraInfo>());
		return;
	}

	Array devices_array = v_devices;
	Vector<CameraInfo> camera_info;
	for (Variant device_variant : devices_array) {
		if (device_variant.get_type() != Variant::DICTIONARY) {
			continue;
		}

		Dictionary device_dict = device_variant;
		Variant id_variant = device_dict.get(KEY_ID, Variant());
		Variant label_variant = device_dict.get(KEY_LABEL, Variant());
		if (id_variant.get_type() != Variant::STRING || label_variant.get_type() != Variant::STRING) {
			WARN_PRINT("Camera info entry missing required keys (id, label).");
			continue;
		}

		CameraInfo info;
		info.device_id = id_variant;
		info.label = label_variant;

		camera_info.push_back(info);
	}

	on_get_cameras_callback(context, camera_info);
}

void CameraDriverWeb::get_cameras(void *p_context, CameraDriverWebGetCamerasCallback p_callback) {
	godot_js_camera_get_cameras(p_context, (void *)p_callback, &_on_get_cameras_callback);
}

void CameraDriverWeb::get_pixel_data(void *p_context, const String &p_device_id, const int p_width, const int p_height, void (*p_callback)(void *, const uint8_t *, const int, const int, const int, const int, const int, const char *), void (*p_denied_callback)(void *), CameraDriverWebFormatsCallback p_formats_callback) {
	godot_js_camera_get_pixel_data(p_context, p_device_id.utf8().get_data(), p_width, p_height, p_callback, p_denied_callback, p_formats_callback);
}

void CameraDriverWeb::stop_stream(const String &p_device_id) {
	godot_js_camera_stop_stream(p_device_id.utf8().get_data());
}

void CameraDriverWeb::abort_stream(void *p_context) {
	godot_js_camera_abort(p_context);
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
