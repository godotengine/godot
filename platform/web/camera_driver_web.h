/**************************************************************************/
/*  camera_driver_web.h                                                   */
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

#pragma once

#include "godot_camera.h"
#include "godot_js.h"

#include "core/string/ustring.h"
#include "core/templates/vector.h"
#include "core/variant/array.h"

#define KEY_CAMERAS String("cameras")
#define KEY_CAPABILITIES String("capabilities")
#define KEY_ERROR String("error")
#define KEY_HEIGHT String("height")
#define KEY_ID String("id")
#define KEY_INDEX String("index")
#define KEY_LABEL String("label")
#define KEY_MAX String("max")
#define KEY_WIDTH String("width")

struct CapabilityInfo {
	int width;
	int height;
};

struct CameraInfo {
	int index;
	String device_id;
	String label;
	CapabilityInfo capability;
};

using CameraDriverWeb_OnGetCamerasCallback = void (*)(void *context, const Vector<CameraInfo> &camera_info);

class CameraDriverWeb {
private:
	static CameraDriverWeb *singleton;
	static Array _camera_info_key;
	static int _get_max_or_direct(const Variant &p_val);
	WASM_EXPORT static void _on_get_cameras_callback(void *context, void *callback, const char *json_ptr);

public:
	static CameraDriverWeb *get_singleton();
	void get_cameras(void *context, CameraDriverWeb_OnGetCamerasCallback callback);
	void get_pixel_data(void *context, const String &p_device_id, const int width, const int height, CameraLibrary_OnGetPixelDataCallback p_callback, CameraLibrary_OnDeniedCallback p_denied_callback);
	void stop_stream(const String &device_id = String());

	CameraDriverWeb();
	~CameraDriverWeb();
};
