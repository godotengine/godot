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

struct FormatInfo {
	int width;
	int height;
};

struct CameraInfo {
	int index;
	String device_id;
	String label;
	Vector<FormatInfo> formats;
};

using CameraDriverWebGetCamerasCallback = void (*)(void *p_context, const Vector<CameraInfo> &p_camera_info);

class CameraDriverWeb {
private:
	static CameraDriverWeb *singleton;
	static Array _camera_info_key;
	static int _get_int_value(const Variant &p_val);
	WASM_EXPORT static void _on_get_cameras_callback(void *context, void *callback, const char *json_ptr);

public:
	static CameraDriverWeb *get_singleton();
	void get_cameras(void *p_context, CameraDriverWebGetCamerasCallback p_callback);
	void get_pixel_data(void *p_context, const String &p_device_id, const int p_width, const int p_height, void (*p_callback)(void *, const uint8_t *, const int, const int, const int, const int, const int, const char *), void (*p_denied_callback)(void *));
	void stop_stream(const String &device_id = String());

	CameraDriverWeb();
	~CameraDriverWeb();
};
