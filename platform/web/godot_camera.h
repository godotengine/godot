/**************************************************************************/
/*  godot_camera.h                                                        */
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

#include <emscripten.h>
#include <cstdint>

#ifdef __cplusplus
extern "C" {
#endif

using CameraLibrary_OnGetCamerasCallback = void (*)(void *, const char *);

using CameraLibrary_OnGetCapabilitiesCallback = void (*)(void *, const char *);

using CameraLibrary_OnGetPixelDataCallback = void (*)(void *, const uint8_t *, const int, const int, const int, const char *);

using CameraLibrary_OnDeniedCallback = void (*)(void *);

extern void godot_js_camera_get_pixel_data(
		void *context,
		const char *p_device_id_ptr,
		const int width,
		const int height,
		CameraLibrary_OnGetPixelDataCallback p_callback_ptr,
		CameraLibrary_OnDeniedCallback p_denied_callback_ptr);

extern void godot_js_camera_stop_stream(const char *p_device_id_ptr = nullptr);

#ifdef __cplusplus
}
#endif
