/**************************************************************************/
/*  bridge_openharmony.h                                                  */
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

#include <rawfile/raw_file_manager.h>
#include <cstdint>

extern "C" {
typedef struct GodotTouchEvent {
	uint32_t type;
	uint32_t id;
	float x;
	float y;
} GodotTouchEvent;

typedef struct GodotKeyEvent {
	uint32_t code;
	char32_t unicode;
	bool pressed;
	bool alt;
	bool ctrl;
	bool shift;
	bool meta;
} GodotKeyEvent;

typedef struct GodotMouseEvent {
	uint32_t type;
	uint32_t button;
	uint32_t mask;
	float x;
	float y;
} GodotMouseEvent;

int64_t godot_init(NativeResourceManager *p_resource_manager, void *p_native_window, int32_t window_id, int64_t window_width, int64_t window_height, const char *p_allowed_permissions);
void godot_touch(GodotTouchEvent *p_event, int count);
void godot_mouse(GodotMouseEvent *p_event);
void godot_key(GodotKeyEvent *p_event);
void godot_resize(uint32_t width, uint32_t height);
void godot_window_event(int32_t event);
}
