/**************************************************************************/
/*  wrapper_openharmony.h                                                 */
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

#include <cstddef>
#include <cstdint>

enum WrapperScreenOrientation {
	WRAPPER_SCREEN_LANDSCAPE,
	WRAPPER_SCREEN_PORTRAIT,
	WRAPPER_SCREEN_REVERSE_LANDSCAPE,
	WRAPPER_SCREEN_REVERSE_PORTRAIT,
};

int ohos_wrapper_get_display_dpi();
float ohos_wrapper_get_display_scaled_density();
float ohos_wrapper_get_display_refresh_rate();
WrapperScreenOrientation ohos_wrapper_get_display_orientation();
void ohos_wrapper_screen_set_keep_on(int32_t window_id, bool p_enable);
bool ohos_wrapper_screen_is_kept_on(int32_t window_id);
int ohos_wrapper_get_keyboard_avoid_area(int32_t window_id);
