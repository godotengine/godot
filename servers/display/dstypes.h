/**************************************************************************/
/*  dstypes.h                                                             */
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

#include "core/input/input.h"

/// Constants for DisplayServer.
/// This file is the 'least common denominator' that can be included without
/// needing to include complex header hierarchies of the DisplayServer.
namespace DSTypes {

using WindowID = int;

enum {
	MAIN_WINDOW_ID = 0,
	INVALID_WINDOW_ID = -1,
	INVALID_INDICATOR_ID = -1
};

using IndicatorID = int;

// Keep the VSyncMode enum values in sync with the `display/window/vsync/vsync_mode`
// project setting hint.
enum VSyncMode {
	VSYNC_DISABLED,
	VSYNC_ENABLED,
	VSYNC_ADAPTIVE,
	VSYNC_MAILBOX
};

enum AccessibilityLiveMode {
	LIVE_OFF,
	LIVE_POLITE,
	LIVE_ASSERTIVE,
};

enum MouseMode {
	MOUSE_MODE_VISIBLE = Input::MOUSE_MODE_VISIBLE,
	MOUSE_MODE_HIDDEN = Input::MOUSE_MODE_HIDDEN,
	MOUSE_MODE_CAPTURED = Input::MOUSE_MODE_CAPTURED,
	MOUSE_MODE_CONFINED = Input::MOUSE_MODE_CONFINED,
	MOUSE_MODE_CONFINED_HIDDEN = Input::MOUSE_MODE_CONFINED_HIDDEN,
	MOUSE_MODE_MAX = Input::MOUSE_MODE_MAX,
};

enum WindowMode {
	WINDOW_MODE_WINDOWED,
	WINDOW_MODE_MINIMIZED,
	WINDOW_MODE_MAXIMIZED,
	WINDOW_MODE_FULLSCREEN,
	WINDOW_MODE_EXCLUSIVE_FULLSCREEN,
};

} //namespace DSTypes
