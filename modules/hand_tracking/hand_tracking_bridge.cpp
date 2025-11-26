/**************************************************************************/
/*  hand_tracking_bridge.cpp                                              */
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

#include "hand_tracking_bridge.h"

#include "core/os/mutex.h"

// Global storage for the latest hand tracking frame
static godot_hand_frame g_latest_frame;
static Mutex g_frame_mutex;
static bool g_frame_available = false;

// Called from Swift/visionOS platform layer
extern "C" void godot_visionos_set_hand_frame(const godot_hand_frame *frame) {
	if (!frame) {
		return;
	}

	MutexLock lock(g_frame_mutex);
	g_latest_frame = *frame;
	g_frame_available = true;
}

// Internal C++ API for retrieving the latest frame
bool hand_tracking_get_latest_frame(godot_hand_frame &out_frame) {
	MutexLock lock(g_frame_mutex);

	if (!g_frame_available) {
		return false;
	}

	out_frame = g_latest_frame;
	return true;
}

// Check if hand tracking data is available
bool hand_tracking_is_available() {
	MutexLock lock(g_frame_mutex);
	return g_frame_available;
}

// Reset/clear hand tracking data
void hand_tracking_clear() {
	MutexLock lock(g_frame_mutex);
	g_frame_available = false;
	// Zero out the frame data
	memset(&g_latest_frame, 0, sizeof(godot_hand_frame));
}
