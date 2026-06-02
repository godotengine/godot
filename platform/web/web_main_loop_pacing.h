/**************************************************************************/
/*  web_main_loop_pacing.h                                                */
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

#include "display_server_web.h"
#include "os_web.h"

#include "core/config/engine.h"
#include "main/main.h"

static inline bool web_main_loop_should_skip_frame(OS_Web *p_os, uint64_t &r_target_ticks) {
#ifndef PROXY_TO_PTHREAD_ENABLED
	uint64_t current_ticks = p_os->get_ticks_usec();
#endif

	bool force_draw = DisplayServerWeb::get_singleton()->check_size_force_redraw();
	if (force_draw) {
		Main::force_redraw();
#ifndef PROXY_TO_PTHREAD_ENABLED
	} else if (current_ticks < r_target_ticks) {
		return true; // Skip frame.
#endif
	}

#ifndef PROXY_TO_PTHREAD_ENABLED
	int max_fps = Engine::get_singleton()->get_max_fps();
	if (max_fps > 0) {
		if (current_ticks - r_target_ticks > 1000000) {
			// When the window loses focus, we stop getting updates and accumulate delay.
			// For this reason, if the difference is too big, we reset target ticks to the current ticks.
			r_target_ticks = current_ticks;
		}
		r_target_ticks += (uint64_t)(1000000 / max_fps);
	}
#endif

	return false;
}
