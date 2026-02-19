/**************************************************************************/
/*  rendering_server_types.h                                              */
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

#include "core/math/rect2.h"
#include "core/math/rect2i.h"
#include "core/math/vector2.h"
#include "core/string/ustring.h"
#include "core/templates/rid.h"
#include "core/templates/vector.h"
#include "servers/rendering/rendering_server_enums.h"

#include <cstdint>

namespace RenderingServerTypes {

/* SHADER API */

struct ShaderNativeSourceCode {
	struct Version {
		struct Stage {
			String name;
			String code;
		};
		Vector<Stage> stages;
	};
	Vector<Version> versions;
};

/* STATUS INFORMATION */

struct FrameProfileArea {
	String name;
	double gpu_msec;
	double cpu_msec;
};

/* COMPOSITOR */

struct BlitToScreen {
	RID render_target;
	Rect2 src_rect = Rect2(0.0, 0.0, 1.0, 1.0);
	Rect2i dst_rect;

	struct {
		bool use_layer = false;
		uint32_t layer = 0;
	} multi_view;

	struct {
		//lens distorted parameters for VR
		bool apply = false;
		Vector2 eye_center;
		float k1 = 0.0;
		float k2 = 0.0;

		float upscale = 1.0;
		float aspect_ratio = 1.0;
	} lens_distortion;
};

/* BACKGROUND */

// Helper for RSE::SplashStretchMode, put here for convenience.
inline Rect2 get_splash_stretched_screen_rect(const Size2 &p_image_size, const Size2 &p_window_size, RSE::SplashStretchMode p_stretch_mode) {
	Size2 imgsize = p_image_size;
	Rect2 screenrect;
	switch (p_stretch_mode) {
		case RSE::SPLASH_STRETCH_MODE_DISABLED: {
			screenrect.size = imgsize;
			screenrect.position = ((p_window_size - screenrect.size) / 2.0).floor();
		} break;
		case RSE::SPLASH_STRETCH_MODE_KEEP: {
			if (p_window_size.width > p_window_size.height) {
				// Scale horizontally.
				screenrect.size.y = p_window_size.height;
				screenrect.size.x = imgsize.width * p_window_size.height / imgsize.height;
				screenrect.position.x = (p_window_size.width - screenrect.size.x) / 2;
			} else {
				// Scale vertically.
				screenrect.size.x = p_window_size.width;
				screenrect.size.y = imgsize.height * p_window_size.width / imgsize.width;
				screenrect.position.y = (p_window_size.height - screenrect.size.y) / 2;
			}
		} break;
		case RSE::SPLASH_STRETCH_MODE_KEEP_WIDTH: {
			// Scale vertically.
			screenrect.size.x = p_window_size.width;
			screenrect.size.y = imgsize.height * p_window_size.width / imgsize.width;
			screenrect.position.y = (p_window_size.height - screenrect.size.y) / 2;
		} break;
		case RSE::SPLASH_STRETCH_MODE_KEEP_HEIGHT: {
			// Scale horizontally.
			screenrect.size.y = p_window_size.height;
			screenrect.size.x = imgsize.width * p_window_size.height / imgsize.height;
			screenrect.position.x = (p_window_size.width - screenrect.size.x) / 2;
		} break;
		case RSE::SPLASH_STRETCH_MODE_COVER: {
			double window_aspect = (double)p_window_size.width / p_window_size.height;
			double img_aspect = imgsize.width / imgsize.height;

			if (window_aspect > img_aspect) {
				// Scale vertically.
				screenrect.size.x = p_window_size.width;
				screenrect.size.y = imgsize.height * p_window_size.width / imgsize.width;
				screenrect.position.y = (p_window_size.height - screenrect.size.y) / 2;
			} else {
				// Scale horizontally.
				screenrect.size.y = p_window_size.height;
				screenrect.size.x = imgsize.width * p_window_size.height / imgsize.height;
				screenrect.position.x = (p_window_size.width - screenrect.size.x) / 2;
			}
		} break;
		case RSE::SPLASH_STRETCH_MODE_IGNORE: {
			screenrect.size.x = p_window_size.width;
			screenrect.size.y = p_window_size.height;
		} break;
	}
	return screenrect;
}

} // namespace RenderingServerTypes
