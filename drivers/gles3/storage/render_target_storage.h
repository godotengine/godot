/*************************************************************************/
/*  render_target_storage.h                                              */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2022 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2022 Godot Engine contributors (cf. AUTHORS.md).   */
/*                                                                       */
/* Permission is hereby granted, free of charge, to any person obtaining */
/* a copy of this software and associated documentation files (the       */
/* "Software"), to deal in the Software without restriction, including   */
/* without limitation the rights to use, copy, modify, merge, publish,   */
/* distribute, sublicense, and/or sell copies of the Software, and to    */
/* permit persons to whom the Software is furnished to do so, subject to */
/* the following conditions:                                             */
/*                                                                       */
/* The above copyright notice and this permission notice shall be        */
/* included in all copies or substantial portions of the Software.       */
/*                                                                       */
/* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,       */
/* EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF    */
/* MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.*/
/* IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY  */
/* CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,  */
/* TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE     */
/* SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.                */
/*************************************************************************/

#ifndef RENDER_TARGET_STORAGE_GLES3_H
#define RENDER_TARGET_STORAGE_GLES3_H

#ifdef GLES3_ENABLED

#include "core/templates/rid_owner.h"
#include "servers/rendering/renderer_compositor.h"
#include "servers/rendering/renderer_storage.h" // included until we move stuff into storage/render_target_storage.h
// #include "servers/rendering/storage/render_target_storage.h"

// This must come first to avoid windows.h mess
#include "platform_config.h"
#ifndef OPENGL_INCLUDE_H
#include <GLES3/gl3.h>
#else
#include OPENGL_INCLUDE_H
#endif

namespace GLES3 {

// NOTE, this class currently is just a container for the the RenderTarget struct and is not yet implemented further, we'll do that next after we finish with TextureStorage

struct RenderTarget {
	RID self;
	GLuint fbo = 0;
	GLuint color = 0;
	GLuint depth = 0;

	GLuint multisample_fbo = 0;
	GLuint multisample_color = 0;
	GLuint multisample_depth = 0;
	bool multisample_active = false;

	struct Effect {
		GLuint fbo = 0;
		int width = 0;
		int height = 0;

		GLuint color = 0;
	};

	Effect copy_screen_effect;

	struct MipMaps {
		struct Size {
			GLuint fbo = 0;
			GLuint color = 0;
			int width = 0;
			int height = 0;
		};

		Vector<Size> sizes;
		GLuint color = 0;
		int levels = 0;
	};

	MipMaps mip_maps[2];

	struct External {
		GLuint fbo = 0;
		GLuint color = 0;
		GLuint depth = 0;
		RID texture;
	} external;

	int x = 0;
	int y = 0;
	int width = 0;
	int height = 0;

	bool flags[RendererStorage::RENDER_TARGET_FLAG_MAX] = {};

	// instead of allocating sized render targets immediately,
	// defer this for faster startup
	bool allocate_is_dirty = false;
	bool used_in_frame = false;
	RS::ViewportMSAA msaa = RS::VIEWPORT_MSAA_DISABLED;

	bool use_fxaa = false;
	bool use_debanding = false;

	RID texture;

	bool used_dof_blur_near = false;
	bool mip_maps_allocated = false;

	Color clear_color = Color(1, 1, 1, 1);
	bool clear_requested = false;

	RenderTarget() {
		for (int i = 0; i < RendererStorage::RENDER_TARGET_FLAG_MAX; ++i) {
			flags[i] = false;
		}
		external.fbo = 0;
	}
};

} // namespace GLES3

#endif // !GLES3_ENABLED

#endif // !RENDER_TARGET_STORAGE_GLES3_H
