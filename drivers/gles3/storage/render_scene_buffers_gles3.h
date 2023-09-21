/**************************************************************************/
/*  render_scene_buffers_gles3.h                                          */
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

#ifndef RENDER_SCENE_BUFFERS_GLES3_H
#define RENDER_SCENE_BUFFERS_GLES3_H

#ifdef GLES3_ENABLED

#include "servers/rendering/storage/render_scene_buffers.h"

#include "platform_gl.h"

class RenderSceneBuffersGLES3 : public RenderSceneBuffers {
	GDCLASS(RenderSceneBuffersGLES3, RenderSceneBuffers);

public:
	// Original implementation, need to investigate which ones we'll keep like this and what we'll change...

	int internal_width = 0;
	int internal_height = 0;
	int width = 0;
	int height = 0;
	//float fsr_sharpness = 0.2f;
	RS::ViewportMSAA msaa = RS::VIEWPORT_MSAA_DISABLED;
	//RS::ViewportScreenSpaceAA screen_space_aa = RS::VIEWPORT_SCREEN_SPACE_AA_DISABLED;
	//bool use_debanding = false;
	uint32_t view_count = 1;

	RID render_target;

	//built-in textures used for ping pong image processing and blurring
	struct Blur {
		RID texture;

		struct Mipmap {
			RID texture;
			int width;
			int height;
			GLuint fbo;
		};

		Vector<Mipmap> mipmaps;
	};

	Blur blur[2]; //the second one starts from the first mipmap

private:
public:
	virtual ~RenderSceneBuffersGLES3();
	virtual void configure(const RenderSceneBuffersConfiguration *p_config) override;

	virtual void set_fsr_sharpness(float p_fsr_sharpness) override{};
	virtual void set_texture_mipmap_bias(float p_texture_mipmap_bias) override{};
	virtual void set_use_debanding(bool p_use_debanding) override{};

	void free_render_buffer_data();
};

#endif // GLES3_ENABLED

#endif // RENDER_SCENE_BUFFERS_GLES3_H
