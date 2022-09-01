/*************************************************************************/
/*  render_scene_buffers_gles3.h                                         */
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

#ifndef RENDER_SCENE_BUFFERS_GLES3_H
#define RENDER_SCENE_BUFFERS_GLES3_H

#ifdef GLES3_ENABLED

#include "servers/rendering/storage/render_scene_buffers.h"

#include "platform_config.h"
#ifndef OPENGL_INCLUDE_H
#include <GLES3/gl3.h>
#else
#include OPENGL_INCLUDE_H
#endif

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
	//uint32_t view_count = 1;

	bool is_transparent = false;

	RID render_target;
	GLuint internal_texture = 0; // Used for rendering when post effects are enabled
	GLuint depth_texture = 0; // Main depth texture
	GLuint framebuffer = 0; // Main framebuffer, contains internal_texture and depth_texture or render_target->color and depth_texture

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
	virtual void configure(RID p_render_target, const Size2i p_internal_size, const Size2i p_target_size, float p_fsr_sharpness, float p_texture_mipmap_bias, RS::ViewportMSAA p_msaa, RenderingServer::ViewportScreenSpaceAA p_screen_space_aa, bool p_use_taa, bool p_use_debanding, uint32_t p_view_count) override;

	virtual void set_fsr_sharpness(float p_fsr_sharpness) override{};
	virtual void set_texture_mipmap_bias(float p_texture_mipmap_bias) override{};
	virtual void set_use_debanding(bool p_use_debanding) override{};

	void free_render_buffer_data();
};

#endif // GLES3_ENABLED

#endif // RENDER_SCENE_BUFFERS_GLES3_H
