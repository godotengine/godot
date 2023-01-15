/**************************************************************************/
/*  render_scene_buffers_gles3.cpp                                        */
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

#ifdef GLES3_ENABLED

#include "render_scene_buffers_gles3.h"
#include "texture_storage.h"

RenderSceneBuffersGLES3::~RenderSceneBuffersGLES3() {
	free_render_buffer_data();
}

void RenderSceneBuffersGLES3::configure(RID p_render_target, const Size2i p_internal_size, const Size2i p_target_size, RS::ViewportScaling3DMode p_scaling_3d_mode, float p_fsr_sharpness, float p_texture_mipmap_bias, RS::ViewportMSAA p_msaa, RenderingServer::ViewportScreenSpaceAA p_screen_space_aa, bool p_use_taa, bool p_use_debanding, uint32_t p_view_count) {
	GLES3::TextureStorage *texture_storage = GLES3::TextureStorage::get_singleton();

	//internal_size.x = p_internal_size.x; // ignore for now
	//internal_size.y = p_internal_size.y;
	width = p_target_size.x;
	height = p_target_size.y;
	//scaling_3d_mode = p_scaling_3d_mode
	//fsr_sharpness = p_fsr_sharpness;
	//texture_mipmap_bias = p_texture_mipmap_bias;
	render_target = p_render_target;
	//msaa = p_msaa;
	//screen_space_aa = p_screen_space_aa;
	//use_debanding = p_use_debanding;
	view_count = p_view_count;

	free_render_buffer_data();

	GLES3::RenderTarget *rt = texture_storage->get_render_target(p_render_target);

	is_transparent = rt->is_transparent;
}

void RenderSceneBuffersGLES3::free_render_buffer_data() {
}

#endif // GLES3_ENABLED
