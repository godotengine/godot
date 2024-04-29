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

void RenderSceneBuffersGLES3::configure(const RenderSceneBuffersConfiguration *p_config) {
	//internal_size.x = p_config->get_internal_size().x; // ignore for now
	//internal_size.y = p_config->get_internal_size().y;
	width = p_config->get_target_size().x;
	height = p_config->get_target_size().y;
	//scaling_3d_mode = p_config->get_scaling_3d_mode()
	//fsr_sharpness = p_config->get_fsr_sharpness();
	//texture_mipmap_bias = p_config->get_texture_mipmap_bias();
	render_target = p_config->get_render_target();
	//msaa = p_config->get_msaa_3d();
	//screen_space_aa = p_config->get_screen_space_aa();
	//use_debanding = p_config->get_use_debanding();
	view_count = p_config->get_view_count();

	free_render_buffer_data();
}

void RenderSceneBuffersGLES3::free_render_buffer_data() {
}

#endif // GLES3_ENABLED
