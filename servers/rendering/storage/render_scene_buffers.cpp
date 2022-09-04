/*************************************************************************/
/*  render_scene_buffers.cpp                                             */
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

#include "render_scene_buffers.h"

void RenderSceneBuffers::_bind_methods() {
	ClassDB::bind_method(D_METHOD("configure", "render_target", "internal_size", "target_size", "fsr_sharpness", "texture_mipmap_bias", "msaa", "screen_space_aa", "use_taa", "use_debanding", "view_count"), &RenderSceneBuffers::configure);
}

void RenderSceneBuffers::configure(RID p_render_target, const Size2i p_internal_size, const Size2i p_target_size, float p_fsr_sharpness, float p_texture_mipmap_bias, RS::ViewportMSAA p_msaa, RenderingServer::ViewportScreenSpaceAA p_screen_space_aa, bool p_use_taa, bool p_use_debanding, uint32_t p_view_count) {
	GDVIRTUAL_CALL(_configure, p_render_target, p_internal_size, p_target_size, p_fsr_sharpness, p_texture_mipmap_bias, p_msaa, p_screen_space_aa, p_use_taa, p_use_debanding, p_view_count);
};

void RenderSceneBuffers::set_fsr_sharpness(float p_fsr_sharpness) {
	GDVIRTUAL_CALL(_set_fsr_sharpness, p_fsr_sharpness);
}

void RenderSceneBuffers::set_texture_mipmap_bias(float p_texture_mipmap_bias) {
	GDVIRTUAL_CALL(_set_texture_mipmap_bias, p_texture_mipmap_bias);
}

void RenderSceneBuffers::set_use_debanding(bool p_use_debanding) {
	GDVIRTUAL_CALL(_set_use_debanding, p_use_debanding);
}
