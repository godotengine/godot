/**************************************************************************/
/*  render_data_extension.cpp                                             */
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

#include "render_data_extension.h"

#include "servers/rendering/rendering_server.h" // IWYU pragma: Needed to bind RSE enums.

#include "core/object/class_db.h"

// RenderSceneBuffersExtension

void RenderSceneBuffersExtension::_bind_methods() {
	GDVIRTUAL_BIND(_configure, "config");
	GDVIRTUAL_BIND(_set_fsr_sharpness, "fsr_sharpness");
	GDVIRTUAL_BIND(_set_texture_mipmap_bias, "texture_mipmap_bias");
	GDVIRTUAL_BIND(_set_anisotropic_filtering_level, "anisotropic_filtering_level");
	GDVIRTUAL_BIND(_set_use_debanding, "use_debanding");
}

void RenderSceneBuffersExtension::configure(const RenderSceneBuffersConfiguration *p_config) {
	GDVIRTUAL_CALL(_configure, p_config);
}

void RenderSceneBuffersExtension::set_fsr_sharpness(float p_fsr_sharpness) {
	GDVIRTUAL_CALL(_set_fsr_sharpness, p_fsr_sharpness);
}

void RenderSceneBuffersExtension::set_texture_mipmap_bias(float p_texture_mipmap_bias) {
	GDVIRTUAL_CALL(_set_texture_mipmap_bias, p_texture_mipmap_bias);
}

void RenderSceneBuffersExtension::set_anisotropic_filtering_level(RSE::ViewportAnisotropicFiltering p_anisotropic_filtering_level) {
	GDVIRTUAL_CALL(_set_anisotropic_filtering_level, p_anisotropic_filtering_level);
}

void RenderSceneBuffersExtension::set_use_debanding(bool p_use_debanding) {
	GDVIRTUAL_CALL(_set_use_debanding, p_use_debanding);
}

// RenderSceneDataExtension

void RenderSceneDataExtension::_bind_methods() {
	GDVIRTUAL_BIND(_get_cam_transform);
	GDVIRTUAL_BIND(_get_cam_projection);
	GDVIRTUAL_BIND(_get_view_count);
	GDVIRTUAL_BIND(_get_view_eye_offset, "view");
	GDVIRTUAL_BIND(_get_view_projection, "view");

	GDVIRTUAL_BIND(_get_uniform_buffer);
}

Transform3D RenderSceneDataExtension::get_cam_transform() const {
	Transform3D ret;
	GDVIRTUAL_CALL(_get_cam_transform, ret);
	return ret;
}

Projection RenderSceneDataExtension::get_cam_projection() const {
	Projection ret;
	GDVIRTUAL_CALL(_get_cam_projection, ret);
	return ret;
}

uint32_t RenderSceneDataExtension::get_view_count() const {
	uint32_t ret = 0;
	GDVIRTUAL_CALL(_get_view_count, ret);
	return ret;
}

Vector3 RenderSceneDataExtension::get_view_eye_offset(uint32_t p_view) const {
	Vector3 ret;
	GDVIRTUAL_CALL(_get_view_eye_offset, p_view, ret);
	return ret;
}

Projection RenderSceneDataExtension::get_view_projection(uint32_t p_view) const {
	Projection ret;
	GDVIRTUAL_CALL(_get_view_projection, p_view, ret);
	return ret;
}

RID RenderSceneDataExtension::get_uniform_buffer() const {
	RID ret;
	GDVIRTUAL_CALL(_get_uniform_buffer, ret);
	return ret;
}

// RenderDataExtension

void RenderDataExtension::_bind_methods() {
	GDVIRTUAL_BIND(_get_render_scene_buffers);
	GDVIRTUAL_BIND(_get_render_scene_data)
	GDVIRTUAL_BIND(_get_environment)
	GDVIRTUAL_BIND(_get_camera_attributes)
}

Ref<RenderSceneBuffers> RenderDataExtension::get_render_scene_buffers() const {
	Ref<RenderSceneBuffers> ret;
	GDVIRTUAL_CALL(_get_render_scene_buffers, ret);
	return ret;
}

RenderSceneData *RenderDataExtension::get_render_scene_data() const {
	RenderSceneData *ret = nullptr;
	GDVIRTUAL_CALL(_get_render_scene_data, ret);
	return ret;
}

RID RenderDataExtension::get_environment() const {
	RID ret;
	GDVIRTUAL_CALL(_get_environment, ret);
	return ret;
}

RID RenderDataExtension::get_camera_attributes() const {
	RID ret;
	GDVIRTUAL_CALL(_get_camera_attributes, ret);
	return ret;
}
