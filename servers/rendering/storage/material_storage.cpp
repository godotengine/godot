/**************************************************************************/
/*  material_storage.cpp                                                  */
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

#include "material_storage.h"

HashMap<StringName, RendererMaterialStorage::BlendData> *RendererMaterialStorage::BlendRegistry::get(const RSE::ShaderMode &p_key) {
	switch (p_key) {
		case RSE::SHADER_CANVAS_ITEM:
			return &canvas_blend_mode;
		case RSE::SHADER_SPATIAL:
			return &spatial_blend_mode;
		case RSE::SHADER_TEXTURE_BLIT:
			return &texture_blit_blend_mode;
		default:
			return nullptr;
	}
}

void RendererMaterialStorage::register_blend_mode(const RSE::ShaderMode p_mode, StringName p_name, RendererMaterialStorage::BlendData p_data, const bool p_shader_enabled) {
	StringName blend_name = "blend_" + String(p_name);
	auto modes = blend_mode_registry.get(p_mode);
	ERR_FAIL_COND_MSG(modes == nullptr, "registry only contains blend modes for shader types canvas_item, spatial, and texture blit.");

	modes->insert(blend_name, p_data);

	if (p_shader_enabled) {
		ShaderTypes::get_singleton()->add_blend_mode(p_mode, p_name);
	}
}

void RendererMaterialStorage::register_blend_mode(const RSE::ShaderMode p_mode, RSE::BlendMode p_blend_mode, RendererMaterialStorage::BlendData p_data, const bool p_shader_enabled) {
	register_blend_mode(p_mode, RenderingServerTypes::blend_mode_to_string(p_blend_mode), p_data, p_shader_enabled);
}

void RendererMaterialStorage::register_blend_mode(const RSE::ShaderMode p_mode, StringName p_name, RenderingDeviceCommons::PipelineColorBlendState::Attachment p_attachment) {
	register_blend_mode(p_mode, p_name, BlendData(p_attachment), true);
}

Vector<StringName> RendererMaterialStorage::get_blend_modes(const RSE::ShaderMode p_mode) {
	Vector<StringName> modes;
	auto el = blend_mode_registry.get(p_mode)->begin();
	while (el) {
		modes.append(el->key);
		++el;
	}
	return modes;
}

RenderingDeviceCommons::PipelineColorBlendState::Attachment RendererMaterialStorage::get_blend_attachment(const RSE::ShaderMode p_mode, StringName p_blend_mode, bool p_transparent) {
	auto attachments = blend_mode_registry.get(p_mode);
	RenderingDeviceCommons::PipelineColorBlendState::Attachment attachment;

	if (attachments->has(p_blend_mode)) {
		RendererMaterialStorage::BlendData *blend = attachments->getptr(p_blend_mode);
		if (p_transparent) {
			attachment = blend->transparent_attachment;
		} else {
			attachment = blend->attachment;
		}
	} else {
		print_line("unrecognized blend mode", p_mode, p_blend_mode);
	}
	return attachment;
}
