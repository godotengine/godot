/*************************************************************************/
/*  texture_storage.cpp                                                  */
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

#include "texture_storage.h"
#include "../effects/copy_effects.h"
#include "material_storage.h"

using namespace RendererRD;

///////////////////////////////////////////////////////////////////////////
// TextureStorage::CanvasTexture

void TextureStorage::CanvasTexture::clear_sets() {
	if (cleared_cache) {
		return;
	}
	for (int i = 1; i < RS::CANVAS_ITEM_TEXTURE_FILTER_MAX; i++) {
		for (int j = 1; j < RS::CANVAS_ITEM_TEXTURE_REPEAT_MAX; j++) {
			if (RD::get_singleton()->uniform_set_is_valid(uniform_sets[i][j])) {
				RD::get_singleton()->free(uniform_sets[i][j]);
				uniform_sets[i][j] = RID();
			}
		}
	}
	cleared_cache = true;
}

TextureStorage::CanvasTexture::~CanvasTexture() {
	clear_sets();
}

///////////////////////////////////////////////////////////////////////////
// TextureStorage::Texture

void TextureStorage::Texture::cleanup() {
	if (RD::get_singleton()->texture_is_valid(rd_texture_srgb)) {
		//erase this first, as it's a dependency of the one below
		RD::get_singleton()->free(rd_texture_srgb);
	}
	if (RD::get_singleton()->texture_is_valid(rd_texture)) {
		RD::get_singleton()->free(rd_texture);
	}
	if (canvas_texture) {
		memdelete(canvas_texture);
	}
}

///////////////////////////////////////////////////////////////////////////
// TextureStorage

TextureStorage *TextureStorage::singleton = nullptr;

TextureStorage *TextureStorage::get_singleton() {
	return singleton;
}

TextureStorage::TextureStorage() {
	singleton = this;

	{ //create default textures

		RD::TextureFormat tformat;
		tformat.format = RD::DATA_FORMAT_R8G8B8A8_UNORM;
		tformat.width = 4;
		tformat.height = 4;
		tformat.usage_bits = RD::TEXTURE_USAGE_SAMPLING_BIT | RD::TEXTURE_USAGE_CAN_UPDATE_BIT;
		tformat.texture_type = RD::TEXTURE_TYPE_2D;

		Vector<uint8_t> pv;
		pv.resize(16 * 4);
		for (int i = 0; i < 16; i++) {
			// Opaque white.
			pv.set(i * 4 + 0, 255);
			pv.set(i * 4 + 1, 255);
			pv.set(i * 4 + 2, 255);
			pv.set(i * 4 + 3, 255);
		}

		{
			Vector<Vector<uint8_t>> vpv;
			vpv.push_back(pv);
			default_rd_textures[DEFAULT_RD_TEXTURE_WHITE] = RD::get_singleton()->texture_create(tformat, RD::TextureView(), vpv);
		}

		for (int i = 0; i < 16; i++) {
			// Opaque black.
			pv.set(i * 4 + 0, 0);
			pv.set(i * 4 + 1, 0);
			pv.set(i * 4 + 2, 0);
			pv.set(i * 4 + 3, 255);
		}

		{
			Vector<Vector<uint8_t>> vpv;
			vpv.push_back(pv);
			default_rd_textures[DEFAULT_RD_TEXTURE_BLACK] = RD::get_singleton()->texture_create(tformat, RD::TextureView(), vpv);
		}

		for (int i = 0; i < 16; i++) {
			// Transparent black.
			pv.set(i * 4 + 0, 0);
			pv.set(i * 4 + 1, 0);
			pv.set(i * 4 + 2, 0);
			pv.set(i * 4 + 3, 0);
		}

		{
			Vector<Vector<uint8_t>> vpv;
			vpv.push_back(pv);
			default_rd_textures[DEFAULT_RD_TEXTURE_TRANSPARENT] = RD::get_singleton()->texture_create(tformat, RD::TextureView(), vpv);
		}

		for (int i = 0; i < 16; i++) {
			// Opaque normal map "flat" color.
			pv.set(i * 4 + 0, 128);
			pv.set(i * 4 + 1, 128);
			pv.set(i * 4 + 2, 255);
			pv.set(i * 4 + 3, 255);
		}

		{
			Vector<Vector<uint8_t>> vpv;
			vpv.push_back(pv);
			default_rd_textures[DEFAULT_RD_TEXTURE_NORMAL] = RD::get_singleton()->texture_create(tformat, RD::TextureView(), vpv);
		}

		for (int i = 0; i < 16; i++) {
			// Opaque flowmap "flat" color.
			pv.set(i * 4 + 0, 255);
			pv.set(i * 4 + 1, 128);
			pv.set(i * 4 + 2, 255);
			pv.set(i * 4 + 3, 255);
		}

		{
			Vector<Vector<uint8_t>> vpv;
			vpv.push_back(pv);
			default_rd_textures[DEFAULT_RD_TEXTURE_ANISO] = RD::get_singleton()->texture_create(tformat, RD::TextureView(), vpv);
		}

		{
			RD::TextureFormat tf;
			tf.format = RD::DATA_FORMAT_D16_UNORM;
			tf.width = 4;
			tf.height = 4;
			tf.usage_bits = RD::TEXTURE_USAGE_SAMPLING_BIT | RD::TEXTURE_USAGE_CAN_UPDATE_BIT | RD::TEXTURE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT;
			tf.texture_type = RD::TEXTURE_TYPE_2D;

			Vector<uint8_t> sv;
			sv.resize(16 * 2);
			uint16_t *ptr = (uint16_t *)sv.ptrw();
			for (int i = 0; i < 16; i++) {
				ptr[i] = Math::make_half_float(1.0f);
			}

			Vector<Vector<uint8_t>> vpv;
			vpv.push_back(sv);
			default_rd_textures[DEFAULT_RD_TEXTURE_DEPTH] = RD::get_singleton()->texture_create(tf, RD::TextureView(), vpv);
		}

		for (int i = 0; i < 16; i++) {
			pv.set(i * 4 + 0, 0);
			pv.set(i * 4 + 1, 0);
			pv.set(i * 4 + 2, 0);
			pv.set(i * 4 + 3, 0);
		}

		default_rd_textures[DEFAULT_RD_TEXTURE_MULTIMESH_BUFFER] = RD::get_singleton()->texture_buffer_create(16, RD::DATA_FORMAT_R8G8B8A8_UNORM, pv);

		for (int i = 0; i < 16; i++) {
			pv.set(i * 4 + 0, 0);
			pv.set(i * 4 + 1, 0);
			pv.set(i * 4 + 2, 0);
			pv.set(i * 4 + 3, 0);
		}

		{
			tformat.format = RD::DATA_FORMAT_R8G8B8A8_UINT;
			Vector<Vector<uint8_t>> vpv;
			vpv.push_back(pv);
			default_rd_textures[DEFAULT_RD_TEXTURE_2D_UINT] = RD::get_singleton()->texture_create(tformat, RD::TextureView(), vpv);
		}
	}

	{ //create default black cubemap array

		RD::TextureFormat tformat;
		tformat.format = RD::DATA_FORMAT_R8G8B8A8_UNORM;
		tformat.width = 4;
		tformat.height = 4;
		tformat.array_layers = 6;
		tformat.usage_bits = RD::TEXTURE_USAGE_SAMPLING_BIT | RD::TEXTURE_USAGE_CAN_UPDATE_BIT;
		tformat.texture_type = RD::TEXTURE_TYPE_CUBE_ARRAY;

		Vector<uint8_t> pv;
		pv.resize(16 * 4);
		for (int i = 0; i < 16; i++) {
			pv.set(i * 4 + 0, 0);
			pv.set(i * 4 + 1, 0);
			pv.set(i * 4 + 2, 0);
			pv.set(i * 4 + 3, 0);
		}

		{
			Vector<Vector<uint8_t>> vpv;
			for (int i = 0; i < 6; i++) {
				vpv.push_back(pv);
			}
			default_rd_textures[DEFAULT_RD_TEXTURE_CUBEMAP_ARRAY_BLACK] = RD::get_singleton()->texture_create(tformat, RD::TextureView(), vpv);
		}
	}

	{ //create default white cubemap array

		RD::TextureFormat tformat;
		tformat.format = RD::DATA_FORMAT_R8G8B8A8_UNORM;
		tformat.width = 4;
		tformat.height = 4;
		tformat.array_layers = 6;
		tformat.usage_bits = RD::TEXTURE_USAGE_SAMPLING_BIT | RD::TEXTURE_USAGE_CAN_UPDATE_BIT;
		tformat.texture_type = RD::TEXTURE_TYPE_CUBE_ARRAY;

		Vector<uint8_t> pv;
		pv.resize(16 * 4);
		for (int i = 0; i < 16; i++) {
			pv.set(i * 4 + 0, 255);
			pv.set(i * 4 + 1, 255);
			pv.set(i * 4 + 2, 255);
			pv.set(i * 4 + 3, 255);
		}

		{
			Vector<Vector<uint8_t>> vpv;
			for (int i = 0; i < 6; i++) {
				vpv.push_back(pv);
			}
			default_rd_textures[DEFAULT_RD_TEXTURE_CUBEMAP_ARRAY_WHITE] = RD::get_singleton()->texture_create(tformat, RD::TextureView(), vpv);
		}
	}

	{ //create default black cubemap

		RD::TextureFormat tformat;
		tformat.format = RD::DATA_FORMAT_R8G8B8A8_UNORM;
		tformat.width = 4;
		tformat.height = 4;
		tformat.array_layers = 6;
		tformat.usage_bits = RD::TEXTURE_USAGE_SAMPLING_BIT | RD::TEXTURE_USAGE_CAN_UPDATE_BIT;
		tformat.texture_type = RD::TEXTURE_TYPE_CUBE;

		Vector<uint8_t> pv;
		pv.resize(16 * 4);
		for (int i = 0; i < 16; i++) {
			pv.set(i * 4 + 0, 0);
			pv.set(i * 4 + 1, 0);
			pv.set(i * 4 + 2, 0);
			pv.set(i * 4 + 3, 0);
		}

		{
			Vector<Vector<uint8_t>> vpv;
			for (int i = 0; i < 6; i++) {
				vpv.push_back(pv);
			}
			default_rd_textures[DEFAULT_RD_TEXTURE_CUBEMAP_BLACK] = RD::get_singleton()->texture_create(tformat, RD::TextureView(), vpv);
		}
	}

	{ //create default white cubemap

		RD::TextureFormat tformat;
		tformat.format = RD::DATA_FORMAT_R8G8B8A8_UNORM;
		tformat.width = 4;
		tformat.height = 4;
		tformat.array_layers = 6;
		tformat.usage_bits = RD::TEXTURE_USAGE_SAMPLING_BIT | RD::TEXTURE_USAGE_CAN_UPDATE_BIT;
		tformat.texture_type = RD::TEXTURE_TYPE_CUBE;

		Vector<uint8_t> pv;
		pv.resize(16 * 4);
		for (int i = 0; i < 16; i++) {
			pv.set(i * 4 + 0, 255);
			pv.set(i * 4 + 1, 255);
			pv.set(i * 4 + 2, 255);
			pv.set(i * 4 + 3, 255);
		}

		{
			Vector<Vector<uint8_t>> vpv;
			for (int i = 0; i < 6; i++) {
				vpv.push_back(pv);
			}
			default_rd_textures[DEFAULT_RD_TEXTURE_CUBEMAP_WHITE] = RD::get_singleton()->texture_create(tformat, RD::TextureView(), vpv);
		}
	}

	{ //create default 3D

		RD::TextureFormat tformat;
		tformat.format = RD::DATA_FORMAT_R8G8B8A8_UNORM;
		tformat.width = 4;
		tformat.height = 4;
		tformat.depth = 4;
		tformat.usage_bits = RD::TEXTURE_USAGE_SAMPLING_BIT | RD::TEXTURE_USAGE_CAN_UPDATE_BIT;
		tformat.texture_type = RD::TEXTURE_TYPE_3D;

		Vector<uint8_t> pv;
		pv.resize(64 * 4);
		for (int i = 0; i < 64; i++) {
			pv.set(i * 4 + 0, 0);
			pv.set(i * 4 + 1, 0);
			pv.set(i * 4 + 2, 0);
			pv.set(i * 4 + 3, 0);
		}

		{
			Vector<Vector<uint8_t>> vpv;
			vpv.push_back(pv);
			default_rd_textures[DEFAULT_RD_TEXTURE_3D_BLACK] = RD::get_singleton()->texture_create(tformat, RD::TextureView(), vpv);
		}
		for (int i = 0; i < 64; i++) {
			pv.set(i * 4 + 0, 255);
			pv.set(i * 4 + 1, 255);
			pv.set(i * 4 + 2, 255);
			pv.set(i * 4 + 3, 255);
		}

		{
			Vector<Vector<uint8_t>> vpv;
			vpv.push_back(pv);
			default_rd_textures[DEFAULT_RD_TEXTURE_3D_WHITE] = RD::get_singleton()->texture_create(tformat, RD::TextureView(), vpv);
		}
	}

	{ //create default array

		RD::TextureFormat tformat;
		tformat.format = RD::DATA_FORMAT_R8G8B8A8_UNORM;
		tformat.width = 4;
		tformat.height = 4;
		tformat.array_layers = 1;
		tformat.usage_bits = RD::TEXTURE_USAGE_SAMPLING_BIT | RD::TEXTURE_USAGE_CAN_UPDATE_BIT;
		tformat.texture_type = RD::TEXTURE_TYPE_2D_ARRAY;

		Vector<uint8_t> pv;
		pv.resize(16 * 4);
		for (int i = 0; i < 16; i++) {
			pv.set(i * 4 + 0, 255);
			pv.set(i * 4 + 1, 255);
			pv.set(i * 4 + 2, 255);
			pv.set(i * 4 + 3, 255);
		}

		{
			Vector<Vector<uint8_t>> vpv;
			vpv.push_back(pv);
			default_rd_textures[DEFAULT_RD_TEXTURE_2D_ARRAY_WHITE] = RD::get_singleton()->texture_create(tformat, RD::TextureView(), vpv);
		}
	}

	{ // default atlas texture
		RD::TextureFormat tformat;
		tformat.format = RD::DATA_FORMAT_R8G8B8A8_UNORM;
		tformat.width = 4;
		tformat.height = 4;
		tformat.usage_bits = RD::TEXTURE_USAGE_SAMPLING_BIT | RD::TEXTURE_USAGE_CAN_UPDATE_BIT;
		tformat.texture_type = RD::TEXTURE_TYPE_2D;

		Vector<uint8_t> pv;
		pv.resize(16 * 4);
		for (int i = 0; i < 16; i++) {
			pv.set(i * 4 + 0, 0);
			pv.set(i * 4 + 1, 0);
			pv.set(i * 4 + 2, 0);
			pv.set(i * 4 + 3, 255);
		}

		{
			Vector<Vector<uint8_t>> vpv;
			vpv.push_back(pv);
			decal_atlas.texture = RD::get_singleton()->texture_create(tformat, RD::TextureView(), vpv);
			decal_atlas.texture_srgb = decal_atlas.texture;
		}
	}

	{ //create default VRS

		RD::TextureFormat tformat;
		tformat.format = RD::DATA_FORMAT_R8_UINT;
		tformat.width = 4;
		tformat.height = 4;
		tformat.usage_bits = RD::TEXTURE_USAGE_COLOR_ATTACHMENT_BIT | RD::TEXTURE_USAGE_SAMPLING_BIT | RD::TEXTURE_USAGE_STORAGE_BIT | RD::TEXTURE_USAGE_CAN_UPDATE_BIT;
		if (RD::get_singleton()->has_feature(RD::SUPPORTS_ATTACHMENT_VRS)) {
			tformat.usage_bits |= RD::TEXTURE_USAGE_VRS_ATTACHMENT_BIT;
		}
		tformat.texture_type = RD::TEXTURE_TYPE_2D;

		Vector<uint8_t> pv;
		pv.resize(4 * 4);
		for (int i = 0; i < 4 * 4; i++) {
			pv.set(i, 0);
		}

		{
			Vector<Vector<uint8_t>> vpv;
			vpv.push_back(pv);
			default_rd_textures[DEFAULT_RD_TEXTURE_VRS] = RD::get_singleton()->texture_create(tformat, RD::TextureView(), vpv);
		}
	}

	{
		Vector<String> sdf_modes;
		sdf_modes.push_back("\n#define MODE_LOAD\n");
		sdf_modes.push_back("\n#define MODE_LOAD_SHRINK\n");
		sdf_modes.push_back("\n#define MODE_PROCESS\n");
		sdf_modes.push_back("\n#define MODE_PROCESS_OPTIMIZED\n");
		sdf_modes.push_back("\n#define MODE_STORE\n");
		sdf_modes.push_back("\n#define MODE_STORE_SHRINK\n");

		rt_sdf.shader.initialize(sdf_modes);

		rt_sdf.shader_version = rt_sdf.shader.version_create();

		for (int i = 0; i < RenderTargetSDF::SHADER_MAX; i++) {
			rt_sdf.pipelines[i] = RD::get_singleton()->compute_pipeline_create(rt_sdf.shader.version_get_shader(rt_sdf.shader_version, i));
		}
	}
}

TextureStorage::~TextureStorage() {
	rt_sdf.shader.version_free(rt_sdf.shader_version);

	if (decal_atlas.textures.size()) {
		ERR_PRINT("Decal Atlas: " + itos(decal_atlas.textures.size()) + " textures were not removed from the atlas.");
	}

	if (decal_atlas.texture.is_valid()) {
		RD::get_singleton()->free(decal_atlas.texture);
	}

	//def textures
	for (int i = 0; i < DEFAULT_RD_TEXTURE_MAX; i++) {
		if (default_rd_textures[i].is_valid()) {
			RD::get_singleton()->free(default_rd_textures[i]);
		}
	}

	singleton = nullptr;
}

bool TextureStorage::can_create_resources_async() const {
	return true;
}

/* Canvas Texture API */

RID TextureStorage::canvas_texture_allocate() {
	return canvas_texture_owner.allocate_rid();
}

void TextureStorage::canvas_texture_initialize(RID p_rid) {
	canvas_texture_owner.initialize_rid(p_rid);
}

void TextureStorage::canvas_texture_free(RID p_rid) {
	canvas_texture_owner.free(p_rid);
}

void TextureStorage::canvas_texture_set_channel(RID p_canvas_texture, RS::CanvasTextureChannel p_channel, RID p_texture) {
	CanvasTexture *ct = canvas_texture_owner.get_or_null(p_canvas_texture);
	ERR_FAIL_NULL(ct);

	switch (p_channel) {
		case RS::CANVAS_TEXTURE_CHANNEL_DIFFUSE: {
			ct->diffuse = p_texture;
		} break;
		case RS::CANVAS_TEXTURE_CHANNEL_NORMAL: {
			ct->normal_map = p_texture;
		} break;
		case RS::CANVAS_TEXTURE_CHANNEL_SPECULAR: {
			ct->specular = p_texture;
		} break;
	}

	ct->clear_sets();
}

void TextureStorage::canvas_texture_set_shading_parameters(RID p_canvas_texture, const Color &p_specular_color, float p_shininess) {
	CanvasTexture *ct = canvas_texture_owner.get_or_null(p_canvas_texture);
	ERR_FAIL_NULL(ct);

	ct->specular_color.r = p_specular_color.r;
	ct->specular_color.g = p_specular_color.g;
	ct->specular_color.b = p_specular_color.b;
	ct->specular_color.a = p_shininess;
	ct->clear_sets();
}

void TextureStorage::canvas_texture_set_texture_filter(RID p_canvas_texture, RS::CanvasItemTextureFilter p_filter) {
	CanvasTexture *ct = canvas_texture_owner.get_or_null(p_canvas_texture);
	ERR_FAIL_NULL(ct);

	ct->texture_filter = p_filter;
	ct->clear_sets();
}

void TextureStorage::canvas_texture_set_texture_repeat(RID p_canvas_texture, RS::CanvasItemTextureRepeat p_repeat) {
	CanvasTexture *ct = canvas_texture_owner.get_or_null(p_canvas_texture);
	ERR_FAIL_NULL(ct);
	ct->texture_repeat = p_repeat;
	ct->clear_sets();
}

bool TextureStorage::canvas_texture_get_uniform_set(RID p_texture, RS::CanvasItemTextureFilter p_base_filter, RS::CanvasItemTextureRepeat p_base_repeat, RID p_base_shader, int p_base_set, RID &r_uniform_set, Size2i &r_size, Color &r_specular_shininess, bool &r_use_normal, bool &r_use_specular) {
	MaterialStorage *material_storage = MaterialStorage::get_singleton();

	CanvasTexture *ct = nullptr;
	Texture *t = get_texture(p_texture);

	// TODO once we have our texture storage split off we'll look into moving this code into canvas_texture

	if (t) {
		//regular texture
		if (!t->canvas_texture) {
			t->canvas_texture = memnew(CanvasTexture);
			t->canvas_texture->diffuse = p_texture;
		}

		ct = t->canvas_texture;
	} else {
		ct = canvas_texture_owner.get_or_null(p_texture);
	}

	if (!ct) {
		return false; //invalid texture RID
	}

	RS::CanvasItemTextureFilter filter = ct->texture_filter != RS::CANVAS_ITEM_TEXTURE_FILTER_DEFAULT ? ct->texture_filter : p_base_filter;
	ERR_FAIL_COND_V(filter == RS::CANVAS_ITEM_TEXTURE_FILTER_DEFAULT, false);

	RS::CanvasItemTextureRepeat repeat = ct->texture_repeat != RS::CANVAS_ITEM_TEXTURE_REPEAT_DEFAULT ? ct->texture_repeat : p_base_repeat;
	ERR_FAIL_COND_V(repeat == RS::CANVAS_ITEM_TEXTURE_REPEAT_DEFAULT, false);

	RID uniform_set = ct->uniform_sets[filter][repeat];
	if (!RD::get_singleton()->uniform_set_is_valid(uniform_set)) {
		//create and update
		Vector<RD::Uniform> uniforms;
		{ //diffuse
			RD::Uniform u;
			u.uniform_type = RD::UNIFORM_TYPE_TEXTURE;
			u.binding = 0;

			t = get_texture(ct->diffuse);
			if (!t) {
				u.append_id(texture_rd_get_default(DEFAULT_RD_TEXTURE_WHITE));
				ct->size_cache = Size2i(1, 1);
			} else {
				u.append_id(t->rd_texture);
				ct->size_cache = Size2i(t->width_2d, t->height_2d);
			}
			uniforms.push_back(u);
		}
		{ //normal
			RD::Uniform u;
			u.uniform_type = RD::UNIFORM_TYPE_TEXTURE;
			u.binding = 1;

			t = get_texture(ct->normal_map);
			if (!t) {
				u.append_id(texture_rd_get_default(DEFAULT_RD_TEXTURE_NORMAL));
				ct->use_normal_cache = false;
			} else {
				u.append_id(t->rd_texture);
				ct->use_normal_cache = true;
			}
			uniforms.push_back(u);
		}
		{ //specular
			RD::Uniform u;
			u.uniform_type = RD::UNIFORM_TYPE_TEXTURE;
			u.binding = 2;

			t = get_texture(ct->specular);
			if (!t) {
				u.append_id(texture_rd_get_default(DEFAULT_RD_TEXTURE_WHITE));
				ct->use_specular_cache = false;
			} else {
				u.append_id(t->rd_texture);
				ct->use_specular_cache = true;
			}
			uniforms.push_back(u);
		}
		{ //sampler
			RD::Uniform u;
			u.uniform_type = RD::UNIFORM_TYPE_SAMPLER;
			u.binding = 3;
			u.append_id(material_storage->sampler_rd_get_default(filter, repeat));
			uniforms.push_back(u);
		}

		uniform_set = RD::get_singleton()->uniform_set_create(uniforms, p_base_shader, p_base_set);
		ct->uniform_sets[filter][repeat] = uniform_set;
		ct->cleared_cache = false;
	}

	r_uniform_set = uniform_set;
	r_size = ct->size_cache;
	r_specular_shininess = ct->specular_color;
	r_use_normal = ct->use_normal_cache;
	r_use_specular = ct->use_specular_cache;

	return true;
}

/* Texture API */

RID TextureStorage::texture_allocate() {
	return texture_owner.allocate_rid();
}

void TextureStorage::texture_free(RID p_texture) {
	Texture *t = texture_owner.get_or_null(p_texture);
	ERR_FAIL_COND(!t);
	ERR_FAIL_COND(t->is_render_target);

	t->cleanup();

	if (t->is_proxy && t->proxy_to.is_valid()) {
		Texture *proxy_to = texture_owner.get_or_null(t->proxy_to);
		if (proxy_to) {
			proxy_to->proxies.erase(p_texture);
		}
	}

	decal_atlas_remove_texture(p_texture);

	for (int i = 0; i < t->proxies.size(); i++) {
		Texture *p = texture_owner.get_or_null(t->proxies[i]);
		ERR_CONTINUE(!p);
		p->proxy_to = RID();
		p->rd_texture = RID();
		p->rd_texture_srgb = RID();
	}

	texture_owner.free(p_texture);
}

void TextureStorage::texture_2d_initialize(RID p_texture, const Ref<Image> &p_image) {
	TextureToRDFormat ret_format;
	Ref<Image> image = _validate_texture_format(p_image, ret_format);

	Texture texture;

	texture.type = TextureStorage::TYPE_2D;

	texture.width = p_image->get_width();
	texture.height = p_image->get_height();
	texture.layers = 1;
	texture.mipmaps = p_image->get_mipmap_count() + 1;
	texture.depth = 1;
	texture.format = p_image->get_format();
	texture.validated_format = image->get_format();

	texture.rd_type = RD::TEXTURE_TYPE_2D;
	texture.rd_format = ret_format.format;
	texture.rd_format_srgb = ret_format.format_srgb;

	RD::TextureFormat rd_format;
	RD::TextureView rd_view;
	{ //attempt register
		rd_format.format = texture.rd_format;
		rd_format.width = texture.width;
		rd_format.height = texture.height;
		rd_format.depth = 1;
		rd_format.array_layers = 1;
		rd_format.mipmaps = texture.mipmaps;
		rd_format.texture_type = texture.rd_type;
		rd_format.samples = RD::TEXTURE_SAMPLES_1;
		rd_format.usage_bits = RD::TEXTURE_USAGE_SAMPLING_BIT | RD::TEXTURE_USAGE_CAN_UPDATE_BIT | RD::TEXTURE_USAGE_CAN_COPY_FROM_BIT;
		if (texture.rd_format_srgb != RD::DATA_FORMAT_MAX) {
			rd_format.shareable_formats.push_back(texture.rd_format);
			rd_format.shareable_formats.push_back(texture.rd_format_srgb);
		}
	}
	{
		rd_view.swizzle_r = ret_format.swizzle_r;
		rd_view.swizzle_g = ret_format.swizzle_g;
		rd_view.swizzle_b = ret_format.swizzle_b;
		rd_view.swizzle_a = ret_format.swizzle_a;
	}
	Vector<uint8_t> data = image->get_data(); //use image data
	Vector<Vector<uint8_t>> data_slices;
	data_slices.push_back(data);
	texture.rd_texture = RD::get_singleton()->texture_create(rd_format, rd_view, data_slices);
	ERR_FAIL_COND(texture.rd_texture.is_null());
	if (texture.rd_format_srgb != RD::DATA_FORMAT_MAX) {
		rd_view.format_override = texture.rd_format_srgb;
		texture.rd_texture_srgb = RD::get_singleton()->texture_create_shared(rd_view, texture.rd_texture);
		if (texture.rd_texture_srgb.is_null()) {
			RD::get_singleton()->free(texture.rd_texture);
			ERR_FAIL_COND(texture.rd_texture_srgb.is_null());
		}
	}

	//used for 2D, overridable
	texture.width_2d = texture.width;
	texture.height_2d = texture.height;
	texture.is_render_target = false;
	texture.rd_view = rd_view;
	texture.is_proxy = false;

	texture_owner.initialize_rid(p_texture, texture);
}

void TextureStorage::texture_2d_layered_initialize(RID p_texture, const Vector<Ref<Image>> &p_layers, RS::TextureLayeredType p_layered_type) {
	ERR_FAIL_COND(p_layers.size() == 0);

	ERR_FAIL_COND(p_layered_type == RS::TEXTURE_LAYERED_CUBEMAP && p_layers.size() != 6);
	ERR_FAIL_COND(p_layered_type == RS::TEXTURE_LAYERED_CUBEMAP_ARRAY && (p_layers.size() < 6 || (p_layers.size() % 6) != 0));

	TextureToRDFormat ret_format;
	Vector<Ref<Image>> images;
	{
		int valid_width = 0;
		int valid_height = 0;
		bool valid_mipmaps = false;
		Image::Format valid_format = Image::FORMAT_MAX;

		for (int i = 0; i < p_layers.size(); i++) {
			ERR_FAIL_COND(p_layers[i]->is_empty());

			if (i == 0) {
				valid_width = p_layers[i]->get_width();
				valid_height = p_layers[i]->get_height();
				valid_format = p_layers[i]->get_format();
				valid_mipmaps = p_layers[i]->has_mipmaps();
			} else {
				ERR_FAIL_COND(p_layers[i]->get_width() != valid_width);
				ERR_FAIL_COND(p_layers[i]->get_height() != valid_height);
				ERR_FAIL_COND(p_layers[i]->get_format() != valid_format);
				ERR_FAIL_COND(p_layers[i]->has_mipmaps() != valid_mipmaps);
			}

			images.push_back(_validate_texture_format(p_layers[i], ret_format));
		}
	}

	Texture texture;

	texture.type = TextureStorage::TYPE_LAYERED;
	texture.layered_type = p_layered_type;

	texture.width = p_layers[0]->get_width();
	texture.height = p_layers[0]->get_height();
	texture.layers = p_layers.size();
	texture.mipmaps = p_layers[0]->get_mipmap_count() + 1;
	texture.depth = 1;
	texture.format = p_layers[0]->get_format();
	texture.validated_format = images[0]->get_format();

	switch (p_layered_type) {
		case RS::TEXTURE_LAYERED_2D_ARRAY: {
			texture.rd_type = RD::TEXTURE_TYPE_2D_ARRAY;
		} break;
		case RS::TEXTURE_LAYERED_CUBEMAP: {
			texture.rd_type = RD::TEXTURE_TYPE_CUBE;
		} break;
		case RS::TEXTURE_LAYERED_CUBEMAP_ARRAY: {
			texture.rd_type = RD::TEXTURE_TYPE_CUBE_ARRAY;
		} break;
	}

	texture.rd_format = ret_format.format;
	texture.rd_format_srgb = ret_format.format_srgb;

	RD::TextureFormat rd_format;
	RD::TextureView rd_view;
	{ //attempt register
		rd_format.format = texture.rd_format;
		rd_format.width = texture.width;
		rd_format.height = texture.height;
		rd_format.depth = 1;
		rd_format.array_layers = texture.layers;
		rd_format.mipmaps = texture.mipmaps;
		rd_format.texture_type = texture.rd_type;
		rd_format.samples = RD::TEXTURE_SAMPLES_1;
		rd_format.usage_bits = RD::TEXTURE_USAGE_SAMPLING_BIT | RD::TEXTURE_USAGE_CAN_UPDATE_BIT | RD::TEXTURE_USAGE_CAN_COPY_FROM_BIT;
		if (texture.rd_format_srgb != RD::DATA_FORMAT_MAX) {
			rd_format.shareable_formats.push_back(texture.rd_format);
			rd_format.shareable_formats.push_back(texture.rd_format_srgb);
		}
	}
	{
		rd_view.swizzle_r = ret_format.swizzle_r;
		rd_view.swizzle_g = ret_format.swizzle_g;
		rd_view.swizzle_b = ret_format.swizzle_b;
		rd_view.swizzle_a = ret_format.swizzle_a;
	}
	Vector<Vector<uint8_t>> data_slices;
	for (int i = 0; i < images.size(); i++) {
		Vector<uint8_t> data = images[i]->get_data(); //use image data
		data_slices.push_back(data);
	}
	texture.rd_texture = RD::get_singleton()->texture_create(rd_format, rd_view, data_slices);
	ERR_FAIL_COND(texture.rd_texture.is_null());
	if (texture.rd_format_srgb != RD::DATA_FORMAT_MAX) {
		rd_view.format_override = texture.rd_format_srgb;
		texture.rd_texture_srgb = RD::get_singleton()->texture_create_shared(rd_view, texture.rd_texture);
		if (texture.rd_texture_srgb.is_null()) {
			RD::get_singleton()->free(texture.rd_texture);
			ERR_FAIL_COND(texture.rd_texture_srgb.is_null());
		}
	}

	//used for 2D, overridable
	texture.width_2d = texture.width;
	texture.height_2d = texture.height;
	texture.is_render_target = false;
	texture.rd_view = rd_view;
	texture.is_proxy = false;

	texture_owner.initialize_rid(p_texture, texture);
}

void TextureStorage::texture_3d_initialize(RID p_texture, Image::Format p_format, int p_width, int p_height, int p_depth, bool p_mipmaps, const Vector<Ref<Image>> &p_data) {
	ERR_FAIL_COND(p_data.size() == 0);

	Image::Image3DValidateError verr = Image::validate_3d_image(p_format, p_width, p_height, p_depth, p_mipmaps, p_data);
	if (verr != Image::VALIDATE_3D_OK) {
		ERR_FAIL_MSG(Image::get_3d_image_validation_error_text(verr));
	}

	TextureToRDFormat ret_format;
	Image::Format validated_format = Image::FORMAT_MAX;
	Vector<uint8_t> all_data;
	uint32_t mipmap_count = 0;
	Vector<Texture::BufferSlice3D> slices;
	{
		Vector<Ref<Image>> images;
		uint32_t all_data_size = 0;
		images.resize(p_data.size());
		for (int i = 0; i < p_data.size(); i++) {
			TextureToRDFormat f;
			images.write[i] = _validate_texture_format(p_data[i], f);
			if (i == 0) {
				ret_format = f;
				validated_format = images[0]->get_format();
			}

			all_data_size += images[i]->get_data().size();
		}

		all_data.resize(all_data_size); //consolidate all data here
		uint32_t offset = 0;
		Size2i prev_size;
		for (int i = 0; i < p_data.size(); i++) {
			uint32_t s = images[i]->get_data().size();

			memcpy(&all_data.write[offset], images[i]->get_data().ptr(), s);
			{
				Texture::BufferSlice3D slice;
				slice.size.width = images[i]->get_width();
				slice.size.height = images[i]->get_height();
				slice.offset = offset;
				slice.buffer_size = s;
				slices.push_back(slice);
			}
			offset += s;

			Size2i img_size(images[i]->get_width(), images[i]->get_height());
			if (img_size != prev_size) {
				mipmap_count++;
			}
			prev_size = img_size;
		}
	}

	Texture texture;

	texture.type = TextureStorage::TYPE_3D;
	texture.width = p_width;
	texture.height = p_height;
	texture.depth = p_depth;
	texture.mipmaps = mipmap_count;
	texture.format = p_data[0]->get_format();
	texture.validated_format = validated_format;

	texture.buffer_size_3d = all_data.size();
	texture.buffer_slices_3d = slices;

	texture.rd_type = RD::TEXTURE_TYPE_3D;
	texture.rd_format = ret_format.format;
	texture.rd_format_srgb = ret_format.format_srgb;

	RD::TextureFormat rd_format;
	RD::TextureView rd_view;
	{ //attempt register
		rd_format.format = texture.rd_format;
		rd_format.width = texture.width;
		rd_format.height = texture.height;
		rd_format.depth = texture.depth;
		rd_format.array_layers = 1;
		rd_format.mipmaps = texture.mipmaps;
		rd_format.texture_type = texture.rd_type;
		rd_format.samples = RD::TEXTURE_SAMPLES_1;
		rd_format.usage_bits = RD::TEXTURE_USAGE_SAMPLING_BIT | RD::TEXTURE_USAGE_CAN_UPDATE_BIT | RD::TEXTURE_USAGE_CAN_COPY_FROM_BIT;
		if (texture.rd_format_srgb != RD::DATA_FORMAT_MAX) {
			rd_format.shareable_formats.push_back(texture.rd_format);
			rd_format.shareable_formats.push_back(texture.rd_format_srgb);
		}
	}
	{
		rd_view.swizzle_r = ret_format.swizzle_r;
		rd_view.swizzle_g = ret_format.swizzle_g;
		rd_view.swizzle_b = ret_format.swizzle_b;
		rd_view.swizzle_a = ret_format.swizzle_a;
	}
	Vector<Vector<uint8_t>> data_slices;
	data_slices.push_back(all_data); //one slice

	texture.rd_texture = RD::get_singleton()->texture_create(rd_format, rd_view, data_slices);
	ERR_FAIL_COND(texture.rd_texture.is_null());
	if (texture.rd_format_srgb != RD::DATA_FORMAT_MAX) {
		rd_view.format_override = texture.rd_format_srgb;
		texture.rd_texture_srgb = RD::get_singleton()->texture_create_shared(rd_view, texture.rd_texture);
		if (texture.rd_texture_srgb.is_null()) {
			RD::get_singleton()->free(texture.rd_texture);
			ERR_FAIL_COND(texture.rd_texture_srgb.is_null());
		}
	}

	//used for 2D, overridable
	texture.width_2d = texture.width;
	texture.height_2d = texture.height;
	texture.is_render_target = false;
	texture.rd_view = rd_view;
	texture.is_proxy = false;

	texture_owner.initialize_rid(p_texture, texture);
}

void TextureStorage::texture_proxy_initialize(RID p_texture, RID p_base) {
	Texture *tex = texture_owner.get_or_null(p_base);
	ERR_FAIL_COND(!tex);
	Texture proxy_tex = *tex;

	proxy_tex.rd_view.format_override = tex->rd_format;
	proxy_tex.rd_texture = RD::get_singleton()->texture_create_shared(proxy_tex.rd_view, tex->rd_texture);
	if (proxy_tex.rd_texture_srgb.is_valid()) {
		proxy_tex.rd_view.format_override = tex->rd_format_srgb;
		proxy_tex.rd_texture_srgb = RD::get_singleton()->texture_create_shared(proxy_tex.rd_view, tex->rd_texture);
	}
	proxy_tex.proxy_to = p_base;
	proxy_tex.is_render_target = false;
	proxy_tex.is_proxy = true;
	proxy_tex.proxies.clear();

	texture_owner.initialize_rid(p_texture, proxy_tex);

	tex->proxies.push_back(p_texture);
}

void TextureStorage::_texture_2d_update(RID p_texture, const Ref<Image> &p_image, int p_layer, bool p_immediate) {
	ERR_FAIL_COND(p_image.is_null() || p_image->is_empty());

	Texture *tex = texture_owner.get_or_null(p_texture);
	ERR_FAIL_COND(!tex);
	ERR_FAIL_COND(tex->is_render_target);
	ERR_FAIL_COND(p_image->get_width() != tex->width || p_image->get_height() != tex->height);
	ERR_FAIL_COND(p_image->get_format() != tex->format);

	if (tex->type == TextureStorage::TYPE_LAYERED) {
		ERR_FAIL_INDEX(p_layer, tex->layers);
	}

#ifdef TOOLS_ENABLED
	tex->image_cache_2d.unref();
#endif
	TextureToRDFormat f;
	Ref<Image> validated = _validate_texture_format(p_image, f);

	RD::get_singleton()->texture_update(tex->rd_texture, p_layer, validated->get_data());
}

void TextureStorage::texture_2d_update(RID p_texture, const Ref<Image> &p_image, int p_layer) {
	_texture_2d_update(p_texture, p_image, p_layer, false);
}

void TextureStorage::texture_3d_update(RID p_texture, const Vector<Ref<Image>> &p_data) {
	Texture *tex = texture_owner.get_or_null(p_texture);
	ERR_FAIL_COND(!tex);
	ERR_FAIL_COND(tex->type != TextureStorage::TYPE_3D);

	Image::Image3DValidateError verr = Image::validate_3d_image(tex->format, tex->width, tex->height, tex->depth, tex->mipmaps > 1, p_data);
	if (verr != Image::VALIDATE_3D_OK) {
		ERR_FAIL_MSG(Image::get_3d_image_validation_error_text(verr));
	}

	Vector<uint8_t> all_data;
	{
		Vector<Ref<Image>> images;
		uint32_t all_data_size = 0;
		images.resize(p_data.size());
		for (int i = 0; i < p_data.size(); i++) {
			Ref<Image> image = p_data[i];
			if (image->get_format() != tex->validated_format) {
				image = image->duplicate();
				image->convert(tex->validated_format);
			}
			all_data_size += images[i]->get_data().size();
			images.push_back(image);
		}

		all_data.resize(all_data_size); //consolidate all data here
		uint32_t offset = 0;

		for (int i = 0; i < p_data.size(); i++) {
			uint32_t s = images[i]->get_data().size();
			memcpy(&all_data.write[offset], images[i]->get_data().ptr(), s);
			offset += s;
		}
	}

	RD::get_singleton()->texture_update(tex->rd_texture, 0, all_data);
}

void TextureStorage::texture_proxy_update(RID p_texture, RID p_proxy_to) {
	Texture *tex = texture_owner.get_or_null(p_texture);
	ERR_FAIL_COND(!tex);
	ERR_FAIL_COND(!tex->is_proxy);
	Texture *proxy_to = texture_owner.get_or_null(p_proxy_to);
	ERR_FAIL_COND(!proxy_to);
	ERR_FAIL_COND(proxy_to->is_proxy);

	if (tex->proxy_to.is_valid()) {
		//unlink proxy
		if (RD::get_singleton()->texture_is_valid(tex->rd_texture)) {
			RD::get_singleton()->free(tex->rd_texture);
			tex->rd_texture = RID();
		}
		if (RD::get_singleton()->texture_is_valid(tex->rd_texture_srgb)) {
			RD::get_singleton()->free(tex->rd_texture_srgb);
			tex->rd_texture_srgb = RID();
		}
		Texture *prev_tex = texture_owner.get_or_null(tex->proxy_to);
		ERR_FAIL_COND(!prev_tex);
		prev_tex->proxies.erase(p_texture);
	}

	*tex = *proxy_to;

	tex->proxy_to = p_proxy_to;
	tex->is_render_target = false;
	tex->is_proxy = true;
	tex->proxies.clear();
	proxy_to->proxies.push_back(p_texture);

	tex->rd_view.format_override = tex->rd_format;
	tex->rd_texture = RD::get_singleton()->texture_create_shared(tex->rd_view, proxy_to->rd_texture);
	if (tex->rd_texture_srgb.is_valid()) {
		tex->rd_view.format_override = tex->rd_format_srgb;
		tex->rd_texture_srgb = RD::get_singleton()->texture_create_shared(tex->rd_view, proxy_to->rd_texture);
	}
}

//these two APIs can be used together or in combination with the others.
void TextureStorage::texture_2d_placeholder_initialize(RID p_texture) {
	//this could be better optimized to reuse an existing image , done this way
	//for now to get it working
	Ref<Image> image;
	image.instantiate();
	image->create(4, 4, false, Image::FORMAT_RGBA8);
	image->fill(Color(1, 0, 1, 1));

	texture_2d_initialize(p_texture, image);
}

void TextureStorage::texture_2d_layered_placeholder_initialize(RID p_texture, RS::TextureLayeredType p_layered_type) {
	//this could be better optimized to reuse an existing image , done this way
	//for now to get it working
	Ref<Image> image;
	image.instantiate();
	image->create(4, 4, false, Image::FORMAT_RGBA8);
	image->fill(Color(1, 0, 1, 1));

	Vector<Ref<Image>> images;
	if (p_layered_type == RS::TEXTURE_LAYERED_2D_ARRAY) {
		images.push_back(image);
	} else {
		//cube
		for (int i = 0; i < 6; i++) {
			images.push_back(image);
		}
	}

	texture_2d_layered_initialize(p_texture, images, p_layered_type);
}

void TextureStorage::texture_3d_placeholder_initialize(RID p_texture) {
	//this could be better optimized to reuse an existing image , done this way
	//for now to get it working
	Ref<Image> image;
	image.instantiate();
	image->create(4, 4, false, Image::FORMAT_RGBA8);
	image->fill(Color(1, 0, 1, 1));

	Vector<Ref<Image>> images;
	//cube
	for (int i = 0; i < 4; i++) {
		images.push_back(image);
	}

	texture_3d_initialize(p_texture, Image::FORMAT_RGBA8, 4, 4, 4, false, images);
}

Ref<Image> TextureStorage::texture_2d_get(RID p_texture) const {
	Texture *tex = texture_owner.get_or_null(p_texture);
	ERR_FAIL_COND_V(!tex, Ref<Image>());

#ifdef TOOLS_ENABLED
	if (tex->image_cache_2d.is_valid() && !tex->is_render_target) {
		return tex->image_cache_2d;
	}
#endif
	Vector<uint8_t> data = RD::get_singleton()->texture_get_data(tex->rd_texture, 0);
	ERR_FAIL_COND_V(data.size() == 0, Ref<Image>());
	Ref<Image> image;
	image.instantiate();
	image->create(tex->width, tex->height, tex->mipmaps > 1, tex->validated_format, data);
	ERR_FAIL_COND_V(image->is_empty(), Ref<Image>());
	if (tex->format != tex->validated_format) {
		image->convert(tex->format);
	}

#ifdef TOOLS_ENABLED
	if (Engine::get_singleton()->is_editor_hint() && !tex->is_render_target) {
		tex->image_cache_2d = image;
	}
#endif

	return image;
}

Ref<Image> TextureStorage::texture_2d_layer_get(RID p_texture, int p_layer) const {
	Texture *tex = texture_owner.get_or_null(p_texture);
	ERR_FAIL_COND_V(!tex, Ref<Image>());

	Vector<uint8_t> data = RD::get_singleton()->texture_get_data(tex->rd_texture, p_layer);
	ERR_FAIL_COND_V(data.size() == 0, Ref<Image>());
	Ref<Image> image;
	image.instantiate();
	image->create(tex->width, tex->height, tex->mipmaps > 1, tex->validated_format, data);
	ERR_FAIL_COND_V(image->is_empty(), Ref<Image>());
	if (tex->format != tex->validated_format) {
		image->convert(tex->format);
	}

	return image;
}

Vector<Ref<Image>> TextureStorage::texture_3d_get(RID p_texture) const {
	Texture *tex = texture_owner.get_or_null(p_texture);
	ERR_FAIL_COND_V(!tex, Vector<Ref<Image>>());
	ERR_FAIL_COND_V(tex->type != TextureStorage::TYPE_3D, Vector<Ref<Image>>());

	Vector<uint8_t> all_data = RD::get_singleton()->texture_get_data(tex->rd_texture, 0);

	ERR_FAIL_COND_V(all_data.size() != (int)tex->buffer_size_3d, Vector<Ref<Image>>());

	Vector<Ref<Image>> ret;

	for (int i = 0; i < tex->buffer_slices_3d.size(); i++) {
		const Texture::BufferSlice3D &bs = tex->buffer_slices_3d[i];
		ERR_FAIL_COND_V(bs.offset >= (uint32_t)all_data.size(), Vector<Ref<Image>>());
		ERR_FAIL_COND_V(bs.offset + bs.buffer_size > (uint32_t)all_data.size(), Vector<Ref<Image>>());
		Vector<uint8_t> sub_region = all_data.slice(bs.offset, bs.offset + bs.buffer_size);

		Ref<Image> img;
		img.instantiate();
		img->create(bs.size.width, bs.size.height, false, tex->validated_format, sub_region);
		ERR_FAIL_COND_V(img->is_empty(), Vector<Ref<Image>>());
		if (tex->format != tex->validated_format) {
			img->convert(tex->format);
		}

		ret.push_back(img);
	}

	return ret;
}

void TextureStorage::texture_replace(RID p_texture, RID p_by_texture) {
	Texture *tex = texture_owner.get_or_null(p_texture);
	ERR_FAIL_COND(!tex);
	ERR_FAIL_COND(tex->proxy_to.is_valid()); //can't replace proxy
	Texture *by_tex = texture_owner.get_or_null(p_by_texture);
	ERR_FAIL_COND(!by_tex);
	ERR_FAIL_COND(by_tex->proxy_to.is_valid()); //can't replace proxy

	if (tex == by_tex) {
		return;
	}

	if (tex->rd_texture_srgb.is_valid()) {
		RD::get_singleton()->free(tex->rd_texture_srgb);
	}
	RD::get_singleton()->free(tex->rd_texture);

	if (tex->canvas_texture) {
		memdelete(tex->canvas_texture);
		tex->canvas_texture = nullptr;
	}

	Vector<RID> proxies_to_update = tex->proxies;
	Vector<RID> proxies_to_redirect = by_tex->proxies;

	*tex = *by_tex;

	tex->proxies = proxies_to_update; //restore proxies, so they can be updated

	if (tex->canvas_texture) {
		tex->canvas_texture->diffuse = p_texture; //update
	}

	for (int i = 0; i < proxies_to_update.size(); i++) {
		texture_proxy_update(proxies_to_update[i], p_texture);
	}
	for (int i = 0; i < proxies_to_redirect.size(); i++) {
		texture_proxy_update(proxies_to_redirect[i], p_texture);
	}
	//delete last, so proxies can be updated
	texture_owner.free(p_by_texture);

	decal_atlas_mark_dirty_on_texture(p_texture);
}

void TextureStorage::texture_set_size_override(RID p_texture, int p_width, int p_height) {
	Texture *tex = texture_owner.get_or_null(p_texture);
	ERR_FAIL_COND(!tex);
	ERR_FAIL_COND(tex->type != TextureStorage::TYPE_2D);

	tex->width_2d = p_width;
	tex->height_2d = p_height;
}

void TextureStorage::texture_set_path(RID p_texture, const String &p_path) {
	Texture *tex = texture_owner.get_or_null(p_texture);
	ERR_FAIL_COND(!tex);

	tex->path = p_path;
}

String TextureStorage::texture_get_path(RID p_texture) const {
	Texture *tex = texture_owner.get_or_null(p_texture);
	ERR_FAIL_COND_V(!tex, String());

	return tex->path;
}

void TextureStorage::texture_set_detect_3d_callback(RID p_texture, RS::TextureDetectCallback p_callback, void *p_userdata) {
	Texture *tex = texture_owner.get_or_null(p_texture);
	ERR_FAIL_COND(!tex);

	tex->detect_3d_callback_ud = p_userdata;
	tex->detect_3d_callback = p_callback;
}

void TextureStorage::texture_set_detect_normal_callback(RID p_texture, RS::TextureDetectCallback p_callback, void *p_userdata) {
	Texture *tex = texture_owner.get_or_null(p_texture);
	ERR_FAIL_COND(!tex);

	tex->detect_normal_callback_ud = p_userdata;
	tex->detect_normal_callback = p_callback;
}

void TextureStorage::texture_set_detect_roughness_callback(RID p_texture, RS::TextureDetectRoughnessCallback p_callback, void *p_userdata) {
	Texture *tex = texture_owner.get_or_null(p_texture);
	ERR_FAIL_COND(!tex);

	tex->detect_roughness_callback_ud = p_userdata;
	tex->detect_roughness_callback = p_callback;
}

void TextureStorage::texture_debug_usage(List<RS::TextureInfo> *r_info) {
}

void TextureStorage::texture_set_force_redraw_if_visible(RID p_texture, bool p_enable) {
}

Size2 TextureStorage::texture_size_with_proxy(RID p_proxy) {
	return texture_2d_get_size(p_proxy);
}

Ref<Image> TextureStorage::_validate_texture_format(const Ref<Image> &p_image, TextureToRDFormat &r_format) {
	Ref<Image> image = p_image->duplicate();

	switch (p_image->get_format()) {
		case Image::FORMAT_L8: {
			r_format.format = RD::DATA_FORMAT_R8_UNORM;
			r_format.swizzle_r = RD::TEXTURE_SWIZZLE_R;
			r_format.swizzle_g = RD::TEXTURE_SWIZZLE_R;
			r_format.swizzle_b = RD::TEXTURE_SWIZZLE_R;
			r_format.swizzle_a = RD::TEXTURE_SWIZZLE_ONE;
		} break; //luminance
		case Image::FORMAT_LA8: {
			r_format.format = RD::DATA_FORMAT_R8G8_UNORM;
			r_format.swizzle_r = RD::TEXTURE_SWIZZLE_R;
			r_format.swizzle_g = RD::TEXTURE_SWIZZLE_R;
			r_format.swizzle_b = RD::TEXTURE_SWIZZLE_R;
			r_format.swizzle_a = RD::TEXTURE_SWIZZLE_G;
		} break; //luminance-alpha
		case Image::FORMAT_R8: {
			r_format.format = RD::DATA_FORMAT_R8_UNORM;
			r_format.swizzle_r = RD::TEXTURE_SWIZZLE_R;
			r_format.swizzle_g = RD::TEXTURE_SWIZZLE_ZERO;
			r_format.swizzle_b = RD::TEXTURE_SWIZZLE_ZERO;
			r_format.swizzle_a = RD::TEXTURE_SWIZZLE_ONE;
		} break;
		case Image::FORMAT_RG8: {
			r_format.format = RD::DATA_FORMAT_R8G8_UNORM;
			r_format.swizzle_r = RD::TEXTURE_SWIZZLE_R;
			r_format.swizzle_g = RD::TEXTURE_SWIZZLE_G;
			r_format.swizzle_b = RD::TEXTURE_SWIZZLE_ZERO;
			r_format.swizzle_a = RD::TEXTURE_SWIZZLE_ONE;
		} break;
		case Image::FORMAT_RGB8: {
			//this format is not mandatory for specification, check if supported first
			if (false && RD::get_singleton()->texture_is_format_supported_for_usage(RD::DATA_FORMAT_R8G8B8_UNORM, RD::TEXTURE_USAGE_SAMPLING_BIT | RD::TEXTURE_USAGE_CAN_UPDATE_BIT) && RD::get_singleton()->texture_is_format_supported_for_usage(RD::DATA_FORMAT_R8G8B8_SRGB, RD::TEXTURE_USAGE_SAMPLING_BIT | RD::TEXTURE_USAGE_CAN_UPDATE_BIT)) {
				r_format.format = RD::DATA_FORMAT_R8G8B8_UNORM;
				r_format.format_srgb = RD::DATA_FORMAT_R8G8B8_SRGB;
			} else {
				//not supported, reconvert
				r_format.format = RD::DATA_FORMAT_R8G8B8A8_UNORM;
				r_format.format_srgb = RD::DATA_FORMAT_R8G8B8A8_SRGB;
				image->convert(Image::FORMAT_RGBA8);
			}
			r_format.swizzle_r = RD::TEXTURE_SWIZZLE_R;
			r_format.swizzle_g = RD::TEXTURE_SWIZZLE_G;
			r_format.swizzle_b = RD::TEXTURE_SWIZZLE_B;
			r_format.swizzle_a = RD::TEXTURE_SWIZZLE_ONE;

		} break;
		case Image::FORMAT_RGBA8: {
			r_format.format = RD::DATA_FORMAT_R8G8B8A8_UNORM;
			r_format.format_srgb = RD::DATA_FORMAT_R8G8B8A8_SRGB;
			r_format.swizzle_r = RD::TEXTURE_SWIZZLE_R;
			r_format.swizzle_g = RD::TEXTURE_SWIZZLE_G;
			r_format.swizzle_b = RD::TEXTURE_SWIZZLE_B;
			r_format.swizzle_a = RD::TEXTURE_SWIZZLE_A;
		} break;
		case Image::FORMAT_RGBA4444: {
			r_format.format = RD::DATA_FORMAT_B4G4R4A4_UNORM_PACK16;
			r_format.swizzle_r = RD::TEXTURE_SWIZZLE_B; //needs swizzle
			r_format.swizzle_g = RD::TEXTURE_SWIZZLE_G;
			r_format.swizzle_b = RD::TEXTURE_SWIZZLE_R;
			r_format.swizzle_a = RD::TEXTURE_SWIZZLE_A;
		} break;
		case Image::FORMAT_RGB565: {
			r_format.format = RD::DATA_FORMAT_B5G6R5_UNORM_PACK16;
			r_format.swizzle_r = RD::TEXTURE_SWIZZLE_B;
			r_format.swizzle_g = RD::TEXTURE_SWIZZLE_G;
			r_format.swizzle_b = RD::TEXTURE_SWIZZLE_R;
			r_format.swizzle_a = RD::TEXTURE_SWIZZLE_A;
		} break;
		case Image::FORMAT_RF: {
			r_format.format = RD::DATA_FORMAT_R32_SFLOAT;
			r_format.swizzle_r = RD::TEXTURE_SWIZZLE_R;
			r_format.swizzle_g = RD::TEXTURE_SWIZZLE_ZERO;
			r_format.swizzle_b = RD::TEXTURE_SWIZZLE_ZERO;
			r_format.swizzle_a = RD::TEXTURE_SWIZZLE_ONE;
		} break; //float
		case Image::FORMAT_RGF: {
			r_format.format = RD::DATA_FORMAT_R32G32_SFLOAT;
			r_format.swizzle_r = RD::TEXTURE_SWIZZLE_R;
			r_format.swizzle_g = RD::TEXTURE_SWIZZLE_G;
			r_format.swizzle_b = RD::TEXTURE_SWIZZLE_ZERO;
			r_format.swizzle_a = RD::TEXTURE_SWIZZLE_ONE;
		} break;
		case Image::FORMAT_RGBF: {
			//this format is not mandatory for specification, check if supported first
			if (RD::get_singleton()->texture_is_format_supported_for_usage(RD::DATA_FORMAT_R32G32B32_SFLOAT, RD::TEXTURE_USAGE_SAMPLING_BIT | RD::TEXTURE_USAGE_CAN_UPDATE_BIT)) {
				r_format.format = RD::DATA_FORMAT_R32G32B32_SFLOAT;
			} else {
				//not supported, reconvert
				r_format.format = RD::DATA_FORMAT_R32G32B32A32_SFLOAT;
				image->convert(Image::FORMAT_RGBAF);
			}

			r_format.swizzle_r = RD::TEXTURE_SWIZZLE_R;
			r_format.swizzle_g = RD::TEXTURE_SWIZZLE_G;
			r_format.swizzle_b = RD::TEXTURE_SWIZZLE_B;
			r_format.swizzle_a = RD::TEXTURE_SWIZZLE_ONE;
		} break;
		case Image::FORMAT_RGBAF: {
			r_format.format = RD::DATA_FORMAT_R32G32B32A32_SFLOAT;
			r_format.swizzle_r = RD::TEXTURE_SWIZZLE_R;
			r_format.swizzle_g = RD::TEXTURE_SWIZZLE_G;
			r_format.swizzle_b = RD::TEXTURE_SWIZZLE_B;
			r_format.swizzle_a = RD::TEXTURE_SWIZZLE_A;

		} break;
		case Image::FORMAT_RH: {
			r_format.format = RD::DATA_FORMAT_R16_SFLOAT;
			r_format.swizzle_r = RD::TEXTURE_SWIZZLE_R;
			r_format.swizzle_g = RD::TEXTURE_SWIZZLE_ZERO;
			r_format.swizzle_b = RD::TEXTURE_SWIZZLE_ZERO;
			r_format.swizzle_a = RD::TEXTURE_SWIZZLE_ONE;

		} break; //half float
		case Image::FORMAT_RGH: {
			r_format.format = RD::DATA_FORMAT_R16G16_SFLOAT;
			r_format.swizzle_r = RD::TEXTURE_SWIZZLE_R;
			r_format.swizzle_g = RD::TEXTURE_SWIZZLE_G;
			r_format.swizzle_b = RD::TEXTURE_SWIZZLE_ZERO;
			r_format.swizzle_a = RD::TEXTURE_SWIZZLE_ONE;

		} break;
		case Image::FORMAT_RGBH: {
			//this format is not mandatory for specification, check if supported first
			if (RD::get_singleton()->texture_is_format_supported_for_usage(RD::DATA_FORMAT_R16G16B16_SFLOAT, RD::TEXTURE_USAGE_SAMPLING_BIT | RD::TEXTURE_USAGE_CAN_UPDATE_BIT)) {
				r_format.format = RD::DATA_FORMAT_R16G16B16_SFLOAT;
			} else {
				//not supported, reconvert
				r_format.format = RD::DATA_FORMAT_R16G16B16A16_SFLOAT;
				image->convert(Image::FORMAT_RGBAH);
			}

			r_format.swizzle_r = RD::TEXTURE_SWIZZLE_R;
			r_format.swizzle_g = RD::TEXTURE_SWIZZLE_G;
			r_format.swizzle_b = RD::TEXTURE_SWIZZLE_B;
			r_format.swizzle_a = RD::TEXTURE_SWIZZLE_ONE;
		} break;
		case Image::FORMAT_RGBAH: {
			r_format.format = RD::DATA_FORMAT_R16G16B16A16_SFLOAT;
			r_format.swizzle_r = RD::TEXTURE_SWIZZLE_R;
			r_format.swizzle_g = RD::TEXTURE_SWIZZLE_G;
			r_format.swizzle_b = RD::TEXTURE_SWIZZLE_B;
			r_format.swizzle_a = RD::TEXTURE_SWIZZLE_A;

		} break;
		case Image::FORMAT_RGBE9995: {
			r_format.format = RD::DATA_FORMAT_E5B9G9R9_UFLOAT_PACK32;
#ifndef _MSC_VER
#warning TODO need to make a function in Image to swap bits for this
#endif
			r_format.swizzle_r = RD::TEXTURE_SWIZZLE_IDENTITY;
			r_format.swizzle_g = RD::TEXTURE_SWIZZLE_IDENTITY;
			r_format.swizzle_b = RD::TEXTURE_SWIZZLE_IDENTITY;
			r_format.swizzle_a = RD::TEXTURE_SWIZZLE_IDENTITY;
		} break;
		case Image::FORMAT_DXT1: {
			if (RD::get_singleton()->texture_is_format_supported_for_usage(RD::DATA_FORMAT_BC1_RGB_UNORM_BLOCK, RD::TEXTURE_USAGE_SAMPLING_BIT | RD::TEXTURE_USAGE_CAN_UPDATE_BIT)) {
				r_format.format = RD::DATA_FORMAT_BC1_RGB_UNORM_BLOCK;
				r_format.format_srgb = RD::DATA_FORMAT_BC1_RGB_SRGB_BLOCK;
			} else {
				//not supported, reconvert
				r_format.format = RD::DATA_FORMAT_R8G8B8A8_UNORM;
				r_format.format_srgb = RD::DATA_FORMAT_R8G8B8A8_SRGB;
				image->decompress();
				image->convert(Image::FORMAT_RGBA8);
			}
			r_format.swizzle_r = RD::TEXTURE_SWIZZLE_R;
			r_format.swizzle_g = RD::TEXTURE_SWIZZLE_G;
			r_format.swizzle_b = RD::TEXTURE_SWIZZLE_B;
			r_format.swizzle_a = RD::TEXTURE_SWIZZLE_ONE;

		} break; //s3tc bc1
		case Image::FORMAT_DXT3: {
			if (RD::get_singleton()->texture_is_format_supported_for_usage(RD::DATA_FORMAT_BC2_UNORM_BLOCK, RD::TEXTURE_USAGE_SAMPLING_BIT | RD::TEXTURE_USAGE_CAN_UPDATE_BIT)) {
				r_format.format = RD::DATA_FORMAT_BC2_UNORM_BLOCK;
				r_format.format_srgb = RD::DATA_FORMAT_BC2_SRGB_BLOCK;
			} else {
				//not supported, reconvert
				r_format.format = RD::DATA_FORMAT_R8G8B8A8_UNORM;
				r_format.format_srgb = RD::DATA_FORMAT_R8G8B8A8_SRGB;
				image->decompress();
				image->convert(Image::FORMAT_RGBA8);
			}
			r_format.swizzle_r = RD::TEXTURE_SWIZZLE_R;
			r_format.swizzle_g = RD::TEXTURE_SWIZZLE_G;
			r_format.swizzle_b = RD::TEXTURE_SWIZZLE_B;
			r_format.swizzle_a = RD::TEXTURE_SWIZZLE_A;

		} break; //bc2
		case Image::FORMAT_DXT5: {
			if (RD::get_singleton()->texture_is_format_supported_for_usage(RD::DATA_FORMAT_BC3_UNORM_BLOCK, RD::TEXTURE_USAGE_SAMPLING_BIT | RD::TEXTURE_USAGE_CAN_UPDATE_BIT)) {
				r_format.format = RD::DATA_FORMAT_BC3_UNORM_BLOCK;
				r_format.format_srgb = RD::DATA_FORMAT_BC3_SRGB_BLOCK;
			} else {
				//not supported, reconvert
				r_format.format = RD::DATA_FORMAT_R8G8B8A8_UNORM;
				r_format.format_srgb = RD::DATA_FORMAT_R8G8B8A8_SRGB;
				image->decompress();
				image->convert(Image::FORMAT_RGBA8);
			}
			r_format.swizzle_r = RD::TEXTURE_SWIZZLE_R;
			r_format.swizzle_g = RD::TEXTURE_SWIZZLE_G;
			r_format.swizzle_b = RD::TEXTURE_SWIZZLE_B;
			r_format.swizzle_a = RD::TEXTURE_SWIZZLE_A;
		} break; //bc3
		case Image::FORMAT_RGTC_R: {
			if (RD::get_singleton()->texture_is_format_supported_for_usage(RD::DATA_FORMAT_BC4_UNORM_BLOCK, RD::TEXTURE_USAGE_SAMPLING_BIT | RD::TEXTURE_USAGE_CAN_UPDATE_BIT)) {
				r_format.format = RD::DATA_FORMAT_BC4_UNORM_BLOCK;
			} else {
				//not supported, reconvert
				r_format.format = RD::DATA_FORMAT_R8_UNORM;
				image->decompress();
				image->convert(Image::FORMAT_R8);
			}
			r_format.swizzle_r = RD::TEXTURE_SWIZZLE_R;
			r_format.swizzle_g = RD::TEXTURE_SWIZZLE_ZERO;
			r_format.swizzle_b = RD::TEXTURE_SWIZZLE_ZERO;
			r_format.swizzle_a = RD::TEXTURE_SWIZZLE_ONE;

		} break;
		case Image::FORMAT_RGTC_RG: {
			if (RD::get_singleton()->texture_is_format_supported_for_usage(RD::DATA_FORMAT_BC5_UNORM_BLOCK, RD::TEXTURE_USAGE_SAMPLING_BIT | RD::TEXTURE_USAGE_CAN_UPDATE_BIT)) {
				r_format.format = RD::DATA_FORMAT_BC5_UNORM_BLOCK;
			} else {
				//not supported, reconvert
				r_format.format = RD::DATA_FORMAT_R8G8_UNORM;
				image->decompress();
				image->convert(Image::FORMAT_RG8);
			}
			r_format.swizzle_r = RD::TEXTURE_SWIZZLE_R;
			r_format.swizzle_g = RD::TEXTURE_SWIZZLE_G;
			r_format.swizzle_b = RD::TEXTURE_SWIZZLE_ZERO;
			r_format.swizzle_a = RD::TEXTURE_SWIZZLE_ONE;

		} break;
		case Image::FORMAT_BPTC_RGBA: {
			if (RD::get_singleton()->texture_is_format_supported_for_usage(RD::DATA_FORMAT_BC7_UNORM_BLOCK, RD::TEXTURE_USAGE_SAMPLING_BIT | RD::TEXTURE_USAGE_CAN_UPDATE_BIT)) {
				r_format.format = RD::DATA_FORMAT_BC7_UNORM_BLOCK;
				r_format.format_srgb = RD::DATA_FORMAT_BC7_SRGB_BLOCK;
			} else {
				//not supported, reconvert
				r_format.format = RD::DATA_FORMAT_R8G8B8A8_UNORM;
				r_format.format_srgb = RD::DATA_FORMAT_R8G8B8A8_SRGB;
				image->decompress();
				image->convert(Image::FORMAT_RGBA8);
			}
			r_format.swizzle_r = RD::TEXTURE_SWIZZLE_R;
			r_format.swizzle_g = RD::TEXTURE_SWIZZLE_G;
			r_format.swizzle_b = RD::TEXTURE_SWIZZLE_B;
			r_format.swizzle_a = RD::TEXTURE_SWIZZLE_A;

		} break; //btpc bc7
		case Image::FORMAT_BPTC_RGBF: {
			if (RD::get_singleton()->texture_is_format_supported_for_usage(RD::DATA_FORMAT_BC6H_SFLOAT_BLOCK, RD::TEXTURE_USAGE_SAMPLING_BIT | RD::TEXTURE_USAGE_CAN_UPDATE_BIT)) {
				r_format.format = RD::DATA_FORMAT_BC6H_SFLOAT_BLOCK;
			} else {
				//not supported, reconvert
				r_format.format = RD::DATA_FORMAT_R16G16B16A16_SFLOAT;
				image->decompress();
				image->convert(Image::FORMAT_RGBAH);
			}
			r_format.swizzle_r = RD::TEXTURE_SWIZZLE_R;
			r_format.swizzle_g = RD::TEXTURE_SWIZZLE_G;
			r_format.swizzle_b = RD::TEXTURE_SWIZZLE_B;
			r_format.swizzle_a = RD::TEXTURE_SWIZZLE_ONE;
		} break; //float bc6h
		case Image::FORMAT_BPTC_RGBFU: {
			if (RD::get_singleton()->texture_is_format_supported_for_usage(RD::DATA_FORMAT_BC6H_UFLOAT_BLOCK, RD::TEXTURE_USAGE_SAMPLING_BIT | RD::TEXTURE_USAGE_CAN_UPDATE_BIT)) {
				r_format.format = RD::DATA_FORMAT_BC6H_UFLOAT_BLOCK;
			} else {
				//not supported, reconvert
				r_format.format = RD::DATA_FORMAT_R16G16B16A16_SFLOAT;
				image->decompress();
				image->convert(Image::FORMAT_RGBAH);
			}
			r_format.swizzle_r = RD::TEXTURE_SWIZZLE_R;
			r_format.swizzle_g = RD::TEXTURE_SWIZZLE_G;
			r_format.swizzle_b = RD::TEXTURE_SWIZZLE_B;
			r_format.swizzle_a = RD::TEXTURE_SWIZZLE_ONE;
		} break; //unsigned float bc6hu
		case Image::FORMAT_ETC2_R11: {
			if (RD::get_singleton()->texture_is_format_supported_for_usage(RD::DATA_FORMAT_EAC_R11_UNORM_BLOCK, RD::TEXTURE_USAGE_SAMPLING_BIT | RD::TEXTURE_USAGE_CAN_UPDATE_BIT)) {
				r_format.format = RD::DATA_FORMAT_EAC_R11_UNORM_BLOCK;
			} else {
				//not supported, reconvert
				r_format.format = RD::DATA_FORMAT_R8_UNORM;
				image->decompress();
				image->convert(Image::FORMAT_R8);
			}
			r_format.swizzle_r = RD::TEXTURE_SWIZZLE_R;
			r_format.swizzle_g = RD::TEXTURE_SWIZZLE_ZERO;
			r_format.swizzle_b = RD::TEXTURE_SWIZZLE_ZERO;
			r_format.swizzle_a = RD::TEXTURE_SWIZZLE_ONE;

		} break; //etc2
		case Image::FORMAT_ETC2_R11S: {
			if (RD::get_singleton()->texture_is_format_supported_for_usage(RD::DATA_FORMAT_EAC_R11_SNORM_BLOCK, RD::TEXTURE_USAGE_SAMPLING_BIT | RD::TEXTURE_USAGE_CAN_UPDATE_BIT)) {
				r_format.format = RD::DATA_FORMAT_EAC_R11_SNORM_BLOCK;
			} else {
				//not supported, reconvert
				r_format.format = RD::DATA_FORMAT_R8_SNORM;
				image->decompress();
				image->convert(Image::FORMAT_R8);
			}
			r_format.swizzle_r = RD::TEXTURE_SWIZZLE_R;
			r_format.swizzle_g = RD::TEXTURE_SWIZZLE_ZERO;
			r_format.swizzle_b = RD::TEXTURE_SWIZZLE_ZERO;
			r_format.swizzle_a = RD::TEXTURE_SWIZZLE_ONE;
		} break; //signed: {} break; NOT srgb.
		case Image::FORMAT_ETC2_RG11: {
			if (RD::get_singleton()->texture_is_format_supported_for_usage(RD::DATA_FORMAT_EAC_R11G11_UNORM_BLOCK, RD::TEXTURE_USAGE_SAMPLING_BIT | RD::TEXTURE_USAGE_CAN_UPDATE_BIT)) {
				r_format.format = RD::DATA_FORMAT_EAC_R11G11_UNORM_BLOCK;
			} else {
				//not supported, reconvert
				r_format.format = RD::DATA_FORMAT_R8G8_UNORM;
				image->decompress();
				image->convert(Image::FORMAT_RG8);
			}
			r_format.swizzle_r = RD::TEXTURE_SWIZZLE_R;
			r_format.swizzle_g = RD::TEXTURE_SWIZZLE_G;
			r_format.swizzle_b = RD::TEXTURE_SWIZZLE_ZERO;
			r_format.swizzle_a = RD::TEXTURE_SWIZZLE_ONE;
		} break;
		case Image::FORMAT_ETC2_RG11S: {
			if (RD::get_singleton()->texture_is_format_supported_for_usage(RD::DATA_FORMAT_EAC_R11G11_SNORM_BLOCK, RD::TEXTURE_USAGE_SAMPLING_BIT | RD::TEXTURE_USAGE_CAN_UPDATE_BIT)) {
				r_format.format = RD::DATA_FORMAT_EAC_R11G11_SNORM_BLOCK;
			} else {
				//not supported, reconvert
				r_format.format = RD::DATA_FORMAT_R8G8_SNORM;
				image->decompress();
				image->convert(Image::FORMAT_RG8);
			}
			r_format.swizzle_r = RD::TEXTURE_SWIZZLE_R;
			r_format.swizzle_g = RD::TEXTURE_SWIZZLE_G;
			r_format.swizzle_b = RD::TEXTURE_SWIZZLE_ZERO;
			r_format.swizzle_a = RD::TEXTURE_SWIZZLE_ONE;
		} break;
		case Image::FORMAT_ETC:
		case Image::FORMAT_ETC2_RGB8: {
			//ETC2 is backwards compatible with ETC1, and all modern platforms support it
			if (RD::get_singleton()->texture_is_format_supported_for_usage(RD::DATA_FORMAT_ETC2_R8G8B8_UNORM_BLOCK, RD::TEXTURE_USAGE_SAMPLING_BIT | RD::TEXTURE_USAGE_CAN_UPDATE_BIT)) {
				r_format.format = RD::DATA_FORMAT_ETC2_R8G8B8_UNORM_BLOCK;
				r_format.format_srgb = RD::DATA_FORMAT_ETC2_R8G8B8_SRGB_BLOCK;
			} else {
				//not supported, reconvert
				r_format.format = RD::DATA_FORMAT_R8G8B8A8_UNORM;
				r_format.format_srgb = RD::DATA_FORMAT_R8G8B8A8_SRGB;
				image->decompress();
				image->convert(Image::FORMAT_RGBA8);
			}
			r_format.swizzle_r = RD::TEXTURE_SWIZZLE_R;
			r_format.swizzle_g = RD::TEXTURE_SWIZZLE_G;
			r_format.swizzle_b = RD::TEXTURE_SWIZZLE_B;
			r_format.swizzle_a = RD::TEXTURE_SWIZZLE_ONE;

		} break;
		case Image::FORMAT_ETC2_RGBA8: {
			if (RD::get_singleton()->texture_is_format_supported_for_usage(RD::DATA_FORMAT_ETC2_R8G8B8A8_UNORM_BLOCK, RD::TEXTURE_USAGE_SAMPLING_BIT | RD::TEXTURE_USAGE_CAN_UPDATE_BIT)) {
				r_format.format = RD::DATA_FORMAT_ETC2_R8G8B8A8_UNORM_BLOCK;
				r_format.format_srgb = RD::DATA_FORMAT_ETC2_R8G8B8A8_SRGB_BLOCK;
			} else {
				//not supported, reconvert
				r_format.format = RD::DATA_FORMAT_R8G8B8A8_UNORM;
				r_format.format_srgb = RD::DATA_FORMAT_R8G8B8A8_SRGB;
				image->decompress();
				image->convert(Image::FORMAT_RGBA8);
			}
			r_format.swizzle_r = RD::TEXTURE_SWIZZLE_R;
			r_format.swizzle_g = RD::TEXTURE_SWIZZLE_G;
			r_format.swizzle_b = RD::TEXTURE_SWIZZLE_B;
			r_format.swizzle_a = RD::TEXTURE_SWIZZLE_A;
		} break;
		case Image::FORMAT_ETC2_RGB8A1: {
			if (RD::get_singleton()->texture_is_format_supported_for_usage(RD::DATA_FORMAT_ETC2_R8G8B8A1_UNORM_BLOCK, RD::TEXTURE_USAGE_SAMPLING_BIT | RD::TEXTURE_USAGE_CAN_UPDATE_BIT)) {
				r_format.format = RD::DATA_FORMAT_ETC2_R8G8B8A1_UNORM_BLOCK;
				r_format.format_srgb = RD::DATA_FORMAT_ETC2_R8G8B8A1_SRGB_BLOCK;
			} else {
				//not supported, reconvert
				r_format.format = RD::DATA_FORMAT_R8G8B8A8_UNORM;
				r_format.format_srgb = RD::DATA_FORMAT_R8G8B8A8_SRGB;
				image->decompress();
				image->convert(Image::FORMAT_RGBA8);
			}
			r_format.swizzle_r = RD::TEXTURE_SWIZZLE_R;
			r_format.swizzle_g = RD::TEXTURE_SWIZZLE_G;
			r_format.swizzle_b = RD::TEXTURE_SWIZZLE_B;
			r_format.swizzle_a = RD::TEXTURE_SWIZZLE_A;
		} break;
		case Image::FORMAT_ETC2_RA_AS_RG: {
			if (RD::get_singleton()->texture_is_format_supported_for_usage(RD::DATA_FORMAT_ETC2_R8G8B8A8_UNORM_BLOCK, RD::TEXTURE_USAGE_SAMPLING_BIT | RD::TEXTURE_USAGE_CAN_UPDATE_BIT)) {
				r_format.format = RD::DATA_FORMAT_ETC2_R8G8B8A8_UNORM_BLOCK;
				r_format.format_srgb = RD::DATA_FORMAT_ETC2_R8G8B8A8_SRGB_BLOCK;
			} else {
				//not supported, reconvert
				r_format.format = RD::DATA_FORMAT_R8G8B8A8_UNORM;
				r_format.format_srgb = RD::DATA_FORMAT_R8G8B8A8_SRGB;
				image->decompress();
				image->convert(Image::FORMAT_RGBA8);
			}
			r_format.swizzle_r = RD::TEXTURE_SWIZZLE_R;
			r_format.swizzle_g = RD::TEXTURE_SWIZZLE_A;
			r_format.swizzle_b = RD::TEXTURE_SWIZZLE_ZERO;
			r_format.swizzle_a = RD::TEXTURE_SWIZZLE_ONE;
		} break;
		case Image::FORMAT_DXT5_RA_AS_RG: {
			if (RD::get_singleton()->texture_is_format_supported_for_usage(RD::DATA_FORMAT_BC3_UNORM_BLOCK, RD::TEXTURE_USAGE_SAMPLING_BIT | RD::TEXTURE_USAGE_CAN_UPDATE_BIT)) {
				r_format.format = RD::DATA_FORMAT_BC3_UNORM_BLOCK;
				r_format.format_srgb = RD::DATA_FORMAT_BC3_SRGB_BLOCK;
			} else {
				//not supported, reconvert
				r_format.format = RD::DATA_FORMAT_R8G8B8A8_UNORM;
				r_format.format_srgb = RD::DATA_FORMAT_R8G8B8A8_SRGB;
				image->decompress();
				image->convert(Image::FORMAT_RGBA8);
			}
			r_format.swizzle_r = RD::TEXTURE_SWIZZLE_R;
			r_format.swizzle_g = RD::TEXTURE_SWIZZLE_A;
			r_format.swizzle_b = RD::TEXTURE_SWIZZLE_ZERO;
			r_format.swizzle_a = RD::TEXTURE_SWIZZLE_ONE;
		} break;

		default: {
		}
	}

	return image;
}

/* DECAL API */

RID TextureStorage::decal_atlas_get_texture() const {
	return decal_atlas.texture;
}

RID TextureStorage::decal_atlas_get_texture_srgb() const {
	return decal_atlas.texture_srgb;
}

RID TextureStorage::decal_allocate() {
	return decal_owner.allocate_rid();
}

void TextureStorage::decal_initialize(RID p_decal) {
	decal_owner.initialize_rid(p_decal, Decal());
}

void TextureStorage::decal_free(RID p_rid) {
	Decal *decal = decal_owner.get_or_null(p_rid);
	for (int i = 0; i < RS::DECAL_TEXTURE_MAX; i++) {
		if (decal->textures[i].is_valid() && owns_texture(decal->textures[i])) {
			texture_remove_from_decal_atlas(decal->textures[i]);
		}
	}
	decal->dependency.deleted_notify(p_rid);
	decal_owner.free(p_rid);
}

void TextureStorage::decal_set_extents(RID p_decal, const Vector3 &p_extents) {
	Decal *decal = decal_owner.get_or_null(p_decal);
	ERR_FAIL_COND(!decal);
	decal->extents = p_extents;
	decal->dependency.changed_notify(Dependency::DEPENDENCY_CHANGED_AABB);
}

void TextureStorage::decal_set_texture(RID p_decal, RS::DecalTexture p_type, RID p_texture) {
	Decal *decal = decal_owner.get_or_null(p_decal);
	ERR_FAIL_COND(!decal);
	ERR_FAIL_INDEX(p_type, RS::DECAL_TEXTURE_MAX);

	if (decal->textures[p_type] == p_texture) {
		return;
	}

	ERR_FAIL_COND(p_texture.is_valid() && !owns_texture(p_texture));

	if (decal->textures[p_type].is_valid() && owns_texture(decal->textures[p_type])) {
		texture_remove_from_decal_atlas(decal->textures[p_type]);
	}

	decal->textures[p_type] = p_texture;

	if (decal->textures[p_type].is_valid()) {
		texture_add_to_decal_atlas(decal->textures[p_type]);
	}

	decal->dependency.changed_notify(Dependency::DEPENDENCY_CHANGED_DECAL);
}

void TextureStorage::decal_set_emission_energy(RID p_decal, float p_energy) {
	Decal *decal = decal_owner.get_or_null(p_decal);
	ERR_FAIL_COND(!decal);
	decal->emission_energy = p_energy;
}

void TextureStorage::decal_set_albedo_mix(RID p_decal, float p_mix) {
	Decal *decal = decal_owner.get_or_null(p_decal);
	ERR_FAIL_COND(!decal);
	decal->albedo_mix = p_mix;
}

void TextureStorage::decal_set_modulate(RID p_decal, const Color &p_modulate) {
	Decal *decal = decal_owner.get_or_null(p_decal);
	ERR_FAIL_COND(!decal);
	decal->modulate = p_modulate;
}

void TextureStorage::decal_set_cull_mask(RID p_decal, uint32_t p_layers) {
	Decal *decal = decal_owner.get_or_null(p_decal);
	ERR_FAIL_COND(!decal);
	decal->cull_mask = p_layers;
	decal->dependency.changed_notify(Dependency::DEPENDENCY_CHANGED_AABB);
}

void TextureStorage::decal_set_distance_fade(RID p_decal, bool p_enabled, float p_begin, float p_length) {
	Decal *decal = decal_owner.get_or_null(p_decal);
	ERR_FAIL_COND(!decal);
	decal->distance_fade = p_enabled;
	decal->distance_fade_begin = p_begin;
	decal->distance_fade_length = p_length;
}

void TextureStorage::decal_set_fade(RID p_decal, float p_above, float p_below) {
	Decal *decal = decal_owner.get_or_null(p_decal);
	ERR_FAIL_COND(!decal);
	decal->upper_fade = p_above;
	decal->lower_fade = p_below;
}

void TextureStorage::decal_set_normal_fade(RID p_decal, float p_fade) {
	Decal *decal = decal_owner.get_or_null(p_decal);
	ERR_FAIL_COND(!decal);
	decal->normal_fade = p_fade;
}

void TextureStorage::decal_atlas_mark_dirty_on_texture(RID p_texture) {
	if (decal_atlas.textures.has(p_texture)) {
		//belongs to decal atlas..

		decal_atlas.dirty = true; //mark it dirty since it was most likely modified
	}
}

void TextureStorage::decal_atlas_remove_texture(RID p_texture) {
	if (decal_atlas.textures.has(p_texture)) {
		decal_atlas.textures.erase(p_texture);
		//there is not much a point of making it dirty, just let it be.
	}
}

AABB TextureStorage::decal_get_aabb(RID p_decal) const {
	Decal *decal = decal_owner.get_or_null(p_decal);
	ERR_FAIL_COND_V(!decal, AABB());

	return AABB(-decal->extents, decal->extents * 2.0);
}

Dependency *TextureStorage::decal_get_dependency(RID p_decal) {
	Decal *decal = decal_owner.get_or_null(p_decal);
	ERR_FAIL_COND_V(!decal, nullptr);

	return &decal->dependency;
}

void TextureStorage::update_decal_atlas() {
	RendererRD::CopyEffects *copy_effects = RendererRD::CopyEffects::get_singleton();
	ERR_FAIL_NULL(copy_effects);

	if (!decal_atlas.dirty) {
		return; //nothing to do
	}

	decal_atlas.dirty = false;

	if (decal_atlas.texture.is_valid()) {
		RD::get_singleton()->free(decal_atlas.texture);
		decal_atlas.texture = RID();
		decal_atlas.texture_srgb = RID();
		decal_atlas.texture_mipmaps.clear();
	}

	int border = 1 << decal_atlas.mipmaps;

	if (decal_atlas.textures.size()) {
		//generate atlas
		Vector<DecalAtlas::SortItem> itemsv;
		itemsv.resize(decal_atlas.textures.size());
		int base_size = 8;

		int idx = 0;

		for (const KeyValue<RID, DecalAtlas::Texture> &E : decal_atlas.textures) {
			DecalAtlas::SortItem &si = itemsv.write[idx];

			Texture *src_tex = get_texture(E.key);

			si.size.width = (src_tex->width / border) + 1;
			si.size.height = (src_tex->height / border) + 1;
			si.pixel_size = Size2i(src_tex->width, src_tex->height);

			if (base_size < si.size.width) {
				base_size = nearest_power_of_2_templated(si.size.width);
			}

			si.texture = E.key;
			idx++;
		}

		//sort items by size
		itemsv.sort();

		//attempt to create atlas
		int item_count = itemsv.size();
		DecalAtlas::SortItem *items = itemsv.ptrw();

		int atlas_height = 0;

		while (true) {
			Vector<int> v_offsetsv;
			v_offsetsv.resize(base_size);

			int *v_offsets = v_offsetsv.ptrw();
			memset(v_offsets, 0, sizeof(int) * base_size);

			int max_height = 0;

			for (int i = 0; i < item_count; i++) {
				//best fit
				DecalAtlas::SortItem &si = items[i];
				int best_idx = -1;
				int best_height = 0x7FFFFFFF;
				for (int j = 0; j <= base_size - si.size.width; j++) {
					int height = 0;
					for (int k = 0; k < si.size.width; k++) {
						int h = v_offsets[k + j];
						if (h > height) {
							height = h;
							if (height > best_height) {
								break; //already bad
							}
						}
					}

					if (height < best_height) {
						best_height = height;
						best_idx = j;
					}
				}

				//update
				for (int k = 0; k < si.size.width; k++) {
					v_offsets[k + best_idx] = best_height + si.size.height;
				}

				si.pos.x = best_idx;
				si.pos.y = best_height;

				if (si.pos.y + si.size.height > max_height) {
					max_height = si.pos.y + si.size.height;
				}
			}

			if (max_height <= base_size * 2) {
				atlas_height = max_height;
				break; //good ratio, break;
			}

			base_size *= 2;
		}

		decal_atlas.size.width = base_size * border;
		decal_atlas.size.height = nearest_power_of_2_templated(atlas_height * border);

		for (int i = 0; i < item_count; i++) {
			DecalAtlas::Texture *t = decal_atlas.textures.getptr(items[i].texture);
			t->uv_rect.position = items[i].pos * border + Vector2i(border / 2, border / 2);
			t->uv_rect.size = items[i].pixel_size;

			t->uv_rect.position /= Size2(decal_atlas.size);
			t->uv_rect.size /= Size2(decal_atlas.size);
		}
	} else {
		//use border as size, so it at least has enough mipmaps
		decal_atlas.size.width = border;
		decal_atlas.size.height = border;
	}

	//blit textures

	RD::TextureFormat tformat;
	tformat.format = RD::DATA_FORMAT_R8G8B8A8_UNORM;
	tformat.width = decal_atlas.size.width;
	tformat.height = decal_atlas.size.height;
	tformat.usage_bits = RD::TEXTURE_USAGE_SAMPLING_BIT | RD::TEXTURE_USAGE_COLOR_ATTACHMENT_BIT | RD::TEXTURE_USAGE_CAN_COPY_TO_BIT;
	tformat.texture_type = RD::TEXTURE_TYPE_2D;
	tformat.mipmaps = decal_atlas.mipmaps;
	tformat.shareable_formats.push_back(RD::DATA_FORMAT_R8G8B8A8_UNORM);
	tformat.shareable_formats.push_back(RD::DATA_FORMAT_R8G8B8A8_SRGB);

	decal_atlas.texture = RD::get_singleton()->texture_create(tformat, RD::TextureView());
	RD::get_singleton()->texture_clear(decal_atlas.texture, Color(0, 0, 0, 0), 0, decal_atlas.mipmaps, 0, 1);

	{
		//create the framebuffer

		Size2i s = decal_atlas.size;

		for (int i = 0; i < decal_atlas.mipmaps; i++) {
			DecalAtlas::MipMap mm;
			mm.texture = RD::get_singleton()->texture_create_shared_from_slice(RD::TextureView(), decal_atlas.texture, 0, i);
			Vector<RID> fb;
			fb.push_back(mm.texture);
			mm.fb = RD::get_singleton()->framebuffer_create(fb);
			mm.size = s;
			decal_atlas.texture_mipmaps.push_back(mm);

			s.width = MAX(1, s.width >> 1);
			s.height = MAX(1, s.height >> 1);
		}
		{
			//create the SRGB variant
			RD::TextureView rd_view;
			rd_view.format_override = RD::DATA_FORMAT_R8G8B8A8_SRGB;
			decal_atlas.texture_srgb = RD::get_singleton()->texture_create_shared(rd_view, decal_atlas.texture);
		}
	}

	RID prev_texture;
	for (int i = 0; i < decal_atlas.texture_mipmaps.size(); i++) {
		const DecalAtlas::MipMap &mm = decal_atlas.texture_mipmaps[i];

		Color clear_color(0, 0, 0, 0);

		if (decal_atlas.textures.size()) {
			if (i == 0) {
				Vector<Color> cc;
				cc.push_back(clear_color);

				RD::DrawListID draw_list = RD::get_singleton()->draw_list_begin(mm.fb, RD::INITIAL_ACTION_CLEAR, RD::FINAL_ACTION_READ, RD::INITIAL_ACTION_DROP, RD::FINAL_ACTION_DISCARD, cc);

				for (const KeyValue<RID, DecalAtlas::Texture> &E : decal_atlas.textures) {
					DecalAtlas::Texture *t = decal_atlas.textures.getptr(E.key);
					Texture *src_tex = get_texture(E.key);
					copy_effects->copy_to_atlas_fb(src_tex->rd_texture, mm.fb, t->uv_rect, draw_list, false, t->panorama_to_dp_users > 0);
				}

				RD::get_singleton()->draw_list_end();

				prev_texture = mm.texture;
			} else {
				copy_effects->copy_to_fb_rect(prev_texture, mm.fb, Rect2i(Point2i(), mm.size));
				prev_texture = mm.texture;
			}
		} else {
			RD::get_singleton()->texture_clear(mm.texture, clear_color, 0, 1, 0, 1);
		}
	}
}

void TextureStorage::texture_add_to_decal_atlas(RID p_texture, bool p_panorama_to_dp) {
	if (!decal_atlas.textures.has(p_texture)) {
		DecalAtlas::Texture t;
		t.users = 1;
		t.panorama_to_dp_users = p_panorama_to_dp ? 1 : 0;
		decal_atlas.textures[p_texture] = t;
		decal_atlas.dirty = true;
	} else {
		DecalAtlas::Texture *t = decal_atlas.textures.getptr(p_texture);
		t->users++;
		if (p_panorama_to_dp) {
			t->panorama_to_dp_users++;
		}
	}
}

void TextureStorage::texture_remove_from_decal_atlas(RID p_texture, bool p_panorama_to_dp) {
	DecalAtlas::Texture *t = decal_atlas.textures.getptr(p_texture);
	ERR_FAIL_COND(!t);
	t->users--;
	if (p_panorama_to_dp) {
		ERR_FAIL_COND(t->panorama_to_dp_users == 0);
		t->panorama_to_dp_users--;
	}
	if (t->users == 0) {
		decal_atlas.textures.erase(p_texture);
		//do not mark it dirty, there is no need to since it remains working
	}
}

/* RENDER TARGET API */

void TextureStorage::_clear_render_target(RenderTarget *rt) {
	//free in reverse dependency order
	if (rt->framebuffer.is_valid()) {
		RD::get_singleton()->free(rt->framebuffer);
		rt->framebuffer_uniform_set = RID(); //chain deleted
	}

	if (rt->color.is_valid()) {
		RD::get_singleton()->free(rt->color);
	}
	rt->color_slices.clear(); // these are automatically freed.

	if (rt->color_multisample.is_valid()) {
		RD::get_singleton()->free(rt->color_multisample);
	}

	if (rt->backbuffer.is_valid()) {
		RD::get_singleton()->free(rt->backbuffer);
		rt->backbuffer = RID();
		rt->backbuffer_mipmaps.clear();
		rt->backbuffer_uniform_set = RID(); //chain deleted
	}

	_render_target_clear_sdf(rt);

	rt->framebuffer = RID();
	rt->color = RID();
	rt->color_multisample = RID();
}

void TextureStorage::_update_render_target(RenderTarget *rt) {
	if (rt->texture.is_null()) {
		//create a placeholder until updated
		rt->texture = texture_allocate();
		texture_2d_placeholder_initialize(rt->texture);
		Texture *tex = get_texture(rt->texture);
		tex->is_render_target = true;
	}

	_clear_render_target(rt);

	if (rt->size.width == 0 || rt->size.height == 0) {
		return;
	}
	//until we implement support for HDR monitors (and render target is attached to screen), this is enough.
	rt->color_format = RD::DATA_FORMAT_R8G8B8A8_UNORM;
	rt->color_format_srgb = RD::DATA_FORMAT_R8G8B8A8_SRGB;
	rt->image_format = rt->is_transparent ? Image::FORMAT_RGBA8 : Image::FORMAT_RGB8;

	RD::TextureFormat rd_color_attachment_format;
	RD::TextureView rd_view;
	{ //attempt register
		rd_color_attachment_format.format = rt->color_format;
		rd_color_attachment_format.width = rt->size.width;
		rd_color_attachment_format.height = rt->size.height;
		rd_color_attachment_format.depth = 1;
		rd_color_attachment_format.array_layers = rt->view_count; // for stereo we create two (or more) layers, need to see if we can make fallback work like this too if we don't have multiview
		rd_color_attachment_format.mipmaps = 1;
		if (rd_color_attachment_format.array_layers > 1) { // why are we not using rt->texture_type ??
			rd_color_attachment_format.texture_type = RD::TEXTURE_TYPE_2D_ARRAY;
		} else {
			rd_color_attachment_format.texture_type = RD::TEXTURE_TYPE_2D;
		}
		rd_color_attachment_format.samples = RD::TEXTURE_SAMPLES_1;
		rd_color_attachment_format.usage_bits = RD::TEXTURE_USAGE_SAMPLING_BIT | RD::TEXTURE_USAGE_COLOR_ATTACHMENT_BIT | RD::TEXTURE_USAGE_CAN_COPY_FROM_BIT;
		rd_color_attachment_format.usage_bits |= RD::TEXTURE_USAGE_STORAGE_BIT; // FIXME we need this only when FSR is enabled
		rd_color_attachment_format.shareable_formats.push_back(rt->color_format);
		rd_color_attachment_format.shareable_formats.push_back(rt->color_format_srgb);
		if (rt->msaa != RS::VIEWPORT_MSAA_DISABLED) {
			rd_color_attachment_format.is_resolve_buffer = true;
		}
	}

	rt->color = RD::get_singleton()->texture_create(rd_color_attachment_format, rd_view);
	ERR_FAIL_COND(rt->color.is_null());

	Vector<RID> fb_textures;

	if (rt->msaa != RS::VIEWPORT_MSAA_DISABLED) {
		// Use the texture format of the color attachment for the multisample color attachment.
		RD::TextureFormat rd_color_multisample_format = rd_color_attachment_format;
		const RD::TextureSamples texture_samples[RS::VIEWPORT_MSAA_MAX] = {
			RD::TEXTURE_SAMPLES_1,
			RD::TEXTURE_SAMPLES_2,
			RD::TEXTURE_SAMPLES_4,
			RD::TEXTURE_SAMPLES_8,
		};
		rd_color_multisample_format.samples = texture_samples[rt->msaa];
		RD::TextureView rd_view_multisample;
		rd_color_multisample_format.is_resolve_buffer = false;
		rt->color_multisample = RD::get_singleton()->texture_create(rd_color_multisample_format, rd_view_multisample);
		fb_textures.push_back(rt->color_multisample);
		ERR_FAIL_COND(rt->color_multisample.is_null());
	}
	fb_textures.push_back(rt->color);
	rt->framebuffer = RD::get_singleton()->framebuffer_create(fb_textures, RenderingDevice::INVALID_ID, rt->view_count);
	if (rt->framebuffer.is_null()) {
		_clear_render_target(rt);
		ERR_FAIL_COND(rt->framebuffer.is_null());
	}

	{ //update texture

		Texture *tex = get_texture(rt->texture);

		//free existing textures
		if (RD::get_singleton()->texture_is_valid(tex->rd_texture)) {
			RD::get_singleton()->free(tex->rd_texture);
		}
		if (RD::get_singleton()->texture_is_valid(tex->rd_texture_srgb)) {
			RD::get_singleton()->free(tex->rd_texture_srgb);
		}

		tex->rd_texture = RID();
		tex->rd_texture_srgb = RID();

		//create shared textures to the color buffer,
		//so transparent can be supported
		RD::TextureView view;
		view.format_override = rt->color_format;
		if (!rt->is_transparent) {
			view.swizzle_a = RD::TEXTURE_SWIZZLE_ONE;
		}
		tex->rd_texture = RD::get_singleton()->texture_create_shared(view, rt->color);
		if (rt->color_format_srgb != RD::DATA_FORMAT_MAX) {
			view.format_override = rt->color_format_srgb;
			tex->rd_texture_srgb = RD::get_singleton()->texture_create_shared(view, rt->color);
		}
		tex->rd_view = view;
		tex->width = rt->size.width;
		tex->height = rt->size.height;
		tex->width_2d = rt->size.width;
		tex->height_2d = rt->size.height;
		tex->rd_format = rt->color_format;
		tex->rd_format_srgb = rt->color_format_srgb;
		tex->format = rt->image_format;

		Vector<RID> proxies = tex->proxies; //make a copy, since update may change it
		for (int i = 0; i < proxies.size(); i++) {
			texture_proxy_update(proxies[i], rt->texture);
		}
	}
}

void TextureStorage::_create_render_target_backbuffer(RenderTarget *rt) {
	ERR_FAIL_COND(rt->backbuffer.is_valid());

	uint32_t mipmaps_required = Image::get_image_required_mipmaps(rt->size.width, rt->size.height, Image::FORMAT_RGBA8);
	RD::TextureFormat tf;
	tf.format = rt->color_format;
	tf.width = rt->size.width;
	tf.height = rt->size.height;
	tf.texture_type = RD::TEXTURE_TYPE_2D;
	tf.usage_bits = RD::TEXTURE_USAGE_COLOR_ATTACHMENT_BIT | RD::TEXTURE_USAGE_STORAGE_BIT | RD::TEXTURE_USAGE_SAMPLING_BIT | RD::TEXTURE_USAGE_CAN_COPY_TO_BIT;
	tf.mipmaps = mipmaps_required;

	rt->backbuffer = RD::get_singleton()->texture_create(tf, RD::TextureView());
	RD::get_singleton()->set_resource_name(rt->backbuffer, "Render Target Back Buffer");
	rt->backbuffer_mipmap0 = RD::get_singleton()->texture_create_shared_from_slice(RD::TextureView(), rt->backbuffer, 0, 0);
	RD::get_singleton()->set_resource_name(rt->backbuffer_mipmap0, "Back Buffer slice mipmap 0");

	{
		Vector<RID> fb_tex;
		fb_tex.push_back(rt->backbuffer_mipmap0);
		rt->backbuffer_fb = RD::get_singleton()->framebuffer_create(fb_tex);
	}

	if (rt->framebuffer_uniform_set.is_valid() && RD::get_singleton()->uniform_set_is_valid(rt->framebuffer_uniform_set)) {
		//the new one will require the backbuffer.
		RD::get_singleton()->free(rt->framebuffer_uniform_set);
		rt->framebuffer_uniform_set = RID();
	}
	//create mipmaps
	for (uint32_t i = 1; i < mipmaps_required; i++) {
		RID mipmap = RD::get_singleton()->texture_create_shared_from_slice(RD::TextureView(), rt->backbuffer, 0, i);
		RD::get_singleton()->set_resource_name(mipmap, "Back Buffer slice mip: " + itos(i));

		rt->backbuffer_mipmaps.push_back(mipmap);
	}
}

RID TextureStorage::render_target_create() {
	RenderTarget render_target;

	render_target.was_used = false;
	render_target.clear_requested = false;

	_update_render_target(&render_target);
	return render_target_owner.make_rid(render_target);
}

void TextureStorage::render_target_free(RID p_rid) {
	RenderTarget *rt = render_target_owner.get_or_null(p_rid);

	_clear_render_target(rt);

	if (rt->texture.is_valid()) {
		Texture *tex = get_texture(rt->texture);
		tex->is_render_target = false;
		texture_free(rt->texture);
	}

	render_target_owner.free(p_rid);
}

void TextureStorage::render_target_set_position(RID p_render_target, int p_x, int p_y) {
	//unused for this render target
}

void TextureStorage::render_target_set_size(RID p_render_target, int p_width, int p_height, uint32_t p_view_count) {
	RenderTarget *rt = render_target_owner.get_or_null(p_render_target);
	ERR_FAIL_COND(!rt);
	if (rt->size.x != p_width || rt->size.y != p_height || rt->view_count != p_view_count) {
		rt->size.x = p_width;
		rt->size.y = p_height;
		rt->view_count = p_view_count;
		_update_render_target(rt);
	}
}

RID TextureStorage::render_target_get_texture(RID p_render_target) {
	RenderTarget *rt = render_target_owner.get_or_null(p_render_target);
	ERR_FAIL_COND_V(!rt, RID());

	return rt->texture;
}

void TextureStorage::render_target_set_external_texture(RID p_render_target, unsigned int p_texture_id) {
}

void TextureStorage::render_target_set_transparent(RID p_render_target, bool p_is_transparent) {
	RenderTarget *rt = render_target_owner.get_or_null(p_render_target);
	ERR_FAIL_COND(!rt);
	rt->is_transparent = p_is_transparent;
	_update_render_target(rt);
}

void TextureStorage::render_target_set_direct_to_screen(RID p_render_target, bool p_value) {
}

bool TextureStorage::render_target_was_used(RID p_render_target) {
	RenderTarget *rt = render_target_owner.get_or_null(p_render_target);
	ERR_FAIL_COND_V(!rt, false);
	return rt->was_used;
}

void TextureStorage::render_target_set_as_unused(RID p_render_target) {
	RenderTarget *rt = render_target_owner.get_or_null(p_render_target);
	ERR_FAIL_COND(!rt);
	rt->was_used = false;
}

void TextureStorage::render_target_set_msaa(RID p_render_target, RS::ViewportMSAA p_msaa) {
	RenderTarget *rt = render_target_owner.get_or_null(p_render_target);
	ERR_FAIL_COND(!rt);
	if (p_msaa == rt->msaa) {
		return;
	}

	rt->msaa = p_msaa;
	_update_render_target(rt);
}

Size2 TextureStorage::render_target_get_size(RID p_render_target) {
	RenderTarget *rt = render_target_owner.get_or_null(p_render_target);
	ERR_FAIL_COND_V(!rt, Size2());

	return rt->size;
}

RID TextureStorage::render_target_get_rd_framebuffer(RID p_render_target) {
	RenderTarget *rt = render_target_owner.get_or_null(p_render_target);
	ERR_FAIL_COND_V(!rt, RID());

	return rt->framebuffer;
}

RID TextureStorage::render_target_get_rd_texture(RID p_render_target) {
	RenderTarget *rt = render_target_owner.get_or_null(p_render_target);
	ERR_FAIL_COND_V(!rt, RID());

	return rt->color;
}

RID TextureStorage::render_target_get_rd_texture_slice(RID p_render_target, uint32_t p_layer) {
	RenderTarget *rt = render_target_owner.get_or_null(p_render_target);
	ERR_FAIL_COND_V(!rt, RID());

	if (rt->view_count == 1) {
		return rt->color;
	} else {
		ERR_FAIL_UNSIGNED_INDEX_V(p_layer, rt->view_count, RID());
		if (rt->color_slices.size() == 0) {
			for (uint32_t v = 0; v < rt->view_count; v++) {
				RID slice = RD::get_singleton()->texture_create_shared_from_slice(RD::TextureView(), rt->color, v, 0);
				rt->color_slices.push_back(slice);
			}
		}
		return rt->color_slices[p_layer];
	}
}

RID TextureStorage::render_target_get_rd_backbuffer(RID p_render_target) {
	RenderTarget *rt = render_target_owner.get_or_null(p_render_target);
	ERR_FAIL_COND_V(!rt, RID());
	return rt->backbuffer;
}

RID TextureStorage::render_target_get_rd_backbuffer_framebuffer(RID p_render_target) {
	RenderTarget *rt = render_target_owner.get_or_null(p_render_target);
	ERR_FAIL_COND_V(!rt, RID());

	if (!rt->backbuffer.is_valid()) {
		_create_render_target_backbuffer(rt);
	}

	return rt->backbuffer_fb;
}

void TextureStorage::render_target_request_clear(RID p_render_target, const Color &p_clear_color) {
	RenderTarget *rt = render_target_owner.get_or_null(p_render_target);
	ERR_FAIL_COND(!rt);
	rt->clear_requested = true;
	rt->clear_color = p_clear_color;
}

bool TextureStorage::render_target_is_clear_requested(RID p_render_target) {
	RenderTarget *rt = render_target_owner.get_or_null(p_render_target);
	ERR_FAIL_COND_V(!rt, false);
	return rt->clear_requested;
}

Color TextureStorage::render_target_get_clear_request_color(RID p_render_target) {
	RenderTarget *rt = render_target_owner.get_or_null(p_render_target);
	ERR_FAIL_COND_V(!rt, Color());
	return rt->clear_color;
}

void TextureStorage::render_target_disable_clear_request(RID p_render_target) {
	RenderTarget *rt = render_target_owner.get_or_null(p_render_target);
	ERR_FAIL_COND(!rt);
	rt->clear_requested = false;
}

void TextureStorage::render_target_do_clear_request(RID p_render_target) {
	RenderTarget *rt = render_target_owner.get_or_null(p_render_target);
	ERR_FAIL_COND(!rt);
	if (!rt->clear_requested) {
		return;
	}
	Vector<Color> clear_colors;
	clear_colors.push_back(rt->clear_color);
	RD::get_singleton()->draw_list_begin(rt->framebuffer, RD::INITIAL_ACTION_CLEAR, RD::FINAL_ACTION_READ, RD::INITIAL_ACTION_KEEP, RD::FINAL_ACTION_DISCARD, clear_colors);
	RD::get_singleton()->draw_list_end();
	rt->clear_requested = false;
}

void TextureStorage::render_target_set_sdf_size_and_scale(RID p_render_target, RS::ViewportSDFOversize p_size, RS::ViewportSDFScale p_scale) {
	RenderTarget *rt = render_target_owner.get_or_null(p_render_target);
	ERR_FAIL_COND(!rt);
	if (rt->sdf_oversize == p_size && rt->sdf_scale == p_scale) {
		return;
	}

	rt->sdf_oversize = p_size;
	rt->sdf_scale = p_scale;

	_render_target_clear_sdf(rt);
}

Rect2i TextureStorage::_render_target_get_sdf_rect(const RenderTarget *rt) const {
	Size2i margin;
	int scale;
	switch (rt->sdf_oversize) {
		case RS::VIEWPORT_SDF_OVERSIZE_100_PERCENT: {
			scale = 100;
		} break;
		case RS::VIEWPORT_SDF_OVERSIZE_120_PERCENT: {
			scale = 120;
		} break;
		case RS::VIEWPORT_SDF_OVERSIZE_150_PERCENT: {
			scale = 150;
		} break;
		case RS::VIEWPORT_SDF_OVERSIZE_200_PERCENT: {
			scale = 200;
		} break;
		default: {
		}
	}

	margin = (rt->size * scale / 100) - rt->size;

	Rect2i r(Vector2i(), rt->size);
	r.position -= margin;
	r.size += margin * 2;

	return r;
}

Rect2i TextureStorage::render_target_get_sdf_rect(RID p_render_target) const {
	const RenderTarget *rt = render_target_owner.get_or_null(p_render_target);
	ERR_FAIL_COND_V(!rt, Rect2i());

	return _render_target_get_sdf_rect(rt);
}

void TextureStorage::render_target_mark_sdf_enabled(RID p_render_target, bool p_enabled) {
	RenderTarget *rt = render_target_owner.get_or_null(p_render_target);
	ERR_FAIL_COND(!rt);

	rt->sdf_enabled = p_enabled;
}

bool TextureStorage::render_target_is_sdf_enabled(RID p_render_target) const {
	const RenderTarget *rt = render_target_owner.get_or_null(p_render_target);
	ERR_FAIL_COND_V(!rt, false);

	return rt->sdf_enabled;
}

RID TextureStorage::render_target_get_sdf_texture(RID p_render_target) {
	RenderTarget *rt = render_target_owner.get_or_null(p_render_target);
	ERR_FAIL_COND_V(!rt, RID());
	if (rt->sdf_buffer_read.is_null()) {
		// no texture, create a dummy one for the 2D uniform set
		RD::TextureFormat tformat;
		tformat.format = RD::DATA_FORMAT_R8G8B8A8_UNORM;
		tformat.width = 4;
		tformat.height = 4;
		tformat.usage_bits = RD::TEXTURE_USAGE_SAMPLING_BIT;
		tformat.texture_type = RD::TEXTURE_TYPE_2D;

		Vector<uint8_t> pv;
		pv.resize(16 * 4);
		memset(pv.ptrw(), 0, 16 * 4);
		Vector<Vector<uint8_t>> vpv;

		rt->sdf_buffer_read = RD::get_singleton()->texture_create(tformat, RD::TextureView(), vpv);
	}

	return rt->sdf_buffer_read;
}

void TextureStorage::_render_target_allocate_sdf(RenderTarget *rt) {
	ERR_FAIL_COND(rt->sdf_buffer_write_fb.is_valid());
	if (rt->sdf_buffer_read.is_valid()) {
		RD::get_singleton()->free(rt->sdf_buffer_read);
		rt->sdf_buffer_read = RID();
	}

	Size2i size = _render_target_get_sdf_rect(rt).size;

	RD::TextureFormat tformat;
	tformat.format = RD::DATA_FORMAT_R8_UNORM;
	tformat.width = size.width;
	tformat.height = size.height;
	tformat.usage_bits = RD::TEXTURE_USAGE_SAMPLING_BIT | RD::TEXTURE_USAGE_STORAGE_BIT | RD::TEXTURE_USAGE_COLOR_ATTACHMENT_BIT;
	tformat.texture_type = RD::TEXTURE_TYPE_2D;

	rt->sdf_buffer_write = RD::get_singleton()->texture_create(tformat, RD::TextureView());

	{
		Vector<RID> write_fb;
		write_fb.push_back(rt->sdf_buffer_write);
		rt->sdf_buffer_write_fb = RD::get_singleton()->framebuffer_create(write_fb);
	}

	int scale;
	switch (rt->sdf_scale) {
		case RS::VIEWPORT_SDF_SCALE_100_PERCENT: {
			scale = 100;
		} break;
		case RS::VIEWPORT_SDF_SCALE_50_PERCENT: {
			scale = 50;
		} break;
		case RS::VIEWPORT_SDF_SCALE_25_PERCENT: {
			scale = 25;
		} break;
		default: {
			scale = 100;
		} break;
	}

	rt->process_size = size * scale / 100;
	rt->process_size.x = MAX(rt->process_size.x, 1);
	rt->process_size.y = MAX(rt->process_size.y, 1);

	tformat.format = RD::DATA_FORMAT_R16G16_SINT;
	tformat.width = rt->process_size.width;
	tformat.height = rt->process_size.height;
	tformat.usage_bits = RD::TEXTURE_USAGE_STORAGE_BIT;

	rt->sdf_buffer_process[0] = RD::get_singleton()->texture_create(tformat, RD::TextureView());
	rt->sdf_buffer_process[1] = RD::get_singleton()->texture_create(tformat, RD::TextureView());

	tformat.format = RD::DATA_FORMAT_R16_SNORM;
	tformat.usage_bits = RD::TEXTURE_USAGE_SAMPLING_BIT | RD::TEXTURE_USAGE_STORAGE_BIT;

	rt->sdf_buffer_read = RD::get_singleton()->texture_create(tformat, RD::TextureView());

	{
		Vector<RD::Uniform> uniforms;
		{
			RD::Uniform u;
			u.uniform_type = RD::UNIFORM_TYPE_IMAGE;
			u.binding = 1;
			u.append_id(rt->sdf_buffer_write);
			uniforms.push_back(u);
		}
		{
			RD::Uniform u;
			u.uniform_type = RD::UNIFORM_TYPE_IMAGE;
			u.binding = 2;
			u.append_id(rt->sdf_buffer_read);
			uniforms.push_back(u);
		}
		{
			RD::Uniform u;
			u.uniform_type = RD::UNIFORM_TYPE_IMAGE;
			u.binding = 3;
			u.append_id(rt->sdf_buffer_process[0]);
			uniforms.push_back(u);
		}
		{
			RD::Uniform u;
			u.uniform_type = RD::UNIFORM_TYPE_IMAGE;
			u.binding = 4;
			u.append_id(rt->sdf_buffer_process[1]);
			uniforms.push_back(u);
		}

		rt->sdf_buffer_process_uniform_sets[0] = RD::get_singleton()->uniform_set_create(uniforms, rt_sdf.shader.version_get_shader(rt_sdf.shader_version, 0), 0);
		RID aux2 = uniforms.write[2].get_id(0);
		RID aux3 = uniforms.write[3].get_id(0);
		uniforms.write[2].set_id(0, aux3);
		uniforms.write[3].set_id(0, aux2);
		rt->sdf_buffer_process_uniform_sets[1] = RD::get_singleton()->uniform_set_create(uniforms, rt_sdf.shader.version_get_shader(rt_sdf.shader_version, 0), 0);
	}
}

void TextureStorage::_render_target_clear_sdf(RenderTarget *rt) {
	if (rt->sdf_buffer_read.is_valid()) {
		RD::get_singleton()->free(rt->sdf_buffer_read);
		rt->sdf_buffer_read = RID();
	}
	if (rt->sdf_buffer_write_fb.is_valid()) {
		RD::get_singleton()->free(rt->sdf_buffer_write);
		RD::get_singleton()->free(rt->sdf_buffer_process[0]);
		RD::get_singleton()->free(rt->sdf_buffer_process[1]);
		rt->sdf_buffer_write = RID();
		rt->sdf_buffer_write_fb = RID();
		rt->sdf_buffer_process[0] = RID();
		rt->sdf_buffer_process[1] = RID();
		rt->sdf_buffer_process_uniform_sets[0] = RID();
		rt->sdf_buffer_process_uniform_sets[1] = RID();
	}
}

RID TextureStorage::render_target_get_sdf_framebuffer(RID p_render_target) {
	RenderTarget *rt = render_target_owner.get_or_null(p_render_target);
	ERR_FAIL_COND_V(!rt, RID());

	if (rt->sdf_buffer_write_fb.is_null()) {
		_render_target_allocate_sdf(rt);
	}

	return rt->sdf_buffer_write_fb;
}
void TextureStorage::render_target_sdf_process(RID p_render_target) {
	RenderTarget *rt = render_target_owner.get_or_null(p_render_target);
	ERR_FAIL_COND(!rt);
	ERR_FAIL_COND(rt->sdf_buffer_write_fb.is_null());

	RenderTargetSDF::PushConstant push_constant;

	Rect2i r = _render_target_get_sdf_rect(rt);

	push_constant.size[0] = r.size.width;
	push_constant.size[1] = r.size.height;
	push_constant.stride = 0;
	push_constant.shift = 0;
	push_constant.base_size[0] = r.size.width;
	push_constant.base_size[1] = r.size.height;

	bool shrink = false;

	switch (rt->sdf_scale) {
		case RS::VIEWPORT_SDF_SCALE_50_PERCENT: {
			push_constant.size[0] >>= 1;
			push_constant.size[1] >>= 1;
			push_constant.shift = 1;
			shrink = true;
		} break;
		case RS::VIEWPORT_SDF_SCALE_25_PERCENT: {
			push_constant.size[0] >>= 2;
			push_constant.size[1] >>= 2;
			push_constant.shift = 2;
			shrink = true;
		} break;
		default: {
		};
	}

	RD::ComputeListID compute_list = RD::get_singleton()->compute_list_begin();

	/* Load */

	RD::get_singleton()->compute_list_bind_compute_pipeline(compute_list, rt_sdf.pipelines[shrink ? RenderTargetSDF::SHADER_LOAD_SHRINK : RenderTargetSDF::SHADER_LOAD]);
	RD::get_singleton()->compute_list_bind_uniform_set(compute_list, rt->sdf_buffer_process_uniform_sets[1], 0); //fill [0]
	RD::get_singleton()->compute_list_set_push_constant(compute_list, &push_constant, sizeof(RenderTargetSDF::PushConstant));

	RD::get_singleton()->compute_list_dispatch_threads(compute_list, push_constant.size[0], push_constant.size[1], 1);

	/* Process */

	int stride = nearest_power_of_2_templated(MAX(push_constant.size[0], push_constant.size[1]) / 2);

	RD::get_singleton()->compute_list_bind_compute_pipeline(compute_list, rt_sdf.pipelines[RenderTargetSDF::SHADER_PROCESS]);

	RD::get_singleton()->compute_list_add_barrier(compute_list);
	bool swap = false;

	//jumpflood
	while (stride > 0) {
		RD::get_singleton()->compute_list_bind_uniform_set(compute_list, rt->sdf_buffer_process_uniform_sets[swap ? 1 : 0], 0);
		push_constant.stride = stride;
		RD::get_singleton()->compute_list_set_push_constant(compute_list, &push_constant, sizeof(RenderTargetSDF::PushConstant));
		RD::get_singleton()->compute_list_dispatch_threads(compute_list, push_constant.size[0], push_constant.size[1], 1);
		stride /= 2;
		swap = !swap;
		RD::get_singleton()->compute_list_add_barrier(compute_list);
	}

	/* Store */

	RD::get_singleton()->compute_list_bind_compute_pipeline(compute_list, rt_sdf.pipelines[shrink ? RenderTargetSDF::SHADER_STORE_SHRINK : RenderTargetSDF::SHADER_STORE]);
	RD::get_singleton()->compute_list_bind_uniform_set(compute_list, rt->sdf_buffer_process_uniform_sets[swap ? 1 : 0], 0);
	RD::get_singleton()->compute_list_set_push_constant(compute_list, &push_constant, sizeof(RenderTargetSDF::PushConstant));
	RD::get_singleton()->compute_list_dispatch_threads(compute_list, push_constant.size[0], push_constant.size[1], 1);

	RD::get_singleton()->compute_list_end();
}

void TextureStorage::render_target_copy_to_back_buffer(RID p_render_target, const Rect2i &p_region, bool p_gen_mipmaps) {
	CopyEffects *copy_effects = CopyEffects::get_singleton();
	ERR_FAIL_NULL(copy_effects);

	RenderTarget *rt = render_target_owner.get_or_null(p_render_target);
	ERR_FAIL_COND(!rt);
	if (!rt->backbuffer.is_valid()) {
		_create_render_target_backbuffer(rt);
	}

	Rect2i region;
	if (p_region == Rect2i()) {
		region.size = rt->size;
	} else {
		region = Rect2i(Size2i(), rt->size).intersection(p_region);
		if (region.size == Size2i()) {
			return; //nothing to do
		}
	}

	// TODO figure out stereo support here

	//single texture copy for backbuffer
	//RD::get_singleton()->texture_copy(rt->color, rt->backbuffer_mipmap0, Vector3(region.position.x, region.position.y, 0), Vector3(region.position.x, region.position.y, 0), Vector3(region.size.x, region.size.y, 1), 0, 0, 0, 0, true);
	copy_effects->copy_to_rect(rt->color, rt->backbuffer_mipmap0, region, false, false, false, true, true);

	if (!p_gen_mipmaps) {
		return;
	}
	RD::get_singleton()->draw_command_begin_label("Gaussian Blur Mipmaps");
	//then mipmap blur
	RID prev_texture = rt->color; //use color, not backbuffer, as bb has mipmaps.

	for (int i = 0; i < rt->backbuffer_mipmaps.size(); i++) {
		region.position.x >>= 1;
		region.position.y >>= 1;
		region.size.x = MAX(1, region.size.x >> 1);
		region.size.y = MAX(1, region.size.y >> 1);

		RID mipmap = rt->backbuffer_mipmaps[i];
		copy_effects->gaussian_blur(prev_texture, mipmap, region, true);
		prev_texture = mipmap;
	}
	RD::get_singleton()->draw_command_end_label();
}

void TextureStorage::render_target_clear_back_buffer(RID p_render_target, const Rect2i &p_region, const Color &p_color) {
	RenderTarget *rt = render_target_owner.get_or_null(p_render_target);
	ERR_FAIL_COND(!rt);

	CopyEffects *copy_effects = CopyEffects::get_singleton();
	ERR_FAIL_NULL(copy_effects);

	if (!rt->backbuffer.is_valid()) {
		_create_render_target_backbuffer(rt);
	}

	Rect2i region;
	if (p_region == Rect2i()) {
		region.size = rt->size;
	} else {
		region = Rect2i(Size2i(), rt->size).intersection(p_region);
		if (region.size == Size2i()) {
			return; //nothing to do
		}
	}

	//single texture copy for backbuffer
	copy_effects->set_color(rt->backbuffer_mipmap0, p_color, region, true);
}

void TextureStorage::render_target_gen_back_buffer_mipmaps(RID p_render_target, const Rect2i &p_region) {
	RenderTarget *rt = render_target_owner.get_or_null(p_render_target);
	ERR_FAIL_COND(!rt);

	CopyEffects *copy_effects = CopyEffects::get_singleton();
	ERR_FAIL_NULL(copy_effects);

	if (!rt->backbuffer.is_valid()) {
		_create_render_target_backbuffer(rt);
	}

	Rect2i region;
	if (p_region == Rect2i()) {
		region.size = rt->size;
	} else {
		region = Rect2i(Size2i(), rt->size).intersection(p_region);
		if (region.size == Size2i()) {
			return; //nothing to do
		}
	}
	RD::get_singleton()->draw_command_begin_label("Gaussian Blur Mipmaps2");
	//then mipmap blur
	RID prev_texture = rt->backbuffer_mipmap0;

	for (int i = 0; i < rt->backbuffer_mipmaps.size(); i++) {
		region.position.x >>= 1;
		region.position.y >>= 1;
		region.size.x = MAX(1, region.size.x >> 1);
		region.size.y = MAX(1, region.size.y >> 1);

		RID mipmap = rt->backbuffer_mipmaps[i];
		copy_effects->gaussian_blur(prev_texture, mipmap, region, true);
		prev_texture = mipmap;
	}
	RD::get_singleton()->draw_command_end_label();
}

RID TextureStorage::render_target_get_framebuffer_uniform_set(RID p_render_target) {
	RenderTarget *rt = render_target_owner.get_or_null(p_render_target);
	ERR_FAIL_COND_V(!rt, RID());
	return rt->framebuffer_uniform_set;
}
RID TextureStorage::render_target_get_backbuffer_uniform_set(RID p_render_target) {
	RenderTarget *rt = render_target_owner.get_or_null(p_render_target);
	ERR_FAIL_COND_V(!rt, RID());
	return rt->backbuffer_uniform_set;
}

void TextureStorage::render_target_set_framebuffer_uniform_set(RID p_render_target, RID p_uniform_set) {
	RenderTarget *rt = render_target_owner.get_or_null(p_render_target);
	ERR_FAIL_COND(!rt);
	rt->framebuffer_uniform_set = p_uniform_set;
}

void TextureStorage::render_target_set_backbuffer_uniform_set(RID p_render_target, RID p_uniform_set) {
	RenderTarget *rt = render_target_owner.get_or_null(p_render_target);
	ERR_FAIL_COND(!rt);
	rt->backbuffer_uniform_set = p_uniform_set;
}

void TextureStorage::render_target_set_vrs_mode(RID p_render_target, RS::ViewportVRSMode p_mode) {
	RenderTarget *rt = render_target_owner.get_or_null(p_render_target);
	ERR_FAIL_COND(!rt);

	rt->vrs_mode = p_mode;
}

void TextureStorage::render_target_set_vrs_texture(RID p_render_target, RID p_texture) {
	RenderTarget *rt = render_target_owner.get_or_null(p_render_target);
	ERR_FAIL_COND(!rt);

	rt->vrs_texture = p_texture;
}

RS::ViewportVRSMode TextureStorage::render_target_get_vrs_mode(RID p_render_target) const {
	RenderTarget *rt = render_target_owner.get_or_null(p_render_target);
	ERR_FAIL_COND_V(!rt, RS::VIEWPORT_VRS_DISABLED);

	return rt->vrs_mode;
}

RID TextureStorage::render_target_get_vrs_texture(RID p_render_target) const {
	RenderTarget *rt = render_target_owner.get_or_null(p_render_target);
	ERR_FAIL_COND_V(!rt, RID());

	return rt->vrs_texture;
}
