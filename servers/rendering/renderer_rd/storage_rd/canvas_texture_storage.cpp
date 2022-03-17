/*************************************************************************/
/*  canvas_texture_storage.cpp                                           */
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

#include "canvas_texture_storage.h"
#include "texture_storage.h"

// Until we move things into their own storage classes, also include our old class
#include "servers/rendering/renderer_rd/renderer_storage_rd.h"

using namespace RendererRD;

///////////////////////////////////////////////////////////////////////////
// CanvasTexture

void CanvasTexture::clear_sets() {
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

CanvasTexture::~CanvasTexture() {
	clear_sets();
}

///////////////////////////////////////////////////////////////////////////
// CanvasTextureStorage

CanvasTextureStorage *CanvasTextureStorage::singleton = nullptr;

CanvasTextureStorage *CanvasTextureStorage::get_singleton() {
	return singleton;
}

CanvasTextureStorage::CanvasTextureStorage() {
	singleton = this;
}

CanvasTextureStorage::~CanvasTextureStorage() {
	singleton = nullptr;
}

RID CanvasTextureStorage::canvas_texture_allocate() {
	return canvas_texture_owner.allocate_rid();
}

void CanvasTextureStorage::canvas_texture_initialize(RID p_rid) {
	canvas_texture_owner.initialize_rid(p_rid);
}

void CanvasTextureStorage::canvas_texture_free(RID p_rid) {
	canvas_texture_owner.free(p_rid);
}

void CanvasTextureStorage::canvas_texture_set_channel(RID p_canvas_texture, RS::CanvasTextureChannel p_channel, RID p_texture) {
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

void CanvasTextureStorage::canvas_texture_set_shading_parameters(RID p_canvas_texture, const Color &p_specular_color, float p_shininess) {
	CanvasTexture *ct = canvas_texture_owner.get_or_null(p_canvas_texture);
	ERR_FAIL_NULL(ct);

	ct->specular_color.r = p_specular_color.r;
	ct->specular_color.g = p_specular_color.g;
	ct->specular_color.b = p_specular_color.b;
	ct->specular_color.a = p_shininess;
	ct->clear_sets();
}

void CanvasTextureStorage::canvas_texture_set_texture_filter(RID p_canvas_texture, RS::CanvasItemTextureFilter p_filter) {
	CanvasTexture *ct = canvas_texture_owner.get_or_null(p_canvas_texture);
	ERR_FAIL_NULL(ct);

	ct->texture_filter = p_filter;
	ct->clear_sets();
}

void CanvasTextureStorage::canvas_texture_set_texture_repeat(RID p_canvas_texture, RS::CanvasItemTextureRepeat p_repeat) {
	CanvasTexture *ct = canvas_texture_owner.get_or_null(p_canvas_texture);
	ERR_FAIL_NULL(ct);
	ct->texture_repeat = p_repeat;
	ct->clear_sets();
}

bool CanvasTextureStorage::canvas_texture_get_uniform_set(RID p_texture, RS::CanvasItemTextureFilter p_base_filter, RS::CanvasItemTextureRepeat p_base_repeat, RID p_base_shader, int p_base_set, RID &r_uniform_set, Size2i &r_size, Color &r_specular_shininess, bool &r_use_normal, bool &r_use_specular) {
	RendererStorageRD *storage = RendererStorageRD::base_singleton;

	CanvasTexture *ct = nullptr;
	TextureStorage *texture_storage = TextureStorage::get_singleton();
	Texture *t = texture_storage->get_texture(p_texture);

	// TODO once we have our texture storage split off we'll look into moving this code into canvas_texture

	if (t) {
		//regular texture
		if (!t->canvas_texture) {
			t->canvas_texture = memnew(CanvasTexture);
			t->canvas_texture->diffuse = p_texture;
		}

		ct = t->canvas_texture;
	} else {
		ct = get_canvas_texture(p_texture);
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

			t = texture_storage->get_texture(ct->diffuse);
			if (!t) {
				u.append_id(texture_storage->texture_rd_get_default(DEFAULT_RD_TEXTURE_WHITE));
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

			t = texture_storage->get_texture(ct->normal_map);
			if (!t) {
				u.append_id(texture_storage->texture_rd_get_default(DEFAULT_RD_TEXTURE_NORMAL));
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

			t = texture_storage->get_texture(ct->specular);
			if (!t) {
				u.append_id(texture_storage->texture_rd_get_default(DEFAULT_RD_TEXTURE_WHITE));
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
			u.append_id(storage->sampler_rd_get_default(filter, repeat));
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
