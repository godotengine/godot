/*************************************************************************/
/*  decal_atlas_storage.cpp                                              */
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

#include "decal_atlas_storage.h"
#include "texture_storage.h"

// Should be able to remove this once we move effects into their own file and include the correct effects
#include "servers/rendering/renderer_rd/renderer_storage_rd.h"

using namespace RendererRD;

DecalAtlasStorage *DecalAtlasStorage::singleton = nullptr;

DecalAtlasStorage *DecalAtlasStorage::get_singleton() {
	return singleton;
}

DecalAtlasStorage::DecalAtlasStorage() {
	singleton = this;

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
			//take the chance and initialize decal atlas to something
			Vector<Vector<uint8_t>> vpv;
			vpv.push_back(pv);
			decal_atlas.texture = RD::get_singleton()->texture_create(tformat, RD::TextureView(), vpv);
			decal_atlas.texture_srgb = decal_atlas.texture;
		}
	}
}

DecalAtlasStorage::~DecalAtlasStorage() {
	if (decal_atlas.textures.size()) {
		ERR_PRINT("Decal Atlas: " + itos(decal_atlas.textures.size()) + " textures were not removed from the atlas.");
	}

	if (decal_atlas.texture.is_valid()) {
		RD::get_singleton()->free(decal_atlas.texture);
	}

	singleton = nullptr;
}

RID DecalAtlasStorage::decal_atlas_get_texture() const {
	return decal_atlas.texture;
}

RID DecalAtlasStorage::decal_atlas_get_texture_srgb() const {
	return decal_atlas.texture_srgb;
}

RID DecalAtlasStorage::decal_allocate() {
	return decal_owner.allocate_rid();
}

void DecalAtlasStorage::decal_initialize(RID p_decal) {
	decal_owner.initialize_rid(p_decal, Decal());
}

void DecalAtlasStorage::decal_free(RID p_rid) {
	Decal *decal = decal_owner.get_or_null(p_rid);
	for (int i = 0; i < RS::DECAL_TEXTURE_MAX; i++) {
		if (decal->textures[i].is_valid() && TextureStorage::get_singleton()->owns_texture(decal->textures[i])) {
			texture_remove_from_decal_atlas(decal->textures[i]);
		}
	}
	decal->dependency.deleted_notify(p_rid);
	decal_owner.free(p_rid);
}

void DecalAtlasStorage::decal_set_extents(RID p_decal, const Vector3 &p_extents) {
	Decal *decal = decal_owner.get_or_null(p_decal);
	ERR_FAIL_COND(!decal);
	decal->extents = p_extents;
	decal->dependency.changed_notify(RendererStorage::DEPENDENCY_CHANGED_AABB);
}

void DecalAtlasStorage::decal_set_texture(RID p_decal, RS::DecalTexture p_type, RID p_texture) {
	Decal *decal = decal_owner.get_or_null(p_decal);
	ERR_FAIL_COND(!decal);
	ERR_FAIL_INDEX(p_type, RS::DECAL_TEXTURE_MAX);

	if (decal->textures[p_type] == p_texture) {
		return;
	}

	ERR_FAIL_COND(p_texture.is_valid() && !TextureStorage::get_singleton()->owns_texture(p_texture));

	if (decal->textures[p_type].is_valid() && TextureStorage::get_singleton()->owns_texture(decal->textures[p_type])) {
		texture_remove_from_decal_atlas(decal->textures[p_type]);
	}

	decal->textures[p_type] = p_texture;

	if (decal->textures[p_type].is_valid()) {
		texture_add_to_decal_atlas(decal->textures[p_type]);
	}

	decal->dependency.changed_notify(RendererStorage::DEPENDENCY_CHANGED_DECAL);
}

void DecalAtlasStorage::decal_set_emission_energy(RID p_decal, float p_energy) {
	Decal *decal = decal_owner.get_or_null(p_decal);
	ERR_FAIL_COND(!decal);
	decal->emission_energy = p_energy;
}

void DecalAtlasStorage::decal_set_albedo_mix(RID p_decal, float p_mix) {
	Decal *decal = decal_owner.get_or_null(p_decal);
	ERR_FAIL_COND(!decal);
	decal->albedo_mix = p_mix;
}

void DecalAtlasStorage::decal_set_modulate(RID p_decal, const Color &p_modulate) {
	Decal *decal = decal_owner.get_or_null(p_decal);
	ERR_FAIL_COND(!decal);
	decal->modulate = p_modulate;
}

void DecalAtlasStorage::decal_set_cull_mask(RID p_decal, uint32_t p_layers) {
	Decal *decal = decal_owner.get_or_null(p_decal);
	ERR_FAIL_COND(!decal);
	decal->cull_mask = p_layers;
	decal->dependency.changed_notify(RendererStorage::DEPENDENCY_CHANGED_AABB);
}

void DecalAtlasStorage::decal_set_distance_fade(RID p_decal, bool p_enabled, float p_begin, float p_length) {
	Decal *decal = decal_owner.get_or_null(p_decal);
	ERR_FAIL_COND(!decal);
	decal->distance_fade = p_enabled;
	decal->distance_fade_begin = p_begin;
	decal->distance_fade_length = p_length;
}

void DecalAtlasStorage::decal_set_fade(RID p_decal, float p_above, float p_below) {
	Decal *decal = decal_owner.get_or_null(p_decal);
	ERR_FAIL_COND(!decal);
	decal->upper_fade = p_above;
	decal->lower_fade = p_below;
}

void DecalAtlasStorage::decal_set_normal_fade(RID p_decal, float p_fade) {
	Decal *decal = decal_owner.get_or_null(p_decal);
	ERR_FAIL_COND(!decal);
	decal->normal_fade = p_fade;
}

void DecalAtlasStorage::decal_atlas_mark_dirty_on_texture(RID p_texture) {
	if (decal_atlas.textures.has(p_texture)) {
		//belongs to decal atlas..

		decal_atlas.dirty = true; //mark it dirty since it was most likely modified
	}
}

void DecalAtlasStorage::decal_atlas_remove_texture(RID p_texture) {
	if (decal_atlas.textures.has(p_texture)) {
		decal_atlas.textures.erase(p_texture);
		//there is not much a point of making it dirty, just let it be.
	}
}

AABB DecalAtlasStorage::decal_get_aabb(RID p_decal) const {
	Decal *decal = decal_owner.get_or_null(p_decal);
	ERR_FAIL_COND_V(!decal, AABB());

	return AABB(-decal->extents, decal->extents * 2.0);
}

void DecalAtlasStorage::update_decal_atlas() {
	EffectsRD *effects = RendererStorageRD::base_singleton->get_effects();

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
		const RID *K = nullptr;

		int idx = 0;
		while ((K = decal_atlas.textures.next(K))) {
			DecalAtlas::SortItem &si = itemsv.write[idx];

			Texture *src_tex = TextureStorage::get_singleton()->get_texture(*K);

			si.size.width = (src_tex->width / border) + 1;
			si.size.height = (src_tex->height / border) + 1;
			si.pixel_size = Size2i(src_tex->width, src_tex->height);

			if (base_size < si.size.width) {
				base_size = nearest_power_of_2_templated(si.size.width);
			}

			si.texture = *K;
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

				const RID *K = nullptr;
				while ((K = decal_atlas.textures.next(K))) {
					DecalAtlas::Texture *t = decal_atlas.textures.getptr(*K);
					Texture *src_tex = TextureStorage::get_singleton()->get_texture(*K);
					effects->copy_to_atlas_fb(src_tex->rd_texture, mm.fb, t->uv_rect, draw_list, false, t->panorama_to_dp_users > 0);
				}

				RD::get_singleton()->draw_list_end();

				prev_texture = mm.texture;
			} else {
				effects->copy_to_fb_rect(prev_texture, mm.fb, Rect2i(Point2i(), mm.size));
				prev_texture = mm.texture;
			}
		} else {
			RD::get_singleton()->texture_clear(mm.texture, clear_color, 0, 1, 0, 1);
		}
	}
}

void DecalAtlasStorage::texture_add_to_decal_atlas(RID p_texture, bool p_panorama_to_dp) {
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

void DecalAtlasStorage::texture_remove_from_decal_atlas(RID p_texture, bool p_panorama_to_dp) {
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
