/*************************************************************************/
/*  decal_atlas_storage.h                                                */
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

#ifndef DECAL_ATLAS_STORAGE_RD_H
#define DECAL_ATLAS_STORAGE_RD_H

#include "core/templates/rid_owner.h"
#include "servers/rendering/renderer_storage.h"
#include "servers/rendering/storage/decal_atlas_storage.h"

namespace RendererRD {

struct DecalAtlas {
	struct Texture {
		int panorama_to_dp_users;
		int users;
		Rect2 uv_rect;
	};

	struct SortItem {
		RID texture;
		Size2i pixel_size;
		Size2i size;
		Point2i pos;

		bool operator<(const SortItem &p_item) const {
			//sort larger to smaller
			if (size.height == p_item.size.height) {
				return size.width > p_item.size.width;
			} else {
				return size.height > p_item.size.height;
			}
		}
	};

	HashMap<RID, Texture> textures;
	bool dirty = true;
	int mipmaps = 5;

	RID texture;
	RID texture_srgb;
	struct MipMap {
		RID fb;
		RID texture;
		Size2i size;
	};
	Vector<MipMap> texture_mipmaps;

	Size2i size;
};

struct Decal {
	Vector3 extents = Vector3(1, 1, 1);
	RID textures[RS::DECAL_TEXTURE_MAX];
	float emission_energy = 1.0;
	float albedo_mix = 1.0;
	Color modulate = Color(1, 1, 1, 1);
	uint32_t cull_mask = (1 << 20) - 1;
	float upper_fade = 0.3;
	float lower_fade = 0.3;
	bool distance_fade = false;
	float distance_fade_begin = 10;
	float distance_fade_length = 1;
	float normal_fade = 0.0;

	RendererStorage::Dependency dependency;
};

class DecalAtlasStorage : public RendererDecalAtlasStorage {
private:
	static DecalAtlasStorage *singleton;

	DecalAtlas decal_atlas;

	mutable RID_Owner<Decal, true> decal_owner;

public:
	static DecalAtlasStorage *get_singleton();

	void update_decal_atlas();

	DecalAtlasStorage();
	virtual ~DecalAtlasStorage();

	Decal *get_decal(RID p_rid) { return decal_owner.get_or_null(p_rid); };
	bool owns_decal(RID p_rid) { return decal_owner.owns(p_rid); };

	RID decal_atlas_get_texture() const;
	RID decal_atlas_get_texture_srgb() const;
	_FORCE_INLINE_ Rect2 decal_atlas_get_texture_rect(RID p_texture) {
		DecalAtlas::Texture *t = decal_atlas.textures.getptr(p_texture);
		if (!t) {
			return Rect2();
		}

		return t->uv_rect;
	}

	virtual RID decal_allocate() override;
	virtual void decal_initialize(RID p_decal) override;
	virtual void decal_free(RID p_rid) override;

	virtual void decal_set_extents(RID p_decal, const Vector3 &p_extents) override;
	virtual void decal_set_texture(RID p_decal, RS::DecalTexture p_type, RID p_texture) override;
	virtual void decal_set_emission_energy(RID p_decal, float p_energy) override;
	virtual void decal_set_albedo_mix(RID p_decal, float p_mix) override;
	virtual void decal_set_modulate(RID p_decal, const Color &p_modulate) override;
	virtual void decal_set_cull_mask(RID p_decal, uint32_t p_layers) override;
	virtual void decal_set_distance_fade(RID p_decal, bool p_enabled, float p_begin, float p_length) override;
	virtual void decal_set_fade(RID p_decal, float p_above, float p_below) override;
	virtual void decal_set_normal_fade(RID p_decal, float p_fade) override;

	void decal_atlas_mark_dirty_on_texture(RID p_texture);
	void decal_atlas_remove_texture(RID p_texture);

	virtual void texture_add_to_decal_atlas(RID p_texture, bool p_panorama_to_dp = false) override;
	virtual void texture_remove_from_decal_atlas(RID p_texture, bool p_panorama_to_dp = false) override;

	_FORCE_INLINE_ Vector3 decal_get_extents(RID p_decal) {
		const Decal *decal = decal_owner.get_or_null(p_decal);
		return decal->extents;
	}

	_FORCE_INLINE_ RID decal_get_texture(RID p_decal, RS::DecalTexture p_texture) {
		const Decal *decal = decal_owner.get_or_null(p_decal);
		return decal->textures[p_texture];
	}

	_FORCE_INLINE_ Color decal_get_modulate(RID p_decal) {
		const Decal *decal = decal_owner.get_or_null(p_decal);
		return decal->modulate;
	}

	_FORCE_INLINE_ float decal_get_emission_energy(RID p_decal) {
		const Decal *decal = decal_owner.get_or_null(p_decal);
		return decal->emission_energy;
	}

	_FORCE_INLINE_ float decal_get_albedo_mix(RID p_decal) {
		const Decal *decal = decal_owner.get_or_null(p_decal);
		return decal->albedo_mix;
	}

	_FORCE_INLINE_ uint32_t decal_get_cull_mask(RID p_decal) {
		const Decal *decal = decal_owner.get_or_null(p_decal);
		return decal->cull_mask;
	}

	_FORCE_INLINE_ float decal_get_upper_fade(RID p_decal) {
		const Decal *decal = decal_owner.get_or_null(p_decal);
		return decal->upper_fade;
	}

	_FORCE_INLINE_ float decal_get_lower_fade(RID p_decal) {
		const Decal *decal = decal_owner.get_or_null(p_decal);
		return decal->lower_fade;
	}

	_FORCE_INLINE_ float decal_get_normal_fade(RID p_decal) {
		const Decal *decal = decal_owner.get_or_null(p_decal);
		return decal->normal_fade;
	}

	_FORCE_INLINE_ bool decal_is_distance_fade_enabled(RID p_decal) {
		const Decal *decal = decal_owner.get_or_null(p_decal);
		return decal->distance_fade;
	}

	_FORCE_INLINE_ float decal_get_distance_fade_begin(RID p_decal) {
		const Decal *decal = decal_owner.get_or_null(p_decal);
		return decal->distance_fade_begin;
	}

	_FORCE_INLINE_ float decal_get_distance_fade_length(RID p_decal) {
		const Decal *decal = decal_owner.get_or_null(p_decal);
		return decal->distance_fade_length;
	}

	virtual AABB decal_get_aabb(RID p_decal) const override;
};

} // namespace RendererRD

#endif // !DECAL_ATLAS_STORAGE_RD_H
