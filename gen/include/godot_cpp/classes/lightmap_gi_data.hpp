/**************************************************************************/
/*  lightmap_gi_data.hpp                                                  */
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

// THIS FILE IS GENERATED. EDITS WILL BE LOST.

#pragma once

#include <godot_cpp/classes/ref.hpp>
#include <godot_cpp/classes/resource.hpp>
#include <godot_cpp/variant/node_path.hpp>
#include <godot_cpp/variant/typed_array.hpp>

#include <godot_cpp/core/class_db.hpp>

#include <type_traits>

namespace godot {

struct Rect2;
class TextureLayered;

class LightmapGIData : public Resource {
	GDEXTENSION_CLASS(LightmapGIData, Resource)

public:
	enum ShadowmaskMode {
		SHADOWMASK_MODE_NONE = 0,
		SHADOWMASK_MODE_REPLACE = 1,
		SHADOWMASK_MODE_OVERLAY = 2,
	};

	void set_lightmap_textures(const TypedArray<Ref<TextureLayered>> &p_light_textures);
	TypedArray<Ref<TextureLayered>> get_lightmap_textures() const;
	void set_shadowmask_textures(const TypedArray<Ref<TextureLayered>> &p_shadowmask_textures);
	TypedArray<Ref<TextureLayered>> get_shadowmask_textures() const;
	void set_uses_spherical_harmonics(bool p_uses_spherical_harmonics);
	bool is_using_spherical_harmonics() const;
	void add_user(const NodePath &p_path, const Rect2 &p_uv_scale, int32_t p_slice_index, int32_t p_sub_instance);
	int32_t get_user_count() const;
	NodePath get_user_path(int32_t p_user_idx) const;
	void clear_users();
	void set_light_texture(const Ref<TextureLayered> &p_light_texture);
	Ref<TextureLayered> get_light_texture() const;

protected:
	template <typename T, typename B>
	static void register_virtuals() {
		Resource::register_virtuals<T, B>();
	}

public:
};

} // namespace godot

VARIANT_ENUM_CAST(LightmapGIData::ShadowmaskMode);

