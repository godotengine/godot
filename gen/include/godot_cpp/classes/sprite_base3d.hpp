/**************************************************************************/
/*  sprite_base3d.hpp                                                     */
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

#include <godot_cpp/classes/base_material3d.hpp>
#include <godot_cpp/classes/geometry_instance3d.hpp>
#include <godot_cpp/classes/ref.hpp>
#include <godot_cpp/variant/color.hpp>
#include <godot_cpp/variant/rect2.hpp>
#include <godot_cpp/variant/vector2.hpp>
#include <godot_cpp/variant/vector3.hpp>

#include <godot_cpp/core/class_db.hpp>

#include <type_traits>

namespace godot {

class TriangleMesh;

class SpriteBase3D : public GeometryInstance3D {
	GDEXTENSION_CLASS(SpriteBase3D, GeometryInstance3D)

public:
	enum DrawFlags {
		FLAG_TRANSPARENT = 0,
		FLAG_SHADED = 1,
		FLAG_DOUBLE_SIDED = 2,
		FLAG_DISABLE_DEPTH_TEST = 3,
		FLAG_FIXED_SIZE = 4,
		FLAG_MAX = 5,
	};

	enum AlphaCutMode {
		ALPHA_CUT_DISABLED = 0,
		ALPHA_CUT_DISCARD = 1,
		ALPHA_CUT_OPAQUE_PREPASS = 2,
		ALPHA_CUT_HASH = 3,
	};

	void set_centered(bool p_centered);
	bool is_centered() const;
	void set_offset(const Vector2 &p_offset);
	Vector2 get_offset() const;
	void set_flip_h(bool p_flip_h);
	bool is_flipped_h() const;
	void set_flip_v(bool p_flip_v);
	bool is_flipped_v() const;
	void set_modulate(const Color &p_modulate);
	Color get_modulate() const;
	void set_render_priority(int32_t p_priority);
	int32_t get_render_priority() const;
	void set_pixel_size(float p_pixel_size);
	float get_pixel_size() const;
	void set_axis(Vector3::Axis p_axis);
	Vector3::Axis get_axis() const;
	void set_draw_flag(SpriteBase3D::DrawFlags p_flag, bool p_enabled);
	bool get_draw_flag(SpriteBase3D::DrawFlags p_flag) const;
	void set_alpha_cut_mode(SpriteBase3D::AlphaCutMode p_mode);
	SpriteBase3D::AlphaCutMode get_alpha_cut_mode() const;
	void set_alpha_scissor_threshold(float p_threshold);
	float get_alpha_scissor_threshold() const;
	void set_alpha_hash_scale(float p_threshold);
	float get_alpha_hash_scale() const;
	void set_alpha_antialiasing(BaseMaterial3D::AlphaAntiAliasing p_alpha_aa);
	BaseMaterial3D::AlphaAntiAliasing get_alpha_antialiasing() const;
	void set_alpha_antialiasing_edge(float p_edge);
	float get_alpha_antialiasing_edge() const;
	void set_billboard_mode(BaseMaterial3D::BillboardMode p_mode);
	BaseMaterial3D::BillboardMode get_billboard_mode() const;
	void set_texture_filter(BaseMaterial3D::TextureFilter p_mode);
	BaseMaterial3D::TextureFilter get_texture_filter() const;
	Rect2 get_item_rect() const;
	Ref<TriangleMesh> generate_triangle_mesh() const;

protected:
	template <typename T, typename B>
	static void register_virtuals() {
		GeometryInstance3D::register_virtuals<T, B>();
	}

public:
};

} // namespace godot

VARIANT_ENUM_CAST(SpriteBase3D::DrawFlags);
VARIANT_ENUM_CAST(SpriteBase3D::AlphaCutMode);

