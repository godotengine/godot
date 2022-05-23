/*************************************************************************/
/*  label_3d.h                                                           */
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

#ifndef LABEL_3D_H
#define LABEL_3D_H

#include "scene/3d/visual_instance.h"
#include "scene/resources/font.h"

class Label3D : public GeometryInstance {
	GDCLASS(Label3D, GeometryInstance);

public:
	enum DrawFlags {
		FLAG_SHADED,
		FLAG_DOUBLE_SIDED,
		FLAG_DISABLE_DEPTH_TEST,
		FLAG_FIXED_SIZE,
		FLAG_MAX
	};

	enum AlphaCutMode {
		ALPHA_CUT_DISABLED,
		ALPHA_CUT_DISCARD,
		ALPHA_CUT_OPAQUE_PREPASS
	};

	enum Align {

		ALIGN_LEFT,
		ALIGN_CENTER,
		ALIGN_RIGHT,
		ALIGN_FILL
	};

	enum VAlign {

		VALIGN_TOP,
		VALIGN_CENTER,
		VALIGN_BOTTOM,
		VALIGN_FILL
	};

private:
	real_t pixel_size = 0.01;
	bool flags[FLAG_MAX] = {};
	AlphaCutMode alpha_cut = ALPHA_CUT_DISABLED;
	float alpha_scissor_threshold = 0.5;

	AABB aabb;

	mutable Ref<TriangleMesh> triangle_mesh;
	RID mesh;
	struct SurfaceData {
		PoolVector3Array mesh_vertices;
		PoolVector3Array mesh_normals;
		PoolRealArray mesh_tangents;
		PoolColorArray mesh_colors;
		PoolVector2Array mesh_uvs;
		PoolIntArray indices;
		int offset = 0;
		float z_shift = 0.0;
		RID material;
	};
	HashMap<uint64_t, SurfaceData> surfaces;

	struct WordCache {
		enum {
			CHAR_NEWLINE = -1,
			CHAR_WRAPLINE = -2
		};
		int char_pos; // if -1, then newline
		int word_len;
		int pixel_width;
		int space_count;
		WordCache *next;
		WordCache() {
			char_pos = 0;
			word_len = 0;
			pixel_width = 0;
			next = nullptr;
			space_count = 0;
		}
	};
	bool word_cache_dirty = true;

	WordCache *word_cache = nullptr;
	int line_count = 0;

	Align horizontal_alignment = ALIGN_CENTER;
	VAlign vertical_alignment = VALIGN_CENTER;
	String text;
	String xl_text;
	bool uppercase = false;

	bool autowrap = false;
	float width = 500.0;

	Ref<Font> font_override;
	mutable Ref<Font> theme_font;
	Color modulate = Color(1, 1, 1, 1);
	Point2 lbl_offset;
	int outline_render_priority = -1;
	int render_priority = 0;

	Color outline_modulate = Color(0, 0, 0, 1);

	float line_spacing = 0.f;

	RID base_material;
	SpatialMaterial::BillboardMode billboard_mode = SpatialMaterial::BILLBOARD_DISABLED;

	bool pending_update = false;

	void regenerate_word_cache();
	int get_longest_line_width() const;
	float _generate_glyph_surfaces(const Ref<Font> &p_font, CharType p_char, CharType p_next, Vector2 p_offset, const Color &p_modulate, int p_priority, bool p_outline);

protected:
	void _notification(int p_what);

	static void _bind_methods();

	void _validate_property(PropertyInfo &property) const;

	void _im_update();
	void _font_changed();
	void _queue_update();

	void _shape();

public:
	void set_horizontal_alignment(Align p_alignment);
	Align get_horizontal_alignment() const;

	void set_vertical_alignment(VAlign p_alignment);
	VAlign get_vertical_alignment() const;

	void set_render_priority(int p_priority);
	int get_render_priority() const;

	void set_outline_render_priority(int p_priority);
	int get_outline_render_priority() const;

	void set_text(const String &p_string);
	String get_text() const;

	void set_uppercase(bool p_uppercase);
	bool is_uppercase() const;

	void set_font(const Ref<Font> &p_font);
	Ref<Font> get_font() const;
	Ref<Font> _get_font_or_default() const;

	void set_line_spacing(float p_size);
	float get_line_spacing() const;

	void set_modulate(const Color &p_color);
	Color get_modulate() const;

	void set_outline_modulate(const Color &p_color);
	Color get_outline_modulate() const;

	void set_autowrap(bool p_mode);
	bool get_autowrap() const;

	void set_width(float p_width);
	float get_width() const;

	void set_pixel_size(real_t p_amount);
	real_t get_pixel_size() const;

	void set_offset(const Point2 &p_offset);
	Point2 get_offset() const;

	void set_draw_flag(DrawFlags p_flag, bool p_enable);
	bool get_draw_flag(DrawFlags p_flag) const;

	void set_alpha_cut_mode(AlphaCutMode p_mode);
	AlphaCutMode get_alpha_cut_mode() const;

	void set_alpha_scissor_threshold(float p_threshold);
	float get_alpha_scissor_threshold() const;

	void set_billboard_mode(SpatialMaterial::BillboardMode p_mode);
	SpatialMaterial::BillboardMode get_billboard_mode() const;

	virtual AABB get_aabb() const;
	Ref<TriangleMesh> generate_triangle_mesh() const;

	virtual PoolVector<Face3> get_faces(uint32_t p_usage_flags) const;

	Label3D();
	~Label3D();
};

VARIANT_ENUM_CAST(Label3D::DrawFlags);
VARIANT_ENUM_CAST(Label3D::AlphaCutMode);
VARIANT_ENUM_CAST(Label3D::Align);
VARIANT_ENUM_CAST(Label3D::VAlign);

#endif // LABEL_3D_H
