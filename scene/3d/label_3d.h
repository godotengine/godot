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

#include "scene/3d/visual_instance_3d.h"
#include "scene/resources/font.h"
#include "scene/resources/text_paragraph.h"

class Label3D : public GeometryInstance3D {
	GDCLASS(Label3D, GeometryInstance3D);

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

private:
	real_t pixel_size = 0.005;
	bool flags[FLAG_MAX] = {};
	AlphaCutMode alpha_cut = ALPHA_CUT_DISABLED;
	float alpha_scissor_threshold = 0.5;

	AABB aabb;

	mutable Ref<TriangleMesh> triangle_mesh;
	RID mesh;
	struct SurfaceData {
		PackedVector3Array mesh_vertices;
		PackedVector3Array mesh_normals;
		PackedFloat32Array mesh_tangents;
		PackedColorArray mesh_colors;
		PackedVector2Array mesh_uvs;
		PackedInt32Array indices;
		int offset = 0;
		float z_shift = 0.0;
		RID material;
	};

	struct SurfaceKey {
		uint64_t texture_id;
		int32_t priority;
		int32_t outline_size;

		bool operator==(const SurfaceKey &p_b) const {
			return (texture_id == p_b.texture_id) && (priority == p_b.priority) && (outline_size == p_b.outline_size);
		}

		SurfaceKey(uint64_t p_texture_id, int p_priority, int p_outline_size) {
			texture_id = p_texture_id;
			priority = p_priority;
			outline_size = p_outline_size;
		}
	};

	struct SurfaceKeyHasher {
		_FORCE_INLINE_ static uint32_t hash(const SurfaceKey &p_a) {
			return hash_murmur3_buffer(&p_a, sizeof(SurfaceKey));
		}
	};

	HashMap<SurfaceKey, SurfaceData, SurfaceKeyHasher> surfaces;

	String text;
	String xl_text;
	TextServer::AutowrapMode autowrap_mode = TextServer::AUTOWRAP_OFF;
	bool uppercase = false;

	bool text_set = false;
	Ref<TextParagraph> text_para;

	Point2 lbl_offset;

	Ref<Font> font_override;
	mutable Ref<Font> theme_font;

	int font_size = 32;
	Color modulate = Color(1, 1, 1, 1);
	int render_priority = 0;

	int outline_size = 12;
	Color outline_modulate = Color(0, 0, 0, 1);
	int outline_render_priority = -1;

	String language;
	TextServer::StructuredTextParser st_parser = TextServer::STRUCTURED_TEXT_DEFAULT;
	Array st_args;

	RID base_material;
	StandardMaterial3D::BillboardMode billboard_mode = StandardMaterial3D::BILLBOARD_DISABLED;
	StandardMaterial3D::TextureFilter texture_filter = StandardMaterial3D::TEXTURE_FILTER_LINEAR_WITH_MIPMAPS;

	bool pending_update = false;

	void _generate_glyph_surfaces(const Glyph &p_glyph, const Vector2 &p_offset, const Color &p_modulate, int p_priority = 0, int p_outline_size = 0);
	void _update_text();
	void _update_fonts();
	void _invalidate_fonts();

protected:
	GDVIRTUAL2RC(Array, _structured_text_parser, Array, String)

	void _notification(int p_what);
	static void _bind_methods();

	void _validate_property(PropertyInfo &p_property) const;

	void _im_update();
	void _queue_update();

public:
	virtual PackedStringArray get_configuration_warnings() const override;

	void set_horizontal_alignment(HorizontalAlignment p_alignment);
	HorizontalAlignment get_horizontal_alignment() const;

	void set_vertical_alignment(VerticalAlignment p_alignment);
	VerticalAlignment get_vertical_alignment() const;

	void set_render_priority(int p_priority);
	int get_render_priority() const;

	void set_outline_render_priority(int p_priority);
	int get_outline_render_priority() const;

	void set_text(const String &p_string);
	String get_text() const;

	void set_text_direction(TextServer::Direction p_text_direction);
	TextServer::Direction get_text_direction() const;

	void set_orientation(TextServer::Orientation p_orientation);
	TextServer::Orientation get_orientation() const;

	void set_uniform_line_height(bool p_enabled);
	bool get_uniform_line_height() const;

	void set_invert_line_order(bool p_enabled);
	bool get_invert_line_order() const;

	void set_language(const String &p_language);
	String get_language() const;

	void set_structured_text_bidi_override(TextServer::StructuredTextParser p_parser);
	TextServer::StructuredTextParser get_structured_text_bidi_override() const;

	void set_structured_text_bidi_override_options(Array p_args);
	Array get_structured_text_bidi_override_options() const;

	void set_uppercase(bool p_uppercase);
	bool is_uppercase() const;

	void set_font(const Ref<Font> &p_font);
	Ref<Font> get_font() const;
	Ref<Font> _get_font_or_default() const;

	void set_font_size(int p_size);
	int get_font_size() const;

	void set_outline_size(int p_size);
	int get_outline_size() const;

	void set_line_spacing(float p_size);
	float get_line_spacing() const;

	void set_modulate(const Color &p_color);
	Color get_modulate() const;

	void set_outline_modulate(const Color &p_color);
	Color get_outline_modulate() const;

	void set_autowrap_mode(TextServer::AutowrapMode p_mode);
	TextServer::AutowrapMode get_autowrap_mode() const;

	void set_width(float p_width);
	float get_width() const;

	void set_height(float p_height);
	float get_height() const;

	void set_text_overrun_behavior(TextServer::OverrunBehavior p_behavior);
	TextServer::OverrunBehavior get_text_overrun_behavior() const;

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

	void set_billboard_mode(StandardMaterial3D::BillboardMode p_mode);
	StandardMaterial3D::BillboardMode get_billboard_mode() const;

	void set_texture_filter(StandardMaterial3D::TextureFilter p_filter);
	StandardMaterial3D::TextureFilter get_texture_filter() const;

	virtual AABB get_aabb() const override;
	Ref<TriangleMesh> generate_triangle_mesh() const;

	Label3D();
	~Label3D();
};

VARIANT_ENUM_CAST(Label3D::DrawFlags);
VARIANT_ENUM_CAST(Label3D::AlphaCutMode);

#endif // LABEL_3D_H
