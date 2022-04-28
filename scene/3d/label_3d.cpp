/*************************************************************************/
/*  label_3d.cpp                                                         */
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

#include "label_3d.h"

#include "core/core_string_names.h"
#include "scene/resources/theme.h"
#include "scene/scene_string_names.h"

void Label3D::_bind_methods() {
	ClassDB::bind_method(D_METHOD("set_horizontal_alignment", "alignment"), &Label3D::set_horizontal_alignment);
	ClassDB::bind_method(D_METHOD("get_horizontal_alignment"), &Label3D::get_horizontal_alignment);

	ClassDB::bind_method(D_METHOD("set_vertical_alignment", "alignment"), &Label3D::set_vertical_alignment);
	ClassDB::bind_method(D_METHOD("get_vertical_alignment"), &Label3D::get_vertical_alignment);

	ClassDB::bind_method(D_METHOD("set_modulate", "modulate"), &Label3D::set_modulate);
	ClassDB::bind_method(D_METHOD("get_modulate"), &Label3D::get_modulate);

	ClassDB::bind_method(D_METHOD("set_outline_modulate", "modulate"), &Label3D::set_outline_modulate);
	ClassDB::bind_method(D_METHOD("get_outline_modulate"), &Label3D::get_outline_modulate);

	ClassDB::bind_method(D_METHOD("set_text", "text"), &Label3D::set_text);
	ClassDB::bind_method(D_METHOD("get_text"), &Label3D::get_text);

	ClassDB::bind_method(D_METHOD("set_text_direction", "direction"), &Label3D::set_text_direction);
	ClassDB::bind_method(D_METHOD("get_text_direction"), &Label3D::get_text_direction);

	ClassDB::bind_method(D_METHOD("set_opentype_feature", "tag", "value"), &Label3D::set_opentype_feature);
	ClassDB::bind_method(D_METHOD("get_opentype_feature", "tag"), &Label3D::get_opentype_feature);
	ClassDB::bind_method(D_METHOD("clear_opentype_features"), &Label3D::clear_opentype_features);

	ClassDB::bind_method(D_METHOD("set_language", "language"), &Label3D::set_language);
	ClassDB::bind_method(D_METHOD("get_language"), &Label3D::get_language);

	ClassDB::bind_method(D_METHOD("set_structured_text_bidi_override", "parser"), &Label3D::set_structured_text_bidi_override);
	ClassDB::bind_method(D_METHOD("get_structured_text_bidi_override"), &Label3D::get_structured_text_bidi_override);

	ClassDB::bind_method(D_METHOD("set_structured_text_bidi_override_options", "args"), &Label3D::set_structured_text_bidi_override_options);
	ClassDB::bind_method(D_METHOD("get_structured_text_bidi_override_options"), &Label3D::get_structured_text_bidi_override_options);

	ClassDB::bind_method(D_METHOD("set_uppercase", "enable"), &Label3D::set_uppercase);
	ClassDB::bind_method(D_METHOD("is_uppercase"), &Label3D::is_uppercase);

	ClassDB::bind_method(D_METHOD("set_font", "font"), &Label3D::set_font);
	ClassDB::bind_method(D_METHOD("get_font"), &Label3D::get_font);

	ClassDB::bind_method(D_METHOD("set_font_size", "size"), &Label3D::set_font_size);
	ClassDB::bind_method(D_METHOD("get_font_size"), &Label3D::get_font_size);

	ClassDB::bind_method(D_METHOD("set_outline_size", "outline_size"), &Label3D::set_outline_size);
	ClassDB::bind_method(D_METHOD("get_outline_size"), &Label3D::get_outline_size);

	ClassDB::bind_method(D_METHOD("set_line_spacing", "line_spacing"), &Label3D::set_line_spacing);
	ClassDB::bind_method(D_METHOD("get_line_spacing"), &Label3D::get_line_spacing);

	ClassDB::bind_method(D_METHOD("set_autowrap_mode", "autowrap_mode"), &Label3D::set_autowrap_mode);
	ClassDB::bind_method(D_METHOD("get_autowrap_mode"), &Label3D::get_autowrap_mode);

	ClassDB::bind_method(D_METHOD("set_width", "width"), &Label3D::set_width);
	ClassDB::bind_method(D_METHOD("get_width"), &Label3D::get_width);

	ClassDB::bind_method(D_METHOD("set_pixel_size", "pixel_size"), &Label3D::set_pixel_size);
	ClassDB::bind_method(D_METHOD("get_pixel_size"), &Label3D::get_pixel_size);

	ClassDB::bind_method(D_METHOD("set_offset", "offset"), &Label3D::set_offset);
	ClassDB::bind_method(D_METHOD("get_offset"), &Label3D::get_offset);

	ClassDB::bind_method(D_METHOD("set_draw_flag", "flag", "enabled"), &Label3D::set_draw_flag);
	ClassDB::bind_method(D_METHOD("get_draw_flag", "flag"), &Label3D::get_draw_flag);

	ClassDB::bind_method(D_METHOD("set_billboard_mode", "mode"), &Label3D::set_billboard_mode);
	ClassDB::bind_method(D_METHOD("get_billboard_mode"), &Label3D::get_billboard_mode);

	ClassDB::bind_method(D_METHOD("set_alpha_cut_mode", "mode"), &Label3D::set_alpha_cut_mode);
	ClassDB::bind_method(D_METHOD("get_alpha_cut_mode"), &Label3D::get_alpha_cut_mode);

	ClassDB::bind_method(D_METHOD("set_alpha_scissor_threshold", "threshold"), &Label3D::set_alpha_scissor_threshold);
	ClassDB::bind_method(D_METHOD("get_alpha_scissor_threshold"), &Label3D::get_alpha_scissor_threshold);

	ClassDB::bind_method(D_METHOD("set_texture_filter", "mode"), &Label3D::set_texture_filter);
	ClassDB::bind_method(D_METHOD("get_texture_filter"), &Label3D::get_texture_filter);

	ClassDB::bind_method(D_METHOD("generate_triangle_mesh"), &Label3D::generate_triangle_mesh);

	ClassDB::bind_method(D_METHOD("_queue_update"), &Label3D::_queue_update);
	ClassDB::bind_method(D_METHOD("_font_changed"), &Label3D::_font_changed);
	ClassDB::bind_method(D_METHOD("_im_update"), &Label3D::_im_update);

	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "pixel_size", PROPERTY_HINT_RANGE, "0.0001,128,0.0001"), "set_pixel_size", "get_pixel_size");
	ADD_PROPERTY(PropertyInfo(Variant::VECTOR2, "offset"), "set_offset", "get_offset");

	ADD_GROUP("Flags", "");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "billboard", PROPERTY_HINT_ENUM, "Disabled,Enabled,Y-Billboard"), "set_billboard_mode", "get_billboard_mode");
	ADD_PROPERTYI(PropertyInfo(Variant::BOOL, "shaded"), "set_draw_flag", "get_draw_flag", FLAG_SHADED);
	ADD_PROPERTYI(PropertyInfo(Variant::BOOL, "double_sided"), "set_draw_flag", "get_draw_flag", FLAG_DOUBLE_SIDED);
	ADD_PROPERTYI(PropertyInfo(Variant::BOOL, "no_depth_test"), "set_draw_flag", "get_draw_flag", FLAG_DISABLE_DEPTH_TEST);
	ADD_PROPERTYI(PropertyInfo(Variant::BOOL, "fixed_size"), "set_draw_flag", "get_draw_flag", FLAG_FIXED_SIZE);
	ADD_PROPERTY(PropertyInfo(Variant::INT, "alpha_cut", PROPERTY_HINT_ENUM, "Disabled,Discard,Opaque Pre-Pass"), "set_alpha_cut_mode", "get_alpha_cut_mode");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "alpha_scissor_threshold", PROPERTY_HINT_RANGE, "0,1,0.01"), "set_alpha_scissor_threshold", "get_alpha_scissor_threshold");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "texture_filter", PROPERTY_HINT_ENUM, "Nearest,Linear,Nearest Mipmap,Linear Mipmap,Nearest Mipmap Anisotropic,Linear Mipmap Anisotropic"), "set_texture_filter", "get_texture_filter");

	ADD_GROUP("Text", "");
	ADD_PROPERTY(PropertyInfo(Variant::COLOR, "modulate"), "set_modulate", "get_modulate");
	ADD_PROPERTY(PropertyInfo(Variant::COLOR, "outline_modulate"), "set_outline_modulate", "get_outline_modulate");
	ADD_PROPERTY(PropertyInfo(Variant::STRING, "text", PROPERTY_HINT_MULTILINE_TEXT, ""), "set_text", "get_text");
	ADD_PROPERTY(PropertyInfo(Variant::OBJECT, "font", PROPERTY_HINT_RESOURCE_TYPE, "Font"), "set_font", "get_font");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "font_size", PROPERTY_HINT_RANGE, "1,127,1"), "set_font_size", "get_font_size");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "outline_size", PROPERTY_HINT_RANGE, "0,127,1"), "set_outline_size", "get_outline_size");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "horizontal_alignment", PROPERTY_HINT_ENUM, "Left,Center,Right,Fill"), "set_horizontal_alignment", "get_horizontal_alignment");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "vertical_alignment", PROPERTY_HINT_ENUM, "Top,Center,Bottom"), "set_vertical_alignment", "get_vertical_alignment");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "uppercase"), "set_uppercase", "is_uppercase");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "line_spacing"), "set_line_spacing", "get_line_spacing");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "autowrap_mode", PROPERTY_HINT_ENUM, "Off,Arbitrary,Word,Word (Smart)"), "set_autowrap_mode", "get_autowrap_mode");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "width"), "set_width", "get_width");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "structured_text_bidi_override", PROPERTY_HINT_ENUM, "Default,URI,File,Email,List,None,Custom"), "set_structured_text_bidi_override", "get_structured_text_bidi_override");
	ADD_PROPERTY(PropertyInfo(Variant::ARRAY, "structured_text_bidi_override_options"), "set_structured_text_bidi_override_options", "get_structured_text_bidi_override_options");

	ADD_GROUP("Locale", "");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "text_direction", PROPERTY_HINT_ENUM, "Auto,Left-to-Right,Right-to-Left"), "set_text_direction", "get_text_direction");
	ADD_PROPERTY(PropertyInfo(Variant::STRING, "language", PROPERTY_HINT_LOCALE_ID, ""), "set_language", "get_language");

	BIND_ENUM_CONSTANT(AUTOWRAP_OFF);
	BIND_ENUM_CONSTANT(AUTOWRAP_ARBITRARY);
	BIND_ENUM_CONSTANT(AUTOWRAP_WORD);
	BIND_ENUM_CONSTANT(AUTOWRAP_WORD_SMART);

	BIND_ENUM_CONSTANT(FLAG_SHADED);
	BIND_ENUM_CONSTANT(FLAG_DOUBLE_SIDED);
	BIND_ENUM_CONSTANT(FLAG_DISABLE_DEPTH_TEST);
	BIND_ENUM_CONSTANT(FLAG_FIXED_SIZE);
	BIND_ENUM_CONSTANT(FLAG_MAX);

	BIND_ENUM_CONSTANT(ALPHA_CUT_DISABLED);
	BIND_ENUM_CONSTANT(ALPHA_CUT_DISCARD);
	BIND_ENUM_CONSTANT(ALPHA_CUT_OPAQUE_PREPASS);
}

bool Label3D::_set(const StringName &p_name, const Variant &p_value) {
	String str = p_name;
	if (str.begins_with("opentype_features/")) {
		String name = str.get_slicec('/', 1);
		int32_t tag = TS->name_to_tag(name);
		int value = p_value;
		if (value == -1) {
			if (opentype_features.has(tag)) {
				opentype_features.erase(tag);
				dirty_font = true;
				_queue_update();
			}
		} else {
			if (!opentype_features.has(tag) || (int)opentype_features[tag] != value) {
				opentype_features[tag] = value;
				dirty_font = true;
				_queue_update();
			}
		}
		notify_property_list_changed();
		return true;
	}

	return false;
}

bool Label3D::_get(const StringName &p_name, Variant &r_ret) const {
	String str = p_name;
	if (str.begins_with("opentype_features/")) {
		String name = str.get_slicec('/', 1);
		int32_t tag = TS->name_to_tag(name);
		if (opentype_features.has(tag)) {
			r_ret = opentype_features[tag];
			return true;
		} else {
			r_ret = -1;
			return true;
		}
	}
	return false;
}

void Label3D::_get_property_list(List<PropertyInfo> *p_list) const {
	for (const Variant *ftr = opentype_features.next(nullptr); ftr != nullptr; ftr = opentype_features.next(ftr)) {
		String name = TS->tag_to_name(*ftr);
		p_list->push_back(PropertyInfo(Variant::INT, "opentype_features/" + name));
	}
	p_list->push_back(PropertyInfo(Variant::NIL, "opentype_features/_new", PROPERTY_HINT_NONE, "", PROPERTY_USAGE_EDITOR));
}

void Label3D::_validate_property(PropertyInfo &property) const {
	if (property.name == "material_override" || property.name == "material_overlay") {
		property.usage = PROPERTY_USAGE_NO_EDITOR;
	}
}

void Label3D::_notification(int p_what) {
	switch (p_what) {
		case NOTIFICATION_ENTER_TREE: {
			if (!pending_update) {
				_im_update();
			}
		} break;
		case NOTIFICATION_TRANSLATION_CHANGED: {
			String new_text = tr(text);
			if (new_text == xl_text) {
				return; // Nothing new.
			}
			xl_text = new_text;
			dirty_text = true;
			_queue_update();
		} break;
	}
}

void Label3D::_im_update() {
	_shape();

	triangle_mesh.unref();
	update_gizmos();

	pending_update = false;
}

void Label3D::_queue_update() {
	if (pending_update) {
		return;
	}

	pending_update = true;
	call_deferred(SceneStringNames::get_singleton()->_im_update);
}

AABB Label3D::get_aabb() const {
	return aabb;
}

Ref<TriangleMesh> Label3D::generate_triangle_mesh() const {
	if (triangle_mesh.is_valid()) {
		return triangle_mesh;
	}

	Ref<Font> font = _get_font_or_default();
	if (font.is_null()) {
		return Ref<TriangleMesh>();
	}

	Vector<Vector3> faces;
	faces.resize(6);
	Vector3 *facesw = faces.ptrw();

	float total_h = 0.0;
	float max_line_w = 0.0;
	for (int i = 0; i < lines_rid.size(); i++) {
		total_h += TS->shaped_text_get_size(lines_rid[i]).y + font->get_spacing(TextServer::SPACING_TOP) + font->get_spacing(TextServer::SPACING_BOTTOM) + line_spacing;
		max_line_w = MAX(max_line_w, TS->shaped_text_get_width(lines_rid[i]));
	}

	float vbegin = 0;
	switch (vertical_alignment) {
		case VERTICAL_ALIGNMENT_FILL:
		case VERTICAL_ALIGNMENT_TOP: {
			// Nothing.
		} break;
		case VERTICAL_ALIGNMENT_CENTER: {
			vbegin = (total_h - line_spacing) / 2.0;
		} break;
		case VERTICAL_ALIGNMENT_BOTTOM: {
			vbegin = (total_h - line_spacing);
		} break;
	}

	Vector2 offset = Vector2(0, vbegin);
	switch (horizontal_alignment) {
		case HORIZONTAL_ALIGNMENT_LEFT:
			break;
		case HORIZONTAL_ALIGNMENT_FILL:
		case HORIZONTAL_ALIGNMENT_CENTER: {
			offset.x = -max_line_w / 2.0;
		} break;
		case HORIZONTAL_ALIGNMENT_RIGHT: {
			offset.x = -max_line_w;
		} break;
	}

	Rect2 final_rect = Rect2(offset + lbl_offset, Size2(max_line_w, total_h));

	if (final_rect.size.x == 0 || final_rect.size.y == 0) {
		return Ref<TriangleMesh>();
	}

	real_t pixel_size = get_pixel_size();

	Vector2 vertices[4] = {

		(final_rect.position + Vector2(0, -final_rect.size.y)) * pixel_size,
		(final_rect.position + Vector2(final_rect.size.x, -final_rect.size.y)) * pixel_size,
		(final_rect.position + Vector2(final_rect.size.x, 0)) * pixel_size,
		final_rect.position * pixel_size,

	};

	static const int indices[6] = {
		0, 1, 2,
		0, 2, 3
	};

	for (int j = 0; j < 6; j++) {
		int i = indices[j];
		Vector3 vtx;
		vtx[0] = vertices[i][0];
		vtx[1] = vertices[i][1];
		facesw[j] = vtx;
	}

	triangle_mesh = Ref<TriangleMesh>(memnew(TriangleMesh));
	triangle_mesh->create(faces);

	return triangle_mesh;
}

void Label3D::_generate_glyph_surfaces(const Glyph &p_glyph, Vector2 &r_offset, const Color &p_modulate, int p_priority, int p_outline_size) {
	for (int j = 0; j < p_glyph.repeat; j++) {
		Vector2 gl_of;
		Vector2 gl_sz;
		Rect2 gl_uv;
		Size2 texs;
		RID tex;

		if (p_glyph.font_rid != RID()) {
			tex = TS->font_get_glyph_texture_rid(p_glyph.font_rid, Vector2i(p_glyph.font_size, p_outline_size), p_glyph.index);
			if (tex != RID()) {
				gl_of = (TS->font_get_glyph_offset(p_glyph.font_rid, Vector2i(p_glyph.font_size, p_outline_size), p_glyph.index) + Vector2(p_glyph.x_off, p_glyph.y_off)) * pixel_size;
				gl_sz = TS->font_get_glyph_size(p_glyph.font_rid, Vector2i(p_glyph.font_size, p_outline_size), p_glyph.index) * pixel_size;
				gl_uv = TS->font_get_glyph_uv_rect(p_glyph.font_rid, Vector2i(p_glyph.font_size, p_outline_size), p_glyph.index);
				texs = TS->font_get_glyph_texture_size(p_glyph.font_rid, Vector2i(p_glyph.font_size, p_outline_size), p_glyph.index);
			}
		} else {
			gl_sz = TS->get_hex_code_box_size(p_glyph.font_size, p_glyph.index) * pixel_size;
			gl_of = Vector2(0, -gl_sz.y);
		}

		bool msdf = TS->font_is_multichannel_signed_distance_field(p_glyph.font_rid);

		uint64_t mat_hash;
		if (tex != RID()) {
			mat_hash = hash_one_uint64(tex.get_id());
		} else {
			mat_hash = hash_one_uint64(0);
		}
		mat_hash = hash_djb2_one_64(p_priority | (p_outline_size << 31), mat_hash);

		if (!surfaces.has(mat_hash)) {
			SurfaceData surf;
			surf.material = RenderingServer::get_singleton()->material_create();
			// Set defaults for material, names need to match up those in StandardMaterial3D
			RS::get_singleton()->material_set_param(surf.material, "albedo", Color(1, 1, 1, 1));
			RS::get_singleton()->material_set_param(surf.material, "specular", 0.5);
			RS::get_singleton()->material_set_param(surf.material, "metallic", 0.0);
			RS::get_singleton()->material_set_param(surf.material, "roughness", 1.0);
			RS::get_singleton()->material_set_param(surf.material, "uv1_offset", Vector3(0, 0, 0));
			RS::get_singleton()->material_set_param(surf.material, "uv1_scale", Vector3(1, 1, 1));
			RS::get_singleton()->material_set_param(surf.material, "uv2_offset", Vector3(0, 0, 0));
			RS::get_singleton()->material_set_param(surf.material, "uv2_scale", Vector3(1, 1, 1));
			RS::get_singleton()->material_set_param(surf.material, "alpha_scissor_threshold", alpha_scissor_threshold);
			if (msdf) {
				RS::get_singleton()->material_set_param(surf.material, "msdf_pixel_range", TS->font_get_msdf_pixel_range(p_glyph.font_rid));
				RS::get_singleton()->material_set_param(surf.material, "msdf_outline_size", p_outline_size);
			}

			RID shader_rid;
			StandardMaterial3D::get_material_for_2d(get_draw_flag(FLAG_SHADED), true, get_draw_flag(FLAG_DOUBLE_SIDED), get_alpha_cut_mode() == ALPHA_CUT_DISCARD, get_alpha_cut_mode() == ALPHA_CUT_OPAQUE_PREPASS, get_billboard_mode() == StandardMaterial3D::BILLBOARD_ENABLED, get_billboard_mode() == StandardMaterial3D::BILLBOARD_FIXED_Y, msdf, get_draw_flag(FLAG_DISABLE_DEPTH_TEST), get_draw_flag(FLAG_FIXED_SIZE), texture_filter, &shader_rid);

			RS::get_singleton()->material_set_shader(surf.material, shader_rid);
			RS::get_singleton()->material_set_param(surf.material, "texture_albedo", tex);
			if (get_alpha_cut_mode() == ALPHA_CUT_DISABLED) {
				RS::get_singleton()->material_set_render_priority(surf.material, p_priority);
			} else {
				surf.z_shift = p_priority * pixel_size;
			}

			surfaces[mat_hash] = surf;
		}
		SurfaceData &s = surfaces[mat_hash];

		s.mesh_vertices.resize((s.offset + 1) * 4);
		s.mesh_normals.resize((s.offset + 1) * 4);
		s.mesh_tangents.resize((s.offset + 1) * 16);
		s.mesh_colors.resize((s.offset + 1) * 4);
		s.mesh_uvs.resize((s.offset + 1) * 4);

		s.mesh_vertices.write[(s.offset * 4) + 3] = Vector3(r_offset.x + gl_of.x, r_offset.y - gl_of.y - gl_sz.y, s.z_shift);
		s.mesh_vertices.write[(s.offset * 4) + 2] = Vector3(r_offset.x + gl_of.x + gl_sz.x, r_offset.y - gl_of.y - gl_sz.y, s.z_shift);
		s.mesh_vertices.write[(s.offset * 4) + 1] = Vector3(r_offset.x + gl_of.x + gl_sz.x, r_offset.y - gl_of.y, s.z_shift);
		s.mesh_vertices.write[(s.offset * 4) + 0] = Vector3(r_offset.x + gl_of.x, r_offset.y - gl_of.y, s.z_shift);

		for (int i = 0; i < 4; i++) {
			s.mesh_normals.write[(s.offset * 4) + i] = Vector3(0.0, 0.0, 1.0);
			s.mesh_tangents.write[(s.offset * 16) + (i * 4) + 0] = 1.0;
			s.mesh_tangents.write[(s.offset * 16) + (i * 4) + 1] = 0.0;
			s.mesh_tangents.write[(s.offset * 16) + (i * 4) + 2] = 0.0;
			s.mesh_tangents.write[(s.offset * 16) + (i * 4) + 3] = 1.0;
			s.mesh_colors.write[(s.offset * 4) + i] = p_modulate;
			s.mesh_uvs.write[(s.offset * 4) + i] = Vector2();

			if (aabb == AABB()) {
				aabb.position = s.mesh_vertices[(s.offset * 4) + i];
			} else {
				aabb.expand_to(s.mesh_vertices[(s.offset * 4) + i]);
			}
		}

		if (tex != RID()) {
			s.mesh_uvs.write[(s.offset * 4) + 3] = Vector2(gl_uv.position.x / texs.x, (gl_uv.position.y + gl_uv.size.y) / texs.y);
			s.mesh_uvs.write[(s.offset * 4) + 2] = Vector2((gl_uv.position.x + gl_uv.size.x) / texs.x, (gl_uv.position.y + gl_uv.size.y) / texs.y);
			s.mesh_uvs.write[(s.offset * 4) + 1] = Vector2((gl_uv.position.x + gl_uv.size.x) / texs.x, gl_uv.position.y / texs.y);
			s.mesh_uvs.write[(s.offset * 4) + 0] = Vector2(gl_uv.position.x / texs.x, gl_uv.position.y / texs.y);
		}

		s.indices.resize((s.offset + 1) * 6);
		s.indices.write[(s.offset * 6) + 0] = (s.offset * 4) + 0;
		s.indices.write[(s.offset * 6) + 1] = (s.offset * 4) + 1;
		s.indices.write[(s.offset * 6) + 2] = (s.offset * 4) + 2;
		s.indices.write[(s.offset * 6) + 3] = (s.offset * 4) + 0;
		s.indices.write[(s.offset * 6) + 4] = (s.offset * 4) + 2;
		s.indices.write[(s.offset * 6) + 5] = (s.offset * 4) + 3;

		s.offset++;
		r_offset.x += p_glyph.advance * pixel_size;
	}
}

void Label3D::_shape() {
	// Clear mesh.
	RS::get_singleton()->mesh_clear(mesh);
	aabb = AABB();

	// Clear materials.
	for (Map<uint64_t, SurfaceData>::Element *E = surfaces.front(); E; E = E->next()) {
		RenderingServer::get_singleton()->free(E->get().material);
	}
	surfaces.clear();

	Ref<Font> font = _get_font_or_default();
	ERR_FAIL_COND(font.is_null());

	// Update text buffer.
	if (dirty_text) {
		TS->shaped_text_clear(text_rid);
		TS->shaped_text_set_direction(text_rid, text_direction);

		String text = (uppercase) ? TS->string_to_upper(xl_text, language) : xl_text;
		TS->shaped_text_add_string(text_rid, text, font->get_rids(), font_size, opentype_features, language);

		Array stt;
		if (st_parser == TextServer::STRUCTURED_TEXT_CUSTOM) {
			GDVIRTUAL_CALL(_structured_text_parser, st_args, text, stt);
		} else {
			stt = TS->parse_structured_text(st_parser, st_args, text);
		}
		TS->shaped_text_set_bidi_override(text_rid, stt);

		dirty_text = false;
		dirty_font = false;
		dirty_lines = true;
	} else if (dirty_font) {
		int spans = TS->shaped_get_span_count(text_rid);
		for (int i = 0; i < spans; i++) {
			TS->shaped_set_span_update_font(text_rid, i, font->get_rids(), font_size, opentype_features);
		}

		dirty_font = false;
		dirty_lines = true;
	}

	if (dirty_lines) {
		for (int i = 0; i < lines_rid.size(); i++) {
			TS->free_rid(lines_rid[i]);
		}
		lines_rid.clear();

		uint16_t autowrap_flags = TextServer::BREAK_MANDATORY;
		switch (autowrap_mode) {
			case AUTOWRAP_WORD_SMART:
				autowrap_flags = TextServer::BREAK_WORD_BOUND_ADAPTIVE | TextServer::BREAK_MANDATORY;
				break;
			case AUTOWRAP_WORD:
				autowrap_flags = TextServer::BREAK_WORD_BOUND | TextServer::BREAK_MANDATORY;
				break;
			case AUTOWRAP_ARBITRARY:
				autowrap_flags = TextServer::BREAK_GRAPHEME_BOUND | TextServer::BREAK_MANDATORY;
				break;
			case AUTOWRAP_OFF:
				break;
		}
		PackedInt32Array line_breaks = TS->shaped_text_get_line_breaks(text_rid, width, 0, autowrap_flags);

		float max_line_w = 0.0;
		for (int i = 0; i < line_breaks.size(); i = i + 2) {
			RID line = TS->shaped_text_substr(text_rid, line_breaks[i], line_breaks[i + 1] - line_breaks[i]);
			max_line_w = MAX(max_line_w, TS->shaped_text_get_width(line));
			lines_rid.push_back(line);
		}

		if (horizontal_alignment == HORIZONTAL_ALIGNMENT_FILL) {
			for (int i = 0; i < lines_rid.size() - 1; i++) {
				TS->shaped_text_fit_to_width(lines_rid[i], (width > 0) ? width : max_line_w, TextServer::JUSTIFICATION_WORD_BOUND | TextServer::JUSTIFICATION_KASHIDA);
			}
		}
		dirty_lines = false;
	}

	// Generate surfaces and materials.
	float total_h = 0.0;
	for (int i = 0; i < lines_rid.size(); i++) {
		total_h += (TS->shaped_text_get_size(lines_rid[i]).y + font->get_spacing(TextServer::SPACING_TOP) + font->get_spacing(TextServer::SPACING_BOTTOM) + line_spacing) * pixel_size;
	}

	float vbegin = 0.0;
	switch (vertical_alignment) {
		case VERTICAL_ALIGNMENT_FILL:
		case VERTICAL_ALIGNMENT_TOP: {
			// Nothing.
		} break;
		case VERTICAL_ALIGNMENT_CENTER: {
			vbegin = (total_h - line_spacing * pixel_size) / 2.0;
		} break;
		case VERTICAL_ALIGNMENT_BOTTOM: {
			vbegin = (total_h - line_spacing * pixel_size);
		} break;
	}

	Vector2 offset = Vector2(0, vbegin + lbl_offset.y * pixel_size);
	for (int i = 0; i < lines_rid.size(); i++) {
		const Glyph *glyphs = TS->shaped_text_get_glyphs(lines_rid[i]);
		int gl_size = TS->shaped_text_get_glyph_count(lines_rid[i]);
		float line_width = TS->shaped_text_get_width(lines_rid[i]) * pixel_size;

		switch (horizontal_alignment) {
			case HORIZONTAL_ALIGNMENT_LEFT:
				offset.x = 0.0;
				break;
			case HORIZONTAL_ALIGNMENT_FILL:
			case HORIZONTAL_ALIGNMENT_CENTER: {
				offset.x = -line_width / 2.0;
			} break;
			case HORIZONTAL_ALIGNMENT_RIGHT: {
				offset.x = -line_width;
			} break;
		}
		offset.x += lbl_offset.x * pixel_size;
		offset.y -= (TS->shaped_text_get_ascent(lines_rid[i]) + font->get_spacing(TextServer::SPACING_TOP)) * pixel_size;

		if (outline_modulate.a != 0.0 && outline_size > 0) {
			// Outline surfaces.
			Vector2 ol_offset = offset;
			for (int j = 0; j < gl_size; j++) {
				_generate_glyph_surfaces(glyphs[j], ol_offset, outline_modulate, -1, outline_size);
			}
		}

		// Main text surfaces.
		for (int j = 0; j < gl_size; j++) {
			_generate_glyph_surfaces(glyphs[j], offset, modulate, 0);
		}
		offset.y -= (TS->shaped_text_get_descent(lines_rid[i]) + line_spacing + font->get_spacing(TextServer::SPACING_BOTTOM)) * pixel_size;
	}

	for (Map<uint64_t, SurfaceData>::Element *E = surfaces.front(); E; E = E->next()) {
		Array mesh_array;
		mesh_array.resize(RS::ARRAY_MAX);
		mesh_array[RS::ARRAY_VERTEX] = E->get().mesh_vertices;
		mesh_array[RS::ARRAY_NORMAL] = E->get().mesh_normals;
		mesh_array[RS::ARRAY_TANGENT] = E->get().mesh_tangents;
		mesh_array[RS::ARRAY_COLOR] = E->get().mesh_colors;
		mesh_array[RS::ARRAY_TEX_UV] = E->get().mesh_uvs;
		mesh_array[RS::ARRAY_INDEX] = E->get().indices;

		RS::SurfaceData sd;
		RS::get_singleton()->mesh_create_surface_data_from_arrays(&sd, RS::PRIMITIVE_TRIANGLES, mesh_array);

		sd.material = E->get().material;

		RS::get_singleton()->mesh_add_surface(mesh, sd);
	}
}

void Label3D::set_text(const String &p_string) {
	text = p_string;
	xl_text = tr(p_string);
	dirty_text = true;
	_queue_update();
}

String Label3D::get_text() const {
	return text;
}

void Label3D::set_horizontal_alignment(HorizontalAlignment p_alignment) {
	ERR_FAIL_INDEX((int)p_alignment, 4);
	if (horizontal_alignment != p_alignment) {
		if (horizontal_alignment == HORIZONTAL_ALIGNMENT_FILL || p_alignment == HORIZONTAL_ALIGNMENT_FILL) {
			dirty_lines = true; // Reshape lines.
		}
		horizontal_alignment = p_alignment;
		_queue_update();
	}
}

HorizontalAlignment Label3D::get_horizontal_alignment() const {
	return horizontal_alignment;
}

void Label3D::set_vertical_alignment(VerticalAlignment p_alignment) {
	ERR_FAIL_INDEX((int)p_alignment, 4);
	if (vertical_alignment != p_alignment) {
		vertical_alignment = p_alignment;
		_queue_update();
	}
}

VerticalAlignment Label3D::get_vertical_alignment() const {
	return vertical_alignment;
}

void Label3D::set_text_direction(TextServer::Direction p_text_direction) {
	ERR_FAIL_COND((int)p_text_direction < -1 || (int)p_text_direction > 3);
	if (text_direction != p_text_direction) {
		text_direction = p_text_direction;
		dirty_text = true;
		_queue_update();
	}
}

TextServer::Direction Label3D::get_text_direction() const {
	return text_direction;
}

void Label3D::clear_opentype_features() {
	opentype_features.clear();
	dirty_font = true;
	_queue_update();
}

void Label3D::set_opentype_feature(const String &p_name, int p_value) {
	int32_t tag = TS->name_to_tag(p_name);
	if (!opentype_features.has(tag) || (int)opentype_features[tag] != p_value) {
		opentype_features[tag] = p_value;
		dirty_font = true;
		_queue_update();
	}
}

int Label3D::get_opentype_feature(const String &p_name) const {
	int32_t tag = TS->name_to_tag(p_name);
	if (!opentype_features.has(tag)) {
		return -1;
	}
	return opentype_features[tag];
}

void Label3D::set_language(const String &p_language) {
	if (language != p_language) {
		language = p_language;
		dirty_text = true;
		_queue_update();
	}
}

String Label3D::get_language() const {
	return language;
}

void Label3D::set_structured_text_bidi_override(TextServer::StructuredTextParser p_parser) {
	if (st_parser != p_parser) {
		st_parser = p_parser;
		dirty_text = true;
		_queue_update();
	}
}

TextServer::StructuredTextParser Label3D::get_structured_text_bidi_override() const {
	return st_parser;
}

void Label3D::set_structured_text_bidi_override_options(Array p_args) {
	if (st_args != p_args) {
		st_args = p_args;
		dirty_text = true;
		_queue_update();
	}
}

Array Label3D::get_structured_text_bidi_override_options() const {
	return st_args;
}

void Label3D::set_uppercase(bool p_uppercase) {
	if (uppercase != p_uppercase) {
		uppercase = p_uppercase;
		dirty_text = true;
		_queue_update();
	}
}

bool Label3D::is_uppercase() const {
	return uppercase;
}

void Label3D::_font_changed() {
	dirty_font = true;
	_queue_update();
}

void Label3D::set_font(const Ref<Font> &p_font) {
	if (font_override != p_font) {
		if (font_override.is_valid()) {
			font_override->disconnect(CoreStringNames::get_singleton()->changed, Callable(this, "_font_changed"));
		}
		font_override = p_font;
		dirty_font = true;
		if (font_override.is_valid()) {
			font_override->connect(CoreStringNames::get_singleton()->changed, Callable(this, "_font_changed"));
		}
		_queue_update();
	}
}

Ref<Font> Label3D::get_font() const {
	return font_override;
}

Ref<Font> Label3D::_get_font_or_default() const {
	if (font_override.is_valid() && font_override->get_data_count() > 0) {
		return font_override;
	}

	// Check the project-defined Theme resource.
	if (Theme::get_project_default().is_valid()) {
		List<StringName> theme_types;
		Theme::get_project_default()->get_type_dependencies(get_class_name(), StringName(), &theme_types);

		for (const StringName &E : theme_types) {
			if (Theme::get_project_default()->has_theme_item(Theme::DATA_TYPE_FONT, "font", E)) {
				return Theme::get_project_default()->get_theme_item(Theme::DATA_TYPE_FONT, "font", E);
			}
		}
	}

	// Lastly, fall back on the items defined in the default Theme, if they exist.
	{
		List<StringName> theme_types;
		Theme::get_default()->get_type_dependencies(get_class_name(), StringName(), &theme_types);

		for (const StringName &E : theme_types) {
			if (Theme::get_default()->has_theme_item(Theme::DATA_TYPE_FONT, "font", E)) {
				return Theme::get_default()->get_theme_item(Theme::DATA_TYPE_FONT, "font", E);
			}
		}
	}

	// If they don't exist, use any type to return the default/empty value.
	return Theme::get_default()->get_theme_item(Theme::DATA_TYPE_FONT, "font", StringName());
}

void Label3D::set_font_size(int p_size) {
	if (font_size != p_size) {
		font_size = p_size;
		dirty_font = true;
		_queue_update();
	}
}

int Label3D::get_font_size() const {
	return font_size;
}

void Label3D::set_outline_size(int p_size) {
	if (outline_size != p_size) {
		outline_size = p_size;
		_queue_update();
	}
}

int Label3D::get_outline_size() const {
	return outline_size;
}

void Label3D::set_modulate(const Color &p_color) {
	if (modulate != p_color) {
		modulate = p_color;
		_queue_update();
	}
}

Color Label3D::get_modulate() const {
	return modulate;
}

void Label3D::set_outline_modulate(const Color &p_color) {
	if (outline_modulate != p_color) {
		outline_modulate = p_color;
		_queue_update();
	}
}

Color Label3D::get_outline_modulate() const {
	return outline_modulate;
}

void Label3D::set_autowrap_mode(Label3D::AutowrapMode p_mode) {
	if (autowrap_mode != p_mode) {
		autowrap_mode = p_mode;
		dirty_lines = true;
		_queue_update();
	}
}

Label3D::AutowrapMode Label3D::get_autowrap_mode() const {
	return autowrap_mode;
}

void Label3D::set_width(float p_width) {
	if (width != p_width) {
		width = p_width;
		dirty_lines = true;
		_queue_update();
	}
}

float Label3D::get_width() const {
	return width;
}

void Label3D::set_pixel_size(real_t p_amount) {
	if (pixel_size != p_amount) {
		pixel_size = p_amount;
		_queue_update();
	}
}

real_t Label3D::get_pixel_size() const {
	return pixel_size;
}

void Label3D::set_offset(const Point2 &p_offset) {
	if (lbl_offset != p_offset) {
		lbl_offset = p_offset;
		_queue_update();
	}
}

Point2 Label3D::get_offset() const {
	return lbl_offset;
}

void Label3D::set_line_spacing(float p_line_spacing) {
	if (line_spacing != p_line_spacing) {
		line_spacing = p_line_spacing;
		_queue_update();
	}
}

float Label3D::get_line_spacing() const {
	return line_spacing;
}

void Label3D::set_draw_flag(DrawFlags p_flag, bool p_enable) {
	ERR_FAIL_INDEX(p_flag, FLAG_MAX);
	if (flags[p_flag] != p_enable) {
		flags[p_flag] = p_enable;
		_queue_update();
	}
}

bool Label3D::get_draw_flag(DrawFlags p_flag) const {
	ERR_FAIL_INDEX_V(p_flag, FLAG_MAX, false);
	return flags[p_flag];
}

void Label3D::set_billboard_mode(StandardMaterial3D::BillboardMode p_mode) {
	ERR_FAIL_INDEX(p_mode, 3);
	if (billboard_mode != p_mode) {
		billboard_mode = p_mode;
		_queue_update();
	}
}

StandardMaterial3D::BillboardMode Label3D::get_billboard_mode() const {
	return billboard_mode;
}

void Label3D::set_alpha_cut_mode(AlphaCutMode p_mode) {
	ERR_FAIL_INDEX(p_mode, 3);
	if (alpha_cut != p_mode) {
		alpha_cut = p_mode;
		_queue_update();
	}
}

void Label3D::set_texture_filter(StandardMaterial3D::TextureFilter p_filter) {
	if (texture_filter != p_filter) {
		texture_filter = p_filter;
		_queue_update();
	}
}

StandardMaterial3D::TextureFilter Label3D::get_texture_filter() const {
	return texture_filter;
}

Label3D::AlphaCutMode Label3D::get_alpha_cut_mode() const {
	return alpha_cut;
}

void Label3D::set_alpha_scissor_threshold(float p_threshold) {
	if (alpha_scissor_threshold != p_threshold) {
		alpha_scissor_threshold = p_threshold;
		_queue_update();
	}
}

float Label3D::get_alpha_scissor_threshold() const {
	return alpha_scissor_threshold;
}

Label3D::Label3D() {
	for (int i = 0; i < FLAG_MAX; i++) {
		flags[i] = (i == FLAG_DOUBLE_SIDED);
	}

	text_rid = TS->create_shaped_text();

	mesh = RenderingServer::get_singleton()->mesh_create();

	set_cast_shadows_setting(SHADOW_CASTING_SETTING_OFF);
	set_base(mesh);
}

Label3D::~Label3D() {
	for (int i = 0; i < lines_rid.size(); i++) {
		TS->free_rid(lines_rid[i]);
	}
	lines_rid.clear();

	TS->free_rid(text_rid);

	RenderingServer::get_singleton()->free(mesh);
	for (Map<uint64_t, SurfaceData>::Element *E = surfaces.front(); E; E = E->next()) {
		RenderingServer::get_singleton()->free(E->get().material);
	}
	surfaces.clear();
}
