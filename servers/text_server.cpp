/*************************************************************************/
/*  text_server.cpp                                                      */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2021 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2021 Godot Engine contributors (cf. AUTHORS.md).   */
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

#include "servers/text_server.h"
#include "scene/main/canvas_item.h"

TextServerManager *TextServerManager::singleton = nullptr;
TextServer *TextServerManager::server = nullptr;
TextServerManager::TextServerCreate TextServerManager::server_create_functions[TextServerManager::MAX_SERVERS];
int TextServerManager::server_create_count = 0;

void TextServerManager::_bind_methods() {
	ClassDB::bind_method(D_METHOD("get_interface_count"), &TextServerManager::_get_interface_count);
	ClassDB::bind_method(D_METHOD("get_interface_name", "index"), &TextServerManager::_get_interface_name);
	ClassDB::bind_method(D_METHOD("get_interface_features", "index"), &TextServerManager::_get_interface_features);
	ClassDB::bind_method(D_METHOD("get_interface", "index"), &TextServerManager::_get_interface);
	ClassDB::bind_method(D_METHOD("get_interfaces"), &TextServerManager::_get_interfaces);
	ClassDB::bind_method(D_METHOD("find_interface", "name"), &TextServerManager::_find_interface);

	ClassDB::bind_method(D_METHOD("set_primary_interface", "index"), &TextServerManager::_set_primary_interface);
	ClassDB::bind_method(D_METHOD("get_primary_interface"), &TextServerManager::_get_primary_interface);
}

void TextServerManager::register_create_function(const String &p_name, uint32_t p_features, TextServerManager::CreateFunction p_function, void *p_user_data) {
	ERR_FAIL_COND(server_create_count == MAX_SERVERS);
	server_create_functions[server_create_count].name = p_name;
	server_create_functions[server_create_count].create_function = p_function;
	server_create_functions[server_create_count].user_data = p_user_data;
	server_create_functions[server_create_count].features = p_features;
	server_create_count++;
}

int TextServerManager::get_interface_count() {
	return server_create_count;
}

String TextServerManager::get_interface_name(int p_index) {
	ERR_FAIL_INDEX_V(p_index, server_create_count, String());
	return server_create_functions[p_index].name;
}

uint32_t TextServerManager::get_interface_features(int p_index) {
	ERR_FAIL_INDEX_V(p_index, server_create_count, 0);
	return server_create_functions[p_index].features;
}

TextServer *TextServerManager::initialize(int p_index, Error &r_error) {
	ERR_FAIL_INDEX_V(p_index, server_create_count, nullptr);
	if (server_create_functions[p_index].instance == nullptr) {
		server_create_functions[p_index].instance = server_create_functions[p_index].create_function(r_error, server_create_functions[p_index].user_data);
		if (server_create_functions[p_index].instance != nullptr) {
			server_create_functions[p_index].instance->load_support_data(""); // Try loading default data.
		}
	}
	if (server_create_functions[p_index].instance != nullptr) {
		server = server_create_functions[p_index].instance;
		if (OS::get_singleton()->get_main_loop()) {
			OS::get_singleton()->get_main_loop()->notification(MainLoop::NOTIFICATION_TEXT_SERVER_CHANGED);
		}
	}
	return server_create_functions[p_index].instance;
}

TextServer *TextServerManager::get_primary_interface() {
	return server;
}

int TextServerManager::_get_interface_count() const {
	return server_create_count;
}

String TextServerManager::_get_interface_name(int p_index) const {
	return get_interface_name(p_index);
}

uint32_t TextServerManager::_get_interface_features(int p_index) const {
	return get_interface_features(p_index);
}

TextServer *TextServerManager::_get_interface(int p_index) const {
	ERR_FAIL_INDEX_V(p_index, server_create_count, nullptr);
	if (server_create_functions[p_index].instance == nullptr) {
		Error error;
		server_create_functions[p_index].instance = server_create_functions[p_index].create_function(error, server_create_functions[p_index].user_data);
		if (server_create_functions[p_index].instance != nullptr) {
			server_create_functions[p_index].instance->load_support_data(""); // Try loading default data.
		}
	}
	return server_create_functions[p_index].instance;
}

TextServer *TextServerManager::_find_interface(const String &p_name) const {
	for (int i = 0; i < server_create_count; i++) {
		if (server_create_functions[i].name == p_name) {
			return _get_interface(i);
		}
	}
	return nullptr;
}

Array TextServerManager::_get_interfaces() const {
	Array ret;

	for (int i = 0; i < server_create_count; i++) {
		Dictionary iface_info;

		iface_info["id"] = i;
		iface_info["name"] = server_create_functions[i].name;

		ret.push_back(iface_info);
	};

	return ret;
};

bool TextServerManager::_set_primary_interface(int p_index) {
	Error error;
	TextServerManager::initialize(p_index, error);
	return (error == OK);
}

TextServer *TextServerManager::_get_primary_interface() const {
	return server;
}

TextServerManager::TextServerManager() {
	singleton = this;
}

TextServerManager::~TextServerManager() {
	singleton = nullptr;
	for (int i = 0; i < server_create_count; i++) {
		if (server_create_functions[i].instance != nullptr) {
			memdelete(server_create_functions[i].instance);
			server_create_functions[i].instance = nullptr;
		}
	}
}

/*************************************************************************/

bool TextServer::Glyph::operator==(const Glyph &p_a) const {
	return (p_a.index == index) && (p_a.font_rid == font_rid) && (p_a.font_size == font_size) && (p_a.start == start);
}

bool TextServer::Glyph::operator!=(const Glyph &p_a) const {
	return (p_a.index != index) || (p_a.font_rid != font_rid) || (p_a.font_size != font_size) || (p_a.start != start);
}

bool TextServer::Glyph::operator<(const Glyph &p_a) const {
	if (p_a.start == start) {
		if (p_a.count == count) {
			if ((p_a.flags & GRAPHEME_IS_VIRTUAL) == GRAPHEME_IS_VIRTUAL) {
				return true;
			} else {
				return false;
			}
		}
		return p_a.count > count;
	}
	return p_a.start < start;
}

bool TextServer::Glyph::operator>(const Glyph &p_a) const {
	if (p_a.start == start) {
		if (p_a.count == count) {
			if ((p_a.flags & GRAPHEME_IS_VIRTUAL) == GRAPHEME_IS_VIRTUAL) {
				return false;
			} else {
				return true;
			}
		}
		return p_a.count < count;
	}
	return p_a.start > start;
}

void TextServer::_bind_methods() {
	ClassDB::bind_method(D_METHOD("has_feature", "feature"), &TextServer::has_feature);
	ClassDB::bind_method(D_METHOD("get_name"), &TextServer::get_name);
	ClassDB::bind_method(D_METHOD("load_support_data", "filename"), &TextServer::load_support_data);

	ClassDB::bind_method(D_METHOD("is_locale_right_to_left", "locale"), &TextServer::is_locale_right_to_left);

	ClassDB::bind_method(D_METHOD("name_to_tag", "name"), &TextServer::name_to_tag);
	ClassDB::bind_method(D_METHOD("tag_to_name", "tag"), &TextServer::tag_to_name);

	ClassDB::bind_method(D_METHOD("has", "rid"), &TextServer::has);
	ClassDB::bind_method(D_METHOD("free_rid", "rid"), &TextServer::free); // shouldn't conflict with Object::free()

	/* Font Interface */

	ClassDB::bind_method(D_METHOD("create_font"), &TextServer::create_font);

	ClassDB::bind_method(D_METHOD("font_set_data", "data"), &TextServer::font_set_data);

	ClassDB::bind_method(D_METHOD("font_set_antialiased", "font_rid", "antialiased"), &TextServer::font_set_antialiased);
	ClassDB::bind_method(D_METHOD("font_is_antialiased", "font_rid"), &TextServer::font_is_antialiased);

	ClassDB::bind_method(D_METHOD("font_set_multichannel_signed_distance_field", "font_rid", "msdf"), &TextServer::font_set_multichannel_signed_distance_field);
	ClassDB::bind_method(D_METHOD("font_is_multichannel_signed_distance_field", "font_rid"), &TextServer::font_is_multichannel_signed_distance_field);

	ClassDB::bind_method(D_METHOD("font_set_msdf_pixel_range", "font_rid", "msdf_pixel_range"), &TextServer::font_set_msdf_pixel_range);
	ClassDB::bind_method(D_METHOD("font_get_msdf_pixel_range", "font_rid"), &TextServer::font_get_msdf_pixel_range);

	ClassDB::bind_method(D_METHOD("font_set_msdf_size", "font_rid", "msdf_size"), &TextServer::font_set_msdf_size);
	ClassDB::bind_method(D_METHOD("font_get_msdf_size", "font_rid"), &TextServer::font_get_msdf_size);

	ClassDB::bind_method(D_METHOD("font_set_fixed_size", "font_rid", "fixed_size"), &TextServer::font_set_fixed_size);
	ClassDB::bind_method(D_METHOD("font_get_fixed_size", "font_rid"), &TextServer::font_get_fixed_size);

	ClassDB::bind_method(D_METHOD("font_set_force_autohinter", "font_rid", "force_autohinter"), &TextServer::font_set_force_autohinter);
	ClassDB::bind_method(D_METHOD("font_is_force_autohinter", "font_rid"), &TextServer::font_is_force_autohinter);

	ClassDB::bind_method(D_METHOD("font_set_hinting", "font_rid", "_hinting"), &TextServer::font_set_hinting);
	ClassDB::bind_method(D_METHOD("font_get_hinting", "font_rid"), &TextServer::font_get_hinting);

	ClassDB::bind_method(D_METHOD("font_set_variation_coordinates", "font_rid", "variation_coordinates"), &TextServer::font_set_variation_coordinates);
	ClassDB::bind_method(D_METHOD("font_get_variation_coordinates", "font_rid"), &TextServer::font_get_variation_coordinates);

	ClassDB::bind_method(D_METHOD("font_set_oversampling", "font_rid", "oversampling"), &TextServer::font_set_oversampling);
	ClassDB::bind_method(D_METHOD("font_get_oversampling", "font_rid"), &TextServer::font_get_oversampling);

	ClassDB::bind_method(D_METHOD("font_get_size_cache_list", "font_rid"), &TextServer::font_get_size_cache_list);
	ClassDB::bind_method(D_METHOD("font_clear_size_cache", "font_rid"), &TextServer::font_clear_size_cache);
	ClassDB::bind_method(D_METHOD("font_remove_size_cache", "font_rid", "size"), &TextServer::font_remove_size_cache);

	ClassDB::bind_method(D_METHOD("font_set_ascent", "font_rid", "size", "ascent"), &TextServer::font_set_ascent);
	ClassDB::bind_method(D_METHOD("font_get_ascent", "font_rid", "size"), &TextServer::font_get_ascent);

	ClassDB::bind_method(D_METHOD("font_set_descent", "font_rid", "size", "descent"), &TextServer::font_set_descent);
	ClassDB::bind_method(D_METHOD("font_get_descent", "font_rid", "size"), &TextServer::font_get_descent);

	ClassDB::bind_method(D_METHOD("font_set_underline_position", "font_rid", "size", "underline_position"), &TextServer::font_set_underline_position);
	ClassDB::bind_method(D_METHOD("font_get_underline_position", "font_rid", "size"), &TextServer::font_get_underline_position);

	ClassDB::bind_method(D_METHOD("font_set_underline_thickness", "font_rid", "size", "underline_thickness"), &TextServer::font_set_underline_thickness);
	ClassDB::bind_method(D_METHOD("font_get_underline_thickness", "font_rid", "size"), &TextServer::font_get_underline_thickness);

	ClassDB::bind_method(D_METHOD("font_set_scale", "font_rid", "size", "scale"), &TextServer::font_set_scale);
	ClassDB::bind_method(D_METHOD("font_get_scale", "font_rid", "size"), &TextServer::font_get_scale);

	ClassDB::bind_method(D_METHOD("font_set_spacing", "font_rid", "size", "spacing", "value"), &TextServer::font_set_spacing);
	ClassDB::bind_method(D_METHOD("font_get_spacing", "font_rid", "size", "spacing"), &TextServer::font_get_spacing);

	ClassDB::bind_method(D_METHOD("font_get_texture_count", "font_rid", "size"), &TextServer::font_get_texture_count);
	ClassDB::bind_method(D_METHOD("font_clear_textures", "font_rid", "size"), &TextServer::font_clear_textures);
	ClassDB::bind_method(D_METHOD("font_remove_texture", "font_rid", "size", "texture_index"), &TextServer::font_remove_texture);

	ClassDB::bind_method(D_METHOD("font_set_texture_image", "font_rid", "size", "texture_index", "image"), &TextServer::font_set_texture_image);
	ClassDB::bind_method(D_METHOD("font_get_texture_image", "font_rid", "size", "texture_index"), &TextServer::font_get_texture_image);

	ClassDB::bind_method(D_METHOD("font_set_texture_offsets", "font_rid", "size", "texture_index", "offset"), &TextServer::font_set_texture_offsets);
	ClassDB::bind_method(D_METHOD("font_get_texture_offsets", "font_rid", "size", "texture_index"), &TextServer::font_get_texture_offsets);

	ClassDB::bind_method(D_METHOD("font_get_glyph_list", "font_rid", "size"), &TextServer::font_get_glyph_list);
	ClassDB::bind_method(D_METHOD("font_clear_glyphs", "font_rid", "size"), &TextServer::font_clear_glyphs);
	ClassDB::bind_method(D_METHOD("font_remove_glyph", "font_rid", "size", "glyph"), &TextServer::font_remove_glyph);

	ClassDB::bind_method(D_METHOD("font_get_glyph_advance", "font_rid", "size", "glyph"), &TextServer::font_get_glyph_advance);
	ClassDB::bind_method(D_METHOD("font_set_glyph_advance", "font_rid", "size", "glyph", "advance"), &TextServer::font_set_glyph_advance);

	ClassDB::bind_method(D_METHOD("font_get_glyph_offset", "font_rid", "size", "glyph"), &TextServer::font_get_glyph_offset);
	ClassDB::bind_method(D_METHOD("font_set_glyph_offset", "font_rid", "size", "glyph", "offset"), &TextServer::font_set_glyph_offset);

	ClassDB::bind_method(D_METHOD("font_get_glyph_size", "font_rid", "size", "glyph"), &TextServer::font_get_glyph_size);
	ClassDB::bind_method(D_METHOD("font_set_glyph_size", "font_rid", "size", "glyph", "gl_size"), &TextServer::font_set_glyph_size);

	ClassDB::bind_method(D_METHOD("font_get_glyph_uv_rect", "font_rid", "size", "glyph"), &TextServer::font_get_glyph_uv_rect);
	ClassDB::bind_method(D_METHOD("font_set_glyph_uv_rect", "font_rid", "size", "glyph", "uv_rect"), &TextServer::font_set_glyph_uv_rect);

	ClassDB::bind_method(D_METHOD("font_get_glyph_texture_idx", "font_rid", "size", "glyph"), &TextServer::font_get_glyph_texture_idx);
	ClassDB::bind_method(D_METHOD("font_set_glyph_texture_idx", "font_rid", "size", "glyph", "texture_idx"), &TextServer::font_set_glyph_texture_idx);

	ClassDB::bind_method(D_METHOD("font_get_glyph_contours", "font", "size", "index"), &TextServer::_font_get_glyph_contours);

	ClassDB::bind_method(D_METHOD("font_get_kerning_list", "font_rid", "size"), &TextServer::font_get_kerning_list);
	ClassDB::bind_method(D_METHOD("font_clear_kerning_map", "font_rid", "size"), &TextServer::font_clear_kerning_map);
	ClassDB::bind_method(D_METHOD("font_remove_kerning", "font_rid", "size", "glyph_pair"), &TextServer::font_remove_kerning);

	ClassDB::bind_method(D_METHOD("font_set_kerning", "font_rid", "size", "glyph_pair", "kerning"), &TextServer::font_set_kerning);
	ClassDB::bind_method(D_METHOD("font_get_kerning", "font_rid", "size", "glyph_pair"), &TextServer::font_get_kerning);

	ClassDB::bind_method(D_METHOD("font_get_glyph_index", "font_rid", "size", "char", "variation_selector"), &TextServer::font_get_glyph_index);

	ClassDB::bind_method(D_METHOD("font_has_char", "font_rid", "char"), &TextServer::font_has_char);
	ClassDB::bind_method(D_METHOD("font_get_supported_chars", "font_rid"), &TextServer::font_get_supported_chars);

	ClassDB::bind_method(D_METHOD("font_render_range", "font_rid", "size", "start", "end"), &TextServer::font_render_range);
	ClassDB::bind_method(D_METHOD("font_render_glyph", "font_rid", "size", "index"), &TextServer::font_render_glyph);

	ClassDB::bind_method(D_METHOD("font_draw_glyph", "font_rid", "canvas", "size", "pos", "index", "color"), &TextServer::font_draw_glyph, DEFVAL(Color(1, 1, 1)));
	ClassDB::bind_method(D_METHOD("font_draw_glyph_outline", "font_rid", "canvas", "size", "outline_size", "pos", "index", "color"), &TextServer::font_draw_glyph_outline, DEFVAL(Color(1, 1, 1)));

	ClassDB::bind_method(D_METHOD("font_is_language_supported", "font_rid", "language"), &TextServer::font_is_language_supported);
	ClassDB::bind_method(D_METHOD("font_set_language_support_override", "font_rid", "language", "supported"), &TextServer::font_set_language_support_override);
	ClassDB::bind_method(D_METHOD("font_get_language_support_override", "font_rid", "language"), &TextServer::font_get_language_support_override);
	ClassDB::bind_method(D_METHOD("font_remove_language_support_override", "font_rid", "language"), &TextServer::font_remove_language_support_override);
	ClassDB::bind_method(D_METHOD("font_get_language_support_overrides", "font_rid"), &TextServer::font_get_language_support_overrides);

	ClassDB::bind_method(D_METHOD("font_is_script_supported", "font_rid", "script"), &TextServer::font_is_script_supported);
	ClassDB::bind_method(D_METHOD("font_set_script_support_override", "font_rid", "script", "supported"), &TextServer::font_set_script_support_override);
	ClassDB::bind_method(D_METHOD("font_get_script_support_override", "font_rid", "script"), &TextServer::font_get_script_support_override);
	ClassDB::bind_method(D_METHOD("font_remove_script_support_override", "font_rid", "script"), &TextServer::font_remove_script_support_override);
	ClassDB::bind_method(D_METHOD("font_get_script_support_overrides", "font_rid"), &TextServer::font_get_script_support_overrides);

	ClassDB::bind_method(D_METHOD("font_supported_feature_list", "font_rid"), &TextServer::font_supported_feature_list);
	ClassDB::bind_method(D_METHOD("font_supported_variation_list", "font_rid"), &TextServer::font_supported_variation_list);

	ClassDB::bind_method(D_METHOD("font_get_global_oversampling"), &TextServer::font_get_global_oversampling);
	ClassDB::bind_method(D_METHOD("font_set_global_oversampling", "oversampling"), &TextServer::font_set_global_oversampling);

	ClassDB::bind_method(D_METHOD("get_hex_code_box_size", "size", "index"), &TextServer::get_hex_code_box_size);
	ClassDB::bind_method(D_METHOD("draw_hex_code_box", "canvas", "size", "pos", "index", "color"), &TextServer::draw_hex_code_box);

	/* Shaped text buffer interface */

	ClassDB::bind_method(D_METHOD("create_shaped_text", "direction", "orientation"), &TextServer::create_shaped_text, DEFVAL(DIRECTION_AUTO), DEFVAL(ORIENTATION_HORIZONTAL));

	ClassDB::bind_method(D_METHOD("shaped_text_clear", "rid"), &TextServer::shaped_text_clear);

	ClassDB::bind_method(D_METHOD("shaped_text_set_direction", "shaped", "direction"), &TextServer::shaped_text_set_direction, DEFVAL(DIRECTION_AUTO));
	ClassDB::bind_method(D_METHOD("shaped_text_get_direction", "shaped"), &TextServer::shaped_text_get_direction);

	ClassDB::bind_method(D_METHOD("shaped_text_set_bidi_override", "shaped", "override"), &TextServer::_shaped_text_set_bidi_override);

	ClassDB::bind_method(D_METHOD("shaped_text_set_orientation", "shaped", "orientation"), &TextServer::shaped_text_set_orientation, DEFVAL(ORIENTATION_HORIZONTAL));
	ClassDB::bind_method(D_METHOD("shaped_text_get_orientation", "shaped"), &TextServer::shaped_text_get_orientation);

	ClassDB::bind_method(D_METHOD("shaped_text_set_preserve_invalid", "shaped", "enabled"), &TextServer::shaped_text_set_preserve_invalid);
	ClassDB::bind_method(D_METHOD("shaped_text_get_preserve_invalid", "shaped"), &TextServer::shaped_text_get_preserve_invalid);

	ClassDB::bind_method(D_METHOD("shaped_text_set_preserve_control", "shaped", "enabled"), &TextServer::shaped_text_set_preserve_control);
	ClassDB::bind_method(D_METHOD("shaped_text_get_preserve_control", "shaped"), &TextServer::shaped_text_get_preserve_control);

	ClassDB::bind_method(D_METHOD("shaped_text_add_string", "shaped", "text", "fonts", "size", "opentype_features", "language"), &TextServer::shaped_text_add_string, DEFVAL(Dictionary()), DEFVAL(""));
	ClassDB::bind_method(D_METHOD("shaped_text_add_object", "shaped", "key", "size", "inline_align", "length"), &TextServer::shaped_text_add_object, DEFVAL(INLINE_ALIGN_CENTER), DEFVAL(1));
	ClassDB::bind_method(D_METHOD("shaped_text_resize_object", "shaped", "key", "size", "inline_align"), &TextServer::shaped_text_resize_object, DEFVAL(INLINE_ALIGN_CENTER));

	ClassDB::bind_method(D_METHOD("shaped_text_substr", "shaped", "start", "length"), &TextServer::shaped_text_substr);
	ClassDB::bind_method(D_METHOD("shaped_text_get_parent", "shaped"), &TextServer::shaped_text_get_parent);
	ClassDB::bind_method(D_METHOD("shaped_text_fit_to_width", "shaped", "width", "jst_flags"), &TextServer::shaped_text_fit_to_width, DEFVAL(JUSTIFICATION_WORD_BOUND | JUSTIFICATION_KASHIDA));
	ClassDB::bind_method(D_METHOD("shaped_text_tab_align", "shaped", "tab_stops"), &TextServer::shaped_text_tab_align);

	ClassDB::bind_method(D_METHOD("shaped_text_shape", "shaped"), &TextServer::shaped_text_shape);
	ClassDB::bind_method(D_METHOD("shaped_text_is_ready", "shaped"), &TextServer::shaped_text_is_ready);

	ClassDB::bind_method(D_METHOD("shaped_text_get_glyphs", "shaped"), &TextServer::_shaped_text_get_glyphs);

	ClassDB::bind_method(D_METHOD("shaped_text_get_range", "shaped"), &TextServer::shaped_text_get_range);
	ClassDB::bind_method(D_METHOD("shaped_text_get_line_breaks_adv", "shaped", "width", "start", "once", "break_flags"), &TextServer::_shaped_text_get_line_breaks_adv, DEFVAL(0), DEFVAL(true), DEFVAL(BREAK_MANDATORY | BREAK_WORD_BOUND));
	ClassDB::bind_method(D_METHOD("shaped_text_get_line_breaks", "shaped", "width", "start", "break_flags"), &TextServer::_shaped_text_get_line_breaks, DEFVAL(0), DEFVAL(BREAK_MANDATORY | BREAK_WORD_BOUND));
	ClassDB::bind_method(D_METHOD("shaped_text_get_word_breaks", "shaped"), &TextServer::_shaped_text_get_word_breaks);

	ClassDB::bind_method(D_METHOD("shaped_text_overrun_trim_to_width", "shaped", "width", "overrun_trim_flags"), &TextServer::shaped_text_overrun_trim_to_width, DEFVAL(0), DEFVAL(OVERRUN_NO_TRIMMING));

	ClassDB::bind_method(D_METHOD("shaped_text_get_objects", "shaped"), &TextServer::shaped_text_get_objects);
	ClassDB::bind_method(D_METHOD("shaped_text_get_object_rect", "shaped", "key"), &TextServer::shaped_text_get_object_rect);

	ClassDB::bind_method(D_METHOD("shaped_text_get_size", "shaped"), &TextServer::shaped_text_get_size);
	ClassDB::bind_method(D_METHOD("shaped_text_get_ascent", "shaped"), &TextServer::shaped_text_get_ascent);
	ClassDB::bind_method(D_METHOD("shaped_text_get_descent", "shaped"), &TextServer::shaped_text_get_descent);
	ClassDB::bind_method(D_METHOD("shaped_text_get_width", "shaped"), &TextServer::shaped_text_get_width);
	ClassDB::bind_method(D_METHOD("shaped_text_get_underline_position", "shaped"), &TextServer::shaped_text_get_underline_position);
	ClassDB::bind_method(D_METHOD("shaped_text_get_underline_thickness", "shaped"), &TextServer::shaped_text_get_underline_thickness);

	ClassDB::bind_method(D_METHOD("shaped_text_get_carets", "shaped", "position"), &TextServer::_shaped_text_get_carets);
	ClassDB::bind_method(D_METHOD("shaped_text_get_selection", "shaped", "start", "end"), &TextServer::_shaped_text_get_selection);

	ClassDB::bind_method(D_METHOD("shaped_text_hit_test_grapheme", "shaped", "coords"), &TextServer::shaped_text_hit_test_grapheme);
	ClassDB::bind_method(D_METHOD("shaped_text_hit_test_position", "shaped", "coords"), &TextServer::shaped_text_hit_test_position);

	ClassDB::bind_method(D_METHOD("shaped_text_next_grapheme_pos", "shaped", "pos"), &TextServer::shaped_text_next_grapheme_pos);
	ClassDB::bind_method(D_METHOD("shaped_text_prev_grapheme_pos", "shaped", "pos"), &TextServer::shaped_text_prev_grapheme_pos);

	ClassDB::bind_method(D_METHOD("shaped_text_draw", "shaped", "canvas", "pos", "clip_l", "clip_r", "color"), &TextServer::shaped_text_draw, DEFVAL(-1), DEFVAL(-1), DEFVAL(Color(1, 1, 1)));
	ClassDB::bind_method(D_METHOD("shaped_text_draw_outline", "shaped", "canvas", "pos", "clip_l", "clip_r", "outline_size", "color"), &TextServer::shaped_text_draw_outline, DEFVAL(-1), DEFVAL(-1), DEFVAL(1), DEFVAL(Color(1, 1, 1)));

	ClassDB::bind_method(D_METHOD("shaped_text_get_dominant_direciton_in_range", "shaped", "start", "end"), &TextServer::shaped_text_get_dominant_direciton_in_range);

	ClassDB::bind_method(D_METHOD("format_number", "number", "language"), &TextServer::format_number, DEFVAL(""));
	ClassDB::bind_method(D_METHOD("parse_number", "number", "language"), &TextServer::parse_number, DEFVAL(""));
	ClassDB::bind_method(D_METHOD("percent_sign", "language"), &TextServer::percent_sign, DEFVAL(""));

	/* Direction */
	BIND_ENUM_CONSTANT(DIRECTION_AUTO);
	BIND_ENUM_CONSTANT(DIRECTION_LTR);
	BIND_ENUM_CONSTANT(DIRECTION_RTL);

	/* Orientation */
	BIND_ENUM_CONSTANT(ORIENTATION_HORIZONTAL);
	BIND_ENUM_CONSTANT(ORIENTATION_VERTICAL);

	/* JustificationFlag */
	BIND_ENUM_CONSTANT(JUSTIFICATION_NONE);
	BIND_ENUM_CONSTANT(JUSTIFICATION_KASHIDA);
	BIND_ENUM_CONSTANT(JUSTIFICATION_WORD_BOUND);
	BIND_ENUM_CONSTANT(JUSTIFICATION_TRIM_EDGE_SPACES);
	BIND_ENUM_CONSTANT(JUSTIFICATION_AFTER_LAST_TAB);

	/* LineBreakFlag */
	BIND_ENUM_CONSTANT(BREAK_NONE);
	BIND_ENUM_CONSTANT(BREAK_MANDATORY);
	BIND_ENUM_CONSTANT(BREAK_WORD_BOUND);
	BIND_ENUM_CONSTANT(BREAK_GRAPHEME_BOUND);

	/* TextOverrunFlag */
	BIND_ENUM_CONSTANT(OVERRUN_NO_TRIMMING);
	BIND_ENUM_CONSTANT(OVERRUN_TRIM);
	BIND_ENUM_CONSTANT(OVERRUN_TRIM_WORD_ONLY);
	BIND_ENUM_CONSTANT(OVERRUN_ADD_ELLIPSIS);
	BIND_ENUM_CONSTANT(OVERRUN_ENFORCE_ELLIPSIS);

	/* GraphemeFlag */
	BIND_ENUM_CONSTANT(GRAPHEME_IS_RTL);
	BIND_ENUM_CONSTANT(GRAPHEME_IS_VIRTUAL);
	BIND_ENUM_CONSTANT(GRAPHEME_IS_SPACE);
	BIND_ENUM_CONSTANT(GRAPHEME_IS_BREAK_HARD);
	BIND_ENUM_CONSTANT(GRAPHEME_IS_BREAK_SOFT);
	BIND_ENUM_CONSTANT(GRAPHEME_IS_TAB);
	BIND_ENUM_CONSTANT(GRAPHEME_IS_ELONGATION);
	BIND_ENUM_CONSTANT(GRAPHEME_IS_PUNCTUATION);
	BIND_ENUM_CONSTANT(GRAPHEME_IS_UNDERSCORE);
	BIND_ENUM_CONSTANT(GRAPHEME_IS_CONNECTED);

	/* Hinting */
	BIND_ENUM_CONSTANT(HINTING_NONE);
	BIND_ENUM_CONSTANT(HINTING_LIGHT);
	BIND_ENUM_CONSTANT(HINTING_NORMAL);

	/* Feature */
	BIND_ENUM_CONSTANT(FEATURE_BIDI_LAYOUT);
	BIND_ENUM_CONSTANT(FEATURE_VERTICAL_LAYOUT);
	BIND_ENUM_CONSTANT(FEATURE_SHAPING);
	BIND_ENUM_CONSTANT(FEATURE_KASHIDA_JUSTIFICATION);
	BIND_ENUM_CONSTANT(FEATURE_BREAK_ITERATORS);
	BIND_ENUM_CONSTANT(FEATURE_FONT_SYSTEM);
	BIND_ENUM_CONSTANT(FEATURE_FONT_VARIABLE);
	BIND_ENUM_CONSTANT(FEATURE_USE_SUPPORT_DATA);

	/* FT Contour Point Types */
	BIND_ENUM_CONSTANT(CONTOUR_CURVE_TAG_ON);
	BIND_ENUM_CONSTANT(CONTOUR_CURVE_TAG_OFF_CONIC);
	BIND_ENUM_CONSTANT(CONTOUR_CURVE_TAG_OFF_CUBIC);

	/* Font Spacing*/
	BIND_ENUM_CONSTANT(SPACING_GLYPH);
	BIND_ENUM_CONSTANT(SPACING_SPACE);
	BIND_ENUM_CONSTANT(SPACING_TOP);
	BIND_ENUM_CONSTANT(SPACING_BOTTOM);
}

Vector3 TextServer::hex_code_box_font_size[2] = { Vector3(5, 5, 1), Vector3(10, 10, 2) };
Ref<CanvasTexture> TextServer::hex_code_box_font_tex[2] = { nullptr, nullptr };

void TextServer::initialize_hex_code_box_fonts() {
	static unsigned int tamsyn5x9_png_len = 175;
	static unsigned char tamsyn5x9_png[] = {
		0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a, 0x00, 0x00, 0x00, 0x0d,
		0x49, 0x48, 0x44, 0x52, 0x00, 0x00, 0x00, 0x50, 0x00, 0x00, 0x00, 0x05,
		0x04, 0x03, 0x00, 0x00, 0x00, 0x20, 0x7c, 0x76, 0xda, 0x00, 0x00, 0x00,
		0x0f, 0x50, 0x4c, 0x54, 0x45, 0xfd, 0x07, 0x00, 0x00, 0x00, 0x00, 0x06,
		0x7e, 0x74, 0x00, 0x40, 0xc6, 0xff, 0xff, 0xff, 0x47, 0x9a, 0xd4, 0xc7,
		0x00, 0x00, 0x00, 0x01, 0x74, 0x52, 0x4e, 0x53, 0x00, 0x40, 0xe6, 0xd8,
		0x66, 0x00, 0x00, 0x00, 0x4e, 0x49, 0x44, 0x41, 0x54, 0x08, 0x1d, 0x05,
		0xc1, 0x21, 0x01, 0x00, 0x00, 0x00, 0x83, 0x30, 0x04, 0xc1, 0x10, 0xef,
		0x9f, 0xe9, 0x1b, 0x86, 0x2c, 0x17, 0xb9, 0xcc, 0x65, 0x0c, 0x73, 0x38,
		0xc7, 0xe6, 0x22, 0x19, 0x88, 0x98, 0x10, 0x48, 0x4a, 0x29, 0x85, 0x14,
		0x02, 0x89, 0x10, 0xa3, 0x1c, 0x0b, 0x31, 0xd6, 0xe6, 0x08, 0x69, 0x39,
		0x48, 0x44, 0xa0, 0x0d, 0x4a, 0x22, 0xa1, 0x94, 0x42, 0x0a, 0x01, 0x63,
		0x6d, 0x0e, 0x72, 0x18, 0x61, 0x8c, 0x74, 0x38, 0xc7, 0x26, 0x1c, 0xf3,
		0x71, 0x16, 0x15, 0x27, 0x6a, 0xc2, 0x2f, 0x00, 0x00, 0x00, 0x00, 0x49,
		0x45, 0x4e, 0x44, 0xae, 0x42, 0x60, 0x82
	};

	static unsigned int tamsyn10x20_png_len = 270;
	static unsigned char tamsyn10x20_png[] = {
		0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a, 0x00, 0x00, 0x00, 0x0d,
		0x49, 0x48, 0x44, 0x52, 0x00, 0x00, 0x00, 0xa0, 0x00, 0x00, 0x00, 0x0a,
		0x04, 0x03, 0x00, 0x00, 0x00, 0xc1, 0x66, 0x48, 0x96, 0x00, 0x00, 0x00,
		0x0f, 0x50, 0x4c, 0x54, 0x45, 0x00, 0x00, 0x00, 0xf9, 0x07, 0x00, 0x5d,
		0x71, 0xa5, 0x00, 0x00, 0x00, 0xff, 0xff, 0xff, 0x49, 0xdb, 0xcb, 0x7f,
		0x00, 0x00, 0x00, 0x01, 0x74, 0x52, 0x4e, 0x53, 0x00, 0x40, 0xe6, 0xd8,
		0x66, 0x00, 0x00, 0x00, 0xad, 0x49, 0x44, 0x41, 0x54, 0x28, 0xcf, 0xa5,
		0x92, 0x4b, 0x0e, 0x03, 0x31, 0x08, 0x43, 0xdf, 0x82, 0x83, 0x79, 0xe1,
		0xfb, 0x9f, 0xa9, 0x0b, 0x3e, 0x61, 0xa6, 0x1f, 0x55, 0xad, 0x14, 0x31,
		0x66, 0x42, 0x1c, 0x70, 0x0c, 0xb6, 0x00, 0x01, 0xb6, 0x08, 0xdb, 0x00,
		0x8d, 0xc2, 0x14, 0xb2, 0x55, 0xa1, 0xfe, 0x09, 0xc2, 0x26, 0xdc, 0x25,
		0x75, 0x22, 0x97, 0x1a, 0x25, 0x77, 0x28, 0x31, 0x02, 0x80, 0xc8, 0xdd,
		0x2c, 0x11, 0x1a, 0x54, 0x9f, 0xc8, 0xa2, 0x8a, 0x06, 0xa9, 0x93, 0x22,
		0xbd, 0xd4, 0xd0, 0x0c, 0xcf, 0x81, 0x2b, 0xca, 0xbb, 0x83, 0xe0, 0x10,
		0xe6, 0xad, 0xff, 0x10, 0x2a, 0x66, 0x34, 0x41, 0x58, 0x35, 0x54, 0x49,
		0x5a, 0x63, 0xa5, 0xc2, 0x87, 0xab, 0x52, 0x76, 0x9a, 0xba, 0xc6, 0xf4,
		0x75, 0x7a, 0x9e, 0x3c, 0x46, 0x86, 0x5c, 0xa3, 0xfd, 0x87, 0x0e, 0x75,
		0x08, 0x7b, 0xee, 0x7e, 0xea, 0x21, 0x5c, 0x4f, 0xf6, 0xc5, 0xc8, 0x4b,
		0xb9, 0x11, 0xf2, 0xd6, 0xe1, 0x8f, 0x84, 0x62, 0x7b, 0x67, 0xf9, 0x24,
		0xde, 0x6d, 0xbc, 0xb2, 0xcd, 0xb1, 0xf3, 0xf2, 0x2f, 0xe8, 0xe2, 0xe4,
		0xae, 0x4b, 0x4f, 0xcf, 0x2b, 0xdc, 0x8d, 0x0d, 0xf0, 0x00, 0x8f, 0x22,
		0x26, 0x65, 0x75, 0x8a, 0xe6, 0x84, 0x00, 0x00, 0x00, 0x00, 0x49, 0x45,
		0x4e, 0x44, 0xae, 0x42, 0x60, 0x82
	};

	if (RenderingServer::get_singleton() != nullptr) {
		Vector<uint8_t> hex_box_data;

		Ref<Image> image;
		image.instantiate();

		Ref<ImageTexture> hex_code_image_tex[2];

		hex_box_data.resize(tamsyn5x9_png_len);
		memcpy(hex_box_data.ptrw(), tamsyn5x9_png, tamsyn5x9_png_len);
		image->load_png_from_buffer(hex_box_data);
		hex_code_image_tex[0].instantiate();
		hex_code_image_tex[0]->create_from_image(image);
		hex_code_box_font_tex[0].instantiate();
		hex_code_box_font_tex[0]->set_diffuse_texture(hex_code_image_tex[0]);
		hex_code_box_font_tex[0]->set_texture_filter(CanvasItem::TEXTURE_FILTER_NEAREST);
		hex_box_data.clear();

		hex_box_data.resize(tamsyn10x20_png_len);
		memcpy(hex_box_data.ptrw(), tamsyn10x20_png, tamsyn10x20_png_len);
		image->load_png_from_buffer(hex_box_data);
		hex_code_image_tex[1].instantiate();
		hex_code_image_tex[1]->create_from_image(image);
		hex_code_box_font_tex[1].instantiate();
		hex_code_box_font_tex[1]->set_diffuse_texture(hex_code_image_tex[1]);
		hex_code_box_font_tex[1]->set_texture_filter(CanvasItem::TEXTURE_FILTER_NEAREST);
		hex_box_data.clear();
	}
}

void TextServer::finish_hex_code_box_fonts() {
	if (hex_code_box_font_tex[0].is_valid()) {
		hex_code_box_font_tex[0].unref();
	}
	if (hex_code_box_font_tex[1].is_valid()) {
		hex_code_box_font_tex[1].unref();
	}
}

Vector2 TextServer::get_hex_code_box_size(int p_size, char32_t p_index) const {
	int fnt = (p_size < 20) ? 0 : 1;

	real_t w = ((p_index <= 0xFF) ? 1 : ((p_index <= 0xFFFF) ? 2 : 3)) * hex_code_box_font_size[fnt].x;
	real_t h = 2 * hex_code_box_font_size[fnt].y;
	return Vector2(w + 4, h + 3 + 2 * hex_code_box_font_size[fnt].z);
}

void TextServer::draw_hex_code_box(RID p_canvas, int p_size, const Vector2 &p_pos, char32_t p_index, const Color &p_color) const {
	int fnt = (p_size < 20) ? 0 : 1;

	ERR_FAIL_COND(hex_code_box_font_tex[fnt].is_null());

	uint8_t a = p_index & 0x0F;
	uint8_t b = (p_index >> 4) & 0x0F;
	uint8_t c = (p_index >> 8) & 0x0F;
	uint8_t d = (p_index >> 12) & 0x0F;
	uint8_t e = (p_index >> 16) & 0x0F;
	uint8_t f = (p_index >> 20) & 0x0F;

	Vector2 pos = p_pos;
	Rect2 dest = Rect2(Vector2(), Vector2(hex_code_box_font_size[fnt].x, hex_code_box_font_size[fnt].y));

	real_t w = ((p_index <= 0xFF) ? 1 : ((p_index <= 0xFFFF) ? 2 : 3)) * hex_code_box_font_size[fnt].x;
	real_t h = 2 * hex_code_box_font_size[fnt].y;

	pos.y -= Math::floor((h + 3 + hex_code_box_font_size[fnt].z) * 0.75);

	RenderingServer::get_singleton()->canvas_item_add_rect(p_canvas, Rect2(pos + Point2(0, 0), Size2(1, h + 2 + 2 * hex_code_box_font_size[fnt].z)), p_color);
	RenderingServer::get_singleton()->canvas_item_add_rect(p_canvas, Rect2(pos + Point2(w + 2, 0), Size2(1, h + 2 + 2 * hex_code_box_font_size[fnt].z)), p_color);
	RenderingServer::get_singleton()->canvas_item_add_rect(p_canvas, Rect2(pos + Point2(0, 0), Size2(w + 2, 1)), p_color);
	RenderingServer::get_singleton()->canvas_item_add_rect(p_canvas, Rect2(pos + Point2(0, h + 2 + 2 * hex_code_box_font_size[fnt].z), Size2(w + 2, 1)), p_color);

	pos += Point2(2, 2);
	if (p_index <= 0xFF) {
		dest.position = pos + Vector2(hex_code_box_font_size[fnt].x, hex_code_box_font_size[fnt].y) * Point2(0, 0);
		RenderingServer::get_singleton()->canvas_item_add_texture_rect_region(p_canvas, dest, hex_code_box_font_tex[fnt]->get_rid(), Rect2(Point2(b * hex_code_box_font_size[fnt].x, 0), dest.size), p_color, false, false);
		dest.position = pos + Vector2(hex_code_box_font_size[fnt].x, hex_code_box_font_size[fnt].y) * Point2(0, 1) + Point2(0, hex_code_box_font_size[fnt].z);
		RenderingServer::get_singleton()->canvas_item_add_texture_rect_region(p_canvas, dest, hex_code_box_font_tex[fnt]->get_rid(), Rect2(Point2(a * hex_code_box_font_size[fnt].x, 0), dest.size), p_color, false, false);
	} else if (p_index <= 0xFFFF) {
		dest.position = pos + Vector2(hex_code_box_font_size[fnt].x, hex_code_box_font_size[fnt].y) * Point2(0, 0);
		RenderingServer::get_singleton()->canvas_item_add_texture_rect_region(p_canvas, dest, hex_code_box_font_tex[fnt]->get_rid(), Rect2(Point2(d * hex_code_box_font_size[fnt].x, 0), dest.size), p_color, false, false);
		dest.position = pos + Vector2(hex_code_box_font_size[fnt].x, hex_code_box_font_size[fnt].y) * Point2(1, 0);
		RenderingServer::get_singleton()->canvas_item_add_texture_rect_region(p_canvas, dest, hex_code_box_font_tex[fnt]->get_rid(), Rect2(Point2(c * hex_code_box_font_size[fnt].x, 0), dest.size), p_color, false, false);
		dest.position = pos + Vector2(hex_code_box_font_size[fnt].x, hex_code_box_font_size[fnt].y) * Point2(0, 1) + Point2(0, hex_code_box_font_size[fnt].z);
		RenderingServer::get_singleton()->canvas_item_add_texture_rect_region(p_canvas, dest, hex_code_box_font_tex[fnt]->get_rid(), Rect2(Point2(b * hex_code_box_font_size[fnt].x, 0), dest.size), p_color, false, false);
		dest.position = pos + Vector2(hex_code_box_font_size[fnt].x, hex_code_box_font_size[fnt].y) * Point2(1, 1) + Point2(0, hex_code_box_font_size[fnt].z);
		RenderingServer::get_singleton()->canvas_item_add_texture_rect_region(p_canvas, dest, hex_code_box_font_tex[fnt]->get_rid(), Rect2(Point2(a * hex_code_box_font_size[fnt].x, 0), dest.size), p_color, false, false);
	} else {
		dest.position = pos + Vector2(hex_code_box_font_size[fnt].x, hex_code_box_font_size[fnt].y) * Point2(0, 0);
		RenderingServer::get_singleton()->canvas_item_add_texture_rect_region(p_canvas, dest, hex_code_box_font_tex[fnt]->get_rid(), Rect2(Point2(f * hex_code_box_font_size[fnt].x, 0), dest.size), p_color, false, false);
		dest.position = pos + Vector2(hex_code_box_font_size[fnt].x, hex_code_box_font_size[fnt].y) * Point2(1, 0);
		RenderingServer::get_singleton()->canvas_item_add_texture_rect_region(p_canvas, dest, hex_code_box_font_tex[fnt]->get_rid(), Rect2(Point2(e * hex_code_box_font_size[fnt].x, 0), dest.size), p_color, false, false);
		dest.position = pos + Vector2(hex_code_box_font_size[fnt].x, hex_code_box_font_size[fnt].y) * Point2(2, 0);
		RenderingServer::get_singleton()->canvas_item_add_texture_rect_region(p_canvas, dest, hex_code_box_font_tex[fnt]->get_rid(), Rect2(Point2(d * hex_code_box_font_size[fnt].x, 0), dest.size), p_color, false, false);
		dest.position = pos + Vector2(hex_code_box_font_size[fnt].x, hex_code_box_font_size[fnt].y) * Point2(0, 1) + Point2(0, hex_code_box_font_size[fnt].z);
		RenderingServer::get_singleton()->canvas_item_add_texture_rect_region(p_canvas, dest, hex_code_box_font_tex[fnt]->get_rid(), Rect2(Point2(c * hex_code_box_font_size[fnt].x, 0), dest.size), p_color, false, false);
		dest.position = pos + Vector2(hex_code_box_font_size[fnt].x, hex_code_box_font_size[fnt].y) * Point2(1, 1) + Point2(0, hex_code_box_font_size[fnt].z);
		RenderingServer::get_singleton()->canvas_item_add_texture_rect_region(p_canvas, dest, hex_code_box_font_tex[fnt]->get_rid(), Rect2(Point2(b * hex_code_box_font_size[fnt].x, 0), dest.size), p_color, false, false);
		dest.position = pos + Vector2(hex_code_box_font_size[fnt].x, hex_code_box_font_size[fnt].y) * Point2(2, 1) + Point2(0, hex_code_box_font_size[fnt].z);
		RenderingServer::get_singleton()->canvas_item_add_texture_rect_region(p_canvas, dest, hex_code_box_font_tex[fnt]->get_rid(), Rect2(Point2(a * hex_code_box_font_size[fnt].x, 0), dest.size), p_color, false, false);
	}
}

Vector<Vector2i> TextServer::shaped_text_get_line_breaks_adv(RID p_shaped, const Vector<real_t> &p_width, int p_start, bool p_once, uint8_t /*TextBreakFlag*/ p_break_flags) const {
	Vector<Vector2i> lines;

	ERR_FAIL_COND_V(p_width.is_empty(), lines);

	const_cast<TextServer *>(this)->shaped_text_update_breaks(p_shaped);
	const Vector<Glyph> &logical = const_cast<TextServer *>(this)->shaped_text_sort_logical(p_shaped);
	const Vector2i &range = shaped_text_get_range(p_shaped);

	real_t width = 0.f;
	int line_start = MAX(p_start, range.x);
	int last_safe_break = -1;
	int chunk = 0;

	int l_size = logical.size();
	const Glyph *l_gl = logical.ptr();

	for (int i = 0; i < l_size; i++) {
		if (l_gl[i].start < p_start) {
			continue;
		}
		if (l_gl[i].count > 0) {
			if ((p_width[chunk] > 0) && (width + l_gl[i].advance > p_width[chunk]) && (last_safe_break >= 0)) {
				lines.push_back(Vector2i(line_start, l_gl[last_safe_break].end));
				line_start = l_gl[last_safe_break].end;
				i = last_safe_break;
				last_safe_break = -1;
				width = 0;
				chunk++;
				if (chunk >= p_width.size()) {
					chunk = 0;
					if (p_once) {
						return lines;
					}
				}
				continue;
			}
			if ((p_break_flags & BREAK_MANDATORY) == BREAK_MANDATORY) {
				if ((l_gl[i].flags & GRAPHEME_IS_BREAK_HARD) == GRAPHEME_IS_BREAK_HARD) {
					lines.push_back(Vector2i(line_start, l_gl[i].end));
					line_start = l_gl[i].end;
					last_safe_break = -1;
					width = 0;
					chunk = 0;
					if (p_once) {
						return lines;
					}
					continue;
				}
			}
			if ((p_break_flags & BREAK_WORD_BOUND) == BREAK_WORD_BOUND) {
				if ((l_gl[i].flags & GRAPHEME_IS_BREAK_SOFT) == GRAPHEME_IS_BREAK_SOFT) {
					last_safe_break = i;
				}
			}
			if ((p_break_flags & BREAK_GRAPHEME_BOUND) == BREAK_GRAPHEME_BOUND) {
				last_safe_break = i;
			}
		}
		width += l_gl[i].advance;
	}

	if (l_size > 0) {
		lines.push_back(Vector2i(line_start, range.y));
	} else {
		lines.push_back(Vector2i(0, 0));
	}

	return lines;
}

Vector<Vector2i> TextServer::shaped_text_get_line_breaks(RID p_shaped, real_t p_width, int p_start, uint8_t /*TextBreakFlag*/ p_break_flags) const {
	Vector<Vector2i> lines;

	const_cast<TextServer *>(this)->shaped_text_update_breaks(p_shaped);
	const Vector<Glyph> &logical = const_cast<TextServer *>(this)->shaped_text_sort_logical(p_shaped);
	const Vector2i &range = shaped_text_get_range(p_shaped);

	real_t width = 0.f;
	int line_start = MAX(p_start, range.x);
	int last_safe_break = -1;
	int word_count = 0;
	int l_size = logical.size();
	const Glyph *l_gl = logical.ptr();

	for (int i = 0; i < l_size; i++) {
		if (l_gl[i].start < p_start) {
			continue;
		}
		if (l_gl[i].count > 0) {
			if ((p_width > 0) && (width + l_gl[i].advance * l_gl[i].repeat > p_width) && (last_safe_break >= 0)) {
				lines.push_back(Vector2i(line_start, l_gl[last_safe_break].end));
				line_start = l_gl[last_safe_break].end;
				i = last_safe_break;
				last_safe_break = -1;
				width = 0;
				word_count = 0;
				continue;
			}
			if ((p_break_flags & BREAK_MANDATORY) == BREAK_MANDATORY) {
				if ((l_gl[i].flags & GRAPHEME_IS_BREAK_HARD) == GRAPHEME_IS_BREAK_HARD) {
					lines.push_back(Vector2i(line_start, l_gl[i].end));
					line_start = l_gl[i].end;
					last_safe_break = -1;
					width = 0;
					continue;
				}
			}
			if ((p_break_flags & BREAK_WORD_BOUND) == BREAK_WORD_BOUND) {
				if ((l_gl[i].flags & GRAPHEME_IS_BREAK_SOFT) == GRAPHEME_IS_BREAK_SOFT) {
					last_safe_break = i;
					word_count++;
				}
			}
			if (((p_break_flags & BREAK_WORD_BOUND_ADAPTIVE) == BREAK_WORD_BOUND_ADAPTIVE) && word_count == 0) {
				last_safe_break = i;
			}
			if ((p_break_flags & BREAK_GRAPHEME_BOUND) == BREAK_GRAPHEME_BOUND) {
				last_safe_break = i;
			}
		}
		width += l_gl[i].advance * l_gl[i].repeat;
	}

	if (l_size > 0) {
		if (lines.size() == 0 || lines[lines.size() - 1].y < range.y) {
			lines.push_back(Vector2i(line_start, range.y));
		}
	} else {
		lines.push_back(Vector2i(0, 0));
	}

	return lines;
}

Vector<Vector2i> TextServer::shaped_text_get_word_breaks(RID p_shaped, int p_grapheme_flags) const {
	Vector<Vector2i> words;

	const_cast<TextServer *>(this)->shaped_text_update_justification_ops(p_shaped);
	const Vector<Glyph> &logical = const_cast<TextServer *>(this)->shaped_text_sort_logical(p_shaped);
	const Vector2i &range = shaped_text_get_range(p_shaped);

	int word_start = range.x;

	int l_size = logical.size();
	const Glyph *l_gl = logical.ptr();

	for (int i = 0; i < l_size; i++) {
		if (l_gl[i].count > 0) {
			if ((l_gl[i].flags & p_grapheme_flags) != 0) {
				words.push_back(Vector2i(word_start, l_gl[i].start));
				word_start = l_gl[i].end;
			}
		}
	}
	if (l_size > 0) {
		words.push_back(Vector2i(word_start, range.y));
	}

	return words;
}

TextServer::TrimData TextServer::shaped_text_get_trim_data(RID p_shaped) const {
	WARN_PRINT("Getting overrun data not supported by this TextServer.");
	return TrimData();
}

void TextServer::shaped_text_get_carets(RID p_shaped, int p_position, Rect2 &p_leading_caret, Direction &p_leading_dir, Rect2 &p_trailing_caret, Direction &p_trailing_dir) const {
	Vector<Rect2> carets;
	const Vector<TextServer::Glyph> visual = shaped_text_get_glyphs(p_shaped);
	TextServer::Orientation orientation = shaped_text_get_orientation(p_shaped);
	const Vector2 &range = shaped_text_get_range(p_shaped);
	real_t ascent = shaped_text_get_ascent(p_shaped);
	real_t descent = shaped_text_get_descent(p_shaped);
	real_t height = (ascent + descent) / 2;

	real_t off = 0.0f;
	p_leading_dir = DIRECTION_AUTO;
	p_trailing_dir = DIRECTION_AUTO;

	int v_size = visual.size();
	const Glyph *glyphs = visual.ptr();

	for (int i = 0; i < v_size; i++) {
		if (glyphs[i].count > 0) {
			// Caret before grapheme (top / left).
			if (p_position == glyphs[i].start && ((glyphs[i].flags & GRAPHEME_IS_VIRTUAL) != GRAPHEME_IS_VIRTUAL)) {
				Rect2 cr;
				if (orientation == ORIENTATION_HORIZONTAL) {
					if (glyphs[i].start == range.x) {
						cr.size.y = height * 2;
					} else {
						cr.size.y = height;
					}
					cr.position.y = -ascent;
					cr.position.x = off;
					if ((glyphs[i].flags & GRAPHEME_IS_RTL) == GRAPHEME_IS_RTL) {
						p_trailing_dir = DIRECTION_RTL;
						for (int j = 0; j < glyphs[i].count; j++) {
							cr.position.x += glyphs[i + j].advance * glyphs[i + j].repeat;
							cr.size.x -= glyphs[i + j].advance * glyphs[i + j].repeat;
						}
					} else {
						p_trailing_dir = DIRECTION_LTR;
						for (int j = 0; j < glyphs[i].count; j++) {
							cr.size.x += glyphs[i + j].advance * glyphs[i + j].repeat;
						}
					}
				} else {
					if (glyphs[i].start == range.x) {
						cr.size.x = height * 2;
					} else {
						cr.size.x = height;
					}
					cr.position.x = -ascent;
					cr.position.y = off;
					if ((glyphs[i].flags & GRAPHEME_IS_RTL) == GRAPHEME_IS_RTL) {
						p_trailing_dir = DIRECTION_RTL;
						for (int j = 0; j < glyphs[i].count; j++) {
							cr.position.y += glyphs[i + j].advance * glyphs[i + j].repeat;
							cr.size.y -= glyphs[i + j].advance * glyphs[i + j].repeat;
						}
					} else {
						p_trailing_dir = DIRECTION_LTR;
						for (int j = 0; j < glyphs[i].count; j++) {
							cr.size.y += glyphs[i + j].advance * glyphs[i + j].repeat;
						}
					}
				}
				p_trailing_caret = cr;
			}
			// Caret after grapheme (bottom / right).
			if (p_position == glyphs[i].end && ((glyphs[i].flags & GRAPHEME_IS_VIRTUAL) != GRAPHEME_IS_VIRTUAL)) {
				Rect2 cr;
				if (orientation == ORIENTATION_HORIZONTAL) {
					if (glyphs[i].end == range.y) {
						cr.size.y = height * 2;
						cr.position.y = -ascent;
					} else {
						cr.size.y = height;
						cr.position.y = -ascent + height;
					}
					cr.position.x = off;
					if ((glyphs[i].flags & GRAPHEME_IS_RTL) != GRAPHEME_IS_RTL) {
						p_leading_dir = DIRECTION_LTR;
						for (int j = 0; j < glyphs[i].count; j++) {
							cr.position.x += glyphs[i + j].advance * glyphs[i + j].repeat;
							cr.size.x -= glyphs[i + j].advance * glyphs[i + j].repeat;
						}
					} else {
						p_leading_dir = DIRECTION_RTL;
						for (int j = 0; j < glyphs[i].count; j++) {
							cr.size.x += glyphs[i + j].advance * glyphs[i + j].repeat;
						}
					}
				} else {
					cr.size.y = 1.0f;
					if (glyphs[i].end == range.y) {
						cr.size.x = height * 2;
						cr.position.x = -ascent;
					} else {
						cr.size.x = height;
						cr.position.x = -ascent + height;
					}
					cr.position.y = off;
					if ((glyphs[i].flags & GRAPHEME_IS_RTL) != GRAPHEME_IS_RTL) {
						p_leading_dir = DIRECTION_LTR;
						for (int j = 0; j < glyphs[i].count; j++) {
							cr.position.y += glyphs[i + j].advance * glyphs[i + j].repeat;
							cr.size.y -= glyphs[i + j].advance * glyphs[i + j].repeat;
						}
					} else {
						p_leading_dir = DIRECTION_RTL;
						for (int j = 0; j < glyphs[i].count; j++) {
							cr.size.y += glyphs[i + j].advance * glyphs[i + j].repeat;
						}
					}
				}
				p_leading_caret = cr;
			}
			// Caret inside grapheme (middle).
			if (p_position > glyphs[i].start && p_position < glyphs[i].end && (glyphs[i].flags & GRAPHEME_IS_VIRTUAL) != GRAPHEME_IS_VIRTUAL) {
				real_t advance = 0.f;
				for (int j = 0; j < glyphs[i].count; j++) {
					advance += glyphs[i + j].advance * glyphs[i + j].repeat;
				}
				real_t char_adv = advance / (real_t)(glyphs[i].end - glyphs[i].start);
				Rect2 cr;
				if (orientation == ORIENTATION_HORIZONTAL) {
					cr.size.x = 1.0f;
					cr.size.y = height * 2;
					cr.position.y = -ascent;
					if ((glyphs[i].flags & GRAPHEME_IS_RTL) == GRAPHEME_IS_RTL) {
						cr.position.x = off + char_adv * (glyphs[i].end - p_position);
					} else {
						cr.position.x = off + char_adv * (p_position - glyphs[i].start);
					}
				} else {
					cr.size.y = 1.0f;
					cr.size.x = height * 2;
					cr.position.x = -ascent;
					if ((glyphs[i].flags & GRAPHEME_IS_RTL) == GRAPHEME_IS_RTL) {
						cr.position.y = off + char_adv * (glyphs[i].end - p_position);
					} else {
						cr.position.y = off + char_adv * (p_position - glyphs[i].start);
					}
				}
				p_trailing_caret = cr;
				p_leading_caret = cr;
			}
		}
		off += glyphs[i].advance * glyphs[i].repeat;
	}
}

TextServer::Direction TextServer::shaped_text_get_dominant_direciton_in_range(RID p_shaped, int p_start, int p_end) const {
	const Vector<TextServer::Glyph> visual = shaped_text_get_glyphs(p_shaped);

	if (p_start == p_end) {
		return DIRECTION_AUTO;
	}

	int start = MIN(p_start, p_end);
	int end = MAX(p_start, p_end);

	int rtl = 0;
	int ltr = 0;

	int v_size = visual.size();
	const Glyph *glyphs = visual.ptr();

	for (int i = 0; i < v_size; i++) {
		if ((glyphs[i].end > start) && (glyphs[i].start < end)) {
			if (glyphs[i].count > 0) {
				if ((glyphs[i].flags & GRAPHEME_IS_RTL) == GRAPHEME_IS_RTL) {
					rtl++;
				} else {
					ltr++;
				}
			}
		}
	}
	if (ltr == rtl) {
		return DIRECTION_AUTO;
	} else if (ltr > rtl) {
		return DIRECTION_LTR;
	} else {
		return DIRECTION_RTL;
	}
}

Vector<Vector2> TextServer::shaped_text_get_selection(RID p_shaped, int p_start, int p_end) const {
	Vector<Vector2> ranges;
	const Vector<TextServer::Glyph> visual = shaped_text_get_glyphs(p_shaped);

	if (p_start == p_end) {
		return ranges;
	}

	int start = MIN(p_start, p_end);
	int end = MAX(p_start, p_end);

	int v_size = visual.size();
	const Glyph *glyphs = visual.ptr();

	real_t off = 0.0f;
	for (int i = 0; i < v_size; i++) {
		for (int k = 0; k < glyphs[i].repeat; k++) {
			if ((glyphs[i].count > 0) && ((glyphs[i].index != 0) || ((glyphs[i].flags & GRAPHEME_IS_SPACE) == GRAPHEME_IS_SPACE))) {
				if (glyphs[i].start < end && glyphs[i].end > start) {
					// Grapheme fully in selection range.
					if (glyphs[i].start >= start && glyphs[i].end <= end) {
						real_t advance = 0.f;
						for (int j = 0; j < glyphs[i].count; j++) {
							advance += glyphs[i + j].advance;
						}
						ranges.push_back(Vector2(off, off + advance));
					}
					// Only start of grapheme is in selection range.
					if (glyphs[i].start >= start && glyphs[i].end > end) {
						real_t advance = 0.f;
						for (int j = 0; j < glyphs[i].count; j++) {
							advance += glyphs[i + j].advance;
						}
						real_t char_adv = advance / (real_t)(glyphs[i].end - glyphs[i].start);
						if ((glyphs[i].flags & GRAPHEME_IS_RTL) == GRAPHEME_IS_RTL) {
							ranges.push_back(Vector2(off + char_adv * (glyphs[i].end - end), off + advance));
						} else {
							ranges.push_back(Vector2(off, off + char_adv * (end - glyphs[i].start)));
						}
					}
					// Only end of grapheme is in selection range.
					if (glyphs[i].start < start && glyphs[i].end <= end) {
						real_t advance = 0.f;
						for (int j = 0; j < glyphs[i].count; j++) {
							advance += glyphs[i + j].advance;
						}
						real_t char_adv = advance / (real_t)(glyphs[i].end - glyphs[i].start);
						if ((glyphs[i].flags & GRAPHEME_IS_RTL) == GRAPHEME_IS_RTL) {
							ranges.push_back(Vector2(off, off + char_adv * (start - glyphs[i].start)));
						} else {
							ranges.push_back(Vector2(off + char_adv * (glyphs[i].end - start), off + advance));
						}
					}
					// Selection range is within grapheme.
					if (glyphs[i].start < start && glyphs[i].end > end) {
						real_t advance = 0.f;
						for (int j = 0; j < glyphs[i].count; j++) {
							advance += glyphs[i + j].advance;
						}
						real_t char_adv = advance / (real_t)(glyphs[i].end - glyphs[i].start);
						if ((glyphs[i].flags & GRAPHEME_IS_RTL) == GRAPHEME_IS_RTL) {
							ranges.push_back(Vector2(off + char_adv * (glyphs[i].end - end), off + char_adv * (glyphs[i].end - start)));
						} else {
							ranges.push_back(Vector2(off + char_adv * (start - glyphs[i].start), off + char_adv * (end - glyphs[i].start)));
						}
					}
				}
			}
			off += glyphs[i].advance;
		}
	}

	// Merge intersecting ranges.
	int i = 0;
	while (i < ranges.size()) {
		i++;
	}
	i = 0;
	while (i < ranges.size()) {
		int j = i + 1;
		while (j < ranges.size()) {
			if (Math::is_equal_approx(ranges[i].y, ranges[j].x, (real_t)UNIT_EPSILON)) {
				ranges.write[i].y = ranges[j].y;
				ranges.remove(j);
				continue;
			}
			j++;
		}
		i++;
	}

	return ranges;
}

int TextServer::shaped_text_hit_test_grapheme(RID p_shaped, real_t p_coords) const {
	const Vector<TextServer::Glyph> visual = shaped_text_get_glyphs(p_shaped);

	// Exact grapheme hit test, return -1 if missed.
	real_t off = 0.0f;

	int v_size = visual.size();
	const Glyph *glyphs = visual.ptr();

	for (int i = 0; i < v_size; i++) {
		for (int j = 0; j < glyphs[i].repeat; j++) {
			if (p_coords >= off && p_coords < off + glyphs[i].advance) {
				return i;
			}
			off += glyphs[i].advance;
		}
	}
	return -1;
}

int TextServer::shaped_text_hit_test_position(RID p_shaped, real_t p_coords) const {
	const Vector<TextServer::Glyph> visual = shaped_text_get_glyphs(p_shaped);

	int v_size = visual.size();
	const Glyph *glyphs = visual.ptr();

	// Cursor placement hit test.

	// Place caret to the left of the leftmost grapheme, or to position 0 if string is empty.
	if (p_coords <= 0) {
		if (v_size > 0) {
			if ((glyphs[0].flags & GRAPHEME_IS_RTL) == GRAPHEME_IS_RTL) {
				return glyphs[0].end;
			} else {
				return glyphs[0].start;
			}
		} else {
			return 0;
		}
	}

	// Place caret to the right of the rightmost grapheme, or to position 0 if string is empty.
	if (p_coords >= shaped_text_get_width(p_shaped)) {
		if (v_size > 0) {
			if ((glyphs[v_size - 1].flags & GRAPHEME_IS_RTL) == GRAPHEME_IS_RTL) {
				return glyphs[v_size - 1].start;
			} else {
				return glyphs[v_size - 1].end;
			}
		} else {
			return 0;
		}
	}

	real_t off = 0.0f;
	for (int i = 0; i < v_size; i++) {
		if (glyphs[i].count > 0) {
			real_t advance = 0.f;
			for (int j = 0; j < glyphs[i].count; j++) {
				advance += glyphs[i + j].advance * glyphs[i + j].repeat;
			}
			if (((glyphs[i].flags & GRAPHEME_IS_VIRTUAL) == GRAPHEME_IS_VIRTUAL) && (p_coords >= off && p_coords < off + advance)) {
				if ((glyphs[i].flags & GRAPHEME_IS_RTL) == GRAPHEME_IS_RTL) {
					return glyphs[i].end;
				} else {
					return glyphs[i].start;
				}
			}
			// Place caret to the left of clicked grapheme.
			if (p_coords >= off && p_coords < off + advance / 2) {
				if ((glyphs[i].flags & GRAPHEME_IS_RTL) == GRAPHEME_IS_RTL) {
					return glyphs[i].end;
				} else {
					return glyphs[i].start;
				}
			}
			// Place caret to the right of clicked grapheme.
			if (p_coords >= off + advance / 2 && p_coords < off + advance) {
				if ((glyphs[i].flags & GRAPHEME_IS_RTL) == GRAPHEME_IS_RTL) {
					return glyphs[i].start;
				} else {
					return glyphs[i].end;
				}
			}
		}
		off += glyphs[i].advance * glyphs[i].repeat;
	}
	return 0;
}

int TextServer::shaped_text_next_grapheme_pos(RID p_shaped, int p_pos) {
	const Vector<TextServer::Glyph> visual = shaped_text_get_glyphs(p_shaped);
	int v_size = visual.size();
	const Glyph *glyphs = visual.ptr();
	for (int i = 0; i < v_size; i++) {
		if (p_pos >= glyphs[i].start && p_pos < glyphs[i].end) {
			return glyphs[i].end;
		}
	}
	return p_pos;
}

int TextServer::shaped_text_prev_grapheme_pos(RID p_shaped, int p_pos) {
	const Vector<TextServer::Glyph> visual = shaped_text_get_glyphs(p_shaped);
	int v_size = visual.size();
	const Glyph *glyphs = visual.ptr();
	for (int i = 0; i < v_size; i++) {
		if (p_pos > glyphs[i].start && p_pos <= glyphs[i].end) {
			return glyphs[i].start;
		}
	}

	return p_pos;
}

void TextServer::shaped_text_draw(RID p_shaped, RID p_canvas, const Vector2 &p_pos, real_t p_clip_l, real_t p_clip_r, const Color &p_color) const {
	const Vector<TextServer::Glyph> visual = shaped_text_get_glyphs(p_shaped);
	TextServer::Orientation orientation = shaped_text_get_orientation(p_shaped);
	bool hex_codes = shaped_text_get_preserve_control(p_shaped) || shaped_text_get_preserve_invalid(p_shaped);

	bool rtl = shaped_text_get_direction(p_shaped) == DIRECTION_RTL;
	TrimData trim_data = shaped_text_get_trim_data(p_shaped);

	int v_size = visual.size();
	const Glyph *glyphs = visual.ptr();

	Vector2 ofs = p_pos;
	// Draw RTL ellipsis string when needed.
	if (rtl && trim_data.ellipsis_pos >= 0) {
		for (int i = trim_data.ellipsis_glyph_buf.size() - 1; i >= 0; i--) {
			for (int j = 0; j < trim_data.ellipsis_glyph_buf[i].repeat; j++) {
				font_draw_glyph(trim_data.ellipsis_glyph_buf[i].font_rid, p_canvas, trim_data.ellipsis_glyph_buf[i].font_size, ofs + Vector2(trim_data.ellipsis_glyph_buf[i].x_off, trim_data.ellipsis_glyph_buf[i].y_off), trim_data.ellipsis_glyph_buf[i].index, p_color);
				if (orientation == ORIENTATION_HORIZONTAL) {
					ofs.x += trim_data.ellipsis_glyph_buf[i].advance;
				} else {
					ofs.y += trim_data.ellipsis_glyph_buf[i].advance;
				}
			}
		}
	}
	// Draw at the baseline.
	for (int i = 0; i < v_size; i++) {
		for (int j = 0; j < glyphs[i].repeat; j++) {
			if (p_clip_r > 0) {
				// Clip right / bottom.
				if (orientation == ORIENTATION_HORIZONTAL) {
					if (ofs.x - p_pos.x > p_clip_r) {
						return;
					}
				} else {
					if (ofs.y - p_pos.y > p_clip_r) {
						return;
					}
				}
			}
			if (p_clip_l > 0) {
				// Clip left / top.
				if (orientation == ORIENTATION_HORIZONTAL) {
					if (ofs.x - p_pos.x < p_clip_l) {
						ofs.x += glyphs[i].advance;
						continue;
					}
				} else {
					if (ofs.y - p_pos.y < p_clip_l) {
						ofs.y += glyphs[i].advance;
						continue;
					}
				}
			}
			if (trim_data.trim_pos >= 0) {
				if (rtl) {
					if (i < trim_data.trim_pos && (glyphs[j].flags & TextServer::GRAPHEME_IS_VIRTUAL) != TextServer::GRAPHEME_IS_VIRTUAL) {
						continue;
					}
				} else {
					if (i >= trim_data.trim_pos && (glyphs[j].flags & TextServer::GRAPHEME_IS_VIRTUAL) != TextServer::GRAPHEME_IS_VIRTUAL) {
						break;
					}
				}
			}

			if (glyphs[i].font_rid != RID()) {
				font_draw_glyph(glyphs[i].font_rid, p_canvas, glyphs[i].font_size, ofs + Vector2(glyphs[i].x_off, glyphs[i].y_off), glyphs[i].index, p_color);
			} else if (hex_codes && ((glyphs[i].flags & GRAPHEME_IS_VIRTUAL) != GRAPHEME_IS_VIRTUAL)) {
				TextServer::draw_hex_code_box(p_canvas, glyphs[i].font_size, ofs + Vector2(glyphs[i].x_off, glyphs[i].y_off), glyphs[i].index, p_color);
			}
			if (orientation == ORIENTATION_HORIZONTAL) {
				ofs.x += glyphs[i].advance;
			} else {
				ofs.y += glyphs[i].advance;
			}
		}
	}
	// Draw LTR ellipsis string when needed.
	if (!rtl && trim_data.ellipsis_pos >= 0) {
		for (int i = 0; i < trim_data.ellipsis_glyph_buf.size(); i++) {
			for (int j = 0; j < trim_data.ellipsis_glyph_buf[i].repeat; j++) {
				font_draw_glyph(trim_data.ellipsis_glyph_buf[i].font_rid, p_canvas, trim_data.ellipsis_glyph_buf[i].font_size, ofs + Vector2(trim_data.ellipsis_glyph_buf[i].x_off, trim_data.ellipsis_glyph_buf[i].y_off), trim_data.ellipsis_glyph_buf[i].index, p_color);
				if (orientation == ORIENTATION_HORIZONTAL) {
					ofs.x += trim_data.ellipsis_glyph_buf[i].advance;
				} else {
					ofs.y += trim_data.ellipsis_glyph_buf[i].advance;
				}
			}
		}
	}
}

void TextServer::shaped_text_draw_outline(RID p_shaped, RID p_canvas, const Vector2 &p_pos, real_t p_clip_l, real_t p_clip_r, int p_outline_size, const Color &p_color) const {
	const Vector<TextServer::Glyph> visual = shaped_text_get_glyphs(p_shaped);
	TextServer::Orientation orientation = shaped_text_get_orientation(p_shaped);

	bool rtl = (shaped_text_get_direction(p_shaped) == DIRECTION_RTL);
	TrimData trim_data = shaped_text_get_trim_data(p_shaped);

	int v_size = visual.size();
	const Glyph *glyphs = visual.ptr();
	Vector2 ofs = p_pos;
	// Draw RTL ellipsis string when needed.
	if (rtl && trim_data.ellipsis_pos >= 0) {
		for (int i = trim_data.ellipsis_glyph_buf.size() - 1; i >= 0; i--) {
			for (int j = 0; j < trim_data.ellipsis_glyph_buf[i].repeat; j++) {
				font_draw_glyph(trim_data.ellipsis_glyph_buf[i].font_rid, p_canvas, trim_data.ellipsis_glyph_buf[i].font_size, ofs + Vector2(trim_data.ellipsis_glyph_buf[i].x_off, trim_data.ellipsis_glyph_buf[i].y_off), trim_data.ellipsis_glyph_buf[i].index, p_color);
				if (orientation == ORIENTATION_HORIZONTAL) {
					ofs.x += trim_data.ellipsis_glyph_buf[i].advance;
				} else {
					ofs.y += trim_data.ellipsis_glyph_buf[i].advance;
				}
			}
		}
	}
	// Draw at the baseline.
	for (int i = 0; i < v_size; i++) {
		for (int j = 0; j < glyphs[i].repeat; j++) {
			if (p_clip_r > 0) {
				// Clip right / bottom.
				if (orientation == ORIENTATION_HORIZONTAL) {
					if (ofs.x - p_pos.x > p_clip_r) {
						return;
					}
				} else {
					if (ofs.y - p_pos.y > p_clip_r) {
						return;
					}
				}
			}
			if (p_clip_l > 0) {
				// Clip left / top.
				if (orientation == ORIENTATION_HORIZONTAL) {
					if (ofs.x - p_pos.x < p_clip_l) {
						ofs.x += glyphs[i].advance;
						continue;
					}
				} else {
					if (ofs.y - p_pos.y < p_clip_l) {
						ofs.y += glyphs[i].advance;
						continue;
					}
				}
			}
			if (trim_data.trim_pos >= 0) {
				if (rtl) {
					if (i < trim_data.trim_pos) {
						continue;
					}
				} else {
					if (i >= trim_data.trim_pos && (glyphs[j].flags & TextServer::GRAPHEME_IS_VIRTUAL) != TextServer::GRAPHEME_IS_VIRTUAL) {
						break;
					}
				}
			}
			if (glyphs[i].font_rid != RID()) {
				font_draw_glyph_outline(glyphs[i].font_rid, p_canvas, glyphs[i].font_size, p_outline_size, ofs + Vector2(glyphs[i].x_off, glyphs[i].y_off), glyphs[i].index, p_color);
			}
			if (orientation == ORIENTATION_HORIZONTAL) {
				ofs.x += glyphs[i].advance;
			} else {
				ofs.y += glyphs[i].advance;
			}
		}
	}
	// Draw LTR ellipsis string when needed.
	if (!rtl && trim_data.ellipsis_pos >= 0) {
		for (int i = 0; i < trim_data.ellipsis_glyph_buf.size(); i++) {
			for (int j = 0; j < trim_data.ellipsis_glyph_buf[i].repeat; j++) {
				font_draw_glyph(trim_data.ellipsis_glyph_buf[i].font_rid, p_canvas, trim_data.ellipsis_glyph_buf[i].font_size, ofs + Vector2(trim_data.ellipsis_glyph_buf[i].x_off, trim_data.ellipsis_glyph_buf[i].y_off), trim_data.ellipsis_glyph_buf[i].index, p_color);
				if (orientation == ORIENTATION_HORIZONTAL) {
					ofs.x += trim_data.ellipsis_glyph_buf[i].advance;
				} else {
					ofs.y += trim_data.ellipsis_glyph_buf[i].advance;
				}
			}
		}
	}
}

Dictionary TextServer::_font_get_glyph_contours(RID p_font, int p_size, int32_t p_index) const {
	Vector<Vector3> points;
	Vector<int32_t> contours;
	bool orientation;
	bool ok = font_get_glyph_contours(p_font, p_size, p_index, points, contours, orientation);
	Dictionary out;

	if (ok) {
		out["points"] = points;
		out["contours"] = contours;
		out["orientation"] = orientation;
	}
	return out;
}

void TextServer::_shaped_text_set_bidi_override(RID p_shaped, const Array &p_override) {
	Vector<Vector2i> overrides;
	for (int i = 0; i < p_override.size(); i++) {
		overrides.push_back(p_override[i]);
	}
	shaped_text_set_bidi_override(p_shaped, overrides);
}

Array TextServer::_shaped_text_get_glyphs(RID p_shaped) const {
	Array ret;

	Vector<Glyph> glyphs = shaped_text_get_glyphs(p_shaped);
	for (int i = 0; i < glyphs.size(); i++) {
		Dictionary glyph;

		glyph["start"] = glyphs[i].start;
		glyph["end"] = glyphs[i].end;
		glyph["repeat"] = glyphs[i].repeat;
		glyph["count"] = glyphs[i].count;
		glyph["flags"] = glyphs[i].flags;
		glyph["offset"] = Vector2(glyphs[i].x_off, glyphs[i].y_off);
		glyph["advance"] = glyphs[i].advance;
		glyph["font_rid"] = glyphs[i].font_rid;
		glyph["font_size"] = glyphs[i].font_size;
		glyph["index"] = glyphs[i].index;

		ret.push_back(glyph);
	}

	return ret;
}

Array TextServer::_shaped_text_get_line_breaks_adv(RID p_shaped, const PackedFloat32Array &p_width, int p_start, bool p_once, uint8_t p_break_flags) const {
	Array ret;

	Vector<Vector2i> lines = shaped_text_get_line_breaks_adv(p_shaped, p_width, p_start, p_once, p_break_flags);
	for (int i = 0; i < lines.size(); i++) {
		ret.push_back(lines[i]);
	}

	return ret;
}

Array TextServer::_shaped_text_get_line_breaks(RID p_shaped, real_t p_width, int p_start, uint8_t p_break_flags) const {
	Array ret;

	Vector<Vector2i> lines = shaped_text_get_line_breaks(p_shaped, p_width, p_start, p_break_flags);
	for (int i = 0; i < lines.size(); i++) {
		ret.push_back(lines[i]);
	}

	return ret;
}

Array TextServer::_shaped_text_get_word_breaks(RID p_shaped) const {
	Array ret;

	Vector<Vector2i> words = shaped_text_get_word_breaks(p_shaped);
	for (int i = 0; i < words.size(); i++) {
		ret.push_back(words[i]);
	}

	return ret;
}

Dictionary TextServer::_shaped_text_get_carets(RID p_shaped, int p_position) const {
	Dictionary ret;

	Rect2 l_caret, t_caret;
	Direction l_dir, t_dir;
	shaped_text_get_carets(p_shaped, p_position, l_caret, l_dir, t_caret, t_dir);

	ret["leading_rect"] = l_caret;
	ret["leading_direction"] = l_dir;
	ret["trailing_rect"] = t_caret;
	ret["trailing_direction"] = t_dir;

	return ret;
}

Array TextServer::_shaped_text_get_selection(RID p_shaped, int p_start, int p_end) const {
	Array ret;

	Vector<Vector2> ranges = shaped_text_get_selection(p_shaped, p_start, p_end);
	for (int i = 0; i < ranges.size(); i++) {
		ret.push_back(ranges[i]);
	}

	return ret;
}

TextServer::TextServer() {
}

TextServer::~TextServer() {
}
