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
#include "servers/rendering_server.h"

TextServerManager *TextServerManager::singleton = nullptr;

void TextServerManager::_bind_methods() {
	ClassDB::bind_method(D_METHOD("add_interface", "interface"), &TextServerManager::add_interface);
	ClassDB::bind_method(D_METHOD("get_interface_count"), &TextServerManager::get_interface_count);
	ClassDB::bind_method(D_METHOD("remove_interface", "interface"), &TextServerManager::remove_interface);
	ClassDB::bind_method(D_METHOD("get_interface", "idx"), &TextServerManager::get_interface);
	ClassDB::bind_method(D_METHOD("get_interfaces"), &TextServerManager::get_interfaces);
	ClassDB::bind_method(D_METHOD("find_interface", "name"), &TextServerManager::find_interface);

	ClassDB::bind_method(D_METHOD("set_primary_interface", "index"), &TextServerManager::set_primary_interface);
	ClassDB::bind_method(D_METHOD("get_primary_interface"), &TextServerManager::_get_primary_interface);

	ADD_SIGNAL(MethodInfo("interface_added", PropertyInfo(Variant::STRING_NAME, "interface_name")));
	ADD_SIGNAL(MethodInfo("interface_removed", PropertyInfo(Variant::STRING_NAME, "interface_name")));
}

void TextServerManager::add_interface(const Ref<TextServer> &p_interface) {
	ERR_FAIL_COND(p_interface.is_null());

	for (int i = 0; i < interfaces.size(); i++) {
		if (interfaces[i] == p_interface) {
			ERR_PRINT("TextServer: Interface was already added.");
			return;
		};
	};

	interfaces.push_back(p_interface);
	print_verbose("TextServer: Added interface \"" + p_interface->get_name() + "\"");
	emit_signal(SNAME("interface_added"), p_interface->get_name());
}

void TextServerManager::remove_interface(const Ref<TextServer> &p_interface) {
	ERR_FAIL_COND(p_interface.is_null());
	ERR_FAIL_COND_MSG(p_interface == primary_interface, "TextServer: Can't remove primary interface.");

	int idx = -1;
	for (int i = 0; i < interfaces.size(); i++) {
		if (interfaces[i] == p_interface) {
			idx = i;
			break;
		};
	};

	ERR_FAIL_COND(idx == -1);
	print_verbose("TextServer: Removed interface \"" + p_interface->get_name() + "\"");
	emit_signal(SNAME("interface_removed"), p_interface->get_name());
	interfaces.remove(idx);
}

int TextServerManager::get_interface_count() const {
	return interfaces.size();
}

Ref<TextServer> TextServerManager::get_interface(int p_index) const {
	ERR_FAIL_INDEX_V(p_index, interfaces.size(), nullptr);
	return interfaces[p_index];
}

Ref<TextServer> TextServerManager::find_interface(const String &p_name) const {
	int idx = -1;
	for (int i = 0; i < interfaces.size(); i++) {
		if (interfaces[i]->get_name() == p_name) {
			idx = i;
			break;
		};
	};

	ERR_FAIL_COND_V(idx == -1, nullptr);
	return interfaces[idx];
}

Array TextServerManager::get_interfaces() const {
	Array ret;

	for (int i = 0; i < interfaces.size(); i++) {
		Dictionary iface_info;

		iface_info["id"] = i;
		iface_info["name"] = interfaces[i]->get_name();

		ret.push_back(iface_info);
	};

	return ret;
}

Ref<TextServer> TextServerManager::_get_primary_interface() const {
	return primary_interface;
}

void TextServerManager::set_primary_interface(const Ref<TextServer> &p_primary_interface) {
	if (p_primary_interface.is_null()) {
		print_verbose("TextServer: Clearing primary interface");
		primary_interface.unref();
	} else {
		primary_interface = p_primary_interface;
		print_verbose("TextServer: Primary interface set to: \"" + primary_interface->get_name() + "\".");

		if (OS::get_singleton()->get_main_loop()) {
			OS::get_singleton()->get_main_loop()->notification(MainLoop::NOTIFICATION_TEXT_SERVER_CHANGED);
		}
	}
}

TextServerManager::TextServerManager() {
	singleton = this;
}

TextServerManager::~TextServerManager() {
	if (primary_interface.is_valid()) {
		primary_interface.unref();
	}
	while (interfaces.size() > 0) {
		interfaces.remove(0);
	}
	singleton = nullptr;
}

/*************************************************************************/

bool Glyph::operator==(const Glyph &p_a) const {
	return (p_a.index == index) && (p_a.font_rid == font_rid) && (p_a.font_size == font_size) && (p_a.start == start);
}

bool Glyph::operator!=(const Glyph &p_a) const {
	return (p_a.index != index) || (p_a.font_rid != font_rid) || (p_a.font_size != font_size) || (p_a.start != start);
}

bool Glyph::operator<(const Glyph &p_a) const {
	if (p_a.start == start) {
		if (p_a.count == count) {
			if ((p_a.flags & TextServer::GRAPHEME_IS_VIRTUAL) == TextServer::GRAPHEME_IS_VIRTUAL) {
				return true;
			} else {
				return false;
			}
		}
		return p_a.count > count;
	}
	return p_a.start < start;
}

bool Glyph::operator>(const Glyph &p_a) const {
	if (p_a.start == start) {
		if (p_a.count == count) {
			if ((p_a.flags & TextServer::GRAPHEME_IS_VIRTUAL) == TextServer::GRAPHEME_IS_VIRTUAL) {
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
	ClassDB::bind_method(D_METHOD("get_features"), &TextServer::get_features);
	ClassDB::bind_method(D_METHOD("load_support_data", "filename"), &TextServer::load_support_data);

	ClassDB::bind_method(D_METHOD("get_support_data_filename"), &TextServer::get_support_data_filename);
	ClassDB::bind_method(D_METHOD("get_support_data_info"), &TextServer::get_support_data_info);
	ClassDB::bind_method(D_METHOD("save_support_data", "filename"), &TextServer::save_support_data);

	ClassDB::bind_method(D_METHOD("is_locale_right_to_left", "locale"), &TextServer::is_locale_right_to_left);

	ClassDB::bind_method(D_METHOD("name_to_tag", "name"), &TextServer::name_to_tag);
	ClassDB::bind_method(D_METHOD("tag_to_name", "tag"), &TextServer::tag_to_name);

	ClassDB::bind_method(D_METHOD("has", "rid"), &TextServer::has);
	ClassDB::bind_method(D_METHOD("free_rid", "rid"), &TextServer::free); // shouldn't conflict with Object::free()

	/* Font Interface */

	ClassDB::bind_method(D_METHOD("create_font"), &TextServer::create_font);

	ClassDB::bind_method(D_METHOD("font_set_data", "font_rid", "data"), &TextServer::font_set_data);

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

	ClassDB::bind_method(D_METHOD("font_get_glyph_contours", "font", "size", "index"), &TextServer::font_get_glyph_contours);

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

	ClassDB::bind_method(D_METHOD("shaped_text_set_bidi_override", "shaped", "override"), &TextServer::shaped_text_set_bidi_override);

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

	ClassDB::bind_method(D_METHOD("shaped_text_get_glyphs", "shaped"), &TextServer::_shaped_text_get_glyphs_wrapper);
	ClassDB::bind_method(D_METHOD("shaped_text_sort_logical", "shaped"), &TextServer::_shaped_text_sort_logical_wrapper);
	ClassDB::bind_method(D_METHOD("shaped_text_get_glyph_count", "shaped"), &TextServer::shaped_text_get_glyph_count);

	ClassDB::bind_method(D_METHOD("shaped_text_get_range", "shaped"), &TextServer::shaped_text_get_range);
	ClassDB::bind_method(D_METHOD("shaped_text_get_line_breaks_adv", "shaped", "width", "start", "once", "break_flags"), &TextServer::shaped_text_get_line_breaks_adv, DEFVAL(0), DEFVAL(true), DEFVAL(BREAK_MANDATORY | BREAK_WORD_BOUND));
	ClassDB::bind_method(D_METHOD("shaped_text_get_line_breaks", "shaped", "width", "start", "break_flags"), &TextServer::shaped_text_get_line_breaks, DEFVAL(0), DEFVAL(BREAK_MANDATORY | BREAK_WORD_BOUND));
	ClassDB::bind_method(D_METHOD("shaped_text_get_word_breaks", "shaped", "grapheme_flags"), &TextServer::shaped_text_get_word_breaks);

	ClassDB::bind_method(D_METHOD("shaped_text_get_trim_pos", "shaped"), &TextServer::shaped_text_get_trim_pos);
	ClassDB::bind_method(D_METHOD("shaped_text_get_ellipsis_pos", "shaped"), &TextServer::shaped_text_get_ellipsis_pos);
	ClassDB::bind_method(D_METHOD("shaped_text_get_ellipsis_glyphs", "shaped"), &TextServer::_shaped_text_get_ellipsis_glyphs_wrapper);
	ClassDB::bind_method(D_METHOD("shaped_text_get_ellipsis_glyph_count", "shaped"), &TextServer::shaped_text_get_ellipsis_glyph_count);

	ClassDB::bind_method(D_METHOD("shaped_text_overrun_trim_to_width", "shaped", "width", "overrun_trim_flags"), &TextServer::shaped_text_overrun_trim_to_width, DEFVAL(0), DEFVAL(OVERRUN_NO_TRIMMING));

	ClassDB::bind_method(D_METHOD("shaped_text_get_objects", "shaped"), &TextServer::shaped_text_get_objects);
	ClassDB::bind_method(D_METHOD("shaped_text_get_object_rect", "shaped", "key"), &TextServer::shaped_text_get_object_rect);

	ClassDB::bind_method(D_METHOD("shaped_text_get_size", "shaped"), &TextServer::shaped_text_get_size);
	ClassDB::bind_method(D_METHOD("shaped_text_get_ascent", "shaped"), &TextServer::shaped_text_get_ascent);
	ClassDB::bind_method(D_METHOD("shaped_text_get_descent", "shaped"), &TextServer::shaped_text_get_descent);
	ClassDB::bind_method(D_METHOD("shaped_text_get_width", "shaped"), &TextServer::shaped_text_get_width);
	ClassDB::bind_method(D_METHOD("shaped_text_get_underline_position", "shaped"), &TextServer::shaped_text_get_underline_position);
	ClassDB::bind_method(D_METHOD("shaped_text_get_underline_thickness", "shaped"), &TextServer::shaped_text_get_underline_thickness);

	ClassDB::bind_method(D_METHOD("shaped_text_get_carets", "shaped", "position"), &TextServer::_shaped_text_get_carets_wrapper);
	ClassDB::bind_method(D_METHOD("shaped_text_get_selection", "shaped", "start", "end"), &TextServer::shaped_text_get_selection);

	ClassDB::bind_method(D_METHOD("shaped_text_hit_test_grapheme", "shaped", "coords"), &TextServer::shaped_text_hit_test_grapheme);
	ClassDB::bind_method(D_METHOD("shaped_text_hit_test_position", "shaped", "coords"), &TextServer::shaped_text_hit_test_position);

	ClassDB::bind_method(D_METHOD("shaped_text_next_grapheme_pos", "shaped", "pos"), &TextServer::shaped_text_next_grapheme_pos);
	ClassDB::bind_method(D_METHOD("shaped_text_prev_grapheme_pos", "shaped", "pos"), &TextServer::shaped_text_prev_grapheme_pos);

	ClassDB::bind_method(D_METHOD("shaped_text_draw", "shaped", "canvas", "pos", "clip_l", "clip_r", "color"), &TextServer::shaped_text_draw, DEFVAL(-1), DEFVAL(-1), DEFVAL(Color(1, 1, 1)));
	ClassDB::bind_method(D_METHOD("shaped_text_draw_outline", "shaped", "canvas", "pos", "clip_l", "clip_r", "outline_size", "color"), &TextServer::shaped_text_draw_outline, DEFVAL(-1), DEFVAL(-1), DEFVAL(1), DEFVAL(Color(1, 1, 1)));

	ClassDB::bind_method(D_METHOD("shaped_text_get_dominant_direction_in_range", "shaped", "start", "end"), &TextServer::shaped_text_get_dominant_direction_in_range);

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
	BIND_ENUM_CONSTANT(JUSTIFICATION_CONSTRAIN_ELLIPSIS);

	/* LineBreakFlag */
	BIND_ENUM_CONSTANT(BREAK_NONE);
	BIND_ENUM_CONSTANT(BREAK_MANDATORY);
	BIND_ENUM_CONSTANT(BREAK_WORD_BOUND);
	BIND_ENUM_CONSTANT(BREAK_GRAPHEME_BOUND);
	BIND_ENUM_CONSTANT(BREAK_WORD_BOUND_ADAPTIVE);

	/* TextOverrunFlag */
	BIND_ENUM_CONSTANT(OVERRUN_NO_TRIMMING);
	BIND_ENUM_CONSTANT(OVERRUN_TRIM);
	BIND_ENUM_CONSTANT(OVERRUN_TRIM_WORD_ONLY);
	BIND_ENUM_CONSTANT(OVERRUN_ADD_ELLIPSIS);
	BIND_ENUM_CONSTANT(OVERRUN_ENFORCE_ELLIPSIS);
	BIND_ENUM_CONSTANT(OVERRUN_JUSTIFICATION_AWARE);

	/* GraphemeFlag */
	BIND_ENUM_CONSTANT(GRAPHEME_IS_VALID);
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

Vector2 TextServer::get_hex_code_box_size(int p_size, char32_t p_index) const {
	int w = ((p_index <= 0xFF) ? 1 : ((p_index <= 0xFFFF) ? 2 : 3));
	int sp = MAX(0, w - 1);
	int sz = MAX(1, Math::round(p_size / 15.f));

	return Vector2(4 + 3 * w + sp + 1, 15) * sz;
}

void TextServer::_draw_hex_code_box_number(RID p_canvas, int p_size, const Vector2 &p_pos, uint8_t p_index, const Color &p_color) const {
	static uint8_t chars[] = { 0x7E, 0x30, 0x6D, 0x79, 0x33, 0x5B, 0x5F, 0x70, 0x7F, 0x7B, 0x77, 0x1F, 0x4E, 0x3D, 0x4F, 0x47, 0x00 };
	uint8_t x = chars[p_index];
	if (x & (1 << 6)) {
		RenderingServer::get_singleton()->canvas_item_add_rect(p_canvas, Rect2(p_pos, Size2(3, 1) * p_size), p_color);
	}
	if (x & (1 << 5)) {
		RenderingServer::get_singleton()->canvas_item_add_rect(p_canvas, Rect2(p_pos + Point2(2, 0) * p_size, Size2(1, 3) * p_size), p_color);
	}
	if (x & (1 << 4)) {
		RenderingServer::get_singleton()->canvas_item_add_rect(p_canvas, Rect2(p_pos + Point2(2, 2) * p_size, Size2(1, 3) * p_size), p_color);
	}
	if (x & (1 << 3)) {
		RenderingServer::get_singleton()->canvas_item_add_rect(p_canvas, Rect2(p_pos + Point2(0, 4) * p_size, Size2(3, 1) * p_size), p_color);
	}
	if (x & (1 << 2)) {
		RenderingServer::get_singleton()->canvas_item_add_rect(p_canvas, Rect2(p_pos + Point2(0, 2) * p_size, Size2(1, 3) * p_size), p_color);
	}
	if (x & (1 << 1)) {
		RenderingServer::get_singleton()->canvas_item_add_rect(p_canvas, Rect2(p_pos, Size2(1, 3) * p_size), p_color);
	}
	if (x & (1 << 0)) {
		RenderingServer::get_singleton()->canvas_item_add_rect(p_canvas, Rect2(p_pos + Point2(0, 2) * p_size, Size2(3, 1) * p_size), p_color);
	}
}

void TextServer::draw_hex_code_box(RID p_canvas, int p_size, const Vector2 &p_pos, char32_t p_index, const Color &p_color) const {
	if (p_index == 0) {
		return;
	}

	int w = ((p_index <= 0xFF) ? 1 : ((p_index <= 0xFFFF) ? 2 : 3));
	int sp = MAX(0, w - 1);
	int sz = MAX(1, Math::round(p_size / 15.f));

	Size2 size = Vector2(4 + 3 * w + sp, 15) * sz;
	Point2 pos = p_pos - Point2i(0, size.y * 0.85);

	// Draw frame.
	RenderingServer::get_singleton()->canvas_item_add_rect(p_canvas, Rect2(pos + Point2(0, 0), Size2(sz, size.y)), p_color);
	RenderingServer::get_singleton()->canvas_item_add_rect(p_canvas, Rect2(pos + Point2(size.x - sz, 0), Size2(sz, size.y)), p_color);
	RenderingServer::get_singleton()->canvas_item_add_rect(p_canvas, Rect2(pos + Point2(0, 0), Size2(size.x, sz)), p_color);
	RenderingServer::get_singleton()->canvas_item_add_rect(p_canvas, Rect2(pos + Point2(0, size.y - sz), Size2(size.x, sz)), p_color);

	uint8_t a = p_index & 0x0F;
	uint8_t b = (p_index >> 4) & 0x0F;
	uint8_t c = (p_index >> 8) & 0x0F;
	uint8_t d = (p_index >> 12) & 0x0F;
	uint8_t e = (p_index >> 16) & 0x0F;
	uint8_t f = (p_index >> 20) & 0x0F;

	// Draw hex code.
	if (p_index <= 0xFF) {
		_draw_hex_code_box_number(p_canvas, sz, pos + Point2(2, 2) * sz, b, p_color);
		_draw_hex_code_box_number(p_canvas, sz, pos + Point2(2, 8) * sz, a, p_color);
	} else if (p_index <= 0xFFFF) {
		_draw_hex_code_box_number(p_canvas, sz, pos + Point2(2, 2) * sz, d, p_color);
		_draw_hex_code_box_number(p_canvas, sz, pos + Point2(6, 2) * sz, c, p_color);
		_draw_hex_code_box_number(p_canvas, sz, pos + Point2(2, 8) * sz, b, p_color);
		_draw_hex_code_box_number(p_canvas, sz, pos + Point2(6, 8) * sz, a, p_color);
	} else {
		_draw_hex_code_box_number(p_canvas, sz, pos + Point2(2, 2) * sz, f, p_color);
		_draw_hex_code_box_number(p_canvas, sz, pos + Point2(6, 2) * sz, e, p_color);
		_draw_hex_code_box_number(p_canvas, sz, pos + Point2(10, 2) * sz, d, p_color);
		_draw_hex_code_box_number(p_canvas, sz, pos + Point2(2, 8) * sz, c, p_color);
		_draw_hex_code_box_number(p_canvas, sz, pos + Point2(6, 8) * sz, b, p_color);
		_draw_hex_code_box_number(p_canvas, sz, pos + Point2(10, 8) * sz, a, p_color);
	}
}

PackedInt32Array TextServer::shaped_text_get_line_breaks_adv(RID p_shaped, const PackedFloat32Array &p_width, int p_start, bool p_once, uint16_t /*TextBreakFlag*/ p_break_flags) const {
	PackedInt32Array lines;

	ERR_FAIL_COND_V(p_width.is_empty(), lines);

	const_cast<TextServer *>(this)->shaped_text_update_breaks(p_shaped);
	const Vector2i &range = shaped_text_get_range(p_shaped);

	real_t width = 0.f;
	int line_start = MAX(p_start, range.x);
	int last_safe_break = -1;
	int chunk = 0;

	int l_size = shaped_text_get_glyph_count(p_shaped);
	const Glyph *l_gl = const_cast<TextServer *>(this)->shaped_text_sort_logical(p_shaped);

	for (int i = 0; i < l_size; i++) {
		if (l_gl[i].start < p_start) {
			continue;
		}
		if (l_gl[i].count > 0) {
			if ((p_width[chunk] > 0) && (width + l_gl[i].advance > p_width[chunk]) && (last_safe_break >= 0)) {
				lines.push_back(line_start);
				lines.push_back(l_gl[last_safe_break].end);
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
					lines.push_back(line_start);
					lines.push_back(l_gl[i].end);
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
		if (lines.size() == 0 || lines[lines.size() - 1] < range.y) {
			lines.push_back(line_start);
			lines.push_back(range.y);
		}
	} else {
		lines.push_back(0);
		lines.push_back(0);
	}

	return lines;
}

PackedInt32Array TextServer::shaped_text_get_line_breaks(RID p_shaped, real_t p_width, int p_start, uint16_t /*TextBreakFlag*/ p_break_flags) const {
	PackedInt32Array lines;

	const_cast<TextServer *>(this)->shaped_text_update_breaks(p_shaped);
	const Vector2i &range = shaped_text_get_range(p_shaped);

	real_t width = 0.f;
	int line_start = MAX(p_start, range.x);
	int last_safe_break = -1;
	int word_count = 0;

	int l_size = shaped_text_get_glyph_count(p_shaped);
	const Glyph *l_gl = const_cast<TextServer *>(this)->shaped_text_sort_logical(p_shaped);

	for (int i = 0; i < l_size; i++) {
		if (l_gl[i].start < p_start) {
			continue;
		}
		if (l_gl[i].count > 0) {
			if ((p_width > 0) && (width + l_gl[i].advance * l_gl[i].repeat > p_width) && (last_safe_break >= 0)) {
				lines.push_back(line_start);
				lines.push_back(l_gl[last_safe_break].end);
				line_start = l_gl[last_safe_break].end;
				i = last_safe_break;
				last_safe_break = -1;
				width = 0;
				word_count = 0;
				continue;
			}
			if ((p_break_flags & BREAK_MANDATORY) == BREAK_MANDATORY) {
				if ((l_gl[i].flags & GRAPHEME_IS_BREAK_HARD) == GRAPHEME_IS_BREAK_HARD) {
					lines.push_back(line_start);
					lines.push_back(l_gl[i].end);
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
		if (lines.size() == 0 || lines[lines.size() - 1] < range.y) {
			lines.push_back(line_start);
			lines.push_back(range.y);
		}
	} else {
		lines.push_back(0);
		lines.push_back(0);
	}

	return lines;
}

PackedInt32Array TextServer::shaped_text_get_word_breaks(RID p_shaped, int p_grapheme_flags) const {
	PackedInt32Array words;

	const_cast<TextServer *>(this)->shaped_text_update_justification_ops(p_shaped);
	const Vector2i &range = shaped_text_get_range(p_shaped);

	int word_start = range.x;

	int l_size = shaped_text_get_glyph_count(p_shaped);
	const Glyph *l_gl = const_cast<TextServer *>(this)->shaped_text_sort_logical(p_shaped);

	for (int i = 0; i < l_size; i++) {
		if (l_gl[i].count > 0) {
			if ((l_gl[i].flags & p_grapheme_flags) != 0) {
				words.push_back(word_start);
				words.push_back(l_gl[i].start);
				word_start = l_gl[i].end;
			}
		}
	}
	if (l_size > 0) {
		words.push_back(word_start);
		words.push_back(range.y);
	}

	return words;
}

CaretInfo TextServer::shaped_text_get_carets(RID p_shaped, int p_position) const {
	Vector<Rect2> carets;

	TextServer::Orientation orientation = shaped_text_get_orientation(p_shaped);
	const Vector2 &range = shaped_text_get_range(p_shaped);
	real_t ascent = shaped_text_get_ascent(p_shaped);
	real_t descent = shaped_text_get_descent(p_shaped);
	real_t height = (ascent + descent) / 2;

	real_t off = 0.0f;
	CaretInfo caret;
	caret.l_dir = DIRECTION_AUTO;
	caret.t_dir = DIRECTION_AUTO;

	int v_size = shaped_text_get_glyph_count(p_shaped);
	const Glyph *glyphs = shaped_text_get_glyphs(p_shaped);

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
						caret.t_dir = DIRECTION_RTL;
						for (int j = 0; j < glyphs[i].count; j++) {
							cr.position.x += glyphs[i + j].advance * glyphs[i + j].repeat;
							cr.size.x -= glyphs[i + j].advance * glyphs[i + j].repeat;
						}
					} else {
						caret.t_dir = DIRECTION_LTR;
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
						caret.t_dir = DIRECTION_RTL;
						for (int j = 0; j < glyphs[i].count; j++) {
							cr.position.y += glyphs[i + j].advance * glyphs[i + j].repeat;
							cr.size.y -= glyphs[i + j].advance * glyphs[i + j].repeat;
						}
					} else {
						caret.t_dir = DIRECTION_LTR;
						for (int j = 0; j < glyphs[i].count; j++) {
							cr.size.y += glyphs[i + j].advance * glyphs[i + j].repeat;
						}
					}
				}
				caret.t_caret = cr;
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
						caret.l_dir = DIRECTION_LTR;
						for (int j = 0; j < glyphs[i].count; j++) {
							cr.position.x += glyphs[i + j].advance * glyphs[i + j].repeat;
							cr.size.x -= glyphs[i + j].advance * glyphs[i + j].repeat;
						}
					} else {
						caret.l_dir = DIRECTION_RTL;
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
						caret.l_dir = DIRECTION_LTR;
						for (int j = 0; j < glyphs[i].count; j++) {
							cr.position.y += glyphs[i + j].advance * glyphs[i + j].repeat;
							cr.size.y -= glyphs[i + j].advance * glyphs[i + j].repeat;
						}
					} else {
						caret.l_dir = DIRECTION_RTL;
						for (int j = 0; j < glyphs[i].count; j++) {
							cr.size.y += glyphs[i + j].advance * glyphs[i + j].repeat;
						}
					}
				}
				caret.l_caret = cr;
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
				caret.t_caret = cr;
				caret.l_caret = cr;
			}
		}
		off += glyphs[i].advance * glyphs[i].repeat;
	}
	return caret;
}

Dictionary TextServer::_shaped_text_get_carets_wrapper(RID p_shaped, int p_position) const {
	Dictionary ret;

	CaretInfo caret = shaped_text_get_carets(p_shaped, p_position);

	ret["leading_rect"] = caret.l_caret;
	ret["leading_direction"] = caret.l_dir;
	ret["trailing_rect"] = caret.t_caret;
	ret["trailing_direction"] = caret.t_dir;

	return ret;
}

TextServer::Direction TextServer::shaped_text_get_dominant_direction_in_range(RID p_shaped, int p_start, int p_end) const {
	if (p_start == p_end) {
		return DIRECTION_AUTO;
	}

	int start = MIN(p_start, p_end);
	int end = MAX(p_start, p_end);

	int rtl = 0;
	int ltr = 0;

	int v_size = shaped_text_get_glyph_count(p_shaped);
	const Glyph *glyphs = shaped_text_get_glyphs(p_shaped);

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

	if (p_start == p_end) {
		return ranges;
	}

	int start = MIN(p_start, p_end);
	int end = MAX(p_start, p_end);

	int v_size = shaped_text_get_glyph_count(p_shaped);
	const Glyph *glyphs = shaped_text_get_glyphs(p_shaped);

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
	// Exact grapheme hit test, return -1 if missed.
	real_t off = 0.0f;

	int v_size = shaped_text_get_glyph_count(p_shaped);
	const Glyph *glyphs = shaped_text_get_glyphs(p_shaped);

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
	int v_size = shaped_text_get_glyph_count(p_shaped);
	const Glyph *glyphs = shaped_text_get_glyphs(p_shaped);

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

int TextServer::shaped_text_next_grapheme_pos(RID p_shaped, int p_pos) const {
	int v_size = shaped_text_get_glyph_count(p_shaped);
	const Glyph *glyphs = shaped_text_get_glyphs(p_shaped);
	for (int i = 0; i < v_size; i++) {
		if (p_pos >= glyphs[i].start && p_pos < glyphs[i].end) {
			return glyphs[i].end;
		}
	}
	return p_pos;
}

int TextServer::shaped_text_prev_grapheme_pos(RID p_shaped, int p_pos) const {
	int v_size = shaped_text_get_glyph_count(p_shaped);
	const Glyph *glyphs = shaped_text_get_glyphs(p_shaped);
	for (int i = 0; i < v_size; i++) {
		if (p_pos > glyphs[i].start && p_pos <= glyphs[i].end) {
			return glyphs[i].start;
		}
	}

	return p_pos;
}

void TextServer::shaped_text_draw(RID p_shaped, RID p_canvas, const Vector2 &p_pos, real_t p_clip_l, real_t p_clip_r, const Color &p_color) const {
	TextServer::Orientation orientation = shaped_text_get_orientation(p_shaped);
	bool hex_codes = shaped_text_get_preserve_control(p_shaped) || shaped_text_get_preserve_invalid(p_shaped);

	bool rtl = shaped_text_get_direction(p_shaped) == DIRECTION_RTL;

	int ellipsis_pos = shaped_text_get_ellipsis_pos(p_shaped);
	int trim_pos = shaped_text_get_trim_pos(p_shaped);

	const Glyph *ellipsis_glyphs = shaped_text_get_ellipsis_glyphs(p_shaped);
	int ellipsis_gl_size = shaped_text_get_ellipsis_glyph_count(p_shaped);

	int v_size = shaped_text_get_glyph_count(p_shaped);
	const Glyph *glyphs = shaped_text_get_glyphs(p_shaped);

	Vector2 ofs = p_pos;
	// Draw RTL ellipsis string when needed.
	if (rtl && ellipsis_pos >= 0) {
		for (int i = ellipsis_gl_size - 1; i >= 0; i--) {
			for (int j = 0; j < ellipsis_glyphs[i].repeat; j++) {
				font_draw_glyph(ellipsis_glyphs[i].font_rid, p_canvas, ellipsis_glyphs[i].font_size, ofs + Vector2(ellipsis_glyphs[i].x_off, ellipsis_glyphs[i].y_off), ellipsis_glyphs[i].index, p_color);
				if (orientation == ORIENTATION_HORIZONTAL) {
					ofs.x += ellipsis_glyphs[i].advance;
				} else {
					ofs.y += ellipsis_glyphs[i].advance;
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
			if (trim_pos >= 0) {
				if (rtl) {
					if (i < trim_pos && (glyphs[j].flags & TextServer::GRAPHEME_IS_VIRTUAL) != TextServer::GRAPHEME_IS_VIRTUAL) {
						continue;
					}
				} else {
					if (i >= trim_pos && (glyphs[j].flags & TextServer::GRAPHEME_IS_VIRTUAL) != TextServer::GRAPHEME_IS_VIRTUAL) {
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
	if (!rtl && ellipsis_pos >= 0) {
		for (int i = 0; i < ellipsis_gl_size; i++) {
			for (int j = 0; j < ellipsis_glyphs[i].repeat; j++) {
				font_draw_glyph(ellipsis_glyphs[i].font_rid, p_canvas, ellipsis_glyphs[i].font_size, ofs + Vector2(ellipsis_glyphs[i].x_off, ellipsis_glyphs[i].y_off), ellipsis_glyphs[i].index, p_color);
				if (orientation == ORIENTATION_HORIZONTAL) {
					ofs.x += ellipsis_glyphs[i].advance;
				} else {
					ofs.y += ellipsis_glyphs[i].advance;
				}
			}
		}
	}
}

void TextServer::shaped_text_draw_outline(RID p_shaped, RID p_canvas, const Vector2 &p_pos, real_t p_clip_l, real_t p_clip_r, int p_outline_size, const Color &p_color) const {
	TextServer::Orientation orientation = shaped_text_get_orientation(p_shaped);

	bool rtl = (shaped_text_get_direction(p_shaped) == DIRECTION_RTL);

	int ellipsis_pos = shaped_text_get_ellipsis_pos(p_shaped);
	int trim_pos = shaped_text_get_trim_pos(p_shaped);

	const Glyph *ellipsis_glyphs = shaped_text_get_ellipsis_glyphs(p_shaped);
	int ellipsis_gl_size = shaped_text_get_ellipsis_glyph_count(p_shaped);

	int v_size = shaped_text_get_glyph_count(p_shaped);
	const Glyph *glyphs = shaped_text_get_glyphs(p_shaped);
	Vector2 ofs = p_pos;
	// Draw RTL ellipsis string when needed.
	if (rtl && ellipsis_pos >= 0) {
		for (int i = ellipsis_gl_size - 1; i >= 0; i--) {
			for (int j = 0; j < ellipsis_glyphs[i].repeat; j++) {
				font_draw_glyph(ellipsis_glyphs[i].font_rid, p_canvas, ellipsis_glyphs[i].font_size, ofs + Vector2(ellipsis_glyphs[i].x_off, ellipsis_glyphs[i].y_off), ellipsis_glyphs[i].index, p_color);
				if (orientation == ORIENTATION_HORIZONTAL) {
					ofs.x += ellipsis_glyphs[i].advance;
				} else {
					ofs.y += ellipsis_glyphs[i].advance;
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
			if (trim_pos >= 0) {
				if (rtl) {
					if (i < trim_pos) {
						continue;
					}
				} else {
					if (i >= trim_pos && (glyphs[j].flags & TextServer::GRAPHEME_IS_VIRTUAL) != TextServer::GRAPHEME_IS_VIRTUAL) {
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
	if (!rtl && ellipsis_pos >= 0) {
		for (int i = 0; i < ellipsis_gl_size; i++) {
			for (int j = 0; j < ellipsis_glyphs[i].repeat; j++) {
				font_draw_glyph(ellipsis_glyphs[i].font_rid, p_canvas, ellipsis_glyphs[i].font_size, ofs + Vector2(ellipsis_glyphs[i].x_off, ellipsis_glyphs[i].y_off), ellipsis_glyphs[i].index, p_color);
				if (orientation == ORIENTATION_HORIZONTAL) {
					ofs.x += ellipsis_glyphs[i].advance;
				} else {
					ofs.y += ellipsis_glyphs[i].advance;
				}
			}
		}
	}
}

Array TextServer::_shaped_text_get_glyphs_wrapper(RID p_shaped) const {
	Array ret;

	const Glyph *glyphs = shaped_text_get_glyphs(p_shaped);
	int gl_size = shaped_text_get_glyph_count(p_shaped);
	for (int i = 0; i < gl_size; i++) {
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

Array TextServer::_shaped_text_sort_logical_wrapper(RID p_shaped) {
	Array ret;

	const Glyph *glyphs = shaped_text_sort_logical(p_shaped);
	int gl_size = shaped_text_get_glyph_count(p_shaped);
	for (int i = 0; i < gl_size; i++) {
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

Array TextServer::_shaped_text_get_ellipsis_glyphs_wrapper(RID p_shaped) const {
	Array ret;

	const Glyph *glyphs = shaped_text_get_ellipsis_glyphs(p_shaped);
	int gl_size = shaped_text_get_ellipsis_glyph_count(p_shaped);
	for (int i = 0; i < gl_size; i++) {
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

TextServer::TextServer() {
}

TextServer::~TextServer() {
}
