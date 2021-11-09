/*************************************************************************/
/*  font.cpp                                                             */
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

#include "font.h"

#include "core/io/image_loader.h"
#include "core/io/resource_loader.h"
#include "core/string/translation.h"
#include "core/templates/hashfuncs.h"
#include "scene/resources/text_line.h"
#include "scene/resources/text_paragraph.h"

_FORCE_INLINE_ void FontData::_clear_cache() {
	for (int i = 0; i < cache.size(); i++) {
		if (cache[i].is_valid()) {
			TS->free(cache[i]);
			cache.write[i] = RID();
		}
	}
}

_FORCE_INLINE_ void FontData::_ensure_rid(int p_cache_index) const {
	if (unlikely(p_cache_index >= cache.size())) {
		cache.resize(p_cache_index + 1);
	}
	if (unlikely(!cache[p_cache_index].is_valid())) {
		cache.write[p_cache_index] = TS->create_font();
		TS->font_set_data_ptr(cache[p_cache_index], data_ptr, data_size);
		TS->font_set_antialiased(cache[p_cache_index], antialiased);
		TS->font_set_multichannel_signed_distance_field(cache[p_cache_index], msdf);
		TS->font_set_msdf_pixel_range(cache[p_cache_index], msdf_pixel_range);
		TS->font_set_msdf_size(cache[p_cache_index], msdf_size);
		TS->font_set_fixed_size(cache[p_cache_index], fixed_size);
		TS->font_set_force_autohinter(cache[p_cache_index], force_autohinter);
		TS->font_set_hinting(cache[p_cache_index], hinting);
		TS->font_set_oversampling(cache[p_cache_index], oversampling);
	}
}

void FontData::_bind_methods() {
	ClassDB::bind_method(D_METHOD("load_bitmap_font", "path"), &FontData::load_bitmap_font);
	ClassDB::bind_method(D_METHOD("load_dynamic_font", "path"), &FontData::load_dynamic_font);

	ClassDB::bind_method(D_METHOD("set_data", "data"), &FontData::set_data);
	ClassDB::bind_method(D_METHOD("get_data"), &FontData::get_data);

	ClassDB::bind_method(D_METHOD("set_antialiased", "antialiased"), &FontData::set_antialiased);
	ClassDB::bind_method(D_METHOD("is_antialiased"), &FontData::is_antialiased);

	ClassDB::bind_method(D_METHOD("set_font_name", "name"), &FontData::set_font_name);
	ClassDB::bind_method(D_METHOD("get_font_name"), &FontData::get_font_name);

	ClassDB::bind_method(D_METHOD("set_font_style_name", "name"), &FontData::set_font_style_name);
	ClassDB::bind_method(D_METHOD("get_font_style_name"), &FontData::get_font_style_name);

	ClassDB::bind_method(D_METHOD("set_font_style", "style"), &FontData::set_font_style);
	ClassDB::bind_method(D_METHOD("get_font_style"), &FontData::get_font_style);

	ClassDB::bind_method(D_METHOD("set_multichannel_signed_distance_field", "msdf"), &FontData::set_multichannel_signed_distance_field);
	ClassDB::bind_method(D_METHOD("is_multichannel_signed_distance_field"), &FontData::is_multichannel_signed_distance_field);

	ClassDB::bind_method(D_METHOD("set_msdf_pixel_range", "msdf_pixel_range"), &FontData::set_msdf_pixel_range);
	ClassDB::bind_method(D_METHOD("get_msdf_pixel_range"), &FontData::get_msdf_pixel_range);

	ClassDB::bind_method(D_METHOD("set_msdf_size", "msdf_size"), &FontData::set_msdf_size);
	ClassDB::bind_method(D_METHOD("get_msdf_size"), &FontData::get_msdf_size);

	ClassDB::bind_method(D_METHOD("set_force_autohinter", "force_autohinter"), &FontData::set_force_autohinter);
	ClassDB::bind_method(D_METHOD("is_force_autohinter"), &FontData::is_force_autohinter);

	ClassDB::bind_method(D_METHOD("set_hinting", "hinting"), &FontData::set_hinting);
	ClassDB::bind_method(D_METHOD("get_hinting"), &FontData::get_hinting);

	ClassDB::bind_method(D_METHOD("set_oversampling", "oversampling"), &FontData::set_oversampling);
	ClassDB::bind_method(D_METHOD("get_oversampling"), &FontData::get_oversampling);

	ClassDB::bind_method(D_METHOD("find_cache", "variation_coordinates"), &FontData::find_cache);

	ClassDB::bind_method(D_METHOD("get_cache_count"), &FontData::get_cache_count);
	ClassDB::bind_method(D_METHOD("clear_cache"), &FontData::clear_cache);
	ClassDB::bind_method(D_METHOD("remove_cache", "cache_index"), &FontData::remove_cache);

	ClassDB::bind_method(D_METHOD("get_size_cache_list", "cache_index"), &FontData::get_size_cache_list);
	ClassDB::bind_method(D_METHOD("clear_size_cache", "cache_index"), &FontData::clear_size_cache);
	ClassDB::bind_method(D_METHOD("remove_size_cache", "cache_index", "size"), &FontData::remove_size_cache);

	ClassDB::bind_method(D_METHOD("set_variation_coordinates", "cache_index", "variation_coordinates"), &FontData::set_variation_coordinates);
	ClassDB::bind_method(D_METHOD("get_variation_coordinates", "cache_index"), &FontData::get_variation_coordinates);

	ClassDB::bind_method(D_METHOD("set_ascent", "cache_index", "size", "ascent"), &FontData::set_ascent);
	ClassDB::bind_method(D_METHOD("get_ascent", "cache_index", "size"), &FontData::get_ascent);

	ClassDB::bind_method(D_METHOD("set_descent", "cache_index", "size", "descent"), &FontData::set_descent);
	ClassDB::bind_method(D_METHOD("get_descent", "cache_index", "size"), &FontData::get_descent);

	ClassDB::bind_method(D_METHOD("set_underline_position", "cache_index", "size", "underline_position"), &FontData::set_underline_position);
	ClassDB::bind_method(D_METHOD("get_underline_position", "cache_index", "size"), &FontData::get_underline_position);

	ClassDB::bind_method(D_METHOD("set_underline_thickness", "cache_index", "size", "underline_thickness"), &FontData::set_underline_thickness);
	ClassDB::bind_method(D_METHOD("get_underline_thickness", "cache_index", "size"), &FontData::get_underline_thickness);

	ClassDB::bind_method(D_METHOD("set_scale", "cache_index", "size", "scale"), &FontData::set_scale);
	ClassDB::bind_method(D_METHOD("get_scale", "cache_index", "size"), &FontData::get_scale);

	ClassDB::bind_method(D_METHOD("set_spacing", "cache_index", "size", "spacing_type", "value"), &FontData::set_spacing);
	ClassDB::bind_method(D_METHOD("get_spacing", "cache_index", "size", "spacing_type"), &FontData::get_spacing);

	ClassDB::bind_method(D_METHOD("get_texture_count", "cache_index", "size"), &FontData::get_texture_count);
	ClassDB::bind_method(D_METHOD("clear_textures", "cache_index", "size"), &FontData::clear_textures);
	ClassDB::bind_method(D_METHOD("remove_texture", "cache_index", "size", "texture_index"), &FontData::remove_texture);

	ClassDB::bind_method(D_METHOD("set_texture_image", "cache_index", "size", "texture_index", "image"), &FontData::set_texture_image);
	ClassDB::bind_method(D_METHOD("get_texture_image", "cache_index", "size", "texture_index"), &FontData::get_texture_image);

	ClassDB::bind_method(D_METHOD("set_texture_offsets", "cache_index", "size", "texture_index", "offset"), &FontData::set_texture_offsets);
	ClassDB::bind_method(D_METHOD("get_texture_offsets", "cache_index", "size", "texture_index"), &FontData::get_texture_offsets);

	ClassDB::bind_method(D_METHOD("get_glyph_list", "cache_index", "size"), &FontData::get_glyph_list);
	ClassDB::bind_method(D_METHOD("clear_glyphs", "cache_index", "size"), &FontData::clear_glyphs);
	ClassDB::bind_method(D_METHOD("remove_glyph", "cache_index", "size", "glyph"), &FontData::remove_glyph);

	ClassDB::bind_method(D_METHOD("set_glyph_advance", "cache_index", "size", "glyph", "advance"), &FontData::set_glyph_advance);
	ClassDB::bind_method(D_METHOD("get_glyph_advance", "cache_index", "size", "glyph"), &FontData::get_glyph_advance);

	ClassDB::bind_method(D_METHOD("set_glyph_offset", "cache_index", "size", "glyph", "offset"), &FontData::set_glyph_offset);
	ClassDB::bind_method(D_METHOD("get_glyph_offset", "cache_index", "size", "glyph"), &FontData::get_glyph_offset);

	ClassDB::bind_method(D_METHOD("set_glyph_size", "cache_index", "size", "glyph", "gl_size"), &FontData::set_glyph_size);
	ClassDB::bind_method(D_METHOD("get_glyph_size", "cache_index", "size", "glyph"), &FontData::get_glyph_size);

	ClassDB::bind_method(D_METHOD("set_glyph_uv_rect", "cache_index", "size", "glyph", "uv_rect"), &FontData::set_glyph_uv_rect);
	ClassDB::bind_method(D_METHOD("get_glyph_uv_rect", "cache_index", "size", "glyph"), &FontData::get_glyph_uv_rect);

	ClassDB::bind_method(D_METHOD("set_glyph_texture_idx", "cache_index", "size", "glyph", "texture_idx"), &FontData::set_glyph_texture_idx);
	ClassDB::bind_method(D_METHOD("get_glyph_texture_idx", "cache_index", "size", "glyph"), &FontData::get_glyph_texture_idx);

	ClassDB::bind_method(D_METHOD("get_kerning_list", "cache_index", "size"), &FontData::get_kerning_list);
	ClassDB::bind_method(D_METHOD("clear_kerning_map", "cache_index", "size"), &FontData::clear_kerning_map);
	ClassDB::bind_method(D_METHOD("remove_kerning", "cache_index", "size", "glyph_pair"), &FontData::remove_kerning);

	ClassDB::bind_method(D_METHOD("set_kerning", "cache_index", "size", "glyph_pair", "kerning"), &FontData::set_kerning);
	ClassDB::bind_method(D_METHOD("get_kerning", "cache_index", "size", "glyph_pair"), &FontData::get_kerning);

	ClassDB::bind_method(D_METHOD("render_range", "cache_index", "size", "start", "end"), &FontData::render_range);
	ClassDB::bind_method(D_METHOD("render_glyph", "cache_index", "size", "index"), &FontData::render_glyph);

	ClassDB::bind_method(D_METHOD("get_cache_rid", "cache_index"), &FontData::get_cache_rid);

	ClassDB::bind_method(D_METHOD("is_language_supported", "language"), &FontData::is_language_supported);
	ClassDB::bind_method(D_METHOD("set_language_support_override", "language", "supported"), &FontData::set_language_support_override);
	ClassDB::bind_method(D_METHOD("get_language_support_override", "language"), &FontData::get_language_support_override);
	ClassDB::bind_method(D_METHOD("remove_language_support_override", "language"), &FontData::remove_language_support_override);
	ClassDB::bind_method(D_METHOD("get_language_support_overrides"), &FontData::get_language_support_overrides);

	ClassDB::bind_method(D_METHOD("is_script_supported", "script"), &FontData::is_script_supported);
	ClassDB::bind_method(D_METHOD("set_script_support_override", "script", "supported"), &FontData::set_script_support_override);
	ClassDB::bind_method(D_METHOD("get_script_support_override", "script"), &FontData::get_script_support_override);
	ClassDB::bind_method(D_METHOD("remove_script_support_override", "script"), &FontData::remove_script_support_override);
	ClassDB::bind_method(D_METHOD("get_script_support_overrides"), &FontData::get_script_support_overrides);

	ClassDB::bind_method(D_METHOD("has_char", "char"), &FontData::has_char);
	ClassDB::bind_method(D_METHOD("get_supported_chars"), &FontData::get_supported_chars);

	ClassDB::bind_method(D_METHOD("get_glyph_index", "size", "char", "variation_selector"), &FontData::get_glyph_index);

	ClassDB::bind_method(D_METHOD("get_supported_feature_list"), &FontData::get_supported_feature_list);
	ClassDB::bind_method(D_METHOD("get_supported_variation_list"), &FontData::get_supported_variation_list);
}

bool FontData::_set(const StringName &p_name, const Variant &p_value) {
	Vector<String> tokens = p_name.operator String().split("/");
	if (tokens.size() == 1) {
		if (tokens[0] == "data") {
			set_data(p_value);
			return true;
		} else if (tokens[0] == "antialiased") {
			set_antialiased(p_value);
			return true;
		} else if (tokens[0] == "font_name") {
			set_font_name(p_value);
			return true;
		} else if (tokens[0] == "style_name") {
			set_font_style_name(p_value);
			return true;
		} else if (tokens[0] == "font_style") {
			set_font_style(p_value);
			return true;
		} else if (tokens[0] == "multichannel_signed_distance_field") {
			set_multichannel_signed_distance_field(p_value);
			return true;
		} else if (tokens[0] == "msdf_pixel_range") {
			set_msdf_pixel_range(p_value);
			return true;
		} else if (tokens[0] == "msdf_size") {
			set_msdf_size(p_value);
			return true;
		} else if (tokens[0] == "fixed_size") {
			set_fixed_size(p_value);
			return true;
		} else if (tokens[0] == "hinting") {
			set_hinting((TextServer::Hinting)p_value.operator int());
			return true;
		} else if (tokens[0] == "force_autohinter") {
			set_force_autohinter(p_value);
			return true;
		} else if (tokens[0] == "oversampling") {
			set_oversampling(p_value);
			return true;
		}
	} else if (tokens.size() == 2 && tokens[0] == "language_support_override") {
		String lang = tokens[1];
		set_language_support_override(lang, p_value);
		return true;
	} else if (tokens.size() == 2 && tokens[0] == "script_support_override") {
		String script = tokens[1];
		set_script_support_override(script, p_value);
		return true;
	} else if (tokens.size() >= 3 && tokens[0] == "cache") {
		int cache_index = tokens[1].to_int();
		if (tokens.size() == 3 && tokens[2] == "variation_coordinates") {
			set_variation_coordinates(cache_index, p_value);
			return true;
		}
		if (tokens.size() >= 5) {
			Vector2i sz = Vector2i(tokens[2].to_int(), tokens[3].to_int());
			if (tokens[4] == "ascent") {
				set_ascent(cache_index, sz.x, p_value);
				return true;
			} else if (tokens[4] == "descent") {
				set_descent(cache_index, sz.x, p_value);
				return true;
			} else if (tokens[4] == "underline_position") {
				set_underline_position(cache_index, sz.x, p_value);
				return true;
			} else if (tokens[4] == "underline_thickness") {
				set_underline_thickness(cache_index, sz.x, p_value);
				return true;
			} else if (tokens[4] == "scale") {
				set_scale(cache_index, sz.x, p_value);
				return true;
			} else if (tokens[4] == "spacing_glyph") {
				set_spacing(cache_index, sz.x, TextServer::SPACING_GLYPH, p_value);
				return true;
			} else if (tokens[4] == "spacing_space") {
				set_spacing(cache_index, sz.x, TextServer::SPACING_SPACE, p_value);
				return true;
			} else if (tokens.size() == 7 && tokens[4] == "textures") {
				int texture_index = tokens[5].to_int();
				if (tokens[6] == "image") {
					set_texture_image(cache_index, sz, texture_index, p_value);
					return true;
				} else if (tokens[6] == "offsets") {
					set_texture_offsets(cache_index, sz, texture_index, p_value);
					return true;
				}
			} else if (tokens.size() == 7 && tokens[4] == "glyphs") {
				int32_t glyph_index = tokens[5].to_int();
				if (tokens[6] == "advance") {
					set_glyph_advance(cache_index, sz.x, glyph_index, p_value);
					return true;
				} else if (tokens[6] == "offset") {
					set_glyph_offset(cache_index, sz, glyph_index, p_value);
					return true;
				} else if (tokens[6] == "size") {
					set_glyph_size(cache_index, sz, glyph_index, p_value);
					return true;
				} else if (tokens[6] == "uv_rect") {
					set_glyph_uv_rect(cache_index, sz, glyph_index, p_value);
					return true;
				} else if (tokens[6] == "texture_idx") {
					set_glyph_texture_idx(cache_index, sz, glyph_index, p_value);
					return true;
				}
			} else if (tokens.size() == 7 && tokens[4] == "kerning_overrides") {
				Vector2i gp = Vector2i(tokens[5].to_int(), tokens[6].to_int());
				set_kerning(cache_index, sz.x, gp, p_value);
				return true;
			}
		}
	}
	return false;
}

bool FontData::_get(const StringName &p_name, Variant &r_ret) const {
	Vector<String> tokens = p_name.operator String().split("/");
	if (tokens.size() == 1) {
		if (tokens[0] == "data") {
			r_ret = get_data();
			return true;
		} else if (tokens[0] == "antialiased") {
			r_ret = is_antialiased();
			return true;
		} else if (tokens[0] == "font_name") {
			r_ret = get_font_name();
			return true;
		} else if (tokens[0] == "style_name") {
			r_ret = get_font_style_name();
			return true;
		} else if (tokens[0] == "font_style") {
			r_ret = get_font_style();
			return true;
		} else if (tokens[0] == "multichannel_signed_distance_field") {
			r_ret = is_multichannel_signed_distance_field();
			return true;
		} else if (tokens[0] == "msdf_pixel_range") {
			r_ret = get_msdf_pixel_range();
			return true;
		} else if (tokens[0] == "msdf_size") {
			r_ret = get_msdf_size();
			return true;
		} else if (tokens[0] == "fixed_size") {
			r_ret = get_fixed_size();
			return true;
		} else if (tokens[0] == "hinting") {
			r_ret = get_hinting();
			return true;
		} else if (tokens[0] == "force_autohinter") {
			r_ret = is_force_autohinter();
			return true;
		} else if (tokens[0] == "oversampling") {
			r_ret = get_oversampling();
			return true;
		}
	} else if (tokens.size() == 2 && tokens[0] == "language_support_override") {
		String lang = tokens[1];
		r_ret = get_language_support_override(lang);
		return true;
	} else if (tokens.size() == 2 && tokens[0] == "script_support_override") {
		String script = tokens[1];
		r_ret = get_script_support_override(script);
		return true;
	} else if (tokens.size() >= 3 && tokens[0] == "cache") {
		int cache_index = tokens[1].to_int();
		if (tokens.size() == 3 && tokens[2] == "variation_coordinates") {
			r_ret = get_variation_coordinates(cache_index);
			return true;
		}
		if (tokens.size() >= 5) {
			Vector2i sz = Vector2i(tokens[2].to_int(), tokens[3].to_int());
			if (tokens[4] == "ascent") {
				r_ret = get_ascent(cache_index, sz.x);
				return true;
			} else if (tokens[4] == "descent") {
				r_ret = get_descent(cache_index, sz.x);
				return true;
			} else if (tokens[4] == "underline_position") {
				r_ret = get_underline_position(cache_index, sz.x);
				return true;
			} else if (tokens[4] == "underline_thickness") {
				r_ret = get_underline_thickness(cache_index, sz.x);
				return true;
			} else if (tokens[4] == "scale") {
				r_ret = get_scale(cache_index, sz.x);
				return true;
			} else if (tokens[4] == "spacing_glyph") {
				r_ret = get_spacing(cache_index, sz.x, TextServer::SPACING_GLYPH);
				return true;
			} else if (tokens[4] == "spacing_space") {
				r_ret = get_spacing(cache_index, sz.x, TextServer::SPACING_SPACE);
				return true;
			} else if (tokens.size() == 7 && tokens[4] == "textures") {
				int texture_index = tokens[5].to_int();
				if (tokens[6] == "image") {
					r_ret = get_texture_image(cache_index, sz, texture_index);
					return true;
				} else if (tokens[6] == "offsets") {
					r_ret = get_texture_offsets(cache_index, sz, texture_index);
					return true;
				}
			} else if (tokens.size() == 7 && tokens[4] == "glyphs") {
				int32_t glyph_index = tokens[5].to_int();
				if (tokens[6] == "advance") {
					r_ret = get_glyph_advance(cache_index, sz.x, glyph_index);
					return true;
				} else if (tokens[6] == "offset") {
					r_ret = get_glyph_offset(cache_index, sz, glyph_index);
					return true;
				} else if (tokens[6] == "size") {
					r_ret = get_glyph_size(cache_index, sz, glyph_index);
					return true;
				} else if (tokens[6] == "uv_rect") {
					r_ret = get_glyph_uv_rect(cache_index, sz, glyph_index);
					return true;
				} else if (tokens[6] == "texture_idx") {
					r_ret = get_glyph_texture_idx(cache_index, sz, glyph_index);
					return true;
				}
			} else if (tokens.size() == 7 && tokens[4] == "kerning_overrides") {
				Vector2i gp = Vector2i(tokens[5].to_int(), tokens[6].to_int());
				r_ret = get_kerning(cache_index, sz.x, gp);
				return true;
			}
		}
	}
	return false;
}

void FontData::_get_property_list(List<PropertyInfo> *p_list) const {
	p_list->push_back(PropertyInfo(Variant::PACKED_BYTE_ARRAY, "data", PROPERTY_HINT_NONE, "", PROPERTY_USAGE_STORAGE));

	p_list->push_back(PropertyInfo(Variant::STRING, "font_name", PROPERTY_HINT_NONE, "", PROPERTY_USAGE_STORAGE));
	p_list->push_back(PropertyInfo(Variant::STRING, "style_name", PROPERTY_HINT_NONE, "", PROPERTY_USAGE_STORAGE));
	p_list->push_back(PropertyInfo(Variant::INT, "font_style", PROPERTY_HINT_NONE, "", PROPERTY_USAGE_STORAGE));
	p_list->push_back(PropertyInfo(Variant::BOOL, "antialiased", PROPERTY_HINT_NONE, "", PROPERTY_USAGE_STORAGE));
	p_list->push_back(PropertyInfo(Variant::BOOL, "multichannel_signed_distance_field", PROPERTY_HINT_NONE, "", PROPERTY_USAGE_STORAGE));
	p_list->push_back(PropertyInfo(Variant::INT, "msdf_pixel_range", PROPERTY_HINT_NONE, "", PROPERTY_USAGE_STORAGE));
	p_list->push_back(PropertyInfo(Variant::INT, "msdf_size", PROPERTY_HINT_NONE, "", PROPERTY_USAGE_STORAGE));
	p_list->push_back(PropertyInfo(Variant::INT, "fixed_size", PROPERTY_HINT_NONE, "", PROPERTY_USAGE_STORAGE));
	p_list->push_back(PropertyInfo(Variant::INT, "hinting", PROPERTY_HINT_ENUM, "None,Light,Normal", PROPERTY_USAGE_STORAGE));
	p_list->push_back(PropertyInfo(Variant::BOOL, "force_autohinter", PROPERTY_HINT_NONE, "", PROPERTY_USAGE_STORAGE));
	p_list->push_back(PropertyInfo(Variant::FLOAT, "oversampling", PROPERTY_HINT_NONE, "", PROPERTY_USAGE_STORAGE));

	Vector<String> lang_over = get_language_support_overrides();
	for (int i = 0; i < lang_over.size(); i++) {
		p_list->push_back(PropertyInfo(Variant::BOOL, "language_support_override/" + lang_over[i], PROPERTY_HINT_NONE, "", PROPERTY_USAGE_STORAGE));
	}
	Vector<String> scr_over = get_script_support_overrides();
	for (int i = 0; i < scr_over.size(); i++) {
		p_list->push_back(PropertyInfo(Variant::BOOL, "script_support_override/" + scr_over[i], PROPERTY_HINT_NONE, "", PROPERTY_USAGE_STORAGE));
	}
	for (int i = 0; i < cache.size(); i++) {
		String prefix = "cache/" + itos(i) + "/";
		Array sizes = get_size_cache_list(i);
		p_list->push_back(PropertyInfo(Variant::DICTIONARY, prefix + "variation_coordinates", PROPERTY_HINT_NONE, "", PROPERTY_USAGE_STORAGE));
		for (int j = 0; j < sizes.size(); j++) {
			Vector2i sz = sizes[j];
			String prefix_sz = prefix + itos(sz.x) + "/" + itos(sz.y) + "/";
			if (sz.y == 0) {
				p_list->push_back(PropertyInfo(Variant::FLOAT, prefix_sz + "ascent", PROPERTY_HINT_NONE, "", PROPERTY_USAGE_STORAGE));
				p_list->push_back(PropertyInfo(Variant::FLOAT, prefix_sz + "descent", PROPERTY_HINT_NONE, "", PROPERTY_USAGE_STORAGE));
				p_list->push_back(PropertyInfo(Variant::FLOAT, prefix_sz + "underline_position", PROPERTY_HINT_NONE, "", PROPERTY_USAGE_STORAGE));
				p_list->push_back(PropertyInfo(Variant::FLOAT, prefix_sz + "underline_thickness", PROPERTY_HINT_NONE, "", PROPERTY_USAGE_STORAGE));
				p_list->push_back(PropertyInfo(Variant::FLOAT, prefix_sz + "scale", PROPERTY_HINT_NONE, "", PROPERTY_USAGE_STORAGE));
				p_list->push_back(PropertyInfo(Variant::BOOL, prefix_sz + "spacing_glyph", PROPERTY_HINT_NONE, "", PROPERTY_USAGE_STORAGE));
				p_list->push_back(PropertyInfo(Variant::BOOL, prefix_sz + "spacing_space", PROPERTY_HINT_NONE, "", PROPERTY_USAGE_STORAGE));
			}

			int tx_cnt = get_texture_count(i, sz);
			for (int k = 0; k < tx_cnt; k++) {
				p_list->push_back(PropertyInfo(Variant::PACKED_INT32_ARRAY, prefix_sz + "textures/" + itos(k) + "/offsets", PROPERTY_HINT_NONE, "", PROPERTY_USAGE_STORAGE));
				p_list->push_back(PropertyInfo(Variant::OBJECT, prefix_sz + "textures/" + itos(k) + "/image", PROPERTY_HINT_RESOURCE_TYPE, "Image", PROPERTY_USAGE_STORAGE | PROPERTY_USAGE_RESOURCE_NOT_PERSISTENT));
			}
			Array glyphs = get_glyph_list(i, sz);
			for (int k = 0; k < glyphs.size(); k++) {
				const int32_t &gl = glyphs[k];
				if (sz.y == 0) {
					p_list->push_back(PropertyInfo(Variant::VECTOR2, prefix_sz + "glyphs/" + itos(gl) + "/advance", PROPERTY_HINT_NONE, "", PROPERTY_USAGE_STORAGE));
				}
				p_list->push_back(PropertyInfo(Variant::VECTOR2, prefix_sz + "glyphs/" + itos(gl) + "/offset", PROPERTY_HINT_NONE, "", PROPERTY_USAGE_STORAGE));
				p_list->push_back(PropertyInfo(Variant::VECTOR2, prefix_sz + "glyphs/" + itos(gl) + "/size", PROPERTY_HINT_NONE, "", PROPERTY_USAGE_STORAGE));
				p_list->push_back(PropertyInfo(Variant::RECT2, prefix_sz + "glyphs/" + itos(gl) + "/uv_rect", PROPERTY_HINT_NONE, "", PROPERTY_USAGE_STORAGE));
				p_list->push_back(PropertyInfo(Variant::INT, prefix_sz + "glyphs/" + itos(gl) + "/texture_idx", PROPERTY_HINT_NONE, "", PROPERTY_USAGE_STORAGE));
			}
			if (sz.y == 0) {
				Array kerning_map = get_kerning_list(i, sz.x);
				for (int k = 0; k < kerning_map.size(); k++) {
					const Vector2i &gl_pair = kerning_map[k];
					p_list->push_back(PropertyInfo(Variant::VECTOR2, prefix_sz + "kerning_overrides/" + itos(gl_pair.x) + "/" + itos(gl_pair.y), PROPERTY_HINT_NONE, "", PROPERTY_USAGE_STORAGE));
				}
			}
		}
	}
}

void FontData::reset_state() {
	_clear_cache();
	data.clear();
	data_ptr = nullptr;
	data_size = 0;
	cache.clear();

	antialiased = true;
	msdf = false;
	force_autohinter = false;
	hinting = TextServer::HINTING_LIGHT;
	msdf_pixel_range = 14;
	msdf_size = 128;
	fixed_size = 0;
	oversampling = 0.f;
}

void FontData::_convert_packed_8bit(Ref<Image> &p_source, int p_page, int p_sz) {
	int w = p_source->get_width();
	int h = p_source->get_height();

	PackedByteArray imgdata = p_source->get_data();
	const uint8_t *r = imgdata.ptr();

	PackedByteArray imgdata_r;
	imgdata_r.resize(w * h * 2);
	uint8_t *wr = imgdata_r.ptrw();

	PackedByteArray imgdata_g;
	imgdata_g.resize(w * h * 2);
	uint8_t *wg = imgdata_g.ptrw();

	PackedByteArray imgdata_b;
	imgdata_b.resize(w * h * 2);
	uint8_t *wb = imgdata_b.ptrw();

	PackedByteArray imgdata_a;
	imgdata_a.resize(w * h * 2);
	uint8_t *wa = imgdata_a.ptrw();

	for (int i = 0; i < h; i++) {
		for (int j = 0; j < w; j++) {
			int ofs_src = (i * w + j) * 4;
			int ofs_dst = (i * w + j) * 2;
			wr[ofs_dst + 0] = 255;
			wr[ofs_dst + 1] = r[ofs_src + 0];
			wg[ofs_dst + 0] = 255;
			wg[ofs_dst + 1] = r[ofs_src + 1];
			wb[ofs_dst + 0] = 255;
			wb[ofs_dst + 1] = r[ofs_src + 2];
			wa[ofs_dst + 0] = 255;
			wa[ofs_dst + 1] = r[ofs_src + 3];
		}
	}
	Ref<Image> img_r = memnew(Image(w, h, 0, Image::FORMAT_LA8, imgdata_r));
	set_texture_image(0, Vector2i(p_sz, 0), p_page * 4 + 0, img_r);
	Ref<Image> img_g = memnew(Image(w, h, 0, Image::FORMAT_LA8, imgdata_g));
	set_texture_image(0, Vector2i(p_sz, 0), p_page * 4 + 1, img_g);
	Ref<Image> img_b = memnew(Image(w, h, 0, Image::FORMAT_LA8, imgdata_b));
	set_texture_image(0, Vector2i(p_sz, 0), p_page * 4 + 2, img_b);
	Ref<Image> img_a = memnew(Image(w, h, 0, Image::FORMAT_LA8, imgdata_a));
	set_texture_image(0, Vector2i(p_sz, 0), p_page * 4 + 3, img_a);
}

void FontData::_convert_packed_4bit(Ref<Image> &p_source, int p_page, int p_sz) {
	int w = p_source->get_width();
	int h = p_source->get_height();

	PackedByteArray imgdata = p_source->get_data();
	const uint8_t *r = imgdata.ptr();

	PackedByteArray imgdata_r;
	imgdata_r.resize(w * h * 2);
	uint8_t *wr = imgdata_r.ptrw();

	PackedByteArray imgdata_g;
	imgdata_g.resize(w * h * 2);
	uint8_t *wg = imgdata_g.ptrw();

	PackedByteArray imgdata_b;
	imgdata_b.resize(w * h * 2);
	uint8_t *wb = imgdata_b.ptrw();

	PackedByteArray imgdata_a;
	imgdata_a.resize(w * h * 2);
	uint8_t *wa = imgdata_a.ptrw();

	PackedByteArray imgdata_ro;
	imgdata_ro.resize(w * h * 2);
	uint8_t *wro = imgdata_ro.ptrw();

	PackedByteArray imgdata_go;
	imgdata_go.resize(w * h * 2);
	uint8_t *wgo = imgdata_go.ptrw();

	PackedByteArray imgdata_bo;
	imgdata_bo.resize(w * h * 2);
	uint8_t *wbo = imgdata_bo.ptrw();

	PackedByteArray imgdata_ao;
	imgdata_ao.resize(w * h * 2);
	uint8_t *wao = imgdata_ao.ptrw();

	for (int i = 0; i < h; i++) {
		for (int j = 0; j < w; j++) {
			int ofs_src = (i * w + j) * 4;
			int ofs_dst = (i * w + j) * 2;
			wr[ofs_dst + 0] = 255;
			wro[ofs_dst + 0] = 255;
			if (r[ofs_src + 0] > 0x0F) {
				wr[ofs_dst + 1] = (r[ofs_src + 0] - 0x0F) * 2;
				wro[ofs_dst + 1] = 0;
			} else {
				wr[ofs_dst + 1] = 0;
				wro[ofs_dst + 1] = r[ofs_src + 0] * 2;
			}
			wg[ofs_dst + 0] = 255;
			wgo[ofs_dst + 0] = 255;
			if (r[ofs_src + 1] > 0x0F) {
				wg[ofs_dst + 1] = (r[ofs_src + 1] - 0x0F) * 2;
				wgo[ofs_dst + 1] = 0;
			} else {
				wg[ofs_dst + 1] = 0;
				wgo[ofs_dst + 1] = r[ofs_src + 1] * 2;
			}
			wb[ofs_dst + 0] = 255;
			wbo[ofs_dst + 0] = 255;
			if (r[ofs_src + 2] > 0x0F) {
				wb[ofs_dst + 1] = (r[ofs_src + 2] - 0x0F) * 2;
				wbo[ofs_dst + 1] = 0;
			} else {
				wb[ofs_dst + 1] = 0;
				wbo[ofs_dst + 1] = r[ofs_src + 2] * 2;
			}
			wa[ofs_dst + 0] = 255;
			wao[ofs_dst + 0] = 255;
			if (r[ofs_src + 3] > 0x0F) {
				wa[ofs_dst + 1] = (r[ofs_src + 3] - 0x0F) * 2;
				wao[ofs_dst + 1] = 0;
			} else {
				wa[ofs_dst + 1] = 0;
				wao[ofs_dst + 1] = r[ofs_src + 3] * 2;
			}
		}
	}
	Ref<Image> img_r = memnew(Image(w, h, 0, Image::FORMAT_LA8, imgdata_r));
	set_texture_image(0, Vector2i(p_sz, 0), p_page * 4 + 0, img_r);
	Ref<Image> img_g = memnew(Image(w, h, 0, Image::FORMAT_LA8, imgdata_g));
	set_texture_image(0, Vector2i(p_sz, 0), p_page * 4 + 1, img_g);
	Ref<Image> img_b = memnew(Image(w, h, 0, Image::FORMAT_LA8, imgdata_b));
	set_texture_image(0, Vector2i(p_sz, 0), p_page * 4 + 2, img_b);
	Ref<Image> img_a = memnew(Image(w, h, 0, Image::FORMAT_LA8, imgdata_a));
	set_texture_image(0, Vector2i(p_sz, 0), p_page * 4 + 3, img_a);

	Ref<Image> img_ro = memnew(Image(w, h, 0, Image::FORMAT_LA8, imgdata_ro));
	set_texture_image(0, Vector2i(p_sz, 1), p_page * 4 + 0, img_ro);
	Ref<Image> img_go = memnew(Image(w, h, 0, Image::FORMAT_LA8, imgdata_go));
	set_texture_image(0, Vector2i(p_sz, 1), p_page * 4 + 1, img_go);
	Ref<Image> img_bo = memnew(Image(w, h, 0, Image::FORMAT_LA8, imgdata_bo));
	set_texture_image(0, Vector2i(p_sz, 1), p_page * 4 + 2, img_bo);
	Ref<Image> img_ao = memnew(Image(w, h, 0, Image::FORMAT_LA8, imgdata_ao));
	set_texture_image(0, Vector2i(p_sz, 1), p_page * 4 + 3, img_ao);
}

void FontData::_convert_rgba_4bit(Ref<Image> &p_source, int p_page, int p_sz) {
	int w = p_source->get_width();
	int h = p_source->get_height();

	PackedByteArray imgdata = p_source->get_data();
	const uint8_t *r = imgdata.ptr();

	PackedByteArray imgdata_g;
	imgdata_g.resize(w * h * 4);
	uint8_t *wg = imgdata_g.ptrw();

	PackedByteArray imgdata_o;
	imgdata_o.resize(w * h * 4);
	uint8_t *wo = imgdata_o.ptrw();

	for (int i = 0; i < h; i++) {
		for (int j = 0; j < w; j++) {
			int ofs = (i * w + j) * 4;

			if (r[ofs + 0] > 0x7F) {
				wg[ofs + 0] = r[ofs + 0];
				wo[ofs + 0] = 0;
			} else {
				wg[ofs + 0] = 0;
				wo[ofs + 0] = r[ofs + 0] * 2;
			}
			if (r[ofs + 1] > 0x7F) {
				wg[ofs + 1] = r[ofs + 1];
				wo[ofs + 1] = 0;
			} else {
				wg[ofs + 1] = 0;
				wo[ofs + 1] = r[ofs + 1] * 2;
			}
			if (r[ofs + 2] > 0x7F) {
				wg[ofs + 2] = r[ofs + 2];
				wo[ofs + 2] = 0;
			} else {
				wg[ofs + 2] = 0;
				wo[ofs + 2] = r[ofs + 2] * 2;
			}
			if (r[ofs + 3] > 0x7F) {
				wg[ofs + 3] = r[ofs + 3];
				wo[ofs + 3] = 0;
			} else {
				wg[ofs + 3] = 0;
				wo[ofs + 3] = r[ofs + 3] * 2;
			}
		}
	}
	Ref<Image> img_g = memnew(Image(w, h, 0, Image::FORMAT_RGBA8, imgdata_g));
	set_texture_image(0, Vector2i(p_sz, 0), p_page, img_g);

	Ref<Image> img_o = memnew(Image(w, h, 0, Image::FORMAT_RGBA8, imgdata_o));
	set_texture_image(0, Vector2i(p_sz, 1), p_page, img_o);
}

void FontData::_convert_mono_8bit(Ref<Image> &p_source, int p_page, int p_ch, int p_sz, int p_ol) {
	int w = p_source->get_width();
	int h = p_source->get_height();

	PackedByteArray imgdata = p_source->get_data();
	const uint8_t *r = imgdata.ptr();

	int size = 4;
	if (p_source->get_format() == Image::FORMAT_L8) {
		size = 1;
		p_ch = 0;
	}

	PackedByteArray imgdata_g;
	imgdata_g.resize(w * h * 2);
	uint8_t *wg = imgdata_g.ptrw();

	for (int i = 0; i < h; i++) {
		for (int j = 0; j < w; j++) {
			int ofs_src = (i * w + j) * size;
			int ofs_dst = (i * w + j) * 2;
			wg[ofs_dst + 0] = 255;
			wg[ofs_dst + 1] = r[ofs_src + p_ch];
		}
	}
	Ref<Image> img_g = memnew(Image(w, h, 0, Image::FORMAT_LA8, imgdata_g));
	set_texture_image(0, Vector2i(p_sz, p_ol), p_page, img_g);
}

void FontData::_convert_mono_4bit(Ref<Image> &p_source, int p_page, int p_ch, int p_sz, int p_ol) {
	int w = p_source->get_width();
	int h = p_source->get_height();

	PackedByteArray imgdata = p_source->get_data();
	const uint8_t *r = imgdata.ptr();

	int size = 4;
	if (p_source->get_format() == Image::FORMAT_L8) {
		size = 1;
		p_ch = 0;
	}

	PackedByteArray imgdata_g;
	imgdata_g.resize(w * h * 2);
	uint8_t *wg = imgdata_g.ptrw();

	PackedByteArray imgdata_o;
	imgdata_o.resize(w * h * 2);
	uint8_t *wo = imgdata_o.ptrw();

	for (int i = 0; i < h; i++) {
		for (int j = 0; j < w; j++) {
			int ofs_src = (i * w + j) * size;
			int ofs_dst = (i * w + j) * 2;
			wg[ofs_dst + 0] = 255;
			wo[ofs_dst + 0] = 255;
			if (r[ofs_src + p_ch] > 0x7F) {
				wg[ofs_dst + 1] = r[ofs_src + p_ch];
				wo[ofs_dst + 1] = 0;
			} else {
				wg[ofs_dst + 1] = 0;
				wo[ofs_dst + 1] = r[ofs_src + p_ch] * 2;
			}
		}
	}
	Ref<Image> img_g = memnew(Image(w, h, 0, Image::FORMAT_LA8, imgdata_g));
	set_texture_image(0, Vector2i(p_sz, 0), p_page, img_g);

	Ref<Image> img_o = memnew(Image(w, h, 0, Image::FORMAT_LA8, imgdata_o));
	set_texture_image(0, Vector2i(p_sz, p_ol), p_page, img_o);
}

/*************************************************************************/

Error FontData::load_bitmap_font(const String &p_path) {
	reset_state();

	antialiased = false;
	msdf = false;
	force_autohinter = false;
	hinting = TextServer::HINTING_NONE;
	oversampling = 1.0f;

	FileAccessRef f = FileAccess::open(p_path, FileAccess::READ);
	if (f == nullptr) {
		ERR_FAIL_V_MSG(ERR_CANT_CREATE, TTR("Cannot open font from file ") + "\"" + p_path + "\".");
	}

	int base_size = 16;
	int height = 0;
	int ascent = 0;
	int outline = 0;
	uint32_t st_flags = 0;
	String font_name;

	bool packed = false;
	uint8_t ch[4] = { 0, 0, 0, 0 }; // RGBA
	int first_gl_ch = -1;
	int first_ol_ch = -1;
	int first_cm_ch = -1;

	unsigned char magic[4];
	f->get_buffer((unsigned char *)&magic, 4);
	if (magic[0] == 'B' && magic[1] == 'M' && magic[2] == 'F') {
		// Binary BMFont file.
		ERR_FAIL_COND_V_MSG(magic[3] != 3, ERR_CANT_CREATE, vformat(TTR("Version %d of BMFont is not supported."), (int)magic[3]));

		uint8_t block_type = f->get_8();
		uint32_t block_size = f->get_32();
		while (!f->eof_reached()) {
			uint64_t off = f->get_position();
			switch (block_type) {
				case 1: /* info */ {
					ERR_FAIL_COND_V_MSG(block_size < 15, ERR_CANT_CREATE, TTR("Invalid BMFont info block size."));
					base_size = f->get_16();
					uint8_t flags = f->get_8();
					ERR_FAIL_COND_V_MSG(flags & 0x02, ERR_CANT_CREATE, TTR("Non-unicode version of BMFont is not supported."));
					if (flags & (1 << 3)) {
						st_flags |= TextServer::FONT_BOLD;
					}
					if (flags & (1 << 2)) {
						st_flags |= TextServer::FONT_ITALIC;
					}
					f->get_8(); // non-unicode charset, skip
					f->get_16(); // stretch_h, skip
					f->get_8(); // aa, skip
					f->get_32(); // padding, skip
					f->get_16(); // spacing, skip
					outline = f->get_8();
					// font name
					PackedByteArray name_data;
					name_data.resize(block_size - 14);
					f->get_buffer(name_data.ptrw(), block_size - 14);
					font_name = String::utf8((const char *)name_data.ptr(), block_size - 14);
					set_fixed_size(base_size);
				} break;
				case 2: /* common */ {
					ERR_FAIL_COND_V_MSG(block_size != 15, ERR_CANT_CREATE, TTR("Invalid BMFont common block size."));
					height = f->get_16();
					ascent = f->get_16();
					f->get_32(); // scale, skip
					f->get_16(); // pages, skip
					uint8_t flags = f->get_8();
					packed = (flags & 0x01);
					ch[3] = f->get_8();
					ch[0] = f->get_8();
					ch[1] = f->get_8();
					ch[2] = f->get_8();
					for (int i = 0; i < 4; i++) {
						if (ch[i] == 0 && first_gl_ch == -1) {
							first_gl_ch = i;
						}
						if (ch[i] == 1 && first_ol_ch == -1) {
							first_ol_ch = i;
						}
						if (ch[i] == 2 && first_cm_ch == -1) {
							first_cm_ch = i;
						}
					}
				} break;
				case 3: /* pages */ {
					int page = 0;
					CharString cs;
					char32_t c = f->get_8();
					while (!f->eof_reached() && f->get_position() <= off + block_size) {
						if (c == '\0') {
							String base_dir = p_path.get_base_dir();
							String file = base_dir.plus_file(String::utf8(cs.ptr(), cs.length()));
							if (RenderingServer::get_singleton() != nullptr) {
								Ref<Image> img;
								img.instantiate();
								Error err = ImageLoader::load_image(file, img);
								ERR_FAIL_COND_V_MSG(err != OK, ERR_FILE_CANT_READ, TTR("Can't load font texture: ") + "\"" + file + "\".");

								if (packed) {
									if (ch[3] == 0) { // 4 x 8 bit monochrome, no outline
										outline = 0;
										ERR_FAIL_COND_V_MSG(img->get_format() != Image::FORMAT_RGBA8, ERR_FILE_CANT_READ, TTR("Unsupported BMFont texture format."));
										_convert_packed_8bit(img, page, base_size);
									} else if ((ch[3] == 2) && (outline > 0)) { // 4 x 4 bit monochrome, gl + outline
										ERR_FAIL_COND_V_MSG(img->get_format() != Image::FORMAT_RGBA8, ERR_FILE_CANT_READ, TTR("Unsupported BMFont texture format."));
										_convert_packed_4bit(img, page, base_size);
									} else {
										ERR_FAIL_V_MSG(ERR_CANT_CREATE, TTR("Unsupported BMFont texture format."));
									}
								} else {
									if ((ch[0] == 0) && (ch[1] == 0) && (ch[2] == 0) && (ch[3] == 0)) { // RGBA8 color, no outline
										outline = 0;
										ERR_FAIL_COND_V_MSG(img->get_format() != Image::FORMAT_RGBA8, ERR_FILE_CANT_READ, TTR("Unsupported BMFont texture format."));
										set_texture_image(0, Vector2i(base_size, 0), page, img);
									} else if ((ch[0] == 2) && (ch[1] == 2) && (ch[2] == 2) && (ch[3] == 2) && (outline > 0)) { // RGBA4 color, gl + outline
										ERR_FAIL_COND_V_MSG(img->get_format() != Image::FORMAT_RGBA8, ERR_FILE_CANT_READ, TTR("Unsupported BMFont texture format."));
										_convert_rgba_4bit(img, page, base_size);
									} else if ((first_gl_ch >= 0) && (first_ol_ch >= 0) && (outline > 0)) { // 1 x 8 bit monochrome, gl + outline
										ERR_FAIL_COND_V_MSG(img->get_format() != Image::FORMAT_RGBA8 && img->get_format() != Image::FORMAT_L8, ERR_FILE_CANT_READ, TTR("Unsupported BMFont texture format."));
										_convert_mono_8bit(img, page, first_gl_ch, base_size, 0);
										_convert_mono_8bit(img, page, first_ol_ch, base_size, 1);
									} else if ((first_cm_ch >= 0) && (outline > 0)) { // 1 x 4 bit monochrome, gl + outline
										ERR_FAIL_COND_V_MSG(img->get_format() != Image::FORMAT_RGBA8 && img->get_format() != Image::FORMAT_L8, ERR_FILE_CANT_READ, TTR("Unsupported BMFont texture format."));
										_convert_mono_4bit(img, page, first_cm_ch, base_size, 1);
									} else if (first_gl_ch >= 0) { // 1 x 8 bit monochrome, no outline
										outline = 0;
										ERR_FAIL_COND_V_MSG(img->get_format() != Image::FORMAT_RGBA8 && img->get_format() != Image::FORMAT_L8, ERR_FILE_CANT_READ, TTR("Unsupported BMFont texture format."));
										_convert_mono_8bit(img, page, first_gl_ch, base_size, 0);
									} else {
										ERR_FAIL_V_MSG(ERR_CANT_CREATE, TTR("Unsupported BMFont texture format."));
									}
								}
							}
							page++;
							cs = "";
						} else {
							cs += c;
						}
						c = f->get_8();
					}
				} break;
				case 4: /* chars */ {
					int char_count = block_size / 20;
					for (int i = 0; i < char_count; i++) {
						Vector2 advance;
						Vector2 size;
						Vector2 offset;
						Rect2 uv_rect;

						char32_t idx = f->get_32();
						uv_rect.position.x = (int16_t)f->get_16();
						uv_rect.position.y = (int16_t)f->get_16();
						uv_rect.size.width = (int16_t)f->get_16();
						size.width = uv_rect.size.width;
						uv_rect.size.height = (int16_t)f->get_16();
						size.height = uv_rect.size.height;
						offset.x = (int16_t)f->get_16();
						offset.y = (int16_t)f->get_16() - ascent;
						advance.x = (int16_t)f->get_16();
						if (advance.x < 0) {
							advance.x = size.width + 1;
						}

						int texture_idx = f->get_8();
						uint8_t channel = f->get_8();

						ERR_FAIL_COND_V_MSG(!packed && channel != 15, ERR_CANT_CREATE, TTR("Invalid glyph channel."));
						int ch_off = 0;
						switch (channel) {
							case 1:
								ch_off = 2;
								break; // B
							case 2:
								ch_off = 1;
								break; // G
							case 4:
								ch_off = 0;
								break; // R
							case 8:
								ch_off = 3;
								break; // A
							default:
								ch_off = 0;
								break;
						}
						set_glyph_advance(0, base_size, idx, advance);
						set_glyph_offset(0, Vector2i(base_size, 0), idx, offset);
						set_glyph_size(0, Vector2i(base_size, 0), idx, size);
						set_glyph_uv_rect(0, Vector2i(base_size, 0), idx, uv_rect);
						set_glyph_texture_idx(0, Vector2i(base_size, 0), idx, texture_idx * (packed ? 4 : 1) + ch_off);
						if (outline > 0) {
							set_glyph_offset(0, Vector2i(base_size, 1), idx, offset);
							set_glyph_size(0, Vector2i(base_size, 1), idx, size);
							set_glyph_uv_rect(0, Vector2i(base_size, 1), idx, uv_rect);
							set_glyph_texture_idx(0, Vector2i(base_size, 1), idx, texture_idx * (packed ? 4 : 1) + ch_off);
						}
					}
				} break;
				case 5: /* kerning */ {
					int pair_count = block_size / 10;
					for (int i = 0; i < pair_count; i++) {
						Vector2i kpk;
						kpk.x = f->get_32();
						kpk.y = f->get_32();
						set_kerning(0, base_size, kpk, Vector2((int16_t)f->get_16(), 0));
					}
				} break;
				default: {
					ERR_FAIL_V_MSG(ERR_CANT_CREATE, TTR("Invalid BMFont block type."));
				} break;
			}
			f->seek(off + block_size);
			block_type = f->get_8();
			block_size = f->get_32();
		}

	} else {
		// Text BMFont file.
		f->seek(0);
		while (true) {
			String line = f->get_line();

			int delimiter = line.find(" ");
			String type = line.substr(0, delimiter);
			int pos = delimiter + 1;
			Map<String, String> keys;

			while (pos < line.size() && line[pos] == ' ') {
				pos++;
			}

			while (pos < line.size()) {
				int eq = line.find("=", pos);
				if (eq == -1) {
					break;
				}
				String key = line.substr(pos, eq - pos);
				int end = -1;
				String value;
				if (line[eq + 1] == '"') {
					end = line.find("\"", eq + 2);
					if (end == -1) {
						break;
					}
					value = line.substr(eq + 2, end - 1 - eq - 1);
					pos = end + 1;
				} else {
					end = line.find(" ", eq + 1);
					if (end == -1) {
						end = line.size();
					}
					value = line.substr(eq + 1, end - eq);
					pos = end;
				}

				while (pos < line.size() && line[pos] == ' ') {
					pos++;
				}

				keys[key] = value;
			}

			if (type == "info") {
				if (keys.has("size")) {
					base_size = keys["size"].to_int();
					set_fixed_size(base_size);
				}
				if (keys.has("outline")) {
					outline = keys["outline"].to_int();
				}
				if (keys.has("bold")) {
					if (keys["bold"].to_int()) {
						st_flags |= TextServer::FONT_BOLD;
					}
				}
				if (keys.has("italic")) {
					if (keys["italic"].to_int()) {
						st_flags |= TextServer::FONT_ITALIC;
					}
				}
				if (keys.has("face")) {
					font_name = keys["face"];
				}
				ERR_FAIL_COND_V_MSG((!keys.has("unicode") || keys["unicode"].to_int() != 1), ERR_CANT_CREATE, TTR("Non-unicode version of BMFont is not supported."));
			} else if (type == "common") {
				if (keys.has("lineHeight")) {
					height = keys["lineHeight"].to_int();
				}
				if (keys.has("base")) {
					ascent = keys["base"].to_int();
				}
				if (keys.has("packed")) {
					packed = (keys["packed"].to_int() == 1);
				}
				if (keys.has("alphaChnl")) {
					ch[3] = keys["alphaChnl"].to_int();
				}
				if (keys.has("redChnl")) {
					ch[0] = keys["redChnl"].to_int();
				}
				if (keys.has("greenChnl")) {
					ch[1] = keys["greenChnl"].to_int();
				}
				if (keys.has("blueChnl")) {
					ch[2] = keys["blueChnl"].to_int();
				}
				for (int i = 0; i < 4; i++) {
					if (ch[i] == 0 && first_gl_ch == -1) {
						first_gl_ch = i;
					}
					if (ch[i] == 1 && first_ol_ch == -1) {
						first_ol_ch = i;
					}
					if (ch[i] == 2 && first_cm_ch == -1) {
						first_cm_ch = i;
					}
				}
			} else if (type == "page") {
				int page = 0;
				if (keys.has("id")) {
					page = keys["id"].to_int();
				}
				if (keys.has("file")) {
					String base_dir = p_path.get_base_dir();
					String file = base_dir.plus_file(keys["file"]);
					if (RenderingServer::get_singleton() != nullptr) {
						Ref<Image> img;
						img.instantiate();
						Error err = ImageLoader::load_image(file, img);
						ERR_FAIL_COND_V_MSG(err != OK, ERR_FILE_CANT_READ, TTR("Can't load font texture: ") + "\"" + file + "\".");
						if (packed) {
							if (ch[3] == 0) { // 4 x 8 bit monochrome, no outline
								outline = 0;
								ERR_FAIL_COND_V_MSG(img->get_format() != Image::FORMAT_RGBA8, ERR_FILE_CANT_READ, TTR("Unsupported BMFont texture format."));
								_convert_packed_8bit(img, page, base_size);
							} else if ((ch[3] == 2) && (outline > 0)) { // 4 x 4 bit monochrome, gl + outline
								ERR_FAIL_COND_V_MSG(img->get_format() != Image::FORMAT_RGBA8, ERR_FILE_CANT_READ, TTR("Unsupported BMFont texture format."));
								_convert_packed_4bit(img, page, base_size);
							} else {
								ERR_FAIL_V_MSG(ERR_CANT_CREATE, TTR("Unsupported BMFont texture format."));
							}
						} else {
							if ((ch[0] == 0) && (ch[1] == 0) && (ch[2] == 0) && (ch[3] == 0)) { // RGBA8 color, no outline
								outline = 0;
								ERR_FAIL_COND_V_MSG(img->get_format() != Image::FORMAT_RGBA8, ERR_FILE_CANT_READ, TTR("Unsupported BMFont texture format."));
								set_texture_image(0, Vector2i(base_size, 0), page, img);
							} else if ((ch[0] == 2) && (ch[1] == 2) && (ch[2] == 2) && (ch[3] == 2) && (outline > 0)) { // RGBA4 color, gl + outline
								ERR_FAIL_COND_V_MSG(img->get_format() != Image::FORMAT_RGBA8, ERR_FILE_CANT_READ, TTR("Unsupported BMFont texture format."));
								_convert_rgba_4bit(img, page, base_size);
							} else if ((first_gl_ch >= 0) && (first_ol_ch >= 0) && (outline > 0)) { // 1 x 8 bit monochrome, gl + outline
								ERR_FAIL_COND_V_MSG(img->get_format() != Image::FORMAT_RGBA8 && img->get_format() != Image::FORMAT_L8, ERR_FILE_CANT_READ, TTR("Unsupported BMFont texture format."));
								_convert_mono_8bit(img, page, first_gl_ch, base_size, 0);
								_convert_mono_8bit(img, page, first_ol_ch, base_size, 1);
							} else if ((first_cm_ch >= 0) && (outline > 0)) { // 1 x 4 bit monochrome, gl + outline
								ERR_FAIL_COND_V_MSG(img->get_format() != Image::FORMAT_RGBA8 && img->get_format() != Image::FORMAT_L8, ERR_FILE_CANT_READ, TTR("Unsupported BMFont texture format."));
								_convert_mono_4bit(img, page, first_cm_ch, base_size, 1);
							} else if (first_gl_ch >= 0) { // 1 x 8 bit monochrome, no outline
								outline = 0;
								ERR_FAIL_COND_V_MSG(img->get_format() != Image::FORMAT_RGBA8 && img->get_format() != Image::FORMAT_L8, ERR_FILE_CANT_READ, TTR("Unsupported BMFont texture format."));
								_convert_mono_8bit(img, page, first_gl_ch, base_size, 0);
							} else {
								ERR_FAIL_V_MSG(ERR_CANT_CREATE, TTR("Unsupported BMFont texture format."));
							}
						}
					}
				}
			} else if (type == "char") {
				char32_t idx = 0;
				Vector2 advance;
				Vector2 size;
				Vector2 offset;
				Rect2 uv_rect;
				int texture_idx = -1;
				uint8_t channel = 15;

				if (keys.has("id")) {
					idx = keys["id"].to_int();
				}
				if (keys.has("x")) {
					uv_rect.position.x = keys["x"].to_int();
				}
				if (keys.has("y")) {
					uv_rect.position.y = keys["y"].to_int();
				}
				if (keys.has("width")) {
					uv_rect.size.width = keys["width"].to_int();
					size.width = keys["width"].to_int();
				}
				if (keys.has("height")) {
					uv_rect.size.height = keys["height"].to_int();
					size.height = keys["height"].to_int();
				}
				if (keys.has("xoffset")) {
					offset.x = keys["xoffset"].to_int();
				}
				if (keys.has("yoffset")) {
					offset.y = keys["yoffset"].to_int() - ascent;
				}
				if (keys.has("page")) {
					texture_idx = keys["page"].to_int();
				}
				if (keys.has("xadvance")) {
					advance.x = keys["xadvance"].to_int();
				}
				if (advance.x < 0) {
					advance.x = size.width + 1;
				}
				if (keys.has("chnl")) {
					channel = keys["chnl"].to_int();
				}

				ERR_FAIL_COND_V_MSG(!packed && channel != 15, ERR_CANT_CREATE, TTR("Invalid glyph channel."));
				int ch_off = 0;
				switch (channel) {
					case 1:
						ch_off = 2;
						break; // B
					case 2:
						ch_off = 1;
						break; // G
					case 4:
						ch_off = 0;
						break; // R
					case 8:
						ch_off = 3;
						break; // A
					default:
						ch_off = 0;
						break;
				}
				set_glyph_advance(0, base_size, idx, advance);
				set_glyph_offset(0, Vector2i(base_size, 0), idx, offset);
				set_glyph_size(0, Vector2i(base_size, 0), idx, size);
				set_glyph_uv_rect(0, Vector2i(base_size, 0), idx, uv_rect);
				set_glyph_texture_idx(0, Vector2i(base_size, 0), idx, texture_idx * (packed ? 4 : 1) + ch_off);
				if (outline > 0) {
					set_glyph_offset(0, Vector2i(base_size, 1), idx, offset);
					set_glyph_size(0, Vector2i(base_size, 1), idx, size);
					set_glyph_uv_rect(0, Vector2i(base_size, 1), idx, uv_rect);
					set_glyph_texture_idx(0, Vector2i(base_size, 1), idx, texture_idx * (packed ? 4 : 1) + ch_off);
				}
			} else if (type == "kerning") {
				Vector2i kpk;
				if (keys.has("first")) {
					kpk.x = keys["first"].to_int();
				}
				if (keys.has("second")) {
					kpk.y = keys["second"].to_int();
				}
				if (keys.has("amount")) {
					set_kerning(0, base_size, kpk, Vector2(keys["amount"].to_int(), 0));
				}
			}

			if (f->eof_reached()) {
				break;
			}
		}
	}

	set_font_name(font_name);
	set_font_style(st_flags);
	set_ascent(0, base_size, ascent);
	set_descent(0, base_size, height - ascent);

	return OK;
}

Error FontData::load_dynamic_font(const String &p_path) {
	reset_state();

	Vector<uint8_t> data = FileAccess::get_file_as_array(p_path);
	set_data(data);

	return OK;
}

void FontData::set_data_ptr(const uint8_t *p_data, size_t p_size) {
	data.clear();
	data_ptr = p_data;
	data_size = p_size;

	if (data_ptr != nullptr) {
		for (int i = 0; i < cache.size(); i++) {
			if (cache[i].is_valid()) {
				TS->font_set_data_ptr(cache[i], data_ptr, data_size);
			}
		}
	}
}

void FontData::set_data(const PackedByteArray &p_data) {
	data = p_data;
	data_ptr = data.ptr();
	data_size = data.size();

	if (data_ptr != nullptr) {
		for (int i = 0; i < cache.size(); i++) {
			if (cache[i].is_valid()) {
				TS->font_set_data_ptr(cache[i], data_ptr, data_size);
			}
		}
	}
}

PackedByteArray FontData::get_data() const {
	if (unlikely((size_t)data.size() != data_size)) {
		PackedByteArray *data_w = const_cast<PackedByteArray *>(&data);
		data_w->resize(data_size);
		memcpy(data_w->ptrw(), data_ptr, data_size);
	}
	return data;
}

void FontData::set_font_name(const String &p_name) {
	_ensure_rid(0);
	TS->font_set_name(cache[0], p_name);
}

String FontData::get_font_name() const {
	_ensure_rid(0);
	return TS->font_get_name(cache[0]);
}

void FontData::set_font_style_name(const String &p_name) {
	_ensure_rid(0);
	TS->font_set_style_name(cache[0], p_name);
}

String FontData::get_font_style_name() const {
	_ensure_rid(0);
	return TS->font_get_style_name(cache[0]);
}

void FontData::set_font_style(uint32_t p_style) {
	_ensure_rid(0);
	TS->font_set_style(cache[0], p_style);
}

uint32_t FontData::get_font_style() const {
	_ensure_rid(0);
	return TS->font_get_style(cache[0]);
}

void FontData::set_antialiased(bool p_antialiased) {
	if (antialiased != p_antialiased) {
		antialiased = p_antialiased;
		for (int i = 0; i < cache.size(); i++) {
			_ensure_rid(i);
			TS->font_set_antialiased(cache[i], antialiased);
		}
		emit_changed();
	}
}

bool FontData::is_antialiased() const {
	return antialiased;
}

void FontData::set_multichannel_signed_distance_field(bool p_msdf) {
	if (msdf != p_msdf) {
		msdf = p_msdf;
		for (int i = 0; i < cache.size(); i++) {
			_ensure_rid(i);
			TS->font_set_multichannel_signed_distance_field(cache[i], msdf);
		}
		emit_changed();
	}
}

bool FontData::is_multichannel_signed_distance_field() const {
	return msdf;
}

void FontData::set_msdf_pixel_range(int p_msdf_pixel_range) {
	if (msdf_pixel_range != p_msdf_pixel_range) {
		msdf_pixel_range = p_msdf_pixel_range;
		for (int i = 0; i < cache.size(); i++) {
			_ensure_rid(i);
			TS->font_set_msdf_pixel_range(cache[i], msdf_pixel_range);
		}
		emit_changed();
	}
}

int FontData::get_msdf_pixel_range() const {
	return msdf_pixel_range;
}

void FontData::set_msdf_size(int p_msdf_size) {
	if (msdf_size != p_msdf_size) {
		msdf_size = p_msdf_size;
		for (int i = 0; i < cache.size(); i++) {
			_ensure_rid(i);
			TS->font_set_msdf_size(cache[i], msdf_size);
		}
		emit_changed();
	}
}

int FontData::get_msdf_size() const {
	return msdf_size;
}

void FontData::set_fixed_size(int p_fixed_size) {
	if (fixed_size != p_fixed_size) {
		fixed_size = p_fixed_size;
		for (int i = 0; i < cache.size(); i++) {
			_ensure_rid(i);
			TS->font_set_fixed_size(cache[i], fixed_size);
		}
		emit_changed();
	}
}

int FontData::get_fixed_size() const {
	return fixed_size;
}

void FontData::set_force_autohinter(bool p_force_autohinter) {
	if (force_autohinter != p_force_autohinter) {
		force_autohinter = p_force_autohinter;
		for (int i = 0; i < cache.size(); i++) {
			_ensure_rid(i);
			TS->font_set_force_autohinter(cache[i], force_autohinter);
		}
		emit_changed();
	}
}

bool FontData::is_force_autohinter() const {
	return force_autohinter;
}

void FontData::set_hinting(TextServer::Hinting p_hinting) {
	if (hinting != p_hinting) {
		hinting = p_hinting;
		for (int i = 0; i < cache.size(); i++) {
			_ensure_rid(i);
			TS->font_set_hinting(cache[i], hinting);
		}
		emit_changed();
	}
}

TextServer::Hinting FontData::get_hinting() const {
	return hinting;
}

void FontData::set_oversampling(real_t p_oversampling) {
	if (oversampling != p_oversampling) {
		oversampling = p_oversampling;
		for (int i = 0; i < cache.size(); i++) {
			_ensure_rid(i);
			TS->font_set_oversampling(cache[i], oversampling);
		}
		emit_changed();
	}
}

real_t FontData::get_oversampling() const {
	return oversampling;
}

RID FontData::find_cache(const Dictionary &p_variation_coordinates) const {
	// Find existing variation cache.
	const Dictionary &supported_coords = get_supported_variation_list();
	for (int i = 0; i < cache.size(); i++) {
		if (cache[i].is_valid()) {
			const Dictionary &cache_var = TS->font_get_variation_coordinates(cache[i]);
			bool match = true;
			for (const Variant *V = supported_coords.next(nullptr); V && match; V = supported_coords.next(V)) {
				const Vector3 &def = supported_coords[*V];

				real_t c_v = def.z;
				if (cache_var.has(*V)) {
					real_t val = cache_var[*V];
					c_v = CLAMP(val, def.x, def.y);
				}
				if (cache_var.has(TS->tag_to_name(*V))) {
					real_t val = cache_var[TS->tag_to_name(*V)];
					c_v = CLAMP(val, def.x, def.y);
				}

				real_t s_v = def.z;
				if (p_variation_coordinates.has(*V)) {
					real_t val = p_variation_coordinates[*V];
					s_v = CLAMP(val, def.x, def.y);
				}
				if (p_variation_coordinates.has(TS->tag_to_name(*V))) {
					real_t val = p_variation_coordinates[TS->tag_to_name(*V)];
					s_v = CLAMP(val, def.x, def.y);
				}

				match = match && (c_v == s_v);
			}
			if (match) {
				return cache[i];
			}
		}
	}

	// Create new variation cache.
	int idx = cache.size();
	_ensure_rid(idx);
	TS->font_set_variation_coordinates(cache[idx], p_variation_coordinates);
	return cache[idx];
}

int FontData::get_cache_count() const {
	return cache.size();
}

void FontData::clear_cache() {
	_clear_cache();
	cache.clear();
}

void FontData::remove_cache(int p_cache_index) {
	ERR_FAIL_INDEX(p_cache_index, cache.size());
	if (cache[p_cache_index].is_valid()) {
		TS->free(cache.write[p_cache_index]);
	}
	cache.remove(p_cache_index);
	emit_changed();
}

Array FontData::get_size_cache_list(int p_cache_index) const {
	ERR_FAIL_COND_V(p_cache_index < 0, Array());
	_ensure_rid(p_cache_index);
	return TS->font_get_size_cache_list(cache[p_cache_index]);
}

void FontData::clear_size_cache(int p_cache_index) {
	ERR_FAIL_COND(p_cache_index < 0);
	_ensure_rid(p_cache_index);
	TS->font_clear_size_cache(cache[p_cache_index]);
}

void FontData::remove_size_cache(int p_cache_index, const Vector2i &p_size) {
	ERR_FAIL_COND(p_cache_index < 0);
	_ensure_rid(p_cache_index);
	TS->font_remove_size_cache(cache[p_cache_index], p_size);
}

void FontData::set_variation_coordinates(int p_cache_index, const Dictionary &p_variation_coordinates) {
	ERR_FAIL_COND(p_cache_index < 0);
	_ensure_rid(p_cache_index);
	TS->font_set_variation_coordinates(cache[p_cache_index], p_variation_coordinates);
	emit_changed();
}

Dictionary FontData::get_variation_coordinates(int p_cache_index) const {
	ERR_FAIL_COND_V(p_cache_index < 0, Dictionary());
	_ensure_rid(p_cache_index);
	return TS->font_get_variation_coordinates(cache[p_cache_index]);
}

void FontData::set_ascent(int p_cache_index, int p_size, real_t p_ascent) {
	ERR_FAIL_COND(p_cache_index < 0);
	_ensure_rid(p_cache_index);
	TS->font_set_ascent(cache[p_cache_index], p_size, p_ascent);
}

real_t FontData::get_ascent(int p_cache_index, int p_size) const {
	ERR_FAIL_COND_V(p_cache_index < 0, 0.f);
	_ensure_rid(p_cache_index);
	return TS->font_get_ascent(cache[p_cache_index], p_size);
}

void FontData::set_descent(int p_cache_index, int p_size, real_t p_descent) {
	ERR_FAIL_COND(p_cache_index < 0);
	_ensure_rid(p_cache_index);
	TS->font_set_descent(cache[p_cache_index], p_size, p_descent);
}

real_t FontData::get_descent(int p_cache_index, int p_size) const {
	ERR_FAIL_COND_V(p_cache_index < 0, 0.f);
	_ensure_rid(p_cache_index);
	return TS->font_get_descent(cache[p_cache_index], p_size);
}

void FontData::set_underline_position(int p_cache_index, int p_size, real_t p_underline_position) {
	ERR_FAIL_COND(p_cache_index < 0);
	_ensure_rid(p_cache_index);
	TS->font_set_underline_position(cache[p_cache_index], p_size, p_underline_position);
}

real_t FontData::get_underline_position(int p_cache_index, int p_size) const {
	ERR_FAIL_COND_V(p_cache_index < 0, 0.f);
	_ensure_rid(p_cache_index);
	return TS->font_get_underline_position(cache[p_cache_index], p_size);
}

void FontData::set_underline_thickness(int p_cache_index, int p_size, real_t p_underline_thickness) {
	ERR_FAIL_COND(p_cache_index < 0);
	_ensure_rid(p_cache_index);
	TS->font_set_underline_thickness(cache[p_cache_index], p_size, p_underline_thickness);
}

real_t FontData::get_underline_thickness(int p_cache_index, int p_size) const {
	ERR_FAIL_COND_V(p_cache_index < 0, 0.f);
	_ensure_rid(p_cache_index);
	return TS->font_get_underline_thickness(cache[p_cache_index], p_size);
}

void FontData::set_scale(int p_cache_index, int p_size, real_t p_scale) {
	ERR_FAIL_COND(p_cache_index < 0);
	_ensure_rid(p_cache_index);
	TS->font_set_scale(cache[p_cache_index], p_size, p_scale);
}

real_t FontData::get_scale(int p_cache_index, int p_size) const {
	ERR_FAIL_COND_V(p_cache_index < 0, 0.f);
	_ensure_rid(p_cache_index);
	return TS->font_get_scale(cache[p_cache_index], p_size);
}

void FontData::set_spacing(int p_cache_index, int p_size, TextServer::SpacingType p_spacing, int p_value) {
	ERR_FAIL_COND(p_cache_index < 0);
	_ensure_rid(p_cache_index);
	TS->font_set_spacing(cache[p_cache_index], p_size, p_spacing, p_value);
}

int FontData::get_spacing(int p_cache_index, int p_size, TextServer::SpacingType p_spacing) const {
	ERR_FAIL_COND_V(p_cache_index < 0, 0);
	_ensure_rid(p_cache_index);
	return TS->font_get_spacing(cache[p_cache_index], p_size, p_spacing);
}

int FontData::get_texture_count(int p_cache_index, const Vector2i &p_size) const {
	ERR_FAIL_COND_V(p_cache_index < 0, 0);
	_ensure_rid(p_cache_index);
	return TS->font_get_texture_count(cache[p_cache_index], p_size);
}

void FontData::clear_textures(int p_cache_index, const Vector2i &p_size) {
	ERR_FAIL_COND(p_cache_index < 0);
	_ensure_rid(p_cache_index);
	TS->font_clear_textures(cache[p_cache_index], p_size);
}

void FontData::remove_texture(int p_cache_index, const Vector2i &p_size, int p_texture_index) {
	ERR_FAIL_COND(p_cache_index < 0);
	_ensure_rid(p_cache_index);
	TS->font_remove_texture(cache[p_cache_index], p_size, p_texture_index);
}

void FontData::set_texture_image(int p_cache_index, const Vector2i &p_size, int p_texture_index, const Ref<Image> &p_image) {
	ERR_FAIL_COND(p_cache_index < 0);
	_ensure_rid(p_cache_index);
	TS->font_set_texture_image(cache[p_cache_index], p_size, p_texture_index, p_image);
}

Ref<Image> FontData::get_texture_image(int p_cache_index, const Vector2i &p_size, int p_texture_index) const {
	ERR_FAIL_COND_V(p_cache_index < 0, Ref<Image>());
	_ensure_rid(p_cache_index);
	return TS->font_get_texture_image(cache[p_cache_index], p_size, p_texture_index);
}

void FontData::set_texture_offsets(int p_cache_index, const Vector2i &p_size, int p_texture_index, const PackedInt32Array &p_offset) {
	ERR_FAIL_COND(p_cache_index < 0);
	_ensure_rid(p_cache_index);
	TS->font_set_texture_offsets(cache[p_cache_index], p_size, p_texture_index, p_offset);
}

PackedInt32Array FontData::get_texture_offsets(int p_cache_index, const Vector2i &p_size, int p_texture_index) const {
	ERR_FAIL_COND_V(p_cache_index < 0, PackedInt32Array());
	_ensure_rid(p_cache_index);
	return TS->font_get_texture_offsets(cache[p_cache_index], p_size, p_texture_index);
}

Array FontData::get_glyph_list(int p_cache_index, const Vector2i &p_size) const {
	ERR_FAIL_COND_V(p_cache_index < 0, Array());
	_ensure_rid(p_cache_index);
	return TS->font_get_glyph_list(cache[p_cache_index], p_size);
}

void FontData::clear_glyphs(int p_cache_index, const Vector2i &p_size) {
	ERR_FAIL_COND(p_cache_index < 0);
	_ensure_rid(p_cache_index);
	TS->font_clear_glyphs(cache[p_cache_index], p_size);
}

void FontData::remove_glyph(int p_cache_index, const Vector2i &p_size, int32_t p_glyph) {
	ERR_FAIL_COND(p_cache_index < 0);
	_ensure_rid(p_cache_index);
	TS->font_remove_glyph(cache[p_cache_index], p_size, p_glyph);
}

void FontData::set_glyph_advance(int p_cache_index, int p_size, int32_t p_glyph, const Vector2 &p_advance) {
	ERR_FAIL_COND(p_cache_index < 0);
	_ensure_rid(p_cache_index);
	TS->font_set_glyph_advance(cache[p_cache_index], p_size, p_glyph, p_advance);
}

Vector2 FontData::get_glyph_advance(int p_cache_index, int p_size, int32_t p_glyph) const {
	ERR_FAIL_COND_V(p_cache_index < 0, Vector2());
	_ensure_rid(p_cache_index);
	return TS->font_get_glyph_advance(cache[p_cache_index], p_size, p_glyph);
}

void FontData::set_glyph_offset(int p_cache_index, const Vector2i &p_size, int32_t p_glyph, const Vector2 &p_offset) {
	ERR_FAIL_COND(p_cache_index < 0);
	_ensure_rid(p_cache_index);
	TS->font_set_glyph_offset(cache[p_cache_index], p_size, p_glyph, p_offset);
}

Vector2 FontData::get_glyph_offset(int p_cache_index, const Vector2i &p_size, int32_t p_glyph) const {
	ERR_FAIL_COND_V(p_cache_index < 0, Vector2());
	_ensure_rid(p_cache_index);
	return TS->font_get_glyph_offset(cache[p_cache_index], p_size, p_glyph);
}

void FontData::set_glyph_size(int p_cache_index, const Vector2i &p_size, int32_t p_glyph, const Vector2 &p_gl_size) {
	ERR_FAIL_COND(p_cache_index < 0);
	_ensure_rid(p_cache_index);
	TS->font_set_glyph_size(cache[p_cache_index], p_size, p_glyph, p_gl_size);
}

Vector2 FontData::get_glyph_size(int p_cache_index, const Vector2i &p_size, int32_t p_glyph) const {
	ERR_FAIL_COND_V(p_cache_index < 0, Vector2());
	_ensure_rid(p_cache_index);
	return TS->font_get_glyph_size(cache[p_cache_index], p_size, p_glyph);
}

void FontData::set_glyph_uv_rect(int p_cache_index, const Vector2i &p_size, int32_t p_glyph, const Rect2 &p_uv_rect) {
	ERR_FAIL_COND(p_cache_index < 0);
	_ensure_rid(p_cache_index);
	TS->font_set_glyph_uv_rect(cache[p_cache_index], p_size, p_glyph, p_uv_rect);
}

Rect2 FontData::get_glyph_uv_rect(int p_cache_index, const Vector2i &p_size, int32_t p_glyph) const {
	ERR_FAIL_COND_V(p_cache_index < 0, Rect2());
	_ensure_rid(p_cache_index);
	return TS->font_get_glyph_uv_rect(cache[p_cache_index], p_size, p_glyph);
}

void FontData::set_glyph_texture_idx(int p_cache_index, const Vector2i &p_size, int32_t p_glyph, int p_texture_idx) {
	ERR_FAIL_COND(p_cache_index < 0);
	_ensure_rid(p_cache_index);
	TS->font_set_glyph_texture_idx(cache[p_cache_index], p_size, p_glyph, p_texture_idx);
}

int FontData::get_glyph_texture_idx(int p_cache_index, const Vector2i &p_size, int32_t p_glyph) const {
	ERR_FAIL_COND_V(p_cache_index < 0, 0);
	_ensure_rid(p_cache_index);
	return TS->font_get_glyph_texture_idx(cache[p_cache_index], p_size, p_glyph);
}

Array FontData::get_kerning_list(int p_cache_index, int p_size) const {
	ERR_FAIL_COND_V(p_cache_index < 0, Array());
	_ensure_rid(p_cache_index);
	return TS->font_get_kerning_list(cache[p_cache_index], p_size);
}

void FontData::clear_kerning_map(int p_cache_index, int p_size) {
	ERR_FAIL_COND(p_cache_index < 0);
	_ensure_rid(p_cache_index);
	TS->font_clear_kerning_map(cache[p_cache_index], p_size);
}

void FontData::remove_kerning(int p_cache_index, int p_size, const Vector2i &p_glyph_pair) {
	ERR_FAIL_COND(p_cache_index < 0);
	_ensure_rid(p_cache_index);
	TS->font_remove_kerning(cache[p_cache_index], p_size, p_glyph_pair);
}

void FontData::set_kerning(int p_cache_index, int p_size, const Vector2i &p_glyph_pair, const Vector2 &p_kerning) {
	ERR_FAIL_COND(p_cache_index < 0);
	_ensure_rid(p_cache_index);
	TS->font_set_kerning(cache[p_cache_index], p_size, p_glyph_pair, p_kerning);
}

Vector2 FontData::get_kerning(int p_cache_index, int p_size, const Vector2i &p_glyph_pair) const {
	ERR_FAIL_COND_V(p_cache_index < 0, Vector2());
	_ensure_rid(p_cache_index);
	return TS->font_get_kerning(cache[p_cache_index], p_size, p_glyph_pair);
}

void FontData::render_range(int p_cache_index, const Vector2i &p_size, char32_t p_start, char32_t p_end) {
	ERR_FAIL_COND(p_cache_index < 0);
	_ensure_rid(p_cache_index);
	TS->font_render_range(cache[p_cache_index], p_size, p_start, p_end);
}

void FontData::render_glyph(int p_cache_index, const Vector2i &p_size, int32_t p_index) {
	ERR_FAIL_COND(p_cache_index < 0);
	_ensure_rid(p_cache_index);
	TS->font_render_glyph(cache[p_cache_index], p_size, p_index);
}

RID FontData::get_cache_rid(int p_cache_index) const {
	ERR_FAIL_COND_V(p_cache_index < 0, RID());
	_ensure_rid(p_cache_index);
	return cache[p_cache_index];
}

bool FontData::is_language_supported(const String &p_language) const {
	_ensure_rid(0);
	return TS->font_is_language_supported(cache[0], p_language);
}

void FontData::set_language_support_override(const String &p_language, bool p_supported) {
	_ensure_rid(0);
	TS->font_set_language_support_override(cache[0], p_language, p_supported);
}

bool FontData::get_language_support_override(const String &p_language) const {
	_ensure_rid(0);
	return TS->font_get_language_support_override(cache[0], p_language);
}

void FontData::remove_language_support_override(const String &p_language) {
	_ensure_rid(0);
	TS->font_remove_language_support_override(cache[0], p_language);
}

Vector<String> FontData::get_language_support_overrides() const {
	_ensure_rid(0);
	return TS->font_get_language_support_overrides(cache[0]);
}

bool FontData::is_script_supported(const String &p_script) const {
	_ensure_rid(0);
	return TS->font_is_script_supported(cache[0], p_script);
}

void FontData::set_script_support_override(const String &p_script, bool p_supported) {
	_ensure_rid(0);
	TS->font_set_script_support_override(cache[0], p_script, p_supported);
}

bool FontData::get_script_support_override(const String &p_script) const {
	_ensure_rid(0);
	return TS->font_get_script_support_override(cache[0], p_script);
}

void FontData::remove_script_support_override(const String &p_script) {
	_ensure_rid(0);
	TS->font_remove_script_support_override(cache[0], p_script);
}

Vector<String> FontData::get_script_support_overrides() const {
	_ensure_rid(0);
	return TS->font_get_script_support_overrides(cache[0]);
}

bool FontData::has_char(char32_t p_char) const {
	_ensure_rid(0);
	return TS->font_has_char(cache[0], p_char);
}

String FontData::get_supported_chars() const {
	_ensure_rid(0);
	return TS->font_get_supported_chars(cache[0]);
}

int32_t FontData::get_glyph_index(int p_size, char32_t p_char, char32_t p_variation_selector) const {
	_ensure_rid(0);
	return TS->font_get_glyph_index(cache[0], p_size, p_char, p_variation_selector);
}

Dictionary FontData::get_supported_feature_list() const {
	_ensure_rid(0);
	return TS->font_supported_feature_list(cache[0]);
}

Dictionary FontData::get_supported_variation_list() const {
	_ensure_rid(0);
	return TS->font_supported_variation_list(cache[0]);
}

FontData::FontData() {
	/* NOP */
}

FontData::~FontData() {
	_clear_cache();
}

/*************************************************************************/

void Font::_data_changed() {
	for (int i = 0; i < rids.size(); i++) {
		rids.write[i] = RID();
	}
	emit_changed();
}

void Font::_ensure_rid(int p_index) const {
	// Find or create cache record.
	if (!rids[p_index].is_valid() && data[p_index].is_valid()) {
		rids.write[p_index] = data[p_index]->find_cache(variation_coordinates);
	}
}

void Font::_bind_methods() {
	ClassDB::bind_method(D_METHOD("add_data", "data"), &Font::add_data);
	ClassDB::bind_method(D_METHOD("set_data", "idx", "data"), &Font::set_data);
	ClassDB::bind_method(D_METHOD("get_data_count"), &Font::get_data_count);
	ClassDB::bind_method(D_METHOD("get_data", "idx"), &Font::get_data);
	ClassDB::bind_method(D_METHOD("get_data_rid", "idx"), &Font::get_data_rid);
	ClassDB::bind_method(D_METHOD("clear_data"), &Font::clear_data);
	ClassDB::bind_method(D_METHOD("remove_data", "idx"), &Font::remove_data);

	ClassDB::bind_method(D_METHOD("set_variation_coordinates", "variation_coordinates"), &Font::set_variation_coordinates);
	ClassDB::bind_method(D_METHOD("get_variation_coordinates"), &Font::get_variation_coordinates);
	ADD_PROPERTY(PropertyInfo(Variant::DICTIONARY, "variation_coordinates"), "set_variation_coordinates", "get_variation_coordinates");

	ClassDB::bind_method(D_METHOD("set_spacing", "spacing", "value"), &Font::set_spacing);
	ClassDB::bind_method(D_METHOD("get_spacing", "spacing"), &Font::get_spacing);

	ADD_GROUP("Extra Spacing", "spacing");
	ADD_PROPERTYI(PropertyInfo(Variant::INT, "spacing_top"), "set_spacing", "get_spacing", TextServer::SPACING_TOP);
	ADD_PROPERTYI(PropertyInfo(Variant::INT, "spacing_bottom"), "set_spacing", "get_spacing", TextServer::SPACING_BOTTOM);

	ClassDB::bind_method(D_METHOD("get_height", "size"), &Font::get_height, DEFVAL(DEFAULT_FONT_SIZE));
	ClassDB::bind_method(D_METHOD("get_ascent", "size"), &Font::get_ascent, DEFVAL(DEFAULT_FONT_SIZE));
	ClassDB::bind_method(D_METHOD("get_descent", "size"), &Font::get_descent, DEFVAL(DEFAULT_FONT_SIZE));
	ClassDB::bind_method(D_METHOD("get_underline_position", "size"), &Font::get_underline_position, DEFVAL(DEFAULT_FONT_SIZE));
	ClassDB::bind_method(D_METHOD("get_underline_thickness", "size"), &Font::get_underline_thickness, DEFVAL(DEFAULT_FONT_SIZE));

	ClassDB::bind_method(D_METHOD("get_string_size", "text", "size", "align", "width", "flags"), &Font::get_string_size, DEFVAL(DEFAULT_FONT_SIZE), DEFVAL(HALIGN_LEFT), DEFVAL(-1), DEFVAL(TextServer::JUSTIFICATION_KASHIDA | TextServer::JUSTIFICATION_WORD_BOUND));
	ClassDB::bind_method(D_METHOD("get_multiline_string_size", "text", "width", "size", "flags"), &Font::get_multiline_string_size, DEFVAL(-1), DEFVAL(DEFAULT_FONT_SIZE), DEFVAL(TextServer::BREAK_MANDATORY | TextServer::BREAK_WORD_BOUND));

	ClassDB::bind_method(D_METHOD("draw_string", "canvas_item", "pos", "text", "align", "width", "size", "modulate", "outline_size", "outline_modulate", "flags"), &Font::draw_string, DEFVAL(HALIGN_LEFT), DEFVAL(-1), DEFVAL(DEFAULT_FONT_SIZE), DEFVAL(Color(1, 1, 1)), DEFVAL(0), DEFVAL(Color(1, 1, 1, 0)), DEFVAL(TextServer::JUSTIFICATION_KASHIDA | TextServer::JUSTIFICATION_WORD_BOUND));
	ClassDB::bind_method(D_METHOD("draw_multiline_string", "canvas_item", "pos", "text", "align", "width", "max_lines", "size", "modulate", "outline_size", "outline_modulate", "flags"), &Font::draw_multiline_string, DEFVAL(HALIGN_LEFT), DEFVAL(-1), DEFVAL(-1), DEFVAL(DEFAULT_FONT_SIZE), DEFVAL(Color(1, 1, 1)), DEFVAL(0), DEFVAL(Color(1, 1, 1, 0)), DEFVAL(TextServer::BREAK_MANDATORY | TextServer::BREAK_WORD_BOUND | TextServer::JUSTIFICATION_KASHIDA | TextServer::JUSTIFICATION_WORD_BOUND));

	ClassDB::bind_method(D_METHOD("get_char_size", "char", "next", "size"), &Font::get_char_size, DEFVAL(0), DEFVAL(DEFAULT_FONT_SIZE));
	ClassDB::bind_method(D_METHOD("draw_char", "canvas_item", "pos", "char", "next", "size", "modulate", "outline_size", "outline_modulate"), &Font::draw_char, DEFVAL(0), DEFVAL(DEFAULT_FONT_SIZE), DEFVAL(Color(1, 1, 1)), DEFVAL(0), DEFVAL(Color(1, 1, 1, 0)));

	ClassDB::bind_method(D_METHOD("has_char", "char"), &Font::has_char);
	ClassDB::bind_method(D_METHOD("get_supported_chars"), &Font::get_supported_chars);

	ClassDB::bind_method(D_METHOD("update_changes"), &Font::update_changes);
}

bool Font::_set(const StringName &p_name, const Variant &p_value) {
	Vector<String> tokens = p_name.operator String().split("/");
#ifndef DISABLE_DEPRECATED
	if (tokens.size() == 1 && tokens[0] == "font_data") {
		// Compatibility, DynamicFont main data.
		Ref<FontData> fd = p_value;
		if (fd.is_valid()) {
			add_data(fd);
			return true;
		}
		return false;
	} else if (tokens.size() == 2 && tokens[0] == "fallback") {
		// Compatibility, DynamicFont fallback data.
		Ref<FontData> fd = p_value;
		if (fd.is_valid()) {
			add_data(fd);
			return true;
		}
		return false;
	} else if (tokens.size() == 1 && tokens[0] == "fallback") {
		// Compatibility, BitmapFont fallback data.
		Ref<Font> f = p_value;
		if (f.is_valid()) {
			for (int i = 0; i < f->get_data_count(); i++) {
				add_data(f->get_data(i));
			}
			return true;
		}
		return false;
	}
#endif /* DISABLE_DEPRECATED */
	if (tokens.size() == 2 && tokens[0] == "data") {
		int idx = tokens[1].to_int();
		Ref<FontData> fd = p_value;
		if (fd.is_valid()) {
			if (idx == data.size()) {
				add_data(fd);
				return true;
			} else if (idx >= 0 && idx < data.size()) {
				set_data(idx, fd);
				return true;
			} else {
				return false;
			}
		} else if (idx >= 0 && idx < data.size()) {
			remove_data(idx);
			return true;
		}
	}
	return false;
}

bool Font::_get(const StringName &p_name, Variant &r_ret) const {
	Vector<String> tokens = p_name.operator String().split("/");
	if (tokens.size() == 2 && tokens[0] == "data") {
		int idx = tokens[1].to_int();

		if (idx == data.size()) {
			r_ret = Ref<FontData>();
			return true;
		} else if (idx >= 0 && idx < data.size()) {
			r_ret = get_data(idx);
			return true;
		}
	}

	return false;
}

void Font::_get_property_list(List<PropertyInfo> *p_list) const {
	for (int i = 0; i < data.size(); i++) {
		p_list->push_back(PropertyInfo(Variant::OBJECT, "data/" + itos(i), PROPERTY_HINT_RESOURCE_TYPE, "FontData"));
	}
	p_list->push_back(PropertyInfo(Variant::OBJECT, "data/" + itos(data.size()), PROPERTY_HINT_RESOURCE_TYPE, "FontData"));
}

void Font::reset_state() {
	for (int i = 0; i < data.size(); i++) {
		if (data[i].is_valid()) {
			data.write[i]->connect(SNAME("changed"), callable_mp(this, &Font::_data_changed), varray(), CONNECT_REFERENCE_COUNTED);
		}
	}
	cache.clear();
	cache_wrap.clear();
	data.clear();
	rids.clear();

	variation_coordinates.clear();
	spacing_bottom = 0;
	spacing_top = 0;
}

Dictionary Font::get_feature_list() const {
	Dictionary out;
	for (int i = 0; i < data.size(); i++) {
		Dictionary data_ftrs = data[i]->get_supported_feature_list();
		for (const Variant *ftr = data_ftrs.next(nullptr); ftr != nullptr; ftr = data_ftrs.next(ftr)) {
			out[*ftr] = data_ftrs[*ftr];
		}
	}
	return out;
}

void Font::add_data(const Ref<FontData> &p_data) {
	ERR_FAIL_COND(p_data.is_null());
	data.push_back(p_data);
	rids.push_back(RID());

	if (data[data.size() - 1].is_valid()) {
		data.write[data.size() - 1]->connect(SNAME("changed"), callable_mp(this, &Font::_data_changed), varray(), CONNECT_REFERENCE_COUNTED);
		Dictionary data_var_list = p_data->get_supported_variation_list();
		for (int j = 0; j < data_var_list.size(); j++) {
			int32_t tag = data_var_list.get_key_at_index(j);
			Vector3i value = data_var_list.get_value_at_index(j);
			if (!variation_coordinates.has(tag) && !variation_coordinates.has(TS->tag_to_name(tag))) {
				variation_coordinates[TS->tag_to_name(tag)] = value.z;
			}
		}
	}

	cache.clear();
	cache_wrap.clear();

	emit_changed();
	notify_property_list_changed();
}

void Font::set_data(int p_idx, const Ref<FontData> &p_data) {
	ERR_FAIL_COND(p_data.is_null());
	ERR_FAIL_INDEX(p_idx, data.size());

	if (data[p_idx].is_valid()) {
		data.write[p_idx]->disconnect(SNAME("changed"), callable_mp(this, &Font::_data_changed));
	}

	data.write[p_idx] = p_data;
	rids.write[p_idx] = RID();
	Dictionary data_var_list = p_data->get_supported_variation_list();
	for (int j = 0; j < data_var_list.size(); j++) {
		int32_t tag = data_var_list.get_key_at_index(j);
		Vector3i value = data_var_list.get_value_at_index(j);
		if (!variation_coordinates.has(tag) && !variation_coordinates.has(TS->tag_to_name(tag))) {
			variation_coordinates[TS->tag_to_name(tag)] = value.z;
		}
	}

	if (data[p_idx].is_valid()) {
		data.write[p_idx]->connect(SNAME("changed"), callable_mp(this, &Font::_data_changed), varray(), CONNECT_REFERENCE_COUNTED);
	}

	cache.clear();
	cache_wrap.clear();

	emit_changed();
	notify_property_list_changed();
}

int Font::get_data_count() const {
	return data.size();
}

Ref<FontData> Font::get_data(int p_idx) const {
	ERR_FAIL_INDEX_V(p_idx, data.size(), Ref<FontData>());
	return data[p_idx];
}

RID Font::get_data_rid(int p_idx) const {
	ERR_FAIL_INDEX_V(p_idx, data.size(), RID());
	_ensure_rid(p_idx);
	return rids[p_idx];
}

void Font::clear_data() {
	for (int i = 0; i < data.size(); i++) {
		if (data[i].is_valid()) {
			data.write[i]->connect(SNAME("changed"), callable_mp(this, &Font::_data_changed), varray(), CONNECT_REFERENCE_COUNTED);
		}
	}
	data.clear();
	rids.clear();
}

void Font::remove_data(int p_idx) {
	ERR_FAIL_INDEX(p_idx, data.size());

	if (data[p_idx].is_valid()) {
		data.write[p_idx]->disconnect(SNAME("changed"), callable_mp(this, &Font::_data_changed));
	}

	data.remove(p_idx);
	rids.remove(p_idx);

	cache.clear();
	cache_wrap.clear();

	emit_changed();
	notify_property_list_changed();
}

void Font::set_variation_coordinates(const Dictionary &p_variation_coordinates) {
	_data_changed();
	variation_coordinates = p_variation_coordinates;
}

Dictionary Font::get_variation_coordinates() const {
	return variation_coordinates;
}

void Font::set_spacing(TextServer::SpacingType p_spacing, int p_value) {
	_data_changed();
	switch (p_spacing) {
		case TextServer::SPACING_TOP: {
			spacing_top = p_value;
		} break;
		case TextServer::SPACING_BOTTOM: {
			spacing_bottom = p_value;
		} break;
		default: {
			ERR_FAIL_MSG("Invalid spacing type: " + itos(p_spacing));
		} break;
	}
}

int Font::get_spacing(TextServer::SpacingType p_spacing) const {
	switch (p_spacing) {
		case TextServer::SPACING_TOP: {
			return spacing_top;
		} break;
		case TextServer::SPACING_BOTTOM: {
			return spacing_bottom;
		} break;
		default: {
			ERR_FAIL_V_MSG(0, "Invalid spacing type: " + itos(p_spacing));
		} break;
	}
}

real_t Font::get_height(int p_size) const {
	real_t ret = 0.f;
	for (int i = 0; i < data.size(); i++) {
		_ensure_rid(i);
		ret = MAX(ret, TS->font_get_ascent(rids[i], p_size) + TS->font_get_descent(rids[i], p_size));
	}
	return ret + spacing_bottom + spacing_top;
}

real_t Font::get_ascent(int p_size) const {
	real_t ret = 0.f;
	for (int i = 0; i < data.size(); i++) {
		_ensure_rid(i);
		ret = MAX(ret, TS->font_get_ascent(rids[i], p_size));
	}
	return ret + spacing_top;
}

real_t Font::get_descent(int p_size) const {
	real_t ret = 0.f;
	for (int i = 0; i < data.size(); i++) {
		_ensure_rid(i);
		ret = MAX(ret, TS->font_get_descent(rids[i], p_size));
	}
	return ret + spacing_bottom;
}

real_t Font::get_underline_position(int p_size) const {
	real_t ret = 0.f;
	for (int i = 0; i < data.size(); i++) {
		_ensure_rid(i);
		ret = MAX(ret, TS->font_get_underline_position(rids[i], p_size));
	}
	return ret + spacing_top;
}

real_t Font::get_underline_thickness(int p_size) const {
	real_t ret = 0.f;
	for (int i = 0; i < data.size(); i++) {
		_ensure_rid(i);
		ret = MAX(ret, TS->font_get_underline_thickness(rids[i], p_size));
	}
	return ret;
}

Size2 Font::get_string_size(const String &p_text, int p_size, HAlign p_align, real_t p_width, uint16_t p_flags) const {
	ERR_FAIL_COND_V(data.is_empty(), Size2());

	for (int i = 0; i < data.size(); i++) {
		_ensure_rid(i);
	}

	uint64_t hash = p_text.hash64();
	if (p_align == HALIGN_FILL) {
		hash = hash_djb2_one_64(hash_djb2_one_float(p_width), hash);
		hash = hash_djb2_one_64(p_flags, hash);
	}
	hash = hash_djb2_one_64(p_size, hash);

	Ref<TextLine> buffer;
	if (cache.has(hash)) {
		buffer = cache.get(hash);
	} else {
		buffer.instantiate();
		buffer->add_string(p_text, Ref<Font>(this), p_size, Dictionary(), TranslationServer::get_singleton()->get_tool_locale());
		cache.insert(hash, buffer);
	}
	return buffer->get_size();
}

Size2 Font::get_multiline_string_size(const String &p_text, real_t p_width, int p_size, uint16_t p_flags) const {
	ERR_FAIL_COND_V(data.is_empty(), Size2());

	for (int i = 0; i < data.size(); i++) {
		_ensure_rid(i);
	}

	uint64_t hash = p_text.hash64();
	uint64_t wrp_hash = hash_djb2_one_64(hash_djb2_one_float(p_width), hash);
	wrp_hash = hash_djb2_one_64(p_flags, wrp_hash);
	wrp_hash = hash_djb2_one_64(p_size, wrp_hash);

	Ref<TextParagraph> lines_buffer;
	if (cache_wrap.has(wrp_hash)) {
		lines_buffer = cache_wrap.get(wrp_hash);
	} else {
		lines_buffer.instantiate();
		lines_buffer->add_string(p_text, Ref<Font>(this), p_size, Dictionary(), TranslationServer::get_singleton()->get_tool_locale());
		lines_buffer->set_width(p_width);
		lines_buffer->set_flags(p_flags);
		cache_wrap.insert(wrp_hash, lines_buffer);
	}

	Size2 ret;
	for (int i = 0; i < lines_buffer->get_line_count(); i++) {
		Size2 line_size = lines_buffer->get_line_size(i);
		if (lines_buffer->get_orientation() == TextServer::ORIENTATION_HORIZONTAL) {
			ret.x = MAX(ret.x, line_size.x);
			ret.y += line_size.y;
		} else {
			ret.y = MAX(ret.y, line_size.y);
			ret.x += line_size.x;
		}
	}
	return ret;
}

void Font::draw_string(RID p_canvas_item, const Point2 &p_pos, const String &p_text, HAlign p_align, real_t p_width, int p_size, const Color &p_modulate, int p_outline_size, const Color &p_outline_modulate, uint16_t p_flags) const {
	ERR_FAIL_COND(data.is_empty());

	for (int i = 0; i < data.size(); i++) {
		_ensure_rid(i);
	}

	uint64_t hash = p_text.hash64();
	if (p_align == HALIGN_FILL) {
		hash = hash_djb2_one_64(hash_djb2_one_float(p_width), hash);
		hash = hash_djb2_one_64(p_flags, hash);
	}
	hash = hash_djb2_one_64(p_size, hash);

	Ref<TextLine> buffer;
	if (cache.has(hash)) {
		buffer = cache.get(hash);
	} else {
		buffer.instantiate();
		buffer->add_string(p_text, Ref<Font>(this), p_size, Dictionary(), TranslationServer::get_singleton()->get_tool_locale());
		cache.insert(hash, buffer);
	}

	Vector2 ofs = p_pos;
	if (buffer->get_orientation() == TextServer::ORIENTATION_HORIZONTAL) {
		ofs.y -= buffer->get_line_ascent();
	} else {
		ofs.x -= buffer->get_line_ascent();
	}

	buffer->set_width(p_width);
	buffer->set_align(p_align);
	buffer->set_flags(p_flags);

	if (p_outline_size > 0 && p_outline_modulate.a != 0.0f) {
		buffer->draw_outline(p_canvas_item, ofs, p_outline_size, p_outline_modulate);
	}
	buffer->draw(p_canvas_item, ofs, p_modulate);
}

void Font::draw_multiline_string(RID p_canvas_item, const Point2 &p_pos, const String &p_text, HAlign p_align, float p_width, int p_max_lines, int p_size, const Color &p_modulate, int p_outline_size, const Color &p_outline_modulate, uint16_t p_flags) const {
	ERR_FAIL_COND(data.is_empty());

	for (int i = 0; i < data.size(); i++) {
		_ensure_rid(i);
	}

	uint64_t hash = p_text.hash64();
	uint64_t wrp_hash = hash_djb2_one_64(hash_djb2_one_float(p_width), hash);
	wrp_hash = hash_djb2_one_64(p_flags, wrp_hash);
	wrp_hash = hash_djb2_one_64(p_size, wrp_hash);

	Ref<TextParagraph> lines_buffer;
	if (cache_wrap.has(wrp_hash)) {
		lines_buffer = cache_wrap.get(wrp_hash);
	} else {
		lines_buffer.instantiate();
		lines_buffer->add_string(p_text, Ref<Font>(this), p_size, Dictionary(), TranslationServer::get_singleton()->get_tool_locale());
		lines_buffer->set_width(p_width);
		lines_buffer->set_flags(p_flags);
		cache_wrap.insert(wrp_hash, lines_buffer);
	}

	lines_buffer->set_align(p_align);

	Vector2 lofs = p_pos;
	for (int i = 0; i < lines_buffer->get_line_count(); i++) {
		if (lines_buffer->get_orientation() == TextServer::ORIENTATION_HORIZONTAL) {
			if (i == 0) {
				lofs.y -= lines_buffer->get_line_ascent(0);
			}
		} else {
			if (i == 0) {
				lofs.x -= lines_buffer->get_line_ascent(0);
			}
		}
		if (p_width > 0) {
			lines_buffer->set_align(p_align);
		}

		if (p_outline_size > 0 && p_outline_modulate.a != 0.0f) {
			lines_buffer->draw_line_outline(p_canvas_item, lofs, i, p_outline_size, p_outline_modulate);
		}
		lines_buffer->draw_line(p_canvas_item, lofs, i, p_modulate);

		Size2 line_size = lines_buffer->get_line_size(i);
		if (lines_buffer->get_orientation() == TextServer::ORIENTATION_HORIZONTAL) {
			lofs.y += line_size.y;
		} else {
			lofs.x += line_size.x;
		}

		if ((p_max_lines > 0) && (i >= p_max_lines)) {
			return;
		}
	}
}

Size2 Font::get_char_size(char32_t p_char, char32_t p_next, int p_size) const {
	for (int i = 0; i < data.size(); i++) {
		_ensure_rid(i);
		if (data[i]->has_char(p_char)) {
			int32_t glyph_a = TS->font_get_glyph_index(rids[i], p_size, p_char, 0);
			Size2 ret = Size2(TS->font_get_glyph_advance(rids[i], p_size, glyph_a).x, TS->font_get_ascent(rids[i], p_size) + TS->font_get_descent(rids[i], p_size));
			if ((p_next != 0) && data[i]->has_char(p_next)) {
				int32_t glyph_b = TS->font_get_glyph_index(rids[i], p_size, p_next, 0);
				ret.x -= TS->font_get_kerning(rids[i], p_size, Vector2i(glyph_a, glyph_b)).x;
			}
			return ret;
		}
	}
	return Size2();
}

real_t Font::draw_char(RID p_canvas_item, const Point2 &p_pos, char32_t p_char, char32_t p_next, int p_size, const Color &p_modulate, int p_outline_size, const Color &p_outline_modulate) const {
	for (int i = 0; i < data.size(); i++) {
		_ensure_rid(i);
		if (data[i]->has_char(p_char)) {
			int32_t glyph_a = TS->font_get_glyph_index(rids[i], p_size, p_char, 0);
			real_t ret = TS->font_get_glyph_advance(rids[i], p_size, glyph_a).x;
			if ((p_next != 0) && data[i]->has_char(p_next)) {
				int32_t glyph_b = TS->font_get_glyph_index(rids[i], p_size, p_next, 0);
				ret -= TS->font_get_kerning(rids[i], p_size, Vector2i(glyph_a, glyph_b)).x;
			}

			if (p_outline_size > 0 && p_outline_modulate.a != 0.0f) {
				TS->font_draw_glyph_outline(rids[i], p_canvas_item, p_size, p_outline_size, p_pos, glyph_a, p_outline_modulate);
			}
			TS->font_draw_glyph(rids[i], p_canvas_item, p_size, p_pos, glyph_a, p_modulate);
			return ret;
		}
	}
	return 0;
}

bool Font::has_char(char32_t p_char) const {
	for (int i = 0; i < data.size(); i++) {
		if (data[i]->has_char(p_char))
			return true;
	}
	return false;
}

String Font::get_supported_chars() const {
	String chars;
	for (int i = 0; i < data.size(); i++) {
		String data_chars = data[i]->get_supported_chars();
		for (int j = 0; j < data_chars.length(); j++) {
			if (chars.find_char(data_chars[j]) == -1) {
				chars += data_chars[j];
			}
		}
	}
	return chars;
}

Vector<RID> Font::get_rids() const {
	for (int i = 0; i < data.size(); i++) {
		_ensure_rid(i);
	}
	return rids;
}

void Font::update_changes() {
	emit_changed();
}

Font::Font() {
	cache.set_capacity(128);
	cache_wrap.set_capacity(32);
}

Font::~Font() {
	clear_data();
	cache.clear();
	cache_wrap.clear();
}
