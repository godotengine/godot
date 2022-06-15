/*************************************************************************/
/*  font.cpp                                                             */
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

#include "font.h"

#include "core/io/image_loader.h"
#include "core/io/resource_loader.h"
#include "core/string/translation.h"
#include "core/templates/hash_map.h"
#include "core/templates/hashfuncs.h"
#include "scene/resources/text_line.h"
#include "scene/resources/text_paragraph.h"
#include "scene/resources/theme.h"

/*************************************************************************/
/*  Font                                                                 */
/*************************************************************************/

_FORCE_INLINE_ void Font::_clear_cache() {
	for (int i = 0; i < cache.size(); i++) {
		if (cache[i].is_valid()) {
			TS->free_rid(cache[i]);
			cache.write[i] = RID();
		}
	}
}

_FORCE_INLINE_ void Font::_ensure_rid(int p_cache_index) const {
	if (unlikely(p_cache_index >= cache.size())) {
		cache.resize(p_cache_index + 1);
	}
	if (unlikely(!cache[p_cache_index].is_valid())) {
		cache.write[p_cache_index] = TS->create_font();
		TS->font_set_data_ptr(cache[p_cache_index], data_ptr, data_size);
		TS->font_set_face_index(cache[p_cache_index], face_index);
		TS->font_set_antialiased(cache[p_cache_index], antialiased);
		TS->font_set_generate_mipmaps(cache[p_cache_index], mipmaps);
		TS->font_set_multichannel_signed_distance_field(cache[p_cache_index], msdf);
		TS->font_set_msdf_pixel_range(cache[p_cache_index], msdf_pixel_range);
		TS->font_set_msdf_size(cache[p_cache_index], msdf_size);
		TS->font_set_fixed_size(cache[p_cache_index], fixed_size);
		TS->font_set_force_autohinter(cache[p_cache_index], force_autohinter);
		TS->font_set_hinting(cache[p_cache_index], hinting);
		TS->font_set_subpixel_positioning(cache[p_cache_index], subpixel_positioning);
		TS->font_set_oversampling(cache[p_cache_index], oversampling);
	}
}

void Font::_bind_methods() {
	ClassDB::bind_method(D_METHOD("load_bitmap_font", "path"), &Font::load_bitmap_font);
	ClassDB::bind_method(D_METHOD("load_dynamic_font", "path"), &Font::load_dynamic_font);

	ClassDB::bind_method(D_METHOD("set_data", "data"), &Font::set_data);
	ClassDB::bind_method(D_METHOD("get_data"), &Font::get_data);

	ClassDB::bind_method(D_METHOD("set_antialiased", "antialiased"), &Font::set_antialiased);
	ClassDB::bind_method(D_METHOD("is_antialiased"), &Font::is_antialiased);

	ClassDB::bind_method(D_METHOD("set_generate_mipmaps", "generate_mipmaps"), &Font::set_generate_mipmaps);
	ClassDB::bind_method(D_METHOD("get_generate_mipmaps"), &Font::get_generate_mipmaps);

	ClassDB::bind_method(D_METHOD("set_font_name", "name"), &Font::set_font_name);
	ClassDB::bind_method(D_METHOD("get_font_name"), &Font::get_font_name);

	ClassDB::bind_method(D_METHOD("set_font_style_name", "name"), &Font::set_font_style_name);
	ClassDB::bind_method(D_METHOD("get_font_style_name"), &Font::get_font_style_name);

	ClassDB::bind_method(D_METHOD("set_font_style", "style"), &Font::set_font_style);
	ClassDB::bind_method(D_METHOD("get_font_style"), &Font::get_font_style);

	ClassDB::bind_method(D_METHOD("set_multichannel_signed_distance_field", "msdf"), &Font::set_multichannel_signed_distance_field);
	ClassDB::bind_method(D_METHOD("is_multichannel_signed_distance_field"), &Font::is_multichannel_signed_distance_field);

	ClassDB::bind_method(D_METHOD("set_msdf_pixel_range", "msdf_pixel_range"), &Font::set_msdf_pixel_range);
	ClassDB::bind_method(D_METHOD("get_msdf_pixel_range"), &Font::get_msdf_pixel_range);

	ClassDB::bind_method(D_METHOD("set_msdf_size", "msdf_size"), &Font::set_msdf_size);
	ClassDB::bind_method(D_METHOD("get_msdf_size"), &Font::get_msdf_size);

	ClassDB::bind_method(D_METHOD("set_fixed_size", "fixed_size"), &Font::set_fixed_size);
	ClassDB::bind_method(D_METHOD("get_fixed_size"), &Font::get_fixed_size);

	ClassDB::bind_method(D_METHOD("set_force_autohinter", "force_autohinter"), &Font::set_force_autohinter);
	ClassDB::bind_method(D_METHOD("is_force_autohinter"), &Font::is_force_autohinter);

	ClassDB::bind_method(D_METHOD("set_hinting", "hinting"), &Font::set_hinting);
	ClassDB::bind_method(D_METHOD("get_hinting"), &Font::get_hinting);

	ClassDB::bind_method(D_METHOD("set_subpixel_positioning", "subpixel_positioning"), &Font::set_subpixel_positioning);
	ClassDB::bind_method(D_METHOD("get_subpixel_positioning"), &Font::get_subpixel_positioning);

	ClassDB::bind_method(D_METHOD("set_oversampling", "oversampling"), &Font::set_oversampling);
	ClassDB::bind_method(D_METHOD("get_oversampling"), &Font::get_oversampling);

	ClassDB::bind_method(D_METHOD("set_fallbacks", "fallbacks"), &Font::set_fallbacks);
	ClassDB::bind_method(D_METHOD("get_fallbacks"), &Font::get_fallbacks);

	ClassDB::bind_method(D_METHOD("find_cache", "variation_coordinates", "face_index", "embolden", "transform"), &Font::find_cache, DEFVAL(0), DEFVAL(0.0), DEFVAL(Transform2D()));

	ClassDB::bind_method(D_METHOD("get_cache_count"), &Font::get_cache_count);
	ClassDB::bind_method(D_METHOD("clear_cache"), &Font::clear_cache);
	ClassDB::bind_method(D_METHOD("remove_cache", "cache_index"), &Font::remove_cache);

	ClassDB::bind_method(D_METHOD("get_size_cache_list", "cache_index"), &Font::get_size_cache_list);
	ClassDB::bind_method(D_METHOD("clear_size_cache", "cache_index"), &Font::clear_size_cache);
	ClassDB::bind_method(D_METHOD("remove_size_cache", "cache_index", "size"), &Font::remove_size_cache);

	ClassDB::bind_method(D_METHOD("set_variation_coordinates", "cache_index", "variation_coordinates"), &Font::set_variation_coordinates);
	ClassDB::bind_method(D_METHOD("get_variation_coordinates", "cache_index"), &Font::get_variation_coordinates);

	ClassDB::bind_method(D_METHOD("set_embolden", "cache_index", "strength"), &Font::set_embolden);
	ClassDB::bind_method(D_METHOD("get_embolden", "cache_index"), &Font::get_embolden);

	ClassDB::bind_method(D_METHOD("set_transform", "cache_index", "transform"), &Font::set_transform);
	ClassDB::bind_method(D_METHOD("get_transform", "cache_index"), &Font::get_transform);

	ClassDB::bind_method(D_METHOD("set_face_index", "cache_index", "face_index"), &Font::set_face_index);
	ClassDB::bind_method(D_METHOD("get_face_index", "cache_index"), &Font::get_face_index);

	ClassDB::bind_method(D_METHOD("get_face_count"), &Font::get_face_count);

	ClassDB::bind_method(D_METHOD("set_ascent", "cache_index", "size", "ascent"), &Font::set_ascent);
	ClassDB::bind_method(D_METHOD("get_ascent", "cache_index", "size"), &Font::get_ascent);

	ClassDB::bind_method(D_METHOD("set_descent", "cache_index", "size", "descent"), &Font::set_descent);
	ClassDB::bind_method(D_METHOD("get_descent", "cache_index", "size"), &Font::get_descent);

	ClassDB::bind_method(D_METHOD("set_underline_position", "cache_index", "size", "underline_position"), &Font::set_underline_position);
	ClassDB::bind_method(D_METHOD("get_underline_position", "cache_index", "size"), &Font::get_underline_position);

	ClassDB::bind_method(D_METHOD("set_underline_thickness", "cache_index", "size", "underline_thickness"), &Font::set_underline_thickness);
	ClassDB::bind_method(D_METHOD("get_underline_thickness", "cache_index", "size"), &Font::get_underline_thickness);

	ClassDB::bind_method(D_METHOD("set_scale", "cache_index", "size", "scale"), &Font::set_scale);
	ClassDB::bind_method(D_METHOD("get_scale", "cache_index", "size"), &Font::get_scale);

	ClassDB::bind_method(D_METHOD("get_texture_count", "cache_index", "size"), &Font::get_texture_count);
	ClassDB::bind_method(D_METHOD("clear_textures", "cache_index", "size"), &Font::clear_textures);
	ClassDB::bind_method(D_METHOD("remove_texture", "cache_index", "size", "texture_index"), &Font::remove_texture);

	ClassDB::bind_method(D_METHOD("set_texture_image", "cache_index", "size", "texture_index", "image"), &Font::set_texture_image);
	ClassDB::bind_method(D_METHOD("get_texture_image", "cache_index", "size", "texture_index"), &Font::get_texture_image);

	ClassDB::bind_method(D_METHOD("set_texture_offsets", "cache_index", "size", "texture_index", "offset"), &Font::set_texture_offsets);
	ClassDB::bind_method(D_METHOD("get_texture_offsets", "cache_index", "size", "texture_index"), &Font::get_texture_offsets);

	ClassDB::bind_method(D_METHOD("get_glyph_list", "cache_index", "size"), &Font::get_glyph_list);
	ClassDB::bind_method(D_METHOD("clear_glyphs", "cache_index", "size"), &Font::clear_glyphs);
	ClassDB::bind_method(D_METHOD("remove_glyph", "cache_index", "size", "glyph"), &Font::remove_glyph);

	ClassDB::bind_method(D_METHOD("set_glyph_advance", "cache_index", "size", "glyph", "advance"), &Font::set_glyph_advance);
	ClassDB::bind_method(D_METHOD("get_glyph_advance", "cache_index", "size", "glyph"), &Font::get_glyph_advance);

	ClassDB::bind_method(D_METHOD("set_glyph_offset", "cache_index", "size", "glyph", "offset"), &Font::set_glyph_offset);
	ClassDB::bind_method(D_METHOD("get_glyph_offset", "cache_index", "size", "glyph"), &Font::get_glyph_offset);

	ClassDB::bind_method(D_METHOD("set_glyph_size", "cache_index", "size", "glyph", "gl_size"), &Font::set_glyph_size);
	ClassDB::bind_method(D_METHOD("get_glyph_size", "cache_index", "size", "glyph"), &Font::get_glyph_size);

	ClassDB::bind_method(D_METHOD("set_glyph_uv_rect", "cache_index", "size", "glyph", "uv_rect"), &Font::set_glyph_uv_rect);
	ClassDB::bind_method(D_METHOD("get_glyph_uv_rect", "cache_index", "size", "glyph"), &Font::get_glyph_uv_rect);

	ClassDB::bind_method(D_METHOD("set_glyph_texture_idx", "cache_index", "size", "glyph", "texture_idx"), &Font::set_glyph_texture_idx);
	ClassDB::bind_method(D_METHOD("get_glyph_texture_idx", "cache_index", "size", "glyph"), &Font::get_glyph_texture_idx);

	ClassDB::bind_method(D_METHOD("get_kerning_list", "cache_index", "size"), &Font::get_kerning_list);
	ClassDB::bind_method(D_METHOD("clear_kerning_map", "cache_index", "size"), &Font::clear_kerning_map);
	ClassDB::bind_method(D_METHOD("remove_kerning", "cache_index", "size", "glyph_pair"), &Font::remove_kerning);

	ClassDB::bind_method(D_METHOD("set_kerning", "cache_index", "size", "glyph_pair", "kerning"), &Font::set_kerning);
	ClassDB::bind_method(D_METHOD("get_kerning", "cache_index", "size", "glyph_pair"), &Font::get_kerning);

	ClassDB::bind_method(D_METHOD("render_range", "cache_index", "size", "start", "end"), &Font::render_range);
	ClassDB::bind_method(D_METHOD("render_glyph", "cache_index", "size", "index"), &Font::render_glyph);

	ClassDB::bind_method(D_METHOD("get_cache_rid", "cache_index"), &Font::get_cache_rid);

	ClassDB::bind_method(D_METHOD("is_language_supported", "language"), &Font::is_language_supported);
	ClassDB::bind_method(D_METHOD("set_language_support_override", "language", "supported"), &Font::set_language_support_override);
	ClassDB::bind_method(D_METHOD("get_language_support_override", "language"), &Font::get_language_support_override);
	ClassDB::bind_method(D_METHOD("remove_language_support_override", "language"), &Font::remove_language_support_override);
	ClassDB::bind_method(D_METHOD("get_language_support_overrides"), &Font::get_language_support_overrides);

	ClassDB::bind_method(D_METHOD("is_script_supported", "script"), &Font::is_script_supported);
	ClassDB::bind_method(D_METHOD("set_script_support_override", "script", "supported"), &Font::set_script_support_override);
	ClassDB::bind_method(D_METHOD("get_script_support_override", "script"), &Font::get_script_support_override);
	ClassDB::bind_method(D_METHOD("remove_script_support_override", "script"), &Font::remove_script_support_override);
	ClassDB::bind_method(D_METHOD("get_script_support_overrides"), &Font::get_script_support_overrides);

	ClassDB::bind_method(D_METHOD("set_opentype_feature_overrides", "overrides"), &Font::set_opentype_feature_overrides);
	ClassDB::bind_method(D_METHOD("get_opentype_feature_overrides"), &Font::get_opentype_feature_overrides);

	ClassDB::bind_method(D_METHOD("has_char", "char"), &Font::has_char);
	ClassDB::bind_method(D_METHOD("get_supported_chars"), &Font::get_supported_chars);

	ClassDB::bind_method(D_METHOD("get_glyph_index", "size", "char", "variation_selector"), &Font::get_glyph_index);

	ClassDB::bind_method(D_METHOD("get_supported_feature_list"), &Font::get_supported_feature_list);
	ClassDB::bind_method(D_METHOD("get_supported_variation_list"), &Font::get_supported_variation_list);

	ADD_PROPERTY(PropertyInfo(Variant::PACKED_BYTE_ARRAY, "data", PROPERTY_HINT_NONE, "", PROPERTY_USAGE_STORAGE), "set_data", "get_data");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "generate_mipmaps", PROPERTY_HINT_NONE, "", PROPERTY_USAGE_STORAGE), "set_generate_mipmaps", "get_generate_mipmaps");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "antialiased", PROPERTY_HINT_NONE, "", PROPERTY_USAGE_STORAGE), "set_antialiased", "is_antialiased");
	ADD_PROPERTY(PropertyInfo(Variant::STRING, "font_name", PROPERTY_HINT_NONE, "", PROPERTY_USAGE_STORAGE), "set_font_name", "get_font_name");
	ADD_PROPERTY(PropertyInfo(Variant::STRING, "style_name", PROPERTY_HINT_NONE, "", PROPERTY_USAGE_STORAGE), "set_font_style_name", "get_font_style_name");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "font_style", PROPERTY_HINT_NONE, "", PROPERTY_USAGE_STORAGE), "set_font_style", "get_font_style");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "subpixel_positioning", PROPERTY_HINT_ENUM, "Disabled,Auto,One half of a pixel,One quarter of a pixel", PROPERTY_USAGE_STORAGE), "set_subpixel_positioning", "get_subpixel_positioning");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "multichannel_signed_distance_field", PROPERTY_HINT_NONE, "", PROPERTY_USAGE_STORAGE), "set_multichannel_signed_distance_field", "is_multichannel_signed_distance_field");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "msdf_pixel_range", PROPERTY_HINT_NONE, "", PROPERTY_USAGE_STORAGE), "set_msdf_pixel_range", "get_msdf_pixel_range");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "msdf_size", PROPERTY_HINT_NONE, "", PROPERTY_USAGE_STORAGE), "set_msdf_size", "get_msdf_size");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "force_autohinter", PROPERTY_HINT_NONE, "", PROPERTY_USAGE_STORAGE), "set_force_autohinter", "is_force_autohinter");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "hinting", PROPERTY_HINT_ENUM, "None,Light,Normal", PROPERTY_USAGE_STORAGE), "set_hinting", "get_hinting");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "oversampling", PROPERTY_HINT_NONE, "", PROPERTY_USAGE_STORAGE), "set_oversampling", "get_oversampling");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "fixed_size", PROPERTY_HINT_NONE, "", PROPERTY_USAGE_STORAGE), "set_fixed_size", "get_fixed_size");
	ADD_PROPERTY(PropertyInfo(Variant::DICTIONARY, "opentype_feature_overrides", PROPERTY_HINT_NONE, "", PROPERTY_USAGE_STORAGE), "set_opentype_feature_overrides", "get_opentype_feature_overrides");
	ADD_PROPERTY(PropertyInfo(Variant::ARRAY, "fallbacks", PROPERTY_HINT_ARRAY_TYPE, vformat("%s/%s:%s", Variant::OBJECT, PROPERTY_HINT_RESOURCE_TYPE, "Font")), "set_fallbacks", "get_fallbacks");
}

bool Font::_set(const StringName &p_name, const Variant &p_value) {
	Vector<String> tokens = p_name.operator String().split("/");

#ifndef DISABLE_DEPRECATED
	if (tokens.size() == 1 && tokens[0] == "font_path") {
		// Compatibility, DynamicFontData.
		load_dynamic_font(p_value);
	} else if (tokens.size() == 1 && tokens[0] == "override_oversampling") {
		set_oversampling(p_value);
	}
#endif

	if (tokens.size() == 2 && tokens[0] == "language_support_override") {
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
		} else if (tokens.size() == 3 && tokens[2] == "embolden") {
			set_embolden(cache_index, p_value);
			return true;
		} else if (tokens.size() == 3 && tokens[2] == "face_index") {
			set_face_index(cache_index, p_value);
			return true;
		} else if (tokens.size() == 3 && tokens[2] == "transform") {
			set_transform(cache_index, p_value);
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

bool Font::_get(const StringName &p_name, Variant &r_ret) const {
	Vector<String> tokens = p_name.operator String().split("/");
	if (tokens.size() == 2 && tokens[0] == "language_support_override") {
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
		} else if (tokens.size() == 3 && tokens[2] == "embolden") {
			r_ret = get_embolden(cache_index);
			return true;
		} else if (tokens.size() == 3 && tokens[2] == "face_index") {
			r_ret = get_face_index(cache_index);
			return true;
		} else if (tokens.size() == 3 && tokens[2] == "transform") {
			r_ret = get_transform(cache_index);
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

void Font::_get_property_list(List<PropertyInfo> *p_list) const {
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
		p_list->push_back(PropertyInfo(Variant::INT, "face_index", PROPERTY_HINT_NONE, "", PROPERTY_USAGE_STORAGE));
		p_list->push_back(PropertyInfo(Variant::FLOAT, "embolden", PROPERTY_HINT_RANGE, "-2,2,0.01", PROPERTY_USAGE_STORAGE));
		p_list->push_back(PropertyInfo(Variant::TRANSFORM2D, "transform", PROPERTY_HINT_NONE, "", PROPERTY_USAGE_STORAGE));

		for (int j = 0; j < sizes.size(); j++) {
			Vector2i sz = sizes[j];
			String prefix_sz = prefix + itos(sz.x) + "/" + itos(sz.y) + "/";
			if (sz.y == 0) {
				p_list->push_back(PropertyInfo(Variant::FLOAT, prefix_sz + "ascent", PROPERTY_HINT_NONE, "", PROPERTY_USAGE_STORAGE));
				p_list->push_back(PropertyInfo(Variant::FLOAT, prefix_sz + "descent", PROPERTY_HINT_NONE, "", PROPERTY_USAGE_STORAGE));
				p_list->push_back(PropertyInfo(Variant::FLOAT, prefix_sz + "underline_position", PROPERTY_HINT_NONE, "", PROPERTY_USAGE_STORAGE));
				p_list->push_back(PropertyInfo(Variant::FLOAT, prefix_sz + "underline_thickness", PROPERTY_HINT_NONE, "", PROPERTY_USAGE_STORAGE));
				p_list->push_back(PropertyInfo(Variant::FLOAT, prefix_sz + "scale", PROPERTY_HINT_NONE, "", PROPERTY_USAGE_STORAGE));
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

void Font::_fallback_changed() {
	emit_changed();
	notify_property_list_changed();
}

void Font::reset_state() {
	_clear_cache();
	data.clear();
	data_ptr = nullptr;
	data_size = 0;
	face_index = 0;
	cache.clear();

	antialiased = true;
	mipmaps = false;
	msdf = false;
	force_autohinter = false;
	hinting = TextServer::HINTING_LIGHT;
	subpixel_positioning = TextServer::SUBPIXEL_POSITIONING_DISABLED;
	msdf_pixel_range = 14;
	msdf_size = 128;
	fixed_size = 0;
	oversampling = 0.f;
}

void Font::_convert_packed_8bit(Ref<Image> &p_source, int p_page, int p_sz) {
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

void Font::_convert_packed_4bit(Ref<Image> &p_source, int p_page, int p_sz) {
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

void Font::_convert_rgba_4bit(Ref<Image> &p_source, int p_page, int p_sz) {
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

void Font::_convert_mono_8bit(Ref<Image> &p_source, int p_page, int p_ch, int p_sz, int p_ol) {
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

void Font::_convert_mono_4bit(Ref<Image> &p_source, int p_page, int p_ch, int p_sz, int p_ol) {
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

Error Font::load_bitmap_font(const String &p_path) {
	reset_state();

	antialiased = false;
	mipmaps = false;
	msdf = false;
	force_autohinter = false;
	hinting = TextServer::HINTING_NONE;
	oversampling = 1.0f;

	Ref<FileAccess> f = FileAccess::open(p_path, FileAccess::READ);
	ERR_FAIL_COND_V_MSG(f.is_null(), ERR_CANT_CREATE, vformat(RTR("Cannot open font from file: %s."), p_path));

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
		ERR_FAIL_COND_V_MSG(magic[3] != 3, ERR_CANT_CREATE, vformat(RTR("Version %d of BMFont is not supported."), (int)magic[3]));

		uint8_t block_type = f->get_8();
		uint32_t block_size = f->get_32();
		while (!f->eof_reached()) {
			uint64_t off = f->get_position();
			switch (block_type) {
				case 1: /* info */ {
					ERR_FAIL_COND_V_MSG(block_size < 15, ERR_CANT_CREATE, RTR("Invalid BMFont info block size."));
					base_size = f->get_16();
					uint8_t flags = f->get_8();
					ERR_FAIL_COND_V_MSG(flags & 0x02, ERR_CANT_CREATE, RTR("Non-unicode version of BMFont is not supported."));
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
					ERR_FAIL_COND_V_MSG(block_size != 15, ERR_CANT_CREATE, RTR("Invalid BMFont common block size."));
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
								ERR_FAIL_COND_V_MSG(err != OK, ERR_FILE_CANT_READ, vformat(RTR("Can't load font texture: %s."), file));

								if (packed) {
									if (ch[3] == 0) { // 4 x 8 bit monochrome, no outline
										outline = 0;
										ERR_FAIL_COND_V_MSG(img->get_format() != Image::FORMAT_RGBA8, ERR_FILE_CANT_READ, RTR("Unsupported BMFont texture format."));
										_convert_packed_8bit(img, page, base_size);
									} else if ((ch[3] == 2) && (outline > 0)) { // 4 x 4 bit monochrome, gl + outline
										ERR_FAIL_COND_V_MSG(img->get_format() != Image::FORMAT_RGBA8, ERR_FILE_CANT_READ, RTR("Unsupported BMFont texture format."));
										_convert_packed_4bit(img, page, base_size);
									} else {
										ERR_FAIL_V_MSG(ERR_CANT_CREATE, RTR("Unsupported BMFont texture format."));
									}
								} else {
									if ((ch[0] == 0) && (ch[1] == 0) && (ch[2] == 0) && (ch[3] == 0)) { // RGBA8 color, no outline
										outline = 0;
										ERR_FAIL_COND_V_MSG(img->get_format() != Image::FORMAT_RGBA8, ERR_FILE_CANT_READ, RTR("Unsupported BMFont texture format."));
										set_texture_image(0, Vector2i(base_size, 0), page, img);
									} else if ((ch[0] == 2) && (ch[1] == 2) && (ch[2] == 2) && (ch[3] == 2) && (outline > 0)) { // RGBA4 color, gl + outline
										ERR_FAIL_COND_V_MSG(img->get_format() != Image::FORMAT_RGBA8, ERR_FILE_CANT_READ, RTR("Unsupported BMFont texture format."));
										_convert_rgba_4bit(img, page, base_size);
									} else if ((first_gl_ch >= 0) && (first_ol_ch >= 0) && (outline > 0)) { // 1 x 8 bit monochrome, gl + outline
										ERR_FAIL_COND_V_MSG(img->get_format() != Image::FORMAT_RGBA8 && img->get_format() != Image::FORMAT_L8, ERR_FILE_CANT_READ, RTR("Unsupported BMFont texture format."));
										_convert_mono_8bit(img, page, first_gl_ch, base_size, 0);
										_convert_mono_8bit(img, page, first_ol_ch, base_size, 1);
									} else if ((first_cm_ch >= 0) && (outline > 0)) { // 1 x 4 bit monochrome, gl + outline
										ERR_FAIL_COND_V_MSG(img->get_format() != Image::FORMAT_RGBA8 && img->get_format() != Image::FORMAT_L8, ERR_FILE_CANT_READ, RTR("Unsupported BMFont texture format."));
										_convert_mono_4bit(img, page, first_cm_ch, base_size, 1);
									} else if (first_gl_ch >= 0) { // 1 x 8 bit monochrome, no outline
										outline = 0;
										ERR_FAIL_COND_V_MSG(img->get_format() != Image::FORMAT_RGBA8 && img->get_format() != Image::FORMAT_L8, ERR_FILE_CANT_READ, RTR("Unsupported BMFont texture format."));
										_convert_mono_8bit(img, page, first_gl_ch, base_size, 0);
									} else {
										ERR_FAIL_V_MSG(ERR_CANT_CREATE, RTR("Unsupported BMFont texture format."));
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

						ERR_FAIL_COND_V_MSG(!packed && channel != 15, ERR_CANT_CREATE, RTR("Invalid glyph channel."));
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
					ERR_FAIL_V_MSG(ERR_CANT_CREATE, RTR("Invalid BMFont block type."));
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
			HashMap<String, String> keys;

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
				ERR_FAIL_COND_V_MSG((!keys.has("unicode") || keys["unicode"].to_int() != 1), ERR_CANT_CREATE, RTR("Non-unicode version of BMFont is not supported."));
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
						ERR_FAIL_COND_V_MSG(err != OK, ERR_FILE_CANT_READ, vformat(RTR("Can't load font texture: %s."), file));
						if (packed) {
							if (ch[3] == 0) { // 4 x 8 bit monochrome, no outline
								outline = 0;
								ERR_FAIL_COND_V_MSG(img->get_format() != Image::FORMAT_RGBA8, ERR_FILE_CANT_READ, RTR("Unsupported BMFont texture format."));
								_convert_packed_8bit(img, page, base_size);
							} else if ((ch[3] == 2) && (outline > 0)) { // 4 x 4 bit monochrome, gl + outline
								ERR_FAIL_COND_V_MSG(img->get_format() != Image::FORMAT_RGBA8, ERR_FILE_CANT_READ, RTR("Unsupported BMFont texture format."));
								_convert_packed_4bit(img, page, base_size);
							} else {
								ERR_FAIL_V_MSG(ERR_CANT_CREATE, RTR("Unsupported BMFont texture format."));
							}
						} else {
							if ((ch[0] == 0) && (ch[1] == 0) && (ch[2] == 0) && (ch[3] == 0)) { // RGBA8 color, no outline
								outline = 0;
								ERR_FAIL_COND_V_MSG(img->get_format() != Image::FORMAT_RGBA8, ERR_FILE_CANT_READ, RTR("Unsupported BMFont texture format."));
								set_texture_image(0, Vector2i(base_size, 0), page, img);
							} else if ((ch[0] == 2) && (ch[1] == 2) && (ch[2] == 2) && (ch[3] == 2) && (outline > 0)) { // RGBA4 color, gl + outline
								ERR_FAIL_COND_V_MSG(img->get_format() != Image::FORMAT_RGBA8, ERR_FILE_CANT_READ, RTR("Unsupported BMFont texture format."));
								_convert_rgba_4bit(img, page, base_size);
							} else if ((first_gl_ch >= 0) && (first_ol_ch >= 0) && (outline > 0)) { // 1 x 8 bit monochrome, gl + outline
								ERR_FAIL_COND_V_MSG(img->get_format() != Image::FORMAT_RGBA8 && img->get_format() != Image::FORMAT_L8, ERR_FILE_CANT_READ, RTR("Unsupported BMFont texture format."));
								_convert_mono_8bit(img, page, first_gl_ch, base_size, 0);
								_convert_mono_8bit(img, page, first_ol_ch, base_size, 1);
							} else if ((first_cm_ch >= 0) && (outline > 0)) { // 1 x 4 bit monochrome, gl + outline
								ERR_FAIL_COND_V_MSG(img->get_format() != Image::FORMAT_RGBA8 && img->get_format() != Image::FORMAT_L8, ERR_FILE_CANT_READ, RTR("Unsupported BMFont texture format."));
								_convert_mono_4bit(img, page, first_cm_ch, base_size, 1);
							} else if (first_gl_ch >= 0) { // 1 x 8 bit monochrome, no outline
								outline = 0;
								ERR_FAIL_COND_V_MSG(img->get_format() != Image::FORMAT_RGBA8 && img->get_format() != Image::FORMAT_L8, ERR_FILE_CANT_READ, RTR("Unsupported BMFont texture format."));
								_convert_mono_8bit(img, page, first_gl_ch, base_size, 0);
							} else {
								ERR_FAIL_V_MSG(ERR_CANT_CREATE, RTR("Unsupported BMFont texture format."));
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

				ERR_FAIL_COND_V_MSG(!packed && channel != 15, ERR_CANT_CREATE, RTR("Invalid glyph channel."));
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

Error Font::load_dynamic_font(const String &p_path) {
	reset_state();

	Vector<uint8_t> data = FileAccess::get_file_as_array(p_path);
	set_data(data);

	return OK;
}

void Font::set_data_ptr(const uint8_t *p_data, size_t p_size) {
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

void Font::set_data(const PackedByteArray &p_data) {
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

PackedByteArray Font::get_data() const {
	if (unlikely((size_t)data.size() != data_size)) {
		PackedByteArray *data_w = const_cast<PackedByteArray *>(&data);
		data_w->resize(data_size);
		memcpy(data_w->ptrw(), data_ptr, data_size);
	}
	return data;
}

void Font::set_font_name(const String &p_name) {
	_ensure_rid(0);
	TS->font_set_name(cache[0], p_name);
}

String Font::get_font_name() const {
	_ensure_rid(0);
	return TS->font_get_name(cache[0]);
}

void Font::set_font_style_name(const String &p_name) {
	_ensure_rid(0);
	TS->font_set_style_name(cache[0], p_name);
}

String Font::get_font_style_name() const {
	_ensure_rid(0);
	return TS->font_get_style_name(cache[0]);
}

void Font::set_font_style(uint32_t p_style) {
	_ensure_rid(0);
	TS->font_set_style(cache[0], p_style);
}

uint32_t Font::get_font_style() const {
	_ensure_rid(0);
	return TS->font_get_style(cache[0]);
}

void Font::set_antialiased(bool p_antialiased) {
	if (antialiased != p_antialiased) {
		antialiased = p_antialiased;
		for (int i = 0; i < cache.size(); i++) {
			_ensure_rid(i);
			TS->font_set_antialiased(cache[i], antialiased);
		}
		emit_changed();
	}
}

bool Font::is_antialiased() const {
	return antialiased;
}

void Font::set_generate_mipmaps(bool p_generate_mipmaps) {
	if (mipmaps != p_generate_mipmaps) {
		mipmaps = p_generate_mipmaps;
		for (int i = 0; i < cache.size(); i++) {
			_ensure_rid(i);
			TS->font_set_generate_mipmaps(cache[i], mipmaps);
		}
		emit_changed();
	}
}

bool Font::get_generate_mipmaps() const {
	return mipmaps;
}

void Font::set_multichannel_signed_distance_field(bool p_msdf) {
	if (msdf != p_msdf) {
		msdf = p_msdf;
		for (int i = 0; i < cache.size(); i++) {
			_ensure_rid(i);
			TS->font_set_multichannel_signed_distance_field(cache[i], msdf);
		}
		emit_changed();
	}
}

bool Font::is_multichannel_signed_distance_field() const {
	return msdf;
}

void Font::set_msdf_pixel_range(int p_msdf_pixel_range) {
	if (msdf_pixel_range != p_msdf_pixel_range) {
		msdf_pixel_range = p_msdf_pixel_range;
		for (int i = 0; i < cache.size(); i++) {
			_ensure_rid(i);
			TS->font_set_msdf_pixel_range(cache[i], msdf_pixel_range);
		}
		emit_changed();
	}
}

int Font::get_msdf_pixel_range() const {
	return msdf_pixel_range;
}

void Font::set_msdf_size(int p_msdf_size) {
	if (msdf_size != p_msdf_size) {
		msdf_size = p_msdf_size;
		for (int i = 0; i < cache.size(); i++) {
			_ensure_rid(i);
			TS->font_set_msdf_size(cache[i], msdf_size);
		}
		emit_changed();
	}
}

int Font::get_msdf_size() const {
	return msdf_size;
}

void Font::set_fixed_size(int p_fixed_size) {
	if (fixed_size != p_fixed_size) {
		fixed_size = p_fixed_size;
		for (int i = 0; i < cache.size(); i++) {
			_ensure_rid(i);
			TS->font_set_fixed_size(cache[i], fixed_size);
		}
		emit_changed();
	}
}

int Font::get_fixed_size() const {
	return fixed_size;
}

void Font::set_force_autohinter(bool p_force_autohinter) {
	if (force_autohinter != p_force_autohinter) {
		force_autohinter = p_force_autohinter;
		for (int i = 0; i < cache.size(); i++) {
			_ensure_rid(i);
			TS->font_set_force_autohinter(cache[i], force_autohinter);
		}
		emit_changed();
	}
}

bool Font::is_force_autohinter() const {
	return force_autohinter;
}

void Font::set_hinting(TextServer::Hinting p_hinting) {
	if (hinting != p_hinting) {
		hinting = p_hinting;
		for (int i = 0; i < cache.size(); i++) {
			_ensure_rid(i);
			TS->font_set_hinting(cache[i], hinting);
		}
		emit_changed();
	}
}

TextServer::Hinting Font::get_hinting() const {
	return hinting;
}

void Font::set_subpixel_positioning(TextServer::SubpixelPositioning p_subpixel) {
	if (subpixel_positioning != p_subpixel) {
		subpixel_positioning = p_subpixel;
		for (int i = 0; i < cache.size(); i++) {
			_ensure_rid(i);
			TS->font_set_subpixel_positioning(cache[i], subpixel_positioning);
		}
		emit_changed();
	}
}

TextServer::SubpixelPositioning Font::get_subpixel_positioning() const {
	return subpixel_positioning;
}

void Font::set_oversampling(real_t p_oversampling) {
	if (oversampling != p_oversampling) {
		oversampling = p_oversampling;
		for (int i = 0; i < cache.size(); i++) {
			_ensure_rid(i);
			TS->font_set_oversampling(cache[i], oversampling);
		}
		emit_changed();
	}
}

real_t Font::get_oversampling() const {
	return oversampling;
}

bool Font::_is_cyclic(const Ref<Font> &p_fb, int p_depth) const {
	ERR_FAIL_COND_V(p_depth > 64, false);
	for (int i = 0; i < p_fb->fallbacks.size(); i++) {
		const Ref<Font> &f = p_fb->fallbacks[i];
		if (f == this) {
			return true;
		}
		return _is_cyclic(f, p_depth + 1);
	}
	return false;
}

void Font::set_fallbacks(const TypedArray<Font> &p_fallbacks) {
	ERR_FAIL_COND(_is_cyclic(this, 0));
	for (int i = 0; i < fallbacks.size(); i++) {
		Ref<Font> fd = fallbacks[i];
		fd->disconnect(SNAME("changed"), callable_mp(this, &Font::_fallback_changed));
	}
	fallbacks = p_fallbacks;
	for (int i = 0; i < fallbacks.size(); i++) {
		Ref<Font> fd = fallbacks[i];
		fd->connect(SNAME("changed"), callable_mp(this, &Font::_fallback_changed), varray(), CONNECT_REFERENCE_COUNTED);
	}
}

TypedArray<Font> Font::get_fallbacks() const {
	return fallbacks;
}

RID Font::find_cache(const Dictionary &p_variation_coordinates, int p_face_index, float p_strength, Transform2D p_transform) const {
	// Find existing variation cache.
	const Dictionary &supported_coords = get_supported_variation_list();
	for (int i = 0; i < cache.size(); i++) {
		if (cache[i].is_valid()) {
			const Dictionary &cache_var = TS->font_get_variation_coordinates(cache[i]);
			bool match = true;
			match = match && (TS->font_get_face_index(cache[i]) == p_face_index);
			match = match && (TS->font_get_embolden(cache[i]) == p_strength);
			match = match && (TS->font_get_transform(cache[i]) == p_transform);
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
	TS->font_set_face_index(cache[idx], p_face_index);
	TS->font_set_embolden(cache[idx], p_strength);
	TS->font_set_transform(cache[idx], p_transform);
	return cache[idx];
}

int Font::get_cache_count() const {
	return cache.size();
}

void Font::clear_cache() {
	_clear_cache();
	cache.clear();
	emit_changed();
}

void Font::remove_cache(int p_cache_index) {
	ERR_FAIL_INDEX(p_cache_index, cache.size());
	if (cache[p_cache_index].is_valid()) {
		TS->free_rid(cache.write[p_cache_index]);
	}
	cache.remove_at(p_cache_index);
	emit_changed();
}

Array Font::get_size_cache_list(int p_cache_index) const {
	ERR_FAIL_COND_V(p_cache_index < 0, Array());
	_ensure_rid(p_cache_index);
	return TS->font_get_size_cache_list(cache[p_cache_index]);
}

void Font::clear_size_cache(int p_cache_index) {
	ERR_FAIL_COND(p_cache_index < 0);
	_ensure_rid(p_cache_index);
	TS->font_clear_size_cache(cache[p_cache_index]);
}

void Font::remove_size_cache(int p_cache_index, const Vector2i &p_size) {
	ERR_FAIL_COND(p_cache_index < 0);
	_ensure_rid(p_cache_index);
	TS->font_remove_size_cache(cache[p_cache_index], p_size);
}

void Font::set_variation_coordinates(int p_cache_index, const Dictionary &p_variation_coordinates) {
	ERR_FAIL_COND(p_cache_index < 0);
	_ensure_rid(p_cache_index);
	TS->font_set_variation_coordinates(cache[p_cache_index], p_variation_coordinates);
}

Dictionary Font::get_variation_coordinates(int p_cache_index) const {
	ERR_FAIL_COND_V(p_cache_index < 0, Dictionary());
	_ensure_rid(p_cache_index);
	return TS->font_get_variation_coordinates(cache[p_cache_index]);
}

void Font::set_embolden(int p_cache_index, float p_strength) {
	ERR_FAIL_COND(p_cache_index < 0);
	_ensure_rid(p_cache_index);
	TS->font_set_embolden(cache[p_cache_index], p_strength);
}

float Font::get_embolden(int p_cache_index) const {
	ERR_FAIL_COND_V(p_cache_index < 0, 0.f);
	_ensure_rid(p_cache_index);
	return TS->font_get_embolden(cache[p_cache_index]);
}

void Font::set_transform(int p_cache_index, Transform2D p_transform) {
	ERR_FAIL_COND(p_cache_index < 0);
	_ensure_rid(p_cache_index);
	TS->font_set_transform(cache[p_cache_index], p_transform);
}

Transform2D Font::get_transform(int p_cache_index) const {
	ERR_FAIL_COND_V(p_cache_index < 0, Transform2D());
	_ensure_rid(p_cache_index);
	return TS->font_get_transform(cache[p_cache_index]);
}

void Font::set_face_index(int p_cache_index, int64_t p_index) {
	ERR_FAIL_COND(p_cache_index < 0);
	ERR_FAIL_COND(p_index < 0);
	ERR_FAIL_COND(p_index >= 0x7FFF);

	_ensure_rid(p_cache_index);
	TS->font_set_face_index(cache[p_cache_index], p_index);
}

int64_t Font::get_face_index(int p_cache_index) const {
	ERR_FAIL_COND_V(p_cache_index < 0, 0);
	_ensure_rid(p_cache_index);
	return TS->font_get_face_index(cache[p_cache_index]);
}

int64_t Font::get_face_count() const {
	_ensure_rid(0);
	return TS->font_get_face_count(cache[0]);
}

void Font::set_ascent(int p_cache_index, int p_size, real_t p_ascent) {
	ERR_FAIL_COND(p_cache_index < 0);
	_ensure_rid(p_cache_index);
	TS->font_set_ascent(cache[p_cache_index], p_size, p_ascent);
}

real_t Font::get_ascent(int p_cache_index, int p_size) const {
	ERR_FAIL_COND_V(p_cache_index < 0, 0.f);
	_ensure_rid(p_cache_index);
	return TS->font_get_ascent(cache[p_cache_index], p_size);
}

void Font::set_descent(int p_cache_index, int p_size, real_t p_descent) {
	ERR_FAIL_COND(p_cache_index < 0);
	_ensure_rid(p_cache_index);
	TS->font_set_descent(cache[p_cache_index], p_size, p_descent);
}

real_t Font::get_descent(int p_cache_index, int p_size) const {
	ERR_FAIL_COND_V(p_cache_index < 0, 0.f);
	_ensure_rid(p_cache_index);
	return TS->font_get_descent(cache[p_cache_index], p_size);
}

void Font::set_underline_position(int p_cache_index, int p_size, real_t p_underline_position) {
	ERR_FAIL_COND(p_cache_index < 0);
	_ensure_rid(p_cache_index);
	TS->font_set_underline_position(cache[p_cache_index], p_size, p_underline_position);
}

real_t Font::get_underline_position(int p_cache_index, int p_size) const {
	ERR_FAIL_COND_V(p_cache_index < 0, 0.f);
	_ensure_rid(p_cache_index);
	return TS->font_get_underline_position(cache[p_cache_index], p_size);
}

void Font::set_underline_thickness(int p_cache_index, int p_size, real_t p_underline_thickness) {
	ERR_FAIL_COND(p_cache_index < 0);
	_ensure_rid(p_cache_index);
	TS->font_set_underline_thickness(cache[p_cache_index], p_size, p_underline_thickness);
}

real_t Font::get_underline_thickness(int p_cache_index, int p_size) const {
	ERR_FAIL_COND_V(p_cache_index < 0, 0.f);
	_ensure_rid(p_cache_index);
	return TS->font_get_underline_thickness(cache[p_cache_index], p_size);
}

void Font::set_scale(int p_cache_index, int p_size, real_t p_scale) {
	ERR_FAIL_COND(p_cache_index < 0);
	_ensure_rid(p_cache_index);
	TS->font_set_scale(cache[p_cache_index], p_size, p_scale);
}

real_t Font::get_scale(int p_cache_index, int p_size) const {
	ERR_FAIL_COND_V(p_cache_index < 0, 0.f);
	_ensure_rid(p_cache_index);
	return TS->font_get_scale(cache[p_cache_index], p_size);
}

int Font::get_texture_count(int p_cache_index, const Vector2i &p_size) const {
	ERR_FAIL_COND_V(p_cache_index < 0, 0);
	_ensure_rid(p_cache_index);
	return TS->font_get_texture_count(cache[p_cache_index], p_size);
}

void Font::clear_textures(int p_cache_index, const Vector2i &p_size) {
	ERR_FAIL_COND(p_cache_index < 0);
	_ensure_rid(p_cache_index);
	TS->font_clear_textures(cache[p_cache_index], p_size);
}

void Font::remove_texture(int p_cache_index, const Vector2i &p_size, int p_texture_index) {
	ERR_FAIL_COND(p_cache_index < 0);
	_ensure_rid(p_cache_index);
	TS->font_remove_texture(cache[p_cache_index], p_size, p_texture_index);
}

void Font::set_texture_image(int p_cache_index, const Vector2i &p_size, int p_texture_index, const Ref<Image> &p_image) {
	ERR_FAIL_COND(p_cache_index < 0);
	_ensure_rid(p_cache_index);
	TS->font_set_texture_image(cache[p_cache_index], p_size, p_texture_index, p_image);
}

Ref<Image> Font::get_texture_image(int p_cache_index, const Vector2i &p_size, int p_texture_index) const {
	ERR_FAIL_COND_V(p_cache_index < 0, Ref<Image>());
	_ensure_rid(p_cache_index);
	return TS->font_get_texture_image(cache[p_cache_index], p_size, p_texture_index);
}

void Font::set_texture_offsets(int p_cache_index, const Vector2i &p_size, int p_texture_index, const PackedInt32Array &p_offset) {
	ERR_FAIL_COND(p_cache_index < 0);
	_ensure_rid(p_cache_index);
	TS->font_set_texture_offsets(cache[p_cache_index], p_size, p_texture_index, p_offset);
}

PackedInt32Array Font::get_texture_offsets(int p_cache_index, const Vector2i &p_size, int p_texture_index) const {
	ERR_FAIL_COND_V(p_cache_index < 0, PackedInt32Array());
	_ensure_rid(p_cache_index);
	return TS->font_get_texture_offsets(cache[p_cache_index], p_size, p_texture_index);
}

Array Font::get_glyph_list(int p_cache_index, const Vector2i &p_size) const {
	ERR_FAIL_COND_V(p_cache_index < 0, Array());
	_ensure_rid(p_cache_index);
	return TS->font_get_glyph_list(cache[p_cache_index], p_size);
}

void Font::clear_glyphs(int p_cache_index, const Vector2i &p_size) {
	ERR_FAIL_COND(p_cache_index < 0);
	_ensure_rid(p_cache_index);
	TS->font_clear_glyphs(cache[p_cache_index], p_size);
}

void Font::remove_glyph(int p_cache_index, const Vector2i &p_size, int32_t p_glyph) {
	ERR_FAIL_COND(p_cache_index < 0);
	_ensure_rid(p_cache_index);
	TS->font_remove_glyph(cache[p_cache_index], p_size, p_glyph);
}

void Font::set_glyph_advance(int p_cache_index, int p_size, int32_t p_glyph, const Vector2 &p_advance) {
	ERR_FAIL_COND(p_cache_index < 0);
	_ensure_rid(p_cache_index);
	TS->font_set_glyph_advance(cache[p_cache_index], p_size, p_glyph, p_advance);
}

Vector2 Font::get_glyph_advance(int p_cache_index, int p_size, int32_t p_glyph) const {
	ERR_FAIL_COND_V(p_cache_index < 0, Vector2());
	_ensure_rid(p_cache_index);
	return TS->font_get_glyph_advance(cache[p_cache_index], p_size, p_glyph);
}

void Font::set_glyph_offset(int p_cache_index, const Vector2i &p_size, int32_t p_glyph, const Vector2 &p_offset) {
	ERR_FAIL_COND(p_cache_index < 0);
	_ensure_rid(p_cache_index);
	TS->font_set_glyph_offset(cache[p_cache_index], p_size, p_glyph, p_offset);
}

Vector2 Font::get_glyph_offset(int p_cache_index, const Vector2i &p_size, int32_t p_glyph) const {
	ERR_FAIL_COND_V(p_cache_index < 0, Vector2());
	_ensure_rid(p_cache_index);
	return TS->font_get_glyph_offset(cache[p_cache_index], p_size, p_glyph);
}

void Font::set_glyph_size(int p_cache_index, const Vector2i &p_size, int32_t p_glyph, const Vector2 &p_gl_size) {
	ERR_FAIL_COND(p_cache_index < 0);
	_ensure_rid(p_cache_index);
	TS->font_set_glyph_size(cache[p_cache_index], p_size, p_glyph, p_gl_size);
}

Vector2 Font::get_glyph_size(int p_cache_index, const Vector2i &p_size, int32_t p_glyph) const {
	ERR_FAIL_COND_V(p_cache_index < 0, Vector2());
	_ensure_rid(p_cache_index);
	return TS->font_get_glyph_size(cache[p_cache_index], p_size, p_glyph);
}

void Font::set_glyph_uv_rect(int p_cache_index, const Vector2i &p_size, int32_t p_glyph, const Rect2 &p_uv_rect) {
	ERR_FAIL_COND(p_cache_index < 0);
	_ensure_rid(p_cache_index);
	TS->font_set_glyph_uv_rect(cache[p_cache_index], p_size, p_glyph, p_uv_rect);
}

Rect2 Font::get_glyph_uv_rect(int p_cache_index, const Vector2i &p_size, int32_t p_glyph) const {
	ERR_FAIL_COND_V(p_cache_index < 0, Rect2());
	_ensure_rid(p_cache_index);
	return TS->font_get_glyph_uv_rect(cache[p_cache_index], p_size, p_glyph);
}

void Font::set_glyph_texture_idx(int p_cache_index, const Vector2i &p_size, int32_t p_glyph, int p_texture_idx) {
	ERR_FAIL_COND(p_cache_index < 0);
	_ensure_rid(p_cache_index);
	TS->font_set_glyph_texture_idx(cache[p_cache_index], p_size, p_glyph, p_texture_idx);
}

int Font::get_glyph_texture_idx(int p_cache_index, const Vector2i &p_size, int32_t p_glyph) const {
	ERR_FAIL_COND_V(p_cache_index < 0, 0);
	_ensure_rid(p_cache_index);
	return TS->font_get_glyph_texture_idx(cache[p_cache_index], p_size, p_glyph);
}

Array Font::get_kerning_list(int p_cache_index, int p_size) const {
	ERR_FAIL_COND_V(p_cache_index < 0, Array());
	_ensure_rid(p_cache_index);
	return TS->font_get_kerning_list(cache[p_cache_index], p_size);
}

void Font::clear_kerning_map(int p_cache_index, int p_size) {
	ERR_FAIL_COND(p_cache_index < 0);
	_ensure_rid(p_cache_index);
	TS->font_clear_kerning_map(cache[p_cache_index], p_size);
}

void Font::remove_kerning(int p_cache_index, int p_size, const Vector2i &p_glyph_pair) {
	ERR_FAIL_COND(p_cache_index < 0);
	_ensure_rid(p_cache_index);
	TS->font_remove_kerning(cache[p_cache_index], p_size, p_glyph_pair);
}

void Font::set_kerning(int p_cache_index, int p_size, const Vector2i &p_glyph_pair, const Vector2 &p_kerning) {
	ERR_FAIL_COND(p_cache_index < 0);
	_ensure_rid(p_cache_index);
	TS->font_set_kerning(cache[p_cache_index], p_size, p_glyph_pair, p_kerning);
}

Vector2 Font::get_kerning(int p_cache_index, int p_size, const Vector2i &p_glyph_pair) const {
	ERR_FAIL_COND_V(p_cache_index < 0, Vector2());
	_ensure_rid(p_cache_index);
	return TS->font_get_kerning(cache[p_cache_index], p_size, p_glyph_pair);
}

void Font::render_range(int p_cache_index, const Vector2i &p_size, char32_t p_start, char32_t p_end) {
	ERR_FAIL_COND(p_cache_index < 0);
	_ensure_rid(p_cache_index);
	TS->font_render_range(cache[p_cache_index], p_size, p_start, p_end);
}

void Font::render_glyph(int p_cache_index, const Vector2i &p_size, int32_t p_index) {
	ERR_FAIL_COND(p_cache_index < 0);
	_ensure_rid(p_cache_index);
	TS->font_render_glyph(cache[p_cache_index], p_size, p_index);
}

RID Font::get_cache_rid(int p_cache_index) const {
	ERR_FAIL_COND_V(p_cache_index < 0, RID());
	_ensure_rid(p_cache_index);
	return cache[p_cache_index];
}

bool Font::is_language_supported(const String &p_language) const {
	_ensure_rid(0);
	return TS->font_is_language_supported(cache[0], p_language);
}

void Font::set_language_support_override(const String &p_language, bool p_supported) {
	_ensure_rid(0);
	TS->font_set_language_support_override(cache[0], p_language, p_supported);
}

bool Font::get_language_support_override(const String &p_language) const {
	_ensure_rid(0);
	return TS->font_get_language_support_override(cache[0], p_language);
}

void Font::remove_language_support_override(const String &p_language) {
	_ensure_rid(0);
	TS->font_remove_language_support_override(cache[0], p_language);
}

Vector<String> Font::get_language_support_overrides() const {
	_ensure_rid(0);
	return TS->font_get_language_support_overrides(cache[0]);
}

bool Font::is_script_supported(const String &p_script) const {
	_ensure_rid(0);
	return TS->font_is_script_supported(cache[0], p_script);
}

void Font::set_script_support_override(const String &p_script, bool p_supported) {
	_ensure_rid(0);
	TS->font_set_script_support_override(cache[0], p_script, p_supported);
}

bool Font::get_script_support_override(const String &p_script) const {
	_ensure_rid(0);
	return TS->font_get_script_support_override(cache[0], p_script);
}

void Font::remove_script_support_override(const String &p_script) {
	_ensure_rid(0);
	TS->font_remove_script_support_override(cache[0], p_script);
}

Vector<String> Font::get_script_support_overrides() const {
	_ensure_rid(0);
	return TS->font_get_script_support_overrides(cache[0]);
}

void Font::set_opentype_feature_overrides(const Dictionary &p_overrides) {
	_ensure_rid(0);
	TS->font_set_opentype_feature_overrides(cache[0], p_overrides);
}

Dictionary Font::get_opentype_feature_overrides() const {
	_ensure_rid(0);
	return TS->font_get_opentype_feature_overrides(cache[0]);
}

bool Font::has_char(char32_t p_char) const {
	_ensure_rid(0);
	return TS->font_has_char(cache[0], p_char);
}

String Font::get_supported_chars() const {
	_ensure_rid(0);
	return TS->font_get_supported_chars(cache[0]);
}

int32_t Font::get_glyph_index(int p_size, char32_t p_char, char32_t p_variation_selector) const {
	_ensure_rid(0);
	return TS->font_get_glyph_index(cache[0], p_size, p_char, p_variation_selector);
}

Dictionary Font::get_supported_feature_list() const {
	_ensure_rid(0);
	return TS->font_supported_feature_list(cache[0]);
}

Dictionary Font::get_supported_variation_list() const {
	_ensure_rid(0);
	return TS->font_supported_variation_list(cache[0]);
}

Font::Font() {
	/* NOP */
}

Font::~Font() {
	_clear_cache();
}

/*************************************************************************/
/*  FontConfig                                                           */
/*************************************************************************/

void FontConfig::_bind_methods() {
	ClassDB::bind_method(D_METHOD("set_font", "font"), &FontConfig::set_font);
	ClassDB::bind_method(D_METHOD("get_font"), &FontConfig::get_font);

	ClassDB::bind_method(D_METHOD("set_size", "size"), &FontConfig::set_size);
	ClassDB::bind_method(D_METHOD("get_size"), &FontConfig::get_size);

	ClassDB::bind_method(D_METHOD("set_variation_opentype", "coords"), &FontConfig::set_variation_opentype);
	ClassDB::bind_method(D_METHOD("get_variation_opentype"), &FontConfig::get_variation_opentype);

	ClassDB::bind_method(D_METHOD("set_variation_embolden", "strength"), &FontConfig::set_variation_embolden);
	ClassDB::bind_method(D_METHOD("get_variation_embolden"), &FontConfig::get_variation_embolden);

	ClassDB::bind_method(D_METHOD("set_variation_face_index", "face_index"), &FontConfig::set_variation_face_index);
	ClassDB::bind_method(D_METHOD("get_variation_face_index"), &FontConfig::get_variation_face_index);

	ClassDB::bind_method(D_METHOD("set_variation_transform", "transform"), &FontConfig::set_variation_transform);
	ClassDB::bind_method(D_METHOD("get_variation_transform"), &FontConfig::get_variation_transform);

	ClassDB::bind_method(D_METHOD("set_variation_customize_fallbacks", "enabled"), &FontConfig::set_variation_customize_fallbacks);
	ClassDB::bind_method(D_METHOD("get_variation_customize_fallbacks"), &FontConfig::get_variation_customize_fallbacks);

	ClassDB::bind_method(D_METHOD("set_opentype_features", "features"), &FontConfig::set_opentype_features);
	ClassDB::bind_method(D_METHOD("get_opentype_features"), &FontConfig::get_opentype_features);

	ClassDB::bind_method(D_METHOD("set_spacing", "spacing", "value"), &FontConfig::set_spacing);
	ClassDB::bind_method(D_METHOD("get_spacing", "spacing"), &FontConfig::get_spacing);

	ClassDB::bind_method(D_METHOD("get_rids"), &FontConfig::get_rids);
	ClassDB::bind_method(D_METHOD("set_cache_capacity", "single_line", "multi_line"), &FontConfig::set_cache_capacity);

	ClassDB::bind_method(D_METHOD("get_height"), &FontConfig::get_height);
	ClassDB::bind_method(D_METHOD("get_ascent"), &FontConfig::get_ascent);
	ClassDB::bind_method(D_METHOD("get_descent"), &FontConfig::get_descent);
	ClassDB::bind_method(D_METHOD("get_underline_position"), &FontConfig::get_underline_position);
	ClassDB::bind_method(D_METHOD("get_underline_thickness"), &FontConfig::get_underline_thickness);

	ClassDB::bind_method(D_METHOD("get_string_size", "text", "alignment", "width", "flags", "direction", "orientation"), &FontConfig::get_string_size, DEFVAL(HORIZONTAL_ALIGNMENT_LEFT), DEFVAL(-1), DEFVAL(TextServer::JUSTIFICATION_KASHIDA | TextServer::JUSTIFICATION_WORD_BOUND), DEFVAL(TextServer::DIRECTION_AUTO), DEFVAL(TextServer::ORIENTATION_HORIZONTAL));
	ClassDB::bind_method(D_METHOD("get_multiline_string_size", "text", "alignment", "width", "max_lines", "flags", "direction", "orientation"), &FontConfig::get_multiline_string_size, DEFVAL(HORIZONTAL_ALIGNMENT_LEFT), DEFVAL(-1), DEFVAL(-1), DEFVAL(TextServer::BREAK_MANDATORY | TextServer::BREAK_WORD_BOUND), DEFVAL(TextServer::DIRECTION_AUTO), DEFVAL(TextServer::ORIENTATION_HORIZONTAL));

	ClassDB::bind_method(D_METHOD("draw_string", "canvas_item", "pos", "text", "alignment", "width", "modulate", "flags", "direction", "orientation"), &FontConfig::draw_string, DEFVAL(HORIZONTAL_ALIGNMENT_LEFT), DEFVAL(-1), DEFVAL(Color(1.0, 1.0, 1.0)), DEFVAL(TextServer::JUSTIFICATION_KASHIDA | TextServer::JUSTIFICATION_WORD_BOUND), DEFVAL(TextServer::DIRECTION_AUTO), DEFVAL(TextServer::ORIENTATION_HORIZONTAL));
	ClassDB::bind_method(D_METHOD("draw_multiline_string", "canvas_item", "pos", "text", "alignment", "width", "max_lines", "modulate", "flags", "direction", "orientation"), &FontConfig::draw_multiline_string, DEFVAL(HORIZONTAL_ALIGNMENT_LEFT), DEFVAL(-1), DEFVAL(-1), DEFVAL(Color(1.0, 1.0, 1.0)), DEFVAL(TextServer::BREAK_MANDATORY | TextServer::BREAK_WORD_BOUND | TextServer::JUSTIFICATION_KASHIDA | TextServer::JUSTIFICATION_WORD_BOUND), DEFVAL(TextServer::DIRECTION_AUTO), DEFVAL(TextServer::ORIENTATION_HORIZONTAL));

	ClassDB::bind_method(D_METHOD("draw_string_outline", "canvas_item", "pos", "text", "alignment", "width", "size", "modulate", "flags", "direction", "orientation"), &FontConfig::draw_string_outline, DEFVAL(HORIZONTAL_ALIGNMENT_LEFT), DEFVAL(-1), DEFVAL(1), DEFVAL(Color(1.0, 1.0, 1.0)), DEFVAL(TextServer::JUSTIFICATION_KASHIDA | TextServer::JUSTIFICATION_WORD_BOUND), DEFVAL(TextServer::DIRECTION_AUTO), DEFVAL(TextServer::ORIENTATION_HORIZONTAL));
	ClassDB::bind_method(D_METHOD("draw_multiline_string_outline", "canvas_item", "pos", "text", "alignment", "width", "max_lines", "size", "modulate", "flags", "direction", "orientation"), &FontConfig::draw_multiline_string_outline, DEFVAL(HORIZONTAL_ALIGNMENT_LEFT), DEFVAL(-1), DEFVAL(-1), DEFVAL(1), DEFVAL(Color(1.0, 1.0, 1.0)), DEFVAL(TextServer::BREAK_MANDATORY | TextServer::BREAK_WORD_BOUND | TextServer::JUSTIFICATION_KASHIDA | TextServer::JUSTIFICATION_WORD_BOUND), DEFVAL(TextServer::DIRECTION_AUTO), DEFVAL(TextServer::ORIENTATION_HORIZONTAL));

	ClassDB::bind_method(D_METHOD("has_char", "char"), &FontConfig::has_char);
	ClassDB::bind_method(D_METHOD("get_supported_chars"), &FontConfig::get_supported_chars);

	ClassDB::bind_method(D_METHOD("get_char_size", "char"), &FontConfig::get_char_size);
	ClassDB::bind_method(D_METHOD("draw_char", "canvas_item", "pos", "char", "modulate"), &FontConfig::draw_char, DEFVAL(Color(1.0, 1.0, 1.0)));
	ClassDB::bind_method(D_METHOD("draw_char_outline", "canvas_item", "pos", "char", "size", "modulate"), &FontConfig::draw_char_outline, DEFVAL(-1), DEFVAL(Color(1.0, 1.0, 1.0)));

	ADD_PROPERTY(PropertyInfo(Variant::OBJECT, "font", PROPERTY_HINT_RESOURCE_TYPE, "Font"), "set_font", "get_font");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "size", PROPERTY_HINT_RANGE, "1,2048,1,or_greater,suffix:px"), "set_size", "get_size");

	ADD_GROUP("Variation", "variation");
	ADD_PROPERTY(PropertyInfo(Variant::DICTIONARY, "variation_opentype"), "set_variation_opentype", "get_variation_opentype");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "variation_face_index"), "set_variation_face_index", "get_variation_face_index");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "variation_embolden", PROPERTY_HINT_RANGE, "-2,2,0.01"), "set_variation_embolden", "get_variation_embolden");
	ADD_PROPERTY(PropertyInfo(Variant::TRANSFORM2D, "variation_transform", PROPERTY_HINT_NONE, "suffix:px"), "set_variation_transform", "get_variation_transform");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "variation_customize_fallbacks"), "set_variation_customize_fallbacks", "get_variation_customize_fallbacks");

	ADD_GROUP("OpenType Features", "opentype");
	ADD_PROPERTY(PropertyInfo(Variant::DICTIONARY, "opentype_features"), "set_opentype_features", "get_opentype_features");

	ADD_GROUP("Extra Spacing", "spacing");
	ADD_PROPERTYI(PropertyInfo(Variant::INT, "spacing_glyph", PROPERTY_HINT_NONE, "suffix:px"), "set_spacing", "get_spacing", TextServer::SPACING_GLYPH);
	ADD_PROPERTYI(PropertyInfo(Variant::INT, "spacing_space", PROPERTY_HINT_NONE, "suffix:px"), "set_spacing", "get_spacing", TextServer::SPACING_SPACE);
	ADD_PROPERTYI(PropertyInfo(Variant::INT, "spacing_top", PROPERTY_HINT_NONE, "suffix:px"), "set_spacing", "get_spacing", TextServer::SPACING_TOP);
	ADD_PROPERTYI(PropertyInfo(Variant::INT, "spacing_bottom", PROPERTY_HINT_NONE, "suffix:px"), "set_spacing", "get_spacing", TextServer::SPACING_BOTTOM);
}

bool FontConfig::_set(const StringName &p_name, const Variant &p_value) {
	if (!variation_customize_fallbacks) {
		return false;
	}

	Vector<String> tokens = p_name.operator String().split("/");

#ifndef DISABLE_DEPRECATED
	if (tokens.size() == 1 && tokens[0] == "font_data") {
		// Compatibility, DynamicFont.
		Ref<Font> fd = p_value;
		if (fd.is_valid()) {
			set_font(fd);
			return true;
		}
		return false;
	} else if (tokens.size() == 2 && tokens[0] == "fallback") {
		// Compatibility, DynamicFont.
		Ref<Font> fd = p_value;
		if (fd.is_valid() && font.is_valid()) {
			TypedArray<Font> fb = font->get_fallbacks();
			fb.push_back(fd);
			font->set_fallbacks(fb);
			return true;
		}
		return false;
	} else if (tokens.size() == 1 && tokens[0] == "textures") {
		// Compatibility, BitmapFont.
		if (!font.is_valid()) {
			font.instantiate();
		}
		font->set_fixed_size(16);
		Array textures = p_value;
		for (int i = 0; i < textures.size(); i++) {
			Ref<ImageTexture> tex = textures[i];
			ERR_CONTINUE(!tex.is_valid());
			font->set_texture_image(0, Vector2i(16, 0), i, tex->get_image());
		}
	} else if (tokens.size() == 1 && tokens[0] == "chars") {
		// Compatibility, BitmapFont.
		if (!font.is_valid()) {
			font.instantiate();
		}
		font->set_fixed_size(16);
		PackedInt32Array arr = p_value;
		int len = arr.size();
		ERR_FAIL_COND_V(len % 9, false);
		if (!len) {
			return false;
		}
		int chars = len / 9;
		for (int i = 0; i < chars; i++) {
			const int32_t *data = &arr[i * 9];
			char32_t c = data[0];
			font->set_glyph_texture_idx(0, Vector2i(16, 0), c, data[1]);
			font->set_glyph_uv_rect(0, Vector2i(16, 0), c, Rect2(data[2], data[3], data[4], data[5]));
			font->set_glyph_offset(0, Vector2i(16, 0), c, Size2(data[6], data[7]));
			font->set_glyph_advance(0, 16, c, Vector2(data[8], 0));
		}
	} else if (tokens.size() == 1 && tokens[0] == "kernings") {
		// Compatibility, BitmapFont.
		if (!font.is_valid()) {
			font.instantiate();
		}
		font->set_fixed_size(16);
		PackedInt32Array arr = p_value;
		int len = arr.size();
		ERR_FAIL_COND_V(len % 3, false);
		if (!len) {
			return false;
		}
		for (int i = 0; i < len / 3; i++) {
			const int32_t *data = &arr[i * 3];
			font->set_kerning(0, 16, Vector2i(data[0], data[1]), Vector2(data[2], 0));
		}
	} else if (tokens.size() == 1 && tokens[0] == "height") {
		// Compatibility, BitmapFont.
		bmp_height = p_value;
		if (!font.is_valid()) {
			font.instantiate();
		}
		font->set_fixed_size(16);
		font->set_descent(0, 16, bmp_height - bmp_ascent);
	} else if (tokens.size() == 1 && tokens[0] == "ascent") {
		// Compatibility, BitmapFont.
		bmp_ascent = p_value;
		if (!font.is_valid()) {
			font.instantiate();
		}
		font->set_fixed_size(16);
		font->set_ascent(0, 16, bmp_ascent);
		font->set_descent(0, 16, bmp_height - bmp_ascent);
	} else if (tokens.size() == 1 && tokens[0] == "fallback") {
		// Compatibility, BitmapFont.
		if (!font.is_valid()) {
			font.instantiate();
		}
		font->set_fixed_size(16);
		Ref<FontConfig> f = p_value;
		if (f.is_valid() && f->get_font().is_valid()) {
			TypedArray<Font> fb = font->get_fallbacks();
			TypedArray<Font> fb_bmp = f->get_font()->get_fallbacks();
			for (int i = 0; i < fb_bmp.size(); i++) {
				fb.push_back(fb_bmp[i]);
			}
			font->set_fallbacks(fb);
			return true;
		}
		return false;
	}
#endif /* DISABLE_DEPRECATED */

	if (tokens.size() >= 1 && tokens[0] == "variation_fallback") {
		Ref<Font> fd = _get_font_or_default();
		for (int i = 1; i < tokens.size() - 1; i++) {
			ERR_FAIL_COND_V(fd.is_null(), false);

			int idx = tokens[i].to_int();
			const TypedArray<Font> &fallbacks = fd->get_fallbacks();
			if (idx < fallbacks.size()) {
				fd = fallbacks[idx];
			}
		}
		if (fd.is_valid()) {
			if (tokens[tokens.size() - 1] == "variation_opentype") {
				fallback_variations[fd].opentype = p_value;
				_update_rids();
				return true;
			} else if (tokens[tokens.size() - 1] == "variation_embolden") {
				fallback_variations[fd].embolden = p_value;
				_update_rids();
				return true;
			} else if (tokens[tokens.size() - 1] == "variation_face_index") {
				fallback_variations[fd].face_index = p_value;
				_update_rids();
				return true;
			} else if (tokens[tokens.size() - 1] == "variation_transform") {
				fallback_variations[fd].transform = p_value;
				_update_rids();
				return true;
			}
		}
	}
	return false;
}

bool FontConfig::_get(const StringName &p_name, Variant &r_ret) const {
	if (!variation_customize_fallbacks) {
		return false;
	}

	Vector<String> tokens = p_name.operator String().split("/");
	if (tokens.size() >= 1 && tokens[0] == "variation_fallback") {
		Ref<Font> fd = _get_font_or_default();
		for (int i = 1; i < tokens.size() - 1; i++) {
			ERR_FAIL_COND_V(fd.is_null(), false);

			int idx = tokens[i].to_int();
			const TypedArray<Font> &fallbacks = fd->get_fallbacks();
			if (idx < fallbacks.size()) {
				fd = fallbacks[idx];
			}
		}
		if (fd.is_valid()) {
			if (tokens[tokens.size() - 1] == "variation_opentype") {
				r_ret = fallback_variations[fd].opentype;
				return true;
			} else if (tokens[tokens.size() - 1] == "variation_embolden") {
				r_ret = fallback_variations[fd].embolden;
				return true;
			} else if (tokens[tokens.size() - 1] == "variation_face_index") {
				r_ret = fallback_variations[fd].face_index;
				return true;
			} else if (tokens[tokens.size() - 1] == "variation_transform") {
				r_ret = fallback_variations[fd].transform;
				return true;
			}
		}
	}
	return false;
}

void FontConfig::_get_property_list_fb(List<PropertyInfo> *p_list, const String &p_path, const Ref<Font> &p_fb, int p_depth) const {
	ERR_FAIL_COND(p_depth > 64);
	ERR_FAIL_COND(p_list == nullptr);
	if (p_fb.is_valid()) {
		p_list->push_back(PropertyInfo(Variant::NIL, vformat("%s (%s)", p_fb->get_font_name(), p_fb->get_font_style_name()), PROPERTY_HINT_NONE, vformat("%s/variation", p_path), PROPERTY_USAGE_SUBGROUP));
		p_list->push_back(PropertyInfo(Variant::DICTIONARY, vformat("%s/variation_opentype", p_path)));
		p_list->push_back(PropertyInfo(Variant::FLOAT, vformat("%s/variation_face_index", p_path)));
		p_list->push_back(PropertyInfo(Variant::FLOAT, vformat("%s/variation_embolden", p_path), PROPERTY_HINT_RANGE, "-2,2,0.01"));
		p_list->push_back(PropertyInfo(Variant::TRANSFORM2D, vformat("%s/variation_transform", p_path)));

		const TypedArray<Font> &fallbacks = p_fb->get_fallbacks();
		for (int i = 0; i < fallbacks.size(); i++) {
			_get_property_list_fb(p_list, vformat("%s/%d", p_path, i), fallbacks[i], p_depth + 1);
		}
	}
}

void FontConfig::_get_property_list(List<PropertyInfo> *p_list) const {
	ERR_FAIL_COND(p_list == nullptr);
	Ref<Font> f = _get_font_or_default();
	if (variation_customize_fallbacks && f.is_valid()) {
		p_list->push_back(PropertyInfo(Variant::NIL, "Variation", PROPERTY_HINT_NONE, "variation", PROPERTY_USAGE_GROUP));
		const TypedArray<Font> &fallbacks = f->get_fallbacks();
		for (int i = 0; i < fallbacks.size(); i++) {
			_get_property_list_fb(p_list, vformat("variation_fallback/%d", i), fallbacks[i], 0);
		}
	}
}

void FontConfig::_font_update_fb(const Ref<Font> &p_fb, const HashMap<Ref<Font>, Variation, VariantHasher, VariantComparator> &p_old_map, int p_depth) {
	ERR_FAIL_COND(p_depth > 64);
	if (p_fb.is_valid()) {
		if (p_old_map.has(p_fb)) {
			fallback_variations[p_fb] = p_old_map[p_fb];
		} else {
			fallback_variations[p_fb] = Variation();
		}
		const TypedArray<Font> &fallbacks = p_fb->get_fallbacks();
		for (int i = 0; i < fallbacks.size(); i++) {
			_font_update_fb(fallbacks[i], p_old_map, p_depth + 1);
		}
	}
}

void FontConfig::_font_changed() {
	HashMap<Ref<Font>, Variation, VariantHasher, VariantComparator> old_map = fallback_variations;
	fallback_variations.clear();

	if (variation_customize_fallbacks) {
		_font_update_fb(_get_font_or_default(), old_map, 0);
	}

	_update_rids();
	notify_property_list_changed();
}

void FontConfig::_update_rids_fb(const Ref<Font> &p_fb, int p_depth) {
	ERR_FAIL_COND(p_depth > 64);
	if (p_fb.is_valid()) {
		if (variation_customize_fallbacks) {
			const Variation &fallback_variation = fallback_variations[p_fb];
			rids.push_back(p_fb->find_cache(fallback_variation.opentype, fallback_variation.face_index, fallback_variation.embolden, fallback_variation.transform));
		} else {
			rids.push_back(p_fb->find_cache(variation.opentype, variation.face_index, variation.embolden, variation.transform));
		}
		const TypedArray<Font> &fallbacks = p_fb->get_fallbacks();
		for (int i = 0; i < fallbacks.size(); i++) {
			_update_rids_fb(fallbacks[i], p_depth + 1);
		}
	}
}

void FontConfig::_update_rids() {
	cache.clear();
	cache_wrap.clear();

	rids.clear();
	Ref<Font> f = _get_font_or_default();
	if (f.is_valid()) {
		rids.push_back(f->find_cache(variation.opentype, variation.face_index, variation.embolden, variation.transform));
		const TypedArray<Font> &fallbacks = f->get_fallbacks();
		for (int i = 0; i < fallbacks.size(); i++) {
			_update_rids_fb(fallbacks[i], 0);
		}
	}

	emit_changed();
}

void FontConfig::reset_state() {
	cache.clear();
	cache_wrap.clear();

	rids.clear();
	if (font.is_valid()) {
		font->disconnect(SNAME("changed"), callable_mp(this, &FontConfig::_font_changed));
		font.unref();
	}

	variation = Variation();
	variation_customize_fallbacks = false;
	fallback_variations.clear();

	opentype_features = Dictionary();

	for (int i = 0; i < TextServer::SPACING_MAX; i++) {
		extra_spacing[i] = 0;
	}
}

void FontConfig::set_font(const Ref<Font> &p_font) {
	if (font != p_font) {
		if (font.is_valid()) {
			font->disconnect(SNAME("changed"), callable_mp(this, &FontConfig::_font_changed));
		}
		font = p_font;
		if (font.is_valid()) {
			font->connect(SNAME("changed"), callable_mp(this, &FontConfig::_font_changed), varray(), CONNECT_REFERENCE_COUNTED);
		}
		_font_changed();
	}
}

Ref<Font> FontConfig::_get_font_or_default() const {
	if (font.is_valid()) {
		return font;
	}
	if (Theme::get_project_default().is_valid()) {
		if (Theme::get_project_default()->has_theme_item(Theme::DATA_TYPE_FONT, "font", StringName())) {
			Ref<FontConfig> fc = Theme::get_project_default()->get_theme_item(Theme::DATA_TYPE_FONT, "font", StringName());
			if (fc.is_valid() && fc->get_font().is_valid()) {
				return fc->get_font();
			}
		}
	}
	if (Theme::get_default().is_valid()) {
		if (Theme::get_default()->has_theme_item(Theme::DATA_TYPE_FONT, "font", StringName())) {
			Ref<FontConfig> fc = Theme::get_default()->get_theme_item(Theme::DATA_TYPE_FONT, "font", StringName());
			if (fc.is_valid() && fc->get_font().is_valid()) {
				return fc->get_font();
			}
		}
	}
	if (Theme::get_fallback_font().is_valid()) {
		return Theme::get_fallback_font()->get_font();
	}
	return Ref<Font>();
}

Ref<Font> FontConfig::get_font() const {
	return font;
}

void FontConfig::set_size(int p_size) {
	if (font_size != p_size) {
		font_size = p_size;
		_font_changed();
	}
}

int FontConfig::get_size() const {
	return font_size;
}

void FontConfig::set_variation_opentype(const Dictionary &p_coords) {
	if (variation.opentype != p_coords) {
		variation.opentype = p_coords;
		_update_rids();
	}
}

Dictionary FontConfig::get_variation_opentype() const {
	return variation.opentype;
}

void FontConfig::set_variation_embolden(float p_strength) {
	if (variation.embolden != p_strength) {
		variation.embolden = p_strength;
		_update_rids();
	}
}

float FontConfig::get_variation_embolden() const {
	return variation.embolden;
}

void FontConfig::set_variation_transform(Transform2D p_transform) {
	if (variation.transform != p_transform) {
		variation.transform = p_transform;
		_update_rids();
	}
}

Transform2D FontConfig::get_variation_transform() const {
	return variation.transform;
}

void FontConfig::set_variation_face_index(int p_face_index) {
	if (variation.face_index != p_face_index) {
		variation.face_index = p_face_index;
		_update_rids();
	}
}

int FontConfig::get_variation_face_index() const {
	return variation.face_index;
}

void FontConfig::set_variation_customize_fallbacks(bool p_enabled) {
	if (variation_customize_fallbacks != p_enabled) {
		variation_customize_fallbacks = p_enabled;
		_font_changed();
	}
}

bool FontConfig::get_variation_customize_fallbacks() const {
	return variation_customize_fallbacks;
}

void FontConfig::set_opentype_features(const Dictionary &p_features) {
	if (opentype_features != p_features) {
		opentype_features = p_features;
		_font_changed();
	}
}

Dictionary FontConfig::get_opentype_features() const {
	return opentype_features;
}

void FontConfig::set_spacing(TextServer::SpacingType p_spacing, int p_value) {
	ERR_FAIL_INDEX((int)p_spacing, TextServer::SPACING_MAX);
	if (extra_spacing[p_spacing] != p_value) {
		extra_spacing[p_spacing] = p_value;
		_font_changed();
	}
}

int FontConfig::get_spacing(TextServer::SpacingType p_spacing) const {
	ERR_FAIL_INDEX_V((int)p_spacing, TextServer::SPACING_MAX, 0);
	return extra_spacing[p_spacing];
}

TypedArray<RID> FontConfig::get_rids() const {
	return rids;
}

void FontConfig::set_cache_capacity(int p_single_line, int p_multi_line) {
	cache.set_capacity(p_single_line);
	cache_wrap.set_capacity(p_multi_line);
}

real_t FontConfig::get_height() const {
	real_t ret = 0.f;
	for (int i = 0; i < rids.size(); i++) {
		ret = MAX(ret, TS->font_get_ascent(rids[i], font_size) + TS->font_get_descent(rids[i], font_size));
	}
	return ret + extra_spacing[TextServer::SPACING_BOTTOM] + extra_spacing[TextServer::SPACING_TOP];
}

real_t FontConfig::get_ascent() const {
	real_t ret = 0.f;
	for (int i = 0; i < rids.size(); i++) {
		ret = MAX(ret, TS->font_get_ascent(rids[i], font_size));
	}
	return ret + extra_spacing[TextServer::SPACING_TOP];
}

real_t FontConfig::get_descent() const {
	real_t ret = 0.f;
	for (int i = 0; i < rids.size(); i++) {
		ret = MAX(ret, TS->font_get_descent(rids[i], font_size));
	}
	return ret + extra_spacing[TextServer::SPACING_BOTTOM];
}

real_t FontConfig::get_underline_position() const {
	real_t ret = 0.f;
	for (int i = 0; i < rids.size(); i++) {
		ret = MAX(ret, TS->font_get_underline_position(rids[i], font_size));
	}
	return ret + extra_spacing[TextServer::SPACING_TOP];
}

real_t FontConfig::get_underline_thickness() const {
	real_t ret = 0.f;
	for (int i = 0; i < rids.size(); i++) {
		ret = MAX(ret, TS->font_get_underline_thickness(rids[i], font_size));
	}
	return ret;
}

// Drawing string.
Size2 FontConfig::get_string_size(const String &p_text, HorizontalAlignment p_alignment, float p_width, uint16_t p_flags, TextServer::Direction p_direction, TextServer::Orientation p_orientation) const {
	uint64_t hash = p_text.hash64();
	if (p_alignment == HORIZONTAL_ALIGNMENT_FILL) {
		hash = hash_djb2_one_64(hash_djb2_one_float(p_width), hash);
		hash = hash_djb2_one_64(p_flags, hash);
		hash = hash_djb2_one_64(p_direction, hash);
		hash = hash_djb2_one_64(p_orientation, hash);
	}

	Ref<TextLine> buffer;
	if (cache.has(hash)) {
		buffer = cache.get(hash);
	} else {
		buffer.instantiate();
		buffer->set_direction(p_direction);
		buffer->set_orientation(p_orientation);
		buffer->add_string(p_text, Ref<FontConfig>(this));
		cache.insert(hash, buffer);
	}
	return buffer->get_size();
}

Size2 FontConfig::get_multiline_string_size(const String &p_text, HorizontalAlignment p_alignment, float p_width, int p_max_lines, uint16_t p_flags, TextServer::Direction p_direction, TextServer::Orientation p_orientation) const {
	uint64_t hash = p_text.hash64();
	hash = hash_djb2_one_64(hash_djb2_one_float(p_width), hash);
	hash = hash_djb2_one_64(p_flags, hash);
	hash = hash_djb2_one_64(p_direction, hash);
	hash = hash_djb2_one_64(p_orientation, hash);

	Ref<TextParagraph> lines_buffer;
	if (cache_wrap.has(hash)) {
		lines_buffer = cache_wrap.get(hash);
	} else {
		lines_buffer.instantiate();
		lines_buffer->set_direction(p_direction);
		lines_buffer->set_orientation(p_orientation);
		lines_buffer->add_string(p_text, Ref<FontConfig>(this));
		lines_buffer->set_width(p_width);
		lines_buffer->set_flags(p_flags);
		cache_wrap.insert(hash, lines_buffer);
	}

	lines_buffer->set_alignment(p_alignment);
	lines_buffer->set_max_lines_visible(p_max_lines);

	return lines_buffer->get_size();
}

void FontConfig::draw_string(RID p_canvas_item, const Point2 &p_pos, const String &p_text, HorizontalAlignment p_alignment, float p_width, const Color &p_modulate, uint16_t p_flags, TextServer::Direction p_direction, TextServer::Orientation p_orientation) const {
	uint64_t hash = p_text.hash64();
	if (p_alignment == HORIZONTAL_ALIGNMENT_FILL) {
		hash = hash_djb2_one_64(hash_djb2_one_float(p_width), hash);
		hash = hash_djb2_one_64(p_flags, hash);
	}

	Ref<TextLine> buffer;
	if (cache.has(hash)) {
		buffer = cache.get(hash);
	} else {
		buffer.instantiate();
		buffer->set_direction(p_direction);
		buffer->set_orientation(p_orientation);
		buffer->add_string(p_text, Ref<FontConfig>(this));
		cache.insert(hash, buffer);
	}

	Vector2 ofs = p_pos;
	if (p_orientation == TextServer::ORIENTATION_HORIZONTAL) {
		ofs.y -= buffer->get_line_ascent();
	} else {
		ofs.x -= buffer->get_line_ascent();
	}

	buffer->set_width(p_width);
	buffer->set_horizontal_alignment(p_alignment);
	buffer->set_flags(p_flags);

	buffer->draw(p_canvas_item, ofs, p_modulate);
}

void FontConfig::draw_multiline_string(RID p_canvas_item, const Point2 &p_pos, const String &p_text, HorizontalAlignment p_alignment, float p_width, int p_max_lines, const Color &p_modulate, uint16_t p_flags, TextServer::Direction p_direction, TextServer::Orientation p_orientation) const {
	uint64_t hash = p_text.hash64();
	hash = hash_djb2_one_64(hash_djb2_one_float(p_width), hash);
	hash = hash_djb2_one_64(p_flags, hash);
	hash = hash_djb2_one_64(p_direction, hash);
	hash = hash_djb2_one_64(p_orientation, hash);

	Ref<TextParagraph> lines_buffer;
	if (cache_wrap.has(hash)) {
		lines_buffer = cache_wrap.get(hash);
	} else {
		lines_buffer.instantiate();
		lines_buffer->set_direction(p_direction);
		lines_buffer->set_orientation(p_orientation);
		lines_buffer->add_string(p_text, Ref<FontConfig>(this));
		lines_buffer->set_width(p_width);
		lines_buffer->set_flags(p_flags);
		cache_wrap.insert(hash, lines_buffer);
	}

	Vector2 ofs = p_pos;
	if (p_orientation == TextServer::ORIENTATION_HORIZONTAL) {
		ofs.y -= lines_buffer->get_line_ascent(0);
	} else {
		ofs.x -= lines_buffer->get_line_ascent(0);
	}

	lines_buffer->set_alignment(p_alignment);
	lines_buffer->set_max_lines_visible(p_max_lines);

	lines_buffer->draw(p_canvas_item, ofs, p_modulate);
}

void FontConfig::draw_string_outline(RID p_canvas_item, const Point2 &p_pos, const String &p_text, HorizontalAlignment p_alignment, float p_width, int p_size, const Color &p_modulate, uint16_t p_flags, TextServer::Direction p_direction, TextServer::Orientation p_orientation) const {
	uint64_t hash = p_text.hash64();
	if (p_alignment == HORIZONTAL_ALIGNMENT_FILL) {
		hash = hash_djb2_one_64(hash_djb2_one_float(p_width), hash);
		hash = hash_djb2_one_64(p_flags, hash);
	}

	Ref<TextLine> buffer;
	if (cache.has(hash)) {
		buffer = cache.get(hash);
	} else {
		buffer.instantiate();
		buffer->set_direction(p_direction);
		buffer->set_orientation(p_orientation);
		buffer->add_string(p_text, Ref<FontConfig>(this));
		cache.insert(hash, buffer);
	}

	Vector2 ofs = p_pos;
	if (p_orientation == TextServer::ORIENTATION_HORIZONTAL) {
		ofs.y -= buffer->get_line_ascent();
	} else {
		ofs.x -= buffer->get_line_ascent();
	}

	buffer->set_width(p_width);
	buffer->set_horizontal_alignment(p_alignment);
	buffer->set_flags(p_flags);

	buffer->draw_outline(p_canvas_item, ofs, p_size, p_modulate);
}

void FontConfig::draw_multiline_string_outline(RID p_canvas_item, const Point2 &p_pos, const String &p_text, HorizontalAlignment p_alignment, float p_width, int p_max_lines, int p_size, const Color &p_modulate, uint16_t p_flags, TextServer::Direction p_direction, TextServer::Orientation p_orientation) const {
	uint64_t hash = p_text.hash64();
	hash = hash_djb2_one_64(hash_djb2_one_float(p_width), hash);
	hash = hash_djb2_one_64(p_flags, hash);
	hash = hash_djb2_one_64(p_direction, hash);
	hash = hash_djb2_one_64(p_orientation, hash);

	Ref<TextParagraph> lines_buffer;
	if (cache_wrap.has(hash)) {
		lines_buffer = cache_wrap.get(hash);
	} else {
		lines_buffer.instantiate();
		lines_buffer->set_direction(p_direction);
		lines_buffer->set_orientation(p_orientation);
		lines_buffer->add_string(p_text, Ref<FontConfig>(this));
		lines_buffer->set_width(p_width);
		lines_buffer->set_flags(p_flags);
		cache_wrap.insert(hash, lines_buffer);
	}

	Vector2 ofs = p_pos;
	if (p_orientation == TextServer::ORIENTATION_HORIZONTAL) {
		ofs.y -= lines_buffer->get_line_ascent(0);
	} else {
		ofs.x -= lines_buffer->get_line_ascent(0);
	}

	lines_buffer->set_alignment(p_alignment);
	lines_buffer->set_max_lines_visible(p_max_lines);

	lines_buffer->draw_outline(p_canvas_item, ofs, p_size, p_modulate);
}

// Helper functions.
bool FontConfig::has_char(char32_t p_char) const {
	for (int i = 0; i < rids.size(); i++) {
		if (TS->font_has_char(rids[i], p_char)) {
			return true;
		}
	}
	return false;
}

String FontConfig::get_supported_chars() const {
	String chars;
	for (int i = 0; i < rids.size(); i++) {
		String data_chars = TS->font_get_supported_chars(rids[i]);
		for (int j = 0; j < data_chars.length(); j++) {
			if (chars.find_char(data_chars[j]) == -1) {
				chars += data_chars[j];
			}
		}
	}
	return chars;
}

// Drawing char.
Size2 FontConfig::get_char_size(char32_t p_char) const {
	for (int i = 0; i < rids.size(); i++) {
		if (TS->font_has_char(rids[i], p_char)) {
			int32_t glyph = TS->font_get_glyph_index(rids[i], font_size, p_char, 0);
			return Size2(TS->font_get_glyph_advance(rids[i], font_size, glyph).x, get_height());
		}
	}
	return Size2();
}

real_t FontConfig::draw_char(RID p_canvas_item, const Point2 &p_pos, char32_t p_char, const Color &p_modulate) const {
	for (int i = 0; i < rids.size(); i++) {
		if (TS->font_has_char(rids[i], p_char)) {
			int32_t glyph = TS->font_get_glyph_index(rids[i], font_size, p_char, 0);
			TS->font_draw_glyph(rids[i], p_canvas_item, font_size, p_pos, glyph, p_modulate);
			return TS->font_get_glyph_advance(rids[i], font_size, glyph).x;
		}
	}
	return 0.f;
}

real_t FontConfig::draw_char_outline(RID p_canvas_item, const Point2 &p_pos, char32_t p_char, int p_size, const Color &p_modulate) const {
	for (int i = 0; i < rids.size(); i++) {
		if (TS->font_has_char(rids[i], p_char)) {
			int32_t glyph = TS->font_get_glyph_index(rids[i], font_size, p_char, 0);
			TS->font_draw_glyph_outline(rids[i], p_canvas_item, font_size, p_size, p_pos, glyph, p_modulate);
			return TS->font_get_glyph_advance(rids[i], font_size, glyph).x;
		}
	}
	return 0.f;
}

FontConfig::FontConfig() {
	cache.set_capacity(64);
	cache_wrap.set_capacity(16);

	for (int i = 0; i < TextServer::SPACING_MAX; i++) {
		extra_spacing[i] = 0;
	}

	_update_rids();
	notify_property_list_changed();
}

FontConfig::~FontConfig() {
}
