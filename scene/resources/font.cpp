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
	ClassDB::bind_method(D_METHOD("set_data", "data"), &FontData::set_data);
	ClassDB::bind_method(D_METHOD("get_data"), &FontData::get_data);

	ClassDB::bind_method(D_METHOD("set_antialiased", "antialiased"), &FontData::set_antialiased);
	ClassDB::bind_method(D_METHOD("is_antialiased"), &FontData::is_antialiased);

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
	oversampling = 0.f;
}

/*************************************************************************/

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
