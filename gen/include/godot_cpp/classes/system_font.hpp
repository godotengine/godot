/**************************************************************************/
/*  system_font.hpp                                                       */
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

#include <godot_cpp/classes/font.hpp>
#include <godot_cpp/classes/ref.hpp>
#include <godot_cpp/classes/text_server.hpp>
#include <godot_cpp/variant/packed_string_array.hpp>

#include <godot_cpp/core/class_db.hpp>

#include <type_traits>

namespace godot {

class SystemFont : public Font {
	GDEXTENSION_CLASS(SystemFont, Font)

public:
	void set_antialiasing(TextServer::FontAntialiasing p_antialiasing);
	TextServer::FontAntialiasing get_antialiasing() const;
	void set_disable_embedded_bitmaps(bool p_disable_embedded_bitmaps);
	bool get_disable_embedded_bitmaps() const;
	void set_generate_mipmaps(bool p_generate_mipmaps);
	bool get_generate_mipmaps() const;
	void set_allow_system_fallback(bool p_allow_system_fallback);
	bool is_allow_system_fallback() const;
	void set_force_autohinter(bool p_force_autohinter);
	bool is_force_autohinter() const;
	void set_modulate_color_glyphs(bool p_modulate);
	bool is_modulate_color_glyphs() const;
	void set_hinting(TextServer::Hinting p_hinting);
	TextServer::Hinting get_hinting() const;
	void set_subpixel_positioning(TextServer::SubpixelPositioning p_subpixel_positioning);
	TextServer::SubpixelPositioning get_subpixel_positioning() const;
	void set_keep_rounding_remainders(bool p_keep_rounding_remainders);
	bool get_keep_rounding_remainders() const;
	void set_multichannel_signed_distance_field(bool p_msdf);
	bool is_multichannel_signed_distance_field() const;
	void set_msdf_pixel_range(int32_t p_msdf_pixel_range);
	int32_t get_msdf_pixel_range() const;
	void set_msdf_size(int32_t p_msdf_size);
	int32_t get_msdf_size() const;
	void set_oversampling(float p_oversampling);
	float get_oversampling() const;
	PackedStringArray get_font_names() const;
	void set_font_names(const PackedStringArray &p_names);
	bool get_font_italic() const;
	void set_font_italic(bool p_italic);
	void set_font_weight(int32_t p_weight);
	void set_font_stretch(int32_t p_stretch);

protected:
	template <typename T, typename B>
	static void register_virtuals() {
		Font::register_virtuals<T, B>();
	}

public:
};

} // namespace godot

