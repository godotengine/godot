/**************************************************************************/
/*  font_variation.hpp                                                    */
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
#include <godot_cpp/variant/dictionary.hpp>
#include <godot_cpp/variant/transform2d.hpp>

#include <godot_cpp/core/class_db.hpp>

#include <type_traits>

namespace godot {

class FontVariation : public Font {
	GDEXTENSION_CLASS(FontVariation, Font)

public:
	void set_base_font(const Ref<Font> &p_font);
	Ref<Font> get_base_font() const;
	void set_variation_opentype(const Dictionary &p_coords);
	Dictionary get_variation_opentype() const;
	void set_variation_embolden(float p_strength);
	float get_variation_embolden() const;
	void set_variation_face_index(int32_t p_face_index);
	int32_t get_variation_face_index() const;
	void set_variation_transform(const Transform2D &p_transform);
	Transform2D get_variation_transform() const;
	void set_opentype_features(const Dictionary &p_features);
	void set_spacing(TextServer::SpacingType p_spacing, int32_t p_value);
	void set_baseline_offset(float p_baseline_offset);
	float get_baseline_offset() const;

protected:
	template <typename T, typename B>
	static void register_virtuals() {
		Font::register_virtuals<T, B>();
	}

public:
};

} // namespace godot

