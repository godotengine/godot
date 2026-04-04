/**************************************************************************/
/*  text_mesh.hpp                                                         */
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

#include <godot_cpp/classes/global_constants.hpp>
#include <godot_cpp/classes/primitive_mesh.hpp>
#include <godot_cpp/classes/ref.hpp>
#include <godot_cpp/classes/text_server.hpp>
#include <godot_cpp/variant/array.hpp>
#include <godot_cpp/variant/string.hpp>
#include <godot_cpp/variant/vector2.hpp>

#include <godot_cpp/core/class_db.hpp>

#include <type_traits>

namespace godot {

class Font;

class TextMesh : public PrimitiveMesh {
	GDEXTENSION_CLASS(TextMesh, PrimitiveMesh)

public:
	void set_horizontal_alignment(HorizontalAlignment p_alignment);
	HorizontalAlignment get_horizontal_alignment() const;
	void set_vertical_alignment(VerticalAlignment p_alignment);
	VerticalAlignment get_vertical_alignment() const;
	void set_text(const String &p_text);
	String get_text() const;
	void set_font(const Ref<Font> &p_font);
	Ref<Font> get_font() const;
	void set_font_size(int32_t p_font_size);
	int32_t get_font_size() const;
	void set_line_spacing(float p_line_spacing);
	float get_line_spacing() const;
	void set_autowrap_mode(TextServer::AutowrapMode p_autowrap_mode);
	TextServer::AutowrapMode get_autowrap_mode() const;
	void set_justification_flags(BitField<TextServer::JustificationFlag> p_justification_flags);
	BitField<TextServer::JustificationFlag> get_justification_flags() const;
	void set_depth(float p_depth);
	float get_depth() const;
	void set_width(float p_width);
	float get_width() const;
	void set_pixel_size(float p_pixel_size);
	float get_pixel_size() const;
	void set_offset(const Vector2 &p_offset);
	Vector2 get_offset() const;
	void set_curve_step(float p_curve_step);
	float get_curve_step() const;
	void set_text_direction(TextServer::Direction p_direction);
	TextServer::Direction get_text_direction() const;
	void set_language(const String &p_language);
	String get_language() const;
	void set_structured_text_bidi_override(TextServer::StructuredTextParser p_parser);
	TextServer::StructuredTextParser get_structured_text_bidi_override() const;
	void set_structured_text_bidi_override_options(const Array &p_args);
	Array get_structured_text_bidi_override_options() const;
	void set_uppercase(bool p_enable);
	bool is_uppercase() const;

protected:
	template <typename T, typename B>
	static void register_virtuals() {
		PrimitiveMesh::register_virtuals<T, B>();
	}

public:
};

} // namespace godot

