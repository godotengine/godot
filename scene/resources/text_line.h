/*************************************************************************/
/*  text_line.h                                                          */
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

#ifndef TEXT_LINE_H
#define TEXT_LINE_H

#include "scene/resources/font.h"
#include "servers/text_server.h"

/*************************************************************************/

class TextLine : public RefCounted {
	GDCLASS(TextLine, RefCounted);

public:
	enum OverrunBehavior {
		OVERRUN_NO_TRIMMING,
		OVERRUN_TRIM_CHAR,
		OVERRUN_TRIM_WORD,
		OVERRUN_TRIM_ELLIPSIS,
		OVERRUN_TRIM_WORD_ELLIPSIS,
	};

private:
	RID rid;
	int spacing_top = 0;
	int spacing_bottom = 0;

	bool dirty = true;

	float width = -1.0;
	uint16_t flags = TextServer::JUSTIFICATION_WORD_BOUND | TextServer::JUSTIFICATION_KASHIDA;
	HorizontalAlignment alignment = HORIZONTAL_ALIGNMENT_LEFT;
	OverrunBehavior overrun_behavior = OVERRUN_TRIM_ELLIPSIS;

	Vector<float> tab_stops;

protected:
	static void _bind_methods();

	void _shape();

public:
	RID get_rid() const;

	void clear();

	void set_direction(TextServer::Direction p_direction);
	TextServer::Direction get_direction() const;

	void set_bidi_override(const Array &p_override);

	void set_orientation(TextServer::Orientation p_orientation);
	TextServer::Orientation get_orientation() const;

	void set_preserve_invalid(bool p_enabled);
	bool get_preserve_invalid() const;

	void set_preserve_control(bool p_enabled);
	bool get_preserve_control() const;

	bool add_string(const String &p_text, const Ref<Font> &p_fonts, int p_size, const Dictionary &p_opentype_features = Dictionary(), const String &p_language = "");
	bool add_object(Variant p_key, const Size2 &p_size, InlineAlignment p_inline_align = INLINE_ALIGNMENT_CENTER, int p_length = 1);
	bool resize_object(Variant p_key, const Size2 &p_size, InlineAlignment p_inline_align = INLINE_ALIGNMENT_CENTER);

	void set_horizontal_alignment(HorizontalAlignment p_alignment);
	HorizontalAlignment get_horizontal_alignment() const;

	void tab_align(const Vector<float> &p_tab_stops);

	void set_flags(uint16_t p_flags);
	uint16_t get_flags() const;

	void set_text_overrun_behavior(OverrunBehavior p_behavior);
	OverrunBehavior get_text_overrun_behavior() const;

	void set_width(float p_width);
	float get_width() const;

	Array get_objects() const;
	Rect2 get_object_rect(Variant p_key) const;

	Size2 get_size() const;

	float get_line_ascent() const;
	float get_line_descent() const;
	float get_line_width() const;
	float get_line_underline_position() const;
	float get_line_underline_thickness() const;

	void draw(RID p_canvas, const Vector2 &p_pos, const Color &p_color = Color(1, 1, 1)) const;
	void draw_outline(RID p_canvas, const Vector2 &p_pos, int p_outline_size = 1, const Color &p_color = Color(1, 1, 1)) const;

	int hit_test(float p_coords) const;

	TextLine(const String &p_text, const Ref<Font> &p_fonts, int p_size, const Dictionary &p_opentype_features = Dictionary(), const String &p_language = "", TextServer::Direction p_direction = TextServer::DIRECTION_AUTO, TextServer::Orientation p_orientation = TextServer::ORIENTATION_HORIZONTAL);
	TextLine();
	~TextLine();
};

VARIANT_ENUM_CAST(TextLine::OverrunBehavior);

#endif // TEXT_LINE_H
