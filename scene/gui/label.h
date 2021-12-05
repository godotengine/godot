/*************************************************************************/
/*  label.h                                                              */
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

#ifndef LABEL_H
#define LABEL_H

#include "scene/gui/control.h"

class Label : public Control {
	GDCLASS(Label, Control);

public:
	enum Align {
		ALIGN_LEFT,
		ALIGN_CENTER,
		ALIGN_RIGHT,
		ALIGN_FILL
	};

	enum VAlign {
		VALIGN_TOP,
		VALIGN_CENTER,
		VALIGN_BOTTOM,
		VALIGN_FILL
	};

	enum AutowrapMode {
		AUTOWRAP_OFF,
		AUTOWRAP_ARBITRARY,
		AUTOWRAP_WORD,
		AUTOWRAP_WORD_SMART
	};

	enum OverrunBehavior {
		OVERRUN_NO_TRIMMING,
		OVERRUN_TRIM_CHAR,
		OVERRUN_TRIM_WORD,
		OVERRUN_TRIM_ELLIPSIS,
		OVERRUN_TRIM_WORD_ELLIPSIS,
	};

	enum VisibleCharactersBehavior {
		VC_CHARS_BEFORE_SHAPING,
		VC_CHARS_AFTER_SHAPING,
		VC_GLYPHS_AUTO,
		VC_GLYPHS_LTR,
		VC_GLYPHS_RTL,
	};

private:
	Align align = ALIGN_LEFT;
	VAlign valign = VALIGN_TOP;
	String text;
	String xl_text;
	AutowrapMode autowrap_mode = AUTOWRAP_OFF;
	bool clip = false;
	OverrunBehavior overrun_behavior = OVERRUN_NO_TRIMMING;
	Size2 minsize;
	bool uppercase = false;

	bool lines_dirty = true;
	bool dirty = true;
	RID text_rid;
	Vector<RID> lines_rid;

	Dictionary opentype_features;
	String language;
	TextDirection text_direction = TEXT_DIRECTION_AUTO;
	Control::StructuredTextParser st_parser = STRUCTURED_TEXT_DEFAULT;
	Array st_args;

	float percent_visible = 1.0;

	VisibleCharactersBehavior visible_chars_behavior = VC_CHARS_BEFORE_SHAPING;
	int visible_chars = -1;
	int lines_skipped = 0;
	int max_lines_visible = -1;

	void _update_visible();
	void _shape();

protected:
	void _notification(int p_what);

	static void _bind_methods();

	bool _set(const StringName &p_name, const Variant &p_value);
	bool _get(const StringName &p_name, Variant &r_ret) const;
	void _get_property_list(List<PropertyInfo> *p_list) const;

public:
	virtual Size2 get_minimum_size() const override;

	void set_align(Align p_align);
	Align get_align() const;

	void set_valign(VAlign p_align);
	VAlign get_valign() const;

	void set_text(const String &p_string);
	String get_text() const;

	void set_text_direction(TextDirection p_text_direction);
	TextDirection get_text_direction() const;

	void set_opentype_feature(const String &p_name, int p_value);
	int get_opentype_feature(const String &p_name) const;
	void clear_opentype_features();

	void set_language(const String &p_language);
	String get_language() const;

	void set_structured_text_bidi_override(Control::StructuredTextParser p_parser);
	Control::StructuredTextParser get_structured_text_bidi_override() const;

	void set_structured_text_bidi_override_options(Array p_args);
	Array get_structured_text_bidi_override_options() const;

	void set_autowrap_mode(AutowrapMode p_mode);
	AutowrapMode get_autowrap_mode() const;

	void set_uppercase(bool p_uppercase);
	bool is_uppercase() const;

	VisibleCharactersBehavior get_visible_characters_behavior() const;
	void set_visible_characters_behavior(VisibleCharactersBehavior p_behavior);

	void set_visible_characters(int p_amount);
	int get_visible_characters() const;
	int get_total_character_count() const;

	void set_clip_text(bool p_clip);
	bool is_clipping_text() const;

	void set_text_overrun_behavior(OverrunBehavior p_behavior);
	OverrunBehavior get_text_overrun_behavior() const;

	void set_percent_visible(float p_percent);
	float get_percent_visible() const;

	void set_lines_skipped(int p_lines);
	int get_lines_skipped() const;

	void set_max_lines_visible(int p_lines);
	int get_max_lines_visible() const;

	int get_line_height(int p_line = -1) const;
	int get_line_count() const;
	int get_visible_line_count() const;

	Label(const String &p_text = String());
	~Label();
};

VARIANT_ENUM_CAST(Label::Align);
VARIANT_ENUM_CAST(Label::VAlign);
VARIANT_ENUM_CAST(Label::AutowrapMode);
VARIANT_ENUM_CAST(Label::OverrunBehavior);
VARIANT_ENUM_CAST(Label::VisibleCharactersBehavior);

#endif
