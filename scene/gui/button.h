/*************************************************************************/
/*  button.h                                                             */
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

#ifndef BUTTON_H
#define BUTTON_H

#include "scene/gui/base_button.h"
#include "scene/resources/text_paragraph.h"

class Button : public BaseButton {
	GDCLASS(Button, BaseButton);

public:
	enum TextAlign {
		ALIGN_LEFT,
		ALIGN_CENTER,
		ALIGN_RIGHT
	};

private:
	bool flat = false;
	String text;
	String xl_text;
	Ref<TextParagraph> text_buf;

	Dictionary opentype_features;
	String language;
	TextDirection text_direction = TEXT_DIRECTION_AUTO;

	Ref<Texture2D> icon;
	bool expand_icon = false;
	bool clip_text = false;
	TextAlign align = ALIGN_CENTER;
	float _internal_margin[4] = {};

	void _shape();

protected:
	void _set_internal_margin(Side p_side, float p_value);
	void _notification(int p_what);
	static void _bind_methods();

	bool _set(const StringName &p_name, const Variant &p_value);
	bool _get(const StringName &p_name, Variant &r_ret) const;
	void _get_property_list(List<PropertyInfo> *p_list) const;

public:
	virtual Size2 get_minimum_size() const override;

	void set_text(const String &p_text);
	String get_text() const;

	void set_text_direction(TextDirection p_text_direction);
	TextDirection get_text_direction() const;

	void set_opentype_feature(const String &p_name, int p_value);
	int get_opentype_feature(const String &p_name) const;
	void clear_opentype_features();

	void set_language(const String &p_language);
	String get_language() const;

	void set_icon(const Ref<Texture2D> &p_icon);
	Ref<Texture2D> get_icon() const;

	void set_expand_icon(bool p_expand_icon);
	bool is_expand_icon() const;

	void set_flat(bool p_flat);
	bool is_flat() const;

	void set_clip_text(bool p_clip_text);
	bool get_clip_text() const;

	void set_text_align(TextAlign p_align);
	TextAlign get_text_align() const;

	Button(const String &p_text = String());
	~Button();
};

VARIANT_ENUM_CAST(Button::TextAlign);

#endif
