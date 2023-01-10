/**************************************************************************/
/*  button.h                                                              */
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

#ifndef BUTTON_H
#define BUTTON_H

#include "scene/gui/base_button.h"

class Button : public BaseButton {
	GDCLASS(Button, BaseButton);

public:
	enum TextAlign {
		ALIGN_LEFT,
		ALIGN_CENTER,
		ALIGN_RIGHT
	};

private:
	bool flat;
	String text;
	String xl_text;
	Ref<Texture> icon;
	bool expand_icon;
	bool clip_text;
	TextAlign align;
	TextAlign icon_align;
	float _internal_margin[4];

protected:
	void _set_internal_margin(Margin p_margin, float p_value);
	void _notification(int p_what);
	static void _bind_methods();

public:
	virtual Size2 get_minimum_size() const;

	void set_text(const String &p_text);
	String get_text() const;

	void set_icon(const Ref<Texture> &p_icon);
	Ref<Texture> get_icon() const;

	void set_expand_icon(bool p_expand_icon);
	bool is_expand_icon() const;

	void set_flat(bool p_flat);
	bool is_flat() const;

	void set_clip_text(bool p_clip_text);
	bool get_clip_text() const;

	void set_text_align(TextAlign p_align);
	TextAlign get_text_align() const;

	void set_icon_align(TextAlign p_align);
	TextAlign get_icon_align() const;

	Button(const String &p_text = String());
	~Button();
};

VARIANT_ENUM_CAST(Button::TextAlign);

#endif // BUTTON_H
