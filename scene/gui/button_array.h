/*************************************************************************/
/*  button_array.h                                                       */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                    http://www.godotengine.org                         */
/*************************************************************************/
/* Copyright (c) 2007-2017 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2017 Godot Engine contributors (cf. AUTHORS.md)    */
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
#ifndef BUTTON_ARRAY_H
#define BUTTON_ARRAY_H

#include "scene/gui/control.h"

class ButtonArray : public Control {

	GDCLASS(ButtonArray, Control);

public:
	enum Align {
		ALIGN_BEGIN,
		ALIGN_CENTER,
		ALIGN_END,
		ALIGN_FILL,
		ALIGN_EXPAND_FILL
	};

private:
	Orientation orientation;
	Align align;

	struct Button {

		String text;
		String xl_text;
		String tooltip;
		Ref<Texture> icon;
		mutable int _ms_cache;
		mutable int _pos_cache;
		mutable int _size_cache;
	};

	int selected;
	int hover;
	bool flat;
	double min_button_size;

	Vector<Button> buttons;

protected:
	bool _set(const StringName &p_name, const Variant &p_value);
	bool _get(const StringName &p_name, Variant &r_ret) const;
	void _get_property_list(List<PropertyInfo> *p_list) const;

	void _notification(int p_what);
	static void _bind_methods();

public:
	void _gui_input(const InputEvent &p_event);

	void set_align(Align p_align);
	Align get_align() const;

	void set_flat(bool p_flat);
	bool is_flat() const;

	void add_button(const String &p_button, const String &p_tooltip = "");
	void add_icon_button(const Ref<Texture> &p_icon, const String &p_button = "", const String &p_tooltip = "");

	void set_button_text(int p_button, const String &p_text);
	void set_button_tooltip(int p_button, const String &p_text);
	void set_button_icon(int p_button, const Ref<Texture> &p_icon);

	String get_button_text(int p_button) const;
	String get_button_tooltip(int p_button) const;
	Ref<Texture> get_button_icon(int p_button) const;

	int get_selected() const;
	int get_hovered() const;
	void set_selected(int p_selected);

	int get_button_count() const;

	void erase_button(int p_button);
	void clear();

	virtual Size2 get_minimum_size() const;

	virtual void get_translatable_strings(List<String> *p_strings) const;
	virtual String get_tooltip(const Point2 &p_pos) const;

	ButtonArray(Orientation p_orientation = HORIZONTAL);
};

class HButtonArray : public ButtonArray {
	GDCLASS(HButtonArray, ButtonArray);

public:
	HButtonArray()
		: ButtonArray(HORIZONTAL){};
};

class VButtonArray : public ButtonArray {
	GDCLASS(VButtonArray, ButtonArray);

public:
	VButtonArray()
		: ButtonArray(VERTICAL){};
};

#endif // BUTTON_ARRAY_H
