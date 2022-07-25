/*************************************************************************/
/*  option_button.h                                                      */
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

#ifndef OPTION_BUTTON_H
#define OPTION_BUTTON_H

#include "scene/gui/button.h"
#include "scene/gui/popup_menu.h"

class OptionButton : public Button {
	GDCLASS(OptionButton, Button);

	PopupMenu *popup;
	int current;

	void _focused(int p_which);
	void _selected(int p_which);
	void _select(int p_which, bool p_emit = false);
	void _select_int(int p_which);

	Array _get_items() const;
	void _set_items(const Array &p_items);

	virtual void pressed();

protected:
	Size2 get_minimum_size() const;
	void _notification(int p_what);
	static void _bind_methods();

public:
	void add_icon_item(const Ref<Texture> &p_icon, const String &p_label, int p_id = -1);
	void add_item(const String &p_label, int p_id = -1);

	void set_item_text(int p_idx, const String &p_text);
	void set_item_icon(int p_idx, const Ref<Texture> &p_icon);
	void set_item_id(int p_idx, int p_id);
	void set_item_metadata(int p_idx, const Variant &p_metadata);
	void set_item_disabled(int p_idx, bool p_disabled);
	void set_item_tooltip(int p_idx, const String &p_tooltip);

	String get_item_text(int p_idx) const;
	Ref<Texture> get_item_icon(int p_idx) const;
	int get_item_id(int p_idx) const;
	int get_item_index(int p_id) const;
	Variant get_item_metadata(int p_idx) const;
	bool is_item_disabled(int p_idx) const;
	String get_item_tooltip(int p_idx) const;

	int get_item_count() const;

	void add_separator();

	void clear();

	void select(int p_idx);
	int get_selected() const;
	int get_selected_id() const;
	Variant get_selected_metadata() const;

	void remove_item(int p_idx);

	PopupMenu *get_popup() const;

	virtual void get_translatable_strings(List<String> *p_strings) const;

	OptionButton();
	~OptionButton();
};

#endif // OPTION_BUTTON_H
