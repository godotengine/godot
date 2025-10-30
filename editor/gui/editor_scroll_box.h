/**************************************************************************/
/*  editor_scroll_box.h                                                   */
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

#pragma once

#include "scene/gui/box_container.h"

class Button;
class ScrollContainer;

class EditorScrollBox : public HBoxContainer {
	GDCLASS(EditorScrollBox, HBoxContainer)

	struct ThemeCache {
		Ref<Texture2D> arrow_left;
		Ref<Texture2D> arrow_right;
	} theme_cache;

	Button *left_button = nullptr;
	Button *right_button = nullptr;
	ScrollContainer *scroll_container = nullptr;
	Control *control = nullptr;

	void _scroll(bool p_right);
	void _accessibility_action_scroll_right(const Variant &p_data);
	void _accessibility_action_scroll_left(const Variant &p_data);
	void _update_buttons();
	void _update_disabled_buttons();
	void _update_buttons_icon_and_tooltip();
	void _update_scroll_container();

protected:
	void _notification(int p_what);
	static void _bind_methods();

public:
	void set_control(Control *p_control);
	Control *get_control() const { return control; }
	bool has_control() const;

	void ensure_control_visible(ObjectID p_id);

	Button *get_left_button() const;
	Button *get_right_button() const;

	ScrollContainer *get_scroll_container() const { return scroll_container; }

	EditorScrollBox();
};
