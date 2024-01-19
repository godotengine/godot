/**************************************************************************/
/*  compat_window_wrapper.h                                                      */
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

// * This is a port of a WindowWrapper from the Godot Engine to use with godot-cpp.

#ifndef COMPAT_WINDOW_WRAPPER_H
#define COMPAT_WINDOW_WRAPPER_H

#ifdef LIMBOAI_MODULE
#include "editor/window_wrapper.h"

#define CompatWindowWrapper WindowWrapper
#define CompatShortcutBin ShortcutBin
#define CompatScreenSelect ScreenSelect

#endif // LIMBOAI_MODULE

#ifdef LIMBOAI_GDEXTENSION
#include <godot_cpp/classes/h_box_container.hpp>
#include <godot_cpp/classes/input_event.hpp>
#include <godot_cpp/classes/label.hpp>
#include <godot_cpp/classes/margin_container.hpp>
#include <godot_cpp/classes/menu_button.hpp>
#include <godot_cpp/classes/panel.hpp>
#include <godot_cpp/classes/popup.hpp>
#include <godot_cpp/classes/shortcut.hpp>
#include <godot_cpp/classes/window.hpp>
#include <godot_cpp/variant/rect2.hpp>

using namespace godot;

// Capture all shortcut events not handled by other nodes.
class CompatShortcutBin : public Node {
	GDCLASS(CompatShortcutBin, Node);

protected:
	virtual void _notification(int what);

	static void _bind_methods();

public:
	virtual void _shortcut_input(const Ref<InputEvent> &p_event) override;
};

class CompatWindowWrapper : public MarginContainer {
	GDCLASS(CompatWindowWrapper, MarginContainer);

	Control *wrapped_control = nullptr;
	MarginContainer *margins = nullptr;
	godot::Window *window = nullptr;

	Panel *window_background = nullptr;

	Ref<Shortcut> enable_shortcut;

	Rect2 _get_default_window_rect() const;
	Node *_get_wrapped_control_parent() const;

	void _set_window_enabled_with_rect(bool p_visible, const Rect2 p_rect);
	void _set_window_rect(const Rect2 p_rect);

protected:
	static void _bind_methods();
	void _notification(int p_what);

public:
	virtual void _shortcut_input(const Ref<InputEvent> &p_event) override;

	void set_wrapped_control(Control *p_control, const Ref<Shortcut> &p_enable_shortcut = Ref<Shortcut>());
	Control *get_wrapped_control() const;
	Control *release_wrapped_control();

	bool is_window_available() const;

	bool get_window_enabled() const;
	void set_window_enabled(bool p_enabled);

	Rect2i get_window_rect() const;
	int get_window_screen() const;

	void restore_window(const Rect2i &p_rect, int p_screen = -1);
	void restore_window_from_saved_position(const Rect2 p_window_rect, int p_screen, const Rect2 p_screen_rect);
	void enable_window_on_screen(int p_screen = -1, bool p_auto_scale = false);

	void set_window_title(const String p_title);
	void set_margins_enabled(bool p_enabled);

	CompatWindowWrapper();
};

class CompatScreenSelect : public Button {
	GDCLASS(CompatScreenSelect, Button);

	Popup *popup = nullptr;
	Panel *popup_background = nullptr;
	godot::HBoxContainer *screen_list = nullptr;

	void _build_advanced_menu();

	void _emit_screen_signal(int p_screen_idx);
	void _handle_mouse_shortcut(const Ref<InputEvent> &p_event);
	void _show_popup();

protected:
	static void _bind_methods();

	void _notification(int p_what);

public:
	virtual void _pressed() override;

	CompatScreenSelect();
};

#endif // ! LIMBOAI_GDEXTENSION

#endif // COMPAT_WINDOW_WRAPPER_H
