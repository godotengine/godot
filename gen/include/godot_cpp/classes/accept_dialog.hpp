/**************************************************************************/
/*  accept_dialog.hpp                                                     */
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

#include <godot_cpp/classes/window.hpp>
#include <godot_cpp/variant/string.hpp>

#include <godot_cpp/core/class_db.hpp>

#include <type_traits>

namespace godot {

class Button;
class Label;
class LineEdit;

class AcceptDialog : public Window {
	GDEXTENSION_CLASS(AcceptDialog, Window)

public:
	Button *get_ok_button();
	Label *get_label();
	void set_hide_on_ok(bool p_enabled);
	bool get_hide_on_ok() const;
	void set_close_on_escape(bool p_enabled);
	bool get_close_on_escape() const;
	Button *add_button(const String &p_text, bool p_right = false, const String &p_action = String());
	Button *add_cancel_button(const String &p_name);
	void remove_button(Button *p_button);
	void register_text_enter(LineEdit *p_line_edit);
	void set_text(const String &p_text);
	String get_text() const;
	void set_autowrap(bool p_autowrap);
	bool has_autowrap();
	void set_ok_button_text(const String &p_text);
	String get_ok_button_text() const;

protected:
	template <typename T, typename B>
	static void register_virtuals() {
		Window::register_virtuals<T, B>();
	}

public:
};

} // namespace godot

