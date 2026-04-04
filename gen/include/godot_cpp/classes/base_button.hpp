/**************************************************************************/
/*  base_button.hpp                                                       */
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

#include <godot_cpp/classes/control.hpp>
#include <godot_cpp/classes/global_constants.hpp>
#include <godot_cpp/classes/ref.hpp>

#include <godot_cpp/core/class_db.hpp>

#include <type_traits>

namespace godot {

class ButtonGroup;
class Shortcut;

class BaseButton : public Control {
	GDEXTENSION_CLASS(BaseButton, Control)

public:
	enum DrawMode {
		DRAW_NORMAL = 0,
		DRAW_PRESSED = 1,
		DRAW_HOVER = 2,
		DRAW_DISABLED = 3,
		DRAW_HOVER_PRESSED = 4,
	};

	enum ActionMode {
		ACTION_MODE_BUTTON_PRESS = 0,
		ACTION_MODE_BUTTON_RELEASE = 1,
	};

	void set_pressed(bool p_pressed);
	bool is_pressed() const;
	void set_pressed_no_signal(bool p_pressed);
	bool is_hovered() const;
	void set_toggle_mode(bool p_enabled);
	bool is_toggle_mode() const;
	void set_shortcut_in_tooltip(bool p_enabled);
	bool is_shortcut_in_tooltip_enabled() const;
	void set_disabled(bool p_disabled);
	bool is_disabled() const;
	void set_action_mode(BaseButton::ActionMode p_mode);
	BaseButton::ActionMode get_action_mode() const;
	void set_button_mask(BitField<MouseButtonMask> p_mask);
	BitField<MouseButtonMask> get_button_mask() const;
	BaseButton::DrawMode get_draw_mode() const;
	void set_keep_pressed_outside(bool p_enabled);
	bool is_keep_pressed_outside() const;
	void set_shortcut_feedback(bool p_enabled);
	bool is_shortcut_feedback() const;
	void set_shortcut(const Ref<Shortcut> &p_shortcut);
	Ref<Shortcut> get_shortcut() const;
	void set_button_group(const Ref<ButtonGroup> &p_button_group);
	Ref<ButtonGroup> get_button_group() const;
	virtual void _pressed();
	virtual void _toggled(bool p_toggled_on);

protected:
	template <typename T, typename B>
	static void register_virtuals() {
		Control::register_virtuals<T, B>();
		if constexpr (!std::is_same_v<decltype(&B::_pressed), decltype(&T::_pressed)>) {
			BIND_VIRTUAL_METHOD(T, _pressed, 3218959716);
		}
		if constexpr (!std::is_same_v<decltype(&B::_toggled), decltype(&T::_toggled)>) {
			BIND_VIRTUAL_METHOD(T, _toggled, 2586408642);
		}
	}

public:
};

} // namespace godot

VARIANT_ENUM_CAST(BaseButton::DrawMode);
VARIANT_ENUM_CAST(BaseButton::ActionMode);

