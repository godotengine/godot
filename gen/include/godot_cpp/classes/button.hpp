/**************************************************************************/
/*  button.hpp                                                            */
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

#include <godot_cpp/classes/base_button.hpp>
#include <godot_cpp/classes/control.hpp>
#include <godot_cpp/classes/global_constants.hpp>
#include <godot_cpp/classes/ref.hpp>
#include <godot_cpp/classes/text_server.hpp>
#include <godot_cpp/variant/string.hpp>

#include <godot_cpp/core/class_db.hpp>

#include <type_traits>

namespace godot {

class Texture2D;

class Button : public BaseButton {
	GDEXTENSION_CLASS(Button, BaseButton)

public:
	void set_text(const String &p_text);
	String get_text() const;
	void set_text_overrun_behavior(TextServer::OverrunBehavior p_overrun_behavior);
	TextServer::OverrunBehavior get_text_overrun_behavior() const;
	void set_autowrap_mode(TextServer::AutowrapMode p_autowrap_mode);
	TextServer::AutowrapMode get_autowrap_mode() const;
	void set_autowrap_trim_flags(BitField<TextServer::LineBreakFlag> p_autowrap_trim_flags);
	BitField<TextServer::LineBreakFlag> get_autowrap_trim_flags() const;
	void set_text_direction(Control::TextDirection p_direction);
	Control::TextDirection get_text_direction() const;
	void set_language(const String &p_language);
	String get_language() const;
	void set_button_icon(const Ref<Texture2D> &p_texture);
	Ref<Texture2D> get_button_icon() const;
	void set_flat(bool p_enabled);
	bool is_flat() const;
	void set_clip_text(bool p_enabled);
	bool get_clip_text() const;
	void set_text_alignment(HorizontalAlignment p_alignment);
	HorizontalAlignment get_text_alignment() const;
	void set_icon_alignment(HorizontalAlignment p_icon_alignment);
	HorizontalAlignment get_icon_alignment() const;
	void set_vertical_icon_alignment(VerticalAlignment p_vertical_icon_alignment);
	VerticalAlignment get_vertical_icon_alignment() const;
	void set_expand_icon(bool p_enabled);
	bool is_expand_icon() const;

protected:
	template <typename T, typename B>
	static void register_virtuals() {
		BaseButton::register_virtuals<T, B>();
	}

public:
};

} // namespace godot

