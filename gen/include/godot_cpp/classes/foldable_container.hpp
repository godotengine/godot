/**************************************************************************/
/*  foldable_container.hpp                                                */
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

#include <godot_cpp/classes/container.hpp>
#include <godot_cpp/classes/control.hpp>
#include <godot_cpp/classes/global_constants.hpp>
#include <godot_cpp/classes/ref.hpp>
#include <godot_cpp/classes/text_server.hpp>
#include <godot_cpp/variant/string.hpp>

#include <godot_cpp/core/class_db.hpp>

#include <type_traits>

namespace godot {

class FoldableGroup;

class FoldableContainer : public Container {
	GDEXTENSION_CLASS(FoldableContainer, Container)

public:
	enum TitlePosition {
		POSITION_TOP = 0,
		POSITION_BOTTOM = 1,
	};

	void fold();
	void expand();
	void set_folded(bool p_folded);
	bool is_folded() const;
	void set_foldable_group(const Ref<FoldableGroup> &p_button_group);
	Ref<FoldableGroup> get_foldable_group() const;
	void set_title(const String &p_text);
	String get_title() const;
	void set_title_alignment(HorizontalAlignment p_alignment);
	HorizontalAlignment get_title_alignment() const;
	void set_language(const String &p_language);
	String get_language() const;
	void set_title_text_direction(Control::TextDirection p_text_direction);
	Control::TextDirection get_title_text_direction() const;
	void set_title_text_overrun_behavior(TextServer::OverrunBehavior p_overrun_behavior);
	TextServer::OverrunBehavior get_title_text_overrun_behavior() const;
	void set_title_position(FoldableContainer::TitlePosition p_title_position);
	FoldableContainer::TitlePosition get_title_position() const;
	void add_title_bar_control(Control *p_control);
	void remove_title_bar_control(Control *p_control);

protected:
	template <typename T, typename B>
	static void register_virtuals() {
		Container::register_virtuals<T, B>();
	}

public:
};

} // namespace godot

VARIANT_ENUM_CAST(FoldableContainer::TitlePosition);

