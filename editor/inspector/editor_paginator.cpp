/**************************************************************************/
/*  editor_paginator.cpp                                                  */
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

#include "editor_paginator.h"

#include "core/object/callable_mp.h"
#include "scene/gui/button.h"
#include "scene/gui/label.h"
#include "scene/gui/line_edit.h"

void EditorPaginator::_first_page_button_pressed() {
	emit_signal("page_changed", 0);
}

void EditorPaginator::_prev_page_button_pressed() {
	emit_signal("page_changed", MAX(0, page - 1));
}

void EditorPaginator::_page_line_edit_text_submitted(const String &p_text) {
	if (p_text.is_valid_int()) {
		int new_page = p_text.to_int() - 1;
		new_page = MIN(MAX(0, new_page), max_page);
		page_line_edit->set_text(Variant(new_page));
		emit_signal("page_changed", new_page);
	} else {
		page_line_edit->set_text(Variant(page));
	}
}

void EditorPaginator::_next_page_button_pressed() {
	emit_signal("page_changed", MIN(max_page, page + 1));
}

void EditorPaginator::_last_page_button_pressed() {
	emit_signal("page_changed", max_page);
}

void EditorPaginator::update(int p_page, int p_max_page) {
	page = p_page;
	max_page = p_max_page;

	// Update buttons.
	first_page_button->set_disabled(page == 0);
	prev_page_button->set_disabled(page == 0);
	next_page_button->set_disabled(page == max_page);
	last_page_button->set_disabled(page == max_page);

	// Update page number and page count.
	page_line_edit->set_text(vformat("%d", page + 1));
	page_count_label->set_text(vformat("/ %d", max_page + 1));
}

void EditorPaginator::_notification(int p_what) {
	switch (p_what) {
		case NOTIFICATION_THEME_CHANGED: {
			first_page_button->set_button_icon(get_editor_theme_icon(SNAME("PageFirst")));
			prev_page_button->set_button_icon(get_editor_theme_icon(SNAME("PagePrevious")));
			next_page_button->set_button_icon(get_editor_theme_icon(SNAME("PageNext")));
			last_page_button->set_button_icon(get_editor_theme_icon(SNAME("PageLast")));
		} break;
	}
}

void EditorPaginator::_bind_methods() {
	ADD_SIGNAL(MethodInfo("page_changed", PropertyInfo(Variant::INT, "page")));
}

EditorPaginator::EditorPaginator() {
	set_h_size_flags(SIZE_EXPAND_FILL);
	set_alignment(ALIGNMENT_CENTER);

	first_page_button = memnew(Button);
	first_page_button->set_accessibility_name(TTRC("First Page"));
	first_page_button->set_flat(true);
	first_page_button->connect(SceneStringName(pressed), callable_mp(this, &EditorPaginator::_first_page_button_pressed));
	add_child(first_page_button);

	prev_page_button = memnew(Button);
	prev_page_button->set_accessibility_name(TTRC("Previous Page"));
	prev_page_button->set_flat(true);
	prev_page_button->connect(SceneStringName(pressed), callable_mp(this, &EditorPaginator::_prev_page_button_pressed));
	add_child(prev_page_button);

	page_line_edit = memnew(LineEdit);
	page_line_edit->set_accessibility_name(TTRC("Page Number"));
	page_line_edit->connect(SceneStringName(text_submitted), callable_mp(this, &EditorPaginator::_page_line_edit_text_submitted));
	page_line_edit->add_theme_constant_override("minimum_character_width", 2);
	add_child(page_line_edit);

	page_count_label = memnew(Label);
	page_count_label->set_focus_mode(FOCUS_ACCESSIBILITY);
	add_child(page_count_label);

	next_page_button = memnew(Button);
	prev_page_button->set_accessibility_name(TTRC("Next Page"));
	next_page_button->set_flat(true);
	next_page_button->connect(SceneStringName(pressed), callable_mp(this, &EditorPaginator::_next_page_button_pressed));
	add_child(next_page_button);

	last_page_button = memnew(Button);
	last_page_button->set_accessibility_name(TTRC("Last Page"));
	last_page_button->set_flat(true);
	last_page_button->connect(SceneStringName(pressed), callable_mp(this, &EditorPaginator::_last_page_button_pressed));
	add_child(last_page_button);
}
