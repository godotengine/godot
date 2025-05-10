/**************************************************************************/
/*  editor_asset_library_pagination.cpp                                   */
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

#include "editor_asset_library_pagination.h"

#include "editor/themes/editor_scale.h"
#include "scene/gui/button.h"
#include "scene/gui/separator.h"

static Button *_make_pagination_button(Node *p_parent) {
	Button *btn = memnew(Button);
	btn->set_auto_translate_mode(Button::AUTO_TRANSLATE_MODE_DISABLED);
	btn->set_theme_type_variation("PanelBackgroundButton");
	p_parent->add_child(btn);
	return btn;
}

void EditorAssetLibraryPagination::_bind_methods() {
	ADD_SIGNAL(MethodInfo("page_pressed", PropertyInfo(Variant::INT, "page")));
}

void EditorAssetLibraryPagination::_notification(int p_what) {
	switch (p_what) {
		case NOTIFICATION_TRANSLATION_CHANGED: {
			btn_first->set_text(TTR("First", "Pagination"));
			btn_prev->set_text(TTR("Previous", "Pagination"));
			btn_next->set_text(TTR("Next", "Pagination"));
			btn_last->set_text(TTR("Last", "Pagination"));
		} break;
	}
}

void EditorAssetLibraryPagination::_go_first() {
	ERR_FAIL_COND(page <= 0);
	emit_signal(SNAME("page_pressed"), 0);
}

void EditorAssetLibraryPagination::_go_prev() {
	ERR_FAIL_COND(page <= 0);
	emit_signal(SNAME("page_pressed"), page - 1);
}

void EditorAssetLibraryPagination::_go_next() {
	ERR_FAIL_COND(page + 1 >= page_count);
	emit_signal(SNAME("page_pressed"), page + 1);
}

void EditorAssetLibraryPagination::_go_last() {
	ERR_FAIL_COND(page + 1 >= page_count);
	emit_signal(SNAME("page_pressed"), page_count - 1);
}

void EditorAssetLibraryPagination::_go_page(int p_page) {
	ERR_FAIL_INDEX(p_page, page_count);
	emit_signal(SNAME("page_pressed"), p_page);
}

void EditorAssetLibraryPagination::setup(int p_page, int p_page_count) {
	page = p_page;
	page_count = p_page_count;

	while (page_numbers->get_child_count() > 0) {
		memdelete(page_numbers->get_child(0));
	}

	if (page_count < 2) {
		hide();
		return;
	}

	const int from = MAX(0, page - (5 / EDSCALE));
	const int to = MIN(from + (10 / EDSCALE), page_count);

	const bool no_prev = page <= 0;
	btn_first->set_disabled(no_prev);
	btn_first->set_focus_mode(no_prev ? FOCUS_ACCESSIBILITY : FOCUS_ALL);
	btn_prev->set_disabled(no_prev);
	btn_prev->set_focus_mode(no_prev ? FOCUS_ACCESSIBILITY : FOCUS_ALL);

	for (int i = from; i < to; i++) {
		Button *current = _make_pagination_button(page_numbers);
		current->set_custom_minimum_size(Size2(30 * EDSCALE, 0));
		current->set_text(itos(i + 1));
		current->set_disabled(i == page);
		current->set_focus_mode(i == page ? FOCUS_ACCESSIBILITY : FOCUS_ALL);
		current->connect(SceneStringName(pressed), callable_mp(this, &EditorAssetLibraryPagination::_go_page).bind(i));
	}

	const bool no_next = page + 1 >= page_count;
	btn_next->set_disabled(no_next);
	btn_next->set_focus_mode(no_next ? FOCUS_ACCESSIBILITY : FOCUS_ALL);
	btn_last->set_disabled(no_next);
	btn_last->set_focus_mode(no_next ? FOCUS_ACCESSIBILITY : FOCUS_ALL);

	show();
}

EditorAssetLibraryPagination::EditorAssetLibraryPagination() {
	set_alignment(ALIGNMENT_CENTER);
	add_theme_constant_override("separation", 5 * EDSCALE);

	btn_first = _make_pagination_button(this);
	btn_first->connect(SceneStringName(pressed), callable_mp(this, &EditorAssetLibraryPagination::_go_first));
	btn_prev = _make_pagination_button(this);
	btn_prev->connect(SceneStringName(pressed), callable_mp(this, &EditorAssetLibraryPagination::_go_prev));

	add_child(memnew(VSeparator));

	page_numbers = memnew(HBoxContainer);
	page_numbers->add_theme_constant_override("separation", 5 * EDSCALE);
	add_child(page_numbers);

	add_child(memnew(VSeparator));

	btn_next = _make_pagination_button(this);
	btn_next->connect(SceneStringName(pressed), callable_mp(this, &EditorAssetLibraryPagination::_go_next));
	btn_last = _make_pagination_button(this);
	btn_last->connect(SceneStringName(pressed), callable_mp(this, &EditorAssetLibraryPagination::_go_last));

	hide();
}
