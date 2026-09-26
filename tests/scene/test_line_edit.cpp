/**************************************************************************/
/*  test_line_edit.cpp                                                    */
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

#include "tests/test_macros.h"

TEST_FORCE_LINK(test_line_edit)

#ifndef ADVANCED_GUI_DISABLED

#include "core/math/math_defs.h"
#include "scene/gui/line_edit.h"
#include "scene/main/scene_tree.h"
#include "scene/main/window.h"
#include "tests/signal_watcher.h"

namespace TestLineEdit {

TEST_CASE("[SceneTree][LineEdit] Default properties") {
	LineEdit *le = memnew(LineEdit);

	CHECK(le->get_text().is_empty());
	CHECK(le->is_editable()); // Constructor calls set_editable(true).
	CHECK(le->get_max_length() == 0); // 0 means unlimited.
	CHECK_FALSE(le->is_secret());
	CHECK_FALSE(le->has_selection());
	CHECK(le->get_caret_column() == 0);

	memdelete(le);
}

TEST_CASE("[SceneTree][LineEdit] Placeholder set via constructor") {
	LineEdit *le = memnew(LineEdit("placeholder"));
	CHECK(le->get_placeholder() == "placeholder");
	memdelete(le);
}

TEST_CASE("[SceneTree][LineEdit] Text and caret basics") {
	LineEdit *le = memnew(LineEdit);

	le->set_text("hello");
	CHECK(le->get_text() == "hello");

	le->set_caret_column(100);
	CHECK(le->get_caret_column() == 5);

	le->set_caret_column(-10);
	CHECK(le->get_caret_column() == 0);

	memdelete(le);
}

TEST_CASE("[SceneTree][LineEdit] max_length clamps set_text") {
	LineEdit *le = memnew(LineEdit);
	le->set_max_length(3);
	le->set_text("abcdef");
	CHECK(le->get_text() == "abc");
	CHECK(le->get_max_length() == 3);

	memdelete(le);
}

TEST_CASE("[SceneTree][LineEdit] insert_text_at_caret truncates and emits text_change_rejected") {
	LineEdit *le = memnew(LineEdit);
	le->set_max_length(4);
	le->set_text("ab");
	le->set_caret_column(2);

	SIGNAL_WATCH(le, "text_change_rejected");
	le->insert_text_at_caret("cde");
	SIGNAL_CHECK("text_change_rejected", { { String("e") } });
	CHECK(le->get_text() == "abcd");
	CHECK(le->get_caret_column() == 4);

	SIGNAL_UNWATCH(le, "text_change_rejected");
	memdelete(le);
}

TEST_CASE("[SceneTree][LineEdit] clear emits text_changed with empty string") {
	LineEdit *le = memnew(LineEdit);
	le->set_text("x");

	SIGNAL_WATCH(le, "text_changed");
	le->clear();
	SIGNAL_CHECK("text_changed", { { String("") } });
	CHECK(le->get_text().is_empty());

	SIGNAL_UNWATCH(le, "text_changed");
	memdelete(le);
}

TEST_CASE("[SceneTree][LineEdit] delete_text removes middle substring") {
	LineEdit *le = memnew(LineEdit);
	le->set_text("abcdef");
	le->delete_text(2, 4);
	CHECK(le->get_text() == "abef");

	memdelete(le);
}

TEST_CASE("[SceneTree][LineEdit] delete_char removes one character before caret") {
	LineEdit *le = memnew(LineEdit);
	le->set_text("ab");
	le->set_caret_column(2);
	le->delete_char();
	CHECK(le->get_text() == "a");
	CHECK(le->get_caret_column() == 1);

	memdelete(le);
}

TEST_CASE("[SceneTree][LineEdit] Secret mode") {
	LineEdit *le = memnew(LineEdit);

	SUBCASE("secret mode preserves underlying text") {
		le->set_secret(true);
		le->set_text("secret");
		CHECK(le->is_secret());
		CHECK(le->get_text() == "secret");

		le->set_secret_character("*");
		CHECK(le->get_secret_character() == "*");
	}

	SUBCASE("secret_character longer than one character is truncated on set") {
		le->set_secret_character("**");
		CHECK(le->get_secret_character() == "*");
		// Stored value is already one character; configuration warnings are not used for this path.
		CHECK(le->get_configuration_warnings().is_empty());
	}

	memdelete(le);
}

TEST_CASE("[SceneTree][LineEdit] expand_to_text_length affects combined minimum width in tree") {
	LineEdit *le = memnew(LineEdit);
	Window *root = SceneTree::get_singleton()->get_root();
	root->add_child(le);

	le->set_text("Wide enough sample text for width comparison.");
	SceneTree::get_singleton()->process(0);
	const real_t w_off = le->get_combined_minimum_size().width;

	le->set_expand_to_text_length_enabled(true);
	SceneTree::get_singleton()->process(0);
	const real_t w_on = le->get_combined_minimum_size().width;

	CHECK(w_on >= w_off);

	memdelete(le);
}

TEST_CASE("[SceneTree][LineEdit] Selection") {
	LineEdit *le = memnew(LineEdit);
	le->set_text("abcdef");

	SUBCASE("select and deselect") {
		le->select(1, 3);
		CHECK(le->has_selection());
		CHECK(le->get_selected_text() == "bc");
		CHECK(le->get_selection_from_column() == 1);
		CHECK(le->get_selection_to_column() == 3);

		le->deselect();
		CHECK_FALSE(le->has_selection());
	}

	SUBCASE("select_all covers the full text") {
		le->select_all();
		CHECK(le->has_selection());
		CHECK(le->get_selected_text() == "abcdef");
	}

	SUBCASE("select(0, 0) clears selection") {
		le->select(0, 4);
		le->select(0, 0);
		CHECK_FALSE(le->has_selection());
	}

	SUBCASE("set_text_with_selection clamps selection to new length") {
		le->set_text("hello");
		le->select(1, 4);
		le->set_text_with_selection("ab");
		CHECK(le->get_text() == "ab");
		CHECK(le->has_selection());
		CHECK(le->get_selection_from_column() == 1);
		CHECK(le->get_selection_to_column() == 2);
	}

	SUBCASE("selection_delete removes selected range") {
		le->select(2, 5);
		le->selection_delete();
		CHECK(le->get_text() == "abf");
		CHECK_FALSE(le->has_selection());
	}

	memdelete(le);
}

TEST_CASE("[SceneTree][LineEdit] Undo and redo") {
	LineEdit *le = memnew(LineEdit);
	// Each set_text clears the undo stack; use a single set_text then undo/redo.
	le->set_text("ab");

	SUBCASE("undo reverts to empty and redo restores text") {
		CHECK(le->has_undo());

		le->undo();
		CHECK(le->get_text().is_empty());
		CHECK(le->has_redo());

		le->redo();
		CHECK(le->get_text() == "ab");
	}

	SUBCASE("undo is a no-op when not editable") {
		le->set_editable(false);
		le->undo();
		CHECK(le->get_text() == "ab");
	}

	memdelete(le);
}

} // namespace TestLineEdit

#endif // ADVANCED_GUI_DISABLED
