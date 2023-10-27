/**************************************************************************/
/*  test_text_edit.h                                                      */
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

#ifndef TEST_TEXT_EDIT_H
#define TEST_TEXT_EDIT_H

#include "scene/gui/text_edit.h"

#include "tests/test_macros.h"

namespace TestTextEdit {

TEST_CASE("[SceneTree][TextEdit] text entry") {
	SceneTree::get_singleton()->get_root()->set_physics_object_picking(false);
	TextEdit *text_edit = memnew(TextEdit);
	SceneTree::get_singleton()->get_root()->add_child(text_edit);
	text_edit->grab_focus();

	Array empty_signal_args;
	empty_signal_args.push_back(Array());

	SUBCASE("[TextEdit] text entry") {
		SIGNAL_WATCH(text_edit, "text_set");
		SIGNAL_WATCH(text_edit, "text_changed");
		SIGNAL_WATCH(text_edit, "lines_edited_from");
		SIGNAL_WATCH(text_edit, "caret_changed");

		Array args1;
		args1.push_back(0);
		args1.push_back(0);
		Array lines_edited_args;
		lines_edited_args.push_back(args1);
		lines_edited_args.push_back(args1.duplicate());

		SUBCASE("[TextEdit] clear and set text") {
			// "text_changed" should not be emitted on clear / set.
			text_edit->clear();
			MessageQueue::get_singleton()->flush();
			CHECK(text_edit->get_text() == "");
			CHECK(text_edit->get_caret_column() == 0);
			CHECK(text_edit->get_line_count() == 1);
			SIGNAL_CHECK("text_set", empty_signal_args);
			SIGNAL_CHECK("lines_edited_from", lines_edited_args);
			SIGNAL_CHECK_FALSE("caret_changed");
			SIGNAL_CHECK_FALSE("text_changed");

			text_edit->set_text("test text");
			MessageQueue::get_singleton()->flush();
			CHECK(text_edit->get_text() == "test text");
			CHECK(text_edit->get_caret_column() == 0);
			CHECK(text_edit->get_line_count() == 1);
			SIGNAL_CHECK("text_set", empty_signal_args);
			SIGNAL_CHECK("lines_edited_from", lines_edited_args);
			SIGNAL_CHECK("caret_changed", empty_signal_args);
			SIGNAL_CHECK_FALSE("text_changed");

			text_edit->clear();
			MessageQueue::get_singleton()->flush();
			CHECK(text_edit->get_text() == "");
			CHECK(text_edit->get_caret_column() == 0);
			SIGNAL_CHECK("text_set", empty_signal_args);
			SIGNAL_CHECK("lines_edited_from", lines_edited_args);
			SIGNAL_CHECK_FALSE("caret_changed");
			SIGNAL_CHECK_FALSE("text_changed");

			// Can undo / redo words when editable.
			text_edit->undo();
			MessageQueue::get_singleton()->flush();
			CHECK(text_edit->get_text() == "test text");
			CHECK(text_edit->get_caret_column() == 0);
			SIGNAL_CHECK("lines_edited_from", lines_edited_args);
			SIGNAL_CHECK("text_changed", empty_signal_args);
			SIGNAL_CHECK_FALSE("caret_changed");
			SIGNAL_CHECK_FALSE("text_set");

			text_edit->redo();
			MessageQueue::get_singleton()->flush();
			CHECK(text_edit->get_text() == "");
			CHECK(text_edit->get_caret_column() == 0);
			SIGNAL_CHECK("lines_edited_from", lines_edited_args);
			SIGNAL_CHECK("text_changed", empty_signal_args);
			SIGNAL_CHECK_FALSE("caret_changed");
			SIGNAL_CHECK_FALSE("text_set");

			// Cannot undo when not-editable but should still clear.
			text_edit->undo();
			MessageQueue::get_singleton()->flush();
			CHECK(text_edit->get_text() == "test text");
			CHECK(text_edit->get_caret_column() == 0);
			SIGNAL_CHECK("lines_edited_from", lines_edited_args);
			SIGNAL_CHECK("text_changed", empty_signal_args);
			SIGNAL_CHECK_FALSE("caret_changed");
			SIGNAL_CHECK_FALSE("text_set");

			// Clear.
			text_edit->set_editable(false);

			Array lines_edited_clear_args;
			Array new_args = args1.duplicate();
			new_args[0] = 1;
			lines_edited_clear_args.push_back(new_args);

			text_edit->clear();
			MessageQueue::get_singleton()->flush();
			CHECK(text_edit->get_text() == "");
			CHECK(text_edit->get_caret_column() == 0);
			SIGNAL_CHECK("text_set", empty_signal_args);
			SIGNAL_CHECK("lines_edited_from", lines_edited_clear_args);
			SIGNAL_CHECK_FALSE("caret_changed");
			SIGNAL_CHECK_FALSE("text_changed");

			text_edit->set_editable(true);

			text_edit->undo();
			MessageQueue::get_singleton()->flush();
			CHECK(text_edit->get_text() == "");
			CHECK(text_edit->get_caret_column() == 0);
			SIGNAL_CHECK_FALSE("text_set");
			SIGNAL_CHECK_FALSE("lines_edited_from");
			SIGNAL_CHECK_FALSE("text_changed");
			SIGNAL_CHECK_FALSE("caret_changed");

			// Can still undo set_text.
			text_edit->set_editable(false);

			text_edit->set_text("test text");
			MessageQueue::get_singleton()->flush();
			CHECK(text_edit->get_text() == "test text");
			CHECK(text_edit->get_caret_column() == 0);
			SIGNAL_CHECK("text_set", empty_signal_args);
			SIGNAL_CHECK("lines_edited_from", lines_edited_args);
			SIGNAL_CHECK("caret_changed", empty_signal_args);
			SIGNAL_CHECK_FALSE("text_changed");

			text_edit->set_editable(true);

			text_edit->undo();
			MessageQueue::get_singleton()->flush();
			CHECK(text_edit->get_text() == "");
			CHECK(text_edit->get_caret_column() == 0);
			SIGNAL_CHECK("lines_edited_from", lines_edited_args);
			SIGNAL_CHECK("text_changed", empty_signal_args);
			SIGNAL_CHECK_FALSE("caret_changed");
			SIGNAL_CHECK_FALSE("text_set");

			// Any selections are removed.
			text_edit->set_text("test text");
			MessageQueue::get_singleton()->flush();
			text_edit->select_all();
			SIGNAL_CHECK("caret_changed", empty_signal_args);
			MessageQueue::get_singleton()->flush();
			CHECK(text_edit->get_text() == "test text");
			CHECK(text_edit->get_caret_column() == 9);
			CHECK(text_edit->has_selection());
			SIGNAL_CHECK("text_set", empty_signal_args);
			SIGNAL_CHECK("lines_edited_from", lines_edited_args);
			SIGNAL_CHECK("caret_changed", empty_signal_args);
			SIGNAL_CHECK_FALSE("text_changed");

			text_edit->set_text("test");
			MessageQueue::get_singleton()->flush();
			CHECK(text_edit->get_text() == "test");
			CHECK(text_edit->get_caret_column() == 0);
			CHECK_FALSE(text_edit->has_selection());
			SIGNAL_CHECK("text_set", empty_signal_args);
			SIGNAL_CHECK("lines_edited_from", lines_edited_args);
			SIGNAL_CHECK("caret_changed", empty_signal_args);
			SIGNAL_CHECK_FALSE("text_changed");

			text_edit->select_all();
			MessageQueue::get_singleton()->flush();
			SIGNAL_CHECK("caret_changed", empty_signal_args);
			CHECK(text_edit->has_selection());

			text_edit->clear();
			MessageQueue::get_singleton()->flush();
			CHECK(text_edit->get_text() == "");
			CHECK(text_edit->get_caret_column() == 0);
			CHECK_FALSE(text_edit->has_selection());
			SIGNAL_CHECK("text_set", empty_signal_args);
			SIGNAL_CHECK("lines_edited_from", lines_edited_args);
			SIGNAL_CHECK("caret_changed", empty_signal_args);
			SIGNAL_CHECK_FALSE("text_changed");
		}

		SUBCASE("[TextEdit] set and get line") {
			// Set / Get line is 0 indexed.
			text_edit->set_line(1, "test");
			MessageQueue::get_singleton()->flush();
			CHECK(text_edit->get_text() == "");
			SIGNAL_CHECK_FALSE("lines_edited_from");
			SIGNAL_CHECK_FALSE("text_set");
			SIGNAL_CHECK_FALSE("text_changed");
			SIGNAL_CHECK_FALSE("caret_changed");

			text_edit->set_line(0, "test");
			MessageQueue::get_singleton()->flush();
			CHECK(text_edit->get_text() == "test");
			CHECK(text_edit->get_line(0) == "test");
			CHECK(text_edit->get_line(1) == "");
			SIGNAL_CHECK("lines_edited_from", lines_edited_args);
			SIGNAL_CHECK("text_changed", empty_signal_args);
			SIGNAL_CHECK_FALSE("text_set");
			SIGNAL_CHECK_FALSE("caret_changed");

			// Setting to a longer line, caret and selections should be preserved.
			text_edit->select_all();
			MessageQueue::get_singleton()->flush();
			CHECK(text_edit->has_selection());
			SIGNAL_CHECK("caret_changed", empty_signal_args);

			text_edit->set_line(0, "test text");
			MessageQueue::get_singleton()->flush();
			CHECK(text_edit->get_line(0) == "test text");
			CHECK(text_edit->has_selection());
			CHECK(text_edit->get_selected_text() == "test");
			SIGNAL_CHECK("lines_edited_from", lines_edited_args);
			SIGNAL_CHECK("text_changed", empty_signal_args);
			SIGNAL_CHECK_FALSE("caret_changed");
			SIGNAL_CHECK_FALSE("text_set");

			// Setting to a shorter line, selection and caret should be adjusted. Also works if not editable.
			text_edit->set_editable(false);
			text_edit->set_line(0, "te");
			MessageQueue::get_singleton()->flush();
			CHECK(text_edit->get_line(0) == "te");
			CHECK(text_edit->has_selection());
			CHECK(text_edit->get_selected_text() == "te");
			CHECK(text_edit->get_caret_column() == 2);
			SIGNAL_CHECK("lines_edited_from", lines_edited_args);
			SIGNAL_CHECK("caret_changed", empty_signal_args);
			SIGNAL_CHECK("text_changed", empty_signal_args);
			SIGNAL_CHECK_FALSE("text_set");
			text_edit->set_editable(true);

			// Undo / redo should work.
			text_edit->undo();
			MessageQueue::get_singleton()->flush();
			CHECK(text_edit->get_line(0) == "test text");
			CHECK(text_edit->has_selection());
			CHECK(text_edit->get_selected_text() == "test");
			CHECK(text_edit->get_caret_column() == 4);
			SIGNAL_CHECK("lines_edited_from", lines_edited_args);
			SIGNAL_CHECK("caret_changed", empty_signal_args);
			SIGNAL_CHECK("text_changed", empty_signal_args);
			SIGNAL_CHECK_FALSE("text_set");

			text_edit->redo();
			MessageQueue::get_singleton()->flush();
			CHECK(text_edit->get_line(0) == "te");
			CHECK(text_edit->has_selection());
			CHECK(text_edit->get_caret_column() == 2);
			SIGNAL_CHECK("lines_edited_from", lines_edited_args);
			SIGNAL_CHECK("caret_changed", empty_signal_args);
			SIGNAL_CHECK("text_changed", empty_signal_args);
			SIGNAL_CHECK_FALSE("text_set");

			// Out of range.
			ERR_PRINT_OFF;
			text_edit->set_line(-1, "test");
			MessageQueue::get_singleton()->flush();
			CHECK(text_edit->get_line(0) == "te");
			SIGNAL_CHECK_FALSE("lines_edited_from");
			SIGNAL_CHECK_FALSE("caret_changed");
			SIGNAL_CHECK_FALSE("text_changed");
			SIGNAL_CHECK_FALSE("text_set");

			text_edit->set_line(1, "test");
			MessageQueue::get_singleton()->flush();
			CHECK(text_edit->get_line(0) == "te");
			SIGNAL_CHECK_FALSE("lines_edited_from");
			SIGNAL_CHECK_FALSE("caret_changed");
			SIGNAL_CHECK_FALSE("text_changed");
			SIGNAL_CHECK_FALSE("text_set");

			ERR_PRINT_ON;
		}

		SUBCASE("[TextEdit] swap lines") {
			((Array)lines_edited_args[1])[1] = 1;

			text_edit->set_text("testing\nswap");
			MessageQueue::get_singleton()->flush();
			CHECK(text_edit->get_text() == "testing\nswap");
			SIGNAL_CHECK("text_set", empty_signal_args);
			SIGNAL_CHECK("lines_edited_from", lines_edited_args);
			SIGNAL_CHECK("caret_changed", empty_signal_args);
			SIGNAL_CHECK_FALSE("text_changed");

			text_edit->set_caret_column(text_edit->get_line(0).length());
			MessageQueue::get_singleton()->flush();
			SIGNAL_CHECK("caret_changed", empty_signal_args);

			((Array)lines_edited_args[1])[1] = 0;
			Array swap_args;
			swap_args.push_back(1);
			swap_args.push_back(1);
			lines_edited_args.push_back(swap_args);
			lines_edited_args.push_back(swap_args);

			// Order does not matter. Should also work if not editable.
			text_edit->set_editable(false);
			text_edit->swap_lines(1, 0);
			MessageQueue::get_singleton()->flush();
			CHECK(text_edit->get_text() == "swap\ntesting");
			SIGNAL_CHECK("lines_edited_from", lines_edited_args);
			SIGNAL_CHECK("caret_changed", empty_signal_args);
			SIGNAL_CHECK("text_changed", empty_signal_args);
			SIGNAL_CHECK_FALSE("text_set");
			text_edit->set_editable(true);

			lines_edited_args.reverse();

			// Single undo/redo action
			text_edit->undo();
			MessageQueue::get_singleton()->flush();
			CHECK(text_edit->get_text() == "testing\nswap");
			SIGNAL_CHECK("lines_edited_from", lines_edited_args);
			SIGNAL_CHECK("caret_changed", empty_signal_args);
			SIGNAL_CHECK("text_changed", empty_signal_args);
			SIGNAL_CHECK_FALSE("text_set");

			lines_edited_args.reverse();

			text_edit->redo();
			MessageQueue::get_singleton()->flush();
			CHECK(text_edit->get_text() == "swap\ntesting");
			SIGNAL_CHECK("lines_edited_from", lines_edited_args);
			SIGNAL_CHECK("caret_changed", empty_signal_args);
			SIGNAL_CHECK("text_changed", empty_signal_args);
			SIGNAL_CHECK_FALSE("text_set");

			// Out of range.
			ERR_PRINT_OFF;
			text_edit->swap_lines(-1, 0);
			CHECK(text_edit->get_text() == "swap\ntesting");
			SIGNAL_CHECK_FALSE("lines_edited_from");
			SIGNAL_CHECK_FALSE("caret_changed");
			SIGNAL_CHECK_FALSE("text_changed");
			SIGNAL_CHECK_FALSE("text_set");

			text_edit->swap_lines(0, -1);
			CHECK(text_edit->get_text() == "swap\ntesting");
			SIGNAL_CHECK_FALSE("lines_edited_from");
			SIGNAL_CHECK_FALSE("caret_changed");
			SIGNAL_CHECK_FALSE("text_changed");
			SIGNAL_CHECK_FALSE("text_set");

			text_edit->swap_lines(2, 0);
			CHECK(text_edit->get_text() == "swap\ntesting");
			SIGNAL_CHECK_FALSE("lines_edited_from");
			SIGNAL_CHECK_FALSE("caret_changed");
			SIGNAL_CHECK_FALSE("text_changed");
			SIGNAL_CHECK_FALSE("text_set");

			text_edit->swap_lines(0, 2);
			CHECK(text_edit->get_text() == "swap\ntesting");
			SIGNAL_CHECK_FALSE("lines_edited_from");
			SIGNAL_CHECK_FALSE("caret_changed");
			SIGNAL_CHECK_FALSE("text_changed");
			SIGNAL_CHECK_FALSE("text_set");
			ERR_PRINT_ON;
		}

		SUBCASE("[TextEdit] insert line at") {
			((Array)lines_edited_args[1])[1] = 1;

			text_edit->set_text("testing\nswap");
			MessageQueue::get_singleton()->flush();
			CHECK(text_edit->get_text() == "testing\nswap");
			SIGNAL_CHECK("text_set", empty_signal_args);
			SIGNAL_CHECK("lines_edited_from", lines_edited_args);
			SIGNAL_CHECK("caret_changed", empty_signal_args);
			SIGNAL_CHECK_FALSE("text_changed");

			text_edit->select_all();
			MessageQueue::get_singleton()->flush();
			CHECK(text_edit->has_selection());
			CHECK(text_edit->get_selection_from_line() == 0);
			CHECK(text_edit->get_selection_to_line() == 1);
			SIGNAL_CHECK("caret_changed", empty_signal_args);

			// Insert before should move caret and selection, and works when not editable.
			text_edit->set_editable(false);
			lines_edited_args.remove_at(0);
			text_edit->insert_line_at(0, "new");
			MessageQueue::get_singleton()->flush();
			CHECK(text_edit->get_text() == "new\ntesting\nswap");
			CHECK(text_edit->get_caret_line() == 2);
			CHECK(text_edit->get_caret_column() == text_edit->get_line(2).size() - 1);
			CHECK(text_edit->has_selection());
			CHECK(text_edit->get_selection_from_line() == 1);
			CHECK(text_edit->get_selection_to_line() == 2);
			SIGNAL_CHECK("lines_edited_from", lines_edited_args);
			SIGNAL_CHECK("caret_changed", empty_signal_args);
			SIGNAL_CHECK("text_changed", empty_signal_args);
			SIGNAL_CHECK_FALSE("text_set");
			text_edit->set_editable(true);

			// Can undo/redo as single action.
			((Array)lines_edited_args[0])[0] = 1;
			((Array)lines_edited_args[0])[1] = 0;
			text_edit->undo();
			MessageQueue::get_singleton()->flush();
			CHECK(text_edit->get_text() == "testing\nswap");
			CHECK(text_edit->has_selection());
			SIGNAL_CHECK("lines_edited_from", lines_edited_args);
			SIGNAL_CHECK("caret_changed", empty_signal_args);
			SIGNAL_CHECK("text_changed", empty_signal_args);
			SIGNAL_CHECK_FALSE("text_set");

			((Array)lines_edited_args[0])[0] = 0;
			((Array)lines_edited_args[0])[1] = 1;
			text_edit->redo();
			MessageQueue::get_singleton()->flush();
			CHECK(text_edit->get_text() == "new\ntesting\nswap");
			CHECK(text_edit->has_selection());
			SIGNAL_CHECK("lines_edited_from", lines_edited_args);
			SIGNAL_CHECK("caret_changed", empty_signal_args);
			SIGNAL_CHECK("text_changed", empty_signal_args);
			SIGNAL_CHECK_FALSE("text_set");

			// Adding inside selection extends selection.
			text_edit->select_all();
			MessageQueue::get_singleton()->flush();
			CHECK(text_edit->has_selection());
			CHECK(text_edit->get_selection_from_line() == 0);
			CHECK(text_edit->get_selection_to_line() == 2);
			SIGNAL_CHECK_FALSE("caret_changed");

			((Array)lines_edited_args[0])[0] = 2;
			((Array)lines_edited_args[0])[1] = 3;
			text_edit->insert_line_at(2, "after");
			MessageQueue::get_singleton()->flush();
			CHECK(text_edit->get_text() == "new\ntesting\nafter\nswap");
			CHECK(text_edit->get_caret_line() == 3);
			CHECK(text_edit->get_caret_column() == text_edit->get_line(3).size() - 1);
			CHECK(text_edit->has_selection());
			CHECK(text_edit->get_selection_from_line() == 0);
			CHECK(text_edit->get_selection_to_line() == 3);
			SIGNAL_CHECK("lines_edited_from", lines_edited_args);
			SIGNAL_CHECK("caret_changed", empty_signal_args);
			SIGNAL_CHECK("text_changed", empty_signal_args);
			SIGNAL_CHECK_FALSE("text_set");

			// Out of range.
			ERR_PRINT_OFF;
			text_edit->insert_line_at(-1, "after");
			CHECK(text_edit->get_text() == "new\ntesting\nafter\nswap");
			SIGNAL_CHECK_FALSE("lines_edited_from");
			SIGNAL_CHECK_FALSE("caret_changed");
			SIGNAL_CHECK_FALSE("text_changed");
			SIGNAL_CHECK_FALSE("text_set");

			text_edit->insert_line_at(4, "after");
			CHECK(text_edit->get_text() == "new\ntesting\nafter\nswap");
			SIGNAL_CHECK_FALSE("lines_edited_from");
			SIGNAL_CHECK_FALSE("caret_changed");
			SIGNAL_CHECK_FALSE("text_changed");
			SIGNAL_CHECK_FALSE("text_set");
			ERR_PRINT_ON;
		}

		SUBCASE("[TextEdit] insert line at caret") {
			lines_edited_args.pop_back();
			((Array)lines_edited_args[0])[1] = 1;

			text_edit->insert_text_at_caret("testing\nswap");
			MessageQueue::get_singleton()->flush();
			CHECK(text_edit->get_text() == "testing\nswap");
			CHECK(text_edit->get_caret_line() == 1);
			CHECK(text_edit->get_caret_column() == text_edit->get_line(1).size() - 1);
			SIGNAL_CHECK("lines_edited_from", lines_edited_args);
			SIGNAL_CHECK("text_changed", empty_signal_args);
			SIGNAL_CHECK("caret_changed", empty_signal_args);
			SIGNAL_CHECK_FALSE("text_set");

			text_edit->set_caret_line(0, false);
			text_edit->set_caret_column(2);
			SIGNAL_DISCARD("caret_changed");

			((Array)lines_edited_args[0])[1] = 0;
			text_edit->insert_text_at_caret("mid");
			MessageQueue::get_singleton()->flush();
			CHECK(text_edit->get_text() == "temidsting\nswap");
			CHECK(text_edit->get_caret_line() == 0);
			CHECK(text_edit->get_caret_column() == 5);
			SIGNAL_CHECK("lines_edited_from", lines_edited_args);
			SIGNAL_CHECK("text_changed", empty_signal_args);
			SIGNAL_CHECK("caret_changed", empty_signal_args);
			SIGNAL_CHECK_FALSE("text_set");

			text_edit->select(0, 0, 0, text_edit->get_line(0).length());
			CHECK(text_edit->has_selection());
			lines_edited_args.push_back(args1.duplicate());

			text_edit->set_editable(false);
			text_edit->insert_text_at_caret("new line");
			MessageQueue::get_singleton()->flush();
			CHECK(text_edit->get_text() == "new line\nswap");
			CHECK(text_edit->get_caret_line() == 0);
			CHECK(text_edit->get_caret_column() == text_edit->get_line(0).size() - 1);
			CHECK_FALSE(text_edit->has_selection());
			SIGNAL_CHECK("lines_edited_from", lines_edited_args);
			SIGNAL_CHECK("text_changed", empty_signal_args);
			SIGNAL_CHECK("caret_changed", empty_signal_args);
			SIGNAL_CHECK_FALSE("text_set");
			text_edit->set_editable(true);

			text_edit->undo();
			MessageQueue::get_singleton()->flush();
			CHECK(text_edit->get_text() == "temidsting\nswap");
			CHECK(text_edit->get_caret_line() == 0);
			CHECK(text_edit->get_caret_column() == 5);
			CHECK(text_edit->has_selection());
			SIGNAL_CHECK("lines_edited_from", lines_edited_args);
			SIGNAL_CHECK("text_changed", empty_signal_args);
			SIGNAL_CHECK("caret_changed", empty_signal_args);
			SIGNAL_CHECK_FALSE("text_set");

			text_edit->redo();
			MessageQueue::get_singleton()->flush();
			CHECK(text_edit->get_text() == "new line\nswap");
			CHECK(text_edit->get_caret_line() == 0);
			CHECK(text_edit->get_caret_column() == 8);
			CHECK_FALSE(text_edit->has_selection());
			SIGNAL_CHECK("lines_edited_from", lines_edited_args);
			SIGNAL_CHECK("text_changed", empty_signal_args);
			SIGNAL_CHECK("caret_changed", empty_signal_args);
			SIGNAL_CHECK_FALSE("text_set");
		}

		SIGNAL_UNWATCH(text_edit, "text_set");
		SIGNAL_UNWATCH(text_edit, "text_changed");
		SIGNAL_UNWATCH(text_edit, "lines_edited_from");
		SIGNAL_UNWATCH(text_edit, "caret_changed");
	}

	SUBCASE("[TextEdit] indent level") {
		CHECK(text_edit->get_indent_level(0) == 0);
		CHECK(text_edit->get_first_non_whitespace_column(0) == 0);

		text_edit->set_line(0, "a");
		CHECK(text_edit->get_indent_level(0) == 0);
		CHECK(text_edit->get_first_non_whitespace_column(0) == 0);

		text_edit->set_line(0, "\t");
		CHECK(text_edit->get_indent_level(0) == 4);
		CHECK(text_edit->get_first_non_whitespace_column(0) == 1);

		text_edit->set_tab_size(8);
		CHECK(text_edit->get_indent_level(0) == 8);

		text_edit->set_line(0, "\t a");
		CHECK(text_edit->get_first_non_whitespace_column(0) == 2);
		CHECK(text_edit->get_indent_level(0) == 9);
	}

	SUBCASE("[TextEdit] selection") {
		SIGNAL_WATCH(text_edit, "text_set");
		SIGNAL_WATCH(text_edit, "text_changed");
		SIGNAL_WATCH(text_edit, "lines_edited_from");
		SIGNAL_WATCH(text_edit, "caret_changed");

		Array args1;
		args1.push_back(0);
		args1.push_back(0);
		Array lines_edited_args;
		lines_edited_args.push_back(args1);
		lines_edited_args.push_back(args1.duplicate());

		SUBCASE("[TextEdit] select all") {
			text_edit->select_all();
			CHECK_FALSE(text_edit->has_selection());
			ERR_PRINT_OFF;
			CHECK(text_edit->get_selection_from_line() == -1);
			CHECK(text_edit->get_selection_from_column() == -1);
			CHECK(text_edit->get_selection_to_line() == -1);
			CHECK(text_edit->get_selection_to_column() == -1);
			CHECK(text_edit->get_selected_text() == "");
			ERR_PRINT_ON;

			text_edit->set_text("test\nselection");
			SEND_GUI_ACTION("ui_text_select_all");
			CHECK(text_edit->get_viewport()->is_input_handled());
			MessageQueue::get_singleton()->flush();
			CHECK(text_edit->get_selected_text() == "test\nselection");
			CHECK(text_edit->has_selection());
			CHECK(text_edit->get_selection_from_line() == 0);
			CHECK(text_edit->get_selection_from_column() == 0);
			CHECK(text_edit->get_selection_to_line() == 1);
			CHECK(text_edit->get_selection_to_column() == 9);
			CHECK(text_edit->get_selection_mode() == TextEdit::SelectionMode::SELECTION_MODE_SHIFT);
			CHECK(text_edit->get_caret_line() == 1);
			CHECK(text_edit->get_caret_column() == 9);
			SIGNAL_CHECK("caret_changed", empty_signal_args);

			text_edit->set_caret_line(0);
			text_edit->set_caret_column(0);
			text_edit->set_selecting_enabled(false);
			CHECK_FALSE(text_edit->has_selection());
			CHECK(text_edit->get_selected_text() == "");
			CHECK(text_edit->get_caret_line() == 0);
			CHECK(text_edit->get_caret_column() == 0);

			text_edit->select_all();
			CHECK_FALSE(text_edit->has_selection());
			CHECK(text_edit->get_selected_text() == "");
			CHECK(text_edit->get_caret_line() == 0);
			CHECK(text_edit->get_caret_column() == 0);
		}

		SUBCASE("[TextEdit] select word under caret") {
			text_edit->set_text("\ntest   test\ntest   test");

			text_edit->set_caret_column(0);
			text_edit->set_caret_line(1);

			text_edit->add_caret(2, 0);
			text_edit->add_caret(2, 2);
			CHECK(text_edit->get_caret_count() == 3);

			MessageQueue::get_singleton()->flush();

			SIGNAL_DISCARD("text_set");
			SIGNAL_DISCARD("text_changed");
			SIGNAL_DISCARD("lines_edited_from");
			SIGNAL_DISCARD("caret_changed");

			text_edit->select_word_under_caret();
			CHECK(text_edit->has_selection(0));
			CHECK(text_edit->get_selected_text(0) == "test");
			CHECK(text_edit->get_selection_from_line(0) == 1);
			CHECK(text_edit->get_selection_from_column(0) == 0);
			CHECK(text_edit->get_selection_to_line(0) == 1);
			CHECK(text_edit->get_selection_to_column(0) == 4);
			CHECK(text_edit->get_caret_line(0) == 1);
			CHECK(text_edit->get_caret_column(0) == 4);

			CHECK(text_edit->has_selection(1));
			CHECK(text_edit->get_selected_text(1) == "test");
			CHECK(text_edit->get_selection_from_line(1) == 2);
			CHECK(text_edit->get_selection_from_column(1) == 0);
			CHECK(text_edit->get_selection_to_line(1) == 2);
			CHECK(text_edit->get_selection_to_column(1) == 4);
			CHECK(text_edit->get_caret_line(1) == 2);
			CHECK(text_edit->get_caret_column(1) == 4);

			CHECK(text_edit->get_caret_count() == 2);

			text_edit->select_word_under_caret();
			CHECK_FALSE(text_edit->has_selection());
			CHECK(text_edit->get_selected_text() == "");

			SEND_GUI_ACTION("ui_text_select_word_under_caret");
			CHECK(text_edit->get_viewport()->is_input_handled());
			MessageQueue::get_singleton()->flush();
			CHECK(text_edit->has_selection(0));
			CHECK(text_edit->get_selected_text(0) == "test");
			CHECK(text_edit->get_selection_from_line(0) == 1);
			CHECK(text_edit->get_selection_from_column(0) == 0);
			CHECK(text_edit->get_selection_to_line(0) == 1);
			CHECK(text_edit->get_selection_to_column(0) == 4);
			CHECK(text_edit->get_caret_line(0) == 1);
			CHECK(text_edit->get_caret_column(0) == 4);

			CHECK(text_edit->has_selection(1));
			CHECK(text_edit->get_selected_text(1) == "test");
			CHECK(text_edit->get_selection_from_line(1) == 2);
			CHECK(text_edit->get_selection_from_column(1) == 0);
			CHECK(text_edit->get_selection_to_line(1) == 2);
			CHECK(text_edit->get_selection_to_column(1) == 4);
			CHECK(text_edit->get_caret_line(1) == 2);
			CHECK(text_edit->get_caret_column(1) == 4);

			CHECK(text_edit->get_selected_text() == "test\ntest");
			SIGNAL_CHECK("caret_changed", empty_signal_args);

			text_edit->set_selecting_enabled(false);
			text_edit->select_word_under_caret();
			CHECK_FALSE(text_edit->has_selection());
			CHECK(text_edit->get_selected_text() == "");
			CHECK(text_edit->get_caret_line(0) == 1);
			CHECK(text_edit->get_caret_column(0) == 4);
			CHECK(text_edit->get_caret_line(1) == 2);
			CHECK(text_edit->get_caret_column(1) == 4);
			SIGNAL_CHECK_FALSE("caret_changed");
			text_edit->set_selecting_enabled(true);

			text_edit->set_caret_line(1, false, true, 0, 0);
			text_edit->set_caret_column(5, false, 0);

			text_edit->set_caret_line(2, false, true, 0, 1);
			text_edit->set_caret_column(5, false, 1);

			text_edit->select_word_under_caret();
			CHECK_FALSE(text_edit->has_selection());
			CHECK(text_edit->get_selected_text() == "");

			text_edit->select_word_under_caret();
			CHECK_FALSE(text_edit->has_selection());
			CHECK(text_edit->get_selected_text() == "");
			CHECK(text_edit->get_caret_line(0) == 1);
			CHECK(text_edit->get_caret_column(0) == 5);
			CHECK(text_edit->get_caret_line(1) == 2);
			CHECK(text_edit->get_caret_column(1) == 5);
			SIGNAL_CHECK_FALSE("caret_changed");
		}

		SUBCASE("[TextEdit] add selection for next occurrence") {
			text_edit->set_text("\ntest   other_test\nrandom   test\nword test word nonrandom");
			text_edit->set_caret_column(0);
			text_edit->set_caret_line(1);

			// First selection made by the implicit select_word_under_caret call
			text_edit->add_selection_for_next_occurrence();
			CHECK(text_edit->get_caret_count() == 1);
			CHECK(text_edit->get_selected_text(0) == "test");
			CHECK(text_edit->get_selection_from_line(0) == 1);
			CHECK(text_edit->get_selection_from_column(0) == 0);
			CHECK(text_edit->get_selection_to_line(0) == 1);
			CHECK(text_edit->get_selection_to_column(0) == 4);
			CHECK(text_edit->get_caret_line(0) == 1);
			CHECK(text_edit->get_caret_column(0) == 4);

			text_edit->add_selection_for_next_occurrence();
			CHECK(text_edit->get_caret_count() == 2);
			CHECK(text_edit->get_selected_text(1) == "test");
			CHECK(text_edit->get_selection_from_line(1) == 1);
			CHECK(text_edit->get_selection_from_column(1) == 13);
			CHECK(text_edit->get_selection_to_line(1) == 1);
			CHECK(text_edit->get_selection_to_column(1) == 17);
			CHECK(text_edit->get_caret_line(1) == 1);
			CHECK(text_edit->get_caret_column(1) == 17);

			text_edit->add_selection_for_next_occurrence();
			CHECK(text_edit->get_caret_count() == 3);
			CHECK(text_edit->get_selected_text(2) == "test");
			CHECK(text_edit->get_selection_from_line(2) == 2);
			CHECK(text_edit->get_selection_from_column(2) == 9);
			CHECK(text_edit->get_selection_to_line(2) == 2);
			CHECK(text_edit->get_selection_to_column(2) == 13);
			CHECK(text_edit->get_caret_line(2) == 2);
			CHECK(text_edit->get_caret_column(2) == 13);

			text_edit->add_selection_for_next_occurrence();
			CHECK(text_edit->get_caret_count() == 4);
			CHECK(text_edit->get_selected_text(3) == "test");
			CHECK(text_edit->get_selection_from_line(3) == 3);
			CHECK(text_edit->get_selection_from_column(3) == 5);
			CHECK(text_edit->get_selection_to_line(3) == 3);
			CHECK(text_edit->get_selection_to_column(3) == 9);
			CHECK(text_edit->get_caret_line(3) == 3);
			CHECK(text_edit->get_caret_column(3) == 9);

			// A different word with a new manually added caret
			text_edit->add_caret(2, 1);
			text_edit->select(2, 0, 2, 4, 4);
			CHECK(text_edit->get_selected_text(4) == "rand");

			text_edit->add_selection_for_next_occurrence();
			CHECK(text_edit->get_caret_count() == 6);
			CHECK(text_edit->get_selected_text(5) == "rand");
			CHECK(text_edit->get_selection_from_line(5) == 3);
			CHECK(text_edit->get_selection_from_column(5) == 18);
			CHECK(text_edit->get_selection_to_line(5) == 3);
			CHECK(text_edit->get_selection_to_column(5) == 22);
			CHECK(text_edit->get_caret_line(5) == 3);
			CHECK(text_edit->get_caret_column(5) == 22);

			// Make sure the previous selections are still active
			CHECK(text_edit->get_selected_text(0) == "test");
			CHECK(text_edit->get_selected_text(1) == "test");
			CHECK(text_edit->get_selected_text(2) == "test");
			CHECK(text_edit->get_selected_text(3) == "test");
		}

		SUBCASE("[TextEdit] deselect on focus loss") {
			text_edit->set_text("test");

			text_edit->set_deselect_on_focus_loss_enabled(true);
			CHECK(text_edit->is_deselect_on_focus_loss_enabled());

			text_edit->grab_focus();
			text_edit->select_all();
			CHECK(text_edit->has_focus());
			CHECK(text_edit->has_selection());

			text_edit->release_focus();
			CHECK_FALSE(text_edit->has_focus());
			CHECK_FALSE(text_edit->has_selection());

			text_edit->set_deselect_on_focus_loss_enabled(false);
			CHECK_FALSE(text_edit->is_deselect_on_focus_loss_enabled());

			text_edit->grab_focus();
			text_edit->select_all();
			CHECK(text_edit->has_focus());
			CHECK(text_edit->has_selection());

			text_edit->release_focus();
			CHECK_FALSE(text_edit->has_focus());
			CHECK(text_edit->has_selection());

			text_edit->set_deselect_on_focus_loss_enabled(true);
			CHECK_FALSE(text_edit->has_selection());
		}

		SUBCASE("[TextEdit] key select") {
			text_edit->set_text("test");

			text_edit->grab_focus();
			SEND_GUI_KEY_EVENT(Key::RIGHT | KeyModifierMask::SHIFT)
			CHECK(text_edit->has_selection());
			CHECK(text_edit->get_selected_text() == "t");

#ifdef MACOS_ENABLED
			SEND_GUI_KEY_EVENT(Key::RIGHT | KeyModifierMask::SHIFT | KeyModifierMask::ALT)
#else
			SEND_GUI_KEY_EVENT(Key::RIGHT | KeyModifierMask::SHIFT | KeyModifierMask::CMD_OR_CTRL)
#endif
			CHECK(text_edit->has_selection());
			CHECK(text_edit->get_selected_text() == "test");

			SEND_GUI_KEY_EVENT(Key::LEFT | KeyModifierMask::SHIFT)
			CHECK(text_edit->has_selection());
			CHECK(text_edit->get_selected_text() == "tes");

#ifdef MACOS_ENABLED
			SEND_GUI_KEY_EVENT(Key::LEFT | KeyModifierMask::SHIFT | KeyModifierMask::ALT)
#else
			SEND_GUI_KEY_EVENT(Key::LEFT | KeyModifierMask::SHIFT | KeyModifierMask::CMD_OR_CTRL)
#endif
			CHECK_FALSE(text_edit->has_selection());
			CHECK(text_edit->get_selected_text() == "");

			SEND_GUI_KEY_EVENT(Key::RIGHT | KeyModifierMask::SHIFT)
			CHECK(text_edit->has_selection());
			CHECK(text_edit->get_selected_text() == "t");

			SEND_GUI_KEY_EVENT(Key::RIGHT)
			CHECK_FALSE(text_edit->has_selection());
			CHECK(text_edit->get_selected_text() == "");

			SEND_GUI_KEY_EVENT(Key::LEFT | KeyModifierMask::SHIFT)
			CHECK(text_edit->has_selection());
			CHECK(text_edit->get_selected_text() == "t");

			SEND_GUI_KEY_EVENT(Key::LEFT)
			CHECK_FALSE(text_edit->has_selection());
			CHECK(text_edit->get_selected_text() == "");

			text_edit->set_selecting_enabled(false);
			SEND_GUI_KEY_EVENT(Key::RIGHT | KeyModifierMask::SHIFT)
			CHECK_FALSE(text_edit->has_selection());
			CHECK(text_edit->get_selected_text() == "");
			text_edit->set_selecting_enabled(true);
		}

		SUBCASE("[TextEdit] mouse drag select") {
			/* Set size for mouse input. */
			text_edit->set_size(Size2(200, 200));

			text_edit->set_text("this is some text\nfor selection");
			text_edit->grab_focus();
			MessageQueue::get_singleton()->flush();

			SEND_GUI_MOUSE_BUTTON_EVENT(text_edit->get_pos_at_line_column(0, 1), MouseButton::LEFT, MouseButtonMask::LEFT, Key::NONE);
			SEND_GUI_MOUSE_MOTION_EVENT(text_edit->get_pos_at_line_column(0, 7), MouseButtonMask::LEFT, Key::NONE);
			CHECK(text_edit->has_selection());
			CHECK(text_edit->get_selected_text() == "for s");
			CHECK(text_edit->get_selection_mode() == TextEdit::SELECTION_MODE_POINTER);
			CHECK(text_edit->get_selection_from_line() == 1);
			CHECK(text_edit->get_selection_from_column() == 0);
			CHECK(text_edit->get_selection_to_line() == 1);
			CHECK(text_edit->get_selection_to_column() == 5);
			CHECK(text_edit->get_caret_line() == 1);
			CHECK(text_edit->get_caret_column() == 5);

			SEND_GUI_MOUSE_BUTTON_EVENT(text_edit->get_pos_at_line_column(0, 9), MouseButton::LEFT, MouseButtonMask::LEFT, Key::NONE);
			CHECK_FALSE(text_edit->has_selection());

			text_edit->set_selecting_enabled(false);
			SEND_GUI_MOUSE_BUTTON_EVENT(text_edit->get_pos_at_line_column(0, 1), MouseButton::LEFT, MouseButtonMask::LEFT, Key::NONE);
			SEND_GUI_MOUSE_MOTION_EVENT(text_edit->get_pos_at_line_column(0, 7), MouseButtonMask::LEFT, Key::NONE);
			CHECK_FALSE(text_edit->has_selection());
			CHECK(text_edit->get_caret_line() == 1);
			CHECK(text_edit->get_caret_column() == 5);
			text_edit->set_selecting_enabled(true);
		}

		SUBCASE("[TextEdit] mouse word select") {
			/* Set size for mouse input. */
			text_edit->set_size(Size2(200, 200));

			text_edit->set_text("this is some text\nfor selection");
			MessageQueue::get_singleton()->flush();
			SIGNAL_DISCARD("caret_changed");

			SEND_GUI_DOUBLE_CLICK(text_edit->get_pos_at_line_column(0, 2), Key::NONE);
			CHECK(text_edit->has_selection());
			CHECK(text_edit->get_selected_text() == "for");
			CHECK(text_edit->get_selection_mode() == TextEdit::SELECTION_MODE_WORD);
			CHECK(text_edit->get_selection_from_line() == 1);
			CHECK(text_edit->get_selection_from_column() == 0);
			CHECK(text_edit->get_selection_to_line() == 1);
			CHECK(text_edit->get_selection_to_column() == 3);
			CHECK(text_edit->get_caret_line() == 1);
			CHECK(text_edit->get_caret_column() == 3);
			SIGNAL_CHECK("caret_changed", empty_signal_args);

			SEND_GUI_MOUSE_MOTION_EVENT(text_edit->get_pos_at_line_column(0, 7), MouseButtonMask::LEFT, Key::NONE);
			CHECK(text_edit->has_selection());
			CHECK(text_edit->get_selected_text() == "for selection");
			CHECK(text_edit->get_selection_mode() == TextEdit::SELECTION_MODE_WORD);
			CHECK(text_edit->get_selection_from_line() == 1);
			CHECK(text_edit->get_selection_from_column() == 0);
			CHECK(text_edit->get_selection_to_line() == 1);
			CHECK(text_edit->get_selection_to_column() == 13);
			CHECK(text_edit->get_caret_line() == 1);
			CHECK(text_edit->get_caret_column() == 13);
			SIGNAL_CHECK("caret_changed", empty_signal_args);

			Point2i line_0 = text_edit->get_pos_at_line_column(0, 0);
			line_0.y /= 2;
			SEND_GUI_MOUSE_BUTTON_EVENT(line_0, MouseButton::LEFT, MouseButtonMask::LEFT, Key::NONE);
			CHECK_FALSE(text_edit->has_selection());

			text_edit->set_selecting_enabled(false);
			SEND_GUI_DOUBLE_CLICK(text_edit->get_pos_at_line_column(0, 2), Key::NONE);
			CHECK_FALSE(text_edit->has_selection());
			CHECK(text_edit->get_caret_line() == 1);
			CHECK(text_edit->get_caret_column() == 3);
			text_edit->set_selecting_enabled(true);
		}

		SUBCASE("[TextEdit] mouse line select") {
			/* Set size for mouse input. */
			text_edit->set_size(Size2(200, 200));

			text_edit->set_text("this is some text\nfor selection");
			MessageQueue::get_singleton()->flush();

			SEND_GUI_DOUBLE_CLICK(text_edit->get_pos_at_line_column(0, 2), Key::NONE);
			SEND_GUI_MOUSE_BUTTON_EVENT(text_edit->get_pos_at_line_column(0, 2), MouseButton::LEFT, MouseButtonMask::LEFT, Key::NONE);
			CHECK(text_edit->has_selection());
			CHECK(text_edit->get_selected_text() == "for selection");
			CHECK(text_edit->get_selection_mode() == TextEdit::SELECTION_MODE_LINE);
			CHECK(text_edit->get_selection_from_line() == 1);
			CHECK(text_edit->get_selection_from_column() == 0);
			CHECK(text_edit->get_selection_to_line() == 1);
			CHECK(text_edit->get_selection_to_column() == 13);
			CHECK(text_edit->get_caret_line() == 1);
			CHECK(text_edit->get_caret_column() == 0);

			Point2i line_0 = text_edit->get_pos_at_line_column(0, 0);
			line_0.y /= 2;
			SEND_GUI_MOUSE_BUTTON_EVENT(line_0, MouseButton::LEFT, MouseButtonMask::LEFT, Key::NONE);
			CHECK_FALSE(text_edit->has_selection());

			text_edit->set_selecting_enabled(false);
			SEND_GUI_DOUBLE_CLICK(text_edit->get_pos_at_line_column(0, 2), Key::NONE);
			SEND_GUI_MOUSE_BUTTON_EVENT(text_edit->get_pos_at_line_column(0, 2), MouseButton::LEFT, MouseButtonMask::LEFT, Key::NONE);
			CHECK_FALSE(text_edit->has_selection());
			CHECK(text_edit->get_caret_line() == 1);
			CHECK(text_edit->get_caret_column() == 0);
			text_edit->set_selecting_enabled(true);
		}

		SUBCASE("[TextEdit] mouse shift click select") {
			/* Set size for mouse input. */
			text_edit->set_size(Size2(200, 200));

			text_edit->set_text("this is some text\nfor selection");
			MessageQueue::get_singleton()->flush();

			SEND_GUI_MOUSE_BUTTON_EVENT(text_edit->get_pos_at_line_column(0, 0), MouseButton::LEFT, MouseButtonMask::LEFT, Key::NONE);
			SEND_GUI_MOUSE_BUTTON_EVENT(text_edit->get_pos_at_line_column(0, 7), MouseButton::LEFT, MouseButtonMask::LEFT, Key::NONE | KeyModifierMask::SHIFT);
			CHECK(text_edit->has_selection());
			CHECK(text_edit->get_selected_text() == "for s");
			CHECK(text_edit->get_selection_mode() == TextEdit::SELECTION_MODE_POINTER);
			CHECK(text_edit->get_selection_from_line() == 1);
			CHECK(text_edit->get_selection_from_column() == 0);
			CHECK(text_edit->get_selection_to_line() == 1);
			CHECK(text_edit->get_selection_to_column() == 5);
			CHECK(text_edit->get_caret_line() == 1);
			CHECK(text_edit->get_caret_column() == 5);

			SEND_GUI_MOUSE_BUTTON_EVENT(text_edit->get_pos_at_line_column(0, 9), MouseButton::LEFT, MouseButtonMask::LEFT, Key::NONE);
			CHECK_FALSE(text_edit->has_selection());

			text_edit->set_selecting_enabled(false);
			SEND_GUI_MOUSE_BUTTON_EVENT(text_edit->get_pos_at_line_column(0, 0), MouseButton::LEFT, MouseButtonMask::LEFT, Key::NONE);
			SEND_GUI_MOUSE_BUTTON_EVENT(text_edit->get_pos_at_line_column(0, 7), MouseButton::LEFT, MouseButtonMask::LEFT, Key::NONE | KeyModifierMask::SHIFT);
			CHECK_FALSE(text_edit->has_selection());
			CHECK(text_edit->get_caret_line() == 1);
			CHECK(text_edit->get_caret_column() == 5);
			text_edit->set_selecting_enabled(true);
		}

		SUBCASE("[TextEdit] select and deselect") {
			text_edit->set_text("this is some text\nfor selection");
			MessageQueue::get_singleton()->flush();

			text_edit->select(-1, -1, 500, 500);
			CHECK(text_edit->has_selection());
			CHECK(text_edit->get_selected_text() == "this is some text\nfor selection");

			text_edit->deselect();
			CHECK_FALSE(text_edit->has_selection());

			text_edit->select(500, 500, -1, -1);
			CHECK(text_edit->has_selection());
			CHECK(text_edit->get_selected_text() == "this is some text\nfor selection");

			text_edit->deselect();
			CHECK_FALSE(text_edit->has_selection());

			text_edit->select(0, 4, 0, 8);
			CHECK(text_edit->has_selection());
			CHECK(text_edit->get_selected_text() == " is ");

			text_edit->deselect();
			CHECK_FALSE(text_edit->has_selection());

			text_edit->select(0, 8, 0, 4);
			CHECK(text_edit->has_selection());
			CHECK(text_edit->get_selected_text() == " is ");

			text_edit->set_selecting_enabled(false);
			CHECK_FALSE(text_edit->has_selection());
			text_edit->select(0, 8, 0, 4);
			CHECK_FALSE(text_edit->has_selection());
			text_edit->set_selecting_enabled(true);

			text_edit->select(0, 8, 0, 4);
			CHECK(text_edit->has_selection());
			SEND_GUI_ACTION("ui_text_caret_right");
			CHECK_FALSE(text_edit->has_selection());

			text_edit->delete_selection();
			CHECK(text_edit->get_text() == "this is some text\nfor selection");
			CHECK(text_edit->get_caret_line() == 0);
			CHECK(text_edit->get_caret_column() == 8);

			text_edit->select(0, 8, 0, 4);
			CHECK(text_edit->has_selection());
			SEND_GUI_ACTION("ui_text_backspace");
			CHECK(text_edit->get_text() == "thissome text\nfor selection");
			CHECK(text_edit->get_caret_line() == 0);
			CHECK(text_edit->get_caret_column() == 4);

			text_edit->undo();
			CHECK(text_edit->has_selection());
			CHECK(text_edit->get_text() == "this is some text\nfor selection");
			CHECK(text_edit->get_caret_line() == 0);
			CHECK(text_edit->get_caret_column() == 8);

			text_edit->redo();
			CHECK_FALSE(text_edit->has_selection());
			CHECK(text_edit->get_text() == "thissome text\nfor selection");
			CHECK(text_edit->get_caret_line() == 0);
			CHECK(text_edit->get_caret_column() == 4);

			text_edit->undo();
			CHECK(text_edit->has_selection());
			CHECK(text_edit->get_text() == "this is some text\nfor selection");
			CHECK(text_edit->get_caret_line() == 0);
			CHECK(text_edit->get_caret_column() == 8);

			text_edit->select(0, 8, 0, 4);
			CHECK(text_edit->has_selection());

			text_edit->delete_selection();
			CHECK_FALSE(text_edit->has_selection());
			CHECK(text_edit->get_text() == "thissome text\nfor selection");

			text_edit->undo();
			CHECK(text_edit->has_selection());
			CHECK(text_edit->get_text() == "this is some text\nfor selection");
			CHECK(text_edit->get_caret_line() == 0);
			CHECK(text_edit->get_caret_column() == 8);

			text_edit->redo();
			CHECK_FALSE(text_edit->has_selection());
			CHECK(text_edit->get_text() == "thissome text\nfor selection");
			CHECK(text_edit->get_caret_line() == 0);
			CHECK(text_edit->get_caret_column() == 4);

			text_edit->undo();
			CHECK(text_edit->has_selection());
			CHECK(text_edit->get_text() == "this is some text\nfor selection");
			CHECK(text_edit->get_caret_line() == 0);
			CHECK(text_edit->get_caret_column() == 8);

			text_edit->set_editable(false);
			text_edit->delete_selection();
			text_edit->set_editable(false);
			CHECK_FALSE(text_edit->has_selection());
			CHECK(text_edit->get_text() == "thissome text\nfor selection");

			text_edit->undo();
			CHECK_FALSE(text_edit->has_selection());
			CHECK(text_edit->get_text() == "thissome text\nfor selection");
		}

		// Add readonly test?
		SUBCASE("[TextEdit] text drag") {
			TextEdit *target_text_edit = memnew(TextEdit);
			SceneTree::get_singleton()->get_root()->add_child(target_text_edit);

			target_text_edit->set_size(Size2(200, 200));
			target_text_edit->set_position(Point2(400, 0));

			text_edit->set_size(Size2(200, 200));

			CHECK_FALSE(text_edit->is_mouse_over_selection());
			text_edit->set_text("drag me");
			text_edit->select_all();
			text_edit->grab_click_focus();
			MessageQueue::get_singleton()->flush();

			Point2i line_0 = text_edit->get_pos_at_line_column(0, 0);
			line_0.y /= 2;
			SEND_GUI_MOUSE_BUTTON_EVENT(line_0, MouseButton::LEFT, MouseButtonMask::LEFT, Key::NONE);
			CHECK(text_edit->is_mouse_over_selection());
			SEND_GUI_MOUSE_MOTION_EVENT(text_edit->get_pos_at_line_column(0, 7), MouseButtonMask::LEFT, Key::NONE);
			CHECK(text_edit->get_viewport()->gui_is_dragging());
			CHECK(text_edit->get_viewport()->gui_get_drag_data() == "drag me");

			line_0 = target_text_edit->get_pos_at_line_column(0, 0);
			line_0.y /= 2;
			line_0.x += 401; // As empty add one.
			SEND_GUI_MOUSE_MOTION_EVENT(line_0, MouseButtonMask::LEFT, Key::NONE);
			CHECK(text_edit->get_viewport()->gui_is_dragging());

			SEND_GUI_MOUSE_BUTTON_RELEASED_EVENT(line_0, MouseButton::LEFT, MouseButtonMask::NONE, Key::NONE);

			CHECK_FALSE(text_edit->get_viewport()->gui_is_dragging());
			CHECK(text_edit->get_text() == "");
			CHECK(target_text_edit->get_text() == "drag me");

			memdelete(target_text_edit);
		}

		SIGNAL_UNWATCH(text_edit, "text_set");
		SIGNAL_UNWATCH(text_edit, "text_changed");
		SIGNAL_UNWATCH(text_edit, "lines_edited_from");
		SIGNAL_UNWATCH(text_edit, "caret_changed");
	}

	SUBCASE("[TextEdit] overridable actions") {
		SIGNAL_WATCH(text_edit, "text_set");
		SIGNAL_WATCH(text_edit, "text_changed");
		SIGNAL_WATCH(text_edit, "lines_edited_from");
		SIGNAL_WATCH(text_edit, "caret_changed");

		Array args1;
		args1.push_back(0);
		args1.push_back(0);
		Array lines_edited_args;
		lines_edited_args.push_back(args1);

		SUBCASE("[TextEdit] backspace") {
			text_edit->set_text("this is\nsome\n");
			text_edit->set_caret_line(0);
			text_edit->set_caret_column(0);
			MessageQueue::get_singleton()->flush();

			SIGNAL_DISCARD("text_set");
			SIGNAL_DISCARD("text_changed");
			SIGNAL_DISCARD("lines_edited_from");
			SIGNAL_DISCARD("caret_changed");

			text_edit->backspace();
			MessageQueue::get_singleton()->flush();
			SIGNAL_CHECK_FALSE("text_changed");
			SIGNAL_CHECK_FALSE("caret_changed");
			SIGNAL_CHECK_FALSE("lines_edited_from");

			text_edit->set_caret_line(2);
			text_edit->set_caret_column(0);
			MessageQueue::get_singleton()->flush();
			SIGNAL_DISCARD("caret_changed");

			((Array)lines_edited_args[0])[0] = 2;
			((Array)lines_edited_args[0])[1] = 1;
			text_edit->backspace();
			MessageQueue::get_singleton()->flush();

			CHECK(text_edit->get_text() == "this is\nsome");
			CHECK(text_edit->get_caret_line() == 1);
			CHECK(text_edit->get_caret_column() == 4);
			SIGNAL_CHECK("caret_changed", empty_signal_args);
			SIGNAL_CHECK("text_changed", empty_signal_args);
			SIGNAL_CHECK("lines_edited_from", lines_edited_args);

			((Array)lines_edited_args[0])[0] = 1;
			text_edit->backspace();
			MessageQueue::get_singleton()->flush();

			CHECK(text_edit->get_text() == "this is\nsom");
			CHECK(text_edit->get_caret_line() == 1);
			CHECK(text_edit->get_caret_column() == 3);
			SIGNAL_CHECK("caret_changed", empty_signal_args);
			SIGNAL_CHECK("text_changed", empty_signal_args);
			SIGNAL_CHECK("lines_edited_from", lines_edited_args);

			text_edit->end_complex_operation();
			text_edit->select(1, 0, 1, 3);
			text_edit->backspace();
			MessageQueue::get_singleton()->flush();

			CHECK(text_edit->get_text() == "this is\n");
			CHECK(text_edit->get_caret_line() == 1);
			CHECK(text_edit->get_caret_column() == 0);
			SIGNAL_CHECK("caret_changed", empty_signal_args);
			SIGNAL_CHECK("text_changed", empty_signal_args);
			SIGNAL_CHECK("lines_edited_from", lines_edited_args);

			text_edit->set_editable(false);
			text_edit->backspace();
			text_edit->set_editable(true);
			MessageQueue::get_singleton()->flush();

			CHECK(text_edit->get_text() == "this is\n");
			CHECK(text_edit->get_caret_line() == 1);
			CHECK(text_edit->get_caret_column() == 0);
			SIGNAL_CHECK_FALSE("text_changed");
			SIGNAL_CHECK_FALSE("caret_changed");
			SIGNAL_CHECK_FALSE("lines_edited_from");

			text_edit->undo();
			MessageQueue::get_singleton()->flush();
			CHECK(text_edit->get_text() == "this is\nsom");
			CHECK(text_edit->get_caret_line() == 1);
			CHECK(text_edit->get_caret_column() == 3);
			SIGNAL_CHECK("caret_changed", empty_signal_args);
			SIGNAL_CHECK("text_changed", empty_signal_args);
			SIGNAL_CHECK("lines_edited_from", lines_edited_args);
		}

		SUBCASE("[TextEdit] cut") {
			text_edit->set_text("this is\nsome\n");
			text_edit->set_caret_line(0);
			text_edit->set_caret_column(6);
			MessageQueue::get_singleton()->flush();

			SIGNAL_DISCARD("text_set");
			SIGNAL_DISCARD("text_changed");
			SIGNAL_DISCARD("lines_edited_from");
			SIGNAL_DISCARD("caret_changed");

			ERR_PRINT_OFF;
			text_edit->cut();
			MessageQueue::get_singleton()->flush();
			ERR_PRINT_ON; // Can't check display server content.

			((Array)lines_edited_args[0])[0] = 1;
			CHECK(text_edit->get_text() == "some\n");
			CHECK(text_edit->get_caret_line() == 0);
			CHECK(text_edit->get_caret_column() == 4);
			SIGNAL_CHECK("caret_changed", empty_signal_args);
			SIGNAL_CHECK("text_changed", empty_signal_args);
			SIGNAL_CHECK("lines_edited_from", lines_edited_args);

			((Array)lines_edited_args[0])[0] = 0;
			((Array)lines_edited_args[0])[1] = 1;
			text_edit->undo();
			MessageQueue::get_singleton()->flush();
			CHECK(text_edit->get_text() == "this is\nsome\n");
			CHECK(text_edit->get_caret_line() == 0);
			CHECK(text_edit->get_caret_column() == 6);
			SIGNAL_CHECK("caret_changed", empty_signal_args);
			SIGNAL_CHECK("text_changed", empty_signal_args);
			SIGNAL_CHECK("lines_edited_from", lines_edited_args);

			((Array)lines_edited_args[0])[0] = 1;
			((Array)lines_edited_args[0])[1] = 0;
			text_edit->redo();
			MessageQueue::get_singleton()->flush();
			CHECK(text_edit->get_text() == "some\n");
			CHECK(text_edit->get_caret_line() == 0);
			CHECK(text_edit->get_caret_column() == 4);
			SIGNAL_CHECK("caret_changed", empty_signal_args);
			SIGNAL_CHECK("text_changed", empty_signal_args);
			SIGNAL_CHECK("lines_edited_from", lines_edited_args);

			text_edit->set_text("this is\nsome\n");
			MessageQueue::get_singleton()->flush();

			SIGNAL_DISCARD("text_set");
			SIGNAL_DISCARD("text_changed");
			SIGNAL_DISCARD("lines_edited_from");
			SIGNAL_DISCARD("caret_changed");

			((Array)lines_edited_args[0])[0] = 0;
			text_edit->select(0, 5, 0, 7);
			ERR_PRINT_OFF;
			SEND_GUI_ACTION("ui_cut");
			CHECK(text_edit->get_viewport()->is_input_handled());
			MessageQueue::get_singleton()->flush();
			ERR_PRINT_ON; // Can't check display server content.
			CHECK(text_edit->get_text() == "this \nsome\n");
			CHECK(text_edit->get_caret_line() == 0);
			CHECK(text_edit->get_caret_column() == 5);
			SIGNAL_CHECK("caret_changed", empty_signal_args);
			SIGNAL_CHECK("text_changed", empty_signal_args);
			SIGNAL_CHECK("lines_edited_from", lines_edited_args);

			text_edit->set_editable(false);
			text_edit->cut();
			MessageQueue::get_singleton()->flush();
			text_edit->set_editable(true);
			CHECK(text_edit->get_text() == "this \nsome\n");
			CHECK(text_edit->get_caret_line() == 0);
			CHECK(text_edit->get_caret_column() == 5);
			SIGNAL_CHECK_FALSE("caret_changed");
			SIGNAL_CHECK_FALSE("text_changed");
			SIGNAL_CHECK_FALSE("lines_edited_from");
		}

		SUBCASE("[TextEdit] copy") {
			// TODO: Cannot test need display server support.
		}

		SUBCASE("[TextEdit] paste") {
			// TODO: Cannot test need display server support.
		}

		SUBCASE("[TextEdit] paste primary") {
			// TODO: Cannot test need display server support.
		}

		SIGNAL_UNWATCH(text_edit, "text_set");
		SIGNAL_UNWATCH(text_edit, "text_changed");
		SIGNAL_UNWATCH(text_edit, "lines_edited_from");
		SIGNAL_UNWATCH(text_edit, "caret_changed");
	}

	// Add undo / redo tests?
	SUBCASE("[TextEdit] input") {
		SIGNAL_WATCH(text_edit, "text_set");
		SIGNAL_WATCH(text_edit, "text_changed");
		SIGNAL_WATCH(text_edit, "lines_edited_from");
		SIGNAL_WATCH(text_edit, "caret_changed");

		Array args1;
		args1.push_back(0);
		args1.push_back(0);
		Array lines_edited_args;
		lines_edited_args.push_back(args1);

		SUBCASE("[TextEdit] ui_text_newline_above") {
			text_edit->set_text("this is some test text.\nthis is some test text.");
			text_edit->select(0, 0, 0, 4);
			text_edit->set_caret_column(4);

			text_edit->add_caret(1, 4);
			text_edit->select(1, 0, 1, 4, 1);
			CHECK(text_edit->get_caret_count() == 2);

			MessageQueue::get_singleton()->flush();

			SIGNAL_DISCARD("text_set");
			SIGNAL_DISCARD("text_changed");
			SIGNAL_DISCARD("lines_edited_from");
			SIGNAL_DISCARD("caret_changed");

			// For the second caret.
			Array args2;
			args2.push_back(0);
			args2.push_back(1);
			lines_edited_args.push_front(args2);

			((Array)lines_edited_args[1])[1] = 1;
			SEND_GUI_ACTION("ui_text_newline_above");
			CHECK(text_edit->get_viewport()->is_input_handled());
			CHECK(text_edit->get_text() == "\nthis is some test text.\n\nthis is some test text.");
			CHECK(text_edit->get_caret_line() == 0);
			CHECK(text_edit->get_caret_column() == 0);
			CHECK_FALSE(text_edit->has_selection(0));

			CHECK(text_edit->get_caret_line(1) == 2);
			CHECK(text_edit->get_caret_column(1) == 0);
			CHECK_FALSE(text_edit->has_selection(1));
			SIGNAL_CHECK("caret_changed", empty_signal_args);
			SIGNAL_CHECK("text_changed", empty_signal_args);
			SIGNAL_CHECK("lines_edited_from", lines_edited_args);

			text_edit->set_caret_line(1);
			text_edit->set_caret_column(4);

			text_edit->set_caret_line(3, false, true, 0, 1);
			text_edit->set_caret_column(4, false, 1);
			MessageQueue::get_singleton()->flush();
			SIGNAL_DISCARD("caret_changed");

			text_edit->set_editable(false);
			SEND_GUI_ACTION("ui_text_newline_above");
			CHECK(text_edit->get_viewport()->is_input_handled());
			CHECK(text_edit->get_text() == "\nthis is some test text.\n\nthis is some test text.");
			CHECK(text_edit->get_caret_line() == 1);
			CHECK(text_edit->get_caret_column() == 4);
			CHECK_FALSE(text_edit->has_selection(0));

			CHECK(text_edit->get_caret_line(1) == 3);
			CHECK(text_edit->get_caret_column(1) == 4);
			CHECK_FALSE(text_edit->has_selection(1));
			SIGNAL_CHECK_FALSE("caret_changed");
			SIGNAL_CHECK_FALSE("text_changed");
			SIGNAL_CHECK_FALSE("lines_edited_from");
			text_edit->set_editable(true);

			((Array)lines_edited_args[0])[0] = 2;
			((Array)lines_edited_args[0])[1] = 3;

			SEND_GUI_ACTION("ui_text_newline_above");
			CHECK(text_edit->get_viewport()->is_input_handled());
			CHECK(text_edit->get_text() == "\n\nthis is some test text.\n\n\nthis is some test text.");
			CHECK(text_edit->get_caret_line() == 1);
			CHECK(text_edit->get_caret_column() == 0);
			CHECK_FALSE(text_edit->has_selection(0));

			CHECK(text_edit->get_caret_line(1) == 4);
			CHECK(text_edit->get_caret_column(1) == 0);
			CHECK_FALSE(text_edit->has_selection(1));
			SIGNAL_CHECK("caret_changed", empty_signal_args);
			SIGNAL_CHECK("text_changed", empty_signal_args);
			SIGNAL_CHECK("lines_edited_from", lines_edited_args);
		}

		SUBCASE("[TextEdit] ui_text_newline_blank") {
			text_edit->set_text("this is some test text.\nthis is some test text.");
			text_edit->select(0, 0, 0, 4);
			text_edit->set_caret_column(4);

			text_edit->add_caret(1, 4);
			text_edit->select(1, 0, 1, 4, 1);
			CHECK(text_edit->get_caret_count() == 2);

			MessageQueue::get_singleton()->flush();

			SIGNAL_DISCARD("text_set");
			SIGNAL_DISCARD("text_changed");
			SIGNAL_DISCARD("lines_edited_from");
			SIGNAL_DISCARD("caret_changed");

			// For the second caret.
			Array args2;
			args2.push_back(1);
			args2.push_back(2);
			lines_edited_args.push_front(args2);

			((Array)lines_edited_args[1])[1] = 1;
			SEND_GUI_ACTION("ui_text_newline_blank");
			CHECK(text_edit->get_viewport()->is_input_handled());
			CHECK(text_edit->get_text() == "this is some test text.\n\nthis is some test text.\n");
			CHECK(text_edit->get_caret_line() == 1);
			CHECK(text_edit->get_caret_column() == 0);
			CHECK_FALSE(text_edit->has_selection(0));

			CHECK(text_edit->get_caret_line(1) == 3);
			CHECK(text_edit->get_caret_column(1) == 0);
			CHECK_FALSE(text_edit->has_selection(1));
			SIGNAL_CHECK("caret_changed", empty_signal_args);
			SIGNAL_CHECK("text_changed", empty_signal_args);
			SIGNAL_CHECK("lines_edited_from", lines_edited_args);

			text_edit->set_editable(false);
			SEND_GUI_ACTION("ui_text_newline_blank");
			CHECK(text_edit->get_viewport()->is_input_handled());
			CHECK(text_edit->get_text() == "this is some test text.\n\nthis is some test text.\n");
			CHECK(text_edit->get_caret_line() == 1);
			CHECK(text_edit->get_caret_column() == 0);
			CHECK_FALSE(text_edit->has_selection(0));

			CHECK(text_edit->get_caret_line(1) == 3);
			CHECK(text_edit->get_caret_column(1) == 0);
			CHECK_FALSE(text_edit->has_selection(1));
			SIGNAL_CHECK_FALSE("caret_changed");
			SIGNAL_CHECK_FALSE("text_changed");
			SIGNAL_CHECK_FALSE("lines_edited_from");
			text_edit->set_editable(true);
		}

		SUBCASE("[TextEdit] ui_text_newline") {
			text_edit->set_text("this is some test text.\nthis is some test text.");
			text_edit->select(0, 0, 0, 4);
			text_edit->set_caret_column(4);

			text_edit->add_caret(1, 4);
			text_edit->select(1, 0, 1, 4, 1);
			CHECK(text_edit->get_caret_count() == 2);

			MessageQueue::get_singleton()->flush();

			SIGNAL_DISCARD("text_set");
			SIGNAL_DISCARD("text_changed");
			SIGNAL_DISCARD("lines_edited_from");
			SIGNAL_DISCARD("caret_changed");

			// For the second caret.
			Array args2;
			args2.push_back(1);
			args2.push_back(1);
			lines_edited_args.push_front(args2);
			lines_edited_args.push_front(args2.duplicate());
			((Array)lines_edited_args[1])[1] = 2;

			lines_edited_args.push_back(lines_edited_args[2].duplicate());
			((Array)lines_edited_args[3])[1] = 1;

			SEND_GUI_ACTION("ui_text_newline");
			CHECK(text_edit->get_viewport()->is_input_handled());
			CHECK(text_edit->get_text() == "\n is some test text.\n\n is some test text.");
			CHECK(text_edit->get_caret_line() == 1);
			CHECK(text_edit->get_caret_column() == 0);
			CHECK_FALSE(text_edit->has_selection(0));

			CHECK(text_edit->get_caret_line(1) == 3);
			CHECK(text_edit->get_caret_column(1) == 0);
			CHECK_FALSE(text_edit->has_selection(1));
			SIGNAL_CHECK("caret_changed", empty_signal_args);
			SIGNAL_CHECK("text_changed", empty_signal_args);
			SIGNAL_CHECK("lines_edited_from", lines_edited_args);

			text_edit->set_editable(false);
			SEND_GUI_ACTION("ui_text_newline");
			CHECK(text_edit->get_viewport()->is_input_handled());
			CHECK(text_edit->get_text() == "\n is some test text.\n\n is some test text.");
			CHECK(text_edit->get_caret_line() == 1);
			CHECK(text_edit->get_caret_column() == 0);
			CHECK_FALSE(text_edit->has_selection(0));

			CHECK(text_edit->get_caret_line(1) == 3);
			CHECK(text_edit->get_caret_column(1) == 0);
			CHECK_FALSE(text_edit->has_selection(1));
			SIGNAL_CHECK_FALSE("caret_changed");
			SIGNAL_CHECK_FALSE("text_changed");
			SIGNAL_CHECK_FALSE("lines_edited_from");
			text_edit->set_editable(true);
		}

		SUBCASE("[TextEdit] ui_text_backspace_all_to_left") {
			text_edit->set_text("\nthis is some test text.\n\nthis is some test text.");
			text_edit->select(1, 0, 1, 4);
			text_edit->set_caret_line(1);
			text_edit->set_caret_column(4);

			text_edit->add_caret(3, 4);
			text_edit->select(3, 0, 3, 4, 1);
			CHECK(text_edit->get_caret_count() == 2);

			MessageQueue::get_singleton()->flush();

			Ref<InputEvent> tmpevent = InputEventKey::create_reference(Key::BACKSPACE | KeyModifierMask::ALT | KeyModifierMask::CMD_OR_CTRL);
			InputMap::get_singleton()->action_add_event("ui_text_backspace_all_to_left", tmpevent);

			SIGNAL_DISCARD("text_set");
			SIGNAL_DISCARD("text_changed");
			SIGNAL_DISCARD("lines_edited_from");
			SIGNAL_DISCARD("caret_changed");

			// For the second caret.
			Array args2;
			args2.push_back(3);
			args2.push_back(3);
			lines_edited_args.push_front(args2);

			// With selection should be a normal backspace.
			((Array)lines_edited_args[1])[0] = 1;
			((Array)lines_edited_args[1])[1] = 1;

			SEND_GUI_ACTION("ui_text_backspace_all_to_left");
			CHECK(text_edit->get_viewport()->is_input_handled());
			CHECK(text_edit->get_text() == "\n is some test text.\n\n is some test text.");
			CHECK(text_edit->get_caret_line() == 1);
			CHECK(text_edit->get_caret_column() == 0);
			CHECK_FALSE(text_edit->has_selection(0));

			CHECK(text_edit->get_caret_line(1) == 3);
			CHECK(text_edit->get_caret_column(1) == 0);
			CHECK_FALSE(text_edit->has_selection(1));
			SIGNAL_CHECK("caret_changed", empty_signal_args);
			SIGNAL_CHECK("text_changed", empty_signal_args);
			SIGNAL_CHECK("lines_edited_from", lines_edited_args);

			((Array)lines_edited_args[0])[1] = 2;
			((Array)lines_edited_args[1])[1] = 0;

			// Start of line should also be a normal backspace.
			SEND_GUI_ACTION("ui_text_backspace_all_to_left");
			CHECK(text_edit->get_viewport()->is_input_handled());
			CHECK(text_edit->get_text() == " is some test text.\n is some test text.");
			CHECK(text_edit->get_caret_line() == 0);
			CHECK(text_edit->get_caret_column() == 0);
			CHECK_FALSE(text_edit->has_selection(0));

			CHECK(text_edit->get_caret_line(1) == 1);
			CHECK(text_edit->get_caret_column(1) == 0);
			CHECK_FALSE(text_edit->has_selection(1));
			SIGNAL_CHECK("caret_changed", empty_signal_args);
			SIGNAL_CHECK("text_changed", empty_signal_args);
			SIGNAL_CHECK("lines_edited_from", lines_edited_args);

			text_edit->set_caret_column(text_edit->get_line(0).length());
			text_edit->set_caret_column(text_edit->get_line(1).length(), false, 1);
			MessageQueue::get_singleton()->flush();

			SIGNAL_DISCARD("text_set");
			SIGNAL_DISCARD("text_changed");
			SIGNAL_DISCARD("lines_edited_from");
			SIGNAL_DISCARD("caret_changed");

			text_edit->set_editable(false);
			SEND_GUI_ACTION("ui_text_backspace_all_to_left");
			CHECK(text_edit->get_viewport()->is_input_handled());
			CHECK(text_edit->get_text() == " is some test text.\n is some test text.");
			CHECK(text_edit->get_caret_line() == 0);
			CHECK(text_edit->get_caret_column() == text_edit->get_line(0).length());
			CHECK_FALSE(text_edit->has_selection(0));

			CHECK(text_edit->get_caret_line(1) == 1);
			CHECK(text_edit->get_caret_column(1) == text_edit->get_line(1).length());
			CHECK_FALSE(text_edit->has_selection(1));
			SIGNAL_CHECK_FALSE("caret_changed");
			SIGNAL_CHECK_FALSE("text_changed");
			SIGNAL_CHECK_FALSE("lines_edited_from");
			text_edit->set_editable(true);

			((Array)lines_edited_args[0])[0] = 1;
			((Array)lines_edited_args[0])[1] = 1;
			((Array)lines_edited_args[1])[0] = 0;

			SEND_GUI_ACTION("ui_text_backspace_all_to_left");
			CHECK(text_edit->get_viewport()->is_input_handled());
			CHECK(text_edit->get_text() == "\n");
			CHECK(text_edit->get_caret_line() == 0);
			CHECK(text_edit->get_caret_column() == 0);
			CHECK_FALSE(text_edit->has_selection(0));

			CHECK(text_edit->get_caret_line(1) == 1);
			CHECK(text_edit->get_caret_column(1) == 0);
			CHECK_FALSE(text_edit->has_selection(1));
			SIGNAL_CHECK("caret_changed", empty_signal_args);
			SIGNAL_CHECK("text_changed", empty_signal_args);
			SIGNAL_CHECK("lines_edited_from", lines_edited_args);

			InputMap::get_singleton()->action_erase_event("ui_text_backspace_all_to_left", tmpevent);
		}

		SUBCASE("[TextEdit] ui_text_backspace_word") {
			text_edit->set_text("\nthis is some test text.\n\nthis is some test text.");
			text_edit->select(1, 0, 1, 4);
			text_edit->set_caret_line(1);
			text_edit->set_caret_column(4);

			text_edit->add_caret(3, 4);
			text_edit->select(3, 0, 3, 4, 1);
			CHECK(text_edit->get_caret_count() == 2);
			MessageQueue::get_singleton()->flush();

			SIGNAL_DISCARD("text_set");
			SIGNAL_DISCARD("text_changed");
			SIGNAL_DISCARD("lines_edited_from");
			SIGNAL_DISCARD("caret_changed");

			// For the second caret.
			Array args2;
			args2.push_back(3);
			args2.push_back(3);
			lines_edited_args.push_front(args2);

			// With selection should be a normal backspace.
			((Array)lines_edited_args[1])[0] = 1;
			((Array)lines_edited_args[1])[1] = 1;

			SEND_GUI_ACTION("ui_text_backspace_word");
			CHECK(text_edit->get_viewport()->is_input_handled());
			CHECK(text_edit->get_text() == "\n is some test text.\n\n is some test text.");
			CHECK(text_edit->get_caret_line() == 1);
			CHECK(text_edit->get_caret_column() == 0);
			CHECK_FALSE(text_edit->has_selection(0));

			CHECK(text_edit->get_caret_line(1) == 3);
			CHECK(text_edit->get_caret_column(1) == 0);
			CHECK_FALSE(text_edit->has_selection(1));
			SIGNAL_CHECK("caret_changed", empty_signal_args);
			SIGNAL_CHECK("text_changed", empty_signal_args);
			SIGNAL_CHECK("lines_edited_from", lines_edited_args);
			text_edit->end_complex_operation();

			((Array)lines_edited_args[0])[1] = 2;
			((Array)lines_edited_args[1])[1] = 0;

			// Start of line should also be a normal backspace.
			SEND_GUI_ACTION("ui_text_backspace_word");
			CHECK(text_edit->get_viewport()->is_input_handled());
			CHECK(text_edit->get_text() == " is some test text.\n is some test text.");
			CHECK(text_edit->get_caret_line() == 0);
			CHECK(text_edit->get_caret_column() == 0);
			CHECK_FALSE(text_edit->has_selection(0));

			CHECK(text_edit->get_caret_line(1) == 1);
			CHECK(text_edit->get_caret_column(1) == 0);
			CHECK_FALSE(text_edit->has_selection(1));
			SIGNAL_CHECK("caret_changed", empty_signal_args);
			SIGNAL_CHECK("text_changed", empty_signal_args);
			SIGNAL_CHECK("lines_edited_from", lines_edited_args);

			text_edit->set_editable(false);
			SEND_GUI_ACTION("ui_text_backspace_word");
			CHECK(text_edit->get_viewport()->is_input_handled());
			CHECK(text_edit->get_text() == " is some test text.\n is some test text.");
			CHECK(text_edit->get_caret_line() == 0);
			CHECK(text_edit->get_caret_column() == 0);
			CHECK_FALSE(text_edit->has_selection(0));

			CHECK(text_edit->get_caret_line(1) == 1);
			CHECK(text_edit->get_caret_column(1) == 0);
			CHECK_FALSE(text_edit->has_selection(1));
			SIGNAL_CHECK_FALSE("caret_changed");
			SIGNAL_CHECK_FALSE("text_changed");
			SIGNAL_CHECK_FALSE("lines_edited_from");
			text_edit->set_editable(true);

			text_edit->set_caret_column(text_edit->get_line(0).length());
			text_edit->set_caret_column(text_edit->get_line(1).length(), false, 1);
			MessageQueue::get_singleton()->flush();

			SIGNAL_DISCARD("text_set");
			SIGNAL_DISCARD("text_changed");
			SIGNAL_DISCARD("lines_edited_from");
			SIGNAL_DISCARD("caret_changed");

			((Array)lines_edited_args[0])[0] = 1;
			((Array)lines_edited_args[0])[1] = 1;
			((Array)lines_edited_args[1])[0] = 0;

			SEND_GUI_ACTION("ui_text_backspace_word");
			CHECK(text_edit->get_viewport()->is_input_handled());
			CHECK(text_edit->get_text() == " is some test \n is some test ");
			CHECK(text_edit->get_caret_line() == 0);
			CHECK(text_edit->get_caret_column() == 14);
			CHECK_FALSE(text_edit->has_selection(0));

			CHECK(text_edit->get_caret_line(1) == 1);
			CHECK(text_edit->get_caret_column(1) == 14);
			CHECK_FALSE(text_edit->has_selection(1));
			SIGNAL_CHECK("caret_changed", empty_signal_args);
			SIGNAL_CHECK("text_changed", empty_signal_args);
			SIGNAL_CHECK("lines_edited_from", lines_edited_args);
		}

		SUBCASE("[TextEdit] ui_text_backspace_word same line") {
			text_edit->set_text("test test test");
			text_edit->set_caret_column(4);
			text_edit->add_caret(0, 9);
			text_edit->add_caret(0, 15);

			// For the second caret.
			Array args2;
			args2.push_back(0);
			lines_edited_args.push_front(args2);

			// For the third caret.
			Array args3;
			args2.push_back(0);
			lines_edited_args.push_front(args2);

			CHECK(text_edit->get_caret_count() == 3);
			MessageQueue::get_singleton()->flush();

			SIGNAL_DISCARD("text_set");
			SIGNAL_DISCARD("text_changed");
			SIGNAL_DISCARD("lines_edited_from");
			SIGNAL_DISCARD("caret_changed");

			SEND_GUI_ACTION("ui_text_backspace_word");
			CHECK(text_edit->get_viewport()->is_input_handled());
			CHECK(text_edit->get_text() == "  ");
			CHECK(text_edit->get_caret_line() == 0);
			CHECK(text_edit->get_caret_column() == 0);
			CHECK_FALSE(text_edit->has_selection(0));

			CHECK(text_edit->get_caret_line(1) == 0);
			CHECK(text_edit->get_caret_column(1) == 1);
			CHECK_FALSE(text_edit->has_selection(1));

			CHECK(text_edit->get_caret_line(2) == 0);
			CHECK(text_edit->get_caret_column(2) == 2);
			CHECK_FALSE(text_edit->has_selection(1));

			SIGNAL_CHECK("caret_changed", empty_signal_args);
			SIGNAL_CHECK("text_changed", empty_signal_args);
			SIGNAL_CHECK("lines_edited_from", lines_edited_args);
		}

		SUBCASE("[TextEdit] ui_text_backspace") {
			text_edit->set_text("\nthis is some test text.\n\nthis is some test text.");
			text_edit->select(1, 0, 1, 4);
			text_edit->set_caret_line(1);
			text_edit->set_caret_column(4);

			text_edit->add_caret(3, 4);
			text_edit->select(3, 0, 3, 4, 1);
			CHECK(text_edit->get_caret_count() == 2);

			MessageQueue::get_singleton()->flush();

			SIGNAL_DISCARD("text_set");
			SIGNAL_DISCARD("text_changed");
			SIGNAL_DISCARD("lines_edited_from");
			SIGNAL_DISCARD("caret_changed");

			// For the second caret.
			Array args2;
			args2.push_back(3);
			args2.push_back(3);
			lines_edited_args.push_front(args2);

			// With selection should be a normal backspace.
			((Array)lines_edited_args[1])[0] = 1;
			((Array)lines_edited_args[1])[1] = 1;

			SEND_GUI_ACTION("ui_text_backspace");
			CHECK(text_edit->get_viewport()->is_input_handled());
			CHECK(text_edit->get_text() == "\n is some test text.\n\n is some test text.");
			CHECK(text_edit->get_caret_line() == 1);
			CHECK(text_edit->get_caret_column() == 0);
			CHECK_FALSE(text_edit->has_selection(0));

			CHECK(text_edit->get_caret_line(1) == 3);
			CHECK(text_edit->get_caret_column(1) == 0);
			CHECK_FALSE(text_edit->has_selection(1));
			SIGNAL_CHECK("caret_changed", empty_signal_args);
			SIGNAL_CHECK("text_changed", empty_signal_args);
			SIGNAL_CHECK("lines_edited_from", lines_edited_args);

			((Array)lines_edited_args[0])[1] = 2;
			((Array)lines_edited_args[1])[1] = 0;

			// Start of line should also be a normal backspace.
			SEND_GUI_ACTION("ui_text_backspace");
			CHECK(text_edit->get_viewport()->is_input_handled());
			CHECK(text_edit->get_text() == " is some test text.\n is some test text.");
			CHECK(text_edit->get_caret_line() == 0);
			CHECK(text_edit->get_caret_column() == 0);
			CHECK_FALSE(text_edit->has_selection(0));

			CHECK(text_edit->get_caret_line(1) == 1);
			CHECK(text_edit->get_caret_column(1) == 0);
			CHECK_FALSE(text_edit->has_selection(1));
			SIGNAL_CHECK("caret_changed", empty_signal_args);
			SIGNAL_CHECK("text_changed", empty_signal_args);
			SIGNAL_CHECK("lines_edited_from", lines_edited_args);

			text_edit->set_caret_column(text_edit->get_line(0).length());
			text_edit->set_caret_column(text_edit->get_line(1).length(), false, 1);
			MessageQueue::get_singleton()->flush();

			SIGNAL_DISCARD("text_set");
			SIGNAL_DISCARD("text_changed");
			SIGNAL_DISCARD("lines_edited_from");
			SIGNAL_DISCARD("caret_changed");

			text_edit->set_editable(false);
			SEND_GUI_ACTION("ui_text_backspace");
			CHECK(text_edit->get_viewport()->is_input_handled());
			CHECK(text_edit->get_text() == " is some test text.\n is some test text.");
			CHECK(text_edit->get_caret_line() == 0);
			CHECK(text_edit->get_caret_column() == text_edit->get_line(0).length());
			CHECK_FALSE(text_edit->has_selection(0));

			CHECK(text_edit->get_caret_line(1) == 1);
			CHECK(text_edit->get_caret_column(1) == text_edit->get_line(1).length());
			CHECK_FALSE(text_edit->has_selection(1));
			SIGNAL_CHECK_FALSE("caret_changed");
			SIGNAL_CHECK_FALSE("text_changed");
			SIGNAL_CHECK_FALSE("lines_edited_from");
			text_edit->set_editable(true);

			((Array)lines_edited_args[0])[0] = 1;
			((Array)lines_edited_args[0])[1] = 1;
			((Array)lines_edited_args[1])[0] = 0;

			SEND_GUI_ACTION("ui_text_backspace");
			CHECK(text_edit->get_viewport()->is_input_handled());
			CHECK(text_edit->get_text() == " is some test text\n is some test text");
			CHECK(text_edit->get_caret_line() == 0);
			CHECK(text_edit->get_caret_column() == 18);
			CHECK_FALSE(text_edit->has_selection(0));

			CHECK(text_edit->get_caret_line(1) == 1);
			CHECK(text_edit->get_caret_column(1) == 18);
			CHECK_FALSE(text_edit->has_selection(1));
			SIGNAL_CHECK("caret_changed", empty_signal_args);
			SIGNAL_CHECK("text_changed", empty_signal_args);
			SIGNAL_CHECK("lines_edited_from", lines_edited_args);

			// Select the entire text, from right to left
			text_edit->select(0, 18, 0, 0);
			text_edit->set_caret_line(0);
			text_edit->set_caret_column(0);

			text_edit->select(1, 18, 1, 0, 1);
			text_edit->set_caret_line(1, false, true, 0, 1);
			text_edit->set_caret_column(0, false, 1);
			MessageQueue::get_singleton()->flush();

			SIGNAL_DISCARD("text_set");
			SIGNAL_DISCARD("text_changed");
			SIGNAL_DISCARD("lines_edited_from");
			SIGNAL_DISCARD("caret_changed");

			SEND_GUI_ACTION("ui_text_backspace");
			CHECK(text_edit->get_text() == "\n");
			CHECK(text_edit->get_caret_line() == 0);
			CHECK(text_edit->get_caret_column() == 0);
			CHECK(text_edit->get_caret_line(1) == 1);
			CHECK(text_edit->get_caret_column(1) == 0);
			SIGNAL_CHECK_FALSE("caret_changed");
			SIGNAL_CHECK("text_changed", empty_signal_args);
			SIGNAL_CHECK("lines_edited_from", lines_edited_args);
		}

		SUBCASE("[TextEdit] ui_text_delete_all_to_right") {
			Ref<InputEvent> tmpevent = InputEventKey::create_reference(Key::BACKSPACE | KeyModifierMask::ALT | KeyModifierMask::CMD_OR_CTRL);
			InputMap::get_singleton()->action_add_event("ui_text_delete_all_to_right", tmpevent);

			text_edit->set_text("this is some test text.\nthis is some test text.\n");
			text_edit->select(0, 0, 0, 4);
			text_edit->set_caret_line(0);
			text_edit->set_caret_column(4);

			text_edit->add_caret(1, 4);
			text_edit->select(1, 0, 1, 4, 1);
			CHECK(text_edit->get_caret_count() == 2);

			MessageQueue::get_singleton()->flush();

			SIGNAL_DISCARD("text_set");
			SIGNAL_DISCARD("text_changed");
			SIGNAL_DISCARD("lines_edited_from");
			SIGNAL_DISCARD("caret_changed");

			// For the second caret.
			Array args2;
			args2.push_back(1);
			args2.push_back(1);
			lines_edited_args.push_front(args2);

			// With selection should be a normal delete.
			SEND_GUI_ACTION("ui_text_delete_all_to_right");
			CHECK(text_edit->get_viewport()->is_input_handled());
			CHECK(text_edit->get_text() == " is some test text.\n is some test text.\n");
			CHECK(text_edit->get_caret_line() == 0);
			CHECK(text_edit->get_caret_column() == 0);
			CHECK_FALSE(text_edit->has_selection(0));

			CHECK(text_edit->get_caret_line(1) == 1);
			CHECK(text_edit->get_caret_column(1) == 0);
			CHECK_FALSE(text_edit->has_selection(1));
			SIGNAL_CHECK("caret_changed", empty_signal_args);
			SIGNAL_CHECK("text_changed", empty_signal_args);
			SIGNAL_CHECK("lines_edited_from", lines_edited_args);

			// End of line should not do anything.
			text_edit->set_caret_column(text_edit->get_line(0).length());
			text_edit->set_caret_column(text_edit->get_line(1).length(), false, 1);
			MessageQueue::get_singleton()->flush();

			SIGNAL_DISCARD("text_set");
			SIGNAL_DISCARD("text_changed");
			SIGNAL_DISCARD("lines_edited_from");
			SIGNAL_DISCARD("caret_changed");

			SEND_GUI_ACTION("ui_text_delete_all_to_right");
			CHECK(text_edit->get_viewport()->is_input_handled());
			CHECK(text_edit->get_text() == " is some test text.\n is some test text.\n");
			CHECK(text_edit->get_caret_line() == 0);
			CHECK(text_edit->get_caret_column() == text_edit->get_line(0).length());
			CHECK_FALSE(text_edit->has_selection(0));

			CHECK(text_edit->get_caret_line(1) == 1);
			CHECK(text_edit->get_caret_column(1) == text_edit->get_line(1).length());
			CHECK_FALSE(text_edit->has_selection(1));
			SIGNAL_CHECK_FALSE("caret_changed");
			SIGNAL_CHECK_FALSE("text_changed");
			SIGNAL_CHECK_FALSE("lines_edited_from");

			text_edit->set_caret_column(0);
			text_edit->set_caret_column(0, false, 1);
			MessageQueue::get_singleton()->flush();

			SIGNAL_DISCARD("text_set");
			SIGNAL_DISCARD("text_changed");
			SIGNAL_DISCARD("lines_edited_from");
			SIGNAL_DISCARD("caret_changed");

			text_edit->set_editable(false);
			SEND_GUI_ACTION("ui_text_delete_all_to_right");
			CHECK(text_edit->get_viewport()->is_input_handled());
			CHECK(text_edit->get_text() == " is some test text.\n is some test text.\n");
			CHECK(text_edit->get_caret_line() == 0);
			CHECK(text_edit->get_caret_column() == 0);
			CHECK_FALSE(text_edit->has_selection(0));

			CHECK(text_edit->get_caret_line(1) == 1);
			CHECK(text_edit->get_caret_column(1) == 0);
			CHECK_FALSE(text_edit->has_selection(1));
			SIGNAL_CHECK_FALSE("caret_changed");
			SIGNAL_CHECK_FALSE("text_changed");
			SIGNAL_CHECK_FALSE("lines_edited_from");
			text_edit->set_editable(true);

			SEND_GUI_ACTION("ui_text_delete_all_to_right");
			CHECK(text_edit->get_viewport()->is_input_handled());
			CHECK(text_edit->get_text() == "\n\n");
			CHECK(text_edit->get_caret_line() == 0);
			CHECK(text_edit->get_caret_column() == 0);
			CHECK_FALSE(text_edit->has_selection(0));

			CHECK(text_edit->get_caret_line(1) == 1);
			CHECK(text_edit->get_caret_column(1) == 0);
			CHECK_FALSE(text_edit->has_selection(1));
			SIGNAL_CHECK_FALSE("caret_changed");
			SIGNAL_CHECK("text_changed", empty_signal_args);
			SIGNAL_CHECK("lines_edited_from", lines_edited_args);

			InputMap::get_singleton()->action_erase_event("ui_text_delete_all_to_right", tmpevent);
		}

		SUBCASE("[TextEdit] ui_text_delete_word") {
			text_edit->set_caret_mid_grapheme_enabled(true);
			CHECK(text_edit->is_caret_mid_grapheme_enabled());

			text_edit->set_text("this ffi some test text.\n\nthis ffi some test text.\n");
			text_edit->select(0, 0, 0, 4);
			text_edit->set_caret_line(0);
			text_edit->set_caret_column(4);

			text_edit->add_caret(2, 4);
			text_edit->select(2, 0, 2, 4, 1);
			CHECK(text_edit->get_caret_count() == 2);

			MessageQueue::get_singleton()->flush();

			SIGNAL_DISCARD("text_set");
			SIGNAL_DISCARD("text_changed");
			SIGNAL_DISCARD("lines_edited_from");
			SIGNAL_DISCARD("caret_changed");

			// For the second caret.
			Array args2;
			args2.push_back(2);
			args2.push_back(2);
			lines_edited_args.push_front(args2);

			// With selection should be a normal delete.
			SEND_GUI_ACTION("ui_text_delete_word");
			CHECK(text_edit->get_viewport()->is_input_handled());
			CHECK(text_edit->get_text() == " ffi some test text.\n\n ffi some test text.\n");
			CHECK(text_edit->get_caret_line() == 0);
			CHECK(text_edit->get_caret_column() == 0);
			CHECK_FALSE(text_edit->has_selection(0));

			CHECK(text_edit->get_caret_line(1) == 2);
			CHECK(text_edit->get_caret_column(1) == 0);
			CHECK_FALSE(text_edit->has_selection(1));
			SIGNAL_CHECK("caret_changed", empty_signal_args);
			SIGNAL_CHECK("text_changed", empty_signal_args);
			SIGNAL_CHECK("lines_edited_from", lines_edited_args);

			// With selection should be a normal delete.
			((Array)lines_edited_args[0])[0] = 3;
			((Array)lines_edited_args[1])[0] = 1;
			text_edit->set_caret_column(text_edit->get_line(0).length());
			text_edit->set_caret_column(text_edit->get_line(2).length(), false, 1);
			MessageQueue::get_singleton()->flush();

			SIGNAL_DISCARD("text_set");
			SIGNAL_DISCARD("text_changed");
			SIGNAL_DISCARD("lines_edited_from");
			SIGNAL_DISCARD("caret_changed");

			SEND_GUI_ACTION("ui_text_delete_word");
			CHECK(text_edit->get_viewport()->is_input_handled());
			CHECK(text_edit->get_text() == " ffi some test text.\n ffi some test text.");
			CHECK(text_edit->get_caret_line() == 0);
			CHECK(text_edit->get_caret_column() == text_edit->get_line(0).length());
			CHECK_FALSE(text_edit->has_selection());

			CHECK(text_edit->get_caret_line(1) == 1);
			CHECK(text_edit->get_caret_column(1) == text_edit->get_line(1).length());
			CHECK_FALSE(text_edit->has_selection(0));
			SIGNAL_CHECK("caret_changed", empty_signal_args);
			SIGNAL_CHECK("text_changed", empty_signal_args);
			SIGNAL_CHECK("lines_edited_from", lines_edited_args);

			((Array)lines_edited_args[1])[0] = 0;
			((Array)lines_edited_args[0])[0] = 1;
			((Array)lines_edited_args[0])[1] = 1;
			text_edit->set_caret_column(0);
			text_edit->set_caret_column(0, false, 1);
			MessageQueue::get_singleton()->flush();

			SIGNAL_DISCARD("text_set");
			SIGNAL_DISCARD("text_changed");
			SIGNAL_DISCARD("lines_edited_from");
			SIGNAL_DISCARD("caret_changed");

			text_edit->set_editable(false);
			SEND_GUI_ACTION("ui_text_delete_word");
			CHECK(text_edit->get_viewport()->is_input_handled());
			CHECK(text_edit->get_text() == " ffi some test text.\n ffi some test text.");
			CHECK(text_edit->get_caret_line() == 0);
			CHECK(text_edit->get_caret_column() == 0);
			CHECK_FALSE(text_edit->has_selection(0));

			CHECK(text_edit->get_caret_line(1) == 1);
			CHECK(text_edit->get_caret_column(1) == 0);
			CHECK_FALSE(text_edit->has_selection(1));
			SIGNAL_CHECK_FALSE("caret_changed");
			SIGNAL_CHECK_FALSE("text_changed");
			SIGNAL_CHECK_FALSE("lines_edited_from");
			text_edit->set_editable(true);

			SEND_GUI_ACTION("ui_text_delete_word");
			CHECK(text_edit->get_viewport()->is_input_handled());
			CHECK(text_edit->get_text() == " some test text.\n some test text.");
			CHECK(text_edit->get_caret_line() == 0);
			CHECK(text_edit->get_caret_column() == 0);
			CHECK_FALSE(text_edit->has_selection(0));

			CHECK(text_edit->get_caret_line(1) == 1);
			CHECK(text_edit->get_caret_column(1) == 0);
			CHECK_FALSE(text_edit->has_selection(1));
			SIGNAL_CHECK_FALSE("caret_changed");
			SIGNAL_CHECK("text_changed", empty_signal_args);
			SIGNAL_CHECK("lines_edited_from", lines_edited_args);
		}

		SUBCASE("[TextEdit] ui_text_delete") {
			text_edit->set_caret_mid_grapheme_enabled(true);
			CHECK(text_edit->is_caret_mid_grapheme_enabled());

			text_edit->set_text("this ffi some test text.\nthis ffi some test text.");
			text_edit->select(0, 0, 0, 4);
			text_edit->set_caret_line(0);
			text_edit->set_caret_column(4);

			text_edit->add_caret(1, 4);
			text_edit->select(1, 0, 1, 4, 1);
			CHECK(text_edit->get_caret_count() == 2);

			MessageQueue::get_singleton()->flush();

			SIGNAL_DISCARD("text_set");
			SIGNAL_DISCARD("text_changed");
			SIGNAL_DISCARD("lines_edited_from");
			SIGNAL_DISCARD("caret_changed");

			// For the second caret.
			Array args2;
			args2.push_back(1);
			args2.push_back(1);
			lines_edited_args.push_front(args2);

			// With selection should be a normal delete.
			SEND_GUI_ACTION("ui_text_delete");
			CHECK(text_edit->get_viewport()->is_input_handled());
			CHECK(text_edit->get_text() == " ffi some test text.\n ffi some test text.");
			CHECK(text_edit->get_caret_line() == 0);
			CHECK(text_edit->get_caret_column() == 0);
			CHECK_FALSE(text_edit->has_selection(0));

			CHECK(text_edit->get_caret_line(1) == 1);
			CHECK(text_edit->get_caret_column(1) == 0);
			CHECK_FALSE(text_edit->has_selection(1));
			SIGNAL_CHECK("caret_changed", empty_signal_args);
			SIGNAL_CHECK("text_changed", empty_signal_args);
			SIGNAL_CHECK("lines_edited_from", lines_edited_args);

			// With selection should be a normal delete.
			lines_edited_args.remove_at(0);
			((Array)lines_edited_args[0])[0] = 1;
			text_edit->set_caret_column(text_edit->get_line(1).length(), false, 1);
			text_edit->set_caret_column(text_edit->get_line(0).length());
			MessageQueue::get_singleton()->flush();

			SIGNAL_DISCARD("text_set");
			SIGNAL_DISCARD("text_changed");
			SIGNAL_DISCARD("lines_edited_from");
			SIGNAL_DISCARD("caret_changed");

			SEND_GUI_ACTION("ui_text_delete");
			CHECK(text_edit->get_viewport()->is_input_handled());
			CHECK(text_edit->get_text() == " ffi some test text. ffi some test text.");
			CHECK(text_edit->get_caret_line() == 0);
			CHECK(text_edit->get_caret_column() == 20);
			CHECK_FALSE(text_edit->has_selection(0));
			SIGNAL_CHECK("caret_changed", empty_signal_args);
			SIGNAL_CHECK("text_changed", empty_signal_args);
			SIGNAL_CHECK("lines_edited_from", lines_edited_args);

			// Caret should be removed due to column preservation.
			CHECK(text_edit->get_caret_count() == 1);

			// Lets add it back.
			text_edit->set_caret_column(0);
			text_edit->add_caret(0, 20);

			((Array)lines_edited_args[0])[0] = 0;
			lines_edited_args.push_back(args2);
			((Array)lines_edited_args[1])[0] = 0;
			((Array)lines_edited_args[1])[1] = 0;
			MessageQueue::get_singleton()->flush();

			SIGNAL_DISCARD("text_set");
			SIGNAL_DISCARD("text_changed");
			SIGNAL_DISCARD("lines_edited_from");
			SIGNAL_DISCARD("caret_changed");

			text_edit->set_editable(false);
			SEND_GUI_ACTION("ui_text_delete");
			CHECK(text_edit->get_viewport()->is_input_handled());
			CHECK(text_edit->get_text() == " ffi some test text. ffi some test text.");
			CHECK(text_edit->get_caret_line() == 0);
			CHECK(text_edit->get_caret_column() == 0);
			CHECK_FALSE(text_edit->has_selection(0));

			CHECK(text_edit->get_caret_line(1) == 0);
			CHECK(text_edit->get_caret_column(1) == 20);
			CHECK_FALSE(text_edit->has_selection(1));
			SIGNAL_CHECK_FALSE("caret_changed");
			SIGNAL_CHECK_FALSE("text_changed");
			SIGNAL_CHECK_FALSE("lines_edited_from");
			text_edit->set_editable(true);

			text_edit->start_action(TextEdit::EditAction::ACTION_NONE);

			SEND_GUI_ACTION("ui_text_delete");
			CHECK(text_edit->get_viewport()->is_input_handled());
			CHECK(text_edit->get_text() == "ffi some test text.ffi some test text.");
			CHECK(text_edit->get_caret_line() == 0);
			CHECK(text_edit->get_caret_column() == 0);
			CHECK_FALSE(text_edit->has_selection(0));

			CHECK(text_edit->get_caret_line(1) == 0);
			CHECK(text_edit->get_caret_column(1) == 19);
			CHECK_FALSE(text_edit->has_selection(0));
			SIGNAL_CHECK("caret_changed", empty_signal_args);
			SIGNAL_CHECK("text_changed", empty_signal_args);
			SIGNAL_CHECK("lines_edited_from", lines_edited_args);

			text_edit->start_action(TextEdit::EditAction::ACTION_NONE);

			SEND_GUI_ACTION("ui_text_delete");
			CHECK(text_edit->get_viewport()->is_input_handled());
			CHECK(text_edit->get_text() == "fi some test text.fi some test text.");
			CHECK(text_edit->get_caret_line() == 0);
			CHECK(text_edit->get_caret_column() == 0);
			CHECK_FALSE(text_edit->has_selection(0));

			CHECK(text_edit->get_caret_line(1) == 0);
			CHECK(text_edit->get_caret_column(1) == 18);
			CHECK_FALSE(text_edit->has_selection(1));
			SIGNAL_CHECK("caret_changed", empty_signal_args);
			SIGNAL_CHECK("text_changed", empty_signal_args);
			SIGNAL_CHECK("lines_edited_from", lines_edited_args);
		}

		SUBCASE("[TextEdit] ui_text_caret_word_left") {
			text_edit->set_text("\nthis is some test text.\nthis is some test text.");
			text_edit->set_caret_line(1);
			text_edit->set_caret_column(7);

			text_edit->add_caret(2, 7);
			CHECK(text_edit->get_caret_count() == 2);
			MessageQueue::get_singleton()->flush();

			SIGNAL_DISCARD("text_set");
			SIGNAL_DISCARD("text_changed");
			SIGNAL_DISCARD("lines_edited_from");
			SIGNAL_DISCARD("caret_changed");

			// Shift should select.
#ifdef MACOS_ENABLED
			SEND_GUI_KEY_EVENT(Key::LEFT | KeyModifierMask::ALT | KeyModifierMask::SHIFT);
#else
			SEND_GUI_KEY_EVENT(Key::LEFT | KeyModifierMask::CMD_OR_CTRL | KeyModifierMask::SHIFT);
#endif
			CHECK(text_edit->get_viewport()->is_input_handled());
			CHECK(text_edit->get_caret_line() == 1);
			CHECK(text_edit->get_caret_column() == 5);
			CHECK(text_edit->get_selected_text(0) == "is");
			CHECK(text_edit->has_selection(0));

			CHECK(text_edit->get_caret_line(1) == 2);
			CHECK(text_edit->get_caret_column(1) == 5);
			CHECK(text_edit->get_selected_text(1) == "is");
			CHECK(text_edit->has_selection(1));

			SIGNAL_CHECK("caret_changed", empty_signal_args);
			SIGNAL_CHECK_FALSE("text_changed");
			SIGNAL_CHECK_FALSE("lines_edited_from");

			// Should still move caret with selection.
			SEND_GUI_ACTION("ui_text_caret_word_left");
			CHECK(text_edit->get_viewport()->is_input_handled());
			CHECK(text_edit->get_caret_line() == 1);
			CHECK(text_edit->get_caret_column() == 0);
			CHECK_FALSE(text_edit->has_selection(0));

			CHECK(text_edit->get_caret_line(1) == 2);
			CHECK(text_edit->get_caret_column(1) == 0);
			CHECK_FALSE(text_edit->has_selection(1));

			SIGNAL_CHECK("caret_changed", empty_signal_args);
			SIGNAL_CHECK_FALSE("text_changed");
			SIGNAL_CHECK_FALSE("lines_edited_from");

			// Normal word left.
			SEND_GUI_ACTION("ui_text_caret_word_left");
			CHECK(text_edit->get_viewport()->is_input_handled());
			CHECK(text_edit->get_caret_line() == 0);
			CHECK(text_edit->get_caret_column() == 0);
			CHECK_FALSE(text_edit->has_selection(0));

			CHECK(text_edit->get_caret_line(1) == 1);
			CHECK(text_edit->get_caret_column(1) == 23);
			CHECK_FALSE(text_edit->has_selection(1));
			SIGNAL_CHECK("caret_changed", empty_signal_args);
			SIGNAL_CHECK_FALSE("text_changed");
			SIGNAL_CHECK_FALSE("lines_edited_from");
		}

		SUBCASE("[TextEdit] ui_text_caret_left") {
			text_edit->set_text("\nthis is some test text.\nthis is some test text.");
			text_edit->set_caret_line(1);
			text_edit->set_caret_column(7);
			text_edit->select(1, 2, 1, 7);

			text_edit->add_caret(2, 7);
			text_edit->select(2, 2, 2, 7, 1);
			CHECK(text_edit->get_caret_count() == 2);

			MessageQueue::get_singleton()->flush();

			SIGNAL_DISCARD("text_set");
			SIGNAL_DISCARD("text_changed");
			SIGNAL_DISCARD("lines_edited_from");
			SIGNAL_DISCARD("caret_changed");

			// Normal left should deselect and place at selection start.
			SEND_GUI_ACTION("ui_text_caret_left");
			CHECK(text_edit->get_viewport()->is_input_handled());

			CHECK(text_edit->get_caret_line() == 1);
			CHECK(text_edit->get_caret_column() == 2);
			CHECK_FALSE(text_edit->has_selection(0));

			CHECK(text_edit->get_caret_line(1) == 2);
			CHECK(text_edit->get_caret_column(1) == 2);
			CHECK_FALSE(text_edit->has_selection(1));

			SIGNAL_CHECK("caret_changed", empty_signal_args);
			SIGNAL_CHECK_FALSE("text_changed");
			SIGNAL_CHECK_FALSE("lines_edited_from");

			// With shift should select.
			SEND_GUI_KEY_EVENT(Key::LEFT | KeyModifierMask::SHIFT);
			CHECK(text_edit->get_viewport()->is_input_handled());
			CHECK(text_edit->get_caret_line() == 1);
			CHECK(text_edit->get_caret_column() == 1);
			CHECK(text_edit->get_selected_text(0) == "h");
			CHECK(text_edit->has_selection(0));

			CHECK(text_edit->get_caret_line(1) == 2);
			CHECK(text_edit->get_caret_column(1) == 1);
			CHECK(text_edit->get_selected_text(1) == "h");
			CHECK(text_edit->has_selection(1));

			SIGNAL_CHECK("caret_changed", empty_signal_args);
			SIGNAL_CHECK_FALSE("text_changed");
			SIGNAL_CHECK_FALSE("lines_edited_from");

			// All ready at select left, should only deselect.
			SEND_GUI_ACTION("ui_text_caret_left");
			CHECK(text_edit->get_viewport()->is_input_handled());
			CHECK(text_edit->get_caret_line() == 1);
			CHECK(text_edit->get_caret_column() == 1);
			CHECK_FALSE(text_edit->has_selection(0));

			CHECK(text_edit->get_caret_line(1) == 2);
			CHECK(text_edit->get_caret_column(1) == 1);
			CHECK_FALSE(text_edit->has_selection(1));

			SIGNAL_CHECK_FALSE("caret_changed");
			SIGNAL_CHECK_FALSE("text_changed");
			SIGNAL_CHECK_FALSE("lines_edited_from");

			// Normal left.
			SEND_GUI_ACTION("ui_text_caret_left");
			CHECK(text_edit->get_viewport()->is_input_handled());
			CHECK(text_edit->get_caret_line() == 1);
			CHECK(text_edit->get_caret_column() == 0);
			CHECK_FALSE(text_edit->has_selection(0));

			CHECK(text_edit->get_caret_line(1) == 2);
			CHECK(text_edit->get_caret_column(1) == 0);
			CHECK_FALSE(text_edit->has_selection());
			SIGNAL_CHECK("caret_changed", empty_signal_args);
			SIGNAL_CHECK_FALSE("text_changed");
			SIGNAL_CHECK_FALSE("lines_edited_from");

			// Left at col 0 should go up a line.
			SEND_GUI_ACTION("ui_text_caret_left");
			CHECK(text_edit->get_viewport()->is_input_handled());
			CHECK(text_edit->get_caret_line() == 0);
			CHECK(text_edit->get_caret_column() == 0);
			CHECK_FALSE(text_edit->has_selection(0));

			CHECK(text_edit->get_caret_line(1) == 1);
			CHECK(text_edit->get_caret_column(1) == 23);
			CHECK_FALSE(text_edit->has_selection(1));
			SIGNAL_CHECK("caret_changed", empty_signal_args);
			SIGNAL_CHECK_FALSE("text_changed");
			SIGNAL_CHECK_FALSE("lines_edited_from");
		}

		SUBCASE("[TextEdit] ui_text_caret_word_right") {
			text_edit->set_text("this is some test text\n\nthis is some test text\n");
			text_edit->set_caret_line(0);
			text_edit->set_caret_column(13);

			text_edit->add_caret(2, 13);
			CHECK(text_edit->get_caret_count() == 2);

			MessageQueue::get_singleton()->flush();

			SIGNAL_DISCARD("text_set");
			SIGNAL_DISCARD("text_changed");
			SIGNAL_DISCARD("lines_edited_from");
			SIGNAL_DISCARD("caret_changed");

			// Shift should select.
#ifdef MACOS_ENABLED
			SEND_GUI_KEY_EVENT(Key::RIGHT | KeyModifierMask::ALT | KeyModifierMask::SHIFT);
#else
			SEND_GUI_KEY_EVENT(Key::RIGHT | KeyModifierMask::CMD_OR_CTRL | KeyModifierMask::SHIFT);
#endif
			CHECK(text_edit->get_viewport()->is_input_handled());
			CHECK(text_edit->get_caret_line() == 0);
			CHECK(text_edit->get_caret_column() == 17);
			CHECK(text_edit->get_selected_text(0) == "test");
			CHECK(text_edit->has_selection(0));

			CHECK(text_edit->get_caret_line(1) == 2);
			CHECK(text_edit->get_caret_column(1) == 17);
			CHECK(text_edit->get_selected_text(1) == "test");
			CHECK(text_edit->has_selection(1));

			SIGNAL_CHECK("caret_changed", empty_signal_args);
			SIGNAL_CHECK_FALSE("text_changed");
			SIGNAL_CHECK_FALSE("lines_edited_from");

			// Should still move caret with selection.
			SEND_GUI_ACTION("ui_text_caret_word_right");
			CHECK(text_edit->get_viewport()->is_input_handled());
			CHECK(text_edit->get_caret_line() == 0);
			CHECK(text_edit->get_caret_column() == 22);
			CHECK_FALSE(text_edit->has_selection(0));

			CHECK(text_edit->get_caret_line(1) == 2);
			CHECK(text_edit->get_caret_column(1) == 22);
			CHECK_FALSE(text_edit->has_selection(1));

			SIGNAL_CHECK("caret_changed", empty_signal_args);
			SIGNAL_CHECK_FALSE("text_changed");
			SIGNAL_CHECK_FALSE("lines_edited_from");

			// Normal word right.
			SEND_GUI_ACTION("ui_text_caret_word_right");
			CHECK(text_edit->get_viewport()->is_input_handled());
			CHECK(text_edit->get_caret_line() == 1);
			CHECK(text_edit->get_caret_column() == 0);
			CHECK_FALSE(text_edit->has_selection(0));

			CHECK(text_edit->get_caret_line(1) == 3);
			CHECK(text_edit->get_caret_column(1) == 0);
			CHECK_FALSE(text_edit->has_selection(1));
			SIGNAL_CHECK("caret_changed", empty_signal_args);
			SIGNAL_CHECK_FALSE("text_changed");
			SIGNAL_CHECK_FALSE("lines_edited_from");
		}

		SUBCASE("[TextEdit] ui_text_caret_right") {
			text_edit->set_text("this is some test text\n\nthis is some test text\n");
			text_edit->set_caret_line(0);
			text_edit->set_caret_column(16);
			text_edit->select(0, 16, 0, 20);

			text_edit->add_caret(2, 16);
			text_edit->select(2, 16, 2, 20, 1);
			CHECK(text_edit->get_caret_count() == 2);

			MessageQueue::get_singleton()->flush();

			SIGNAL_DISCARD("text_set");
			SIGNAL_DISCARD("text_changed");
			SIGNAL_DISCARD("lines_edited_from");
			SIGNAL_DISCARD("caret_changed");

			// Normal right should deselect and place at selection start.
			SEND_GUI_ACTION("ui_text_caret_right");
			CHECK(text_edit->get_viewport()->is_input_handled());
			CHECK(text_edit->get_caret_line() == 0);
			CHECK(text_edit->get_caret_column() == 20);
			CHECK_FALSE(text_edit->has_selection(0));

			CHECK(text_edit->get_caret_line(1) == 2);
			CHECK(text_edit->get_caret_column(1) == 20);
			CHECK_FALSE(text_edit->has_selection(1));

			SIGNAL_CHECK("caret_changed", empty_signal_args);
			SIGNAL_CHECK_FALSE("text_changed");
			SIGNAL_CHECK_FALSE("lines_edited_from");

			// With shift should select.
			SEND_GUI_KEY_EVENT(Key::RIGHT | KeyModifierMask::SHIFT);
			CHECK(text_edit->get_viewport()->is_input_handled());
			CHECK(text_edit->get_caret_line() == 0);
			CHECK(text_edit->get_caret_column() == 21);
			CHECK(text_edit->get_selected_text(0) == "x");
			CHECK(text_edit->has_selection(0));

			CHECK(text_edit->get_caret_line(1) == 2);
			CHECK(text_edit->get_caret_column(1) == 21);
			CHECK(text_edit->get_selected_text(1) == "x");
			CHECK(text_edit->has_selection(1));

			SIGNAL_CHECK("caret_changed", empty_signal_args);
			SIGNAL_CHECK_FALSE("text_changed");
			SIGNAL_CHECK_FALSE("lines_edited_from");

			// All ready at select right, should only deselect.
			SEND_GUI_ACTION("ui_text_caret_right");
			CHECK(text_edit->get_viewport()->is_input_handled());
			CHECK(text_edit->get_caret_line() == 0);
			CHECK(text_edit->get_caret_column() == 21);
			CHECK_FALSE(text_edit->has_selection(0));

			CHECK(text_edit->get_caret_line(1) == 2);
			CHECK(text_edit->get_caret_column(1) == 21);
			CHECK_FALSE(text_edit->has_selection(1));
			SIGNAL_CHECK_FALSE("caret_changed");
			SIGNAL_CHECK_FALSE("text_changed");
			SIGNAL_CHECK_FALSE("lines_edited_from");

			// Normal right.
			SEND_GUI_ACTION("ui_text_caret_right");
			CHECK(text_edit->get_viewport()->is_input_handled());
			CHECK(text_edit->get_caret_line() == 0);
			CHECK(text_edit->get_caret_column() == 22);
			CHECK_FALSE(text_edit->has_selection(0));

			CHECK(text_edit->get_caret_line(1) == 2);
			CHECK(text_edit->get_caret_column(1) == 22);
			CHECK_FALSE(text_edit->has_selection(1));
			SIGNAL_CHECK("caret_changed", empty_signal_args);
			SIGNAL_CHECK_FALSE("text_changed");
			SIGNAL_CHECK_FALSE("lines_edited_from");

			// Right at end col should go down a line.
			SEND_GUI_ACTION("ui_text_caret_right");
			CHECK(text_edit->get_viewport()->is_input_handled());
			CHECK(text_edit->get_caret_line() == 1);
			CHECK(text_edit->get_caret_column() == 0);
			CHECK_FALSE(text_edit->has_selection(0));

			CHECK(text_edit->get_caret_line(1) == 3);
			CHECK(text_edit->get_caret_column(1) == 0);
			CHECK_FALSE(text_edit->has_selection(1));
			SIGNAL_CHECK("caret_changed", empty_signal_args);
			SIGNAL_CHECK_FALSE("text_changed");
			SIGNAL_CHECK_FALSE("lines_edited_from");
		}

		SUBCASE("[TextEdit] ui_text_caret_up") {
			text_edit->set_line_wrapping_mode(TextEdit::LineWrappingMode::LINE_WRAPPING_BOUNDARY);

			text_edit->set_size(Size2(110, 100));
			text_edit->set_text("this is some\nother test\nlines\ngo here\nthis is some\nother test\nlines\ngo here");
			text_edit->set_caret_line(3);
			text_edit->set_caret_column(7);

			text_edit->add_caret(7, 7);
			CHECK(text_edit->get_caret_count() == 2);

			MessageQueue::get_singleton()->flush();

			CHECK(text_edit->is_line_wrapped(0));
			SIGNAL_DISCARD("text_set");
			SIGNAL_DISCARD("text_changed");
			SIGNAL_DISCARD("lines_edited_from");
			SIGNAL_DISCARD("caret_changed");

			// Select + up should select everything to the left on that line.
			SEND_GUI_KEY_EVENT(Key::UP | KeyModifierMask::SHIFT);
			CHECK(text_edit->get_viewport()->is_input_handled());
			CHECK(text_edit->get_caret_line() == 2);
			CHECK(text_edit->get_caret_column() == 5);
			CHECK(text_edit->get_selected_text(0) == "\ngo here");
			CHECK(text_edit->has_selection(0));

			CHECK(text_edit->get_caret_line(1) == 6);
			CHECK(text_edit->get_caret_column(1) == 5);
			CHECK(text_edit->get_selected_text(1) == "\ngo here");
			CHECK(text_edit->has_selection(1));
			SIGNAL_CHECK("caret_changed", empty_signal_args);
			SIGNAL_CHECK_FALSE("text_changed");
			SIGNAL_CHECK_FALSE("lines_edited_from");

			// Should deselect and move up.
			SEND_GUI_ACTION("ui_text_caret_up");
			CHECK(text_edit->get_viewport()->is_input_handled());
			CHECK(text_edit->get_caret_line() == 1);
			CHECK(text_edit->get_caret_column() == 8);
			CHECK_FALSE(text_edit->has_selection(0));

			CHECK(text_edit->get_caret_line(1) == 5);
			CHECK(text_edit->get_caret_column(1) == 8);
			CHECK_FALSE(text_edit->has_selection(1));
			SIGNAL_CHECK("caret_changed", empty_signal_args);
			SIGNAL_CHECK_FALSE("text_changed");
			SIGNAL_CHECK_FALSE("lines_edited_from");

			// Normal up over wrapped line.
			SEND_GUI_ACTION("ui_text_caret_up");
			CHECK(text_edit->get_viewport()->is_input_handled());
			CHECK(text_edit->get_caret_line() == 0);
			CHECK(text_edit->get_caret_column() == 12);
			CHECK_FALSE(text_edit->has_selection(0));

			CHECK(text_edit->get_caret_line(1) == 4);
			CHECK(text_edit->get_caret_column(1) == 12);
			CHECK_FALSE(text_edit->has_selection(1));
			SIGNAL_CHECK("caret_changed", empty_signal_args);
			SIGNAL_CHECK_FALSE("text_changed");
			SIGNAL_CHECK_FALSE("lines_edited_from");
			text_edit->set_caret_column(12, false);

			// Normal up over wrapped line to line 0.
			SEND_GUI_ACTION("ui_text_caret_up");
			CHECK(text_edit->get_viewport()->is_input_handled());
			CHECK(text_edit->get_caret_line() == 0);
			CHECK(text_edit->get_caret_column() == 7);
			CHECK_FALSE(text_edit->has_selection(0));

			CHECK(text_edit->get_caret_line(1) == 4);
			CHECK(text_edit->get_caret_column(1) == 7);
			CHECK_FALSE(text_edit->has_selection(1));
			SIGNAL_CHECK("caret_changed", empty_signal_args);
			SIGNAL_CHECK_FALSE("text_changed");
			SIGNAL_CHECK_FALSE("lines_edited_from");
		}

		SUBCASE("[TextEdit] ui_text_caret_down") {
			text_edit->set_line_wrapping_mode(TextEdit::LineWrappingMode::LINE_WRAPPING_BOUNDARY);

			text_edit->set_size(Size2(110, 100));
			text_edit->set_text("go here\nlines\nother test\nthis is some\ngo here\nlines\nother test\nthis is some");
			text_edit->set_caret_line(0);
			text_edit->set_caret_column(7);

			text_edit->add_caret(4, 7);
			CHECK(text_edit->get_caret_count() == 2);

			MessageQueue::get_singleton()->flush();

			CHECK(text_edit->is_line_wrapped(3));
			SIGNAL_DISCARD("text_set");
			SIGNAL_DISCARD("text_changed");
			SIGNAL_DISCARD("lines_edited_from");
			SIGNAL_DISCARD("caret_changed");

			// Select + down should select everything to the right on that line.
			SEND_GUI_KEY_EVENT(Key::DOWN | KeyModifierMask::SHIFT);
			CHECK(text_edit->get_viewport()->is_input_handled());
			CHECK(text_edit->get_caret_line() == 1);
			CHECK(text_edit->get_caret_column() == 5);
			CHECK(text_edit->get_selected_text(0) == "\nlines");
			CHECK(text_edit->has_selection(0));

			CHECK(text_edit->get_caret_line(1) == 5);
			CHECK(text_edit->get_caret_column(1) == 5);
			CHECK(text_edit->get_selected_text(1) == "\nlines");
			CHECK(text_edit->has_selection(1));
			SIGNAL_CHECK("caret_changed", empty_signal_args);
			SIGNAL_CHECK_FALSE("text_changed");
			SIGNAL_CHECK_FALSE("lines_edited_from");

			// Should deselect and move down.
			SEND_GUI_ACTION("ui_text_caret_down");
			CHECK(text_edit->get_viewport()->is_input_handled());
			CHECK(text_edit->get_caret_line() == 2);
			CHECK(text_edit->get_caret_column() == 8);
			CHECK_FALSE(text_edit->has_selection(0));

			CHECK(text_edit->get_caret_line(1) == 6);
			CHECK(text_edit->get_caret_column(1) == 8);
			CHECK_FALSE(text_edit->has_selection(1));
			SIGNAL_CHECK("caret_changed", empty_signal_args);
			SIGNAL_CHECK_FALSE("text_changed");
			SIGNAL_CHECK_FALSE("lines_edited_from");

			// Normal down over wrapped line.
			SEND_GUI_ACTION("ui_text_caret_down");
			CHECK(text_edit->get_viewport()->is_input_handled());
			CHECK(text_edit->get_caret_line() == 3);
			CHECK(text_edit->get_caret_column() == 7);
			CHECK_FALSE(text_edit->has_selection(0));

			CHECK(text_edit->get_caret_line(1) == 7);
			CHECK(text_edit->get_caret_column(1) == 7);
			CHECK_FALSE(text_edit->has_selection(1));
			SIGNAL_CHECK("caret_changed", empty_signal_args);
			SIGNAL_CHECK_FALSE("text_changed");
			SIGNAL_CHECK_FALSE("lines_edited_from");
			text_edit->set_caret_column(7, false);

			// Normal down over wrapped line to last wrapped line.
			SEND_GUI_ACTION("ui_text_caret_down");
			CHECK(text_edit->get_viewport()->is_input_handled());
			CHECK(text_edit->get_caret_line() == 3);
			CHECK(text_edit->get_caret_column() == 12);
			CHECK_FALSE(text_edit->has_selection(0));

			CHECK(text_edit->get_caret_line(1) == 7);
			CHECK(text_edit->get_caret_column(1) == 12);
			CHECK_FALSE(text_edit->has_selection(1));
			SIGNAL_CHECK("caret_changed", empty_signal_args);
			SIGNAL_CHECK_FALSE("text_changed");
			SIGNAL_CHECK_FALSE("lines_edited_from");
		}

		SUBCASE("[TextEdit] ui_text_caret_document_start") {
			text_edit->set_line_wrapping_mode(TextEdit::LineWrappingMode::LINE_WRAPPING_BOUNDARY);

			text_edit->set_size(Size2(110, 100));
			text_edit->set_text("this is some\nother test\nlines\ngo here");
			text_edit->set_caret_line(4);
			text_edit->set_caret_column(7);

			text_edit->add_caret(3, 2);
			CHECK(text_edit->get_caret_count() == 2);

			MessageQueue::get_singleton()->flush();

			CHECK(text_edit->is_line_wrapped(0));
			SIGNAL_DISCARD("text_set");
			SIGNAL_DISCARD("text_changed");
			SIGNAL_DISCARD("lines_edited_from");
			SIGNAL_DISCARD("caret_changed");

#ifdef MACOS_ENABLED
			SEND_GUI_KEY_EVENT(Key::UP | KeyModifierMask::CMD_OR_CTRL | KeyModifierMask::SHIFT);
#else
			SEND_GUI_KEY_EVENT(Key::HOME | KeyModifierMask::CMD_OR_CTRL | KeyModifierMask::SHIFT);
#endif
			CHECK(text_edit->get_viewport()->is_input_handled());
			CHECK(text_edit->get_text() == "this is some\nother test\nlines\ngo here");
			CHECK(text_edit->get_caret_line() == 0);
			CHECK(text_edit->get_caret_column() == 0);
			CHECK(text_edit->get_selected_text() == "this is some\nother test\nlines\ngo here");
			CHECK(text_edit->has_selection());
			SIGNAL_CHECK("caret_changed", empty_signal_args);
			SIGNAL_CHECK_FALSE("text_changed");
			SIGNAL_CHECK_FALSE("lines_edited_from");
			CHECK(text_edit->get_caret_count() == 1);

			SEND_GUI_ACTION("ui_text_caret_document_start");
			CHECK(text_edit->get_viewport()->is_input_handled());
			CHECK(text_edit->get_text() == "this is some\nother test\nlines\ngo here");
			CHECK(text_edit->get_caret_line() == 0);
			CHECK(text_edit->get_caret_column() == 0);
			CHECK_FALSE(text_edit->has_selection());
			SIGNAL_CHECK_FALSE("caret_changed");
			SIGNAL_CHECK_FALSE("text_changed");
			SIGNAL_CHECK_FALSE("lines_edited_from");
		}

		SUBCASE("[TextEdit] ui_text_caret_document_end") {
			text_edit->set_line_wrapping_mode(TextEdit::LineWrappingMode::LINE_WRAPPING_BOUNDARY);

			text_edit->set_size(Size2(110, 100));
			text_edit->set_text("go here\nlines\nother test\nthis is some");
			text_edit->set_caret_line(0);
			text_edit->set_caret_column(0);

			text_edit->add_caret(1, 0);
			CHECK(text_edit->get_caret_count() == 2);
			MessageQueue::get_singleton()->flush();

			CHECK(text_edit->is_line_wrapped(3));
			SIGNAL_DISCARD("text_set");
			SIGNAL_DISCARD("text_changed");
			SIGNAL_DISCARD("lines_edited_from");
			SIGNAL_DISCARD("caret_changed");

#ifdef MACOS_ENABLED
			SEND_GUI_KEY_EVENT(Key::DOWN | KeyModifierMask::CMD_OR_CTRL | KeyModifierMask::SHIFT);
#else
			SEND_GUI_KEY_EVENT(Key::END | KeyModifierMask::CMD_OR_CTRL | KeyModifierMask::SHIFT);
#endif
			CHECK(text_edit->get_viewport()->is_input_handled());
			CHECK(text_edit->get_text() == "go here\nlines\nother test\nthis is some");
			CHECK(text_edit->get_caret_line() == 3);
			CHECK(text_edit->get_caret_column() == 12);
			CHECK(text_edit->get_selected_text() == "go here\nlines\nother test\nthis is some");
			CHECK(text_edit->has_selection());
			SIGNAL_CHECK("caret_changed", empty_signal_args);
			SIGNAL_CHECK_FALSE("text_changed");
			SIGNAL_CHECK_FALSE("lines_edited_from");
			CHECK(text_edit->get_caret_count() == 1);

			SEND_GUI_ACTION("ui_text_caret_document_end");
			CHECK(text_edit->get_viewport()->is_input_handled());
			CHECK(text_edit->get_text() == "go here\nlines\nother test\nthis is some");
			CHECK(text_edit->get_caret_line() == 3);
			CHECK(text_edit->get_caret_column() == 12);
			CHECK_FALSE(text_edit->has_selection());
			SIGNAL_CHECK_FALSE("caret_changed");
			SIGNAL_CHECK_FALSE("text_changed");
			SIGNAL_CHECK_FALSE("lines_edited_from");
		}

		SUBCASE("[TextEdit] ui_text_caret_line_start") {
			text_edit->set_line_wrapping_mode(TextEdit::LineWrappingMode::LINE_WRAPPING_BOUNDARY);

			text_edit->set_size(Size2(110, 100));
			text_edit->set_text("  this is some\n  this is some");
			text_edit->set_caret_line(0);
			text_edit->set_caret_column(text_edit->get_line(0).length());

			text_edit->add_caret(1, text_edit->get_line(1).length());
			CHECK(text_edit->get_caret_count() == 2);
			MessageQueue::get_singleton()->flush();

			CHECK(text_edit->is_line_wrapped(0));
			SIGNAL_DISCARD("text_set");
			SIGNAL_DISCARD("text_changed");
			SIGNAL_DISCARD("lines_edited_from");
			SIGNAL_DISCARD("caret_changed");

#ifdef MACOS_ENABLED
			SEND_GUI_KEY_EVENT(Key::LEFT | KeyModifierMask::CMD_OR_CTRL | KeyModifierMask::SHIFT);
#else
			SEND_GUI_KEY_EVENT(Key::HOME | KeyModifierMask::SHIFT);
#endif
			CHECK(text_edit->get_viewport()->is_input_handled());
			CHECK(text_edit->get_caret_line() == 0);
			CHECK(text_edit->get_caret_column() == 10);
			CHECK(text_edit->has_selection(0));
			CHECK(text_edit->get_selected_text(0) == "some");

			CHECK(text_edit->get_caret_line(1) == 1);
			CHECK(text_edit->get_caret_column(1) == 10);
			CHECK(text_edit->has_selection(1));
			CHECK(text_edit->get_selected_text(1) == "some");
			SIGNAL_CHECK("caret_changed", empty_signal_args);
			SIGNAL_CHECK_FALSE("text_changed");
			SIGNAL_CHECK_FALSE("lines_edited_from");

			SEND_GUI_ACTION("ui_text_caret_line_start");
			CHECK(text_edit->get_viewport()->is_input_handled());
			CHECK(text_edit->get_caret_line() == 0);
			CHECK(text_edit->get_caret_column() == 2);
			CHECK_FALSE(text_edit->has_selection(0));

			CHECK(text_edit->get_caret_line(1) == 1);
			CHECK(text_edit->get_caret_column(1) == 2);
			CHECK_FALSE(text_edit->has_selection(1));
			SIGNAL_CHECK("caret_changed", empty_signal_args);
			SIGNAL_CHECK_FALSE("text_changed");
			SIGNAL_CHECK_FALSE("lines_edited_from");

			SEND_GUI_ACTION("ui_text_caret_line_start");
			CHECK(text_edit->get_viewport()->is_input_handled());
			CHECK(text_edit->get_caret_line() == 0);
			CHECK(text_edit->get_caret_column() == 0);
			CHECK_FALSE(text_edit->has_selection(0));

			CHECK(text_edit->get_caret_line(1) == 1);
			CHECK(text_edit->get_caret_column(1) == 0);
			CHECK_FALSE(text_edit->has_selection(1));
			SIGNAL_CHECK("caret_changed", empty_signal_args);
			SIGNAL_CHECK_FALSE("text_changed");
			SIGNAL_CHECK_FALSE("lines_edited_from");

			SEND_GUI_ACTION("ui_text_caret_line_start");
			CHECK(text_edit->get_viewport()->is_input_handled());
			CHECK(text_edit->get_caret_line() == 0);
			CHECK(text_edit->get_caret_column() == 2);
			CHECK_FALSE(text_edit->has_selection(0));

			CHECK(text_edit->get_caret_line(1) == 1);
			CHECK(text_edit->get_caret_column(1) == 2);
			CHECK_FALSE(text_edit->has_selection(1));
			SIGNAL_CHECK("caret_changed", empty_signal_args);
			SIGNAL_CHECK_FALSE("text_changed");
			SIGNAL_CHECK_FALSE("lines_edited_from");
		}

		SUBCASE("[TextEdit] ui_text_caret_line_end") {
			text_edit->set_line_wrapping_mode(TextEdit::LineWrappingMode::LINE_WRAPPING_BOUNDARY);

			text_edit->set_size(Size2(110, 100));
			text_edit->set_text("  this is some\n  this is some");
			text_edit->set_caret_line(0);
			text_edit->set_caret_column(0);

			text_edit->add_caret(1, 0);
			CHECK(text_edit->get_caret_count() == 2);
			MessageQueue::get_singleton()->flush();

			CHECK(text_edit->is_line_wrapped(0));
			SIGNAL_DISCARD("text_set");
			SIGNAL_DISCARD("text_changed");
			SIGNAL_DISCARD("lines_edited_from");
			SIGNAL_DISCARD("caret_changed");

#ifdef MACOS_ENABLED
			SEND_GUI_KEY_EVENT(Key::RIGHT | KeyModifierMask::CMD_OR_CTRL | KeyModifierMask::SHIFT);
#else
			SEND_GUI_KEY_EVENT(Key::END | KeyModifierMask::SHIFT);
#endif
			CHECK(text_edit->get_viewport()->is_input_handled());
			CHECK(text_edit->get_caret_line() == 0);
			CHECK(text_edit->get_caret_column() == 9);
			CHECK(text_edit->has_selection(0));
			CHECK(text_edit->get_selected_text(0) == "  this is");

			CHECK(text_edit->get_caret_line(1) == 1);
			CHECK(text_edit->get_caret_column(1) == 9);
			CHECK(text_edit->has_selection(1));
			CHECK(text_edit->get_selected_text(1) == "  this is");
			SIGNAL_CHECK("caret_changed", empty_signal_args);
			SIGNAL_CHECK_FALSE("text_changed");
			SIGNAL_CHECK_FALSE("lines_edited_from");

			SEND_GUI_ACTION("ui_text_caret_line_end");
			CHECK(text_edit->get_viewport()->is_input_handled());
			CHECK(text_edit->get_caret_line() == 0);
			CHECK(text_edit->get_caret_column() == text_edit->get_line(0).length());
			CHECK_FALSE(text_edit->has_selection(0));

			CHECK(text_edit->get_viewport()->is_input_handled());
			CHECK(text_edit->get_caret_line(1) == 1);
			CHECK(text_edit->get_caret_column(1) == text_edit->get_line(1).length());
			CHECK_FALSE(text_edit->has_selection(1));
			SIGNAL_CHECK("caret_changed", empty_signal_args);
			SIGNAL_CHECK_FALSE("text_changed");
			SIGNAL_CHECK_FALSE("lines_edited_from");
		}

		SUBCASE("[TextEdit] unicode") {
			text_edit->set_text("\n");
			text_edit->set_caret_line(0);
			text_edit->set_caret_column(0);

			text_edit->add_caret(1, 0);
			CHECK(text_edit->get_caret_count() == 2);
			text_edit->insert_text_at_caret("a");
			MessageQueue::get_singleton()->flush();

			SIGNAL_DISCARD("text_set");
			SIGNAL_DISCARD("text_changed");
			SIGNAL_DISCARD("lines_edited_from");
			SIGNAL_DISCARD("caret_changed");

			// For the second caret.
			Array args2;
			args2.push_back(1);
			args2.push_back(1);
			lines_edited_args.push_front(args2);

			SEND_GUI_KEY_EVENT(Key::A);
			CHECK(text_edit->get_viewport()->is_input_handled());
			CHECK(text_edit->get_text() == "aA\naA");
			CHECK(text_edit->get_caret_column() == 2);
			CHECK(text_edit->get_caret_column(1) == 2);
			SIGNAL_CHECK("caret_changed", empty_signal_args);
			SIGNAL_CHECK("text_changed", empty_signal_args);
			SIGNAL_CHECK("lines_edited_from", lines_edited_args);

			text_edit->set_editable(false);
			SEND_GUI_KEY_EVENT(Key::A);
			CHECK_FALSE(text_edit->get_viewport()->is_input_handled()); // Should this be handled?
			CHECK(text_edit->get_text() == "aA\naA");
			CHECK(text_edit->get_caret_column() == 2);
			CHECK(text_edit->get_caret_column(1) == 2);
			SIGNAL_CHECK_FALSE("caret_changed");
			SIGNAL_CHECK_FALSE("text_changed");
			SIGNAL_CHECK_FALSE("lines_edited_from");
			text_edit->set_editable(true);

			lines_edited_args.push_back(lines_edited_args[1].duplicate());
			lines_edited_args.push_front(args2.duplicate());

			text_edit->select(0, 0, 0, 1);
			text_edit->select(1, 0, 1, 1, 1);
			SEND_GUI_KEY_EVENT(Key::B);
			CHECK(text_edit->get_viewport()->is_input_handled());
			CHECK(text_edit->get_text() == "BA\nBA");
			CHECK(text_edit->get_caret_column() == 1);
			CHECK(text_edit->get_caret_column(1) == 1);
			SIGNAL_CHECK("caret_changed", empty_signal_args);
			SIGNAL_CHECK("text_changed", empty_signal_args);
			SIGNAL_CHECK("lines_edited_from", lines_edited_args);

			SEND_GUI_ACTION("ui_text_toggle_insert_mode");
			CHECK(text_edit->is_overtype_mode_enabled());

			SEND_GUI_KEY_EVENT(Key::B);
			CHECK(text_edit->get_viewport()->is_input_handled());
			CHECK(text_edit->get_text() == "BB\nBB");
			CHECK(text_edit->get_caret_column() == 2);
			SIGNAL_CHECK("caret_changed", empty_signal_args);
			SIGNAL_CHECK("text_changed", empty_signal_args);
			SIGNAL_CHECK("lines_edited_from", lines_edited_args);

			text_edit->select(0, 0, 0, 1);
			text_edit->select(1, 0, 1, 1, 1);
			SEND_GUI_KEY_EVENT(Key::A);
			CHECK(text_edit->get_viewport()->is_input_handled());
			CHECK(text_edit->get_text() == "AB\nAB");
			CHECK(text_edit->get_caret_column() == 1);
			CHECK(text_edit->get_caret_column(1) == 1);
			SIGNAL_CHECK("caret_changed", empty_signal_args);
			SIGNAL_CHECK("text_changed", empty_signal_args);
			SIGNAL_CHECK("lines_edited_from", lines_edited_args);
			text_edit->set_overtype_mode_enabled(false);
			CHECK_FALSE(text_edit->is_overtype_mode_enabled());

			lines_edited_args.remove_at(0);
			lines_edited_args.remove_at(1);

			SEND_GUI_KEY_EVENT(Key::TAB);
			CHECK(text_edit->get_viewport()->is_input_handled());
			CHECK(text_edit->get_text() == "A\tB\nA\tB");
			CHECK(text_edit->get_caret_column() == 2);
			CHECK(text_edit->get_caret_column(1) == 2);
			SIGNAL_CHECK("caret_changed", empty_signal_args);
			SIGNAL_CHECK("text_changed", empty_signal_args);
			SIGNAL_CHECK("lines_edited_from", lines_edited_args);
		}

		SIGNAL_UNWATCH(text_edit, "text_set");
		SIGNAL_UNWATCH(text_edit, "text_changed");
		SIGNAL_UNWATCH(text_edit, "lines_edited_from");
		SIGNAL_UNWATCH(text_edit, "caret_changed");
	}

	memdelete(text_edit);
}

TEST_CASE("[SceneTree][TextEdit] context menu") {
	TextEdit *text_edit = memnew(TextEdit);
	SceneTree::get_singleton()->get_root()->add_child(text_edit);

	text_edit->set_size(Size2(800, 200));
	text_edit->set_line(0, "Lorem ipsum dolor sit amet, consectetur adipiscing elit. Donec vasius mattis leo, sed porta ex lacinia bibendum. Nunc bibendum pellentesque.");
	MessageQueue::get_singleton()->flush();

	text_edit->set_context_menu_enabled(false);
	CHECK_FALSE(text_edit->is_context_menu_enabled());

	CHECK_FALSE(text_edit->is_menu_visible());
	SEND_GUI_MOUSE_BUTTON_EVENT(Point2i(600, 10), MouseButton::RIGHT, MouseButtonMask::RIGHT, Key::NONE);
	CHECK_FALSE(text_edit->is_menu_visible());

	text_edit->set_context_menu_enabled(true);
	CHECK(text_edit->is_context_menu_enabled());

	CHECK_FALSE(text_edit->is_menu_visible());
	SEND_GUI_MOUSE_BUTTON_EVENT(Point2i(700, 10), MouseButton::RIGHT, MouseButtonMask::RIGHT, Key::NONE);
	CHECK(text_edit->is_menu_visible());

	memdelete(text_edit);
}

TEST_CASE("[SceneTree][TextEdit] versioning") {
	TextEdit *text_edit = memnew(TextEdit);
	SceneTree::get_singleton()->get_root()->add_child(text_edit);

	// Action undo / redo states are tested in the action test e.g selection_delete.
	CHECK_FALSE(text_edit->has_undo());
	CHECK_FALSE(text_edit->has_redo());
	CHECK(text_edit->get_version() == 0);
	CHECK(text_edit->get_saved_version() == 0);

	text_edit->begin_complex_operation();
	text_edit->begin_complex_operation();
	text_edit->begin_complex_operation();

	text_edit->insert_text_at_caret("test");
	CHECK(text_edit->get_version() == 1);
	CHECK(text_edit->get_saved_version() == 0);
	CHECK(text_edit->has_undo());
	CHECK_FALSE(text_edit->has_redo());

	text_edit->end_complex_operation();

	// Can undo and redo mid op.
	text_edit->insert_text_at_caret(" nested");
	CHECK(text_edit->get_version() == 2);
	CHECK(text_edit->get_saved_version() == 0);
	CHECK(text_edit->has_undo());
	CHECK_FALSE(text_edit->has_redo());
	text_edit->undo();

	CHECK(text_edit->has_redo());
	text_edit->redo();

	text_edit->end_complex_operation();

	text_edit->insert_text_at_caret(" ops");
	CHECK(text_edit->get_version() == 3);
	CHECK(text_edit->get_saved_version() == 0);
	CHECK(text_edit->has_undo());
	CHECK_FALSE(text_edit->has_redo());

	text_edit->end_complex_operation();

	text_edit->tag_saved_version();
	CHECK(text_edit->get_saved_version() == 3);

	text_edit->undo();
	CHECK(text_edit->get_line(0) == "");
	CHECK(text_edit->get_version() == 0);
	CHECK(text_edit->get_saved_version() == 3);
	CHECK_FALSE(text_edit->has_undo());
	CHECK(text_edit->has_redo());

	text_edit->redo();
	CHECK(text_edit->get_line(0) == "test ops nested");
	CHECK(text_edit->get_version() == 3);
	CHECK(text_edit->get_saved_version() == 3);
	CHECK(text_edit->has_undo());
	CHECK_FALSE(text_edit->has_redo());

	text_edit->clear_undo_history();
	CHECK_FALSE(text_edit->has_undo());
	CHECK_FALSE(text_edit->has_redo());
	CHECK(text_edit->get_version() == 3); // Should this be cleared?
	CHECK(text_edit->get_saved_version() == 0);

	memdelete(text_edit);
}

TEST_CASE("[SceneTree][TextEdit] search") {
	TextEdit *text_edit = memnew(TextEdit);
	SceneTree::get_singleton()->get_root()->add_child(text_edit);

	text_edit->set_text("hay needle, hay\nHAY NEEDLE, HAY\nwordword.word.word");
	int length = text_edit->get_line(1).length();

	CHECK(text_edit->search("test", 0, 0, 0) == Point2i(-1, -1));
	CHECK(text_edit->search("test", TextEdit::SEARCH_MATCH_CASE, 0, 0) == Point2i(-1, -1));
	CHECK(text_edit->search("test", TextEdit::SEARCH_WHOLE_WORDS, 0, 0) == Point2i(-1, -1));
	CHECK(text_edit->search("test", TextEdit::SEARCH_BACKWARDS, 0, 0) == Point2i(-1, -1));

	CHECK(text_edit->search("test", 0, 1, length) == Point2i(-1, -1));
	CHECK(text_edit->search("test", TextEdit::SEARCH_MATCH_CASE, 1, length) == Point2i(-1, -1));
	CHECK(text_edit->search("test", TextEdit::SEARCH_WHOLE_WORDS, 1, length) == Point2i(-1, -1));
	CHECK(text_edit->search("test", TextEdit::SEARCH_BACKWARDS, 1, length) == Point2i(-1, -1));

	CHECK(text_edit->search("needle", 0, 0, 0) == Point2i(4, 0));
	CHECK(text_edit->search("needle", 0, 1, length) == Point2i(4, 0));
	CHECK(text_edit->search("needle", 0, 0, 5) == Point2i(4, 1));
	CHECK(text_edit->search("needle", TextEdit::SEARCH_BACKWARDS, 0, 0) == Point2i(4, 1));
	CHECK(text_edit->search("needle", TextEdit::SEARCH_BACKWARDS, 1, 5) == Point2i(4, 1));
	CHECK(text_edit->search("needle", TextEdit::SEARCH_BACKWARDS, 1, 3) == Point2i(4, 0));

	CHECK(text_edit->search("needle", TextEdit::SEARCH_MATCH_CASE, 0, 0) == Point2i(4, 0));
	CHECK(text_edit->search("needle", TextEdit::SEARCH_MATCH_CASE | TextEdit::SEARCH_BACKWARDS, 0, 0) == Point2i(4, 0));

	CHECK(text_edit->search("needle", TextEdit::SEARCH_WHOLE_WORDS | TextEdit::SEARCH_MATCH_CASE, 0, 0) == Point2i(4, 0));
	CHECK(text_edit->search("needle", TextEdit::SEARCH_WHOLE_WORDS | TextEdit::SEARCH_MATCH_CASE | TextEdit::SEARCH_BACKWARDS, 0, 0) == Point2i(4, 0));

	CHECK(text_edit->search("need", TextEdit::SEARCH_MATCH_CASE, 0, 0) == Point2i(4, 0));
	CHECK(text_edit->search("need", TextEdit::SEARCH_MATCH_CASE | TextEdit::SEARCH_BACKWARDS, 0, 0) == Point2i(4, 0));

	CHECK(text_edit->search("need", TextEdit::SEARCH_WHOLE_WORDS | TextEdit::SEARCH_MATCH_CASE, 0, 0) == Point2i(-1, -1));
	CHECK(text_edit->search("need", TextEdit::SEARCH_WHOLE_WORDS | TextEdit::SEARCH_MATCH_CASE | TextEdit::SEARCH_BACKWARDS, 0, 0) == Point2i(-1, -1));

	CHECK(text_edit->search("word", TextEdit::SEARCH_WHOLE_WORDS, 2, 0) == Point2i(9, 2));
	CHECK(text_edit->search("word", TextEdit::SEARCH_WHOLE_WORDS, 2, 10) == Point2i(14, 2));
	CHECK(text_edit->search(".word", TextEdit::SEARCH_WHOLE_WORDS, 2, 0) == Point2i(8, 2));
	CHECK(text_edit->search("word.", TextEdit::SEARCH_WHOLE_WORDS, 2, 0) == Point2i(9, 2));

	ERR_PRINT_OFF;
	CHECK(text_edit->search("", 0, 0, 0) == Point2i(-1, -1));
	CHECK(text_edit->search("needle", 0, -1, 0) == Point2i(-1, -1));
	CHECK(text_edit->search("needle", 0, 0, -1) == Point2i(-1, -1));
	CHECK(text_edit->search("needle", 0, 100, 0) == Point2i(-1, -1));
	CHECK(text_edit->search("needle", 0, 0, 100) == Point2i(-1, -1));
	ERR_PRINT_ON;

	memdelete(text_edit);
}

TEST_CASE("[SceneTree][TextEdit] mouse") {
	TextEdit *text_edit = memnew(TextEdit);
	SceneTree::get_singleton()->get_root()->add_child(text_edit);

	text_edit->set_size(Size2(800, 200));

	CHECK(text_edit->get_rect_at_line_column(0, 0).get_position() == Point2i(0, 0));

	text_edit->set_line(0, "A");
	MessageQueue::get_singleton()->flush();
	CHECK(text_edit->get_rect_at_line_column(0, 1).get_position().x > 0);

	text_edit->clear(); // Necessary, otherwise the following test cases fail.

	text_edit->set_line(0, "Lorem ipsum dolor sit amet, consectetur adipiscing elit. Donec vasius mattis leo, sed porta ex lacinia bibendum. Nunc bibendum pellentesque.");
	MessageQueue::get_singleton()->flush();

	CHECK(text_edit->get_word_at_pos(text_edit->get_pos_at_line_column(0, 1)) == "Lorem");
	CHECK(text_edit->get_word_at_pos(text_edit->get_pos_at_line_column(0, 9)) == "ipsum");

	ERR_PRINT_OFF;
	CHECK(text_edit->get_pos_at_line_column(0, -1) == Point2i(-1, -1));
	CHECK(text_edit->get_pos_at_line_column(-1, 0) == Point2i(-1, -1));
	CHECK(text_edit->get_pos_at_line_column(-1, -1) == Point2i(-1, -1));

	CHECK(text_edit->get_pos_at_line_column(0, 500) == Point2i(-1, -1));
	CHECK(text_edit->get_pos_at_line_column(2, 0) == Point2i(-1, -1));
	CHECK(text_edit->get_pos_at_line_column(2, 500) == Point2i(-1, -1));

	// Out of view.
	CHECK(text_edit->get_pos_at_line_column(0, text_edit->get_line(0).length() - 1) == Point2i(-1, -1));
	ERR_PRINT_ON;

	// Add method to get drawn column count?
	Point2i start_pos = text_edit->get_pos_at_line_column(0, 0);
	Point2i end_pos = text_edit->get_pos_at_line_column(0, 105);

	CHECK(text_edit->get_line_column_at_pos(Point2i(start_pos.x, start_pos.y)) == Point2i(0, 0));
	CHECK(text_edit->get_line_column_at_pos(Point2i(end_pos.x, end_pos.y)) == Point2i(104, 0));

	// Should this return Point2i(-1, -1) if its also < 0 not just > vis_lines.
	CHECK(text_edit->get_line_column_at_pos(Point2i(end_pos.x - 100, end_pos.y), false) == Point2i(90, 0));
	CHECK(text_edit->get_line_column_at_pos(Point2i(end_pos.x, end_pos.y + 100), false) == Point2i(-1, -1));
	CHECK(text_edit->get_line_column_at_pos(Point2i(end_pos.x - 100, end_pos.y + 100), false) == Point2i(-1, -1));
	CHECK(text_edit->get_line_column_at_pos(Point2i(end_pos.x, end_pos.y - 100), false) == Point2i(104, 0));
	CHECK(text_edit->get_line_column_at_pos(Point2i(end_pos.x - 100, end_pos.y - 100), false) == Point2i(90, 0));

	CHECK(text_edit->get_line_column_at_pos(Point2i(end_pos.x - 100, end_pos.y)) == Point2i(90, 0));
	CHECK(text_edit->get_line_column_at_pos(Point2i(end_pos.x, end_pos.y + 100)) == Point2i(140, 0));
	CHECK(text_edit->get_line_column_at_pos(Point2i(end_pos.x - 100, end_pos.y + 100)) == Point2i(140, 0));
	CHECK(text_edit->get_line_column_at_pos(Point2i(end_pos.x, end_pos.y - 100)) == Point2i(104, 0));
	CHECK(text_edit->get_line_column_at_pos(Point2i(end_pos.x - 100, end_pos.y - 100)) == Point2i(90, 0));

	memdelete(text_edit);
}

TEST_CASE("[SceneTree][TextEdit] caret") {
	TextEdit *text_edit = memnew(TextEdit);
	text_edit->set_context_menu_enabled(false); // Prohibit sending InputEvents to the context menu.
	SceneTree::get_singleton()->get_root()->add_child(text_edit);

	text_edit->set_size(Size2(800, 200));
	text_edit->grab_focus();
	text_edit->set_line(0, "ffi");

	text_edit->set_caret_mid_grapheme_enabled(true);
	CHECK(text_edit->is_caret_mid_grapheme_enabled());

	SEND_GUI_ACTION("ui_text_caret_right");
	CHECK(text_edit->get_caret_column() == 1);

	SEND_GUI_ACTION("ui_text_caret_right");
	CHECK(text_edit->get_caret_column() == 2);

	SEND_GUI_ACTION("ui_text_caret_right");
	CHECK(text_edit->get_caret_column() == 3);

	SEND_GUI_ACTION("ui_text_caret_left");
	CHECK(text_edit->get_caret_column() == 2);

	text_edit->set_line(0, "Lorem  ipsum dolor sit amet, consectetur adipiscing elit. Donec vasius mattis leo, sed porta ex lacinia bibendum. Nunc bibendum pellentesque.");
	for (int i = 0; i < 3; i++) {
		text_edit->insert_line_at(0, "Lorem  ipsum dolor sit amet, consectetur adipiscing elit. Donec vasius mattis leo, sed porta ex lacinia bibendum. Nunc bibendum pellentesque.");
	}
	MessageQueue::get_singleton()->flush();

	text_edit->set_caret_blink_enabled(false);
	CHECK_FALSE(text_edit->is_caret_blink_enabled());

	text_edit->set_caret_blink_enabled(true);
	CHECK(text_edit->is_caret_blink_enabled());

	text_edit->set_caret_blink_interval(10);
	CHECK(text_edit->get_caret_blink_interval() == 10);

	ERR_PRINT_OFF;
	text_edit->set_caret_blink_interval(-1);
	CHECK(text_edit->get_caret_blink_interval() == 10);

	text_edit->set_caret_blink_interval(0);
	CHECK(text_edit->get_caret_blink_interval() == 10);
	ERR_PRINT_ON;

	text_edit->set_caret_type(TextEdit::CaretType::CARET_TYPE_LINE);
	CHECK(text_edit->get_caret_type() == TextEdit::CaretType::CARET_TYPE_LINE);

	text_edit->set_caret_type(TextEdit::CaretType::CARET_TYPE_BLOCK);
	CHECK(text_edit->get_caret_type() == TextEdit::CaretType::CARET_TYPE_BLOCK);

	text_edit->set_caret_type(TextEdit::CaretType::CARET_TYPE_LINE);
	CHECK(text_edit->get_caret_type() == TextEdit::CaretType::CARET_TYPE_LINE);

	int caret_col = text_edit->get_caret_column();
	text_edit->set_move_caret_on_right_click_enabled(false);
	CHECK_FALSE(text_edit->is_move_caret_on_right_click_enabled());

	SEND_GUI_MOUSE_BUTTON_EVENT(Point2i(100, 1), MouseButton::RIGHT, MouseButtonMask::RIGHT, Key::NONE);
	CHECK(text_edit->get_caret_column() == caret_col);

	text_edit->set_move_caret_on_right_click_enabled(true);
	CHECK(text_edit->is_move_caret_on_right_click_enabled());

	SEND_GUI_MOUSE_BUTTON_EVENT(Point2i(100, 1), MouseButton::RIGHT, MouseButtonMask::RIGHT, Key::NONE);
	CHECK(text_edit->get_caret_column() != caret_col);

	text_edit->set_move_caret_on_right_click_enabled(false);
	CHECK_FALSE(text_edit->is_move_caret_on_right_click_enabled());

	text_edit->set_caret_column(0);
	CHECK(text_edit->get_word_under_caret() == "Lorem");

	text_edit->set_caret_column(4);
	CHECK(text_edit->get_word_under_caret() == "Lorem");

	// Should this work?
	text_edit->set_caret_column(5);
	CHECK(text_edit->get_word_under_caret() == "");

	text_edit->set_caret_column(6);
	CHECK(text_edit->get_word_under_caret() == "");

	text_edit->set_caret_line(1);
	CHECK(text_edit->get_caret_line() == 1);

	text_edit->set_caret_line(-1);
	CHECK(text_edit->get_caret_line() == 0);
	text_edit->set_caret_line(100);
	CHECK(text_edit->get_caret_line() == 3);

	text_edit->set_caret_column(-1);
	CHECK(text_edit->get_caret_column() == 0);
	text_edit->set_caret_column(10000000);
	CHECK(text_edit->get_caret_column() == 141);

	memdelete(text_edit);
}

TEST_CASE("[SceneTree][TextEdit] multicaret") {
	TextEdit *text_edit = memnew(TextEdit);
	SceneTree::get_singleton()->get_root()->add_child(text_edit);
	text_edit->set_multiple_carets_enabled(true);

	Array empty_signal_args;
	empty_signal_args.push_back(Array());

	SIGNAL_WATCH(text_edit, "caret_changed");

	text_edit->set_text("this is\nsome test\ntext");
	text_edit->set_caret_line(0);
	text_edit->set_caret_column(0);
	MessageQueue::get_singleton()->flush();
	SIGNAL_DISCARD("caret_changed");

	SUBCASE("[TextEdit] add remove caret") {
		// Overlapping
		CHECK(text_edit->add_caret(0, 0) == -1);
		MessageQueue::get_singleton()->flush();
		SIGNAL_CHECK_FALSE("caret_changed");

		// Selection
		text_edit->select(0, 0, 2, 4);
		CHECK(text_edit->add_caret(0, 0) == -1);
		CHECK(text_edit->add_caret(2, 4) == -1);
		CHECK(text_edit->add_caret(1, 2) == -1);

		// Out of bounds
		CHECK(text_edit->add_caret(-1, 0) == -1);
		CHECK(text_edit->add_caret(5, 0) == -1);
		CHECK(text_edit->add_caret(0, 100) == -1);

		MessageQueue::get_singleton()->flush();
		SIGNAL_CHECK_FALSE("caret_changed");

		CHECK(text_edit->get_caret_count() == 1);

		text_edit->deselect();
		SIGNAL_CHECK_FALSE("caret_changed");

		CHECK(text_edit->add_caret(0, 1) == 1);
		MessageQueue::get_singleton()->flush();
		SIGNAL_CHECK("caret_changed", empty_signal_args);
		CHECK(text_edit->get_caret_count() == 2);

		CHECK(text_edit->get_caret_line(0) == 0);
		CHECK(text_edit->get_caret_column(0) == 0);

		CHECK(text_edit->get_caret_line(1) == 0);
		CHECK(text_edit->get_caret_column(1) == 1);

		ERR_PRINT_OFF;
		text_edit->remove_caret(-1);
		text_edit->remove_caret(5);
		ERR_PRINT_ON;
		CHECK(text_edit->get_caret_count() == 2);
		SIGNAL_CHECK_FALSE("caret_changed");

		text_edit->remove_caret(0);
		SIGNAL_CHECK_FALSE("caret_changed");
		CHECK(text_edit->get_caret_count() == 1);
		CHECK(text_edit->get_caret_line(0) == 0);
		CHECK(text_edit->get_caret_column(0) == 1);

		ERR_PRINT_OFF;
		text_edit->remove_caret(0);
		CHECK(text_edit->get_caret_count() == 1);
		ERR_PRINT_ON;
	}

	SUBCASE("[TextEdit] caret index edit order") {
		Vector<int> caret_index_get_order;
		caret_index_get_order.push_back(1);
		caret_index_get_order.push_back(0);

		CHECK(text_edit->add_caret(1, 0));
		CHECK(text_edit->get_caret_count() == 2);
		CHECK(text_edit->get_caret_index_edit_order() == caret_index_get_order);

		text_edit->remove_secondary_carets();
		text_edit->set_caret_line(1);
		CHECK(text_edit->add_caret(0, 0));
		CHECK(text_edit->get_caret_count() == 2);

		caret_index_get_order.write[0] = 0;
		caret_index_get_order.write[1] = 1;
		CHECK(text_edit->get_caret_index_edit_order() == caret_index_get_order);
	}

	SUBCASE("[TextEdit] add caret at carets") {
		text_edit->remove_secondary_carets();
		text_edit->set_caret_line(1);
		text_edit->set_caret_column(9);

		text_edit->add_caret_at_carets(true);
		CHECK(text_edit->get_caret_count() == 2);
		CHECK(text_edit->get_caret_line(1) == 2);
		CHECK(text_edit->get_caret_column(1) == 4);

		text_edit->add_caret_at_carets(true);
		CHECK(text_edit->get_caret_count() == 2);

		text_edit->add_caret_at_carets(false);
		CHECK(text_edit->get_caret_count() == 3);
		CHECK(text_edit->get_caret_line(2) == 0);
		CHECK(text_edit->get_caret_column(2) == 7);

		text_edit->remove_secondary_carets();
		text_edit->set_caret_line(0);
		text_edit->set_caret_column(4);
		text_edit->select(0, 0, 0, 4);
		text_edit->add_caret_at_carets(true);
		CHECK(text_edit->get_caret_count() == 2);
		CHECK(text_edit->get_selection_from_line(1) == 1);
		CHECK(text_edit->get_selection_to_line(1) == 1);
		CHECK(text_edit->get_selection_from_column(1) == 0);
		CHECK(text_edit->get_selection_to_column(1) == 3);

		text_edit->add_caret_at_carets(true);
		CHECK(text_edit->get_caret_count() == 3);
		CHECK(text_edit->get_selection_from_line(2) == 2);
		CHECK(text_edit->get_selection_to_line(2) == 2);
		CHECK(text_edit->get_selection_from_column(2) == 0);
		CHECK(text_edit->get_selection_to_column(2) == 4);
	}

	memdelete(text_edit);
}

TEST_CASE("[SceneTree][TextEdit] line wrapping") {
	TextEdit *text_edit = memnew(TextEdit);
	SceneTree::get_singleton()->get_root()->add_child(text_edit);
	text_edit->grab_focus();

	// Set size for boundary.
	text_edit->set_size(Size2(800, 200));
	text_edit->set_line(0, "Lorem ipsum dolor sit amet, consectetur adipiscing elit. Donec vasius mattis leo, sed porta ex lacinia bibendum. Nunc bibendum pellentesque.");
	CHECK_FALSE(text_edit->is_line_wrapped(0));
	CHECK(text_edit->get_line_wrap_count(0) == 0);
	CHECK(text_edit->get_line_wrap_index_at_column(0, 130) == 0);
	CHECK(text_edit->get_line_wrapped_text(0).size() == 1);

	SIGNAL_WATCH(text_edit, "text_set");
	SIGNAL_WATCH(text_edit, "text_changed");
	SIGNAL_WATCH(text_edit, "lines_edited_from");
	SIGNAL_WATCH(text_edit, "caret_changed");

	text_edit->set_line_wrapping_mode(TextEdit::LineWrappingMode::LINE_WRAPPING_BOUNDARY);
	SIGNAL_CHECK_FALSE("text_set");
	SIGNAL_CHECK_FALSE("text_changed");
	SIGNAL_CHECK_FALSE("lines_edited_from");
	SIGNAL_CHECK_FALSE("caret_changed");

	CHECK(text_edit->is_line_wrapped(0));
	CHECK(text_edit->get_line_wrap_count(0) == 1);
	CHECK(text_edit->get_line_wrap_index_at_column(0, 130) == 1);
	CHECK(text_edit->get_line_wrapped_text(0).size() == 2);

	SIGNAL_UNWATCH(text_edit, "text_set");
	SIGNAL_UNWATCH(text_edit, "text_changed");
	SIGNAL_UNWATCH(text_edit, "lines_edited_from");
	SIGNAL_UNWATCH(text_edit, "caret_changed");

	ERR_PRINT_OFF;
	CHECK_FALSE(text_edit->is_line_wrapped(-1));
	CHECK_FALSE(text_edit->is_line_wrapped(1));
	CHECK(text_edit->get_line_wrap_count(-1) == 0);
	CHECK(text_edit->get_line_wrap_count(1) == 0);
	CHECK(text_edit->get_line_wrap_index_at_column(-1, 0) == 0);
	CHECK(text_edit->get_line_wrap_index_at_column(0, -1) == 0);
	CHECK(text_edit->get_line_wrap_index_at_column(1, 0) == 0);
	CHECK(text_edit->get_line_wrap_index_at_column(0, 10000) == 0);
	CHECK(text_edit->get_line_wrapped_text(-1).size() == 0);
	CHECK(text_edit->get_line_wrapped_text(1).size() == 0);
	ERR_PRINT_ON;

	memdelete(text_edit);
}

TEST_CASE("[SceneTree][TextEdit] viewport") {
	TextEdit *text_edit = memnew(TextEdit);
	SceneTree::get_singleton()->get_root()->add_child(text_edit);

	// No subcases here for performance.
	text_edit->set_size(Size2(800, 600));
	for (int i = 0; i < 50; i++) {
		text_edit->insert_line_at(0, "Lorem ipsum dolor sit amet, consectetur adipiscing elit. Donec vasius mattis leo, sed porta ex lacinia bibendum. Nunc bibendum pellentesque.");
	}
	MessageQueue::get_singleton()->flush();

	const int visible_lines = text_edit->get_visible_line_count();
	const int total_visible_lines = text_edit->get_total_visible_line_count();
	CHECK(total_visible_lines == 51);

	// First visible line.
	CHECK(text_edit->get_first_visible_line() == 0);
	CHECK(text_edit->get_v_scroll() == 0);
	CHECK(text_edit->get_last_full_visible_line() == visible_lines - 1);
	CHECK(text_edit->get_last_full_visible_line_wrap_index() == 0);

	text_edit->set_line_as_first_visible(visible_lines);
	MessageQueue::get_singleton()->flush();
	CHECK(text_edit->get_first_visible_line() == visible_lines);
	CHECK(text_edit->get_v_scroll() == visible_lines);
	CHECK(text_edit->get_last_full_visible_line() == (visible_lines * 2) - 1);
	CHECK(text_edit->get_last_full_visible_line_wrap_index() == 0);

	ERR_PRINT_OFF;
	text_edit->set_line_as_first_visible(-1);
	text_edit->set_line_as_first_visible(500);
	text_edit->set_line_as_first_visible(0, -1);
	text_edit->set_line_as_first_visible(0, 500);
	CHECK(text_edit->get_first_visible_line() == visible_lines);
	ERR_PRINT_ON;

	// Wrap.
	text_edit->set_line_wrapping_mode(TextEdit::LineWrappingMode::LINE_WRAPPING_BOUNDARY);
	MessageQueue::get_singleton()->flush();
	CHECK(text_edit->get_total_visible_line_count() > total_visible_lines);

	text_edit->set_line_as_first_visible(5, 1);
	MessageQueue::get_singleton()->flush();
	CHECK(text_edit->get_first_visible_line() == 5);
	CHECK(text_edit->get_v_scroll() == 11);
	CHECK(text_edit->get_last_full_visible_line() == visible_lines - 6);
	CHECK(text_edit->get_last_full_visible_line_wrap_index() == 1);

	// Reset.
	text_edit->set_line_wrapping_mode(TextEdit::LineWrappingMode::LINE_WRAPPING_NONE);
	MessageQueue::get_singleton()->flush();
	CHECK(text_edit->get_total_visible_line_count() == total_visible_lines);
	text_edit->set_line_as_first_visible(0);
	MessageQueue::get_singleton()->flush();
	CHECK(text_edit->get_first_visible_line() == 0);
	CHECK(text_edit->get_v_scroll() == 0);
	CHECK(text_edit->get_last_full_visible_line() == visible_lines - 1);
	CHECK(text_edit->get_last_full_visible_line_wrap_index() == 0);

	// Last visible line.
	text_edit->set_line_as_last_visible(visible_lines * 2);
	MessageQueue::get_singleton()->flush();
	CHECK(text_edit->get_first_visible_line() == visible_lines);
	CHECK(text_edit->get_v_scroll() == visible_lines);
	CHECK(text_edit->get_last_full_visible_line() == (visible_lines * 2) - 1);
	CHECK(text_edit->get_last_full_visible_line_wrap_index() == 0);

	ERR_PRINT_OFF;
	text_edit->set_line_as_last_visible(-1);
	text_edit->set_line_as_last_visible(500);
	text_edit->set_line_as_last_visible(0, -1);
	text_edit->set_line_as_last_visible(0, 500);
	CHECK(text_edit->get_first_visible_line() == visible_lines);
	CHECK(text_edit->get_last_full_visible_line() == (visible_lines * 2) - 1);
	ERR_PRINT_ON;

	// Wrap.
	text_edit->set_line_wrapping_mode(TextEdit::LineWrappingMode::LINE_WRAPPING_BOUNDARY);
	MessageQueue::get_singleton()->flush();
	CHECK(text_edit->get_total_visible_line_count() > total_visible_lines);

	text_edit->set_line_as_last_visible(visible_lines + 5, 1);
	MessageQueue::get_singleton()->flush();
	CHECK(text_edit->get_first_visible_line() == 16);
	CHECK(text_edit->get_v_scroll() == 32.0);
	CHECK(text_edit->get_last_full_visible_line() == visible_lines + 5);
	CHECK(text_edit->get_last_full_visible_line_wrap_index() == 0);

	// Reset.
	text_edit->set_line_wrapping_mode(TextEdit::LineWrappingMode::LINE_WRAPPING_NONE);
	MessageQueue::get_singleton()->flush();
	CHECK(text_edit->get_total_visible_line_count() == total_visible_lines);
	text_edit->set_line_as_first_visible(0);
	MessageQueue::get_singleton()->flush();
	CHECK(text_edit->get_first_visible_line() == 0);
	CHECK(text_edit->get_v_scroll() == 0);
	CHECK(text_edit->get_last_full_visible_line() == visible_lines - 1);
	CHECK(text_edit->get_last_full_visible_line_wrap_index() == 0);

	// Center.
	text_edit->set_line_as_center_visible(visible_lines + (visible_lines / 2));
	MessageQueue::get_singleton()->flush();
	CHECK(text_edit->get_first_visible_line() == visible_lines);
	CHECK(text_edit->get_v_scroll() == visible_lines);
	CHECK(text_edit->get_last_full_visible_line() == (visible_lines * 2) - 1);
	CHECK(text_edit->get_last_full_visible_line_wrap_index() == 0);

	ERR_PRINT_OFF;
	text_edit->set_line_as_last_visible(-1);
	text_edit->set_line_as_last_visible(500);
	text_edit->set_line_as_last_visible(0, -1);
	text_edit->set_line_as_last_visible(0, 500);
	CHECK(text_edit->get_first_visible_line() == visible_lines);
	CHECK(text_edit->get_last_full_visible_line() == (visible_lines * 2) - 1);
	ERR_PRINT_ON;

	// Wrap.
	text_edit->set_line_wrapping_mode(TextEdit::LineWrappingMode::LINE_WRAPPING_BOUNDARY);
	MessageQueue::get_singleton()->flush();
	CHECK(text_edit->get_total_visible_line_count() > total_visible_lines);

	text_edit->set_line_as_center_visible(visible_lines + (visible_lines / 2) + 5, 1);
	MessageQueue::get_singleton()->flush();
	CHECK(text_edit->get_first_visible_line() == visible_lines + (visible_lines / 2));
	CHECK(text_edit->get_v_scroll() == (visible_lines * 3));
	CHECK(text_edit->get_last_full_visible_line() == (visible_lines * 2) - 1);
	CHECK(text_edit->get_last_full_visible_line_wrap_index() == 1);

	// Scroll past eof.
	int line_count = text_edit->get_line_count();
	text_edit->set_scroll_past_end_of_file_enabled(true);
	MessageQueue::get_singleton()->flush();
	text_edit->set_line_as_center_visible(line_count - 1);
	MessageQueue::get_singleton()->flush();

	CHECK(text_edit->get_first_visible_line() == (visible_lines * 2) + 3);
	CHECK(text_edit->get_v_scroll() == (visible_lines * 4) + 6);
	CHECK(text_edit->get_last_full_visible_line() == (visible_lines * 2) + 8);
	CHECK(text_edit->get_last_full_visible_line_wrap_index() == 0);

	text_edit->set_scroll_past_end_of_file_enabled(false);
	MessageQueue::get_singleton()->flush();
	CHECK(text_edit->get_first_visible_line() == (visible_lines * 2) + 3);
	CHECK(text_edit->get_v_scroll() == (visible_lines * 4) - 4);
	CHECK(text_edit->get_last_full_visible_line() == (visible_lines * 2) + 8);
	CHECK(text_edit->get_last_full_visible_line_wrap_index() == 0);

	// Reset.
	text_edit->set_line_wrapping_mode(TextEdit::LineWrappingMode::LINE_WRAPPING_NONE);
	MessageQueue::get_singleton()->flush();
	CHECK(text_edit->get_total_visible_line_count() == total_visible_lines);
	text_edit->set_line_as_first_visible(0);
	MessageQueue::get_singleton()->flush();
	CHECK(text_edit->get_first_visible_line() == 0);
	CHECK(text_edit->get_v_scroll() == 0);
	CHECK(text_edit->get_last_full_visible_line() == visible_lines - 1);
	CHECK(text_edit->get_last_full_visible_line_wrap_index() == 0);

	// Auto adjust - todo: horizontal scroll.
	// Below.
	MessageQueue::get_singleton()->flush();
	CHECK_FALSE(text_edit->is_caret_visible());
	text_edit->set_caret_line(visible_lines + 5, false);
	CHECK_FALSE(text_edit->is_caret_visible());
	text_edit->adjust_viewport_to_caret();
	MessageQueue::get_singleton()->flush();
	CHECK(text_edit->is_caret_visible());
	CHECK(text_edit->get_first_visible_line() == 5);
	CHECK(text_edit->get_v_scroll() == 5);
	CHECK(text_edit->get_last_full_visible_line() == (visible_lines - 1) + 5);
	CHECK(text_edit->get_last_full_visible_line_wrap_index() == 0);

	text_edit->center_viewport_to_caret();
	MessageQueue::get_singleton()->flush();
	CHECK(text_edit->get_first_visible_line() == visible_lines - 5);
	CHECK(text_edit->get_v_scroll() == visible_lines - 5);
	CHECK(text_edit->get_last_full_visible_line() == (visible_lines * 2) - 6);
	CHECK(text_edit->get_last_full_visible_line_wrap_index() == 0);

	// Caret visible, do nothing.
	text_edit->adjust_viewport_to_caret();
	MessageQueue::get_singleton()->flush();
	CHECK(text_edit->get_first_visible_line() == visible_lines - 5);
	CHECK(text_edit->get_v_scroll() == visible_lines - 5);
	CHECK(text_edit->get_last_full_visible_line() == (visible_lines * 2) - 6);
	CHECK(text_edit->get_last_full_visible_line_wrap_index() == 0);

	// Above.
	text_edit->set_caret_line(1, false);
	MessageQueue::get_singleton()->flush();
	text_edit->adjust_viewport_to_caret();
	MessageQueue::get_singleton()->flush();
	CHECK(text_edit->is_caret_visible());
	CHECK(text_edit->get_first_visible_line() == 1);
	CHECK(text_edit->get_v_scroll() == 1);
	CHECK(text_edit->get_last_full_visible_line() == visible_lines);
	CHECK(text_edit->get_last_full_visible_line_wrap_index() == 0);
	CHECK(text_edit->get_caret_wrap_index() == 0);

	text_edit->set_line_as_first_visible(0);
	MessageQueue::get_singleton()->flush();
	CHECK(text_edit->get_first_visible_line() == 0);
	CHECK(text_edit->get_v_scroll() == 0);
	CHECK(text_edit->get_last_full_visible_line() == visible_lines - 1);
	CHECK(text_edit->get_last_full_visible_line_wrap_index() == 0);

	text_edit->adjust_viewport_to_caret();
	MessageQueue::get_singleton()->flush();
	CHECK(text_edit->get_first_visible_line() == 0);
	CHECK(text_edit->get_v_scroll() == 0);
	CHECK(text_edit->get_last_full_visible_line() == visible_lines - 1);
	CHECK(text_edit->get_last_full_visible_line_wrap_index() == 0);

	// Wrap
	text_edit->set_line_wrapping_mode(TextEdit::LineWrappingMode::LINE_WRAPPING_BOUNDARY);
	MessageQueue::get_singleton()->flush();
	CHECK(text_edit->get_total_visible_line_count() > total_visible_lines);

	text_edit->set_caret_line(visible_lines + 5, false, true, 1);
	MessageQueue::get_singleton()->flush();
	text_edit->adjust_viewport_to_caret();
	MessageQueue::get_singleton()->flush();

	CHECK(text_edit->get_first_visible_line() == (visible_lines / 2) + 6);
	CHECK(text_edit->get_v_scroll() == (visible_lines + (visible_lines / 2)) + 1);
	CHECK(text_edit->get_last_full_visible_line() == (visible_lines) + 5);
	CHECK(text_edit->get_last_full_visible_line_wrap_index() == 0);
	CHECK(text_edit->get_caret_wrap_index() == 1);

	text_edit->center_viewport_to_caret();
	MessageQueue::get_singleton()->flush();
	CHECK(text_edit->get_first_visible_line() == visible_lines);
	CHECK(text_edit->get_v_scroll() == (visible_lines * 2) + 1);
	CHECK(text_edit->get_last_full_visible_line() == (visible_lines * 2) - 11);
	CHECK(text_edit->get_last_full_visible_line_wrap_index() == 1);

	// Caret visible, do nothing.
	text_edit->adjust_viewport_to_caret();
	MessageQueue::get_singleton()->flush();
	CHECK(text_edit->get_first_visible_line() == visible_lines);
	CHECK(text_edit->get_v_scroll() == (visible_lines * 2) + 1);
	CHECK(text_edit->get_last_full_visible_line() == (visible_lines * 2) - 11);
	CHECK(text_edit->get_last_full_visible_line_wrap_index() == 1);

	// Above.
	text_edit->set_caret_line(1, false, true, 1);
	MessageQueue::get_singleton()->flush();
	text_edit->adjust_viewport_to_caret();
	MessageQueue::get_singleton()->flush();
	CHECK(text_edit->is_caret_visible());
	CHECK(text_edit->get_first_visible_line() == 1);
	CHECK(text_edit->get_v_scroll() == 3);
	CHECK(text_edit->get_last_full_visible_line() == (visible_lines / 2) + 1);
	CHECK(text_edit->get_last_full_visible_line_wrap_index() == 1);
	CHECK(text_edit->get_caret_wrap_index() == 1);

	text_edit->set_line_as_first_visible(0);
	MessageQueue::get_singleton()->flush();
	CHECK(text_edit->is_caret_visible());
	CHECK(text_edit->get_first_visible_line() == 0);
	CHECK(text_edit->get_v_scroll() == 0);
	CHECK(text_edit->get_last_full_visible_line() == visible_lines - 11);
	CHECK(text_edit->get_last_full_visible_line_wrap_index() == 0);

	text_edit->adjust_viewport_to_caret();
	MessageQueue::get_singleton()->flush();
	CHECK(text_edit->get_first_visible_line() == 0);
	CHECK(text_edit->get_v_scroll() == 0);
	CHECK(text_edit->get_last_full_visible_line() == visible_lines - 11);
	CHECK(text_edit->get_last_full_visible_line_wrap_index() == 0);

	// Reset.
	text_edit->set_line_wrapping_mode(TextEdit::LineWrappingMode::LINE_WRAPPING_NONE);
	MessageQueue::get_singleton()->flush();
	CHECK(text_edit->get_total_visible_line_count() == total_visible_lines);
	text_edit->set_line_as_first_visible(0);
	MessageQueue::get_singleton()->flush();
	CHECK(text_edit->get_first_visible_line() == 0);
	CHECK(text_edit->get_v_scroll() == 0);
	CHECK(text_edit->get_last_full_visible_line() == visible_lines - 1);
	CHECK(text_edit->get_last_full_visible_line_wrap_index() == 0);
	CHECK(text_edit->get_caret_wrap_index() == 0);

	// Smooth scroll.
	text_edit->set_v_scroll_speed(10);
	CHECK(text_edit->get_v_scroll_speed() == 10);
	ERR_PRINT_OFF;
	text_edit->set_v_scroll_speed(-1);
	CHECK(text_edit->get_v_scroll_speed() == 10);

	text_edit->set_v_scroll_speed(0);
	CHECK(text_edit->get_v_scroll_speed() == 10);

	text_edit->set_v_scroll_speed(1);
	CHECK(text_edit->get_v_scroll_speed() == 1);
	ERR_PRINT_ON;

	// Scroll.
	int v_scroll = text_edit->get_v_scroll();
	SEND_GUI_MOUSE_BUTTON_EVENT(Point2i(10, 10), MouseButton::WHEEL_DOWN, 0, Key::NONE);
	CHECK(text_edit->get_v_scroll() > v_scroll);
	SEND_GUI_MOUSE_BUTTON_EVENT(Point2i(10, 10), MouseButton::WHEEL_UP, 0, Key::NONE);
	CHECK(text_edit->get_v_scroll() == v_scroll);

	// smooth scroll speed.
	text_edit->set_smooth_scroll_enabled(true);

	v_scroll = text_edit->get_v_scroll();
	SEND_GUI_MOUSE_BUTTON_EVENT(Point2i(10, 10), MouseButton::WHEEL_DOWN, 0, Key::NONE);
	text_edit->notification(TextEdit::NOTIFICATION_INTERNAL_PHYSICS_PROCESS);
	CHECK(text_edit->get_v_scroll() >= v_scroll);
	SEND_GUI_MOUSE_BUTTON_EVENT(Point2i(10, 10), MouseButton::WHEEL_UP, 0, Key::NONE);
	text_edit->notification(TextEdit::NOTIFICATION_INTERNAL_PHYSICS_PROCESS);
	CHECK(text_edit->get_v_scroll() == v_scroll);

	v_scroll = text_edit->get_v_scroll();
	text_edit->set_v_scroll_speed(10000);
	SEND_GUI_MOUSE_BUTTON_EVENT(Point2i(10, 10), MouseButton::WHEEL_DOWN, 0, Key::NONE);
	text_edit->notification(TextEdit::NOTIFICATION_INTERNAL_PHYSICS_PROCESS);
	CHECK(text_edit->get_v_scroll() >= v_scroll);
	SEND_GUI_MOUSE_BUTTON_EVENT(Point2i(10, 10), MouseButton::WHEEL_UP, 0, Key::NONE);
	text_edit->notification(TextEdit::NOTIFICATION_INTERNAL_PHYSICS_PROCESS);
	CHECK(text_edit->get_v_scroll() == v_scroll);

	ERR_PRINT_OFF;
	CHECK(text_edit->get_scroll_pos_for_line(-1) == 0);
	CHECK(text_edit->get_scroll_pos_for_line(1000) == 0);
	CHECK(text_edit->get_scroll_pos_for_line(1, -1) == 0);
	CHECK(text_edit->get_scroll_pos_for_line(1, 100) == 0);
	ERR_PRINT_ON;

	text_edit->set_h_scroll(-100);
	CHECK(text_edit->get_h_scroll() == 0);

	text_edit->set_h_scroll(10000000);
	CHECK(text_edit->get_h_scroll() == 306);
	CHECK(text_edit->get_h_scroll_bar()->get_combined_minimum_size().x == 8);

	text_edit->set_h_scroll(-100);
	CHECK(text_edit->get_h_scroll() == 0);

	text_edit->set_smooth_scroll_enabled(false);

	CHECK(text_edit->get_first_visible_line() == 0);
	CHECK(text_edit->get_v_scroll() == 0);
	CHECK(text_edit->get_last_full_visible_line() == visible_lines - 1);
	CHECK(text_edit->get_last_full_visible_line_wrap_index() == 0);

	text_edit->grab_focus();
	SEND_GUI_ACTION("ui_text_scroll_down");
	CHECK(text_edit->get_viewport()->is_input_handled());
	CHECK(text_edit->get_caret_line() == 1);
	CHECK(text_edit->get_first_visible_line() == 1);
	CHECK(text_edit->get_v_scroll() == 1);
	CHECK(text_edit->get_last_full_visible_line() == visible_lines);
	CHECK(text_edit->get_last_full_visible_line_wrap_index() == 0);
	CHECK(text_edit->get_caret_wrap_index() == 0);

	SEND_GUI_ACTION("ui_text_scroll_up");
	CHECK(text_edit->get_viewport()->is_input_handled());
	CHECK(text_edit->get_caret_line() == 1);
	CHECK(text_edit->get_first_visible_line() == 0);
	CHECK(text_edit->get_v_scroll() == 0);
	CHECK(text_edit->get_last_full_visible_line() == visible_lines - 1);
	CHECK(text_edit->get_last_full_visible_line_wrap_index() == 0);
	CHECK(text_edit->get_caret_wrap_index() == 0);

	// Page down, similar to VSCode, to end of page then scroll.
	SEND_GUI_ACTION("ui_text_caret_page_down");
	CHECK(text_edit->get_viewport()->is_input_handled());
	CHECK(text_edit->get_caret_line() == 21);
	CHECK(text_edit->get_first_visible_line() == 0);
	CHECK(text_edit->get_v_scroll() == 0);
	CHECK(text_edit->get_last_full_visible_line() == visible_lines - 1);
	CHECK(text_edit->get_last_full_visible_line_wrap_index() == 0);
	CHECK(text_edit->get_caret_wrap_index() == 0);

	SEND_GUI_ACTION("ui_text_caret_page_down");
	CHECK(text_edit->get_viewport()->is_input_handled());
	CHECK(text_edit->get_caret_line() == 41);
	CHECK(text_edit->get_first_visible_line() == 20);
	CHECK(text_edit->get_v_scroll() == 20);
	CHECK(text_edit->get_last_full_visible_line() == (visible_lines - 1) * 2);
	CHECK(text_edit->get_last_full_visible_line_wrap_index() == 0);
	CHECK(text_edit->get_caret_wrap_index() == 0);

	SEND_GUI_ACTION("ui_text_caret_page_up");
	CHECK(text_edit->get_viewport()->is_input_handled());
	CHECK(text_edit->get_caret_line() == 21);
	CHECK(text_edit->get_first_visible_line() == 20);
	CHECK(text_edit->get_v_scroll() == 20);
	CHECK(text_edit->get_last_full_visible_line() == (visible_lines - 1) * 2);
	CHECK(text_edit->get_last_full_visible_line_wrap_index() == 0);
	CHECK(text_edit->get_caret_wrap_index() == 0);

	SEND_GUI_ACTION("ui_text_caret_page_up");
	CHECK(text_edit->get_viewport()->is_input_handled());
	CHECK(text_edit->get_caret_line() == 1);
	CHECK(text_edit->get_first_visible_line() == 1);
	CHECK(text_edit->get_v_scroll() == 1);
	CHECK(text_edit->get_last_full_visible_line() == visible_lines);
	CHECK(text_edit->get_last_full_visible_line_wrap_index() == 0);
	CHECK(text_edit->get_caret_wrap_index() == 0);

	text_edit->set_line_wrapping_mode(TextEdit::LineWrappingMode::LINE_WRAPPING_NONE);
	MessageQueue::get_singleton()->flush();

	text_edit->grab_focus();
	SEND_GUI_ACTION("ui_text_scroll_down");
	CHECK(text_edit->get_viewport()->is_input_handled());
	CHECK(text_edit->get_caret_line() == 2);
	CHECK(text_edit->get_first_visible_line() == 2);
	CHECK(text_edit->get_v_scroll() == 2);
	CHECK(text_edit->get_last_full_visible_line() == visible_lines + 1);
	CHECK(text_edit->get_last_full_visible_line_wrap_index() == 0);
	CHECK(text_edit->get_caret_wrap_index() == 0);

	SEND_GUI_ACTION("ui_text_scroll_up");
	CHECK(text_edit->get_viewport()->is_input_handled());
	CHECK(text_edit->get_caret_line() == 2);
	CHECK(text_edit->get_first_visible_line() == 1);
	CHECK(text_edit->get_v_scroll() == 1);
	CHECK(text_edit->get_last_full_visible_line() == visible_lines);
	CHECK(text_edit->get_last_full_visible_line_wrap_index() == 0);
	CHECK(text_edit->get_caret_wrap_index() == 0);

	// Page down, similar to VSCode, to end of page then scroll.
	SEND_GUI_ACTION("ui_text_caret_page_down");
	CHECK(text_edit->get_viewport()->is_input_handled());
	CHECK(text_edit->get_caret_line() == 22);
	CHECK(text_edit->get_first_visible_line() == 1);
	CHECK(text_edit->get_v_scroll() == 1);
	CHECK(text_edit->get_last_full_visible_line() == visible_lines);
	CHECK(text_edit->get_last_full_visible_line_wrap_index() == 0);
	CHECK(text_edit->get_caret_wrap_index() == 0);

	SEND_GUI_ACTION("ui_text_caret_page_down");
	CHECK(text_edit->get_viewport()->is_input_handled());
	CHECK(text_edit->get_caret_line() == 42);
	CHECK(text_edit->get_first_visible_line() == 21);
	CHECK(text_edit->get_v_scroll() == 21);
	CHECK(text_edit->get_last_full_visible_line() == (visible_lines * 2) - 1);
	CHECK(text_edit->get_last_full_visible_line_wrap_index() == 0);
	CHECK(text_edit->get_caret_wrap_index() == 0);

	SEND_GUI_ACTION("ui_text_caret_page_up");
	CHECK(text_edit->get_viewport()->is_input_handled());
	CHECK(text_edit->get_caret_line() == 22);
	CHECK(text_edit->get_first_visible_line() == 21);
	CHECK(text_edit->get_v_scroll() == 21);
	CHECK(text_edit->get_last_full_visible_line() == (visible_lines * 2) - 1);
	CHECK(text_edit->get_last_full_visible_line_wrap_index() == 0);
	CHECK(text_edit->get_caret_wrap_index() == 0);

	SEND_GUI_ACTION("ui_text_caret_page_up");
	CHECK(text_edit->get_viewport()->is_input_handled());
	CHECK(text_edit->get_caret_line() == 2);
	CHECK(text_edit->get_first_visible_line() == 2);
	CHECK(text_edit->get_v_scroll() == 2);
	CHECK(text_edit->get_last_full_visible_line() == visible_lines + 1);
	CHECK(text_edit->get_last_full_visible_line_wrap_index() == 0);
	CHECK(text_edit->get_caret_wrap_index() == 0);

	// Typing and undo / redo should adjust viewport
	text_edit->set_caret_line(0);
	text_edit->set_caret_column(0);
	text_edit->set_line_as_first_visible(5);
	MessageQueue::get_singleton()->flush();
	CHECK(text_edit->get_first_visible_line() == 5);

	SEND_GUI_KEY_EVENT(Key::A);
	CHECK(text_edit->get_first_visible_line() == 0);

	text_edit->set_line_as_first_visible(5);
	MessageQueue::get_singleton()->flush();
	CHECK(text_edit->get_first_visible_line() == 5);

	text_edit->undo();
	MessageQueue::get_singleton()->flush();
	CHECK(text_edit->get_first_visible_line() == 0);

	text_edit->set_line_as_first_visible(5);
	MessageQueue::get_singleton()->flush();
	CHECK(text_edit->get_first_visible_line() == 5);

	text_edit->redo();
	MessageQueue::get_singleton()->flush();
	CHECK(text_edit->get_first_visible_line() == 0);

	memdelete(text_edit);
}

TEST_CASE("[SceneTree][TextEdit] setter getters") {
	TextEdit *text_edit = memnew(TextEdit);
	SceneTree::get_singleton()->get_root()->add_child(text_edit);

	SUBCASE("[TextEdit] set and get placeholder") {
		text_edit->set_placeholder("test\nplaceholder");
		CHECK(text_edit->get_placeholder() == "test\nplaceholder");

		CHECK(text_edit->get_text() == "");
		CHECK(text_edit->get_line_count() == 1);
		CHECK(text_edit->get_last_full_visible_line() == 0);
	}

	SUBCASE("[TextEdit] highlight current line") {
		text_edit->set_highlight_current_line(true);
		CHECK(text_edit->is_highlight_current_line_enabled());
		text_edit->set_highlight_current_line(false);
		CHECK_FALSE(text_edit->is_highlight_current_line_enabled());
	}

	SUBCASE("[TextEdit] highlight all occurrences") {
		text_edit->set_highlight_all_occurrences(true);
		CHECK(text_edit->is_highlight_all_occurrences_enabled());
		text_edit->set_highlight_all_occurrences(false);
		CHECK_FALSE(text_edit->is_highlight_all_occurrences_enabled());
	}

	SUBCASE("[TextEdit] draw control chars") {
		text_edit->set_draw_control_chars(true);
		CHECK(text_edit->get_draw_control_chars());
		text_edit->set_draw_control_chars(false);
		CHECK_FALSE(text_edit->get_draw_control_chars());
	}

	SUBCASE("[TextEdit] draw tabs") {
		text_edit->set_draw_tabs(true);
		CHECK(text_edit->is_drawing_tabs());
		text_edit->set_draw_tabs(false);
		CHECK_FALSE(text_edit->is_drawing_tabs());
	}

	SUBCASE("[TextEdit] draw spaces") {
		text_edit->set_draw_spaces(true);
		CHECK(text_edit->is_drawing_spaces());
		text_edit->set_draw_spaces(false);
		CHECK_FALSE(text_edit->is_drawing_spaces());
	}

	SUBCASE("[TextEdit] draw minimao") {
		text_edit->set_draw_minimap(true);
		CHECK(text_edit->is_drawing_minimap());
		text_edit->set_draw_minimap(false);
		CHECK_FALSE(text_edit->is_drawing_minimap());
	}

	SUBCASE("[TextEdit] minimap width") {
		text_edit->set_minimap_width(-1);
		CHECK(text_edit->get_minimap_width() == -1);
		text_edit->set_minimap_width(1000);
		CHECK(text_edit->get_minimap_width() == 1000);
	}

	SUBCASE("[TextEdit] line color background") {
		ERR_PRINT_OFF;
		text_edit->set_line_background_color(-1, Color("#ff0000"));
		text_edit->set_line_background_color(0, Color("#00ff00"));
		text_edit->set_line_background_color(1, Color("#0000ff"));

		CHECK(text_edit->get_line_background_color(-1) == Color());
		CHECK(text_edit->get_line_background_color(0) == Color("#00ff00"));
		CHECK(text_edit->get_line_background_color(1) == Color());
		ERR_PRINT_ON;

		text_edit->set_line_background_color(0, Color("#ffff00"));
		CHECK(text_edit->get_line_background_color(0) == Color("#ffff00"));
	}

	memdelete(text_edit);
}

TEST_CASE("[SceneTree][TextEdit] gutters") {
	TextEdit *text_edit = memnew(TextEdit);
	SceneTree::get_singleton()->get_root()->add_child(text_edit);

	Array empty_signal_args;
	empty_signal_args.push_back(Array());

	SIGNAL_WATCH(text_edit, "gutter_clicked");
	SIGNAL_WATCH(text_edit, "gutter_added");
	SIGNAL_WATCH(text_edit, "gutter_removed");

	SUBCASE("[TextEdit] gutter add and remove") {
		text_edit->add_gutter();
		CHECK(text_edit->get_gutter_count() == 1);
		CHECK(text_edit->get_gutter_width(0) == 24);
		CHECK(text_edit->get_total_gutter_width() == 24 + 2);
		SIGNAL_CHECK("gutter_added", empty_signal_args);

		text_edit->set_gutter_name(0, "test_gutter");
		CHECK(text_edit->get_gutter_name(0) == "test_gutter");

		text_edit->set_gutter_width(0, 10);
		CHECK(text_edit->get_gutter_width(0) == 10);
		CHECK(text_edit->get_total_gutter_width() == 10 + 2);

		text_edit->add_gutter(-100);
		text_edit->set_gutter_width(1, 10);
		CHECK(text_edit->get_gutter_width(1) == 10);
		CHECK(text_edit->get_total_gutter_width() == 20 + 2);
		CHECK(text_edit->get_gutter_count() == 2);
		CHECK(text_edit->get_gutter_name(0) == "test_gutter");
		SIGNAL_CHECK("gutter_added", empty_signal_args);

		text_edit->set_gutter_draw(1, false);
		CHECK(text_edit->get_total_gutter_width() == 10 + 2);

		text_edit->add_gutter(100);
		CHECK(text_edit->get_gutter_count() == 3);
		CHECK(text_edit->get_gutter_width(2) == 24);
		CHECK(text_edit->get_total_gutter_width() == 34 + 2);
		CHECK(text_edit->get_gutter_name(0) == "test_gutter");
		SIGNAL_CHECK("gutter_added", empty_signal_args);

		text_edit->add_gutter(0);
		CHECK(text_edit->get_gutter_count() == 4);
		CHECK(text_edit->get_gutter_width(0) == 24);
		CHECK(text_edit->get_total_gutter_width() == 58 + 2);
		CHECK(text_edit->get_gutter_name(1) == "test_gutter");
		SIGNAL_CHECK("gutter_added", empty_signal_args);

		text_edit->remove_gutter(2);
		CHECK(text_edit->get_gutter_name(1) == "test_gutter");
		CHECK(text_edit->get_gutter_count() == 3);
		CHECK(text_edit->get_total_gutter_width() == 58 + 2);
		SIGNAL_CHECK("gutter_removed", empty_signal_args);

		text_edit->remove_gutter(0);
		CHECK(text_edit->get_gutter_name(0) == "test_gutter");
		CHECK(text_edit->get_gutter_count() == 2);
		CHECK(text_edit->get_total_gutter_width() == 34 + 2);
		SIGNAL_CHECK("gutter_removed", empty_signal_args);

		ERR_PRINT_OFF;
		text_edit->remove_gutter(-1);
		SIGNAL_CHECK_FALSE("gutter_removed");

		text_edit->remove_gutter(100);
		SIGNAL_CHECK_FALSE("gutter_removed");

		CHECK(text_edit->get_gutter_name(-1) == "");
		CHECK(text_edit->get_gutter_name(100) == "");
		ERR_PRINT_ON;
	}

	SUBCASE("[TextEdit] gutter data") {
		text_edit->add_gutter();
		CHECK(text_edit->get_gutter_count() == 1);
		SIGNAL_CHECK("gutter_added", empty_signal_args);

		text_edit->set_gutter_name(0, "test_gutter");
		CHECK(text_edit->get_gutter_name(0) == "test_gutter");

		text_edit->set_gutter_width(0, 10);
		CHECK(text_edit->get_gutter_width(0) == 10);

		text_edit->set_gutter_clickable(0, true);
		CHECK(text_edit->is_gutter_clickable(0));

		text_edit->set_gutter_overwritable(0, true);
		CHECK(text_edit->is_gutter_overwritable(0));

		text_edit->set_gutter_type(0, TextEdit::GutterType::GUTTER_TYPE_CUSTOM);
		CHECK(text_edit->get_gutter_type(0) == TextEdit::GutterType::GUTTER_TYPE_CUSTOM);

		text_edit->set_text("test\ntext");

		ERR_PRINT_OFF;
		text_edit->set_line_gutter_metadata(1, 0, "test");
		text_edit->set_line_gutter_metadata(0, -1, "test");
		text_edit->set_line_gutter_metadata(0, 2, "test");
		text_edit->set_line_gutter_metadata(2, 0, "test");
		text_edit->set_line_gutter_metadata(-1, 0, "test");

		CHECK(text_edit->get_line_gutter_metadata(1, 0) == "test");
		CHECK(text_edit->get_line_gutter_metadata(0, -1) == "");
		CHECK(text_edit->get_line_gutter_metadata(0, 2) == "");
		CHECK(text_edit->get_line_gutter_metadata(2, 0) == "");
		CHECK(text_edit->get_line_gutter_metadata(-1, 0) == "");

		text_edit->set_line_gutter_text(1, 0, "test");
		text_edit->set_line_gutter_text(0, -1, "test");
		text_edit->set_line_gutter_text(0, 2, "test");
		text_edit->set_line_gutter_text(2, 0, "test");
		text_edit->set_line_gutter_text(-1, 0, "test");

		CHECK(text_edit->get_line_gutter_text(1, 0) == "test");
		CHECK(text_edit->get_line_gutter_text(0, -1) == "");
		CHECK(text_edit->get_line_gutter_text(0, 2) == "");
		CHECK(text_edit->get_line_gutter_text(2, 0) == "");
		CHECK(text_edit->get_line_gutter_text(-1, 0) == "");

		text_edit->set_line_gutter_item_color(1, 0, Color(1, 0, 0));
		text_edit->set_line_gutter_item_color(0, -1, Color(1, 0, 0));
		text_edit->set_line_gutter_item_color(0, 2, Color(1, 0, 0));
		text_edit->set_line_gutter_item_color(2, 0, Color(1, 0, 0));
		text_edit->set_line_gutter_item_color(-1, 0, Color(1, 0, 0));

		CHECK(text_edit->get_line_gutter_item_color(1, 0) == Color(1, 0, 0));
		CHECK(text_edit->get_line_gutter_item_color(0, -1) == Color());
		CHECK(text_edit->get_line_gutter_item_color(0, 2) == Color());
		CHECK(text_edit->get_line_gutter_item_color(2, 0) == Color());
		CHECK(text_edit->get_line_gutter_item_color(-1, 0) == Color());

		text_edit->set_line_gutter_clickable(1, 0, true);
		text_edit->set_line_gutter_clickable(0, -1, true);
		text_edit->set_line_gutter_clickable(0, 2, true);
		text_edit->set_line_gutter_clickable(2, 0, true);
		text_edit->set_line_gutter_clickable(-1, 0, true);

		CHECK(text_edit->is_line_gutter_clickable(1, 0) == true);
		CHECK(text_edit->is_line_gutter_clickable(0, -1) == false);
		CHECK(text_edit->is_line_gutter_clickable(0, 2) == false);
		CHECK(text_edit->is_line_gutter_clickable(2, 0) == false);
		CHECK(text_edit->is_line_gutter_clickable(-1, 0) == false);
		ERR_PRINT_ON;

		// Merging tested via CodeEdit gutters.
	}

	SIGNAL_UNWATCH(text_edit, "gutter_clicked");
	SIGNAL_UNWATCH(text_edit, "gutter_added");
	SIGNAL_UNWATCH(text_edit, "gutter_removed");
	memdelete(text_edit);
}

} // namespace TestTextEdit

#endif // TEST_TEXT_EDIT_H
