/**************************************************************************/
/*  test_code_edit.h                                                      */
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

#ifndef TEST_CODE_EDIT_H
#define TEST_CODE_EDIT_H

#include "scene/gui/code_edit.h"

#include "tests/test_macros.h"

namespace TestCodeEdit {

TEST_CASE("[SceneTree][CodeEdit] line gutters") {
	CodeEdit *code_edit = memnew(CodeEdit);
	SceneTree::get_singleton()->get_root()->add_child(code_edit);
	code_edit->grab_focus();

	SUBCASE("[CodeEdit] breakpoints") {
		SIGNAL_WATCH(code_edit, "breakpoint_toggled");

		SUBCASE("[CodeEdit] draw breakpoints gutter") {
			code_edit->set_draw_breakpoints_gutter(false);
			CHECK_FALSE(code_edit->is_drawing_breakpoints_gutter());

			code_edit->set_draw_breakpoints_gutter(true);
			CHECK(code_edit->is_drawing_breakpoints_gutter());
		}

		SUBCASE("[CodeEdit] set line as breakpoint") {
			/* Out of bounds. */
			ERR_PRINT_OFF;

			code_edit->set_line_as_breakpoint(-1, true);
			CHECK_FALSE(code_edit->is_line_breakpointed(-1));
			SIGNAL_CHECK_FALSE("breakpoint_toggled");

			code_edit->set_line_as_breakpoint(1, true);
			CHECK_FALSE(code_edit->is_line_breakpointed(1));
			SIGNAL_CHECK_FALSE("breakpoint_toggled");

			ERR_PRINT_ON;

			Array arg1;
			arg1.push_back(0);
			Array args;
			args.push_back(arg1);

			code_edit->set_line_as_breakpoint(0, true);
			CHECK(code_edit->is_line_breakpointed(0));
			CHECK(code_edit->get_breakpointed_lines()[0] == 0);
			SIGNAL_CHECK("breakpoint_toggled", args);

			code_edit->set_line_as_breakpoint(0, false);
			CHECK_FALSE(code_edit->is_line_breakpointed(0));
			SIGNAL_CHECK("breakpoint_toggled", args);
		}

		SUBCASE("[CodeEdit] clear breakpointed lines") {
			code_edit->clear_breakpointed_lines();
			SIGNAL_CHECK_FALSE("breakpoint_toggled");

			Array arg1;
			arg1.push_back(0);
			Array args;
			args.push_back(arg1);

			code_edit->set_line_as_breakpoint(0, true);
			CHECK(code_edit->is_line_breakpointed(0));
			SIGNAL_CHECK("breakpoint_toggled", args);

			code_edit->clear_breakpointed_lines();
			CHECK_FALSE(code_edit->is_line_breakpointed(0));
			SIGNAL_CHECK("breakpoint_toggled", args);
		}

		SUBCASE("[CodeEdit] breakpoints and set text") {
			Array arg1;
			arg1.push_back(0);
			Array args;
			args.push_back(arg1);

			code_edit->set_text("test\nline");
			code_edit->set_line_as_breakpoint(0, true);
			CHECK(code_edit->is_line_breakpointed(0));
			SIGNAL_CHECK("breakpoint_toggled", args);

			/* breakpoint on lines that still exist are kept. */
			code_edit->set_text("");
			MessageQueue::get_singleton()->flush();
			CHECK(code_edit->is_line_breakpointed(0));
			SIGNAL_CHECK_FALSE("breakpoint_toggled");

			/* breakpoint on lines that are removed should also be removed. */
			code_edit->clear_breakpointed_lines();
			SIGNAL_DISCARD("breakpoint_toggled")

			((Array)args[0])[0] = 1;
			code_edit->set_text("test\nline");
			code_edit->set_line_as_breakpoint(1, true);
			CHECK(code_edit->is_line_breakpointed(1));
			SIGNAL_CHECK("breakpoint_toggled", args);

			code_edit->set_text("");
			MessageQueue::get_singleton()->flush();
			CHECK_FALSE(code_edit->is_line_breakpointed(0));
			ERR_PRINT_OFF;
			CHECK_FALSE(code_edit->is_line_breakpointed(1));
			ERR_PRINT_ON;
			SIGNAL_CHECK("breakpoint_toggled", args);
		}

		SUBCASE("[CodeEdit] breakpoints and clear") {
			Array arg1;
			arg1.push_back(0);
			Array args;
			args.push_back(arg1);

			code_edit->set_text("test\nline");
			code_edit->set_line_as_breakpoint(0, true);
			CHECK(code_edit->is_line_breakpointed(0));
			SIGNAL_CHECK("breakpoint_toggled", args);

			/* breakpoint on lines that still exist are removed. */
			code_edit->clear();
			MessageQueue::get_singleton()->flush();
			CHECK_FALSE(code_edit->is_line_breakpointed(0));
			SIGNAL_CHECK("breakpoint_toggled", args);

			/* breakpoint on lines that are removed should also be removed. */
			code_edit->clear_breakpointed_lines();
			SIGNAL_DISCARD("breakpoint_toggled")

			((Array)args[0])[0] = 1;
			code_edit->set_text("test\nline");
			code_edit->set_line_as_breakpoint(1, true);
			CHECK(code_edit->is_line_breakpointed(1));
			SIGNAL_CHECK("breakpoint_toggled", args);

			code_edit->clear();
			MessageQueue::get_singleton()->flush();
			CHECK_FALSE(code_edit->is_line_breakpointed(0));
			ERR_PRINT_OFF;
			CHECK_FALSE(code_edit->is_line_breakpointed(1));
			ERR_PRINT_ON;
			SIGNAL_CHECK("breakpoint_toggled", args);
		}

		SUBCASE("[CodeEdit] breakpoints and new lines no text") {
			Array arg1;
			arg1.push_back(0);
			Array args;
			args.push_back(arg1);

			/* No text moves breakpoint. */
			code_edit->set_line_as_breakpoint(0, true);
			CHECK(code_edit->is_line_breakpointed(0));
			SIGNAL_CHECK("breakpoint_toggled", args);

			/* Normal. */
			((Array)args[0])[0] = 0;
			Array arg2;
			arg2.push_back(1);
			args.push_back(arg2);

			SEND_GUI_ACTION("ui_text_newline");
			CHECK(code_edit->get_line_count() == 2);
			CHECK_FALSE(code_edit->is_line_breakpointed(0));
			CHECK(code_edit->is_line_breakpointed(1));
			SIGNAL_CHECK("breakpoint_toggled", args);

			/* Non-Breaking. */
			((Array)args[0])[0] = 1;
			((Array)args[1])[0] = 2;
			SEND_GUI_ACTION("ui_text_newline_blank");
			CHECK(code_edit->get_line_count() == 3);
			CHECK_FALSE(code_edit->is_line_breakpointed(1));
			CHECK(code_edit->is_line_breakpointed(2));
			SIGNAL_CHECK("breakpoint_toggled", args);

			/* Above. */
			((Array)args[0])[0] = 2;
			((Array)args[1])[0] = 3;
			SEND_GUI_ACTION("ui_text_newline_above");
			CHECK(code_edit->get_line_count() == 4);
			CHECK_FALSE(code_edit->is_line_breakpointed(2));
			CHECK(code_edit->is_line_breakpointed(3));
			SIGNAL_CHECK("breakpoint_toggled", args);
		}

		SUBCASE("[CodeEdit] breakpoints and new lines with text") {
			Array arg1;
			arg1.push_back(0);
			Array args;
			args.push_back(arg1);

			/* Having text does not move breakpoint. */
			code_edit->insert_text_at_caret("text");
			code_edit->set_line_as_breakpoint(0, true);
			CHECK(code_edit->is_line_breakpointed(0));
			SIGNAL_CHECK("breakpoint_toggled", args);

			/* Normal. */
			SEND_GUI_ACTION("ui_text_newline");
			CHECK(code_edit->get_line_count() == 2);
			CHECK(code_edit->is_line_breakpointed(0));
			CHECK_FALSE(code_edit->is_line_breakpointed(1));
			SIGNAL_CHECK_FALSE("breakpoint_toggled");

			/* Non-Breaking. */
			code_edit->set_caret_line(0);
			SEND_GUI_ACTION("ui_text_newline_blank");
			CHECK(code_edit->get_line_count() == 3);
			CHECK(code_edit->is_line_breakpointed(0));
			CHECK_FALSE(code_edit->is_line_breakpointed(1));
			SIGNAL_CHECK_FALSE("breakpoint_toggled");

			/* Above does move. */
			((Array)args[0])[0] = 0;
			Array arg2;
			arg2.push_back(1);
			args.push_back(arg2);

			code_edit->set_caret_line(0);
			SEND_GUI_ACTION("ui_text_newline_above");
			CHECK(code_edit->get_line_count() == 4);
			CHECK_FALSE(code_edit->is_line_breakpointed(0));
			CHECK(code_edit->is_line_breakpointed(1));
			SIGNAL_CHECK("breakpoint_toggled", args);
		}

		SUBCASE("[CodeEdit] breakpoints and backspace") {
			Array arg1;
			arg1.push_back(1);
			Array args;
			args.push_back(arg1);

			code_edit->set_text("\n\n");
			code_edit->set_line_as_breakpoint(1, true);
			CHECK(code_edit->is_line_breakpointed(1));
			SIGNAL_CHECK("breakpoint_toggled", args);

			code_edit->set_caret_line(2);

			/* backspace onto line does not remove breakpoint */
			SEND_GUI_ACTION("ui_text_backspace");
			CHECK(code_edit->is_line_breakpointed(1));
			SIGNAL_CHECK_FALSE("breakpoint_toggled");

			/* backspace on breakpointed line removes it */
			SEND_GUI_ACTION("ui_text_backspace");
			CHECK_FALSE(code_edit->is_line_breakpointed(0));
			ERR_PRINT_OFF;
			CHECK_FALSE(code_edit->is_line_breakpointed(1));
			ERR_PRINT_ON;
			SIGNAL_CHECK("breakpoint_toggled", args);

			/* Backspace above breakpointed line moves it. */
			((Array)args[0])[0] = 2;

			code_edit->set_text("\n\n");
			code_edit->set_line_as_breakpoint(2, true);
			CHECK(code_edit->is_line_breakpointed(2));
			SIGNAL_CHECK("breakpoint_toggled", args);

			code_edit->set_caret_line(1);

			Array arg2;
			arg2.push_back(1);
			args.push_back(arg2);
			SEND_GUI_ACTION("ui_text_backspace");
			ERR_PRINT_OFF;
			CHECK_FALSE(code_edit->is_line_breakpointed(2));
			ERR_PRINT_ON;
			CHECK(code_edit->is_line_breakpointed(1));
			SIGNAL_CHECK("breakpoint_toggled", args);
		}

		SUBCASE("[CodeEdit] breakpoints and delete") {
			Array arg1;
			arg1.push_back(1);
			Array args;
			args.push_back(arg1);

			code_edit->set_text("\n\n");
			code_edit->set_line_as_breakpoint(1, true);
			CHECK(code_edit->is_line_breakpointed(1));
			SIGNAL_CHECK("breakpoint_toggled", args);
			code_edit->set_caret_line(1);

			/* Delete onto breakpointed lines does not remove it. */
			SEND_GUI_ACTION("ui_text_delete");
			CHECK(code_edit->get_line_count() == 2);
			CHECK(code_edit->is_line_breakpointed(1));
			SIGNAL_CHECK_FALSE("breakpoint_toggled");

			/* Delete moving breakpointed line up removes it. */
			code_edit->set_caret_line(0);
			SEND_GUI_ACTION("ui_text_delete");
			CHECK(code_edit->get_line_count() == 1);
			ERR_PRINT_OFF;
			CHECK_FALSE(code_edit->is_line_breakpointed(1));
			ERR_PRINT_ON;
			SIGNAL_CHECK("breakpoint_toggled", args);

			/* Delete above breakpointed line moves it. */
			((Array)args[0])[0] = 2;

			code_edit->set_text("\n\n");
			code_edit->set_line_as_breakpoint(2, true);
			CHECK(code_edit->is_line_breakpointed(2));
			SIGNAL_CHECK("breakpoint_toggled", args);

			code_edit->set_caret_line(0);

			Array arg2;
			arg2.push_back(1);
			args.push_back(arg2);
			SEND_GUI_ACTION("ui_text_delete");
			ERR_PRINT_OFF;
			CHECK_FALSE(code_edit->is_line_breakpointed(2));
			ERR_PRINT_ON;
			CHECK(code_edit->is_line_breakpointed(1));
			SIGNAL_CHECK("breakpoint_toggled", args);
		}

		SUBCASE("[CodeEdit] breakpoints and delete selection") {
			Array arg1;
			arg1.push_back(1);
			Array args;
			args.push_back(arg1);

			code_edit->set_text("\n\n");
			code_edit->set_line_as_breakpoint(1, true);
			CHECK(code_edit->is_line_breakpointed(1));
			SIGNAL_CHECK("breakpoint_toggled", args);

			code_edit->select(0, 0, 2, 0);
			code_edit->delete_selection();
			MessageQueue::get_singleton()->flush();
			CHECK_FALSE(code_edit->is_line_breakpointed(0));
			SIGNAL_CHECK("breakpoint_toggled", args);

			/* Should handle breakpoint move when deleting selection by adding less text then removed. */
			((Array)args[0])[0] = 9;

			code_edit->set_text("\n\n\n\n\n\n\n\n\n");
			code_edit->set_line_as_breakpoint(9, true);
			CHECK(code_edit->is_line_breakpointed(9));
			SIGNAL_CHECK("breakpoint_toggled", args);

			code_edit->select(0, 0, 6, 0);

			Array arg2;
			arg2.push_back(4);
			args.push_back(arg2);
			SEND_GUI_ACTION("ui_text_newline");
			ERR_PRINT_OFF;
			CHECK_FALSE(code_edit->is_line_breakpointed(9));
			ERR_PRINT_ON;
			CHECK(code_edit->is_line_breakpointed(4));
			SIGNAL_CHECK("breakpoint_toggled", args);

			/* Should handle breakpoint move when deleting selection by adding more text then removed. */
			((Array)args[0])[0] = 9;
			((Array)args[1])[0] = 14;

			code_edit->insert_text_at_caret("\n\n\n\n\n");
			MessageQueue::get_singleton()->flush();
			SIGNAL_DISCARD("breakpoint_toggled")
			CHECK(code_edit->is_line_breakpointed(9));

			code_edit->select(0, 0, 6, 0);
			code_edit->insert_text_at_caret("\n\n\n\n\n\n\n\n\n\n\n");
			MessageQueue::get_singleton()->flush();
			CHECK(code_edit->is_line_breakpointed(14));
			SIGNAL_CHECK("breakpoint_toggled", args);
		}

		SUBCASE("[CodeEdit] breakpoints and undo") {
			Array arg1;
			arg1.push_back(1);
			Array args;
			args.push_back(arg1);

			code_edit->set_text("\n\n");
			code_edit->set_line_as_breakpoint(1, true);
			CHECK(code_edit->is_line_breakpointed(1));
			SIGNAL_CHECK("breakpoint_toggled", args);

			code_edit->select(0, 0, 2, 0);
			code_edit->delete_selection();
			MessageQueue::get_singleton()->flush();
			CHECK_FALSE(code_edit->is_line_breakpointed(0));
			SIGNAL_CHECK("breakpoint_toggled", args);

			/* Undo does not restore breakpoint. */
			code_edit->undo();
			CHECK_FALSE(code_edit->is_line_breakpointed(1));
			SIGNAL_CHECK_FALSE("breakpoint_toggled");
		}

		SIGNAL_UNWATCH(code_edit, "breakpoint_toggled");
	}

	SUBCASE("[CodeEdit] bookmarks") {
		SUBCASE("[CodeEdit] draw bookmarks gutter") {
			code_edit->set_draw_bookmarks_gutter(false);
			CHECK_FALSE(code_edit->is_drawing_bookmarks_gutter());

			code_edit->set_draw_bookmarks_gutter(true);
			CHECK(code_edit->is_drawing_bookmarks_gutter());
		}

		SUBCASE("[CodeEdit] set line as bookmarks") {
			/* Out of bounds. */
			ERR_PRINT_OFF;

			code_edit->set_line_as_bookmarked(-1, true);
			CHECK_FALSE(code_edit->is_line_bookmarked(-1));

			code_edit->set_line_as_bookmarked(1, true);
			CHECK_FALSE(code_edit->is_line_bookmarked(1));

			ERR_PRINT_ON;

			code_edit->set_line_as_bookmarked(0, true);
			CHECK(code_edit->get_bookmarked_lines()[0] == 0);
			CHECK(code_edit->is_line_bookmarked(0));

			code_edit->set_line_as_bookmarked(0, false);
			CHECK_FALSE(code_edit->is_line_bookmarked(0));
		}

		SUBCASE("[CodeEdit] clear bookmarked lines") {
			code_edit->clear_bookmarked_lines();

			code_edit->set_line_as_bookmarked(0, true);
			CHECK(code_edit->is_line_bookmarked(0));

			code_edit->clear_bookmarked_lines();
			CHECK_FALSE(code_edit->is_line_bookmarked(0));
		}

		SUBCASE("[CodeEdit] bookmarks and set text") {
			code_edit->set_text("test\nline");
			code_edit->set_line_as_bookmarked(0, true);
			CHECK(code_edit->is_line_bookmarked(0));

			/* bookmarks on lines that still exist are kept. */
			code_edit->set_text("");
			MessageQueue::get_singleton()->flush();
			CHECK(code_edit->is_line_bookmarked(0));

			/* bookmarks on lines that are removed should also be removed. */
			code_edit->clear_bookmarked_lines();

			code_edit->set_text("test\nline");
			code_edit->set_line_as_bookmarked(1, true);
			CHECK(code_edit->is_line_bookmarked(1));

			code_edit->set_text("");
			MessageQueue::get_singleton()->flush();
			CHECK_FALSE(code_edit->is_line_bookmarked(0));
			ERR_PRINT_OFF;
			CHECK_FALSE(code_edit->is_line_bookmarked(1));
			ERR_PRINT_ON;
		}

		SUBCASE("[CodeEdit] bookmarks and clear") {
			code_edit->set_text("test\nline");
			code_edit->set_line_as_bookmarked(0, true);
			CHECK(code_edit->is_line_bookmarked(0));

			/* bookmarks on lines that still exist are removed. */
			code_edit->clear();
			MessageQueue::get_singleton()->flush();
			CHECK_FALSE(code_edit->is_line_bookmarked(0));

			/* bookmarks on lines that are removed should also be removed. */
			code_edit->clear_bookmarked_lines();

			code_edit->set_text("test\nline");
			code_edit->set_line_as_bookmarked(1, true);
			CHECK(code_edit->is_line_bookmarked(1));

			code_edit->clear();
			MessageQueue::get_singleton()->flush();
			CHECK_FALSE(code_edit->is_line_bookmarked(0));
			ERR_PRINT_OFF;
			CHECK_FALSE(code_edit->is_line_bookmarked(1));
			ERR_PRINT_ON;
		}

		SUBCASE("[CodeEdit] bookmarks and new lines no text") {
			/* No text moves bookmarks. */
			code_edit->set_line_as_bookmarked(0, true);
			CHECK(code_edit->is_line_bookmarked(0));

			/* Normal. */
			SEND_GUI_ACTION("ui_text_newline");
			CHECK(code_edit->get_line_count() == 2);
			CHECK_FALSE(code_edit->is_line_bookmarked(0));
			CHECK(code_edit->is_line_bookmarked(1));

			/* Non-Breaking. */
			SEND_GUI_ACTION("ui_text_newline_blank");
			CHECK(code_edit->get_line_count() == 3);
			CHECK_FALSE(code_edit->is_line_bookmarked(1));
			CHECK(code_edit->is_line_bookmarked(2));

			/* Above. */
			SEND_GUI_ACTION("ui_text_newline_above");
			CHECK(code_edit->get_line_count() == 4);
			CHECK_FALSE(code_edit->is_line_bookmarked(2));
			CHECK(code_edit->is_line_bookmarked(3));
		}

		SUBCASE("[CodeEdit] bookmarks and new lines with text") {
			/* Having text does not move bookmark. */
			code_edit->insert_text_at_caret("text");
			code_edit->set_line_as_bookmarked(0, true);
			CHECK(code_edit->is_line_bookmarked(0));

			/* Normal. */
			SEND_GUI_ACTION("ui_text_newline");
			CHECK(code_edit->get_line_count() == 2);
			CHECK(code_edit->is_line_bookmarked(0));
			CHECK_FALSE(code_edit->is_line_bookmarked(1));

			/* Non-Breaking. */
			code_edit->set_caret_line(0);
			SEND_GUI_ACTION("ui_text_newline_blank");
			CHECK(code_edit->get_line_count() == 3);
			CHECK(code_edit->is_line_bookmarked(0));
			CHECK_FALSE(code_edit->is_line_bookmarked(1));

			/* Above does move. */
			code_edit->set_caret_line(0);
			SEND_GUI_ACTION("ui_text_newline_above");
			CHECK(code_edit->get_line_count() == 4);
			CHECK_FALSE(code_edit->is_line_bookmarked(0));
			CHECK(code_edit->is_line_bookmarked(1));
		}

		SUBCASE("[CodeEdit] bookmarks and backspace") {
			code_edit->set_text("\n\n");
			code_edit->set_line_as_bookmarked(1, true);
			CHECK(code_edit->is_line_bookmarked(1));

			code_edit->set_caret_line(2);

			/* backspace onto line does not remove bookmark */
			SEND_GUI_ACTION("ui_text_backspace");
			CHECK(code_edit->is_line_bookmarked(1));

			/* backspace on bookmarked line removes it */
			SEND_GUI_ACTION("ui_text_backspace");
			CHECK_FALSE(code_edit->is_line_bookmarked(0));
			ERR_PRINT_OFF;
			CHECK_FALSE(code_edit->is_line_bookmarked(1));
			ERR_PRINT_ON;
		}

		SUBCASE("[CodeEdit] bookmarks and delete") {
			code_edit->set_text("\n\n");
			code_edit->set_line_as_bookmarked(1, true);
			CHECK(code_edit->is_line_bookmarked(1));
			code_edit->set_caret_line(1);

			/* Delete onto bookmarked lines does not remove it. */
			SEND_GUI_ACTION("ui_text_delete");
			CHECK(code_edit->get_line_count() == 2);
			CHECK(code_edit->is_line_bookmarked(1));

			/* Delete moving bookmarked line up removes it. */
			code_edit->set_caret_line(0);
			SEND_GUI_ACTION("ui_text_delete");
			CHECK(code_edit->get_line_count() == 1);
			ERR_PRINT_OFF;
			CHECK_FALSE(code_edit->is_line_bookmarked(1));
			ERR_PRINT_ON;
		}

		SUBCASE("[CodeEdit] bookmarks and delete selection") {
			code_edit->set_text("\n\n");
			code_edit->set_line_as_bookmarked(1, true);
			CHECK(code_edit->is_line_bookmarked(1));

			code_edit->select(0, 0, 2, 0);
			code_edit->delete_selection();
			MessageQueue::get_singleton()->flush();
			CHECK_FALSE(code_edit->is_line_bookmarked(0));
		}

		SUBCASE("[CodeEdit] bookmarks and undo") {
			code_edit->set_text("\n\n");
			code_edit->set_line_as_bookmarked(1, true);
			CHECK(code_edit->is_line_bookmarked(1));

			code_edit->select(0, 0, 2, 0);
			code_edit->delete_selection();
			MessageQueue::get_singleton()->flush();
			CHECK_FALSE(code_edit->is_line_bookmarked(0));

			/* Undo does not restore bookmark. */
			code_edit->undo();
			CHECK_FALSE(code_edit->is_line_bookmarked(1));
		}
	}

	SUBCASE("[CodeEdit] executing lines") {
		SUBCASE("[CodeEdit] draw executing lines gutter") {
			code_edit->set_draw_executing_lines_gutter(false);
			CHECK_FALSE(code_edit->is_drawing_executing_lines_gutter());

			code_edit->set_draw_executing_lines_gutter(true);
			CHECK(code_edit->is_drawing_executing_lines_gutter());
		}

		SUBCASE("[CodeEdit] set line as executing lines") {
			/* Out of bounds. */
			ERR_PRINT_OFF;

			code_edit->set_line_as_executing(-1, true);
			CHECK_FALSE(code_edit->is_line_executing(-1));

			code_edit->set_line_as_executing(1, true);
			CHECK_FALSE(code_edit->is_line_executing(1));

			ERR_PRINT_ON;

			code_edit->set_line_as_executing(0, true);
			CHECK(code_edit->get_executing_lines()[0] == 0);
			CHECK(code_edit->is_line_executing(0));

			code_edit->set_line_as_executing(0, false);
			CHECK_FALSE(code_edit->is_line_executing(0));
		}

		SUBCASE("[CodeEdit] clear executing lines lines") {
			code_edit->clear_executing_lines();

			code_edit->set_line_as_executing(0, true);
			CHECK(code_edit->is_line_executing(0));

			code_edit->clear_executing_lines();
			CHECK_FALSE(code_edit->is_line_executing(0));
		}

		SUBCASE("[CodeEdit] executing lines and set text") {
			code_edit->set_text("test\nline");
			code_edit->set_line_as_executing(0, true);
			CHECK(code_edit->is_line_executing(0));

			/* executing on lines that still exist are kept. */
			code_edit->set_text("");
			MessageQueue::get_singleton()->flush();
			CHECK(code_edit->is_line_executing(0));

			/* executing on lines that are removed should also be removed. */
			code_edit->clear_executing_lines();

			code_edit->set_text("test\nline");
			code_edit->set_line_as_executing(1, true);
			CHECK(code_edit->is_line_executing(1));

			code_edit->set_text("");
			MessageQueue::get_singleton()->flush();
			CHECK_FALSE(code_edit->is_line_executing(0));
			ERR_PRINT_OFF;
			CHECK_FALSE(code_edit->is_line_executing(1));
			ERR_PRINT_ON;
		}

		SUBCASE("[CodeEdit] executing lines and clear") {
			code_edit->set_text("test\nline");
			code_edit->set_line_as_executing(0, true);
			CHECK(code_edit->is_line_executing(0));

			/* executing on lines that still exist are removed. */
			code_edit->clear();
			MessageQueue::get_singleton()->flush();
			CHECK_FALSE(code_edit->is_line_executing(0));

			/* executing on lines that are removed should also be removed. */
			code_edit->clear_executing_lines();

			code_edit->set_text("test\nline");
			code_edit->set_line_as_executing(1, true);
			CHECK(code_edit->is_line_executing(1));

			code_edit->clear();
			MessageQueue::get_singleton()->flush();
			CHECK_FALSE(code_edit->is_line_executing(0));
			ERR_PRINT_OFF;
			CHECK_FALSE(code_edit->is_line_executing(1));
			ERR_PRINT_ON;
		}

		SUBCASE("[CodeEdit] executing lines and new lines no text") {
			/* No text moves executing lines. */
			code_edit->set_line_as_executing(0, true);
			CHECK(code_edit->is_line_executing(0));

			/* Normal. */
			SEND_GUI_ACTION("ui_text_newline");
			CHECK(code_edit->get_line_count() == 2);
			CHECK_FALSE(code_edit->is_line_executing(0));
			CHECK(code_edit->is_line_executing(1));

			/* Non-Breaking. */
			SEND_GUI_ACTION("ui_text_newline_blank");
			CHECK(code_edit->get_line_count() == 3);
			CHECK_FALSE(code_edit->is_line_executing(1));
			CHECK(code_edit->is_line_executing(2));

			/* Above. */
			SEND_GUI_ACTION("ui_text_newline_above");
			CHECK(code_edit->get_line_count() == 4);
			CHECK_FALSE(code_edit->is_line_executing(2));
			CHECK(code_edit->is_line_executing(3));
		}

		SUBCASE("[CodeEdit] executing lines and new lines with text") {
			/* Having text does not move executing lines. */
			code_edit->insert_text_at_caret("text");
			code_edit->set_line_as_executing(0, true);
			CHECK(code_edit->is_line_executing(0));

			/* Normal. */
			SEND_GUI_ACTION("ui_text_newline");
			CHECK(code_edit->get_line_count() == 2);
			CHECK(code_edit->is_line_executing(0));
			CHECK_FALSE(code_edit->is_line_executing(1));

			/* Non-Breaking. */
			code_edit->set_caret_line(0);
			SEND_GUI_ACTION("ui_text_newline_blank");
			CHECK(code_edit->get_line_count() == 3);
			CHECK(code_edit->is_line_executing(0));
			CHECK_FALSE(code_edit->is_line_executing(1));

			/* Above does move. */
			code_edit->set_caret_line(0);
			SEND_GUI_ACTION("ui_text_newline_above");
			CHECK(code_edit->get_line_count() == 4);
			CHECK_FALSE(code_edit->is_line_executing(0));
			CHECK(code_edit->is_line_executing(1));
		}

		SUBCASE("[CodeEdit] executing lines and backspace") {
			code_edit->set_text("\n\n");
			code_edit->set_line_as_executing(1, true);
			CHECK(code_edit->is_line_executing(1));

			code_edit->set_caret_line(2);

			/* backspace onto line does not remove executing lines. */
			SEND_GUI_ACTION("ui_text_backspace");
			CHECK(code_edit->is_line_executing(1));

			/* backspace on executing line removes it */
			SEND_GUI_ACTION("ui_text_backspace");
			CHECK_FALSE(code_edit->is_line_executing(0));
			ERR_PRINT_OFF;
			CHECK_FALSE(code_edit->is_line_executing(1));
			ERR_PRINT_ON;
		}

		SUBCASE("[CodeEdit] executing lines and delete") {
			code_edit->set_text("\n\n");
			code_edit->set_line_as_executing(1, true);
			CHECK(code_edit->is_line_executing(1));
			code_edit->set_caret_line(1);

			/* Delete onto executing lines does not remove it. */
			SEND_GUI_ACTION("ui_text_delete");
			CHECK(code_edit->get_line_count() == 2);
			CHECK(code_edit->is_line_executing(1));

			/* Delete moving executing line up removes it. */
			code_edit->set_caret_line(0);
			SEND_GUI_ACTION("ui_text_delete");
			CHECK(code_edit->get_line_count() == 1);
			ERR_PRINT_OFF;
			CHECK_FALSE(code_edit->is_line_executing(1));
			ERR_PRINT_ON;
		}

		SUBCASE("[CodeEdit] executing lines and delete selection") {
			code_edit->set_text("\n\n");
			code_edit->set_line_as_executing(1, true);
			CHECK(code_edit->is_line_executing(1));

			code_edit->select(0, 0, 2, 0);
			code_edit->delete_selection();
			MessageQueue::get_singleton()->flush();
			CHECK_FALSE(code_edit->is_line_executing(0));
		}

		SUBCASE("[CodeEdit] executing lines and undo") {
			code_edit->set_text("\n\n");
			code_edit->set_line_as_executing(1, true);
			CHECK(code_edit->is_line_executing(1));

			code_edit->select(0, 0, 2, 0);
			code_edit->delete_selection();
			MessageQueue::get_singleton()->flush();
			CHECK_FALSE(code_edit->is_line_executing(0));

			/* Undo does not restore executing lines. */
			code_edit->undo();
			CHECK_FALSE(code_edit->is_line_executing(1));
		}
	}

	SUBCASE("[CodeEdit] line numbers") {
		SUBCASE("[CodeEdit] draw line numbers gutter and padding") {
			code_edit->set_draw_line_numbers(false);
			CHECK_FALSE(code_edit->is_draw_line_numbers_enabled());

			code_edit->set_draw_line_numbers(true);
			CHECK(code_edit->is_draw_line_numbers_enabled());

			code_edit->set_line_numbers_zero_padded(false);
			CHECK_FALSE(code_edit->is_line_numbers_zero_padded());

			code_edit->set_line_numbers_zero_padded(true);
			CHECK(code_edit->is_line_numbers_zero_padded());

			code_edit->set_line_numbers_zero_padded(false);
			CHECK_FALSE(code_edit->is_line_numbers_zero_padded());

			code_edit->set_draw_line_numbers(false);
			CHECK_FALSE(code_edit->is_draw_line_numbers_enabled());

			code_edit->set_line_numbers_zero_padded(true);
			CHECK(code_edit->is_line_numbers_zero_padded());
		}
	}

	SUBCASE("[CodeEdit] line folding") {
		SUBCASE("[CodeEdit] draw line folding gutter") {
			code_edit->set_draw_fold_gutter(false);
			CHECK_FALSE(code_edit->is_drawing_fold_gutter());

			code_edit->set_draw_fold_gutter(true);
			CHECK(code_edit->is_drawing_fold_gutter());
		}
	}

	memdelete(code_edit);
}

TEST_CASE("[SceneTree][CodeEdit] delimiters") {
	CodeEdit *code_edit = memnew(CodeEdit);
	SceneTree::get_singleton()->get_root()->add_child(code_edit);
	code_edit->grab_focus();

	const Point2 OUTSIDE_DELIMETER = Point2(-1, -1);

	code_edit->clear_string_delimiters();
	code_edit->clear_comment_delimiters();

	SUBCASE("[CodeEdit] add and remove delimiters") {
		SUBCASE("[CodeEdit] add and remove string delimiters") {
			/* Add a delimiter.*/
			code_edit->add_string_delimiter("\"", "\"", false);
			CHECK(code_edit->has_string_delimiter("\""));
			CHECK(code_edit->get_string_delimiters().size() == 1);

			ERR_PRINT_OFF;

			/* Adding a duplicate start key is not allowed. */
			code_edit->add_string_delimiter("\"", "\'", false);
			CHECK(code_edit->get_string_delimiters().size() == 1);

			/* Adding a duplicate end key is allowed. */
			code_edit->add_string_delimiter("'", "\"", false);
			CHECK(code_edit->has_string_delimiter("'"));
			CHECK(code_edit->get_string_delimiters().size() == 2);

			/* Both start and end keys have to be symbols. */
			code_edit->add_string_delimiter("f", "\"", false);
			CHECK_FALSE(code_edit->has_string_delimiter("f"));
			CHECK(code_edit->get_string_delimiters().size() == 2);

			code_edit->add_string_delimiter("f", "\"", false);
			CHECK_FALSE(code_edit->has_string_delimiter("f"));
			CHECK(code_edit->get_string_delimiters().size() == 2);

			code_edit->add_string_delimiter("@", "f", false);
			CHECK_FALSE(code_edit->has_string_delimiter("@"));
			CHECK(code_edit->get_string_delimiters().size() == 2);

			code_edit->add_string_delimiter("f", "f", false);
			CHECK_FALSE(code_edit->has_string_delimiter("f"));
			CHECK(code_edit->get_string_delimiters().size() == 2);

			/* Blank start keys are not allowed */
			code_edit->add_string_delimiter("", "#", false);
			CHECK_FALSE(code_edit->has_string_delimiter("#"));
			CHECK(code_edit->get_string_delimiters().size() == 2);

			ERR_PRINT_ON;

			/* Blank end keys are allowed. */
			code_edit->add_string_delimiter("#", "", false);
			CHECK(code_edit->has_string_delimiter("#"));
			CHECK(code_edit->get_string_delimiters().size() == 3);

			/* Remove a delimiter. */
			code_edit->remove_string_delimiter("#");
			CHECK_FALSE(code_edit->has_string_delimiter("#"));
			CHECK(code_edit->get_string_delimiters().size() == 2);

			/* Set should override existing, and test multiline */
			TypedArray<String> delimiters;
			delimiters.push_back("^^ ^^");

			code_edit->set_string_delimiters(delimiters);
			CHECK_FALSE(code_edit->has_string_delimiter("\""));
			CHECK(code_edit->has_string_delimiter("^^"));
			CHECK(code_edit->get_string_delimiters().size() == 1);

			/* clear should remove all. */
			code_edit->clear_string_delimiters();
			CHECK_FALSE(code_edit->has_string_delimiter("^^"));
			CHECK(code_edit->get_string_delimiters().size() == 0);
		}

		SUBCASE("[CodeEdit] add and remove comment delimiters") {
			/* Add a delimiter.*/
			code_edit->add_comment_delimiter("\"", "\"", false);
			CHECK(code_edit->has_comment_delimiter("\""));
			CHECK(code_edit->get_comment_delimiters().size() == 1);

			ERR_PRINT_OFF;

			/* Adding a duplicate start key is not allowed. */
			code_edit->add_comment_delimiter("\"", "\'", false);
			CHECK(code_edit->get_comment_delimiters().size() == 1);

			/* Adding a duplicate end key is allowed. */
			code_edit->add_comment_delimiter("'", "\"", false);
			CHECK(code_edit->has_comment_delimiter("'"));
			CHECK(code_edit->get_comment_delimiters().size() == 2);

			/* Both start and end keys have to be symbols. */
			code_edit->add_comment_delimiter("f", "\"", false);
			CHECK_FALSE(code_edit->has_comment_delimiter("f"));
			CHECK(code_edit->get_comment_delimiters().size() == 2);

			code_edit->add_comment_delimiter("f", "\"", false);
			CHECK_FALSE(code_edit->has_comment_delimiter("f"));
			CHECK(code_edit->get_comment_delimiters().size() == 2);

			code_edit->add_comment_delimiter("@", "f", false);
			CHECK_FALSE(code_edit->has_comment_delimiter("@"));
			CHECK(code_edit->get_comment_delimiters().size() == 2);

			code_edit->add_comment_delimiter("f", "f", false);
			CHECK_FALSE(code_edit->has_comment_delimiter("f"));
			CHECK(code_edit->get_comment_delimiters().size() == 2);

			/* Blank start keys are not allowed. */
			code_edit->add_comment_delimiter("", "#", false);
			CHECK_FALSE(code_edit->has_comment_delimiter("#"));
			CHECK(code_edit->get_comment_delimiters().size() == 2);

			ERR_PRINT_ON;

			/* Blank end keys are allowed. */
			code_edit->add_comment_delimiter("#", "", false);
			CHECK(code_edit->has_comment_delimiter("#"));
			CHECK(code_edit->get_comment_delimiters().size() == 3);

			/* Remove a delimiter. */
			code_edit->remove_comment_delimiter("#");
			CHECK_FALSE(code_edit->has_comment_delimiter("#"));
			CHECK(code_edit->get_comment_delimiters().size() == 2);

			/* Set should override existing, and test multiline. */
			TypedArray<String> delimiters;
			delimiters.push_back("^^ ^^");

			code_edit->set_comment_delimiters(delimiters);
			CHECK_FALSE(code_edit->has_comment_delimiter("\""));
			CHECK(code_edit->has_comment_delimiter("^^"));
			CHECK(code_edit->get_comment_delimiters().size() == 1);

			/* clear should remove all. */
			code_edit->clear_comment_delimiters();
			CHECK_FALSE(code_edit->has_comment_delimiter("^^"));
			CHECK(code_edit->get_comment_delimiters().size() == 0);
		}

		SUBCASE("[CodeEdit] add and remove mixed delimiters") {
			code_edit->add_comment_delimiter("#", "", false);
			CHECK(code_edit->has_comment_delimiter("#"));
			CHECK(code_edit->get_comment_delimiters().size() == 1);

			ERR_PRINT_OFF;

			/* Disallow adding a string with the same start key as comment. */
			code_edit->add_string_delimiter("#", "", false);
			CHECK_FALSE(code_edit->has_string_delimiter("#"));
			CHECK(code_edit->get_string_delimiters().size() == 0);

			code_edit->add_string_delimiter("\"", "\"", false);
			CHECK(code_edit->has_string_delimiter("\""));
			CHECK(code_edit->get_comment_delimiters().size() == 1);

			/* Disallow adding a comment with the same start key as string. */
			code_edit->add_comment_delimiter("\"", "", false);
			CHECK_FALSE(code_edit->has_comment_delimiter("\""));
			CHECK(code_edit->get_comment_delimiters().size() == 1);

			ERR_PRINT_ON;

			/* Cannot remove string with remove comment. */
			code_edit->remove_comment_delimiter("\"");
			CHECK(code_edit->has_string_delimiter("\""));
			CHECK(code_edit->get_string_delimiters().size() == 1);

			/* Cannot remove comment with remove string. */
			code_edit->remove_string_delimiter("#");
			CHECK(code_edit->has_comment_delimiter("#"));
			CHECK(code_edit->get_comment_delimiters().size() == 1);

			/* Clear comments leave strings. */
			code_edit->clear_comment_delimiters();
			CHECK(code_edit->has_string_delimiter("\""));
			CHECK(code_edit->get_string_delimiters().size() == 1);

			/* Clear string leave comments. */
			code_edit->add_comment_delimiter("#", "", false);
			CHECK(code_edit->has_comment_delimiter("#"));
			CHECK(code_edit->get_comment_delimiters().size() == 1);

			code_edit->clear_string_delimiters();
			CHECK(code_edit->has_comment_delimiter("#"));
			CHECK(code_edit->get_comment_delimiters().size() == 1);
		}
	}

	SUBCASE("[CodeEdit] single line delimiters") {
		SUBCASE("[CodeEdit] single line string delimiters") {
			/* Blank end key should set lineonly to true. */
			code_edit->add_string_delimiter("#", "", false);
			CHECK(code_edit->has_string_delimiter("#"));
			CHECK(code_edit->get_string_delimiters().size() == 1);

			/* Insert line above, line with string then line below. */
			code_edit->insert_text_at_caret(" \n#\n ");

			/* Check line above is not in string. */
			CHECK(code_edit->is_in_string(0, 1) == -1);
			CHECK(code_edit->get_delimiter_start_position(0, 1) == OUTSIDE_DELIMETER);
			CHECK(code_edit->get_delimiter_end_position(0, 1) == OUTSIDE_DELIMETER);

			/* Check column before start key is not in string. */
			CHECK(code_edit->is_in_string(1, 0) == -1);
			CHECK(code_edit->get_delimiter_start_position(1, 0) == OUTSIDE_DELIMETER);
			CHECK(code_edit->get_delimiter_end_position(1, 0) == OUTSIDE_DELIMETER);

			/* Check column after start key is in string and start / end positions are correct. */
			CHECK(code_edit->is_in_string(1, 1) != -1);
			CHECK(code_edit->get_delimiter_start_position(1, 1) == Point2(1, 1));
			CHECK(code_edit->get_delimiter_end_position(1, 1) == Point2(2, 1));

			/* Check line after is not in string. */
			CHECK(code_edit->is_in_string(2, 1) == -1);
			CHECK(code_edit->get_delimiter_start_position(2, 1) == OUTSIDE_DELIMETER);
			CHECK(code_edit->get_delimiter_end_position(2, 1) == OUTSIDE_DELIMETER);

			/* Check region metadata. */
			int idx = code_edit->is_in_string(1, 1);
			CHECK(code_edit->get_delimiter_start_key(idx) == "#");
			CHECK(code_edit->get_delimiter_end_key(idx) == "");

			/* Check nested strings are handled correctly. */
			code_edit->set_text(" \n#  # \n ");

			/* Check line above is not in string. */
			CHECK(code_edit->is_in_string(0, 1) == -1);
			CHECK(code_edit->get_delimiter_start_position(0, 1) == OUTSIDE_DELIMETER);
			CHECK(code_edit->get_delimiter_end_position(0, 1) == OUTSIDE_DELIMETER);

			/* Check column before first start key is not in string. */
			CHECK(code_edit->is_in_string(1, 0) == -1);
			CHECK(code_edit->get_delimiter_start_position(1, 0) == OUTSIDE_DELIMETER);
			CHECK(code_edit->get_delimiter_end_position(1, 0) == OUTSIDE_DELIMETER);

			/* Check column after the first start key is in string and start / end positions are correct. */
			CHECK(code_edit->is_in_string(1, 1) != -1);
			CHECK(code_edit->get_delimiter_start_position(1, 1) == Point2(1, 1));
			CHECK(code_edit->get_delimiter_end_position(1, 1) == Point2(6, 1));

			/* Check column after the second start key returns data for the first. */
			CHECK(code_edit->is_in_string(1, 5) != -1);
			CHECK(code_edit->get_delimiter_start_position(1, 5) == Point2(1, 1));
			CHECK(code_edit->get_delimiter_end_position(1, 5) == Point2(6, 1));

			/* Check line after is not in string. */
			CHECK(code_edit->is_in_string(2, 1) == -1);
			CHECK(code_edit->get_delimiter_start_position(2, 1) == OUTSIDE_DELIMETER);
			CHECK(code_edit->get_delimiter_end_position(2, 1) == OUTSIDE_DELIMETER);

			/* Check is in string with no column returns true if entire line is comment excluding whitespace. */
			code_edit->set_text(" \n  #  # \n ");
			CHECK(code_edit->is_in_string(1) != -1);

			code_edit->set_text(" \n  text #  # \n ");
			CHECK(code_edit->is_in_string(1) == -1);

			/* Removing delimiter should update. */
			code_edit->set_text(" \n  #  # \n ");

			code_edit->remove_string_delimiter("#");
			CHECK_FALSE(code_edit->has_string_delimiter("$"));
			CHECK(code_edit->get_string_delimiters().size() == 0);

			CHECK(code_edit->is_in_string(1) == -1);

			/* Adding and clear should update. */
			code_edit->add_string_delimiter("#", "", false);
			CHECK(code_edit->has_string_delimiter("#"));
			CHECK(code_edit->get_string_delimiters().size() == 1);
			CHECK(code_edit->is_in_string(1) != -1);

			code_edit->clear_string_delimiters();
			CHECK_FALSE(code_edit->has_string_delimiter("$"));
			CHECK(code_edit->get_string_delimiters().size() == 0);

			CHECK(code_edit->is_in_string(1) == -1);
		}

		SUBCASE("[CodeEdit] single line comment delimiters") {
			/* Blank end key should set lineonly to true. */
			code_edit->add_comment_delimiter("#", "", false);
			CHECK(code_edit->has_comment_delimiter("#"));
			CHECK(code_edit->get_comment_delimiters().size() == 1);

			/* Insert line above, line with comment then line below. */
			code_edit->insert_text_at_caret(" \n#\n ");

			/* Check line above is not in comment. */
			CHECK(code_edit->is_in_comment(0, 1) == -1);
			CHECK(code_edit->get_delimiter_start_position(0, 1) == OUTSIDE_DELIMETER);
			CHECK(code_edit->get_delimiter_end_position(0, 1) == OUTSIDE_DELIMETER);

			/* Check column before start key is not in comment. */
			CHECK(code_edit->is_in_comment(1, 0) == -1);
			CHECK(code_edit->get_delimiter_start_position(1, 0) == OUTSIDE_DELIMETER);
			CHECK(code_edit->get_delimiter_end_position(1, 0) == OUTSIDE_DELIMETER);

			/* Check column after start key is in comment and start / end positions are correct. */
			CHECK(code_edit->is_in_comment(1, 1) != -1);
			CHECK(code_edit->get_delimiter_start_position(1, 1) == Point2(1, 1));
			CHECK(code_edit->get_delimiter_end_position(1, 1) == Point2(2, 1));

			/* Check line after is not in comment. */
			CHECK(code_edit->is_in_comment(2, 1) == -1);
			CHECK(code_edit->get_delimiter_start_position(2, 1) == OUTSIDE_DELIMETER);
			CHECK(code_edit->get_delimiter_end_position(2, 1) == OUTSIDE_DELIMETER);

			/* Check region metadata. */
			int idx = code_edit->is_in_comment(1, 1);
			CHECK(code_edit->get_delimiter_start_key(idx) == "#");
			CHECK(code_edit->get_delimiter_end_key(idx) == "");

			/* Check nested comments are handled correctly. */
			code_edit->set_text(" \n#  # \n ");

			/* Check line above is not in comment. */
			CHECK(code_edit->is_in_comment(0, 1) == -1);
			CHECK(code_edit->get_delimiter_start_position(0, 1) == OUTSIDE_DELIMETER);
			CHECK(code_edit->get_delimiter_end_position(0, 1) == OUTSIDE_DELIMETER);

			/* Check column before first start key is not in comment. */
			CHECK(code_edit->is_in_comment(1, 0) == -1);
			CHECK(code_edit->get_delimiter_start_position(1, 0) == OUTSIDE_DELIMETER);
			CHECK(code_edit->get_delimiter_end_position(1, 0) == OUTSIDE_DELIMETER);

			/* Check column after the first start key is in comment and start / end positions are correct. */
			CHECK(code_edit->is_in_comment(1, 1) != -1);
			CHECK(code_edit->get_delimiter_start_position(1, 1) == Point2(1, 1));
			CHECK(code_edit->get_delimiter_end_position(1, 1) == Point2(6, 1));

			/* Check column after the second start key returns data for the first. */
			CHECK(code_edit->is_in_comment(1, 5) != -1);
			CHECK(code_edit->get_delimiter_start_position(1, 5) == Point2(1, 1));
			CHECK(code_edit->get_delimiter_end_position(1, 5) == Point2(6, 1));

			/* Check line after is not in comment. */
			CHECK(code_edit->is_in_comment(2, 1) == -1);
			CHECK(code_edit->get_delimiter_start_position(2, 1) == OUTSIDE_DELIMETER);
			CHECK(code_edit->get_delimiter_end_position(2, 1) == OUTSIDE_DELIMETER);

			/* Check is in comment with no column returns true if entire line is comment excluding whitespace. */
			code_edit->set_text(" \n  #  # \n ");
			CHECK(code_edit->is_in_comment(1) != -1);

			code_edit->set_text(" \n  text #  # \n ");
			CHECK(code_edit->is_in_comment(1) == -1);

			/* Removing delimiter should update. */
			code_edit->set_text(" \n  #  # \n ");

			code_edit->remove_comment_delimiter("#");
			CHECK_FALSE(code_edit->has_comment_delimiter("$"));
			CHECK(code_edit->get_comment_delimiters().size() == 0);

			CHECK(code_edit->is_in_comment(1) == -1);

			/* Adding and clear should update. */
			code_edit->add_comment_delimiter("#", "", false);
			CHECK(code_edit->has_comment_delimiter("#"));
			CHECK(code_edit->get_comment_delimiters().size() == 1);
			CHECK(code_edit->is_in_comment(1) != -1);

			code_edit->clear_comment_delimiters();
			CHECK_FALSE(code_edit->has_comment_delimiter("$"));
			CHECK(code_edit->get_comment_delimiters().size() == 0);

			CHECK(code_edit->is_in_comment(1) == -1);
		}

		SUBCASE("[CodeEdit] single line mixed delimiters") {
			/* Blank end key should set lineonly to true. */
			/* Add string delimiter. */
			code_edit->add_string_delimiter("&", "", false);
			CHECK(code_edit->has_string_delimiter("&"));
			CHECK(code_edit->get_string_delimiters().size() == 1);

			/* Add comment delimiter. */
			code_edit->add_comment_delimiter("#", "", false);
			CHECK(code_edit->has_comment_delimiter("#"));
			CHECK(code_edit->get_comment_delimiters().size() == 1);

			/* Nest a string delimiter inside a comment. */
			code_edit->set_text(" \n#  & \n ");

			/* Check line above is not in comment. */
			CHECK(code_edit->is_in_comment(0, 1) == -1);
			CHECK(code_edit->get_delimiter_start_position(0, 1) == OUTSIDE_DELIMETER);
			CHECK(code_edit->get_delimiter_end_position(0, 1) == OUTSIDE_DELIMETER);

			/* Check column before first start key is not in comment. */
			CHECK(code_edit->is_in_comment(1, 0) == -1);
			CHECK(code_edit->get_delimiter_start_position(1, 0) == OUTSIDE_DELIMETER);
			CHECK(code_edit->get_delimiter_end_position(1, 0) == OUTSIDE_DELIMETER);

			/* Check column after the first start key is in comment and start / end positions are correct. */
			CHECK(code_edit->is_in_comment(1, 1) != -1);
			CHECK(code_edit->get_delimiter_start_position(1, 1) == Point2(1, 1));
			CHECK(code_edit->get_delimiter_end_position(1, 1) == Point2(6, 1));

			/* Check column after the second start key returns data for the first, and does not state string. */
			CHECK(code_edit->is_in_comment(1, 5) != -1);
			CHECK(code_edit->get_delimiter_start_position(1, 5) == Point2(1, 1));
			CHECK(code_edit->get_delimiter_end_position(1, 5) == Point2(6, 1));
			CHECK(code_edit->is_in_string(1, 5) == -1);

			/* Check line after is not in comment. */
			CHECK(code_edit->is_in_comment(2, 1) == -1);
			CHECK(code_edit->get_delimiter_start_position(2, 1) == OUTSIDE_DELIMETER);
			CHECK(code_edit->get_delimiter_end_position(2, 1) == OUTSIDE_DELIMETER);

			/* Remove the comment delimiter. */
			code_edit->remove_comment_delimiter("#");
			CHECK_FALSE(code_edit->has_comment_delimiter("$"));
			CHECK(code_edit->get_comment_delimiters().size() == 0);

			/* The "first" comment region is no longer valid. */
			CHECK(code_edit->is_in_comment(1, 1) == -1);
			CHECK(code_edit->get_delimiter_start_position(1, 1) == OUTSIDE_DELIMETER);
			CHECK(code_edit->get_delimiter_end_position(1, 1) == OUTSIDE_DELIMETER);

			/* The "second" region as string is now valid. */
			CHECK(code_edit->is_in_string(1, 5) != -1);
			CHECK(code_edit->get_delimiter_start_position(1, 5) == Point2(4, 1));
			CHECK(code_edit->get_delimiter_end_position(1, 5) == Point2(6, 1));
		}
	}

	SUBCASE("[CodeEdit] multiline delimiters") {
		SUBCASE("[CodeEdit] multiline string delimiters") {
			code_edit->clear_string_delimiters();
			code_edit->clear_comment_delimiters();

			/* Add string delimiter. */
			code_edit->add_string_delimiter("#", "#", false);
			CHECK(code_edit->has_string_delimiter("#"));
			CHECK(code_edit->get_string_delimiters().size() == 1);

			/* First test over a single line. */
			code_edit->set_text(" \n #  # \n ");

			/* Check line above is not in string. */
			CHECK(code_edit->is_in_string(0, 1) == -1);
			CHECK(code_edit->get_delimiter_start_position(0, 1) == OUTSIDE_DELIMETER);
			CHECK(code_edit->get_delimiter_end_position(0, 1) == OUTSIDE_DELIMETER);

			/* Check column before start key is not in string. */
			CHECK(code_edit->is_in_string(1, 0) == -1);
			CHECK(code_edit->get_delimiter_start_position(1, 0) == OUTSIDE_DELIMETER);
			CHECK(code_edit->get_delimiter_end_position(1, 0) == OUTSIDE_DELIMETER);

			/* Check column before closing delimiter is in string. */
			CHECK(code_edit->is_in_string(1, 2) != -1);
			CHECK(code_edit->get_delimiter_start_position(1, 2) == Point2(2, 1));
			CHECK(code_edit->get_delimiter_end_position(1, 2) == Point2(5, 1));

			/* Check column after end key is not in string. */
			CHECK(code_edit->is_in_string(1, 6) == -1);
			CHECK(code_edit->get_delimiter_start_position(1, 6) == OUTSIDE_DELIMETER);
			CHECK(code_edit->get_delimiter_end_position(1, 6) == OUTSIDE_DELIMETER);

			/* Check line after is not in string. */
			CHECK(code_edit->is_in_string(2, 1) == -1);
			CHECK(code_edit->get_delimiter_start_position(2, 1) == OUTSIDE_DELIMETER);
			CHECK(code_edit->get_delimiter_end_position(2, 1) == OUTSIDE_DELIMETER);

			/* Check the region metadata. */
			int idx = code_edit->is_in_string(1, 2);
			CHECK(code_edit->get_delimiter_start_key(idx) == "#");
			CHECK(code_edit->get_delimiter_end_key(idx) == "#");

			/* Next test over a multiple blank lines. */
			code_edit->set_text(" \n # \n\n # \n ");

			/* Check line above is not in string. */
			CHECK(code_edit->is_in_string(0, 1) == -1);
			CHECK(code_edit->get_delimiter_start_position(0, 1) == OUTSIDE_DELIMETER);
			CHECK(code_edit->get_delimiter_end_position(0, 1) == OUTSIDE_DELIMETER);

			/* Check column before start key is not in string. */
			CHECK(code_edit->is_in_string(1, 0) == -1);
			CHECK(code_edit->get_delimiter_start_position(1, 0) == OUTSIDE_DELIMETER);
			CHECK(code_edit->get_delimiter_end_position(1, 0) == OUTSIDE_DELIMETER);

			/* Check column just after start key is in string. */
			CHECK(code_edit->is_in_string(1, 2) != -1);
			CHECK(code_edit->get_delimiter_start_position(1, 2) == Point2(2, 1));
			CHECK(code_edit->get_delimiter_end_position(1, 2) == Point2(2, 3));

			/* Check blank middle line. */
			CHECK(code_edit->is_in_string(2, 0) != -1);
			CHECK(code_edit->get_delimiter_start_position(2, 0) == Point2(2, 1));
			CHECK(code_edit->get_delimiter_end_position(2, 0) == Point2(2, 3));

			/* Check column just before end key is in string. */
			CHECK(code_edit->is_in_string(3, 0) != -1);
			CHECK(code_edit->get_delimiter_start_position(3, 0) == Point2(2, 1));
			CHECK(code_edit->get_delimiter_end_position(3, 0) == Point2(2, 3));

			/* Check column after end key is not in string. */
			CHECK(code_edit->is_in_string(3, 3) == -1);
			CHECK(code_edit->get_delimiter_start_position(3, 3) == OUTSIDE_DELIMETER);
			CHECK(code_edit->get_delimiter_end_position(3, 3) == OUTSIDE_DELIMETER);

			/* Check line after is not in string. */
			CHECK(code_edit->is_in_string(4, 1) == -1);
			CHECK(code_edit->get_delimiter_start_position(4, 1) == OUTSIDE_DELIMETER);
			CHECK(code_edit->get_delimiter_end_position(4, 1) == OUTSIDE_DELIMETER);

			/* Next test over a multiple non-blank lines. */
			code_edit->set_text(" \n # \n \n # \n ");

			/* Check line above is not in string. */
			CHECK(code_edit->is_in_string(0, 1) == -1);
			CHECK(code_edit->get_delimiter_start_position(0, 1) == OUTSIDE_DELIMETER);
			CHECK(code_edit->get_delimiter_end_position(0, 1) == OUTSIDE_DELIMETER);

			/* Check column before start key is not in string. */
			CHECK(code_edit->is_in_string(1, 0) == -1);
			CHECK(code_edit->get_delimiter_start_position(1, 0) == OUTSIDE_DELIMETER);
			CHECK(code_edit->get_delimiter_end_position(1, 0) == OUTSIDE_DELIMETER);

			/* Check column just after start key is in string. */
			CHECK(code_edit->is_in_string(1, 2) != -1);
			CHECK(code_edit->get_delimiter_start_position(1, 2) == Point2(2, 1));
			CHECK(code_edit->get_delimiter_end_position(1, 2) == Point2(2, 3));

			/* Check middle line. */
			CHECK(code_edit->is_in_string(2, 0) != -1);
			CHECK(code_edit->get_delimiter_start_position(2, 0) == Point2(2, 1));
			CHECK(code_edit->get_delimiter_end_position(2, 0) == Point2(2, 3));

			/* Check column just before end key is in string. */
			CHECK(code_edit->is_in_string(3, 0) != -1);
			CHECK(code_edit->get_delimiter_start_position(3, 0) == Point2(2, 1));
			CHECK(code_edit->get_delimiter_end_position(3, 0) == Point2(2, 3));

			/* Check column after end key is not in string. */
			CHECK(code_edit->is_in_string(3, 3) == -1);
			CHECK(code_edit->get_delimiter_start_position(3, 3) == OUTSIDE_DELIMETER);
			CHECK(code_edit->get_delimiter_end_position(3, 3) == OUTSIDE_DELIMETER);

			/* Check line after is not in string. */
			CHECK(code_edit->is_in_string(4, 1) == -1);
			CHECK(code_edit->get_delimiter_start_position(4, 1) == OUTSIDE_DELIMETER);
			CHECK(code_edit->get_delimiter_end_position(4, 1) == OUTSIDE_DELIMETER);

			/* check the region metadata. */
			idx = code_edit->is_in_string(1, 2);
			CHECK(code_edit->get_delimiter_start_key(idx) == "#");
			CHECK(code_edit->get_delimiter_end_key(idx) == "#");

			/* Next test nested strings. */
			code_edit->add_string_delimiter("^", "^", false);
			CHECK(code_edit->has_string_delimiter("^"));
			CHECK(code_edit->get_string_delimiters().size() == 2);

			code_edit->set_text(" \n # ^\n \n^ # \n ");

			/* Check line above is not in string. */
			CHECK(code_edit->is_in_string(0, 1) == -1);
			CHECK(code_edit->get_delimiter_start_position(0, 1) == OUTSIDE_DELIMETER);
			CHECK(code_edit->get_delimiter_end_position(0, 1) == OUTSIDE_DELIMETER);

			/* Check column before start key is not in string. */
			CHECK(code_edit->is_in_string(1, 0) == -1);
			CHECK(code_edit->get_delimiter_start_position(1, 0) == OUTSIDE_DELIMETER);
			CHECK(code_edit->get_delimiter_end_position(1, 0) == OUTSIDE_DELIMETER);

			/* Check column just after start key is in string. */
			CHECK(code_edit->is_in_string(1, 2) != -1);
			CHECK(code_edit->get_delimiter_start_position(1, 2) == Point2(2, 1));
			CHECK(code_edit->get_delimiter_end_position(1, 2) == Point2(3, 3));

			/* Check middle line. */
			CHECK(code_edit->is_in_string(2, 0) != -1);
			CHECK(code_edit->get_delimiter_start_position(2, 0) == Point2(2, 1));
			CHECK(code_edit->get_delimiter_end_position(2, 0) == Point2(3, 3));

			/* Check column just before end key is in string. */
			CHECK(code_edit->is_in_string(3, 0) != -1);
			CHECK(code_edit->get_delimiter_start_position(3, 0) == Point2(2, 1));
			CHECK(code_edit->get_delimiter_end_position(3, 0) == Point2(3, 3));

			/* Check column after end key is not in string. */
			CHECK(code_edit->is_in_string(3, 3) == -1);
			CHECK(code_edit->get_delimiter_start_position(3, 3) == OUTSIDE_DELIMETER);
			CHECK(code_edit->get_delimiter_end_position(3, 3) == OUTSIDE_DELIMETER);

			/* Check line after is not in string. */
			CHECK(code_edit->is_in_string(4, 1) == -1);
			CHECK(code_edit->get_delimiter_start_position(4, 1) == OUTSIDE_DELIMETER);
			CHECK(code_edit->get_delimiter_end_position(4, 1) == OUTSIDE_DELIMETER);

			/* check the region metadata. */
			idx = code_edit->is_in_string(1, 2);
			CHECK(code_edit->get_delimiter_start_key(idx) == "#");
			CHECK(code_edit->get_delimiter_end_key(idx) == "#");

			/* Next test no end key. */
			code_edit->set_text(" \n # \n ");

			/* check the region metadata. */
			idx = code_edit->is_in_string(1, 2);
			CHECK(code_edit->get_delimiter_start_position(1, 2) == Point2(2, 1));
			CHECK(code_edit->get_delimiter_end_position(1, 2) == Point2(-1, -1));
			CHECK(code_edit->get_delimiter_start_key(idx) == "#");
			CHECK(code_edit->get_delimiter_end_key(idx) == "#");

			/* Check is in string with no column returns true if entire line is string excluding whitespace. */
			code_edit->set_text(" \n # \n\n #\n ");
			CHECK(code_edit->is_in_string(1) != -1);
			CHECK(code_edit->is_in_string(2) != -1);
			CHECK(code_edit->is_in_string(3) != -1);

			code_edit->set_text(" \n test # \n\n # test \n ");
			CHECK(code_edit->is_in_string(1) == -1);
			CHECK(code_edit->is_in_string(2) != -1);
			CHECK(code_edit->is_in_string(3) == -1);
		}

		SUBCASE("[CodeEdit] multiline comment delimiters") {
			/* Add comment delimiter. */
			code_edit->add_comment_delimiter("#", "#", false);
			CHECK(code_edit->has_comment_delimiter("#"));
			CHECK(code_edit->get_comment_delimiters().size() == 1);

			/* First test over a single line. */
			code_edit->set_text(" \n #  # \n ");

			/* Check line above is not in comment. */
			CHECK(code_edit->is_in_comment(0, 1) == -1);
			CHECK(code_edit->get_delimiter_start_position(0, 1) == OUTSIDE_DELIMETER);
			CHECK(code_edit->get_delimiter_end_position(0, 1) == OUTSIDE_DELIMETER);

			/* Check column before start key is not in comment. */
			CHECK(code_edit->is_in_comment(1, 0) == -1);
			CHECK(code_edit->get_delimiter_start_position(1, 0) == OUTSIDE_DELIMETER);
			CHECK(code_edit->get_delimiter_end_position(1, 0) == OUTSIDE_DELIMETER);

			/* Check column before closing delimiter is in comment. */
			CHECK(code_edit->is_in_comment(1, 2) != -1);
			CHECK(code_edit->get_delimiter_start_position(1, 2) == Point2(2, 1));
			CHECK(code_edit->get_delimiter_end_position(1, 2) == Point2(5, 1));

			/* Check column after end key is not in comment. */
			CHECK(code_edit->is_in_comment(1, 6) == -1);
			CHECK(code_edit->get_delimiter_start_position(1, 6) == OUTSIDE_DELIMETER);
			CHECK(code_edit->get_delimiter_end_position(1, 6) == OUTSIDE_DELIMETER);

			/* Check line after is not in comment. */
			CHECK(code_edit->is_in_comment(2, 1) == -1);
			CHECK(code_edit->get_delimiter_start_position(2, 1) == OUTSIDE_DELIMETER);
			CHECK(code_edit->get_delimiter_end_position(2, 1) == OUTSIDE_DELIMETER);

			/* Check the region metadata. */
			int idx = code_edit->is_in_comment(1, 2);
			CHECK(code_edit->get_delimiter_start_key(idx) == "#");
			CHECK(code_edit->get_delimiter_end_key(idx) == "#");

			/* Next test over a multiple blank lines. */
			code_edit->set_text(" \n # \n\n # \n ");

			/* Check line above is not in comment. */
			CHECK(code_edit->is_in_comment(0, 1) == -1);
			CHECK(code_edit->get_delimiter_start_position(0, 1) == OUTSIDE_DELIMETER);
			CHECK(code_edit->get_delimiter_end_position(0, 1) == OUTSIDE_DELIMETER);

			/* Check column before start key is not in comment. */
			CHECK(code_edit->is_in_comment(1, 0) == -1);
			CHECK(code_edit->get_delimiter_start_position(1, 0) == OUTSIDE_DELIMETER);
			CHECK(code_edit->get_delimiter_end_position(1, 0) == OUTSIDE_DELIMETER);

			/* Check column just after start key is in comment. */
			CHECK(code_edit->is_in_comment(1, 2) != -1);
			CHECK(code_edit->get_delimiter_start_position(1, 2) == Point2(2, 1));
			CHECK(code_edit->get_delimiter_end_position(1, 2) == Point2(2, 3));

			/* Check blank middle line. */
			CHECK(code_edit->is_in_comment(2, 0) != -1);
			CHECK(code_edit->get_delimiter_start_position(2, 0) == Point2(2, 1));
			CHECK(code_edit->get_delimiter_end_position(2, 0) == Point2(2, 3));

			/* Check column just before end key is in comment. */
			CHECK(code_edit->is_in_comment(3, 0) != -1);
			CHECK(code_edit->get_delimiter_start_position(3, 0) == Point2(2, 1));
			CHECK(code_edit->get_delimiter_end_position(3, 0) == Point2(2, 3));

			/* Check column after end key is not in comment. */
			CHECK(code_edit->is_in_comment(3, 3) == -1);
			CHECK(code_edit->get_delimiter_start_position(3, 3) == OUTSIDE_DELIMETER);
			CHECK(code_edit->get_delimiter_end_position(3, 3) == OUTSIDE_DELIMETER);

			/* Check line after is not in comment. */
			CHECK(code_edit->is_in_comment(4, 1) == -1);
			CHECK(code_edit->get_delimiter_start_position(4, 1) == OUTSIDE_DELIMETER);
			CHECK(code_edit->get_delimiter_end_position(4, 1) == OUTSIDE_DELIMETER);

			/* Next test over a multiple non-blank lines. */
			code_edit->set_text(" \n # \n \n # \n ");

			/* Check line above is not in comment. */
			CHECK(code_edit->is_in_comment(0, 1) == -1);
			CHECK(code_edit->get_delimiter_start_position(0, 1) == OUTSIDE_DELIMETER);
			CHECK(code_edit->get_delimiter_end_position(0, 1) == OUTSIDE_DELIMETER);

			/* Check column before start key is not in comment. */
			CHECK(code_edit->is_in_comment(1, 0) == -1);
			CHECK(code_edit->get_delimiter_start_position(1, 0) == OUTSIDE_DELIMETER);
			CHECK(code_edit->get_delimiter_end_position(1, 0) == OUTSIDE_DELIMETER);

			/* Check column just after start key is in comment. */
			CHECK(code_edit->is_in_comment(1, 2) != -1);
			CHECK(code_edit->get_delimiter_start_position(1, 2) == Point2(2, 1));
			CHECK(code_edit->get_delimiter_end_position(1, 2) == Point2(2, 3));

			/* Check middle line. */
			CHECK(code_edit->is_in_comment(2, 0) != -1);
			CHECK(code_edit->get_delimiter_start_position(2, 0) == Point2(2, 1));
			CHECK(code_edit->get_delimiter_end_position(2, 0) == Point2(2, 3));

			/* Check column just before end key is in comment. */
			CHECK(code_edit->is_in_comment(3, 0) != -1);
			CHECK(code_edit->get_delimiter_start_position(3, 0) == Point2(2, 1));
			CHECK(code_edit->get_delimiter_end_position(3, 0) == Point2(2, 3));

			/* Check column after end key is not in comment. */
			CHECK(code_edit->is_in_comment(3, 3) == -1);
			CHECK(code_edit->get_delimiter_start_position(3, 3) == OUTSIDE_DELIMETER);
			CHECK(code_edit->get_delimiter_end_position(3, 3) == OUTSIDE_DELIMETER);

			/* Check line after is not in comment. */
			CHECK(code_edit->is_in_comment(4, 1) == -1);
			CHECK(code_edit->get_delimiter_start_position(4, 1) == OUTSIDE_DELIMETER);
			CHECK(code_edit->get_delimiter_end_position(4, 1) == OUTSIDE_DELIMETER);

			/* check the region metadata. */
			idx = code_edit->is_in_comment(1, 2);
			CHECK(code_edit->get_delimiter_start_key(idx) == "#");
			CHECK(code_edit->get_delimiter_end_key(idx) == "#");

			/* Next test nested comments. */
			code_edit->add_comment_delimiter("^", "^", false);
			CHECK(code_edit->has_comment_delimiter("^"));
			CHECK(code_edit->get_comment_delimiters().size() == 2);

			code_edit->set_text(" \n # ^\n \n^ # \n ");

			/* Check line above is not in comment. */
			CHECK(code_edit->is_in_comment(0, 1) == -1);
			CHECK(code_edit->get_delimiter_start_position(0, 1) == OUTSIDE_DELIMETER);
			CHECK(code_edit->get_delimiter_end_position(0, 1) == OUTSIDE_DELIMETER);

			/* Check column before start key is not in comment. */
			CHECK(code_edit->is_in_comment(1, 0) == -1);
			CHECK(code_edit->get_delimiter_start_position(1, 0) == OUTSIDE_DELIMETER);
			CHECK(code_edit->get_delimiter_end_position(1, 0) == OUTSIDE_DELIMETER);

			/* Check column just after start key is in comment. */
			CHECK(code_edit->is_in_comment(1, 2) != -1);
			CHECK(code_edit->get_delimiter_start_position(1, 2) == Point2(2, 1));
			CHECK(code_edit->get_delimiter_end_position(1, 2) == Point2(3, 3));

			/* Check middle line. */
			CHECK(code_edit->is_in_comment(2, 0) != -1);
			CHECK(code_edit->get_delimiter_start_position(2, 0) == Point2(2, 1));
			CHECK(code_edit->get_delimiter_end_position(2, 0) == Point2(3, 3));

			/* Check column just before end key is in comment. */
			CHECK(code_edit->is_in_comment(3, 0) != -1);
			CHECK(code_edit->get_delimiter_start_position(3, 0) == Point2(2, 1));
			CHECK(code_edit->get_delimiter_end_position(3, 0) == Point2(3, 3));

			/* Check column after end key is not in comment. */
			CHECK(code_edit->is_in_comment(3, 3) == -1);
			CHECK(code_edit->get_delimiter_start_position(3, 3) == OUTSIDE_DELIMETER);
			CHECK(code_edit->get_delimiter_end_position(3, 3) == OUTSIDE_DELIMETER);

			/* Check line after is not in comment. */
			CHECK(code_edit->is_in_comment(4, 1) == -1);
			CHECK(code_edit->get_delimiter_start_position(4, 1) == OUTSIDE_DELIMETER);
			CHECK(code_edit->get_delimiter_end_position(4, 1) == OUTSIDE_DELIMETER);

			/* check the region metadata. */
			idx = code_edit->is_in_comment(1, 2);
			CHECK(code_edit->get_delimiter_start_key(idx) == "#");
			CHECK(code_edit->get_delimiter_end_key(idx) == "#");

			/* Next test no end key. */
			code_edit->set_text(" \n # \n ");

			/* check the region metadata. */
			idx = code_edit->is_in_comment(1, 2);
			CHECK(code_edit->get_delimiter_start_position(1, 2) == Point2(2, 1));
			CHECK(code_edit->get_delimiter_end_position(1, 2) == Point2(-1, -1));
			CHECK(code_edit->get_delimiter_start_key(idx) == "#");
			CHECK(code_edit->get_delimiter_end_key(idx) == "#");

			/* Check is in comment with no column returns true if entire line is comment excluding whitespace. */
			code_edit->set_text(" \n # \n\n #\n ");
			CHECK(code_edit->is_in_comment(1) != -1);
			CHECK(code_edit->is_in_comment(2) != -1);
			CHECK(code_edit->is_in_comment(3) != -1);

			code_edit->set_text(" \n test # \n\n # test \n ");
			CHECK(code_edit->is_in_comment(1) == -1);
			CHECK(code_edit->is_in_comment(2) != -1);
			CHECK(code_edit->is_in_comment(3) == -1);
		}

		SUBCASE("[CodeEdit] multiline mixed delimiters") {
			/* Add comment delimiter. */
			code_edit->add_comment_delimiter("#", "#", false);
			CHECK(code_edit->has_comment_delimiter("#"));
			CHECK(code_edit->get_comment_delimiters().size() == 1);

			/* Add string delimiter. */
			code_edit->add_string_delimiter("^", "^", false);
			CHECK(code_edit->has_string_delimiter("^"));
			CHECK(code_edit->get_string_delimiters().size() == 1);

			/* Nest a string inside a comment. */
			code_edit->set_text(" \n # ^\n \n^ # \n ");

			/* Check line above is not in comment. */
			CHECK(code_edit->is_in_comment(0, 1) == -1);
			CHECK(code_edit->get_delimiter_start_position(0, 1) == OUTSIDE_DELIMETER);
			CHECK(code_edit->get_delimiter_end_position(0, 1) == OUTSIDE_DELIMETER);

			/* Check column before start key is not in comment. */
			CHECK(code_edit->is_in_comment(1, 0) == -1);
			CHECK(code_edit->get_delimiter_start_position(1, 0) == OUTSIDE_DELIMETER);
			CHECK(code_edit->get_delimiter_end_position(1, 0) == OUTSIDE_DELIMETER);

			/* Check column just after start key is in comment. */
			CHECK(code_edit->is_in_comment(1, 2) != -1);
			CHECK(code_edit->get_delimiter_start_position(1, 2) == Point2(2, 1));
			CHECK(code_edit->get_delimiter_end_position(1, 2) == Point2(3, 3));

			/* Check middle line. */
			CHECK(code_edit->is_in_comment(2, 0) != -1);
			CHECK(code_edit->get_delimiter_start_position(2, 0) == Point2(2, 1));
			CHECK(code_edit->get_delimiter_end_position(2, 0) == Point2(3, 3));

			/* Check column just before end key is in comment. */
			CHECK(code_edit->is_in_comment(3, 0) != -1);
			CHECK(code_edit->get_delimiter_start_position(3, 0) == Point2(2, 1));
			CHECK(code_edit->get_delimiter_end_position(3, 0) == Point2(3, 3));

			/* Check column after end key is not in comment. */
			CHECK(code_edit->is_in_comment(3, 3) == -1);
			CHECK(code_edit->get_delimiter_start_position(3, 3) == OUTSIDE_DELIMETER);
			CHECK(code_edit->get_delimiter_end_position(3, 3) == OUTSIDE_DELIMETER);

			/* Check line after is not in comment. */
			CHECK(code_edit->is_in_comment(4, 1) == -1);
			CHECK(code_edit->get_delimiter_start_position(4, 1) == OUTSIDE_DELIMETER);
			CHECK(code_edit->get_delimiter_end_position(4, 1) == OUTSIDE_DELIMETER);

			/* check the region metadata. */
			int idx = code_edit->is_in_comment(1, 2);
			CHECK(code_edit->get_delimiter_start_key(idx) == "#");
			CHECK(code_edit->get_delimiter_end_key(idx) == "#");

			/* Check is in comment with no column returns true as inner delimiter should not be counted. */
			CHECK(code_edit->is_in_comment(1) != -1);
			CHECK(code_edit->is_in_comment(2) != -1);
			CHECK(code_edit->is_in_comment(3) != -1);
		}
	}

	memdelete(code_edit);
}

TEST_CASE("[SceneTree][CodeEdit] indent") {
	CodeEdit *code_edit = memnew(CodeEdit);
	SceneTree::get_singleton()->get_root()->add_child(code_edit);
	code_edit->grab_focus();

	SUBCASE("[CodeEdit] indent settings") {
		code_edit->set_indent_size(10);
		CHECK(code_edit->get_indent_size() == 10);
		CHECK(code_edit->get_tab_size() == 10);

		code_edit->set_auto_indent_enabled(false);
		CHECK_FALSE(code_edit->is_auto_indent_enabled());

		code_edit->set_auto_indent_enabled(true);
		CHECK(code_edit->is_auto_indent_enabled());

		code_edit->set_indent_using_spaces(false);
		CHECK_FALSE(code_edit->is_indent_using_spaces());

		code_edit->set_indent_using_spaces(true);
		CHECK(code_edit->is_indent_using_spaces());

		/* Only the first char is registered. */
		TypedArray<String> auto_indent_prefixes;
		auto_indent_prefixes.push_back("::");
		auto_indent_prefixes.push_back("s");
		auto_indent_prefixes.push_back("1");
		code_edit->set_auto_indent_prefixes(auto_indent_prefixes);

		auto_indent_prefixes = code_edit->get_auto_indent_prefixes();
		CHECK(auto_indent_prefixes.has(":"));
		CHECK(auto_indent_prefixes.has("s"));
		CHECK(auto_indent_prefixes.has("1"));
	}

	SUBCASE("[CodeEdit] indent tabs") {
		code_edit->set_indent_size(4);
		code_edit->set_auto_indent_enabled(true);
		code_edit->set_indent_using_spaces(false);

		/* Do nothing if not editable. */
		code_edit->set_editable(false);

		code_edit->do_indent();
		CHECK(code_edit->get_line(0).is_empty());

		code_edit->indent_lines();
		CHECK(code_edit->get_line(0).is_empty());

		code_edit->set_editable(true);

		/* Simple indent. */
		code_edit->do_indent();
		CHECK(code_edit->get_line(0) == "\t");

		/* Check input action. */
		SEND_GUI_ACTION("ui_text_indent");
		CHECK(code_edit->get_line(0) == "\t\t");

		/* Insert in place. */
		code_edit->set_text("");
		code_edit->insert_text_at_caret("test");
		code_edit->do_indent();
		CHECK(code_edit->get_line(0) == "test\t");

		/* Indent lines does entire line and works without selection. */
		code_edit->set_text("");
		code_edit->insert_text_at_caret("test");
		code_edit->indent_lines();
		CHECK(code_edit->get_line(0) == "\ttest");

		/* Selection does entire line. */
		code_edit->set_text("test");
		code_edit->select_all();
		code_edit->do_indent();
		CHECK(code_edit->get_line(0) == "\ttest");

		/* Handles multiple lines. */
		code_edit->set_text("test\ntext");
		code_edit->select_all();
		code_edit->do_indent();
		CHECK(code_edit->get_line(0) == "\ttest");
		CHECK(code_edit->get_line(1) == "\ttext");

		/* Do not indent line if last col is zero. */
		code_edit->set_text("test\ntext");
		code_edit->select(0, 0, 1, 0);
		code_edit->do_indent();
		CHECK(code_edit->get_line(0) == "\ttest");
		CHECK(code_edit->get_line(1) == "text");

		/* Indent even if last column of first line. */
		code_edit->set_text("test\ntext");
		code_edit->select(0, 4, 1, 0);
		code_edit->do_indent();
		CHECK(code_edit->get_line(0) == "\ttest");
		CHECK(code_edit->get_line(1) == "text");

		/* Check selection is adjusted. */
		code_edit->set_text("test");
		code_edit->select(0, 1, 0, 2);
		code_edit->do_indent();
		CHECK(code_edit->get_selection_from_column() == 2);
		CHECK(code_edit->get_selection_to_column() == 3);
		CHECK(code_edit->get_line(0) == "\ttest");
		code_edit->undo();
	}

	SUBCASE("[CodeEdit] indent spaces") {
		code_edit->set_indent_size(4);
		code_edit->set_auto_indent_enabled(true);
		code_edit->set_indent_using_spaces(true);

		/* Do nothing if not editable. */
		code_edit->set_editable(false);

		code_edit->do_indent();
		CHECK(code_edit->get_line(0).is_empty());

		code_edit->indent_lines();
		CHECK(code_edit->get_line(0).is_empty());

		code_edit->set_editable(true);

		/* Simple indent. */
		code_edit->do_indent();
		CHECK(code_edit->get_line(0) == "    ");

		/* Check input action. */
		SEND_GUI_ACTION("ui_text_indent");
		CHECK(code_edit->get_line(0) == "        ");

		/* Insert in place. */
		code_edit->set_text("");
		code_edit->insert_text_at_caret("test");
		code_edit->do_indent();
		CHECK(code_edit->get_line(0) == "test    ");

		/* Indent lines does entire line and works without selection. */
		code_edit->set_text("");
		code_edit->insert_text_at_caret("test");
		code_edit->indent_lines();
		CHECK(code_edit->get_line(0) == "    test");

		/* Selection does entire line. */
		code_edit->set_text("test");
		code_edit->select_all();
		code_edit->do_indent();
		CHECK(code_edit->get_line(0) == "    test");

		/* single indent only add required spaces. */
		code_edit->set_text(" test");
		code_edit->select_all();
		code_edit->do_indent();
		CHECK(code_edit->get_line(0) == "    test");

		/* Handles multiple lines. */
		code_edit->set_text("test\ntext");
		code_edit->select_all();
		code_edit->do_indent();
		CHECK(code_edit->get_line(0) == "    test");
		CHECK(code_edit->get_line(1) == "    text");

		/* Do not indent line if last col is zero. */
		code_edit->set_text("test\ntext");
		code_edit->select(0, 0, 1, 0);
		code_edit->do_indent();
		CHECK(code_edit->get_line(0) == "    test");
		CHECK(code_edit->get_line(1) == "text");

		/* Indent even if last column of first line. */
		code_edit->set_text("test\ntext");
		code_edit->select(0, 4, 1, 0);
		code_edit->do_indent();
		CHECK(code_edit->get_line(0) == "    test");
		CHECK(code_edit->get_line(1) == "text");

		/* Check selection is adjusted. */
		code_edit->set_text("test");
		code_edit->select(0, 1, 0, 2);
		code_edit->do_indent();
		CHECK(code_edit->get_selection_from_column() == 5);
		CHECK(code_edit->get_selection_to_column() == 6);
		CHECK(code_edit->get_line(0) == "    test");
	}

	SUBCASE("[CodeEdit] unindent tabs") {
		code_edit->set_indent_size(4);
		code_edit->set_auto_indent_enabled(true);
		code_edit->set_indent_using_spaces(false);

		/* Do nothing if not editable. */
		code_edit->set_text("\t");

		code_edit->set_editable(false);

		code_edit->unindent_lines();
		CHECK(code_edit->get_line(0) == "\t");

		code_edit->unindent_lines();
		CHECK(code_edit->get_line(0) == "\t");

		code_edit->set_editable(true);

		/* Simple unindent. */
		code_edit->unindent_lines();
		CHECK(code_edit->get_line(0) == "");

		/* Backspace does a simple unindent. */
		code_edit->set_text("");
		code_edit->insert_text_at_caret("\t");
		code_edit->backspace();
		CHECK(code_edit->get_line(0) == "");

		/* Unindent lines does entire line and works without selection. */
		code_edit->set_text("");
		code_edit->insert_text_at_caret("\ttest");
		code_edit->unindent_lines();
		CHECK(code_edit->get_line(0) == "test");

		/* Caret on col zero unindent line. */
		code_edit->set_text("\t\ttest");
		code_edit->unindent_lines();
		CHECK(code_edit->get_line(0) == "\ttest");

		/* Check input action. */
		code_edit->set_text("\t\ttest");
		SEND_GUI_ACTION("ui_text_dedent");
		CHECK(code_edit->get_line(0) == "\ttest");

		/* Selection does entire line. */
		code_edit->set_text("\t\ttest");
		code_edit->select_all();
		code_edit->unindent_lines();
		CHECK(code_edit->get_line(0) == "\ttest");

		/* Handles multiple lines. */
		code_edit->set_text("\ttest\n\ttext");
		code_edit->select_all();
		code_edit->unindent_lines();
		CHECK(code_edit->get_line(0) == "test");
		CHECK(code_edit->get_line(1) == "text");

		/* Do not unindent line if last col is zero. */
		code_edit->set_text("\ttest\n\ttext");
		code_edit->select(0, 0, 1, 0);
		code_edit->unindent_lines();
		CHECK(code_edit->get_line(0) == "test");
		CHECK(code_edit->get_line(1) == "\ttext");

		/* Unindent even if last column of first line. */
		code_edit->set_text("\ttest\n\ttext");
		code_edit->select(0, 5, 1, 1);
		code_edit->unindent_lines();
		CHECK(code_edit->get_line(0) == "test");
		CHECK(code_edit->get_line(1) == "text");

		/* Check selection is adjusted. */
		code_edit->set_text("\ttest");
		code_edit->select(0, 1, 0, 2);
		code_edit->unindent_lines();
		CHECK(code_edit->get_selection_from_column() == 0);
		CHECK(code_edit->get_selection_to_column() == 1);
		CHECK(code_edit->get_line(0) == "test");
	}

	SUBCASE("[CodeEdit] unindent spaces") {
		code_edit->set_indent_size(4);
		code_edit->set_auto_indent_enabled(true);
		code_edit->set_indent_using_spaces(true);

		/* Do nothing if not editable. */
		code_edit->set_text("    ");

		code_edit->set_editable(false);

		code_edit->unindent_lines();
		CHECK(code_edit->get_line(0) == "    ");

		code_edit->unindent_lines();
		CHECK(code_edit->get_line(0) == "    ");

		code_edit->set_editable(true);

		/* Simple unindent. */
		code_edit->unindent_lines();
		CHECK(code_edit->get_line(0) == "");

		/* Backspace does a simple unindent. */
		code_edit->set_text("");
		code_edit->insert_text_at_caret("    ");
		code_edit->backspace();
		CHECK(code_edit->get_line(0) == "");

		/* Backspace with letter. */
		code_edit->set_text("");
		code_edit->insert_text_at_caret("    a");
		code_edit->backspace();
		CHECK(code_edit->get_line(0) == "    ");

		/* Unindent lines does entire line and works without selection. */
		code_edit->set_text("");
		code_edit->insert_text_at_caret("    test");
		code_edit->unindent_lines();
		CHECK(code_edit->get_line(0) == "test");

		/* Caret on col zero unindent line. */
		code_edit->set_text("        test");
		code_edit->unindent_lines();
		CHECK(code_edit->get_line(0) == "    test");

		/* Only as far as needed */
		code_edit->set_text("       test");
		code_edit->unindent_lines();
		CHECK(code_edit->get_line(0) == "    test");

		/* Check input action. */
		code_edit->set_text("        test");
		SEND_GUI_ACTION("ui_text_dedent");
		CHECK(code_edit->get_line(0) == "    test");

		/* Selection does entire line. */
		code_edit->set_text("        test");
		code_edit->select_all();
		code_edit->unindent_lines();
		CHECK(code_edit->get_line(0) == "    test");

		/* Handles multiple lines. */
		code_edit->set_text("    test\n    text");
		code_edit->select_all();
		code_edit->unindent_lines();
		CHECK(code_edit->get_line(0) == "test");
		CHECK(code_edit->get_line(1) == "text");

		/* Do not unindent line if last col is zero. */
		code_edit->set_text("    test\n    text");
		code_edit->select(0, 0, 1, 0);
		code_edit->unindent_lines();
		CHECK(code_edit->get_line(0) == "test");
		CHECK(code_edit->get_line(1) == "    text");

		/* Unindent even if last column of first line. */
		code_edit->set_text("    test\n    text");
		code_edit->select(0, 5, 1, 1);
		code_edit->unindent_lines();
		CHECK(code_edit->get_line(0) == "test");
		CHECK(code_edit->get_line(1) == "text");

		/* Check selection is adjusted. */
		code_edit->set_text("    test");
		code_edit->select(0, 4, 0, 5);
		code_edit->unindent_lines();
		CHECK(code_edit->get_selection_from_column() == 0);
		CHECK(code_edit->get_selection_to_column() == 1);
		CHECK(code_edit->get_line(0) == "test");
	}

	SUBCASE("[CodeEdit] auto indent") {
		SUBCASE("[CodeEdit] auto indent tabs") {
			code_edit->set_indent_size(4);
			code_edit->set_auto_indent_enabled(true);
			code_edit->set_indent_using_spaces(false);

			/* Simple indent on new line. */
			code_edit->set_text("");
			code_edit->insert_text_at_caret("test:");
			SEND_GUI_ACTION("ui_text_newline");
			CHECK(code_edit->get_line(0) == "test:");
			CHECK(code_edit->get_line(1) == "\t");

			/* new blank line should still indent. */
			code_edit->set_text("");
			code_edit->insert_text_at_caret("test:");
			SEND_GUI_ACTION("ui_text_newline_blank");
			CHECK(code_edit->get_line(0) == "test:");
			CHECK(code_edit->get_line(1) == "\t");

			/* new line above should not indent. */
			code_edit->set_text("");
			code_edit->insert_text_at_caret("test:");
			SEND_GUI_ACTION("ui_text_newline_above");
			CHECK(code_edit->get_line(0) == "");
			CHECK(code_edit->get_line(1) == "test:");

			/* Whitespace between symbol and caret is okay. */
			code_edit->set_text("");
			code_edit->insert_text_at_caret("test:  ");
			SEND_GUI_ACTION("ui_text_newline");
			CHECK(code_edit->get_line(0) == "test:  ");
			CHECK(code_edit->get_line(1) == "\t");

			/* Comment between symbol and caret is okay. */
			code_edit->add_comment_delimiter("#", "");
			code_edit->set_text("");
			code_edit->insert_text_at_caret("test: # comment");
			SEND_GUI_ACTION("ui_text_newline");
			CHECK(code_edit->get_line(0) == "test: # comment");
			CHECK(code_edit->get_line(1) == "\t");
			code_edit->remove_comment_delimiter("#");

			/* Strings between symbol and caret are not okay. */
			code_edit->add_string_delimiter("#", "");
			code_edit->set_text("");
			code_edit->insert_text_at_caret("test: # string");
			SEND_GUI_ACTION("ui_text_newline");
			CHECK(code_edit->get_line(0) == "test: # string");
			CHECK(code_edit->get_line(1) == "");
			code_edit->remove_string_delimiter("#");

			/* Non-whitespace prevents auto-indentation. */
			code_edit->add_comment_delimiter("#", "");
			code_edit->set_text("");
			code_edit->insert_text_at_caret("test := 0 # comment");
			SEND_GUI_ACTION("ui_text_newline");
			CHECK(code_edit->get_line(0) == "test := 0 # comment");
			CHECK(code_edit->get_line(1) == "");
			code_edit->remove_comment_delimiter("#");

			/* Even when there's no comments. */
			code_edit->set_text("");
			code_edit->insert_text_at_caret("test := 0");
			SEND_GUI_ACTION("ui_text_newline");
			CHECK(code_edit->get_line(0) == "test := 0");
			CHECK(code_edit->get_line(1) == "");

			/* If between brace pairs an extra line is added. */
			code_edit->set_text("");
			code_edit->insert_text_at_caret("test{}");
			code_edit->set_caret_column(5);
			SEND_GUI_ACTION("ui_text_newline");
			CHECK(code_edit->get_line(0) == "test{");
			CHECK(code_edit->get_line(1) == "\t");
			CHECK(code_edit->get_line(2) == "}");

			/* Except when we are going above. */
			code_edit->set_text("");
			code_edit->insert_text_at_caret("test{}");
			code_edit->set_caret_column(5);
			SEND_GUI_ACTION("ui_text_newline_above");
			CHECK(code_edit->get_line(0) == "");
			CHECK(code_edit->get_line(1) == "test{}");

			/* or below. */
			code_edit->set_text("");
			code_edit->insert_text_at_caret("test{}");
			code_edit->set_caret_column(5);
			SEND_GUI_ACTION("ui_text_newline_blank");
			CHECK(code_edit->get_line(0) == "test{}");
			CHECK(code_edit->get_line(1) == "");
		}

		SUBCASE("[CodeEdit] auto indent spaces") {
			code_edit->set_indent_size(4);
			code_edit->set_auto_indent_enabled(true);
			code_edit->set_indent_using_spaces(true);

			/* Simple indent on new line. */
			code_edit->set_text("");
			code_edit->insert_text_at_caret("test:");
			SEND_GUI_ACTION("ui_text_newline");
			CHECK(code_edit->get_line(0) == "test:");
			CHECK(code_edit->get_line(1) == "    ");

			/* new blank line should still indent. */
			code_edit->set_text("");
			code_edit->insert_text_at_caret("test:");
			SEND_GUI_ACTION("ui_text_newline_blank");
			CHECK(code_edit->get_line(0) == "test:");
			CHECK(code_edit->get_line(1) == "    ");

			/* new line above should not indent. */
			code_edit->set_text("");
			code_edit->insert_text_at_caret("test:");
			SEND_GUI_ACTION("ui_text_newline_above");
			CHECK(code_edit->get_line(0) == "");
			CHECK(code_edit->get_line(1) == "test:");

			/* Whitespace between symbol and caret is okay. */
			code_edit->set_text("");
			code_edit->insert_text_at_caret("test:  ");
			SEND_GUI_ACTION("ui_text_newline");
			CHECK(code_edit->get_line(0) == "test:  ");
			CHECK(code_edit->get_line(1) == "    ");

			/* Comment between symbol and caret is okay. */
			code_edit->add_comment_delimiter("#", "");
			code_edit->set_text("");
			code_edit->insert_text_at_caret("test: # comment");
			SEND_GUI_ACTION("ui_text_newline");
			CHECK(code_edit->get_line(0) == "test: # comment");
			CHECK(code_edit->get_line(1) == "    ");
			code_edit->remove_comment_delimiter("#");

			/* Strings between symbol and caret are not okay. */
			code_edit->add_string_delimiter("#", "");
			code_edit->set_text("");
			code_edit->insert_text_at_caret("test: # string");
			SEND_GUI_ACTION("ui_text_newline");
			CHECK(code_edit->get_line(0) == "test: # string");
			CHECK(code_edit->get_line(1) == "");
			code_edit->remove_string_delimiter("#");

			/* Non-whitespace prevents auto-indentation. */
			code_edit->add_comment_delimiter("#", "");
			code_edit->set_text("");
			code_edit->insert_text_at_caret("test := 0 # comment");
			SEND_GUI_ACTION("ui_text_newline");
			CHECK(code_edit->get_line(0) == "test := 0 # comment");
			CHECK(code_edit->get_line(1) == "");
			code_edit->remove_comment_delimiter("#");

			/* Even when there's no comments. */
			code_edit->set_text("");
			code_edit->insert_text_at_caret("test := 0");
			SEND_GUI_ACTION("ui_text_newline");
			CHECK(code_edit->get_line(0) == "test := 0");
			CHECK(code_edit->get_line(1) == "");

			/* If between brace pairs an extra line is added. */
			code_edit->set_text("");
			code_edit->insert_text_at_caret("test{}");
			code_edit->set_caret_column(5);
			SEND_GUI_ACTION("ui_text_newline");
			CHECK(code_edit->get_line(0) == "test{");
			CHECK(code_edit->get_line(1) == "    ");
			CHECK(code_edit->get_line(2) == "}");

			/* Except when we are going above. */
			code_edit->set_text("");
			code_edit->insert_text_at_caret("test{}");
			code_edit->set_caret_column(5);
			SEND_GUI_ACTION("ui_text_newline_above");
			CHECK(code_edit->get_line(0) == "");
			CHECK(code_edit->get_line(1) == "test{}");

			/* or below. */
			code_edit->set_text("");
			code_edit->insert_text_at_caret("test{}");
			code_edit->set_caret_column(5);
			SEND_GUI_ACTION("ui_text_newline_blank");
			CHECK(code_edit->get_line(0) == "test{}");
			CHECK(code_edit->get_line(1) == "");

			/* If there is something after a colon
			and there is a colon in the comment it
			should not indent. */
			code_edit->add_comment_delimiter("#", "");
			code_edit->set_text("");
			code_edit->insert_text_at_caret("test:test#:");
			SEND_GUI_ACTION("ui_text_newline");
			CHECK(code_edit->get_line(0) == "test:test#:");
			CHECK(code_edit->get_line(1) == "");
			code_edit->remove_comment_delimiter("#");
		}
	}

	SUBCASE("[CodeEdit] convert indent to tabs") {
		code_edit->set_indent_size(4);
		code_edit->set_indent_using_spaces(false);

		// Only line.
		code_edit->insert_text_at_caret("        test");
		code_edit->set_caret_line(0);
		code_edit->set_caret_column(8);
		code_edit->select(0, 8, 0, 9);
		code_edit->convert_indent();
		CHECK(code_edit->get_line(0) == "\t\ttest");
		CHECK(code_edit->get_caret_column() == 2);
		CHECK(code_edit->get_selection_from_column() == 2);
		CHECK(code_edit->get_selection_to_column() == 3);

		// First line.
		code_edit->set_text("");
		code_edit->insert_text_at_caret("        test\n");
		code_edit->set_caret_line(0);
		code_edit->set_caret_column(8);
		code_edit->select(0, 8, 0, 9);
		code_edit->convert_indent();
		CHECK(code_edit->get_line(0) == "\t\ttest");
		CHECK(code_edit->get_caret_column() == 2);
		CHECK(code_edit->get_selection_from_column() == 2);
		CHECK(code_edit->get_selection_to_column() == 3);

		// Middle line.
		code_edit->set_text("");
		code_edit->insert_text_at_caret("\n        test\n");
		code_edit->set_caret_line(1);
		code_edit->set_caret_column(8);
		code_edit->select(1, 8, 1, 9);
		code_edit->convert_indent();
		CHECK(code_edit->get_line(1) == "\t\ttest");
		CHECK(code_edit->get_caret_column() == 2);
		CHECK(code_edit->get_selection_from_column() == 2);
		CHECK(code_edit->get_selection_to_column() == 3);

		// End line.
		code_edit->set_text("");
		code_edit->insert_text_at_caret("\n        test");
		code_edit->set_caret_line(1);
		code_edit->set_caret_column(8);
		code_edit->select(1, 8, 1, 9);
		code_edit->convert_indent();
		CHECK(code_edit->get_line(1) == "\t\ttest");
		CHECK(code_edit->get_caret_column() == 2);
		CHECK(code_edit->get_selection_from_column() == 2);
		CHECK(code_edit->get_selection_to_column() == 3);

		// Within provided range.
		code_edit->set_text("");
		code_edit->insert_text_at_caret("    test\n        test\n");
		code_edit->set_caret_line(1);
		code_edit->set_caret_column(8);
		code_edit->select(1, 8, 1, 9);
		code_edit->convert_indent(1, 1);
		CHECK(code_edit->get_line(0) == "    test");
		CHECK(code_edit->get_line(1) == "\t\ttest");
		CHECK(code_edit->get_caret_column() == 2);
		CHECK(code_edit->get_selection_from_column() == 2);
		CHECK(code_edit->get_selection_to_column() == 3);
	}

	SUBCASE("[CodeEdit] convert indent to spaces") {
		code_edit->set_indent_size(4);
		code_edit->set_indent_using_spaces(true);

		// Only line.
		code_edit->insert_text_at_caret("\t\ttest");
		code_edit->set_caret_line(0);
		code_edit->set_caret_column(2);
		code_edit->select(0, 2, 0, 3);
		code_edit->convert_indent();
		CHECK(code_edit->get_line(0) == "        test");
		CHECK(code_edit->get_caret_column() == 8);
		CHECK(code_edit->get_selection_from_column() == 8);
		CHECK(code_edit->get_selection_to_column() == 9);

		// First line.
		code_edit->set_text("");
		code_edit->insert_text_at_caret("\t\ttest\n");
		code_edit->set_caret_line(0);
		code_edit->set_caret_column(2);
		code_edit->select(0, 2, 0, 3);
		code_edit->convert_indent();
		CHECK(code_edit->get_line(0) == "        test");
		CHECK(code_edit->get_caret_column() == 8);
		CHECK(code_edit->get_selection_from_column() == 8);
		CHECK(code_edit->get_selection_to_column() == 9);

		// Middle line.
		code_edit->set_text("");
		code_edit->insert_text_at_caret("\n\t\ttest\n");
		code_edit->set_caret_line(1);
		code_edit->set_caret_column(2);
		code_edit->select(1, 2, 1, 3);
		code_edit->convert_indent();
		CHECK(code_edit->get_line(1) == "        test");
		CHECK(code_edit->get_caret_column() == 8);
		CHECK(code_edit->get_selection_from_column() == 8);
		CHECK(code_edit->get_selection_to_column() == 9);

		// End line.
		code_edit->set_text("");
		code_edit->insert_text_at_caret("\n\t\ttest");
		code_edit->set_caret_line(1);
		code_edit->set_caret_column(2);
		code_edit->select(1, 2, 1, 3);
		code_edit->convert_indent();
		CHECK(code_edit->get_line(1) == "        test");
		CHECK(code_edit->get_caret_column() == 8);
		CHECK(code_edit->get_selection_from_column() == 8);
		CHECK(code_edit->get_selection_to_column() == 9);

		// Within provided range.
		code_edit->set_text("");
		code_edit->insert_text_at_caret("\ttest\n\t\ttest\n");
		code_edit->set_caret_line(1);
		code_edit->set_caret_column(2);
		code_edit->select(1, 2, 1, 3);
		code_edit->convert_indent(1, 1);
		CHECK(code_edit->get_line(0) == "\ttest");
		CHECK(code_edit->get_line(1) == "        test");
		CHECK(code_edit->get_caret_column() == 8);
		CHECK(code_edit->get_selection_from_column() == 8);
		CHECK(code_edit->get_selection_to_column() == 9);

		// Outside of range.
		ERR_PRINT_OFF;
		code_edit->convert_indent(0, 4);
		code_edit->convert_indent(4, 5);
		code_edit->convert_indent(4, 1);
		ERR_PRINT_ON;
	}

	memdelete(code_edit);
}

TEST_CASE("[SceneTree][CodeEdit] folding") {
	CodeEdit *code_edit = memnew(CodeEdit);
	SceneTree::get_singleton()->get_root()->add_child(code_edit);
	code_edit->grab_focus();

	SUBCASE("[CodeEdit] folding settings") {
		code_edit->set_line_folding_enabled(true);
		CHECK(code_edit->is_line_folding_enabled());

		code_edit->set_line_folding_enabled(false);
		CHECK_FALSE(code_edit->is_line_folding_enabled());
	}

	SUBCASE("[CodeEdit] folding") {
		code_edit->set_line_folding_enabled(true);

		// No indent.
		code_edit->set_text("line1\nline2\nline3");
		for (int i = 0; i < 2; i++) {
			CHECK_FALSE(code_edit->can_fold_line(i));
			code_edit->fold_line(i);
			CHECK_FALSE(code_edit->is_line_folded(i));
		}
		CHECK(code_edit->get_next_visible_line_offset_from(1, 1) == 1);

		// Indented lines.
		code_edit->set_text("\tline1\n\tline2\n\tline3");
		for (int i = 0; i < 2; i++) {
			CHECK_FALSE(code_edit->can_fold_line(i));
			code_edit->fold_line(i);
			CHECK_FALSE(code_edit->is_line_folded(i));
		}
		CHECK(code_edit->get_next_visible_line_offset_from(1, 1) == 1);

		// Indent.
		code_edit->set_text("line1\n\tline2\nline3");
		CHECK(code_edit->can_fold_line(0));
		for (int i = 1; i < 2; i++) {
			CHECK_FALSE(code_edit->can_fold_line(i));
			code_edit->fold_line(i);
			CHECK_FALSE(code_edit->is_line_folded(i));
		}
		code_edit->fold_line(0);
		CHECK(code_edit->is_line_folded(0));
		CHECK_FALSE(code_edit->is_line_folded(1));
		CHECK_FALSE(code_edit->is_line_folded(2));
		CHECK(code_edit->get_next_visible_line_offset_from(1, 1) == 2);

		// Indent with blank lines.
		code_edit->set_text("line1\n\tline2\n\n\nline3");
		CHECK(code_edit->can_fold_line(0));
		for (int i = 1; i < 2; i++) {
			CHECK_FALSE(code_edit->can_fold_line(i));
			code_edit->fold_line(i);
			CHECK_FALSE(code_edit->is_line_folded(i));
		}
		code_edit->fold_line(0);
		CHECK(code_edit->is_line_folded(0));
		CHECK_FALSE(code_edit->is_line_folded(1));
		CHECK_FALSE(code_edit->is_line_folded(2));
		CHECK(code_edit->get_next_visible_line_offset_from(1, 1) == 2);

		// Nested indents.
		code_edit->set_text("line1\n\tline2\n\t\tline3\nline4");
		CHECK(code_edit->can_fold_line(0));
		CHECK(code_edit->can_fold_line(1));
		for (int i = 2; i < 3; i++) {
			CHECK_FALSE(code_edit->can_fold_line(i));
			code_edit->fold_line(i);
			CHECK_FALSE(code_edit->is_line_folded(i));
		}
		code_edit->fold_line(1);
		CHECK_FALSE(code_edit->is_line_folded(0));
		CHECK(code_edit->is_line_folded(1));
		CHECK_FALSE(code_edit->is_line_folded(2));
		CHECK_FALSE(code_edit->is_line_folded(3));
		CHECK(code_edit->get_next_visible_line_offset_from(2, 1) == 2);

		code_edit->fold_line(0);
		CHECK(code_edit->is_line_folded(0));
		CHECK_FALSE(code_edit->is_line_folded(1));
		CHECK_FALSE(code_edit->is_line_folded(2));
		CHECK_FALSE(code_edit->is_line_folded(3));
		CHECK(code_edit->get_next_visible_line_offset_from(1, 1) == 3);

		// Check metadata.
		CHECK(code_edit->get_folded_lines().size() == 1);
		CHECK((int)code_edit->get_folded_lines()[0] == 0);

		// Cannot unfold nested.
		code_edit->unfold_line(1);
		CHECK_FALSE(code_edit->is_line_folded(0));
		CHECK_FALSE(code_edit->is_line_folded(1));
		CHECK(code_edit->get_next_visible_line_offset_from(1, 1) == 1);

		// (un)Fold all / toggle.
		code_edit->unfold_line(0);
		CHECK_FALSE(code_edit->is_line_folded(0));
		CHECK_FALSE(code_edit->is_line_folded(1));
		CHECK(code_edit->get_next_visible_line_offset_from(1, 1) == 1);

		// Check metadata.
		CHECK(code_edit->get_folded_lines().size() == 0);

		code_edit->fold_all_lines();
		CHECK(code_edit->is_line_folded(0));
		CHECK_FALSE(code_edit->is_line_folded(1));
		CHECK(code_edit->get_next_visible_line_offset_from(1, 1) == 3);

		code_edit->unfold_all_lines();
		CHECK_FALSE(code_edit->is_line_folded(0));
		CHECK_FALSE(code_edit->is_line_folded(1));
		CHECK(code_edit->get_next_visible_line_offset_from(1, 1) == 1);

		code_edit->toggle_foldable_line(0);
		CHECK(code_edit->is_line_folded(0));
		CHECK_FALSE(code_edit->is_line_folded(1));
		CHECK(code_edit->get_next_visible_line_offset_from(1, 1) == 3);

		// Can also unfold from hidden line.
		code_edit->unfold_line(1);
		CHECK_FALSE(code_edit->is_line_folded(0));
		CHECK_FALSE(code_edit->is_line_folded(1));
		CHECK(code_edit->get_next_visible_line_offset_from(1, 1) == 1);

		// Blank lines.
		code_edit->set_text("line1\n\tline2\n\n\n\ttest\n\nline3");
		CHECK(code_edit->can_fold_line(0));
		for (int i = 1; i < code_edit->get_line_count(); i++) {
			CHECK_FALSE(code_edit->can_fold_line(i));
			code_edit->fold_line(i);
			CHECK_FALSE(code_edit->is_line_folded(i));
		}
		code_edit->fold_line(0);
		CHECK(code_edit->is_line_folded(0));
		for (int i = 1; i < code_edit->get_line_count(); i++) {
			CHECK_FALSE(code_edit->is_line_folded(i));
		}
		CHECK(code_edit->get_next_visible_line_offset_from(1, 1) == 5);

		// End of file.
		code_edit->set_text("line1\n\tline2");
		CHECK(code_edit->can_fold_line(0));
		CHECK_FALSE(code_edit->can_fold_line(1));
		code_edit->fold_line(1);
		CHECK_FALSE(code_edit->is_line_folded(1));
		code_edit->fold_line(0);
		CHECK(code_edit->is_line_folded(0));
		CHECK_FALSE(code_edit->is_line_folded(1));
		CHECK(code_edit->get_next_visible_line_offset_from(1, 1) == 1);

		// Comment & string blocks.
		// Single line block
		code_edit->add_comment_delimiter("#", "", true);
		code_edit->set_text("#line1\n#\tline2");
		CHECK(code_edit->can_fold_line(0));
		CHECK_FALSE(code_edit->can_fold_line(1));
		code_edit->fold_line(1);
		CHECK_FALSE(code_edit->is_line_folded(1));
		code_edit->fold_line(0);
		CHECK(code_edit->is_line_folded(0));
		CHECK_FALSE(code_edit->is_line_folded(1));
		CHECK(code_edit->get_next_visible_line_offset_from(1, 1) == 1);

		// Has to be full line.
		code_edit->set_text("test #line1\n#\tline2");
		CHECK_FALSE(code_edit->can_fold_line(0));
		CHECK_FALSE(code_edit->can_fold_line(1));
		code_edit->fold_line(1);
		CHECK_FALSE(code_edit->is_line_folded(1));
		code_edit->fold_line(0);
		CHECK_FALSE(code_edit->is_line_folded(0));
		CHECK_FALSE(code_edit->is_line_folded(1));
		CHECK(code_edit->get_next_visible_line_offset_from(1, 1) == 1);

		code_edit->set_text("#line1\ntest #\tline2");
		CHECK_FALSE(code_edit->can_fold_line(0));
		CHECK_FALSE(code_edit->can_fold_line(1));
		code_edit->fold_line(1);
		CHECK_FALSE(code_edit->is_line_folded(1));
		code_edit->fold_line(0);
		CHECK_FALSE(code_edit->is_line_folded(0));
		CHECK_FALSE(code_edit->is_line_folded(1));
		CHECK(code_edit->get_next_visible_line_offset_from(1, 1) == 1);

		// String.
		code_edit->add_string_delimiter("^", "", true);
		code_edit->set_text("^line1\n^\tline2");
		CHECK(code_edit->can_fold_line(0));
		CHECK_FALSE(code_edit->can_fold_line(1));
		code_edit->fold_line(1);
		CHECK_FALSE(code_edit->is_line_folded(1));
		code_edit->fold_line(0);
		CHECK(code_edit->is_line_folded(0));
		CHECK_FALSE(code_edit->is_line_folded(1));
		CHECK(code_edit->get_next_visible_line_offset_from(1, 1) == 1);

		// Has to be full line.
		code_edit->set_text("test ^line1\n^\tline2");
		CHECK_FALSE(code_edit->can_fold_line(0));
		CHECK_FALSE(code_edit->can_fold_line(1));
		code_edit->fold_line(1);
		CHECK_FALSE(code_edit->is_line_folded(1));
		code_edit->fold_line(0);
		CHECK_FALSE(code_edit->is_line_folded(0));
		CHECK_FALSE(code_edit->is_line_folded(1));
		CHECK(code_edit->get_next_visible_line_offset_from(1, 1) == 1);

		code_edit->set_text("^line1\ntest ^\tline2");
		CHECK_FALSE(code_edit->can_fold_line(0));
		CHECK_FALSE(code_edit->can_fold_line(1));
		code_edit->fold_line(1);
		CHECK_FALSE(code_edit->is_line_folded(1));
		code_edit->fold_line(0);
		CHECK_FALSE(code_edit->is_line_folded(0));
		CHECK_FALSE(code_edit->is_line_folded(1));
		CHECK(code_edit->get_next_visible_line_offset_from(1, 1) == 1);

		// Multiline blocks.
		code_edit->add_comment_delimiter("&", "&", false);
		code_edit->set_text("&line1\n\tline2&\nline3");
		CHECK(code_edit->can_fold_line(0));
		CHECK_FALSE(code_edit->can_fold_line(1));
		code_edit->fold_line(1);
		CHECK_FALSE(code_edit->is_line_folded(1));
		code_edit->fold_line(0);
		CHECK(code_edit->is_line_folded(0));
		CHECK_FALSE(code_edit->is_line_folded(1));
		CHECK(code_edit->get_next_visible_line_offset_from(1, 1) == 2);

		// Multiline comment before last line.
		code_edit->set_text("&line1\nline2&\ntest");
		CHECK(code_edit->can_fold_line(0));
		CHECK_FALSE(code_edit->can_fold_line(2));
		code_edit->fold_line(1);
		CHECK_FALSE(code_edit->is_line_folded(1));
		code_edit->fold_line(0);
		CHECK(code_edit->is_line_folded(0));
		CHECK(code_edit->get_next_visible_line_offset_from(1, 1) == 2);

		// Has to be full line.
		code_edit->set_text("test &line1\n\tline2&");
		CHECK_FALSE(code_edit->can_fold_line(0));
		CHECK_FALSE(code_edit->can_fold_line(1));
		code_edit->fold_line(1);
		CHECK_FALSE(code_edit->is_line_folded(1));
		code_edit->fold_line(0);
		CHECK_FALSE(code_edit->is_line_folded(0));
		CHECK_FALSE(code_edit->is_line_folded(1));
		CHECK(code_edit->get_next_visible_line_offset_from(1, 1) == 1);

		code_edit->set_text("&line1\n\tline2& test");
		CHECK_FALSE(code_edit->can_fold_line(0));
		CHECK_FALSE(code_edit->can_fold_line(1));
		code_edit->fold_line(1);
		CHECK_FALSE(code_edit->is_line_folded(1));
		code_edit->fold_line(0);
		CHECK_FALSE(code_edit->is_line_folded(0));
		CHECK_FALSE(code_edit->is_line_folded(1));
		CHECK(code_edit->get_next_visible_line_offset_from(1, 1) == 1);

		// Strings.
		code_edit->add_string_delimiter("$", "$", false);
		code_edit->set_text("$line1\n\tline2$");
		CHECK(code_edit->can_fold_line(0));
		CHECK_FALSE(code_edit->can_fold_line(1));
		code_edit->fold_line(1);
		CHECK_FALSE(code_edit->is_line_folded(1));
		code_edit->fold_line(0);
		CHECK(code_edit->is_line_folded(0));
		CHECK_FALSE(code_edit->is_line_folded(1));
		CHECK(code_edit->get_next_visible_line_offset_from(1, 1) == 1);

		// Has to be full line.
		code_edit->set_text("test $line1\n\tline2$");
		CHECK_FALSE(code_edit->can_fold_line(0));
		CHECK_FALSE(code_edit->can_fold_line(1));
		code_edit->fold_line(1);
		CHECK_FALSE(code_edit->is_line_folded(1));
		code_edit->fold_line(0);
		CHECK_FALSE(code_edit->is_line_folded(0));
		CHECK_FALSE(code_edit->is_line_folded(1));
		CHECK(code_edit->get_next_visible_line_offset_from(1, 1) == 1);

		code_edit->set_text("$line1\n\tline2$ test");
		CHECK_FALSE(code_edit->can_fold_line(0));
		CHECK_FALSE(code_edit->can_fold_line(1));
		code_edit->fold_line(1);
		CHECK_FALSE(code_edit->is_line_folded(1));
		code_edit->fold_line(0);
		CHECK_FALSE(code_edit->is_line_folded(0));
		CHECK_FALSE(code_edit->is_line_folded(1));
		CHECK(code_edit->get_next_visible_line_offset_from(1, 1) == 1);

		// Non-indented comments/strings.
		// Single line
		code_edit->set_text("test\n\tline1\n#line1\n#line2\n\ttest");
		CHECK(code_edit->can_fold_line(0));
		CHECK_FALSE(code_edit->can_fold_line(1));
		code_edit->fold_line(1);
		CHECK_FALSE(code_edit->is_line_folded(1));
		code_edit->fold_line(0);
		CHECK(code_edit->is_line_folded(0));
		CHECK_FALSE(code_edit->is_line_folded(1));
		CHECK(code_edit->get_next_visible_line_offset_from(1, 1) == 4);

		code_edit->set_text("test\n\tline1\n^line1\n^line2\n\ttest");
		CHECK(code_edit->can_fold_line(0));
		CHECK_FALSE(code_edit->can_fold_line(1));
		code_edit->fold_line(1);
		CHECK_FALSE(code_edit->is_line_folded(1));
		code_edit->fold_line(0);
		CHECK(code_edit->is_line_folded(0));
		CHECK_FALSE(code_edit->is_line_folded(1));
		CHECK(code_edit->get_next_visible_line_offset_from(1, 1) == 4);

		// Indent level 0->1, comment after lines
		code_edit->set_text("line1\n\tline2\n#test");
		CHECK(code_edit->can_fold_line(0));
		CHECK_FALSE(code_edit->can_fold_line(1));
		code_edit->fold_line(1);
		CHECK_FALSE(code_edit->is_line_folded(1));
		code_edit->fold_line(0);
		CHECK(code_edit->is_line_folded(0));
		CHECK_FALSE(code_edit->is_line_folded(1));
		CHECK(code_edit->get_next_visible_line_offset_from(1, 1) == 2);

		// Indent level 0->1, comment between lines
		code_edit->set_text("line1\n#test\n\tline2\nline3");
		CHECK(code_edit->can_fold_line(0));
		CHECK_FALSE(code_edit->can_fold_line(2));
		code_edit->fold_line(2);
		CHECK_FALSE(code_edit->is_line_folded(2));
		code_edit->fold_line(0);
		CHECK(code_edit->is_line_folded(0));
		CHECK_FALSE(code_edit->is_line_folded(2));
		CHECK(code_edit->get_next_visible_line_offset_from(1, 1) == 3);

		// Indent level 1->2, comment after lines
		code_edit->set_text("\tline1\n\t\tline2\n#test");
		CHECK(code_edit->can_fold_line(0));
		CHECK_FALSE(code_edit->can_fold_line(1));
		code_edit->fold_line(1);
		CHECK_FALSE(code_edit->is_line_folded(1));
		code_edit->fold_line(0);
		CHECK(code_edit->is_line_folded(0));
		CHECK_FALSE(code_edit->is_line_folded(1));
		CHECK(code_edit->get_next_visible_line_offset_from(1, 1) == 2);

		// Indent level 1->2, comment between lines
		code_edit->set_text("\tline1\n#test\n\t\tline2\nline3");
		CHECK(code_edit->can_fold_line(0));
		CHECK_FALSE(code_edit->can_fold_line(2));
		code_edit->fold_line(2);
		CHECK_FALSE(code_edit->is_line_folded(2));
		code_edit->fold_line(0);
		CHECK(code_edit->is_line_folded(0));
		CHECK_FALSE(code_edit->is_line_folded(2));
		CHECK(code_edit->get_next_visible_line_offset_from(1, 1) == 3);

		// Multiline
		code_edit->set_text("test\n\tline1\n&line1\nline2&\n\ttest");
		CHECK(code_edit->can_fold_line(0));
		CHECK_FALSE(code_edit->can_fold_line(1));
		code_edit->fold_line(1);
		CHECK_FALSE(code_edit->is_line_folded(1));
		code_edit->fold_line(0);
		CHECK(code_edit->is_line_folded(0));
		CHECK_FALSE(code_edit->is_line_folded(1));
		CHECK(code_edit->get_next_visible_line_offset_from(1, 1) == 4);

		code_edit->set_text("test\n\tline1\n$line1\nline2$\n\ttest");
		CHECK(code_edit->can_fold_line(0));
		CHECK_FALSE(code_edit->can_fold_line(1));
		code_edit->fold_line(1);
		CHECK_FALSE(code_edit->is_line_folded(1));
		code_edit->fold_line(0);
		CHECK(code_edit->is_line_folded(0));
		CHECK_FALSE(code_edit->is_line_folded(1));
		CHECK(code_edit->get_next_visible_line_offset_from(1, 1) == 4);
	}

	memdelete(code_edit);
}

TEST_CASE("[SceneTree][CodeEdit] region folding") {
	CodeEdit *code_edit = memnew(CodeEdit);
	SceneTree::get_singleton()->get_root()->add_child(code_edit);
	code_edit->grab_focus();

	SUBCASE("[CodeEdit] region folding") {
		code_edit->set_line_folding_enabled(true);

		// Region tag detection.
		code_edit->set_text("#region region_name\nline2\n#endregion");
		code_edit->clear_comment_delimiters();
		code_edit->add_comment_delimiter("#", "");
		CHECK(code_edit->is_line_code_region_start(0));
		CHECK_FALSE(code_edit->is_line_code_region_start(1));
		CHECK_FALSE(code_edit->is_line_code_region_start(2));
		CHECK_FALSE(code_edit->is_line_code_region_end(0));
		CHECK_FALSE(code_edit->is_line_code_region_end(1));
		CHECK(code_edit->is_line_code_region_end(2));

		// Region tag customization.
		code_edit->set_text("#region region_name\nline2\n#endregion\n#open region_name\nline2\n#close");
		code_edit->clear_comment_delimiters();
		code_edit->add_comment_delimiter("#", "");
		CHECK(code_edit->is_line_code_region_start(0));
		CHECK(code_edit->is_line_code_region_end(2));
		CHECK_FALSE(code_edit->is_line_code_region_start(3));
		CHECK_FALSE(code_edit->is_line_code_region_end(5));
		code_edit->set_code_region_tags("open", "close");
		CHECK_FALSE(code_edit->is_line_code_region_start(0));
		CHECK_FALSE(code_edit->is_line_code_region_end(2));
		CHECK(code_edit->is_line_code_region_start(3));
		CHECK(code_edit->is_line_code_region_end(5));
		code_edit->set_code_region_tags("region", "endregion");

		// Setting identical start and end region tags should fail.
		CHECK(code_edit->get_code_region_start_tag() == "region");
		CHECK(code_edit->get_code_region_end_tag() == "endregion");
		ERR_PRINT_OFF;
		code_edit->set_code_region_tags("same_tag", "same_tag");
		ERR_PRINT_ON;
		CHECK(code_edit->get_code_region_start_tag() == "region");
		CHECK(code_edit->get_code_region_end_tag() == "endregion");

		// Region creation with selection adds start / close region lines.
		code_edit->set_text("line1\nline2\nline3");
		code_edit->clear_comment_delimiters();
		code_edit->add_comment_delimiter("#", "");
		code_edit->select(1, 0, 1, 4);
		code_edit->create_code_region();
		CHECK(code_edit->is_line_code_region_start(1));
		CHECK(code_edit->get_line(2).contains("line2"));
		CHECK(code_edit->is_line_code_region_end(3));

		// Region creation without any selection has no effect.
		code_edit->set_text("line1\nline2\nline3");
		code_edit->clear_comment_delimiters();
		code_edit->add_comment_delimiter("#", "");
		code_edit->create_code_region();
		CHECK(code_edit->get_text() == "line1\nline2\nline3");

		// Region creation with multiple selections.
		code_edit->set_text("line1\nline2\nline3");
		code_edit->clear_comment_delimiters();
		code_edit->add_comment_delimiter("#", "");
		code_edit->select(0, 0, 0, 4, 0);
		code_edit->add_caret(2, 5);
		code_edit->select(2, 0, 2, 5, 1);
		code_edit->create_code_region();
		CHECK(code_edit->get_text() == "#region New Code Region\nline1\n#endregion\nline2\n#region New Code Region\nline3\n#endregion");

		// Two selections on the same line create only one region.
		code_edit->set_text("test line1\ntest line2\ntest line3");
		code_edit->clear_comment_delimiters();
		code_edit->add_comment_delimiter("#", "");
		code_edit->select(0, 0, 1, 2, 0);
		code_edit->add_caret(1, 4);
		code_edit->select(1, 4, 2, 5, 1);
		code_edit->create_code_region();
		CHECK(code_edit->get_text() == "#region New Code Region\ntest line1\ntest line2\ntest line3\n#endregion");

		// Region tag with // comment delimiter.
		code_edit->set_text("//region region_name\nline2\n//endregion");
		code_edit->clear_comment_delimiters();
		code_edit->add_comment_delimiter("//", "");
		CHECK(code_edit->is_line_code_region_start(0));
		CHECK(code_edit->is_line_code_region_end(2));

		// Creating region with no valid one line comment delimiter has no effect.
		code_edit->set_text("line1\nline2\nline3");
		code_edit->clear_comment_delimiters();
		code_edit->create_code_region();
		CHECK(code_edit->get_text() == "line1\nline2\nline3");
		code_edit->add_comment_delimiter("/*", "*/");
		code_edit->create_code_region();
		CHECK(code_edit->get_text() == "line1\nline2\nline3");

		// Choose one line comment delimiter.
		code_edit->set_text("//region region_name\nline2\n//endregion");
		code_edit->clear_comment_delimiters();
		code_edit->add_comment_delimiter("/*", "*/");
		code_edit->add_comment_delimiter("//", "");
		CHECK(code_edit->is_line_code_region_start(0));
		CHECK(code_edit->is_line_code_region_end(2));

		// Update code region delimiter when removing comment delimiter.
		code_edit->set_text("#region region_name\nline2\n#endregion\n//region region_name\nline2\n//endregion");
		code_edit->clear_comment_delimiters();
		code_edit->add_comment_delimiter("//", "");
		code_edit->add_comment_delimiter("#", ""); // A shorter delimiter has higher priority.
		CHECK(code_edit->is_line_code_region_start(0));
		CHECK(code_edit->is_line_code_region_end(2));
		CHECK_FALSE(code_edit->is_line_code_region_start(3));
		CHECK_FALSE(code_edit->is_line_code_region_end(5));
		code_edit->remove_comment_delimiter("#");
		CHECK_FALSE(code_edit->is_line_code_region_start(0));
		CHECK_FALSE(code_edit->is_line_code_region_end(2));
		CHECK(code_edit->is_line_code_region_start(3));
		CHECK(code_edit->is_line_code_region_end(5));

		// Update code region delimiter when clearing comment delimiters.
		code_edit->set_text("//region region_name\nline2\n//endregion");
		code_edit->clear_comment_delimiters();
		code_edit->add_comment_delimiter("//", "");
		CHECK(code_edit->is_line_code_region_start(0));
		CHECK(code_edit->is_line_code_region_end(2));
		code_edit->clear_comment_delimiters();
		CHECK_FALSE(code_edit->is_line_code_region_start(0));
		CHECK_FALSE(code_edit->is_line_code_region_end(2));

		// Fold region.
		code_edit->clear_comment_delimiters();
		code_edit->add_comment_delimiter("#", "");
		code_edit->set_text("#region region_name\nline2\nline3\n#endregion\nvisible line");
		CHECK(code_edit->can_fold_line(0));
		for (int i = 1; i < 5; i++) {
			CHECK_FALSE(code_edit->can_fold_line(i));
		}
		for (int i = 0; i < 5; i++) {
			CHECK_FALSE(code_edit->is_line_folded(i));
		}
		code_edit->fold_line(0);
		CHECK(code_edit->is_line_folded(0));
		CHECK(code_edit->get_next_visible_line_offset_from(1, 1) == 4);

		// Region with no end can't be folded.
		ERR_PRINT_OFF;
		code_edit->clear_comment_delimiters();
		code_edit->add_comment_delimiter("#", "");
		code_edit->set_text("#region region_name\nline2\nline3\n#bad_end_tag\nvisible line");
		CHECK_FALSE(code_edit->can_fold_line(0));
		ERR_PRINT_ON;

		// Bad nested region can't be folded.
		ERR_PRINT_OFF;
		code_edit->clear_comment_delimiters();
		code_edit->add_comment_delimiter("#", "");
		code_edit->set_text("#region without end\n#region region2\nline3\n#endregion\n#no_end");
		CHECK_FALSE(code_edit->can_fold_line(0));
		CHECK(code_edit->can_fold_line(1));
		ERR_PRINT_ON;

		// Nested region folding.
		ERR_PRINT_OFF;
		code_edit->clear_comment_delimiters();
		code_edit->add_comment_delimiter("#", "");
		code_edit->set_text("#region region1\n#region region2\nline3\n#endregion\n#endregion");
		CHECK(code_edit->can_fold_line(0));
		CHECK(code_edit->can_fold_line(1));
		code_edit->fold_line(1);
		CHECK(code_edit->get_next_visible_line_offset_from(2, 1) == 3);
		code_edit->fold_line(0);
		CHECK(code_edit->get_next_visible_line_offset_from(1, 1) == 4);
		ERR_PRINT_ON;

		// Unfolding a line inside a region unfold whole region.
		code_edit->clear_comment_delimiters();
		code_edit->add_comment_delimiter("#", "");
		code_edit->set_text("#region region\ninside\nline3\n#endregion\nvisible");
		code_edit->fold_line(0);
		CHECK(code_edit->is_line_folded(0));
		CHECK(code_edit->get_next_visible_line_offset_from(1, 1) == 4);
		code_edit->unfold_line(1);
		CHECK_FALSE(code_edit->is_line_folded(0));
	}

	memdelete(code_edit);
}

TEST_CASE("[SceneTree][CodeEdit] completion") {
	CodeEdit *code_edit = memnew(CodeEdit);
	SceneTree::get_singleton()->get_root()->add_child(code_edit);
	code_edit->grab_focus();

	SUBCASE("[CodeEdit] auto brace completion") {
		code_edit->set_auto_brace_completion_enabled(true);
		CHECK(code_edit->is_auto_brace_completion_enabled());

		code_edit->set_highlight_matching_braces_enabled(true);
		CHECK(code_edit->is_highlight_matching_braces_enabled());

		/* Try setters, any length. */
		Dictionary auto_brace_completion_pairs;
		auto_brace_completion_pairs["["] = "]";
		auto_brace_completion_pairs["'"] = "'";
		auto_brace_completion_pairs[";"] = "'";
		auto_brace_completion_pairs["'''"] = "'''";
		code_edit->set_auto_brace_completion_pairs(auto_brace_completion_pairs);
		CHECK(code_edit->get_auto_brace_completion_pairs().size() == 4);
		CHECK(code_edit->get_auto_brace_completion_pairs()["["] == "]");
		CHECK(code_edit->get_auto_brace_completion_pairs()["'"] == "'");
		CHECK(code_edit->get_auto_brace_completion_pairs()[";"] == "'");
		CHECK(code_edit->get_auto_brace_completion_pairs()["'''"] == "'''");

		ERR_PRINT_OFF;

		/* No duplicate start keys. */
		code_edit->add_auto_brace_completion_pair("[", "]");
		CHECK(code_edit->get_auto_brace_completion_pairs().size() == 4);

		/* No empty keys. */
		code_edit->add_auto_brace_completion_pair("[", "");
		CHECK(code_edit->get_auto_brace_completion_pairs().size() == 4);

		code_edit->add_auto_brace_completion_pair("", "]");
		CHECK(code_edit->get_auto_brace_completion_pairs().size() == 4);

		code_edit->add_auto_brace_completion_pair("", "");
		CHECK(code_edit->get_auto_brace_completion_pairs().size() == 4);

		/* Must be a symbol. */
		code_edit->add_auto_brace_completion_pair("a", "]");
		CHECK(code_edit->get_auto_brace_completion_pairs().size() == 4);

		code_edit->add_auto_brace_completion_pair("[", "a");
		CHECK(code_edit->get_auto_brace_completion_pairs().size() == 4);

		code_edit->add_auto_brace_completion_pair("a", "a");
		CHECK(code_edit->get_auto_brace_completion_pairs().size() == 4);

		ERR_PRINT_ON;

		/* Check metadata. */
		CHECK(code_edit->has_auto_brace_completion_open_key("["));
		CHECK(code_edit->has_auto_brace_completion_open_key("'"));
		CHECK(code_edit->has_auto_brace_completion_open_key(";"));
		CHECK(code_edit->has_auto_brace_completion_open_key("'''"));
		CHECK_FALSE(code_edit->has_auto_brace_completion_open_key("("));

		CHECK(code_edit->has_auto_brace_completion_close_key("]"));
		CHECK(code_edit->has_auto_brace_completion_close_key("'"));
		CHECK(code_edit->has_auto_brace_completion_close_key("'''"));
		CHECK_FALSE(code_edit->has_auto_brace_completion_close_key(")"));

		CHECK(code_edit->get_auto_brace_completion_close_key("[") == "]");
		CHECK(code_edit->get_auto_brace_completion_close_key("'") == "'");
		CHECK(code_edit->get_auto_brace_completion_close_key(";") == "'");
		CHECK(code_edit->get_auto_brace_completion_close_key("'''") == "'''");
		CHECK(code_edit->get_auto_brace_completion_close_key("(").is_empty());

		/* Check typing inserts closing pair. */
		code_edit->clear();
		SEND_GUI_KEY_EVENT(Key::BRACKETLEFT);
		CHECK(code_edit->get_line(0) == "[]");

		/* Should first match and insert smaller key. */
		code_edit->clear();
		SEND_GUI_KEY_EVENT(Key::APOSTROPHE);
		CHECK(code_edit->get_line(0) == "''");
		CHECK(code_edit->get_caret_column() == 1);

		/* Move out from center, Should match and insert larger key. */
		SEND_GUI_ACTION("ui_text_caret_right");
		SEND_GUI_KEY_EVENT(Key::APOSTROPHE);
		CHECK(code_edit->get_line(0) == "''''''");
		CHECK(code_edit->get_caret_column() == 3);

		/* Backspace should remove all. */
		SEND_GUI_ACTION("ui_text_backspace");
		CHECK(code_edit->get_line(0).is_empty());

		/* If in between and typing close key should "skip". */
		SEND_GUI_KEY_EVENT(Key::BRACKETLEFT);
		CHECK(code_edit->get_line(0) == "[]");
		CHECK(code_edit->get_caret_column() == 1);
		SEND_GUI_KEY_EVENT(Key::BRACKETRIGHT);
		CHECK(code_edit->get_line(0) == "[]");
		CHECK(code_edit->get_caret_column() == 2);

		/* If current is char and inserting a string, do not autocomplete. */
		code_edit->clear();
		SEND_GUI_KEY_EVENT(Key::A);
		SEND_GUI_KEY_EVENT(Key::APOSTROPHE);
		CHECK(code_edit->get_line(0) == "A'");

		/* If in comment, do not complete. */
		code_edit->add_comment_delimiter("#", "");
		code_edit->clear();
		SEND_GUI_KEY_EVENT(Key::NUMBERSIGN);
		SEND_GUI_KEY_EVENT(Key::APOSTROPHE);
		CHECK(code_edit->get_line(0) == "#'");

		/* If in string, and inserting string do not complete. */
		code_edit->clear();
		SEND_GUI_KEY_EVENT(Key::APOSTROPHE);
		SEND_GUI_KEY_EVENT(Key::QUOTEDBL);
		CHECK(code_edit->get_line(0) == "'\"'");

		/* Wrap single line selection with brackets */
		code_edit->clear();
		code_edit->insert_text_at_caret("abc");
		code_edit->select_all();
		SEND_GUI_KEY_EVENT(Key::BRACKETLEFT);
		CHECK(code_edit->get_line(0) == "[abc]");

		/* Caret should be after the last character of the single line selection */
		CHECK(code_edit->get_caret_column() == 4);

		/* Wrap multi line selection with brackets */
		code_edit->clear();
		code_edit->insert_text_at_caret("abc\nabc");
		code_edit->select_all();
		SEND_GUI_KEY_EVENT(Key::BRACKETLEFT);
		CHECK(code_edit->get_text() == "[abc\nabc]");

		/* Caret should be after the last character of the multi line selection */
		CHECK(code_edit->get_caret_line() == 1);
		CHECK(code_edit->get_caret_column() == 3);

		/* If inserted character is not a auto brace completion open key, replace selected text with the inserted character */
		code_edit->clear();
		code_edit->insert_text_at_caret("abc");
		code_edit->select_all();
		SEND_GUI_KEY_EVENT(Key::KEY_1);
		CHECK(code_edit->get_text() == "1");

		/* If potential multichar and single brace completion is matched, it should wrap the single.  */
		code_edit->clear();
		code_edit->insert_text_at_caret("\'\'abc");
		code_edit->select(0, 2, 0, 5);
		SEND_GUI_KEY_EVENT(Key::APOSTROPHE);
		CHECK(code_edit->get_text() == "\'\'\'abc\'");

		/* If only the potential multichar brace completion is matched, it does not wrap or complete. */
		auto_brace_completion_pairs.erase("\'");
		code_edit->set_auto_brace_completion_pairs(auto_brace_completion_pairs);
		CHECK_FALSE(code_edit->has_auto_brace_completion_open_key("\'"));

		code_edit->clear();
		code_edit->insert_text_at_caret("\'\'abc");
		code_edit->select(0, 2, 0, 5);
		SEND_GUI_KEY_EVENT(Key::APOSTROPHE);
		CHECK(code_edit->get_text() == "\'\'\'");
	}

	SUBCASE("[CodeEdit] autocomplete with brace completion") {
		code_edit->set_auto_brace_completion_enabled(true);
		CHECK(code_edit->is_auto_brace_completion_enabled());

		code_edit->insert_text_at_caret("(te)");
		code_edit->set_caret_column(3);

		// Full completion.
		code_edit->add_code_completion_option(CodeEdit::CodeCompletionKind::KIND_FUNCTION, "test()", "test()");
		code_edit->update_code_completion_options();
		code_edit->confirm_code_completion();
		CHECK(code_edit->get_line(0) == "(test())");
		CHECK(code_edit->get_caret_column() == 7);
		code_edit->undo();

		// With "arg".
		code_edit->add_code_completion_option(CodeEdit::CodeCompletionKind::KIND_FUNCTION, "test(", "test(");
		code_edit->update_code_completion_options();
		code_edit->confirm_code_completion();
		CHECK(code_edit->get_line(0) == "(test())");
		CHECK(code_edit->get_caret_column() == 6);
		code_edit->undo();

		// brace completion disabled
		code_edit->set_auto_brace_completion_enabled(false);

		// Full completion.
		code_edit->add_code_completion_option(CodeEdit::CodeCompletionKind::KIND_FUNCTION, "test()", "test()");
		code_edit->update_code_completion_options();
		code_edit->confirm_code_completion();
		CHECK(code_edit->get_line(0) == "(test())");
		CHECK(code_edit->get_caret_column() == 7);
		code_edit->undo();

		// With "arg".
		code_edit->add_code_completion_option(CodeEdit::CodeCompletionKind::KIND_FUNCTION, "test(", "test(");
		code_edit->update_code_completion_options();
		code_edit->confirm_code_completion();
		CHECK(code_edit->get_line(0) == "(test()");
		CHECK(code_edit->get_caret_column() == 6);

		// String
		code_edit->set_auto_brace_completion_enabled(true);
		code_edit->clear();
		code_edit->insert_text_at_caret("\"\"");
		code_edit->set_caret_column(1);

		// Full completion.
		code_edit->add_code_completion_option(CodeEdit::CodeCompletionKind::KIND_NODE_PATH, "\"test\"", "\"test\"");
		code_edit->update_code_completion_options();
		code_edit->confirm_code_completion();
		CHECK(code_edit->get_line(0) == "\"test\"");
		CHECK(code_edit->get_caret_column() == 6);
		code_edit->undo();

		// With "arg".
		code_edit->add_code_completion_option(CodeEdit::CodeCompletionKind::KIND_NODE_PATH, "\"test", "\"test");
		code_edit->update_code_completion_options();
		code_edit->confirm_code_completion();
		CHECK(code_edit->get_line(0) == "\"\"test\"");
		CHECK(code_edit->get_caret_column() == 7);
		code_edit->undo();

		// brace completion disabled
		code_edit->set_auto_brace_completion_enabled(false);

		// Full completion.
		code_edit->add_code_completion_option(CodeEdit::CodeCompletionKind::KIND_NODE_PATH, "\"test\"", "\"test\"");
		code_edit->update_code_completion_options();
		code_edit->confirm_code_completion();
		CHECK(code_edit->get_line(0) == "\"test\"");
		CHECK(code_edit->get_caret_column() == 6);
		code_edit->undo();

		// With "arg".
		code_edit->add_code_completion_option(CodeEdit::CodeCompletionKind::KIND_NODE_PATH, "\"test", "\"test");
		code_edit->update_code_completion_options();
		code_edit->confirm_code_completion();
		CHECK(code_edit->get_line(0) == "\"\"test\"");
		CHECK(code_edit->get_caret_column() == 7);
		code_edit->undo();
	}

	SUBCASE("[CodeEdit] autocomplete") {
		code_edit->set_code_completion_enabled(true);
		CHECK(code_edit->is_code_completion_enabled());

		/* Set prefixes, single char only, disallow empty. */
		TypedArray<String> completion_prefixes;
		completion_prefixes.push_back("");
		completion_prefixes.push_back(".");
		completion_prefixes.push_back(".");
		completion_prefixes.push_back(",,");

		ERR_PRINT_OFF;
		code_edit->set_code_completion_prefixes(completion_prefixes);
		ERR_PRINT_ON;
		completion_prefixes = code_edit->get_code_completion_prefixes();
		CHECK(completion_prefixes.size() == 2);
		CHECK(completion_prefixes.has("."));
		CHECK(completion_prefixes.has(","));

		code_edit->set_text("test\ntest");
		CHECK(code_edit->get_text_for_code_completion() == String::chr(0xFFFF) + "test\ntest");
	}

	SUBCASE("[CodeEdit] autocomplete request") {
		SIGNAL_WATCH(code_edit, "code_completion_requested");
		code_edit->set_code_completion_enabled(true);

		Array signal_args;
		signal_args.push_back(Array());

		/* Force request. */
		code_edit->request_code_completion();
		SIGNAL_CHECK_FALSE("code_completion_requested");
		code_edit->request_code_completion(true);
		SIGNAL_CHECK("code_completion_requested", signal_args);

		/* Manual request should force. */
		SEND_GUI_ACTION("ui_text_completion_query");
		SIGNAL_CHECK("code_completion_requested", signal_args);

		/* Insert prefix. */
		TypedArray<String> completion_prefixes;
		completion_prefixes.push_back(".");
		code_edit->set_code_completion_prefixes(completion_prefixes);

		code_edit->insert_text_at_caret(".");
		code_edit->request_code_completion();
		SIGNAL_CHECK("code_completion_requested", signal_args);

		/* Should work with space too. */
		code_edit->insert_text_at_caret(" ");
		code_edit->request_code_completion();
		SIGNAL_CHECK("code_completion_requested", signal_args);

		/* Should work when complete ends with prefix. */
		code_edit->clear();
		code_edit->insert_text_at_caret("t");
		code_edit->add_code_completion_option(CodeEdit::CodeCompletionKind::KIND_CLASS, "test.", "test.");
		code_edit->update_code_completion_options();
		code_edit->confirm_code_completion();
		CHECK(code_edit->get_line(0) == "test.");
		SIGNAL_CHECK("code_completion_requested", signal_args);

		SIGNAL_UNWATCH(code_edit, "code_completion_requested");
	}

	SUBCASE("[CodeEdit] autocomplete completion") {
		if (TS->has_feature(TextServer::FEATURE_FONT_DYNAMIC) && TS->has_feature(TextServer::FEATURE_SIMPLE_LAYOUT)) {
			CHECK(code_edit->get_code_completion_selected_index() == -1);
			code_edit->set_code_completion_enabled(true);
			CHECK(code_edit->get_code_completion_selected_index() == -1);

			code_edit->update_code_completion_options();
			code_edit->set_code_completion_selected_index(1);
			CHECK(code_edit->get_code_completion_selected_index() == -1);
			CHECK(code_edit->get_code_completion_option(0).size() == 0);
			CHECK(code_edit->get_code_completion_options().size() == 0);

			/* Adding does not update the list. */
			code_edit->add_code_completion_option(CodeEdit::CodeCompletionKind::KIND_VARIABLE, "item_0.", "item_0");

			code_edit->set_code_completion_selected_index(1);
			CHECK(code_edit->get_code_completion_selected_index() == -1);
			CHECK(code_edit->get_code_completion_option(0).size() == 0);
			CHECK(code_edit->get_code_completion_options().size() == 0);

			/* After update, pending add should not be counted, */
			/* also does not work on col 0                      */
			int before_text_caret_column = code_edit->get_caret_column();
			code_edit->insert_text_at_caret("i");

			code_edit->update_code_completion_options();
			code_edit->add_code_completion_option(CodeEdit::CodeCompletionKind::KIND_CLASS, "item_0.", "item_0", Color(1, 0, 0), Ref<Resource>(), Color(1, 0, 0));
			code_edit->add_code_completion_option(CodeEdit::CodeCompletionKind::KIND_VARIABLE, "item_1.", "item_1");
			code_edit->add_code_completion_option(CodeEdit::CodeCompletionKind::KIND_VARIABLE, "item_2.", "item_2");

			ERR_PRINT_OFF;
			code_edit->set_code_completion_selected_index(1);
			ERR_PRINT_ON;
			CHECK(code_edit->get_code_completion_selected_index() == 0);
			CHECK(code_edit->get_code_completion_option(0).size() == 7);
			CHECK(code_edit->get_code_completion_options().size() == 1);

			/* Check cancel closes completion. */
			SEND_GUI_ACTION("ui_cancel");
			CHECK(code_edit->get_code_completion_selected_index() == -1);

			code_edit->update_code_completion_options();
			CHECK(code_edit->get_code_completion_selected_index() == 0);
			code_edit->set_code_completion_selected_index(1);
			CHECK(code_edit->get_code_completion_selected_index() == 1);
			CHECK(code_edit->get_code_completion_option(0).size() == 7);
			CHECK(code_edit->get_code_completion_options().size() == 3);

			/* Check data. */
			Dictionary option = code_edit->get_code_completion_option(0);
			CHECK((int)option["kind"] == (int)CodeEdit::CodeCompletionKind::KIND_CLASS);
			CHECK(option["display_text"] == "item_0.");
			CHECK(option["insert_text"] == "item_0");
			CHECK(option["font_color"] == Color(1, 0, 0));
			CHECK(option["icon"] == Ref<Resource>());
			CHECK(option["default_value"] == Color(1, 0, 0));

			/* Set size for mouse input. */
			code_edit->set_size(Size2(100, 100));

			/* Test home and end keys close the completion and move the caret */
			/* => ui_text_caret_line_start */
			code_edit->set_caret_column(before_text_caret_column + 1);
			code_edit->add_code_completion_option(CodeEdit::CodeCompletionKind::KIND_CLASS, "item_0.", "item_0", Color(1, 0, 0), Ref<Resource>(), Color(1, 0, 0));
			code_edit->add_code_completion_option(CodeEdit::CodeCompletionKind::KIND_VARIABLE, "item_1.", "item_1");
			code_edit->add_code_completion_option(CodeEdit::CodeCompletionKind::KIND_VARIABLE, "item_2.", "item_2");

			code_edit->update_code_completion_options();

			SEND_GUI_ACTION("ui_text_caret_line_start");
			code_edit->update_code_completion_options();
			CHECK(code_edit->get_code_completion_selected_index() == -1);
			CHECK(code_edit->get_caret_column() == 0);

			/* => ui_text_caret_line_end */
			code_edit->set_caret_column(before_text_caret_column + 1);
			code_edit->add_code_completion_option(CodeEdit::CodeCompletionKind::KIND_CLASS, "item_0.", "item_0", Color(1, 0, 0), Ref<Resource>(), Color(1, 0, 0));
			code_edit->add_code_completion_option(CodeEdit::CodeCompletionKind::KIND_VARIABLE, "item_1.", "item_1");
			code_edit->add_code_completion_option(CodeEdit::CodeCompletionKind::KIND_VARIABLE, "item_2.", "item_2");

			code_edit->update_code_completion_options();

			SEND_GUI_ACTION("ui_text_caret_line_end");
			code_edit->update_code_completion_options();
			CHECK(code_edit->get_code_completion_selected_index() == -1);
			CHECK(code_edit->get_caret_column() == before_text_caret_column + 1);

			/* Check input. */
			code_edit->set_caret_column(before_text_caret_column + 1);
			code_edit->add_code_completion_option(CodeEdit::CodeCompletionKind::KIND_CLASS, "item_0.", "item_0", Color(1, 0, 0), Ref<Resource>(), Color(1, 0, 0));
			code_edit->add_code_completion_option(CodeEdit::CodeCompletionKind::KIND_VARIABLE, "item_1.", "item_1");
			code_edit->add_code_completion_option(CodeEdit::CodeCompletionKind::KIND_VARIABLE, "item_2.", "item_2");

			code_edit->update_code_completion_options();

			SEND_GUI_ACTION("ui_page_down");
			CHECK(code_edit->get_code_completion_selected_index() == 2);

			SEND_GUI_ACTION("ui_page_up");
			CHECK(code_edit->get_code_completion_selected_index() == 0);

			SEND_GUI_ACTION("ui_up");
			CHECK(code_edit->get_code_completion_selected_index() == 2);

			SEND_GUI_ACTION("ui_down");
			CHECK(code_edit->get_code_completion_selected_index() == 0);

			SEND_GUI_KEY_EVENT(Key::T);
			CHECK(code_edit->get_code_completion_selected_index() == 0);

			SEND_GUI_ACTION("ui_left");
			CHECK(code_edit->get_code_completion_selected_index() == 0);

			SEND_GUI_ACTION("ui_right");
			CHECK(code_edit->get_code_completion_selected_index() == 0);

			SEND_GUI_ACTION("ui_text_backspace");
			CHECK(code_edit->get_code_completion_selected_index() == 0);

			Point2 caret_pos = code_edit->get_caret_draw_pos();
			caret_pos.y += code_edit->get_line_height();
			SEND_GUI_MOUSE_BUTTON_EVENT(caret_pos, MouseButton::WHEEL_DOWN, 0, Key::NONE);
			CHECK(code_edit->get_code_completion_selected_index() == 1);

			SEND_GUI_MOUSE_BUTTON_EVENT(caret_pos, MouseButton::WHEEL_UP, 0, Key::NONE);
			CHECK(code_edit->get_code_completion_selected_index() == 0);

			/* Single click selects. */
			caret_pos.y += code_edit->get_line_height() * 2;
			SEND_GUI_MOUSE_BUTTON_EVENT(caret_pos, MouseButton::LEFT, MouseButtonMask::LEFT, Key::NONE);
			CHECK(code_edit->get_code_completion_selected_index() == 2);

			/* Double click inserts. */
			SEND_GUI_DOUBLE_CLICK(caret_pos, Key::NONE);
			CHECK(code_edit->get_code_completion_selected_index() == -1);
			CHECK(code_edit->get_line(0) == "item_2");

			code_edit->set_auto_brace_completion_enabled(false);

			/* Does nothing in readonly. */
			code_edit->undo();
			code_edit->add_code_completion_option(CodeEdit::CodeCompletionKind::KIND_CLASS, "item_0.", "item_0");
			code_edit->update_code_completion_options();
			code_edit->set_editable(false);
			code_edit->confirm_code_completion();
			code_edit->set_editable(true);
			CHECK(code_edit->get_line(0) == "i");

			/* Replace */
			code_edit->clear();
			code_edit->insert_text_at_caret("item_1 test");
			code_edit->set_caret_column(2);
			code_edit->add_code_completion_option(CodeEdit::CodeCompletionKind::KIND_CLASS, "item_0.", "item_0");
			code_edit->update_code_completion_options();
			SEND_GUI_ACTION("ui_text_completion_replace");
			CHECK(code_edit->get_line(0) == "item_0 test");

			/* Replace string. */
			code_edit->clear();
			code_edit->insert_text_at_caret("\"item_1 test\"");
			code_edit->set_caret_column(2);
			code_edit->add_code_completion_option(CodeEdit::CodeCompletionKind::KIND_CLASS, "item_0.", "item_0");
			code_edit->update_code_completion_options();
			SEND_GUI_ACTION("ui_text_completion_replace");
			CHECK(code_edit->get_line(0) == "\"item_0\"");

			/* Normal replace if no end is given. */
			code_edit->clear();
			code_edit->insert_text_at_caret("\"item_1 test");
			code_edit->set_caret_column(2);
			code_edit->add_code_completion_option(CodeEdit::CodeCompletionKind::KIND_CLASS, "item_0.", "item_0");
			code_edit->update_code_completion_options();
			SEND_GUI_ACTION("ui_text_completion_replace");
			CHECK(code_edit->get_line(0) == "\"item_0\" test");

			/* Insert at completion. */
			code_edit->clear();
			code_edit->insert_text_at_caret("item_1 test");
			code_edit->set_caret_column(2);
			code_edit->add_code_completion_option(CodeEdit::CodeCompletionKind::KIND_CLASS, "item_0.", "item_0");
			code_edit->update_code_completion_options();
			SEND_GUI_ACTION("ui_text_completion_accept");
			CHECK(code_edit->get_line(0) == "item_01 test");

			/* Insert at completion with string should have same output. */
			code_edit->clear();
			code_edit->insert_text_at_caret("\"item_1 test\"");
			code_edit->set_caret_column(2);
			code_edit->add_code_completion_option(CodeEdit::CodeCompletionKind::KIND_CLASS, "item_0.", "item_0");
			code_edit->update_code_completion_options();
			SEND_GUI_ACTION("ui_text_completion_accept");
			CHECK(code_edit->get_line(0) == "\"item_0\"1 test\"");

			/* Merge symbol at end on insert text. */
			/* End on completion entry. */
			code_edit->clear();
			code_edit->insert_text_at_caret("item_1 test");
			code_edit->set_caret_column(2);
			code_edit->add_code_completion_option(CodeEdit::CodeCompletionKind::KIND_CLASS, "item_0(", "item_0(");
			code_edit->update_code_completion_options();
			SEND_GUI_ACTION("ui_text_completion_replace");
			CHECK(code_edit->get_line(0) == "item_0( test");
			CHECK(code_edit->get_caret_column() == 7);

			/* End of text*/
			code_edit->clear();
			code_edit->insert_text_at_caret("item_1( test");
			code_edit->set_caret_column(2);
			code_edit->add_code_completion_option(CodeEdit::CodeCompletionKind::KIND_CLASS, "item_0", "item_0");
			code_edit->update_code_completion_options();
			SEND_GUI_ACTION("ui_text_completion_replace");
			CHECK(code_edit->get_line(0) == "item_0( test");
			CHECK(code_edit->get_caret_column() == 6);

			/* End of both. */
			code_edit->clear();
			code_edit->insert_text_at_caret("item_1( test");
			code_edit->set_caret_column(2);
			code_edit->add_code_completion_option(CodeEdit::CodeCompletionKind::KIND_CLASS, "item_0(", "item_0(");
			code_edit->update_code_completion_options();
			SEND_GUI_ACTION("ui_text_completion_replace");
			CHECK(code_edit->get_line(0) == "item_0( test");
			CHECK(code_edit->get_caret_column() == 7);

			/* Full set. */
			/* End on completion entry. */
			code_edit->clear();
			code_edit->insert_text_at_caret("item_1 test");
			code_edit->set_caret_column(2);
			code_edit->add_code_completion_option(CodeEdit::CodeCompletionKind::KIND_CLASS, "item_0()", "item_0()");
			code_edit->update_code_completion_options();
			SEND_GUI_ACTION("ui_text_completion_replace");
			CHECK(code_edit->get_line(0) == "item_0() test");
			CHECK(code_edit->get_caret_column() == 8);

			/* End of text*/
			code_edit->clear();
			code_edit->insert_text_at_caret("item_1() test");
			code_edit->set_caret_column(2);
			code_edit->add_code_completion_option(CodeEdit::CodeCompletionKind::KIND_CLASS, "item_0", "item_0");
			code_edit->update_code_completion_options();
			SEND_GUI_ACTION("ui_text_completion_replace");
			CHECK(code_edit->get_line(0) == "item_0() test");
			CHECK(code_edit->get_caret_column() == 6);

			/* End of both. */
			code_edit->clear();
			code_edit->insert_text_at_caret("item_1() test");
			code_edit->set_caret_column(2);
			code_edit->add_code_completion_option(CodeEdit::CodeCompletionKind::KIND_CLASS, "item_0()", "item_0()");
			code_edit->update_code_completion_options();
			SEND_GUI_ACTION("ui_text_completion_replace");
			CHECK(code_edit->get_line(0) == "item_0() test");
			CHECK(code_edit->get_caret_column() == 8);

			/* Autobrace completion. */
			code_edit->set_auto_brace_completion_enabled(true);

			/* End on completion entry. */
			code_edit->clear();
			code_edit->insert_text_at_caret("item_1 test");
			code_edit->set_caret_column(2);
			code_edit->add_code_completion_option(CodeEdit::CodeCompletionKind::KIND_CLASS, "item_0(", "item_0(");
			code_edit->update_code_completion_options();
			SEND_GUI_ACTION("ui_text_completion_replace");
			CHECK(code_edit->get_line(0) == "item_0() test");
			CHECK(code_edit->get_caret_column() == 7);

			/* End of text*/
			code_edit->clear();
			code_edit->insert_text_at_caret("item_1( test");
			code_edit->set_caret_column(2);
			code_edit->add_code_completion_option(CodeEdit::CodeCompletionKind::KIND_CLASS, "item_0", "item_0");
			code_edit->update_code_completion_options();
			SEND_GUI_ACTION("ui_text_completion_replace");
			CHECK(code_edit->get_line(0) == "item_0( test");
			CHECK(code_edit->get_caret_column() == 6);

			/* End of both. */
			code_edit->clear();
			code_edit->insert_text_at_caret("item_1( test");
			code_edit->set_caret_column(2);
			code_edit->add_code_completion_option(CodeEdit::CodeCompletionKind::KIND_CLASS, "item_0(", "item_0(");
			code_edit->update_code_completion_options();
			SEND_GUI_ACTION("ui_text_completion_replace");
			CHECK(code_edit->get_line(0) == "item_0( test");
			CHECK(code_edit->get_caret_column() == 7);

			/* Full set. */
			/* End on completion entry. */
			code_edit->clear();
			code_edit->insert_text_at_caret("item_1 test");
			code_edit->set_caret_column(2);
			code_edit->add_code_completion_option(CodeEdit::CodeCompletionKind::KIND_CLASS, "item_0()", "item_0()");
			code_edit->update_code_completion_options();
			SEND_GUI_ACTION("ui_text_completion_replace");
			CHECK(code_edit->get_line(0) == "item_0() test");
			CHECK(code_edit->get_caret_column() == 8);

			/* End of text*/
			code_edit->clear();
			code_edit->insert_text_at_caret("item_1() test");
			code_edit->set_caret_column(2);
			code_edit->add_code_completion_option(CodeEdit::CodeCompletionKind::KIND_CLASS, "item_0", "item_0");
			code_edit->update_code_completion_options();
			SEND_GUI_ACTION("ui_text_completion_replace");
			CHECK(code_edit->get_line(0) == "item_0() test");
			CHECK(code_edit->get_caret_column() == 6);

			/* End of both. */
			code_edit->clear();
			code_edit->insert_text_at_caret("item_1() test");
			code_edit->set_caret_column(2);
			code_edit->add_code_completion_option(CodeEdit::CodeCompletionKind::KIND_CLASS, "item_0()", "item_0()");
			code_edit->update_code_completion_options();
			SEND_GUI_ACTION("ui_text_completion_replace");
			CHECK(code_edit->get_line(0) == "item_0() test");
			CHECK(code_edit->get_caret_column() == 8);
		}
	}

	SUBCASE("[CodeEdit] autocomplete suggestion order") {
		/* Prefer less fragmented suggestion. */
		code_edit->clear();
		code_edit->insert_text_at_caret("te");
		code_edit->set_caret_column(2);
		code_edit->add_code_completion_option(CodeEdit::CodeCompletionKind::KIND_VARIABLE, "test", "test");
		code_edit->add_code_completion_option(CodeEdit::CodeCompletionKind::KIND_VARIABLE, "tset", "tset");
		code_edit->update_code_completion_options();
		code_edit->confirm_code_completion();
		CHECK(code_edit->get_line(0) == "test");

		/* Prefer suggestion starting with the string to complete (matching start). */
		code_edit->clear();
		code_edit->insert_text_at_caret("te");
		code_edit->set_caret_column(2);
		code_edit->add_code_completion_option(CodeEdit::CodeCompletionKind::KIND_VARIABLE, "test", "test");
		code_edit->add_code_completion_option(CodeEdit::CodeCompletionKind::KIND_VARIABLE, "stest", "stest");
		code_edit->update_code_completion_options();
		code_edit->confirm_code_completion();
		CHECK(code_edit->get_line(0) == "test");

		/* Prefer less fragment over matching start. */
		code_edit->clear();
		code_edit->insert_text_at_caret("te");
		code_edit->set_caret_column(2);
		code_edit->add_code_completion_option(CodeEdit::CodeCompletionKind::KIND_VARIABLE, "tset", "tset");
		code_edit->add_code_completion_option(CodeEdit::CodeCompletionKind::KIND_VARIABLE, "stest", "stest");
		code_edit->update_code_completion_options();
		code_edit->confirm_code_completion();
		CHECK(code_edit->get_line(0) == "stest");

		/* Prefer good capitalization. */
		code_edit->clear();
		code_edit->insert_text_at_caret("te");
		code_edit->set_caret_column(2);
		code_edit->add_code_completion_option(CodeEdit::CodeCompletionKind::KIND_VARIABLE, "test", "test");
		code_edit->add_code_completion_option(CodeEdit::CodeCompletionKind::KIND_VARIABLE, "Test", "Test");
		code_edit->update_code_completion_options();
		code_edit->confirm_code_completion();
		CHECK(code_edit->get_line(0) == "test");

		/* Prefer matching start over good capitalization. */
		code_edit->clear();
		code_edit->insert_text_at_caret("te");
		code_edit->set_caret_column(2);
		code_edit->add_code_completion_option(CodeEdit::CodeCompletionKind::KIND_VARIABLE, "Test", "Test");
		code_edit->add_code_completion_option(CodeEdit::CodeCompletionKind::KIND_VARIABLE, "stest_bis", "test_bis");
		code_edit->update_code_completion_options();
		code_edit->confirm_code_completion();
		CHECK(code_edit->get_line(0) == "Test");

		/* Prefer closer location. */
		code_edit->clear();
		code_edit->insert_text_at_caret("te");
		code_edit->set_caret_column(2);
		code_edit->add_code_completion_option(CodeEdit::CodeCompletionKind::KIND_VARIABLE, "test", "test");
		code_edit->add_code_completion_option(CodeEdit::CodeCompletionKind::KIND_VARIABLE, "test_bis", "test_bis", Color(1, 1, 1), Ref<Resource>(), Variant::NIL, CodeEdit::LOCATION_LOCAL);
		code_edit->update_code_completion_options();
		code_edit->confirm_code_completion();
		CHECK(code_edit->get_line(0) == "test_bis");

		/* Prefer good capitalization over location. */
		code_edit->clear();
		code_edit->insert_text_at_caret("te");
		code_edit->set_caret_column(2);
		code_edit->add_code_completion_option(CodeEdit::CodeCompletionKind::KIND_VARIABLE, "test", "test");
		code_edit->add_code_completion_option(CodeEdit::CodeCompletionKind::KIND_VARIABLE, "Test", "Test", Color(1, 1, 1), Ref<Resource>(), Variant::NIL, CodeEdit::LOCATION_LOCAL);
		code_edit->update_code_completion_options();
		code_edit->confirm_code_completion();
		CHECK(code_edit->get_line(0) == "test");

		/* Prefer the start of the string to complete being closest to the start of the suggestion (closest to start). */
		code_edit->clear();
		code_edit->insert_text_at_caret("te");
		code_edit->set_caret_column(2);
		code_edit->add_code_completion_option(CodeEdit::CodeCompletionKind::KIND_VARIABLE, "stest", "stest");
		code_edit->add_code_completion_option(CodeEdit::CodeCompletionKind::KIND_VARIABLE, "sstest", "sstest");
		code_edit->update_code_completion_options();
		code_edit->confirm_code_completion();
		CHECK(code_edit->get_line(0) == "stest");

		/* Prefer location over closest to start. */
		code_edit->clear();
		code_edit->insert_text_at_caret("te");
		code_edit->set_caret_column(2);
		code_edit->add_code_completion_option(CodeEdit::CodeCompletionKind::KIND_VARIABLE, "stest", "stest");
		code_edit->add_code_completion_option(CodeEdit::CodeCompletionKind::KIND_VARIABLE, "sstest", "sstest", Color(1, 1, 1), Ref<Resource>(), Variant::NIL, CodeEdit::LOCATION_LOCAL);
		code_edit->update_code_completion_options();
		code_edit->confirm_code_completion();
		CHECK(code_edit->get_line(0) == "sstest");
	}

	SUBCASE("[CodeEdit] autocomplete currently selected option") {
		code_edit->set_code_completion_enabled(true);
		REQUIRE(code_edit->is_code_completion_enabled());

		// Initially select item 0.
		code_edit->insert_text_at_caret("te");
		code_edit->set_caret_column(2);
		code_edit->add_code_completion_option(CodeEdit::CodeCompletionKind::KIND_VARIABLE, "te1", "te1");
		code_edit->add_code_completion_option(CodeEdit::CodeCompletionKind::KIND_VARIABLE, "te3", "te3");
		code_edit->update_code_completion_options();
		CHECK_MESSAGE(code_edit->get_code_completion_selected_index() == 0, "Initially selected item should be 0.");

		// After adding later options shouldn't update selection.
		code_edit->add_code_completion_option(CodeEdit::CodeCompletionKind::KIND_VARIABLE, "te1", "te1");
		code_edit->add_code_completion_option(CodeEdit::CodeCompletionKind::KIND_VARIABLE, "te3", "te3");
		code_edit->add_code_completion_option(CodeEdit::CodeCompletionKind::KIND_VARIABLE, "te4", "te4"); // Added te4.
		code_edit->update_code_completion_options();
		CHECK_MESSAGE(code_edit->get_code_completion_selected_index() == 0, "Adding later options shouldn't update selection.");

		code_edit->set_code_completion_selected_index(2);
		code_edit->add_code_completion_option(CodeEdit::CodeCompletionKind::KIND_VARIABLE, "te1", "te1");
		code_edit->add_code_completion_option(CodeEdit::CodeCompletionKind::KIND_VARIABLE, "te3", "te3");
		code_edit->add_code_completion_option(CodeEdit::CodeCompletionKind::KIND_VARIABLE, "te4", "te4");
		code_edit->add_code_completion_option(CodeEdit::CodeCompletionKind::KIND_VARIABLE, "te5", "te5"); // Added te5.
		code_edit->add_code_completion_option(CodeEdit::CodeCompletionKind::KIND_VARIABLE, "te6", "te6"); // Added te6.
		code_edit->update_code_completion_options();
		CHECK_MESSAGE(code_edit->get_code_completion_selected_index() == 2, "Adding later options shouldn't update selection.");

		// Removing elements after selected element shouldn't update selection.
		code_edit->add_code_completion_option(CodeEdit::CodeCompletionKind::KIND_VARIABLE, "te1", "te1");
		code_edit->add_code_completion_option(CodeEdit::CodeCompletionKind::KIND_VARIABLE, "te3", "te3");
		code_edit->add_code_completion_option(CodeEdit::CodeCompletionKind::KIND_VARIABLE, "te4", "te4");
		code_edit->add_code_completion_option(CodeEdit::CodeCompletionKind::KIND_VARIABLE, "te5", "te5"); // Removed te6.
		code_edit->update_code_completion_options();
		CHECK_MESSAGE(code_edit->get_code_completion_selected_index() == 2, "Removing elements after selected element shouldn't update selection.");

		// Changing elements after selected element shouldn't update selection.
		code_edit->add_code_completion_option(CodeEdit::CodeCompletionKind::KIND_VARIABLE, "te1", "te1");
		code_edit->add_code_completion_option(CodeEdit::CodeCompletionKind::KIND_VARIABLE, "te3", "te3");
		code_edit->add_code_completion_option(CodeEdit::CodeCompletionKind::KIND_VARIABLE, "te4", "te4");
		code_edit->add_code_completion_option(CodeEdit::CodeCompletionKind::KIND_VARIABLE, "te6", "te6"); // Changed te5->te6.
		code_edit->update_code_completion_options();
		CHECK_MESSAGE(code_edit->get_code_completion_selected_index() == 2, "Changing elements after selected element shouldn't update selection.");

		// Changing elements before selected element should reset selection.
		code_edit->add_code_completion_option(CodeEdit::CodeCompletionKind::KIND_VARIABLE, "te2", "te2"); // Changed te1->te2.
		code_edit->add_code_completion_option(CodeEdit::CodeCompletionKind::KIND_VARIABLE, "te3", "te3");
		code_edit->add_code_completion_option(CodeEdit::CodeCompletionKind::KIND_VARIABLE, "te4", "te4");
		code_edit->add_code_completion_option(CodeEdit::CodeCompletionKind::KIND_VARIABLE, "te6", "te6");
		code_edit->update_code_completion_options();
		CHECK_MESSAGE(code_edit->get_code_completion_selected_index() == 0, "Changing elements before selected element should reset selection.");

		// Removing elements before selected element should reset selection.
		code_edit->set_code_completion_selected_index(2);
		code_edit->add_code_completion_option(CodeEdit::CodeCompletionKind::KIND_VARIABLE, "te3", "te3"); // Removed te2.
		code_edit->add_code_completion_option(CodeEdit::CodeCompletionKind::KIND_VARIABLE, "te4", "te4");
		code_edit->add_code_completion_option(CodeEdit::CodeCompletionKind::KIND_VARIABLE, "te6", "te6");
		code_edit->update_code_completion_options();
		CHECK_MESSAGE(code_edit->get_code_completion_selected_index() == 0, "Removing elements before selected element should reset selection.");
	}

	memdelete(code_edit);
}

TEST_CASE("[SceneTree][CodeEdit] symbol lookup") {
	CodeEdit *code_edit = memnew(CodeEdit);
	SceneTree::get_singleton()->get_root()->add_child(code_edit);
	code_edit->grab_focus();

	code_edit->set_symbol_lookup_on_click_enabled(true);
	CHECK(code_edit->is_symbol_lookup_on_click_enabled());

	if (TS->has_feature(TextServer::FEATURE_FONT_DYNAMIC) && TS->has_feature(TextServer::FEATURE_SIMPLE_LAYOUT)) {
		/* Set size for mouse input. */
		code_edit->set_size(Size2(100, 100));

		code_edit->set_text("this is some text");

		Point2 caret_pos = code_edit->get_caret_draw_pos();
		caret_pos.x += 60;
		SEND_GUI_MOUSE_BUTTON_EVENT(caret_pos, MouseButton::NONE, 0, Key::NONE);
		CHECK(code_edit->get_text_for_symbol_lookup() == "this is s" + String::chr(0xFFFF) + "ome text");

		SIGNAL_WATCH(code_edit, "symbol_validate");

#ifdef MACOS_ENABLED
		SEND_GUI_KEY_EVENT(Key::META);
#else
		SEND_GUI_KEY_EVENT(Key::CTRL);
#endif

		Array signal_args;
		Array arg;
		arg.push_back("some");
		signal_args.push_back(arg);
		SIGNAL_CHECK("symbol_validate", signal_args);

		SIGNAL_UNWATCH(code_edit, "symbol_validate");
	}

	memdelete(code_edit);
}

TEST_CASE("[SceneTree][CodeEdit] line length guidelines") {
	CodeEdit *code_edit = memnew(CodeEdit);
	SceneTree::get_singleton()->get_root()->add_child(code_edit);
	code_edit->grab_focus();

	TypedArray<int> guide_lines;

	code_edit->set_line_length_guidelines(guide_lines);
	CHECK(code_edit->get_line_length_guidelines().size() == 0);

	guide_lines.push_back(80);
	guide_lines.push_back(120);

	/* Order should be preserved. */
	code_edit->set_line_length_guidelines(guide_lines);
	CHECK((int)code_edit->get_line_length_guidelines()[0] == 80);
	CHECK((int)code_edit->get_line_length_guidelines()[1] == 120);

	memdelete(code_edit);
}

TEST_CASE("[SceneTree][CodeEdit] Backspace delete") {
	CodeEdit *code_edit = memnew(CodeEdit);
	SceneTree::get_singleton()->get_root()->add_child(code_edit);
	code_edit->grab_focus();

	/* Backspace with selection on first line. */
	code_edit->set_text("");
	code_edit->insert_text_at_caret("test backspace");
	code_edit->select(0, 0, 0, 5);
	code_edit->backspace();
	CHECK(code_edit->get_line(0) == "backspace");

	/* Backspace with selection on first line and caret at the beginning of file. */
	code_edit->set_text("");
	code_edit->insert_text_at_caret("test backspace");
	code_edit->select(0, 0, 0, 5);
	code_edit->set_caret_column(0);
	code_edit->backspace();
	CHECK(code_edit->get_line(0) == "backspace");

	/* Move caret up to the previous line on backspace if caret is at the first column. */
	code_edit->set_text("");
	code_edit->insert_text_at_caret("line 1\nline 2");
	code_edit->set_caret_line(1);
	code_edit->set_caret_column(0);
	code_edit->backspace();
	CHECK(code_edit->get_line(0) == "line 1line 2");
	CHECK(code_edit->get_caret_line() == 0);
	CHECK(code_edit->get_caret_column() == 6);

	/* Backspace delete all text if all text is selected. */
	code_edit->set_text("");
	code_edit->insert_text_at_caret("line 1\nline 2\nline 3");
	code_edit->select_all();
	code_edit->backspace();
	CHECK(code_edit->get_text().is_empty());

	/* Backspace at the beginning without selection has no effect. */
	code_edit->set_text("");
	code_edit->insert_text_at_caret("line 1\nline 2\nline 3");
	code_edit->set_caret_line(0);
	code_edit->set_caret_column(0);
	code_edit->backspace();
	CHECK(code_edit->get_text() == "line 1\nline 2\nline 3");

	memdelete(code_edit);
}

TEST_CASE("[SceneTree][CodeEdit] New Line") {
	CodeEdit *code_edit = memnew(CodeEdit);
	SceneTree::get_singleton()->get_root()->add_child(code_edit);
	code_edit->grab_focus();

	/* Add a new line. */
	code_edit->set_text("");
	code_edit->insert_text_at_caret("test new line");
	code_edit->set_caret_line(0);
	code_edit->set_caret_column(13);
	SEND_GUI_ACTION("ui_text_newline");
	CHECK(code_edit->get_line(0) == "test new line");
	CHECK(code_edit->get_line(1) == "");

	/* Split line with new line. */
	code_edit->set_text("");
	code_edit->insert_text_at_caret("test new line");
	code_edit->set_caret_line(0);
	code_edit->set_caret_column(5);
	SEND_GUI_ACTION("ui_text_newline");
	CHECK(code_edit->get_line(0) == "test ");
	CHECK(code_edit->get_line(1) == "new line");

	/* Delete selection and split with new line. */
	code_edit->set_text("");
	code_edit->insert_text_at_caret("test new line");
	code_edit->select(0, 0, 0, 5);
	SEND_GUI_ACTION("ui_text_newline");
	CHECK(code_edit->get_line(0) == "");
	CHECK(code_edit->get_line(1) == "new line");

	/* Blank new line below with selection should not split. */
	code_edit->set_text("");
	code_edit->insert_text_at_caret("test new line");
	code_edit->select(0, 0, 0, 5);
	SEND_GUI_ACTION("ui_text_newline_blank");
	CHECK(code_edit->get_line(0) == "test new line");
	CHECK(code_edit->get_line(1) == "");

	/* Blank new line above with selection should not split. */
	code_edit->set_text("");
	code_edit->insert_text_at_caret("test new line");
	code_edit->select(0, 0, 0, 5);
	SEND_GUI_ACTION("ui_text_newline_above");
	CHECK(code_edit->get_line(0) == "");
	CHECK(code_edit->get_line(1) == "test new line");

	memdelete(code_edit);
}

TEST_CASE("[SceneTree][CodeEdit] Duplicate Lines") {
	CodeEdit *code_edit = memnew(CodeEdit);
	SceneTree::get_singleton()->get_root()->add_child(code_edit);
	code_edit->grab_focus();

	code_edit->set_text(R"(extends Node

func _ready():
	var a := len(OS.get_cmdline_args())
	var b := get_child_count()
	var c := a + b
	for i in range(c):
		print("This is the solution: ", sin(i))
	var pos = get_index() - 1
	print("Make sure this exits: %b" % pos)
)");

	/* Duplicate a single line without selection. */
	code_edit->set_caret_line(0);
	code_edit->duplicate_lines();
	CHECK(code_edit->get_line(0) == "extends Node");
	CHECK(code_edit->get_line(1) == "extends Node");
	CHECK(code_edit->get_line(2) == "");

	/* Duplicate multiple lines with selection. */
	code_edit->set_caret_line(6);
	code_edit->set_caret_column(15);
	code_edit->select(4, 8, 6, 15);
	code_edit->duplicate_lines();
	CHECK(code_edit->get_line(6) == "\tvar c := a + b");
	CHECK(code_edit->get_line(7) == "\tvar a := len(OS.get_cmdline_args())");
	CHECK(code_edit->get_line(8) == "\tvar b := get_child_count()");
	CHECK(code_edit->get_line(9) == "\tvar c := a + b");
	CHECK(code_edit->get_line(10) == "\tfor i in range(c):");

	/* Duplicate single lines with multiple carets. */
	code_edit->deselect();
	code_edit->set_caret_line(10);
	code_edit->set_caret_column(1);
	code_edit->add_caret(11, 2);
	code_edit->add_caret(12, 1);
	code_edit->duplicate_lines();
	CHECK(code_edit->get_line(9) == "\tvar c := a + b");
	CHECK(code_edit->get_line(10) == "\tfor i in range(c):");
	CHECK(code_edit->get_line(11) == "\tfor i in range(c):");
	CHECK(code_edit->get_line(12) == "\t\tprint(\"This is the solution: \", sin(i))");
	CHECK(code_edit->get_line(13) == "\t\tprint(\"This is the solution: \", sin(i))");
	CHECK(code_edit->get_line(14) == "\tvar pos = get_index() - 1");
	CHECK(code_edit->get_line(15) == "\tvar pos = get_index() - 1");
	CHECK(code_edit->get_line(16) == "\tprint(\"Make sure this exits: %b\" % pos)");

	/* Duplicate multiple lines with multiple carets. */
	code_edit->select(0, 0, 1, 2, 0);
	code_edit->select(3, 0, 4, 2, 1);
	code_edit->select(16, 0, 17, 0, 2);
	code_edit->set_caret_line(1, false, true, 0, 0);
	code_edit->set_caret_column(2, false, 0);
	code_edit->set_caret_line(4, false, true, 0, 1);
	code_edit->set_caret_column(2, false, 1);
	code_edit->set_caret_line(17, false, true, 0, 2);
	code_edit->set_caret_column(0, false, 2);
	code_edit->duplicate_lines();
	CHECK(code_edit->get_line(1) == "extends Node");
	CHECK(code_edit->get_line(2) == "extends Node");
	CHECK(code_edit->get_line(3) == "extends Node");
	CHECK(code_edit->get_line(4) == "");
	CHECK(code_edit->get_line(6) == "\tvar a := len(OS.get_cmdline_args())");
	CHECK(code_edit->get_line(7) == "func _ready():");
	CHECK(code_edit->get_line(8) == "\tvar a := len(OS.get_cmdline_args())");
	CHECK(code_edit->get_line(9) == "\tvar b := get_child_count()");
	CHECK(code_edit->get_line(20) == "\tprint(\"Make sure this exits: %b\" % pos)");
	CHECK(code_edit->get_line(21) == "");
	CHECK(code_edit->get_line(22) == "\tprint(\"Make sure this exits: %b\" % pos)");
	CHECK(code_edit->get_line(23) == "");

	memdelete(code_edit);
}

} // namespace TestCodeEdit

#endif // TEST_CODE_EDIT_H
