/*************************************************************************/
/*  test_code_edit.h                                                     */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2021 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2021 Godot Engine contributors (cf. AUTHORS.md).   */
/*                                                                       */
/* Permission is hereby granted, free of charge, to any person obtaining */
/* a copy of this software and associated documentation files (the       */
/* "Software"), to deal in the Software without restriction, including   */
/* without limitation the rights to use, copy, modify, merge, publish,   */
/* distribute, sublicense, and/or sell copies of the Software, and to    */
/* permit persons to whom the Software is furnished to do so, subject to */
/* the following conditions:                                             */
/*                                                                       */
/* The above copyright notice and this permission notice shall be        */
/* included in all copies or substantial portions of the Software.       */
/*                                                                       */
/* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,       */
/* EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF    */
/* MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.*/
/* IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY  */
/* CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,  */
/* TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE     */
/* SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.                */
/*************************************************************************/

#ifndef TEST_CODE_EDIT_H
#define TEST_CODE_EDIT_H

#include "scene/gui/code_edit.h"

#include "tests/test_macros.h"

namespace TestCodeEdit {

TEST_CASE("[SceneTree][CodeEdit] line gutters") {
	CodeEdit *code_edit = memnew(CodeEdit);
	SceneTree::get_singleton()->get_root()->add_child(code_edit);

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
			CHECK(code_edit->get_breakpointed_lines()[0] == Variant(0));
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

			SEND_GUI_ACTION(code_edit, "ui_text_newline");
			CHECK(code_edit->get_line_count() == 2);
			CHECK_FALSE(code_edit->is_line_breakpointed(0));
			CHECK(code_edit->is_line_breakpointed(1));
			SIGNAL_CHECK("breakpoint_toggled", args);

			/* Non-Breaking. */
			((Array)args[0])[0] = 1;
			((Array)args[1])[0] = 2;
			SEND_GUI_ACTION(code_edit, "ui_text_newline_blank");
			CHECK(code_edit->get_line_count() == 3);
			CHECK_FALSE(code_edit->is_line_breakpointed(1));
			CHECK(code_edit->is_line_breakpointed(2));
			SIGNAL_CHECK("breakpoint_toggled", args);

			/* Above. */
			((Array)args[0])[0] = 2;
			((Array)args[1])[0] = 3;
			SEND_GUI_ACTION(code_edit, "ui_text_newline_above");
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
			SEND_GUI_ACTION(code_edit, "ui_text_newline");
			CHECK(code_edit->get_line_count() == 2);
			CHECK(code_edit->is_line_breakpointed(0));
			CHECK_FALSE(code_edit->is_line_breakpointed(1));
			SIGNAL_CHECK_FALSE("breakpoint_toggled");

			/* Non-Breaking. */
			code_edit->set_caret_line(0);
			SEND_GUI_ACTION(code_edit, "ui_text_newline_blank");
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
			SEND_GUI_ACTION(code_edit, "ui_text_newline_above");
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
			SEND_GUI_ACTION(code_edit, "ui_text_backspace");
			CHECK(code_edit->is_line_breakpointed(1));
			SIGNAL_CHECK_FALSE("breakpoint_toggled");

			/* backspace on breakpointed line removes it */
			SEND_GUI_ACTION(code_edit, "ui_text_backspace");
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
			SEND_GUI_ACTION(code_edit, "ui_text_backspace");
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
			SEND_GUI_ACTION(code_edit, "ui_text_delete");
			CHECK(code_edit->get_line_count() == 2);
			CHECK(code_edit->is_line_breakpointed(1));
			SIGNAL_CHECK_FALSE("breakpoint_toggled");

			/* Delete moving breakpointed line up removes it. */
			code_edit->set_caret_line(0);
			SEND_GUI_ACTION(code_edit, "ui_text_delete");
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
			SEND_GUI_ACTION(code_edit, "ui_text_delete");
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
			SEND_GUI_ACTION(code_edit, "ui_text_newline");
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
			CHECK(code_edit->get_bookmarked_lines()[0] == Variant(0));
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
			SEND_GUI_ACTION(code_edit, "ui_text_newline");
			CHECK(code_edit->get_line_count() == 2);
			CHECK_FALSE(code_edit->is_line_bookmarked(0));
			CHECK(code_edit->is_line_bookmarked(1));

			/* Non-Breaking. */
			SEND_GUI_ACTION(code_edit, "ui_text_newline_blank");
			CHECK(code_edit->get_line_count() == 3);
			CHECK_FALSE(code_edit->is_line_bookmarked(1));
			CHECK(code_edit->is_line_bookmarked(2));

			/* Above. */
			SEND_GUI_ACTION(code_edit, "ui_text_newline_above");
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
			SEND_GUI_ACTION(code_edit, "ui_text_newline");
			CHECK(code_edit->get_line_count() == 2);
			CHECK(code_edit->is_line_bookmarked(0));
			CHECK_FALSE(code_edit->is_line_bookmarked(1));

			/* Non-Breaking. */
			code_edit->set_caret_line(0);
			SEND_GUI_ACTION(code_edit, "ui_text_newline_blank");
			CHECK(code_edit->get_line_count() == 3);
			CHECK(code_edit->is_line_bookmarked(0));
			CHECK_FALSE(code_edit->is_line_bookmarked(1));

			/* Above does move. */
			code_edit->set_caret_line(0);
			SEND_GUI_ACTION(code_edit, "ui_text_newline_above");
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
			SEND_GUI_ACTION(code_edit, "ui_text_backspace");
			CHECK(code_edit->is_line_bookmarked(1));

			/* backspace on bookmarked line removes it */
			SEND_GUI_ACTION(code_edit, "ui_text_backspace");
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
			SEND_GUI_ACTION(code_edit, "ui_text_delete");
			CHECK(code_edit->get_line_count() == 2);
			CHECK(code_edit->is_line_bookmarked(1));

			/* Delete moving bookmarked line up removes it. */
			code_edit->set_caret_line(0);
			SEND_GUI_ACTION(code_edit, "ui_text_delete");
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
			CHECK(code_edit->get_executing_lines()[0] == Variant(0));
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
			SEND_GUI_ACTION(code_edit, "ui_text_newline");
			CHECK(code_edit->get_line_count() == 2);
			CHECK_FALSE(code_edit->is_line_executing(0));
			CHECK(code_edit->is_line_executing(1));

			/* Non-Breaking. */
			SEND_GUI_ACTION(code_edit, "ui_text_newline_blank");
			CHECK(code_edit->get_line_count() == 3);
			CHECK_FALSE(code_edit->is_line_executing(1));
			CHECK(code_edit->is_line_executing(2));

			/* Above. */
			SEND_GUI_ACTION(code_edit, "ui_text_newline_above");
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
			SEND_GUI_ACTION(code_edit, "ui_text_newline");
			CHECK(code_edit->get_line_count() == 2);
			CHECK(code_edit->is_line_executing(0));
			CHECK_FALSE(code_edit->is_line_executing(1));

			/* Non-Breaking. */
			code_edit->set_caret_line(0);
			SEND_GUI_ACTION(code_edit, "ui_text_newline_blank");
			CHECK(code_edit->get_line_count() == 3);
			CHECK(code_edit->is_line_executing(0));
			CHECK_FALSE(code_edit->is_line_executing(1));

			/* Above does move. */
			code_edit->set_caret_line(0);
			SEND_GUI_ACTION(code_edit, "ui_text_newline_above");
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
			SEND_GUI_ACTION(code_edit, "ui_text_backspace");
			CHECK(code_edit->is_line_executing(1));

			/* backspace on executing line removes it */
			SEND_GUI_ACTION(code_edit, "ui_text_backspace");
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
			SEND_GUI_ACTION(code_edit, "ui_text_delete");
			CHECK(code_edit->get_line_count() == 2);
			CHECK(code_edit->is_line_executing(1));

			/* Delete moving executing line up removes it. */
			code_edit->set_caret_line(0);
			SEND_GUI_ACTION(code_edit, "ui_text_delete");
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

			/* Check nested strings are handeled correctly. */
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

			/* Check is in string with no column retruns true if entire line is comment excluding whitespace. */
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

			/* Check nested comments are handeled correctly. */
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

			/* Check is in comment with no column retruns true if entire line is comment excluding whitespace. */
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

			/* Check is in string with no column retruns true if entire line is string excluding whitespace. */
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

			/* Check is in comment with no column retruns true if entire line is comment excluding whitespace. */
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

			/* Check is in comment with no column retruns true as inner delimiter should not be counted. */
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
		SEND_GUI_ACTION(code_edit, "ui_text_indent");
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
		SEND_GUI_ACTION(code_edit, "ui_text_indent");
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

		code_edit->do_unindent();
		CHECK(code_edit->get_line(0) == "\t");

		code_edit->unindent_lines();
		CHECK(code_edit->get_line(0) == "\t");

		code_edit->set_editable(true);

		/* Simple unindent. */
		code_edit->do_unindent();
		CHECK(code_edit->get_line(0) == "");

		/* Should inindent inplace. */
		code_edit->set_text("");
		code_edit->insert_text_at_caret("test\t");

		code_edit->do_unindent();
		CHECK(code_edit->get_line(0) == "test");

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
		code_edit->do_unindent();
		CHECK(code_edit->get_line(0) == "\ttest");

		/* Check input action. */
		code_edit->set_text("\t\ttest");
		SEND_GUI_ACTION(code_edit, "ui_text_dedent");
		CHECK(code_edit->get_line(0) == "\ttest");

		/* Selection does entire line. */
		code_edit->set_text("\t\ttest");
		code_edit->select_all();
		code_edit->do_unindent();
		CHECK(code_edit->get_line(0) == "\ttest");

		/* Handles multiple lines. */
		code_edit->set_text("\ttest\n\ttext");
		code_edit->select_all();
		code_edit->do_unindent();
		CHECK(code_edit->get_line(0) == "test");
		CHECK(code_edit->get_line(1) == "text");

		/* Do not unindent line if last col is zero. */
		code_edit->set_text("\ttest\n\ttext");
		code_edit->select(0, 0, 1, 0);
		code_edit->do_unindent();
		CHECK(code_edit->get_line(0) == "test");
		CHECK(code_edit->get_line(1) == "\ttext");

		/* Unindent even if last column of first line. */
		code_edit->set_text("\ttest\n\ttext");
		code_edit->select(0, 5, 1, 1);
		code_edit->do_unindent();
		CHECK(code_edit->get_line(0) == "test");
		CHECK(code_edit->get_line(1) == "text");

		/* Check selection is adjusted. */
		code_edit->set_text("\ttest");
		code_edit->select(0, 1, 0, 2);
		code_edit->do_unindent();
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

		code_edit->do_unindent();
		CHECK(code_edit->get_line(0) == "    ");

		code_edit->unindent_lines();
		CHECK(code_edit->get_line(0) == "    ");

		code_edit->set_editable(true);

		/* Simple unindent. */
		code_edit->do_unindent();
		CHECK(code_edit->get_line(0) == "");

		/* Should inindent inplace. */
		code_edit->set_text("");
		code_edit->insert_text_at_caret("test    ");

		code_edit->do_unindent();
		CHECK(code_edit->get_line(0) == "test");

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
		code_edit->do_unindent();
		CHECK(code_edit->get_line(0) == "    test");

		/* Only as far as needed */
		code_edit->set_text("       test");
		code_edit->do_unindent();
		CHECK(code_edit->get_line(0) == "    test");

		/* Check input action. */
		code_edit->set_text("        test");
		SEND_GUI_ACTION(code_edit, "ui_text_dedent");
		CHECK(code_edit->get_line(0) == "    test");

		/* Selection does entire line. */
		code_edit->set_text("        test");
		code_edit->select_all();
		code_edit->do_unindent();
		CHECK(code_edit->get_line(0) == "    test");

		/* Handles multiple lines. */
		code_edit->set_text("    test\n    text");
		code_edit->select_all();
		code_edit->do_unindent();
		CHECK(code_edit->get_line(0) == "test");
		CHECK(code_edit->get_line(1) == "text");

		/* Do not unindent line if last col is zero. */
		code_edit->set_text("    test\n    text");
		code_edit->select(0, 0, 1, 0);
		code_edit->do_unindent();
		CHECK(code_edit->get_line(0) == "test");
		CHECK(code_edit->get_line(1) == "    text");

		/* Unindent even if last column of first line. */
		code_edit->set_text("    test\n    text");
		code_edit->select(0, 5, 1, 1);
		code_edit->do_unindent();
		CHECK(code_edit->get_line(0) == "test");
		CHECK(code_edit->get_line(1) == "text");

		/* Check selection is adjusted. */
		code_edit->set_text("    test");
		code_edit->select(0, 4, 0, 5);
		code_edit->do_unindent();
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
			SEND_GUI_ACTION(code_edit, "ui_text_newline");
			CHECK(code_edit->get_line(0) == "test:");
			CHECK(code_edit->get_line(1) == "\t");

			/* new blank line should still indent. */
			code_edit->set_text("");
			code_edit->insert_text_at_caret("test:");
			SEND_GUI_ACTION(code_edit, "ui_text_newline_blank");
			CHECK(code_edit->get_line(0) == "test:");
			CHECK(code_edit->get_line(1) == "\t");

			/* new line above should not indent. */
			code_edit->set_text("");
			code_edit->insert_text_at_caret("test:");
			SEND_GUI_ACTION(code_edit, "ui_text_newline_above");
			CHECK(code_edit->get_line(0) == "");
			CHECK(code_edit->get_line(1) == "test:");

			/* Whitespace between symbol and caret is okay. */
			code_edit->set_text("");
			code_edit->insert_text_at_caret("test:  ");
			SEND_GUI_ACTION(code_edit, "ui_text_newline");
			CHECK(code_edit->get_line(0) == "test:  ");
			CHECK(code_edit->get_line(1) == "\t");

			/* Comment between symbol and caret is okay. */
			code_edit->add_comment_delimiter("#", "");
			code_edit->set_text("");
			code_edit->insert_text_at_caret("test: # comment");
			SEND_GUI_ACTION(code_edit, "ui_text_newline");
			CHECK(code_edit->get_line(0) == "test: # comment");
			CHECK(code_edit->get_line(1) == "\t");
			code_edit->remove_comment_delimiter("#");

			/* Strings between symbol and caret are not okay. */
			code_edit->add_string_delimiter("#", "");
			code_edit->set_text("");
			code_edit->insert_text_at_caret("test: # string");
			SEND_GUI_ACTION(code_edit, "ui_text_newline");
			CHECK(code_edit->get_line(0) == "test: # string");
			CHECK(code_edit->get_line(1) == "");
			code_edit->remove_comment_delimiter("#");

			/* If between brace pairs an extra line is added. */
			code_edit->set_text("");
			code_edit->insert_text_at_caret("test{}");
			code_edit->set_caret_column(5);
			SEND_GUI_ACTION(code_edit, "ui_text_newline");
			CHECK(code_edit->get_line(0) == "test{");
			CHECK(code_edit->get_line(1) == "\t");
			CHECK(code_edit->get_line(2) == "}");

			/* Except when we are going above. */
			code_edit->set_text("");
			code_edit->insert_text_at_caret("test{}");
			code_edit->set_caret_column(5);
			SEND_GUI_ACTION(code_edit, "ui_text_newline_above");
			CHECK(code_edit->get_line(0) == "");
			CHECK(code_edit->get_line(1) == "test{}");

			/* or below. */
			code_edit->set_text("");
			code_edit->insert_text_at_caret("test{}");
			code_edit->set_caret_column(5);
			SEND_GUI_ACTION(code_edit, "ui_text_newline_blank");
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
			SEND_GUI_ACTION(code_edit, "ui_text_newline");
			CHECK(code_edit->get_line(0) == "test:");
			CHECK(code_edit->get_line(1) == "    ");

			/* new blank line should still indent. */
			code_edit->set_text("");
			code_edit->insert_text_at_caret("test:");
			SEND_GUI_ACTION(code_edit, "ui_text_newline_blank");
			CHECK(code_edit->get_line(0) == "test:");
			CHECK(code_edit->get_line(1) == "    ");

			/* new line above should not indent. */
			code_edit->set_text("");
			code_edit->insert_text_at_caret("test:");
			SEND_GUI_ACTION(code_edit, "ui_text_newline_above");
			CHECK(code_edit->get_line(0) == "");
			CHECK(code_edit->get_line(1) == "test:");

			/* Whitespace between symbol and caret is okay. */
			code_edit->set_text("");
			code_edit->insert_text_at_caret("test:  ");
			SEND_GUI_ACTION(code_edit, "ui_text_newline");
			CHECK(code_edit->get_line(0) == "test:  ");
			CHECK(code_edit->get_line(1) == "    ");

			/* Comment between symbol and caret is okay. */
			code_edit->add_comment_delimiter("#", "");
			code_edit->set_text("");
			code_edit->insert_text_at_caret("test: # comment");
			SEND_GUI_ACTION(code_edit, "ui_text_newline");
			CHECK(code_edit->get_line(0) == "test: # comment");
			CHECK(code_edit->get_line(1) == "    ");
			code_edit->remove_comment_delimiter("#");

			/* Strings between symbol and caret are not okay. */
			code_edit->add_string_delimiter("#", "");
			code_edit->set_text("");
			code_edit->insert_text_at_caret("test: # string");
			SEND_GUI_ACTION(code_edit, "ui_text_newline");
			CHECK(code_edit->get_line(0) == "test: # string");
			CHECK(code_edit->get_line(1) == "");
			code_edit->remove_comment_delimiter("#");

			/* If between brace pairs an extra line is added. */
			code_edit->set_text("");
			code_edit->insert_text_at_caret("test{}");
			code_edit->set_caret_column(5);
			SEND_GUI_ACTION(code_edit, "ui_text_newline");
			CHECK(code_edit->get_line(0) == "test{");
			CHECK(code_edit->get_line(1) == "    ");
			CHECK(code_edit->get_line(2) == "}");

			/* Except when we are going above. */
			code_edit->set_text("");
			code_edit->insert_text_at_caret("test{}");
			code_edit->set_caret_column(5);
			SEND_GUI_ACTION(code_edit, "ui_text_newline_above");
			CHECK(code_edit->get_line(0) == "");
			CHECK(code_edit->get_line(1) == "test{}");

			/* or below. */
			code_edit->set_text("");
			code_edit->insert_text_at_caret("test{}");
			code_edit->set_caret_column(5);
			SEND_GUI_ACTION(code_edit, "ui_text_newline_blank");
			CHECK(code_edit->get_line(0) == "test{}");
			CHECK(code_edit->get_line(1) == "");
		}
	}

	memdelete(code_edit);
}

TEST_CASE("[SceneTree][CodeEdit] folding") {
	CodeEdit *code_edit = memnew(CodeEdit);
	SceneTree::get_singleton()->get_root()->add_child(code_edit);

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
		CHECK(code_edit->get_next_visible_line_offset_from(1, 1) == 6);

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
		code_edit->set_text("&line1\n\tline2&");
		CHECK(code_edit->can_fold_line(0));
		CHECK_FALSE(code_edit->can_fold_line(1));
		code_edit->fold_line(1);
		CHECK_FALSE(code_edit->is_line_folded(1));
		code_edit->fold_line(0);
		CHECK(code_edit->is_line_folded(0));
		CHECK_FALSE(code_edit->is_line_folded(1));
		CHECK(code_edit->get_next_visible_line_offset_from(1, 1) == 1);

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

		// Non-indented comments/ strings.
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

TEST_CASE("[SceneTree][CodeEdit] completion") {
	CodeEdit *code_edit = memnew(CodeEdit);
	SceneTree::get_singleton()->get_root()->add_child(code_edit);

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
		SEND_GUI_KEY_EVENT(code_edit, KEY_BRACKETLEFT);
		CHECK(code_edit->get_line(0) == "[]");

		/* Should first match and insert smaller key. */
		code_edit->clear();
		SEND_GUI_KEY_EVENT(code_edit, KEY_APOSTROPHE);
		CHECK(code_edit->get_line(0) == "''");
		CHECK(code_edit->get_caret_column() == 1);

		/* Move out from centre, Should match and insert larger key. */
		SEND_GUI_ACTION(code_edit, "ui_text_caret_right");
		SEND_GUI_KEY_EVENT(code_edit, KEY_APOSTROPHE);
		CHECK(code_edit->get_line(0) == "''''''");
		CHECK(code_edit->get_caret_column() == 3);

		/* Backspace should remove all. */
		SEND_GUI_ACTION(code_edit, "ui_text_backspace");
		CHECK(code_edit->get_line(0).is_empty());

		/* If in between and typing close key should "skip". */
		SEND_GUI_KEY_EVENT(code_edit, KEY_BRACKETLEFT);
		CHECK(code_edit->get_line(0) == "[]");
		CHECK(code_edit->get_caret_column() == 1);
		SEND_GUI_KEY_EVENT(code_edit, KEY_BRACKETRIGHT);
		CHECK(code_edit->get_line(0) == "[]");
		CHECK(code_edit->get_caret_column() == 2);

		/* If current is char and inserting a string, do not autocomplete. */
		code_edit->clear();
		SEND_GUI_KEY_EVENT(code_edit, KEY_A);
		SEND_GUI_KEY_EVENT(code_edit, KEY_APOSTROPHE);
		CHECK(code_edit->get_line(0) == "A'");

		/* If in comment, do not complete. */
		code_edit->add_comment_delimiter("#", "");
		code_edit->clear();
		SEND_GUI_KEY_EVENT(code_edit, KEY_NUMBERSIGN);
		SEND_GUI_KEY_EVENT(code_edit, KEY_APOSTROPHE);
		CHECK(code_edit->get_line(0) == "#'");

		/* If in string, and inserting string do not complete. */
		code_edit->clear();
		SEND_GUI_KEY_EVENT(code_edit, KEY_APOSTROPHE);
		SEND_GUI_KEY_EVENT(code_edit, KEY_QUOTEDBL);
		CHECK(code_edit->get_line(0) == "'\"'");
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
		SIGNAL_WATCH(code_edit, "request_code_completion");
		code_edit->set_code_completion_enabled(true);

		Array signal_args;
		signal_args.push_back(Array());

		/* Force request. */
		code_edit->request_code_completion();
		SIGNAL_CHECK_FALSE("request_code_completion");
		code_edit->request_code_completion(true);
		SIGNAL_CHECK("request_code_completion", signal_args);

		/* Manual request should force. */
		SEND_GUI_ACTION(code_edit, "ui_text_completion_query");
		SIGNAL_CHECK("request_code_completion", signal_args);

		/* Insert prefix. */
		TypedArray<String> completion_prefixes;
		completion_prefixes.push_back(".");
		code_edit->set_code_completion_prefixes(completion_prefixes);

		code_edit->insert_text_at_caret(".");
		code_edit->request_code_completion();
		SIGNAL_CHECK("request_code_completion", signal_args);

		/* Should work with space too. */
		code_edit->insert_text_at_caret(" ");
		code_edit->request_code_completion();
		SIGNAL_CHECK("request_code_completion", signal_args);

		/* Should work when complete ends with prefix. */
		code_edit->clear();
		code_edit->insert_text_at_caret("t");
		code_edit->add_code_completion_option(CodeEdit::CodeCompletionKind::KIND_CLASS, "test.", "test.");
		code_edit->update_code_completion_options();
		code_edit->confirm_code_completion();
		CHECK(code_edit->get_line(0) == "test.");
		SIGNAL_CHECK("request_code_completion", signal_args);

		SIGNAL_UNWATCH(code_edit, "request_code_completion");
	}

	SUBCASE("[CodeEdit] autocomplete completion") {
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
		code_edit->insert_text_at_caret("i");
		code_edit->update_code_completion_options();
		code_edit->add_code_completion_option(CodeEdit::CodeCompletionKind::KIND_CLASS, "item_0.", "item_0", Color(1, 0, 0), RES(), Color(1, 0, 0));
		code_edit->add_code_completion_option(CodeEdit::CodeCompletionKind::KIND_VARIABLE, "item_1.", "item_1");
		code_edit->add_code_completion_option(CodeEdit::CodeCompletionKind::KIND_VARIABLE, "item_2.", "item_2");

		ERR_PRINT_OFF;
		code_edit->set_code_completion_selected_index(1);
		ERR_PRINT_ON;
		CHECK(code_edit->get_code_completion_selected_index() == 0);
		CHECK(code_edit->get_code_completion_option(0).size() == 6);
		CHECK(code_edit->get_code_completion_options().size() == 1);

		/* Check cancel closes completion. */
		SEND_GUI_ACTION(code_edit, "ui_cancel");
		CHECK(code_edit->get_code_completion_selected_index() == -1);

		code_edit->update_code_completion_options();
		CHECK(code_edit->get_code_completion_selected_index() == 0);
		code_edit->set_code_completion_selected_index(1);
		CHECK(code_edit->get_code_completion_selected_index() == 1);
		CHECK(code_edit->get_code_completion_option(0).size() == 6);
		CHECK(code_edit->get_code_completion_options().size() == 3);

		/* Check data. */
		Dictionary option = code_edit->get_code_completion_option(0);
		CHECK((int)option["kind"] == (int)CodeEdit::CodeCompletionKind::KIND_CLASS);
		CHECK(option["display_text"] == "item_0.");
		CHECK(option["insert_text"] == "item_0");
		CHECK(option["font_color"] == Color(1, 0, 0));
		CHECK(option["icon"] == RES());
		CHECK(option["default_value"] == Color(1, 0, 0));

		/* Set size for mouse input. */
		code_edit->set_size(Size2(100, 100));

		/* Check input. */
		SEND_GUI_ACTION(code_edit, "ui_end");
		CHECK(code_edit->get_code_completion_selected_index() == 2);

		SEND_GUI_ACTION(code_edit, "ui_home");
		CHECK(code_edit->get_code_completion_selected_index() == 0);

		SEND_GUI_ACTION(code_edit, "ui_page_down");
		CHECK(code_edit->get_code_completion_selected_index() == 2);

		SEND_GUI_ACTION(code_edit, "ui_page_up");
		CHECK(code_edit->get_code_completion_selected_index() == 0);

		SEND_GUI_ACTION(code_edit, "ui_up");
		CHECK(code_edit->get_code_completion_selected_index() == 2);

		SEND_GUI_ACTION(code_edit, "ui_down");
		CHECK(code_edit->get_code_completion_selected_index() == 0);

		SEND_GUI_KEY_EVENT(code_edit, KEY_T);
		CHECK(code_edit->get_code_completion_selected_index() == 0);

		SEND_GUI_ACTION(code_edit, "ui_left");
		CHECK(code_edit->get_code_completion_selected_index() == 0);

		SEND_GUI_ACTION(code_edit, "ui_right");
		CHECK(code_edit->get_code_completion_selected_index() == 0);

		SEND_GUI_ACTION(code_edit, "ui_text_backspace");
		CHECK(code_edit->get_code_completion_selected_index() == 0);

		Point2 caret_pos = code_edit->get_caret_draw_pos();
		caret_pos.y -= code_edit->get_line_height();
		SEND_GUI_MOUSE_EVENT(code_edit, caret_pos, MOUSE_BUTTON_WHEEL_DOWN, MOUSE_BUTTON_NONE);
		CHECK(code_edit->get_code_completion_selected_index() == 1);

		SEND_GUI_MOUSE_EVENT(code_edit, caret_pos, MOUSE_BUTTON_WHEEL_UP, MOUSE_BUTTON_NONE);
		CHECK(code_edit->get_code_completion_selected_index() == 0);

		/* Single click selects. */
		SEND_GUI_MOUSE_EVENT(code_edit, caret_pos, MOUSE_BUTTON_LEFT, MOUSE_BUTTON_MASK_LEFT);
		CHECK(code_edit->get_code_completion_selected_index() == 2);

		/* Double click inserts. */
		SEND_GUI_DOUBLE_CLICK(code_edit, caret_pos);
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
		SEND_GUI_ACTION(code_edit, "ui_text_completion_replace");
		CHECK(code_edit->get_line(0) == "item_0 test");

		/* Replace string. */
		code_edit->clear();
		code_edit->insert_text_at_caret("\"item_1 test\"");
		code_edit->set_caret_column(2);
		code_edit->add_code_completion_option(CodeEdit::CodeCompletionKind::KIND_CLASS, "item_0.", "item_0");
		code_edit->update_code_completion_options();
		SEND_GUI_ACTION(code_edit, "ui_text_completion_replace");
		CHECK(code_edit->get_line(0) == "\"item_0\"");

		/* Normal replace if no end is given. */
		code_edit->clear();
		code_edit->insert_text_at_caret("\"item_1 test");
		code_edit->set_caret_column(2);
		code_edit->add_code_completion_option(CodeEdit::CodeCompletionKind::KIND_CLASS, "item_0.", "item_0");
		code_edit->update_code_completion_options();
		SEND_GUI_ACTION(code_edit, "ui_text_completion_replace");
		CHECK(code_edit->get_line(0) == "\"item_0\" test");

		/* Insert at completion. */
		code_edit->clear();
		code_edit->insert_text_at_caret("item_1 test");
		code_edit->set_caret_column(2);
		code_edit->add_code_completion_option(CodeEdit::CodeCompletionKind::KIND_CLASS, "item_0.", "item_0");
		code_edit->update_code_completion_options();
		SEND_GUI_ACTION(code_edit, "ui_text_completion_accept");
		CHECK(code_edit->get_line(0) == "item_01 test");

		/* Insert at completion with string should have same output. */
		code_edit->clear();
		code_edit->insert_text_at_caret("\"item_1 test\"");
		code_edit->set_caret_column(2);
		code_edit->add_code_completion_option(CodeEdit::CodeCompletionKind::KIND_CLASS, "item_0.", "item_0");
		code_edit->update_code_completion_options();
		SEND_GUI_ACTION(code_edit, "ui_text_completion_accept");
		CHECK(code_edit->get_line(0) == "\"item_0\"1 test\"");

		/* Merge symbol at end on insert text. */
		/* End on completion entry. */
		code_edit->clear();
		code_edit->insert_text_at_caret("item_1 test");
		code_edit->set_caret_column(2);
		code_edit->add_code_completion_option(CodeEdit::CodeCompletionKind::KIND_CLASS, "item_0(", "item_0(");
		code_edit->update_code_completion_options();
		SEND_GUI_ACTION(code_edit, "ui_text_completion_replace");
		CHECK(code_edit->get_line(0) == "item_0( test");
		CHECK(code_edit->get_caret_column() == 7);

		/* End of text*/
		code_edit->clear();
		code_edit->insert_text_at_caret("item_1( test");
		code_edit->set_caret_column(2);
		code_edit->add_code_completion_option(CodeEdit::CodeCompletionKind::KIND_CLASS, "item_0", "item_0");
		code_edit->update_code_completion_options();
		SEND_GUI_ACTION(code_edit, "ui_text_completion_replace");
		CHECK(code_edit->get_line(0) == "item_0( test");
		CHECK(code_edit->get_caret_column() == 6);

		/* End of both. */
		code_edit->clear();
		code_edit->insert_text_at_caret("item_1( test");
		code_edit->set_caret_column(2);
		code_edit->add_code_completion_option(CodeEdit::CodeCompletionKind::KIND_CLASS, "item_0(", "item_0(");
		code_edit->update_code_completion_options();
		SEND_GUI_ACTION(code_edit, "ui_text_completion_replace");
		CHECK(code_edit->get_line(0) == "item_0( test");
		CHECK(code_edit->get_caret_column() == 7);

		/* Full set. */
		/* End on completion entry. */
		code_edit->clear();
		code_edit->insert_text_at_caret("item_1 test");
		code_edit->set_caret_column(2);
		code_edit->add_code_completion_option(CodeEdit::CodeCompletionKind::KIND_CLASS, "item_0()", "item_0()");
		code_edit->update_code_completion_options();
		SEND_GUI_ACTION(code_edit, "ui_text_completion_replace");
		CHECK(code_edit->get_line(0) == "item_0() test");
		CHECK(code_edit->get_caret_column() == 8);

		/* End of text*/
		code_edit->clear();
		code_edit->insert_text_at_caret("item_1() test");
		code_edit->set_caret_column(2);
		code_edit->add_code_completion_option(CodeEdit::CodeCompletionKind::KIND_CLASS, "item_0", "item_0");
		code_edit->update_code_completion_options();
		SEND_GUI_ACTION(code_edit, "ui_text_completion_replace");
		CHECK(code_edit->get_line(0) == "item_0() test");
		CHECK(code_edit->get_caret_column() == 6);

		/* End of both. */
		code_edit->clear();
		code_edit->insert_text_at_caret("item_1() test");
		code_edit->set_caret_column(2);
		code_edit->add_code_completion_option(CodeEdit::CodeCompletionKind::KIND_CLASS, "item_0()", "item_0()");
		code_edit->update_code_completion_options();
		SEND_GUI_ACTION(code_edit, "ui_text_completion_replace");
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
		SEND_GUI_ACTION(code_edit, "ui_text_completion_replace");
		CHECK(code_edit->get_line(0) == "item_0() test");
		CHECK(code_edit->get_caret_column() == 7);

		/* End of text*/
		code_edit->clear();
		code_edit->insert_text_at_caret("item_1( test");
		code_edit->set_caret_column(2);
		code_edit->add_code_completion_option(CodeEdit::CodeCompletionKind::KIND_CLASS, "item_0", "item_0");
		code_edit->update_code_completion_options();
		SEND_GUI_ACTION(code_edit, "ui_text_completion_replace");
		CHECK(code_edit->get_line(0) == "item_0( test");
		CHECK(code_edit->get_caret_column() == 6);

		/* End of both. */
		code_edit->clear();
		code_edit->insert_text_at_caret("item_1( test");
		code_edit->set_caret_column(2);
		code_edit->add_code_completion_option(CodeEdit::CodeCompletionKind::KIND_CLASS, "item_0(", "item_0(");
		code_edit->update_code_completion_options();
		SEND_GUI_ACTION(code_edit, "ui_text_completion_replace");
		CHECK(code_edit->get_line(0) == "item_0( test");
		CHECK(code_edit->get_caret_column() == 7);

		/* Full set. */
		/* End on completion entry. */
		code_edit->clear();
		code_edit->insert_text_at_caret("item_1 test");
		code_edit->set_caret_column(2);
		code_edit->add_code_completion_option(CodeEdit::CodeCompletionKind::KIND_CLASS, "item_0()", "item_0()");
		code_edit->update_code_completion_options();
		SEND_GUI_ACTION(code_edit, "ui_text_completion_replace");
		CHECK(code_edit->get_line(0) == "item_0() test");
		CHECK(code_edit->get_caret_column() == 8);

		/* End of text*/
		code_edit->clear();
		code_edit->insert_text_at_caret("item_1() test");
		code_edit->set_caret_column(2);
		code_edit->add_code_completion_option(CodeEdit::CodeCompletionKind::KIND_CLASS, "item_0", "item_0");
		code_edit->update_code_completion_options();
		SEND_GUI_ACTION(code_edit, "ui_text_completion_replace");
		CHECK(code_edit->get_line(0) == "item_0() test");
		CHECK(code_edit->get_caret_column() == 6);

		/* End of both. */
		code_edit->clear();
		code_edit->insert_text_at_caret("item_1() test");
		code_edit->set_caret_column(2);
		code_edit->add_code_completion_option(CodeEdit::CodeCompletionKind::KIND_CLASS, "item_0()", "item_0()");
		code_edit->update_code_completion_options();
		SEND_GUI_ACTION(code_edit, "ui_text_completion_replace");
		CHECK(code_edit->get_line(0) == "item_0() test");
		CHECK(code_edit->get_caret_column() == 8);
	}

	memdelete(code_edit);
}

TEST_CASE("[SceneTree][CodeEdit] symbol lookup") {
	CodeEdit *code_edit = memnew(CodeEdit);
	SceneTree::get_singleton()->get_root()->add_child(code_edit);

	code_edit->set_symbol_lookup_on_click_enabled(true);
	CHECK(code_edit->is_symbol_lookup_on_click_enabled());

	/* Set size for mouse input. */
	code_edit->set_size(Size2(100, 100));

	code_edit->set_text("this is some text");

	Point2 caret_pos = code_edit->get_caret_draw_pos();
	caret_pos.x += 55;
	SEND_GUI_MOUSE_EVENT(code_edit, caret_pos, MOUSE_BUTTON_NONE, MOUSE_BUTTON_NONE);
	CHECK(code_edit->get_text_for_symbol_lookup() == "this is s" + String::chr(0xFFFF) + "ome text");

	SIGNAL_WATCH(code_edit, "symbol_validate");

#ifdef OSX_ENABLED
	SEND_GUI_KEY_EVENT(code_edit, KEY_META);
#else
	SEND_GUI_KEY_EVENT(code_edit, KEY_CTRL);
#endif

	Array signal_args;
	Array arg;
	arg.push_back("some");
	signal_args.push_back(arg);
	SIGNAL_CHECK("symbol_validate", signal_args);

	SIGNAL_UNWATCH(code_edit, "symbol_validate");

	memdelete(code_edit);
}

TEST_CASE("[SceneTree][CodeEdit] line length guidelines") {
	CodeEdit *code_edit = memnew(CodeEdit);
	SceneTree::get_singleton()->get_root()->add_child(code_edit);

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

	/* Move caret up to the previous line on backspace if carret is at the first column. */
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
	CHECK(code_edit->get_text() == "");

	/* Backspace at the beginning without selection has no effect. */
	code_edit->set_text("");
	code_edit->insert_text_at_caret("line 1\nline 2\nline 3");
	code_edit->set_caret_line(0);
	code_edit->set_caret_column(0);
	code_edit->backspace();
	CHECK(code_edit->get_text() == "line 1\nline 2\nline 3");

	memdelete(code_edit);
}

} // namespace TestCodeEdit

#endif // TEST_CODE_EDIT_H
