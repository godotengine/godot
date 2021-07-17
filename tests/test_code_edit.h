/*************************************************************************/
/*  test_code_edit.h                                                   */
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

#include "core/input/input_map.h"
#include "core/object/message_queue.h"
#include "core/os/keyboard.h"
#include "core/string/string_builder.h"
#include "scene/gui/code_edit.h"
#include "scene/resources/default_theme/default_theme.h"

#include "tests/test_macros.h"

namespace TestCodeEdit {

TEST_CASE("[SceneTree][CodeEdit] line metadata") {
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
		}

		SIGNAL_UNWATCH(code_edit, "breakpoint_toggled");
	}

	memdelete(code_edit);
}

} // namespace TestCodeEdit

#endif // TEST_CODE_EDIT_H
