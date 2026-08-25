/**************************************************************************/
/*  test_editor_help.cpp                                                  */
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

TEST_FORCE_LINK(test_editor_help)

#include "editor/doc/editor_help.h"
#include "scene/gui/control.h"
#include "scene/main/scene_tree.h"
#include "scene/main/timer.h"
#include "tests/display_server_mock.h"

namespace TestEditorHelp {

TEST_CASE("[Editor][EditorHelpBitTooltip] Ignore false target mouse exit") {
	Control *target = memnew(Control);
	target->set_position(Point2(100, 100));
	target->set_size(Size2(100, 100));
	SceneTree::get_singleton()->get_root()->add_child(target);

	EditorHelpBitTooltip *tooltip = memnew(EditorHelpBitTooltip(target));
	Timer *timer = nullptr;
	for (int i = 0; i < tooltip->get_child_count(); i++) {
		timer = Object::cast_to<Timer>(tooltip->get_child(i));
		if (timer) {
			break;
		}
	}
	if (!timer) {
		memdelete(tooltip);
		memdelete(target);
		FAIL_CHECK("The tooltip dismissal timer was not found.");
		return;
	}
	target->add_child(tooltip);

	SEND_GUI_MOUSE_MOTION_EVENT(Point2(150, 150), MouseButtonMask::NONE, Key::NONE);
	target->emit_signal(SceneStringName(mouse_exited));
	CHECK_UNARY(timer->is_stopped());

	SEND_GUI_MOUSE_MOTION_EVENT(Point2(250, 250), MouseButtonMask::NONE, Key::NONE);
	CHECK_UNARY_FALSE(timer->is_stopped());

	memdelete(target);
}

} // namespace TestEditorHelp
