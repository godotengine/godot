/**************************************************************************/
/*  test_viewport.h                                                       */
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

#ifndef TEST_POPUP_H
#define TEST_POPUP_H

#include "scene/gui/popup.h"
#include "scene/gui/control.h"
#include "scene/main/window.h"
#include "core/object/worker_thread_pool.h"

#include "tests/test_macros.h"
#include "tests/test_tools.h"


namespace TestPopup {

TEST_CASE("[SceneTree][Popup]") {
	Window *root = SceneTree::get_singleton()->get_root();
	root->set_embedding_subwindows(true);
	Control *parent = memnew(Control);
	Popup *popup = memnew(Popup);

	// Scene tree:
	// - root (Window) (Default visible: true)
	//   - parent (Control) (Default visible: true)
	//     - popup (Popup) (Default visible: false)

	root->add_child(parent);
	parent->add_child(popup);
	ErrorDetector ed;

	SUBCASE("[_initialize_visiable_parents] Calling twice without deinitializing should first disconnect current connections (fix #87626).") {
		popup->set_visible(true);
		CHECK_FALSE(ed.has_error);
		ed.clear();
		ERR_PRINT_OFF;
		popup->notification(popup->NOTIFICATION_VISIBILITY_CHANGED);
		ERR_PRINT_ON;
		CHECK_FALSE(ed.has_error);
		ed.clear();
	}

}

} // namespace TestPopup

#endif // TEST_POPUP_H
