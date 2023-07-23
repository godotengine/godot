/**************************************************************************/
/*  test_popup_menu.h                                                     */
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

#ifndef TEST_POPUP_MENU_H
#define TEST_POPUP_MENU_H

#include "scene/gui/popup_menu.h"

#include "tests/test_macros.h"

namespace TestPopupMenu {

TEST_CASE("[SceneTree][PopupMenu] Mouse handling") {
	SceneTree *tree = SceneTree::get_singleton();
	Window *root = tree->get_root();
	const double timer_delay = 0.4; // Default delay value in PopupMenu::PopupMenu is 0.3, so giving it here a bit more time.

	PopupMenu *pm = memnew(PopupMenu);
	PopupMenu *s1 = memnew(PopupMenu);
	PopupMenu *s2 = memnew(PopupMenu);
	s1->set_name("S1");
	s2->set_name("S2");

	pm->add_item("Item 1"); // a1
	pm->add_submenu_item("Submenu 1", "S1"); // a2
	s1->add_item("Sub 1"); // b1
	s1->add_submenu_item("Submenu 2", "S2"); // b2
	s2->add_item("Sub A"); // c1
	s2->add_item("Sub B"); // c2
	s1->add_child(s2);
	pm->add_child(s1);
	root->add_child(pm);

	Point2i on_a1 = Point2i(15, 15);
	Point2i on_a2 = Point2i(15, 40);
	Point2i on_b1 = Point2i(150, 40);
	Point2i on_b2 = Point2i(150, 75);
	Point2i on_c1 = Point2i(90, 75);
	Point2i on_c2 = Point2i(90, 100);

	SUBCASE("[PopupMenu] Focused element on mouse move") {
		pm->show();
		tree->process(timer_delay); // Initial delay for getting nodes ready.
		CHECK_EQ(pm->get_focused_item(), -1);
		SEND_GUI_MOUSE_MOTION_EVENT(on_a1, MouseButtonMask::NONE, Key::NONE);
		CHECK_EQ(pm->get_focused_item(), 0);
		SEND_GUI_MOUSE_MOTION_EVENT(on_a2, MouseButtonMask::NONE, Key::NONE);
		tree->process(timer_delay); // Delay for opening submenu.
		CHECK_EQ(pm->get_focused_item(), 1);
		CHECK_EQ(s1->get_focused_item(), -1);
		SEND_GUI_MOUSE_MOTION_EVENT(on_b1, MouseButtonMask::NONE, Key::NONE);
		CHECK_EQ(pm->get_focused_item(), 1);
		CHECK_EQ(s1->get_focused_item(), 0);
		SEND_GUI_MOUSE_MOTION_EVENT(on_b2, MouseButtonMask::NONE, Key::NONE);
		tree->process(timer_delay); // Delay for opening submenu.
		CHECK_EQ(s1->get_focused_item(), 1);
		CHECK_EQ(s2->get_focused_item(), -1);
		SEND_GUI_MOUSE_MOTION_EVENT(on_c1, MouseButtonMask::NONE, Key::NONE);
		CHECK_EQ(pm->get_focused_item(), 1);
		CHECK_EQ(s1->get_focused_item(), 1);
		CHECK_EQ(s2->get_focused_item(), 0);
		SEND_GUI_MOUSE_MOTION_EVENT(on_c2, MouseButtonMask::NONE, Key::NONE);
		CHECK_EQ(s1->get_focused_item(), 1);
		CHECK_EQ(s2->get_focused_item(), 1);
		SEND_GUI_MOUSE_MOTION_EVENT(on_b1, MouseButtonMask::NONE, Key::NONE);
		//CHECK_EQ(s1->get_focused_item(), 0); // GH-70361 currently expected to fail.
	}

	memdelete(s2);
	memdelete(s1);
	memdelete(pm);
}

} // namespace TestPopupMenu

#endif // TEST_POPUP_MENU_H
