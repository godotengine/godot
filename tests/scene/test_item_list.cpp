/**************************************************************************/
/*  test_item_list.cpp                                                    */
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

TEST_FORCE_LINK(test_item_list)

#ifndef ADVANCED_GUI_DISABLED

#include "scene/gui/item_list.h"
#include "scene/main/scene_tree.h"
#include "scene/main/window.h"
#include "tests/display_server_mock.h"

namespace TestItemList {

TEST_CASE("[SceneTree][ItemList] Arrow keys after incremental search") {
	ItemList *list = memnew(ItemList);
	list->set_size(Size2(200, 200));
	list->add_item("apple");
	list->add_item("Banana");
	list->add_item("avocado");
	list->add_item("cherry");
	SceneTree::get_singleton()->get_root()->add_child(list);
	list->grab_focus();

	SUBCASE("Arrow keys move to the next match, ignoring case") {
		// Key::A types an uppercase "A", items are lowercase.
		SEND_GUI_KEY_EVENT(Key::A);
		CHECK_EQ(list->get_current(), 0);

		SEND_GUI_KEY_EVENT(Key::DOWN);
		SEND_GUI_KEY_UP_EVENT(Key::DOWN);
		CHECK_EQ(list->get_current(), 2);

		SEND_GUI_KEY_EVENT(Key::UP);
		SEND_GUI_KEY_UP_EVENT(Key::UP);
		CHECK_EQ(list->get_current(), 0);
	}

	SUBCASE("Down arrow still navigates when there is no next match") {
		SEND_GUI_KEY_EVENT(Key::B);
		CHECK_EQ(list->get_current(), 1);

		SEND_GUI_KEY_EVENT(Key::DOWN);
		SEND_GUI_KEY_UP_EVENT(Key::DOWN);
		CHECK_EQ(list->get_current(), 2);
	}

	SUBCASE("Up arrow still navigates when there is no previous match") {
		SEND_GUI_KEY_EVENT(Key::B);
		CHECK_EQ(list->get_current(), 1);

		SEND_GUI_KEY_EVENT(Key::UP);
		SEND_GUI_KEY_UP_EVENT(Key::UP);
		CHECK_EQ(list->get_current(), 0);
	}

	SUBCASE("Arrow keys navigate normally once the matches run out") {
		SEND_GUI_KEY_EVENT(Key::A);
		SEND_GUI_KEY_EVENT(Key::DOWN);
		SEND_GUI_KEY_UP_EVENT(Key::DOWN);
		SEND_GUI_KEY_EVENT(Key::DOWN);
		SEND_GUI_KEY_UP_EVENT(Key::DOWN);
		CHECK_EQ(list->get_current(), 3);

		SEND_GUI_KEY_EVENT(Key::UP);
		SEND_GUI_KEY_UP_EVENT(Key::UP);
		SEND_GUI_KEY_EVENT(Key::UP);
		SEND_GUI_KEY_UP_EVENT(Key::UP);
		CHECK_EQ(list->get_current(), 1);
	}

	memdelete(list);
}

} // namespace TestItemList

#endif // ADVANCED_GUI_DISABLED
