/**************************************************************************/
/*  test_scroll_container.cpp                                             */
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

TEST_FORCE_LINK(test_scroll_container)

#include "scene/gui/control.h"
#include "scene/gui/scroll_container.h"
#include "scene/main/window.h"

namespace TestScrollContainer {

TEST_CASE("[SceneTree][ScrollContainer] Scroll modes") {
	ScrollContainer *test_container = memnew(ScrollContainer);
	Window *root = SceneTree::get_singleton()->get_root();
	root->add_child(test_container);
	Control *test_control = memnew(Control);
	test_container->add_child(test_control);
	test_control->set_custom_minimum_size(Size2(100, 100));

	test_container->set_size(Size2(50, 50));
	SceneTree::get_singleton()->process(0);

	CHECK_MESSAGE(
			test_container->get_h_scroll_bar()->is_visible(),
			"H Scrollbar displays when ScrollContainer width is smaller than contents width with SCROLL_MODE_AUTO.");

	CHECK_MESSAGE(
			test_container->get_v_scroll_bar()->is_visible(),
			"V Scrollbar displays when ScrollContainer height is smaller than contents height with SCROLL_MODE_AUTO.");

	test_container->set_size(Size2(150, 150));
	SceneTree::get_singleton()->process(0);

	CHECK_MESSAGE(
			!test_container->get_h_scroll_bar()->is_visible(),
			"H Scrollbar does not display when ScrollContainer width is larger than contents width with SCROLL_MODE_AUTO.");

	CHECK_MESSAGE(
			!test_container->get_v_scroll_bar()->is_visible(),
			"V Scrollbar does not display when ScrollContainer height is larger than contents height with SCROLL_MODE_AUTO.");

	test_container->set_horizontal_scroll_mode(ScrollContainer::SCROLL_MODE_DISABLED);
	test_container->set_vertical_scroll_mode(ScrollContainer::SCROLL_MODE_DISABLED);

	test_container->set_size(Size2(50, 50));
	SceneTree::get_singleton()->process(0);

	CHECK_MESSAGE(
			!test_container->get_h_scroll_bar()->is_visible(),
			"H Scrollbar does not display when ScrollContainer width is smaller than contents width with SCROLL_MODE_DISABLED.");

	CHECK_MESSAGE(
			!test_container->get_v_scroll_bar()->is_visible(),
			"V Scrollbar does not display when ScrollContainer height is smaller than contents height with SCROLL_MODE_DISABLED.");

	test_container->set_size(Size2(150, 150));
	SceneTree::get_singleton()->process(0);

	CHECK_MESSAGE(
			!test_container->get_h_scroll_bar()->is_visible(),
			"H Scrollbar does not display when ScrollContainer width is larger than contents width with SCROLL_MODE_DISABLED.");

	CHECK_MESSAGE(
			!test_container->get_v_scroll_bar()->is_visible(),
			"V Scrollbar does not display when ScrollContainer height is larger than contents height with SCROLL_MODE_DISABLED.");

	test_container->set_horizontal_scroll_mode(ScrollContainer::SCROLL_MODE_SHOW_ALWAYS);
	test_container->set_vertical_scroll_mode(ScrollContainer::SCROLL_MODE_SHOW_ALWAYS);

	test_container->set_size(Size2(50, 50));
	SceneTree::get_singleton()->process(0);

	CHECK_MESSAGE(
			test_container->get_h_scroll_bar()->is_visible(),
			"H Scrollbar displays when ScrollContainer width is smaller than contents width with SCROLL_MODE_SHOW_ALWAYS.");

	CHECK_MESSAGE(
			test_container->get_v_scroll_bar()->is_visible(),
			"V Scrollbar displays when ScrollContainer height is smaller than contents height with SCROLL_MODE_SHOW_ALWAYS.");

	test_container->set_size(Size2(150, 150));
	SceneTree::get_singleton()->process(0);

	CHECK_MESSAGE(
			test_container->get_h_scroll_bar()->is_visible(),
			"H Scrollbar displays when ScrollContainer width is larger than contents width with SCROLL_MODE_SHOW_ALWAYS.");

	CHECK_MESSAGE(
			test_container->get_v_scroll_bar()->is_visible(),
			"V Scrollbar displays when ScrollContainer height is larger than contents height with SCROLL_MODE_SHOW_ALWAYS.");

	test_container->set_horizontal_scroll_mode(ScrollContainer::SCROLL_MODE_SHOW_NEVER);
	test_container->set_vertical_scroll_mode(ScrollContainer::SCROLL_MODE_SHOW_NEVER);

	test_container->set_size(Size2(50, 50));
	SceneTree::get_singleton()->process(0);

	CHECK_MESSAGE(
			!test_container->get_h_scroll_bar()->is_visible(),
			"H Scrollbar does not display when ScrollContainer width is smaller than contents width with SCROLL_MODE_SHOW_NEVER.");

	CHECK_MESSAGE(
			!test_container->get_v_scroll_bar()->is_visible(),
			"V Scrollbar does not display when ScrollContainer height is smaller than contents height with SCROLL_MODE_SHOW_NEVER.");

	test_container->set_size(Size2(150, 150));
	SceneTree::get_singleton()->process(0);

	CHECK_MESSAGE(
			!test_container->get_h_scroll_bar()->is_visible(),
			"H Scrollbar does not display when ScrollContainer width is larger than contents width with SCROLL_MODE_SHOW_NEVER.");

	CHECK_MESSAGE(
			!test_container->get_v_scroll_bar()->is_visible(),
			"V Scrollbar does not display when ScrollContainer height is larger than contents height with SCROLL_MODE_SHOW_NEVER.");

	test_container->set_horizontal_scroll_mode(ScrollContainer::SCROLL_MODE_RESERVE);
	test_container->set_vertical_scroll_mode(ScrollContainer::SCROLL_MODE_RESERVE);

	test_container->set_size(Size2(50, 50));
	SceneTree::get_singleton()->process(0);

	CHECK_MESSAGE(
			test_container->get_h_scroll_bar()->is_visible(),
			"H Scrollbar displays when ScrollContainer width is smaller than contents width with SCROLL_MODE_RESERVE.");

	CHECK_MESSAGE(
			test_container->get_v_scroll_bar()->is_visible(),
			"V Scrollbar displays when ScrollContainer height is smaller than contents height with SCROLL_MODE_RESERVE.");

	test_container->set_size(Size2(150, 150));
	SceneTree::get_singleton()->process(0);

	CHECK_MESSAGE(
			!test_container->get_h_scroll_bar()->is_visible(),
			"H Scrollbar does not display when ScrollContainer width is larger than contents width with SCROLL_MODE_RESERVE.");

	CHECK_MESSAGE(
			!test_container->get_v_scroll_bar()->is_visible(),
			"V Scrollbar does not display when ScrollContainer height is larger than contents height with SCROLL_MODE_RESERVE.");

	test_container->set_horizontal_scroll_mode(ScrollContainer::SCROLL_MODE_MAXIMIZE_FIRST);
	test_container->set_vertical_scroll_mode(ScrollContainer::SCROLL_MODE_MAXIMIZE_FIRST);
	SceneTree::get_singleton()->process(0);

	CHECK_MESSAGE(
			!test_container->get_h_scroll_bar()->is_visible(),
			"H Scrollbar does not display when ScrollContainer maximum width is 0 with SCROLL_MODE_MAXIMIZE_FIRST.");

	CHECK_MESSAGE(
			!test_container->get_v_scroll_bar()->is_visible(),
			"V Scrollbar does not display when ScrollContainer maximum height is 0 with SCROLL_MODE_MAXIMIZE_FIRST.");

	test_container->set_custom_maximum_size(Size2(50, 50));
	SceneTree::get_singleton()->process(0);

	CHECK_MESSAGE(
			test_container->get_h_scroll_bar()->is_visible(),
			"H Scrollbar displays when ScrollContainer maximum width is smaller than contents width with SCROLL_MODE_MAXIMIZE_FIRST.");

	CHECK_MESSAGE(
			test_container->get_v_scroll_bar()->is_visible(),
			"V Scrollbar displays when ScrollContainer maximum height is smaller than contents height with SCROLL_MODE_MAXIMIZE_FIRST.");

	test_container->set_custom_maximum_size(Size2(150, 150));
	SceneTree::get_singleton()->process(0);

	CHECK_MESSAGE(
			!test_container->get_h_scroll_bar()->is_visible(),
			"H Scrollbar does not display when ScrollContainer maximum width is greater than contents width with SCROLL_MODE_MAXIMIZE_FIRST.");

	CHECK_MESSAGE(
			!test_container->get_v_scroll_bar()->is_visible(),
			"V Scrollbar does not display when ScrollContainer maximum height is greater than contents height with SCROLL_MODE_MAXIMIZE_FIRST.");

	memdelete(test_control);
	memdelete(test_container);
}

} // namespace TestScrollContainer
