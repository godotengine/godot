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
#include "scene/main/scene_tree.h"
#include "scene/main/window.h"

namespace TestScrollContainer {

TEST_CASE("[SceneTree][ScrollContainer] Default behavior") {
	ScrollContainer *test_container = memnew(ScrollContainer);
	Window *root = SceneTree::get_singleton()->get_root();
	root->add_child(test_container);

	Control *test_control = memnew(Control);
	test_control->set_custom_minimum_size(Size2(20, 20));
	test_container->add_child(test_control);

	test_container->set_size(Size2(100, 100));
	SceneTree::get_singleton()->process(0);

	CHECK_MESSAGE(
			test_container->get_horizontal_scroll_mode() == ScrollContainer::SCROLL_MODE_AUTO,
			"Horizontal scroll mode is SCROLL_MODE_AUTO by default.");

	CHECK_MESSAGE(
			test_container->get_vertical_scroll_mode() == ScrollContainer::SCROLL_MODE_AUTO,
			"Vertical scroll mode is SCROLL_MODE_AUTO by default.");

	CHECK_MESSAGE(
			test_container->get_scroll_hint_mode() == ScrollContainer::SCROLL_HINT_MODE_DISABLED,
			"Scroll hint mode is SCROLL_HINT_MODE_DISABLED by default.");

	CHECK_MESSAGE(
			!test_container->is_scroll_hint_tiled(),
			"tile_scroll_hint is disabled by default.");

	CHECK_MESSAGE(
			!test_container->is_following_focus(),
			"follow_focus is disabled by default.");

	CHECK_MESSAGE(
			!test_container->is_scroll_horizontal_by_default(),
			"scroll_horizontal_by_default is disabled by default.");

	CHECK_MESSAGE(
			!test_container->get_draw_focus_border(),
			"draw_focus_border is disabled by default.");

	CHECK_MESSAGE(
			test_container->is_clipping_contents(),
			"clip_contents is enabled by default.");

	CHECK_MESSAGE(
			test_container->get_h_scroll() == 0,
			"Horizontal scroll starts at 0 by default.");

	CHECK_MESSAGE(
			test_container->get_v_scroll() == 0,
			"Vertical scroll starts at 0 by default.");

	CHECK_MESSAGE(
			test_container->get_h_scroll_bar() != nullptr,
			"Horizontal scrollbar is created by default.");

	CHECK_MESSAGE(
			test_container->get_v_scroll_bar() != nullptr,
			"Vertical scrollbar is created by default.");

	CHECK_MESSAGE(
			!test_container->get_h_scroll_bar()->is_visible(),
			"Horizontal scrollbar is hidden in auto mode when content fits.");

	CHECK_MESSAGE(
			!test_container->get_v_scroll_bar()->is_visible(),
			"Vertical scrollbar is hidden in auto mode when content fits.");

	memdelete(test_control);
	memdelete(test_container);
}

TEST_CASE("[SceneTree][ScrollContainer] Scroll modes") {
	ScrollContainer *test_container = memnew(ScrollContainer);
	Window *root = SceneTree::get_singleton()->get_root();
	root->add_child(test_container);
	Control *test_control = memnew(Control);
	test_container->add_child(test_control);
	test_control->set_custom_minimum_size(Size2(100, 100));
	Control *target_control = memnew(Control);
	target_control->set_custom_minimum_size(Size2(20, 20));
	target_control->set_position(Point2(80, 80));
	target_control->set_size(Size2(20, 20));
	test_control->add_child(target_control);

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

	test_container->set_h_scroll(0);
	test_container->set_v_scroll(0);
	int disabled_h_scroll_before = test_container->get_h_scroll();
	int disabled_v_scroll_before = test_container->get_v_scroll();
	test_container->ensure_control_visible(target_control);
	SceneTree::get_singleton()->process(0);

	CHECK_MESSAGE(
			test_container->get_h_scroll() == disabled_h_scroll_before,
			"SCROLL_MODE_DISABLED does not change horizontal scroll when ensure_control_visible() targets an out-of-view descendant.");

	CHECK_MESSAGE(
			test_container->get_v_scroll() == disabled_v_scroll_before,
			"SCROLL_MODE_DISABLED does not change vertical scroll when ensure_control_visible() targets an out-of-view descendant.");

	test_container->set_h_scroll(0);
	test_container->set_v_scroll(0);

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

	test_container->set_h_scroll(0);
	test_container->set_v_scroll(0);
	int show_never_h_scroll_before = test_container->get_h_scroll();
	int show_never_v_scroll_before = test_container->get_v_scroll();
	test_container->ensure_control_visible(target_control);
	SceneTree::get_singleton()->process(0);

	CHECK_MESSAGE(
			test_container->get_h_scroll() > show_never_h_scroll_before,
			"SCROLL_MODE_SHOW_NEVER keeps horizontal scrollbar hidden while still allowing programmatic scrolling.");

	CHECK_MESSAGE(
			test_container->get_v_scroll() > show_never_v_scroll_before,
			"SCROLL_MODE_SHOW_NEVER keeps vertical scrollbar hidden while still allowing programmatic scrolling.");

	test_container->set_h_scroll(0);
	test_container->set_v_scroll(0);

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

	memdelete(test_control);
	memdelete(test_container);
}

TEST_CASE("[SceneTree][ScrollContainer] Scroll values and property setters") {
	ScrollContainer *test_container = memnew(ScrollContainer);
	Window *root = SceneTree::get_singleton()->get_root();
	root->add_child(test_container);

	Control *test_control = memnew(Control);
	test_control->set_custom_minimum_size(Size2(300, 300));
	test_container->add_child(test_control);

	test_container->set_size(Size2(100, 100));
	SceneTree::get_singleton()->process(0);

	test_container->set_h_scroll(25);
	test_container->set_v_scroll(35);

	CHECK_MESSAGE(
			test_container->get_h_scroll() == 25,
			"Horizontal scroll value is updated by set_h_scroll().");

	CHECK_MESSAGE(
			test_container->get_v_scroll() == 35,
			"Vertical scroll value is updated by set_v_scroll().");

	test_container->set_horizontal_custom_step(7.0f);
	test_container->set_vertical_custom_step(9.0f);

	CHECK_MESSAGE(
			Math::is_equal_approx(test_container->get_horizontal_custom_step(), 7.0f),
			"Horizontal custom step is updated correctly.");

	CHECK_MESSAGE(
			Math::is_equal_approx(test_container->get_vertical_custom_step(), 9.0f),
			"Vertical custom step is updated correctly.");

	test_container->set_deadzone(13);
	CHECK_MESSAGE(
			test_container->get_deadzone() == 13,
			"Deadzone value is updated correctly.");

	test_container->set_scroll_horizontal_by_default(true);
	CHECK_MESSAGE(
			test_container->is_scroll_horizontal_by_default(),
			"scroll_horizontal_by_default is enabled correctly.");

	test_container->set_follow_focus(true);
	CHECK_MESSAGE(
			test_container->is_following_focus(),
			"follow_focus is enabled correctly.");

	test_container->set_scroll_hint_mode(ScrollContainer::SCROLL_HINT_MODE_ALL);
	CHECK_MESSAGE(
			test_container->get_scroll_hint_mode() == ScrollContainer::SCROLL_HINT_MODE_ALL,
			"Scroll hint mode is updated correctly.");

	test_container->set_tile_scroll_hint(true);
	CHECK_MESSAGE(
			test_container->is_scroll_hint_tiled(),
			"Tiled scroll hint mode is enabled correctly.");

	test_container->set_draw_focus_border(true);
	CHECK_MESSAGE(
			test_container->get_draw_focus_border(),
			"draw_focus_border is enabled correctly.");

	memdelete(test_control);
	memdelete(test_container);
}

TEST_CASE("[SceneTree][ScrollContainer] ensure_control_visible and follow_focus") {
	ScrollContainer *test_container = memnew(ScrollContainer);
	Window *root = SceneTree::get_singleton()->get_root();
	root->add_child(test_container);

	Control *content_control = memnew(Control);
	content_control->set_custom_minimum_size(Size2(300, 300));
	test_container->add_child(content_control);

	Control *target_control = memnew(Control);
	target_control->set_focus_mode(Control::FOCUS_ALL);
	target_control->set_custom_minimum_size(Size2(20, 20));
	target_control->set_position(Point2(260, 250));
	target_control->set_size(Size2(20, 20));
	content_control->add_child(target_control);

	test_container->set_size(Size2(100, 100));
	test_container->set_h_scroll(0);
	test_container->set_v_scroll(0);
	SceneTree::get_singleton()->process(0);

	CHECK_MESSAGE(
			test_container->get_h_scroll() == 0,
			"Horizontal scroll starts at 0 for this setup.");

	CHECK_MESSAGE(
			test_container->get_v_scroll() == 0,
			"Vertical scroll starts at 0 for this setup.");

	test_container->ensure_control_visible(target_control);
	SceneTree::get_singleton()->process(0);

	CHECK_MESSAGE(
			test_container->get_h_scroll() > 0,
			"ensure_control_visible() scrolls horizontally when target control is outside the visible area.");

	CHECK_MESSAGE(
			test_container->get_v_scroll() > 0,
			"ensure_control_visible() scrolls vertically when target control is outside the visible area.");

	Rect2 target_rect(target_control->get_position(), target_control->get_size());
	real_t side_margin = test_container->get_v_scroll_bar()->is_visible() ? test_container->get_v_scroll_bar()->get_size().x : 0.0f;
	real_t bottom_margin = test_container->get_h_scroll_bar()->is_visible() ? test_container->get_h_scroll_bar()->get_size().y : 0.0f;
	real_t visible_left = test_container->get_h_scroll();
	real_t visible_top = test_container->get_v_scroll();
	real_t visible_right = visible_left + test_container->get_size().x - side_margin;
	real_t visible_bottom = visible_top + test_container->get_size().y - bottom_margin;

	CHECK_MESSAGE(
			(target_rect.position.x >= visible_left && (target_rect.position.x + target_rect.size.x) <= visible_right),
			"ensure_control_visible() fully reveals target control horizontally.");

	CHECK_MESSAGE(
			(target_rect.position.y >= visible_top && (target_rect.position.y + target_rect.size.y) <= visible_bottom),
			"ensure_control_visible() fully reveals target control vertically.");

	test_container->set_h_scroll(0);
	test_container->set_v_scroll(0);
	test_container->set_follow_focus(true);
	SceneTree::get_singleton()->process(0);

	target_control->grab_focus();
	SceneTree::get_singleton()->process(0);

	CHECK_MESSAGE(
			test_container->get_h_scroll() > 0,
			"follow_focus scrolls horizontally to focused descendants.");

	CHECK_MESSAGE(
			test_container->get_v_scroll() > 0,
			"follow_focus scrolls vertically to focused descendants.");

	visible_left = test_container->get_h_scroll();
	visible_top = test_container->get_v_scroll();
	visible_right = visible_left + test_container->get_size().x - side_margin;
	visible_bottom = visible_top + test_container->get_size().y - bottom_margin;

	CHECK_MESSAGE(
			(target_rect.position.x >= visible_left && (target_rect.position.x + target_rect.size.x) <= visible_right),
			"follow_focus fully reveals focused target control horizontally.");

	CHECK_MESSAGE(
			(target_rect.position.y >= visible_top && (target_rect.position.y + target_rect.size.y) <= visible_bottom),
			"follow_focus fully reveals focused target control vertically.");

	memdelete(target_control);
	memdelete(content_control);
	memdelete(test_container);
}

TEST_CASE("[SceneTree][ScrollContainer] Configuration warnings") {
	ScrollContainer *test_container = memnew(ScrollContainer);
	Window *root = SceneTree::get_singleton()->get_root();
	root->add_child(test_container);

	CHECK_MESSAGE(
			test_container->get_configuration_warnings().size() == 1,
			"ScrollContainer reports one configuration warning when it has no content child.");

	Control *child_control_1 = memnew(Control);
	Control *child_control_2 = memnew(Control);
	test_container->add_child(child_control_1);

	CHECK_MESSAGE(
			test_container->get_configuration_warnings().is_empty(),
			"ScrollContainer reports no configuration warning when it has exactly one content child.");

	test_container->add_child(child_control_2);

	CHECK_MESSAGE(
			test_container->get_configuration_warnings().size() == 1,
			"ScrollContainer reports one configuration warning when it has more than one content child.");

	memdelete(child_control_2);
	memdelete(child_control_1);
	memdelete(test_container);
}

} // namespace TestScrollContainer
