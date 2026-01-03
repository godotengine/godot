/**************************************************************************/
/*  test_split_container.h                                                */
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

#pragma once

#include "scene/gui/split_container.h"
#include "scene/main/window.h"

#include "tests/test_macros.h"

namespace TestSplitContainer {

#define CHECK_RECTS(m_rects, m_child_rects)                                                                                                                   \
	CHECK(m_rects.size() == m_child_rects.size());                                                                                                            \
	for (int i = 0; i < (int)m_child_rects.size(); i++) {                                                                                                     \
		CHECK_MESSAGE(m_child_rects[i] == m_rects[i], vformat("Child %s is the wrong size. Child rect: %s, expected: %s.", i, m_child_rects[i], m_rects[i])); \
	}

static inline Vector<Rect2> get_rects_multi(SplitContainer *p_sc, const Vector<int> &p_positions, int p_sep, bool p_horizontal = true) {
	// p_positions is the top/left side of the dragger.
	Vector<Rect2> rects;
	int last_pos = 0;
	for (int i = 0; i < (int)p_positions.size() + 1; i++) {
		const int pos = i >= (int)p_positions.size() ? p_sc->get_size()[p_horizontal ? 0 : 1] : p_positions[i];
		Rect2 rect;
		if (p_horizontal) {
			rect.size.y = p_sc->get_size().y;
			rect.position.x = last_pos;
			rect.size.x = pos - last_pos;
		} else {
			rect.size.x = p_sc->get_size().x;
			rect.position.y = last_pos;
			rect.size.y = pos - last_pos;
		}
		rects.push_back(rect);
		last_pos = pos + p_sep;
	}
	return rects;
}

static inline Vector<Rect2> get_rects(SplitContainer *p_sc, int p_position, int p_sep, bool p_horizontal = true) {
	return get_rects_multi(p_sc, Vector<int>({ p_position }), p_sep, p_horizontal);
}

static inline Vector<Rect2> get_rects_multi_rtl(SplitContainer *p_sc, const Vector<int> &p_positions, int p_sep) {
	Vector<Rect2> rects;
	int last_pos = p_sc->get_size().x;
	for (int i = 0; i < (int)p_positions.size() + 1; i++) {
		const int pos = i >= (int)p_positions.size() ? 0 : p_sc->get_size().x - p_positions[i];
		Rect2 rect;
		rect.size.y = p_sc->get_size().y;
		rect.position.x = pos;
		rect.size.x = last_pos - pos;
		rects.push_back(rect);
		last_pos = pos - p_sep;
	}
	return rects;
}

static inline Vector<Rect2> get_rects_rtl(SplitContainer *p_sc, int p_position, int p_sep) {
	return get_rects_multi_rtl(p_sc, Vector<int>({ p_position }), p_sep);
}

static inline Vector<Rect2> get_child_rects(SplitContainer *p_sc) {
	Vector<Rect2> rects;
	for (int i = 0; i < p_sc->get_child_count(false); i++) {
		Control *c = Object::cast_to<Control>(p_sc->get_child(i, false));
		if (!c || !c->is_visible()) {
			continue;
		}
		rects.push_back(c->get_rect());
	}
	return rects;
}

static inline void set_size_flags(SplitContainer *p_sc, Vector<float> p_expand_ratios, bool p_horizontal = true) {
	for (int i = 0; i < p_sc->get_child_count(false); i++) {
		Control *c = Object::cast_to<Control>(p_sc->get_child(i, false));
		if (!c || !c->is_visible()) {
			continue;
		}
		float ratio = p_expand_ratios[i];
		if (p_horizontal) {
			c->set_h_size_flags(ratio > 0 ? Control::SIZE_EXPAND_FILL : Control::SIZE_FILL);
		} else {
			c->set_v_size_flags(ratio > 0 ? Control::SIZE_EXPAND_FILL : Control::SIZE_FILL);
		}
		if (ratio > 0) {
			c->set_stretch_ratio(ratio);
		}
	}
	MessageQueue::get_singleton()->flush();
}

TEST_CASE("[SceneTree][SplitContainer] Add and remove children") {
	SplitContainer *split_container = memnew(SplitContainer);
	split_container->set_size(Size2(500, 500));
	SceneTree::get_singleton()->get_root()->add_child(split_container);

	SUBCASE("[SplitContainer] One child") {
		Control *child_a = memnew(Control);
		split_container->add_child(child_a);
		MessageQueue::get_singleton()->flush();

		// One child will fill the entire area.
		CHECK(child_a->get_rect() == split_container->get_rect());

		split_container->set_vertical(true);
		CHECK(child_a->get_rect() == split_container->get_rect());

		memdelete(child_a);
	}

	SUBCASE("[SplitContainer] Preserve split offset") {
		// The split offset is preserved through adding, removing, and changing visibility of children.
		split_container->set_split_offset(100);
		CHECK(split_container->get_split_offset() == 100);

		Control *child_a = memnew(Control);
		split_container->add_child(child_a);
		MessageQueue::get_singleton()->flush();
		CHECK(split_container->get_split_offset() == 100);

		Control *child_b = memnew(Control);
		split_container->add_child(child_b);
		MessageQueue::get_singleton()->flush();
		CHECK(split_container->get_split_offset() == 100);

		child_a->hide();
		MessageQueue::get_singleton()->flush();
		CHECK(split_container->get_split_offset() == 100);

		child_b->hide();
		MessageQueue::get_singleton()->flush();
		CHECK(split_container->get_split_offset() == 100);

		child_b->show();
		MessageQueue::get_singleton()->flush();
		CHECK(split_container->get_split_offset() == 100);

		child_a->show();
		MessageQueue::get_singleton()->flush();
		CHECK(split_container->get_split_offset() == 100);

		split_container->remove_child(child_a);
		MessageQueue::get_singleton()->flush();
		CHECK(split_container->get_split_offset() == 100);

		split_container->add_child(child_a);
		MessageQueue::get_singleton()->flush();
		CHECK(split_container->get_split_offset() == 100);

		memdelete(child_a);
		memdelete(child_b);
	}

	memdelete(split_container);
}

TEST_CASE("[SceneTree][SplitContainer] Dragger visibility") {
	SplitContainer *split_container = memnew(SplitContainer);
	split_container->set_size(Size2(500, 500));
	SceneTree::get_singleton()->get_root()->add_child(split_container);
	Control *child_a = memnew(Control);
	Control *child_b = memnew(Control);
	split_container->add_child(child_a);
	split_container->add_child(child_b);
	SplitContainerDragger *dragger = Object::cast_to<SplitContainerDragger>(split_container->get_child(2, true));

	split_container->add_theme_constant_override("autohide", 0);
	MessageQueue::get_singleton()->flush();

	const int sep_constant = split_container->get_theme_constant("separation");
	const Size2i sep = Size2i(MAX(sep_constant, split_container->get_theme_icon("h_grabber")->get_width()), MAX(sep_constant, split_container->get_theme_icon("v_grabber")->get_height()));

	SUBCASE("[SplitContainer] Visibility based on child count") {
		split_container->remove_child(child_a);
		split_container->remove_child(child_b);
		MessageQueue::get_singleton()->flush();

		// No children, not visible.
		CHECK_FALSE(dragger->is_visible());

		// Add one child, not visible.
		split_container->add_child(child_a);
		MessageQueue::get_singleton()->flush();
		CHECK_FALSE(dragger->is_visible());

		// Two children, visible.
		split_container->add_child(child_b);
		MessageQueue::get_singleton()->flush();
		CHECK(dragger->is_visible());

		// Remove a child, not visible.
		split_container->remove_child(child_b);
		MessageQueue::get_singleton()->flush();
		CHECK_FALSE(dragger->is_visible());
	}

	SUBCASE("[SplitContainer] Set dragger visibility") {
		split_container->set_dragger_visibility(SplitContainer::DRAGGER_HIDDEN);
		MessageQueue::get_singleton()->flush();
		CHECK_RECTS(get_rects(split_container, 0, sep.x), get_child_rects(split_container));
		// Can't check the visibility since it happens in draw.

		split_container->set_dragger_visibility(SplitContainer::DRAGGER_HIDDEN_COLLAPSED);
		MessageQueue::get_singleton()->flush();
		CHECK_RECTS(get_rects(split_container, 0, 0), get_child_rects(split_container));

		split_container->set_dragger_visibility(SplitContainer::DRAGGER_VISIBLE);
		MessageQueue::get_singleton()->flush();
		CHECK_RECTS(get_rects(split_container, 0, sep.x), get_child_rects(split_container));
	}

	SUBCASE("[SplitContainer] Not visible when collapsed") {
		split_container->set_collapsed(true);
		MessageQueue::get_singleton()->flush();
		CHECK_FALSE(dragger->is_visible());

		split_container->set_collapsed(false);
		MessageQueue::get_singleton()->flush();
		CHECK(dragger->is_visible());
	}

	memdelete(child_a);
	memdelete(child_b);
	memdelete(split_container);
}

TEST_CASE("[SceneTree][SplitContainer] Collapsed") {
	DisplayServerMock *DS = (DisplayServerMock *)(DisplayServer::get_singleton());

	SplitContainer *split_container = memnew(SplitContainer);
	split_container->set_size(Size2(500, 500));
	SceneTree::get_singleton()->get_root()->add_child(split_container);
	Control *child_a = memnew(Control);
	split_container->add_child(child_a);
	Control *child_b = memnew(Control);
	split_container->add_child(child_b);
	MessageQueue::get_singleton()->flush();

	const int sep_constant = split_container->get_theme_constant("separation");
	const Size2i sep = Size2i(MAX(sep_constant, split_container->get_theme_icon("h_grabber")->get_width()), MAX(sep_constant, split_container->get_theme_icon("v_grabber")->get_height()));

	SUBCASE("[SplitContainer] Dragging and cursor") {
		split_container->set_collapsed(true);

		// Cursor is default.
		SEND_GUI_MOUSE_MOTION_EVENT(Point2(1, 1), MouseButtonMask::NONE, Key::NONE);
		CHECK(DS->get_cursor_shape() == DisplayServer::CURSOR_ARROW);

		// Dragger is disabled, cannot drag.
		SEND_GUI_MOUSE_BUTTON_EVENT(Point2(1, 1), MouseButton::LEFT, MouseButtonMask::LEFT, Key::NONE);
		MessageQueue::get_singleton()->flush();
		CHECK_RECTS(get_rects(split_container, 0, sep.x), get_child_rects(split_container));
		CHECK(split_container->get_split_offset() == 0);
		SEND_GUI_MOUSE_MOTION_EVENT(Point2(10, 1), MouseButtonMask::LEFT, Key::NONE);
		MessageQueue::get_singleton()->flush();
		CHECK_RECTS(get_rects(split_container, 0, sep.x), get_child_rects(split_container));
		CHECK(split_container->get_split_offset() == 0);
	}

	SUBCASE("[SplitContainer] No expand flags") {
		int def_pos = 0;

		split_container->set_split_offset(10);
		MessageQueue::get_singleton()->flush();
		CHECK_RECTS(get_rects(split_container, def_pos + 10, sep.x), get_child_rects(split_container));

		split_container->set_collapsed(true);
		MessageQueue::get_singleton()->flush();

		// The split offset is treated as 0 when collapsed.
		CHECK_RECTS(get_rects(split_container, def_pos, sep.x), get_child_rects(split_container));
		CHECK(split_container->get_split_offset() == 10);
	}

	SUBCASE("[SplitContainer] First child expanded") {
		int def_pos = split_container->get_size().x - sep.x;
		child_a->set_h_size_flags(Control::SIZE_EXPAND_FILL);
		split_container->set_split_offset(-10);
		MessageQueue::get_singleton()->flush();

		CHECK_RECTS(get_rects(split_container, def_pos - 10, sep.x), get_child_rects(split_container));

		split_container->set_collapsed(true);
		MessageQueue::get_singleton()->flush();

		// The split offset is treated as 0 when collapsed.
		CHECK_RECTS(get_rects(split_container, def_pos, sep.x), get_child_rects(split_container));
		CHECK(split_container->get_split_offset() == -10);
	}

	SUBCASE("[SplitContainer] Second child expanded") {
		int def_pos = 0;
		child_b->set_h_size_flags(Control::SIZE_EXPAND_FILL);
		split_container->set_split_offset(10);
		MessageQueue::get_singleton()->flush();

		CHECK_RECTS(get_rects(split_container, def_pos + 10, sep.x), get_child_rects(split_container));

		split_container->set_collapsed(true);
		MessageQueue::get_singleton()->flush();

		// The split offset is treated as 0 when collapsed.
		CHECK_RECTS(get_rects(split_container, def_pos, sep.x), get_child_rects(split_container));
		CHECK(split_container->get_split_offset() == 10);
	}

	SUBCASE("[SplitContainer] Both children expanded") {
		int def_pos = (split_container->get_size().y - sep.y) / 2;
		child_a->set_h_size_flags(Control::SIZE_EXPAND_FILL);
		child_b->set_h_size_flags(Control::SIZE_EXPAND_FILL);
		split_container->set_split_offset(10);
		MessageQueue::get_singleton()->flush();

		CHECK_RECTS(get_rects(split_container, def_pos + 10, sep.x), get_child_rects(split_container));

		split_container->set_collapsed(true);
		MessageQueue::get_singleton()->flush();

		// The split offset is treated as 0 when collapsed.
		CHECK_RECTS(get_rects(split_container, def_pos, sep.x), get_child_rects(split_container));
		CHECK(split_container->get_split_offset() == 10);
	}

	memdelete(child_a);
	memdelete(child_b);
	memdelete(split_container);
}

TEST_CASE("[SceneTree][SplitContainer] Cursor shape") {
	DisplayServerMock *DS = (DisplayServerMock *)(DisplayServer::get_singleton());

	SplitContainer *split_container = memnew(SplitContainer);
	split_container->set_size(Size2(500, 500));
	SceneTree::get_singleton()->get_root()->add_child(split_container);
	Control *child_a = memnew(Control);
	split_container->add_child(child_a);
	Control *child_b = memnew(Control);
	split_container->add_child(child_b);
	MessageQueue::get_singleton()->flush();

	Point2 on_dragger = Point2(1, 1);
	Point2 not_on_dragger = Point2(50, 50);

	// Default cursor shape.
	SEND_GUI_MOUSE_MOTION_EVENT(not_on_dragger, MouseButtonMask::NONE, Key::NONE);
	CHECK(DS->get_cursor_shape() == DisplayServer::CURSOR_ARROW);

	// Horizontal cursor shape.
	SEND_GUI_MOUSE_MOTION_EVENT(on_dragger, MouseButtonMask::NONE, Key::NONE);
	CHECK(DS->get_cursor_shape() == DisplayServer::CURSOR_HSPLIT);

	// Vertical cursor shape.
	split_container->set_vertical(true);
	SEND_GUI_MOUSE_MOTION_EVENT(on_dragger, MouseButtonMask::NONE, Key::NONE);
	CHECK(DS->get_cursor_shape() == DisplayServer::CURSOR_VSPLIT);

	// Move off, default cursor shape.
	SEND_GUI_MOUSE_MOTION_EVENT(not_on_dragger, MouseButtonMask::NONE, Key::NONE);
	CHECK(DS->get_cursor_shape() == DisplayServer::CURSOR_ARROW);

	memdelete(child_a);
	memdelete(child_b);
	memdelete(split_container);
}

TEST_CASE("[SceneTree][SplitContainer] Two children") {
	SplitContainer *split_container = memnew(SplitContainer);
	split_container->set_size(Size2(500, 500));
	SceneTree::get_singleton()->get_root()->add_child(split_container);
	Control *child_a = memnew(Control);
	Control *child_b = memnew(Control);
	split_container->add_child(child_a);
	split_container->add_child(child_b);
	MessageQueue::get_singleton()->flush();

	const int sep_constant = split_container->get_theme_constant("separation");
	const Size2i sep = Size2i(MAX(sep_constant, split_container->get_theme_icon("h_grabber")->get_width()), MAX(sep_constant, split_container->get_theme_icon("v_grabber")->get_height()));

	SUBCASE("[SplitContainer] Minimum size") {
		// Minimum size is the sum of both children's minimum sizes and the separator depending on the vertical axis.
		child_a->set_custom_minimum_size(Size2(100, 200));
		child_b->set_custom_minimum_size(Size2(100, 200));
		MessageQueue::get_singleton()->flush();

		Size2 min_size = split_container->get_minimum_size();
		CHECK(min_size.x == 200 + sep.x);
		CHECK(min_size.y == 200);

		split_container->set_vertical(true);
		MessageQueue::get_singleton()->flush();
		min_size = split_container->get_minimum_size();
		CHECK(min_size.x == 100);
		CHECK(min_size.y == 400 + sep.y);
	}

	SUBCASE("[SplitContainer] Default position") {
		SUBCASE("[SplitContainer] Vertical") {
			// Make sure clamping the split offset doesn't change it or the position.
			split_container->set_vertical(true);

			// No expand flags set.
			MessageQueue::get_singleton()->flush();
			int def_pos = 0;
			CHECK_RECTS(get_rects(split_container, def_pos, sep.y, false), get_child_rects(split_container));
			split_container->clamp_split_offset();
			MessageQueue::get_singleton()->flush();
			CHECK(split_container->get_split_offset() == 0);
			CHECK_RECTS(get_rects(split_container, def_pos, sep.y, false), get_child_rects(split_container));

			// First expand flags set.
			child_a->set_v_size_flags(Control::SIZE_EXPAND_FILL);
			child_b->set_v_size_flags(Control::SIZE_FILL);
			MessageQueue::get_singleton()->flush();
			def_pos = split_container->get_size().y - sep.y;
			CHECK_RECTS(get_rects(split_container, def_pos, sep.y, false), get_child_rects(split_container));
			split_container->clamp_split_offset();
			MessageQueue::get_singleton()->flush();
			CHECK(split_container->get_split_offset() == 0);
			CHECK_RECTS(get_rects(split_container, def_pos, sep.y, false), get_child_rects(split_container));

			// Second expand flags set.
			child_a->set_v_size_flags(Control::SIZE_FILL);
			child_b->set_v_size_flags(Control::SIZE_EXPAND_FILL);
			MessageQueue::get_singleton()->flush();
			def_pos = 0;
			CHECK_RECTS(get_rects(split_container, 0, sep.y, false), get_child_rects(split_container));
			split_container->clamp_split_offset();
			MessageQueue::get_singleton()->flush();
			CHECK(split_container->get_split_offset() == 0);
			CHECK_RECTS(get_rects(split_container, def_pos, sep.y, false), get_child_rects(split_container));

			// Both expand flags set.
			child_a->set_v_size_flags(Control::SIZE_EXPAND_FILL);
			child_b->set_v_size_flags(Control::SIZE_EXPAND_FILL);
			MessageQueue::get_singleton()->flush();
			def_pos = (split_container->get_size().y - sep.y) / 2;
			CHECK_RECTS(get_rects(split_container, def_pos, sep.y, false), get_child_rects(split_container));
			split_container->clamp_split_offset();
			MessageQueue::get_singleton()->flush();
			CHECK(split_container->get_split_offset() == 0);
			CHECK_RECTS(get_rects(split_container, def_pos, sep.y, false), get_child_rects(split_container));

			// Unequal stretch ratios.
			child_a->set_stretch_ratio(2.0);
			MessageQueue::get_singleton()->flush();
			def_pos = (split_container->get_size().y * 2 / 3) - sep.y / 2;
			CHECK_RECTS(get_rects(split_container, def_pos, sep.y, false), get_child_rects(split_container));
			split_container->clamp_split_offset();
			MessageQueue::get_singleton()->flush();
			CHECK(split_container->get_split_offset() == 0);
			CHECK_RECTS(get_rects(split_container, def_pos, sep.y, false), get_child_rects(split_container));
		}

		SUBCASE("[SplitContainer] Right to left") {
			split_container->set_layout_direction(Control::LAYOUT_DIRECTION_RTL);
			split_container->set_position(Point2(0, 0));

			// No expand flags set.
			MessageQueue::get_singleton()->flush();
			int def_pos = 0;
			CHECK_RECTS(get_rects_rtl(split_container, def_pos, sep.y), get_child_rects(split_container));
			split_container->clamp_split_offset();
			MessageQueue::get_singleton()->flush();
			CHECK(split_container->get_split_offset() == 0);
			CHECK_RECTS(get_rects_rtl(split_container, def_pos, sep.y), get_child_rects(split_container));

			// First expand flags set.
			child_a->set_h_size_flags(Control::SIZE_EXPAND_FILL);
			child_b->set_h_size_flags(Control::SIZE_FILL);
			MessageQueue::get_singleton()->flush();
			def_pos = split_container->get_size().y - sep.y;
			CHECK_RECTS(get_rects_rtl(split_container, def_pos, sep.y), get_child_rects(split_container));
			split_container->clamp_split_offset();
			MessageQueue::get_singleton()->flush();
			CHECK(split_container->get_split_offset() == 0);
			CHECK_RECTS(get_rects_rtl(split_container, def_pos, sep.y), get_child_rects(split_container));

			// Second expand flags set.
			child_a->set_h_size_flags(Control::SIZE_FILL);
			child_b->set_h_size_flags(Control::SIZE_EXPAND_FILL);
			MessageQueue::get_singleton()->flush();
			def_pos = 0;
			CHECK_RECTS(get_rects_rtl(split_container, def_pos, sep.y), get_child_rects(split_container));
			split_container->clamp_split_offset();
			MessageQueue::get_singleton()->flush();
			CHECK(split_container->get_split_offset() == 0);
			CHECK_RECTS(get_rects_rtl(split_container, def_pos, sep.y), get_child_rects(split_container));

			// Both expand flags set.
			child_a->set_h_size_flags(Control::SIZE_EXPAND_FILL);
			child_b->set_h_size_flags(Control::SIZE_EXPAND_FILL);
			MessageQueue::get_singleton()->flush();
			def_pos = (split_container->get_size().y - sep.y) / 2;
			CHECK_RECTS(get_rects_rtl(split_container, def_pos, sep.y), get_child_rects(split_container));
			split_container->clamp_split_offset();
			MessageQueue::get_singleton()->flush();
			CHECK(split_container->get_split_offset() == 0);
			CHECK_RECTS(get_rects_rtl(split_container, def_pos, sep.y), get_child_rects(split_container));

			// Unequal stretch ratios.
			child_a->set_stretch_ratio(2.0);
			MessageQueue::get_singleton()->flush();
			def_pos = (split_container->get_size().y * 2 / 3) - sep.y / 2;
			CHECK_RECTS(get_rects_rtl(split_container, def_pos, sep.y), get_child_rects(split_container));
			split_container->clamp_split_offset();
			MessageQueue::get_singleton()->flush();
			CHECK(split_container->get_split_offset() == 0);
			CHECK_RECTS(get_rects_rtl(split_container, def_pos, sep.y), get_child_rects(split_container));
		}

		SUBCASE("[SplitContainer] No expand flags") {
			int def_pos = 0;

			CHECK(split_container->get_split_offset() == 0);
			CHECK_RECTS(get_rects(split_container, def_pos, sep.x), get_child_rects(split_container));

			split_container->clamp_split_offset();
			MessageQueue::get_singleton()->flush();
			CHECK(split_container->get_split_offset() == 0);
			CHECK_RECTS(get_rects(split_container, def_pos, sep.x), get_child_rects(split_container));

			// Minimum sizes affect default position.

			// First child with minimum size.
			child_a->set_custom_minimum_size(Size2(400, 0));
			MessageQueue::get_singleton()->flush();
			def_pos = 400;
			CHECK_RECTS(get_rects(split_container, def_pos, sep.x), get_child_rects(split_container));
			split_container->clamp_split_offset();
			MessageQueue::get_singleton()->flush();
			CHECK(split_container->get_split_offset() == def_pos);
			CHECK_RECTS(get_rects(split_container, def_pos, sep.x), get_child_rects(split_container));

			// Second child with minimum size.
			child_a->set_custom_minimum_size(Size2(0, 0));
			child_b->set_custom_minimum_size(Size2(400, 0));
			MessageQueue::get_singleton()->flush();
			def_pos = split_container->get_size().x - 400 - sep.x;
			CHECK_RECTS(get_rects(split_container, def_pos, sep.x), get_child_rects(split_container));
			split_container->clamp_split_offset();
			MessageQueue::get_singleton()->flush();
			CHECK(split_container->get_split_offset() == def_pos);
			CHECK_RECTS(get_rects(split_container, def_pos, sep.x), get_child_rects(split_container));

			// Both children with minimum size.
			child_a->set_custom_minimum_size(Size2(200, 0));
			child_b->set_custom_minimum_size(Size2(288, 0));
			MessageQueue::get_singleton()->flush();
			def_pos = 200;
			CHECK_RECTS(get_rects(split_container, def_pos, sep.x), get_child_rects(split_container));
			split_container->clamp_split_offset();
			MessageQueue::get_singleton()->flush();
			CHECK(split_container->get_split_offset() == def_pos);
			CHECK_RECTS(get_rects(split_container, def_pos, sep.x), get_child_rects(split_container));
		}

		SUBCASE("[SplitContainer] First child expanded") {
			const int def_pos = split_container->get_size().x - sep.x;
			child_a->set_h_size_flags(Control::SIZE_EXPAND_FILL);
			MessageQueue::get_singleton()->flush();

			CHECK_RECTS(get_rects(split_container, def_pos, sep.x), get_child_rects(split_container));

			split_container->clamp_split_offset();
			MessageQueue::get_singleton()->flush();
			CHECK(split_container->get_split_offset() == 0);
			CHECK_RECTS(get_rects(split_container, def_pos, sep.x), get_child_rects(split_container));

			// Minimum sizes affect default position.

			// First child with minimum size.
			child_a->set_custom_minimum_size(Size2(400, 0));
			MessageQueue::get_singleton()->flush();
			CHECK_RECTS(get_rects(split_container, def_pos, sep.x), get_child_rects(split_container));
			split_container->clamp_split_offset();
			MessageQueue::get_singleton()->flush();
			CHECK(split_container->get_split_offset() == 0);
			CHECK_RECTS(get_rects(split_container, def_pos, sep.x), get_child_rects(split_container));

			// Second child with minimum size.
			child_a->set_custom_minimum_size(Size2(0, 0));
			child_b->set_custom_minimum_size(Size2(400, 0));
			MessageQueue::get_singleton()->flush();
			int pos = split_container->get_size().x - 400 - sep.x;
			CHECK_RECTS(get_rects(split_container, pos, sep.x), get_child_rects(split_container));
			split_container->clamp_split_offset();
			MessageQueue::get_singleton()->flush();
			CHECK(split_container->get_split_offset() == pos - def_pos);
			CHECK_RECTS(get_rects(split_container, pos, sep.x), get_child_rects(split_container));

			// Both children with minimum size.
			child_a->set_custom_minimum_size(Size2(200, 0));
			child_b->set_custom_minimum_size(Size2(288, 0));
			MessageQueue::get_singleton()->flush();
			pos = 200;
			CHECK_RECTS(get_rects(split_container, pos, sep.x), get_child_rects(split_container));
			split_container->clamp_split_offset();
			MessageQueue::get_singleton()->flush();
			CHECK(split_container->get_split_offset() == pos - def_pos);
			CHECK_RECTS(get_rects(split_container, pos, sep.x), get_child_rects(split_container));
		}

		SUBCASE("[SplitContainer] Second child expanded") {
			int def_pos = 0;
			child_b->set_h_size_flags(Control::SIZE_EXPAND_FILL);
			MessageQueue::get_singleton()->flush();

			CHECK_RECTS(get_rects(split_container, def_pos, sep.x), get_child_rects(split_container));

			split_container->clamp_split_offset();
			MessageQueue::get_singleton()->flush();
			CHECK(split_container->get_split_offset() == 0);
			CHECK_RECTS(get_rects(split_container, def_pos, sep.x), get_child_rects(split_container));

			// Minimum sizes affect default position.

			// First child with minimum size.
			child_a->set_custom_minimum_size(Size2(400, 0));
			MessageQueue::get_singleton()->flush();
			def_pos = 400;
			CHECK_RECTS(get_rects(split_container, def_pos, sep.x), get_child_rects(split_container));
			split_container->clamp_split_offset();
			MessageQueue::get_singleton()->flush();
			CHECK(split_container->get_split_offset() == def_pos);
			CHECK_RECTS(get_rects(split_container, def_pos, sep.x), get_child_rects(split_container));

			// Second child with minimum size.
			child_a->set_custom_minimum_size(Size2(0, 0));
			child_b->set_custom_minimum_size(Size2(400, 0));
			MessageQueue::get_singleton()->flush();
			def_pos = 500 - 400 - sep.x;
			CHECK_RECTS(get_rects(split_container, def_pos, sep.x), get_child_rects(split_container));
			split_container->clamp_split_offset();
			MessageQueue::get_singleton()->flush();
			CHECK(split_container->get_split_offset() == def_pos);
			CHECK_RECTS(get_rects(split_container, def_pos, sep.x), get_child_rects(split_container));

			// Both children with minimum size.
			child_a->set_custom_minimum_size(Size2(200, 0));
			child_b->set_custom_minimum_size(Size2(288, 0));
			MessageQueue::get_singleton()->flush();
			def_pos = 200;
			CHECK_RECTS(get_rects(split_container, def_pos, sep.x), get_child_rects(split_container));
			split_container->clamp_split_offset();
			MessageQueue::get_singleton()->flush();
			CHECK(split_container->get_split_offset() == def_pos);
			CHECK_RECTS(get_rects(split_container, def_pos, sep.x), get_child_rects(split_container));
		}

		SUBCASE("[SplitContainer] Both children expanded") {
			const int def_pos = (split_container->get_size().x - sep.x) / 2;
			child_a->set_h_size_flags(Control::SIZE_EXPAND_FILL);
			child_b->set_h_size_flags(Control::SIZE_EXPAND_FILL);
			MessageQueue::get_singleton()->flush();

			CHECK_RECTS(get_rects(split_container, def_pos, sep.x), get_child_rects(split_container));

			split_container->clamp_split_offset();
			MessageQueue::get_singleton()->flush();
			CHECK(split_container->get_split_offset() == 0);
			CHECK_RECTS(get_rects(split_container, def_pos, sep.x), get_child_rects(split_container));

			// Minimum sizes affect default position.

			// First child with minimum size.
			child_a->set_custom_minimum_size(Size2(400, 0));
			MessageQueue::get_singleton()->flush();
			int pos = 400;
			CHECK_RECTS(get_rects(split_container, pos, sep.x), get_child_rects(split_container));
			split_container->clamp_split_offset();
			MessageQueue::get_singleton()->flush();
			CHECK(split_container->get_split_offset() == pos - def_pos);
			CHECK_RECTS(get_rects(split_container, pos, sep.x), get_child_rects(split_container));

			// Second child with minimum size.
			child_a->set_custom_minimum_size(Size2(0, 0));
			child_b->set_custom_minimum_size(Size2(400, 0));
			MessageQueue::get_singleton()->flush();
			pos = split_container->get_size().x - 400 - sep.x;
			CHECK_RECTS(get_rects(split_container, pos, sep.x), get_child_rects(split_container));
			split_container->clamp_split_offset();
			MessageQueue::get_singleton()->flush();
			CHECK(split_container->get_split_offset() == pos - def_pos);
			CHECK_RECTS(get_rects(split_container, pos, sep.x), get_child_rects(split_container));

			// Both children with minimum size.
			child_a->set_custom_minimum_size(Size2(200, 0));
			child_b->set_custom_minimum_size(Size2(288, 0));
			MessageQueue::get_singleton()->flush();
			pos = 200;
			CHECK_RECTS(get_rects(split_container, pos, sep.x), get_child_rects(split_container));
			split_container->clamp_split_offset();
			MessageQueue::get_singleton()->flush();
			CHECK(split_container->get_split_offset() == pos - def_pos);
			CHECK_RECTS(get_rects(split_container, pos, sep.x), get_child_rects(split_container));
		}

		SUBCASE("[SplitContainer] Unequal stretch ratios") {
			const int def_pos = (split_container->get_size().x * 2 / 3) - sep.x / 2;
			child_a->set_h_size_flags(Control::SIZE_EXPAND_FILL);
			child_b->set_h_size_flags(Control::SIZE_EXPAND_FILL);
			child_a->set_stretch_ratio(2.0);
			MessageQueue::get_singleton()->flush();

			CHECK_RECTS(get_rects(split_container, def_pos, sep.x), get_child_rects(split_container));

			split_container->clamp_split_offset();
			MessageQueue::get_singleton()->flush();
			CHECK(split_container->get_split_offset() == 0);
			CHECK_RECTS(get_rects(split_container, def_pos, sep.x), get_child_rects(split_container));

			// Minimum sizes affect default position.

			// First child with minimum size.
			child_a->set_custom_minimum_size(Size2(400, 0));
			MessageQueue::get_singleton()->flush();
			int pos = 400;
			CHECK_RECTS(get_rects(split_container, pos, sep.x), get_child_rects(split_container));
			split_container->clamp_split_offset();
			MessageQueue::get_singleton()->flush();
			CHECK(split_container->get_split_offset() == pos - def_pos);
			CHECK_RECTS(get_rects(split_container, pos, sep.x), get_child_rects(split_container));

			// Second child with minimum size.
			child_a->set_custom_minimum_size(Size2(0, 0));
			child_b->set_custom_minimum_size(Size2(400, 0));
			MessageQueue::get_singleton()->flush();
			pos = split_container->get_size().x - 400 - sep.x;
			CHECK_RECTS(get_rects(split_container, pos, sep.x), get_child_rects(split_container));
			split_container->clamp_split_offset();
			MessageQueue::get_singleton()->flush();
			CHECK(split_container->get_split_offset() == pos - def_pos);
			CHECK_RECTS(get_rects(split_container, pos, sep.x), get_child_rects(split_container));

			// Both children with minimum size.
			child_a->set_custom_minimum_size(Size2(200, 0));
			child_b->set_custom_minimum_size(Size2(288, 0));
			MessageQueue::get_singleton()->flush();
			pos = 200;
			CHECK_RECTS(get_rects(split_container, pos, sep.x), get_child_rects(split_container));
			split_container->clamp_split_offset();
			MessageQueue::get_singleton()->flush();
			CHECK(split_container->get_split_offset() == pos - def_pos);
			CHECK_RECTS(get_rects(split_container, pos, sep.x), get_child_rects(split_container));
		}
	}

	SUBCASE("[SplitContainer] Set split offset") {
		SUBCASE("[SplitContainer] Right to left") {
			split_container->set_layout_direction(Control::LAYOUT_DIRECTION_RTL);
			split_container->set_position(Point2(0, 0));
			int def_pos = 0;
			// Positive.
			split_container->set_split_offset(10);
			MessageQueue::get_singleton()->flush();
			CHECK(split_container->get_split_offset() == 10);
			CHECK_RECTS(get_rects_rtl(split_container, def_pos + 10, sep.y), get_child_rects(split_container));

			// Negative.
			split_container->set_split_offset(-10);
			MessageQueue::get_singleton()->flush();
			CHECK(split_container->get_split_offset() == -10);
			CHECK_RECTS(get_rects_rtl(split_container, def_pos, sep.y), get_child_rects(split_container));
		}

		SUBCASE("[SplitContainer] No expand flags") {
			int def_pos = 0;

			// Positive.
			split_container->set_split_offset(10);
			MessageQueue::get_singleton()->flush();
			CHECK(split_container->get_split_offset() == 10);
			CHECK_RECTS(get_rects(split_container, def_pos + 10, sep.x), get_child_rects(split_container));

			// Negative.
			split_container->set_split_offset(-10);
			MessageQueue::get_singleton()->flush();
			CHECK(split_container->get_split_offset() == -10);
			CHECK_RECTS(get_rects(split_container, def_pos, sep.x), get_child_rects(split_container));

			// Clamped.
			split_container->set_split_offset(1000);
			MessageQueue::get_singleton()->flush();
			CHECK(split_container->get_split_offset() == 1000);
			CHECK_RECTS(get_rects(split_container, split_container->get_size().x - sep.x, sep.x), get_child_rects(split_container));
		}

		SUBCASE("[SplitContainer] First child expanded") {
			int def_pos = split_container->get_size().x - sep.x;
			child_a->set_h_size_flags(Control::SIZE_EXPAND_FILL);
			MessageQueue::get_singleton()->flush();

			// Positive.
			split_container->set_split_offset(10);
			MessageQueue::get_singleton()->flush();
			CHECK(split_container->get_split_offset() == 10);
			CHECK_RECTS(get_rects(split_container, def_pos, sep.x), get_child_rects(split_container));

			// Negative.
			split_container->set_split_offset(-10);
			MessageQueue::get_singleton()->flush();
			CHECK(split_container->get_split_offset() == -10);
			CHECK_RECTS(get_rects(split_container, def_pos - 10, sep.x), get_child_rects(split_container));

			// Clamped.
			split_container->set_split_offset(-1000);
			MessageQueue::get_singleton()->flush();
			CHECK(split_container->get_split_offset() == -1000);
			CHECK_RECTS(get_rects(split_container, 0, sep.x), get_child_rects(split_container));
		}

		SUBCASE("[SplitContainer] Second child expanded") {
			int def_pos = 0;
			child_b->set_h_size_flags(Control::SIZE_EXPAND_FILL);
			MessageQueue::get_singleton()->flush();

			// Positive.
			split_container->set_split_offset(10);
			MessageQueue::get_singleton()->flush();
			CHECK(split_container->get_split_offset() == 10);
			CHECK_RECTS(get_rects(split_container, def_pos + 10, sep.x), get_child_rects(split_container));

			// Negative.
			split_container->set_split_offset(-10);
			MessageQueue::get_singleton()->flush();
			CHECK(split_container->get_split_offset() == -10);
			CHECK_RECTS(get_rects(split_container, def_pos, sep.x), get_child_rects(split_container));

			// Clamped.
			split_container->set_split_offset(1000);
			MessageQueue::get_singleton()->flush();
			CHECK(split_container->get_split_offset() == 1000);
			CHECK_RECTS(get_rects(split_container, split_container->get_size().x - sep.x, sep.x), get_child_rects(split_container));
		}

		SUBCASE("[SplitContainer] Both children expanded") {
			int def_pos = (split_container->get_size().x - sep.x) / 2;
			child_a->set_h_size_flags(Control::SIZE_EXPAND_FILL);
			child_b->set_h_size_flags(Control::SIZE_EXPAND_FILL);
			MessageQueue::get_singleton()->flush();

			// Positive.
			split_container->set_split_offset(10);
			MessageQueue::get_singleton()->flush();
			CHECK(split_container->get_split_offset() == 10);
			CHECK_RECTS(get_rects(split_container, def_pos + 10, sep.x), get_child_rects(split_container));

			// Negative.
			split_container->set_split_offset(-10);
			MessageQueue::get_singleton()->flush();
			CHECK(split_container->get_split_offset() == -10);
			CHECK_RECTS(get_rects(split_container, def_pos - 10, sep.x), get_child_rects(split_container));

			// Clamped positive.
			split_container->set_split_offset(1000);
			MessageQueue::get_singleton()->flush();
			CHECK(split_container->get_split_offset() == 1000);
			CHECK_RECTS(get_rects(split_container, split_container->get_size().x - sep.x, sep.x), get_child_rects(split_container));

			// Clamped negative.
			split_container->set_split_offset(-1000);
			MessageQueue::get_singleton()->flush();
			CHECK(split_container->get_split_offset() == -1000);
			CHECK_RECTS(get_rects(split_container, 0, sep.x), get_child_rects(split_container));
		}

		SUBCASE("[SplitContainer] Unequal stretch ratios") {
			int def_pos = (split_container->get_size().x * 2 / 3) - sep.x / 2;
			child_a->set_h_size_flags(Control::SIZE_EXPAND_FILL);
			child_b->set_h_size_flags(Control::SIZE_EXPAND_FILL);
			child_a->set_stretch_ratio(2.0);
			MessageQueue::get_singleton()->flush();

			// Positive.
			split_container->set_split_offset(10);
			MessageQueue::get_singleton()->flush();
			CHECK(split_container->get_split_offset() == 10);
			CHECK_RECTS(get_rects(split_container, def_pos + 10, sep.x), get_child_rects(split_container));

			// Negative.
			split_container->set_split_offset(-10);
			MessageQueue::get_singleton()->flush();
			CHECK(split_container->get_split_offset() == -10);
			CHECK_RECTS(get_rects(split_container, def_pos - 10, sep.x), get_child_rects(split_container));

			// Clamped positive.
			split_container->set_split_offset(1000);
			MessageQueue::get_singleton()->flush();
			CHECK(split_container->get_split_offset() == 1000);
			CHECK_RECTS(get_rects(split_container, split_container->get_size().x - sep.x, sep.x), get_child_rects(split_container));

			// Clamped negative.
			split_container->set_split_offset(-1000);
			MessageQueue::get_singleton()->flush();
			CHECK(split_container->get_split_offset() == -1000);
			CHECK_RECTS(get_rects(split_container, 0, sep.x), get_child_rects(split_container));
		}
	}

	SUBCASE("[SplitContainer] Keep split offset when changing minimum size") {
		SUBCASE("[SplitContainer] No expand flags") {
			int def_pos = 0;

			split_container->set_split_offset(100);
			child_a->set_custom_minimum_size(Size2(10, 0));
			MessageQueue::get_singleton()->flush();
			CHECK(split_container->get_split_offset() == 100);
			CHECK_RECTS(get_rects(split_container, def_pos + 100, sep.x), get_child_rects(split_container));

			child_a->set_custom_minimum_size(Size2(50, 0));
			MessageQueue::get_singleton()->flush();
			CHECK(split_container->get_split_offset() == 100);
			CHECK_RECTS(get_rects(split_container, def_pos + 100, sep.x), get_child_rects(split_container));
		}

		SUBCASE("[SplitContainer] First child expanded") {
			int def_pos = split_container->get_size().x - sep.x;
			child_a->set_h_size_flags(Control::SIZE_EXPAND_FILL);
			MessageQueue::get_singleton()->flush();

			split_container->set_split_offset(-100);
			child_b->set_custom_minimum_size(Size2(10, 0));
			MessageQueue::get_singleton()->flush();
			CHECK(split_container->get_split_offset() == -100);
			CHECK_RECTS(get_rects(split_container, def_pos - 100, sep.x), get_child_rects(split_container));

			child_b->set_custom_minimum_size(Size2(50, 0));
			MessageQueue::get_singleton()->flush();
			CHECK(split_container->get_split_offset() == -100);
			CHECK_RECTS(get_rects(split_container, def_pos - 100, sep.x), get_child_rects(split_container));
		}

		SUBCASE("[SplitContainer] Second child expanded") {
			int def_pos = 0;
			child_b->set_h_size_flags(Control::SIZE_EXPAND_FILL);
			MessageQueue::get_singleton()->flush();

			split_container->set_split_offset(100);
			child_a->set_custom_minimum_size(Size2(10, 0));
			MessageQueue::get_singleton()->flush();
			CHECK(split_container->get_split_offset() == 100);
			CHECK_RECTS(get_rects(split_container, def_pos + 100, sep.x), get_child_rects(split_container));

			child_a->set_custom_minimum_size(Size2(50, 0));
			MessageQueue::get_singleton()->flush();
			CHECK(split_container->get_split_offset() == 100);
			CHECK_RECTS(get_rects(split_container, def_pos + 100, sep.x), get_child_rects(split_container));
		}

		SUBCASE("[SplitContainer] Both children expanded") {
			int def_pos = (split_container->get_size().x - sep.x) / 2;
			child_a->set_h_size_flags(Control::SIZE_EXPAND_FILL);
			child_b->set_h_size_flags(Control::SIZE_EXPAND_FILL);
			MessageQueue::get_singleton()->flush();

			split_container->set_split_offset(20);
			child_a->set_custom_minimum_size(Size2(10, 0));
			child_b->set_custom_minimum_size(Size2(10, 0));
			MessageQueue::get_singleton()->flush();
			CHECK(split_container->get_split_offset() == 20);
			CHECK_RECTS(get_rects(split_container, def_pos + 20, sep.x), get_child_rects(split_container));

			child_a->set_custom_minimum_size(Size2(50, 0));
			child_b->set_custom_minimum_size(Size2(50, 0));
			MessageQueue::get_singleton()->flush();
			CHECK(split_container->get_split_offset() == 20);
			CHECK_RECTS(get_rects(split_container, def_pos + 20, sep.x), get_child_rects(split_container));
		}
	}

	SUBCASE("[SplitContainer] Keep split offset when changing visibility") {
		SUBCASE("[SplitContainer] No expand flags") {
			int def_pos = 0;
			split_container->set_split_offset(100);

			child_a->hide();
			MessageQueue::get_singleton()->flush();
			CHECK(split_container->get_split_offset() == 100);
			CHECK_RECTS(get_rects_multi(split_container, Vector<int>(), sep.x), get_child_rects(split_container));

			child_b->hide();
			MessageQueue::get_singleton()->flush();
			CHECK(split_container->get_split_offset() == 100);

			child_a->show();
			MessageQueue::get_singleton()->flush();
			CHECK(split_container->get_split_offset() == 100);
			CHECK_RECTS(get_rects_multi(split_container, Vector<int>(), sep.x), get_child_rects(split_container));

			child_b->show();
			MessageQueue::get_singleton()->flush();
			CHECK(split_container->get_split_offset() == 100);
			CHECK_RECTS(get_rects(split_container, def_pos + 100, sep.x), get_child_rects(split_container));
		}

		SUBCASE("[SplitContainer] First child expanded") {
			int def_pos = split_container->get_size().x - sep.x;
			child_a->set_h_size_flags(Control::SIZE_EXPAND_FILL);
			MessageQueue::get_singleton()->flush();
			split_container->set_split_offset(-100);

			child_a->hide();
			MessageQueue::get_singleton()->flush();
			CHECK(split_container->get_split_offset() == -100);
			CHECK_RECTS(get_rects_multi(split_container, Vector<int>(), sep.x), get_child_rects(split_container));

			child_b->hide();
			MessageQueue::get_singleton()->flush();
			CHECK(split_container->get_split_offset() == -100);

			child_a->show();
			MessageQueue::get_singleton()->flush();
			CHECK(split_container->get_split_offset() == -100);
			CHECK_RECTS(get_rects_multi(split_container, Vector<int>(), sep.x), get_child_rects(split_container));

			child_b->show();
			MessageQueue::get_singleton()->flush();
			CHECK(split_container->get_split_offset() == -100);
			CHECK_RECTS(get_rects(split_container, def_pos - 100, sep.x), get_child_rects(split_container));
		}

		SUBCASE("[SplitContainer] Second child expanded") {
			int def_pos = 0;
			child_b->set_h_size_flags(Control::SIZE_EXPAND_FILL);
			MessageQueue::get_singleton()->flush();
			split_container->set_split_offset(100);

			child_a->hide();
			MessageQueue::get_singleton()->flush();
			CHECK(split_container->get_split_offset() == 100);
			CHECK_RECTS(get_rects_multi(split_container, Vector<int>(), sep.x), get_child_rects(split_container));

			child_b->hide();
			MessageQueue::get_singleton()->flush();
			CHECK(split_container->get_split_offset() == 100);

			child_a->show();
			MessageQueue::get_singleton()->flush();
			CHECK(split_container->get_split_offset() == 100);
			CHECK_RECTS(get_rects_multi(split_container, Vector<int>(), sep.x), get_child_rects(split_container));

			child_b->show();
			MessageQueue::get_singleton()->flush();
			CHECK(split_container->get_split_offset() == 100);
			CHECK_RECTS(get_rects(split_container, def_pos + 100, sep.x), get_child_rects(split_container));
		}

		SUBCASE("[SplitContainer] Both children expanded") {
			int def_pos = (split_container->get_size().x - sep.x) / 2;
			child_a->set_h_size_flags(Control::SIZE_EXPAND_FILL);
			child_b->set_h_size_flags(Control::SIZE_EXPAND_FILL);
			MessageQueue::get_singleton()->flush();
			split_container->set_split_offset(20);

			child_a->hide();
			MessageQueue::get_singleton()->flush();
			CHECK(split_container->get_split_offset() == 20);
			CHECK_RECTS(get_rects_multi(split_container, Vector<int>(), sep.x), get_child_rects(split_container));

			child_b->hide();
			MessageQueue::get_singleton()->flush();
			CHECK(split_container->get_split_offset() == 20);

			child_a->show();
			MessageQueue::get_singleton()->flush();
			CHECK(split_container->get_split_offset() == 20);
			CHECK_RECTS(get_rects_multi(split_container, Vector<int>(), sep.x), get_child_rects(split_container));

			child_b->show();
			MessageQueue::get_singleton()->flush();
			CHECK(split_container->get_split_offset() == 20);
			CHECK_RECTS(get_rects(split_container, def_pos + 20, sep.x), get_child_rects(split_container));
		}
	}

	SUBCASE("[SplitContainer] Keep split offset when removing children") {
		SUBCASE("[SplitContainer] No expand flags") {
			int def_pos = 0;
			split_container->set_split_offset(100);

			split_container->remove_child(child_a);
			MessageQueue::get_singleton()->flush();
			CHECK(split_container->get_split_offset() == 100);
			CHECK_RECTS(get_rects_multi(split_container, Vector<int>(), sep.x), get_child_rects(split_container));

			split_container->remove_child(child_b);
			MessageQueue::get_singleton()->flush();
			CHECK(split_container->get_split_offset() == 100);

			split_container->add_child(child_a);
			MessageQueue::get_singleton()->flush();
			CHECK(split_container->get_split_offset() == 100);
			CHECK_RECTS(get_rects_multi(split_container, Vector<int>(), sep.x), get_child_rects(split_container));

			split_container->add_child(child_b);
			MessageQueue::get_singleton()->flush();
			CHECK(split_container->get_split_offset() == 100);
			CHECK_RECTS(get_rects(split_container, def_pos + 100, sep.x), get_child_rects(split_container));
		}

		SUBCASE("[SplitContainer] First child expanded") {
			int def_pos = split_container->get_size().x - sep.x;
			child_a->set_h_size_flags(Control::SIZE_EXPAND_FILL);
			MessageQueue::get_singleton()->flush();
			split_container->set_split_offset(-100);

			split_container->remove_child(child_a);
			MessageQueue::get_singleton()->flush();
			CHECK(split_container->get_split_offset() == -100);
			CHECK_RECTS(get_rects_multi(split_container, Vector<int>(), sep.x), get_child_rects(split_container));

			split_container->remove_child(child_b);
			MessageQueue::get_singleton()->flush();
			CHECK(split_container->get_split_offset() == -100);

			split_container->add_child(child_a);
			MessageQueue::get_singleton()->flush();
			CHECK(split_container->get_split_offset() == -100);
			CHECK_RECTS(get_rects_multi(split_container, Vector<int>(), sep.x), get_child_rects(split_container));

			split_container->add_child(child_b);
			MessageQueue::get_singleton()->flush();
			CHECK(split_container->get_split_offset() == -100);
			CHECK_RECTS(get_rects(split_container, def_pos - 100, sep.x), get_child_rects(split_container));
		}

		SUBCASE("[SplitContainer] Second child expanded") {
			int def_pos = 0;
			child_b->set_h_size_flags(Control::SIZE_EXPAND_FILL);
			MessageQueue::get_singleton()->flush();
			split_container->set_split_offset(100);

			split_container->remove_child(child_a);
			MessageQueue::get_singleton()->flush();
			CHECK(split_container->get_split_offset() == 100);
			CHECK_RECTS(get_rects_multi(split_container, Vector<int>(), sep.x), get_child_rects(split_container));

			split_container->remove_child(child_b);
			MessageQueue::get_singleton()->flush();
			CHECK(split_container->get_split_offset() == 100);

			split_container->add_child(child_a);
			MessageQueue::get_singleton()->flush();
			CHECK(split_container->get_split_offset() == 100);
			CHECK_RECTS(get_rects_multi(split_container, Vector<int>(), sep.x), get_child_rects(split_container));

			split_container->add_child(child_b);
			MessageQueue::get_singleton()->flush();
			CHECK(split_container->get_split_offset() == 100);
			CHECK_RECTS(get_rects(split_container, def_pos + 100, sep.x), get_child_rects(split_container));
		}

		SUBCASE("[SplitContainer] Both children expanded") {
			int def_pos = (split_container->get_size().x - sep.x) / 2;
			child_a->set_h_size_flags(Control::SIZE_EXPAND_FILL);
			child_b->set_h_size_flags(Control::SIZE_EXPAND_FILL);
			MessageQueue::get_singleton()->flush();
			split_container->set_split_offset(20);

			split_container->remove_child(child_a);
			MessageQueue::get_singleton()->flush();
			CHECK(split_container->get_split_offset() == 20);
			CHECK_RECTS(get_rects_multi(split_container, Vector<int>(), sep.x), get_child_rects(split_container));

			split_container->remove_child(child_b);
			MessageQueue::get_singleton()->flush();
			CHECK(split_container->get_split_offset() == 20);

			split_container->add_child(child_a);
			MessageQueue::get_singleton()->flush();
			CHECK(split_container->get_split_offset() == 20);
			CHECK_RECTS(get_rects_multi(split_container, Vector<int>(), sep.x), get_child_rects(split_container));

			split_container->add_child(child_b);
			MessageQueue::get_singleton()->flush();
			CHECK(split_container->get_split_offset() == 20);
			CHECK_RECTS(get_rects(split_container, def_pos + 20, sep.x), get_child_rects(split_container));
		}
	}

	SUBCASE("[SplitContainer] Keep split offset when changing expand flags") {
		int def_pos = 0;
		split_container->set_split_offset(20);
		MessageQueue::get_singleton()->flush();
		CHECK(split_container->get_split_offset() == 20);
		CHECK_RECTS(get_rects(split_container, def_pos + 20, sep.x), get_child_rects(split_container));

		child_a->set_h_size_flags(Control::SIZE_EXPAND_FILL);
		def_pos = split_container->get_size().x - sep.x;
		MessageQueue::get_singleton()->flush();
		CHECK(split_container->get_split_offset() == 20);
		CHECK_RECTS(get_rects(split_container, def_pos, sep.x), get_child_rects(split_container));

		child_a->set_h_size_flags(Control::SIZE_FILL);
		child_b->set_h_size_flags(Control::SIZE_EXPAND_FILL);
		def_pos = 0;
		MessageQueue::get_singleton()->flush();
		CHECK(split_container->get_split_offset() == 20);
		CHECK_RECTS(get_rects(split_container, def_pos + 20, sep.x), get_child_rects(split_container));

		child_a->set_h_size_flags(Control::SIZE_EXPAND_FILL);
		child_b->set_h_size_flags(Control::SIZE_EXPAND_FILL);
		def_pos = (split_container->get_size().x - sep.x) / 2;
		MessageQueue::get_singleton()->flush();
		CHECK(split_container->get_split_offset() == 20);
		CHECK_RECTS(get_rects(split_container, def_pos + 20, sep.x), get_child_rects(split_container));

		child_a->set_h_size_flags(Control::SIZE_FILL);
		child_b->set_h_size_flags(Control::SIZE_FILL);
		def_pos = 0;
		MessageQueue::get_singleton()->flush();
		CHECK(split_container->get_split_offset() == 20);
		CHECK_RECTS(get_rects(split_container, def_pos + 20, sep.x), get_child_rects(split_container));
	}

	SUBCASE("[SplitContainer] Keep split offset when moving children") {
		int def_pos = 0;
		split_container->set_split_offset(100);

		split_container->move_child(child_a, 1);
		MessageQueue::get_singleton()->flush();
		CHECK(split_container->get_split_offset() == 100);
		CHECK_RECTS(get_rects(split_container, def_pos + 100, sep.x), get_child_rects(split_container));
	}

	SUBCASE("[SplitContainer] Resize split container") {
		SUBCASE("[SplitContainer] No expand flags") {
			int def_pos = 0;
			// Increase the size.
			split_container->set_size(Size2(600, 500));
			MessageQueue::get_singleton()->flush();
			CHECK_RECTS(get_rects(split_container, def_pos, sep.x), get_child_rects(split_container));

			// Decrease the size.
			split_container->set_size(Size2(400, 500));
			MessageQueue::get_singleton()->flush();
			CHECK_RECTS(get_rects(split_container, def_pos, sep.x), get_child_rects(split_container));

			// Change size with a split offset.
			split_container->set_split_offset(100);
			MessageQueue::get_singleton()->flush();
			CHECK_RECTS(get_rects(split_container, def_pos + 100, sep.x), get_child_rects(split_container));

			split_container->set_size(Size2(500, 500));
			MessageQueue::get_singleton()->flush();
			CHECK_RECTS(get_rects(split_container, def_pos + 100, sep.x), get_child_rects(split_container));
			CHECK(split_container->get_split_offset() == 100);

			// Change size so that the first child changes size.
			split_container->set_size(Size2(80, 500));
			MessageQueue::get_singleton()->flush();
			CHECK_RECTS(get_rects(split_container, 80 - sep.x, sep.x), get_child_rects(split_container));
			CHECK(split_container->get_split_offset() == 100);

			// Increase size again.
			split_container->set_size(Size2(500, 500));
			MessageQueue::get_singleton()->flush();
			CHECK_RECTS(get_rects(split_container, def_pos + 100, sep.x), get_child_rects(split_container));
			CHECK(split_container->get_split_offset() == 100);
		}

		SUBCASE("[SplitContainer] First child expanded") {
			int def_pos = split_container->get_size().x - sep.x;
			child_a->set_h_size_flags(Control::SIZE_EXPAND_FILL);
			MessageQueue::get_singleton()->flush();

			// Increase the size.
			split_container->set_size(Size2(600, 500));
			def_pos = split_container->get_size().x - sep.x;
			MessageQueue::get_singleton()->flush();
			CHECK_RECTS(get_rects(split_container, def_pos, sep.x), get_child_rects(split_container));

			// Decrease the size.
			split_container->set_size(Size2(400, 500));
			def_pos = split_container->get_size().x - sep.x;
			MessageQueue::get_singleton()->flush();
			CHECK_RECTS(get_rects(split_container, def_pos, sep.x), get_child_rects(split_container));

			// Change size with a split offset.
			split_container->set_split_offset(-100);
			MessageQueue::get_singleton()->flush();
			CHECK_RECTS(get_rects(split_container, def_pos - 100, sep.x), get_child_rects(split_container));

			split_container->set_size(Size2(500, 500));
			def_pos = split_container->get_size().x - sep.x;
			MessageQueue::get_singleton()->flush();
			CHECK_RECTS(get_rects(split_container, def_pos - 100, sep.x), get_child_rects(split_container));
			CHECK(split_container->get_split_offset() == -100);

			// Change size so that the second child changes size.
			split_container->set_size(Size2(80, 500));
			MessageQueue::get_singleton()->flush();
			CHECK_RECTS(get_rects(split_container, 0, sep.x), get_child_rects(split_container));
			CHECK(split_container->get_split_offset() == -100);

			// Increase size again.
			split_container->set_size(Size2(500, 500));
			MessageQueue::get_singleton()->flush();
			CHECK_RECTS(get_rects(split_container, def_pos - 100, sep.x), get_child_rects(split_container));
			CHECK(split_container->get_split_offset() == -100);
		}

		SUBCASE("[SplitContainer] Second child expanded") {
			int def_pos = 0;
			child_b->set_h_size_flags(Control::SIZE_EXPAND_FILL);
			MessageQueue::get_singleton()->flush();

			// Increase the size.
			split_container->set_size(Size2(600, 500));
			MessageQueue::get_singleton()->flush();
			CHECK_RECTS(get_rects(split_container, def_pos, sep.x), get_child_rects(split_container));

			// Decrease the size.
			split_container->set_size(Size2(400, 500));
			MessageQueue::get_singleton()->flush();
			CHECK_RECTS(get_rects(split_container, def_pos, sep.x), get_child_rects(split_container));

			// Change size with a split offset.
			split_container->set_split_offset(100);
			MessageQueue::get_singleton()->flush();
			CHECK_RECTS(get_rects(split_container, def_pos + 100, sep.x), get_child_rects(split_container));

			split_container->set_size(Size2(500, 500));
			MessageQueue::get_singleton()->flush();
			CHECK_RECTS(get_rects(split_container, def_pos + 100, sep.x), get_child_rects(split_container));
			CHECK(split_container->get_split_offset() == 100);

			// Change size so that the first child changes size.
			split_container->set_size(Size2(80, 500));
			MessageQueue::get_singleton()->flush();
			CHECK_RECTS(get_rects(split_container, 80 - sep.x, sep.x), get_child_rects(split_container));
			CHECK(split_container->get_split_offset() == 100);

			// Increase size again.
			split_container->set_size(Size2(500, 500));
			MessageQueue::get_singleton()->flush();
			CHECK_RECTS(get_rects(split_container, def_pos + 100, sep.x), get_child_rects(split_container));
			CHECK(split_container->get_split_offset() == 100);
		}

		SUBCASE("[SplitContainer] Both children expanded") {
			int def_pos = (split_container->get_size().x - sep.x) / 2;
			child_a->set_h_size_flags(Control::SIZE_EXPAND_FILL);
			child_b->set_h_size_flags(Control::SIZE_EXPAND_FILL);
			MessageQueue::get_singleton()->flush();

			// Increase the size.
			split_container->set_size(Size2(600, 500));
			def_pos = (split_container->get_size().x - sep.x) / 2;
			MessageQueue::get_singleton()->flush();
			CHECK_RECTS(get_rects(split_container, def_pos, sep.x), get_child_rects(split_container));

			// Decrease the size.
			split_container->set_size(Size2(400, 500));
			def_pos = (split_container->get_size().x - sep.x) / 2;
			MessageQueue::get_singleton()->flush();
			CHECK_RECTS(get_rects(split_container, def_pos, sep.x), get_child_rects(split_container));

			// Change size with a split offset.
			split_container->set_split_offset(100);
			MessageQueue::get_singleton()->flush();
			CHECK_RECTS(get_rects(split_container, def_pos + 100, sep.x), get_child_rects(split_container));

			split_container->set_size(Size2(500, 500));
			def_pos = (split_container->get_size().x - sep.x) / 2;
			MessageQueue::get_singleton()->flush();
			CHECK_RECTS(get_rects(split_container, def_pos + 100, sep.x), get_child_rects(split_container));
			CHECK(split_container->get_split_offset() == 100);

			// Change size so that the second child is minimized.
			split_container->set_size(Size2(80, 500));
			MessageQueue::get_singleton()->flush();
			CHECK_RECTS(get_rects(split_container, 80 - sep.x, sep.x), get_child_rects(split_container));
			CHECK(split_container->get_split_offset() == 100);

			// Increase size again.
			split_container->set_size(Size2(500, 500));
			def_pos = (split_container->get_size().x - sep.x) / 2;
			MessageQueue::get_singleton()->flush();
			CHECK_RECTS(get_rects(split_container, def_pos + 100, sep.x), get_child_rects(split_container));
			CHECK(split_container->get_split_offset() == 100);
		}

		SUBCASE("[SplitContainer] Unequal stretch ratios") {
			int def_pos = (split_container->get_size().x * 2 / 3) - sep.x / 2;
			child_a->set_h_size_flags(Control::SIZE_EXPAND_FILL);
			child_b->set_h_size_flags(Control::SIZE_EXPAND_FILL);
			child_a->set_stretch_ratio(2.0);
			MessageQueue::get_singleton()->flush();

			// Increase the size.
			split_container->set_size(Size2(600, 500));
			def_pos = (split_container->get_size().x * 2 / 3) - sep.x / 2;
			MessageQueue::get_singleton()->flush();
			CHECK_RECTS(get_rects(split_container, def_pos, sep.x), get_child_rects(split_container));

			// Decrease the size.
			split_container->set_size(Size2(400, 500));
			def_pos = (split_container->get_size().x * 2 / 3) - sep.x / 2;
			MessageQueue::get_singleton()->flush();
			CHECK_RECTS(get_rects(split_container, def_pos, sep.x), get_child_rects(split_container));

			// Change size with a split offset.
			split_container->set_split_offset(100);
			MessageQueue::get_singleton()->flush();
			CHECK_RECTS(get_rects(split_container, def_pos + 100, sep.x), get_child_rects(split_container));

			split_container->set_size(Size2(500, 500));
			def_pos = (split_container->get_size().x * 2 / 3) - sep.x / 2;
			MessageQueue::get_singleton()->flush();
			CHECK_RECTS(get_rects(split_container, def_pos + 100, sep.x), get_child_rects(split_container));
			CHECK(split_container->get_split_offset() == 100);

			// Change size so that the second child is minimized.
			split_container->set_size(Size2(80, 500));
			MessageQueue::get_singleton()->flush();
			CHECK_RECTS(get_rects(split_container, 80 - sep.x, sep.x), get_child_rects(split_container));
			CHECK(split_container->get_split_offset() == 100);

			// Increase size again.
			split_container->set_size(Size2(500, 500));
			def_pos = (split_container->get_size().x * 2 / 3) - sep.x / 2;
			MessageQueue::get_singleton()->flush();
			CHECK_RECTS(get_rects(split_container, def_pos + 100, sep.x), get_child_rects(split_container));
			CHECK(split_container->get_split_offset() == 100);
		}
	}

	SUBCASE("[SplitContainer] Drag") {
		SUBCASE("[SplitContainer] Vertical, no expand flags") {
			SIGNAL_WATCH(split_container, "dragged");
			Array signal_args = { { 0 } };

			split_container->set_vertical(true);
			Point2 mouse_offset = Point2(1, 1);
			int dragger_pos = 0;
			int split_dragger_ofs = 0;

			// Grab the dragger.
			SEND_GUI_MOUSE_BUTTON_EVENT(mouse_offset + Point2(0, dragger_pos), MouseButton::LEFT, MouseButtonMask::LEFT, Key::NONE);
			MessageQueue::get_singleton()->flush();
			CHECK_RECTS(get_rects(split_container, dragger_pos, sep.y, false), get_child_rects(split_container));
			CHECK(split_container->get_split_offset() == dragger_pos + split_dragger_ofs);
			SIGNAL_CHECK_FALSE("dragged");

			// Move the dragger.
			dragger_pos = 10;
			SEND_GUI_MOUSE_MOTION_EVENT(mouse_offset + Point2(0, dragger_pos), MouseButtonMask::LEFT, Key::NONE);
			MessageQueue::get_singleton()->flush();
			CHECK_RECTS(get_rects(split_container, dragger_pos, sep.y, false), get_child_rects(split_container));
			CHECK(split_container->get_split_offset() == dragger_pos + split_dragger_ofs);
			// It is clamped.
			split_container->clamp_split_offset();
			CHECK(split_container->get_split_offset() == dragger_pos + split_dragger_ofs);
			((Array)signal_args[0])[0] = split_container->get_split_offset();
			SIGNAL_CHECK("dragged", signal_args);

			// Move down.
			dragger_pos = 400;
			SEND_GUI_MOUSE_MOTION_EVENT(mouse_offset + Point2(0, dragger_pos), MouseButtonMask::LEFT, Key::NONE);
			MessageQueue::get_singleton()->flush();
			CHECK_RECTS(get_rects(split_container, dragger_pos, sep.y, false), get_child_rects(split_container));
			CHECK(split_container->get_split_offset() == dragger_pos + split_dragger_ofs);
			((Array)signal_args[0])[0] = split_container->get_split_offset();
			SIGNAL_CHECK("dragged", signal_args);

			// Moves even when mouse is outside.
			dragger_pos = split_container->get_size().y - sep.y;
			SEND_GUI_MOUSE_MOTION_EVENT(mouse_offset + Point2(0, 1000), MouseButtonMask::LEFT, Key::NONE);
			MessageQueue::get_singleton()->flush();
			CHECK_RECTS(get_rects(split_container, dragger_pos, sep.y, false), get_child_rects(split_container));
			CHECK(split_container->get_split_offset() == dragger_pos + split_dragger_ofs);
			((Array)signal_args[0])[0] = split_container->get_split_offset();
			SIGNAL_CHECK("dragged", signal_args);

			// Move up.
			dragger_pos = 100;
			SEND_GUI_MOUSE_MOTION_EVENT(mouse_offset + Point2(0, dragger_pos), MouseButtonMask::LEFT, Key::NONE);
			MessageQueue::get_singleton()->flush();
			CHECK_RECTS(get_rects(split_container, dragger_pos, sep.y, false), get_child_rects(split_container));
			CHECK(split_container->get_split_offset() == dragger_pos + split_dragger_ofs);
			((Array)signal_args[0])[0] = split_container->get_split_offset();
			SIGNAL_CHECK("dragged", signal_args);

			// Release.
			SEND_GUI_MOUSE_BUTTON_RELEASED_EVENT(mouse_offset + Point2(0, dragger_pos), MouseButton::LEFT, MouseButtonMask::NONE, Key::NONE);
			MessageQueue::get_singleton()->flush();
			CHECK_RECTS(get_rects(split_container, dragger_pos, sep.y, false), get_child_rects(split_container));
			CHECK(split_container->get_split_offset() == dragger_pos + split_dragger_ofs);
			SIGNAL_CHECK_FALSE("dragged");

			// No longer moves with the mouse.
			SEND_GUI_MOUSE_MOTION_EVENT(mouse_offset + Point2(0, 200), MouseButtonMask::NONE, Key::NONE);
			MessageQueue::get_singleton()->flush();
			CHECK_RECTS(get_rects(split_container, dragger_pos, sep.y, false), get_child_rects(split_container));
			CHECK(split_container->get_split_offset() == dragger_pos + split_dragger_ofs);
			SIGNAL_CHECK_FALSE("dragged");

			SIGNAL_UNWATCH(split_container, "dragged");
		}

		SUBCASE("[SplitContainer] No expand flags") {
			Point2 mouse_offset = Point2(1, 1);
			int dragger_pos = 0;
			int split_dragger_ofs = 0;

			// Grab the dragger.
			SEND_GUI_MOUSE_BUTTON_EVENT(mouse_offset + Point2(dragger_pos, 0), MouseButton::LEFT, MouseButtonMask::LEFT, Key::NONE);
			MessageQueue::get_singleton()->flush();
			CHECK_RECTS(get_rects(split_container, dragger_pos, sep.x), get_child_rects(split_container));
			CHECK(split_container->get_split_offset() == dragger_pos + split_dragger_ofs);

			// Move the dragger.
			dragger_pos = 10;
			SEND_GUI_MOUSE_MOTION_EVENT(mouse_offset + Point2(dragger_pos, 0), MouseButtonMask::LEFT, Key::NONE);
			MessageQueue::get_singleton()->flush();
			CHECK_RECTS(get_rects(split_container, dragger_pos, sep.x), get_child_rects(split_container));
			CHECK(split_container->get_split_offset() == dragger_pos + split_dragger_ofs);
			// It is clamped.
			split_container->clamp_split_offset();
			CHECK(split_container->get_split_offset() == dragger_pos + split_dragger_ofs);

			// Continue moving.
			dragger_pos = 400;
			SEND_GUI_MOUSE_MOTION_EVENT(mouse_offset + Point2(dragger_pos, 0), MouseButtonMask::LEFT, Key::NONE);
			MessageQueue::get_singleton()->flush();
			CHECK_RECTS(get_rects(split_container, dragger_pos, sep.x), get_child_rects(split_container));
			CHECK(split_container->get_split_offset() == dragger_pos + split_dragger_ofs);

			// Moves even when mouse is outside.
			dragger_pos = split_container->get_size().x - sep.x;
			SEND_GUI_MOUSE_MOTION_EVENT(mouse_offset + Point2(1000, 0), MouseButtonMask::LEFT, Key::NONE);
			MessageQueue::get_singleton()->flush();
			CHECK_RECTS(get_rects(split_container, dragger_pos, sep.x), get_child_rects(split_container));
			CHECK(split_container->get_split_offset() == dragger_pos + split_dragger_ofs);

			// Move back in.
			dragger_pos = 100;
			SEND_GUI_MOUSE_MOTION_EVENT(mouse_offset + Point2(dragger_pos, 0), MouseButtonMask::LEFT, Key::NONE);
			MessageQueue::get_singleton()->flush();
			CHECK_RECTS(get_rects(split_container, dragger_pos, sep.x), get_child_rects(split_container));
			CHECK(split_container->get_split_offset() == dragger_pos + split_dragger_ofs);

			// Release.
			SEND_GUI_MOUSE_BUTTON_RELEASED_EVENT(mouse_offset + Point2(dragger_pos, 0), MouseButton::LEFT, MouseButtonMask::NONE, Key::NONE);
			MessageQueue::get_singleton()->flush();
			CHECK_RECTS(get_rects(split_container, dragger_pos, sep.x), get_child_rects(split_container));
			CHECK(split_container->get_split_offset() == dragger_pos + split_dragger_ofs);

			// No longer moves with the mouse.
			SEND_GUI_MOUSE_MOTION_EVENT(mouse_offset + Point2(200, 0), MouseButtonMask::NONE, Key::NONE);
			MessageQueue::get_singleton()->flush();
			CHECK_RECTS(get_rects(split_container, dragger_pos, sep.x), get_child_rects(split_container));
			CHECK(split_container->get_split_offset() == dragger_pos + split_dragger_ofs);
		}

		SUBCASE("[SplitContainer] First child expanded") {
			child_a->set_h_size_flags(Control::SIZE_EXPAND_FILL);
			MessageQueue::get_singleton()->flush();
			Point2 mouse_offset = Point2(1, 1);
			int dragger_pos = split_container->get_size().x - sep.x;
			int split_dragger_ofs = -dragger_pos;

			// Grab the dragger.
			SEND_GUI_MOUSE_BUTTON_EVENT(mouse_offset + Point2(dragger_pos, 0), MouseButton::LEFT, MouseButtonMask::LEFT, Key::NONE);
			MessageQueue::get_singleton()->flush();
			CHECK_RECTS(get_rects(split_container, dragger_pos, sep.x), get_child_rects(split_container));
			CHECK(split_container->get_split_offset() == dragger_pos + split_dragger_ofs);

			// Move the dragger.
			dragger_pos -= 10;
			SEND_GUI_MOUSE_MOTION_EVENT(mouse_offset + Point2(dragger_pos, 0), MouseButtonMask::LEFT, Key::NONE);
			MessageQueue::get_singleton()->flush();
			CHECK_RECTS(get_rects(split_container, dragger_pos, sep.x), get_child_rects(split_container));
			CHECK(split_container->get_split_offset() == dragger_pos + split_dragger_ofs);
			// It is clamped.
			split_container->clamp_split_offset();
			CHECK(split_container->get_split_offset() == dragger_pos + split_dragger_ofs);

			// Continue moving.
			dragger_pos = 400;
			SEND_GUI_MOUSE_MOTION_EVENT(mouse_offset + Point2(dragger_pos, 0), MouseButtonMask::LEFT, Key::NONE);
			MessageQueue::get_singleton()->flush();
			CHECK_RECTS(get_rects(split_container, dragger_pos, sep.x), get_child_rects(split_container));
			CHECK(split_container->get_split_offset() == dragger_pos + split_dragger_ofs);

			// Moves even when mouse is outside.
			dragger_pos = split_container->get_size().x - sep.x;
			SEND_GUI_MOUSE_MOTION_EVENT(mouse_offset + Point2(1000, 0), MouseButtonMask::LEFT, Key::NONE);
			MessageQueue::get_singleton()->flush();
			CHECK_RECTS(get_rects(split_container, dragger_pos, sep.x), get_child_rects(split_container));
			CHECK(split_container->get_split_offset() == dragger_pos + split_dragger_ofs);

			// Move back in.
			dragger_pos = 100;
			SEND_GUI_MOUSE_MOTION_EVENT(mouse_offset + Point2(dragger_pos, 0), MouseButtonMask::LEFT, Key::NONE);
			MessageQueue::get_singleton()->flush();
			CHECK_RECTS(get_rects(split_container, dragger_pos, sep.x), get_child_rects(split_container));
			CHECK(split_container->get_split_offset() == dragger_pos + split_dragger_ofs);

			// Release.
			SEND_GUI_MOUSE_BUTTON_RELEASED_EVENT(mouse_offset + Point2(dragger_pos, 0), MouseButton::LEFT, MouseButtonMask::NONE, Key::NONE);
			MessageQueue::get_singleton()->flush();
			CHECK_RECTS(get_rects(split_container, dragger_pos, sep.x), get_child_rects(split_container));
			CHECK(split_container->get_split_offset() == dragger_pos + split_dragger_ofs);

			// No longer moves with the mouse.
			SEND_GUI_MOUSE_MOTION_EVENT(mouse_offset + Point2(200, 0), MouseButtonMask::NONE, Key::NONE);
			MessageQueue::get_singleton()->flush();
			CHECK_RECTS(get_rects(split_container, dragger_pos, sep.x), get_child_rects(split_container));
			CHECK(split_container->get_split_offset() == dragger_pos + split_dragger_ofs);
		}

		SUBCASE("[SplitContainer] Second child expanded") {
			child_b->set_h_size_flags(Control::SIZE_EXPAND_FILL);
			MessageQueue::get_singleton()->flush();
			Point2 mouse_offset = Point2(1, 1);
			int dragger_pos = 0;
			int split_dragger_ofs = 0;

			// Grab the dragger.
			SEND_GUI_MOUSE_BUTTON_EVENT(mouse_offset + Point2(dragger_pos, 0), MouseButton::LEFT, MouseButtonMask::LEFT, Key::NONE);
			MessageQueue::get_singleton()->flush();
			CHECK_RECTS(get_rects(split_container, dragger_pos, sep.x), get_child_rects(split_container));
			CHECK(split_container->get_split_offset() == dragger_pos + split_dragger_ofs);

			// Move the dragger.
			dragger_pos = 10;
			SEND_GUI_MOUSE_MOTION_EVENT(mouse_offset + Point2(dragger_pos, 0), MouseButtonMask::LEFT, Key::NONE);
			MessageQueue::get_singleton()->flush();
			CHECK_RECTS(get_rects(split_container, dragger_pos, sep.x), get_child_rects(split_container));
			CHECK(split_container->get_split_offset() == dragger_pos + split_dragger_ofs);
			// It is clamped.
			split_container->clamp_split_offset();
			CHECK(split_container->get_split_offset() == dragger_pos + split_dragger_ofs);

			// Continue moving.
			dragger_pos = 400;
			SEND_GUI_MOUSE_MOTION_EVENT(mouse_offset + Point2(dragger_pos, 0), MouseButtonMask::LEFT, Key::NONE);
			MessageQueue::get_singleton()->flush();
			CHECK_RECTS(get_rects(split_container, dragger_pos, sep.x), get_child_rects(split_container));
			CHECK(split_container->get_split_offset() == dragger_pos + split_dragger_ofs);

			// Moves even when mouse is outside.
			dragger_pos = split_container->get_size().x - sep.x;
			SEND_GUI_MOUSE_MOTION_EVENT(mouse_offset + Point2(1000, 0), MouseButtonMask::LEFT, Key::NONE);
			MessageQueue::get_singleton()->flush();
			CHECK_RECTS(get_rects(split_container, dragger_pos, sep.x), get_child_rects(split_container));
			CHECK(split_container->get_split_offset() == dragger_pos + split_dragger_ofs);

			// Move back in.
			dragger_pos = 100;
			SEND_GUI_MOUSE_MOTION_EVENT(mouse_offset + Point2(dragger_pos, 0), MouseButtonMask::LEFT, Key::NONE);
			MessageQueue::get_singleton()->flush();
			CHECK_RECTS(get_rects(split_container, dragger_pos, sep.x), get_child_rects(split_container));
			CHECK(split_container->get_split_offset() == dragger_pos + split_dragger_ofs);

			// Release.
			SEND_GUI_MOUSE_BUTTON_RELEASED_EVENT(mouse_offset + Point2(dragger_pos, 0), MouseButton::LEFT, MouseButtonMask::NONE, Key::NONE);
			MessageQueue::get_singleton()->flush();
			CHECK_RECTS(get_rects(split_container, dragger_pos, sep.x), get_child_rects(split_container));
			CHECK(split_container->get_split_offset() == dragger_pos + split_dragger_ofs);

			// No longer moves with the mouse.
			SEND_GUI_MOUSE_MOTION_EVENT(mouse_offset + Point2(200, 0), MouseButtonMask::NONE, Key::NONE);
			MessageQueue::get_singleton()->flush();
			CHECK_RECTS(get_rects(split_container, dragger_pos, sep.x), get_child_rects(split_container));
			CHECK(split_container->get_split_offset() == dragger_pos + split_dragger_ofs);
		}

		SUBCASE("[SplitContainer] Both children expanded") {
			child_a->set_h_size_flags(Control::SIZE_EXPAND_FILL);
			child_b->set_h_size_flags(Control::SIZE_EXPAND_FILL);
			MessageQueue::get_singleton()->flush();
			Point2 mouse_offset = Point2(1, 1);
			int dragger_pos = (split_container->get_size().x - sep.x) / 2;
			int split_dragger_ofs = -dragger_pos;

			// Grab the dragger.
			SEND_GUI_MOUSE_BUTTON_EVENT(mouse_offset + Point2(dragger_pos, 0), MouseButton::LEFT, MouseButtonMask::LEFT, Key::NONE);
			MessageQueue::get_singleton()->flush();
			CHECK_RECTS(get_rects(split_container, dragger_pos, sep.x), get_child_rects(split_container));
			CHECK(split_container->get_split_offset() == dragger_pos + split_dragger_ofs);

			// Move the dragger.
			dragger_pos += 10;
			SEND_GUI_MOUSE_MOTION_EVENT(mouse_offset + Point2(dragger_pos, 0), MouseButtonMask::LEFT, Key::NONE);
			MessageQueue::get_singleton()->flush();
			CHECK_RECTS(get_rects(split_container, dragger_pos, sep.x), get_child_rects(split_container));
			CHECK(split_container->get_split_offset() == dragger_pos + split_dragger_ofs);
			// It is clamped.
			split_container->clamp_split_offset();
			CHECK(split_container->get_split_offset() == dragger_pos + split_dragger_ofs);

			// Continue moving.
			dragger_pos = 400;
			SEND_GUI_MOUSE_MOTION_EVENT(mouse_offset + Point2(dragger_pos, 0), MouseButtonMask::LEFT, Key::NONE);
			MessageQueue::get_singleton()->flush();
			CHECK_RECTS(get_rects(split_container, dragger_pos, sep.x), get_child_rects(split_container));
			CHECK(split_container->get_split_offset() == dragger_pos + split_dragger_ofs);

			// Moves even when mouse is outside.
			dragger_pos = split_container->get_size().x - sep.x;
			SEND_GUI_MOUSE_MOTION_EVENT(mouse_offset + Point2(1000, 0), MouseButtonMask::LEFT, Key::NONE);
			MessageQueue::get_singleton()->flush();
			CHECK_RECTS(get_rects(split_container, dragger_pos, sep.x), get_child_rects(split_container));
			CHECK(split_container->get_split_offset() == dragger_pos + split_dragger_ofs);

			// Move back in.
			dragger_pos = 100;
			SEND_GUI_MOUSE_MOTION_EVENT(mouse_offset + Point2(dragger_pos, 0), MouseButtonMask::LEFT, Key::NONE);
			MessageQueue::get_singleton()->flush();
			CHECK_RECTS(get_rects(split_container, dragger_pos, sep.x), get_child_rects(split_container));
			CHECK(split_container->get_split_offset() == dragger_pos + split_dragger_ofs);

			// Release.
			SEND_GUI_MOUSE_BUTTON_RELEASED_EVENT(mouse_offset + Point2(dragger_pos, 0), MouseButton::LEFT, MouseButtonMask::NONE, Key::NONE);
			MessageQueue::get_singleton()->flush();
			CHECK_RECTS(get_rects(split_container, dragger_pos, sep.x), get_child_rects(split_container));
			CHECK(split_container->get_split_offset() == dragger_pos + split_dragger_ofs);

			// No longer moves with the mouse.
			SEND_GUI_MOUSE_MOTION_EVENT(mouse_offset + Point2(200, 0), MouseButtonMask::NONE, Key::NONE);
			MessageQueue::get_singleton()->flush();
			CHECK_RECTS(get_rects(split_container, dragger_pos, sep.x), get_child_rects(split_container));
			CHECK(split_container->get_split_offset() == dragger_pos + split_dragger_ofs);
		}
	}

	memdelete(child_b);
	memdelete(child_a);
	memdelete(split_container);
}

TEST_CASE("[SceneTree][SplitContainer] More children") {
	SplitContainer *split_container = memnew(SplitContainer);
	split_container->set_size(Size2(500, 500));
	SceneTree::get_singleton()->get_root()->add_child(split_container);
	Control *child_a = memnew(Control);
	Control *child_b = memnew(Control);
	Control *child_c = memnew(Control);
	split_container->add_child(child_a);
	split_container->add_child(child_b);
	split_container->add_child(child_c);
	Size2i min_size = Size2i(10, 10);
	child_a->set_custom_minimum_size(min_size);
	child_b->set_custom_minimum_size(min_size);
	child_c->set_custom_minimum_size(min_size);
	MessageQueue::get_singleton()->flush();

	const int sep_constant = split_container->get_theme_constant("separation");
	const Size2i sep = Size2i(MAX(sep_constant, split_container->get_theme_icon("h_grabber")->get_width()), MAX(sep_constant, split_container->get_theme_icon("v_grabber")->get_height()));

	SUBCASE("[SplitContainer] Duplicate") {
		// Make sure dynamically added internal draggers duplicate properly.
		SplitContainer *duplicate = (SplitContainer *)(Node *)split_container->duplicate();
		MessageQueue::get_singleton()->flush();
		CHECK(duplicate->get_child_count(false) == split_container->get_child_count(false));
		CHECK(duplicate->get_child_count(true) == split_container->get_child_count(true));
		memdelete(duplicate);
	}

	SUBCASE("[SplitContainer] Default position") {
		CHECK(split_container->get_split_offsets() == Vector<int>({ 0, 0 }));

		set_size_flags(split_container, { -1, -1, -1 }); // None expanded.
		Vector<int> def_pos = { min_size.x, min_size.x * 2 + sep.x };
		CHECK_RECTS(get_rects_multi(split_container, def_pos, sep.x), get_child_rects(split_container));
		split_container->clamp_split_offset();
		MessageQueue::get_singleton()->flush();
		CHECK(split_container->get_split_offsets() == Vector<int>({ min_size.x, min_size.x * 2 + sep.x }));
		CHECK_RECTS(get_rects_multi(split_container, def_pos, sep.x), get_child_rects(split_container));

		split_container->set_split_offsets({ 0, 0 });

		set_size_flags(split_container, { 1, -1, -1 }); // First expanded.
		def_pos = { (int)split_container->get_size().x - sep.x * 2 - min_size.x * 2, (int)split_container->get_size().x - sep.x - min_size.x };
		CHECK_RECTS(get_rects_multi(split_container, def_pos, sep.x), get_child_rects(split_container));
		split_container->clamp_split_offset();
		MessageQueue::get_singleton()->flush();
		CHECK(split_container->get_split_offsets() == Vector<int>({ -min_size.x * 2 - sep.x, -min_size.x }));
		CHECK_RECTS(get_rects_multi(split_container, def_pos, sep.x), get_child_rects(split_container));
		split_container->set_split_offsets({ 0, 0 });

		set_size_flags(split_container, { -1, 1, -1 }); // Second expanded.
		def_pos = { min_size.x, (int)split_container->get_size().x - min_size.x - sep.x };
		CHECK_RECTS(get_rects_multi(split_container, def_pos, sep.x), get_child_rects(split_container));
		split_container->clamp_split_offset();
		MessageQueue::get_singleton()->flush();
		CHECK(split_container->get_split_offsets() == Vector<int>({ min_size.x, -min_size.x }));
		CHECK_RECTS(get_rects_multi(split_container, def_pos, sep.x), get_child_rects(split_container));
		split_container->set_split_offsets({ 0, 0 });

		set_size_flags(split_container, { -1, -1, 1 }); // Third expanded.
		def_pos = { min_size.x, min_size.x * 2 + sep.x };
		CHECK_RECTS(get_rects_multi(split_container, def_pos, sep.x), get_child_rects(split_container));
		split_container->clamp_split_offset();
		MessageQueue::get_singleton()->flush();
		CHECK(split_container->get_split_offsets() == Vector<int>({ min_size.x, min_size.x * 2 + sep.x }));
		CHECK_RECTS(get_rects_multi(split_container, def_pos, sep.x), get_child_rects(split_container));
		split_container->set_split_offsets({ 0, 0 });

		set_size_flags(split_container, { 1, 1, -1 }); // First and second expanded.
		int child_2_expanded_size = ((int)split_container->get_size().x - min_size.x) / 2 - sep.x;
		def_pos = { child_2_expanded_size, (int)split_container->get_size().x - min_size.x - sep.x };
		CHECK_RECTS(get_rects_multi(split_container, def_pos, sep.x), get_child_rects(split_container));
		split_container->clamp_split_offset();
		MessageQueue::get_singleton()->flush();
		CHECK(split_container->get_split_offsets() == Vector<int>({ 0, -min_size.x }));
		CHECK_RECTS(get_rects_multi(split_container, def_pos, sep.x), get_child_rects(split_container));
		split_container->set_split_offsets({ 0, 0 });

		set_size_flags(split_container, { 1, -1, 1 }); // First and third expanded.
		def_pos = { child_2_expanded_size, child_2_expanded_size + min_size.x + sep.x };
		CHECK_RECTS(get_rects_multi(split_container, def_pos, sep.x), get_child_rects(split_container));
		split_container->clamp_split_offset();
		MessageQueue::get_singleton()->flush();
		CHECK(split_container->get_split_offsets() == Vector<int>({ 0, 0 }));
		CHECK_RECTS(get_rects_multi(split_container, def_pos, sep.x), get_child_rects(split_container));
		split_container->set_split_offsets({ 0, 0 });

		set_size_flags(split_container, { -1, 1, 1 }); // Second and third expanded.
		def_pos = { min_size.x, min_size.x + child_2_expanded_size + sep.x };
		CHECK_RECTS(get_rects_multi(split_container, def_pos, sep.x), get_child_rects(split_container));
		split_container->clamp_split_offset();
		MessageQueue::get_singleton()->flush();
		CHECK(split_container->get_split_offsets() == Vector<int>({ min_size.x, 0 }));
		CHECK_RECTS(get_rects_multi(split_container, def_pos, sep.x), get_child_rects(split_container));
		split_container->set_split_offsets({ 0, 0 });

		set_size_flags(split_container, { 1, 1, 1 }); // All expanded.
		int child_3_expanded_size = (split_container->get_size().x - sep.x * 2) / 3;
		// Add 1 due to pixel error accumulation.
		def_pos = { child_3_expanded_size, child_3_expanded_size * 2 + sep.x + 1 };
		CHECK_RECTS(get_rects_multi(split_container, def_pos, sep.x), get_child_rects(split_container));
		split_container->clamp_split_offset();
		MessageQueue::get_singleton()->flush();
		CHECK(split_container->get_split_offsets() == Vector<int>({ 0, 0 }));
		CHECK_RECTS(get_rects_multi(split_container, def_pos, sep.x), get_child_rects(split_container));
		split_container->set_split_offsets({ 0, 0 });

		set_size_flags(split_container, { 1, 2, 3 }); // All expanded, different ratios.
		int child_6_expanded_size = (split_container->get_size().x - sep.x * 2) / 6;
		def_pos = { child_6_expanded_size, child_6_expanded_size * 3 + sep.x + 1 };
		CHECK_RECTS(get_rects_multi(split_container, def_pos, sep.x), get_child_rects(split_container));
		split_container->clamp_split_offset();
		MessageQueue::get_singleton()->flush();
		CHECK(split_container->get_split_offsets() == Vector<int>({ 0, 0 }));
		CHECK_RECTS(get_rects_multi(split_container, def_pos, sep.x), get_child_rects(split_container));
		split_container->set_split_offsets({ 0, 0 });
	}

	SUBCASE("[SplitContainer] Set split offset") {
		const int expanded_single_size = (int)split_container->get_size().x - min_size.x * 2 - sep.x * 2;

		SUBCASE("[SplitContainer] No expand flags") {
			set_size_flags(split_container, { -1, -1, -1 }); // None expanded.

			// First is positive.
			split_container->set_split_offsets({ 50, 0 });
			MessageQueue::get_singleton()->flush();
			CHECK(split_container->get_split_offsets() == Vector<int>{ 50, 0 });
			CHECK_RECTS(get_rects_multi(split_container, { 50, 50 + min_size.x + sep.x }, sep.x), get_child_rects(split_container));

			// Second is positive.
			split_container->set_split_offsets({ 0, 50 });
			MessageQueue::get_singleton()->flush();
			CHECK(split_container->get_split_offsets() == Vector<int>{ 0, 50 });
			CHECK_RECTS(get_rects_multi(split_container, { min_size.x, 50 }, sep.x), get_child_rects(split_container));

			// Both are positive and equal, the first will override since they both start at 0.
			split_container->set_split_offsets({ 50, 50 });
			MessageQueue::get_singleton()->flush();
			CHECK(split_container->get_split_offsets() == Vector<int>{ 50, 50 });
			CHECK_RECTS(get_rects_multi(split_container, { 50, 50 + min_size.x + sep.x }, sep.x), get_child_rects(split_container));

			// Both are negative and clamped.
			split_container->set_split_offsets({ -50, -50 });
			MessageQueue::get_singleton()->flush();
			CHECK(split_container->get_split_offsets() == Vector<int>{ -50, -50 });
			CHECK_RECTS(get_rects_multi(split_container, { min_size.x, min_size.x * 2 + sep.x }, sep.x), get_child_rects(split_container));

			// First positive, second negative. First takes priority.
			split_container->set_split_offsets({ 50, -50 });
			MessageQueue::get_singleton()->flush();
			CHECK(split_container->get_split_offsets() == Vector<int>{ 50, -50 });
			CHECK_RECTS(get_rects_multi(split_container, { 50, 50 + min_size.x + sep.x }, sep.x), get_child_rects(split_container));

			// First is clamped and pushes second to the end.
			split_container->set_split_offsets({ 1000, 0 });
			MessageQueue::get_singleton()->flush();
			CHECK(split_container->get_split_offsets() == Vector<int>{ 1000, 0 });
			CHECK_RECTS(get_rects_multi(split_container, { expanded_single_size, expanded_single_size + min_size.x + sep.x }, sep.x), get_child_rects(split_container));

			// Second is clamped.
			split_container->set_split_offsets({ 0, 1000 });
			MessageQueue::get_singleton()->flush();
			CHECK(split_container->get_split_offsets() == Vector<int>{ 0, 1000 });
			CHECK_RECTS(get_rects_multi(split_container, { min_size.x, expanded_single_size + min_size.x + sep.x }, sep.x), get_child_rects(split_container));

			// Both are clamped positively, first one takes priority.
			split_container->set_split_offsets({ 1000, 1000 });
			MessageQueue::get_singleton()->flush();
			CHECK(split_container->get_split_offsets() == Vector<int>{ 1000, 1000 });
			CHECK_RECTS(get_rects_multi(split_container, { expanded_single_size, expanded_single_size + min_size.x + sep.x }, sep.x), get_child_rects(split_container));
		}

		SUBCASE("[SplitContainer] First child expanded") {
			set_size_flags(split_container, { 1, -1, -1 }); // First expanded.

			// First is positive and clamped.
			split_container->set_split_offsets({ 50, 0 });
			MessageQueue::get_singleton()->flush();
			CHECK(split_container->get_split_offsets() == Vector<int>{ 50, 0 });
			CHECK_RECTS(get_rects_multi(split_container, { expanded_single_size, expanded_single_size + min_size.x + sep.x }, sep.x), get_child_rects(split_container));

			// Second is positive and clamped.
			split_container->set_split_offsets({ 0, 50 });
			MessageQueue::get_singleton()->flush();
			CHECK(split_container->get_split_offsets() == Vector<int>{ 0, 50 });
			CHECK_RECTS(get_rects_multi(split_container, { expanded_single_size, expanded_single_size + min_size.x + sep.x }, sep.x), get_child_rects(split_container));

			// First is negative and moves left.
			split_container->set_split_offsets({ -50, 0 });
			MessageQueue::get_singleton()->flush();
			CHECK(split_container->get_split_offsets() == Vector<int>{ -50, 0 });
			CHECK_RECTS(get_rects_multi(split_container, { (int)split_container->get_size().x - 50 - sep.x, (int)split_container->get_size().x - min_size.x - sep.x }, sep.x), get_child_rects(split_container));

			// Second is negative, but first has priority so it doesn't move.
			split_container->set_split_offsets({ 0, -50 });
			MessageQueue::get_singleton()->flush();
			CHECK(split_container->get_split_offsets() == Vector<int>{ 0, -50 });
			CHECK_RECTS(get_rects_multi(split_container, { expanded_single_size, expanded_single_size + min_size.x + sep.x }, sep.x), get_child_rects(split_container));

			// Both are negative and equal, they move left but the second doesn't move as much as wanted.
			split_container->set_split_offsets({ -50, -50 });
			MessageQueue::get_singleton()->flush();
			CHECK(split_container->get_split_offsets() == Vector<int>{ -50, -50 });
			CHECK_RECTS(get_rects_multi(split_container, { (int)split_container->get_size().x - 50 - sep.x, (int)split_container->get_size().x - 50 + min_size.x }, sep.x), get_child_rects(split_container));

			// Both are negative with space and move left.
			split_container->set_split_offsets({ -100, -50 });
			MessageQueue::get_singleton()->flush();
			CHECK(split_container->get_split_offsets() == Vector<int>{ -100, -50 });
			CHECK_RECTS(get_rects_multi(split_container, { (int)split_container->get_size().x - 100 - sep.x, (int)split_container->get_size().x - 50 - sep.x }, sep.x), get_child_rects(split_container));

			// First moves all the way left.
			split_container->set_split_offsets({ -1000, 0 });
			MessageQueue::get_singleton()->flush();
			CHECK(split_container->get_split_offsets() == Vector<int>{ -1000, 0 });
			CHECK_RECTS(get_rects_multi(split_container, { min_size.x, expanded_single_size + min_size.x + sep.x }, sep.x), get_child_rects(split_container));

			// Second cannot move all the way left since first takes priority.
			split_container->set_split_offsets({ 0, -1000 });
			MessageQueue::get_singleton()->flush();
			CHECK(split_container->get_split_offsets() == Vector<int>{ 0, -1000 });
			CHECK_RECTS(get_rects_multi(split_container, { expanded_single_size, expanded_single_size + min_size.x + sep.x }, sep.x), get_child_rects(split_container));

			// First and second move all the way left.
			split_container->set_split_offsets({ -1000, -1000 });
			MessageQueue::get_singleton()->flush();
			CHECK(split_container->get_split_offsets() == Vector<int>{ -1000, -1000 });
			CHECK_RECTS(get_rects_multi(split_container, { min_size.x, min_size.x * 2 + sep.x }, sep.x), get_child_rects(split_container));
		}

		SUBCASE("[SplitContainer] All children expanded") {
			set_size_flags(split_container, { 1, 1, 1 }); // All expanded.
			const int child_3_expanded_size = (split_container->get_size().x - sep.x * 2) / 3;

			// First is moved positive, does not affect second.
			split_container->set_split_offsets({ 50, 0 });
			MessageQueue::get_singleton()->flush();
			CHECK(split_container->get_split_offsets() == Vector<int>{ 50, 0 });
			CHECK_RECTS(get_rects_multi(split_container, { child_3_expanded_size + 50, child_3_expanded_size * 2 + sep.x + 1 }, sep.x), get_child_rects(split_container));

			// First is moved negative, does not affect second.
			split_container->set_split_offsets({ -50, 0 });
			MessageQueue::get_singleton()->flush();
			CHECK(split_container->get_split_offsets() == Vector<int>{ -50, 0 });
			CHECK_RECTS(get_rects_multi(split_container, { child_3_expanded_size - 50, child_3_expanded_size * 2 + sep.x + 1 }, sep.x), get_child_rects(split_container));

			// Second is moved positive, does not affect first.
			split_container->set_split_offsets({ 0, 50 });
			MessageQueue::get_singleton()->flush();
			CHECK(split_container->get_split_offsets() == Vector<int>{ 0, 50 });
			CHECK_RECTS(get_rects_multi(split_container, { child_3_expanded_size, child_3_expanded_size * 2 + 50 + sep.x + 1 }, sep.x), get_child_rects(split_container));

			// Second is moved negative, does not affect first.
			split_container->set_split_offsets({ 0, -50 });
			MessageQueue::get_singleton()->flush();
			CHECK(split_container->get_split_offsets() == Vector<int>{ 0, -50 });
			CHECK_RECTS(get_rects_multi(split_container, { child_3_expanded_size, child_3_expanded_size * 2 - 50 + sep.x + 1 }, sep.x), get_child_rects(split_container));

			// First is moved positive enough to affect second.
			split_container->set_split_offsets({ 200, 0 });
			MessageQueue::get_singleton()->flush();
			CHECK(split_container->get_split_offsets() == Vector<int>{ 200, 0 });
			CHECK_RECTS(get_rects_multi(split_container, { child_3_expanded_size + 200, child_3_expanded_size + 200 + sep.x + min_size.x }, sep.x), get_child_rects(split_container));

			// Second is moved enough to pass the first, but the first has priority.
			split_container->set_split_offsets({ 0, -200 });
			MessageQueue::get_singleton()->flush();
			CHECK(split_container->get_split_offsets() == Vector<int>{ 0, -200 });
			CHECK_RECTS(get_rects_multi(split_container, { child_3_expanded_size, child_3_expanded_size + min_size.x + sep.x }, sep.x), get_child_rects(split_container));
		}
	}

	SUBCASE("[SplitContainer] Resize") {
		SUBCASE("[SplitContainer] No expand flags") {
			Vector<int> def_pos = { min_size.x, min_size.x * 2 + sep.x };
			// Increase the size.
			split_container->set_size(Size2(600, 500));
			MessageQueue::get_singleton()->flush();
			CHECK_RECTS(get_rects_multi(split_container, def_pos, sep.x), get_child_rects(split_container));

			// Decrease the size.
			split_container->set_size(Size2(400, 500));
			MessageQueue::get_singleton()->flush();
			CHECK_RECTS(get_rects_multi(split_container, def_pos, sep.x), get_child_rects(split_container));

			// Change size with a split offset.
			split_container->set_split_offsets({ 50, 100 });
			split_container->set_size(Size2(500, 500));
			MessageQueue::get_singleton()->flush();
			CHECK_RECTS(get_rects_multi(split_container, { 50, 100 }, sep.x), get_child_rects(split_container));
			CHECK(split_container->get_split_offsets() == Vector<int>{ 50, 100 });

			// Change size so that the second child gets clamped and changes size.
			split_container->set_size(Size2(100, 500));
			MessageQueue::get_singleton()->flush();
			CHECK_RECTS(get_rects_multi(split_container, { 50, 100 - sep.x - min_size.x }, sep.x), get_child_rects(split_container));
			CHECK(split_container->get_split_offsets() == Vector<int>{ 50, 100 });

			// Change size so that the first child changes size.
			split_container->set_size(Size2(60, 500));
			MessageQueue::get_singleton()->flush();
			CHECK_RECTS(get_rects_multi(split_container, { 60 - min_size.x * 2 - sep.x * 2, 60 - sep.x - min_size.x }, sep.x), get_child_rects(split_container));
			CHECK(split_container->get_split_offsets() == Vector<int>{ 50, 100 });

			// Increase size again.
			split_container->set_size(Size2(500, 500));
			MessageQueue::get_singleton()->flush();
			CHECK_RECTS(get_rects_multi(split_container, { 50, 100 }, sep.x), get_child_rects(split_container));
			CHECK(split_container->get_split_offsets() == Vector<int>{ 50, 100 });
		}

		SUBCASE("[SplitContainer] First child expanded") {
			set_size_flags(split_container, { 1, -1, -1 });
			Vector<int> def_pos = { (int)split_container->get_size().x - sep.x * 2 - min_size.x, (int)split_container->get_size().x - sep.x };
			// Increase the size.
			split_container->set_size(Size2(600, 500));
			def_pos = { (int)split_container->get_size().x - sep.x * 2 - min_size.x * 2, (int)split_container->get_size().x - sep.x - min_size.x };
			MessageQueue::get_singleton()->flush();
			CHECK_RECTS(get_rects_multi(split_container, def_pos, sep.x), get_child_rects(split_container));

			// Decrease the size.
			split_container->set_size(Size2(400, 500));
			def_pos = { (int)split_container->get_size().x - sep.x * 2 - min_size.x * 2, (int)split_container->get_size().x - sep.x - min_size.x };
			MessageQueue::get_singleton()->flush();
			CHECK_RECTS(get_rects_multi(split_container, def_pos, sep.x), get_child_rects(split_container));

			// Change size with a split offset.
			split_container->set_split_offsets({ -100, -50 });
			split_container->set_size(Size2(500, 500));
			MessageQueue::get_singleton()->flush();
			CHECK_RECTS(get_rects_multi(split_container, { (int)split_container->get_size().x - 100 - sep.x, (int)split_container->get_size().x - 50 - sep.x }, sep.x), get_child_rects(split_container));
			CHECK(split_container->get_split_offsets() == Vector<int>{ -100, -50 });

			// Change size so that the first child gets clamped and changes size.
			split_container->set_size(Size2(100, 500));
			MessageQueue::get_singleton()->flush();
			CHECK_RECTS(get_rects_multi(split_container, { min_size.x, (int)split_container->get_size().x - 50 - sep.x }, sep.x), get_child_rects(split_container));
			CHECK(split_container->get_split_offsets() == Vector<int>{ -100, -50 });

			// Change size so that the second child changes size.
			split_container->set_size(Size2(50, 500));
			MessageQueue::get_singleton()->flush();
			CHECK_RECTS(get_rects_multi(split_container, { min_size.x, min_size.x * 2 + sep.x }, sep.x), get_child_rects(split_container));
			CHECK(split_container->get_split_offsets() == Vector<int>{ -100, -50 });

			// Increase size again.
			split_container->set_size(Size2(500, 500));
			MessageQueue::get_singleton()->flush();
			CHECK_RECTS(get_rects_multi(split_container, { (int)split_container->get_size().x - 100 - sep.x, (int)split_container->get_size().x - 50 - sep.x }, sep.x), get_child_rects(split_container));
			CHECK(split_container->get_split_offsets() == Vector<int>{ -100, -50 });
		}

		SUBCASE("[SplitContainer] All children expanded") {
			set_size_flags(split_container, { 1, 1, 1 });
			// Increase the size.
			split_container->set_size(Size2(600, 500));
			MessageQueue::get_singleton()->flush();
			int child_3_expanded_size = (split_container->get_size().x - sep.x * 2) / 3;
			Vector<int> def_pos = { child_3_expanded_size, child_3_expanded_size * 2 + sep.x };
			CHECK_RECTS(get_rects_multi(split_container, def_pos, sep.x), get_child_rects(split_container));

			// Decrease the size.
			split_container->set_size(Size2(400, 500));
			MessageQueue::get_singleton()->flush();
			child_3_expanded_size = (split_container->get_size().x - sep.x * 2) / 3;
			def_pos = { child_3_expanded_size, child_3_expanded_size * 2 + sep.x };
			CHECK_RECTS(get_rects_multi(split_container, def_pos, sep.x), get_child_rects(split_container));

			// Change size with a split offset.
			split_container->set_split_offsets({ -50, 50 });
			split_container->set_size(Size2(500, 500));
			MessageQueue::get_singleton()->flush();
			child_3_expanded_size = (split_container->get_size().x - sep.x * 2) / 3;
			def_pos = { child_3_expanded_size, child_3_expanded_size * 2 + sep.x + 1 };
			CHECK_RECTS(get_rects_multi(split_container, { def_pos[0] - 50, def_pos[1] + 50 }, sep.x), get_child_rects(split_container));
			CHECK(split_container->get_split_offsets() == Vector<int>{ -50, 50 });

			// Change size so that the children get clamped and change sizes.
			split_container->set_size(Size2(100, 500));
			MessageQueue::get_singleton()->flush();
			CHECK_RECTS(get_rects_multi(split_container, { min_size.x, (int)split_container->get_size().x - sep.x - min_size.x }, sep.x), get_child_rects(split_container));
			CHECK(split_container->get_split_offsets() == Vector<int>{ -50, 50 });

			// Increase size again.
			split_container->set_size(Size2(500, 500));
			MessageQueue::get_singleton()->flush();
			child_3_expanded_size = (split_container->get_size().x - sep.x * 2) / 3;
			def_pos = { child_3_expanded_size, child_3_expanded_size * 2 + sep.x + 1 };
			CHECK_RECTS(get_rects_multi(split_container, { def_pos[0] - 50, def_pos[1] + 50 }, sep.x), get_child_rects(split_container));
			CHECK(split_container->get_split_offsets() == Vector<int>{ -50, 50 });
		}
	}

	SUBCASE("[SplitContainer] Visibility changes") {
		set_size_flags(split_container, { -1, -1, -1 }); // None expanded.
		split_container->set_split_offsets({ 50, 122 });
		MessageQueue::get_singleton()->flush();
		Vector<int> def_pos = { 50, 122 };
		CHECK_RECTS(get_rects_multi(split_container, def_pos, sep.x), get_child_rects(split_container));

		// Hide and show the first child.
		child_a->set_visible(false);
		MessageQueue::get_singleton()->flush();
		CHECK(split_container->get_split_offsets() == Vector<int>({ 60 }));
		CHECK_RECTS(get_rects_multi(split_container, { 60 }, sep.x), get_child_rects(split_container));

		child_a->set_visible(true);
		MessageQueue::get_singleton()->flush();
		CHECK(split_container->get_split_offsets() == def_pos);
		CHECK_RECTS(get_rects_multi(split_container, def_pos, sep.x), get_child_rects(split_container));

		// Hide and show the second child.
		child_b->set_visible(false);
		MessageQueue::get_singleton()->flush();
		CHECK(split_container->get_split_offsets() == Vector<int>({ 50 }));
		CHECK_RECTS(get_rects_multi(split_container, { 50 }, sep.x), get_child_rects(split_container));
		child_b->set_visible(true);
		MessageQueue::get_singleton()->flush();
		CHECK(split_container->get_split_offsets() == def_pos);
		CHECK_RECTS(get_rects_multi(split_container, def_pos, sep.x), get_child_rects(split_container));

		// Hide and show the last child.
		child_c->set_visible(false);
		MessageQueue::get_singleton()->flush();
		CHECK(split_container->get_split_offsets() == Vector<int>({ 50 }));
		CHECK_RECTS(get_rects_multi(split_container, { 50 }, sep.x), get_child_rects(split_container));
		child_c->set_visible(true);
		MessageQueue::get_singleton()->flush();
		CHECK(split_container->get_split_offsets() == def_pos);
		CHECK_RECTS(get_rects_multi(split_container, def_pos, sep.x), get_child_rects(split_container));

		set_size_flags(split_container, { 1, 1, 1 }); // All expanded.
		split_container->set_split_offsets({ 50, 60 });
		MessageQueue::get_singleton()->flush();
		int child_3_expanded_size = (split_container->get_size().x - sep.x * 2) / 3;
		def_pos = { child_3_expanded_size + 50, child_3_expanded_size * 2 + sep.x + 1 + 60 };
		CHECK_RECTS(get_rects_multi(split_container, def_pos, sep.x), get_child_rects(split_container));

		// Hide and show the first child.
		int child_2_expanded_size = (split_container->get_size().x - sep.x) / 2;
		child_a->set_visible(false);
		MessageQueue::get_singleton()->flush();
		int half_point = (split_container->get_size().x - def_pos[0]) / 2 - sep.x;
		int so = child_3_expanded_size + 11 - half_point; // 11 is from 60 - 50 + 1 to get the second child's size.
		CHECK_RECTS(get_rects_multi(split_container, { child_2_expanded_size + so }, sep.x), get_child_rects(split_container));
		CHECK(split_container->get_split_offsets() == Vector<int>({ so }));
		child_a->set_visible(true);
		MessageQueue::get_singleton()->flush();
		CHECK_RECTS(get_rects_multi(split_container, def_pos, sep.x), get_child_rects(split_container));
		CHECK(split_container->get_split_offsets() == Vector<int>({ 50, 60 }));

		// Hide and show the second child.
		child_b->set_visible(false);
		MessageQueue::get_singleton()->flush();
		half_point = (split_container->get_size().x - (def_pos[1] - def_pos[0] - sep.x)) / 2 - sep.x + 1;
		so = def_pos[0] - half_point;
		CHECK_RECTS(get_rects_multi(split_container, { child_2_expanded_size + so }, sep.x), get_child_rects(split_container));
		CHECK(split_container->get_split_offsets() == Vector<int>({ so }));
		child_b->set_visible(true);
		MessageQueue::get_singleton()->flush();
		// There is lost precision due to SplitContainer using ints, so this is off by one.
		CHECK_RECTS(get_rects_multi(split_container, { def_pos[0] - 1, def_pos[1] - 1 }, sep.x), get_child_rects(split_container));
		CHECK(split_container->get_split_offsets() == Vector<int>({ 49, 59 }));

		// Hide and show the last child.
		split_container->set_split_offsets({ 50, 60 });
		child_c->set_visible(false);
		MessageQueue::get_singleton()->flush();
		half_point = (def_pos[1] - sep.x) / 2 + 1;
		so = def_pos[0] - half_point;
		CHECK_RECTS(get_rects_multi(split_container, { child_2_expanded_size + so }, sep.x), get_child_rects(split_container));
		CHECK(split_container->get_split_offsets() == Vector<int>({ so }));
		child_c->set_visible(true);
		MessageQueue::get_singleton()->flush();
		CHECK_RECTS(get_rects_multi(split_container, { def_pos[0] - 1, def_pos[1] - 1 }, sep.x), get_child_rects(split_container));
		CHECK(split_container->get_split_offsets() == Vector<int>({ 49, 59 }));
	}

	SUBCASE("[SplitContainer] Adjust split offset when moving children") {
		split_container->set_split_offsets({ 50, 80 });
		split_container->move_child(child_a, 1);
		Vector<int> pos = { 30 - sep.x, 80 }; // 30 = 80 - 50.
		MessageQueue::get_singleton()->flush();
		CHECK(split_container->get_split_offsets() == pos);
		CHECK_RECTS(get_rects_multi(split_container, pos, sep.x), get_child_rects(split_container));

		// Move last child to first.
		split_container->set_split_offsets({ 50, 80 });
		split_container->move_child(child_c, 0);
		pos = { (int)split_container->get_size().x - 80 - sep.x, (int)split_container->get_size().x - 30 };
		MessageQueue::get_singleton()->flush();
		CHECK(split_container->get_split_offsets() == pos);
		CHECK_RECTS(get_rects_multi(split_container, pos, sep.x), get_child_rects(split_container));

		// Move it back.
		split_container->move_child(child_c, 2);
		pos = { 50, 80 };
		MessageQueue::get_singleton()->flush();
		CHECK(split_container->get_split_offsets() == pos);
		CHECK_RECTS(get_rects_multi(split_container, pos, sep.x), get_child_rects(split_container));
	}

	SUBCASE("[SplitContainer] Showing child with not enough space shrinks the largest child first") {
		set_size_flags(split_container, { -1, -1, -1 }); // None expanded.

		// Second child is largest.
		child_a->set_visible(false);
		Vector<int> pos = { 360 };
		split_container->set_split_offsets(pos);

		MessageQueue::get_singleton()->flush();
		CHECK(split_container->get_split_offsets() == pos);
		CHECK_RECTS(get_rects_multi(split_container, pos, sep.x), get_child_rects(split_container));

		child_a->set_size(Vector2(100, 100));
		MessageQueue::get_singleton()->flush();
		CHECK(split_container->get_split_offsets() == pos);
		CHECK_RECTS(get_rects_multi(split_container, pos, sep.x), get_child_rects(split_container));

		child_a->set_visible(true);
		pos = { 100, 360 };
		MessageQueue::get_singleton()->flush();
		CHECK(split_container->get_split_offsets() == pos);
		CHECK_RECTS(get_rects_multi(split_container, pos, sep.x), get_child_rects(split_container));

		// Last child is largest.
		child_a->set_visible(false);
		pos = { 60 };
		split_container->set_split_offsets(pos);

		MessageQueue::get_singleton()->flush();
		CHECK(split_container->get_split_offsets() == pos);
		CHECK_RECTS(get_rects_multi(split_container, pos, sep.x), get_child_rects(split_container));

		child_a->set_size(Vector2(100, 100));
		MessageQueue::get_singleton()->flush();
		CHECK(split_container->get_split_offsets() == pos);
		CHECK_RECTS(get_rects_multi(split_container, pos, sep.x), get_child_rects(split_container));

		child_a->set_visible(true);
		pos = { 100, 160 + sep.x };
		MessageQueue::get_singleton()->flush();
		CHECK(split_container->get_split_offsets() == pos);
		CHECK_RECTS(get_rects_multi(split_container, pos, sep.x), get_child_rects(split_container));

		// Both visible children are the same size.
		child_a->set_visible(false);
		pos = { (int)split_container->get_size().x / 2 - sep.x / 2 };
		split_container->set_split_offsets(pos);

		MessageQueue::get_singleton()->flush();
		CHECK(split_container->get_split_offsets() == pos);
		CHECK_RECTS(get_rects_multi(split_container, pos, sep.x), get_child_rects(split_container));
		CHECK(child_b->get_size().x == child_c->get_size().x);

		child_a->set_size(Vector2(100, 100));
		MessageQueue::get_singleton()->flush();
		CHECK(split_container->get_split_offsets() == pos);
		CHECK_RECTS(get_rects_multi(split_container, pos, sep.x), get_child_rects(split_container));

		child_a->set_visible(true);
		pos = { 100, (int)split_container->get_size().x / 2 + 50 };
		MessageQueue::get_singleton()->flush();
		CHECK(split_container->get_split_offsets() == pos);
		CHECK_RECTS(get_rects_multi(split_container, pos, sep.x), get_child_rects(split_container));
		CHECK(child_b->get_size().x == child_c->get_size().x);

		// Second child is slightly larger than the last child.
		child_a->set_visible(false);
		pos = { (int)split_container->get_size().x / 2 - sep.x / 2 + 20 };
		split_container->set_split_offsets(pos);

		MessageQueue::get_singleton()->flush();
		CHECK(split_container->get_split_offsets() == pos);
		CHECK_RECTS(get_rects_multi(split_container, pos, sep.x), get_child_rects(split_container));

		child_a->set_size(Vector2(100, 100));
		MessageQueue::get_singleton()->flush();
		CHECK(split_container->get_split_offsets() == pos);
		CHECK_RECTS(get_rects_multi(split_container, pos, sep.x), get_child_rects(split_container));

		child_a->set_visible(true);
		pos = { 100, (int)split_container->get_size().x / 2 + 50 };
		MessageQueue::get_singleton()->flush();
		CHECK(split_container->get_split_offsets() == pos);
		CHECK_RECTS(get_rects_multi(split_container, pos, sep.x), get_child_rects(split_container));
		CHECK(child_b->get_size().x == child_c->get_size().x);
	}

	memdelete(split_container);
}

} // namespace TestSplitContainer
