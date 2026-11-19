/**************************************************************************/
/*  test_graph_edit.cpp                                                   */
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

TEST_FORCE_LINK(test_graph_edit)

#ifndef ADVANCED_GUI_DISABLED

#include "core/input/input.h"
#include "scene/gui/graph_edit.h"
#include "scene/gui/graph_frame.h"
#include "scene/gui/graph_node.h"
#include "scene/main/scene_tree.h"
#include "scene/main/window.h"

namespace TestGraphEdit {

TEST_CASE("[GraphEdit][SceneTree] Frame contents respect the snapping modifier") {
	bool snapping_enabled = false;
	bool modifier_pressed = false;
	Vector2 expected_position(333, 247);

	SUBCASE("Snapping disabled") {}
	SUBCASE("Modifier enables snapping") {
		modifier_pressed = true;
		expected_position = Vector2(340, 240);
	}
	SUBCASE("Snapping enabled") {
		snapping_enabled = true;
		expected_position = Vector2(340, 240);
	}
	SUBCASE("Modifier disables snapping") {
		snapping_enabled = true;
		modifier_pressed = true;
	}

	for (bool nested : { false, true }) {
		for (float zoom : { 1.0f, 1.5f }) {
			CAPTURE(nested);
			CAPTURE(zoom);
			GraphEdit *graph = memnew(GraphEdit);
			SceneTree::get_singleton()->get_root()->add_child(graph);
			graph->set_size(Size2(1000, 700));
			graph->set_zoom(zoom);
			graph->set_snapping_distance(20);
			graph->set_snapping_enabled(snapping_enabled);

			GraphFrame *frame = memnew(GraphFrame);
			frame->set_name("Frame");
			graph->add_child(frame);
			GraphNode *node = memnew(GraphNode);
			node->set_name("Node");
			graph->add_child(node);
			node->set_position_offset(Vector2(320, 240));
			if (nested) {
				GraphFrame *inner_frame = memnew(GraphFrame);
				inner_frame->set_name("InnerFrame");
				graph->add_child(inner_frame);
				graph->attach_graph_element_to_frame(node->get_name(), inner_frame->get_name());
				graph->attach_graph_element_to_frame(inner_frame->get_name(), frame->get_name());
			} else {
				graph->attach_graph_element_to_frame(node->get_name(), frame->get_name());
			}
			SceneTree::get_singleton()->process(0);

			const Vector2 click_position = frame->get_position() + Vector2(10, 5);
			Ref<InputEventMouseButton> button;
			button.instantiate();
			button->set_button_index(MouseButton::LEFT);
			button->set_pressed(true);
			button->set_position(click_position);
			static_cast<Control *>(graph)->gui_input(button);
			CHECK(frame->is_selected());
			CHECK_FALSE(node->is_selected());

			Ref<InputEventKey> modifier;
			modifier.instantiate();
			modifier->set_keycode(Key::CMD_OR_CTRL);
			modifier->set_pressed(modifier_pressed);
			Input::get_singleton()->parse_input_event(modifier);
			Input::get_singleton()->flush_buffered_events();

			Ref<InputEventMouseMotion> motion;
			motion.instantiate();
			motion->set_button_mask(MouseButtonMask::LEFT);
			motion->set_position(click_position + Vector2(13, 7) * zoom);
			motion->set_relative(Vector2(13, 7) * zoom);
			static_cast<Control *>(graph)->gui_input(motion);
			CHECK(node->get_position_offset().is_equal_approx(expected_position));

			modifier = modifier->duplicate();
			modifier->set_pressed(false);
			Input::get_singleton()->parse_input_event(modifier);
			Input::get_singleton()->flush_buffered_events();
			button->set_pressed(false);
			static_cast<Control *>(graph)->gui_input(button);
			memdelete(graph);
		}
	}
}

} // namespace TestGraphEdit

#endif // ADVANCED_GUI_DISABLED
