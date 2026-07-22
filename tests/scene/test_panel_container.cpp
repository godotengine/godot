/**************************************************************************/
/*  test_panel_container.cpp                                              */
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

TEST_FORCE_LINK(test_panel_container)

#include "scene/gui/control.h"
#include "scene/gui/panel_container.h"
#include "scene/main/scene_tree.h"
#include "scene/main/window.h"
#include "scene/resources/style_box.h"

namespace TestPanelContainer {

TEST_CASE("[SceneTree][PanelContainer] Default properties") {
	PanelContainer *panel_container = memnew(PanelContainer);
	Window *root = SceneTree::get_singleton()->get_root();
	root->add_child(panel_container);
	SceneTree::get_singleton()->process(0);

	CHECK_MESSAGE(
			panel_container->get_mouse_filter() == Control::MOUSE_FILTER_STOP,
			"PanelContainer mouse filter is set to MOUSE_FILTER_STOP by default.");

	memdelete(panel_container);
}

TEST_CASE("[SceneTree][PanelContainer] StyleBox affects layout and minimum size") {
	PanelContainer *panel_container = memnew(PanelContainer);
	Window *root = SceneTree::get_singleton()->get_root();
	root->add_child(panel_container);

	Ref<StyleBoxEmpty> panel_style = memnew(StyleBoxEmpty);
	panel_style->set_content_margin_individual(5, 6, 7, 8);
	panel_container->add_theme_style_override("panel", panel_style);

	Control *child_control = memnew(Control);
	child_control->set_custom_minimum_size(Size2(20, 10));
	panel_container->add_child(child_control);

	panel_container->set_size(Size2(100, 80));
	SceneTree::get_singleton()->process(0);

	CHECK_MESSAGE(
			child_control->get_position().is_equal_approx(Point2(5, 6)),
			"Child control is offset by the panel StyleBox margins.");

	CHECK_MESSAGE(
			child_control->get_size().is_equal_approx(Size2(88, 66)),
			"Child control size is reduced by the panel StyleBox minimum size.");

	CHECK_MESSAGE(
			panel_container->get_minimum_size().is_equal_approx(Size2(32, 24)),
			"Minimum size equals child minimum size plus panel StyleBox minimum size.");

	memdelete(child_control);
	memdelete(panel_container);
}

TEST_CASE("[SceneTree][PanelContainer] Multiple children use maximum minimum size") {
	PanelContainer *panel_container = memnew(PanelContainer);
	Window *root = SceneTree::get_singleton()->get_root();
	root->add_child(panel_container);

	Ref<StyleBoxEmpty> panel_style = memnew(StyleBoxEmpty);
	panel_style->set_content_margin_individual(2, 3, 4, 5);
	panel_container->add_theme_style_override("panel", panel_style);

	Control *child_control_1 = memnew(Control);
	Control *child_control_2 = memnew(Control);
	child_control_1->set_custom_minimum_size(Size2(40, 10));
	child_control_2->set_custom_minimum_size(Size2(20, 50));
	panel_container->add_child(child_control_1);
	panel_container->add_child(child_control_2);
	SceneTree::get_singleton()->process(0);

	CHECK_MESSAGE(
			panel_container->get_minimum_size().is_equal_approx(Size2(46, 58)),
			"Minimum size uses maximum child minimum width/height plus panel StyleBox minimum size.");

	memdelete(child_control_2);
	memdelete(child_control_1);
	memdelete(panel_container);
}

} // namespace TestPanelContainer
