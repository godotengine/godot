/**************************************************************************/
/*  test_label.cpp                                                        */
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

TEST_FORCE_LINK(test_label)

#include "modules/modules_enabled.gen.h" // IWYU pragma: keep. Needed for MODULE_TEXT_SERVER_FB_ENABLED and MODULE_TEXT_SERVER_ADV_ENABLED definitions.

#if defined(MODULE_TEXT_SERVER_FB_ENABLED) || defined(MODULE_TEXT_SERVER_ADV_ENABLED)

#include "scene/gui/box_container.h"
#include "scene/gui/label.h"
#include "scene/gui/margin_container.h"
#include "scene/gui/panel_container.h"
#include "scene/main/scene_tree.h"
#include "scene/main/window.h"

namespace TestLabel {

TEST_CASE("[SceneTree][Label] Autowrapped labels in a maximum-width container") {
	Control *control = memnew(Control);
	PanelContainer *panel = memnew(PanelContainer);
	panel->set_custom_maximum_size(Size2(200, -1));
	panel->set_size(Size2(40, 40));
	control->add_child(panel);
	MarginContainer *margin = memnew(MarginContainer);
	for (const char *side : { "margin_left", "margin_top", "margin_right", "margin_bottom" }) {
		margin->add_theme_constant_override(side, 10);
	}
	panel->add_child(margin);
	VBoxContainer *vbox = memnew(VBoxContainer);
	margin->add_child(vbox);
	Label *heading = memnew(Label);
	heading->set_text("ABC");
	vbox->add_child(heading);
	Label *first = memnew(Label);
	first->set_autowrap_mode(TextServer::AUTOWRAP_WORD_SMART);
	first->set_text("abc");
	vbox->add_child(first);
	Label *second = memnew(Label);
	second->set_autowrap_mode(TextServer::AUTOWRAP_WORD_SMART);
	second->set_text("xyz");
	vbox->add_child(second);
	control->set_anchors_preset(Control::PRESET_CENTER_BOTTOM);
	SceneTree::get_singleton()->get_root()->add_child(control);
	panel->set_position(Point2(-1, -1));
	control->set_position(Point2(101, 101));
	SceneTree::get_singleton()->process(0);
	SceneTree::get_singleton()->process(0);

	INFO("Panel height: ", panel->get_size().height, ", minimum: ", panel->get_combined_minimum_size().height,
			", heading height: ", heading->get_size().height, ", first label height: ", first->get_size().height);
	CHECK(first->get_line_count() == 1);
	CHECK(second->get_line_count() == 1);
	CHECK(first->get_size().height == doctest::Approx(heading->get_size().height));
	CHECK(second->get_size().height == doctest::Approx(heading->get_size().height));
	CHECK(panel->get_size().height == doctest::Approx(panel->get_combined_minimum_size().height));

	memdelete(control);
}

TEST_CASE("[SceneTree][Label] Autowrapping respects a width below the maximum") {
	Label *label = memnew(Label);
	label->set_custom_maximum_size(Size2(200, -1));
	label->set_autowrap_mode(TextServer::AUTOWRAP_WORD_SMART);
	label->set_text("A long line of text that must wrap at the assigned width.");
	SceneTree::get_singleton()->get_root()->add_child(label);
	SceneTree::get_singleton()->process(0);
	label->set_size(Size2(50, label->get_size().height));
	const int narrow_line_count = label->get_line_count();
	CHECK(narrow_line_count > 1);
	label->set_size(Size2(200, label->get_size().height));
	SceneTree::get_singleton()->process(0);
	CHECK(label->get_line_count() < narrow_line_count);
	memdelete(label);
}

TEST_CASE("[SceneTree][Label] Custom minimum size") {
	// This is an anti-regression test case introduced in GH-116640. When the old minimum size behavior is removed, this test case should be removed too.
	Label *test_label = memnew(Label);
	Window *root = SceneTree::get_singleton()->get_root();
	root->add_child(test_label);

	real_t min_width = 50;

	test_label->set_custom_minimum_size(Size2(min_width, 0));
	test_label->set_text("This is a long text that should be wrapped and exceeds minimum size.");
	SceneTree::get_singleton()->process(0);

	CHECK_MESSAGE(
			test_label->get_size().width > min_width,
			"Label width will increase to fit the text with AUTOWRAP_OFF.");

	test_label->set_autowrap_mode(TextServer::AUTOWRAP_ARBITRARY);
	SceneTree::get_singleton()->process(0);

	CHECK_MESSAGE(
			Math::is_equal_approx(test_label->get_size().width, min_width),
			"Label width will be equal to custom minimum width with AUTOWRAP_ARBITRARY.");

	test_label->set_autowrap_mode(TextServer::AUTOWRAP_WORD);
	SceneTree::get_singleton()->process(0);

	CHECK_MESSAGE(
			Math::is_equal_approx(test_label->get_size().width, min_width),
			"Label width will be equal to custom minimum width with AUTOWRAP_WORD.");

	test_label->set_autowrap_mode(TextServer::AUTOWRAP_WORD_SMART);
	SceneTree::get_singleton()->process(0);

	CHECK_MESSAGE(
			Math::is_equal_approx(test_label->get_size().width, min_width),
			"Label width will be equal to custom minimum width with AUTOWRAP_WORD_SMART.");

	test_label->set_autowrap_mode(TextServer::AUTOWRAP_OFF);
	test_label->set_clip_text(true);
	SceneTree::get_singleton()->process(0);

	CHECK_MESSAGE(
			Math::is_equal_approx(test_label->get_size().width, min_width),
			"Label width will be equal to custom minimum width with clip_text.");

	test_label->set_clip_text(false);
	test_label->set_text_overrun_behavior(TextServer::OVERRUN_TRIM_CHAR);
	SceneTree::get_singleton()->process(0);

	CHECK_MESSAGE(
			Math::is_equal_approx(test_label->get_size().width, min_width),
			"Label width will be equal to custom minimum width with OVERRUN_TRIM_CHAR.");

	test_label->set_text_overrun_behavior(TextServer::OVERRUN_TRIM_WORD);
	SceneTree::get_singleton()->process(0);

	CHECK_MESSAGE(
			Math::is_equal_approx(test_label->get_size().width, min_width),
			"Label width will be equal to custom minimum width with OVERRUN_TRIM_WORD.");

	test_label->set_text_overrun_behavior(TextServer::OVERRUN_TRIM_ELLIPSIS);
	SceneTree::get_singleton()->process(0);

	CHECK_MESSAGE(
			Math::is_equal_approx(test_label->get_size().width, min_width),
			"Label width will be equal to custom minimum width with OVERRUN_TRIM_ELLIPSIS.");

	test_label->set_text_overrun_behavior(TextServer::OVERRUN_TRIM_WORD_ELLIPSIS);
	SceneTree::get_singleton()->process(0);

	CHECK_MESSAGE(
			Math::is_equal_approx(test_label->get_size().width, min_width),
			"Label width will be equal to custom minimum width with OVERRUN_TRIM_WORD_ELLIPSIS.");

	test_label->set_text_overrun_behavior(TextServer::OVERRUN_TRIM_ELLIPSIS_FORCE);
	SceneTree::get_singleton()->process(0);

	CHECK_MESSAGE(
			Math::is_equal_approx(test_label->get_size().width, min_width),
			"Label width will be equal to custom minimum width with OVERRUN_TRIM_ELLIPSIS_FORCE.");

	test_label->set_text_overrun_behavior(TextServer::OVERRUN_TRIM_WORD_ELLIPSIS_FORCE);
	SceneTree::get_singleton()->process(0);

	CHECK_MESSAGE(
			Math::is_equal_approx(test_label->get_size().width, min_width),
			"Label width will be equal to custom minimum width with OVERRUN_TRIM_WORD_ELLIPSIS_FORCE.");

	memdelete(test_label);
}

TEST_CASE("[SceneTree][Label] Sizing") {
	Label *test_label = memnew(Label);
	Window *root = SceneTree::get_singleton()->get_root();
	root->add_child(test_label);

	real_t max_width = 50;
	real_t min_width = 25;

	test_label->set_custom_maximum_size(Size2(max_width, -1));
	test_label->set_custom_minimum_size(Size2(min_width, 0));
	test_label->set_text("This is a long text that should be wrapped and exceeds minimum size.");
	SceneTree::get_singleton()->process(0);

	CHECK_MESSAGE(
			test_label->get_size().width <= max_width,
			"Label width will increase up to the custom maximum width with AUTOWRAP_OFF.");
	CHECK_MESSAGE(
			test_label->get_size().width > min_width,
			"Label width will increase beyond the custom minimum width with AUTOWRAP_OFF.");

	test_label->set_autowrap_mode(TextServer::AUTOWRAP_ARBITRARY);
	SceneTree::get_singleton()->process(0);

	CHECK_MESSAGE(
			test_label->get_size().width <= max_width,
			"Label width will increase up to the custom maximum width with AUTOWRAP_ARBITRARY.");
	CHECK_MESSAGE(
			test_label->get_size().width > min_width,
			"Label width will increase beyond the custom minimum width with AUTOWRAP_ARBITRARY.");

	test_label->set_autowrap_mode(TextServer::AUTOWRAP_WORD);
	SceneTree::get_singleton()->process(0);

	CHECK_MESSAGE(
			test_label->get_size().width <= max_width,
			"Label width will increase up to the custom maximum width with AUTOWRAP_WORD.");
	CHECK_MESSAGE(
			test_label->get_size().width > min_width,
			"Label width will increase beyond the custom minimum width with AUTOWRAP_WORD.");

	test_label->set_autowrap_mode(TextServer::AUTOWRAP_WORD_SMART);
	SceneTree::get_singleton()->process(0);

	CHECK_MESSAGE(
			test_label->get_size().width <= max_width,
			"Label width will increase up to the custom maximum width with AUTOWRAP_WORD_SMART.");
	CHECK_MESSAGE(
			test_label->get_size().width > min_width,
			"Label width will increase beyond the custom minimum width with AUTOWRAP_WORD_SMART.");

	test_label->set_autowrap_mode(TextServer::AUTOWRAP_OFF);
	test_label->set_clip_text(true);
	SceneTree::get_singleton()->process(0);

	CHECK_MESSAGE(
			test_label->get_size().width <= max_width,
			"Label width will increase up to the custom maximum width with clip_text.");
	CHECK_MESSAGE(
			test_label->get_size().width > min_width,
			"Label width will increase beyond the custom minimum width with clip_text.");

	test_label->set_clip_text(false);
	test_label->set_text_overrun_behavior(TextServer::OVERRUN_TRIM_CHAR);
	SceneTree::get_singleton()->process(0);

	CHECK_MESSAGE(
			test_label->get_size().width <= max_width,
			"Label width will increase up to the custom maximum width with OVERRUN_TRIM_CHAR.");
	CHECK_MESSAGE(
			test_label->get_size().width > min_width,
			"Label width will increase beyond the custom minimum width with OVERRUN_TRIM_CHAR.");

	test_label->set_text_overrun_behavior(TextServer::OVERRUN_TRIM_WORD);
	SceneTree::get_singleton()->process(0);

	CHECK_MESSAGE(
			test_label->get_size().width <= max_width,
			"Label width will increase up to the custom maximum width with OVERRUN_TRIM_WORD.");
	CHECK_MESSAGE(
			test_label->get_size().width > min_width,
			"Label width will increase beyond the custom minimum width with OVERRUN_TRIM_WORD.");

	test_label->set_text_overrun_behavior(TextServer::OVERRUN_TRIM_ELLIPSIS);
	SceneTree::get_singleton()->process(0);

	CHECK_MESSAGE(
			test_label->get_size().width <= max_width,
			"Label width will increase up to the custom maximum width with OVERRUN_TRIM_ELLIPSIS.");
	CHECK_MESSAGE(
			test_label->get_size().width > min_width,
			"Label width will increase beyond the custom minimum width with OVERRUN_TRIM_ELLIPSIS.");

	test_label->set_text_overrun_behavior(TextServer::OVERRUN_TRIM_WORD_ELLIPSIS);
	SceneTree::get_singleton()->process(0);

	CHECK_MESSAGE(
			test_label->get_size().width <= max_width,
			"Label width will increase up to the custom maximum width with OVERRUN_TRIM_WORD_ELLIPSIS.");
	CHECK_MESSAGE(
			test_label->get_size().width > min_width,
			"Label width will increase beyond the custom minimum width with OVERRUN_TRIM_WORD_ELLIPSIS.");

	test_label->set_text_overrun_behavior(TextServer::OVERRUN_TRIM_ELLIPSIS_FORCE);
	SceneTree::get_singleton()->process(0);

	CHECK_MESSAGE(
			test_label->get_size().width <= max_width,
			"Label width will increase up to the custom maximum width with OVERRUN_TRIM_ELLIPSIS_FORCE.");
	CHECK_MESSAGE(
			test_label->get_size().width > min_width,
			"Label width will increase beyond the custom minimum width with OVERRUN_TRIM_ELLIPSIS_FORCE.");

	test_label->set_text_overrun_behavior(TextServer::OVERRUN_TRIM_WORD_ELLIPSIS_FORCE);
	SceneTree::get_singleton()->process(0);

	CHECK_MESSAGE(
			test_label->get_size().width <= max_width,
			"Label width will increase up to the custom maximum width with OVERRUN_TRIM_WORD_ELLIPSIS_FORCE.");
	CHECK_MESSAGE(
			test_label->get_size().width > min_width,
			"Label width will increase beyond the custom minimum width with OVERRUN_TRIM_WORD_ELLIPSIS_FORCE.");

	memdelete(test_label);
}

} // namespace TestLabel

#endif // MODULE_TEXT_SERVER_FB_ENABLED || MODULE_TEXT_SERVER_ADV_ENABLED
