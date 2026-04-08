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

#if defined(MODULE_TEXT_SERVER_FB_ENABLED) || defined(MODULE_TEXT_SERVER_ADV_ENABLED)

#include "scene/gui/label.h"
#include "scene/main/scene_tree.h"
#include "scene/main/window.h"

namespace TestLabel {

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

	test_label->set_custom_maximum_size(Size2(max_width, 0));
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
