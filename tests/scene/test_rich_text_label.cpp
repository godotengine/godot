/**************************************************************************/
/*  test_rich_text_label.cpp                                              */
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

TEST_FORCE_LINK(test_rich_text_label)

#ifndef ADVANCED_GUI_DISABLED

#include "scene/gui/rich_text_label.h"
#include "scene/main/scene_tree.h"
#include "scene/main/window.h"

namespace TestRichTextLabel {

TEST_CASE("[SceneTree][RichTextLabel] Custom minimum size with fit content") {
	// This is an anti-regression test case introduced in GH-116640. When the old minimum size behavior is removed, this test case should be removed too.
	RichTextLabel *test_label = memnew(RichTextLabel);
	Window *root = SceneTree::get_singleton()->get_root();
	root->add_child(test_label);

	real_t min_width = 50;

	test_label->set_fit_content(true);
	test_label->set_custom_minimum_size(Size2(min_width, 0));
	test_label->set_autowrap_mode(TextServer::AUTOWRAP_OFF);
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

	memdelete(test_label);
}

TEST_CASE("[SceneTree][RichTextLabel] Sizing with fit content") {
	RichTextLabel *test_label = memnew(RichTextLabel);
	Window *root = SceneTree::get_singleton()->get_root();
	root->add_child(test_label);

	real_t max_width = 50;
	real_t min_width = 25;

	test_label->set_fit_content(true);
	test_label->set_custom_maximum_size(Size2(max_width, 0));
	test_label->set_custom_minimum_size(Size2(min_width, 0));
	test_label->set_autowrap_mode(TextServer::AUTOWRAP_OFF);
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

	memdelete(test_label);
}

TEST_CASE("[SceneTree][RichTextLabel] VisibleRatio") {
	RichTextLabel *test_label = memnew(RichTextLabel);
	Window *root = SceneTree::get_singleton()->get_root();
	root->add_child(test_label);

	test_label->set_text("Short text");
	test_label->set_visible_characters(-1);

	CHECK_MESSAGE(
			test_label->get_visible_ratio() == doctest::Approx(1.0),
			"Label visible_ratio is set to 1.0 when visible_characters is set to -1.");

	test_label->set_text("This is a longer text string");

	CHECK_MESSAGE(
			test_label->get_visible_ratio() == doctest::Approx(1.0),
			"Label visible_ratio remains the same when text changes (from -1).");
	CHECK_MESSAGE(
			test_label->get_visible_characters() == -1,
			"Label visible_characters remains at -1 when text changes (from -1).");

	test_label->set_text("Short text");
	test_label->set_visible_characters(6);

	CHECK_MESSAGE(
			test_label->get_visible_ratio() == doctest::Approx(0.6),
			"Label visible_ratio automatically updates when visible_characters is set.");

	test_label->set_text("This is a longer text string");

	// NOTE: This outcome is *different* to Label, which recomputes visible_characters, not visible_ratio.
	// See https://github.com/godotengine/godot/issues/119840
	CHECK_MESSAGE(
			test_label->get_visible_ratio() == doctest::Approx(0.2142857),
			"Label visible_ratio is recomputed when text changes.");
	CHECK_MESSAGE(
			test_label->get_visible_characters() == 6,
			"Label visible_characters remains the same when text changes.");

	test_label->set_text("Paragraph one\nParagraph two");
	test_label->set_visible_characters(6);
	CHECK(test_label->get_visible_ratio() == doctest::Approx(0.2222222));
	test_label->remove_paragraph(1);
	CHECK_MESSAGE(
			test_label->get_visible_ratio() == doctest::Approx(0.4285714),
			"Label visible_ratio is recomputed when paragraph is removed.");

	test_label->set_text("Short text");
	test_label->set_visible_characters(10);

	CHECK_MESSAGE(
			test_label->get_visible_ratio() == doctest::Approx(1.0),
			"Label visible_ratio automatically updates when visible_characters is set to full length.");

	test_label->set_text("This is a longer text string");

	// NOTE: This outcome is again *different* to Label.
	CHECK_MESSAGE(
			test_label->get_visible_ratio() == doctest::Approx(0.3571428),
			"Label visible_ratio is recomputed when text changes (from full length).");
	CHECK_MESSAGE(
			test_label->get_visible_characters() == 10,
			"Label visible_characters remains the same when text changes (from full length).");

	test_label->set_text("");
	test_label->set_visible_characters(3);

	CHECK_MESSAGE(
			test_label->get_visible_ratio() == doctest::Approx(1.0),
			"Label visible_ratio is set to 1.0 when visible_characters is set while text is empty.");

	test_label->set_text("Short text");
	test_label->set_visible_characters(13);

	CHECK_MESSAGE(
			test_label->get_visible_ratio() == doctest::Approx(1.0),
			"Label visible_ratio is set to 1.0 when visible_characters is set too large.");

	test_label->set_text("Short text");
	test_label->set_visible_characters(-2);

	CHECK_MESSAGE(
			test_label->get_visible_ratio() == doctest::Approx(0.0),
			"Label visible_ratio is set to 0.0 when visible_characters is set < -1.");

	test_label->set_visible_characters(6);
	CHECK(test_label->get_visible_ratio() == doctest::Approx(0.6));
	test_label->set_use_bbcode(true);
	test_label->set_text("[b]Formatted[/b] text");
	CHECK_MESSAGE(
			test_label->get_visible_ratio() == doctest::Approx(0.4285714),
			"Label visible_ratio is recomputed when text changes in bbcode mode.");

	memdelete(test_label);
}

} // namespace TestRichTextLabel

#endif // ADVANCED_GUI_DISABLED
