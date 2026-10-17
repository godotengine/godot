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

#include "core/io/image.h"
#include "core/object/callable_mp.h"
#include "scene/gui/box_container.h"
#include "scene/gui/margin_container.h"
#include "scene/gui/panel_container.h"
#include "scene/gui/rich_text_label.h"
#include "scene/gui/texture_rect.h"
#include "scene/main/scene_tree.h"
#include "scene/main/window.h"
#include "scene/resources/image_texture.h"

namespace TestRichTextLabel {

class LayoutMonitor : public Object {
public:
	RichTextLabel *label = nullptr;
	int resize_count = 0;

	void on_resized() {
		resize_count++;
		if (resize_count == 100) {
			label->set_fit_content(false);
		}
	}
};

TEST_CASE("[SceneTree][RichTextLabel] Fit content with a fit-width texture converges") {
	for (bool use_margin_container : { false, true }) {
		for (int sibling_width : { 30, 32 }) {
			for (TextureRect::ExpandMode expand_mode : { TextureRect::EXPAND_FIT_WIDTH, TextureRect::EXPAND_FIT_WIDTH_PROPORTIONAL }) {
				CAPTURE(use_margin_container);
				CAPTURE(sibling_width);
				CAPTURE(expand_mode);

				Container *container = use_margin_container ? static_cast<Container *>(memnew(MarginContainer)) : memnew(PanelContainer);
				container->set_custom_minimum_size(Size2(300, 0));
				container->set_size(Size2(300, 253));
				VBoxContainer *vbox = memnew(VBoxContainer);
				container->add_child(vbox);
				HBoxContainer *hbox = memnew(HBoxContainer);
				vbox->add_child(hbox);

				TextureRect *texture_rect = memnew(TextureRect);
				Ref<Image> image = memnew(Image(8, 8, false, Image::FORMAT_RGB8));
				texture_rect->set_texture(ImageTexture::create_from_image(image));
				texture_rect->set_expand_mode(expand_mode);
				hbox->add_child(texture_rect);

				RichTextLabel *label = memnew(RichTextLabel);
				label->set_h_size_flags(Control::SIZE_EXPAND_FILL);
				label->set_text("QWERTYUIOP");
				label->set_fit_content(true);
				hbox->add_child(label);

				Control *sibling = memnew(Control);
				sibling->set_custom_minimum_size(Size2(sibling_width, 0));
				hbox->add_child(sibling);

				LayoutMonitor *monitor = memnew(LayoutMonitor);
				monitor->label = label;
				label->connect(SceneStringName(resized), callable_mp(monitor, &LayoutMonitor::on_resized));
				SceneTree::get_singleton()->get_root()->add_child(container);
				SceneTree::get_singleton()->process(0);

				CHECK(monitor->resize_count < 100);
				CHECK(label->is_fit_content_enabled());
				CHECK(label->get_size().height >= label->get_content_height());
				CHECK(texture_rect->get_size().width == texture_rect->get_size().height);

				label->set_text("qwertyuiop");
				SceneTree::get_singleton()->process(0);
				CHECK(monitor->resize_count < 100);
				CHECK(label->is_fit_content_enabled());

				int resize_count = monitor->resize_count;
				for (int frame = 0; frame < 3; frame++) {
					SceneTree::get_singleton()->process(0);
				}
				CHECK(monitor->resize_count == resize_count);

				memdelete(container);
				memdelete(monitor);
			}
		}
	}
}

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

TEST_CASE("[SceneTree][RichTextLabel] Font size wrapping paragraph does not add empty first line") {
	RichTextLabel *test_label = memnew(RichTextLabel);
	Window *root = SceneTree::get_singleton()->get_root();
	root->add_child(test_label);

	test_label->set_use_bbcode(true);

	test_label->set_text("[p][font_size=22]aaa[/font_size][/p]");
	SceneTree::get_singleton()->process(0);
	CHECK(test_label->get_paragraph_count() == 1);
	CHECK(test_label->get_character_line(0) == 0);

	test_label->set_text("[font_size=22][p]aaa[/p][/font_size]");
	SceneTree::get_singleton()->process(0);
	CHECK(test_label->get_paragraph_count() == 1);
	CHECK(test_label->get_character_line(0) == 0);

	memdelete(test_label);
}

} // namespace TestRichTextLabel

#endif // ADVANCED_GUI_DISABLED
