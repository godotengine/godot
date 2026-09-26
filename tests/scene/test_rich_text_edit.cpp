/**************************************************************************/
/*  test_rich_text_edit.cpp                                               */
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

TEST_FORCE_LINK(test_rich_text_edit)

#ifndef ADVANCED_GUI_DISABLED

#include "scene/gui/code_edit.h"
#include "scene/gui/rich_text_edit.h"
#include "scene/main/scene_tree.h"
#include "scene/main/window.h"

namespace TestRichTextEdit {

TEST_CASE("[SceneTree][RichTextEdit] Display overflow preserves editing and the document") {
	RichTextEdit *edit = memnew(RichTextEdit);
	SceneTree::get_singleton()->get_root()->add_child(edit);
	edit->set_size(Size2(160, 60));
	edit->set_use_bbcode(true);
	const String bbcode = "[b]A long unwrapped first line of text[/b]\nSecond\nThird\nFourth\nFifth";
	edit->set_bbcode_text(bbcode);
	MessageQueue::get_singleton()->flush();
	const Size2 original_size = edit->get_size();
	const int original_width = edit->get_line_width(0);
	CHECK(edit->get_h_scroll_bar()->is_visible());
	CHECK(edit->get_v_scroll_bar()->is_visible());

	edit->set_caret_line(0);
	edit->set_caret_column(1);
	edit->begin_complex_operation();
	edit->insert_text_at_caret("X");
	edit->end_complex_operation();
	MessageQueue::get_singleton()->flush();
	const PackedByteArray edited_document = edit->get_document_protobuf();
	CHECK(edit->has_undo());

	// The option is inert while editable, regardless of assignment order.
	edit->set_display_overflow_enabled(true);
	MessageQueue::get_singleton()->flush();
	CHECK(edit->get_h_scroll_bar()->is_visible());
	CHECK(edit->get_v_scroll_bar()->is_visible());
	edit->set_h_scroll(30);
	edit->set_v_scroll(1);
	CHECK(edit->get_h_scroll() > 0);
	CHECK(edit->get_v_scroll() > 0);

	edit->set_editable(false);
	MessageQueue::get_singleton()->flush();
	CHECK_FALSE(edit->get_h_scroll_bar()->is_visible());
	CHECK_FALSE(edit->get_v_scroll_bar()->is_visible());
	CHECK(edit->get_h_scroll() == 0);
	CHECK(edit->get_v_scroll() == 0);
	CHECK(edit->get_first_visible_line() == 0);
	CHECK(edit->get_size() == original_size);
	CHECK(edit->get_document_protobuf() == edited_document);

	// Turning the option off also restores the viewport while still read-only.
	edit->set_display_overflow_enabled(false);
	MessageQueue::get_singleton()->flush();
	CHECK(edit->get_h_scroll_bar()->is_visible());
	CHECK(edit->get_v_scroll_bar()->is_visible());
	edit->set_display_overflow_enabled(true);
	for (int i = 0; i < 10; i++) {
		edit->set_editable(true);
		MessageQueue::get_singleton()->flush();
		CHECK(edit->get_h_scroll_bar()->is_visible());
		CHECK(edit->get_v_scroll_bar()->is_visible());
		edit->set_editable(false);
		MessageQueue::get_singleton()->flush();
		CHECK_FALSE(edit->get_h_scroll_bar()->is_visible());
		CHECK_FALSE(edit->get_v_scroll_bar()->is_visible());
	}
	CHECK(edit->get_document_protobuf() == edited_document);
	edit->set_editable(true);
	edit->undo();
	MessageQueue::get_singleton()->flush();
	CHECK(edit->get_bbcode_text() == bbcode);
	CHECK(edit->get_line_width(0) == original_width);
	memdelete(edit);
}

TEST_CASE("[SceneTree][RichTextEdit] Display overflow does not change default text viewports") {
	TextEdit *edit = nullptr;
	SUBCASE("TextEdit") {
		edit = memnew(TextEdit);
	}
	SUBCASE("CodeEdit") {
		edit = memnew(CodeEdit);
	}
	SUBCASE("RichTextEdit without opt-in") {
		edit = memnew(RichTextEdit);
	}
	SceneTree::get_singleton()->get_root()->add_child(edit);
	edit->set_size(Size2(100, 50));
	edit->set_text("A very long first line that must scroll\n2\n3\n4\n5");
	for (bool editable : { true, false, true }) {
		edit->set_editable(editable);
		MessageQueue::get_singleton()->flush();
		CHECK(edit->get_h_scroll_bar()->is_visible());
		CHECK(edit->get_v_scroll_bar()->is_visible());
		edit->set_h_scroll(20);
		CHECK(edit->get_h_scroll() == 20);
	}
	memdelete(edit);
}

TEST_CASE("[SceneTree][RichTextEdit] Display overflow wraps at the authored width") {
	RichTextEdit *edit = memnew(RichTextEdit);
	RichTextEdit *reference = memnew(RichTextEdit);
	Window *root = SceneTree::get_singleton()->get_root();
	for (RichTextEdit *control : { edit, reference }) {
		root->add_child(control);
		control->set_editable(false);
		control->set_line_wrapping_mode(TextEdit::LINE_WRAPPING_BOUNDARY);
		control->set_autowrap_mode(TextServer::AUTOWRAP_WORD_SMART);
		control->set_text("one two three four five six seven eight nine ten eleven twelve");
	}
	edit->set_size(Size2(160, 40));
	reference->set_size(Size2(160, 600));
	edit->set_display_overflow_enabled(true);
	MessageQueue::get_singleton()->flush();
	CHECK(edit->get_size() == Size2(160, 40));
	CHECK(edit->get_line_wrap_count(0) > 0);
	CHECK(edit->get_line_wrapped_text(0) == reference->get_line_wrapped_text(0));
	CHECK_FALSE(edit->get_v_scroll_bar()->is_visible());
	for (int visible : { 0, 4, 12, -1 }) {
		edit->set_visible_characters(visible);
		reference->set_visible_characters(visible);
		MessageQueue::get_singleton()->flush();
		CHECK(edit->get_line_wrapped_text(0) == reference->get_line_wrapped_text(0));
		CHECK(edit->get_size() == Size2(160, 40));
	}
	memdelete(reference);
	memdelete(edit);
}

TEST_CASE("[SceneTree][RichTextEdit] Content height measures without resizing") {
	RichTextEdit *edit = memnew(RichTextEdit);
	RichTextEdit *reference = memnew(RichTextEdit);
	for (RichTextEdit *control : { edit, reference }) {
		SceneTree::get_singleton()->get_root()->add_child(control);
		control->set_use_bbcode(true);
		control->set_editable(false);
		control->set_display_overflow_enabled(true);
		control->set_line_wrapping_mode(TextEdit::LINE_WRAPPING_BOUNDARY);
		control->set_autowrap_mode(TextServer::AUTOWRAP_WORD_SMART);
	}
	reference->set_fit_content_height_enabled(true);
	for (const String &bbcode : { String(), String("one two three four five six seven eight nine ten"), String("[font_size=52]Big[/font_size]\nSmall\n"), String("[quote]A quoted paragraph\nwith another line[/quote]") }) {
		edit->set_bbcode_text(bbcode);
		reference->set_bbcode_text(bbcode);
		for (int width : { 160, 280 }) {
			edit->set_size(Size2(width, 20));
			reference->set_size(Size2(width, 20));
			for (int visible : { 0, 3, -1 }) {
				edit->set_visible_characters(visible);
				reference->set_visible_characters(visible);
				MessageQueue::get_singleton()->flush();
				const Size2 size = edit->get_size();
				const PackedByteArray document = edit->get_document_protobuf();
				const int expected = reference->get_minimum_size().y - reference->get_theme_stylebox("read_only")->get_minimum_size().y;
				CHECK(edit->get_content_height() == expected);
				CHECK(edit->get_size() == size);
				CHECK(edit->get_size() == Size2(width, 20));
				CHECK(edit->get_document_protobuf() == document);
				CHECK(edit->get_visible_characters() == visible);
				CHECK_FALSE(edit->is_fit_content_height_enabled());
			}
		}
	}
	memdelete(reference);
	memdelete(edit);
}

TEST_CASE("[SceneTree][RichTextEdit] Before shaping measures the visible prefix") {
	RichTextEdit *edit = memnew(RichTextEdit);
	RichTextEdit *reference = memnew(RichTextEdit);
	Window *root = SceneTree::get_singleton()->get_root();
	root->add_child(edit);
	root->add_child(reference);
	edit->set_use_bbcode(true);
	reference->set_use_bbcode(true);

	String full_text = "ABCD";
	String prefix_text = "AB";
	int visible_count = 2;

	SUBCASE("Plain text") {
	}
	SUBCASE("Bold text") {
		full_text = "[b]ABCD[/b]";
		prefix_text = "[b]AB[/b]";
	}
	SUBCASE("Multiple style runs") {
		full_text = "[font_size=32]AB[/font_size][b]CD[/b]";
		prefix_text = "[font_size=32]AB[/font_size][b]C[/b]";
		visible_count = 3;
	}
	SUBCASE("RTL text") {
		full_text = U"אבגד";
		prefix_text = U"אב";
		edit->set_text_direction(Control::TEXT_DIRECTION_RTL);
		reference->set_text_direction(Control::TEXT_DIRECTION_RTL);
	}
	SUBCASE("Structured URI") {
		full_text = "https://example.com/path";
		prefix_text = "https://example";
		visible_count = prefix_text.length();
		edit->set_structured_text_bidi_override(TextServer::STRUCTURED_TEXT_URI);
		reference->set_structured_text_bidi_override(TextServer::STRUCTURED_TEXT_URI);
	}

	edit->set_bbcode_text(full_text);
	reference->set_bbcode_text(prefix_text);
	MessageQueue::get_singleton()->flush();
	const int full_width = edit->get_line_width(0);
	const int prefix_width = reference->get_line_width(0);
	CHECK(prefix_width > 0);
	CHECK(full_width > prefix_width);

	edit->set_visible_characters(visible_count);
	CHECK(edit->get_line_width(0) > 0);
	CHECK(edit->get_line_width(0) == prefix_width);
	CHECK(edit->get_bbcode_text() == full_text);

	edit->set_visible_characters(0);
	CHECK(edit->get_line_width(0) == 0);
	edit->set_visible_characters(visible_count);
	CHECK(edit->get_line_width(0) == prefix_width);

	edit->set_visible_characters_behavior(TextServer::VC_CHARS_AFTER_SHAPING);
	CHECK(edit->get_line_width(0) == full_width);
	edit->set_visible_characters_behavior(TextServer::VC_CHARS_BEFORE_SHAPING);
	CHECK(edit->get_line_width(0) == prefix_width);
	edit->set_visible_characters(-1);
	CHECK(edit->get_line_width(0) == full_width);

	memdelete(reference);
	memdelete(edit);
}

TEST_CASE("[SceneTree][RichTextEdit] Before shaping counts newlines and hides unrevealed lines") {
	RichTextEdit *edit = memnew(RichTextEdit);
	RichTextEdit *reference = memnew(RichTextEdit);
	Window *root = SceneTree::get_singleton()->get_root();
	root->add_child(edit);
	root->add_child(reference);
	edit->set_text("AB\nCD");
	reference->set_text("C");
	MessageQueue::get_singleton()->flush();
	const int first_width = edit->get_line_width(0);
	const int second_width = edit->get_line_width(1);

	edit->set_visible_characters(1);
	CHECK(edit->get_line_width(0) > 0);
	CHECK(edit->get_line_width(0) < first_width);
	CHECK(edit->get_line_width(1) == 0);
	edit->set_visible_characters(3);
	CHECK(edit->get_line_width(0) == first_width);
	CHECK(edit->get_line_width(1) == 0);
	edit->set_visible_characters(4);
	CHECK(edit->get_line_width(1) > 0);
	CHECK(edit->get_line_width(1) == reference->get_line_width(0));
	CHECK(edit->get_line(0) == "AB");
	CHECK(edit->get_line(1) == "CD");
	edit->set_visible_characters(-1);
	CHECK(edit->get_line_width(1) == second_width);

	memdelete(reference);
	memdelete(edit);
}

class PrefixParserEdit : public RichTextEdit {
public:
	mutable String last_parsed_text;

	TypedArray<Vector3i> structured_text_parser(TextServer::StructuredTextParser p_parser_type, const Array &p_args, const String &p_text) const override {
		last_parsed_text = p_text;
		TypedArray<Vector3i> ranges;
		// The full line intentionally has no overrides. Prefix parsing must still run.
		if (p_text.length() < 4) {
			ranges.push_back(Vector3i(0, p_text.length(), TextServer::DIRECTION_LTR));
		}
		return ranges;
	}
};

TEST_CASE("[SceneTree][RichTextEdit] Before shaping reparses the prefix even without full line overrides") {
	PrefixParserEdit *edit = memnew(PrefixParserEdit);
	RichTextEdit *reference = memnew(RichTextEdit);
	Window *root = SceneTree::get_singleton()->get_root();
	root->add_child(edit);
	root->add_child(reference);
	edit->set_text("ABCD");
	reference->set_text("AB");
	MessageQueue::get_singleton()->flush();

	edit->set_visible_characters(2);
	CHECK(edit->last_parsed_text == "AB");
	CHECK(edit->get_line_width(0) > 0);
	CHECK(edit->get_line_width(0) == reference->get_line_width(0));

	memdelete(reference);
	memdelete(edit);
}

} // namespace TestRichTextEdit

#endif // ADVANCED_GUI_DISABLED
