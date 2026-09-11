/**************************************************************************/
/*  test_link_button.cpp                                                  */
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

TEST_FORCE_LINK(test_link_button)

#include "scene/gui/link_button.h"

namespace TestLinkButton {

TEST_CASE("[SceneTree][LinkButton] Default properties") {
	LinkButton *lb = memnew(LinkButton);

	CHECK(lb->get_text().is_empty());
	CHECK(lb->get_uri().is_empty());
	CHECK(lb->get_underline_mode() == LinkButton::UNDERLINE_MODE_ALWAYS);
	CHECK(lb->get_text_direction() == Control::TEXT_DIRECTION_AUTO);
	CHECK(lb->get_language().is_empty());
	CHECK(lb->get_text_overrun_behavior() == TextServer::OVERRUN_NO_TRIMMING);
	CHECK(lb->get_structured_text_bidi_override() == TextServer::STRUCTURED_TEXT_DEFAULT);
	CHECK(lb->get_structured_text_bidi_override_options().is_empty());

	// Constructor overrides Control defaults.
	CHECK(lb->get_focus_mode() == Control::FOCUS_ACCESSIBILITY);
	CHECK(lb->get_cursor_shape(Point2()) == Control::CURSOR_POINTING_HAND);

	memdelete(lb);
}

TEST_CASE("[SceneTree][LinkButton] Constructor sets text") {
	LinkButton *lb = memnew(LinkButton("Visit us"));

	CHECK(lb->get_text() == "Visit us");

	memdelete(lb);
}

TEST_CASE("[SceneTree][LinkButton] Text round-trip and idempotency") {
	LinkButton *lb = memnew(LinkButton);

	lb->set_text("click here");
	CHECK(lb->get_text() == "click here");

	// Setting the same text again is a no-op (exercises the early-return path).
	lb->set_text("click here");
	CHECK(lb->get_text() == "click here");

	lb->set_text("");
	CHECK(lb->get_text().is_empty());

	memdelete(lb);
}

TEST_CASE("[SceneTree][LinkButton] Overrun trimming zeroes minimum width") {
	LinkButton *lb = memnew(LinkButton);

	lb->set_text("A");

	// When overrun trimming is active the widget defers width to the container,
	// so get_combined_minimum_size() must report zero width regardless of content.
	lb->set_text_overrun_behavior(TextServer::OVERRUN_TRIM_CHAR);
	CHECK(lb->get_combined_minimum_size().width == 0);

	// Restoring to no-trimming removes the zero override.
	lb->set_text_overrun_behavior(TextServer::OVERRUN_NO_TRIMMING);
	CHECK(lb->get_combined_minimum_size().width >= 0);

	// Re-enabling trimming must zero it again (reversibility).
	lb->set_text_overrun_behavior(TextServer::OVERRUN_TRIM_WORD);
	CHECK(lb->get_combined_minimum_size().width == 0);

	memdelete(lb);
}

TEST_CASE("[SceneTree][LinkButton] URI round-trip") {
	LinkButton *lb = memnew(LinkButton);

	lb->set_uri("https://godotengine.org");
	CHECK(lb->get_uri() == "https://godotengine.org");

	// Setting the same URI again is a no-op (exercises the early-return path).
	lb->set_uri("https://godotengine.org");
	CHECK(lb->get_uri() == "https://godotengine.org");

	lb->set_uri("");
	CHECK(lb->get_uri().is_empty());

	memdelete(lb);
}

TEST_CASE("[SceneTree][LinkButton] Underline mode — all three enum values") {
	LinkButton *lb = memnew(LinkButton);

	lb->set_underline_mode(LinkButton::UNDERLINE_MODE_ON_HOVER);
	CHECK(lb->get_underline_mode() == LinkButton::UNDERLINE_MODE_ON_HOVER);

	lb->set_underline_mode(LinkButton::UNDERLINE_MODE_NEVER);
	CHECK(lb->get_underline_mode() == LinkButton::UNDERLINE_MODE_NEVER);

	lb->set_underline_mode(LinkButton::UNDERLINE_MODE_ALWAYS);
	CHECK(lb->get_underline_mode() == LinkButton::UNDERLINE_MODE_ALWAYS);

	// Setting the same mode again is a no-op (exercises the early-return path).
	lb->set_underline_mode(LinkButton::UNDERLINE_MODE_ALWAYS);
	CHECK(lb->get_underline_mode() == LinkButton::UNDERLINE_MODE_ALWAYS);

	memdelete(lb);
}

TEST_CASE("[SceneTree][LinkButton] Cursor shape: pointing hand when enabled, arrow when disabled") {
	LinkButton *lb = memnew(LinkButton);

	CHECK(lb->get_cursor_shape(Point2()) == Control::CURSOR_POINTING_HAND);

	lb->set_disabled(true);
	CHECK(lb->get_cursor_shape(Point2()) == Control::CURSOR_ARROW);

	lb->set_disabled(false);
	CHECK(lb->get_cursor_shape(Point2()) == Control::CURSOR_POINTING_HAND);

	memdelete(lb);
}

TEST_CASE("[SceneTree][LinkButton] Ellipsis char: round-trip and multi-character clamping") {
	LinkButton *lb = memnew(LinkButton);

	lb->set_ellipsis_char("!");
	CHECK(lb->get_ellipsis_char() == "!");

	// Multi-character input: implementation truncates to the first character.
	// WARN_PRINT fires for this deliberate bad input; suppress it to keep output clean.
	ERR_PRINT_OFF
	lb->set_ellipsis_char("abc");
	ERR_PRINT_ON
	CHECK(lb->get_ellipsis_char() == "a");

	memdelete(lb);
}

TEST_CASE("[SceneTree][LinkButton] Text direction: all four valid values and guard") {
	LinkButton *lb = memnew(LinkButton);

	lb->set_text_direction(Control::TEXT_DIRECTION_LTR);
	CHECK(lb->get_text_direction() == Control::TEXT_DIRECTION_LTR);

	lb->set_text_direction(Control::TEXT_DIRECTION_RTL);
	CHECK(lb->get_text_direction() == Control::TEXT_DIRECTION_RTL);

	lb->set_text_direction(Control::TEXT_DIRECTION_INHERITED);
	CHECK(lb->get_text_direction() == Control::TEXT_DIRECTION_INHERITED);

	lb->set_text_direction(Control::TEXT_DIRECTION_AUTO);
	CHECK(lb->get_text_direction() == Control::TEXT_DIRECTION_AUTO);

	// Out-of-range values are rejected; the stored value must be preserved.
	const Control::TextDirection dir_before = lb->get_text_direction();
	ERR_PRINT_OFF
	lb->set_text_direction(static_cast<Control::TextDirection>(4));
	ERR_PRINT_ON
	CHECK(lb->get_text_direction() == dir_before);

	memdelete(lb);
}

TEST_CASE("[SceneTree][LinkButton] Structured text BiDi override round-trip") {
	LinkButton *lb = memnew(LinkButton);

	lb->set_structured_text_bidi_override(TextServer::STRUCTURED_TEXT_URI);
	CHECK(lb->get_structured_text_bidi_override() == TextServer::STRUCTURED_TEXT_URI);

	// Setting the same parser again is a no-op (exercises the early-return path).
	lb->set_structured_text_bidi_override(TextServer::STRUCTURED_TEXT_URI);
	CHECK(lb->get_structured_text_bidi_override() == TextServer::STRUCTURED_TEXT_URI);

	Array options;
	options.push_back("a");
	options.push_back(2);
	lb->set_structured_text_bidi_override_options(options);
	CHECK(lb->get_structured_text_bidi_override_options() == options);

	// Reset to default parser and empty options; getters must reflect the change.
	lb->set_structured_text_bidi_override(TextServer::STRUCTURED_TEXT_DEFAULT);
	CHECK(lb->get_structured_text_bidi_override() == TextServer::STRUCTURED_TEXT_DEFAULT);

	lb->set_structured_text_bidi_override_options(Array());
	CHECK(lb->get_structured_text_bidi_override_options().is_empty());

	memdelete(lb);
}

TEST_CASE("[SceneTree][LinkButton] Language round-trip") {
	LinkButton *lb = memnew(LinkButton);

	lb->set_language("fr");
	CHECK(lb->get_language() == "fr");

	lb->set_language("");
	CHECK(lb->get_language().is_empty());

	memdelete(lb);
}

} // namespace TestLinkButton
