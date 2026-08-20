/**************************************************************************/
/*  test_reference_rect.cpp                                               */
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

TEST_FORCE_LINK(test_reference_rect)

#include "scene/gui/reference_rect.h"
#include "scene/main/scene_tree.h"
#include "scene/main/window.h"

namespace TestReferenceRect {

TEST_CASE("[SceneTree][ReferenceRect] Default properties") {
	ReferenceRect *reference_rect = memnew(ReferenceRect);

	CHECK(reference_rect->get_border_color() == Color(1, 0, 0, 1));
	CHECK(reference_rect->get_border_width() == 1.0f);
	CHECK(reference_rect->get_editor_only());

	CHECK(reference_rect->get_mouse_filter() == Control::MOUSE_FILTER_STOP);

	CHECK(reference_rect->get_combined_minimum_size() == Size2(0, 0));

	memdelete(reference_rect);
}

TEST_CASE("[SceneTree][ReferenceRect] Border color") {
	ReferenceRect *reference_rect = memnew(ReferenceRect);

	const Color c = Color(0.25f, 0.5f, 0.75f, 0.9f);
	reference_rect->set_border_color(c);
	CHECK(reference_rect->get_border_color() == c);

	memdelete(reference_rect);
}

TEST_CASE("[SceneTree][ReferenceRect] Border width") {
	ReferenceRect *reference_rect = memnew(ReferenceRect);

	SUBCASE("Positive width is stored") {
		reference_rect->set_border_width(3.5f);
		CHECK(reference_rect->get_border_width() == 3.5f);
	}

	SUBCASE("Negative width clamps to zero") {
		reference_rect->set_border_width(-2.0f);
		CHECK(reference_rect->get_border_width() == 0.0f);
	}

	SUBCASE("Width above the inspector hint range is still stored") {
		reference_rect->set_border_width(12.0f);
		CHECK(reference_rect->get_border_width() == 12.0f);
	}

	memdelete(reference_rect);
}

TEST_CASE("[SceneTree][ReferenceRect] Editor only flag") {
	ReferenceRect *reference_rect = memnew(ReferenceRect);

	SUBCASE("editor_only can be disabled") {
		reference_rect->set_editor_only(false);
		CHECK_FALSE(reference_rect->get_editor_only());
	}

	SUBCASE("editor_only can be re-enabled") {
		reference_rect->set_editor_only(false);
		reference_rect->set_editor_only(true);
		CHECK(reference_rect->get_editor_only());
	}

	SUBCASE("Toggling editor_only does not change minimum size") {
		reference_rect->set_editor_only(false);
		CHECK(reference_rect->get_combined_minimum_size() == Size2(0, 0));

		reference_rect->set_editor_only(true);
		CHECK(reference_rect->get_combined_minimum_size() == Size2(0, 0));
	}

	memdelete(reference_rect);
}

TEST_CASE("[SceneTree][ReferenceRect] Minimum size on tree") {
	ReferenceRect *reference_rect = memnew(ReferenceRect);
	Window *root = SceneTree::get_singleton()->get_root();
	root->add_child(reference_rect);

	reference_rect->set_border_width(4.0f);
	reference_rect->set_editor_only(false);
	reference_rect->set_border_color(Color(0, 1, 0, 1));
	SceneTree::get_singleton()->process(0);

	CHECK(reference_rect->get_combined_minimum_size() == Size2(0, 0));

	memdelete(reference_rect);
}

} // namespace TestReferenceRect
