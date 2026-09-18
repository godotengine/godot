/**************************************************************************/
/*  test_item_list.cpp                                                    */
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

TEST_FORCE_LINK(test_item_list)

#ifndef ADVANCED_GUI_DISABLED

#include "core/io/image.h"
#include "scene/gui/item_list.h"
#include "scene/resources/image_texture.h"

namespace TestItemList {

TEST_CASE("[SceneTree][ItemList] Default properties") {
	ItemList *list = memnew(ItemList);

	CHECK(list->get_item_count() == 0);
	CHECK(list->get_current() == -1);
	CHECK_FALSE(list->is_anything_selected());
	CHECK(list->get_selected_items().is_empty());

	CHECK(list->get_select_mode() == ItemList::SELECT_SINGLE);
	CHECK(list->get_icon_mode() == ItemList::ICON_MODE_LEFT);
	CHECK(list->get_scroll_hint_mode() == ItemList::SCROLL_HINT_MODE_DISABLED);

	CHECK(list->get_max_columns() == 1);
	CHECK(list->get_max_text_lines() == 1);
	CHECK(list->get_fixed_column_width() == 0);
	CHECK_FALSE(list->is_same_column_width());
	CHECK_FALSE(list->has_auto_width());
	CHECK_FALSE(list->has_auto_height());
	CHECK(list->has_wraparound_items());

	CHECK(list->get_allow_search());
	CHECK_FALSE(list->get_allow_rmb_select());
	CHECK_FALSE(list->get_allow_reselect());
	CHECK(list->get_icon_scale() == real_t(1.0));

	// Constructor overrides Control defaults.
	CHECK(list->get_focus_mode() == Control::FOCUS_ALL);
	CHECK(list->is_clipping_contents());

	// Internal scrollbars are created in the constructor.
	CHECK(list->get_v_scroll_bar() != nullptr);
	CHECK(list->get_h_scroll_bar() != nullptr);

	memdelete(list);
}

TEST_CASE("[SceneTree][ItemList] Item CRUD and negative indices") {
	ItemList *list = memnew(ItemList);

	CHECK(list->add_item("first") == 0);
	CHECK(list->add_item("second") == 1);
	CHECK(list->add_item("third") == 2);
	CHECK(list->get_item_count() == 3);

	// Negative index addresses from the end (e.g. -1 == last).
	list->set_item_text(-1, "last_renamed");
	CHECK(list->get_item_text(2) == "last_renamed");

	list->remove_item(1);
	CHECK(list->get_item_count() == 2);
	CHECK(list->get_item_text(0) == "first");
	CHECK(list->get_item_text(1) == "last_renamed");

	// `set_item_count` grows the list with default items.
	list->set_item_count(4);
	CHECK(list->get_item_count() == 4);
	list->set_item_text(0, "a");
	list->set_item_text(3, "d");
	CHECK(list->get_item_text(0) == "a");
	CHECK(list->get_item_text(3) == "d");

	// `set_item_count` shrinks the list, dropping trailing items.
	list->set_item_count(2);
	CHECK(list->get_item_count() == 2);
	CHECK(list->get_item_text(0) == "a");

	// `add_item` honors `p_selectable=false`.
	const int unselectable_idx = list->add_item("locked", Ref<Texture2D>(), false);
	CHECK_FALSE(list->is_item_selectable(unselectable_idx));

	// `add_icon_item` round-trips the texture.
	Ref<Image> image = memnew(Image(4, 4, false, Image::FORMAT_RGB8));
	Ref<ImageTexture> tex = ImageTexture::create_from_image(image);
	const int icon_idx = list->add_icon_item(tex);
	CHECK(list->get_item_icon(icon_idx) == tex);

	memdelete(list);
}

TEST_CASE("[SceneTree][ItemList] clear() resets count, current, and selection") {
	ItemList *list = memnew(ItemList);
	list->add_item("a");
	list->add_item("b");
	list->select(1);
	CHECK(list->get_current() == 1);

	list->clear();
	CHECK(list->get_item_count() == 0);
	CHECK(list->get_current() == -1);
	CHECK_FALSE(list->is_anything_selected());

	// `clear()` on an already empty list is safe and idempotent.
	list->clear();
	CHECK(list->get_item_count() == 0);

	memdelete(list);
}

TEST_CASE("[SceneTree][ItemList] remove_item current-row semantics") {
	SUBCASE("Removing the current row clears current and selection (single mode)") {
		ItemList *list = memnew(ItemList);
		list->add_item("a");
		list->add_item("b");
		list->select(1);
		CHECK(list->get_current() == 1);

		list->remove_item(1);
		CHECK(list->get_current() == -1);
		CHECK_FALSE(list->is_anything_selected());

		memdelete(list);
	}

	SUBCASE("Removing a row after the current row preserves current and selection") {
		ItemList *list = memnew(ItemList);
		list->add_item("a");
		list->add_item("b");
		list->add_item("c");
		list->select(0);
		CHECK(list->get_current() == 0);
		CHECK(list->is_selected(0));

		// Removing a row whose index is greater than `current` must not
		// change `current` and must preserve the existing selection.
		list->remove_item(2);
		CHECK(list->get_item_count() == 2);
		CHECK(list->get_current() == 0);
		CHECK(list->is_selected(0));

		memdelete(list);
	}

	SUBCASE("Multi-select: removing the current row keeps other selections") {
		ItemList *list = memnew(ItemList);
		list->set_select_mode(ItemList::SELECT_MULTI);
		list->add_item("a");
		list->add_item("b");
		list->add_item("c");

		list->select(0, false);
		list->select(2, false);
		list->set_current(1);
		CHECK(list->get_current() == 1);

		list->remove_item(1);
		CHECK(list->get_item_count() == 2);
		CHECK(list->get_current() == -1);
		// Surviving items keep their selection state.
		CHECK(list->is_selected(0)); // Was "a".
		CHECK(list->is_selected(1)); // Was "c", now at index 1.

		memdelete(list);
	}
}

TEST_CASE("[SceneTree][ItemList] Selection in SELECT_SINGLE mode") {
	ItemList *list = memnew(ItemList);
	list->add_item("a");
	list->add_item("b");
	list->add_item("c");

	SUBCASE("select() replaces previous selection and updates current") {
		list->select(1);
		CHECK(list->is_selected(1));
		CHECK(list->get_current() == 1);

		list->select(2);
		CHECK(list->is_selected(2));
		CHECK_FALSE(list->is_selected(1));
		CHECK(list->get_current() == 2);
	}

	SUBCASE("deselect() clears selection and resets current to -1") {
		list->select(0);
		CHECK(list->get_current() == 0);

		list->deselect(0);
		CHECK_FALSE(list->is_selected(0));
		CHECK(list->get_current() == -1);
	}

	SUBCASE("select() is a no-op on disabled items and does not change selection") {
		list->select(1);
		CHECK(list->is_selected(1));

		list->set_item_disabled(0, true);
		list->select(0);
		CHECK_FALSE(list->is_selected(0));
		// Previous selection on item 1 is preserved.
		CHECK(list->is_selected(1));
		CHECK(list->get_current() == 1);
	}

	SUBCASE("select() is a no-op on non-selectable items") {
		list->set_item_selectable(2, false);
		list->select(2);
		CHECK_FALSE(list->is_selected(2));
		CHECK(list->get_current() == -1);
	}

	SUBCASE("deselect_all() clears every selection and resets current") {
		list->select(2);
		CHECK(list->is_anything_selected());

		list->deselect_all();
		CHECK_FALSE(list->is_anything_selected());
		CHECK(list->get_current() == -1);
	}

	memdelete(list);
}

TEST_CASE("[SceneTree][ItemList] Selection in SELECT_MULTI mode") {
	ItemList *list = memnew(ItemList);
	list->set_select_mode(ItemList::SELECT_MULTI);
	list->add_item("a");
	list->add_item("b");
	list->add_item("c");

	SUBCASE("select(idx, false) is additive; select(idx) is exclusive") {
		list->select(0, false);
		list->select(2, false);
		CHECK(list->is_selected(0));
		CHECK(list->is_selected(2));
		CHECK_FALSE(list->is_selected(1));

		const Vector<int> selected = list->get_selected_items();
		REQUIRE(selected.size() == 2);
		CHECK(selected[0] == 0);
		CHECK(selected[1] == 2);

		// Default `select(idx)` (p_single=true) is exclusive even in multi
		// mode: it clears other selections and sets `current`.
		list->select(1);
		CHECK(list->is_selected(1));
		CHECK_FALSE(list->is_selected(0));
		CHECK_FALSE(list->is_selected(2));
		CHECK(list->get_current() == 1);
	}

	SUBCASE("set_current() does not select in multi mode") {
		list->select(0, false);
		list->set_current(2);
		CHECK(list->get_current() == 2);
		// `set_current` must not implicitly select in multi mode.
		CHECK_FALSE(list->is_selected(2));
		CHECK(list->is_selected(0));
	}

	SUBCASE("deselect() clears one item without touching others") {
		list->select(0, false);
		list->select(2, false);

		list->deselect(0);
		CHECK_FALSE(list->is_selected(0));
		CHECK(list->is_selected(2));
	}

	SUBCASE("deselect_all() clears every selection and resets current") {
		list->select(0, false);
		list->select(2, false);
		list->set_current(1);

		list->deselect_all();
		CHECK_FALSE(list->is_anything_selected());
		CHECK(list->get_current() == -1);
	}

	memdelete(list);
}

TEST_CASE("[SceneTree][ItemList] move_item updates current for the first selected row") {
	SUBCASE("Single selection: current follows the moved row") {
		ItemList *list = memnew(ItemList);
		list->add_item("a");
		list->add_item("b");
		list->add_item("c");
		list->select(2);
		CHECK(list->get_current() == 2);

		list->move_item(2, 0);
		CHECK(list->get_item_text(0) == "c");
		CHECK(list->get_item_text(1) == "a");
		CHECK(list->get_item_text(2) == "b");
		CHECK(list->is_selected(0));
		CHECK(list->get_current() == 0);

		memdelete(list);
	}

	SUBCASE("Multi mode: current updates only when first selected matches p_from_idx") {
		ItemList *list = memnew(ItemList);
		list->set_select_mode(ItemList::SELECT_MULTI);
		list->add_item("a");
		list->add_item("b");
		list->add_item("c");
		list->select(0, false);
		list->select(2, false);
		// `set_current` does not select in multi mode.
		list->set_current(1);
		CHECK(list->get_current() == 1);

		// First selected is index 0; moving index 2 must NOT change current.
		list->move_item(2, 0);
		CHECK(list->get_item_text(0) == "c");
		CHECK(list->get_current() == 1);

		memdelete(list);
	}
}

TEST_CASE("[SceneTree][ItemList] sort_items_by_text") {
	SUBCASE("Single mode: re-syncs current to the still-selected row") {
		ItemList *list = memnew(ItemList);
		list->add_item("z");
		list->add_item("m");
		list->add_item("a");
		list->select(1);
		CHECK(list->get_item_text(list->get_current()) == "m");

		list->sort_items_by_text();
		CHECK(list->get_item_text(0) == "a");
		CHECK(list->get_item_text(1) == "m");
		CHECK(list->get_item_text(2) == "z");
		CHECK(list->is_selected(1));
		CHECK(list->get_current() == 1);

		memdelete(list);
	}

	SUBCASE("Multi mode: selection flags travel with items, current is not adjusted") {
		ItemList *list = memnew(ItemList);
		list->set_select_mode(ItemList::SELECT_MULTI);
		list->add_item("z");
		list->add_item("m");
		list->add_item("a");
		list->select(0, false); // "z"
		list->select(2, false); // "a"
		CHECK(list->get_current() == -1);

		list->sort_items_by_text();
		CHECK(list->get_item_text(0) == "a");
		CHECK(list->get_item_text(1) == "m");
		CHECK(list->get_item_text(2) == "z");
		// "a" and "z" stay selected; "m" remains unselected.
		CHECK(list->is_selected(0));
		CHECK_FALSE(list->is_selected(1));
		CHECK(list->is_selected(2));
		// Multi-mode sort intentionally does not touch `current`.
		CHECK(list->get_current() == -1);

		memdelete(list);
	}
}

TEST_CASE("[SceneTree][ItemList] find_metadata") {
	ItemList *list = memnew(ItemList);
	list->add_item("a");
	list->add_item("b");
	list->add_item("c");
	list->set_item_metadata(0, 42);
	list->set_item_metadata(1, String("tag"));
	list->set_item_metadata(2, Vector2(1, 2));

	CHECK(list->find_metadata(42) == 0);
	CHECK(list->find_metadata(String("tag")) == 1);
	CHECK(list->find_metadata(Vector2(1, 2)) == 2);
	CHECK(list->find_metadata(999) == -1);
	CHECK(list->find_metadata(String("missing")) == -1);

	memdelete(list);
}

TEST_CASE("[SceneTree][ItemList] Guards reject invalid input without mutating state") {
	ItemList *list = memnew(ItemList);

	// `set_max_text_lines(0)` is rejected; default of 1 is preserved.
	CHECK(list->get_max_text_lines() == 1);
	ERR_PRINT_OFF
	list->set_max_text_lines(0);
	ERR_PRINT_ON
	CHECK(list->get_max_text_lines() == 1);

	// `set_fixed_column_width(<0)` is rejected; default of 0 is preserved.
	CHECK(list->get_fixed_column_width() == 0);
	ERR_PRINT_OFF
	list->set_fixed_column_width(-1);
	ERR_PRINT_ON
	CHECK(list->get_fixed_column_width() == 0);

	list->add_item("x");

	// Out-of-range `TextDirection` is rejected; per-item value is preserved.
	const Control::TextDirection dir_before = list->get_item_text_direction(0);
	ERR_PRINT_OFF
	list->set_item_text_direction(0, static_cast<Control::TextDirection>(4));
	ERR_PRINT_ON
	CHECK(list->get_item_text_direction(0) == dir_before);

	// NaN icon scale is rejected; previous value is preserved.
	const real_t scale_before = list->get_icon_scale();
	ERR_PRINT_OFF
	list->set_icon_scale(NAN);
	ERR_PRINT_ON
	CHECK(list->get_icon_scale() == scale_before);

	// Out-of-range item access on `remove_item` is rejected without crashing.
	const int count_before = list->get_item_count();
	ERR_PRINT_OFF
	list->remove_item(99);
	ERR_PRINT_ON
	CHECK(list->get_item_count() == count_before);

	memdelete(list);
}

} // namespace TestItemList

#endif // ADVANCED_GUI_DISABLED
