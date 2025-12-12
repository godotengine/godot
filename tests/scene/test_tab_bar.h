/**************************************************************************/
/*  test_tab_bar.h                                                        */
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

#pragma once

#include "scene/gui/tab_bar.h"
#include "scene/main/window.h"

#include "tests/test_macros.h"

namespace TestTabBar {
TEST_CASE("[SceneTree][TabBar] tab operations") {
	TabBar *tab_bar = memnew(TabBar);
	SceneTree::get_singleton()->get_root()->add_child(tab_bar);
	tab_bar->set_clip_tabs(false);
	MessageQueue::get_singleton()->flush();

	SIGNAL_WATCH(tab_bar, "tab_selected");
	SIGNAL_WATCH(tab_bar, "tab_changed");

	SUBCASE("[TabBar] no tabs") {
		CHECK(tab_bar->get_tab_count() == 0);
		CHECK(tab_bar->get_current_tab() == -1);
		CHECK(tab_bar->get_previous_tab() == -1);
	}

	SUBCASE("[TabBar] add tabs disallowing deselection") {
		tab_bar->set_deselect_enabled(false);
		tab_bar->add_tab("tab0");
		CHECK(tab_bar->get_tab_count() == 1);
		CHECK(tab_bar->get_current_tab() == 0);
		CHECK(tab_bar->get_previous_tab() == -1);
		SIGNAL_CHECK("tab_selected", { { 0 } });
		SIGNAL_CHECK("tab_changed", { { 0 } });

		tab_bar->add_tab("tab1");
		CHECK(tab_bar->get_tab_count() == 2);
		CHECK(tab_bar->get_current_tab() == 0);
		CHECK(tab_bar->get_previous_tab() == -1);
		SIGNAL_CHECK_FALSE("tab_selected");
		SIGNAL_CHECK_FALSE("tab_changed");

		tab_bar->add_tab("tab2");
		CHECK(tab_bar->get_tab_count() == 3);
		CHECK(tab_bar->get_current_tab() == 0);
		CHECK(tab_bar->get_previous_tab() == -1);
		SIGNAL_CHECK_FALSE("tab_selected");
		SIGNAL_CHECK_FALSE("tab_changed");

		CHECK(tab_bar->get_tab_title(0) == "tab0");
		CHECK(tab_bar->get_tab_tooltip(0) == "");
		CHECK(tab_bar->get_tab_text_direction(0) == Control::TEXT_DIRECTION_INHERITED);
		CHECK_FALSE(tab_bar->is_tab_disabled(0));
		CHECK_FALSE(tab_bar->is_tab_hidden(0));

		CHECK(tab_bar->get_tab_title(1) == "tab1");
		CHECK(tab_bar->get_tab_tooltip(1) == "");
		CHECK(tab_bar->get_tab_text_direction(1) == Control::TEXT_DIRECTION_INHERITED);
		CHECK_FALSE(tab_bar->is_tab_disabled(1));
		CHECK_FALSE(tab_bar->is_tab_hidden(1));

		CHECK(tab_bar->get_tab_title(2) == "tab2");
		CHECK(tab_bar->get_tab_tooltip(2) == "");
		CHECK(tab_bar->get_tab_text_direction(2) == Control::TEXT_DIRECTION_INHERITED);
		CHECK_FALSE(tab_bar->is_tab_disabled(2));
		CHECK_FALSE(tab_bar->is_tab_hidden(2));
	}

	SUBCASE("[TabBar] add tabs allowing deselection") {
		tab_bar->set_deselect_enabled(true);
		tab_bar->add_tab("tab0");
		CHECK(tab_bar->get_tab_count() == 1);
		CHECK(tab_bar->get_current_tab() == -1);
		CHECK(tab_bar->get_previous_tab() == -1);
		SIGNAL_CHECK_FALSE("tab_selected");
		SIGNAL_CHECK_FALSE("tab_changed");

		tab_bar->add_tab("tab1");
		CHECK(tab_bar->get_tab_count() == 2);
		CHECK(tab_bar->get_current_tab() == -1);
		CHECK(tab_bar->get_previous_tab() == -1);
		SIGNAL_CHECK_FALSE("tab_selected");
		SIGNAL_CHECK_FALSE("tab_changed");
	}

	SUBCASE("[TabBar] set tab count") {
		// Adds multiple tabs at once.
		tab_bar->set_tab_count(3);
		CHECK(tab_bar->get_tab_count() == 3);
		CHECK(tab_bar->get_current_tab() == 0);
		CHECK(tab_bar->get_previous_tab() == -1);
		SIGNAL_CHECK_FALSE("tab_selected");
		SIGNAL_CHECK_FALSE("tab_changed");

		CHECK(tab_bar->get_tab_title(0) == "");
		CHECK(tab_bar->get_tab_tooltip(0) == "");
		CHECK(tab_bar->get_tab_text_direction(0) == Control::TEXT_DIRECTION_INHERITED);
		CHECK_FALSE(tab_bar->is_tab_disabled(0));
		CHECK_FALSE(tab_bar->is_tab_hidden(0));

		CHECK(tab_bar->get_tab_title(1) == "");
		CHECK(tab_bar->get_tab_tooltip(1) == "");
		CHECK(tab_bar->get_tab_text_direction(1) == Control::TEXT_DIRECTION_INHERITED);
		CHECK_FALSE(tab_bar->is_tab_disabled(1));
		CHECK_FALSE(tab_bar->is_tab_hidden(1));

		CHECK(tab_bar->get_tab_title(2) == "");
		CHECK(tab_bar->get_tab_tooltip(2) == "");
		CHECK(tab_bar->get_tab_text_direction(2) == Control::TEXT_DIRECTION_INHERITED);
		CHECK_FALSE(tab_bar->is_tab_disabled(2));
		CHECK_FALSE(tab_bar->is_tab_hidden(2));

		// Setting to less tabs than there are removes from the end.
		tab_bar->set_tab_title(0, "tab0");
		tab_bar->set_tab_title(1, "tab1");
		tab_bar->set_tab_title(2, "tab2");

		tab_bar->set_tab_count(2);
		CHECK(tab_bar->get_tab_count() == 2);
		CHECK(tab_bar->get_current_tab() == 0);
		CHECK(tab_bar->get_previous_tab() == -1);
		CHECK(tab_bar->get_tab_title(0) == "tab0");
		CHECK(tab_bar->get_tab_title(1) == "tab1");

		// Remove all tabs.
		tab_bar->set_tab_count(0);
		CHECK(tab_bar->get_tab_count() == 0);
		CHECK(tab_bar->get_current_tab() == -1);
		CHECK(tab_bar->get_previous_tab() == -1);
	}

	SUBCASE("[TabBar] clear tabs") {
		CHECK(tab_bar->get_tab_count() == 0);
		CHECK(tab_bar->get_current_tab() == -1);
		CHECK(tab_bar->get_previous_tab() == -1);

		tab_bar->set_tab_count(2);
		CHECK(tab_bar->get_tab_count() == 2);
		CHECK(tab_bar->get_current_tab() == 0);
		CHECK(tab_bar->get_previous_tab() == -1);
		SIGNAL_DISCARD("tab_selected");
		SIGNAL_DISCARD("tab_changed");

		tab_bar->clear_tabs();
		CHECK(tab_bar->get_tab_count() == 0);
		CHECK(tab_bar->get_current_tab() == -1);
		CHECK(tab_bar->get_previous_tab() == -1);
		SIGNAL_CHECK_FALSE("tab_selected");
		SIGNAL_CHECK_FALSE("tab_changed");
	}

	SUBCASE("[TabBar] remove tabs") {
		tab_bar->add_tab("tab0");
		tab_bar->add_tab("tab1");
		tab_bar->add_tab("tab2");
		tab_bar->set_current_tab(1);
		CHECK(tab_bar->get_tab_count() == 3);
		CHECK(tab_bar->get_current_tab() == 1);
		CHECK(tab_bar->get_previous_tab() == 0);
		SIGNAL_DISCARD("tab_selected");
		SIGNAL_DISCARD("tab_changed");

		// Remove first tab.
		tab_bar->remove_tab(0);
		CHECK(tab_bar->get_tab_count() == 2);
		CHECK(tab_bar->get_tab_title(0) == "tab1");
		CHECK(tab_bar->get_tab_title(1) == "tab2");
		CHECK(tab_bar->get_current_tab() == 0);
		CHECK(tab_bar->get_previous_tab() == 0);
		SIGNAL_CHECK_FALSE("tab_selected");
		SIGNAL_CHECK_FALSE("tab_changed");

		// Remove last tab.
		tab_bar->remove_tab(1);
		CHECK(tab_bar->get_tab_count() == 1);
		CHECK(tab_bar->get_tab_title(0) == "tab1");
		CHECK(tab_bar->get_current_tab() == 0);
		CHECK(tab_bar->get_previous_tab() == 0);
		SIGNAL_CHECK_FALSE("tab_selected");
		SIGNAL_CHECK_FALSE("tab_changed");

		// Remove only tab.
		tab_bar->remove_tab(0);
		CHECK(tab_bar->get_tab_count() == 0);
		CHECK(tab_bar->get_current_tab() == -1);
		CHECK(tab_bar->get_previous_tab() == -1);
		SIGNAL_CHECK_FALSE("tab_selected");
		SIGNAL_CHECK("tab_changed", { { -1 } });

		// Remove current tab when there are other tabs.
		tab_bar->add_tab("tab0");
		tab_bar->add_tab("tab1");
		tab_bar->add_tab("tab2");
		tab_bar->set_current_tab(1);
		tab_bar->set_current_tab(2);
		CHECK(tab_bar->get_tab_count() == 3);
		CHECK(tab_bar->get_current_tab() == 2);
		CHECK(tab_bar->get_previous_tab() == 1);
		SIGNAL_DISCARD("tab_selected");
		SIGNAL_DISCARD("tab_changed");

		tab_bar->remove_tab(2);
		CHECK(tab_bar->get_tab_count() == 2);
		CHECK(tab_bar->get_current_tab() == 1);
		CHECK(tab_bar->get_previous_tab() == 1);
		SIGNAL_CHECK_FALSE("tab_selected");
		SIGNAL_CHECK("tab_changed", { { 1 } });
	}

	SUBCASE("[TabBar] move tabs") {
		tab_bar->add_tab("tab0");
		tab_bar->add_tab("tab1");
		tab_bar->add_tab("tab2");
		tab_bar->set_current_tab(1);
		CHECK(tab_bar->get_current_tab() == 1);
		CHECK(tab_bar->get_previous_tab() == 0);
		SIGNAL_DISCARD("tab_selected");
		SIGNAL_DISCARD("tab_changed");

		// Don't move if index is the same.
		tab_bar->move_tab(0, 0);
		CHECK(tab_bar->get_tab_title(0) == "tab0");
		CHECK(tab_bar->get_tab_title(1) == "tab1");
		CHECK(tab_bar->get_tab_title(2) == "tab2");
		CHECK(tab_bar->get_current_tab() == 1);
		CHECK(tab_bar->get_previous_tab() == 0);
		SIGNAL_CHECK_FALSE("tab_selected");
		SIGNAL_CHECK_FALSE("tab_changed");

		// Move the first tab to the end.
		tab_bar->move_tab(0, 2);
		CHECK(tab_bar->get_tab_title(0) == "tab1");
		CHECK(tab_bar->get_tab_title(1) == "tab2");
		CHECK(tab_bar->get_tab_title(2) == "tab0");
		CHECK(tab_bar->get_current_tab() == 0);
		CHECK(tab_bar->get_previous_tab() == 2);
		SIGNAL_CHECK_FALSE("tab_selected");
		SIGNAL_CHECK_FALSE("tab_changed");

		// Move the second tab to the front.
		tab_bar->move_tab(1, 0);
		CHECK(tab_bar->get_tab_title(0) == "tab2");
		CHECK(tab_bar->get_tab_title(1) == "tab1");
		CHECK(tab_bar->get_tab_title(2) == "tab0");
		CHECK(tab_bar->get_current_tab() == 1);
		CHECK(tab_bar->get_previous_tab() == 2);
		SIGNAL_CHECK_FALSE("tab_selected");
		SIGNAL_CHECK_FALSE("tab_changed");
	}

	SUBCASE("[TabBar] set current tab") {
		tab_bar->add_tab("tab0");
		tab_bar->add_tab("tab1");
		tab_bar->add_tab("tab2");
		CHECK(tab_bar->get_current_tab() == 0);
		CHECK(tab_bar->get_previous_tab() == -1);
		SIGNAL_CHECK("tab_selected", { { 0 } });
		SIGNAL_CHECK("tab_changed", { { 0 } });

		// Set the current tab.
		tab_bar->set_current_tab(1);
		CHECK(tab_bar->get_current_tab() == 1);
		CHECK(tab_bar->get_previous_tab() == 0);
		SIGNAL_CHECK("tab_selected", { { 1 } });
		SIGNAL_CHECK("tab_changed", { { 1 } });

		// Set to same tab.
		tab_bar->set_current_tab(1);
		CHECK(tab_bar->get_current_tab() == 1);
		CHECK(tab_bar->get_previous_tab() == 1);
		SIGNAL_CHECK("tab_selected", { { 1 } });
		SIGNAL_CHECK_FALSE("tab_changed");

		// Out of bounds.
		ERR_PRINT_OFF;
		tab_bar->set_current_tab(-5);
		CHECK(tab_bar->get_current_tab() == 1);
		CHECK(tab_bar->get_previous_tab() == 1);
		SIGNAL_CHECK_FALSE("tab_selected");
		SIGNAL_CHECK_FALSE("tab_changed");

		tab_bar->set_current_tab(5);
		CHECK(tab_bar->get_current_tab() == 1);
		CHECK(tab_bar->get_previous_tab() == 1);
		SIGNAL_CHECK_FALSE("tab_selected");
		SIGNAL_CHECK_FALSE("tab_changed");
		ERR_PRINT_ON;
	}

	SUBCASE("[TabBar] deselection enabled") {
		tab_bar->add_tab("tab0");
		tab_bar->add_tab("tab1");
		tab_bar->add_tab("tab2");
		CHECK(tab_bar->get_current_tab() == 0);
		CHECK(tab_bar->get_previous_tab() == -1);
		SIGNAL_DISCARD("tab_selected");
		SIGNAL_DISCARD("tab_changed");

		// Setting deselect enabled doesn't change current tab.
		tab_bar->set_deselect_enabled(true);
		CHECK(tab_bar->get_deselect_enabled());
		CHECK(tab_bar->get_current_tab() == 0);
		CHECK(tab_bar->get_previous_tab() == -1);
		SIGNAL_CHECK_FALSE("tab_selected");
		SIGNAL_CHECK_FALSE("tab_changed");

		// Can deselect all tabs by setting current to -1.
		tab_bar->set_current_tab(-1);
		CHECK(tab_bar->get_current_tab() == -1);
		CHECK(tab_bar->get_previous_tab() == 0);
		SIGNAL_CHECK("tab_selected", { { -1 } });
		SIGNAL_CHECK("tab_changed", { { -1 } });

		// Adding first tab will NOT change the current tab. (stays deselected)
		tab_bar->clear_tabs();
		CHECK(tab_bar->get_current_tab() == -1);
		CHECK(tab_bar->get_previous_tab() == -1);

		tab_bar->add_tab("tab0");
		tab_bar->add_tab("tab1");
		tab_bar->add_tab("tab2");
		CHECK(tab_bar->get_tab_count() == 3);
		CHECK(tab_bar->get_current_tab() == -1);
		CHECK(tab_bar->get_previous_tab() == -1);
		SIGNAL_CHECK_FALSE("tab_selected");
		SIGNAL_CHECK_FALSE("tab_changed");

		tab_bar->set_current_tab(-1);
		SIGNAL_DISCARD("tab_selected");
		SIGNAL_DISCARD("tab_changed");

		// Disabling while at -1 will select the first available tab.
		tab_bar->set_deselect_enabled(false);
		CHECK_FALSE(tab_bar->get_deselect_enabled());
		CHECK(tab_bar->get_current_tab() == 0);
		CHECK(tab_bar->get_previous_tab() == -1);
		SIGNAL_CHECK("tab_selected", { { 0 } });
		SIGNAL_CHECK("tab_changed", { { 0 } });

		// Cannot set to -1 if disabled.
		ERR_PRINT_OFF;
		tab_bar->set_current_tab(-1);
		CHECK(tab_bar->get_current_tab() == 0);
		CHECK(tab_bar->get_previous_tab() == -1);
		SIGNAL_CHECK_FALSE("tab_selected");
		SIGNAL_CHECK_FALSE("tab_changed");
		ERR_PRINT_ON;

		// Disabling while at -1 skips any disabled or hidden tabs.
		tab_bar->set_deselect_enabled(true);
		tab_bar->set_tab_disabled(0, true);
		tab_bar->set_tab_hidden(1, true);
		tab_bar->set_current_tab(-1);
		SIGNAL_DISCARD("tab_selected");
		SIGNAL_DISCARD("tab_changed");
		tab_bar->set_deselect_enabled(false);
		CHECK(tab_bar->get_current_tab() == 2);
		CHECK(tab_bar->get_previous_tab() == -1);
		SIGNAL_CHECK("tab_selected", { { 2 } });
		SIGNAL_CHECK("tab_changed", { { 2 } });
	}

	SUBCASE("[TabBar] hidden tabs") {
		tab_bar->add_tab("tab0");
		tab_bar->add_tab("tab1");
		tab_bar->add_tab("tab2");
		tab_bar->set_current_tab(1);
		CHECK(tab_bar->get_current_tab() == 1);
		CHECK(tab_bar->get_previous_tab() == 0);
		CHECK_FALSE(tab_bar->is_tab_hidden(1));
		SIGNAL_DISCARD("tab_selected");
		SIGNAL_DISCARD("tab_changed");

		MessageQueue::get_singleton()->flush();
		Vector<Rect2> tab_rects = {
			tab_bar->get_tab_rect(0),
			tab_bar->get_tab_rect(1),
			tab_bar->get_tab_rect(2)
		};

		// Hiding a tab does not affect current tab.
		tab_bar->set_tab_hidden(1, true);
		CHECK(tab_bar->is_tab_hidden(1));
		CHECK(tab_bar->get_current_tab() == 1);
		CHECK(tab_bar->get_previous_tab() == 0);
		SIGNAL_CHECK_FALSE("tab_selected");
		SIGNAL_CHECK_FALSE("tab_changed");

		// The tabs after are moved over.
		MessageQueue::get_singleton()->flush();
		CHECK(tab_bar->get_tab_rect(0) == tab_rects[0]);
		CHECK(tab_bar->get_tab_rect(2) == tab_rects[1]);

		// Unhiding a tab does not affect current tab.
		tab_bar->set_tab_hidden(1, false);
		CHECK_FALSE(tab_bar->is_tab_hidden(1));
		CHECK(tab_bar->get_current_tab() == 1);
		CHECK(tab_bar->get_previous_tab() == 0);
		SIGNAL_CHECK_FALSE("tab_selected");
		SIGNAL_CHECK_FALSE("tab_changed");

		// The tabs are back where they were.
		MessageQueue::get_singleton()->flush();
		CHECK(tab_bar->get_tab_rect(0) == tab_rects[0]);
		CHECK(tab_bar->get_tab_rect(1) == tab_rects[1]);
		CHECK(tab_bar->get_tab_rect(2) == tab_rects[2]);
	}

	SUBCASE("[TabBar] disabled tabs") {
		tab_bar->add_tab("tab0");
		tab_bar->add_tab("tab1");
		tab_bar->add_tab("tab2");
		CHECK_FALSE(tab_bar->is_tab_disabled(1));
		tab_bar->set_current_tab(1);
		CHECK(tab_bar->get_current_tab() == 1);
		CHECK(tab_bar->get_previous_tab() == 0);
		CHECK_FALSE(tab_bar->is_tab_hidden(1));
		SIGNAL_DISCARD("tab_selected");
		SIGNAL_DISCARD("tab_changed");

		// Disabling a tab does not affect current tab.
		tab_bar->set_tab_disabled(1, true);
		CHECK(tab_bar->is_tab_disabled(1));
		CHECK(tab_bar->get_current_tab() == 1);
		CHECK(tab_bar->get_previous_tab() == 0);
		SIGNAL_CHECK_FALSE("tab_selected");
		SIGNAL_CHECK_FALSE("tab_changed");

		// Enabling a tab does not affect current tab.
		tab_bar->set_tab_disabled(1, false);
		CHECK_FALSE(tab_bar->is_tab_disabled(1));
		CHECK(tab_bar->get_current_tab() == 1);
		CHECK(tab_bar->get_previous_tab() == 0);
		SIGNAL_CHECK_FALSE("tab_selected");
		SIGNAL_CHECK_FALSE("tab_changed");
	}

	SUBCASE("[TabBar] select next available") {
		tab_bar->add_tab("tab0");
		tab_bar->add_tab("tab1");
		tab_bar->add_tab("tab2");
		tab_bar->add_tab("tab3");
		tab_bar->add_tab("tab4");
		tab_bar->set_tab_disabled(2, true);
		tab_bar->set_tab_hidden(3, true);
		tab_bar->set_current_tab(0);
		CHECK(tab_bar->get_current_tab() == 0);
		CHECK(tab_bar->get_previous_tab() == 0);
		SIGNAL_DISCARD("tab_selected");
		SIGNAL_DISCARD("tab_changed");

		// Selects the next tab.
		CHECK(tab_bar->select_next_available());
		CHECK(tab_bar->get_current_tab() == 1);
		CHECK(tab_bar->get_previous_tab() == 0);
		SIGNAL_CHECK("tab_selected", { { 1 } });
		SIGNAL_CHECK("tab_changed", { { 1 } });

		// Skips over disabled and hidden tabs.
		CHECK(tab_bar->select_next_available());
		CHECK(tab_bar->get_current_tab() == 4);
		CHECK(tab_bar->get_previous_tab() == 1);
		SIGNAL_CHECK("tab_selected", { { 4 } });
		SIGNAL_CHECK("tab_changed", { { 4 } });

		// Does not wrap around.
		CHECK_FALSE(tab_bar->select_next_available());
		CHECK(tab_bar->get_current_tab() == 4);
		CHECK(tab_bar->get_previous_tab() == 1);
		SIGNAL_CHECK_FALSE("tab_selected");
		SIGNAL_CHECK_FALSE("tab_changed");

		// Fails if there is only one valid tab.
		tab_bar->remove_tab(0);
		tab_bar->remove_tab(3);
		CHECK(tab_bar->get_current_tab() == 0);
		CHECK(tab_bar->get_previous_tab() == 0);
		SIGNAL_DISCARD("tab_selected");
		SIGNAL_DISCARD("tab_changed");

		CHECK_FALSE(tab_bar->select_next_available());
		CHECK(tab_bar->get_current_tab() == 0);
		CHECK(tab_bar->get_previous_tab() == 0);
		SIGNAL_CHECK_FALSE("tab_selected");
		SIGNAL_CHECK_FALSE("tab_changed");

		// Fails if there are no valid tabs.
		tab_bar->remove_tab(0);
		CHECK(tab_bar->get_current_tab() == -1);
		CHECK(tab_bar->get_previous_tab() == 0);
		SIGNAL_DISCARD("tab_selected");
		SIGNAL_DISCARD("tab_changed");

		CHECK_FALSE(tab_bar->select_next_available());
		CHECK(tab_bar->get_current_tab() == -1);
		CHECK(tab_bar->get_previous_tab() == 0);
		SIGNAL_CHECK_FALSE("tab_selected");
		SIGNAL_CHECK_FALSE("tab_changed");

		// Fails if there are no tabs.
		tab_bar->clear_tabs();
		CHECK(tab_bar->get_current_tab() == -1);
		CHECK(tab_bar->get_previous_tab() == -1);
		SIGNAL_DISCARD("tab_selected");
		SIGNAL_DISCARD("tab_changed");

		CHECK_FALSE(tab_bar->select_next_available());
		CHECK(tab_bar->get_current_tab() == -1);
		CHECK(tab_bar->get_previous_tab() == -1);
		SIGNAL_CHECK_FALSE("tab_selected");
		SIGNAL_CHECK_FALSE("tab_changed");
	}

	SUBCASE("[TabBar] select previous available") {
		tab_bar->add_tab("tab0");
		tab_bar->add_tab("tab1");
		tab_bar->add_tab("tab2");
		tab_bar->add_tab("tab3");
		tab_bar->add_tab("tab4");
		tab_bar->set_tab_disabled(1, true);
		tab_bar->set_tab_hidden(2, true);
		tab_bar->set_current_tab(4);
		CHECK(tab_bar->get_current_tab() == 4);
		CHECK(tab_bar->get_previous_tab() == 0);
		SIGNAL_DISCARD("tab_selected");
		SIGNAL_DISCARD("tab_changed");

		// Selects the previous tab.
		CHECK(tab_bar->select_previous_available());
		CHECK(tab_bar->get_current_tab() == 3);
		CHECK(tab_bar->get_previous_tab() == 4);
		SIGNAL_CHECK("tab_selected", { { 3 } });
		SIGNAL_CHECK("tab_changed", { { 3 } });

		// Skips over disabled and hidden tabs.
		CHECK(tab_bar->select_previous_available());
		CHECK(tab_bar->get_current_tab() == 0);
		CHECK(tab_bar->get_previous_tab() == 3);
		SIGNAL_CHECK("tab_selected", { { 0 } });
		SIGNAL_CHECK("tab_changed", { { 0 } });

		// Does not wrap around.
		CHECK_FALSE(tab_bar->select_previous_available());
		CHECK(tab_bar->get_current_tab() == 0);
		CHECK(tab_bar->get_previous_tab() == 3);
		SIGNAL_CHECK_FALSE("tab_selected");
		SIGNAL_CHECK_FALSE("tab_changed");

		// Fails if there is only one valid tab.
		tab_bar->remove_tab(4);
		tab_bar->remove_tab(3);
		CHECK(tab_bar->get_current_tab() == 0);
		CHECK(tab_bar->get_previous_tab() == 2);
		SIGNAL_DISCARD("tab_selected");
		SIGNAL_DISCARD("tab_changed");

		CHECK_FALSE(tab_bar->select_previous_available());
		CHECK(tab_bar->get_current_tab() == 0);
		CHECK(tab_bar->get_previous_tab() == 2);
		SIGNAL_CHECK_FALSE("tab_selected");
		SIGNAL_CHECK_FALSE("tab_changed");

		// Fails if there are no valid tabs.
		tab_bar->remove_tab(0);
		CHECK(tab_bar->get_current_tab() == -1);
		CHECK(tab_bar->get_previous_tab() == 1);
		SIGNAL_DISCARD("tab_selected");
		SIGNAL_DISCARD("tab_changed");

		CHECK_FALSE(tab_bar->select_previous_available());
		CHECK(tab_bar->get_current_tab() == -1);
		CHECK(tab_bar->get_previous_tab() == 1);
		SIGNAL_CHECK_FALSE("tab_selected");
		SIGNAL_CHECK_FALSE("tab_changed");

		// Fails if there are no tabs.
		tab_bar->clear_tabs();
		CHECK(tab_bar->get_current_tab() == -1);
		CHECK(tab_bar->get_previous_tab() == -1);
		SIGNAL_DISCARD("tab_selected");
		SIGNAL_DISCARD("tab_changed");

		CHECK_FALSE(tab_bar->select_previous_available());
		CHECK(tab_bar->get_current_tab() == -1);
		CHECK(tab_bar->get_previous_tab() == -1);
		SIGNAL_CHECK_FALSE("tab_selected");
		SIGNAL_CHECK_FALSE("tab_changed");
	}

	SIGNAL_UNWATCH(tab_bar, "tab_selected");
	SIGNAL_UNWATCH(tab_bar, "tab_changed");

	memdelete(tab_bar);
}

TEST_CASE("[SceneTree][TabBar] initialization") {
	TabBar *tab_bar = memnew(TabBar);

	SIGNAL_WATCH(tab_bar, "tab_selected");
	SIGNAL_WATCH(tab_bar, "tab_changed");

	SUBCASE("[TabBar] current tab can be set before tabs are set") {
		// This queues the current tab to update on when tabs are set.
		tab_bar->set_current_tab(1);
		CHECK(tab_bar->get_current_tab() == -1);
		CHECK(tab_bar->get_previous_tab() == -1);
		SIGNAL_CHECK_FALSE("tab_selected");
		SIGNAL_CHECK_FALSE("tab_changed");

		tab_bar->set_tab_count(2);
		CHECK(tab_bar->get_tab_count() == 2);
		CHECK(tab_bar->get_current_tab() == 1);
		CHECK(tab_bar->get_previous_tab() == 0);
		SIGNAL_CHECK("tab_selected", { { 1 } });
		SIGNAL_CHECK("tab_changed", { { 1 } });

		// Does not work again.
		ERR_PRINT_OFF;
		tab_bar->set_current_tab(2);
		CHECK(tab_bar->get_tab_count() == 2);
		CHECK(tab_bar->get_current_tab() == 1);
		CHECK(tab_bar->get_previous_tab() == 0);
		SIGNAL_CHECK_FALSE("tab_selected");
		SIGNAL_CHECK_FALSE("tab_changed");

		tab_bar->set_tab_count(3);
		CHECK(tab_bar->get_tab_count() == 3);
		CHECK(tab_bar->get_current_tab() == 1);
		CHECK(tab_bar->get_previous_tab() == 0);
		SIGNAL_CHECK_FALSE("tab_selected");
		SIGNAL_CHECK_FALSE("tab_changed");
		ERR_PRINT_ON;
	}

	SUBCASE("[TabBar] setting tabs works normally if no current tab was set") {
		CHECK(tab_bar->get_current_tab() == -1);
		CHECK(tab_bar->get_previous_tab() == -1);

		tab_bar->set_tab_count(2);
		CHECK(tab_bar->get_tab_count() == 2);
		CHECK(tab_bar->get_current_tab() == 0);
		CHECK(tab_bar->get_previous_tab() == -1);
		SIGNAL_CHECK_FALSE("tab_selected");
		SIGNAL_CHECK_FALSE("tab_changed");
	}

	SUBCASE("[TabBar] cannot set current tab to an invalid value before tabs are set") {
		tab_bar->set_current_tab(100);
		CHECK(tab_bar->get_current_tab() == -1);
		CHECK(tab_bar->get_previous_tab() == -1);
		SIGNAL_CHECK_FALSE("tab_selected");
		SIGNAL_CHECK_FALSE("tab_changed");

		// This will print an error message as if `set_current_tab` was called after.
		ERR_PRINT_OFF;
		tab_bar->set_tab_count(2);
		CHECK(tab_bar->get_tab_count() == 2);
		CHECK(tab_bar->get_current_tab() == 0);
		CHECK(tab_bar->get_previous_tab() == -1);
		SIGNAL_CHECK_FALSE("tab_selected");
		SIGNAL_CHECK_FALSE("tab_changed");
		ERR_PRINT_ON;
	}

	SUBCASE("[TabBar] setting the current tab before tabs only works when out of tree") {
		tab_bar->set_current_tab(1);
		CHECK(tab_bar->get_current_tab() == -1);
		CHECK(tab_bar->get_previous_tab() == -1);
		SIGNAL_CHECK_FALSE("tab_selected");
		SIGNAL_CHECK_FALSE("tab_changed");

		SceneTree::get_singleton()->get_root()->add_child(tab_bar);
		MessageQueue::get_singleton()->flush();
		CHECK(tab_bar->get_tab_count() == 0);
		CHECK(tab_bar->get_current_tab() == -1);
		CHECK(tab_bar->get_previous_tab() == -1);
		SIGNAL_CHECK_FALSE("tab_selected");
		SIGNAL_CHECK_FALSE("tab_changed");

		// Works normally.
		tab_bar->set_tab_count(2);
		CHECK(tab_bar->get_tab_count() == 2);
		CHECK(tab_bar->get_current_tab() == 0);
		CHECK(tab_bar->get_previous_tab() == -1);
		SIGNAL_CHECK_FALSE("tab_selected");
		SIGNAL_CHECK_FALSE("tab_changed");
	}

	SIGNAL_UNWATCH(tab_bar, "tab_selected");
	SIGNAL_UNWATCH(tab_bar, "tab_changed");

	memdelete(tab_bar);
}

TEST_CASE("[SceneTree][TabBar] layout and offset") {
	TabBar *tab_bar = memnew(TabBar);
	SceneTree::get_singleton()->get_root()->add_child(tab_bar);

	tab_bar->set_clip_tabs(false);
	tab_bar->add_tab("tab0");
	tab_bar->add_tab("tab1 ");
	tab_bar->add_tab("tab2    ");
	MessageQueue::get_singleton()->flush();
	Size2 all_tabs_size = tab_bar->get_size();

	Vector<Rect2> tab_rects = {
		tab_bar->get_tab_rect(0),
		tab_bar->get_tab_rect(1),
		tab_bar->get_tab_rect(2)
	};

	SUBCASE("[TabBar] tabs are arranged next to each other") {
		// Horizontal positions are next to each other.
		CHECK(tab_rects[0].position.x == 0);
		CHECK(tab_rects[1].position.x == tab_rects[0].size.x);
		CHECK(tab_rects[2].position.x == tab_rects[1].position.x + tab_rects[1].size.x);

		// Fills the entire width.
		CHECK(tab_rects[2].position.x + tab_rects[2].size.x == all_tabs_size.x);

		// Horizontal sizes are positive.
		CHECK(tab_rects[0].size.x > 0);
		CHECK(tab_rects[1].size.x > 0);
		CHECK(tab_rects[2].size.x > 0);

		// Vertical positions are at 0.
		CHECK(tab_rects[0].position.y == 0);
		CHECK(tab_rects[1].position.y == 0);
		CHECK(tab_rects[2].position.y == 0);

		// Vertical sizes are the same.
		CHECK(tab_rects[0].size.y == tab_rects[1].size.y);
		CHECK(tab_rects[1].size.y == tab_rects[2].size.y);
	}

	SUBCASE("[TabBar] tab alignment") {
		// Add extra space so the alignment can be seen.
		tab_bar->set_size(Size2(all_tabs_size.x + 100, all_tabs_size.y));

		// Left alignment.
		tab_bar->set_tab_alignment(TabBar::ALIGNMENT_LEFT);
		MessageQueue::get_singleton()->flush();
		tab_rects = {
			tab_bar->get_tab_rect(0),
			tab_bar->get_tab_rect(1),
			tab_bar->get_tab_rect(2)
		};
		CHECK(tab_bar->get_tab_alignment() == TabBar::ALIGNMENT_LEFT);
		CHECK(tab_rects[0].position.x == 0);
		CHECK(tab_rects[1].position.x == tab_rects[0].size.x);
		CHECK(tab_rects[2].position.x == tab_rects[1].position.x + tab_rects[1].size.x);

		// Right alignment.
		tab_bar->set_tab_alignment(TabBar::ALIGNMENT_RIGHT);
		MessageQueue::get_singleton()->flush();
		tab_rects = {
			tab_bar->get_tab_rect(0),
			tab_bar->get_tab_rect(1),
			tab_bar->get_tab_rect(2)
		};
		CHECK(tab_bar->get_tab_alignment() == TabBar::ALIGNMENT_RIGHT);
		CHECK(tab_rects[2].position.x == tab_bar->get_size().x - tab_rects[2].size.x);
		CHECK(tab_rects[1].position.x == tab_rects[2].position.x - tab_rects[1].size.x);
		CHECK(tab_rects[0].position.x == tab_rects[1].position.x - tab_rects[0].size.x);

		// Center alignment.
		tab_bar->set_tab_alignment(TabBar::ALIGNMENT_CENTER);
		MessageQueue::get_singleton()->flush();
		tab_rects = {
			tab_bar->get_tab_rect(0),
			tab_bar->get_tab_rect(1),
			tab_bar->get_tab_rect(2)
		};
		CHECK(tab_bar->get_tab_alignment() == TabBar::ALIGNMENT_CENTER);
		float center_pos = tab_bar->get_size().x / 2;
		CHECK(tab_rects[0].position.x == center_pos - all_tabs_size.x / 2);
		CHECK(tab_rects[1].position.x == tab_rects[0].position.x + tab_rects[0].size.x);
		CHECK(tab_rects[2].position.x == tab_rects[1].position.x + tab_rects[1].size.x);
	}

	SUBCASE("[TabBar] clip tabs") {
		// Clip tabs disabled means all tabs are visible and the minimum size holds all of them.
		tab_bar->set_clip_tabs(false);
		CHECK_FALSE(tab_bar->get_clip_tabs());
		MessageQueue::get_singleton()->flush();
		CHECK(tab_bar->get_tab_offset() == 0);
		CHECK(tab_bar->get_minimum_size() == tab_bar->get_size());
		CHECK(tab_bar->get_size().x == tab_rects[0].size.x + tab_rects[1].size.x + tab_rects[2].size.x);
		CHECK(tab_bar->get_size().y == MAX(tab_rects[0].size.y, MAX(tab_rects[1].size.y, tab_rects[2].size.y)));

		tab_bar->set_clip_tabs(true);
		CHECK(tab_bar->get_clip_tabs());
		MessageQueue::get_singleton()->flush();
		CHECK(tab_bar->get_tab_offset() == 0);

		// Horizontal size and minimum size get set to the widest tab plus arrow icons.
		const float offset_button_size = tab_bar->get_theme_icon("decrement_icon")->get_width() + tab_bar->get_theme_icon("increment_icon")->get_width();
		CHECK(tab_bar->get_minimum_size().x == offset_button_size + MAX(tab_rects[0].size.x, MAX(tab_rects[1].size.x, tab_rects[2].size.x)));
		CHECK(tab_bar->get_minimum_size().y == all_tabs_size.y);
		CHECK(tab_bar->get_size().x == offset_button_size + MAX(tab_rects[0].size.x, MAX(tab_rects[1].size.x, tab_rects[2].size.x)));
		CHECK(tab_bar->get_size().y == all_tabs_size.y);
	}

	SUBCASE("[TabBar] ensure tab visible") {
		tab_bar->set_scroll_to_selected(false);
		tab_bar->set_clip_tabs(true);

		// Resize tab bar to only be able to fit 2 tabs.
		const float offset_button_size = tab_bar->get_theme_icon("decrement_icon")->get_width() + tab_bar->get_theme_icon("increment_icon")->get_width();
		tab_bar->set_size(Size2(tab_rects[2].size.x + tab_rects[1].size.x + offset_button_size, all_tabs_size.y));
		MessageQueue::get_singleton()->flush();
		CHECK(tab_bar->get_tab_offset() == 0);
		CHECK(tab_bar->get_offset_buttons_visible());

		// Scroll right to a tab that is not visible.
		tab_bar->ensure_tab_visible(2);
		CHECK(tab_bar->get_tab_offset() == 1);
		CHECK(tab_bar->get_tab_rect(1).position.x == 0);
		CHECK(tab_bar->get_tab_rect(2).position.x == tab_rects[1].size.x);

		tab_bar->set_tab_offset(2);
		CHECK(tab_bar->get_tab_offset() == 2);
		CHECK(tab_bar->get_tab_rect(2).position.x == 0);

		// Scroll left to a previous tab.
		tab_bar->ensure_tab_visible(1);
		CHECK(tab_bar->get_tab_offset() == 1);
		CHECK(tab_bar->get_tab_rect(1).position.x == 0);
		CHECK(tab_bar->get_tab_rect(2).position.x == tab_rects[1].size.x);

		// Will not scroll if the tab is already visible.
		tab_bar->ensure_tab_visible(2);
		CHECK(tab_bar->get_tab_offset() == 1);
		CHECK(tab_bar->get_tab_rect(1).position.x == 0);
		CHECK(tab_bar->get_tab_rect(2).position.x == tab_rects[1].size.x);
	}

	SUBCASE("[TabBar] tab separation") {
		tab_bar->set_tab_alignment(TabBar::ALIGNMENT_LEFT);
		tab_bar->set_clip_tabs(false);

		// Set separation to 0 first.
		tab_bar->add_theme_constant_override("tab_separation", 0);
		MessageQueue::get_singleton()->flush();

		tab_rects = {
			tab_bar->get_tab_rect(0),
			tab_bar->get_tab_rect(1),
			tab_bar->get_tab_rect(2)
		};

		// Verify adjacency.
		CHECK(tab_rects[1].position.x == tab_rects[0].position.x + tab_rects[0].size.x);
		CHECK(tab_rects[2].position.x == tab_rects[1].position.x + tab_rects[1].size.x);

		// Apply separation.
		int separation = 20;
		tab_bar->add_theme_constant_override("tab_separation", separation);
		MessageQueue::get_singleton()->flush();

		Vector<Rect2> rects_sep = {
			tab_bar->get_tab_rect(0),
			tab_bar->get_tab_rect(1),
			tab_bar->get_tab_rect(2)
		};

		// Tab 0 should not move.
		CHECK(rects_sep[0].position.x == tab_rects[0].position.x);

		// Tab 1 should move by 1 * separation.
		CHECK(rects_sep[1].position.x == tab_rects[1].position.x + separation);

		// Tab 2 should move by 2 * separation.
		CHECK(rects_sep[2].position.x == tab_rects[2].position.x + (separation * 2));

		// Verify minimum size increased by total separation (separation * (tab_count - 1)).
		Size2 min_size_0 = Size2(tab_rects[0].size.x + tab_rects[1].size.x + tab_rects[2].size.x, tab_rects[0].size.y);
		Size2 expected_min_size = min_size_0;
		expected_min_size.x += separation * 2;

		CHECK(tab_bar->get_minimum_size().x == expected_min_size.x);
	}

	SUBCASE("[TabBar] hidden tabs remove separation") {
		// Setup: Left alignment, no clipping, explicit separation.
		tab_bar->set_tab_alignment(TabBar::ALIGNMENT_LEFT);
		tab_bar->set_clip_tabs(false);
		int separation = 20;
		tab_bar->add_theme_constant_override("tab_separation", separation);

		tab_bar->clear_tabs();
		tab_bar->add_tab("A");
		tab_bar->add_tab("B");
		tab_bar->add_tab("C");
		MessageQueue::get_singleton()->flush();

		// All tabs visible. Layout: [A] - 20px - [B] - 20px - [C].
		tab_rects = {
			tab_bar->get_tab_rect(0),
			tab_bar->get_tab_rect(1),
			tab_bar->get_tab_rect(2)
		};

		CHECK(tab_rects[1].position.x == tab_rects[0].position.x + tab_rects[0].size.x + separation);
		CHECK(tab_rects[2].position.x == tab_rects[1].position.x + tab_rects[1].size.x + separation);

		// Hide middle tab. Layout: [A] - 20px - [C].
		// The separation belonging to B should not be present.
		tab_bar->set_tab_hidden(1, true);
		MessageQueue::get_singleton()->flush();

		tab_rects = {
			tab_bar->get_tab_rect(0),
			tab_bar->get_tab_rect(1),
			tab_bar->get_tab_rect(2)
		};

		// Position of C should now be right after A + 1 separation.
		float expected_c_pos = tab_rects[0].position.x + tab_rects[0].size.x + separation;
		CHECK(tab_rects[2].position.x == expected_c_pos);
	}

	SUBCASE("[TabBar] sizing modes") {
		tab_bar->clear_tabs();
		tab_bar->set_tab_alignment(TabBar::ALIGNMENT_LEFT);
		tab_bar->set_clip_tabs(false);

		// Ensure we have a defined width that is larger than the content.
		tab_bar->set_size(Size2(1000, 100));

		tab_bar->add_tab("S"); // Small
		tab_bar->add_tab("Medium"); // Medium
		tab_bar->add_tab("Large Text"); // Large
		MessageQueue::get_singleton()->flush();

		// Get baseline sizes (Fit Content).
		float width_s = tab_bar->get_tab_rect(0).size.x;
		float width_m = tab_bar->get_tab_rect(1).size.x;
		float width_l = tab_bar->get_tab_rect(2).size.x;

		// Fit Content Sizing: Tabs adjust width to fit their content.
		CHECK(tab_bar->get_tab_sizing() == TabBar::TAB_SIZING_FIT_CONTENT);
		CHECK(width_s < width_m);
		CHECK(width_m < width_l);

		// Uniform Sizing: All tabs should become the size of the largest tab ("Large Text").
		tab_bar->set_tab_sizing(TabBar::TAB_SIZING_UNIFORM);
		MessageQueue::get_singleton()->flush();

		CHECK(tab_bar->get_tab_rect(0).size.x == width_l);
		CHECK(tab_bar->get_tab_rect(1).size.x == width_l);
		CHECK(tab_bar->get_tab_rect(2).size.x == width_l);

		// Justify Sizing: Tabs fill the space, but keep relative size differences based on content.
		tab_bar->set_tab_sizing(TabBar::TAB_SIZING_JUSTIFY);
		MessageQueue::get_singleton()->flush();

		float justify_width_0 = tab_bar->get_tab_rect(0).size.x;
		float justify_width_1 = tab_bar->get_tab_rect(1).size.x;
		float justify_width_2 = tab_bar->get_tab_rect(2).size.x;

		// Order implies size difference is preserved.
		CHECK(justify_width_0 < justify_width_1);
		CHECK(justify_width_1 < justify_width_2);

		// Even the smallest tab should be wider than its natural size because we have 1000px to fill.
		CHECK(justify_width_0 > width_s);

		// Expand Sizing: Tabs should be equal width, dividing the 1000px available space (minus margins/separations).
		tab_bar->set_tab_sizing(TabBar::TAB_SIZING_EXPAND);
		MessageQueue::get_singleton()->flush();

		float expand_width_0 = tab_bar->get_tab_rect(0).size.x;
		float expand_width_1 = tab_bar->get_tab_rect(1).size.x;
		float expand_width_2 = tab_bar->get_tab_rect(2).size.x;

		// Allow 1px difference for rounding errors in distribution.
		CHECK(Math::abs(expand_width_0 - expand_width_1) <= 1.0f);
		CHECK(Math::abs(expand_width_1 - expand_width_2) <= 1.0f);

		// They should be significantly larger than the "Large" natural width because they are filling 1000px.
		CHECK(expand_width_0 > width_l);

		// Verify they fill the width (approximate check).
		float total_width = expand_width_0 + expand_width_1 + expand_width_2;
		CHECK(total_width > 900.0f); // Sanity check against container size.
	}

	SUBCASE("[TabBar] separation reduces expand sizing width") {
		tab_bar->clear_tabs();
		tab_bar->set_size(Size2(200, 50));
		tab_bar->set_tab_sizing(TabBar::TAB_SIZING_EXPAND);

		// Add 2 tabs.
		tab_bar->add_tab("1");
		tab_bar->add_tab("2");
		MessageQueue::get_singleton()->flush();

		// Available space = 200. Each tab = 100.
		tab_bar->add_theme_constant_override("tab_separation", 0);
		MessageQueue::get_singleton()->flush();
		float width_sep_0 = tab_bar->get_tab_rect(0).size.x;
		CHECK(Math::is_equal_approx(width_sep_0, 100.0f));

		// Separation = 50. Available space = 200 - 50 = 150. Each tab = 75.
		int separation = 50;
		tab_bar->add_theme_constant_override("tab_separation", separation);
		MessageQueue::get_singleton()->flush();

		float width_sep_50 = tab_bar->get_tab_rect(0).size.x;
		CHECK(Math::is_equal_approx(width_sep_50, 75.0f));

		// Ensure layout is contiguous: [Tab 75] [Sep 50] [Tab 75] = 200 total.
		Rect2 r1 = tab_bar->get_tab_rect(0);
		Rect2 r2 = tab_bar->get_tab_rect(1);
		CHECK(r2.position.x == r1.size.x + separation);
	}

	SUBCASE("[TabBar] max width caps sizing modes") {
		tab_bar->clear_tabs();
		tab_bar->set_size(Size2(1000, 50));

		// Create a tab with very long text that naturally exceeds the max width.
		tab_bar->add_tab("This is a very long tab title that should exceed 200px");
		MessageQueue::get_singleton()->flush();

		float standard_tab_width = tab_bar->get_tab_rect(0).size.x;

		// Set constraint.
		int max_w = 200;
		tab_bar->set_max_tab_width(max_w);
		MessageQueue::get_singleton()->flush();

		// Verify the unconstrained width is greater than max width, then verify the current width is constrained.
		CHECK(standard_tab_width > max_w);

		// Fit Content: Width should be constrained to 200px.
		CHECK(tab_bar->get_tab_rect(0).size.x == max_w);

		// Uniform: Width should be constrained to 200px.
		tab_bar->set_tab_sizing(TabBar::TAB_SIZING_UNIFORM);
		MessageQueue::get_singleton()->flush();
		CHECK(tab_bar->get_tab_rect(0).size.x == max_w);

		// Justify: Width should be constrained to 200px.
		tab_bar->set_tab_sizing(TabBar::TAB_SIZING_JUSTIFY);
		MessageQueue::get_singleton()->flush();
		CHECK(tab_bar->get_tab_rect(0).size.x == max_w);

		// Expand: Width should be constrained to 200px.
		tab_bar->set_tab_sizing(TabBar::TAB_SIZING_EXPAND);
		MessageQueue::get_singleton()->flush();
		CHECK(tab_bar->get_tab_rect(0).size.x == max_w);
	}

	memdelete(tab_bar);
}

TEST_CASE("[SceneTree][TabBar] Mouse interaction") {
	TabBar *tab_bar = memnew(TabBar);
	SceneTree::get_singleton()->get_root()->add_child(tab_bar);

	tab_bar->set_clip_tabs(false);
	tab_bar->add_tab("tab0");
	tab_bar->add_tab("tab1 ");
	tab_bar->add_tab("tab2    ");
	MessageQueue::get_singleton()->flush();

	Vector<Rect2> tab_rects = {
		tab_bar->get_tab_rect(0),
		tab_bar->get_tab_rect(1),
		tab_bar->get_tab_rect(2)
	};

	SIGNAL_WATCH(tab_bar, "active_tab_rearranged");
	SIGNAL_WATCH(tab_bar, "tab_button_pressed");
	SIGNAL_WATCH(tab_bar, "tab_changed");
	SIGNAL_WATCH(tab_bar, "tab_clicked");
	SIGNAL_WATCH(tab_bar, "tab_close_pressed");
	SIGNAL_WATCH(tab_bar, "tab_hovered");
	SIGNAL_WATCH(tab_bar, "tab_selected");

	SUBCASE("[TabBar] Hover over tabs") {
		// Default is -1.
		CHECK(tab_bar->get_hovered_tab() == -1);

		// Hover over the first tab.
		SEND_GUI_MOUSE_MOTION_EVENT(tab_rects[0].position, MouseButtonMask::NONE, Key::NONE);
		SIGNAL_CHECK("tab_hovered", { { 0 } });
		CHECK(tab_bar->get_hovered_tab() == 0);

		// Hover over same tab won't send signal again.
		SEND_GUI_MOUSE_MOTION_EVENT(tab_rects[0].position + Point2(2, 2), MouseButtonMask::NONE, Key::NONE);
		SIGNAL_CHECK_FALSE("tab_hovered");
		CHECK(tab_bar->get_hovered_tab() == 0);

		// Hover over different tab.
		SEND_GUI_MOUSE_MOTION_EVENT(tab_rects[1].position, MouseButtonMask::NONE, Key::NONE);
		SIGNAL_CHECK("tab_hovered", { { 1 } });
		CHECK(tab_bar->get_hovered_tab() == 1);

		// Exit area.
		SEND_GUI_MOUSE_MOTION_EVENT(Point2(-1, -1), MouseButtonMask::NONE, Key::NONE);
		SIGNAL_CHECK_FALSE("tab_hovered");
		CHECK(tab_bar->get_hovered_tab() == -1);
	}

	SUBCASE("[TabBar] Click to change current") {
		CHECK(tab_bar->get_current_tab() == 0);
		CHECK(tab_bar->get_previous_tab() == -1);
		SIGNAL_DISCARD("tab_selected");
		SIGNAL_DISCARD("tab_changed");

		// Click to set the current tab.
		SEND_GUI_MOUSE_BUTTON_EVENT(tab_rects[1].position, MouseButton::LEFT, MouseButtonMask::LEFT, Key::NONE);
		CHECK(tab_bar->get_current_tab() == 1);
		CHECK(tab_bar->get_previous_tab() == 0);
		SIGNAL_CHECK("tab_selected", { { 1 } });
		SIGNAL_CHECK("tab_changed", { { 1 } });
		SIGNAL_CHECK("tab_clicked", { { 1 } });

		// Click on the same tab.
		SEND_GUI_MOUSE_BUTTON_EVENT(tab_rects[1].position, MouseButton::LEFT, MouseButtonMask::LEFT, Key::NONE);
		CHECK(tab_bar->get_current_tab() == 1);
		CHECK(tab_bar->get_previous_tab() == 1);
		SIGNAL_CHECK("tab_selected", { { 1 } });
		SIGNAL_CHECK_FALSE("tab_changed");
		SIGNAL_CHECK("tab_clicked", { { 1 } });
	}

	SUBCASE("[TabBar] Click on close button") {
		const Size2 close_button_size = tab_bar->get_theme_icon("close_icon")->get_size();
		int h_separation = tab_bar->get_theme_constant("h_separation");
		float margin = tab_bar->get_theme_stylebox("tab_hovered_style")->get_margin(SIDE_RIGHT);

		tab_bar->set_tab_close_display_policy(TabBar::CLOSE_BUTTON_SHOW_ALWAYS);
		MessageQueue::get_singleton()->flush();
		tab_rects = { tab_bar->get_tab_rect(0), tab_bar->get_tab_rect(1), tab_bar->get_tab_rect(2) };

		Point2 cb_pos = Size2(tab_rects[0].get_end().x - close_button_size.x - h_separation - margin + 1, tab_rects[0].position.y + (tab_rects[0].size.y - close_button_size.y) / 2 + 1);
		SEND_GUI_MOUSE_MOTION_EVENT(cb_pos, MouseButtonMask::LEFT, Key::NONE);
		SEND_GUI_MOUSE_BUTTON_EVENT(cb_pos, MouseButton::LEFT, MouseButtonMask::LEFT, Key::NONE);
		SEND_GUI_MOUSE_BUTTON_RELEASED_EVENT(cb_pos, MouseButton::LEFT, MouseButtonMask::NONE, Key::NONE);
		SIGNAL_CHECK("tab_close_pressed", { { 0 } });
		SIGNAL_CHECK_FALSE("tab_clicked");
		SIGNAL_CHECK_FALSE("tab_selected");
		SIGNAL_CHECK_FALSE("tab_changed");
		SIGNAL_CHECK_FALSE("tab_button_pressed");
		// It does not remove the tab.
		CHECK(tab_bar->get_tab_count() == 3);
	}

	SUBCASE("[TabBar] Drag and drop internally") {
		// Cannot drag if not enabled.
		CHECK_FALSE(tab_bar->get_drag_to_rearrange_enabled());
		SEND_GUI_MOUSE_BUTTON_EVENT(tab_rects[0].position, MouseButton::LEFT, MouseButtonMask::LEFT, Key::NONE);
		SEND_GUI_MOUSE_MOTION_EVENT(tab_rects[1].position, MouseButtonMask::LEFT, Key::NONE);
		SIGNAL_CHECK("tab_selected", { { 0 } });
		SIGNAL_CHECK_FALSE("tab_changed");
		CHECK_FALSE(tab_bar->get_viewport()->gui_is_dragging());
		SEND_GUI_MOUSE_BUTTON_RELEASED_EVENT(tab_rects[1].position, MouseButton::LEFT, MouseButtonMask::NONE, Key::NONE);
		CHECK_FALSE(tab_bar->get_viewport()->gui_is_dragging());
		CHECK(tab_bar->get_tab_title(0) == "tab0");
		CHECK(tab_bar->get_tab_title(1) == "tab1 ");
		CHECK(tab_bar->get_tab_title(2) == "tab2    ");
		SIGNAL_CHECK_FALSE("active_tab_rearranged");
		SIGNAL_CHECK_FALSE("tab_selected");
		SIGNAL_CHECK_FALSE("tab_changed");

		tab_bar->set_drag_to_rearrange_enabled(true);
		CHECK(tab_bar->get_drag_to_rearrange_enabled());

		// Release over the same tab to not move.
		SEND_GUI_MOUSE_BUTTON_EVENT(tab_rects[0].position, MouseButton::LEFT, MouseButtonMask::LEFT, Key::NONE);
		SEND_GUI_MOUSE_MOTION_EVENT(tab_rects[1].position, MouseButtonMask::LEFT, Key::NONE);
		SEND_GUI_MOUSE_MOTION_EVENT(tab_rects[0].position, MouseButtonMask::LEFT, Key::NONE);
		SIGNAL_CHECK("tab_selected", { { 0 } });
		SIGNAL_CHECK_FALSE("tab_changed");
		CHECK(tab_bar->get_viewport()->gui_is_dragging());
		SEND_GUI_MOUSE_BUTTON_RELEASED_EVENT(tab_rects[0].position, MouseButton::LEFT, MouseButtonMask::NONE, Key::NONE);
		CHECK_FALSE(tab_bar->get_viewport()->gui_is_dragging());
		CHECK(tab_bar->get_tab_title(0) == "tab0");
		CHECK(tab_bar->get_tab_title(1) == "tab1 ");
		CHECK(tab_bar->get_tab_title(2) == "tab2    ");
		SIGNAL_CHECK_FALSE("active_tab_rearranged");
		SIGNAL_CHECK_FALSE("tab_selected");
		SIGNAL_CHECK_FALSE("tab_changed");

		// Move the first tab after the second.
		SEND_GUI_MOUSE_BUTTON_EVENT(tab_rects[0].position, MouseButton::LEFT, MouseButtonMask::LEFT, Key::NONE);
		SEND_GUI_MOUSE_MOTION_EVENT(tab_rects[1].position, MouseButtonMask::LEFT, Key::NONE);
		SIGNAL_CHECK("tab_selected", { { 0 } });
		SIGNAL_CHECK_FALSE("tab_changed");
		CHECK(tab_bar->get_viewport()->gui_is_dragging());
		SEND_GUI_MOUSE_BUTTON_RELEASED_EVENT(tab_rects[1].position + Point2(tab_rects[1].size.x / 2 + 1, 0), MouseButton::LEFT, MouseButtonMask::NONE, Key::NONE);
		CHECK_FALSE(tab_bar->get_viewport()->gui_is_dragging());
		CHECK(tab_bar->get_tab_title(0) == "tab1 ");
		CHECK(tab_bar->get_tab_title(1) == "tab0");
		CHECK(tab_bar->get_tab_title(2) == "tab2    ");
		CHECK(tab_bar->get_current_tab() == 1);
		SIGNAL_CHECK("active_tab_rearranged", { { 1 } });
		SIGNAL_CHECK("tab_selected", { { 1 } });
		SIGNAL_CHECK_FALSE("tab_changed");

		tab_rects = { tab_bar->get_tab_rect(0), tab_bar->get_tab_rect(1), tab_bar->get_tab_rect(2) };

		// Move the last tab to be the first.
		SEND_GUI_MOUSE_BUTTON_EVENT(tab_rects[2].position, MouseButton::LEFT, MouseButtonMask::LEFT, Key::NONE);
		SEND_GUI_MOUSE_MOTION_EVENT(tab_rects[0].position, MouseButtonMask::LEFT, Key::NONE);
		SIGNAL_CHECK("tab_selected", { { 2 } });
		SIGNAL_CHECK("tab_changed", { { 2 } });
		CHECK(tab_bar->get_viewport()->gui_is_dragging());
		SEND_GUI_MOUSE_BUTTON_RELEASED_EVENT(tab_rects[0].position, MouseButton::LEFT, MouseButtonMask::NONE, Key::NONE);
		CHECK_FALSE(tab_bar->get_viewport()->gui_is_dragging());
		CHECK(tab_bar->get_tab_title(0) == "tab2    ");
		CHECK(tab_bar->get_tab_title(1) == "tab1 ");
		CHECK(tab_bar->get_tab_title(2) == "tab0");
		CHECK(tab_bar->get_current_tab() == 0);
		SIGNAL_CHECK("active_tab_rearranged", { { 0 } });
		SIGNAL_CHECK("tab_selected", { { 0 } });
		SIGNAL_CHECK_FALSE("tab_changed");
	}

	SUBCASE("[TabBar] Drag and drop to different TabBar") {
		TabBar *target_tab_bar = memnew(TabBar);
		SceneTree::get_singleton()->get_root()->add_child(target_tab_bar);

		target_tab_bar->set_clip_tabs(false);
		target_tab_bar->add_tab("other_tab0");
		MessageQueue::get_singleton()->flush();
		target_tab_bar->set_position(tab_bar->get_size());
		MessageQueue::get_singleton()->flush();

		Vector<Rect2> target_tab_rects = { target_tab_bar->get_tab_rect(0) };
		tab_bar->set_drag_to_rearrange_enabled(true);
		tab_bar->set_tabs_rearrange_group(1);

		Point2 target_tab_after_first = target_tab_bar->get_position() + target_tab_rects[0].position + Point2(target_tab_rects[0].size.x / 2 + 1, 0);

		// Cannot drag to another TabBar that does not have drag to rearrange enabled.
		CHECK_FALSE(target_tab_bar->get_drag_to_rearrange_enabled());
		SEND_GUI_MOUSE_BUTTON_EVENT(tab_rects[0].position, MouseButton::LEFT, MouseButtonMask::LEFT, Key::NONE);
		SEND_GUI_MOUSE_MOTION_EVENT(tab_rects[0].position + Point2(20, 0), MouseButtonMask::LEFT, Key::NONE);
		SEND_GUI_MOUSE_MOTION_EVENT(target_tab_after_first, MouseButtonMask::LEFT, Key::NONE);
		SIGNAL_CHECK("tab_selected", { { 0 } });
		SIGNAL_CHECK_FALSE("tab_changed");
		CHECK(tab_bar->get_viewport()->gui_is_dragging());
		SEND_GUI_MOUSE_BUTTON_RELEASED_EVENT(target_tab_after_first, MouseButton::LEFT, MouseButtonMask::NONE, Key::NONE);
		CHECK_FALSE(tab_bar->get_viewport()->gui_is_dragging());
		CHECK(tab_bar->get_tab_count() == 3);
		CHECK(target_tab_bar->get_tab_count() == 1);
		SIGNAL_CHECK_FALSE("active_tab_rearranged");
		SIGNAL_CHECK_FALSE("tab_selected");
		SIGNAL_CHECK_FALSE("tab_changed");

		// Cannot drag to another TabBar that has a tabs rearrange group of -1.
		target_tab_bar->set_drag_to_rearrange_enabled(true);
		tab_bar->set_tabs_rearrange_group(-1);
		SEND_GUI_MOUSE_BUTTON_EVENT(tab_rects[0].position, MouseButton::LEFT, MouseButtonMask::LEFT, Key::NONE);
		SEND_GUI_MOUSE_MOTION_EVENT(tab_rects[0].position + Point2(20, 0), MouseButtonMask::LEFT, Key::NONE);
		SEND_GUI_MOUSE_MOTION_EVENT(target_tab_after_first, MouseButtonMask::LEFT, Key::NONE);
		SIGNAL_CHECK("tab_selected", { { 0 } });
		SIGNAL_CHECK_FALSE("tab_changed");
		CHECK(tab_bar->get_viewport()->gui_is_dragging());
		SEND_GUI_MOUSE_BUTTON_RELEASED_EVENT(target_tab_after_first, MouseButton::LEFT, MouseButtonMask::NONE, Key::NONE);
		CHECK_FALSE(tab_bar->get_viewport()->gui_is_dragging());
		CHECK(tab_bar->get_tab_count() == 3);
		CHECK(target_tab_bar->get_tab_count() == 1);
		SIGNAL_CHECK_FALSE("active_tab_rearranged");
		SIGNAL_CHECK_FALSE("tab_selected");
		SIGNAL_CHECK_FALSE("tab_changed");

		// Cannot drag to another TabBar that has a different tabs rearrange group.
		tab_bar->set_tabs_rearrange_group(1);
		target_tab_bar->set_tabs_rearrange_group(2);
		SEND_GUI_MOUSE_BUTTON_EVENT(tab_rects[0].position, MouseButton::LEFT, MouseButtonMask::LEFT, Key::NONE);
		SEND_GUI_MOUSE_MOTION_EVENT(tab_rects[0].position + Point2(20, 0), MouseButtonMask::LEFT, Key::NONE);
		SEND_GUI_MOUSE_MOTION_EVENT(target_tab_after_first, MouseButtonMask::LEFT, Key::NONE);
		SIGNAL_CHECK("tab_selected", { { 0 } });
		SIGNAL_CHECK_FALSE("tab_changed");
		CHECK(tab_bar->get_viewport()->gui_is_dragging());
		SEND_GUI_MOUSE_BUTTON_RELEASED_EVENT(target_tab_after_first, MouseButton::LEFT, MouseButtonMask::NONE, Key::NONE);
		CHECK_FALSE(tab_bar->get_viewport()->gui_is_dragging());
		CHECK(tab_bar->get_tab_count() == 3);
		CHECK(target_tab_bar->get_tab_count() == 1);
		SIGNAL_CHECK_FALSE("active_tab_rearranged");
		SIGNAL_CHECK_FALSE("tab_selected");
		SIGNAL_CHECK_FALSE("tab_changed");

		// Drag to target container.
		target_tab_bar->set_tabs_rearrange_group(1);
		SEND_GUI_MOUSE_BUTTON_EVENT(tab_rects[0].position, MouseButton::LEFT, MouseButtonMask::LEFT, Key::NONE);
		SEND_GUI_MOUSE_MOTION_EVENT(tab_rects[0].position + Point2(20, 0), MouseButtonMask::LEFT, Key::NONE);
		SEND_GUI_MOUSE_MOTION_EVENT(target_tab_after_first, MouseButtonMask::LEFT, Key::NONE);
		SIGNAL_CHECK("tab_selected", { { 0 } });
		SIGNAL_CHECK_FALSE("tab_changed");
		CHECK(tab_bar->get_viewport()->gui_is_dragging());
		SEND_GUI_MOUSE_BUTTON_RELEASED_EVENT(target_tab_after_first, MouseButton::LEFT, MouseButtonMask::NONE, Key::NONE);
		CHECK_FALSE(tab_bar->get_viewport()->gui_is_dragging());
		CHECK(tab_bar->get_tab_count() == 2);
		CHECK(target_tab_bar->get_tab_count() == 2);
		CHECK(tab_bar->get_tab_title(0) == "tab1 ");
		CHECK(tab_bar->get_tab_title(1) == "tab2    ");
		CHECK(target_tab_bar->get_tab_title(0) == "other_tab0");
		CHECK(target_tab_bar->get_tab_title(1) == "tab0");
		CHECK(tab_bar->get_current_tab() == 0);
		CHECK(target_tab_bar->get_current_tab() == 1);
		SIGNAL_CHECK_FALSE("active_tab_rearranged");
		SIGNAL_CHECK_FALSE("tab_selected"); // Does not send since tab was removed.
		SIGNAL_CHECK("tab_changed", { { 0 } });

		Point2 target_tab = target_tab_bar->get_position();

		// Drag to target container at first index.
		target_tab_bar->set_tabs_rearrange_group(1);
		SEND_GUI_MOUSE_BUTTON_EVENT(tab_rects[0].position, MouseButton::LEFT, MouseButtonMask::LEFT, Key::NONE);
		SEND_GUI_MOUSE_MOTION_EVENT(tab_rects[0].position + Point2(20, 0), MouseButtonMask::LEFT, Key::NONE);
		SEND_GUI_MOUSE_MOTION_EVENT(target_tab, MouseButtonMask::LEFT, Key::NONE);
		SIGNAL_CHECK("tab_selected", { { 0 } });
		SIGNAL_CHECK_FALSE("tab_changed");
		CHECK(tab_bar->get_viewport()->gui_is_dragging());
		SEND_GUI_MOUSE_BUTTON_RELEASED_EVENT(target_tab, MouseButton::LEFT, MouseButtonMask::NONE, Key::NONE);
		CHECK_FALSE(tab_bar->get_viewport()->gui_is_dragging());
		CHECK(tab_bar->get_tab_count() == 1);
		CHECK(target_tab_bar->get_tab_count() == 3);
		CHECK(tab_bar->get_tab_title(0) == "tab2    ");
		CHECK(target_tab_bar->get_tab_title(0) == "tab1 ");
		CHECK(target_tab_bar->get_tab_title(1) == "other_tab0");
		CHECK(target_tab_bar->get_tab_title(2) == "tab0");
		CHECK(tab_bar->get_current_tab() == 0);
		CHECK(target_tab_bar->get_current_tab() == 0);
		SIGNAL_CHECK_FALSE("active_tab_rearranged");
		SIGNAL_CHECK_FALSE("tab_selected"); // Does not send since tab was removed.
		SIGNAL_CHECK("tab_changed", { { 0 } });

		memdelete(target_tab_bar);
	}

	SIGNAL_UNWATCH(tab_bar, "active_tab_rearranged");
	SIGNAL_UNWATCH(tab_bar, "tab_button_pressed");
	SIGNAL_UNWATCH(tab_bar, "tab_changed");
	SIGNAL_UNWATCH(tab_bar, "tab_clicked");
	SIGNAL_UNWATCH(tab_bar, "tab_close_pressed");
	SIGNAL_UNWATCH(tab_bar, "tab_hovered");
	SIGNAL_UNWATCH(tab_bar, "tab_selected");

	memdelete(tab_bar);
}

// FIXME: Add tests for keyboard navigation and other methods.

} // namespace TestTabBar
