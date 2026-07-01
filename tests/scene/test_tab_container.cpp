/**************************************************************************/
/*  test_tab_container.cpp                                                */
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

TEST_FORCE_LINK(test_tab_container)

#ifndef ADVANCED_GUI_DISABLED

#include "scene/gui/box_container.h"
#include "scene/gui/button.h"
#include "scene/gui/popup_menu.h"
#include "scene/gui/tab_container.h"
#include "scene/main/scene_tree.h"
#include "scene/main/window.h"
#include "tests/display_server_mock.h"
#include "tests/signal_watcher.h"

namespace TestTabContainer {

TEST_CASE("[SceneTree][TabContainer] tab operations") {
	TabContainer *tab_container = memnew(TabContainer);
	SceneTree::get_singleton()->get_root()->add_child(tab_container);
	MessageQueue::get_singleton()->flush();
	SIGNAL_WATCH(tab_container, "tab_selected");
	SIGNAL_WATCH(tab_container, "tab_changed");

	Control *tab0 = memnew(Control);
	tab0->set_name("tab0");
	Control *tab1 = memnew(Control);
	tab1->set_name("tab1");
	Control *tab2 = memnew(Control);
	tab2->set_name("tab2");

	SUBCASE("[TabContainer] add tabs by adding children") {
		CHECK(tab_container->get_tab_count() == 0);
		CHECK(tab_container->get_current_tab() == -1);
		CHECK(tab_container->get_previous_tab() == -1);

		// Add first tab child.
		tab_container->add_child(tab0);
		// MessageQueue::get_singleton()->flush();
		CHECK(tab_container->get_tab_count() == 1);
		CHECK(tab_container->get_current_tab() == 0);
		CHECK(tab_container->get_previous_tab() == -1);
		SIGNAL_CHECK("tab_selected", { { 0 } });
		SIGNAL_CHECK("tab_changed", { { 0 } });

		// Add second tab child.
		tab_container->add_child(tab1);
		CHECK(tab_container->get_tab_count() == 2);
		CHECK(tab_container->get_current_tab() == 0);
		CHECK(tab_container->get_previous_tab() == -1);
		SIGNAL_CHECK_FALSE("tab_selected");
		SIGNAL_CHECK_FALSE("tab_changed");

		// Check default values, the title is the name of the child.
		CHECK(tab_container->get_tab_control(0) == tab0);
		CHECK(tab_container->get_tab_idx_from_control(tab0) == 0);
		CHECK(tab_container->get_tab_title(0) == "tab0");
		CHECK(tab_container->get_tab_tooltip(0) == "");
		CHECK_FALSE(tab_container->is_tab_disabled(0));
		CHECK_FALSE(tab_container->is_tab_hidden(0));

		CHECK(tab_container->get_tab_control(1) == tab1);
		CHECK(tab_container->get_tab_idx_from_control(tab1) == 1);
		CHECK(tab_container->get_tab_title(1) == "tab1");
		CHECK(tab_container->get_tab_tooltip(1) == "");
		CHECK_FALSE(tab_container->is_tab_disabled(1));
		CHECK_FALSE(tab_container->is_tab_hidden(1));
	}

	SUBCASE("[TabContainer] remove tabs by removing children") {
		tab_container->add_child(tab0);
		tab_container->add_child(tab1);
		tab_container->add_child(tab2);
		tab_container->set_current_tab(1);
		CHECK(tab_container->get_tab_count() == 3);
		CHECK(tab_container->get_current_tab() == 1);
		CHECK(tab_container->get_previous_tab() == 0);
		SIGNAL_DISCARD("tab_selected");
		SIGNAL_DISCARD("tab_changed");

		// Remove first tab.
		tab_container->remove_child(tab0);
		CHECK(tab_container->get_tab_count() == 2);
		CHECK(tab_container->get_tab_title(0) == "tab1");
		CHECK(tab_container->get_tab_title(1) == "tab2");
		CHECK(tab_container->get_tab_idx_from_control(tab1) == 0);
		CHECK(tab_container->get_tab_idx_from_control(tab2) == 1);
		CHECK(tab_container->get_current_tab() == 0);
		CHECK(tab_container->get_previous_tab() == 0);
		SIGNAL_CHECK_FALSE("tab_selected");
		SIGNAL_CHECK_FALSE("tab_changed");

		// Remove last tab.
		tab_container->remove_child(tab2);
		CHECK(tab_container->get_tab_count() == 1);
		CHECK(tab_container->get_tab_title(0) == "tab1");
		CHECK(tab_container->get_tab_idx_from_control(tab1) == 0);
		CHECK(tab_container->get_current_tab() == 0);
		CHECK(tab_container->get_previous_tab() == 0);
		SIGNAL_CHECK_FALSE("tab_selected");
		SIGNAL_CHECK_FALSE("tab_changed");

		// Remove only tab.
		tab_container->remove_child(tab1);
		CHECK(tab_container->get_tab_count() == 0);
		CHECK(tab_container->get_current_tab() == -1);
		CHECK(tab_container->get_previous_tab() == -1);
		SIGNAL_CHECK_FALSE("tab_selected");
		SIGNAL_CHECK("tab_changed", { { -1 } });

		// Remove current tab when there are other tabs.
		tab_container->add_child(tab0);
		tab_container->add_child(tab1);
		tab_container->add_child(tab2);
		tab_container->set_current_tab(1);
		tab_container->set_current_tab(2);
		CHECK(tab_container->get_tab_count() == 3);
		CHECK(tab_container->get_current_tab() == 2);
		CHECK(tab_container->get_previous_tab() == 1);
		SIGNAL_DISCARD("tab_selected");
		SIGNAL_DISCARD("tab_changed");

		tab_container->remove_child(tab2);
		CHECK(tab_container->get_tab_count() == 2);
		CHECK(tab_container->get_current_tab() == 1);
		CHECK(tab_container->get_previous_tab() == 1);
		SIGNAL_CHECK_FALSE("tab_selected");
		SIGNAL_CHECK("tab_changed", { { 1 } });
	}

	SUBCASE("[TabContainer] move tabs by moving children") {
		tab_container->add_child(tab0);
		tab_container->add_child(tab1);
		tab_container->add_child(tab2);
		tab_container->set_current_tab(1);
		CHECK(tab_container->get_current_tab() == 1);
		CHECK(tab_container->get_previous_tab() == 0);
		SIGNAL_DISCARD("tab_selected");
		SIGNAL_DISCARD("tab_changed");

		// Move the first tab to the end.
		tab_container->move_child(tab0, 2);
		CHECK(tab_container->get_tab_idx_from_control(tab0) == 2);
		CHECK(tab_container->get_tab_idx_from_control(tab1) == 0);
		CHECK(tab_container->get_tab_idx_from_control(tab2) == 1);
		CHECK(tab_container->get_current_tab() == 0);
		CHECK(tab_container->get_previous_tab() == 2);
		SIGNAL_CHECK_FALSE("tab_selected");
		SIGNAL_CHECK_FALSE("tab_changed");

		// Move the second tab to the front.
		tab_container->move_child(tab2, 0);
		CHECK(tab_container->get_tab_idx_from_control(tab0) == 2);
		CHECK(tab_container->get_tab_idx_from_control(tab1) == 1);
		CHECK(tab_container->get_tab_idx_from_control(tab2) == 0);
		CHECK(tab_container->get_current_tab() == 1);
		CHECK(tab_container->get_previous_tab() == 2);
		SIGNAL_CHECK_FALSE("tab_selected");
		SIGNAL_CHECK_FALSE("tab_changed");
	}

	SUBCASE("[TabContainer] set current tab") {
		tab_container->add_child(tab0);
		tab_container->add_child(tab1);
		tab_container->add_child(tab2);
		CHECK(tab_container->get_current_tab() == 0);
		CHECK(tab_container->get_previous_tab() == -1);
		SIGNAL_CHECK("tab_selected", { { 0 } });
		SIGNAL_CHECK("tab_changed", { { 0 } });
		MessageQueue::get_singleton()->flush();
		CHECK(tab0->is_visible());
		CHECK_FALSE(tab1->is_visible());
		CHECK_FALSE(tab2->is_visible());

		// Set the current tab.
		tab_container->set_current_tab(1);
		CHECK(tab_container->get_current_tab() == 1);
		CHECK(tab_container->get_previous_tab() == 0);
		SIGNAL_CHECK("tab_selected", { { 1 } });
		SIGNAL_CHECK("tab_changed", { { 1 } });
		MessageQueue::get_singleton()->flush();
		CHECK_FALSE(tab0->is_visible());
		CHECK(tab1->is_visible());
		CHECK_FALSE(tab2->is_visible());

		// Set to same tab.
		tab_container->set_current_tab(1);
		CHECK(tab_container->get_current_tab() == 1);
		CHECK(tab_container->get_previous_tab() == 1);
		SIGNAL_CHECK("tab_selected", { { 1 } });
		SIGNAL_CHECK_FALSE("tab_changed");
		MessageQueue::get_singleton()->flush();
		CHECK_FALSE(tab0->is_visible());
		CHECK(tab1->is_visible());
		CHECK_FALSE(tab2->is_visible());

		// Out of bounds.
		ERR_PRINT_OFF;
		tab_container->set_current_tab(-5);
		CHECK(tab_container->get_current_tab() == 1);
		CHECK(tab_container->get_previous_tab() == 1);
		SIGNAL_CHECK_FALSE("tab_selected");
		SIGNAL_CHECK_FALSE("tab_changed");
		MessageQueue::get_singleton()->flush();
		CHECK_FALSE(tab0->is_visible());
		CHECK(tab1->is_visible());
		CHECK_FALSE(tab2->is_visible());

		tab_container->set_current_tab(5);
		CHECK(tab_container->get_current_tab() == 1);
		CHECK(tab_container->get_previous_tab() == 1);
		SIGNAL_CHECK_FALSE("tab_selected");
		SIGNAL_CHECK_FALSE("tab_changed");
		MessageQueue::get_singleton()->flush();
		CHECK_FALSE(tab0->is_visible());
		CHECK(tab1->is_visible());
		CHECK_FALSE(tab2->is_visible());
		ERR_PRINT_ON;
	}

	SUBCASE("[TabContainer] change current tab by changing visibility of children") {
		tab_container->add_child(tab0);
		tab_container->add_child(tab1);
		tab_container->add_child(tab2);
		SIGNAL_DISCARD("tab_selected");
		SIGNAL_DISCARD("tab_changed");
		MessageQueue::get_singleton()->flush();
		CHECK(tab0->is_visible());
		CHECK_FALSE(tab1->is_visible());
		CHECK_FALSE(tab2->is_visible());

		// Show a child to make it the current tab.
		tab1->show();
		CHECK(tab_container->get_current_tab() == 1);
		CHECK(tab_container->get_previous_tab() == 0);
		SIGNAL_CHECK("tab_selected", { { 1 } });
		SIGNAL_CHECK("tab_changed", { { 1 } });
		MessageQueue::get_singleton()->flush();
		CHECK_FALSE(tab0->is_visible());
		CHECK(tab1->is_visible());
		CHECK_FALSE(tab2->is_visible());

		// Hide the visible child to select the next tab.
		tab1->hide();
		CHECK(tab_container->get_current_tab() == 2);
		CHECK(tab_container->get_previous_tab() == 1);
		SIGNAL_CHECK("tab_selected", { { 2 } });
		SIGNAL_CHECK("tab_changed", { { 2 } });
		MessageQueue::get_singleton()->flush();
		CHECK_FALSE(tab0->is_visible());
		CHECK_FALSE(tab1->is_visible());
		CHECK(tab2->is_visible());

		// Hide the visible child to select the previous tab if there is no next.
		tab2->hide();
		CHECK(tab_container->get_current_tab() == 1);
		CHECK(tab_container->get_previous_tab() == 2);
		SIGNAL_CHECK("tab_selected", { { 1 } });
		SIGNAL_CHECK("tab_changed", { { 1 } });
		MessageQueue::get_singleton()->flush();
		CHECK_FALSE(tab0->is_visible());
		CHECK(tab1->is_visible());
		CHECK_FALSE(tab2->is_visible());

		// Cannot hide if there is only one valid child since deselection is not enabled.
		tab_container->remove_child(tab1);
		tab_container->remove_child(tab2);
		CHECK(tab_container->get_current_tab() == 0);
		CHECK(tab_container->get_previous_tab() == 0);
		SIGNAL_DISCARD("tab_selected");
		SIGNAL_DISCARD("tab_changed");
		MessageQueue::get_singleton()->flush();
		CHECK(tab0->is_visible());

		tab0->hide();
		CHECK(tab_container->get_current_tab() == 0);
		CHECK(tab_container->get_previous_tab() == 0);
		SIGNAL_CHECK_FALSE("tab_selected");
		SIGNAL_CHECK_FALSE("tab_changed");
		MessageQueue::get_singleton()->flush();
		CHECK(tab0->is_visible());

		// Can hide the last tab if deselection is enabled.
		tab_container->set_deselect_enabled(true);
		tab0->hide();
		CHECK(tab_container->get_current_tab() == -1);
		CHECK(tab_container->get_previous_tab() == 0);
		SIGNAL_CHECK("tab_selected", { { -1 } });
		SIGNAL_CHECK("tab_changed", { { -1 } });
		MessageQueue::get_singleton()->flush();
		CHECK_FALSE(tab0->is_visible());
	}

	SIGNAL_UNWATCH(tab_container, "tab_selected");
	SIGNAL_UNWATCH(tab_container, "tab_changed");

	memdelete(tab2);
	memdelete(tab1);
	memdelete(tab0);
	memdelete(tab_container);
}

TEST_CASE("[SceneTree][TabContainer] initialization") {
	TabContainer *tab_container = memnew(TabContainer);

	Control *tab0 = memnew(Control);
	tab0->set_name("tab0");
	Control *tab1 = memnew(Control);
	tab1->set_name("tab1 ");
	Control *tab2 = memnew(Control);
	tab2->set_name("tab2    ");

	SIGNAL_WATCH(tab_container, "tab_selected");
	SIGNAL_WATCH(tab_container, "tab_changed");

	SUBCASE("[TabContainer] add children before entering tree") {
		CHECK(tab_container->get_current_tab() == -1);
		CHECK(tab_container->get_previous_tab() == -1);

		tab_container->add_child(tab0);
		CHECK(tab_container->get_tab_count() == 1);
		CHECK(tab_container->get_current_tab() == 0);
		CHECK(tab_container->get_previous_tab() == -1);

		tab_container->add_child(tab1);
		CHECK(tab_container->get_tab_count() == 2);
		CHECK(tab_container->get_current_tab() == 0);
		CHECK(tab_container->get_previous_tab() == -1);

		SceneTree::get_singleton()->get_root()->add_child(tab_container);
		MessageQueue::get_singleton()->flush();
		CHECK(tab_container->get_tab_count() == 2);
		CHECK(tab_container->get_current_tab() == 0);
		CHECK(tab_container->get_previous_tab() == -1);
		SIGNAL_CHECK_FALSE("tab_selected");
		SIGNAL_CHECK_FALSE("tab_changed");
		CHECK(tab0->is_visible());
		CHECK_FALSE(tab1->is_visible());
	}

	SUBCASE("[TabContainer] current tab can be set before children are added") {
		// Set the current tab before there are any tabs.
		// This queues the current tab to update on entering the tree.
		tab_container->set_current_tab(1);
		CHECK(tab_container->get_current_tab() == -1);
		CHECK(tab_container->get_previous_tab() == -1);
		SIGNAL_CHECK_FALSE("tab_selected");
		SIGNAL_CHECK_FALSE("tab_changed");

		tab_container->add_child(tab0);
		CHECK(tab_container->get_tab_count() == 1);
		CHECK(tab_container->get_current_tab() == 0);
		CHECK(tab_container->get_previous_tab() == -1);

		tab_container->add_child(tab1);
		CHECK(tab_container->get_tab_count() == 2);
		CHECK(tab_container->get_current_tab() == 0);
		CHECK(tab_container->get_previous_tab() == -1);

		tab_container->add_child(tab2);
		CHECK(tab_container->get_tab_count() == 3);
		CHECK(tab_container->get_current_tab() == 0);
		CHECK(tab_container->get_previous_tab() == -1);
		SIGNAL_CHECK_FALSE("tab_selected");
		SIGNAL_CHECK_FALSE("tab_changed");

		// Current tab is set when entering the tree.
		SceneTree::get_singleton()->get_root()->add_child(tab_container);
		MessageQueue::get_singleton()->flush();
		CHECK(tab_container->get_tab_count() == 3);
		CHECK(tab_container->get_current_tab() == 1);
		CHECK(tab_container->get_previous_tab() == 0);
		SIGNAL_CHECK("tab_selected", { { 1 } });
		SIGNAL_CHECK("tab_changed", { { 1 } });
		CHECK_FALSE(tab0->is_visible());
		CHECK(tab1->is_visible());
		CHECK_FALSE(tab2->is_visible());
	}

	SUBCASE("[TabContainer] cannot set current tab to an invalid value before tabs are set") {
		tab_container->set_current_tab(100);
		CHECK(tab_container->get_current_tab() == -1);
		CHECK(tab_container->get_previous_tab() == -1);
		SIGNAL_CHECK_FALSE("tab_selected");
		SIGNAL_CHECK_FALSE("tab_changed");

		tab_container->add_child(tab0);
		CHECK(tab_container->get_tab_count() == 1);
		CHECK(tab_container->get_current_tab() == 0);
		CHECK(tab_container->get_previous_tab() == -1);
		SIGNAL_CHECK_FALSE("tab_selected");
		SIGNAL_CHECK_FALSE("tab_changed");

		tab_container->add_child(tab1);
		CHECK(tab_container->get_tab_count() == 2);
		CHECK(tab_container->get_current_tab() == 0);
		CHECK(tab_container->get_previous_tab() == -1);
		SIGNAL_CHECK_FALSE("tab_selected");
		SIGNAL_CHECK_FALSE("tab_changed");

		// This will print an error message as if `set_current_tab` was called after.
		ERR_PRINT_OFF;
		SceneTree::get_singleton()->get_root()->add_child(tab_container);
		MessageQueue::get_singleton()->flush();
		CHECK(tab_container->get_tab_count() == 2);
		CHECK(tab_container->get_current_tab() == 0);
		CHECK(tab_container->get_previous_tab() == -1);
		SIGNAL_CHECK_FALSE("tab_selected");
		SIGNAL_CHECK_FALSE("tab_changed");
		ERR_PRINT_ON;
	}

	SUBCASE("[TabContainer] children visibility before entering tree") {
		CHECK(tab_container->get_current_tab() == -1);
		CHECK(tab_container->get_previous_tab() == -1);

		// Adding a hidden child first will change visibility because it is the current tab.
		tab0->hide();
		tab_container->add_child(tab0);
		CHECK(tab_container->get_tab_count() == 1);
		CHECK(tab_container->get_current_tab() == 0);
		CHECK(tab_container->get_previous_tab() == -1);
		MessageQueue::get_singleton()->flush();
		CHECK(tab0->is_visible());

		// Adding a visible child after will hide it because it is not the current tab.
		tab_container->add_child(tab1);
		CHECK(tab_container->get_tab_count() == 2);
		CHECK(tab_container->get_current_tab() == 0);
		CHECK(tab_container->get_previous_tab() == -1);
		MessageQueue::get_singleton()->flush();
		CHECK(tab0->is_visible());
		CHECK_FALSE(tab1->is_visible());

		// Can change current by showing child now after children have been added.
		// This queues the current tab to update on entering the tree.
		tab1->show();
		MessageQueue::get_singleton()->flush();
		CHECK(tab_container->get_tab_count() == 2);
		CHECK(tab_container->get_current_tab() == 0);
		CHECK(tab_container->get_previous_tab() == -1);
		SIGNAL_CHECK_FALSE("tab_selected");
		SIGNAL_CHECK_FALSE("tab_changed");
		CHECK(tab0->is_visible());
		CHECK(tab1->is_visible());

		SceneTree::get_singleton()->get_root()->add_child(tab_container);
		MessageQueue::get_singleton()->flush();
		CHECK(tab_container->get_tab_count() == 2);
		CHECK(tab_container->get_current_tab() == 1);
		CHECK(tab_container->get_previous_tab() == 0);
		SIGNAL_CHECK("tab_selected", { { 1 } });
		SIGNAL_CHECK("tab_changed", { { 1 } });
		CHECK_FALSE(tab0->is_visible());
		CHECK(tab1->is_visible());
	}

	SUBCASE("[TabContainer] setting current tab and changing child visibility after adding") {
		tab_container->add_child(tab0);
		tab_container->add_child(tab1);
		tab_container->add_child(tab2);
		MessageQueue::get_singleton()->flush();
		CHECK(tab_container->get_current_tab() == 0);
		CHECK(tab_container->get_previous_tab() == -1);

		tab2->show();
		MessageQueue::get_singleton()->flush();
		CHECK(tab_container->get_tab_count() == 3);
		CHECK(tab_container->get_current_tab() == 0);
		CHECK(tab_container->get_previous_tab() == -1);
		SIGNAL_CHECK_FALSE("tab_selected");
		SIGNAL_CHECK_FALSE("tab_changed");
		CHECK(tab0->is_visible());
		CHECK_FALSE(tab1->is_visible());
		CHECK(tab2->is_visible());

		// Whichever happens last will have priority.
		tab_container->set_current_tab(1);
		CHECK(tab_container->get_current_tab() == 0);
		CHECK(tab_container->get_previous_tab() == -1);

		// Current tab is set when entering the tree.
		SceneTree::get_singleton()->get_root()->add_child(tab_container);
		MessageQueue::get_singleton()->flush();
		CHECK(tab_container->get_tab_count() == 3);
		CHECK(tab_container->get_current_tab() == 1);
		CHECK(tab_container->get_previous_tab() == 0);
		SIGNAL_CHECK("tab_selected", { { 1 } });
		SIGNAL_CHECK("tab_changed", { { 1 } });
		CHECK_FALSE(tab0->is_visible());
		CHECK(tab1->is_visible());
		CHECK_FALSE(tab2->is_visible());
	}

	SIGNAL_UNWATCH(tab_container, "tab_selected");
	SIGNAL_UNWATCH(tab_container, "tab_changed");

	memdelete(tab2);
	memdelete(tab1);
	memdelete(tab0);
	memdelete(tab_container);
}

TEST_CASE("[SceneTree][TabContainer] layout and offset") {
	TabContainer *tab_container = memnew(TabContainer);
	SceneTree::get_singleton()->get_root()->add_child(tab_container);
	tab_container->set_clip_tabs(false);

	Control *tab0 = memnew(Control);
	tab0->set_name("tab0");
	Control *tab1 = memnew(Control);
	tab1->set_name("tab1 ");
	Control *tab2 = memnew(Control);
	tab2->set_name("tab2    ");

	tab_container->add_child(tab0);
	tab_container->add_child(tab1);
	tab_container->add_child(tab2);

	MessageQueue::get_singleton()->flush();

	Size2 all_tabs_size = tab_container->get_size();
	const float side_margin = tab_container->get_theme_constant("side_margin");

	HBoxContainer *internal_container = tab_container->get_internal_container();
	TabBar *tab_bar = tab_container->get_tab_bar();

	Vector<Rect2> tab_rects = {
		tab_bar->get_tab_rect(0),
		tab_bar->get_tab_rect(1),
		tab_bar->get_tab_rect(2)
	};

	SUBCASE("[TabContainer] tabs are arranged next to each other") {
		// Horizontal positions are next to each other.
		CHECK(tab_rects[0].position.x == 0);
		CHECK(tab_rects[1].position.x == tab_rects[0].size.x);
		CHECK(tab_rects[2].position.x == tab_rects[1].position.x + tab_rects[1].size.x);

		// Fills the entire width.
		CHECK(tab_rects[2].position.x + tab_rects[2].size.x == all_tabs_size.x - side_margin);

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

	SUBCASE("[TabContainer] tab position") {
		float tab_height = tab_rects[0].size.y;
		Ref<StyleBox> panel_style = tab_container->get_theme_stylebox("panel_style");

		// Initial position, same as top position.
		// Tab bar is at the top.
		CHECK(internal_container->get_anchor(SIDE_TOP) == 0);
		CHECK(internal_container->get_anchor(SIDE_BOTTOM) == 0);
		CHECK(internal_container->get_anchor(SIDE_LEFT) == 0);
		CHECK(internal_container->get_anchor(SIDE_RIGHT) == 1);
		CHECK(internal_container->get_offset(SIDE_TOP) == 0);
		CHECK(internal_container->get_offset(SIDE_BOTTOM) == tab_height);
		CHECK(internal_container->get_offset(SIDE_LEFT) == side_margin);
		CHECK(internal_container->get_offset(SIDE_RIGHT) == 0);

		// Child is expanded and below the tab bar.
		CHECK(tab0->get_anchor(SIDE_TOP) == 0);
		CHECK(tab0->get_anchor(SIDE_BOTTOM) == 1);
		CHECK(tab0->get_anchor(SIDE_LEFT) == 0);
		CHECK(tab0->get_anchor(SIDE_RIGHT) == 1);
		CHECK(tab0->get_offset(SIDE_TOP) == tab_height);
		CHECK(tab0->get_offset(SIDE_BOTTOM) == 0);
		CHECK(tab0->get_offset(SIDE_LEFT) == 0);
		CHECK(tab0->get_offset(SIDE_RIGHT) == 0);

		// Bottom position.
		tab_container->set_tabs_position(TabContainer::POSITION_BOTTOM);
		CHECK(tab_container->get_tabs_position() == TabContainer::POSITION_BOTTOM);
		MessageQueue::get_singleton()->flush();

		// Tab bar is at the bottom.
		CHECK(internal_container->get_anchor(SIDE_TOP) == 1);
		CHECK(internal_container->get_anchor(SIDE_BOTTOM) == 1);
		CHECK(internal_container->get_anchor(SIDE_LEFT) == 0);
		CHECK(internal_container->get_anchor(SIDE_RIGHT) == 1);
		CHECK(internal_container->get_offset(SIDE_TOP) == -tab_height);
		CHECK(internal_container->get_offset(SIDE_BOTTOM) == 0);
		CHECK(internal_container->get_offset(SIDE_LEFT) == side_margin);
		CHECK(internal_container->get_offset(SIDE_RIGHT) == 0);

		// Child is expanded and above the tab bar.
		CHECK(tab0->get_anchor(SIDE_TOP) == 0);
		CHECK(tab0->get_anchor(SIDE_BOTTOM) == 1);
		CHECK(tab0->get_anchor(SIDE_LEFT) == 0);
		CHECK(tab0->get_anchor(SIDE_RIGHT) == 1);
		CHECK(tab0->get_offset(SIDE_TOP) == 0);
		CHECK(tab0->get_offset(SIDE_BOTTOM) == -tab_height);
		CHECK(tab0->get_offset(SIDE_LEFT) == 0);
		CHECK(tab0->get_offset(SIDE_RIGHT) == 0);

		// Top position.
		tab_container->set_tabs_position(TabContainer::POSITION_TOP);
		CHECK(tab_container->get_tabs_position() == TabContainer::POSITION_TOP);
		MessageQueue::get_singleton()->flush();

		// Tab bar is at the top.
		CHECK(internal_container->get_anchor(SIDE_TOP) == 0);
		CHECK(internal_container->get_anchor(SIDE_BOTTOM) == 0);
		CHECK(internal_container->get_anchor(SIDE_LEFT) == 0);
		CHECK(internal_container->get_anchor(SIDE_RIGHT) == 1);
		CHECK(internal_container->get_offset(SIDE_TOP) == 0);
		CHECK(internal_container->get_offset(SIDE_BOTTOM) == tab_height);
		CHECK(internal_container->get_offset(SIDE_LEFT) == side_margin);
		CHECK(internal_container->get_offset(SIDE_RIGHT) == 0);

		// Child is expanded and below the tab bar.
		CHECK(tab0->get_anchor(SIDE_TOP) == 0);
		CHECK(tab0->get_anchor(SIDE_BOTTOM) == 1);
		CHECK(tab0->get_anchor(SIDE_LEFT) == 0);
		CHECK(tab0->get_anchor(SIDE_RIGHT) == 1);
		CHECK(tab0->get_offset(SIDE_TOP) == tab_height);
		CHECK(tab0->get_offset(SIDE_BOTTOM) == 0);
		CHECK(tab0->get_offset(SIDE_LEFT) == 0);
		CHECK(tab0->get_offset(SIDE_RIGHT) == 0);

		// Left position.
		tab_container->set_tabs_position(TabContainer::POSITION_LEFT);
		CHECK(tab_container->get_tabs_position() == TabContainer::POSITION_LEFT);
		MessageQueue::get_singleton()->flush();
		CHECK(tab_bar->is_vertical());

		// Tab bar is at the left.
		CHECK(internal_container->get_anchor(SIDE_LEFT) == 0);
		CHECK(internal_container->get_anchor(SIDE_RIGHT) == 0);
		CHECK(internal_container->get_anchor(SIDE_TOP) == 0);
		CHECK(internal_container->get_anchor(SIDE_BOTTOM) == 1);

		// Child is expanded and to the right of the tab bar.
		CHECK(tab0->get_anchor(SIDE_TOP) == 0);
		CHECK(tab0->get_anchor(SIDE_BOTTOM) == 1);
		CHECK(tab0->get_anchor(SIDE_LEFT) == 0);
		CHECK(tab0->get_anchor(SIDE_RIGHT) == 1);
		CHECK(tab0->get_offset(SIDE_LEFT) > 0);
		CHECK(tab0->get_offset(SIDE_RIGHT) == 0);

		// Tabs are arranged vertically.
		tab_rects = { tab_bar->get_tab_rect(0), tab_bar->get_tab_rect(1), tab_bar->get_tab_rect(2) };
		CHECK(tab_rects[0].position.x == tab_rects[1].position.x);
		CHECK(tab_rects[1].position.x == tab_rects[2].position.x);
		CHECK(tab_rects[1].position.y == tab_rects[0].size.y);
		CHECK(tab_rects[2].position.y == tab_rects[1].position.y + tab_rects[1].size.y);

		// Right position.
		tab_container->set_tabs_position(TabContainer::POSITION_RIGHT);
		CHECK(tab_container->get_tabs_position() == TabContainer::POSITION_RIGHT);
		MessageQueue::get_singleton()->flush();
		CHECK(tab_bar->is_vertical());

		// Tab bar is at the right.
		CHECK(internal_container->get_anchor(SIDE_LEFT) == 1);
		CHECK(internal_container->get_anchor(SIDE_RIGHT) == 1);
		CHECK(internal_container->get_anchor(SIDE_TOP) == 0);
		CHECK(internal_container->get_anchor(SIDE_BOTTOM) == 1);

		// Child is expanded and to the left of the tab bar.
		CHECK(tab0->get_anchor(SIDE_TOP) == 0);
		CHECK(tab0->get_anchor(SIDE_BOTTOM) == 1);
		CHECK(tab0->get_anchor(SIDE_LEFT) == 0);
		CHECK(tab0->get_anchor(SIDE_RIGHT) == 1);
		CHECK(tab0->get_offset(SIDE_LEFT) == 0);
		CHECK(tab0->get_offset(SIDE_RIGHT) < 0);
	}

	SUBCASE("[TabContainer] vertical clip tabs keep width and reserve bottom controls row") {
		tab_container->set_tabs_position(TabContainer::POSITION_LEFT);
		tab_container->set_clip_tabs(true);
		tab_container->set_size(Size2(tab_container->get_size().x, tab_bar->get_size().y + 200));
		MessageQueue::get_singleton()->flush();
		CHECK_FALSE(tab_bar->get_offset_buttons_visible());
		const float unclipped_tab_width = tab_bar->get_tab_rect(0).size.x;

		tab_container->set_size(Size2(tab_container->get_size().x, tab_bar->get_tab_rect(0).size.y + 10));
		MessageQueue::get_singleton()->flush();

		CHECK(tab_bar->is_vertical());
		CHECK(tab_bar->get_offset_buttons_visible());
		CHECK(tab_bar->get_tab_rect(0).size.x == unclipped_tab_width);
		CHECK(tab_bar->get_tab_rect(0).get_end().y < tab_bar->get_size().y);

		const Point2 tab_bar_pos = internal_container->get_position() + tab_bar->get_position();
		const float first_tab_right = tab_bar_pos.x + tab_bar->get_tab_rect(0).get_end().x;
		CHECK(first_tab_right <= tab0->get_offset(SIDE_LEFT));

		tab_container->set_clip_tabs(false);
		MessageQueue::get_singleton()->flush();
		const float first_tab_right_no_clip = (internal_container->get_position() + tab_bar->get_position()).x + tab_bar->get_tab_rect(0).get_end().x;
		CHECK(first_tab_right_no_clip <= tab0->get_offset(SIDE_LEFT));
	}

	SUBCASE("[TabContainer] vertical popup button stays left of the centered scroll buttons") {
		tab_container->set_tabs_position(TabContainer::POSITION_LEFT);
		tab_container->set_clip_tabs(true);
		PopupMenu *popup = memnew(PopupMenu);
		tab_container->set_popup(popup);

		tab_container->set_size(Size2(tab_container->get_size().x, tab_bar->get_size().y + 200));
		MessageQueue::get_singleton()->flush();
		tab_bar->set_tab_offset(1);
		tab_container->set_size(Size2(tab_container->get_size().x, tab_bar->get_minimum_size().y + 100));
		MessageQueue::get_singleton()->flush();

		Button *popup_button = nullptr;
		for (int i = 0; i < tab_bar->get_child_count(); i++) {
			popup_button = Object::cast_to<Button>(tab_bar->get_child(i));
			if (popup_button && popup_button->is_visible()) {
				break;
			}
		}
		REQUIRE(popup_button != nullptr);

		const Ref<Texture2D> dec_icon = tab_bar->get_theme_icon("decrement_icon");
		const Ref<Texture2D> inc_icon = tab_bar->get_theme_icon("increment_icon");
		const float row_width = popup_button->get_minimum_size().x + dec_icon->get_width() + inc_icon->get_width();
		const float row_left = (tab_bar->get_size().x - row_width) * 0.5f;
		const int row_height = MAX((int)popup_button->get_size().y, MAX(dec_icon->get_height(), inc_icon->get_height()));
		const int row_top = popup_button->get_position().y - MAX(0, (row_height - (int)popup_button->get_size().y) / 2);

		int last_visible_tab = -1;
		for (int i = tab_bar->get_tab_count() - 1; i >= 0; i--) {
			if (tab_bar->get_tab_rect(i).get_end().y <= row_top) {
				last_visible_tab = i;
				break;
			}
		}
		REQUIRE(last_visible_tab != -1);
		const Rect2 last_visible_tab_rect = tab_bar->get_tab_rect(last_visible_tab);

		CHECK(popup_button->get_position().x == row_left);
		CHECK(row_top == last_visible_tab_rect.get_end().y);
		CHECK(row_top < tab_bar->get_size().y - row_height);
		const int dec_center_y = row_top + row_height / 2;
		const Point2 dec_button_click(row_left + popup_button->get_minimum_size().x + dec_icon->get_width() * 0.5f, dec_center_y);
		SEND_GUI_MOUSE_BUTTON_EVENT(dec_button_click, MouseButton::LEFT, MouseButtonMask::LEFT, Key::NONE);
		CHECK(tab_bar->get_tab_offset() == 0);

		tab_container->set_popup(nullptr);
		memdelete(popup);
	}

	SUBCASE("[TabContainer] vertical non-clip popup row is included in minimum height") {
		tab_container->set_tabs_position(TabContainer::POSITION_LEFT);
		tab_container->set_clip_tabs(false);
		const float tabs_only_height = tab_bar->get_tab_rect(tab_bar->get_tab_count() - 1).get_end().y;
		PopupMenu *popup = memnew(PopupMenu);
		tab_container->set_popup(popup);

		tab_container->set_size(Size2(tab_container->get_size().x, tabs_only_height));
		MessageQueue::get_singleton()->flush();

		Button *popup_button = nullptr;
		for (int i = 0; i < tab_bar->get_child_count(); i++) {
			popup_button = Object::cast_to<Button>(tab_bar->get_child(i));
			if (popup_button && popup_button->is_visible()) {
				break;
			}
		}
		REQUIRE(popup_button != nullptr);

		const float required_height = tab_container->get_minimum_size().y;
		CHECK(required_height > tabs_only_height);

		tab_container->set_size(Size2(tab_container->get_size().x, required_height));
		MessageQueue::get_singleton()->flush();

		const Rect2 last_tab_rect = tab_bar->get_tab_rect(tab_bar->get_tab_count() - 1);
		CHECK(last_tab_rect.get_end().y <= popup_button->get_position().y);

		tab_container->set_popup(nullptr);
		memdelete(popup);
	}

	SUBCASE("[TabContainer] vertical clip popup without scroll buttons stays under the last tab") {
		tab_container->set_tabs_position(TabContainer::POSITION_LEFT);
		tab_container->set_clip_tabs(true);
		PopupMenu *popup = memnew(PopupMenu);
		tab_container->set_popup(popup);

		tab_container->set_size(Size2(tab_container->get_size().x, tab_bar->get_size().y + 200));
		MessageQueue::get_singleton()->flush();
		CHECK_FALSE(tab_bar->get_offset_buttons_visible());

		Button *popup_button = nullptr;
		for (int i = 0; i < tab_bar->get_child_count(); i++) {
			popup_button = Object::cast_to<Button>(tab_bar->get_child(i));
			if (popup_button && popup_button->is_visible()) {
				break;
			}
		}
		REQUIRE(popup_button != nullptr);

		const Rect2 last_tab_rect = tab_bar->get_tab_rect(tab_bar->get_tab_count() - 1);
		CHECK(last_tab_rect.get_end().y <= popup_button->get_position().y);

		tab_container->set_popup(nullptr);
		memdelete(popup);
	}

	SUBCASE("[TabContainer] vertical clip popup stays in bounds by clipping tabs") {
		tab_container->set_clip_tabs(true);

		for (int i = 0; i < 2; i++) {
			tab_container->set_tabs_position(i == 0 ? TabContainer::POSITION_LEFT : TabContainer::POSITION_RIGHT);
			PopupMenu *popup = memnew(PopupMenu);
			tab_container->set_popup(popup);

			MessageQueue::get_singleton()->flush();
			const Rect2 last_tab_rect = tab_bar->get_tab_rect(tab_bar->get_tab_count() - 1);
			tab_container->set_size(Size2(tab_container->get_size().x, last_tab_rect.get_end().y + 1.0f));
			MessageQueue::get_singleton()->flush();

			Button *popup_button = nullptr;
			for (int j = 0; j < tab_bar->get_child_count(); j++) {
				popup_button = Object::cast_to<Button>(tab_bar->get_child(j));
				if (popup_button && popup_button->is_visible()) {
					break;
				}
			}
			REQUIRE(popup_button != nullptr);

			CHECK(tab_bar->get_offset_buttons_visible());
			CHECK(last_tab_rect.get_end().y <= popup_button->get_position().y);

			tab_container->set_popup(nullptr);
			memdelete(popup);
		}
	}

	SUBCASE("[TabContainer] vertical left low max tab width has no gap") {
		tab_container->set_tabs_position(TabContainer::POSITION_LEFT);
		tab_container->set_clip_tabs(true);
		tab_bar->set_max_tab_width(60);
		MessageQueue::get_singleton()->flush();

		const Point2 tab_bar_pos = internal_container->get_position() + tab_bar->get_position();
		const float first_tab_right = tab_bar_pos.x + tab_bar->get_tab_rect(0).get_end().x;
		CHECK(tab0->get_offset(SIDE_LEFT) <= first_tab_right);
	}

	SUBCASE("[TabContainer] vertical clip engages with low custom maximum height") {
		tab_container->set_clip_tabs(true);

		for (int i = 0; i < 2; i++) {
			tab_container->set_tabs_position(i == 0 ? TabContainer::POSITION_LEFT : TabContainer::POSITION_RIGHT);
			tab_container->set_custom_maximum_size(Size2(-1, 40));
			tab_container->set_size(Size2(400, 400));
			MessageQueue::get_singleton()->flush();

			CHECK(tab_bar->is_vertical());
			CHECK(tab_bar->get_offset_buttons_visible());
		}
	}

	SUBCASE("[TabContainer] switching side and top tabs updates child area immediately") {
		tab_container->set_clip_tabs(true);
		tab_container->set_custom_maximum_size(Size2(320, 240));
		tab_container->set_size(Size2(180, 120));

		tab_container->set_tabs_position(TabContainer::POSITION_LEFT);
		MessageQueue::get_singleton()->flush();
		const Size2 left_size = tab0->get_size();

		tab_container->set_tabs_position(TabContainer::POSITION_TOP);
		MessageQueue::get_singleton()->flush();
		const Size2 top_size = tab0->get_size();

		CHECK(top_size.x > left_size.x);
		CHECK(top_size.y < left_size.y);
	}

	SUBCASE("[TabContainer] right tabs stay in bounds with custom maximum width") {
		tab_container->set_tabs_position(TabContainer::POSITION_RIGHT);
		tab_container->set_custom_maximum_size(Size2(120, -1));
		tab_container->set_size(Size2(400, 200));
		MessageQueue::get_singleton()->flush();

		for (int i = 0; i < 10; i++) {
			tab_container->set_size(Size2(400, 160 + i * 5));
			MessageQueue::get_singleton()->flush();

			const Rect2 internal_rect(internal_container->get_position(), internal_container->get_size());
			const Rect2 tabbar_rect(internal_container->get_position() + tab_bar->get_position(), tab_bar->get_size());
			CHECK(tabbar_rect.position.x >= internal_rect.position.x);
			CHECK(tabbar_rect.get_end().x <= internal_rect.get_end().x);
		}
	}

	SUBCASE("[TabContainer] right tabs stay in bounds after resize with custom maximum width") {
		tab_container->set_tabs_position(TabContainer::POSITION_RIGHT);
		MessageQueue::get_singleton()->flush();

		tab_container->set_custom_maximum_size(Size2(120, -1));
		tab_container->set_size(Size2(400, 220));
		MessageQueue::get_singleton()->flush();

		const Rect2 internal_rect(internal_container->get_position(), internal_container->get_size());
		const Rect2 tabbar_rect(internal_container->get_position() + tab_bar->get_position(), tab_bar->get_size());
		CHECK(tabbar_rect.position.x >= internal_rect.position.x);
		CHECK(tabbar_rect.get_end().x <= internal_rect.get_end().x);
	}

	SUBCASE("[TabContainer] clip_tabs prevents Y-axis resize when adding tabs that don't fit") {
		// Enable clip_tabs and set a limited width to force tabs to be clipped
		tab_container->set_clip_tabs(true);
		tab_container->set_custom_maximum_size(Size2(200, -1));
		tab_container->set_size(Size2(200, 300));
		MessageQueue::get_singleton()->flush();

		// Get the current Y size after setup
		float initial_height = tab_container->get_size().height;

		// Add a new child that has a title long enough to not fit horizontally
		Control *new_tab = memnew(Control);
		new_tab->set_name("new_tab_with_very_long_name_that_wont_fit");
		tab_container->add_child(new_tab);
		MessageQueue::get_singleton()->flush();

		// Get the new size
		float final_height = tab_container->get_size().height;

		// The Y-axis size should not have changed since the new tab doesn't require more height
		// (assuming the new tab has the same minimum height requirements as existing tabs)
		CHECK(final_height == initial_height);

		// The tab bar should be using scroll buttons now due to clipping
		CHECK(tab_bar->get_offset_buttons_visible());

		memdelete(new_tab);
	}

	SUBCASE("[TabContainer] clip_tabs allows resize when new tab requires more vertical space") {
		tab_container->set_clip_tabs(true);
		tab_container->set_use_hidden_tabs_for_min_size(true);
		tab_container->set_custom_maximum_size(Size2(200, -1));
		tab_container->set_size(Size2(200, 100));
		MessageQueue::get_singleton()->flush();

		float initial_min_height = tab_container->get_minimum_size().height;

		// Add a child with a larger minimum size (e.g., Panel with padding)
		Control *new_tab = memnew(Control);
		new_tab->set_custom_minimum_size(Size2(50, 300)); // Much larger than tab bar height
		new_tab->set_name("large_tab");
		tab_container->add_child(new_tab);
		MessageQueue::get_singleton()->flush();

		float final_min_height = tab_container->get_minimum_size().height;

		// The minimum size should have increased because the new tab requires more space
		CHECK(final_min_height > initial_min_height);

		memdelete(new_tab);
	}

	memdelete(tab_container);
}

TEST_CASE("[SceneTree][TabContainer] Mouse interaction") {
	TabContainer *tab_container = memnew(TabContainer);
	SceneTree::get_singleton()->get_root()->add_child(tab_container);

	tab_container->set_clip_tabs(false);
	Control *tab0 = memnew(Control);
	tab0->set_name("tab0");
	Control *tab1 = memnew(Control);
	tab1->set_name("tab1 ");
	Control *tab2 = memnew(Control);
	tab2->set_name("tab2    ");

	tab_container->add_child(tab0);
	tab_container->add_child(tab1);
	tab_container->add_child(tab2);

	MessageQueue::get_singleton()->flush();

	const float side_margin = tab_container->get_theme_constant("side_margin");
	Container *internal_container = tab_container->get_internal_container();

	TabBar *tab_bar = tab_container->get_tab_bar();
	const Point2 tab_bar_pos = internal_container->get_position() + tab_bar->get_position();
	Vector<Rect2> tab_rects = {
		tab_bar->get_tab_rect(0),
		tab_bar->get_tab_rect(1),
		tab_bar->get_tab_rect(2)
	};

	SIGNAL_WATCH(tab_container, "active_tab_rearranged");
	SIGNAL_WATCH(tab_container, "tab_changed");
	SIGNAL_WATCH(tab_container, "tab_clicked");
	SIGNAL_WATCH(tab_container, "tab_selected");

	SUBCASE("[TabContainer] Click to change current") {
		CHECK(tab_container->get_current_tab() == 0);
		CHECK(tab_container->get_previous_tab() == -1);
		SIGNAL_DISCARD("tab_selected");
		SIGNAL_DISCARD("tab_changed");

		// Click to set the current tab.
		SEND_GUI_MOUSE_BUTTON_EVENT(tab_bar_pos + tab_rects[1].get_center(), MouseButton::LEFT, MouseButtonMask::LEFT, Key::NONE);
		CHECK(tab_container->get_current_tab() == 1);
		CHECK(tab_container->get_previous_tab() == 0);
		SIGNAL_CHECK("tab_selected", { { 1 } });
		SIGNAL_CHECK("tab_changed", { { 1 } });
		SIGNAL_CHECK("tab_clicked", { { 1 } });

		// Click on the same tab.
		SEND_GUI_MOUSE_BUTTON_EVENT(tab_bar_pos + tab_rects[1].get_center(), MouseButton::LEFT, MouseButtonMask::LEFT, Key::NONE);
		CHECK(tab_container->get_current_tab() == 1);
		CHECK(tab_container->get_previous_tab() == 1);
		SIGNAL_CHECK("tab_selected", { { 1 } });
		SIGNAL_CHECK_FALSE("tab_changed");
		SIGNAL_CHECK("tab_clicked", { { 1 } });

		// Click outside of tabs.
		SEND_GUI_MOUSE_BUTTON_EVENT(Point2(tab_container->get_size().x + 10, 10), MouseButton::LEFT, MouseButtonMask::LEFT, Key::NONE);
		CHECK(tab_container->get_current_tab() == 1);
		CHECK(tab_container->get_previous_tab() == 1);
		SIGNAL_CHECK_FALSE("tab_selected");
		SIGNAL_CHECK_FALSE("tab_changed");
		SIGNAL_CHECK_FALSE("tab_clicked");
	}

	SUBCASE("[TabContainer] Drag and drop internally") {
		// Cannot drag if not enabled.
		CHECK_FALSE(tab_container->get_drag_to_rearrange_enabled());
		SEND_GUI_MOUSE_BUTTON_EVENT(tab_bar_pos + tab_rects[0].get_center(), MouseButton::LEFT, MouseButtonMask::LEFT, Key::NONE);
		SEND_GUI_MOUSE_MOTION_EVENT(tab_bar_pos + tab_rects[1].get_center(), MouseButtonMask::LEFT, Key::NONE);
		SIGNAL_CHECK("tab_selected", { { 0 } });
		SIGNAL_CHECK_FALSE("tab_changed");
		CHECK_FALSE(tab_container->get_viewport()->gui_is_dragging());
		SEND_GUI_MOUSE_BUTTON_RELEASED_EVENT(tab_bar_pos + tab_rects[1].get_center(), MouseButton::LEFT, MouseButtonMask::NONE, Key::NONE);
		CHECK_FALSE(tab_container->get_viewport()->gui_is_dragging());
		CHECK(tab_container->get_tab_idx_from_control(tab0) == 0);
		CHECK(tab_container->get_tab_idx_from_control(tab1) == 1);
		CHECK(tab_container->get_tab_idx_from_control(tab2) == 2);
		SIGNAL_CHECK_FALSE("active_tab_rearranged");
		SIGNAL_CHECK_FALSE("tab_selected");
		SIGNAL_CHECK_FALSE("tab_changed");

		tab_container->set_drag_to_rearrange_enabled(true);
		CHECK(tab_container->get_drag_to_rearrange_enabled());

		// Release over the same tab to not move.
		SEND_GUI_MOUSE_BUTTON_EVENT(tab_bar_pos + tab_rects[0].get_center(), MouseButton::LEFT, MouseButtonMask::LEFT, Key::NONE);
		SEND_GUI_MOUSE_MOTION_EVENT(tab_bar_pos + tab_rects[1].get_center(), MouseButtonMask::LEFT, Key::NONE);
		SEND_GUI_MOUSE_MOTION_EVENT(tab_bar_pos + tab_rects[0].get_center(), MouseButtonMask::LEFT, Key::NONE);
		SIGNAL_CHECK("tab_selected", { { 0 } });
		SIGNAL_CHECK_FALSE("tab_changed");
		CHECK(tab_container->get_viewport()->gui_is_dragging());
		SEND_GUI_MOUSE_BUTTON_RELEASED_EVENT(tab_bar_pos + tab_rects[0].get_center(), MouseButton::LEFT, MouseButtonMask::NONE, Key::NONE);
		CHECK_FALSE(tab_container->get_viewport()->gui_is_dragging());
		CHECK(tab_container->get_tab_idx_from_control(tab0) == 0);
		CHECK(tab_container->get_tab_idx_from_control(tab1) == 1);
		CHECK(tab_container->get_tab_idx_from_control(tab2) == 2);
		SIGNAL_CHECK_FALSE("active_tab_rearranged");
		SIGNAL_CHECK_FALSE("tab_selected");
		SIGNAL_CHECK_FALSE("tab_changed");

		// Move the first tab after the second.
		SEND_GUI_MOUSE_BUTTON_EVENT(tab_bar_pos + tab_rects[0].get_center(), MouseButton::LEFT, MouseButtonMask::LEFT, Key::NONE);
		SEND_GUI_MOUSE_MOTION_EVENT(tab_bar_pos + tab_rects[1].get_center(), MouseButtonMask::LEFT, Key::NONE);
		SIGNAL_CHECK("tab_selected", { { 0 } });
		SIGNAL_CHECK_FALSE("tab_changed");
		CHECK(tab_container->get_viewport()->gui_is_dragging());
		SEND_GUI_MOUSE_BUTTON_RELEASED_EVENT(tab_bar_pos + tab_rects[1].get_center() + Point2(tab_rects[1].size.x / 2 + 1, 0), MouseButton::LEFT, MouseButtonMask::NONE, Key::NONE);
		CHECK_FALSE(tab_container->get_viewport()->gui_is_dragging());
		CHECK(tab_container->get_tab_idx_from_control(tab1) == 0);
		CHECK(tab_container->get_tab_idx_from_control(tab0) == 1);
		CHECK(tab_container->get_tab_idx_from_control(tab2) == 2);
		SIGNAL_CHECK("active_tab_rearranged", { { 1 } });
		SIGNAL_CHECK("tab_selected", { { 1 } });
		SIGNAL_CHECK_FALSE("tab_changed");

		tab_rects = { tab_bar->get_tab_rect(0), tab_bar->get_tab_rect(1), tab_bar->get_tab_rect(2) };

		// Move the last tab to be the first.
		SEND_GUI_MOUSE_BUTTON_EVENT(tab_bar_pos + tab_rects[2].get_center(), MouseButton::LEFT, MouseButtonMask::LEFT, Key::NONE);
		SEND_GUI_MOUSE_MOTION_EVENT(tab_bar_pos + tab_rects[0].get_center(), MouseButtonMask::LEFT, Key::NONE);
		SIGNAL_CHECK("tab_selected", { { 2 } });
		SIGNAL_CHECK("tab_changed", { { 2 } });
		CHECK(tab_container->get_viewport()->gui_is_dragging());
		SEND_GUI_MOUSE_BUTTON_RELEASED_EVENT(tab_bar_pos + tab_rects[0].get_center(), MouseButton::LEFT, MouseButtonMask::NONE, Key::NONE);
		CHECK_FALSE(tab_container->get_viewport()->gui_is_dragging());
		CHECK(tab_container->get_tab_idx_from_control(tab2) == 0);
		CHECK(tab_container->get_tab_idx_from_control(tab1) == 1);
		CHECK(tab_container->get_tab_idx_from_control(tab0) == 2);
		SIGNAL_CHECK("active_tab_rearranged", { { 0 } });
		SIGNAL_CHECK("tab_selected", { { 0 } });
		SIGNAL_CHECK_FALSE("tab_changed");
	}

	SUBCASE("[TabContainer] Drag and drop to different TabContainer") {
		TabContainer *target_tab_container = memnew(TabContainer);
		SceneTree::get_singleton()->get_root()->add_child(target_tab_container);

		target_tab_container->set_clip_tabs(false);
		Control *other_tab0 = memnew(Control);
		other_tab0->set_name("other_tab0");
		target_tab_container->add_child(other_tab0);

		target_tab_container->set_position(tab_container->get_size());
		MessageQueue::get_singleton()->flush();

		Vector<Rect2> target_tab_rects = {
			target_tab_container->get_tab_bar()->get_tab_rect(0)
		};
		tab_container->set_drag_to_rearrange_enabled(true);
		tab_container->set_tabs_rearrange_group(1);

		Point2 target_tab_after_first = Point2(target_tab_container->get_internal_container()->get_offset(SIDE_LEFT), 0) + target_tab_container->get_position() + target_tab_rects[0].position + Point2(target_tab_rects[0].size.x / 2 + 1, 0);

		// Cannot drag to another TabContainer that does not have drag to rearrange enabled.
		CHECK_FALSE(target_tab_container->get_drag_to_rearrange_enabled());
		SEND_GUI_MOUSE_BUTTON_EVENT(Point2(side_margin, 0) + tab_rects[0].position, MouseButton::LEFT, MouseButtonMask::LEFT, Key::NONE);
		SEND_GUI_MOUSE_MOTION_EVENT(Point2(side_margin, 0) + tab_rects[0].position + Point2(20, 0), MouseButtonMask::LEFT, Key::NONE);
		SEND_GUI_MOUSE_MOTION_EVENT(target_tab_after_first, MouseButtonMask::LEFT, Key::NONE);
		SIGNAL_CHECK("tab_selected", { { 0 } });
		SIGNAL_CHECK_FALSE("tab_changed");
		CHECK(tab_container->get_viewport()->gui_is_dragging());
		SEND_GUI_MOUSE_BUTTON_RELEASED_EVENT(target_tab_after_first, MouseButton::LEFT, MouseButtonMask::NONE, Key::NONE);
		CHECK_FALSE(tab_container->get_viewport()->gui_is_dragging());
		CHECK(tab_container->get_tab_count() == 3);
		CHECK(target_tab_container->get_tab_count() == 1);
		SIGNAL_CHECK_FALSE("active_tab_rearranged");
		SIGNAL_CHECK_FALSE("tab_selected");
		SIGNAL_CHECK_FALSE("tab_changed");

		// Cannot drag to another TabContainer that has a tabs rearrange group of -1.
		target_tab_container->set_drag_to_rearrange_enabled(true);
		tab_container->set_tabs_rearrange_group(-1);
		SEND_GUI_MOUSE_BUTTON_EVENT(Point2(side_margin, 0) + tab_rects[0].position, MouseButton::LEFT, MouseButtonMask::LEFT, Key::NONE);
		SEND_GUI_MOUSE_MOTION_EVENT(Point2(side_margin, 0) + tab_rects[0].position + Point2(20, 0), MouseButtonMask::LEFT, Key::NONE);
		SEND_GUI_MOUSE_MOTION_EVENT(target_tab_after_first, MouseButtonMask::LEFT, Key::NONE);
		SIGNAL_CHECK("tab_selected", { { 0 } });
		SIGNAL_CHECK_FALSE("tab_changed");
		CHECK(tab_container->get_viewport()->gui_is_dragging());
		SEND_GUI_MOUSE_BUTTON_RELEASED_EVENT(target_tab_after_first, MouseButton::LEFT, MouseButtonMask::NONE, Key::NONE);
		CHECK_FALSE(tab_container->get_viewport()->gui_is_dragging());
		CHECK(tab_container->get_tab_count() == 3);
		CHECK(target_tab_container->get_tab_count() == 1);
		SIGNAL_CHECK_FALSE("active_tab_rearranged");
		SIGNAL_CHECK_FALSE("tab_selected");
		SIGNAL_CHECK_FALSE("tab_changed");

		// Cannot drag to another TabContainer that has a different tabs rearrange group.
		tab_container->set_tabs_rearrange_group(1);
		target_tab_container->set_tabs_rearrange_group(2);
		SEND_GUI_MOUSE_BUTTON_EVENT(Point2(side_margin, 0) + tab_rects[0].position, MouseButton::LEFT, MouseButtonMask::LEFT, Key::NONE);
		SEND_GUI_MOUSE_MOTION_EVENT(Point2(side_margin, 0) + tab_rects[0].position + Point2(20, 0), MouseButtonMask::LEFT, Key::NONE);
		SEND_GUI_MOUSE_MOTION_EVENT(target_tab_after_first, MouseButtonMask::LEFT, Key::NONE);
		SIGNAL_CHECK("tab_selected", { { 0 } });
		SIGNAL_CHECK_FALSE("tab_changed");
		CHECK(tab_container->get_viewport()->gui_is_dragging());
		SEND_GUI_MOUSE_BUTTON_RELEASED_EVENT(target_tab_after_first, MouseButton::LEFT, MouseButtonMask::NONE, Key::NONE);
		CHECK_FALSE(tab_container->get_viewport()->gui_is_dragging());
		CHECK(tab_container->get_tab_count() == 3);
		CHECK(target_tab_container->get_tab_count() == 1);
		SIGNAL_CHECK_FALSE("active_tab_rearranged");
		SIGNAL_CHECK_FALSE("tab_selected");
		SIGNAL_CHECK_FALSE("tab_changed");

		// Drag to target container.
		target_tab_container->set_tabs_rearrange_group(1);
		SEND_GUI_MOUSE_BUTTON_EVENT(Point2(side_margin, 0) + tab_rects[0].position, MouseButton::LEFT, MouseButtonMask::LEFT, Key::NONE);
		SEND_GUI_MOUSE_MOTION_EVENT(Point2(side_margin, 0) + tab_rects[0].position + Point2(20, 0), MouseButtonMask::LEFT, Key::NONE);
		SEND_GUI_MOUSE_MOTION_EVENT(target_tab_after_first, MouseButtonMask::LEFT, Key::NONE);
		SIGNAL_CHECK("tab_selected", { { 0 } });
		SIGNAL_CHECK_FALSE("tab_changed");
		CHECK(tab_container->get_viewport()->gui_is_dragging());
		SEND_GUI_MOUSE_BUTTON_RELEASED_EVENT(target_tab_after_first, MouseButton::LEFT, MouseButtonMask::NONE, Key::NONE);
		CHECK_FALSE(tab_container->get_viewport()->gui_is_dragging());
		CHECK(tab_container->get_tab_count() == 2);
		CHECK(target_tab_container->get_tab_count() == 2);
		CHECK(tab_container->get_child_count(false) == 2);
		CHECK(target_tab_container->get_child_count(false) == 2);
		CHECK(tab_container->get_tab_idx_from_control(tab1) == 0);
		CHECK(tab_container->get_tab_idx_from_control(tab2) == 1);
		CHECK(target_tab_container->get_tab_idx_from_control(other_tab0) == 0);
		CHECK(target_tab_container->get_tab_idx_from_control(tab0) == 1);
		CHECK(tab_container->get_current_tab() == 0);
		CHECK(target_tab_container->get_current_tab() == 1);
		SIGNAL_CHECK_FALSE("active_tab_rearranged");
		SIGNAL_CHECK_FALSE("tab_selected"); // Does not send since tab was removed.
		SIGNAL_CHECK("tab_changed", { { 0 } });

		Point2 target_tab = Point2(side_margin, 0) + target_tab_container->get_position();

		// Drag to target container at first index.
		target_tab_container->set_tabs_rearrange_group(1);
		SEND_GUI_MOUSE_BUTTON_EVENT(Point2(side_margin, 0) + tab_rects[0].position, MouseButton::LEFT, MouseButtonMask::LEFT, Key::NONE);
		SEND_GUI_MOUSE_MOTION_EVENT(Point2(side_margin, 0) + tab_rects[0].position + Point2(20, 0), MouseButtonMask::LEFT, Key::NONE);
		SEND_GUI_MOUSE_MOTION_EVENT(target_tab, MouseButtonMask::LEFT, Key::NONE);
		SIGNAL_CHECK("tab_selected", { { 0 } });
		SIGNAL_CHECK_FALSE("tab_changed");
		CHECK(tab_container->get_viewport()->gui_is_dragging());
		SEND_GUI_MOUSE_BUTTON_RELEASED_EVENT(target_tab, MouseButton::LEFT, MouseButtonMask::NONE, Key::NONE);
		CHECK_FALSE(tab_container->get_viewport()->gui_is_dragging());
		CHECK(tab_container->get_tab_count() == 1);
		CHECK(target_tab_container->get_tab_count() == 3);
		CHECK(tab_container->get_tab_idx_from_control(tab2) == 0);
		CHECK(target_tab_container->get_tab_idx_from_control(tab1) == 0);
		CHECK(target_tab_container->get_tab_idx_from_control(other_tab0) == 1);
		CHECK(target_tab_container->get_tab_idx_from_control(tab0) == 2);
		CHECK(tab_container->get_current_tab() == 0);
		CHECK(target_tab_container->get_current_tab() == 0);
		SIGNAL_CHECK_FALSE("active_tab_rearranged");
		SIGNAL_CHECK_FALSE("tab_selected"); // Does not send since tab was removed.
		SIGNAL_CHECK("tab_changed", { { 0 } });

		memdelete(target_tab_container);
	}

	SIGNAL_UNWATCH(tab_container, "active_tab_rearranged");
	SIGNAL_UNWATCH(tab_container, "tab_changed");
	SIGNAL_UNWATCH(tab_container, "tab_clicked");
	SIGNAL_UNWATCH(tab_container, "tab_selected");

	memdelete(tab_container);
}

// FIXME: Add tests for keyboard navigation and other methods.

} // namespace TestTabContainer

#endif // ADVANCED_GUI_DISABLED
