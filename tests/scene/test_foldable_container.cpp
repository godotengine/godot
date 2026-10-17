/**************************************************************************/
/*  test_foldable_container.cpp                                           */
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

TEST_FORCE_LINK(test_foldable_container)

#ifndef ADVANCED_GUI_DISABLED

#include "scene/gui/control.h"
#include "scene/gui/foldable_container.h"
#include "scene/main/scene_tree.h"
#include "scene/main/window.h"

namespace TestFoldableContainer {

TEST_CASE("[SceneTree][FoldableContainer] Default properties") {
	FoldableContainer *foldable_container = memnew(FoldableContainer);
	Window *root = SceneTree::get_singleton()->get_root();
	root->add_child(foldable_container);
	SceneTree::get_singleton()->process(0);

	CHECK_MESSAGE(
			!foldable_container->is_folded(),
			"FoldableContainer is expanded by default.");

	CHECK_MESSAGE(
			foldable_container->get_title().is_empty(),
			"FoldableContainer title is empty by default.");

	CHECK_MESSAGE(
			foldable_container->get_title_alignment() == HORIZONTAL_ALIGNMENT_LEFT,
			"FoldableContainer title alignment is set to left by default.");

	CHECK_MESSAGE(
			foldable_container->get_title_position() == FoldableContainer::POSITION_TOP,
			"FoldableContainer title position is set to top by default.");

	CHECK_MESSAGE(
			foldable_container->get_title_text_direction() == Control::TEXT_DIRECTION_AUTO,
			"FoldableContainer title text direction is set to auto by default.");

	CHECK_MESSAGE(
			foldable_container->get_title_text_overrun_behavior() == TextServer::OVERRUN_NO_TRIMMING,
			"FoldableContainer title overrun behavior is set to no trimming by default.");

	CHECK_MESSAGE(
			foldable_container->get_language().is_empty(),
			"FoldableContainer language is empty by default.");

	CHECK_MESSAGE(
			foldable_container->get_focus_mode() == Control::FOCUS_ALL,
			"FoldableContainer focus mode is set to FOCUS_ALL by default.");

	CHECK_MESSAGE(
			foldable_container->get_mouse_filter() == Control::MOUSE_FILTER_STOP,
			"FoldableContainer mouse filter is set to MOUSE_FILTER_STOP by default.");

	CHECK_MESSAGE(
			foldable_container->get_foldable_group().is_null(),
			"FoldableContainer has no foldable group by default.");

	memdelete(foldable_container);
}

TEST_CASE("[SceneTree][FoldableContainer] Fold and expand behavior") {
	FoldableContainer *foldable_container = memnew(FoldableContainer("Section"));
	Window *root = SceneTree::get_singleton()->get_root();
	root->add_child(foldable_container);

	Control *child_control = memnew(Control);
	child_control->set_custom_minimum_size(Size2(120, 80));
	foldable_container->add_child(child_control);
	foldable_container->set_size(Size2(200, 200));
	SceneTree::get_singleton()->process(0);

	Size2 expanded_minimum_size = foldable_container->get_minimum_size();

	CHECK_MESSAGE(
			child_control->is_visible(),
			"Child control is visible when FoldableContainer is expanded.");

	foldable_container->fold();
	SceneTree::get_singleton()->process(0);

	CHECK_MESSAGE(
			foldable_container->is_folded(),
			"FoldableContainer is folded after calling fold().");

	CHECK_MESSAGE(
			!child_control->is_visible(),
			"Child control is hidden when FoldableContainer is folded.");

	CHECK_MESSAGE(
			foldable_container->get_minimum_size().y < expanded_minimum_size.y,
			"Folded minimum size is smaller than expanded minimum size when content is present.");

	foldable_container->expand();
	SceneTree::get_singleton()->process(0);

	CHECK_MESSAGE(
			!foldable_container->is_folded(),
			"FoldableContainer is expanded after calling expand().");

	CHECK_MESSAGE(
			child_control->is_visible(),
			"Child control is visible again after FoldableContainer is expanded.");

	foldable_container->set_folded(true);
	SceneTree::get_singleton()->process(0);

	CHECK_MESSAGE(
			foldable_container->is_folded(),
			"set_folded(true) folds the FoldableContainer.");

	foldable_container->set_folded(false);
	SceneTree::get_singleton()->process(0);

	CHECK_MESSAGE(
			!foldable_container->is_folded(),
			"set_folded(false) expands the FoldableContainer.");

	memdelete(child_control);
	memdelete(foldable_container);
}

TEST_CASE("[SceneTree][FoldableContainer] Title position and title bar controls") {
	FoldableContainer *foldable_container = memnew(FoldableContainer("Section"));
	Window *root = SceneTree::get_singleton()->get_root();
	root->add_child(foldable_container);

	Control *content_control = memnew(Control);
	content_control->set_custom_minimum_size(Size2(40, 40));
	foldable_container->add_child(content_control);

	Control *title_control = memnew(Control);
	title_control->set_custom_minimum_size(Size2(30, 20));
	foldable_container->add_title_bar_control(title_control);

	foldable_container->set_size(Size2(200, 140));
	SceneTree::get_singleton()->process(0);

	CHECK_MESSAGE(
			title_control->get_parent() == foldable_container,
			"Title bar control is reparented to the FoldableContainer when added.");

	real_t top_title_control_y = title_control->get_position().y;
	real_t top_content_y = content_control->get_position().y;

	foldable_container->set_title_position(FoldableContainer::POSITION_BOTTOM);
	SceneTree::get_singleton()->process(0);

	real_t bottom_title_control_y = title_control->get_position().y;
	real_t bottom_content_y = content_control->get_position().y;

	CHECK_MESSAGE(
			bottom_title_control_y > top_title_control_y,
			"Title bar controls move to the bottom when title position is set to POSITION_BOTTOM.");

	CHECK_MESSAGE(
			bottom_content_y < top_content_y,
			"Content area shifts upward when title position is changed from top to bottom.");

	foldable_container->remove_title_bar_control(title_control);

	CHECK_MESSAGE(
			title_control->get_parent() == nullptr,
			"Title bar control is detached from FoldableContainer when removed.");

	memdelete(title_control);
	memdelete(content_control);
	memdelete(foldable_container);
}

TEST_CASE("[SceneTree][FoldableContainer] FoldableGroup behavior") {
	Ref<FoldableGroup> foldable_group;
	foldable_group.instantiate();

	FoldableContainer *container_a = memnew(FoldableContainer("A"));
	FoldableContainer *container_b = memnew(FoldableContainer("B"));
	Window *root = SceneTree::get_singleton()->get_root();
	root->add_child(container_a);
	root->add_child(container_b);

	container_a->set_folded(true);
	container_b->set_folded(true);
	container_a->set_foldable_group(foldable_group);
	container_b->set_foldable_group(foldable_group);
	SceneTree::get_singleton()->process(0);

	CHECK_MESSAGE(
			!container_a->is_folded(),
			"First grouped FoldableContainer auto-expands when allow_folding_all is false and no container is expanded.");

	CHECK_MESSAGE(
			container_b->is_folded(),
			"Second grouped FoldableContainer remains folded when another container in the group is expanded.");

	CHECK_MESSAGE(
			foldable_group->get_expanded_container() == container_a,
			"FoldableGroup tracks the currently expanded container.");

	container_b->set_folded(false);
	SceneTree::get_singleton()->process(0);

	CHECK_MESSAGE(
			container_a->is_folded(),
			"Expanding one grouped FoldableContainer folds the previously expanded container.");

	CHECK_MESSAGE(
			!container_b->is_folded(),
			"Expanded grouped FoldableContainer remains unfolded.");

	CHECK_MESSAGE(
			foldable_group->get_expanded_container() == container_b,
			"FoldableGroup updates expanded container after a different container is expanded.");

	container_b->set_folded(true);
	SceneTree::get_singleton()->process(0);

	CHECK_MESSAGE(
			!container_b->is_folded(),
			"Expanded grouped FoldableContainer cannot be folded when allow_folding_all is false.");

	foldable_group->set_allow_folding_all(true);
	container_b->set_folded(true);
	SceneTree::get_singleton()->process(0);

	CHECK_MESSAGE(
			container_b->is_folded(),
			"Grouped FoldableContainer can be folded when allow_folding_all is true.");

	CHECK_MESSAGE(
			foldable_group->get_expanded_container() == nullptr,
			"FoldableGroup has no expanded container when all grouped containers are folded.");

	memdelete(container_b);
	memdelete(container_a);
}

} // namespace TestFoldableContainer

#endif
