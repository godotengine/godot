/**************************************************************************/
/*  test_animation_node_state_machine.cpp                                 */
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

TEST_FORCE_LINK(test_animation_node_state_machine)

#include "scene/animation/animation_blend_tree.h"
#include "scene/animation/animation_node_state_machine.h"
#include "scene/animation/animation_tree.h"
#include "scene/main/scene_tree.h"
#include "scene/main/window.h"
#include "scene/scene_string_names.h"

namespace TestAnimationNodeStateMachine {

static Ref<AnimationNodeAnimation> make_animation_node() {
	Ref<AnimationNodeAnimation> node;
	node.instantiate();
	node->set_animation("anim");
	return node;
}

static Ref<AnimationNodeStateMachineTransition> make_transition(AnimationNodeStateMachineTransition::SwitchMode p_switch_mode, int p_priority) {
	Ref<AnimationNodeStateMachineTransition> transition;
	transition.instantiate();
	transition->set_advance_mode(AnimationNodeStateMachineTransition::ADVANCE_MODE_AUTO);
	transition->set_switch_mode(p_switch_mode);
	transition->set_priority(p_priority);
	return transition;
}

static AnimationTree *make_tree(const Ref<AnimationNodeStateMachine> &p_state_machine) {
	Ref<Animation> animation;
	animation.instantiate();
	animation->set_length(1.0);
	Ref<AnimationLibrary> library;
	library.instantiate();
	library->add_animation("anim", animation);

	p_state_machine->add_node("A", make_animation_node());
	p_state_machine->add_transition(SceneStringName(Start), "A", make_transition(AnimationNodeStateMachineTransition::SWITCH_MODE_IMMEDIATE, 1));

	AnimationTree *tree = memnew(AnimationTree);
	tree->add_animation_library("", library);
	tree->set_root_animation_node(p_state_machine);
	tree->set_callback_mode_process(AnimationMixer::ANIMATION_CALLBACK_MODE_PROCESS_MANUAL);
	SceneTree::get_singleton()->get_root()->add_child(tree);
	return tree;
}

static StringName get_current_node(AnimationTree *p_tree) {
	Ref<AnimationNodeStateMachinePlayback> playback = p_tree->get("parameters/playback");
	return playback->get_current_node();
}

TEST_CASE("[SceneTree][AnimationNodeStateMachine] At End transition expression is not evaluated before the end") {
	Ref<AnimationNodeStateMachine> state_machine;
	state_machine.instantiate();
	AnimationTree *tree = make_tree(state_machine);

	state_machine->add_node("B", make_animation_node());
	Ref<AnimationNodeStateMachineTransition> at_end = make_transition(AnimationNodeStateMachineTransition::SWITCH_MODE_AT_END, 1);
	// Counts evaluations on the expression base node (the AnimationTree itself by default).
	at_end->set_advance_expression("set_meta(\"calls\", get_meta(\"calls\", 0) + 1) == null");
	state_machine->add_transition("A", "B", at_end);

	for (int i = 0; i < 5; i++) {
		tree->advance(0.1);
	}
	CHECK_EQ(get_current_node(tree), StringName("A"));
	CHECK_EQ(int(tree->get_meta("calls", 0)), 0);

	for (int i = 0; i < 6; i++) {
		tree->advance(0.1);
	}
	CHECK_EQ(get_current_node(tree), StringName("B"));
	CHECK_GE(int(tree->get_meta("calls", 0)), 1);

	memdelete(tree);
}

TEST_CASE("[SceneTree][AnimationNodeStateMachine] At End transition does not block other transitions before the end") {
	Ref<AnimationNodeStateMachine> state_machine;
	state_machine.instantiate();
	AnimationTree *tree = make_tree(state_machine);

	state_machine->add_node("B", make_animation_node());
	state_machine->add_node("C", make_animation_node());
	state_machine->add_transition("A", "B", make_transition(AnimationNodeStateMachineTransition::SWITCH_MODE_AT_END, 0));
	Ref<AnimationNodeStateMachineTransition> immediate = make_transition(AnimationNodeStateMachineTransition::SWITCH_MODE_IMMEDIATE, 1);
	immediate->set_advance_condition("go_c");
	state_machine->add_transition("A", "C", immediate);

	tree->advance(0.1);
	CHECK_EQ(get_current_node(tree), StringName("A"));

	tree->set("parameters/conditions/go_c", true);
	tree->advance(0.1);
	CHECK_EQ(get_current_node(tree), StringName("C"));

	memdelete(tree);
}

TEST_CASE("[SceneTree][AnimationNodeStateMachine] next() can force an At End transition") {
	Ref<AnimationNodeStateMachine> state_machine;
	state_machine.instantiate();
	AnimationTree *tree = make_tree(state_machine);
	state_machine->add_node("B", make_animation_node());
	state_machine->add_transition("A", "B", make_transition(AnimationNodeStateMachineTransition::SWITCH_MODE_AT_END, 1));

	tree->advance(0.1);
	Ref<AnimationNodeStateMachinePlayback> playback = tree->get("parameters/playback");
	CHECK_EQ(playback->get_current_node(), StringName("A"));

	playback->next();
	tree->advance(0.1);
	CHECK_EQ(playback->get_current_node(), StringName("B"));

	memdelete(tree);
}

} // namespace TestAnimationNodeStateMachine
