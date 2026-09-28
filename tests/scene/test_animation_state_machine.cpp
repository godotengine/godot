/**************************************************************************/
/*  test_animation_state_machine.cpp                                      */
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

TEST_FORCE_LINK(test_animation_state_machine)

#include "scene/animation/animation_blend_tree.h"
#include "scene/animation/animation_node_state_machine.h"
#include "scene/animation/animation_tree.h"
#include "scene/main/scene_tree.h"
#include "scene/main/window.h"
#include "scene/resources/animation.h"

namespace TestAnimationStateMachine {

typedef AnimationNodeStateMachineTransition Transition;

// Builds a state machine with the states "A", "B" and "C" (each one plays a 2 seconds long animation).
// The machine starts in "A". The transitions between the states are added by the test.
static Ref<AnimationNodeStateMachine> _create_state_machine() {
	Ref<AnimationNodeStateMachine> state_machine;
	state_machine.instantiate();
	for (const String &state : { "A", "B", "C" }) {
		Ref<AnimationNodeAnimation> animation_node;
		animation_node.instantiate();
		animation_node->set_animation(state);
		state_machine->add_node(state, animation_node);
	}

	Ref<Transition> start_transition;
	start_transition.instantiate();
	start_transition->set_advance_mode(Transition::ADVANCE_MODE_AUTO);
	state_machine->add_transition("Start", "A", start_transition);
	return state_machine;
}

static void _add_transition(const Ref<AnimationNodeStateMachine> &p_state_machine, const StringName &p_from, const StringName &p_to, Transition::SwitchMode p_switch_mode, int p_priority, const StringName &p_condition = StringName()) {
	Ref<Transition> transition;
	transition.instantiate();
	transition->set_advance_mode(Transition::ADVANCE_MODE_AUTO);
	transition->set_switch_mode(p_switch_mode);
	transition->set_priority(p_priority);
	transition->set_advance_condition(p_condition);
	p_state_machine->add_transition(p_from, p_to, transition);
}

static AnimationTree *_create_tree(const Ref<AnimationNodeStateMachine> &p_state_machine) {
	Ref<AnimationLibrary> library;
	library.instantiate();
	for (const String &state : { "A", "B", "C" }) {
		Ref<Animation> animation;
		animation.instantiate();
		animation->set_length(2.0);
		library->add_animation(state, animation);
	}

	AnimationTree *tree = memnew(AnimationTree);
	tree->add_animation_library("", library);
	tree->set_root_animation_node(p_state_machine);
	tree->set_callback_mode_process(AnimationMixer::ANIMATION_CALLBACK_MODE_PROCESS_MANUAL);
	SceneTree::get_singleton()->get_root()->add_child(tree);
	tree->set_active(true);
	return tree;
}

static void _advance(AnimationTree *p_tree, double p_seconds) {
	const double step = 0.1;
	for (double time = 0.0; time < p_seconds - CMP_EPSILON; time += step) {
		p_tree->advance(step);
	}
}

static StringName _get_current_state(AnimationTree *p_tree) {
	Ref<AnimationNodeStateMachinePlayback> playback = p_tree->get("parameters/playback");
	REQUIRE(playback.is_valid());
	return playback->get_current_node();
}

static void _free_tree(AnimationTree *p_tree) {
	SceneTree::get_singleton()->get_root()->remove_child(p_tree);
	memdelete(p_tree);
}

TEST_CASE("[SceneTree][AnimationNodeStateMachine] Waiting \"At End\" transition does not block lower priority transitions") {
	Ref<AnimationNodeStateMachine> state_machine = _create_state_machine();
	_add_transition(state_machine, "A", "B", Transition::SWITCH_MODE_AT_END, 1);
	_add_transition(state_machine, "A", "C", Transition::SWITCH_MODE_IMMEDIATE, 2, "go");

	AnimationTree *tree = _create_tree(state_machine);
	_advance(tree, 0.6);
	CHECK(_get_current_state(tree) == StringName("A"));

	tree->set("parameters/conditions/go", true);
	_advance(tree, 0.2);
	CHECK(_get_current_state(tree) == StringName("C"));

	_free_tree(tree);
}

TEST_CASE("[SceneTree][AnimationNodeStateMachine] \"At End\" transition still waits for the end of the state") {
	Ref<AnimationNodeStateMachine> state_machine = _create_state_machine();
	_add_transition(state_machine, "A", "B", Transition::SWITCH_MODE_AT_END, 1);
	_add_transition(state_machine, "A", "C", Transition::SWITCH_MODE_IMMEDIATE, 2, "go");

	AnimationTree *tree = _create_tree(state_machine);
	_advance(tree, 1.5);
	CHECK(_get_current_state(tree) == StringName("A"));

	_advance(tree, 1.0);
	CHECK(_get_current_state(tree) == StringName("B"));

	_free_tree(tree);
}

TEST_CASE("[SceneTree][AnimationNodeStateMachine] Priority is respected between transitions that can be taken") {
	// The result must not depend on the order in which the transitions were added.
	for (const bool preferred_first : { true, false }) {
		Ref<AnimationNodeStateMachine> state_machine = _create_state_machine();
		if (preferred_first) {
			_add_transition(state_machine, "A", "B", Transition::SWITCH_MODE_IMMEDIATE, 1, "go");
			_add_transition(state_machine, "A", "C", Transition::SWITCH_MODE_IMMEDIATE, 2, "go");
		} else {
			_add_transition(state_machine, "A", "C", Transition::SWITCH_MODE_IMMEDIATE, 2, "go");
			_add_transition(state_machine, "A", "B", Transition::SWITCH_MODE_IMMEDIATE, 1, "go");
		}

		AnimationTree *tree = _create_tree(state_machine);
		_advance(tree, 0.6);
		CHECK(_get_current_state(tree) == StringName("A"));

		tree->set("parameters/conditions/go", true);
		_advance(tree, 0.2);
		CHECK(_get_current_state(tree) == StringName("B"));

		_free_tree(tree);
	}
}

TEST_CASE("[SceneTree][AnimationNodeStateMachine] Explicit next() keeps choosing by priority, ignoring which transition is ready") {
	Ref<AnimationNodeStateMachine> state_machine = _create_state_machine();
	_add_transition(state_machine, "A", "B", Transition::SWITCH_MODE_AT_END, 1);
	_add_transition(state_machine, "A", "C", Transition::SWITCH_MODE_IMMEDIATE, 2, "go");

	AnimationTree *tree = _create_tree(state_machine);
	_advance(tree, 0.1);
	CHECK(_get_current_state(tree) == StringName("A"));

	tree->set("parameters/conditions/go", true);
	Ref<AnimationNodeStateMachinePlayback> playback = tree->get("parameters/playback");
	REQUIRE(playback.is_valid());
	playback->next();
	_advance(tree, 0.1);

	CHECK(_get_current_state(tree) == StringName("B"));

	_free_tree(tree);
}

TEST_CASE("[SceneTree][AnimationNodeStateMachine] Equal priority between ready transitions picks the last one declared") {
	Ref<AnimationNodeStateMachine> state_machine = _create_state_machine();
	_add_transition(state_machine, "A", "B", Transition::SWITCH_MODE_IMMEDIATE, 1, "go");
	_add_transition(state_machine, "A", "C", Transition::SWITCH_MODE_IMMEDIATE, 1, "go");

	AnimationTree *tree = _create_tree(state_machine);
	tree->set("parameters/conditions/go", true);
	_advance(tree, 0.1);

	CHECK(_get_current_state(tree) == StringName("C"));

	_free_tree(tree);
}

TEST_CASE("[SceneTree][AnimationNodeStateMachine] A ready \"Sync\" transition is not blocked by a waiting \"At End\" one") {
	Ref<AnimationNodeStateMachine> state_machine = _create_state_machine();
	_add_transition(state_machine, "A", "B", Transition::SWITCH_MODE_AT_END, 1);
	_add_transition(state_machine, "A", "C", Transition::SWITCH_MODE_SYNC, 2, "go");

	AnimationTree *tree = _create_tree(state_machine);
	_advance(tree, 0.6);
	CHECK(_get_current_state(tree) == StringName("A"));

	tree->set("parameters/conditions/go", true);
	_advance(tree, 0.2);
	CHECK(_get_current_state(tree) == StringName("C"));

	_free_tree(tree);
}

} // namespace TestAnimationStateMachine
