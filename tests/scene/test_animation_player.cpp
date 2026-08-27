/**************************************************************************/
/*  test_animation_player.cpp                                             */
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

TEST_FORCE_LINK(test_animation_player)

#include "scene/animation/animation_player.h"
#include "scene/main/scene_tree.h"
#include "scene/main/window.h"
#include "scene/resources/animation.h"
#include "scene/resources/animation_state_event.h"

namespace TestAnimationPlayer {

class MockTestStateEvent : public AnimationStateEvent {
	GDCLASS(MockTestStateEvent, AnimationStateEvent);

public:
	int start_calls = 0;
	int update_calls = 0;
	int end_calls = 0;
	int cancel_calls = 0;

	virtual void start(const Ref<AnimationStateContext> &p_context) override {
		start_calls++;
	}
	virtual void update(const Ref<AnimationStateContext> &p_context, double p_delta) override {
		update_calls++;
	}
	virtual void end(const Ref<AnimationStateContext> &p_context) override {
		end_calls++;
	}
	virtual void cancel(const Ref<AnimationStateContext> &p_context) override {
		cancel_calls++;
	}
};

TEST_CASE("[AnimationPlayer] get & set default_blend_time") {
	AnimationPlayer *animation_player = memnew(AnimationPlayer);
	animation_player->set_default_blend_time(4.0);

	CHECK(animation_player->get_default_blend_time() == doctest::Approx(4.0f));
	memdelete(animation_player);
}

TEST_CASE("[AnimationPlayer] get & set blend_time") {
	String anim1 = "animation1";
	String anim2 = "animation2";
	const Ref<Animation> animation1 = memnew(Animation);
	const Ref<Animation> animation2 = memnew(Animation);
	const Ref<AnimationLibrary> animation_library = memnew(AnimationLibrary);
	animation_library->add_animation(anim1, animation1);
	animation_library->add_animation(anim2, animation2);

	AnimationPlayer *animation_player = memnew(AnimationPlayer);
	animation_player->add_animation_library("", animation_library);

	animation_player->set_blend_time(anim1, anim2, 4.0);
	CHECK(animation_player->get_blend_time(anim1, anim2) == doctest::Approx(4.0f));
	memdelete(animation_player);
}

TEST_CASE("[SceneTree][AnimationPlayer] State Event playback lifecycle") {
	Node *root = memnew(Node);
	SceneTree::get_singleton()->get_root()->add_child(root);

	AnimationPlayer *player = memnew(AnimationPlayer);
	root->add_child(player);
	player->set_root_node(NodePath(".."));

	Ref<Animation> anim = memnew(Animation);
	anim->set_length(1.0);
	int track = anim->add_track(Animation::TYPE_STATE_EVENT);
	anim->track_set_path(track, NodePath("."));

	Ref<MockTestStateEvent> event = memnew(MockTestStateEvent);
	anim->state_event_track_insert_key(track, 0.2, 0.4, event);

	Ref<AnimationLibrary> lib = memnew(AnimationLibrary);
	lib->add_animation("test", anim);
	player->add_animation_library("", lib);

	player->play("test");

	// Step to 0.1s -> Before key
	player->advance(0.1);
	CHECK(event->start_calls == 0);
	CHECK(event->update_calls == 0);
	CHECK(event->end_calls == 0);

	// Step to 0.3s -> Key is now active (0.2s - 0.6s)
	player->advance(0.2);
	CHECK(event->start_calls == 1);
	CHECK(event->update_calls == 0);
	CHECK(event->end_calls == 0);

	// Step to 0.4s -> Key continues active
	player->advance(0.1);
	CHECK(event->start_calls == 1);
	CHECK(event->update_calls == 1);
	CHECK(event->end_calls == 0);

	// Step to 0.7s -> Key finishes naturally
	player->advance(0.3);
	CHECK(event->start_calls == 1);
	CHECK(event->update_calls == 1);
	CHECK(event->end_calls == 1);

	// Test cancellation when seeked away during active window
	player->seek(0.3, true);
	CHECK(event->start_calls == 2);
	player->stop();
	CHECK(event->cancel_calls == 1);

	SceneTree::get_singleton()->get_root()->remove_child(root);
	memdelete(player);
	memdelete(root);
}

} // namespace TestAnimationPlayer
