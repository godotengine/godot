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

#include "scene/2d/node_2d.h"
#include "scene/animation/animation_player.h"
#include "scene/main/scene_tree.h"
#include "scene/main/window.h"
#include "scene/resources/animation.h"

namespace TestAnimationPlayer {

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

TEST_CASE("[SceneTree][AnimationPlayer] Capture with an unresolved track path") {
	const StringName anim_name = "capture";
	const Ref<Animation> animation = memnew(Animation);
	animation->set_length(1.0);
	int track = animation->add_track(Animation::TYPE_VALUE);
	animation->track_set_path(track, NodePath("Missing:position"));
	animation->value_track_set_update_mode(track, Animation::UPDATE_CAPTURE);
	animation->track_insert_key(track, 0.5, Vector2(1, 1));

	const Ref<AnimationLibrary> animation_library = memnew(AnimationLibrary);
	animation_library->add_animation(anim_name, animation);

	Node2D *owner = memnew(Node2D);
	AnimationPlayer *animation_player = memnew(AnimationPlayer);
	owner->add_child(animation_player);
	SceneTree::get_singleton()->get_root()->add_child(owner);
	animation_player->add_animation_library("", animation_library);

	ERR_PRINT_OFF;
	animation_player->play_with_capture(anim_name, 1.0);
	ERR_PRINT_ON;

	CHECK(animation_player->is_playing());
	memdelete(owner);
}

} // namespace TestAnimationPlayer
