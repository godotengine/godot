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

TEST_CASE("[AnimationPlayer] is & set auto_capture") {
	AnimationPlayer *animation_player = memnew(AnimationPlayer);
	// test default
	CHECK(animation_player->is_auto_capture() == true);

	animation_player->set_auto_capture(false);
	CHECK(animation_player->is_auto_capture() == false);

	animation_player->set_auto_capture(true);
	CHECK(animation_player->is_auto_capture() == true);

	memdelete(animation_player);
}

TEST_CASE("[AnimationPlayer] get & set auto_capture_duration") {
	AnimationPlayer * animation_player = memnew(AnimationPlayer);
	// test default
	CHECK(animation_player->get_auto_capture_duration() == doctest::Approx(-1.0f));

	animation_player->set_auto_capture_duration(4.0f);
	CHECK(animation_player->get_auto_capture_duration() == doctest::Approx(4.0f));

	memdelete(animation_player);
}

TEST_CASE("[AnimationPlayer] get & set auto_capture_transition_type") {
	AnimationPlayer * animation_player = memnew(AnimationPlayer);

	// test default
	CHECK(animation_player->get_auto_capture_transition_type() == Tween::TRANS_LINEAR);

	animation_player->set_auto_capture_transition_type(Tween::TRANS_SINE);
	CHECK(animation_player->get_auto_capture_transition_type() == Tween::TRANS_SINE);

	memdelete(animation_player);
}

TEST_CASE("[AnimationPlayer] get & set auto_capture_ease_type") {
	AnimationPlayer * animation_player = memnew(AnimationPlayer);

	// test default
	CHECK(animation_player->get_auto_capture_ease_type() == Tween::EASE_IN);

	animation_player->set_auto_capture_ease_type(Tween::EASE_OUT);
	CHECK(animation_player->get_auto_capture_ease_type() == Tween::EASE_OUT);

	memdelete(animation_player);
}

TEST_CASE("[AnimationPlayer] get & set speed_scale") {
	AnimationPlayer * animation_player = memnew(AnimationPlayer);

	// test default
	CHECK(animation_player->get_speed_scale() == doctest::Approx(1.0f));

	animation_player->set_speed_scale(2.0f);
	CHECK(animation_player->get_speed_scale() == 2.0f);

	memdelete(animation_player);
}

TEST_CASE("[AnimationPlayer] is & set movie_quit_on_finish_enabled") {
	AnimationPlayer * animation_player = memnew(AnimationPlayer);

	// test default
	CHECK(animation_player->is_movie_quit_on_finish_enabled() == false);

	animation_player->set_movie_quit_on_finish_enabled(true);
	CHECK(animation_player->is_movie_quit_on_finish_enabled() == true);

	animation_player->set_movie_quit_on_finish_enabled(false);
	CHECK(animation_player->is_movie_quit_on_finish_enabled() == false);

	memdelete(animation_player);
}

} // namespace TestAnimationPlayer
