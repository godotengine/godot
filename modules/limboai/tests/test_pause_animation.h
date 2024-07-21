/**
 * test_pause_animation.h
 * =============================================================================
 * Copyright 2021-2024 Serhii Snitsaruk
 *
 * Use of this source code is governed by an MIT-style
 * license that can be found in the LICENSE file or at
 * https://opensource.org/licenses/MIT.
 * =============================================================================
 */

#ifndef TEST_PAUSE_ANIMATION_H
#define TEST_PAUSE_ANIMATION_H

#include "limbo_test.h"

#include "modules/limboai/bt/tasks/bt_task.h"
#include "modules/limboai/bt/tasks/scene/bt_pause_animation.h"

#include "scene/animation/animation_player.h"
#include "scene/main/window.h"
#include "scene/resources/animation.h"
#include "scene/resources/animation_library.h"

namespace TestPauseAnimation {

TEST_CASE("[SceneTree][LimboAI] BTPauseAnimation") {
	AnimationPlayer *player = memnew(AnimationPlayer);
	SceneTree::get_singleton()->get_root()->add_child(player);
	player->set_callback_mode_process(AnimationMixer::ANIMATION_CALLBACK_MODE_PROCESS_IDLE);

	Ref<AnimationLibrary> anim_lib = memnew(AnimationLibrary);
	Ref<Animation> anim = memnew(Animation);
	anim->set_name("test");
	anim->set_length(0.1);
	anim->set_loop_mode(Animation::LOOP_NONE);
	REQUIRE(anim_lib->add_animation("test", anim) == OK);
	REQUIRE(player->add_animation_library("", anim_lib) == OK);
	REQUIRE(player->has_animation("test"));

	Ref<BTPauseAnimation> pa = memnew(BTPauseAnimation);
	Ref<BBNode> player_param = memnew(BBNode);
	pa->set_animation_player(player_param);
	Node *dummy = memnew(Node);
	SceneTree::get_singleton()->get_root()->add_child(dummy);
	Ref<Blackboard> bb = memnew(Blackboard);

	SUBCASE("When AnimationPlayer doesn't exist") {
		player_param->set_saved_value(NodePath("./NotFound"));
		ERR_PRINT_OFF;
		pa->initialize(dummy, bb, dummy);
		CHECK(pa->execute(0.01666) == BTTask::FAILURE);
		ERR_PRINT_ON;
	}
	SUBCASE("When AnimationPlayer exists") {
		player_param->set_saved_value(player->get_path());
		pa->initialize(dummy, bb, dummy);

		SUBCASE("When AnimationPlayer is not playing") {
			REQUIRE_FALSE(player->is_playing());
			CHECK(pa->execute(0.01666) == BTTask::SUCCESS);
		}
		SUBCASE("When AnimationPlayer is playing") {
			player->play("test");
			REQUIRE(player->is_playing());
			CHECK(pa->execute(0.01666) == BTTask::SUCCESS);
			CHECK_FALSE(player->is_playing());
		}
	}

	memdelete(dummy);
	memdelete(player);
}

} //namespace TestPauseAnimation

#endif // TEST_PAUSE_ANIMATION_H
