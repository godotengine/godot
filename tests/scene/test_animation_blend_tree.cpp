/**************************************************************************/
/*  test_animation_blend_tree.cpp                                         */
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

TEST_FORCE_LINK(test_animation_blend_tree)

#include "scene/2d/node_2d.h"
#include "scene/animation/animation_blend_tree.h"
#include "scene/animation/animation_player.h"
#include "scene/main/scene_tree.h"
#include "scene/main/window.h"

namespace TestAnimationBlendTree {

TEST_CASE("[SceneTree][AnimationBlendTree] Create AnimationBlendTree and add AnimationNode") {
	Ref<AnimationNodeBlendTree> blend_tree;
	blend_tree.instantiate();

	// Test initial state.
	CHECK(blend_tree->has_node("output"));
	CHECK_EQ(blend_tree->get_graph_offset(), Vector2(0, 0));
	CHECK_EQ(blend_tree->get_node_list().size(), 1);

	// Test adding animation node.
	Ref<AnimationNodeAnimation> anim_node;
	anim_node.instantiate();
	anim_node->set_animation(StringName("test_animation"));
	Vector2 position(100, 100);
	blend_tree->add_node("test_node", anim_node, position);

	// Test node existence.
	CHECK(blend_tree->has_node("test_node"));
	CHECK_EQ(blend_tree->get_node("test_node"), anim_node);
	CHECK_EQ(blend_tree->get_node_position("test_node"), position);

	// Test node connection on port 0.
	CHECK_EQ(blend_tree->can_connect_node("output", 0, "test_node"), AnimationNodeBlendTree::CONNECTION_OK);
	blend_tree->connect_node("output", 0, "test_node");

	const LocalVector<StringName> *connections = blend_tree->get_node_connection_array("output");
	CHECK_EQ(connections->size(), 1);
	CHECK_EQ(connections->operator[](0), StringName("test_node"));

	// Test node rename.
	blend_tree->rename_node("test_node", "renamed_node");
	CHECK_FALSE(blend_tree->has_node("test_node"));
	CHECK(blend_tree->has_node("renamed_node"));

	connections = blend_tree->get_node_connection_array("output");
	CHECK_EQ(connections->operator[](0), StringName("renamed_node"));

	// Test node removal.
	blend_tree->remove_node("renamed_node");
	CHECK_FALSE(blend_tree->has_node("renamed_node"));

	connections = blend_tree->get_node_connection_array("output");
	CHECK_EQ(connections->operator[](0), StringName());
}

TEST_CASE("[SceneTree][AnimationBlendTree] AnimationTree with a single AnimationNodeAnimation") {
	// Setup test scene.
	Node2D *node_2d = memnew(Node2D);
	node_2d->set_name("Node2D");
	AnimationTree *animation_tree = memnew(AnimationTree);
	AnimationPlayer *animation_player = memnew(AnimationPlayer);
	SceneTree::get_singleton()->get_root()->add_child(node_2d);
	SceneTree::get_singleton()->get_root()->add_child(animation_player);
	SceneTree::get_singleton()->get_root()->add_child(animation_tree);

	const StringName test_animation_name = "test_animation";

	// Setup test animation.
	Ref<Animation> animation;
	animation.instantiate();
	Ref<AnimationLibrary> animation_library;
	animation_library.instantiate();
	animation_library->add_animation(test_animation_name, animation);
	animation_player->add_animation_library("", animation_library);
	int value_track = animation->add_track(Animation::TrackType::TYPE_VALUE);
	animation->track_set_path(value_track, NodePath("Node2D:position:x"));
	animation->set_length(8.0);
	animation->track_insert_key(value_track, 0.0, 0.0f);
	animation->track_insert_key(value_track, 4.0, 0.2f);
	animation->track_insert_key(value_track, 8.0, 1.0f);

	// Setup animation node.
	Ref<AnimationNodeAnimation> anim_node;
	anim_node.instantiate();
	anim_node->set_animation(test_animation_name);

	// Setup animation tree.
	animation_tree->set_animation_player(animation_tree->get_path_to(animation_player));
	animation_tree->set_root_animation_node(anim_node);
	animation_tree->set_active(true);
	// Skip a frame to reset the animation tree.
	SceneTree::get_singleton()->process(0.0);

	// Note: "t" refers to the current playback position in the original animation
	// in the following comments.
	SUBCASE("Continuous update, linear interpolation") {
		// Notable timestamps:
		// t = 2.0 -> position.x = 0.1
		// t = 3.0 -> position.x = 0.15
		// t = 5.0 -> position.x = 0.4
		// t = 6.0 -> position.x = 0.6
		animation->value_track_set_update_mode(value_track, Animation::UPDATE_CONTINUOUS);
		animation->track_set_interpolation_type(value_track, Animation::INTERPOLATION_LINEAR);

		anim_node->set_use_custom_timeline(false);
		anim_node->set_stretch_time_scale(false);
		SUBCASE("Forward playback") {
			anim_node->set_play_mode(AnimationNodeAnimation::PLAY_MODE_FORWARD);

			SceneTree::get_singleton()->process(2.0);
			CHECK(node_2d->get_position().x == doctest::Approx(0.1f)); // t = 2.0
			SceneTree::get_singleton()->process(4.0);
			CHECK(node_2d->get_position().x == doctest::Approx(0.6f)); // t = 6.0
			SceneTree::get_singleton()->process(4.0);
			CHECK(node_2d->get_position().x == doctest::Approx(1.0f)); // t = 8.0
		}

		SUBCASE("Backward playback") {
			anim_node->set_play_mode(AnimationNodeAnimation::PLAY_MODE_BACKWARD);

			SceneTree::get_singleton()->process(2.0);
			CHECK(node_2d->get_position().x == doctest::Approx(0.6f)); // t = 6.0
			SceneTree::get_singleton()->process(4.0);
			CHECK(node_2d->get_position().x == doctest::Approx(0.1f)); // t = 2.0
			SceneTree::get_singleton()->process(4.0);
			CHECK(node_2d->get_position().x == doctest::Approx(0.0f)); // t = 0.0
		}

		SUBCASE("Custom timeline") {
			anim_node->set_use_custom_timeline(true);

			anim_node->set_stretch_time_scale(false);
			anim_node->set_start_offset(2.0);
			anim_node->set_timeline_length(4.0);
			SUBCASE("Forward playback") {
				anim_node->set_play_mode(AnimationNodeAnimation::PLAY_MODE_FORWARD);
				SUBCASE("No looping") {
					anim_node->set_loop_mode(Animation::LOOP_NONE);

					SceneTree::get_singleton()->process(1.0);
					CHECK(node_2d->get_position().x == doctest::Approx(0.15f)); // t = 3.0
					SceneTree::get_singleton()->process(2.0);
					CHECK(node_2d->get_position().x == doctest::Approx(0.4f)); // t = 5.0
					SceneTree::get_singleton()->process(2.0);
					CHECK(node_2d->get_position().x == doctest::Approx(0.6f)); // t = 6.0
				}
				SUBCASE("Linear looping") {
					anim_node->set_loop_mode(Animation::LOOP_LINEAR);

					SceneTree::get_singleton()->process(2.0);
					CHECK(node_2d->get_position().x == doctest::Approx(0.2f)); // t = 4.0
					SceneTree::get_singleton()->process(1.0);
					CHECK(node_2d->get_position().x == doctest::Approx(0.4f)); // t = 5.0
					SceneTree::get_singleton()->process(2.0);
					CHECK(node_2d->get_position().x == doctest::Approx(0.15f)); // t = 6.0
				}
				SUBCASE("Ping Pong looping") {
					anim_node->set_loop_mode(Animation::LOOP_PINGPONG);

					SceneTree::get_singleton()->process(1.0);
					CHECK(node_2d->get_position().x == doctest::Approx(0.15f)); // t = 3.0
					SceneTree::get_singleton()->process(2.0);
					CHECK(node_2d->get_position().x == doctest::Approx(0.4f)); // t = 5.0
					SceneTree::get_singleton()->process(2.0);
					CHECK(node_2d->get_position().x == doctest::Approx(0.4f)); // t = 5.0
					SceneTree::get_singleton()->process(2.0);
					CHECK(node_2d->get_position().x == doctest::Approx(0.15f)); // t = 3.0
				}
			}
			SUBCASE("Backward playback") {
				anim_node->set_play_mode(AnimationNodeAnimation::PLAY_MODE_BACKWARD);
				SUBCASE("No looping") {
					anim_node->set_loop_mode(Animation::LOOP_NONE);

					SceneTree::get_singleton()->process(1.0);
					CHECK(node_2d->get_position().x == doctest::Approx(0.4f)); // t = 5.0
					SceneTree::get_singleton()->process(2.0);
					CHECK(node_2d->get_position().x == doctest::Approx(0.15f)); // t = 3.0
					SceneTree::get_singleton()->process(2.0);
					CHECK(node_2d->get_position().x == doctest::Approx(0.1f)); // t = 2.0
				}
				SUBCASE("Linear looping") {
					anim_node->set_loop_mode(Animation::LOOP_LINEAR);

					SceneTree::get_singleton()->process(2.0);
					CHECK(node_2d->get_position().x == doctest::Approx(0.2f)); // t = 4.0
					SceneTree::get_singleton()->process(1.0);
					CHECK(node_2d->get_position().x == doctest::Approx(0.15f)); // t = 3.0
					SceneTree::get_singleton()->process(2.0);
					CHECK(node_2d->get_position().x == doctest::Approx(0.4f)); // t = 5.0
				}
				SUBCASE("Ping Pong looping") {
					anim_node->set_loop_mode(Animation::LOOP_PINGPONG);

					SceneTree::get_singleton()->process(1.0);
					CHECK(node_2d->get_position().x == doctest::Approx(0.4f)); // t = 5.0
					SceneTree::get_singleton()->process(2.0);
					CHECK(node_2d->get_position().x == doctest::Approx(0.15f)); // t = 3.0
					SceneTree::get_singleton()->process(2.0);
					CHECK(node_2d->get_position().x == doctest::Approx(0.15f)); // t = 3.0
					SceneTree::get_singleton()->process(2.0);
					CHECK(node_2d->get_position().x == doctest::Approx(0.4f)); // t = 5.0
				}
				SUBCASE("With asymmetrical section, no looping") {
					anim_node->set_start_offset(0.0);
					anim_node->set_timeline_length(4.0);
					anim_node->set_loop_mode(Animation::LOOP_NONE);

					SceneTree::get_singleton()->process(2.0);
					CHECK(node_2d->get_position().x == doctest::Approx(0.6f)); // t = 6.0
					SceneTree::get_singleton()->process(1.0);
					CHECK(node_2d->get_position().x == doctest::Approx(0.4f)); // t = 5.0
					SceneTree::get_singleton()->process(2.0);
					CHECK(node_2d->get_position().x == doctest::Approx(0.2f)); // t = 4.0
				}
			}
			SUBCASE("Using Stretch Timelime Scale") {
				anim_node->set_start_offset(2.0);
				anim_node->set_timeline_length(4.0);
				// 1. start_offset should be ignored.
				// 2. timeline_length is exactly half the animation length, so the animation should be played in double speed.
				anim_node->set_stretch_time_scale(true);
				SUBCASE("Forward playback") {
					anim_node->set_play_mode(AnimationNodeAnimation::PLAY_MODE_FORWARD);
					SUBCASE("No looping") {
						anim_node->set_loop_mode(Animation::LOOP_NONE);

						SceneTree::get_singleton()->process(1.0);
						CHECK(node_2d->get_position().x == doctest::Approx(0.1f)); // t = 2.0
						SceneTree::get_singleton()->process(2.0);
						CHECK(node_2d->get_position().x == doctest::Approx(0.6f)); // t = 6.0
						SceneTree::get_singleton()->process(2.0);
						CHECK(node_2d->get_position().x == doctest::Approx(1.0f)); // t = 8.0
					}
					SUBCASE("Linear looping") {
						anim_node->set_loop_mode(Animation::LOOP_LINEAR);

						SceneTree::get_singleton()->process(1.0);
						CHECK(node_2d->get_position().x == doctest::Approx(0.1f)); // t = 2.0
						SceneTree::get_singleton()->process(2.0);
						CHECK(node_2d->get_position().x == doctest::Approx(0.6f)); // t = 6.0
						SceneTree::get_singleton()->process(2.0);
						CHECK(node_2d->get_position().x == doctest::Approx(0.1f)); // t = 2.0
					}
					SUBCASE("Ping Pong looping") {
						anim_node->set_loop_mode(Animation::LOOP_PINGPONG);

						SceneTree::get_singleton()->process(1.0);
						CHECK(node_2d->get_position().x == doctest::Approx(0.1f)); // t = 2.0
						SceneTree::get_singleton()->process(2.0);
						CHECK(node_2d->get_position().x == doctest::Approx(0.6f)); // t = 6.0
						SceneTree::get_singleton()->process(2.0);
						CHECK(node_2d->get_position().x == doctest::Approx(0.6f)); // t = 6.0
						SceneTree::get_singleton()->process(2.0);
						CHECK(node_2d->get_position().x == doctest::Approx(0.1f)); // t = 2.0
					}
				}

				SUBCASE("Backward playback") {
					anim_node->set_play_mode(AnimationNodeAnimation::PLAY_MODE_BACKWARD);
					SUBCASE("No looping") {
						anim_node->set_loop_mode(Animation::LOOP_NONE);

						SceneTree::get_singleton()->process(1.0);
						CHECK(node_2d->get_position().x == doctest::Approx(0.6f)); // t = 6.0
						SceneTree::get_singleton()->process(2.0);
						CHECK(node_2d->get_position().x == doctest::Approx(0.1f)); // t = 2.0
						SceneTree::get_singleton()->process(2.0);
						CHECK(node_2d->get_position().x == doctest::Approx(0.0f)); // t = 0.0
					}
					SUBCASE("Linear looping") {
						anim_node->set_loop_mode(Animation::LOOP_LINEAR);

						SceneTree::get_singleton()->process(1.0);
						CHECK(node_2d->get_position().x == doctest::Approx(0.6f)); // t = 6.0
						SceneTree::get_singleton()->process(2.0);
						CHECK(node_2d->get_position().x == doctest::Approx(0.1f)); // t = 2.0
						SceneTree::get_singleton()->process(2.0);
						CHECK(node_2d->get_position().x == doctest::Approx(0.6f)); // t = 6.0
					}
					SUBCASE("Ping Pong looping") {
						anim_node->set_loop_mode(Animation::LOOP_PINGPONG);

						SceneTree::get_singleton()->process(1.0);
						CHECK(node_2d->get_position().x == doctest::Approx(0.6f)); // t = 6.0
						SceneTree::get_singleton()->process(2.0);
						CHECK(node_2d->get_position().x == doctest::Approx(0.1f)); // t = 2.0
						SceneTree::get_singleton()->process(2.0);
						CHECK(node_2d->get_position().x == doctest::Approx(0.1f)); // t = 2.0
						SceneTree::get_singleton()->process(2.0);
						CHECK(node_2d->get_position().x == doctest::Approx(0.6f)); // t = 6.0
					}
				}
			}
		}
	}

	SUBCASE("Continuous update, nearest interpolation") {
		// When using nearest interpolation, the value of a track at position t is chosen as
		// the value of the nearest keyframe to t whose position is lesser than t.
		// This does not take in account of the playback direction, (e.g., when changing directions in ping pong looping)
		// so we can deduce the answers for each position in advance:
		// t = 1.0 -> position.x = 0.0
		// t = 2.0 -> position.x = 0.0
		// t = 3.0 -> position.x = 0.0
		// t = 5.0 -> position.x = 4.0
		// t = 6.0 -> position.x = 4.0
		// t = 7.0 -> position.x = 4.0
		// t = 8.0 -> position.x = 8.0
		animation->value_track_set_update_mode(value_track, Animation::UPDATE_CONTINUOUS);
		animation->track_set_interpolation_type(value_track, Animation::INTERPOLATION_NEAREST);

		anim_node->set_use_custom_timeline(false);
		anim_node->set_stretch_time_scale(false);
		SUBCASE("Forward playback") {
			anim_node->set_play_mode(AnimationNodeAnimation::PLAY_MODE_FORWARD);

			SceneTree::get_singleton()->process(1.0);
			CHECK(node_2d->get_position().x == doctest::Approx(0.0f)); // t = 1.0
			SceneTree::get_singleton()->process(2.0);
			CHECK(node_2d->get_position().x == doctest::Approx(0.0f)); // t = 3.0
			SceneTree::get_singleton()->process(2.0);
			CHECK(node_2d->get_position().x == doctest::Approx(0.2f)); // t = 5.0
			SceneTree::get_singleton()->process(2.0);
			CHECK(node_2d->get_position().x == doctest::Approx(0.2f)); // t = 7.0
			SceneTree::get_singleton()->process(2.0);
			CHECK(node_2d->get_position().x == doctest::Approx(1.0f)); // t = 8.0
		}

		SUBCASE("Backward playback") {
			anim_node->set_play_mode(AnimationNodeAnimation::PLAY_MODE_BACKWARD);

			SceneTree::get_singleton()->process(1.0);
			CHECK(node_2d->get_position().x == doctest::Approx(0.2f)); // t = 7.0
			SceneTree::get_singleton()->process(2.0);
			CHECK(node_2d->get_position().x == doctest::Approx(0.2f)); // t = 5.0
			SceneTree::get_singleton()->process(2.0);
			CHECK(node_2d->get_position().x == doctest::Approx(0.0f)); // t = 3.0
			SceneTree::get_singleton()->process(2.0);
			CHECK(node_2d->get_position().x == doctest::Approx(0.0f)); // t = 1.0
			SceneTree::get_singleton()->process(2.0);
			CHECK(node_2d->get_position().x == doctest::Approx(0.0f)); // t = 0.0
		}

		SUBCASE("Custom timeline") {
			anim_node->set_use_custom_timeline(true);

			anim_node->set_stretch_time_scale(false);
			anim_node->set_start_offset(1.0);
			anim_node->set_timeline_length(6.0);
			SUBCASE("Forward playback") {
				anim_node->set_play_mode(AnimationNodeAnimation::PLAY_MODE_FORWARD);
				SUBCASE("No looping") {
					anim_node->set_loop_mode(Animation::LOOP_NONE);

					SceneTree::get_singleton()->process(2.0);
					CHECK(node_2d->get_position().x == doctest::Approx(0.0f)); // t = 3.0
					SceneTree::get_singleton()->process(2.0);
					CHECK(node_2d->get_position().x == doctest::Approx(0.2f)); // t = 5.0
					SceneTree::get_singleton()->process(3.0);
					CHECK(node_2d->get_position().x == doctest::Approx(0.2f)); // t = 7.0
				}
				SUBCASE("Linear looping") {
					anim_node->set_loop_mode(Animation::LOOP_LINEAR);

					SceneTree::get_singleton()->process(2.0);
					CHECK(node_2d->get_position().x == doctest::Approx(0.0f)); // t = 3.0
					SceneTree::get_singleton()->process(2.0);
					CHECK(node_2d->get_position().x == doctest::Approx(0.2f)); // t = 5.0
					SceneTree::get_singleton()->process(3.0);
					CHECK(node_2d->get_position().x == doctest::Approx(0.0f)); // t = 1.0
				}
				SUBCASE("Ping Pong looping") {
					anim_node->set_loop_mode(Animation::LOOP_PINGPONG);

					SceneTree::get_singleton()->process(2.0);
					CHECK(node_2d->get_position().x == doctest::Approx(0.0f)); // t = 3.0
					SceneTree::get_singleton()->process(2.0);
					CHECK(node_2d->get_position().x == doctest::Approx(0.2f)); // t = 5.0
					SceneTree::get_singleton()->process(3.0);
					CHECK(node_2d->get_position().x == doctest::Approx(0.2f)); // t = 6.0
					SceneTree::get_singleton()->process(3.0);
					CHECK(node_2d->get_position().x == doctest::Approx(0.0f)); // t = 3.0
				}
			}
			SUBCASE("Backward playback") {
				anim_node->set_play_mode(AnimationNodeAnimation::PLAY_MODE_BACKWARD);
				SUBCASE("No looping") {
					anim_node->set_loop_mode(Animation::LOOP_NONE);

					SceneTree::get_singleton()->process(2.0);
					CHECK(node_2d->get_position().x == doctest::Approx(0.2f)); // t = 5.0
					SceneTree::get_singleton()->process(2.0);
					CHECK(node_2d->get_position().x == doctest::Approx(0.0f)); // t = 3.0
					SceneTree::get_singleton()->process(3.0);
					CHECK(node_2d->get_position().x == doctest::Approx(0.0f)); // t = 1.0
				}
				SUBCASE("Linear looping") {
					anim_node->set_loop_mode(Animation::LOOP_LINEAR);

					SceneTree::get_singleton()->process(2.0);
					CHECK(node_2d->get_position().x == doctest::Approx(0.2f)); // t = 5.0
					SceneTree::get_singleton()->process(2.0);
					CHECK(node_2d->get_position().x == doctest::Approx(0.0f)); // t = 3.0
					SceneTree::get_singleton()->process(3.0);
					CHECK(node_2d->get_position().x == doctest::Approx(0.2f)); // t = 6.0
				}
				SUBCASE("Ping Pong looping") {
					anim_node->set_loop_mode(Animation::LOOP_PINGPONG);

					SceneTree::get_singleton()->process(2.0);
					CHECK(node_2d->get_position().x == doctest::Approx(0.2f)); // t = 5.0
					SceneTree::get_singleton()->process(2.0);
					CHECK(node_2d->get_position().x == doctest::Approx(0.0f)); // t = 3.0
					SceneTree::get_singleton()->process(3.0);
					CHECK(node_2d->get_position().x == doctest::Approx(0.0f)); // t = 2.0
					SceneTree::get_singleton()->process(3.0);
					CHECK(node_2d->get_position().x == doctest::Approx(0.2f)); // t = 5.0
				}
				SUBCASE("With asymmetrical section, no looping") {
					anim_node->set_start_offset(0.0);
					anim_node->set_timeline_length(7.0);
					anim_node->set_loop_mode(Animation::LOOP_NONE);

					SceneTree::get_singleton()->process(1.0);
					CHECK(node_2d->get_position().x == doctest::Approx(0.2f)); // t = 7.0
					SceneTree::get_singleton()->process(2.0);
					CHECK(node_2d->get_position().x == doctest::Approx(0.2f)); // t = 5.0
					SceneTree::get_singleton()->process(2.0);
					CHECK(node_2d->get_position().x == doctest::Approx(0.0f)); // t = 3.0
					SceneTree::get_singleton()->process(4.0);
					CHECK(node_2d->get_position().x == doctest::Approx(0.0f)); // t = 1.0
				}
			}
			SUBCASE("Using Stretch Timelime Scale") {
				anim_node->set_start_offset(2.0);
				anim_node->set_timeline_length(4.0);
				// 1. start_offset should be ignored.
				// 2. timeline_length is exactly half the animation length, so the animation should be played in double speed.
				anim_node->set_stretch_time_scale(true);
				SUBCASE("Forward playback") {
					anim_node->set_play_mode(AnimationNodeAnimation::PLAY_MODE_FORWARD);
					SUBCASE("No looping") {
						anim_node->set_loop_mode(Animation::LOOP_NONE);

						SceneTree::get_singleton()->process(0.5);
						CHECK(node_2d->get_position().x == doctest::Approx(0.0f)); // t = 1.0
						SceneTree::get_singleton()->process(1.0);
						CHECK(node_2d->get_position().x == doctest::Approx(0.0f)); // t = 3.0
						SceneTree::get_singleton()->process(1.0);
						CHECK(node_2d->get_position().x == doctest::Approx(0.2f)); // t = 5.0
						SceneTree::get_singleton()->process(1.0);
						CHECK(node_2d->get_position().x == doctest::Approx(0.2f)); // t = 7.0
						SceneTree::get_singleton()->process(1.0);
						CHECK(node_2d->get_position().x == doctest::Approx(1.0f)); // t = 8.0
					}
					SUBCASE("Linear looping") {
						anim_node->set_loop_mode(Animation::LOOP_LINEAR);

						SceneTree::get_singleton()->process(0.5);
						CHECK(node_2d->get_position().x == doctest::Approx(0.0f)); // t = 1.0
						SceneTree::get_singleton()->process(1.0);
						CHECK(node_2d->get_position().x == doctest::Approx(0.0f)); // t = 3.0
						SceneTree::get_singleton()->process(1.0);
						CHECK(node_2d->get_position().x == doctest::Approx(0.2f)); // t = 5.0
						SceneTree::get_singleton()->process(1.0);
						CHECK(node_2d->get_position().x == doctest::Approx(0.2f)); // t = 7.0
						SceneTree::get_singleton()->process(1.0);
						CHECK(node_2d->get_position().x == doctest::Approx(0.0f)); // t = 1.0
					}
					SUBCASE("Ping Pong looping") {
						anim_node->set_loop_mode(Animation::LOOP_PINGPONG);

						SceneTree::get_singleton()->process(0.5);
						CHECK(node_2d->get_position().x == doctest::Approx(0.0f)); // t = 1.0
						SceneTree::get_singleton()->process(1.0);
						CHECK(node_2d->get_position().x == doctest::Approx(0.0f)); // t = 3.0
						SceneTree::get_singleton()->process(1.0);
						CHECK(node_2d->get_position().x == doctest::Approx(0.2f)); // t = 5.0
						SceneTree::get_singleton()->process(1.0);
						CHECK(node_2d->get_position().x == doctest::Approx(0.2f)); // t = 7.0
						SceneTree::get_singleton()->process(1.0);
						CHECK(node_2d->get_position().x == doctest::Approx(0.2f)); // t = 7.0
					}
				}

				SUBCASE("Backward playback") {
					anim_node->set_play_mode(AnimationNodeAnimation::PLAY_MODE_BACKWARD);
					SUBCASE("No looping") {
						anim_node->set_loop_mode(Animation::LOOP_NONE);

						SceneTree::get_singleton()->process(0.5);
						CHECK(node_2d->get_position().x == doctest::Approx(0.2f)); // t = 7.0
						SceneTree::get_singleton()->process(1.0);
						CHECK(node_2d->get_position().x == doctest::Approx(0.2f)); // t = 5.0
						SceneTree::get_singleton()->process(1.0);
						CHECK(node_2d->get_position().x == doctest::Approx(0.0f)); // t = 3.0
						SceneTree::get_singleton()->process(1.0);
						CHECK(node_2d->get_position().x == doctest::Approx(0.0f)); // t = 1.0
						SceneTree::get_singleton()->process(1.0);
						CHECK(node_2d->get_position().x == doctest::Approx(0.0f)); // t = 0.0
					}
					SUBCASE("Linear looping") {
						anim_node->set_loop_mode(Animation::LOOP_LINEAR);

						SceneTree::get_singleton()->process(0.5);
						CHECK(node_2d->get_position().x == doctest::Approx(0.2f)); // t = 7.0
						SceneTree::get_singleton()->process(1.0);
						CHECK(node_2d->get_position().x == doctest::Approx(0.2f)); // t = 5.0
						SceneTree::get_singleton()->process(1.0);
						CHECK(node_2d->get_position().x == doctest::Approx(0.0f)); // t = 3.0
						SceneTree::get_singleton()->process(1.0);
						CHECK(node_2d->get_position().x == doctest::Approx(0.0f)); // t = 1.0
						SceneTree::get_singleton()->process(1.0);
						CHECK(node_2d->get_position().x == doctest::Approx(0.2f)); // t = 7.0
					}
					SUBCASE("Ping Pong looping") {
						anim_node->set_loop_mode(Animation::LOOP_PINGPONG);

						SceneTree::get_singleton()->process(0.5);
						CHECK(node_2d->get_position().x == doctest::Approx(0.2f)); // t = 7.0
						SceneTree::get_singleton()->process(1.0);
						CHECK(node_2d->get_position().x == doctest::Approx(0.2f)); // t = 5.0
						SceneTree::get_singleton()->process(1.0);
						CHECK(node_2d->get_position().x == doctest::Approx(0.0f)); // t = 3.0
						SceneTree::get_singleton()->process(1.0);
						CHECK(node_2d->get_position().x == doctest::Approx(0.0f)); // t = 1.0
						SceneTree::get_singleton()->process(1.0);
						CHECK(node_2d->get_position().x == doctest::Approx(0.0f)); // t = 1.0
					}
				}
			}
		}
	}

	SUBCASE("Discrete update") {
		animation->value_track_set_update_mode(value_track, Animation::UPDATE_DISCRETE);
		// AnimationTree by default uses ANIMATION_CALLBACK_MODE_DISCRETE_FORCE_CONTINUOUS,
		// which just treats discrete update as continuous update + nearest interpolation.
		// Note that the test cases aren't completely identical to above.
		// Notably, the latest keyframe also respects the playback direction.
		// This makes it impossible to deduce answers for all positions in advance.
		// In the test cases below, comments are added to indicate the latest keyframe's timestamp.
		// ("Latest key is t = XX" refers to the position and not the value)
		// When the playback direction is flipped in ping pong looping, it is also annotated explicitly in the comments.
		CHECK(animation_tree->get_callback_mode_discrete() == AnimationMixer::ANIMATION_CALLBACK_MODE_DISCRETE_FORCE_CONTINUOUS);

		anim_node->set_use_custom_timeline(false);
		anim_node->set_stretch_time_scale(false);
		SUBCASE("Forward playback") {
			anim_node->set_play_mode(AnimationNodeAnimation::PLAY_MODE_FORWARD);

			SceneTree::get_singleton()->process(1.0);
			CHECK(node_2d->get_position().x == doctest::Approx(0.0f)); // t = 1.0 (Latest key is t = 0.0)
			SceneTree::get_singleton()->process(2.0);
			CHECK(node_2d->get_position().x == doctest::Approx(0.0f)); // t = 3.0 (Latest key is t = 0.0)
			SceneTree::get_singleton()->process(2.0);
			CHECK(node_2d->get_position().x == doctest::Approx(0.2f)); // t = 5.0 (Latest key is t = 4.0)
			SceneTree::get_singleton()->process(2.0);
			CHECK(node_2d->get_position().x == doctest::Approx(0.2f)); // t = 7.0 (Latest key is t = 4.0)
			SceneTree::get_singleton()->process(2.0);
			CHECK(node_2d->get_position().x == doctest::Approx(1.0f)); // t = 8.0 (Latest key is t = 8.0)
		}

		SUBCASE("Backward playback") {
			anim_node->set_play_mode(AnimationNodeAnimation::PLAY_MODE_BACKWARD);

			SceneTree::get_singleton()->process(1.0);
			CHECK(node_2d->get_position().x == doctest::Approx(1.0f)); // t = 7.0 (Latest key is t = 8.0)
			SceneTree::get_singleton()->process(2.0);
			CHECK(node_2d->get_position().x == doctest::Approx(1.0f)); // t = 5.0 (Latest key is t = 8.0)
			SceneTree::get_singleton()->process(2.0);
			CHECK(node_2d->get_position().x == doctest::Approx(0.2f)); // t = 3.0 (Latest key is t = 4.0)
			SceneTree::get_singleton()->process(2.0);
			CHECK(node_2d->get_position().x == doctest::Approx(0.2f)); // t = 1.0 (Latest key is t = 4.0)
			SceneTree::get_singleton()->process(2.0);
			CHECK(node_2d->get_position().x == doctest::Approx(0.0f)); // t = 0.0 (Latest key is t = 0.0)
		}

		SUBCASE("Custom timeline") {
			anim_node->set_use_custom_timeline(true);

			anim_node->set_stretch_time_scale(false);
			anim_node->set_start_offset(1.0);
			anim_node->set_timeline_length(6.0);
			SUBCASE("Forward playback") {
				anim_node->set_play_mode(AnimationNodeAnimation::PLAY_MODE_FORWARD);
				SUBCASE("No looping") {
					anim_node->set_loop_mode(Animation::LOOP_NONE);

					SceneTree::get_singleton()->process(2.0);
					CHECK(node_2d->get_position().x == doctest::Approx(0.0f)); // t = 3.0 (Latest key is t = 0.0)
					SceneTree::get_singleton()->process(2.0);
					CHECK(node_2d->get_position().x == doctest::Approx(0.2f)); // t = 5.0 (Latest key is t = 4.0)
					SceneTree::get_singleton()->process(3.0);
					CHECK(node_2d->get_position().x == doctest::Approx(0.2f)); // t = 7.0 (Latest key is t = 4.0)
				}
				SUBCASE("Linear looping") {
					anim_node->set_loop_mode(Animation::LOOP_LINEAR);

					SceneTree::get_singleton()->process(2.0);
					CHECK(node_2d->get_position().x == doctest::Approx(0.0f)); // t = 3.0 (Latest key is t = 0.0)
					SceneTree::get_singleton()->process(2.0);
					CHECK(node_2d->get_position().x == doctest::Approx(0.2f)); // t = 5.0 (Latest key is t = 4.0)
					SceneTree::get_singleton()->process(3.0);
					CHECK(node_2d->get_position().x == doctest::Approx(0.0f)); // t = 1.0 (Latest key is t = 0.0)
				}
				SUBCASE("Ping Pong looping") {
					anim_node->set_loop_mode(Animation::LOOP_PINGPONG);

					SceneTree::get_singleton()->process(2.0);
					CHECK(node_2d->get_position().x == doctest::Approx(0.0f)); // t = 3.0 (Latest key is t = 0.0)
					SceneTree::get_singleton()->process(2.0);
					CHECK(node_2d->get_position().x == doctest::Approx(0.2f)); // t = 5.0 (Latest key is t = 4.0)
					SceneTree::get_singleton()->process(3.0); // Direction flipped.
					CHECK(node_2d->get_position().x == doctest::Approx(1.0f)); // t = 6.0 (Latest key is t = 8.0)
					SceneTree::get_singleton()->process(3.0);
					CHECK(node_2d->get_position().x == doctest::Approx(0.2f)); // t = 3.0 (Latest key is t = 4.0)
				}
			}
			SUBCASE("Backward playback") {
				anim_node->set_play_mode(AnimationNodeAnimation::PLAY_MODE_BACKWARD);
				SUBCASE("No looping") {
					anim_node->set_loop_mode(Animation::LOOP_NONE);

					SceneTree::get_singleton()->process(2.0);
					CHECK(node_2d->get_position().x == doctest::Approx(1.0f)); // t = 5.0 (Latest key is t = 8.0)
					SceneTree::get_singleton()->process(2.0);
					CHECK(node_2d->get_position().x == doctest::Approx(0.2f)); // t = 3.0 (Latest key is t = 4.0)
					SceneTree::get_singleton()->process(3.0);
					CHECK(node_2d->get_position().x == doctest::Approx(0.2f)); // t = 1.0 (Latest key is t = 4.0)
				}
				SUBCASE("Linear looping") {
					anim_node->set_loop_mode(Animation::LOOP_LINEAR);

					SceneTree::get_singleton()->process(2.0);
					CHECK(node_2d->get_position().x == doctest::Approx(1.0f)); // t = 5.0 (Latest key is t = 8.0)
					SceneTree::get_singleton()->process(2.0);
					CHECK(node_2d->get_position().x == doctest::Approx(0.2f)); // t = 3.0 (Latest key is t = 4.0)
					SceneTree::get_singleton()->process(3.0);
					CHECK(node_2d->get_position().x == doctest::Approx(1.0f)); // t = 6.0 (Latest key is t = 8.0)
				}
				SUBCASE("Ping Pong looping") {
					anim_node->set_loop_mode(Animation::LOOP_PINGPONG);

					SceneTree::get_singleton()->process(2.0);
					CHECK(node_2d->get_position().x == doctest::Approx(1.0f)); // t = 5.0 (Latest key is t = 8.0)
					SceneTree::get_singleton()->process(2.0);
					CHECK(node_2d->get_position().x == doctest::Approx(0.2f)); // t = 3.0 (Latest key is t = 4.0)
					SceneTree::get_singleton()->process(3.0); // Direction flipped.
					CHECK(node_2d->get_position().x == doctest::Approx(0.0f)); // t = 2.0 (Latest key is t = 0.0)
					SceneTree::get_singleton()->process(3.0);
					CHECK(node_2d->get_position().x == doctest::Approx(0.2f)); // t = 5.0 (Latest key is t = 4.0)
				}
				SUBCASE("With asymmetrical section, no looping") {
					anim_node->set_start_offset(0.0);
					anim_node->set_timeline_length(7.0);
					anim_node->set_loop_mode(Animation::LOOP_NONE);

					SceneTree::get_singleton()->process(1.0);
					CHECK(node_2d->get_position().x == doctest::Approx(1.0f)); // t = 7.0 (Latest key is t = 8.0)
					SceneTree::get_singleton()->process(2.0);
					CHECK(node_2d->get_position().x == doctest::Approx(1.0f)); // t = 5.0 (Latest key is t = 8.0)
					SceneTree::get_singleton()->process(2.0);
					CHECK(node_2d->get_position().x == doctest::Approx(0.2f)); // t = 3.0 (Latest key is t = 4.0)
					SceneTree::get_singleton()->process(4.0);
					CHECK(node_2d->get_position().x == doctest::Approx(0.2f)); // t = 1.0 (Latest key is t = 4.0)
				}
			}
			SUBCASE("Using Stretch Timelime Scale") {
				anim_node->set_start_offset(2.0);
				anim_node->set_timeline_length(4.0);
				// 1. start_offset should be ignored.
				// 2. timeline_length is exactly half the animation length, so the animation should be played in double speed.
				anim_node->set_stretch_time_scale(true);
				SUBCASE("Forward playback") {
					anim_node->set_play_mode(AnimationNodeAnimation::PLAY_MODE_FORWARD);
					SUBCASE("No looping") {
						anim_node->set_loop_mode(Animation::LOOP_NONE);

						SceneTree::get_singleton()->process(0.5);
						CHECK(node_2d->get_position().x == doctest::Approx(0.0f)); // t = 1.0 (Latest key is t = 0.0)
						SceneTree::get_singleton()->process(1.0);
						CHECK(node_2d->get_position().x == doctest::Approx(0.0f)); // t = 3.0 (Latest key is t = 0.0)
						SceneTree::get_singleton()->process(1.0);
						CHECK(node_2d->get_position().x == doctest::Approx(0.2f)); // t = 5.0 (Latest key is t = 4.0)
						SceneTree::get_singleton()->process(1.0);
						CHECK(node_2d->get_position().x == doctest::Approx(0.2f)); // t = 7.0 (Latest key is t = 4.0)
						SceneTree::get_singleton()->process(1.0);
						CHECK(node_2d->get_position().x == doctest::Approx(1.0f)); // t = 8.0 (Latest key is t = 8.0)
					}
					SUBCASE("Linear looping") {
						anim_node->set_loop_mode(Animation::LOOP_LINEAR);

						SceneTree::get_singleton()->process(0.5);
						CHECK(node_2d->get_position().x == doctest::Approx(0.0f)); // t = 1.0 (Latest key is t = 0.0)
						SceneTree::get_singleton()->process(1.0);
						CHECK(node_2d->get_position().x == doctest::Approx(0.0f)); // t = 3.0 (Latest key is t = 0.0)
						SceneTree::get_singleton()->process(1.0);
						CHECK(node_2d->get_position().x == doctest::Approx(0.2f)); // t = 5.0 (Latest key is t = 4.0)
						SceneTree::get_singleton()->process(1.0);
						CHECK(node_2d->get_position().x == doctest::Approx(0.2f)); // t = 7.0 (Latest key is t = 4.0)
						SceneTree::get_singleton()->process(1.0);
						CHECK(node_2d->get_position().x == doctest::Approx(0.0f)); // t = 1.0 (Latest key is t = 0.0)
					}
					SUBCASE("Ping Pong looping") {
						anim_node->set_loop_mode(Animation::LOOP_PINGPONG);

						SceneTree::get_singleton()->process(0.5);
						CHECK(node_2d->get_position().x == doctest::Approx(0.0f)); // t = 1.0 (Latest key is t = 0.0)
						SceneTree::get_singleton()->process(1.0);
						CHECK(node_2d->get_position().x == doctest::Approx(0.0f)); // t = 3.0 (Latest key is t = 0.0)
						SceneTree::get_singleton()->process(1.0);
						CHECK(node_2d->get_position().x == doctest::Approx(0.2f)); // t = 5.0 (Latest key is t = 4.0)
						SceneTree::get_singleton()->process(1.0);
						CHECK(node_2d->get_position().x == doctest::Approx(0.2f)); // t = 7.0 (Latest key is t = 4.0)
						SceneTree::get_singleton()->process(1.0); // Direction flipped.
						CHECK(node_2d->get_position().x == doctest::Approx(1.0f)); // t = 7.0 (Latest key is t = 8.0)
					}
				}

				SUBCASE("Backward playback") {
					anim_node->set_play_mode(AnimationNodeAnimation::PLAY_MODE_BACKWARD);
					SUBCASE("No looping") {
						anim_node->set_loop_mode(Animation::LOOP_NONE);

						SceneTree::get_singleton()->process(0.5);
						CHECK(node_2d->get_position().x == doctest::Approx(1.0f)); // t = 7.0 (Latest key is t = 8.0)
						SceneTree::get_singleton()->process(1.0);
						CHECK(node_2d->get_position().x == doctest::Approx(1.0f)); // t = 5.0 (Latest key is t = 8.0)
						SceneTree::get_singleton()->process(1.0);
						CHECK(node_2d->get_position().x == doctest::Approx(0.2f)); // t = 3.0 (Latest key is t = 4.0)
						SceneTree::get_singleton()->process(1.0);
						CHECK(node_2d->get_position().x == doctest::Approx(0.2f)); // t = 1.0 (Latest key is t = 4.0)
						SceneTree::get_singleton()->process(1.0);
						CHECK(node_2d->get_position().x == doctest::Approx(0.0f)); // t = 0.0 (Latest key is t = 0.0)
					}
					SUBCASE("Linear looping") {
						anim_node->set_loop_mode(Animation::LOOP_LINEAR);

						SceneTree::get_singleton()->process(0.5);
						CHECK(node_2d->get_position().x == doctest::Approx(1.0f)); // t = 7.0 (Latest key is t = 8.0)
						SceneTree::get_singleton()->process(1.0);
						CHECK(node_2d->get_position().x == doctest::Approx(1.0f)); // t = 5.0 (Latest key is t = 8.0)
						SceneTree::get_singleton()->process(1.0);
						CHECK(node_2d->get_position().x == doctest::Approx(0.2f)); // t = 3.0 (Latest key is t = 4.0)
						SceneTree::get_singleton()->process(1.0);
						CHECK(node_2d->get_position().x == doctest::Approx(0.2f)); // t = 1.0 (Latest key is t = 4.0)
						SceneTree::get_singleton()->process(1.0);
						CHECK(node_2d->get_position().x == doctest::Approx(1.0f)); // t = 7.0 (Latest key is t = 8.0)
					}
					SUBCASE("Ping Pong looping") {
						anim_node->set_loop_mode(Animation::LOOP_PINGPONG);

						SceneTree::get_singleton()->process(0.5);
						CHECK(node_2d->get_position().x == doctest::Approx(1.0f)); // t = 7.0 (Latest key is t = 8.0)
						SceneTree::get_singleton()->process(1.0);
						CHECK(node_2d->get_position().x == doctest::Approx(1.0f)); // t = 5.0 (Latest key is t = 8.0)
						SceneTree::get_singleton()->process(1.0);
						CHECK(node_2d->get_position().x == doctest::Approx(0.2f)); // t = 3.0 (Latest key is t = 4.0)
						SceneTree::get_singleton()->process(1.0);
						CHECK(node_2d->get_position().x == doctest::Approx(0.2f)); // t = 1.0 (Latest key is t = 4.0)
						SceneTree::get_singleton()->process(1.0); // Direction flipped.
						CHECK(node_2d->get_position().x == doctest::Approx(0.0f)); // t = 1.0 (Latest key is t = 0.0)
					}
				}
			}
		}
	}

	memdelete(animation_tree);
	memdelete(animation_player);
	memdelete(node_2d);
}

TEST_CASE("[SceneTree][AnimationBlendTree] AnimationNodeAnimation output should be identical to an equivalent AnimationPlayer") {
	// Setup test scene.
	Node2D *node_2d = memnew(Node2D);
	node_2d->set_name("Node2D");
	AnimationTree *animation_tree = memnew(AnimationTree);
	AnimationPlayer *animation_player = memnew(AnimationPlayer); // Used by AnimationTree.
	AnimationPlayer *animation_player_2 = memnew(AnimationPlayer);
	SceneTree::get_singleton()->get_root()->add_child(node_2d);
	SceneTree::get_singleton()->get_root()->add_child(animation_player);
	SceneTree::get_singleton()->get_root()->add_child(animation_player_2);
	SceneTree::get_singleton()->get_root()->add_child(animation_tree);

	const StringName test_animation_name = "test_animation";

	// Setup test animation for AnimationTree.
	Ref<Animation> animation;
	animation.instantiate();
	Ref<AnimationLibrary> animation_library;
	animation_library.instantiate();
	animation_library->add_animation(test_animation_name, animation);
	animation_player->add_animation_library("", animation_library);
	int value_track = animation->add_track(Animation::TrackType::TYPE_VALUE);
	animation->track_set_path(value_track, NodePath("Node2D:position:x"));
	animation->set_length(8.0);
	animation->track_insert_key(value_track, 0.0, 0.0f);
	animation->track_insert_key(value_track, 4.0, 0.2f);
	animation->track_insert_key(value_track, 8.0, 1.0f);

	// Setup test animation for AnimationPlayer.
	// The animation is identical to AnimationTree's animation,
	// except that position.y is modified instead of position.x.
	Ref<Animation> animation_2;
	animation_2 = animation->duplicate();
	animation_2->track_set_path(value_track, NodePath("Node2D:position:y"));
	Ref<AnimationLibrary> animation_library_2;
	animation_library_2.instantiate();
	animation_library_2->add_animation(test_animation_name, animation_2);
	animation_player_2->add_animation_library("", animation_library_2);

	// Setup animation node.
	Ref<AnimationNodeAnimation> anim_node;
	anim_node.instantiate();
	anim_node->set_animation(test_animation_name);

	// Custom timeline is used to enable loop mode.
	anim_node->set_use_custom_timeline(true);
	anim_node->set_start_offset(0.0);
	anim_node->set_timeline_length(animation->get_length());

	// Setup animation tree.
	animation_tree->set_animation_player(animation_tree->get_path_to(animation_player));
	animation_tree->set_root_animation_node(anim_node);
	animation_tree->set_active(true);

	// Note: To avoid floating point inaccuracy at loop points,
	// the step size and iteration count are picked to avoid integer multiples of the animation length.
	constexpr int ITERATION_COUNT = 6;
	constexpr float STEP_SIZE = 3.0;

	SUBCASE("Forward playback") {
		anim_node->set_play_mode(AnimationNodeAnimation::PLAY_MODE_FORWARD);

		SUBCASE("No looping") {
			animation_2->set_loop_mode(Animation::LOOP_NONE);
			anim_node->set_loop_mode(Animation::LOOP_NONE);
			animation_player_2->stop();
			animation_player_2->play(test_animation_name);
			for (int i = 0; i < ITERATION_COUNT; i++) {
				SceneTree::get_singleton()->process(STEP_SIZE);
				CHECK(node_2d->get_position().x == doctest::Approx(node_2d->get_position().y));
			}
		}

		SUBCASE("Linear looping") {
			animation_2->set_loop_mode(Animation::LOOP_LINEAR);
			anim_node->set_loop_mode(Animation::LOOP_LINEAR);
			animation_player_2->stop();
			animation_player_2->play(test_animation_name);
			for (int i = 0; i < ITERATION_COUNT; i++) {
				SceneTree::get_singleton()->process(STEP_SIZE);
				CHECK(node_2d->get_position().x == doctest::Approx(node_2d->get_position().y));
			}
		}

		SUBCASE("Ping Pong looping") {
			animation_2->set_loop_mode(Animation::LOOP_PINGPONG);
			anim_node->set_loop_mode(Animation::LOOP_PINGPONG);
			animation_player_2->stop();
			animation_player_2->play(test_animation_name);
			for (int i = 0; i < ITERATION_COUNT; i++) {
				SceneTree::get_singleton()->process(STEP_SIZE);
				CHECK(node_2d->get_position().x == doctest::Approx(node_2d->get_position().y));
			}
		}
	}

	SUBCASE("Backward playback") {
		anim_node->set_play_mode(AnimationNodeAnimation::PLAY_MODE_BACKWARD);

		SUBCASE("No looping") {
			animation_2->set_loop_mode(Animation::LOOP_NONE);
			anim_node->set_loop_mode(Animation::LOOP_NONE);
			animation_player_2->stop();
			animation_player_2->play_backwards(test_animation_name);
			for (int i = 0; i < ITERATION_COUNT; i++) {
				SceneTree::get_singleton()->process(STEP_SIZE);
				CHECK(node_2d->get_position().x == doctest::Approx(node_2d->get_position().y));
			}
		}

		SUBCASE("Linear looping") {
			animation_2->set_loop_mode(Animation::LOOP_LINEAR);
			anim_node->set_loop_mode(Animation::LOOP_LINEAR);
			animation_player_2->stop();
			animation_player_2->play_backwards(test_animation_name);
			for (int i = 0; i < ITERATION_COUNT; i++) {
				SceneTree::get_singleton()->process(STEP_SIZE);
				CHECK(node_2d->get_position().x == doctest::Approx(node_2d->get_position().y));
			}
		}

		SUBCASE("Ping Pong looping") {
			animation_2->set_loop_mode(Animation::LOOP_PINGPONG);
			anim_node->set_loop_mode(Animation::LOOP_PINGPONG);
			animation_player_2->stop();
			animation_player_2->play_backwards(test_animation_name);
			for (int i = 0; i < ITERATION_COUNT; i++) {
				SceneTree::get_singleton()->process(STEP_SIZE);
				CHECK(node_2d->get_position().x == doctest::Approx(node_2d->get_position().y));
			}
		}
	}

	memdelete(animation_player_2);
	memdelete(animation_tree);
	memdelete(animation_player);
	memdelete(node_2d);
}

} // namespace TestAnimationBlendTree
