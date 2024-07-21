/**
 * bt_play_animation.cpp
 * =============================================================================
 * Copyright 2021-2024 Serhii Snitsaruk
 *
 * Use of this source code is governed by an MIT-style
 * license that can be found in the LICENSE file or at
 * https://opensource.org/licenses/MIT.
 * =============================================================================
 */

#include "bt_play_animation.h"

//**** Setters / Getters

void BTPlayAnimation::set_animation_player(Ref<BBNode> p_animation_player) {
	animation_player_param = p_animation_player;
	emit_changed();
	if (Engine::get_singleton()->is_editor_hint() && animation_player_param.is_valid()) {
		animation_player_param->connect(LW_NAME(changed), Callable(this, LW_NAME(emit_changed)));
	}
}

void BTPlayAnimation::set_animation_name(StringName p_animation_name) {
	animation_name = p_animation_name;
	emit_changed();
}

void BTPlayAnimation::set_await_completion(double p_await_completion) {
	await_completion = p_await_completion;
	emit_changed();
}

void BTPlayAnimation::set_blend(double p_blend) {
	blend = p_blend;
	emit_changed();
}

void BTPlayAnimation::set_speed(double p_speed) {
	speed = p_speed;
	emit_changed();
}

void BTPlayAnimation::set_from_end(bool p_from_end) {
	from_end = p_from_end;
	emit_changed();
}

//**** Task Implementation

PackedStringArray BTPlayAnimation::get_configuration_warnings() {
	PackedStringArray warnings = BTAction::get_configuration_warnings();
	if (animation_player_param.is_null()) {
		warnings.append("Animation Player parameter is not set.");
	} else {
		if (animation_player_param->get_value_source() == BBParam::SAVED_VALUE && animation_player_param->get_saved_value() == Variant()) {
			warnings.append("Path to AnimationPlayer node is not set.");
		} else if (animation_player_param->get_value_source() == BBParam::BLACKBOARD_VAR && animation_player_param->get_variable() == StringName()) {
			warnings.append("AnimationPlayer blackboard variable is not set.");
		}
	}
	if (animation_name == StringName() && await_completion > 0.0) {
		warnings.append("Animation Name is required in order to wait for the animation to finish.");
	}
	return warnings;
}

String BTPlayAnimation::_generate_name() {
	return "PlayAnimation" +
			(animation_name != StringName() ? vformat(" \"%s\"", animation_name) : "") +
			(blend >= 0.0 ? vformat("  blend: %ss", Math::snapped(blend, 0.001)) : "") +
			(speed != 1.0 ? vformat("  speed: %s", Math::snapped(speed, 0.001)) : "") +
			(from_end != false ? vformat("  from_end: %s", from_end) : "") +
			(await_completion > 0.0 ? vformat("  await_completion: %ss", Math::snapped(await_completion, 0.001)) : "");
}

void BTPlayAnimation::_setup() {
	setup_failed = true;
	ERR_FAIL_COND_MSG(animation_player_param.is_null(), "BTPlayAnimation: AnimationPlayer parameter is not set.");
	animation_player = Object::cast_to<AnimationPlayer>(animation_player_param->get_value(get_scene_root(), get_blackboard()));
	ERR_FAIL_COND_MSG(animation_player == nullptr, "BTPlayAnimation: Failed to get AnimationPlayer.");
	ERR_FAIL_COND_MSG(animation_name != StringName() && !animation_player->has_animation(animation_name), vformat("BTPlayAnimation: Animation not found: %s", animation_name));
	if (animation_name == StringName() && await_completion > 0.0) {
		WARN_PRINT("BTPlayAnimation: Animation Name is required in order to wait for the animation to finish.");
	}
	setup_failed = false;
}

void BTPlayAnimation::_enter() {
	if (!setup_failed) {
		animation_player->play(animation_name, blend, speed, from_end);
	}
}

BT::Status BTPlayAnimation::_tick(double p_delta) {
	ERR_FAIL_COND_V_MSG(setup_failed == true, FAILURE, "BTPlayAnimation: _setup() failed - returning FAILURE.");

	// ! Doing this check instead of using signal due to a bug in Godot: https://github.com/godotengine/godot/issues/76127
	if (animation_player->is_playing() && animation_player->get_assigned_animation() == animation_name) {
		if (get_elapsed_time() < await_completion) {
			return RUNNING;
		} else if (await_completion > 0.0) {
			WARN_PRINT(vformat("BTPlayAnimation: Waiting time for the \"%s\" animation exceeded the allocated %s sec.", animation_name, await_completion));
		}
	}
	return SUCCESS;
}

//**** Godot

void BTPlayAnimation::_bind_methods() {
	ClassDB::bind_method(D_METHOD("set_animation_player", "animation_player"), &BTPlayAnimation::set_animation_player);
	ClassDB::bind_method(D_METHOD("get_animation_player"), &BTPlayAnimation::get_animation_player);
	ClassDB::bind_method(D_METHOD("set_animation_name", "name"), &BTPlayAnimation::set_animation_name);
	ClassDB::bind_method(D_METHOD("get_animation_name"), &BTPlayAnimation::get_animation_name);
	ClassDB::bind_method(D_METHOD("set_await_completion", "time_sec"), &BTPlayAnimation::set_await_completion);
	ClassDB::bind_method(D_METHOD("get_await_completion"), &BTPlayAnimation::get_await_completion);
	ClassDB::bind_method(D_METHOD("set_blend", "blend"), &BTPlayAnimation::set_blend);
	ClassDB::bind_method(D_METHOD("get_blend"), &BTPlayAnimation::get_blend);
	ClassDB::bind_method(D_METHOD("set_speed", "speed"), &BTPlayAnimation::set_speed);
	ClassDB::bind_method(D_METHOD("get_speed"), &BTPlayAnimation::get_speed);
	ClassDB::bind_method(D_METHOD("set_from_end", "from_end"), &BTPlayAnimation::set_from_end);
	ClassDB::bind_method(D_METHOD("get_from_end"), &BTPlayAnimation::get_from_end);

	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "await_completion", PROPERTY_HINT_RANGE, "0.0,100.0"), "set_await_completion", "get_await_completion");
	ADD_PROPERTY(PropertyInfo(Variant::OBJECT, "animation_player", PROPERTY_HINT_RESOURCE_TYPE, "BBNode"), "set_animation_player", "get_animation_player");
	ADD_PROPERTY(PropertyInfo(Variant::STRING_NAME, "animation_name"), "set_animation_name", "get_animation_name");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "blend"), "set_blend", "get_blend");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "speed"), "set_speed", "get_speed");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "from_end"), "set_from_end", "get_from_end");
}
