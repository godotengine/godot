/**
 * bt_stop_animation.cpp
 * =============================================================================
 * Copyright 2021-2024 Serhii Snitsaruk
 *
 * Use of this source code is governed by an MIT-style
 * license that can be found in the LICENSE file or at
 * https://opensource.org/licenses/MIT.
 * =============================================================================
 */

#include "bt_stop_animation.h"

//**** Setters / Getters

void BTStopAnimation::set_animation_player(Ref<BBNode> p_animation_player) {
	animation_player_param = p_animation_player;
	emit_changed();
	if (Engine::get_singleton()->is_editor_hint() && animation_player_param.is_valid()) {
		animation_player_param->connect(LW_NAME(changed), Callable(this, LW_NAME(emit_changed)));
	}
}

void BTStopAnimation::set_animation_name(StringName p_animation_name) {
	animation_name = p_animation_name;
	emit_changed();
}

void BTStopAnimation::set_keep_state(bool p_keep_state) {
	keep_state = p_keep_state;
	emit_changed();
}

//**** Task Implementation

PackedStringArray BTStopAnimation::get_configuration_warnings() {
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
	return warnings;
}

String BTStopAnimation::_generate_name() {
	return "StopAnimation" +
			(animation_name != StringName() ? vformat(" \"%s\"", animation_name) : "") +
			(keep_state ? "  keep_state: true" : "");
}

void BTStopAnimation::_setup() {
	setup_failed = true;
	ERR_FAIL_COND_MSG(animation_player_param.is_null(), "BTStopAnimation: AnimationPlayer parameter is not set.");
	animation_player = Object::cast_to<AnimationPlayer>(animation_player_param->get_value(get_scene_root(), get_blackboard()));
	ERR_FAIL_COND_MSG(animation_player == nullptr, "BTStopAnimation: Failed to get AnimationPlayer.");
	if (animation_name != StringName()) {
		ERR_FAIL_COND_MSG(!animation_player->has_animation(animation_name), vformat("BTStopAnimation: Animation not found: %s", animation_name));
	}
	setup_failed = false;
}

BT::Status BTStopAnimation::_tick(double p_delta) {
	ERR_FAIL_COND_V_MSG(setup_failed == true, FAILURE, "BTStopAnimation: _setup() failed - returning FAILURE.");
	if (animation_player->is_playing() && (animation_name == StringName() || animation_name == animation_player->get_assigned_animation())) {
		animation_player->stop(keep_state);
	}
	return SUCCESS;
}

//**** Godot

void BTStopAnimation::_bind_methods() {
	ClassDB::bind_method(D_METHOD("set_animation_player", "animation_player"), &BTStopAnimation::set_animation_player);
	ClassDB::bind_method(D_METHOD("get_animation_player"), &BTStopAnimation::get_animation_player);
	ClassDB::bind_method(D_METHOD("set_animation_name", "name"), &BTStopAnimation::set_animation_name);
	ClassDB::bind_method(D_METHOD("get_animation_name"), &BTStopAnimation::get_animation_name);
	ClassDB::bind_method(D_METHOD("set_keep_state", "keep_state"), &BTStopAnimation::set_keep_state);
	ClassDB::bind_method(D_METHOD("get_keep_state"), &BTStopAnimation::get_keep_state);

	ADD_PROPERTY(PropertyInfo(Variant::OBJECT, "animation_player", PROPERTY_HINT_RESOURCE_TYPE, "BBNode"), "set_animation_player", "get_animation_player");
	ADD_PROPERTY(PropertyInfo(Variant::STRING_NAME, "animation_name"), "set_animation_name", "get_animation_name");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "keep_state"), "set_keep_state", "get_keep_state");
}
