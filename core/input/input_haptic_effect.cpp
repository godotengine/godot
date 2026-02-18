/**************************************************************************/
/*  input_haptic_effect.cpp                                               */
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

#include "core/input/input_haptic_effect.h"

void InputHapticEffect::_bind_methods() {
	ClassDB::bind_method(D_METHOD("set_direction_radians", "direction_radians"), &InputHapticEffect::set_direction_radians);
	ClassDB::bind_method(D_METHOD("get_direction_radians"), &InputHapticEffect::get_direction_radians);

	ClassDB::bind_method(D_METHOD("set_duration", "duration"), &InputHapticEffect::set_duration);
	ClassDB::bind_method(D_METHOD("get_duration"), &InputHapticEffect::get_duration);

	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "direction_radians"), "set_direction_radians", "get_direction_radians");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "duration"), "set_duration", "get_duration");

	BIND_CONSTANT(HAPTIC_EFFECT_CONSTANT);
	BIND_CONSTANT(HAPTIC_EFFECT_SINE);
	BIND_CONSTANT(HAPTIC_EFFECT_SQUARE);
	BIND_CONSTANT(HAPTIC_EFFECT_TRIANGLE);
	BIND_CONSTANT(HAPTIC_EFFECT_SAWTOOTH_UP);
	BIND_CONSTANT(HAPTIC_EFFECT_SAWTOOTH_DOWN);
	BIND_CONSTANT(HAPTIC_EFFECT_RAMP);
	BIND_CONSTANT(HAPTIC_EFFECT_SPRING);
	BIND_CONSTANT(HAPTIC_EFFECT_DAMPER);
	BIND_CONSTANT(HAPTIC_EFFECT_INERTIA);
	BIND_CONSTANT(HAPTIC_EFFECT_FRICTION);
	BIND_CONSTANT(HAPTIC_EFFECT_LEFT_RIGHT);
	BIND_CONSTANT(HAPTIC_EFFECT_CUSTOM);
}

void InputHapticEffect::set_direction_radians(float p_direction_radians) {
	direction_radians = Math::fmod(p_direction_radians, (float)Math::TAU);
	if (direction_radians < 0.0f) {
		direction_radians += (float)Math::TAU;
	}
}

float InputHapticEffect::get_direction_radians() const {
	return direction_radians;
}

void InputHapticEffect::set_duration(float p_duration) {
	duration = CLAMP(p_duration, 0.0f, (float)(UINT32_MAX / 1000));
}

float InputHapticEffect::get_duration() const {
	return duration;
}

///////////////////////////////////

void InputHapticEffectConstant::_bind_methods() {
	ClassDB::bind_method(D_METHOD("set_force", "force"), &InputHapticEffectConstant::set_force);
	ClassDB::bind_method(D_METHOD("get_force"), &InputHapticEffectConstant::get_force);

	ClassDB::bind_method(D_METHOD("set_attack_length", "attack_length"), &InputHapticEffectConstant::set_attack_length);
	ClassDB::bind_method(D_METHOD("get_attack_length"), &InputHapticEffectConstant::get_attack_length);

	ClassDB::bind_method(D_METHOD("set_attack_level", "attack_level"), &InputHapticEffectConstant::set_attack_level);
	ClassDB::bind_method(D_METHOD("get_attack_level"), &InputHapticEffectConstant::get_attack_level);

	ClassDB::bind_method(D_METHOD("set_fade_length", "fade_length"), &InputHapticEffectConstant::set_fade_length);
	ClassDB::bind_method(D_METHOD("get_fade_length"), &InputHapticEffectConstant::get_fade_length);

	ClassDB::bind_method(D_METHOD("set_fade_level", "fade_level"), &InputHapticEffectConstant::set_fade_level);
	ClassDB::bind_method(D_METHOD("get_fade_level"), &InputHapticEffectConstant::get_fade_level);

	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "force"), "set_force", "get_force");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "attack_length"), "set_attack_length", "get_attack_length");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "attack_level"), "set_attack_level", "get_attack_level");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "fade_length"), "set_fade_length", "get_fade_length");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "fade_level"), "set_fade_level", "get_fade_level");
}

void InputHapticEffectConstant::set_force(float p_force) {
	force = CLAMP(p_force, -1.0, 1.0);
}

float InputHapticEffectConstant::get_force() const {
	return force;
}

void InputHapticEffectConstant::set_attack_length(float p_attack_length) {
	attack_length = CLAMP(p_attack_length, 0.0f, UINT16_MAX / 1000.0f);
}

float InputHapticEffectConstant::get_attack_length() const {
	return attack_length;
}

void InputHapticEffectConstant::set_attack_level(float p_attack_level) {
	attack_level = CLAMP(p_attack_level, 0.0, 1.0);
}

float InputHapticEffectConstant::get_attack_level() const {
	return attack_level;
}

void InputHapticEffectConstant::set_fade_length(float p_fade_length) {
	fade_length = CLAMP(p_fade_length, 0.0f, UINT16_MAX / 1000.0f);
}

float InputHapticEffectConstant::get_fade_length() const {
	return fade_length;
}

void InputHapticEffectConstant::set_fade_level(float p_fade_level) {
	fade_level = CLAMP(p_fade_level, 0.0, 1.0);
}

float InputHapticEffectConstant::get_fade_level() const {
	return fade_level;
}

///////////////////////////////////

void InputHapticEffectPeriodic::_bind_methods() {
	ClassDB::bind_method(D_METHOD("set_period", "period"), &InputHapticEffectPeriodic::set_period);
	ClassDB::bind_method(D_METHOD("get_period"), &InputHapticEffectPeriodic::get_period);

	ClassDB::bind_method(D_METHOD("set_magnitude", "magnitude"), &InputHapticEffectPeriodic::set_magnitude);
	ClassDB::bind_method(D_METHOD("get_magnitude"), &InputHapticEffectPeriodic::get_magnitude);

	ClassDB::bind_method(D_METHOD("set_offset", "offset"), &InputHapticEffectPeriodic::set_offset);
	ClassDB::bind_method(D_METHOD("get_offset"), &InputHapticEffectPeriodic::get_offset);

	ClassDB::bind_method(D_METHOD("set_phase", "phase"), &InputHapticEffectPeriodic::set_phase);
	ClassDB::bind_method(D_METHOD("get_phase"), &InputHapticEffectPeriodic::get_phase);

	ClassDB::bind_method(D_METHOD("set_attack_length", "attack_length"), &InputHapticEffectPeriodic::set_attack_length);
	ClassDB::bind_method(D_METHOD("get_attack_length"), &InputHapticEffectPeriodic::get_attack_length);

	ClassDB::bind_method(D_METHOD("set_attack_level", "attack_level"), &InputHapticEffectPeriodic::set_attack_level);
	ClassDB::bind_method(D_METHOD("get_attack_level"), &InputHapticEffectPeriodic::get_attack_level);

	ClassDB::bind_method(D_METHOD("set_fade_length", "fade_length"), &InputHapticEffectPeriodic::set_fade_length);
	ClassDB::bind_method(D_METHOD("get_fade_length"), &InputHapticEffectPeriodic::get_fade_length);

	ClassDB::bind_method(D_METHOD("set_fade_level", "fade_level"), &InputHapticEffectPeriodic::set_fade_level);
	ClassDB::bind_method(D_METHOD("get_fade_level"), &InputHapticEffectPeriodic::get_fade_level);

	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "period"), "set_period", "get_period");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "magnitude"), "set_magnitude", "get_magnitude");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "offset"), "set_offset", "get_offset");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "phase"), "set_phase", "get_phase");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "attack_length"), "set_attack_length", "get_attack_length");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "attack_level"), "set_attack_level", "get_attack_level");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "fade_length"), "set_fade_length", "get_fade_length");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "fade_level"), "set_fade_level", "get_fade_level");
}

void InputHapticEffectPeriodic::set_type(Type p_type) {
	switch (p_type) {
		case HAPTIC_EFFECT_SINE:
		case HAPTIC_EFFECT_SQUARE:
		case HAPTIC_EFFECT_TRIANGLE:
		case HAPTIC_EFFECT_SAWTOOTH_UP:
		case HAPTIC_EFFECT_SAWTOOTH_DOWN:
			type = p_type;
			break;
		default:
			break;
	}
}

void InputHapticEffectPeriodic::set_period(float p_period) {
	period = CLAMP(p_period, 0.0f, UINT16_MAX / 1000.0f);
}

float InputHapticEffectPeriodic::get_period() const {
	return period;
}

void InputHapticEffectPeriodic::set_magnitude(float p_magnitude) {
	magnitude = CLAMP(p_magnitude, -1.0, 1.0);
}

float InputHapticEffectPeriodic::get_magnitude() const {
	return magnitude;
}

void InputHapticEffectPeriodic::set_offset(float p_offset) {
	offset = CLAMP(p_offset, -1.0, 1.0);
}

float InputHapticEffectPeriodic::get_offset() const {
	return offset;
}

void InputHapticEffectPeriodic::set_phase(float p_phase) {
	phase = Math::fmod(p_phase, (float)Math::TAU);
	if (phase < 0.0f) {
		phase += (float)Math::TAU;
	}
}

float InputHapticEffectPeriodic::get_phase() const {
	return phase;
}

void InputHapticEffectPeriodic::set_attack_length(float p_attack_length) {
	attack_length = CLAMP(p_attack_length, 0.0f, UINT16_MAX / 1000.0f);
}

float InputHapticEffectPeriodic::get_attack_length() const {
	return attack_length;
}

void InputHapticEffectPeriodic::set_attack_level(float p_attack_level) {
	attack_level = CLAMP(p_attack_level, 0.0, 1.0);
}

float InputHapticEffectPeriodic::get_attack_level() const {
	return attack_level;
}

void InputHapticEffectPeriodic::set_fade_length(float p_fade_length) {
	fade_length = CLAMP(p_fade_length, 0.0f, UINT16_MAX / 1000.0f);
}

float InputHapticEffectPeriodic::get_fade_length() const {
	return fade_length;
}

void InputHapticEffectPeriodic::set_fade_level(float p_fade_level) {
	fade_level = CLAMP(p_fade_level, 0.0, 1.0);
}

float InputHapticEffectPeriodic::get_fade_level() const {
	return fade_level;
}

///////////////////////////////////

void InputHapticEffectCondition::_bind_methods() {
	ClassDB::bind_method(D_METHOD("set_right_level", "right_level"), &InputHapticEffectCondition::set_right_level);
	ClassDB::bind_method(D_METHOD("get_right_level"), &InputHapticEffectCondition::get_right_level);

	ClassDB::bind_method(D_METHOD("set_left_level", "left_level"), &InputHapticEffectCondition::set_left_level);
	ClassDB::bind_method(D_METHOD("get_left_level"), &InputHapticEffectCondition::get_left_level);

	ClassDB::bind_method(D_METHOD("set_right_coef", "right_coef"), &InputHapticEffectCondition::set_right_coef);
	ClassDB::bind_method(D_METHOD("get_right_coef"), &InputHapticEffectCondition::get_right_coef);

	ClassDB::bind_method(D_METHOD("set_left_coef", "left_coef"), &InputHapticEffectCondition::set_left_coef);
	ClassDB::bind_method(D_METHOD("get_left_coef"), &InputHapticEffectCondition::get_left_coef);

	ClassDB::bind_method(D_METHOD("set_deadband", "deadband"), &InputHapticEffectCondition::set_deadband);
	ClassDB::bind_method(D_METHOD("get_deadband"), &InputHapticEffectCondition::get_deadband);

	ClassDB::bind_method(D_METHOD("set_center", "center"), &InputHapticEffectCondition::set_center);
	ClassDB::bind_method(D_METHOD("get_center"), &InputHapticEffectCondition::get_center);

	ADD_PROPERTY(PropertyInfo(Variant::VECTOR3, "right_level"), "set_right_level", "get_right_level");
	ADD_PROPERTY(PropertyInfo(Variant::VECTOR3, "left_level"), "set_left_level", "get_left_level");
	ADD_PROPERTY(PropertyInfo(Variant::VECTOR3, "right_coef"), "set_right_coef", "get_right_coef");
	ADD_PROPERTY(PropertyInfo(Variant::VECTOR3, "left_coef"), "set_left_coef", "get_left_coef");
	ADD_PROPERTY(PropertyInfo(Variant::VECTOR3, "deadband"), "set_deadband", "get_deadband");
	ADD_PROPERTY(PropertyInfo(Variant::VECTOR3, "center"), "set_center", "get_center");
}

void InputHapticEffectCondition::set_type(Type p_type) {
	switch (p_type) {
		case HAPTIC_EFFECT_SPRING:
		case HAPTIC_EFFECT_DAMPER:
		case HAPTIC_EFFECT_INERTIA:
		case HAPTIC_EFFECT_FRICTION:
			type = p_type;
			break;
		default:
			break;
	}
}

void InputHapticEffectCondition::set_right_level(const Vector3 &p_right_level) {
	right_level = p_right_level.clampf(0.0, 1.0);
}

Vector3 InputHapticEffectCondition::get_right_level() const {
	return right_level;
}

void InputHapticEffectCondition::set_left_level(const Vector3 &p_left_level) {
	left_level = p_left_level.clampf(0.0, 1.0);
}

Vector3 InputHapticEffectCondition::get_left_level() const {
	return left_level;
}

void InputHapticEffectCondition::set_right_coef(const Vector3 &p_right_coef) {
	right_coef = p_right_coef.clampf(-1.0, 1.0);
}

Vector3 InputHapticEffectCondition::get_right_coef() const {
	return right_coef;
}

void InputHapticEffectCondition::set_left_coef(const Vector3 &p_left_coef) {
	left_coef = p_left_coef.clampf(-1.0, 1.0);
}

Vector3 InputHapticEffectCondition::get_left_coef() const {
	return left_coef;
}

void InputHapticEffectCondition::set_deadband(const Vector3 &p_deadband) {
	deadband = p_deadband.clampf(0.0, 1.0);
}

Vector3 InputHapticEffectCondition::get_deadband() const {
	return deadband;
}

void InputHapticEffectCondition::set_center(const Vector3 &p_center) {
	center = p_center.clampf(-1.0, 1.0);
}

Vector3 InputHapticEffectCondition::get_center() const {
	return center;
}

///////////////////////////////////

void InputHapticEffectLeftRight::_bind_methods() {
	ClassDB::bind_method(D_METHOD("set_large_magnitude", "large_magnitude"), &InputHapticEffectLeftRight::set_large_magnitude);
	ClassDB::bind_method(D_METHOD("get_large_magnitude"), &InputHapticEffectLeftRight::get_large_magnitude);

	ClassDB::bind_method(D_METHOD("set_small_magnitude", "small_magnitude"), &InputHapticEffectLeftRight::set_small_magnitude);
	ClassDB::bind_method(D_METHOD("get_small_magnitude"), &InputHapticEffectLeftRight::get_small_magnitude);

	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "large_magnitude"), "set_large_magnitude", "get_large_magnitude");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "small_magnitude"), "set_small_magnitude", "get_small_magnitude");
}

void InputHapticEffectLeftRight::set_large_magnitude(float p_large_magnitude) {
	large_magnitude = CLAMP(p_large_magnitude, 0.0, 1.0);
}

float InputHapticEffectLeftRight::get_large_magnitude() const {
	return large_magnitude;
}

void InputHapticEffectLeftRight::set_small_magnitude(float p_small_magnitude) {
	small_magnitude = CLAMP(p_small_magnitude, 0.0, 1.0);
}

float InputHapticEffectLeftRight::get_small_magnitude() const {
	return small_magnitude;
}

///////////////////////////////////

void InputHapticEffectRamp::_bind_methods() {
	ClassDB::bind_method(D_METHOD("set_start_strength", "start_strength"), &InputHapticEffectRamp::set_start_strength);
	ClassDB::bind_method(D_METHOD("get_start_strength"), &InputHapticEffectRamp::get_start_strength);

	ClassDB::bind_method(D_METHOD("set_end_strength", "end_strength"), &InputHapticEffectRamp::set_end_strength);
	ClassDB::bind_method(D_METHOD("get_end_strength"), &InputHapticEffectRamp::get_end_strength);

	ClassDB::bind_method(D_METHOD("set_attack_length", "attack_length"), &InputHapticEffectRamp::set_attack_length);
	ClassDB::bind_method(D_METHOD("get_attack_length"), &InputHapticEffectRamp::get_attack_length);

	ClassDB::bind_method(D_METHOD("set_attack_level", "attack_level"), &InputHapticEffectRamp::set_attack_level);
	ClassDB::bind_method(D_METHOD("get_attack_level"), &InputHapticEffectRamp::get_attack_level);

	ClassDB::bind_method(D_METHOD("set_fade_length", "fade_length"), &InputHapticEffectRamp::set_fade_length);
	ClassDB::bind_method(D_METHOD("get_fade_length"), &InputHapticEffectRamp::get_fade_length);

	ClassDB::bind_method(D_METHOD("set_fade_level", "fade_level"), &InputHapticEffectRamp::set_fade_level);
	ClassDB::bind_method(D_METHOD("get_fade_level"), &InputHapticEffectRamp::get_fade_level);

	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "start_strength"), "set_start_strength", "get_start_strength");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "end_strength"), "set_end_strength", "get_end_strength");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "attack_length"), "set_attack_length", "get_attack_length");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "attack_level"), "set_attack_level", "get_attack_level");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "fade_length"), "set_fade_length", "get_fade_length");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "fade_level"), "set_fade_level", "get_fade_level");
}

void InputHapticEffectRamp::set_start_strength(float p_start_strength) {
	start_strength = CLAMP(p_start_strength, -1.0, 1.0);
}

float InputHapticEffectRamp::get_start_strength() const {
	return start_strength;
}

void InputHapticEffectRamp::set_end_strength(float p_end_strength) {
	end_strength = CLAMP(p_end_strength, -1.0, 1.0);
}

float InputHapticEffectRamp::get_end_strength() const {
	return end_strength;
}

void InputHapticEffectRamp::set_attack_length(float p_attack_length) {
	attack_length = CLAMP(p_attack_length, 0.0f, UINT16_MAX / 1000.0f);
}

float InputHapticEffectRamp::get_attack_length() const {
	return attack_length;
}

void InputHapticEffectRamp::set_attack_level(float p_attack_level) {
	attack_level = CLAMP(p_attack_level, 0.0, 1.0);
}

float InputHapticEffectRamp::get_attack_level() const {
	return attack_level;
}

void InputHapticEffectRamp::set_fade_length(float p_fade_length) {
	fade_length = CLAMP(p_fade_length, 0.0f, UINT16_MAX / 1000.0f);
}

float InputHapticEffectRamp::get_fade_length() const {
	return fade_length;
}

void InputHapticEffectRamp::set_fade_level(float p_fade_level) {
	fade_level = CLAMP(p_fade_level, 0.0, 1.0);
}

float InputHapticEffectRamp::get_fade_level() const {
	return fade_level;
}

///////////////////////////////////

void InputHapticEffectCustom::_bind_methods() {
	ClassDB::bind_method(D_METHOD("set_channels", "channels"), &InputHapticEffectCustom::set_channels);
	ClassDB::bind_method(D_METHOD("get_channels"), &InputHapticEffectCustom::get_channels);

	ClassDB::bind_method(D_METHOD("set_samples", "samples"), &InputHapticEffectCustom::set_samples);
	ClassDB::bind_method(D_METHOD("get_samples"), &InputHapticEffectCustom::get_samples);

	ClassDB::bind_method(D_METHOD("set_period", "period"), &InputHapticEffectCustom::set_period);
	ClassDB::bind_method(D_METHOD("get_period"), &InputHapticEffectCustom::get_period);

	ClassDB::bind_method(D_METHOD("set_data", "data"), &InputHapticEffectCustom::set_data);
	ClassDB::bind_method(D_METHOD("get_data"), &InputHapticEffectCustom::get_data);

	ClassDB::bind_method(D_METHOD("set_attack_length", "attack_length"), &InputHapticEffectCustom::set_attack_length);
	ClassDB::bind_method(D_METHOD("get_attack_length"), &InputHapticEffectCustom::get_attack_length);

	ClassDB::bind_method(D_METHOD("set_attack_level", "attack_level"), &InputHapticEffectCustom::set_attack_level);
	ClassDB::bind_method(D_METHOD("get_attack_level"), &InputHapticEffectCustom::get_attack_level);

	ClassDB::bind_method(D_METHOD("set_fade_length", "fade_length"), &InputHapticEffectCustom::set_fade_length);
	ClassDB::bind_method(D_METHOD("get_fade_length"), &InputHapticEffectCustom::get_fade_length);

	ClassDB::bind_method(D_METHOD("set_fade_level", "fade_level"), &InputHapticEffectCustom::set_fade_level);
	ClassDB::bind_method(D_METHOD("get_fade_level"), &InputHapticEffectCustom::get_fade_level);

	ADD_PROPERTY(PropertyInfo(Variant::INT, "channels"), "set_channels", "get_channels");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "samples"), "set_samples", "get_samples");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "period"), "set_period", "get_period");
	ADD_PROPERTY(PropertyInfo(Variant::PACKED_INT32_ARRAY, "data"), "set_data", "get_data");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "attack_length"), "set_attack_length", "get_attack_length");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "attack_level"), "set_attack_level", "get_attack_level");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "fade_length"), "set_fade_length", "get_fade_length");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "fade_level"), "set_fade_level", "get_fade_level");
}

void InputHapticEffectCustom::set_channels(int p_channels) {
	channels = CLAMP(p_channels, 0, UINT8_MAX);
}

int InputHapticEffectCustom::get_channels() const {
	return channels;
}

void InputHapticEffectCustom::set_samples(int p_samples) {
	samples = CLAMP(p_samples, 0, UINT16_MAX);
}

int InputHapticEffectCustom::get_samples() const {
	return samples;
}

void InputHapticEffectCustom::set_period(float p_period) {
	period = CLAMP(p_period, 0.0f, UINT16_MAX / 1000.0f);
}

float InputHapticEffectCustom::get_period() const {
	return period;
}

void InputHapticEffectCustom::set_data(const PackedInt32Array &p_data) {
	data = p_data;
}

PackedInt32Array InputHapticEffectCustom::get_data() const {
	return data;
}

void InputHapticEffectCustom::set_attack_length(float p_attack_length) {
	attack_length = CLAMP(p_attack_length, 0.0f, UINT16_MAX / 1000.0f);
}

float InputHapticEffectCustom::get_attack_length() const {
	return attack_length;
}

void InputHapticEffectCustom::set_attack_level(float p_attack_level) {
	attack_level = CLAMP(p_attack_level, 0.0, 1.0);
}

float InputHapticEffectCustom::get_attack_level() const {
	return attack_level;
}

void InputHapticEffectCustom::set_fade_length(float p_fade_length) {
	fade_length = CLAMP(p_fade_length, 0.0f, UINT16_MAX / 1000.0f);
}

float InputHapticEffectCustom::get_fade_length() const {
	return fade_length;
}

void InputHapticEffectCustom::set_fade_level(float p_fade_level) {
	fade_level = CLAMP(p_fade_level, 0.0, 1.0);
}

float InputHapticEffectCustom::get_fade_level() const {
	return fade_level;
}

///////////////////////////////////
