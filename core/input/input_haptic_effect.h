/**************************************************************************/
/*  input_haptic_effect.h                                                 */
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

#pragma once

#include "core/object/ref_counted.h"

class InputHapticEffect : public RefCounted {
	GDCLASS(InputHapticEffect, RefCounted);

public:
	enum Type {
		HAPTIC_EFFECT_CONSTANT,
		HAPTIC_EFFECT_SINE,
		HAPTIC_EFFECT_SQUARE,
		HAPTIC_EFFECT_TRIANGLE,
		HAPTIC_EFFECT_SAWTOOTH_UP,
		HAPTIC_EFFECT_SAWTOOTH_DOWN,
		HAPTIC_EFFECT_RAMP,
		HAPTIC_EFFECT_SPRING,
		HAPTIC_EFFECT_DAMPER,
		HAPTIC_EFFECT_INERTIA,
		HAPTIC_EFFECT_FRICTION,
		HAPTIC_EFFECT_LEFT_RIGHT,
		HAPTIC_EFFECT_CUSTOM,
	};

	float direction_radians = 0;
	float duration = 0;

protected:
	static void _bind_methods();

public:
	virtual Type get_type() const = 0;

	void set_direction_radians(float p_direction_radians);
	float get_direction_radians() const;

	void set_duration(float p_duration);
	float get_duration() const;
};

class InputHapticEffectConstant : public InputHapticEffect {
	GDCLASS(InputHapticEffectConstant, InputHapticEffect);

	float force = 0;
	float attack_length = 0;
	float attack_level = 0;
	float fade_length = 0;
	float fade_level = 0;

protected:
	static void _bind_methods();

public:
	virtual Type get_type() const override { return HAPTIC_EFFECT_CONSTANT; }

	void set_force(float p_force);
	float get_force() const;

	void set_attack_length(float p_attack_length);
	float get_attack_length() const;

	void set_attack_level(float p_attack_level);
	float get_attack_level() const;

	void set_fade_length(float p_fade_length);
	float get_fade_length() const;

	void set_fade_level(float p_fade_level);
	float get_fade_level() const;
};

class InputHapticEffectPeriodic : public InputHapticEffect {
	GDCLASS(InputHapticEffectPeriodic, InputHapticEffect);

	Type type = HAPTIC_EFFECT_SINE;

	float period = 0;
	float magnitude = 0;
	float offset = 0;
	float phase = 0;

	float attack_length = 0;
	float attack_level = 0;
	float fade_length = 0;
	float fade_level = 0;

protected:
	static void _bind_methods();

public:
	virtual Type get_type() const override { return type; }
	void set_type(Type p_type);

	void set_period(float p_period);
	float get_period() const;

	void set_magnitude(float p_magnitude);
	float get_magnitude() const;

	void set_offset(float p_offset);
	float get_offset() const;

	void set_phase(float p_phase);
	float get_phase() const;

	void set_attack_length(float p_attack_length);
	float get_attack_length() const;

	void set_attack_level(float p_attack_level);
	float get_attack_level() const;

	void set_fade_length(float p_fade_length);
	float get_fade_length() const;

	void set_fade_level(float p_fade_level);
	float get_fade_level() const;
};

class InputHapticEffectCondition : public InputHapticEffect {
	GDCLASS(InputHapticEffectCondition, InputHapticEffect);

	Type type = HAPTIC_EFFECT_SPRING;

	Vector3 right_level;
	Vector3 left_level;
	Vector3 right_coef;
	Vector3 left_coef;
	Vector3 deadband;
	Vector3 center;

protected:
	static void _bind_methods();

public:
	virtual Type get_type() const override { return type; }
	void set_type(Type p_type);

	void set_right_level(const Vector3 &p_right_level);
	Vector3 get_right_level() const;

	void set_left_level(const Vector3 &p_left_level);
	Vector3 get_left_level() const;

	void set_right_coef(const Vector3 &p_right_coef);
	Vector3 get_right_coef() const;

	void set_left_coef(const Vector3 &p_left_coef);
	Vector3 get_left_coef() const;

	void set_deadband(const Vector3 &p_deadband);
	Vector3 get_deadband() const;

	void set_center(const Vector3 &p_center);
	Vector3 get_center() const;
};

class InputHapticEffectLeftRight : public InputHapticEffect {
	GDCLASS(InputHapticEffectLeftRight, InputHapticEffect);

	float large_magnitude = 0;
	float small_magnitude = 0;

protected:
	static void _bind_methods();

public:
	virtual Type get_type() const override { return HAPTIC_EFFECT_LEFT_RIGHT; }

	void set_large_magnitude(float p_large_magnitude);
	float get_large_magnitude() const;

	void set_small_magnitude(float p_small_magnitude);
	float get_small_magnitude() const;
};

class InputHapticEffectRamp : public InputHapticEffect {
	GDCLASS(InputHapticEffectRamp, InputHapticEffect);

	float start_strength = 0;
	float end_strength = 0;

	float attack_length = 0;
	float attack_level = 0;
	float fade_length = 0;
	float fade_level = 0;

protected:
	static void _bind_methods();

public:
	virtual Type get_type() const override { return HAPTIC_EFFECT_RAMP; }

	void set_start_strength(float p_start_strength);
	float get_start_strength() const;

	void set_end_strength(float p_end_strength);
	float get_end_strength() const;

	void set_attack_length(float p_attack_length);
	float get_attack_length() const;

	void set_attack_level(float p_attack_level);
	float get_attack_level() const;

	void set_fade_length(float p_fade_length);
	float get_fade_length() const;

	void set_fade_level(float p_fade_level);
	float get_fade_level() const;
};

class InputHapticEffectCustom : public InputHapticEffect {
	GDCLASS(InputHapticEffectCustom, InputHapticEffect);

	int channels = 0;
	int samples = 0;
	float period = 0;
	PackedInt32Array data;

	float attack_length = 0;
	float attack_level = 0;
	float fade_length = 0;
	float fade_level = 0;

protected:
	static void _bind_methods();

public:
	virtual Type get_type() const override { return HAPTIC_EFFECT_CUSTOM; }

	void set_channels(int p_channels);
	int get_channels() const;

	void set_samples(int p_samples);
	int get_samples() const;

	void set_period(float p_period);
	float get_period() const;

	void set_data(const PackedInt32Array &p_data);
	PackedInt32Array get_data() const;

	void set_attack_length(float p_attack_length);
	float get_attack_length() const;

	void set_attack_level(float p_attack_level);
	float get_attack_level() const;

	void set_fade_length(float p_fade_length);
	float get_fade_length() const;

	void set_fade_level(float p_fade_level);
	float get_fade_level() const;
};
