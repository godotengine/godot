/*************************************************************************/
/*  tween.h                                                              */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2022 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2022 Godot Engine contributors (cf. AUTHORS.md).   */
/*                                                                       */
/* Permission is hereby granted, free of charge, to any person obtaining */
/* a copy of this software and associated documentation files (the       */
/* "Software"), to deal in the Software without restriction, including   */
/* without limitation the rights to use, copy, modify, merge, publish,   */
/* distribute, sublicense, and/or sell copies of the Software, and to    */
/* permit persons to whom the Software is furnished to do so, subject to */
/* the following conditions:                                             */
/*                                                                       */
/* The above copyright notice and this permission notice shall be        */
/* included in all copies or substantial portions of the Software.       */
/*                                                                       */
/* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,       */
/* EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF    */
/* MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.*/
/* IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY  */
/* CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,  */
/* TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE     */
/* SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.                */
/*************************************************************************/

#ifndef TWEEN_H
#define TWEEN_H

#include "scene/main/node.h"

class Tween : public Node {
	GDCLASS(Tween, Node);

public:
	enum TweenProcessMode {
		TWEEN_PROCESS_PHYSICS,
		TWEEN_PROCESS_IDLE,
	};

	enum TransitionType {
		TRANS_LINEAR,
		TRANS_SINE,
		TRANS_QUINT,
		TRANS_QUART,
		TRANS_QUAD,
		TRANS_EXPO,
		TRANS_ELASTIC,
		TRANS_CUBIC,
		TRANS_CIRC,
		TRANS_BOUNCE,
		TRANS_BACK,

		TRANS_COUNT,
	};

	enum EaseType {
		EASE_IN,
		EASE_OUT,
		EASE_IN_OUT,
		EASE_OUT_IN,

		EASE_COUNT,
	};

private:
	enum InterpolateType {

		INTER_PROPERTY,
		INTER_METHOD,
		FOLLOW_PROPERTY,
		FOLLOW_METHOD,
		TARGETING_PROPERTY,
		TARGETING_METHOD,
		INTER_CALLBACK,
	};

	struct InterpolateData {
		bool active;
		InterpolateType type;
		bool finish;
		bool call_deferred;
		real_t elapsed;
		ObjectID id;
		Vector<StringName> key;
		StringName concatenated_key;
		Variant initial_val;
		Variant delta_val;
		Variant final_val;
		ObjectID target_id;
		Vector<StringName> target_key;
		real_t duration;
		TransitionType trans_type;
		EaseType ease_type;
		real_t delay;
		int args;
		Variant arg[VARIANT_ARG_MAX];
		int uid;
		InterpolateData() {
			active = false;
			finish = false;
			call_deferred = false;
			uid = 0;
		}
	};

	String autoplay;
	TweenProcessMode tween_process_mode;
	bool repeat;
	float speed_scale;
	mutable int pending_update;
	int uid;
	bool was_stopped = false;
	List<InterpolateData> interpolates;

	struct PendingCommand {
		StringName key;
		int args;
		Variant arg[10];
	};
	List<PendingCommand> pending_commands;

	void _add_pending_command(StringName p_key, const Variant &p_arg1 = Variant(), const Variant &p_arg2 = Variant(), const Variant &p_arg3 = Variant(), const Variant &p_arg4 = Variant(), const Variant &p_arg5 = Variant(), const Variant &p_arg6 = Variant(), const Variant &p_arg7 = Variant(), const Variant &p_arg8 = Variant(), const Variant &p_arg9 = Variant(), const Variant &p_arg10 = Variant());
	void _process_pending_commands();

	typedef real_t (*interpolater)(real_t t, real_t b, real_t c, real_t d);
	static interpolater interpolaters[TRANS_COUNT][EASE_COUNT];

	Variant &_get_delta_val(InterpolateData &p_data);
	Variant _get_initial_val(const InterpolateData &p_data) const;
	Variant _get_final_val(const InterpolateData &p_data) const;
	Variant _run_equation(InterpolateData &p_data);
	bool _calc_delta_val(const Variant &p_initial_val, const Variant &p_final_val, Variant &p_delta_val);
	bool _apply_tween_value(InterpolateData &p_data, Variant &value);

	void _tween_process(float p_delta);
	void _remove_by_uid(int uid);
	void _push_interpolate_data(InterpolateData &p_data);
	bool _build_interpolation(InterpolateType p_interpolation_type, Object *p_object, NodePath *p_property, StringName *p_method, Variant p_initial_val, Variant p_final_val, real_t p_duration, TransitionType p_trans_type, EaseType p_ease_type, real_t p_delay);

protected:
	bool _set(const StringName &p_name, const Variant &p_value);
	bool _get(const StringName &p_name, Variant &r_ret) const;
	void _get_property_list(List<PropertyInfo> *p_list) const;
	void _notification(int p_what);

	static void _bind_methods();

public:
	static real_t run_equation(Tween::TransitionType p_trans_type, Tween::EaseType p_ease_type, real_t p_time, real_t p_initial, real_t p_delta, real_t p_duration);

	bool is_active() const;
	void set_active(bool p_active);

	bool is_repeat() const;
	void set_repeat(bool p_repeat);

	void set_tween_process_mode(TweenProcessMode p_mode);
	TweenProcessMode get_tween_process_mode() const;

	void set_speed_scale(float p_speed);
	float get_speed_scale() const;

	bool start();
	bool reset(Object *p_object, StringName p_key);
	bool reset_all();
	bool stop(Object *p_object, StringName p_key);
	bool stop_all();
	bool resume(Object *p_object, StringName p_key);
	bool resume_all();
	bool remove(Object *p_object, StringName p_key);
	bool remove_all();

	bool seek(real_t p_time);
	real_t tell() const;
	real_t get_runtime() const;

	bool interpolate_property(Object *p_object, NodePath p_property, Variant p_initial_val, Variant p_final_val, real_t p_duration, TransitionType p_trans_type = TRANS_LINEAR, EaseType p_ease_type = EASE_IN_OUT, real_t p_delay = 0);
	bool interpolate_method(Object *p_object, StringName p_method, Variant p_initial_val, Variant p_final_val, real_t p_duration, TransitionType p_trans_type = TRANS_LINEAR, EaseType p_ease_type = EASE_IN_OUT, real_t p_delay = 0);
	bool interpolate_callback(Object *p_object, real_t p_duration, String p_callback, VARIANT_ARG_DECLARE);
	bool interpolate_deferred_callback(Object *p_object, real_t p_duration, String p_callback, VARIANT_ARG_DECLARE);
	bool follow_property(Object *p_object, NodePath p_property, Variant p_initial_val, Object *p_target, NodePath p_target_property, real_t p_duration, TransitionType p_trans_type = TRANS_LINEAR, EaseType p_ease_type = EASE_IN_OUT, real_t p_delay = 0);
	bool follow_method(Object *p_object, StringName p_method, Variant p_initial_val, Object *p_target, StringName p_target_method, real_t p_duration, TransitionType p_trans_type = TRANS_LINEAR, EaseType p_ease_type = EASE_IN_OUT, real_t p_delay = 0);
	bool targeting_property(Object *p_object, NodePath p_property, Object *p_initial, NodePath p_initial_property, Variant p_final_val, real_t p_duration, TransitionType p_trans_type = TRANS_LINEAR, EaseType p_ease_type = EASE_IN_OUT, real_t p_delay = 0);
	bool targeting_method(Object *p_object, StringName p_method, Object *p_initial, StringName p_initial_method, Variant p_final_val, real_t p_duration, TransitionType p_trans_type = TRANS_LINEAR, EaseType p_ease_type = EASE_IN_OUT, real_t p_delay = 0);

	Tween();
	~Tween();
};

VARIANT_ENUM_CAST(Tween::TweenProcessMode);
VARIANT_ENUM_CAST(Tween::TransitionType);
VARIANT_ENUM_CAST(Tween::EaseType);

#endif
