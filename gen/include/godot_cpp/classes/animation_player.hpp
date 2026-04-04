/**************************************************************************/
/*  animation_player.hpp                                                  */
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

// THIS FILE IS GENERATED. EDITS WILL BE LOST.

#pragma once

#include <godot_cpp/classes/animation_mixer.hpp>
#include <godot_cpp/classes/tween.hpp>
#include <godot_cpp/variant/node_path.hpp>
#include <godot_cpp/variant/string_name.hpp>
#include <godot_cpp/variant/typed_array.hpp>

#include <godot_cpp/core/class_db.hpp>

#include <type_traits>

namespace godot {

class AnimationPlayer : public AnimationMixer {
	GDEXTENSION_CLASS(AnimationPlayer, AnimationMixer)

public:
	enum AnimationProcessCallback {
		ANIMATION_PROCESS_PHYSICS = 0,
		ANIMATION_PROCESS_IDLE = 1,
		ANIMATION_PROCESS_MANUAL = 2,
	};

	enum AnimationMethodCallMode {
		ANIMATION_METHOD_CALL_DEFERRED = 0,
		ANIMATION_METHOD_CALL_IMMEDIATE = 1,
	};

	void animation_set_next(const StringName &p_animation_from, const StringName &p_animation_to);
	StringName animation_get_next(const StringName &p_animation_from) const;
	void set_blend_time(const StringName &p_animation_from, const StringName &p_animation_to, double p_sec);
	double get_blend_time(const StringName &p_animation_from, const StringName &p_animation_to) const;
	void set_default_blend_time(double p_sec);
	double get_default_blend_time() const;
	void set_auto_capture(bool p_auto_capture);
	bool is_auto_capture() const;
	void set_auto_capture_duration(double p_auto_capture_duration);
	double get_auto_capture_duration() const;
	void set_auto_capture_transition_type(Tween::TransitionType p_auto_capture_transition_type);
	Tween::TransitionType get_auto_capture_transition_type() const;
	void set_auto_capture_ease_type(Tween::EaseType p_auto_capture_ease_type);
	Tween::EaseType get_auto_capture_ease_type() const;
	void play(const StringName &p_name = StringName(), double p_custom_blend = -1, float p_custom_speed = 1.0, bool p_from_end = false);
	void play_section_with_markers(const StringName &p_name = StringName(), const StringName &p_start_marker = StringName(), const StringName &p_end_marker = StringName(), double p_custom_blend = -1, float p_custom_speed = 1.0, bool p_from_end = false);
	void play_section(const StringName &p_name = StringName(), double p_start_time = -1, double p_end_time = -1, double p_custom_blend = -1, float p_custom_speed = 1.0, bool p_from_end = false);
	void play_backwards(const StringName &p_name = StringName(), double p_custom_blend = -1);
	void play_section_with_markers_backwards(const StringName &p_name = StringName(), const StringName &p_start_marker = StringName(), const StringName &p_end_marker = StringName(), double p_custom_blend = -1);
	void play_section_backwards(const StringName &p_name = StringName(), double p_start_time = -1, double p_end_time = -1, double p_custom_blend = -1);
	void play_with_capture(const StringName &p_name = StringName(), double p_duration = -1.0, double p_custom_blend = -1, float p_custom_speed = 1.0, bool p_from_end = false, Tween::TransitionType p_trans_type = (Tween::TransitionType)0, Tween::EaseType p_ease_type = (Tween::EaseType)0);
	void pause();
	void stop(bool p_keep_state = false);
	bool is_playing() const;
	bool is_animation_active() const;
	void set_current_animation(const StringName &p_animation);
	StringName get_current_animation() const;
	void set_assigned_animation(const StringName &p_animation);
	StringName get_assigned_animation() const;
	void queue(const StringName &p_name);
	TypedArray<StringName> get_queue();
	void clear_queue();
	void set_speed_scale(float p_speed);
	float get_speed_scale() const;
	float get_playing_speed() const;
	void set_autoplay(const StringName &p_name);
	StringName get_autoplay() const;
	void set_movie_quit_on_finish_enabled(bool p_enabled);
	bool is_movie_quit_on_finish_enabled() const;
	double get_current_animation_position() const;
	double get_current_animation_length() const;
	void set_section_with_markers(const StringName &p_start_marker = StringName(), const StringName &p_end_marker = StringName());
	void set_section(double p_start_time = -1, double p_end_time = -1);
	void reset_section();
	double get_section_start_time() const;
	double get_section_end_time() const;
	bool has_section() const;
	void seek(double p_seconds, bool p_update = false, bool p_update_only = false);
	void set_process_callback(AnimationPlayer::AnimationProcessCallback p_mode);
	AnimationPlayer::AnimationProcessCallback get_process_callback() const;
	void set_method_call_mode(AnimationPlayer::AnimationMethodCallMode p_mode);
	AnimationPlayer::AnimationMethodCallMode get_method_call_mode() const;
	void set_root(const NodePath &p_path);
	NodePath get_root() const;

protected:
	template <typename T, typename B>
	static void register_virtuals() {
		AnimationMixer::register_virtuals<T, B>();
	}

public:
};

} // namespace godot

VARIANT_ENUM_CAST(AnimationPlayer::AnimationProcessCallback);
VARIANT_ENUM_CAST(AnimationPlayer::AnimationMethodCallMode);

