/**************************************************************************/
/*  animation.hpp                                                         */
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

#include <godot_cpp/classes/ref.hpp>
#include <godot_cpp/classes/resource.hpp>
#include <godot_cpp/variant/array.hpp>
#include <godot_cpp/variant/color.hpp>
#include <godot_cpp/variant/node_path.hpp>
#include <godot_cpp/variant/packed_string_array.hpp>
#include <godot_cpp/variant/quaternion.hpp>
#include <godot_cpp/variant/string_name.hpp>
#include <godot_cpp/variant/variant.hpp>
#include <godot_cpp/variant/vector2.hpp>
#include <godot_cpp/variant/vector3.hpp>

#include <godot_cpp/core/class_db.hpp>

#include <type_traits>

namespace godot {

class Animation : public Resource {
	GDEXTENSION_CLASS(Animation, Resource)

public:
	enum TrackType {
		TYPE_VALUE = 0,
		TYPE_POSITION_3D = 1,
		TYPE_ROTATION_3D = 2,
		TYPE_SCALE_3D = 3,
		TYPE_BLEND_SHAPE = 4,
		TYPE_METHOD = 5,
		TYPE_BEZIER = 6,
		TYPE_AUDIO = 7,
		TYPE_ANIMATION = 8,
	};

	enum InterpolationType {
		INTERPOLATION_NEAREST = 0,
		INTERPOLATION_LINEAR = 1,
		INTERPOLATION_CUBIC = 2,
		INTERPOLATION_LINEAR_ANGLE = 3,
		INTERPOLATION_CUBIC_ANGLE = 4,
	};

	enum UpdateMode {
		UPDATE_CONTINUOUS = 0,
		UPDATE_DISCRETE = 1,
		UPDATE_CAPTURE = 2,
	};

	enum LoopMode {
		LOOP_NONE = 0,
		LOOP_LINEAR = 1,
		LOOP_PINGPONG = 2,
	};

	enum LoopedFlag {
		LOOPED_FLAG_NONE = 0,
		LOOPED_FLAG_END = 1,
		LOOPED_FLAG_START = 2,
	};

	enum FindMode {
		FIND_MODE_NEAREST = 0,
		FIND_MODE_APPROX = 1,
		FIND_MODE_EXACT = 2,
	};

	int32_t add_track(Animation::TrackType p_type, int32_t p_at_position = -1);
	void remove_track(int32_t p_track_idx);
	int32_t get_track_count() const;
	Animation::TrackType track_get_type(int32_t p_track_idx) const;
	NodePath track_get_path(int32_t p_track_idx) const;
	void track_set_path(int32_t p_track_idx, const NodePath &p_path);
	int32_t find_track(const NodePath &p_path, Animation::TrackType p_type) const;
	void track_move_up(int32_t p_track_idx);
	void track_move_down(int32_t p_track_idx);
	void track_move_to(int32_t p_track_idx, int32_t p_to_idx);
	void track_swap(int32_t p_track_idx, int32_t p_with_idx);
	void track_set_imported(int32_t p_track_idx, bool p_imported);
	bool track_is_imported(int32_t p_track_idx) const;
	void track_set_enabled(int32_t p_track_idx, bool p_enabled);
	bool track_is_enabled(int32_t p_track_idx) const;
	int32_t position_track_insert_key(int32_t p_track_idx, double p_time, const Vector3 &p_position);
	int32_t rotation_track_insert_key(int32_t p_track_idx, double p_time, const Quaternion &p_rotation);
	int32_t scale_track_insert_key(int32_t p_track_idx, double p_time, const Vector3 &p_scale);
	int32_t blend_shape_track_insert_key(int32_t p_track_idx, double p_time, float p_amount);
	Vector3 position_track_interpolate(int32_t p_track_idx, double p_time_sec, bool p_backward = false) const;
	Quaternion rotation_track_interpolate(int32_t p_track_idx, double p_time_sec, bool p_backward = false) const;
	Vector3 scale_track_interpolate(int32_t p_track_idx, double p_time_sec, bool p_backward = false) const;
	float blend_shape_track_interpolate(int32_t p_track_idx, double p_time_sec, bool p_backward = false) const;
	int32_t track_insert_key(int32_t p_track_idx, double p_time, const Variant &p_key, float p_transition = 1);
	void track_remove_key(int32_t p_track_idx, int32_t p_key_idx);
	void track_remove_key_at_time(int32_t p_track_idx, double p_time);
	void track_set_key_value(int32_t p_track_idx, int32_t p_key, const Variant &p_value);
	void track_set_key_transition(int32_t p_track_idx, int32_t p_key_idx, float p_transition);
	void track_set_key_time(int32_t p_track_idx, int32_t p_key_idx, double p_time);
	float track_get_key_transition(int32_t p_track_idx, int32_t p_key_idx) const;
	int32_t track_get_key_count(int32_t p_track_idx) const;
	Variant track_get_key_value(int32_t p_track_idx, int32_t p_key_idx) const;
	double track_get_key_time(int32_t p_track_idx, int32_t p_key_idx) const;
	int32_t track_find_key(int32_t p_track_idx, double p_time, Animation::FindMode p_find_mode = (Animation::FindMode)0, bool p_limit = false, bool p_backward = false) const;
	void track_set_interpolation_type(int32_t p_track_idx, Animation::InterpolationType p_interpolation);
	Animation::InterpolationType track_get_interpolation_type(int32_t p_track_idx) const;
	void track_set_interpolation_loop_wrap(int32_t p_track_idx, bool p_interpolation);
	bool track_get_interpolation_loop_wrap(int32_t p_track_idx) const;
	bool track_is_compressed(int32_t p_track_idx) const;
	void value_track_set_update_mode(int32_t p_track_idx, Animation::UpdateMode p_mode);
	Animation::UpdateMode value_track_get_update_mode(int32_t p_track_idx) const;
	Variant value_track_interpolate(int32_t p_track_idx, double p_time_sec, bool p_backward = false) const;
	StringName method_track_get_name(int32_t p_track_idx, int32_t p_key_idx) const;
	Array method_track_get_params(int32_t p_track_idx, int32_t p_key_idx) const;
	int32_t bezier_track_insert_key(int32_t p_track_idx, double p_time, float p_value, const Vector2 &p_in_handle = Vector2(0, 0), const Vector2 &p_out_handle = Vector2(0, 0));
	void bezier_track_set_key_value(int32_t p_track_idx, int32_t p_key_idx, float p_value);
	void bezier_track_set_key_in_handle(int32_t p_track_idx, int32_t p_key_idx, const Vector2 &p_in_handle, float p_balanced_value_time_ratio = 1.0);
	void bezier_track_set_key_out_handle(int32_t p_track_idx, int32_t p_key_idx, const Vector2 &p_out_handle, float p_balanced_value_time_ratio = 1.0);
	float bezier_track_get_key_value(int32_t p_track_idx, int32_t p_key_idx) const;
	Vector2 bezier_track_get_key_in_handle(int32_t p_track_idx, int32_t p_key_idx) const;
	Vector2 bezier_track_get_key_out_handle(int32_t p_track_idx, int32_t p_key_idx) const;
	float bezier_track_interpolate(int32_t p_track_idx, double p_time) const;
	int32_t audio_track_insert_key(int32_t p_track_idx, double p_time, const Ref<Resource> &p_stream, float p_start_offset = 0, float p_end_offset = 0);
	void audio_track_set_key_stream(int32_t p_track_idx, int32_t p_key_idx, const Ref<Resource> &p_stream);
	void audio_track_set_key_start_offset(int32_t p_track_idx, int32_t p_key_idx, float p_offset);
	void audio_track_set_key_end_offset(int32_t p_track_idx, int32_t p_key_idx, float p_offset);
	Ref<Resource> audio_track_get_key_stream(int32_t p_track_idx, int32_t p_key_idx) const;
	float audio_track_get_key_start_offset(int32_t p_track_idx, int32_t p_key_idx) const;
	float audio_track_get_key_end_offset(int32_t p_track_idx, int32_t p_key_idx) const;
	void audio_track_set_use_blend(int32_t p_track_idx, bool p_enable);
	bool audio_track_is_use_blend(int32_t p_track_idx) const;
	int32_t animation_track_insert_key(int32_t p_track_idx, double p_time, const StringName &p_animation);
	void animation_track_set_key_animation(int32_t p_track_idx, int32_t p_key_idx, const StringName &p_animation);
	StringName animation_track_get_key_animation(int32_t p_track_idx, int32_t p_key_idx) const;
	void add_marker(const StringName &p_name, double p_time);
	void remove_marker(const StringName &p_name);
	bool has_marker(const StringName &p_name) const;
	StringName get_marker_at_time(double p_time) const;
	StringName get_next_marker(double p_time) const;
	StringName get_prev_marker(double p_time) const;
	double get_marker_time(const StringName &p_name) const;
	PackedStringArray get_marker_names() const;
	Color get_marker_color(const StringName &p_name) const;
	void set_marker_color(const StringName &p_name, const Color &p_color);
	void set_length(float p_time_sec);
	float get_length() const;
	void set_loop_mode(Animation::LoopMode p_loop_mode);
	Animation::LoopMode get_loop_mode() const;
	void set_step(float p_size_sec);
	float get_step() const;
	void clear();
	void copy_track(int32_t p_track_idx, const Ref<Animation> &p_to_animation);
	void optimize(float p_allowed_velocity_err = 0.01, float p_allowed_angular_err = 0.01, int32_t p_precision = 3);
	void compress(uint32_t p_page_size = 8192, uint32_t p_fps = 120, float p_split_tolerance = 4.0);
	bool is_capture_included() const;

protected:
	template <typename T, typename B>
	static void register_virtuals() {
		Resource::register_virtuals<T, B>();
	}

public:
};

} // namespace godot

VARIANT_ENUM_CAST(Animation::TrackType);
VARIANT_ENUM_CAST(Animation::InterpolationType);
VARIANT_ENUM_CAST(Animation::UpdateMode);
VARIANT_ENUM_CAST(Animation::LoopMode);
VARIANT_ENUM_CAST(Animation::LoopedFlag);
VARIANT_ENUM_CAST(Animation::FindMode);

