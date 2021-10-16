/*************************************************************************/
/*  animation.h                                                          */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2021 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2021 Godot Engine contributors (cf. AUTHORS.md).   */
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

#ifndef ANIMATION_H
#define ANIMATION_H

#include "core/io/resource.h"

#define ANIM_MIN_LENGTH 0.001

class Animation : public Resource {
	GDCLASS(Animation, Resource);
	RES_BASE_EXTENSION("anim");

public:
	enum TrackType {
		TYPE_VALUE, ///< Set a value in a property, can be interpolated.
		TYPE_POSITION_3D, ///< Position 3D track
		TYPE_ROTATION_3D, ///< Rotation 3D track
		TYPE_SCALE_3D, ///< Scale 3D track
		TYPE_BLEND_SHAPE, ///< Blend Shape track
		TYPE_METHOD, ///< Call any method on a specific node.
		TYPE_BEZIER, ///< Bezier curve
		TYPE_AUDIO,
		TYPE_ANIMATION,
	};

	enum InterpolationType {
		INTERPOLATION_NEAREST,
		INTERPOLATION_LINEAR,
		INTERPOLATION_CUBIC
	};

	enum UpdateMode {
		UPDATE_CONTINUOUS,
		UPDATE_DISCRETE,
		UPDATE_TRIGGER,
		UPDATE_CAPTURE,

	};

private:
	struct Track {
		TrackType type = TrackType::TYPE_ANIMATION;
		InterpolationType interpolation = INTERPOLATION_LINEAR;
		bool loop_wrap = true;
		NodePath path; // path to something
		bool imported = false;
		bool enabled = true;
		Track() {}
		virtual ~Track() {}
	};

	struct Key {
		real_t transition = 1.0;
		double time = 0.0; // time in secs
	};

	// transform key holds either Vector3 or Quaternion
	template <class T>
	struct TKey : public Key {
		T value;
	};

	const int32_t POSITION_TRACK_SIZE = 5;
	const int32_t ROTATION_TRACK_SIZE = 6;
	const int32_t SCALE_TRACK_SIZE = 5;
	const int32_t BLEND_SHAPE_TRACK_SIZE = 3;

	/* POSITION TRACK */

	struct PositionTrack : public Track {
		Vector<TKey<Vector3>> positions;

		PositionTrack() { type = TYPE_POSITION_3D; }
	};

	/* ROTATION TRACK */

	struct RotationTrack : public Track {
		Vector<TKey<Quaternion>> rotations;

		RotationTrack() { type = TYPE_ROTATION_3D; }
	};

	/* SCALE TRACK */

	struct ScaleTrack : public Track {
		Vector<TKey<Vector3>> scales;
		ScaleTrack() { type = TYPE_SCALE_3D; }
	};

	/* BLEND SHAPE TRACK */

	struct BlendShapeTrack : public Track {
		Vector<TKey<float>> blend_shapes;
		BlendShapeTrack() { type = TYPE_BLEND_SHAPE; }
	};

	/* PROPERTY VALUE TRACK */

	struct ValueTrack : public Track {
		UpdateMode update_mode = UPDATE_CONTINUOUS;
		bool update_on_seek = false;
		Vector<TKey<Variant>> values;

		ValueTrack() {
			type = TYPE_VALUE;
		}
	};

	/* METHOD TRACK */

	struct MethodKey : public Key {
		StringName method;
		Vector<Variant> params;
	};

	struct MethodTrack : public Track {
		Vector<MethodKey> methods;
		MethodTrack() { type = TYPE_METHOD; }
	};

	/* BEZIER TRACK */

	struct BezierKey {
		Vector2 in_handle; //relative (x always <0)
		Vector2 out_handle; //relative (x always >0)
		real_t value = 0.0;
	};

	struct BezierTrack : public Track {
		Vector<TKey<BezierKey>> values;

		BezierTrack() {
			type = TYPE_BEZIER;
		}
	};

	/* AUDIO TRACK */

	struct AudioKey {
		RES stream;
		real_t start_offset = 0.0; //offset from start
		real_t end_offset = 0.0; //offset from end, if 0 then full length or infinite
		AudioKey() {
		}
	};

	struct AudioTrack : public Track {
		Vector<TKey<AudioKey>> values;

		AudioTrack() {
			type = TYPE_AUDIO;
		}
	};

	/* AUDIO TRACK */

	struct AnimationTrack : public Track {
		Vector<TKey<StringName>> values;

		AnimationTrack() {
			type = TYPE_ANIMATION;
		}
	};

	Vector<Track *> tracks;

	/*
	template<class T>
	int _insert_pos(double p_time, T& p_keys);*/

	template <class T>
	void _clear(T &p_keys);

	template <class T, class V>
	int _insert(double p_time, T &p_keys, const V &p_value);

	template <class K>
	inline int _find(const Vector<K> &p_keys, double p_time) const;

	_FORCE_INLINE_ Vector3 _interpolate(const Vector3 &p_a, const Vector3 &p_b, real_t p_c) const;
	_FORCE_INLINE_ Quaternion _interpolate(const Quaternion &p_a, const Quaternion &p_b, real_t p_c) const;
	_FORCE_INLINE_ Variant _interpolate(const Variant &p_a, const Variant &p_b, real_t p_c) const;
	_FORCE_INLINE_ real_t _interpolate(const real_t &p_a, const real_t &p_b, real_t p_c) const;

	_FORCE_INLINE_ Vector3 _cubic_interpolate(const Vector3 &p_pre_a, const Vector3 &p_a, const Vector3 &p_b, const Vector3 &p_post_b, real_t p_c) const;
	_FORCE_INLINE_ Quaternion _cubic_interpolate(const Quaternion &p_pre_a, const Quaternion &p_a, const Quaternion &p_b, const Quaternion &p_post_b, real_t p_c) const;
	_FORCE_INLINE_ Variant _cubic_interpolate(const Variant &p_pre_a, const Variant &p_a, const Variant &p_b, const Variant &p_post_b, real_t p_c) const;
	_FORCE_INLINE_ real_t _cubic_interpolate(const real_t &p_pre_a, const real_t &p_a, const real_t &p_b, const real_t &p_post_b, real_t p_c) const;

	template <class T>
	_FORCE_INLINE_ T _interpolate(const Vector<TKey<T>> &p_keys, double p_time, InterpolationType p_interp, bool p_loop_wrap, bool *p_ok) const;

	template <class T>
	_FORCE_INLINE_ void _track_get_key_indices_in_range(const Vector<T> &p_array, double from_time, double to_time, List<int> *p_indices) const;

	_FORCE_INLINE_ void _value_track_get_key_indices_in_range(const ValueTrack *vt, double from_time, double to_time, List<int> *p_indices) const;
	_FORCE_INLINE_ void _method_track_get_key_indices_in_range(const MethodTrack *mt, double from_time, double to_time, List<int> *p_indices) const;

	double length = 1.0;
	real_t step = 0.1;
	bool loop = false;

	// bind helpers
private:
	Vector<int> _value_track_get_key_indices(int p_track, double p_time, double p_delta) const {
		List<int> idxs;
		value_track_get_key_indices(p_track, p_time, p_delta, &idxs);
		Vector<int> idxr;

		for (int &E : idxs) {
			idxr.push_back(E);
		}
		return idxr;
	}
	Vector<int> _method_track_get_key_indices(int p_track, double p_time, double p_delta) const {
		List<int> idxs;
		method_track_get_key_indices(p_track, p_time, p_delta, &idxs);
		Vector<int> idxr;

		for (int &E : idxs) {
			idxr.push_back(E);
		}
		return idxr;
	}

	bool _position_track_optimize_key(const TKey<Vector3> &t0, const TKey<Vector3> &t1, const TKey<Vector3> &t2, real_t p_alowed_linear_err, real_t p_allowed_angular_error, const Vector3 &p_norm);
	bool _rotation_track_optimize_key(const TKey<Quaternion> &t0, const TKey<Quaternion> &t1, const TKey<Quaternion> &t2, real_t p_allowed_angular_error, float p_max_optimizable_angle);
	bool _scale_track_optimize_key(const TKey<Vector3> &t0, const TKey<Vector3> &t1, const TKey<Vector3> &t2, real_t p_allowed_linear_error);
	bool _blend_shape_track_optimize_key(const TKey<float> &t0, const TKey<float> &t1, const TKey<float> &t2, real_t p_allowed_unit_error);

	void _position_track_optimize(int p_idx, real_t p_allowed_linear_err, real_t p_allowed_angular_err);
	void _rotation_track_optimize(int p_idx, real_t p_allowed_angular_err, real_t p_max_optimizable_angle);
	void _scale_track_optimize(int p_idx, real_t p_allowed_linear_err);
	void _blend_shape_track_optimize(int p_idx, real_t p_allowed_unit_error);

protected:
	bool _set(const StringName &p_name, const Variant &p_value);
	bool _get(const StringName &p_name, Variant &r_ret) const;
	void _get_property_list(List<PropertyInfo> *p_list) const;

	virtual void reset_state() override;

	static void _bind_methods();

public:
	int add_track(TrackType p_type, int p_at_pos = -1);
	void remove_track(int p_track);

	int get_track_count() const;
	TrackType track_get_type(int p_track) const;

	void track_set_path(int p_track, const NodePath &p_path);
	NodePath track_get_path(int p_track) const;
	int find_track(const NodePath &p_path, const TrackType p_type) const;

	void track_move_up(int p_track);
	void track_move_down(int p_track);
	void track_move_to(int p_track, int p_to_index);
	void track_swap(int p_track, int p_with_track);

	void track_set_imported(int p_track, bool p_imported);
	bool track_is_imported(int p_track) const;

	void track_set_enabled(int p_track, bool p_enabled);
	bool track_is_enabled(int p_track) const;

	void track_insert_key(int p_track, double p_time, const Variant &p_key, real_t p_transition = 1);
	void track_set_key_transition(int p_track, int p_key_idx, real_t p_transition);
	void track_set_key_value(int p_track, int p_key_idx, const Variant &p_value);
	void track_set_key_time(int p_track, int p_key_idx, double p_time);
	int track_find_key(int p_track, double p_time, bool p_exact = false) const;
	void track_remove_key(int p_track, int p_idx);
	void track_remove_key_at_time(int p_track, double p_time);
	int track_get_key_count(int p_track) const;
	Variant track_get_key_value(int p_track, int p_key_idx) const;
	double track_get_key_time(int p_track, int p_key_idx) const;
	real_t track_get_key_transition(int p_track, int p_key_idx) const;

	int position_track_insert_key(int p_track, double p_time, const Vector3 &p_position);
	Error position_track_get_key(int p_track, int p_key, Vector3 *r_position) const;
	Error position_track_interpolate(int p_track, double p_time, Vector3 *r_interpolation) const;

	int rotation_track_insert_key(int p_track, double p_time, const Quaternion &p_rotation);
	Error rotation_track_get_key(int p_track, int p_key, Quaternion *r_rotation) const;
	Error rotation_track_interpolate(int p_track, double p_time, Quaternion *r_interpolation) const;

	int scale_track_insert_key(int p_track, double p_time, const Vector3 &p_scale);
	Error scale_track_get_key(int p_track, int p_key, Vector3 *r_scale) const;
	Error scale_track_interpolate(int p_track, double p_time, Vector3 *r_interpolation) const;

	int blend_shape_track_insert_key(int p_track, double p_time, float p_blend);
	Error blend_shape_track_get_key(int p_track, int p_key, float *r_blend) const;
	Error blend_shape_track_interpolate(int p_track, double p_time, float *r_blend) const;

	void track_set_interpolation_type(int p_track, InterpolationType p_interp);
	InterpolationType track_get_interpolation_type(int p_track) const;

	int bezier_track_insert_key(int p_track, double p_time, real_t p_value, const Vector2 &p_in_handle, const Vector2 &p_out_handle);
	void bezier_track_set_key_value(int p_track, int p_index, real_t p_value);
	void bezier_track_set_key_in_handle(int p_track, int p_index, const Vector2 &p_handle);
	void bezier_track_set_key_out_handle(int p_track, int p_index, const Vector2 &p_handle);
	real_t bezier_track_get_key_value(int p_track, int p_index) const;
	Vector2 bezier_track_get_key_in_handle(int p_track, int p_index) const;
	Vector2 bezier_track_get_key_out_handle(int p_track, int p_index) const;

	real_t bezier_track_interpolate(int p_track, double p_time) const;

	int audio_track_insert_key(int p_track, double p_time, const RES &p_stream, real_t p_start_offset = 0, real_t p_end_offset = 0);
	void audio_track_set_key_stream(int p_track, int p_key, const RES &p_stream);
	void audio_track_set_key_start_offset(int p_track, int p_key, real_t p_offset);
	void audio_track_set_key_end_offset(int p_track, int p_key, real_t p_offset);
	RES audio_track_get_key_stream(int p_track, int p_key) const;
	real_t audio_track_get_key_start_offset(int p_track, int p_key) const;
	real_t audio_track_get_key_end_offset(int p_track, int p_key) const;

	int animation_track_insert_key(int p_track, double p_time, const StringName &p_animation);
	void animation_track_set_key_animation(int p_track, int p_key, const StringName &p_animation);
	StringName animation_track_get_key_animation(int p_track, int p_key) const;

	void track_set_interpolation_loop_wrap(int p_track, bool p_enable);
	bool track_get_interpolation_loop_wrap(int p_track) const;

	Variant value_track_interpolate(int p_track, double p_time) const;
	void value_track_get_key_indices(int p_track, double p_time, double p_delta, List<int> *p_indices) const;
	void value_track_set_update_mode(int p_track, UpdateMode p_mode);
	UpdateMode value_track_get_update_mode(int p_track) const;

	void method_track_get_key_indices(int p_track, double p_time, double p_delta, List<int> *p_indices) const;
	Vector<Variant> method_track_get_params(int p_track, int p_key_idx) const;
	StringName method_track_get_name(int p_track, int p_key_idx) const;

	void copy_track(int p_track, Ref<Animation> p_to_animation);

	void track_get_key_indices_in_range(int p_track, double p_time, double p_delta, List<int> *p_indices) const;

	void set_length(real_t p_length);
	real_t get_length() const;

	void set_loop(bool p_enabled);
	bool has_loop() const;

	void set_step(real_t p_step);
	real_t get_step() const;

	void clear();

	void optimize(real_t p_allowed_linear_err = 0.05, real_t p_allowed_angular_err = 0.01, real_t p_max_optimizable_angle = Math_PI * 0.125);

	Animation();
	~Animation();
};

VARIANT_ENUM_CAST(Animation::TrackType);
VARIANT_ENUM_CAST(Animation::InterpolationType);
VARIANT_ENUM_CAST(Animation::UpdateMode);

#endif
