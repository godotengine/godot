/*************************************************************************/
/*  animation.h                                                          */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2017 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2017 Godot Engine contributors (cf. AUTHORS.md)    */
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

#include "resource.h"
/**
	@author Juan Linietsky <reduzio@gmail.com>
*/
class Animation : public Resource {

	GDCLASS(Animation, Resource);
	RES_BASE_EXTENSION("anim");

public:
	enum TrackType {
		TYPE_VALUE, ///< Set a value in a property, can be interpolated.
		TYPE_TRANSFORM, ///< Transform a node or a bone.
		TYPE_METHOD, ///< Call any method on a specific node.
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

	};

private:
	struct Track {

		TrackType type;
		InterpolationType interpolation;
		bool loop_wrap;
		NodePath path; // path to something
		bool imported;
		Track() {
			interpolation = INTERPOLATION_LINEAR;
			imported = false;
			loop_wrap = true;
		}
		virtual ~Track() {}
	};

	struct Key {

		float transition;
		float time; // time in secs
		Key() { transition = 1; }
	};

	// transform key holds either Vector3 or Quaternion
	template <class T>
	struct TKey : public Key {

		float time;
		T value;
	};

	struct TransformKey {

		Vector3 loc;
		Quat rot;
		Vector3 scale;
	};

	/* TRANSFORM TRACK */

	struct TransformTrack : public Track {

		Vector<TKey<TransformKey> > transforms;

		TransformTrack() { type = TYPE_TRANSFORM; }
	};

	/* PROPERTY VALUE TRACK */

	struct ValueTrack : public Track {

		UpdateMode update_mode;
		bool update_on_seek;
		Vector<TKey<Variant> > values;

		ValueTrack() {
			type = TYPE_VALUE;
			update_mode = UPDATE_CONTINUOUS;
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

	Vector<Track *> tracks;

	/*
	template<class T>
	int _insert_pos(float p_time, T& p_keys);*/

	template <class T>
	void _clear(T &p_keys);

	template <class T, class V>
	int _insert(float p_time, T &p_keys, const V &p_value);

	template <class K>
	inline int _find(const Vector<K> &p_keys, float p_time) const;

	_FORCE_INLINE_ Animation::TransformKey _interpolate(const Animation::TransformKey &p_a, const Animation::TransformKey &p_b, float p_c) const;

	_FORCE_INLINE_ Vector3 _interpolate(const Vector3 &p_a, const Vector3 &p_b, float p_c) const;
	_FORCE_INLINE_ Quat _interpolate(const Quat &p_a, const Quat &p_b, float p_c) const;
	_FORCE_INLINE_ Variant _interpolate(const Variant &p_a, const Variant &p_b, float p_c) const;
	_FORCE_INLINE_ float _interpolate(const float &p_a, const float &p_b, float p_c) const;

	_FORCE_INLINE_ Animation::TransformKey _cubic_interpolate(const Animation::TransformKey &p_pre_a, const Animation::TransformKey &p_a, const Animation::TransformKey &p_b, const Animation::TransformKey &p_post_b, float p_c) const;
	_FORCE_INLINE_ Vector3 _cubic_interpolate(const Vector3 &p_pre_a, const Vector3 &p_a, const Vector3 &p_b, const Vector3 &p_post_b, float p_c) const;
	_FORCE_INLINE_ Quat _cubic_interpolate(const Quat &p_pre_a, const Quat &p_a, const Quat &p_b, const Quat &p_post_b, float p_c) const;
	_FORCE_INLINE_ Variant _cubic_interpolate(const Variant &p_pre_a, const Variant &p_a, const Variant &p_b, const Variant &p_post_b, float p_c) const;
	_FORCE_INLINE_ float _cubic_interpolate(const float &p_pre_a, const float &p_a, const float &p_b, const float &p_post_b, float p_c) const;

	template <class T>
	_FORCE_INLINE_ T _interpolate(const Vector<TKey<T> > &p_keys, float p_time, InterpolationType p_interp, bool p_loop_wrap, bool *p_ok) const;

	_FORCE_INLINE_ void _value_track_get_key_indices_in_range(const ValueTrack *vt, float from_time, float to_time, List<int> *p_indices) const;
	_FORCE_INLINE_ void _method_track_get_key_indices_in_range(const MethodTrack *mt, float from_time, float to_time, List<int> *p_indices) const;

	float length;
	float step;
	bool loop;

	// bind helpers
private:
	Array _transform_track_interpolate(int p_track, float p_time) const {
		Vector3 loc;
		Quat rot;
		Vector3 scale;
		transform_track_interpolate(p_track, p_time, &loc, &rot, &scale);
		Array ret;
		ret.push_back(loc);
		ret.push_back(rot);
		ret.push_back(scale);
		return ret;
	}

	PoolVector<int> _value_track_get_key_indices(int p_track, float p_time, float p_delta) const {

		List<int> idxs;
		value_track_get_key_indices(p_track, p_time, p_delta, &idxs);
		PoolVector<int> idxr;

		for (List<int>::Element *E = idxs.front(); E; E = E->next()) {

			idxr.push_back(E->get());
		}
		return idxr;
	}
	PoolVector<int> _method_track_get_key_indices(int p_track, float p_time, float p_delta) const {

		List<int> idxs;
		method_track_get_key_indices(p_track, p_time, p_delta, &idxs);
		PoolVector<int> idxr;

		for (List<int>::Element *E = idxs.front(); E; E = E->next()) {

			idxr.push_back(E->get());
		}
		return idxr;
	}

	bool _transform_track_optimize_key(const TKey<TransformKey> &t0, const TKey<TransformKey> &t1, const TKey<TransformKey> &t2, float p_alowed_linear_err, float p_alowed_angular_err, float p_max_optimizable_angle, const Vector3 &p_norm);
	void _transform_track_optimize(int p_idx, float p_allowed_linear_err = 0.05, float p_allowed_angular_err = 0.01, float p_max_optimizable_angle = Math_PI * 0.125);

protected:
	bool _set(const StringName &p_name, const Variant &p_value);
	bool _get(const StringName &p_name, Variant &r_ret) const;
	void _get_property_list(List<PropertyInfo> *p_list) const;

	static void _bind_methods();

public:
	int add_track(TrackType p_type, int p_at_pos = -1);
	void remove_track(int p_track);

	int get_track_count() const;
	TrackType track_get_type(int p_track) const;

	void track_set_path(int p_track, const NodePath &p_path);
	NodePath track_get_path(int p_track) const;
	int find_track(const NodePath &p_path) const;
	// transform

	void track_move_up(int p_track);
	void track_move_down(int p_track);

	void track_set_imported(int p_track, bool p_imported);
	bool track_is_imported(int p_track) const;

	int transform_track_insert_key(int p_track, float p_time, const Vector3 p_loc, const Quat &p_rot = Quat(), const Vector3 &p_scale = Vector3());
	void track_insert_key(int p_track, float p_time, const Variant &p_key, float p_transition = 1);
	void track_set_key_transition(int p_track, int p_key_idx, float p_transition);
	void track_set_key_value(int p_track, int p_key_idx, const Variant &p_value);
	int track_find_key(int p_track, float p_time, bool p_exact = false) const;
	void track_remove_key(int p_track, int p_idx);
	void track_remove_key_at_position(int p_track, float p_pos);
	int track_get_key_count(int p_track) const;
	Variant track_get_key_value(int p_track, int p_key_idx) const;
	float track_get_key_time(int p_track, int p_key_idx) const;
	float track_get_key_transition(int p_track, int p_key_idx) const;

	Error transform_track_get_key(int p_track, int p_key, Vector3 *r_loc, Quat *r_rot, Vector3 *r_scale) const;
	void track_set_interpolation_type(int p_track, InterpolationType p_interp);
	InterpolationType track_get_interpolation_type(int p_track) const;

	void track_set_interpolation_loop_wrap(int p_track, bool p_enable);
	bool track_get_interpolation_loop_wrap(int p_track) const;

	Error transform_track_interpolate(int p_track, float p_time, Vector3 *r_loc, Quat *r_rot, Vector3 *r_scale) const;

	Variant value_track_interpolate(int p_track, float p_time) const;
	void value_track_get_key_indices(int p_track, float p_time, float p_delta, List<int> *p_indices) const;
	void value_track_set_update_mode(int p_track, UpdateMode p_mode);
	UpdateMode value_track_get_update_mode(int p_track) const;

	void method_track_get_key_indices(int p_track, float p_time, float p_delta, List<int> *p_indices) const;
	Vector<Variant> method_track_get_params(int p_track, int p_key_idx) const;
	StringName method_track_get_name(int p_track, int p_key_idx) const;

	void set_length(float p_length);
	float get_length() const;

	void set_loop(bool p_enabled);
	bool has_loop() const;

	void set_step(float p_step);
	float get_step() const;

	void clear();

	void optimize(float p_allowed_linear_err = 0.05, float p_allowed_angular_err = 0.01, float p_max_optimizable_angle = Math_PI * 0.125);

	Animation();
	~Animation();
};

VARIANT_ENUM_CAST(Animation::TrackType);
VARIANT_ENUM_CAST(Animation::InterpolationType);
VARIANT_ENUM_CAST(Animation::UpdateMode);

#endif
