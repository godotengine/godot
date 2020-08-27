/*************************************************************************/
/*  curve.h                                                              */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2020 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2020 Godot Engine contributors (cf. AUTHORS.md).   */
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

#ifndef CURVE_H
#define CURVE_H

#include "core/resource.h"

// y(x) curve
class Curve : public Resource {
	GDCLASS(Curve, Resource);

public:
	static const int MIN_X = 0.f;
	static const int MAX_X = 1.f;

	static const char *SIGNAL_RANGE_CHANGED;

	enum TangentMode {
		TANGENT_FREE = 0,
		TANGENT_LINEAR,
		TANGENT_MODE_COUNT
	};

	struct Point {
		Vector2 pos;
		real_t left_tangent;
		real_t right_tangent;
		TangentMode left_mode;
		TangentMode right_mode;

		Point() {
			left_tangent = 0;
			right_tangent = 0;
			left_mode = TANGENT_FREE;
			right_mode = TANGENT_FREE;
		}

		Point(Vector2 p_pos,
				real_t p_left = 0,
				real_t p_right = 0,
				TangentMode p_left_mode = TANGENT_FREE,
				TangentMode p_right_mode = TANGENT_FREE) {
			pos = p_pos;
			left_tangent = p_left;
			right_tangent = p_right;
			left_mode = p_left_mode;
			right_mode = p_right_mode;
		}
	};

	Curve();

	int get_point_count() const { return _points.size(); }

	int add_point(Vector2 p_pos,
			real_t left_tangent = 0,
			real_t right_tangent = 0,
			TangentMode left_mode = TANGENT_FREE,
			TangentMode right_mode = TANGENT_FREE);

	void remove_point(int p_index);
	void clear_points();

	int get_index(real_t offset) const;

	void set_point_value(int p_index, real_t pos);
	int set_point_offset(int p_index, float offset);
	Vector2 get_point_position(int p_index) const;

	Point get_point(int p_index) const;

	float get_min_value() const { return _min_value; }
	void set_min_value(float p_min);

	float get_max_value() const { return _max_value; }
	void set_max_value(float p_max);

	real_t interpolate(real_t offset) const;
	real_t interpolate_local_nocheck(int index, real_t local_offset) const;

	void clean_dupes();

	void set_point_left_tangent(int i, real_t tangent);
	void set_point_right_tangent(int i, real_t tangent);
	void set_point_left_mode(int i, TangentMode p_mode);
	void set_point_right_mode(int i, TangentMode p_mode);

	real_t get_point_left_tangent(int i) const;
	real_t get_point_right_tangent(int i) const;
	TangentMode get_point_left_mode(int i) const;
	TangentMode get_point_right_mode(int i) const;

	void update_auto_tangents(int i);

	Array get_data() const;
	void set_data(Array input);

	void bake();
	int get_bake_resolution() const { return _bake_resolution; }
	void set_bake_resolution(int p_resolution);
	real_t interpolate_baked(real_t offset);

	void ensure_default_setup(float p_min, float p_max);

protected:
	static void _bind_methods();

private:
	void mark_dirty();

	Vector<Point> _points;
	bool _baked_cache_dirty;
	Vector<real_t> _baked_cache;
	int _bake_resolution;
	float _min_value;
	float _max_value;
	int _minmax_set_once; // Encodes whether min and max have been set a first time, first bit for min and second for max.
};

VARIANT_ENUM_CAST(Curve::TangentMode)

class Curve2D : public Resource {
	GDCLASS(Curve2D, Resource);

	struct Point {
		Vector2 in;
		Vector2 out;
		Vector2 pos;
	};

	Vector<Point> points;

	struct BakedPoint {
		float ofs;
		Vector2 point;
	};

	mutable bool baked_cache_dirty;
	mutable PackedVector2Array baked_point_cache;
	mutable float baked_max_ofs;

	void _bake() const;

	float bake_interval;

	void _bake_segment2d(Map<float, Vector2> &r_bake, float p_begin, float p_end, const Vector2 &p_a, const Vector2 &p_out, const Vector2 &p_b, const Vector2 &p_in, int p_depth, int p_max_depth, float p_tol) const;
	Dictionary _get_data() const;
	void _set_data(const Dictionary &p_data);

protected:
	static void _bind_methods();

public:
	int get_point_count() const;
	void add_point(const Vector2 &p_pos, const Vector2 &p_in = Vector2(), const Vector2 &p_out = Vector2(), int p_atpos = -1);
	void set_point_position(int p_index, const Vector2 &p_pos);
	Vector2 get_point_position(int p_index) const;
	void set_point_in(int p_index, const Vector2 &p_in);
	Vector2 get_point_in(int p_index) const;
	void set_point_out(int p_index, const Vector2 &p_out);
	Vector2 get_point_out(int p_index) const;
	void remove_point(int p_index);
	void clear_points();

	Vector2 interpolate(int p_index, float p_offset) const;
	Vector2 interpolatef(real_t p_findex) const;

	void set_bake_interval(float p_tolerance);
	float get_bake_interval() const;

	float get_baked_length() const;
	Vector2 interpolate_baked(float p_offset, bool p_cubic = false) const;
	PackedVector2Array get_baked_points() const; //useful for going through
	Vector2 get_closest_point(const Vector2 &p_to_point) const;
	float get_closest_offset(const Vector2 &p_to_point) const;

	PackedVector2Array tessellate(int p_max_stages = 5, float p_tolerance = 4) const; //useful for display

	Curve2D();
};

class Curve3D : public Resource {
	GDCLASS(Curve3D, Resource);

	struct Point {
		Vector3 in;
		Vector3 out;
		Vector3 pos;
		float tilt;

		Point() { tilt = 0; }
	};

	Vector<Point> points;

	struct BakedPoint {
		float ofs;
		Vector3 point;
	};

	mutable bool baked_cache_dirty;
	mutable PackedVector3Array baked_point_cache;
	mutable PackedFloat32Array baked_tilt_cache;
	mutable PackedVector3Array baked_up_vector_cache;
	mutable float baked_max_ofs;

	void _bake() const;

	float bake_interval;
	bool up_vector_enabled;

	void _bake_segment3d(Map<float, Vector3> &r_bake, float p_begin, float p_end, const Vector3 &p_a, const Vector3 &p_out, const Vector3 &p_b, const Vector3 &p_in, int p_depth, int p_max_depth, float p_tol) const;
	Dictionary _get_data() const;
	void _set_data(const Dictionary &p_data);

protected:
	static void _bind_methods();

public:
	int get_point_count() const;
	void add_point(const Vector3 &p_pos, const Vector3 &p_in = Vector3(), const Vector3 &p_out = Vector3(), int p_atpos = -1);
	void set_point_position(int p_index, const Vector3 &p_pos);
	Vector3 get_point_position(int p_index) const;
	void set_point_tilt(int p_index, float p_tilt);
	float get_point_tilt(int p_index) const;
	void set_point_in(int p_index, const Vector3 &p_in);
	Vector3 get_point_in(int p_index) const;
	void set_point_out(int p_index, const Vector3 &p_out);
	Vector3 get_point_out(int p_index) const;
	void remove_point(int p_index);
	void clear_points();

	Vector3 interpolate(int p_index, float p_offset) const;
	Vector3 interpolatef(real_t p_findex) const;

	void set_bake_interval(float p_tolerance);
	float get_bake_interval() const;
	void set_up_vector_enabled(bool p_enable);
	bool is_up_vector_enabled() const;

	float get_baked_length() const;
	Vector3 interpolate_baked(float p_offset, bool p_cubic = false) const;
	float interpolate_baked_tilt(float p_offset) const;
	Vector3 interpolate_baked_up_vector(float p_offset, bool p_apply_tilt = false) const;
	PackedVector3Array get_baked_points() const; //useful for going through
	PackedFloat32Array get_baked_tilts() const; //useful for going through
	PackedVector3Array get_baked_up_vectors() const;
	Vector3 get_closest_point(const Vector3 &p_to_point) const;
	float get_closest_offset(const Vector3 &p_to_point) const;

	PackedVector3Array tessellate(int p_max_stages = 5, float p_tolerance = 4) const; //useful for display

	Curve3D();
};

#endif // CURVE_H
