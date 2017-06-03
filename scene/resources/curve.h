/*************************************************************************/
/*  curve.h                                                              */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                    http://www.godotengine.org                         */
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
#ifndef CURVE_H
#define CURVE_H

#include "resource.h"
#if 0
class Curve2D : public Resource {

	GDCLASS(Curve2D,Resource);

	struct Point {

		Vector2 in;
		Vector2 out;
		Vector2 pos;
	};


	Vector<Point> points;

protected:

	static void _bind_methods();

	void set_points_in(const Vector2Array& p_points_in);
	void set_points_out(const Vector2Array& p_points_out);
	void set_points_pos(const Vector2Array& p_points_pos);

	Vector2Array get_points_in() const;
	Vector2Array get_points_out() const;
	Vector2Array get_points_pos() const;

public:


	int get_point_count() const;
	void add_point(const Vector2& p_pos, const Vector2& p_in=Vector2(), const Vector2& p_out=Vector2());
	void set_point_pos(int p_index, const Vector2& p_pos);
	Vector2 get_point_pos(int p_index) const;
	void set_point_in(int p_index, const Vector2& p_in);
	Vector2 get_point_in(int p_index) const;
	void set_point_out(int p_index, const Vector2& p_out);
	Vector2 get_point_out(int p_index) const;
	void remove_point(int p_index);

	Vector2 interpolate(int p_index, float p_offset) const;
	Vector2 interpolatef(real_t p_findex) const;
	PoolVector<Point2> bake(int p_subdivs=10) const;
	void advance(real_t p_distance,int &r_index, real_t &r_pos) const;
	void get_approx_position_from_offset(real_t p_offset,int &r_index, real_t &r_pos,int p_subdivs=16) const;

	Curve2D();
};

#endif

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
	mutable PoolVector2Array baked_point_cache;
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
	void set_point_pos(int p_index, const Vector2 &p_pos);
	Vector2 get_point_pos(int p_index) const;
	void set_point_in(int p_index, const Vector2 &p_in);
	Vector2 get_point_in(int p_index) const;
	void set_point_out(int p_index, const Vector2 &p_out);
	Vector2 get_point_out(int p_index) const;
	void remove_point(int p_index);
	void clear_points();

	Vector2 interpolate(int p_index, float p_offset) const;
	Vector2 interpolatef(real_t p_findex) const;

	void set_bake_interval(float p_distance);
	float get_bake_interval() const;

	float get_baked_length() const;
	Vector2 interpolate_baked(float p_offset, bool p_cubic = false) const;
	PoolVector2Array get_baked_points() const; //useful for going through

	PoolVector2Array tesselate(int p_max_stages = 5, float p_tolerance = 4) const; //useful for display

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
	mutable PoolVector3Array baked_point_cache;
	mutable PoolRealArray baked_tilt_cache;
	mutable float baked_max_ofs;

	void _bake() const;

	float bake_interval;

	void _bake_segment3d(Map<float, Vector3> &r_bake, float p_begin, float p_end, const Vector3 &p_a, const Vector3 &p_out, const Vector3 &p_b, const Vector3 &p_in, int p_depth, int p_max_depth, float p_tol) const;
	Dictionary _get_data() const;
	void _set_data(const Dictionary &p_data);

protected:
	static void _bind_methods();

public:
	int get_point_count() const;
	void add_point(const Vector3 &p_pos, const Vector3 &p_in = Vector3(), const Vector3 &p_out = Vector3(), int p_atpos = -1);
	void set_point_pos(int p_index, const Vector3 &p_pos);
	Vector3 get_point_pos(int p_index) const;
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

	void set_bake_interval(float p_distance);
	float get_bake_interval() const;

	float get_baked_length() const;
	Vector3 interpolate_baked(float p_offset, bool p_cubic = false) const;
	float interpolate_baked_tilt(float p_offset) const;
	PoolVector3Array get_baked_points() const; //useful for going through
	PoolRealArray get_baked_tilts() const; //useful for going through

	PoolVector3Array tesselate(int p_max_stages = 5, float p_tolerance = 4) const; //useful for display

	Curve3D();
};

#endif // CURVE_H
