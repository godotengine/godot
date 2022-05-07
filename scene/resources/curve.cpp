/*************************************************************************/
/*  curve.cpp                                                            */
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

#include "curve.h"

#include "core/core_string_names.h"

template <class T>
static _FORCE_INLINE_ T _bezier_interp(real_t t, T start, T control_1, T control_2, T end) {
	/* Formula from Wikipedia article on Bezier curves. */
	real_t omt = (1.0 - t);
	real_t omt2 = omt * omt;
	real_t omt3 = omt2 * omt;
	real_t t2 = t * t;
	real_t t3 = t2 * t;

	return start * omt3 + control_1 * omt2 * t * 3.0 + control_2 * omt * t2 * 3.0 + end * t3;
}

const char *Curve::SIGNAL_RANGE_CHANGED = "range_changed";

Curve::Curve() {
	_bake_resolution = 100;
	_baked_cache_dirty = false;
	_min_value = 0;
	_max_value = 1;
	_minmax_set_once = 0b00;
}

int Curve::add_point(Vector2 p_pos, real_t left_tangent, real_t right_tangent, TangentMode left_mode, TangentMode right_mode) {
	// Add a point and preserve order

	// Curve bounds is in 0..1
	if (p_pos.x > MAX_X) {
		p_pos.x = MAX_X;
	} else if (p_pos.x < MIN_X) {
		p_pos.x = MIN_X;
	}

	int ret = -1;

	if (_points.size() == 0) {
		_points.push_back(Point(p_pos, left_tangent, right_tangent, left_mode, right_mode));
		ret = 0;

	} else if (_points.size() == 1) {
		// TODO Is the `else` able to handle this block already?

		real_t diff = p_pos.x - _points[0].pos.x;

		if (diff > 0) {
			_points.push_back(Point(p_pos, left_tangent, right_tangent, left_mode, right_mode));
			ret = 1;
		} else {
			_points.insert(0, Point(p_pos, left_tangent, right_tangent, left_mode, right_mode));
			ret = 0;
		}

	} else {
		int i = get_index(p_pos.x);

		if (i == 0 && p_pos.x < _points[0].pos.x) {
			// Insert before anything else
			_points.insert(0, Point(p_pos, left_tangent, right_tangent, left_mode, right_mode));
			ret = 0;
		} else {
			// Insert between i and i+1
			++i;
			_points.insert(i, Point(p_pos, left_tangent, right_tangent, left_mode, right_mode));
			ret = i;
		}
	}

	update_auto_tangents(ret);

	mark_dirty();

	return ret;
}

int Curve::get_index(real_t offset) const {
	// Lower-bound float binary search

	int imin = 0;
	int imax = _points.size() - 1;

	while (imax - imin > 1) {
		int m = (imin + imax) / 2;

		real_t a = _points[m].pos.x;
		real_t b = _points[m + 1].pos.x;

		if (a < offset && b < offset) {
			imin = m;

		} else if (a > offset) {
			imax = m;

		} else {
			return m;
		}
	}

	// Will happen if the offset is out of bounds
	if (offset > _points[imax].pos.x) {
		return imax;
	}
	return imin;
}

void Curve::clean_dupes() {
	bool dirty = false;

	for (int i = 1; i < _points.size(); ++i) {
		real_t diff = _points[i - 1].pos.x - _points[i].pos.x;
		if (diff <= CMP_EPSILON) {
			_points.remove(i);
			--i;
			dirty = true;
		}
	}

	if (dirty) {
		mark_dirty();
	}
}

void Curve::set_point_left_tangent(int i, real_t tangent) {
	ERR_FAIL_INDEX(i, _points.size());
	_points.write[i].left_tangent = tangent;
	_points.write[i].left_mode = TANGENT_FREE;
	mark_dirty();
}

void Curve::set_point_right_tangent(int i, real_t tangent) {
	ERR_FAIL_INDEX(i, _points.size());
	_points.write[i].right_tangent = tangent;
	_points.write[i].right_mode = TANGENT_FREE;
	mark_dirty();
}

void Curve::set_point_left_mode(int i, TangentMode p_mode) {
	ERR_FAIL_INDEX(i, _points.size());
	_points.write[i].left_mode = p_mode;
	if (i > 0) {
		if (p_mode == TANGENT_LINEAR) {
			Vector2 v = (_points[i - 1].pos - _points[i].pos).normalized();
			_points.write[i].left_tangent = v.y / v.x;
		}
	}
	mark_dirty();
}

void Curve::set_point_right_mode(int i, TangentMode p_mode) {
	ERR_FAIL_INDEX(i, _points.size());
	_points.write[i].right_mode = p_mode;
	if (i + 1 < _points.size()) {
		if (p_mode == TANGENT_LINEAR) {
			Vector2 v = (_points[i + 1].pos - _points[i].pos).normalized();
			_points.write[i].right_tangent = v.y / v.x;
		}
	}
	mark_dirty();
}

real_t Curve::get_point_left_tangent(int i) const {
	ERR_FAIL_INDEX_V(i, _points.size(), 0);
	return _points[i].left_tangent;
}

real_t Curve::get_point_right_tangent(int i) const {
	ERR_FAIL_INDEX_V(i, _points.size(), 0);
	return _points[i].right_tangent;
}

Curve::TangentMode Curve::get_point_left_mode(int i) const {
	ERR_FAIL_INDEX_V(i, _points.size(), TANGENT_FREE);
	return _points[i].left_mode;
}

Curve::TangentMode Curve::get_point_right_mode(int i) const {
	ERR_FAIL_INDEX_V(i, _points.size(), TANGENT_FREE);
	return _points[i].right_mode;
}

void Curve::remove_point(int p_index) {
	ERR_FAIL_INDEX(p_index, _points.size());
	_points.remove(p_index);
	mark_dirty();
}

void Curve::clear_points() {
	_points.clear();
	mark_dirty();
}

void Curve::set_point_value(int p_index, real_t pos) {
	ERR_FAIL_INDEX(p_index, _points.size());
	_points.write[p_index].pos.y = pos;
	update_auto_tangents(p_index);
	mark_dirty();
}

int Curve::set_point_offset(int p_index, float offset) {
	ERR_FAIL_INDEX_V(p_index, _points.size(), -1);
	Point p = _points[p_index];
	remove_point(p_index);
	int i = add_point(Vector2(offset, p.pos.y));
	_points.write[i].left_tangent = p.left_tangent;
	_points.write[i].right_tangent = p.right_tangent;
	_points.write[i].left_mode = p.left_mode;
	_points.write[i].right_mode = p.right_mode;
	if (p_index != i) {
		update_auto_tangents(p_index);
	}
	update_auto_tangents(i);
	return i;
}

Vector2 Curve::get_point_position(int p_index) const {
	ERR_FAIL_INDEX_V(p_index, _points.size(), Vector2(0, 0));
	return _points[p_index].pos;
}

Curve::Point Curve::get_point(int p_index) const {
	ERR_FAIL_INDEX_V(p_index, _points.size(), Point());
	return _points[p_index];
}

void Curve::update_auto_tangents(int i) {
	Point &p = _points.write[i];

	if (i > 0) {
		if (p.left_mode == TANGENT_LINEAR) {
			Vector2 v = (_points[i - 1].pos - p.pos).normalized();
			p.left_tangent = v.y / v.x;
		}
		if (_points[i - 1].right_mode == TANGENT_LINEAR) {
			Vector2 v = (_points[i - 1].pos - p.pos).normalized();
			_points.write[i - 1].right_tangent = v.y / v.x;
		}
	}

	if (i + 1 < _points.size()) {
		if (p.right_mode == TANGENT_LINEAR) {
			Vector2 v = (_points[i + 1].pos - p.pos).normalized();
			p.right_tangent = v.y / v.x;
		}
		if (_points[i + 1].left_mode == TANGENT_LINEAR) {
			Vector2 v = (_points[i + 1].pos - p.pos).normalized();
			_points.write[i + 1].left_tangent = v.y / v.x;
		}
	}
}

#define MIN_Y_RANGE 0.01

void Curve::set_min_value(float p_min) {
	if (_minmax_set_once & 0b11 && p_min > _max_value - MIN_Y_RANGE) {
		_min_value = _max_value - MIN_Y_RANGE;
	} else {
		_minmax_set_once |= 0b10; // first bit is "min set"
		_min_value = p_min;
	}
	// Note: min and max are indicative values,
	// it's still possible that existing points are out of range at this point.
	emit_signal(SIGNAL_RANGE_CHANGED);
}

void Curve::set_max_value(float p_max) {
	if (_minmax_set_once & 0b11 && p_max < _min_value + MIN_Y_RANGE) {
		_max_value = _min_value + MIN_Y_RANGE;
	} else {
		_minmax_set_once |= 0b01; // second bit is "max set"
		_max_value = p_max;
	}
	emit_signal(SIGNAL_RANGE_CHANGED);
}

real_t Curve::interpolate(real_t offset) const {
	if (_points.size() == 0) {
		return 0;
	}
	if (_points.size() == 1) {
		return _points[0].pos.y;
	}

	int i = get_index(offset);

	if (i == _points.size() - 1) {
		return _points[i].pos.y;
	}

	real_t local = offset - _points[i].pos.x;

	if (i == 0 && local <= 0) {
		return _points[0].pos.y;
	}

	return interpolate_local_nocheck(i, local);
}

real_t Curve::interpolate_local_nocheck(int index, real_t local_offset) const {
	const Point a = _points[index];
	const Point b = _points[index + 1];

	/* Cubic bezier
	 *
	 *       ac-----bc
	 *      /         \
	 *     /           \     Here with a.right_tangent > 0
	 *    /             \    and b.left_tangent < 0
	 *   /               \
	 *  a                 b
	 *
	 *  |-d1--|-d2--|-d3--|
	 *
	 * d1 == d2 == d3 == d / 3
	 */

	// Control points are chosen at equal distances
	real_t d = b.pos.x - a.pos.x;
	if (Math::abs(d) <= CMP_EPSILON) {
		return b.pos.y;
	}
	local_offset /= d;
	d /= 3.0;
	real_t yac = a.pos.y + d * a.right_tangent;
	real_t ybc = b.pos.y - d * b.left_tangent;

	real_t y = _bezier_interp(local_offset, a.pos.y, yac, ybc, b.pos.y);

	return y;
}

void Curve::mark_dirty() {
	_baked_cache_dirty = true;
	emit_signal(CoreStringNames::get_singleton()->changed);
}

Array Curve::get_data() const {
	Array output;
	const unsigned int ELEMS = 5;
	output.resize(_points.size() * ELEMS);

	for (int j = 0; j < _points.size(); ++j) {
		const Point p = _points[j];
		int i = j * ELEMS;

		output[i] = p.pos;
		output[i + 1] = p.left_tangent;
		output[i + 2] = p.right_tangent;
		output[i + 3] = p.left_mode;
		output[i + 4] = p.right_mode;
	}

	return output;
}

void Curve::set_data(Array input) {
	const unsigned int ELEMS = 5;
	ERR_FAIL_COND(input.size() % ELEMS != 0);

	_points.clear();

	// Validate input
	for (int i = 0; i < input.size(); i += ELEMS) {
		ERR_FAIL_COND(input[i].get_type() != Variant::VECTOR2);
		ERR_FAIL_COND(!input[i + 1].is_num());
		ERR_FAIL_COND(input[i + 2].get_type() != Variant::REAL);

		ERR_FAIL_COND(input[i + 3].get_type() != Variant::INT);
		int left_mode = input[i + 3];
		ERR_FAIL_COND(left_mode < 0 || left_mode >= TANGENT_MODE_COUNT);

		ERR_FAIL_COND(input[i + 4].get_type() != Variant::INT);
		int right_mode = input[i + 4];
		ERR_FAIL_COND(right_mode < 0 || right_mode >= TANGENT_MODE_COUNT);
	}

	_points.resize(input.size() / ELEMS);

	for (int j = 0; j < _points.size(); ++j) {
		Point &p = _points.write[j];
		int i = j * ELEMS;

		p.pos = input[i];
		p.left_tangent = input[i + 1];
		p.right_tangent = input[i + 2];
		// TODO For some reason the compiler won't convert from Variant to enum
		int left_mode = input[i + 3];
		int right_mode = input[i + 4];
		p.left_mode = (TangentMode)left_mode;
		p.right_mode = (TangentMode)right_mode;
	}

	mark_dirty();
}

void Curve::bake() {
	_baked_cache.clear();

	_baked_cache.resize(_bake_resolution);

	for (int i = 1; i < _bake_resolution - 1; ++i) {
		real_t x = i / static_cast<real_t>(_bake_resolution);
		real_t y = interpolate(x);
		_baked_cache.write[i] = y;
	}

	if (_points.size() != 0) {
		_baked_cache.write[0] = _points[0].pos.y;
		_baked_cache.write[_baked_cache.size() - 1] = _points[_points.size() - 1].pos.y;
	}

	_baked_cache_dirty = false;
}

void Curve::set_bake_resolution(int p_resolution) {
	ERR_FAIL_COND(p_resolution < 1);
	ERR_FAIL_COND(p_resolution > 1000);
	_bake_resolution = p_resolution;
	_baked_cache_dirty = true;
}

real_t Curve::interpolate_baked(real_t offset) {
	if (_baked_cache_dirty) {
		// Last-second bake if not done already
		bake();
	}

	// Special cases if the cache is too small
	if (_baked_cache.size() == 0) {
		if (_points.size() == 0) {
			return 0;
		}
		return _points[0].pos.y;
	} else if (_baked_cache.size() == 1) {
		return _baked_cache[0];
	}

	// Get interpolation index
	real_t fi = offset * _baked_cache.size();
	int i = Math::floor(fi);
	if (i < 0) {
		i = 0;
		fi = 0;
	} else if (i >= _baked_cache.size()) {
		i = _baked_cache.size() - 1;
		fi = 0;
	}

	// Interpolate
	if (i + 1 < _baked_cache.size()) {
		real_t t = fi - i;
		return Math::lerp(_baked_cache[i], _baked_cache[i + 1], t);
	} else {
		return _baked_cache[_baked_cache.size() - 1];
	}
}

void Curve::ensure_default_setup(float p_min, float p_max) {
	if (_points.size() == 0 && _min_value == 0 && _max_value == 1) {
		add_point(Vector2(0, 1));
		add_point(Vector2(1, 1));
		set_min_value(p_min);
		set_max_value(p_max);
	}
}

void Curve::_bind_methods() {
	ClassDB::bind_method(D_METHOD("get_point_count"), &Curve::get_point_count);
	ClassDB::bind_method(D_METHOD("add_point", "position", "left_tangent", "right_tangent", "left_mode", "right_mode"), &Curve::add_point, DEFVAL(0), DEFVAL(0), DEFVAL(TANGENT_FREE), DEFVAL(TANGENT_FREE));
	ClassDB::bind_method(D_METHOD("remove_point", "index"), &Curve::remove_point);
	ClassDB::bind_method(D_METHOD("clear_points"), &Curve::clear_points);
	ClassDB::bind_method(D_METHOD("get_point_position", "index"), &Curve::get_point_position);
	ClassDB::bind_method(D_METHOD("set_point_value", "index", "y"), &Curve::set_point_value);
	ClassDB::bind_method(D_METHOD("set_point_offset", "index", "offset"), &Curve::set_point_offset);
	ClassDB::bind_method(D_METHOD("interpolate", "offset"), &Curve::interpolate);
	ClassDB::bind_method(D_METHOD("interpolate_baked", "offset"), &Curve::interpolate_baked);
	ClassDB::bind_method(D_METHOD("get_point_left_tangent", "index"), &Curve::get_point_left_tangent);
	ClassDB::bind_method(D_METHOD("get_point_right_tangent", "index"), &Curve::get_point_right_tangent);
	ClassDB::bind_method(D_METHOD("get_point_left_mode", "index"), &Curve::get_point_left_mode);
	ClassDB::bind_method(D_METHOD("get_point_right_mode", "index"), &Curve::get_point_right_mode);
	ClassDB::bind_method(D_METHOD("set_point_left_tangent", "index", "tangent"), &Curve::set_point_left_tangent);
	ClassDB::bind_method(D_METHOD("set_point_right_tangent", "index", "tangent"), &Curve::set_point_right_tangent);
	ClassDB::bind_method(D_METHOD("set_point_left_mode", "index", "mode"), &Curve::set_point_left_mode);
	ClassDB::bind_method(D_METHOD("set_point_right_mode", "index", "mode"), &Curve::set_point_right_mode);
	ClassDB::bind_method(D_METHOD("get_min_value"), &Curve::get_min_value);
	ClassDB::bind_method(D_METHOD("set_min_value", "min"), &Curve::set_min_value);
	ClassDB::bind_method(D_METHOD("get_max_value"), &Curve::get_max_value);
	ClassDB::bind_method(D_METHOD("set_max_value", "max"), &Curve::set_max_value);
	ClassDB::bind_method(D_METHOD("clean_dupes"), &Curve::clean_dupes);
	ClassDB::bind_method(D_METHOD("bake"), &Curve::bake);
	ClassDB::bind_method(D_METHOD("get_bake_resolution"), &Curve::get_bake_resolution);
	ClassDB::bind_method(D_METHOD("set_bake_resolution", "resolution"), &Curve::set_bake_resolution);
	ClassDB::bind_method(D_METHOD("_get_data"), &Curve::get_data);
	ClassDB::bind_method(D_METHOD("_set_data", "data"), &Curve::set_data);

	ADD_PROPERTY(PropertyInfo(Variant::REAL, "min_value", PROPERTY_HINT_RANGE, "-1024,1024,0.01"), "set_min_value", "get_min_value");
	ADD_PROPERTY(PropertyInfo(Variant::REAL, "max_value", PROPERTY_HINT_RANGE, "-1024,1024,0.01"), "set_max_value", "get_max_value");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "bake_resolution", PROPERTY_HINT_RANGE, "1,1000,1"), "set_bake_resolution", "get_bake_resolution");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "_data", PROPERTY_HINT_NONE, "", PROPERTY_USAGE_NOEDITOR | PROPERTY_USAGE_INTERNAL), "_set_data", "_get_data");

	ADD_SIGNAL(MethodInfo(SIGNAL_RANGE_CHANGED));

	BIND_ENUM_CONSTANT(TANGENT_FREE);
	BIND_ENUM_CONSTANT(TANGENT_LINEAR);
	BIND_ENUM_CONSTANT(TANGENT_MODE_COUNT);
}

int Curve2D::get_point_count() const {
	return points.size();
}
void Curve2D::add_point(const Vector2 &p_pos, const Vector2 &p_in, const Vector2 &p_out, int p_atpos) {
	Point n;
	n.pos = p_pos;
	n.in = p_in;
	n.out = p_out;
	if (p_atpos >= 0 && p_atpos < points.size()) {
		points.insert(p_atpos, n);
	} else {
		points.push_back(n);
	}

	baked_cache_dirty = true;
	emit_signal(CoreStringNames::get_singleton()->changed);
}

void Curve2D::set_point_position(int p_index, const Vector2 &p_pos) {
	ERR_FAIL_INDEX(p_index, points.size());

	points.write[p_index].pos = p_pos;
	baked_cache_dirty = true;
	emit_signal(CoreStringNames::get_singleton()->changed);
}
Vector2 Curve2D::get_point_position(int p_index) const {
	ERR_FAIL_INDEX_V(p_index, points.size(), Vector2());
	return points[p_index].pos;
}

void Curve2D::set_point_in(int p_index, const Vector2 &p_in) {
	ERR_FAIL_INDEX(p_index, points.size());

	points.write[p_index].in = p_in;
	baked_cache_dirty = true;
	emit_signal(CoreStringNames::get_singleton()->changed);
}
Vector2 Curve2D::get_point_in(int p_index) const {
	ERR_FAIL_INDEX_V(p_index, points.size(), Vector2());
	return points[p_index].in;
}

void Curve2D::set_point_out(int p_index, const Vector2 &p_out) {
	ERR_FAIL_INDEX(p_index, points.size());

	points.write[p_index].out = p_out;
	baked_cache_dirty = true;
	emit_signal(CoreStringNames::get_singleton()->changed);
}

Vector2 Curve2D::get_point_out(int p_index) const {
	ERR_FAIL_INDEX_V(p_index, points.size(), Vector2());
	return points[p_index].out;
}

void Curve2D::remove_point(int p_index) {
	ERR_FAIL_INDEX(p_index, points.size());
	points.remove(p_index);
	baked_cache_dirty = true;
	emit_signal(CoreStringNames::get_singleton()->changed);
}

void Curve2D::clear_points() {
	if (!points.empty()) {
		points.clear();
		baked_cache_dirty = true;
		emit_signal(CoreStringNames::get_singleton()->changed);
	}
}

Vector2 Curve2D::interpolate(int p_index, float p_offset) const {
	int pc = points.size();
	ERR_FAIL_COND_V(pc == 0, Vector2());

	if (p_index >= pc - 1) {
		return points[pc - 1].pos;
	} else if (p_index < 0) {
		return points[0].pos;
	}

	Vector2 p0 = points[p_index].pos;
	Vector2 p1 = p0 + points[p_index].out;
	Vector2 p3 = points[p_index + 1].pos;
	Vector2 p2 = p3 + points[p_index + 1].in;

	return _bezier_interp(p_offset, p0, p1, p2, p3);
}

Vector2 Curve2D::interpolatef(real_t p_findex) const {
	if (p_findex < 0) {
		p_findex = 0;
	} else if (p_findex >= points.size()) {
		p_findex = points.size();
	}

	return interpolate((int)p_findex, Math::fmod(p_findex, (real_t)1.0));
}

void Curve2D::_bake_segment2d(Map<float, Vector2> &r_bake, float p_begin, float p_end, const Vector2 &p_a, const Vector2 &p_out, const Vector2 &p_b, const Vector2 &p_in, int p_depth, int p_max_depth, float p_tol) const {
	float mp = p_begin + (p_end - p_begin) * 0.5;
	Vector2 beg = _bezier_interp(p_begin, p_a, p_a + p_out, p_b + p_in, p_b);
	Vector2 mid = _bezier_interp(mp, p_a, p_a + p_out, p_b + p_in, p_b);
	Vector2 end = _bezier_interp(p_end, p_a, p_a + p_out, p_b + p_in, p_b);

	Vector2 na = (mid - beg).normalized();
	Vector2 nb = (end - mid).normalized();
	float dp = na.dot(nb);

	if (dp < Math::cos(Math::deg2rad(p_tol))) {
		r_bake[mp] = mid;
	}

	if (p_depth < p_max_depth) {
		_bake_segment2d(r_bake, p_begin, mp, p_a, p_out, p_b, p_in, p_depth + 1, p_max_depth, p_tol);
		_bake_segment2d(r_bake, mp, p_end, p_a, p_out, p_b, p_in, p_depth + 1, p_max_depth, p_tol);
	}
}

void Curve2D::_bake() const {
	if (!baked_cache_dirty) {
		return;
	}

	baked_max_ofs = 0;
	baked_cache_dirty = false;

	if (points.size() == 0) {
		baked_point_cache.resize(0);
		return;
	}

	if (points.size() == 1) {
		baked_point_cache.resize(1);
		baked_point_cache.set(0, points[0].pos);
		return;
	}

	Vector2 pos = points[0].pos;
	List<Vector2> pointlist;

	pointlist.push_back(pos); //start always from origin

	for (int i = 0; i < points.size() - 1; i++) {
		float step = 0.1; // at least 10 substeps ought to be enough?
		float p = 0;

		while (p < 1.0) {
			float np = p + step;
			if (np > 1.0) {
				np = 1.0;
			}

			Vector2 npp = _bezier_interp(np, points[i].pos, points[i].pos + points[i].out, points[i + 1].pos + points[i + 1].in, points[i + 1].pos);
			float d = pos.distance_to(npp);

			if (d > bake_interval) {
				// OK! between P and NP there _has_ to be Something, let's go searching!

				int iterations = 10; //lots of detail!

				float low = p;
				float hi = np;
				float mid = low + (hi - low) * 0.5;

				for (int j = 0; j < iterations; j++) {
					npp = _bezier_interp(mid, points[i].pos, points[i].pos + points[i].out, points[i + 1].pos + points[i + 1].in, points[i + 1].pos);
					d = pos.distance_to(npp);

					if (bake_interval < d) {
						hi = mid;
					} else {
						low = mid;
					}
					mid = low + (hi - low) * 0.5;
				}

				pos = npp;
				p = mid;
				pointlist.push_back(pos);
			} else {
				p = np;
			}
		}
	}

	Vector2 lastpos = points[points.size() - 1].pos;

	float rem = pos.distance_to(lastpos);
	baked_max_ofs = (pointlist.size() - 1) * bake_interval + rem;
	pointlist.push_back(lastpos);

	baked_point_cache.resize(pointlist.size());
	PoolVector2Array::Write w = baked_point_cache.write();
	int idx = 0;

	for (List<Vector2>::Element *E = pointlist.front(); E; E = E->next()) {
		w[idx] = E->get();
		idx++;
	}
}

float Curve2D::get_baked_length() const {
	if (baked_cache_dirty) {
		_bake();
	}

	return baked_max_ofs;
}
Vector2 Curve2D::interpolate_baked(float p_offset, bool p_cubic) const {
	if (baked_cache_dirty) {
		_bake();
	}

	//validate//
	int pc = baked_point_cache.size();
	ERR_FAIL_COND_V_MSG(pc == 0, Vector2(), "No points in Curve2D.");

	if (pc == 1) {
		return baked_point_cache.get(0);
	}

	int bpc = baked_point_cache.size();
	PoolVector2Array::Read r = baked_point_cache.read();

	if (p_offset < 0) {
		return r[0];
	}
	if (p_offset >= baked_max_ofs) {
		return r[bpc - 1];
	}

	int idx = Math::floor((double)p_offset / (double)bake_interval);
	float frac = Math::fmod(p_offset, (float)bake_interval);

	if (idx >= bpc - 1) {
		return r[bpc - 1];
	} else if (idx == bpc - 2) {
		if (frac > 0) {
			frac /= Math::fmod(baked_max_ofs, bake_interval);
		}
	} else {
		frac /= bake_interval;
	}

	if (p_cubic) {
		Vector2 pre = idx > 0 ? r[idx - 1] : r[idx];
		Vector2 post = (idx < (bpc - 2)) ? r[idx + 2] : r[idx + 1];
		return r[idx].cubic_interpolate(r[idx + 1], pre, post, frac);
	} else {
		return r[idx].linear_interpolate(r[idx + 1], frac);
	}
}

PoolVector2Array Curve2D::get_baked_points() const {
	if (baked_cache_dirty) {
		_bake();
	}

	return baked_point_cache;
}

void Curve2D::set_bake_interval(float p_tolerance) {
	bake_interval = p_tolerance;
	baked_cache_dirty = true;
	emit_signal(CoreStringNames::get_singleton()->changed);
}

float Curve2D::get_bake_interval() const {
	return bake_interval;
}

Vector2 Curve2D::get_closest_point(const Vector2 &p_to_point) const {
	// Brute force method

	if (baked_cache_dirty) {
		_bake();
	}

	//validate//
	int pc = baked_point_cache.size();
	ERR_FAIL_COND_V_MSG(pc == 0, Vector2(), "No points in Curve2D.");

	if (pc == 1) {
		return baked_point_cache.get(0);
	}

	PoolVector2Array::Read r = baked_point_cache.read();

	Vector2 nearest;
	float nearest_dist = -1.0f;

	for (int i = 0; i < pc - 1; i++) {
		Vector2 origin = r[i];
		Vector2 direction = (r[i + 1] - origin) / bake_interval;

		float d = CLAMP((p_to_point - origin).dot(direction), 0.0f, bake_interval);
		Vector2 proj = origin + direction * d;

		float dist = proj.distance_squared_to(p_to_point);

		if (nearest_dist < 0.0f || dist < nearest_dist) {
			nearest = proj;
			nearest_dist = dist;
		}
	}

	return nearest;
}

float Curve2D::get_closest_offset(const Vector2 &p_to_point) const {
	// Brute force method

	if (baked_cache_dirty) {
		_bake();
	}

	//validate//
	int pc = baked_point_cache.size();
	ERR_FAIL_COND_V_MSG(pc == 0, 0.0f, "No points in Curve2D.");

	if (pc == 1) {
		return 0.0f;
	}

	PoolVector2Array::Read r = baked_point_cache.read();

	float nearest = 0.0f;
	float nearest_dist = -1.0f;
	float offset = 0.0f;

	for (int i = 0; i < pc - 1; i++) {
		Vector2 origin = r[i];
		Vector2 direction = (r[i + 1] - origin) / bake_interval;

		float d = CLAMP((p_to_point - origin).dot(direction), 0.0f, bake_interval);
		Vector2 proj = origin + direction * d;

		float dist = proj.distance_squared_to(p_to_point);

		if (nearest_dist < 0.0f || dist < nearest_dist) {
			nearest = offset + d;
			nearest_dist = dist;
		}

		offset += bake_interval;
	}

	return nearest;
}

Dictionary Curve2D::_get_data() const {
	Dictionary dc;

	PoolVector2Array d;
	d.resize(points.size() * 3);
	PoolVector2Array::Write w = d.write();

	for (int i = 0; i < points.size(); i++) {
		w[i * 3 + 0] = points[i].in;
		w[i * 3 + 1] = points[i].out;
		w[i * 3 + 2] = points[i].pos;
	}

	w = PoolVector2Array::Write();

	dc["points"] = d;

	return dc;
}
void Curve2D::_set_data(const Dictionary &p_data) {
	ERR_FAIL_COND(!p_data.has("points"));

	PoolVector2Array rp = p_data["points"];
	int pc = rp.size();
	ERR_FAIL_COND(pc % 3 != 0);
	points.resize(pc / 3);
	PoolVector2Array::Read r = rp.read();

	for (int i = 0; i < points.size(); i++) {
		points.write[i].in = r[i * 3 + 0];
		points.write[i].out = r[i * 3 + 1];
		points.write[i].pos = r[i * 3 + 2];
	}

	baked_cache_dirty = true;
}

PoolVector2Array Curve2D::tessellate(int p_max_stages, float p_tolerance) const {
	PoolVector2Array tess;

	if (points.size() == 0) {
		return tess;
	}
	Vector<Map<float, Vector2>> midpoints;

	midpoints.resize(points.size() - 1);

	int pc = 1;
	for (int i = 0; i < points.size() - 1; i++) {
		_bake_segment2d(midpoints.write[i], 0, 1, points[i].pos, points[i].out, points[i + 1].pos, points[i + 1].in, 0, p_max_stages, p_tolerance);
		pc++;
		pc += midpoints[i].size();
	}

	tess.resize(pc);
	PoolVector2Array::Write bpw = tess.write();
	bpw[0] = points[0].pos;
	int pidx = 0;

	for (int i = 0; i < points.size() - 1; i++) {
		for (Map<float, Vector2>::Element *E = midpoints[i].front(); E; E = E->next()) {
			pidx++;
			bpw[pidx] = E->get();
		}

		pidx++;
		bpw[pidx] = points[i + 1].pos;
	}

	bpw = PoolVector2Array::Write();

	return tess;
}

void Curve2D::_bind_methods() {
	ClassDB::bind_method(D_METHOD("get_point_count"), &Curve2D::get_point_count);
	ClassDB::bind_method(D_METHOD("add_point", "position", "in", "out", "at_position"), &Curve2D::add_point, DEFVAL(Vector2()), DEFVAL(Vector2()), DEFVAL(-1));
	ClassDB::bind_method(D_METHOD("set_point_position", "idx", "position"), &Curve2D::set_point_position);
	ClassDB::bind_method(D_METHOD("get_point_position", "idx"), &Curve2D::get_point_position);
	ClassDB::bind_method(D_METHOD("set_point_in", "idx", "position"), &Curve2D::set_point_in);
	ClassDB::bind_method(D_METHOD("get_point_in", "idx"), &Curve2D::get_point_in);
	ClassDB::bind_method(D_METHOD("set_point_out", "idx", "position"), &Curve2D::set_point_out);
	ClassDB::bind_method(D_METHOD("get_point_out", "idx"), &Curve2D::get_point_out);
	ClassDB::bind_method(D_METHOD("remove_point", "idx"), &Curve2D::remove_point);
	ClassDB::bind_method(D_METHOD("clear_points"), &Curve2D::clear_points);
	ClassDB::bind_method(D_METHOD("interpolate", "idx", "t"), &Curve2D::interpolate);
	ClassDB::bind_method(D_METHOD("interpolatef", "fofs"), &Curve2D::interpolatef);
	//ClassDB::bind_method(D_METHOD("bake","subdivs"),&Curve2D::bake,DEFVAL(10));
	ClassDB::bind_method(D_METHOD("set_bake_interval", "distance"), &Curve2D::set_bake_interval);
	ClassDB::bind_method(D_METHOD("get_bake_interval"), &Curve2D::get_bake_interval);

	ClassDB::bind_method(D_METHOD("get_baked_length"), &Curve2D::get_baked_length);
	ClassDB::bind_method(D_METHOD("interpolate_baked", "offset", "cubic"), &Curve2D::interpolate_baked, DEFVAL(false));
	ClassDB::bind_method(D_METHOD("get_baked_points"), &Curve2D::get_baked_points);
	ClassDB::bind_method(D_METHOD("get_closest_point", "to_point"), &Curve2D::get_closest_point);
	ClassDB::bind_method(D_METHOD("get_closest_offset", "to_point"), &Curve2D::get_closest_offset);
	ClassDB::bind_method(D_METHOD("tessellate", "max_stages", "tolerance_degrees"), &Curve2D::tessellate, DEFVAL(5), DEFVAL(4));

	ClassDB::bind_method(D_METHOD("_get_data"), &Curve2D::_get_data);
	ClassDB::bind_method(D_METHOD("_set_data"), &Curve2D::_set_data);

	ADD_PROPERTY(PropertyInfo(Variant::REAL, "bake_interval", PROPERTY_HINT_RANGE, "0.01,512,0.01"), "set_bake_interval", "get_bake_interval");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "_data", PROPERTY_HINT_NONE, "", PROPERTY_USAGE_NOEDITOR | PROPERTY_USAGE_INTERNAL), "_set_data", "_get_data");
}

Curve2D::Curve2D() {
	baked_cache_dirty = false;
	baked_max_ofs = 0;
	/*	add_point(Vector2(-1,0,0));
	add_point(Vector2(0,2,0));
	add_point(Vector2(0,3,5));*/
	bake_interval = 5;
}

/***********************************************************************************/
/***********************************************************************************/
/***********************************************************************************/
/***********************************************************************************/
/***********************************************************************************/
/***********************************************************************************/

int Curve3D::get_point_count() const {
	return points.size();
}
void Curve3D::add_point(const Vector3 &p_pos, const Vector3 &p_in, const Vector3 &p_out, int p_atpos) {
	Point n;
	n.pos = p_pos;
	n.in = p_in;
	n.out = p_out;
	if (p_atpos >= 0 && p_atpos < points.size()) {
		points.insert(p_atpos, n);
	} else {
		points.push_back(n);
	}

	baked_cache_dirty = true;
	emit_signal(CoreStringNames::get_singleton()->changed);
}
void Curve3D::set_point_position(int p_index, const Vector3 &p_pos) {
	ERR_FAIL_INDEX(p_index, points.size());

	points.write[p_index].pos = p_pos;
	baked_cache_dirty = true;
	emit_signal(CoreStringNames::get_singleton()->changed);
}
Vector3 Curve3D::get_point_position(int p_index) const {
	ERR_FAIL_INDEX_V(p_index, points.size(), Vector3());
	return points[p_index].pos;
}

void Curve3D::set_point_tilt(int p_index, float p_tilt) {
	ERR_FAIL_INDEX(p_index, points.size());

	points.write[p_index].tilt = p_tilt;
	baked_cache_dirty = true;
	emit_signal(CoreStringNames::get_singleton()->changed);
}
float Curve3D::get_point_tilt(int p_index) const {
	ERR_FAIL_INDEX_V(p_index, points.size(), 0);
	return points[p_index].tilt;
}

void Curve3D::set_point_in(int p_index, const Vector3 &p_in) {
	ERR_FAIL_INDEX(p_index, points.size());

	points.write[p_index].in = p_in;
	baked_cache_dirty = true;
	emit_signal(CoreStringNames::get_singleton()->changed);
}
Vector3 Curve3D::get_point_in(int p_index) const {
	ERR_FAIL_INDEX_V(p_index, points.size(), Vector3());
	return points[p_index].in;
}

void Curve3D::set_point_out(int p_index, const Vector3 &p_out) {
	ERR_FAIL_INDEX(p_index, points.size());

	points.write[p_index].out = p_out;
	baked_cache_dirty = true;
	emit_signal(CoreStringNames::get_singleton()->changed);
}

Vector3 Curve3D::get_point_out(int p_index) const {
	ERR_FAIL_INDEX_V(p_index, points.size(), Vector3());
	return points[p_index].out;
}

void Curve3D::remove_point(int p_index) {
	ERR_FAIL_INDEX(p_index, points.size());
	points.remove(p_index);
	baked_cache_dirty = true;
	emit_signal(CoreStringNames::get_singleton()->changed);
}

void Curve3D::clear_points() {
	if (!points.empty()) {
		points.clear();
		baked_cache_dirty = true;
		emit_signal(CoreStringNames::get_singleton()->changed);
	}
}

Vector3 Curve3D::interpolate(int p_index, float p_offset) const {
	int pc = points.size();
	ERR_FAIL_COND_V(pc == 0, Vector3());

	if (p_index >= pc - 1) {
		return points[pc - 1].pos;
	} else if (p_index < 0) {
		return points[0].pos;
	}

	Vector3 p0 = points[p_index].pos;
	Vector3 p1 = p0 + points[p_index].out;
	Vector3 p3 = points[p_index + 1].pos;
	Vector3 p2 = p3 + points[p_index + 1].in;

	return _bezier_interp(p_offset, p0, p1, p2, p3);
}

Vector3 Curve3D::interpolatef(real_t p_findex) const {
	if (p_findex < 0) {
		p_findex = 0;
	} else if (p_findex >= points.size()) {
		p_findex = points.size();
	}

	return interpolate((int)p_findex, Math::fmod(p_findex, (real_t)1.0));
}

void Curve3D::_bake_segment3d(Map<float, Vector3> &r_bake, float p_begin, float p_end, const Vector3 &p_a, const Vector3 &p_out, const Vector3 &p_b, const Vector3 &p_in, int p_depth, int p_max_depth, float p_tol) const {
	float mp = p_begin + (p_end - p_begin) * 0.5;
	Vector3 beg = _bezier_interp(p_begin, p_a, p_a + p_out, p_b + p_in, p_b);
	Vector3 mid = _bezier_interp(mp, p_a, p_a + p_out, p_b + p_in, p_b);
	Vector3 end = _bezier_interp(p_end, p_a, p_a + p_out, p_b + p_in, p_b);

	Vector3 na = (mid - beg).normalized();
	Vector3 nb = (end - mid).normalized();
	float dp = na.dot(nb);

	if (dp < Math::cos(Math::deg2rad(p_tol))) {
		r_bake[mp] = mid;
	}
	if (p_depth < p_max_depth) {
		_bake_segment3d(r_bake, p_begin, mp, p_a, p_out, p_b, p_in, p_depth + 1, p_max_depth, p_tol);
		_bake_segment3d(r_bake, mp, p_end, p_a, p_out, p_b, p_in, p_depth + 1, p_max_depth, p_tol);
	}
}

void Curve3D::_bake() const {
	if (!baked_cache_dirty) {
		return;
	}

	baked_max_ofs = 0;
	baked_cache_dirty = false;

	if (points.size() == 0) {
		baked_point_cache.resize(0);
		baked_tilt_cache.resize(0);
		baked_up_vector_cache.resize(0);
		return;
	}

	if (points.size() == 1) {
		baked_point_cache.resize(1);
		baked_point_cache.set(0, points[0].pos);
		baked_tilt_cache.resize(1);
		baked_tilt_cache.set(0, points[0].tilt);

		if (up_vector_enabled) {
			baked_up_vector_cache.resize(1);
			baked_up_vector_cache.set(0, Vector3(0, 1, 0));
		} else {
			baked_up_vector_cache.resize(0);
		}

		return;
	}

	Vector3 pos = points[0].pos;
	List<Plane> pointlist;
	pointlist.push_back(Plane(pos, points[0].tilt));

	for (int i = 0; i < points.size() - 1; i++) {
		float step = 0.1; // at least 10 substeps ought to be enough?
		float p = 0;

		while (p < 1.0) {
			float np = p + step;
			if (np > 1.0) {
				np = 1.0;
			}

			Vector3 npp = _bezier_interp(np, points[i].pos, points[i].pos + points[i].out, points[i + 1].pos + points[i + 1].in, points[i + 1].pos);
			float d = pos.distance_to(npp);

			if (d > bake_interval) {
				// OK! between P and NP there _has_ to be Something, let's go searching!

				int iterations = 10; //lots of detail!

				float low = p;
				float hi = np;
				float mid = low + (hi - low) * 0.5;

				for (int j = 0; j < iterations; j++) {
					npp = _bezier_interp(mid, points[i].pos, points[i].pos + points[i].out, points[i + 1].pos + points[i + 1].in, points[i + 1].pos);
					d = pos.distance_to(npp);

					if (bake_interval < d) {
						hi = mid;
					} else {
						low = mid;
					}
					mid = low + (hi - low) * 0.5;
				}

				pos = npp;
				p = mid;
				Plane post;
				post.normal = pos;
				post.d = Math::lerp(points[i].tilt, points[i + 1].tilt, mid);
				pointlist.push_back(post);
			} else {
				p = np;
			}
		}
	}

	Vector3 lastpos = points[points.size() - 1].pos;
	float lastilt = points[points.size() - 1].tilt;

	float rem = pos.distance_to(lastpos);
	baked_max_ofs = (pointlist.size() - 1) * bake_interval + rem;
	pointlist.push_back(Plane(lastpos, lastilt));

	baked_point_cache.resize(pointlist.size());
	PoolVector3Array::Write w = baked_point_cache.write();
	int idx = 0;

	baked_tilt_cache.resize(pointlist.size());
	PoolRealArray::Write wt = baked_tilt_cache.write();

	baked_up_vector_cache.resize(up_vector_enabled ? pointlist.size() : 0);
	PoolVector3Array::Write up_write = baked_up_vector_cache.write();

	Vector3 sideways;
	Vector3 up;
	Vector3 forward;

	Vector3 prev_sideways = Vector3(1, 0, 0);
	Vector3 prev_up = Vector3(0, 1, 0);
	Vector3 prev_forward = Vector3(0, 0, 1);

	for (List<Plane>::Element *E = pointlist.front(); E; E = E->next()) {
		w[idx] = E->get().normal;
		wt[idx] = E->get().d;

		if (!up_vector_enabled) {
			idx++;
			continue;
		}

		forward = idx > 0 ? (w[idx] - w[idx - 1]).normalized() : prev_forward;

		float y_dot = prev_up.dot(forward);

		if (y_dot > (1.0f - CMP_EPSILON)) {
			sideways = prev_sideways;
			up = -prev_forward;
		} else if (y_dot < -(1.0f - CMP_EPSILON)) {
			sideways = prev_sideways;
			up = prev_forward;
		} else {
			sideways = prev_up.cross(forward).normalized();
			up = forward.cross(sideways).normalized();
		}

		if (idx == 1) {
			up_write[0] = up;
		}

		up_write[idx] = up;

		prev_sideways = sideways;
		prev_up = up;
		prev_forward = forward;

		idx++;
	}
}

float Curve3D::get_baked_length() const {
	if (baked_cache_dirty) {
		_bake();
	}

	return baked_max_ofs;
}
Vector3 Curve3D::interpolate_baked(float p_offset, bool p_cubic) const {
	if (baked_cache_dirty) {
		_bake();
	}

	//validate//
	int pc = baked_point_cache.size();
	ERR_FAIL_COND_V_MSG(pc == 0, Vector3(), "No points in Curve3D.");

	if (pc == 1) {
		return baked_point_cache.get(0);
	}

	int bpc = baked_point_cache.size();
	PoolVector3Array::Read r = baked_point_cache.read();

	if (p_offset < 0) {
		return r[0];
	}
	if (p_offset >= baked_max_ofs) {
		return r[bpc - 1];
	}

	int idx = Math::floor((double)p_offset / (double)bake_interval);
	float frac = Math::fmod(p_offset, bake_interval);

	if (idx >= bpc - 1) {
		return r[bpc - 1];
	} else if (idx == bpc - 2) {
		if (frac > 0) {
			frac /= Math::fmod(baked_max_ofs, bake_interval);
		}
	} else {
		frac /= bake_interval;
	}

	if (p_cubic) {
		Vector3 pre = idx > 0 ? r[idx - 1] : r[idx];
		Vector3 post = (idx < (bpc - 2)) ? r[idx + 2] : r[idx + 1];
		return r[idx].cubic_interpolate(r[idx + 1], pre, post, frac);
	} else {
		return r[idx].linear_interpolate(r[idx + 1], frac);
	}
}

float Curve3D::interpolate_baked_tilt(float p_offset) const {
	if (baked_cache_dirty) {
		_bake();
	}

	//validate//
	int pc = baked_tilt_cache.size();
	ERR_FAIL_COND_V_MSG(pc == 0, 0, "No tilts in Curve3D.");

	if (pc == 1) {
		return baked_tilt_cache.get(0);
	}

	int bpc = baked_tilt_cache.size();
	PoolRealArray::Read r = baked_tilt_cache.read();

	if (p_offset < 0) {
		return r[0];
	}
	if (p_offset >= baked_max_ofs) {
		return r[bpc - 1];
	}

	int idx = Math::floor((double)p_offset / (double)bake_interval);
	float frac = Math::fmod(p_offset, bake_interval);

	if (idx >= bpc - 1) {
		return r[bpc - 1];
	} else if (idx == bpc - 2) {
		if (frac > 0) {
			frac /= Math::fmod(baked_max_ofs, bake_interval);
		}
	} else {
		frac /= bake_interval;
	}

	return Math::lerp(r[idx], r[idx + 1], frac);
}

Vector3 Curve3D::interpolate_baked_up_vector(float p_offset, bool p_apply_tilt) const {
	if (baked_cache_dirty) {
		_bake();
	}

	//validate//
	// curve may not have baked up vectors
	int count = baked_up_vector_cache.size();
	ERR_FAIL_COND_V_MSG(count == 0, Vector3(0, 1, 0), "No up vectors in Curve3D.");

	if (count == 1) {
		return baked_up_vector_cache.get(0);
	}

	PoolVector3Array::Read r = baked_up_vector_cache.read();
	PoolVector3Array::Read rp = baked_point_cache.read();
	PoolRealArray::Read rt = baked_tilt_cache.read();

	float offset = CLAMP(p_offset, 0.0f, baked_max_ofs);

	int idx = Math::floor((double)offset / (double)bake_interval);
	float frac = Math::fmod(offset, bake_interval) / bake_interval;

	if (idx == count - 1) {
		return p_apply_tilt ? r[idx].rotated((rp[idx] - rp[idx - 1]).normalized(), rt[idx]) : r[idx];
	}

	Vector3 forward = (rp[idx + 1] - rp[idx]).normalized();
	Vector3 up = r[idx];
	Vector3 up1 = r[idx + 1];

	if (p_apply_tilt) {
		up.rotate(forward, rt[idx]);
		up1.rotate(idx + 2 >= count ? forward : (rp[idx + 2] - rp[idx + 1]).normalized(), rt[idx + 1]);
	}

	Vector3 axis = up.cross(up1);

	if (axis.length_squared() < CMP_EPSILON2) {
		axis = forward;
	} else {
		axis.normalize();
	}

	return up.rotated(axis, up.angle_to(up1) * frac);
}

PoolVector3Array Curve3D::get_baked_points() const {
	if (baked_cache_dirty) {
		_bake();
	}

	return baked_point_cache;
}

PoolRealArray Curve3D::get_baked_tilts() const {
	if (baked_cache_dirty) {
		_bake();
	}

	return baked_tilt_cache;
}

PoolVector3Array Curve3D::get_baked_up_vectors() const {
	if (baked_cache_dirty) {
		_bake();
	}

	return baked_up_vector_cache;
}

Vector3 Curve3D::get_closest_point(const Vector3 &p_to_point) const {
	// Brute force method

	if (baked_cache_dirty) {
		_bake();
	}

	//validate//
	int pc = baked_point_cache.size();
	ERR_FAIL_COND_V_MSG(pc == 0, Vector3(), "No points in Curve3D.");

	if (pc == 1) {
		return baked_point_cache.get(0);
	}

	PoolVector3Array::Read r = baked_point_cache.read();

	Vector3 nearest;
	float nearest_dist = -1.0f;

	for (int i = 0; i < pc - 1; i++) {
		Vector3 origin = r[i];
		Vector3 direction = (r[i + 1] - origin) / bake_interval;

		float d = CLAMP((p_to_point - origin).dot(direction), 0.0f, bake_interval);
		Vector3 proj = origin + direction * d;

		float dist = proj.distance_squared_to(p_to_point);

		if (nearest_dist < 0.0f || dist < nearest_dist) {
			nearest = proj;
			nearest_dist = dist;
		}
	}

	return nearest;
}

float Curve3D::get_closest_offset(const Vector3 &p_to_point) const {
	// Brute force method

	if (baked_cache_dirty) {
		_bake();
	}

	//validate//
	int pc = baked_point_cache.size();
	ERR_FAIL_COND_V_MSG(pc == 0, 0.0f, "No points in Curve3D.");

	if (pc == 1) {
		return 0.0f;
	}

	PoolVector3Array::Read r = baked_point_cache.read();

	float nearest = 0.0f;
	float nearest_dist = -1.0f;
	float offset = 0.0f;

	for (int i = 0; i < pc - 1; i++) {
		Vector3 origin = r[i];
		Vector3 direction = (r[i + 1] - origin) / bake_interval;

		float d = CLAMP((p_to_point - origin).dot(direction), 0.0f, bake_interval);
		Vector3 proj = origin + direction * d;

		float dist = proj.distance_squared_to(p_to_point);

		if (nearest_dist < 0.0f || dist < nearest_dist) {
			nearest = offset + d;
			nearest_dist = dist;
		}

		offset += bake_interval;
	}

	return nearest;
}

void Curve3D::set_bake_interval(float p_tolerance) {
	bake_interval = p_tolerance;
	baked_cache_dirty = true;
	emit_signal(CoreStringNames::get_singleton()->changed);
}

float Curve3D::get_bake_interval() const {
	return bake_interval;
}

void Curve3D::set_up_vector_enabled(bool p_enable) {
	up_vector_enabled = p_enable;
	baked_cache_dirty = true;
	emit_signal(CoreStringNames::get_singleton()->changed);
}

bool Curve3D::is_up_vector_enabled() const {
	return up_vector_enabled;
}

Dictionary Curve3D::_get_data() const {
	Dictionary dc;

	PoolVector3Array d;
	d.resize(points.size() * 3);
	PoolVector3Array::Write w = d.write();
	PoolRealArray t;
	t.resize(points.size());
	PoolRealArray::Write wt = t.write();

	for (int i = 0; i < points.size(); i++) {
		w[i * 3 + 0] = points[i].in;
		w[i * 3 + 1] = points[i].out;
		w[i * 3 + 2] = points[i].pos;
		wt[i] = points[i].tilt;
	}

	w = PoolVector3Array::Write();
	wt = PoolRealArray::Write();

	dc["points"] = d;
	dc["tilts"] = t;

	return dc;
}
void Curve3D::_set_data(const Dictionary &p_data) {
	ERR_FAIL_COND(!p_data.has("points"));
	ERR_FAIL_COND(!p_data.has("tilts"));

	PoolVector3Array rp = p_data["points"];
	int pc = rp.size();
	ERR_FAIL_COND(pc % 3 != 0);
	points.resize(pc / 3);
	PoolVector3Array::Read r = rp.read();
	PoolRealArray rtl = p_data["tilts"];
	PoolRealArray::Read rt = rtl.read();

	for (int i = 0; i < points.size(); i++) {
		points.write[i].in = r[i * 3 + 0];
		points.write[i].out = r[i * 3 + 1];
		points.write[i].pos = r[i * 3 + 2];
		points.write[i].tilt = rt[i];
	}

	baked_cache_dirty = true;
}

PoolVector3Array Curve3D::tessellate(int p_max_stages, float p_tolerance) const {
	PoolVector3Array tess;

	if (points.size() == 0) {
		return tess;
	}
	Vector<Map<float, Vector3>> midpoints;

	midpoints.resize(points.size() - 1);

	int pc = 1;
	for (int i = 0; i < points.size() - 1; i++) {
		_bake_segment3d(midpoints.write[i], 0, 1, points[i].pos, points[i].out, points[i + 1].pos, points[i + 1].in, 0, p_max_stages, p_tolerance);
		pc++;
		pc += midpoints[i].size();
	}

	tess.resize(pc);
	PoolVector3Array::Write bpw = tess.write();
	bpw[0] = points[0].pos;
	int pidx = 0;

	for (int i = 0; i < points.size() - 1; i++) {
		for (Map<float, Vector3>::Element *E = midpoints[i].front(); E; E = E->next()) {
			pidx++;
			bpw[pidx] = E->get();
		}

		pidx++;
		bpw[pidx] = points[i + 1].pos;
	}

	bpw = PoolVector3Array::Write();

	return tess;
}

void Curve3D::_bind_methods() {
	ClassDB::bind_method(D_METHOD("get_point_count"), &Curve3D::get_point_count);
	ClassDB::bind_method(D_METHOD("add_point", "position", "in", "out", "at_position"), &Curve3D::add_point, DEFVAL(Vector3()), DEFVAL(Vector3()), DEFVAL(-1));
	ClassDB::bind_method(D_METHOD("set_point_position", "idx", "position"), &Curve3D::set_point_position);
	ClassDB::bind_method(D_METHOD("get_point_position", "idx"), &Curve3D::get_point_position);
	ClassDB::bind_method(D_METHOD("set_point_tilt", "idx", "tilt"), &Curve3D::set_point_tilt);
	ClassDB::bind_method(D_METHOD("get_point_tilt", "idx"), &Curve3D::get_point_tilt);
	ClassDB::bind_method(D_METHOD("set_point_in", "idx", "position"), &Curve3D::set_point_in);
	ClassDB::bind_method(D_METHOD("get_point_in", "idx"), &Curve3D::get_point_in);
	ClassDB::bind_method(D_METHOD("set_point_out", "idx", "position"), &Curve3D::set_point_out);
	ClassDB::bind_method(D_METHOD("get_point_out", "idx"), &Curve3D::get_point_out);
	ClassDB::bind_method(D_METHOD("remove_point", "idx"), &Curve3D::remove_point);
	ClassDB::bind_method(D_METHOD("clear_points"), &Curve3D::clear_points);
	ClassDB::bind_method(D_METHOD("interpolate", "idx", "t"), &Curve3D::interpolate);
	ClassDB::bind_method(D_METHOD("interpolatef", "fofs"), &Curve3D::interpolatef);
	//ClassDB::bind_method(D_METHOD("bake","subdivs"),&Curve3D::bake,DEFVAL(10));
	ClassDB::bind_method(D_METHOD("set_bake_interval", "distance"), &Curve3D::set_bake_interval);
	ClassDB::bind_method(D_METHOD("get_bake_interval"), &Curve3D::get_bake_interval);
	ClassDB::bind_method(D_METHOD("set_up_vector_enabled", "enable"), &Curve3D::set_up_vector_enabled);
	ClassDB::bind_method(D_METHOD("is_up_vector_enabled"), &Curve3D::is_up_vector_enabled);

	ClassDB::bind_method(D_METHOD("get_baked_length"), &Curve3D::get_baked_length);
	ClassDB::bind_method(D_METHOD("interpolate_baked", "offset", "cubic"), &Curve3D::interpolate_baked, DEFVAL(false));
	ClassDB::bind_method(D_METHOD("interpolate_baked_up_vector", "offset", "apply_tilt"), &Curve3D::interpolate_baked_up_vector, DEFVAL(false));
	ClassDB::bind_method(D_METHOD("get_baked_points"), &Curve3D::get_baked_points);
	ClassDB::bind_method(D_METHOD("get_baked_tilts"), &Curve3D::get_baked_tilts);
	ClassDB::bind_method(D_METHOD("get_baked_up_vectors"), &Curve3D::get_baked_up_vectors);
	ClassDB::bind_method(D_METHOD("get_closest_point", "to_point"), &Curve3D::get_closest_point);
	ClassDB::bind_method(D_METHOD("get_closest_offset", "to_point"), &Curve3D::get_closest_offset);
	ClassDB::bind_method(D_METHOD("tessellate", "max_stages", "tolerance_degrees"), &Curve3D::tessellate, DEFVAL(5), DEFVAL(4));

	ClassDB::bind_method(D_METHOD("_get_data"), &Curve3D::_get_data);
	ClassDB::bind_method(D_METHOD("_set_data"), &Curve3D::_set_data);

	ADD_PROPERTY(PropertyInfo(Variant::REAL, "bake_interval", PROPERTY_HINT_RANGE, "0.01,512,0.01"), "set_bake_interval", "get_bake_interval");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "_data", PROPERTY_HINT_NONE, "", PROPERTY_USAGE_NOEDITOR | PROPERTY_USAGE_INTERNAL), "_set_data", "_get_data");

	ADD_GROUP("Up Vector", "up_vector_");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "up_vector_enabled"), "set_up_vector_enabled", "is_up_vector_enabled");
}

Curve3D::Curve3D() {
	baked_cache_dirty = false;
	baked_max_ofs = 0;
	/*	add_point(Vector3(-1,0,0));
	add_point(Vector3(0,2,0));
	add_point(Vector3(0,3,5));*/
	bake_interval = 0.2;
	up_vector_enabled = true;
}
