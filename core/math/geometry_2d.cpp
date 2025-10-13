/**************************************************************************/
/*  geometry_2d.cpp                                                       */
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

#include "geometry_2d.h"

#include "core/math/delaunay_2d.h"
#include "core/math/math_funcs.h"
#include "core/math/rect2.h"
#include "core/math/triangulate.h"
#include "core/math/vector2i.h"
#include "core/math/vector3i.h"

#include <thirdparty/clipper2/include/clipper2/clipper.h>
#include <thirdparty/misc/polypartition.h>
#define STB_RECT_PACK_IMPLEMENTATION
#include <thirdparty/misc/stb_rect_pack.h>

const int clipper_precision = 5; // Based on CMP_EPSILON.
const double clipper_scale = Math::pow(10.0, clipper_precision);

void Geometry2D::merge_many_polygons(const Vector<Vector<Vector2>> &p_polygons, Vector<Vector<Vector2>> &r_out_polygons, Vector<Vector<Vector2>> &r_out_holes) {
	using namespace Clipper2Lib;

	PathsD subjects;
	for (const Vector<Vector2> &polygon : p_polygons) {
		PathD path(polygon.size());
		for (int i = 0; i < polygon.size(); i++) {
			const Vector2 &point = polygon[i];
			path[i] = PointD(point.x, point.y);
		}
		subjects.push_back(path);
	}

	PathsD solution = Union(subjects, FillRule::NonZero);
	solution = SimplifyPaths(solution, 0.01);

	r_out_polygons.clear();
	r_out_holes.clear();
	for (PathsD::size_type i = 0; i < solution.size(); ++i) {
		PathD &path = solution[i];

		Vector<Point2> output_polygon;
		output_polygon.resize(path.size());
		for (PathsD::size_type j = 0; j < path.size(); ++j) {
			output_polygon.set(j, Vector2(static_cast<real_t>(path[j].x), static_cast<real_t>(path[j].y)));
		}
		if (IsPositive(path)) {
			r_out_polygons.push_back(output_polygon);
		} else {
			r_out_holes.push_back(output_polygon);
		}
	}
}

Vector<Vector<Vector2>> Geometry2D::decompose_many_polygons_in_convex(const Vector<Vector<Point2>> &p_polygons, const Vector<Vector<Point2>> &p_holes) {
	Vector<Vector<Vector2>> decomp;
	List<TPPLPoly> in_poly, out_poly;

	for (const Vector<Vector2> &polygon : p_polygons) {
		TPPLPoly inp;
		inp.Init(polygon.size());
		for (int i = 0; i < polygon.size(); i++) {
			inp.GetPoint(i) = polygon[i];
		}
		inp.SetOrientation(TPPL_ORIENTATION_CCW);
		in_poly.push_back(inp);
	}
	for (const Vector<Vector2> &polygon : p_holes) {
		TPPLPoly inp;
		inp.Init(polygon.size());
		for (int i = 0; i < polygon.size(); i++) {
			inp.GetPoint(i) = polygon[i];
		}
		inp.SetOrientation(TPPL_ORIENTATION_CW);
		inp.SetHole(true);
		in_poly.push_back(inp);
	}
	TPPLPartition tpart;
	if (tpart.ConvexPartition_HM(&in_poly, &out_poly) == 0) { // Failed.
		ERR_PRINT("Convex decomposing failed!");
		return decomp;
	}

	decomp.resize(out_poly.size());
	int idx = 0;
	for (TPPLPoly &tp : out_poly) {
		decomp.write[idx].resize(tp.GetNumPoints());

		for (int64_t i = 0; i < tp.GetNumPoints(); i++) {
			decomp.write[idx].write[i] = tp.GetPoint(i);
		}

		idx++;
	}

	return decomp;
}

Vector<Vector<Vector2>> Geometry2D::decompose_polygon_in_convex(const Vector<Point2> &p_polygon) {
	return Geometry2D::decompose_many_polygons_in_convex({ p_polygon }, {});
}

struct _AtlasWorkRect {
	Size2i s;
	Point2i p;
	int idx = 0;
	_FORCE_INLINE_ bool operator<(const _AtlasWorkRect &p_r) const { return s.width > p_r.s.width; }
};

struct _AtlasWorkRectResult {
	Vector<_AtlasWorkRect> result;
	int max_w = 0;
	int max_h = 0;
};

void Geometry2D::make_atlas(const Vector<Size2i> &p_rects, Vector<Point2i> &r_result, Size2i &r_size) {
	// Super simple, almost brute force scanline stacking fitter.
	// It's pretty basic for now, but it tries to make sure that the aspect ratio of the
	// resulting atlas is somehow square. This is necessary because video cards have limits
	// on texture size (usually 2048 or 4096), so the squarer a texture, the more the chances
	// that it will work in every hardware.
	// For example, it will prioritize a 1024x1024 atlas (works everywhere) instead of a
	// 256x8192 atlas (won't work anywhere).

	ERR_FAIL_COND(p_rects.is_empty());
	for (int i = 0; i < p_rects.size(); i++) {
		ERR_FAIL_COND(p_rects[i].width <= 0);
		ERR_FAIL_COND(p_rects[i].height <= 0);
	}

	Vector<_AtlasWorkRect> wrects;
	wrects.resize(p_rects.size());
	for (int i = 0; i < p_rects.size(); i++) {
		wrects.write[i].s = p_rects[i];
		wrects.write[i].idx = i;
	}
	wrects.sort();
	int widest = wrects[0].s.width;

	Vector<_AtlasWorkRectResult> results;

	for (int i = 0; i <= 12; i++) {
		int w = 1 << i;
		int max_h = 0;
		int max_w = 0;
		if (w < widest) {
			continue;
		}

		Vector<int> hmax;
		hmax.resize(w);
		for (int j = 0; j < w; j++) {
			hmax.write[j] = 0;
		}

		// Place them.
		int ofs = 0;
		int limit_h = 0;
		for (int j = 0; j < wrects.size(); j++) {
			if (ofs + wrects[j].s.width > w) {
				ofs = 0;
			}

			int from_y = 0;
			for (int k = 0; k < wrects[j].s.width; k++) {
				if (hmax[ofs + k] > from_y) {
					from_y = hmax[ofs + k];
				}
			}

			wrects.write[j].p.x = ofs;
			wrects.write[j].p.y = from_y;
			int end_h = from_y + wrects[j].s.height;
			int end_w = ofs + wrects[j].s.width;
			if (ofs == 0) {
				limit_h = end_h;
			}

			for (int k = 0; k < wrects[j].s.width; k++) {
				hmax.write[ofs + k] = end_h;
			}

			if (end_h > max_h) {
				max_h = end_h;
			}

			if (end_w > max_w) {
				max_w = end_w;
			}

			if (ofs == 0 || end_h > limit_h) { // While h limit not reached, keep stacking.
				ofs += wrects[j].s.width;
			}
		}

		_AtlasWorkRectResult result;
		result.result = wrects;
		result.max_h = max_h;
		result.max_w = max_w;
		results.push_back(result);
	}

	// Find the result with the best aspect ratio.

	int best = -1;
	real_t best_aspect = 1e20;

	for (int i = 0; i < results.size(); i++) {
		real_t h = next_power_of_2((uint32_t)results[i].max_h);
		real_t w = next_power_of_2((uint32_t)results[i].max_w);
		real_t aspect = h > w ? h / w : w / h;
		if (aspect < best_aspect) {
			best = i;
			best_aspect = aspect;
		}
	}

	r_result.resize(p_rects.size());

	for (int i = 0; i < p_rects.size(); i++) {
		r_result.write[results[best].result[i].idx] = results[best].result[i].p;
	}

	r_size = Size2(results[best].max_w, results[best].max_h);
}

static inline Vector<Vector<Point2>> _polypaths_do_operation(Geometry2D::PolyBooleanOperation p_op, const Vector<Point2> &p_polypath_a, const Vector<Point2> &p_polypath_b, bool is_a_open = false) {
	using namespace Clipper2Lib;

	ClipType op = ClipType::Union;

	switch (p_op) {
		case Geometry2D::OPERATION_UNION:
			op = ClipType::Union;
			break;
		case Geometry2D::OPERATION_DIFFERENCE:
			op = ClipType::Difference;
			break;
		case Geometry2D::OPERATION_INTERSECTION:
			op = ClipType::Intersection;
			break;
		case Geometry2D::OPERATION_XOR:
			op = ClipType::Xor;
			break;
	}

	PathD path_a(p_polypath_a.size());
	for (int i = 0; i != p_polypath_a.size(); ++i) {
		path_a[i] = PointD(p_polypath_a[i].x, p_polypath_a[i].y);
	}
	PathD path_b(p_polypath_b.size());
	for (int i = 0; i != p_polypath_b.size(); ++i) {
		path_b[i] = PointD(p_polypath_b[i].x, p_polypath_b[i].y);
	}

	ClipperD clp(clipper_precision); // Scale points up internally to attain the desired precision.
	clp.PreserveCollinear(false); // Remove redundant vertices.
	if (is_a_open) {
		clp.AddOpenSubject({ path_a });
	} else {
		clp.AddSubject({ path_a });
	}
	clp.AddClip({ path_b });

	PathsD paths;

	if (is_a_open) {
		PolyTreeD tree; // Needed to populate polylines.
		clp.Execute(op, FillRule::EvenOdd, tree, paths);
	} else {
		clp.Execute(op, FillRule::EvenOdd, paths); // Works on closed polygons only.
	}

	Vector<Vector<Point2>> polypaths;
	polypaths.resize(paths.size());
	for (PathsD::size_type i = 0; i < paths.size(); i++) {
		const PathD &path = paths[i];

		Vector<Vector2> polypath;
		polypath.resize(path.size());
		for (PathsD::size_type j = 0; j < path.size(); ++j) {
			polypath.set(j, Point2(static_cast<real_t>(path[j].x), static_cast<real_t>(path[j].y)));
		}
		polypaths.set(i, polypath);
	}
	return polypaths;
}

static inline Vector<Vector<Point2>> _polypath_offset(const Vector<Point2> &p_polypath, real_t p_delta, Geometry2D::PolyJoinType p_join_type, Geometry2D::PolyEndType p_end_type) {
	using namespace Clipper2Lib;

	JoinType jt = JoinType::Square;

	switch (p_join_type) {
		case Geometry2D::JOIN_SQUARE:
			jt = JoinType::Square;
			break;
		case Geometry2D::JOIN_ROUND:
			jt = JoinType::Round;
			break;
		case Geometry2D::JOIN_MITER:
			jt = JoinType::Miter;
			break;
	}

	EndType et = EndType::Polygon;

	switch (p_end_type) {
		case Geometry2D::END_POLYGON:
			et = EndType::Polygon;
			break;
		case Geometry2D::END_JOINED:
			et = EndType::Joined;
			break;
		case Geometry2D::END_BUTT:
			et = EndType::Butt;
			break;
		case Geometry2D::END_SQUARE:
			et = EndType::Square;
			break;
		case Geometry2D::END_ROUND:
			et = EndType::Round;
			break;
	}

	PathD polypath(p_polypath.size());
	for (int i = 0; i != p_polypath.size(); ++i) {
		polypath[i] = PointD(p_polypath[i].x, p_polypath[i].y);
	}

	// Inflate/deflate.
	PathsD paths = InflatePaths({ polypath }, p_delta, jt, et, 2.0, clipper_precision, 0.25 * clipper_scale);
	// Here the points are scaled up internally and
	// the arc_tolerance is scaled accordingly
	// to attain the desired precision.

	Vector<Vector<Point2>> polypaths;
	polypaths.resize(paths.size());
	for (PathsD::size_type i = 0; i < paths.size(); ++i) {
		const PathD &path = paths[i];

		Vector<Vector2> polypath2;
		polypath2.resize(path.size());
		for (PathsD::size_type j = 0; j < path.size(); ++j) {
			polypath2.set(j, Point2(static_cast<real_t>(path[j].x), static_cast<real_t>(path[j].y)));
		}
		polypaths.set(i, polypath2);
	}
	return polypaths;
}

Vector<Vector3i> Geometry2D::partial_pack_rects(const Vector<Vector2i> &p_sizes, const Size2i &p_atlas_size) {
	Vector<stbrp_node> nodes;
	nodes.resize(p_atlas_size.width);
	memset(nodes.ptrw(), 0, sizeof(stbrp_node) * nodes.size());

	stbrp_context context;
	stbrp_init_target(&context, p_atlas_size.width, p_atlas_size.height, nodes.ptrw(), p_atlas_size.width);

	Vector<stbrp_rect> rects;
	rects.resize(p_sizes.size());

	for (int i = 0; i < p_sizes.size(); i++) {
		rects.write[i].id = i;
		rects.write[i].w = p_sizes[i].width;
		rects.write[i].h = p_sizes[i].height;
		rects.write[i].x = 0;
		rects.write[i].y = 0;
		rects.write[i].was_packed = 0;
	}

	stbrp_pack_rects(&context, rects.ptrw(), rects.size());

	Vector<Vector3i> ret;
	ret.resize(p_sizes.size());

	for (int i = 0; i < p_sizes.size(); i++) {
		ret.write[rects[i].id] = Vector3i(rects[i].x, rects[i].y, rects[i].was_packed != 0 ? 1 : 0);
	}

	return ret;
}

real_t Geometry2D::get_closest_points_between_segments(const Vector2 &p_p1, const Vector2 &p_q1, const Vector2 &p_p2, const Vector2 &p_q2, Vector2 &r_c1, Vector2 &r_c2) {
	Vector2 d1 = p_q1 - p_p1; // Direction vector of segment S1.
	Vector2 d2 = p_q2 - p_p2; // Direction vector of segment S2.
	Vector2 r = p_p1 - p_p2;
	real_t a = d1.dot(d1); // Squared length of segment S1, always nonnegative.
	real_t e = d2.dot(d2); // Squared length of segment S2, always nonnegative.
	real_t f = d2.dot(r);
	real_t s;
	real_t t;
	// Check if either or both segments degenerate into points.
	if (a <= (real_t)CMP_EPSILON && e <= (real_t)CMP_EPSILON) {
		// Both segments degenerate into points.
		r_c1 = p_p1;
		r_c2 = p_p2;
		return Math::sqrt((r_c1 - r_c2).dot(r_c1 - r_c2));
	}
	if (a <= (real_t)CMP_EPSILON) {
		// First segment degenerates into a point.
		s = 0.0;
		t = f / e; // s = 0 => t = (b*s + f) / e = f / e
		t = CLAMP(t, 0.0f, 1.0f);
	} else {
		real_t c = d1.dot(r);
		if (e <= (real_t)CMP_EPSILON) {
			// Second segment degenerates into a point.
			t = 0.0;
			s = CLAMP(-c / a, 0.0f, 1.0f); // t = 0 => s = (b*t - c) / a = -c / a
		} else {
			// The general nondegenerate case starts here.
			real_t b = d1.dot(d2);
			real_t denom = a * e - b * b; // Always nonnegative.
			// If segments not parallel, compute closest point on L1 to L2 and
			// clamp to segment S1. Else pick arbitrary s (here 0).
			if (denom != 0.0f) {
				s = CLAMP((b * f - c * e) / denom, 0.0f, 1.0f);
			} else {
				s = 0.0;
			}
			// Compute point on L2 closest to S1(s) using
			// t = Dot((p_p1 + D1*s) - p_p2,D2) / Dot(D2,D2) = (b*s + f) / e
			t = (b * s + f) / e;

			//If t in [0,1] done. Else clamp t, recompute s for the new value
			// of t using s = Dot((p_p2 + D2*t) - p_p1,D1) / Dot(D1,D1)= (t*b - c) / a
			// and clamp s to [0, 1].
			if (t < 0.0f) {
				t = 0.0;
				s = CLAMP(-c / a, 0.0f, 1.0f);
			} else if (t > 1.0f) {
				t = 1.0;
				s = CLAMP((b - c) / a, 0.0f, 1.0f);
			}
		}
	}
	r_c1 = p_p1 + d1 * s;
	r_c2 = p_p2 + d2 * t;
	return Math::sqrt((r_c1 - r_c2).dot(r_c1 - r_c2));
}

Vector2 Geometry2D::get_closest_point_to_segment(const Vector2 &p_point, const Vector2 &p_segment_a, const Vector2 &p_segment_b) {
	Vector2 p = p_point - p_segment_a;
	Vector2 n = p_segment_b - p_segment_a;
	real_t l2 = n.length_squared();
	if (l2 < 1e-20f) {
		return p_segment_a; // Both points are the same, just give any.
	}

	real_t d = n.dot(p) / l2;

	if (d <= 0.0f) {
		return p_segment_a; // Before first point.
	} else if (d >= 1.0f) {
		return p_segment_b; // After first point.
	}
	return p_segment_a + n * d; // Inside.
}

real_t Geometry2D::get_distance_to_segment(const Vector2 &p_point, const Vector2 &p_segment_a, const Vector2 &p_segment_b) {
	return p_point.distance_to(get_closest_point_to_segment(p_point, p_segment_a, p_segment_b));
}

Vector2 Geometry2D::get_closest_point_to_segment_uncapped(const Vector2 &p_point, const Vector2 &p_segment_a, const Vector2 &p_segment_b) {
	Vector2 p = p_point - p_segment_a;
	Vector2 n = p_segment_b - p_segment_a;
	real_t l2 = n.length_squared();
	if (l2 < 1e-20f) {
		return p_segment_a; // Both points are the same, just give any.
	}

	real_t d = n.dot(p) / l2;

	return p_segment_a + n * d; // Inside.
}

#ifndef DISABLE_DEPRECATED
Vector2 Geometry2D::get_closest_point_to_segment(const Vector2 &p_point, const Vector2 *p_segment) {
	return get_closest_point_to_segment(p_point, p_segment[0], p_segment[1]);
}

real_t Geometry2D::get_distance_to_segment(const Vector2 &p_point, const Vector2 *p_segment) {
	return get_distance_to_segment(p_point, p_segment[0], p_segment[1]);
}

Vector2 Geometry2D::get_closest_point_to_segment_uncapped(const Vector2 &p_point, const Vector2 *p_segment) {
	return get_closest_point_to_segment_uncapped(p_point, p_segment[0], p_segment[1]);
}
#endif // DISABLE_DEPRECATED

bool Geometry2D::is_point_in_triangle(const Vector2 &p_s, const Vector2 &p_a, const Vector2 &p_b, const Vector2 &p_c) {
	Vector2 an = p_a - p_s;
	Vector2 bn = p_b - p_s;
	Vector2 cn = p_c - p_s;

	bool orientation = an.cross(bn) > 0;

	if ((bn.cross(cn) > 0) != orientation) {
		return false;
	}

	return (cn.cross(an) > 0) == orientation;
}

GODOT_MSVC_WARNING_PUSH_AND_IGNORE(4723) // Potential divide by 0. False positive (see: GH-44274).

bool Geometry2D::line_intersects_line(const Vector2 &p_from_a, const Vector2 &p_dir_a, const Vector2 &p_from_b, const Vector2 &p_dir_b, Vector2 &r_result) {
	// See http://paulbourke.net/geometry/pointlineplane/

	const real_t denom = p_dir_b.y * p_dir_a.x - p_dir_b.x * p_dir_a.y;
	if (Math::is_zero_approx(denom)) { // Parallel?
		return false;
	}

	const Vector2 v = p_from_a - p_from_b;
	const real_t t = (p_dir_b.x * v.y - p_dir_b.y * v.x) / denom;
	r_result = p_from_a + t * p_dir_a;
	return true;
}

GODOT_MSVC_WARNING_POP

bool Geometry2D::segment_intersects_segment(const Vector2 &p_from_a, const Vector2 &p_to_a, const Vector2 &p_from_b, const Vector2 &p_to_b, Vector2 *r_result) {
	Vector2 b = p_to_a - p_from_a;
	Vector2 c = p_from_b - p_from_a;
	Vector2 d = p_to_b - p_from_a;

	real_t a_b_len = b.dot(b);
	if (a_b_len <= 0) {
		return false;
	}
	Vector2 b_n = b / a_b_len;
	c = Vector2(c.x * b_n.x + c.y * b_n.y, c.y * b_n.x - c.x * b_n.y);
	d = Vector2(d.x * b_n.x + d.y * b_n.y, d.y * b_n.x - d.x * b_n.y);

	// Fail if c x b and d x b have the same sign (segments don't intersect).
	if ((c.y < (real_t)-CMP_EPSILON && d.y < (real_t)-CMP_EPSILON) || (c.y > (real_t)CMP_EPSILON && d.y > (real_t)CMP_EPSILON)) {
		return false;
	}

	// Fail if segments are parallel or colinear.
	// (when A x b == zero, i.e (c - d) x b == zero, i.e c x b == d x b)
	if (Math::is_equal_approx(c.y, d.y)) {
		return false;
	}

	real_t a_b_pos = d.x + (c.x - d.x) * d.y / (d.y - c.y);

	// Fail if segment c-d crosses line A-b outside of segment A-b.
	if ((a_b_pos < 0) || (a_b_pos > 1)) {
		return false;
	}

	// Apply the discovered position to line A-b in the original coordinate system.
	if (r_result) {
		*r_result = p_from_a + b * a_b_pos;
	}

	return true;
}

bool Geometry2D::is_point_in_circle(const Vector2 &p_point, const Vector2 &p_circle_pos, real_t p_circle_radius) {
	return p_point.distance_squared_to(p_circle_pos) <= p_circle_radius * p_circle_radius;
}

real_t Geometry2D::segment_intersects_circle(const Vector2 &p_from, const Vector2 &p_to, const Vector2 &p_circle_pos, real_t p_circle_radius) {
	Vector2 line_vec = p_to - p_from;
	Vector2 vec_to_line = p_from - p_circle_pos;

	// Create a quadratic formula of the form ax^2 + bx + c = 0
	real_t a = line_vec.dot(line_vec);
	real_t b = 2 * vec_to_line.dot(line_vec);
	real_t c = vec_to_line.dot(vec_to_line) - p_circle_radius * p_circle_radius;

	// Solve for t.
	real_t sqrtterm = b * b - 4 * a * c;

	// If the term we intend to square root is less than 0 then the answer won't be real,
	// so it definitely won't be t in the range 0 to 1.
	if (sqrtterm < 0) {
		return -1;
	}

	// If we can assume that the line segment starts outside the circle (e.g. for continuous time collision detection)
	// then the following can be skipped and we can just return the equivalent of res1.
	sqrtterm = Math::sqrt(sqrtterm);
	real_t res1 = (-b - sqrtterm) / (2 * a);
	real_t res2 = (-b + sqrtterm) / (2 * a);

	if (res1 >= 0 && res1 <= 1) {
		return res1;
	}
	if (res2 >= 0 && res2 <= 1) {
		return res2;
	}
	return -1;
}

bool Geometry2D::segment_intersects_rect(const Vector2 &p_from, const Vector2 &p_to, const Rect2 &p_rect) {
	if (p_rect.has_point(p_from) || p_rect.has_point(p_to)) {
		return true;
	}

	const Vector2 rect_points[4] = {
		p_rect.position,
		p_rect.position + Vector2(p_rect.size.x, 0),
		p_rect.position + p_rect.size,
		p_rect.position + Vector2(0, p_rect.size.y),
	};

	// Check if any of the rect's edges intersect the segment.
	for (int i = 0; i < 4; i++) {
		if (segment_intersects_segment(p_from, p_to, rect_points[i], rect_points[(i + 1) % 4], nullptr)) {
			return true;
		}
	}

	return false;
}

Vector<Vector<Point2>> Geometry2D::merge_polygons(const Vector<Point2> &p_polygon_a, const Vector<Point2> &p_polygon_b) {
	return _polypaths_do_operation(OPERATION_UNION, p_polygon_a, p_polygon_b);
}

Vector<Vector<Point2>> Geometry2D::clip_polygons(const Vector<Point2> &p_polygon_a, const Vector<Point2> &p_polygon_b) {
	return _polypaths_do_operation(OPERATION_DIFFERENCE, p_polygon_a, p_polygon_b);
}

Vector<Vector<Point2>> Geometry2D::intersect_polygons(const Vector<Point2> &p_polygon_a, const Vector<Point2> &p_polygon_b) {
	return _polypaths_do_operation(OPERATION_INTERSECTION, p_polygon_a, p_polygon_b);
}

Vector<Vector<Point2>> Geometry2D::exclude_polygons(const Vector<Point2> &p_polygon_a, const Vector<Point2> &p_polygon_b) {
	return _polypaths_do_operation(OPERATION_XOR, p_polygon_a, p_polygon_b);
}

Vector<Vector<Point2>> Geometry2D::clip_polyline_with_polygon(const Vector<Vector2> &p_polyline, const Vector<Vector2> &p_polygon) {
	return _polypaths_do_operation(OPERATION_DIFFERENCE, p_polyline, p_polygon, true);
}

Vector<Vector<Point2>> Geometry2D::intersect_polyline_with_polygon(const Vector<Vector2> &p_polyline, const Vector<Vector2> &p_polygon) {
	return _polypaths_do_operation(OPERATION_INTERSECTION, p_polyline, p_polygon, true);
}

Vector<Vector<Point2>> Geometry2D::offset_polygon(const Vector<Vector2> &p_polygon, real_t p_delta, PolyJoinType p_join_type) {
	return _polypath_offset(p_polygon, p_delta, p_join_type, END_POLYGON);
}

Vector<Vector<Point2>> Geometry2D::offset_polyline(const Vector<Vector2> &p_polygon, real_t p_delta, PolyJoinType p_join_type, PolyEndType p_end_type) {
	ERR_FAIL_COND_V_MSG(p_end_type == END_POLYGON, Vector<Vector<Point2>>(), "Attempt to offset a polyline like a polygon (use offset_polygon instead).");

	return _polypath_offset(p_polygon, p_delta, p_join_type, p_end_type);
}

Vector<int32_t> Geometry2D::triangulate_delaunay(const Vector<Vector2> &p_points) {
	Vector<Delaunay2D::Triangle> tr = Delaunay2D::triangulate(p_points);
	Vector<int32_t> triangles;

	triangles.resize(3 * tr.size());
	int32_t *ptr = triangles.ptrw();
	for (int32_t i = 0; i < tr.size(); i++) {
		*ptr++ = tr[i].points[0];
		*ptr++ = tr[i].points[1];
		*ptr++ = tr[i].points[2];
	}
	return triangles;
}

Vector<int32_t> Geometry2D::triangulate_polygon(const Vector<Vector2> &p_polygon) {
	Vector<int32_t> triangles;
	if (!Triangulate::triangulate(p_polygon, triangles)) {
		return Vector<int32_t>(); // Fail.
	}
	return triangles;
}

// Assumes cartesian coordinate system with +x to the right, +y up.
// If using screen coordinates (+x to the right, +y down) the result will need to be flipped.
bool Geometry2D::is_polygon_clockwise(const Vector<Vector2> &p_polygon) {
	int c = p_polygon.size();
	if (c < 3) {
		return false;
	}
	const Vector2 *p = p_polygon.ptr();
	real_t sum = 0;
	for (int i = 0; i < c; i++) {
		const Vector2 &v1 = p[i];
		const Vector2 &v2 = p[(i + 1) % c];
		sum += (v2.x - v1.x) * (v2.y + v1.y);
	}

	return sum > 0.0f;
}

// Alternate implementation that should be faster.
bool Geometry2D::is_point_in_polygon(const Vector2 &p_point, const Vector<Vector2> &p_polygon) {
	int c = p_polygon.size();
	if (c < 3) {
		return false;
	}
	const Vector2 *p = p_polygon.ptr();
	Vector2 further_away(-1e20, -1e20);
	Vector2 further_away_opposite(1e20, 1e20);

	for (int i = 0; i < c; i++) {
		further_away = further_away.max(p[i]);
		further_away_opposite = further_away_opposite.min(p[i]);
	}

	// Make point outside that won't intersect with points in segment from p_point.
	further_away += (further_away - further_away_opposite) * Vector2(1.221313, 1.512312);

	int intersections = 0;
	for (int i = 0; i < c; i++) {
		const Vector2 &v1 = p[i];
		const Vector2 &v2 = p[(i + 1) % c];

		Vector2 res;
		if (segment_intersects_segment(v1, v2, p_point, further_away, &res)) {
			intersections++;
			if (res.is_equal_approx(p_point)) {
				// Point is in one of the polygon edges.
				return true;
			}
		}
	}

	return (intersections & 1);
}

bool Geometry2D::is_segment_intersecting_polygon(const Vector2 &p_from, const Vector2 &p_to, const Vector<Vector2> &p_polygon) {
	int c = p_polygon.size();
	const Vector2 *p = p_polygon.ptr();
	for (int i = 0; i < c; i++) {
		const Vector2 &v1 = p[i];
		const Vector2 &v2 = p[(i + 1) % c];
		if (segment_intersects_segment(p_from, p_to, v1, v2, nullptr)) {
			return true;
		}
	}
	return false;
}

real_t Geometry2D::vec2_cross(const Point2 &p_o, const Point2 &p_a, const Point2 &p_b) {
	return (real_t)(p_a.x - p_o.x) * (p_b.y - p_o.y) - (real_t)(p_a.y - p_o.y) * (p_b.x - p_o.x);
}

// Returns a list of points on the convex hull in counter-clockwise order.
// Note: the last point in the returned list is the same as the first one.
Vector<Point2> Geometry2D::convex_hull(Vector<Point2> p_points) {
	int32_t n = p_points.size();
	int32_t k = 0;
	Vector<Point2> h;
	h.resize(2 * n);

	// Sort points lexicographically.
	p_points.sort();

	// Build lower hull.
	for (int i = 0; i < n; ++i) {
		while (k >= 2 && vec2_cross(h[k - 2], h[k - 1], p_points[i]) <= 0) {
			k--;
		}
		h.write[k++] = p_points[i];
	}

	// Build upper hull.
	for (int i = n - 2, t = k + 1; i >= 0; i--) {
		while (k >= t && vec2_cross(h[k - 2], h[k - 1], p_points[i]) <= 0) {
			k--;
		}
		h.write[k++] = p_points[i];
	}

	h.resize(k);
	return h;
}

Vector<Point2i> Geometry2D::bresenham_line(const Point2i &p_from, const Point2i &p_to) {
	Vector<Point2i> points;

	Vector2i delta = (p_to - p_from).abs() * 2;
	Vector2i step = (p_to - p_from).sign();
	Vector2i current = p_from;

	if (delta.x > delta.y) {
		int err = delta.x / 2;

		for (; current.x != p_to.x; current.x += step.x) {
			points.push_back(current);

			err -= delta.y;
			if (err < 0) {
				current.y += step.y;
				err += delta.x;
			}
		}
	} else {
		int err = delta.y / 2;

		for (; current.y != p_to.y; current.y += step.y) {
			points.push_back(current);

			err -= delta.x;
			if (err < 0) {
				current.x += step.x;
				err += delta.y;
			}
		}
	}

	points.push_back(current);

	return points;
}
