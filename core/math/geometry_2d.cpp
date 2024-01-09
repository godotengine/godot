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

#include "core/math/bentley_ottmann.h"
#include "core/math/transform_2d.h"

#include "thirdparty/misc/clipper.hpp"
#include "thirdparty/misc/polypartition.h"
#define STB_RECT_PACK_IMPLEMENTATION
#include "thirdparty/misc/stb_rect_pack.h"

#define SCALE_FACTOR 100000.0 // Based on CMP_EPSILON.

void Geometry2D::triangulate_polygons(const Vector<Vector<Vector2>> &p_polygons, bool p_winding_even_odd, Vector<Vector2> &r_points, Vector<int32_t> &r_triangles) {
	Vector<Vector2> edges;
	Vector<int> winding;
	for (int i = 0; i < p_polygons.size(); i++) {
		const Vector<Vector2> &polygon = p_polygons[i];
		if (polygon.size() < 3) {
			continue;
		}
		for (int j = 1; j < polygon.size(); j++) {
			edges.push_back(polygon[j - 1]);
			edges.push_back(polygon[j]);
			winding.push_back(1);
		}
		edges.push_back(polygon[polygon.size() - 1]);
		edges.push_back(polygon[0]);
		winding.push_back(1);
	}
	BentleyOttmann triangulator(edges, winding, p_winding_even_odd);
	r_points = triangulator.out_points;
	r_triangles = triangulator.out_triangles;
}

Vector<Vector<Vector2>> Geometry2D::decompose_polygon_in_convex(Vector<Point2> polygon) {
	Vector<Vector<Vector2>> decomp;
	List<TPPLPoly> in_poly, out_poly;

	TPPLPoly inp;
	inp.Init(polygon.size());
	for (int i = 0; i < polygon.size(); i++) {
		inp.GetPoint(i) = polygon[i];
	}
	inp.SetOrientation(TPPL_ORIENTATION_CCW);
	in_poly.push_back(inp);
	TPPLPartition tpart;
	if (tpart.ConvexPartition_HM(&in_poly, &out_poly) == 0) { // Failed.
		ERR_PRINT("Convex decomposing failed!");
		return decomp;
	}

	decomp.resize(out_poly.size());
	int idx = 0;
	for (List<TPPLPoly>::Element *I = out_poly.front(); I; I = I->next()) {
		TPPLPoly &tp = I->get();

		decomp.write[idx].resize(tp.GetNumPoints());

		for (int64_t i = 0; i < tp.GetNumPoints(); i++) {
			decomp.write[idx].write[i] = tp.GetPoint(i);
		}

		idx++;
	}

	return decomp;
}

struct _AtlasWorkRect {
	Size2i s;
	Point2i p;
	int idx = 0;
	_FORCE_INLINE_ bool operator<(const _AtlasWorkRect &p_r) const { return s.width > p_r.s.width; };
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

	ERR_FAIL_COND(p_rects.size() == 0);
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
		real_t h = next_power_of_2(results[i].max_h);
		real_t w = next_power_of_2(results[i].max_w);
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

Vector<Vector<Point2>> Geometry2D::_polypaths_do_operation(PolyBooleanOperation p_op, const Vector<Point2> &p_polypath_a, const Vector<Point2> &p_polypath_b, bool is_a_open) {
	using namespace ClipperLib;

	ClipType op = ctUnion;

	switch (p_op) {
		case OPERATION_UNION:
			op = ctUnion;
			break;
		case OPERATION_DIFFERENCE:
			op = ctDifference;
			break;
		case OPERATION_INTERSECTION:
			op = ctIntersection;
			break;
		case OPERATION_XOR:
			op = ctXor;
			break;
	}
	Path path_a, path_b;

	// Need to scale points (Clipper's requirement for robust computation).
	for (int i = 0; i != p_polypath_a.size(); ++i) {
		path_a << IntPoint(p_polypath_a[i].x * (real_t)SCALE_FACTOR, p_polypath_a[i].y * (real_t)SCALE_FACTOR);
	}
	for (int i = 0; i != p_polypath_b.size(); ++i) {
		path_b << IntPoint(p_polypath_b[i].x * (real_t)SCALE_FACTOR, p_polypath_b[i].y * (real_t)SCALE_FACTOR);
	}
	Clipper clp;
	clp.AddPath(path_a, ptSubject, !is_a_open); // Forward compatible with Clipper 10.0.0.
	clp.AddPath(path_b, ptClip, true); // Polylines cannot be set as clip.

	Paths paths;

	if (is_a_open) {
		PolyTree tree; // Needed to populate polylines.
		clp.Execute(op, tree);
		OpenPathsFromPolyTree(tree, paths);
	} else {
		clp.Execute(op, paths); // Works on closed polygons only.
	}
	// Have to scale points down now.
	Vector<Vector<Point2>> polypaths;

	for (Paths::size_type i = 0; i < paths.size(); ++i) {
		Vector<Vector2> polypath;

		const Path &scaled_path = paths[i];

		for (Paths::size_type j = 0; j < scaled_path.size(); ++j) {
			polypath.push_back(Point2(
					static_cast<real_t>(scaled_path[j].X) / (real_t)SCALE_FACTOR,
					static_cast<real_t>(scaled_path[j].Y) / (real_t)SCALE_FACTOR));
		}
		polypaths.push_back(polypath);
	}
	return polypaths;
}

Vector<Vector<Point2>> Geometry2D::_polypath_offset(const Vector<Point2> &p_polypath, real_t p_delta, PolyJoinType p_join_type, PolyEndType p_end_type) {
	using namespace ClipperLib;

	JoinType jt = jtSquare;

	switch (p_join_type) {
		case JOIN_SQUARE:
			jt = jtSquare;
			break;
		case JOIN_ROUND:
			jt = jtRound;
			break;
		case JOIN_MITER:
			jt = jtMiter;
			break;
	}

	EndType et = etClosedPolygon;

	switch (p_end_type) {
		case END_POLYGON:
			et = etClosedPolygon;
			break;
		case END_JOINED:
			et = etClosedLine;
			break;
		case END_BUTT:
			et = etOpenButt;
			break;
		case END_SQUARE:
			et = etOpenSquare;
			break;
		case END_ROUND:
			et = etOpenRound;
			break;
	}
	ClipperOffset co(2.0, 0.25f * (real_t)SCALE_FACTOR); // Defaults from ClipperOffset.
	Path path;

	// Need to scale points (Clipper's requirement for robust computation).
	for (int i = 0; i != p_polypath.size(); ++i) {
		path << IntPoint(p_polypath[i].x * (real_t)SCALE_FACTOR, p_polypath[i].y * (real_t)SCALE_FACTOR);
	}
	co.AddPath(path, jt, et);

	Paths paths;
	co.Execute(paths, p_delta * (real_t)SCALE_FACTOR); // Inflate/deflate.

	// Have to scale points down now.
	Vector<Vector<Point2>> polypaths;

	for (Paths::size_type i = 0; i < paths.size(); ++i) {
		Vector<Vector2> polypath;

		const Path &scaled_path = paths[i];

		for (Paths::size_type j = 0; j < scaled_path.size(); ++j) {
			polypath.push_back(Point2(
					static_cast<real_t>(scaled_path[j].X) / (real_t)SCALE_FACTOR,
					static_cast<real_t>(scaled_path[j].Y) / (real_t)SCALE_FACTOR));
		}
		polypaths.push_back(polypath);
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

static void tessellate_cubic_bezier_in_rect(Vector<Vector2> &r_out, Vector2 p_start, Vector2 p_control1, Vector2 p_control2, Vector2 p_end, Vector2 p_start_transformed, Vector2 p_control1_transformed, Vector2 p_control2_transformed, Vector2 p_end_transformed, const Rect2 &p_limit) {
	while (true) {
		// Stop condition - Completely out of bounds
		real_t limit;
		limit = p_limit.position.x;
		if (p_start_transformed.x < limit && p_control1_transformed.x < limit && p_control2_transformed.x < limit && p_end_transformed.x < limit) {
			break;
		}
		limit = p_limit.position.y;
		if (p_start_transformed.y < limit && p_control1_transformed.y < limit && p_control2_transformed.y < limit && p_end_transformed.y < limit) {
			break;
		}
		limit = p_limit.position.x + p_limit.size.x;
		if (p_start_transformed.x > limit && p_control1_transformed.x > limit && p_control2_transformed.x > limit && p_end_transformed.x > limit) {
			break;
		}
		limit = p_limit.position.y + p_limit.size.y;
		if (p_start_transformed.y > limit && p_control1_transformed.y > limit && p_control2_transformed.y > limit && p_end_transformed.y > limit) {
			break;
		}
		// Stop condition - Sufficiently close to a line
		real_t a = p_control1_transformed.distance_squared_to(p_start_transformed);
		real_t b = p_control1_transformed.distance_squared_to(p_end_transformed);
		real_t c = p_start_transformed.distance_squared_to(p_end_transformed);
		real_t abc = (c - b + a);
		if ((a + b + c < 0.25 || abc * abc - c * (4.0 * a - 1.0) >= 0.0) && c + 0.25 > a && c + 0.25 > b) {
			a = p_control2_transformed.distance_squared_to(p_start_transformed);
			b = p_control2_transformed.distance_squared_to(p_end_transformed);
			abc = (c - b + a);
			if ((a + b + c < 0.25 || abc * abc - c * (4.0 * a - 1.0) >= 0.0) && c + 0.25 > a && c + 0.25 > b) {
				break;
			}
		}
		// Subdivision
		Vector2 start_control1 = 0.5 * (p_start + p_control1);
		Vector2 control1_control2 = 0.5 * (p_control1 + p_control2);
		Vector2 control2_end = 0.5 * (p_control2 + p_end);

		Vector2 start_control1_control2 = 0.5 * (start_control1 + control1_control2);
		Vector2 control1_control2_end = 0.5 * (control1_control2 + control2_end);

		Vector2 start_control1_control2_end = 0.5 * (start_control1_control2 + control1_control2_end);

		Vector2 start_control1_transformed = 0.5 * (p_start_transformed + p_control1_transformed);
		Vector2 control1_control2_transformed = 0.5 * (p_control1_transformed + p_control2_transformed);
		Vector2 control2_end_transformed = 0.5 * (p_control2_transformed + p_end_transformed);

		Vector2 start_control1_control2_transformed = 0.5 * (start_control1_transformed + control1_control2_transformed);
		Vector2 control1_control2_end_transformed = 0.5 * (control1_control2_transformed + control2_end_transformed);

		Vector2 start_control1_control2_end_transformed = 0.5 * (start_control1_control2_transformed + control1_control2_end_transformed);

		tessellate_cubic_bezier_in_rect(r_out, p_start, start_control1, start_control1_control2, start_control1_control2_end, p_start_transformed, start_control1_transformed, start_control1_control2_transformed, start_control1_control2_end_transformed, p_limit);
		p_start = start_control1_control2_end;
		p_control1 = control1_control2_end;
		p_control2 = control2_end;
		p_start_transformed = start_control1_control2_end_transformed;
		p_control1_transformed = control1_control2_end_transformed;
		p_control2_transformed = control2_end_transformed;
	}
	r_out.push_back(p_end);
}

static void tessellate_order5_bezier_in_rect(Vector<Vector2> &r_out, Vector2 p_start, Vector2 p_control1, Vector2 p_control2, Vector2 p_control3, Vector2 p_control4, Vector2 p_end, Vector2 p_start_transformed, Vector2 p_control1_transformed, Vector2 p_control2_transformed, Vector2 p_control3_transformed, Vector2 p_control4_transformed, Vector2 p_end_transformed, const Rect2 &p_limit) {
	while (true) {
		// Stop condition - Completely out of bounds
		real_t limit;
		limit = p_limit.position.x;
		if (p_start_transformed.x < limit && p_control1_transformed.x < limit && p_control2_transformed.x < limit && p_control3_transformed.x < limit && p_control4_transformed.x < limit && p_end_transformed.x < limit) {
			break;
		}
		limit = p_limit.position.y;
		if (p_start_transformed.y < limit && p_control1_transformed.y < limit && p_control2_transformed.y < limit && p_control3_transformed.y < limit && p_control4_transformed.y < limit && p_end_transformed.y < limit) {
			break;
		}
		limit = p_limit.position.x + p_limit.size.x;
		if (p_start_transformed.x > limit && p_control1_transformed.x > limit && p_control2_transformed.x > limit && p_control3_transformed.x > limit && p_control4_transformed.x > limit && p_end_transformed.x > limit) {
			break;
		}
		limit = p_limit.position.y + p_limit.size.y;
		if (p_start_transformed.y > limit && p_control1_transformed.y > limit && p_control2_transformed.y > limit && p_control3_transformed.y > limit && p_control4_transformed.y > limit && p_end_transformed.y > limit) {
			break;
		}
		// Stop condition - Sufficiently close to a line
		real_t a = p_control1_transformed.distance_squared_to(p_start_transformed);
		real_t b = p_control1_transformed.distance_squared_to(p_end_transformed);
		real_t c = p_start_transformed.distance_squared_to(p_end_transformed);
		real_t abc = (c - b + a);
		if ((a + b + c < 0.25 || abc * abc - c * (4.0 * a - 1.0) >= 0.0) && c + 0.25 > a && c + 0.25 > b) {
			a = p_control2_transformed.distance_squared_to(p_start_transformed);
			b = p_control2_transformed.distance_squared_to(p_end_transformed);
			abc = (c - b + a);
			if ((a + b + c < 0.25 || abc * abc - c * (4.0 * a - 1.0) >= 0.0) && c + 0.25 > a && c + 0.25 > b) {
				a = p_control3_transformed.distance_squared_to(p_start_transformed);
				b = p_control3_transformed.distance_squared_to(p_end_transformed);
				abc = (c - b + a);
				if ((a + b + c < 0.25 || abc * abc - c * (4.0 * a - 1.0) >= 0.0) && c + 0.25 > a && c + 0.25 > b) {
					a = p_control4_transformed.distance_squared_to(p_start_transformed);
					b = p_control4_transformed.distance_squared_to(p_end_transformed);
					abc = (c - b + a);
					if ((a + b + c < 0.25 || abc * abc - c * (4.0 * a - 1.0) >= 0.0) && c + 0.25 > a && c + 0.25 > b) {
						break;
					}
				}
			}
		}
		// Subdivision
		Vector2 start_control1 = 0.5 * (p_start + p_control1);
		Vector2 control1_control2 = 0.5 * (p_control1 + p_control2);
		Vector2 control2_control3 = 0.5 * (p_control2 + p_control3);
		Vector2 control3_control4 = 0.5 * (p_control3 + p_control4);
		Vector2 control4_end = 0.5 * (p_control4 + p_end);

		Vector2 start_control1_control2 = 0.5 * (start_control1 + control1_control2);
		Vector2 control1_control2_control3 = 0.5 * (control1_control2 + control2_control3);
		Vector2 control2_control3_control4 = 0.5 * (control2_control3 + control3_control4);
		Vector2 control3_control4_end = 0.5 * (control3_control4 + control4_end);

		Vector2 start_control1_control2_control3 = 0.5 * (start_control1_control2 + control1_control2_control3);
		Vector2 control1_control2_control3_control4 = 0.5 * (control1_control2_control3 + control2_control3_control4);
		Vector2 control2_control3_control4_end = 0.5 * (control2_control3_control4 + control3_control4_end);

		Vector2 start_control1_control2_control3_control4 = 0.5 * (start_control1_control2_control3 + control1_control2_control3_control4);
		Vector2 control1_control2_control3_control4_end = 0.5 * (control1_control2_control3_control4 + control2_control3_control4_end);

		Vector2 start_control1_control2_control3_control4_end = 0.5 * (start_control1_control2_control3_control4 + control1_control2_control3_control4_end);

		Vector2 start_control1_transformed = 0.5 * (p_start_transformed + p_control1_transformed);
		Vector2 control1_control2_transformed = 0.5 * (p_control1_transformed + p_control2_transformed);
		Vector2 control2_control3_transformed = 0.5 * (p_control2_transformed + p_control3_transformed);
		Vector2 control3_control4_transformed = 0.5 * (p_control3_transformed + p_control4_transformed);
		Vector2 control4_end_transformed = 0.5 * (p_control4_transformed + p_end_transformed);

		Vector2 start_control1_control2_transformed = 0.5 * (start_control1_transformed + control1_control2_transformed);
		Vector2 control1_control2_control3_transformed = 0.5 * (control1_control2_transformed + control2_control3_transformed);
		Vector2 control2_control3_control4_transformed = 0.5 * (control2_control3_transformed + control3_control4_transformed);
		Vector2 control3_control4_end_transformed = 0.5 * (control3_control4_transformed + control4_end_transformed);

		Vector2 start_control1_control2_control3_transformed = 0.5 * (start_control1_control2_transformed + control1_control2_control3_transformed);
		Vector2 control1_control2_control3_control4_transformed = 0.5 * (control1_control2_control3_transformed + control2_control3_control4_transformed);
		Vector2 control2_control3_control4_end_transformed = 0.5 * (control2_control3_control4_transformed + control3_control4_end_transformed);

		Vector2 start_control1_control2_control3_control4_transformed = 0.5 * (start_control1_control2_control3_transformed + control1_control2_control3_control4_transformed);
		Vector2 control1_control2_control3_control4_end_transformed = 0.5 * (control1_control2_control3_control4_transformed + control2_control3_control4_end_transformed);

		Vector2 start_control1_control2_control3_control4_end_transformed = 0.5 * (start_control1_control2_control3_control4_transformed + control1_control2_control3_control4_end_transformed);

		tessellate_order5_bezier_in_rect(r_out, p_start, start_control1, start_control1_control2, start_control1_control2_control3, start_control1_control2_control3_control4, start_control1_control2_control3_control4_end, p_start_transformed, start_control1_transformed, start_control1_control2_transformed, start_control1_control2_control3_transformed, start_control1_control2_control3_control4_transformed, start_control1_control2_control3_control4_end_transformed, p_limit);
		p_start = start_control1_control2_control3_control4_end;
		p_control1 = control1_control2_control3_control4_end;
		p_control2 = control2_control3_control4_end;
		p_control3 = control3_control4_end;
		p_control4 = control4_end;
		p_start_transformed = start_control1_control2_control3_control4_end_transformed;
		p_control1_transformed = control1_control2_control3_control4_end_transformed;
		p_control2_transformed = control2_control3_control4_end_transformed;
		p_control3_transformed = control3_control4_end_transformed;
		p_control4_transformed = control4_end_transformed;
	}
	r_out.push_back(p_end);
}

static void tessellate_arc_in_rect(Vector<Vector2> &r_out, const Transform2D &p_axis, const Transform2D &p_axis_transformed, Vector2 p_start, Vector2 p_end, const Rect2 &p_limit) {
	while (true) {
		// Stop condition - Completely out of bounds.
		Rect2 rect;
		rect.position = p_start;
		rect.expand_to(p_end);
		if (!p_limit.intersects_transformed(p_axis_transformed, rect)) {
			break;
		}
		// Stop condition - Sufficiently close to a line.
		Vector2 mid = (p_start + p_end).normalized();
		if (p_axis_transformed.basis_xform(mid - 0.5 * (p_start + p_end)).length_squared() < 0.25) {
			break;
		}
		// Subdivision.
		tessellate_arc_in_rect(r_out, p_axis, p_axis_transformed, p_start, mid, p_limit);
		p_start = mid;
	}
	r_out.push_back(p_axis.xform(p_end));
}

Vector<Vector2> Geometry2D::tessellate_curve_in_rect(const Vector<Vector2> &p_points, const Vector<uint8_t> &p_types, const Transform2D &p_transform, const Rect2 &p_limit, bool use_order5) {
	Vector<Vector2> gather;

	if (p_points.size() < 1) {
		return gather;
	}

	int index = 0;
	gather.push_back(p_points[0]);

	Vector<Vector2> transformed = p_transform.xform(p_points);

	for (int i = 0; i < p_types.size(); i++) {
		for (int j = 0; j < 8; j++) {
			if (p_types[i] & (1 << j)) {
				if (index + 4 >= p_points.size()) {
					goto done;
				}
				Vector2 r1 = p_points[index + 1] - p_points[index + 2];
				Vector2 r2 = p_points[index + 3] - p_points[index + 2];
				Transform2D axis(r1, r2, p_points[index + 2]);
				if (unlikely(axis.determinant() == 0.0)) {
					// The axes are colinear, meaning the arc is either a "line" or a "point".
					// Just connect to the center, and then to the end point.
					gather.push_back(p_points[index + 2]);
					gather.push_back(p_points[index + 4]);
					index += 4;
					continue;
				}
				Transform2D axis_inverse = axis.affine_inverse();
				Transform2D axis_transformed = p_transform * axis;
				Vector2 rstart = axis_inverse.xform(p_points[index]).normalized();
				Vector2 rend = axis_inverse.xform(p_points[index + 4]).normalized();
				gather.push_back(axis.xform(rstart));
				// Limit arc angles to the same quadrant.
				if (rstart.x < 0) {
					if (rstart.y < 0) {
						if (rend.x > 0 || rend.y > 0 || rstart.cross(rend) < 0) {
							tessellate_arc_in_rect(gather, axis, axis_transformed, rstart, Vector2(0.0, -1.0), p_limit);
							if (rend.x < 0 || rend.y > 0) {
								tessellate_arc_in_rect(gather, axis, axis_transformed, Vector2(0.0, -1.0), Vector2(1.0, 0.0), p_limit);
								if (rend.x < 0) {
									tessellate_arc_in_rect(gather, axis, axis_transformed, Vector2(1.0, 0.0), Vector2(0.0, 1.0), p_limit);
									if (rend.y < 0) {
										tessellate_arc_in_rect(gather, axis, axis_transformed, Vector2(0.0, 1.0), Vector2(-1.0, 0.0), p_limit);
										rstart = Vector2(-1.0, 0.0);
									} else {
										rstart = Vector2(0.0, 1.0);
									}
								} else {
									rstart = Vector2(1.0, 0.0);
								}
							} else {
								rstart = Vector2(0.0, -1.0);
							}
						}
					} else {
						if (rend.x > 0 || rend.y < 0 || rstart.cross(rend) < 0) {
							tessellate_arc_in_rect(gather, axis, axis_transformed, rstart, Vector2(-1.0, 0.0), p_limit);
							if (rend.x > 0 || rend.y > 0) {
								tessellate_arc_in_rect(gather, axis, axis_transformed, Vector2(-1.0, 0.0), Vector2(0.0, -1.0), p_limit);
								if (rend.y > 0) {
									tessellate_arc_in_rect(gather, axis, axis_transformed, Vector2(0.0, -1.0), Vector2(1.0, 0.0), p_limit);
									if (rend.x < 0) {
										tessellate_arc_in_rect(gather, axis, axis_transformed, Vector2(1.0, 0.0), Vector2(0.0, 1.0), p_limit);
										rstart = Vector2(0.0, 1.0);
									} else {
										rstart = Vector2(1.0, 0.0);
									}
								} else {
									rstart = Vector2(0.0, -1.0);
								}
							} else {
								rstart = Vector2(-1.0, 0.0);
							}
						}
					}
				} else {
					if (rstart.y < 0) {
						if (rend.x < 0 || rend.y > 0 || rstart.cross(rend) < 0) {
							tessellate_arc_in_rect(gather, axis, axis_transformed, rstart, Vector2(1.0, 0.0), p_limit);
							if (rend.x < 0 || rend.y < 0) {
								tessellate_arc_in_rect(gather, axis, axis_transformed, Vector2(1.0, 0.0), Vector2(0.0, 1.0), p_limit);
								if (rend.y < 0) {
									tessellate_arc_in_rect(gather, axis, axis_transformed, Vector2(0.0, 1.0), Vector2(-1.0, 0.0), p_limit);
									if (rend.x > 0) {
										tessellate_arc_in_rect(gather, axis, axis_transformed, Vector2(-1.0, 0.0), Vector2(0.0, -1.0), p_limit);
										rstart = Vector2(0.0, -1.0);
									} else {
										rstart = Vector2(-1.0, 0.0);
									}
								} else {
									rstart = Vector2(0.0, 1.0);
								}
							} else {
								rstart = Vector2(1.0, 0.0);
							}
						}
					} else {
						if (rend.x < 0 || rend.y < 0 || rstart.cross(rend) < 0) {
							tessellate_arc_in_rect(gather, axis, axis_transformed, rstart, Vector2(0.0, 1.0), p_limit);
							if (rend.x > 0 || rend.y < 0) {
								tessellate_arc_in_rect(gather, axis, axis_transformed, Vector2(0.0, 1.0), Vector2(-1.0, 0.0), p_limit);
								if (rend.x > 0) {
									tessellate_arc_in_rect(gather, axis, axis_transformed, Vector2(-1.0, 0.0), Vector2(0.0, -1.0), p_limit);
									if (rend.y > 0) {
										tessellate_arc_in_rect(gather, axis, axis_transformed, Vector2(0.0, -1.0), Vector2(1.0, 0.0), p_limit);
										rstart = Vector2(1.0, 0.0);
									} else {
										rstart = Vector2(0.0, -1.0);
									}
								} else {
									rstart = Vector2(-1.0, 0.0);
								}
							} else {
								rstart = Vector2(0.0, 1.0);
							}
						}
					}
				}
				tessellate_arc_in_rect(gather, axis, axis_transformed, rstart, rend, p_limit);
				gather.push_back(p_points[index + 4]);
				index += 4;
			} else if (use_order5) {
				if (index + 5 >= p_points.size()) {
					goto done;
				}
				tessellate_order5_bezier_in_rect(gather, p_points[index], p_points[index + 1], p_points[index + 2], p_points[index + 3], p_points[index + 4], p_points[index + 5], transformed[index], transformed[index + 1], transformed[index + 2], transformed[index + 3], transformed[index + 4], transformed[index + 5], p_limit);
				index += 5;
			} else {
				if (index + 3 >= p_points.size()) {
					goto done;
				}
				tessellate_cubic_bezier_in_rect(gather, p_points[index], p_points[index + 1], p_points[index + 2], p_points[index + 3], transformed[index], transformed[index + 1], transformed[index + 2], transformed[index + 3], p_limit);
				index += 3;
			}
		}
	}

	if (use_order5) {
		for (; index + 5 < p_points.size(); index += 5) {
			tessellate_order5_bezier_in_rect(gather, p_points[index], p_points[index + 1], p_points[index + 2], p_points[index + 3], p_points[index + 4], p_points[index + 5], transformed[index], transformed[index + 1], transformed[index + 2], transformed[index + 3], transformed[index + 4], transformed[index + 5], p_limit);
		}
	} else {
		for (; index + 3 < p_points.size(); index += 3) {
			tessellate_cubic_bezier_in_rect(gather, p_points[index], p_points[index + 1], p_points[index + 2], p_points[index + 3], transformed[index], transformed[index + 1], transformed[index + 2], transformed[index + 3], p_limit);
		}
	}

done:

	if (gather.size() < 1) {
		return gather;
	}

	// Optimize vertices too close to each other, or outside the rectangle.
	Vector<Vector2> gather_transformed = p_transform.xform(gather);

	// First pass - Remove points that are too close to each other.
	Vector<int> pass1;
	pass1.push_back(0);
	for (int i = 1; i < gather.size(); i++) {
		if (gather_transformed[i].distance_squared_to(gather_transformed[pass1[pass1.size() - 1]]) < 0.25) {
			continue;
		}
		pass1.push_back(i);
	}
	if (pass1.size() < 2) {
		// The points are all so close they're basically just one point.
		// If that point is outside the limit rect, return an empty result.
		// Otherwise return the one point.
		if (gather_transformed[0].x < p_limit.position.x || gather_transformed[0].y < p_limit.position.y || gather_transformed[0].x > p_limit.position.x + p_limit.size.x || gather_transformed[0].y > p_limit.position.y + p_limit.size.y) {
			return Vector<Vector2>();
		}
		Vector<Vector2> ret;
		ret.push_back(gather[0]);
		return ret;
	}

	// Second pass - Optimize middle points.
	Vector<int> pass2;
	pass2.push_back(pass1[0]);
	pass2.push_back(pass1[1]);
	for (int i = 2; i < pass1.size(); i++) {
		while (pass2.size() > 1) {
			// If three consecutive transformed points are on the same side of the rect,
			// remove the center one.
			Vector2 test1 = gather_transformed[pass2[pass2.size() - 2]];
			Vector2 test2 = gather_transformed[pass2[pass2.size() - 1]];
			Vector2 test3 = gather_transformed[pass1[i]];
			real_t limit;
			limit = p_limit.position.x;
			if (test1.x < limit && test2.x < limit && test3.x < limit) {
				pass2.resize(pass2.size() - 1);
				continue;
			}
			limit = p_limit.position.y;
			if (test1.y < limit && test2.y < limit && test3.y < limit) {
				pass2.resize(pass2.size() - 1);
				continue;
			}
			limit = p_limit.position.x + p_limit.size.x;
			if (test1.x > limit && test2.x > limit && test3.x > limit) {
				pass2.resize(pass2.size() - 1);
				continue;
			}
			limit = p_limit.position.y + p_limit.size.y;
			if (test1.y > limit && test2.y > limit && test3.y > limit) {
				pass2.resize(pass2.size() - 1);
				continue;
			}
			break;
		}
		pass2.push_back(pass1[i]);
	}

	// Third pass - Optimize first and last points.
	int first = 0;
	int last = pass2.size() - 1;

	while (last - first > 1) {
		{
			Vector2 test1 = gather_transformed[pass2[last]];
			Vector2 test2 = gather_transformed[pass2[first]];
			Vector2 test3 = gather_transformed[pass2[first + 1]];
			real_t limit;
			limit = p_limit.position.x;
			if (test1.x < limit && test2.x < limit && test3.x < limit) {
				first++;
				continue;
			}
			limit = p_limit.position.y;
			if (test1.y < limit && test2.y < limit && test3.y < limit) {
				first++;
				continue;
			}
			limit = p_limit.position.x + p_limit.size.x;
			if (test1.x > limit && test2.x > limit && test3.x > limit) {
				first++;
				continue;
			}
			limit = p_limit.position.y + p_limit.size.y;
			if (test1.y > limit && test2.y > limit && test3.y > limit) {
				first++;
				continue;
			}
		}
		{
			Vector2 test1 = gather_transformed[pass2[last - 1]];
			Vector2 test2 = gather_transformed[pass2[last]];
			Vector2 test3 = gather_transformed[pass2[first]];
			real_t limit;
			limit = p_limit.position.x;
			if (test1.x < limit && test2.x < limit && test3.x < limit) {
				last--;
				continue;
			}
			limit = p_limit.position.y;
			if (test1.y < limit && test2.y < limit && test3.y < limit) {
				last--;
				continue;
			}
			limit = p_limit.position.x + p_limit.size.x;
			if (test1.x > limit && test2.x > limit && test3.x > limit) {
				last--;
				continue;
			}
			limit = p_limit.position.y + p_limit.size.y;
			if (test1.y > limit && test2.y > limit && test3.y > limit) {
				last--;
				continue;
			}
		}
		break;
	}

	if (last - first < 2) {
		// Down to just two points. If both are outside, return an empty result.
		Vector2 test1 = gather_transformed[pass2[first]];
		Vector2 test2 = gather_transformed[pass2[last]];
		real_t limit;
		limit = p_limit.position.x;
		if (test1.x < limit && test2.x < limit) {
			return Vector<Vector2>();
		}
		limit = p_limit.position.y;
		if (test1.y < limit && test2.y < limit) {
			return Vector<Vector2>();
		}
		limit = p_limit.position.x + p_limit.size.x;
		if (test1.x > limit && test2.x > limit) {
			return Vector<Vector2>();
		}
		limit = p_limit.position.y + p_limit.size.y;
		if (test1.y > limit && test2.y > limit) {
			return Vector<Vector2>();
		}
	}

	// Final pass - Copy the remaining points, untransformed, to the output.
	last++;
	Vector<Vector2> ret;
	for (int i = first; i < last; i++) {
		ret.push_back(gather[pass2[i]]);
	}
	return ret;
}
