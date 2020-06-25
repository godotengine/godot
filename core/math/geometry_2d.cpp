/*************************************************************************/
/*  geometry_2d.cpp                                                      */
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

#include "geometry_2d.h"

#include "thirdparty/misc/clipper.hpp"
#include "thirdparty/misc/triangulator.h"
#define STB_RECT_PACK_IMPLEMENTATION
#include "thirdparty/misc/stb_rect_pack.h"

#define SCALE_FACTOR 100000.0 // Based on CMP_EPSILON.

Vector<Vector<Vector2>> Geometry2D::decompose_polygon_in_convex(Vector<Point2> polygon) {
	Vector<Vector<Vector2>> decomp;
	List<TriangulatorPoly> in_poly, out_poly;

	TriangulatorPoly inp;
	inp.Init(polygon.size());
	for (int i = 0; i < polygon.size(); i++) {
		inp.GetPoint(i) = polygon[i];
	}
	inp.SetOrientation(TRIANGULATOR_CCW);
	in_poly.push_back(inp);
	TriangulatorPartition tpart;
	if (tpart.ConvexPartition_HM(&in_poly, &out_poly) == 0) { // Failed.
		ERR_PRINT("Convex decomposing failed!");
		return decomp;
	}

	decomp.resize(out_poly.size());
	int idx = 0;
	for (List<TriangulatorPoly>::Element *I = out_poly.front(); I; I = I->next()) {
		TriangulatorPoly &tp = I->get();

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
	int idx;
	_FORCE_INLINE_ bool operator<(const _AtlasWorkRect &p_r) const { return s.width > p_r.s.width; };
};

struct _AtlasWorkRectResult {
	Vector<_AtlasWorkRect> result;
	int max_w;
	int max_h;
};

void Geometry2D::make_atlas(const Vector<Size2i> &p_rects, Vector<Point2i> &r_result, Size2i &r_size) {
	// Super simple, almost brute force scanline stacking fitter.
	// It's pretty basic for now, but it tries to make sure that the aspect ratio of the
	// resulting atlas is somehow square. This is necessary because video cards have limits.
	// On texture size (usually 2048 or 4096), so the more square a texture, the more chances.
	// It will work in every hardware.
	// For example, it will prioritize a 1024x1024 atlas (works everywhere) instead of a
	// 256x8192 atlas (won't work anywhere).

	ERR_FAIL_COND(p_rects.size() == 0);

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
		path_a << IntPoint(p_polypath_a[i].x * SCALE_FACTOR, p_polypath_a[i].y * SCALE_FACTOR);
	}
	for (int i = 0; i != p_polypath_b.size(); ++i) {
		path_b << IntPoint(p_polypath_b[i].x * SCALE_FACTOR, p_polypath_b[i].y * SCALE_FACTOR);
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
					static_cast<real_t>(scaled_path[j].X) / SCALE_FACTOR,
					static_cast<real_t>(scaled_path[j].Y) / SCALE_FACTOR));
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
	ClipperOffset co(2.0, 0.25 * SCALE_FACTOR); // Defaults from ClipperOffset.
	Path path;

	// Need to scale points (Clipper's requirement for robust computation).
	for (int i = 0; i != p_polypath.size(); ++i) {
		path << IntPoint(p_polypath[i].x * SCALE_FACTOR, p_polypath[i].y * SCALE_FACTOR);
	}
	co.AddPath(path, jt, et);

	Paths paths;
	co.Execute(paths, p_delta * SCALE_FACTOR); // Inflate/deflate.

	// Have to scale points down now.
	Vector<Vector<Point2>> polypaths;

	for (Paths::size_type i = 0; i < paths.size(); ++i) {
		Vector<Vector2> polypath;

		const Path &scaled_path = paths[i];

		for (Paths::size_type j = 0; j < scaled_path.size(); ++j) {
			polypath.push_back(Point2(
					static_cast<real_t>(scaled_path[j].X) / SCALE_FACTOR,
					static_cast<real_t>(scaled_path[j].Y) / SCALE_FACTOR));
		}
		polypaths.push_back(polypath);
	}
	return polypaths;
}

Vector<Point2i> Geometry2D::pack_rects(const Vector<Size2i> &p_sizes, const Size2i &p_atlas_size) {
	Vector<stbrp_node> nodes;
	nodes.resize(p_atlas_size.width);

	stbrp_context context;
	stbrp_init_target(&context, p_atlas_size.width, p_atlas_size.height, nodes.ptrw(), p_atlas_size.width);

	Vector<stbrp_rect> rects;
	rects.resize(p_sizes.size());

	for (int i = 0; i < p_sizes.size(); i++) {
		rects.write[i].id = 0;
		rects.write[i].w = p_sizes[i].width;
		rects.write[i].h = p_sizes[i].height;
		rects.write[i].x = 0;
		rects.write[i].y = 0;
		rects.write[i].was_packed = 0;
	}

	int res = stbrp_pack_rects(&context, rects.ptrw(), rects.size());
	if (res == 0) { //pack failed
		return Vector<Point2i>();
	}

	Vector<Point2i> ret;
	ret.resize(p_sizes.size());

	for (int i = 0; i < p_sizes.size(); i++) {
		Point2i r(rects[i].x, rects[i].y);
		ret.write[i] = r;
	}

	return ret;
}

Vector<Vector3i> Geometry2D::partial_pack_rects(const Vector<Vector2i> &p_sizes, const Size2i &p_atlas_size) {
	Vector<stbrp_node> nodes;
	nodes.resize(p_atlas_size.width);
	zeromem(nodes.ptrw(), sizeof(stbrp_node) * nodes.size());

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
