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

#ifdef CLIPPER_ENABLED
#include "thirdparty/clipper2/include/clipper2/clipper.h"
#endif // CLIPPER_ENABLED
#include "thirdparty/misc/polypartition.h"
#define STB_RECT_PACK_IMPLEMENTATION
#include "thirdparty/misc/stb_rect_pack.h"

#define SCALE_FACTOR 100000.0 // Based on CMP_EPSILON.

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
	Vector<Vector<Vector2>> finished_polygons;

#ifdef CLIPPER_ENABLED
	using namespace Clipper2Lib;

	Paths64 polyon_paths_solution;

	ClipType clipper_cliptype = ClipType::Union;
	FillRule clipper_fillrule = FillRule::EvenOdd;

	if (is_a_open) {
		// Polyline with Polygon
		Clipper64 clipper_64;

		Paths64 polyline_paths;
		Paths64 polygon_paths_b;

		Path64 polyline_path_a;
		Path64 polygon_path_b;

		for (const Vector2 &polyline_point : p_polypath_a) {
			const Point64 &point = Point64(polyline_point.x * (real_t)SCALE_FACTOR, polyline_point.y * (real_t)SCALE_FACTOR);
			polyline_path_a.push_back(point);
		}
		polyline_paths.push_back(polyline_path_a);

		for (const Vector2 &polypath_outline_point : p_polypath_b) {
			const Point64 &point = Point64(polypath_outline_point.x * (real_t)SCALE_FACTOR, polypath_outline_point.y * (real_t)SCALE_FACTOR);
			polygon_path_b.push_back(point);
		}
		polygon_paths_b.push_back(polygon_path_b);

		switch (p_op) {
			case OPERATION_UNION:
				// not supported for polyline (in Godot)
				return finished_polygons;

			case OPERATION_DIFFERENCE:
				clipper_cliptype = ClipType::Difference;
				clipper_fillrule = FillRule::EvenOdd;
				clipper_64.AddOpenSubject(polyline_paths);
				clipper_64.AddClip(polygon_paths_b);
				break;

			case OPERATION_INTERSECTION:
				clipper_cliptype = ClipType::Intersection;
				clipper_fillrule = FillRule::EvenOdd;
				clipper_64.AddOpenSubject(polyline_paths);
				clipper_64.AddClip(polygon_paths_b);
				break;

			case OPERATION_XOR:
				// not supported for polyline
				return finished_polygons;
		}

		Paths64 polygon_solution, polyline_solution;
		clipper_64.Execute(clipper_cliptype, clipper_fillrule, polygon_solution, polyline_solution);
		polyon_paths_solution = polyline_solution;

	} else {
		// Polygon with Polygon
		Paths64 polygon_paths;
		Paths64 polygon_clip_paths;

		Path64 polygon_path_a;
		Path64 polygon_path_b;

		for (const Vector2 &polypath_outline_point : p_polypath_a) {
			const Point64 &point = Point64(polypath_outline_point.x * (real_t)SCALE_FACTOR, polypath_outline_point.y * (real_t)SCALE_FACTOR);
			polygon_path_a.push_back(point);
		}
		polygon_paths.push_back(polygon_path_a);

		for (const Vector2 &polypath_outline_point : p_polypath_b) {
			const Point64 &point = Point64(polypath_outline_point.x * (real_t)SCALE_FACTOR, polypath_outline_point.y * (real_t)SCALE_FACTOR);
			polygon_path_b.push_back(point);
		}
		polygon_clip_paths.push_back(polygon_path_b);

		switch (p_op) {
			case OPERATION_UNION:
				clipper_cliptype = ClipType::Union;
				clipper_fillrule = FillRule::NonZero;

				polyon_paths_solution = Union(polygon_paths, polygon_clip_paths, clipper_fillrule);
				break;
			case OPERATION_DIFFERENCE:
				clipper_cliptype = ClipType::Difference;
				clipper_fillrule = FillRule::EvenOdd;

				polyon_paths_solution = Difference(polygon_paths, polygon_clip_paths, clipper_fillrule);
				break;
			case OPERATION_INTERSECTION:
				clipper_cliptype = ClipType::Intersection;
				clipper_fillrule = FillRule::NonZero;

				polyon_paths_solution = Intersect(polygon_paths, polygon_clip_paths, clipper_fillrule);
				break;
			case OPERATION_XOR:
				clipper_cliptype = ClipType::Xor;
				clipper_fillrule = FillRule::NonZero;

				polyon_paths_solution = Xor(polygon_paths, polygon_clip_paths, clipper_fillrule);
				break;
		}
	}

	for (const Path64 &polyon_path : polyon_paths_solution) {
		Vector<Vector2> finished_polygon;
		for (const Point64 &polyon_path_point : polyon_path) {
			finished_polygon.push_back(Vector2(static_cast<real_t>(polyon_path_point.x), static_cast<real_t>(polyon_path_point.y)) / (real_t)SCALE_FACTOR);
		}
		finished_polygons.push_back(finished_polygon);
	}
#endif // CLIPPER_ENABLED

	return finished_polygons;
}

Vector<Vector<Point2>> Geometry2D::_polypath_offset(const Vector<Point2> &p_polypath, real_t p_delta, PolyJoinType p_join_type, PolyEndType p_end_type) {
	Vector<Vector<Vector2>> finished_polygons;

#ifdef CLIPPER_ENABLED
	using namespace Clipper2Lib;

	JoinType clipper_jointype = JoinType::Miter;

	switch (p_join_type) {
		case JOIN_SQUARE:
			clipper_jointype = JoinType::Square;
			break;
		case JOIN_ROUND:
			clipper_jointype = JoinType::Round;
			break;
		case JOIN_MITER:
			clipper_jointype = JoinType::Miter;
			break;
	}

	EndType clipper_endtype = EndType::Polygon;

	switch (p_end_type) {
		case END_POLYGON:
			clipper_endtype = EndType::Polygon;
			break;
		case END_JOINED:
			clipper_endtype = EndType::Joined;
			break;
		case END_BUTT:
			clipper_endtype = EndType::Butt;
			break;
		case END_SQUARE:
			clipper_endtype = EndType::Square;
			break;
		case END_ROUND:
			clipper_endtype = EndType::Round;
			break;
	}

	Paths64 polygon_paths;

	Path64 polygon_path;
	for (const Vector2 &polypath_outline_point : p_polypath) {
		const Point64 &point = Point64(polypath_outline_point.x * (real_t)SCALE_FACTOR, polypath_outline_point.y * (real_t)SCALE_FACTOR);
		polygon_path.push_back(point);
	}
	polygon_paths.push_back(polygon_path);

	Paths64 paths_solution = InflatePaths(polygon_paths, p_delta, clipper_jointype, clipper_endtype);

	for (const Path64 &scaled_path : paths_solution) {
		Vector<Vector2> polypath;
		for (const Point64 &scaled_point : scaled_path) {
			polypath.push_back(Vector2(static_cast<real_t>(scaled_point.x), static_cast<real_t>(scaled_point.y)) / (real_t)SCALE_FACTOR);
		}
		finished_polygons.push_back(polypath);
	}
#endif // CLIPPER_ENABLED

	return finished_polygons;
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
