#include "clipper.h"

#define PRECISION 10000.0

Clipper::Clipper() :
		mode(MODE_CLIP),
		open(false),
		fill_rule(cl::frEvenOdd),
		path_type(cl::ptSubject),
		join_type(cl::kSquare),
		end_type(cl::kPolygon),
		clip_type(cl::ctUnion),
		delta(0.0) {}

//------------------------------------------------------------------------------
// Clipping methods
//------------------------------------------------------------------------------

void Clipper::add_points(const Vector<Vector2> &points) {

	const cl::Path &path = _scale_up(points, PRECISION);

	switch (mode) {

		case MODE_CLIP: {
			cl.AddPath(path, path_type, open);
		} break;

		case MODE_OFFSET: {
			co.AddPath(path, join_type, end_type);
		} break;

		case MODE_TRIANGULATE: {
			ct.AddPath(path, path_type, open);
		} break;
	}
}

void Clipper::execute(bool build_hierarchy) {

	ERR_EXPLAIN("Cannot build hierarchy outside of MODE_CLIP");
	ERR_FAIL_COND(build_hierarchy && mode != MODE_CLIP)

	switch (mode) {

		case MODE_CLIP: {

			if (build_hierarchy) {
				cl.Execute(clip_type, root, solution_open, fill_rule);
				_build_hierarchy(root);
			} else {
				cl.Execute(clip_type, solution_closed, solution_open, fill_rule);
			}
		} break;

		case MODE_OFFSET: {
			co.Execute(solution_closed, delta * PRECISION);
		} break;

		case MODE_TRIANGULATE: {
			ct.Execute(clip_type, solution_closed, fill_rule);
		} break;
	}
}

int Clipper::get_solution_count(SolutionType type) const {

	switch (type) {

		case TYPE_CLOSED: {
			return solution_closed.size();
		} break;

		case TYPE_OPEN: {
			return solution_open.size();
		} break;
	}
	return -1;
}

Vector<Vector2> Clipper::get_solution(int idx, SolutionType type) {

	switch (type) {

		case TYPE_CLOSED: {

			ERR_EXPLAIN("Closed solution not found");
			ERR_FAIL_INDEX_V(idx, solution_closed.size(), Vector<Vector2>());

			return _scale_down(solution_closed[idx], PRECISION);
		} break;

		case TYPE_OPEN: {

			ERR_EXPLAIN("Open solution not found");
			ERR_FAIL_INDEX_V(idx, solution_open.size(), Vector<Vector2>());

			return _scale_down(solution_open[idx], PRECISION);
		} break;
	}
	return Vector<Vector2>();
}

Rect2 Clipper::get_bounds() {

	ERR_EXPLAIN("Cannot get solution bounds in MODE_OFFSET");
	ERR_FAIL_COND_V(mode == MODE_OFFSET, Rect2());

	cl::Rect64 b(0, 0, 0, 0);

	switch (mode) {

		case MODE_CLIP: {
			b = cl.GetBounds();
		} break;

		case MODE_OFFSET: {
		} break;

		case MODE_TRIANGULATE: {
			b = ct.GetBounds();
		} break;
	}

	cl::Point64 pos(b.left, b.top);
	cl::Point64 size(b.right - b.left, b.bottom - b.top);

	Rect2 bounds(
			static_cast<real_t>(pos.x) / PRECISION,
			static_cast<real_t>(pos.y) / PRECISION,
			static_cast<real_t>(size.x) / PRECISION,
			static_cast<real_t>(size.y) / PRECISION);

	return bounds;
}

void Clipper::clear() {

	solution_closed.clear();
	solution_open.clear();

	root.Clear();
	polypaths.clear();
	path_map.clear();

	cl.Clear();
	co.Clear();
	ct.Clear();
}

Vector<int> Clipper::get_hierarchy(int idx) {

	ERR_FAIL_INDEX_V(idx, polypaths.size(), Vector<int>());

	// Returns indices to parent and children of the selected solution

	Vector<int> hierarchy;
	cl::PolyPath *path = polypaths[idx];

	// Parent
	cl::PolyPath *parent = path->GetParent();
	hierarchy.push_back(path_map[parent]);

	// Children
	for (int c = 0; c < path->ChildCount(); ++c) {

		cl::PolyPath &child = path->GetChild(c);
		hierarchy.push_back(path_map[&child]);
	}
	return hierarchy;
}

Vector<Vector2> Clipper::get_parent(int idx) {

	ERR_FAIL_INDEX_V(idx, polypaths.size(), Vector<Vector2>());

	cl::PolyPath *path = polypaths[idx];
	cl::PolyPath *parent = path->GetParent();

	return _scale_down(parent->GetPath(), PRECISION);
}

Vector<Vector2> Clipper::get_child(int idx, int child_idx) {

	ERR_FAIL_INDEX_V(idx, polypaths.size(), Vector<Vector2>());
	ERR_FAIL_INDEX_V(child_idx, polypaths[idx]->ChildCount(), Vector<Vector2>());

	cl::PolyPath *path = polypaths[idx];
	cl::PolyPath &child = path->GetChild(child_idx);

	return _scale_down(child.GetPath(), PRECISION);
}

int Clipper::get_child_count(int idx) {

	ERR_FAIL_INDEX_V(idx, polypaths.size(), -1);

	cl::PolyPath *path = polypaths[idx];

	return path->ChildCount();
}

Array Clipper::get_children(int idx) {

	ERR_FAIL_INDEX_V(idx, polypaths.size(), Array());

	cl::PolyPath *path = polypaths[idx];

	Array children;

	for (int c = 0; c < path->ChildCount(); ++c) {

		cl::PolyPath &child = path->GetChild(c);
		children.push_back(_scale_down(child.GetPath(), PRECISION));
	}
	return children;
}

bool Clipper::is_hole(int idx) {

	ERR_FAIL_INDEX_V(idx, polypaths.size(), false);

	cl::PolyPath *path = polypaths[idx];

	return path->IsHole();
}

//------------------------------------------------------------------------------

void Clipper::_build_hierarchy(cl::PolyPath &p_root) {

	solution_closed.clear();
	polypaths.clear();

	List<cl::PolyPath *> to_visit;
	to_visit.push_back(&p_root);

	for (int idx = -1; !to_visit.empty(); ++idx) {

		cl::PolyPath *parent = to_visit.back()->get();
		to_visit.pop_back();

		for (int c = 0; c < parent->ChildCount(); ++c) {

			cl::PolyPath &child = parent->GetChild(c);
			to_visit.push_back(&child);
		}
		// Ignore root (used as container only, has no boundary)
		if (idx != -1) {
			// Rely on order
			solution_closed.push_back(parent->GetPath());
			polypaths.push_back(parent);
			path_map[parent] = idx;
		}
	}
}

void Clipper::set_mode(ClipMode p_mode, bool reuse_solution) {

	if (mode == p_mode)
		return;

	mode = p_mode;

	if (reuse_solution) {

		// Transfer resulted solution from previous execution
		switch (mode) {

			case MODE_CLIP: {
				cl.AddPaths(solution_closed, path_type, false);
				cl.AddPaths(solution_open, path_type, true);
			} break;

			case MODE_OFFSET: {
				co.AddPaths(solution_closed, join_type, end_type);
			} break;

			case MODE_TRIANGULATE: {
				ct.AddPaths(solution_closed, path_type, false);
			} break;
		}
	}
}

_FORCE_INLINE_ cl::Path Clipper::_scale_up(const Vector<Vector2> &points, real_t scale) {

	cl::Path path;
	path.resize(points.size());

	for (int i = 0; i != points.size(); ++i) {
		path[i] = cl::Point64(
				points[i].x * scale,
				points[i].y * scale);
	}
	return path;
}

_FORCE_INLINE_ Vector<Vector2> Clipper::_scale_down(const cl::Path &path, real_t scale) {

	Vector<Vector2> points;
	points.resize(path.size());

	for (int i = 0; i != path.size(); ++i) {
		points.write[i] = Point2(
				static_cast<real_t>(path[i].x) / scale,
				static_cast<real_t>(path[i].y) / scale);
	}
	return points;
}

void Clipper::_bind_methods() {

	//--------------------------------------------------------------------------
	// Clipper methods
	//--------------------------------------------------------------------------

	ClassDB::bind_method(D_METHOD("add_points", "points"), &Clipper::add_points);
	ClassDB::bind_method(D_METHOD("execute", "build_hierarchy"), &Clipper::execute, DEFVAL(false));

	ClassDB::bind_method(D_METHOD("get_solution_count", "type"), &Clipper::get_solution_count, DEFVAL(TYPE_CLOSED));
	ClassDB::bind_method(D_METHOD("get_solution", "index", "type"), &Clipper::get_solution, DEFVAL(TYPE_CLOSED));

	ClassDB::bind_method(D_METHOD("get_bounds"), &Clipper::get_bounds);
	ClassDB::bind_method(D_METHOD("clear"), &Clipper::clear);

	// Hierarchy
	ClassDB::bind_method(D_METHOD("get_child_count", "idx"), &Clipper::get_child_count);
	ClassDB::bind_method(D_METHOD("get_child", "idx", "child_idx"), &Clipper::get_child);
	ClassDB::bind_method(D_METHOD("get_children", "idx"), &Clipper::get_children);
	ClassDB::bind_method(D_METHOD("get_parent", "idx"), &Clipper::get_parent);
	ClassDB::bind_method(D_METHOD("is_hole", "idx"), &Clipper::is_hole);
	ClassDB::bind_method(D_METHOD("get_hierarchy", "idx"), &Clipper::get_hierarchy);

	ADD_PROPERTY(PropertyInfo(Variant::RECT2, "bounds"), "", "get_bounds");

	//--------------------------------------------------------------------------
	// Configuration methods
	//--------------------------------------------------------------------------
	ClassDB::bind_method(D_METHOD("set_mode", "mode", "reuse_solution"), &Clipper::set_mode, DEFVAL(true));
	ClassDB::bind_method(D_METHOD("get_mode"), &Clipper::get_mode);

	ClassDB::bind_method(D_METHOD("set_open", "is_open"), &Clipper::set_open);
	ClassDB::bind_method(D_METHOD("is_open"), &Clipper::is_open);

	ClassDB::bind_method(D_METHOD("set_path_type", "path_type"), &Clipper::set_path_type);
	ClassDB::bind_method(D_METHOD("get_path_type"), &Clipper::get_path_type);

	ClassDB::bind_method(D_METHOD("set_clip_type", "clip_type"), &Clipper::set_clip_type);
	ClassDB::bind_method(D_METHOD("get_clip_type"), &Clipper::get_clip_type);

	ClassDB::bind_method(D_METHOD("set_fill_rule", "fill_rule"), &Clipper::set_fill_rule);
	ClassDB::bind_method(D_METHOD("get_fill_rule"), &Clipper::get_fill_rule);

	ClassDB::bind_method(D_METHOD("set_join_type", "join_type"), &Clipper::set_join_type);
	ClassDB::bind_method(D_METHOD("get_join_type"), &Clipper::get_join_type);

	ClassDB::bind_method(D_METHOD("set_end_type", "end_type"), &Clipper::set_end_type);
	ClassDB::bind_method(D_METHOD("get_end_type"), &Clipper::get_clip_type);

	ClassDB::bind_method(D_METHOD("set_delta", "delta"), &Clipper::set_delta);
	ClassDB::bind_method(D_METHOD("get_delta"), &Clipper::get_delta);

	ADD_PROPERTY(PropertyInfo(Variant::INT, "mode"), "", "get_mode");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "open"), "set_open", "is_open");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "path_type"), "set_path_type", "get_path_type");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "clip_type"), "set_clip_type", "get_clip_type");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "fill_rule"), "set_fill_rule", "get_fill_rule");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "end_type"), "set_end_type", "get_end_type");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "join_type"), "set_join_type", "get_join_type");
	ADD_PROPERTY(PropertyInfo(Variant::REAL, "delta"), "set_delta", "get_delta");

	//--------------------------------------------------------------------------
	// Enums
	//--------------------------------------------------------------------------
	using namespace clipperlib;
	// Use namespace declaration to avoid having
	// prepended namespace name in enum constants
	BIND_ENUM_CONSTANT(MODE_CLIP);
	BIND_ENUM_CONSTANT(MODE_OFFSET);
	BIND_ENUM_CONSTANT(MODE_TRIANGULATE);

	BIND_ENUM_CONSTANT(TYPE_CLOSED);
	BIND_ENUM_CONSTANT(TYPE_OPEN);

	BIND_ENUM_CONSTANT(ctNone);
	BIND_ENUM_CONSTANT(ctIntersection);
	BIND_ENUM_CONSTANT(ctUnion);
	BIND_ENUM_CONSTANT(ctDifference);
	BIND_ENUM_CONSTANT(ctXor);

	BIND_ENUM_CONSTANT(ptSubject);
	BIND_ENUM_CONSTANT(ptClip);

	BIND_ENUM_CONSTANT(frEvenOdd);
	BIND_ENUM_CONSTANT(frNonZero);
	BIND_ENUM_CONSTANT(frPositive);
	BIND_ENUM_CONSTANT(frNegative);

	BIND_ENUM_CONSTANT(kSquare);
	BIND_ENUM_CONSTANT(kRound);
	BIND_ENUM_CONSTANT(kMiter);

	BIND_ENUM_CONSTANT(kPolygon);
	BIND_ENUM_CONSTANT(kOpenJoined);
	BIND_ENUM_CONSTANT(kOpenButt);
	BIND_ENUM_CONSTANT(kOpenSquare);
	BIND_ENUM_CONSTANT(kOpenRound);
}
