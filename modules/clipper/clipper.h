#ifndef CLIPPER2_H
#define CLIPPER2_H

#include "core/reference.h"

#include "thirdparty/misc/clipper.h"
#include "thirdparty/misc/clipper_offset.h"
#include "thirdparty/misc/clipper_triangulation.h"

namespace cl = clipperlib;

enum ClipMode {
	MODE_CLIP,
	MODE_OFFSET,
	MODE_TRIANGULATE
};

enum SolutionType {
	TYPE_CLOSED,
	TYPE_OPEN
};

class Clipper : public Reference {
	GDCLASS(Clipper, Reference);

public:
	Clipper();

	//--------------------------------------------------------------------------
	// Clipping methods
	//--------------------------------------------------------------------------
	void add_points(const Vector<Vector2> &points);
	void execute(bool build_hierarchy = false);

	int get_solution_count(SolutionType type = TYPE_CLOSED) const;
	Vector<Vector2> get_solution(int idx, SolutionType type = TYPE_CLOSED);

	Rect2 get_bounds();
	void clear();

	// Hierarchy
	Vector<int> get_hierarchy(int idx);

	Vector<Vector2> get_parent(int idx);
	Vector<Vector2> get_child(int idx, int child_idx);
	int get_child_count(int idx);
	Array get_children(int idx);

	bool is_hole(int idx);

	//--------------------------------------------------------------------------
	// Configuration methods
	//--------------------------------------------------------------------------

	// Path and execute configuration, define these before adding new paths
	// Each path added will have the same configuration as the previous one
	// --------------------------------------------------------------------
	void set_mode(ClipMode p_mode, bool reuse_solution = true);
	ClipMode get_mode() const { return mode; }

	void set_open(bool p_is_open) { open = p_is_open; }
	bool is_open() const { return open; }

	void set_path_type(cl::PathType p_path_type) { path_type = p_path_type; }
	cl::PathType get_path_type() const { return path_type; }

	void set_clip_type(cl::ClipType p_clip_type) { clip_type = p_clip_type; }
	cl::ClipType get_clip_type() const { return clip_type; }

	void set_fill_rule(cl::FillRule p_fill_rule) { fill_rule = p_fill_rule; }
	cl::FillRule get_fill_rule() const { return fill_rule; }

	// Only relevant in MODE_OFFSET
	// ----------------------------
	void set_join_type(cl::JoinType p_join_type) { join_type = p_join_type; }
	cl::JoinType get_join_type() const { return join_type; }

	void set_end_type(cl::EndType p_end_type) { end_type = p_end_type; }
	cl::EndType get_end_type() const { return end_type; }

	void set_delta(real_t p_delta) { delta = p_delta; }
	real_t get_delta() const { return delta; }

	//--------------------------------------------------------------------------
protected:
	static void _bind_methods();

	cl::Path _scale_up(const Vector<Vector2> &points, real_t scale);
	Vector<Vector2> _scale_down(const cl::Path &path, real_t scale);

	void _build_hierarchy(cl::PolyPath &p_root);

private:
	bool open;
	cl::PathType path_type;
	cl::ClipType clip_type;
	cl::FillRule fill_rule;
	cl::JoinType join_type;
	cl::EndType end_type;
	real_t delta;

	ClipMode mode;

	cl::Paths solution_closed;
	cl::Paths solution_open;

	cl::PolyPath root;
	Vector<cl::PolyPath *> polypaths;
	Map<cl::PolyPath *, int> path_map;

	cl::Clipper cl;
	cl::ClipperOffset co;
	cl::ClipperTri ct;
};

VARIANT_ENUM_CAST(ClipMode);
VARIANT_ENUM_CAST(SolutionType);

VARIANT_ENUM_CAST(cl::ClipType);
VARIANT_ENUM_CAST(cl::PathType);
VARIANT_ENUM_CAST(cl::FillRule);

VARIANT_ENUM_CAST(cl::JoinType);
VARIANT_ENUM_CAST(cl::EndType);

#endif
