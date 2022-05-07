/*************************************************************************/
/*  occluder_shape_polygon.h                                             */
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

#ifndef OCCLUDER_SHAPE_POLYGON_H
#define OCCLUDER_SHAPE_POLYGON_H

#include "occluder_shape.h"

class OccluderShapePolygon : public OccluderShape {
	GDCLASS(OccluderShapePolygon, OccluderShape);
	OBJ_SAVE_TYPE(OccluderShapePolygon);

	friend class OccluderSpatialGizmo;

	// points in local space of the plane,
	// not necessary in correct winding order
	// (as they can be edited by the user)
	// Note: these are saved by the IDE
	PoolVector<Vector2> _poly_pts_local_raw;
	PoolVector<Vector2> _hole_pts_local_raw;

	// sanitized
	Vector<Vector2> _poly_pts_local;
	Vector<Vector2> _hole_pts_local;
	bool _settings_two_way = true;

#ifdef TOOLS_ENABLED
	AABB _aabb_local;
	void _update_aabb();
#endif

	// mem funcs
	void _sanitize_points();
	void _sanitize_points_internal(const PoolVector<Vector2> &p_from, Vector<Vector2> &r_to);
	static Vector3 _vec2to3(const Vector2 &p_pt) { return Vector3(p_pt.x, p_pt.y, 0.0); }

protected:
	static void _bind_methods();

public:
	// the raw points are used for the IDE Inspector, and also to allow the user
	// to edit the geometry of the poly at runtime (they can also just change the node transform)
	void set_polygon_points(const PoolVector<Vector2> &p_points);
	PoolVector<Vector2> get_polygon_points() const;
	void set_hole_points(const PoolVector<Vector2> &p_points);
	PoolVector<Vector2> get_hole_points() const;

	// primarily for the gizmo
	void set_polygon_point(int p_idx, const Vector2 &p_point);
	void set_hole_point(int p_idx, const Vector2 &p_point);

	void set_two_way(bool p_two_way);
	bool is_two_way() const { return _settings_two_way; }

	void clear();

	void update_shape_to_visual_server();
	virtual Transform center_node(const Transform &p_global_xform, const Transform &p_parent_xform, real_t p_snap);

#ifdef TOOLS_ENABLED
	virtual AABB get_fallback_gizmo_aabb() const;
#endif

	OccluderShapePolygon();
};

#endif // OCCLUDER_SHAPE_POLYGON_H
