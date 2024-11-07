/**************************************************************************/
/*  nav_mesh_queries_3d.h                                                 */
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

#ifndef NAV_MESH_QUERIES_3D_H
#define NAV_MESH_QUERIES_3D_H

#ifndef _3D_DISABLED

#include "../nav_map.h"

class NavMeshQueries3D {
public:
	static Vector3 polygons_get_random_point(const LocalVector<gd::Polygon> &p_polygons, uint32_t p_navigation_layers, bool p_uniformly);

	static Vector<Vector3> polygons_get_path(const LocalVector<gd::Polygon> &p_polygons, Vector3 p_origin, Vector3 p_destination, bool p_optimize, uint32_t p_navigation_layers, Vector<int32_t> *r_path_types, TypedArray<RID> *r_path_rids, Vector<int64_t> *r_path_owners, const Vector3 &p_map_up, uint32_t p_link_polygons_size);
	static Vector3 polygons_get_closest_point_to_segment(const LocalVector<gd::Polygon> &p_polygons, const Vector3 &p_from, const Vector3 &p_to, const bool p_use_collision);
	static Vector3 polygons_get_closest_point(const LocalVector<gd::Polygon> &p_polygons, const Vector3 &p_point);
	static Vector3 polygons_get_closest_point_normal(const LocalVector<gd::Polygon> &p_polygons, const Vector3 &p_point);
	static gd::ClosestPointQueryResult polygons_get_closest_point_info(const LocalVector<gd::Polygon> &p_polygons, const Vector3 &p_point);
	static RID polygons_get_closest_point_owner(const LocalVector<gd::Polygon> &p_polygons, const Vector3 &p_point);

	static void clip_path(const LocalVector<gd::NavigationPoly> &p_navigation_polys, Vector<Vector3> &path, const gd::NavigationPoly *from_poly, const Vector3 &p_to_point, const gd::NavigationPoly *p_to_poly, Vector<int32_t> *r_path_types, TypedArray<RID> *r_path_rids, Vector<int64_t> *r_path_owners, const Vector3 &p_map_up);
};

#endif // _3D_DISABLED

#endif // NAV_MESH_QUERIES_3D_H
