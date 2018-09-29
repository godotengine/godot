/*************************************************************************/
/*  navmesh_query.h                                                             */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2018 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2018 Godot Engine contributors (cf. AUTHORS.md)    */
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
#ifndef NAVMESH_QUERY_H
#define NAVMESH_QUERY_H
#include "core/math/transform.h"
#include "core/object.h"
#include "core/reference.h"
class dtQueryFilter;
class dtNavMeshQuery;
class DetourNavigationMesh;
class DetourNavigationQueryFilter : public Reference {
	GDCLASS(DetourNavigationQueryFilter, Reference);
	/* Detour query filter */
	dtQueryFilter *query_filter;
	static void _bind_methods();

public:
	DetourNavigationQueryFilter();
	~DetourNavigationQueryFilter();
	const inline dtQueryFilter *get() {
		return query_filter;
	}
	void set_area_cost(int area_id, float cost);
	float get_area_cost(int area_id);
};
class DetourNavigationQuery : public Object {
	GDCLASS(DetourNavigationQuery, Object);
	dtNavMeshQuery *navmesh_query;
	/* Navigation mesh transform */
	Transform transform, inverse;
	static void _bind_methods();

protected:
	static const int MAX_POLYS = 2048;
	/* query data */
	class QueryData;
	QueryData *query_data;
	static float random();

public:
	int get_max_polys() const { return MAX_POLYS; }
	typedef uint64_t polyref_t;
	void init(Ref<DetourNavigationMesh> mesh, const Transform &xform);
	Vector3 nearest_point_(const Vector3 &point, const Vector3 &extents, const dtQueryFilter *filter, polyref_t *ppref);
	inline Vector3 nearest_point_(const Vector3 &point, const Vector3 &extents, Ref<DetourNavigationQueryFilter> filter, polyref_t *ppref) {
		return nearest_point_(point, extents, filter->get(), ppref);
	}
	Vector3 nearest_point(const Vector3 &point, const Vector3 &extents, Ref<DetourNavigationQueryFilter> filter);
	Vector3 random_point_(polyref_t *pref, Ref<DetourNavigationQueryFilter> filter);
	Vector3 random_point(Ref<DetourNavigationQueryFilter> filter);
	Vector3 random_point_in_circle_(const Vector3 &center, float radius, const Vector3 &extents, Ref<DetourNavigationQueryFilter> filter, polyref_t *ppref);
	Vector3 random_point_in_circle(const Vector3 &center, float radius, const Vector3 &extents, Ref<DetourNavigationQueryFilter> filter);
	float distance_to_wall_(const Vector3 &point, float radius, const Vector3 &extents,
			Ref<DetourNavigationQueryFilter> filter,
			Vector3 *hit_pos = NULL, Vector3 *hit_normal = NULL);
	float distance_to_wall(const Vector3 &point, float radius, const Vector3 &extents, Ref<DetourNavigationQueryFilter> filter);
	Dictionary distance_to_wall_detailed(const Vector3 &point, float radius, const Vector3 &extents, Ref<DetourNavigationQueryFilter> filter);
	Vector3 raycast_(const Vector3 &start, const Vector3 &end,
			const Vector3 &extents, Ref<DetourNavigationQueryFilter> filter,
			Vector3 *hit_normal);
	Vector<Vector3> raycast(const Vector3 &start, const Vector3 &end, const Vector3 &extents, Ref<DetourNavigationQueryFilter> filter);
	Vector3 move_along_surface_(const Vector3 &start, const Vector3 &end,
			const Vector3 &extents, int max_visited, Ref<DetourNavigationQueryFilter> filter);
	Vector3 move_along_surface(const Vector3 &start, const Vector3 &end,
			const Vector3 &extents, int max_visited, Ref<DetourNavigationQueryFilter> filter);
	Dictionary find_path_(const Vector3 &start, const Vector3 &end, const Vector3 &extents, Ref<DetourNavigationQueryFilter> filter);
	Dictionary find_path(const Vector3 &start, const Vector3 &end, const Vector3 &extents, Ref<DetourNavigationQueryFilter> filter);
	DetourNavigationQuery();
	~DetourNavigationQuery();
};
#endif
