#include "navmesh_query.h"
#include "detour.h"
#include <DetourNavMeshQuery.h>
DetourNavigationQueryFilter::DetourNavigationQueryFilter() :
		Reference() {
	query_filter = memnew(dtQueryFilter());
}
DetourNavigationQueryFilter::~DetourNavigationQueryFilter() {
	memdelete(query_filter);
}
void DetourNavigationQueryFilter::set_area_cost(int area_id, float cost) {
	if (query_filter)
		query_filter->setAreaCost(area_id, cost);
}
float DetourNavigationQueryFilter::get_area_cost(int area_id) {
	if (query_filter)
		return query_filter->getAreaCost(area_id);
	return 1.0f;
}
void DetourNavigationQueryFilter::_bind_methods() {
	ClassDB::bind_method(D_METHOD("set_area_cost", "area_id", "cost"),
			&DetourNavigationQueryFilter::set_area_cost);
	ClassDB::bind_method(D_METHOD("get_area_cost", "area_id"),
			&DetourNavigationQueryFilter::get_area_cost);
}

class DetourNavigationQuery::QueryData {
public:
	Vector3 path_points[MAX_POLYS];
	unsigned char path_flags[MAX_POLYS];
	dtPolyRef polys[MAX_POLYS];
	dtPolyRef path_polys[MAX_POLYS];
};

DetourNavigationQuery::DetourNavigationQuery() :
		Object(),
		query_data(memnew(QueryData)) {}

DetourNavigationQuery::~DetourNavigationQuery() {}
void DetourNavigationQuery::init(Ref<DetourNavigationMesh> mesh,
		const Transform &xform) {
	navmesh_query = dtAllocNavMeshQuery();
	if (!navmesh_query) {
		ERR_PRINT("failed to create navigation query");
		return;
	}
	if (dtStatusFailed(navmesh_query->init(mesh->get_navmesh(), MAX_POLYS))) {
		ERR_PRINT("failed to initialize navigation query");
		return;
	}
	transform = xform;
	inverse = xform.inverse();
}
float DetourNavigationQuery::random() {
	return (float)Math::randf();
}

Vector3 DetourNavigationQuery::nearest_point_(const Vector3 &point,
		const Vector3 &extents,
		const dtQueryFilter *filter,
		uint64_t *ppref) {
	if (!navmesh_query)
		return point;
	Vector3 nearest_point;
	dtPolyRef dtppref = ppref ? (dtPolyRef)(*ppref) : 0;
	navmesh_query->findNearestPoly(&point.coord[0], &extents.coord[0], filter,
			&dtppref, &nearest_point.coord[0]);
	*ppref = dtppref;
	if (*ppref)
		return nearest_point;
	else
		return point;
}

Vector3
DetourNavigationQuery::nearest_point(const Vector3 &point,
		const Vector3 &extents,
		Ref<DetourNavigationQueryFilter> filter) {
	if (!navmesh_query)
		return point;
	Vector3 local_point = inverse.xform(point);
	Vector3 nearest_point;
	polyref_t pref = 0;
	nearest_point = nearest_point_(local_point, extents, filter, &pref);
	// what is dtQueryFilter and how to work with it?
	if (pref)
		return transform.xform(nearest_point);
	else
		return point;
}

Vector3
DetourNavigationQuery::random_point_(polyref_t *pref,
		Ref<DetourNavigationQueryFilter> filter) {
	if (!navmesh_query)
		return Vector3();
	Vector3 point;
	dtPolyRef dtpref = pref ? *pref : 0;
	navmesh_query->findRandomPoint(filter->get(), random, &dtpref,
			&point.coord[0]);
	*pref = dtpref;
	return point;
}

Vector3
DetourNavigationQuery::random_point(Ref<DetourNavigationQueryFilter> filter) {
	polyref_t pref;
	Vector3 point = random_point_(&pref, filter);
	return transform.xform(point);
}

Vector3 DetourNavigationQuery::random_point_in_circle_(
		const Vector3 &center, float radius, const Vector3 &extents,
		Ref<DetourNavigationQueryFilter> filter, polyref_t *ppref) {
	if (!navmesh_query)
		return center;
	dtPolyRef pref;
	navmesh_query->findNearestPoly(&center.coord[0], &extents.coord[0],
			filter->get(), &pref, NULL);
	if (!pref)
		return center;
	Vector3 point = center;
	dtPolyRef dtppref = ppref ? *ppref : 0;
	navmesh_query->findRandomPointAroundCircle(pref, &center.coord[0], radius,
			filter->get(), random, &dtppref,
			&point.coord[0]);
	*ppref = dtppref;
	return point;
}
Vector3 DetourNavigationQuery::random_point_in_circle(
		const Vector3 &center, float radius, const Vector3 &extents,
		Ref<DetourNavigationQueryFilter> filter) {
	Vector3 local_center = inverse.xform(center);
	polyref_t pref2;
	Vector3 point =
			random_point_in_circle_(local_center, radius, extents, filter, &pref2);
	return transform.xform(point);
}
float DetourNavigationQuery::distance_to_wall_(
		const Vector3 &point, float radius, const Vector3 &extents,
		Ref<DetourNavigationQueryFilter> filter, Vector3 *hit_pos,
		Vector3 *hit_normal) {
	if (hit_pos)
		*hit_pos = Vector3();
	if (hit_normal)
		*hit_normal = Vector3(0.0, -1.0, 0.0);
	float distance = radius;
	dtPolyRef pref;
	if (!navmesh_query)
		return distance;
	navmesh_query->findNearestPoly(&point.coord[0], &extents.coord[0],
			filter->get(), &pref, NULL);
	if (!pref)
		return distance;
	navmesh_query->findDistanceToWall(pref, &point.coord[0], radius,
			filter->get(), &distance,
			reinterpret_cast<float *>(hit_pos),
			reinterpret_cast<float *>(hit_normal));
	return distance;
}
Dictionary DetourNavigationQuery::distance_to_wall_detailed(
		const Vector3 &point, float radius, const Vector3 &extents,
		Ref<DetourNavigationQueryFilter> filter) {
	Dictionary ret;
	Vector3 hit_pos = Vector3(), hit_normal = Vector3(0.0, -1.0, 0.0);

	ret["position"] = hit_pos;
	ret["normal"] = hit_normal;
	ret["distance"] = radius;
	Vector3 local_point = inverse.xform(point);
	float dist = distance_to_wall_(local_point, radius, extents, filter, &hit_pos,
			&hit_normal);
	ret["position"] = hit_pos;
	ret["normal"] = hit_normal;
	ret["distance"] = dist;
	return ret;
}
float DetourNavigationQuery::distance_to_wall(
		const Vector3 &point, float radius, const Vector3 &extents,
		Ref<DetourNavigationQueryFilter> filter) {
	Vector3 local_point = inverse.xform(point);
	return distance_to_wall_(local_point, radius, extents, filter);
}

Vector3 DetourNavigationQuery::raycast_(const Vector3 &start,
		const Vector3 &end,
		const Vector3 &extents,
		Ref<DetourNavigationQueryFilter> filter,
		Vector3 *hit_normal) {
	dtPolyRef pref;
	float r;
	if (hit_normal)
		*hit_normal = Vector3();
	navmesh_query->findNearestPoly(&start.coord[0], &extents.coord[0],
			filter->get(), &pref, NULL);
	if (!pref)
		return end;
	int poly_count = 0;
	navmesh_query->raycast(pref, &start.coord[0], &end.coord[0], filter->get(),
			&r, reinterpret_cast<float *>(hit_normal),
			query_data->polys, &poly_count, MAX_POLYS);
	if (r > 1.0f)
		r = 1.0f;
	return start.linear_interpolate(end, r);
}

Vector<Vector3>
DetourNavigationQuery::raycast(const Vector3 &start, const Vector3 &end,
		const Vector3 &extents,
		Ref<DetourNavigationQueryFilter> filter) {
	Vector<Vector3> ret;
	Vector3 normal(0.0, -1.0, 0.0);
	if (!navmesh_query) {
		ret.push_back(end);
		ret.push_back(normal);
		return ret;
	}
	Vector3 local_start = inverse.xform(start);
	Vector3 local_end = inverse.xform(end);
	Vector3 result = raycast_(local_start, local_end, extents, filter, &normal);
	ret.push_back(transform.xform(result));
	ret.push_back(normal);
	return ret;
}
Vector3 DetourNavigationQuery::move_along_surface_(
		const Vector3 &start, const Vector3 &end, const Vector3 &extents,
		int max_visited, Ref<DetourNavigationQueryFilter> filter) {
	dtPolyRef pstart;
	navmesh_query->findNearestPoly(&start.coord[0], &extents.coord[0],
			filter->get(), &pstart, NULL);
	if (!pstart)
		return end;
	Vector3 result;
	int visited = 0;
	Vector<dtPolyRef> visited_ref;
	visited_ref.resize(max_visited);
	navmesh_query->moveAlongSurface(
			pstart, &start.coord[0], &end.coord[0], filter->get(), &result.coord[0],
			max_visited > 0 ? &visited_ref.write[0] : NULL, &visited, max_visited);
	return result;
}
Vector3 DetourNavigationQuery::move_along_surface(
		const Vector3 &start, const Vector3 &end, const Vector3 &extents,
		int max_visited, Ref<DetourNavigationQueryFilter> filter) {
	if (!navmesh_query)
		return end;
#if 0
	/* TODO: are these necessary? */
	Vector3 local_start = inverse.xform(start);
	Vector3 local_end = inverse.xform(end);
#endif
	Vector3 result =
			move_along_surface_(start, end, extents, max_visited, filter);
	return transform.xform(result);
}

Dictionary
DetourNavigationQuery::find_path_(const Vector3 &start, const Vector3 &end,
		const Vector3 &extents,
		Ref<DetourNavigationQueryFilter> filter) {
	Vector<Vector3> points;
	Vector<int> flags;
	Dictionary ret;
	if (!navmesh_query)
		return ret;
	dtPolyRef pstart;
	dtPolyRef pend;
	navmesh_query->findNearestPoly(&start.coord[0], &extents.coord[0],
			filter->get(), &pstart, NULL);
	navmesh_query->findNearestPoly(&end.coord[0], &extents.coord[0],
			filter->get(), &pend, NULL);
	if (!pstart || !pend)
		return ret;
	int num_polys = 0;
	int num_path_points = 0;
	navmesh_query->findPath(pstart, pend, &start.coord[0], &end.coord[0],
			filter->get(), query_data->polys, &num_polys,
			MAX_POLYS);
	if (!num_polys)
		return ret;
	Vector3 actual_end = end;
	if (query_data->polys[num_polys - 1] != pend) {
		Vector3 tmp;
		navmesh_query->closestPointOnPoly(query_data->polys[num_polys - 1],
				&end.coord[0], &tmp.coord[0], NULL);
		actual_end = tmp;
	}
	navmesh_query->findStraightPath(
			&start.coord[0], &actual_end.coord[0], query_data->polys, num_polys,
			&query_data->path_points[0].coord[0], &query_data->path_flags[0],
			query_data->path_polys, &num_path_points, MAX_POLYS);
	for (int i = 0; i < num_path_points; i++) {
		points.push_back(query_data->path_points[i]);
		flags.push_back(query_data->path_flags[i]);
	}
	ret["points"] = points;
	ret["flags"] = flags;
	return ret;
}
Dictionary
DetourNavigationQuery::find_path(const Vector3 &start, const Vector3 &end,
		const Vector3 &extents,
		Ref<DetourNavigationQueryFilter> filter) {
	Vector3 local_start = inverse.xform(start);
	Vector3 local_end = inverse.xform(end);
	Dictionary result = find_path_(local_start, local_end, extents, filter);
	Vector<Vector3> points = result["points"];
	for (int i = 0; i < points.size(); i++)
		points.write[i] = transform.xform(points[i]);
	result["points"] = points;
	return result;
}
void DetourNavigationQuery::_bind_methods() {
	ClassDB::bind_method(D_METHOD("init", "navmesh", "xform"),
			&DetourNavigationQuery::init);
	ClassDB::bind_method(D_METHOD("nearest_point", "point", "extents", "filter"),
			&DetourNavigationQuery::nearest_point);
	ClassDB::bind_method(D_METHOD("random_point", "filter"),
			&DetourNavigationQuery::random_point);
	ClassDB::bind_method(D_METHOD("random_point_in_circle", "center", "radius",
								 "extents", "filter"),
			&DetourNavigationQuery::random_point_in_circle);
	ClassDB::bind_method(
			D_METHOD("distance_to_wall", "point", "radius", "extents", "filter"),
			&DetourNavigationQuery::distance_to_wall);
	ClassDB::bind_method(D_METHOD("distance_to_wall_detailed", "point", "radius",
								 "extents", "filter"),
			&DetourNavigationQuery::distance_to_wall_detailed);
	ClassDB::bind_method(D_METHOD("raycast", "start", "end", "extents", "filter"),
			&DetourNavigationQuery::raycast);
	ClassDB::bind_method(D_METHOD("move_along_surface", "start", "end", "extents",
								 "max_visited", "filter"),
			&DetourNavigationQuery::move_along_surface);
	ClassDB::bind_method(
			D_METHOD("find_path", "start", "end", "extents", "filter"),
			&DetourNavigationQuery::find_path);
}
