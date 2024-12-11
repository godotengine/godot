#include "jolt_concave_polygon_shape_impl_3d.hpp"

#include "servers/jolt_project_settings.hpp"

Variant JoltConcavePolygonShapeImpl3D::get_data() const {
	Dictionary data;
	data["faces"] = faces;
	data["backface_collision"] = back_face_collision;
	return data;
}

void JoltConcavePolygonShapeImpl3D::set_data(const Variant& p_data) {
	ERR_FAIL_COND(p_data.get_type() != Variant::DICTIONARY);

	const Dictionary data = p_data;

	const Variant maybe_faces = data.get("faces", {});
	ERR_FAIL_COND(maybe_faces.get_type() != Variant::PACKED_VECTOR3_ARRAY);

	const Variant maybe_back_face_collision = data.get("backface_collision", {});
	ERR_FAIL_COND(maybe_back_face_collision.get_type() != Variant::BOOL);

	faces = maybe_faces;
	back_face_collision = maybe_back_face_collision;

	aabb = _calculate_aabb();

	destroy();
}

String JoltConcavePolygonShapeImpl3D::to_string() const {
	return vformat("{vertex_count=%d}", faces.size());
}

JPH::ShapeRefC JoltConcavePolygonShapeImpl3D::_build() const {
	const auto vertex_count = (int32_t)faces.size();
	const int32_t face_count = vertex_count / 3;
	const int32_t excess_vertex_count = vertex_count % 3;

	QUIET_FAIL_COND_D(vertex_count == 0);

	ERR_FAIL_COND_D_MSG(
		vertex_count < 3,
		vformat(
			"Godot Jolt failed to build concave polygon shape with %s. "
			"It must have a vertex count of at least 3. "
			"This shape belongs to %s.",
			to_string(),
			_owners_to_string()
		)
	);

	ERR_FAIL_COND_D_MSG(
		excess_vertex_count != 0,
		vformat(
			"Godot Jolt failed to build concave polygon shape with %s. "
			"It must have a vertex count that is divisible by 3. "
			"This shape belongs to %s.",
			to_string(),
			_owners_to_string()
		)
	);

	JPH::TriangleList jolt_faces;
	jolt_faces.reserve((size_t)face_count);

	const Vector3* faces_begin = &faces[0];
	const Vector3* faces_end = faces_begin + vertex_count;
	JPH::uint32 triangle_index = 0;

	for (const Vector3* vertex = faces_begin; vertex != faces_end; vertex += 3) {
		const Vector3* v0 = vertex + 0;
		const Vector3* v1 = vertex + 1;
		const Vector3* v2 = vertex + 2;

		jolt_faces.emplace_back(
			JPH::Float3((float)v2->x, (float)v2->y, (float)v2->z),
			JPH::Float3((float)v1->x, (float)v1->y, (float)v1->z),
			JPH::Float3((float)v0->x, (float)v0->y, (float)v0->z),
			0,
			triangle_index++
		);
	}

	JPH::MeshShapeSettings shape_settings(jolt_faces);
	shape_settings.mActiveEdgeCosThresholdAngle = JoltProjectSettings::get_active_edge_threshold();
	shape_settings.mPerTriangleUserData = JoltProjectSettings::enable_ray_cast_face_index();

	const JPH::ShapeSettings::ShapeResult shape_result = shape_settings.Create();

	ERR_FAIL_COND_D_MSG(
		shape_result.HasError(),
		vformat(
			"Godot Jolt failed to build concave polygon shape with %s. "
			"It returned the following error: '%s'. "
			"This shape belongs to %s.",
			to_string(),
			to_godot(shape_result.GetError()),
			_owners_to_string()
		)
	);

	return JoltShapeImpl3D::with_double_sided(shape_result.Get(), back_face_collision);
}

AABB JoltConcavePolygonShapeImpl3D::_calculate_aabb() const {
	AABB result;

	for (int i = 0; i < faces.size(); ++i) {
		const Vector3& vertex = faces[i];

		if (i == 0) {
			result.position = vertex;
		} else {
			result.expand_to(vertex);
		}
	}

	return result;
}
