#include "jolt_world_boundary_shape_impl_3d.hpp"

#include "servers/jolt_project_settings.hpp"

Variant JoltWorldBoundaryShapeImpl3D::get_data() const {
	return plane;
}

void JoltWorldBoundaryShapeImpl3D::set_data(const Variant& p_data) {
	ERR_FAIL_COND(p_data.get_type() != Variant::PLANE);

	const Plane new_plane = p_data;
	QUIET_FAIL_COND(new_plane == plane);

	plane = p_data;

	destroy();
}

AABB JoltWorldBoundaryShapeImpl3D::get_aabb() const {
	const float size = JoltProjectSettings::get_world_boundary_shape_size();
	const float half_size = size / 2.0f;
	return {Vector3(-half_size, -half_size, -half_size), Vector3(size, half_size, size)};
}

String JoltWorldBoundaryShapeImpl3D::to_string() const {
	return vformat("{plane=%s}", plane);
}

JPH::ShapeRefC JoltWorldBoundaryShapeImpl3D::_build() const {
	const Plane normalized_plane = plane.normalized();

	ERR_FAIL_COND_D_MSG(
		normalized_plane == Plane(),
		vformat(
			"Godot Jolt failed to build world boundary shape with %s. "
			"The plane's normal must not be zero. "
			"This shape belongs to %s.",
			to_string(),
			_owners_to_string()
		)
	);

	const float half_size = JoltProjectSettings::get_world_boundary_shape_size() / 2.0f;
	const JPH::PlaneShapeSettings shape_settings(to_jolt(normalized_plane), nullptr, half_size);
	const JPH::ShapeSettings::ShapeResult shape_result = shape_settings.Create();

	ERR_FAIL_COND_D_MSG(
		shape_result.HasError(),
		vformat(
			"Godot Jolt failed to build world boundary shape with %s. "
			"It returned the following error: '%s'. "
			"This shape belongs to %s.",
			to_string(),
			to_godot(shape_result.GetError()),
			_owners_to_string()
		)
	);

	return shape_result.Get();
}
