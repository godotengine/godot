#include "jolt_world_boundary_shape_impl_3d.hpp"

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

JPH::ShapeRefC JoltWorldBoundaryShapeImpl3D::_build() const {
	ERR_FAIL_D_MSG(vformat(
		"WorldBoundaryShape3D is not supported by Godot Jolt. "
		"Consider using one or more reasonably sized BoxShape3D instead. "
		"This shape belongs to %s.",
		_owners_to_string()
	));
}
